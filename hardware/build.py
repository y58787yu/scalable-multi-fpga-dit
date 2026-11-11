"""Build the fixed Cosmos compute IP and ZCU102 overlay from its sources."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FLOW = ROOT / "flow"
sys.path.insert(0, str(FLOW))
from fast_clock import propagate
from ip_metadata import metadata
from storage_overlay import apply as map_storage


VERSIONS = {"2b": "20.0", "14b": "16.0"}
TOP = "cosmos2b_tile"
MAC = "c209_five_mac_rtl"
BLACKBOXES = (MAC, "c209_gelu_command_rtl", "c209_kv_broadcast_rtl")


def write_text(path, text):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def hls_adapter(rtl):
    """HLS black boxes expose one clock. Restore the real ports after export."""
    external = " input wire mac_fast_clk,mac_fast_rst,\n"
    if rtl.count(external) != 1:
        raise ValueError("The matrix engine clock interface has changed")
    body = rtl.replace(external, "")
    body = re.sub(r"\bmac_fast_clk\b", "ap_clk", body)
    body = re.sub(r"\bmac_fast_rst\b", "ap_rst", body)
    return "// C209_FIVE256_HLS_IMPORT_ONLY_NOT_DEPLOYABLE\n" + body


def bind_leaves(source, output):
    output.mkdir()
    for name in BLACKBOXES:
        spec = json.loads((source / "rtl" / (name + ".json")).read_text(encoding="utf-8"))
        for c_file in spec["c_files"]:
            path = (source / "rtl" / c_file["c_file"]).resolve(strict=True)
            c_file["c_file"] = path.as_posix()
        files = []
        for filename in spec["rtl_files"]:
            path = (source / "rtl" / filename).resolve(strict=True)
            if filename == MAC + ".v":
                adapter = output / filename
                write_text(adapter, hls_adapter(path.read_text(encoding="utf-8")))
                path = adapter
            files.append(path.as_posix())
        spec["rtl_files"] = files
        write_text(output / (name + ".json"), json.dumps(spec, indent=2) + "\n")


def correct_export(ip, source, version):
    folder = ip / "hdl" / "verilog"
    texts = {p.relative_to(ip).as_posix(): p.read_text(encoding="utf-8")
             for p in folder.iterdir() if p.suffix in (".v", ".sv")}
    restored = dict(texts)
    for name in BLACKBOXES:
        recipe = json.loads((source / "rtl" / (name + ".json")).read_text(encoding="utf-8"))
        for filename in recipe["rtl_files"]:
            target = "hdl/verilog/" + filename
            physical = (source / "rtl" / filename).read_text(encoding="utf-8")
            expected = hls_adapter(physical) if filename == MAC + ".v" else physical
            # The two-port shell uses the lexically corrected Verilog wrapper.
            actual = texts[target]
            if name == MAC and filename == MAC + ".v" and version == "20.0":
                actual = actual.replace("2'd1?5'b", "2'd1 ? 5'b").replace("2'd2?5'b", "2'd2 ? 5'b")
            if actual != expected:
                raise ValueError(f"Export differs from the registered RTL: {filename}")
            restored[target] = physical
    patched, _ = propagate(restored, TOP)
    for filename, body in patched.items():
        if body != texts[filename]:
            write_text(ip / filename, body)
    metadata(ip / "component.xml", TOP, version)
    map_storage(ip, source)
    constraint = ip / "constraints" / (TOP + "_ooc.xdc")
    original = constraint.read_text(encoding="utf-8")
    if "mac_fast_clk" in original:
        raise ValueError("An unmodified native HLS export is required")
    write_text(constraint, original + "\ncreate_clock -name mac_fast_clk -period 3.333333 [get_ports mac_fast_clk]\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=VERSIONS)
    parser.add_argument("--output", required=True, type=Path, help="new build directory")
    parser.add_argument("--jobs", type=int, default=4, help="parallel Vivado synthesis runs")
    parser.add_argument("--ip-only", action="store_true", help="stop after compute IP packaging")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    hls, vivado = shutil.which("vitis_hls"), shutil.which("vivado")
    if not hls or not vivado:
        parser.error("source the AMD 2023.2 settings script to make vitis_hls and vivado available")
    xclbinutil = shutil.which("xclbinutil")
    if not args.ip_only and not xclbinutil:
        parser.error("xclbinutil from Vitis 2023.2 is required for the deployment metadata")
    dest = args.output.resolve()
    if dest.exists():
        parser.error("--output must be a new directory")
    dest.mkdir(parents=True)
    source = ROOT / args.model / "compute"
    version = VERSIONS[args.model]
    bind_leaves(source, dest / "leaves")
    env = dict(os.environ, COSMOS_HLS_SOURCE=source.as_posix(),
               COSMOS_HLS_OUTPUT=dest.as_posix(), COSMOS_HLS_IP_VERSION=version)
    subprocess.run([hls, "-f", str(FLOW / "hls.tcl")], env=env, cwd=dest, check=True)
    ip = dest / "hls" / "solution1" / "impl" / "ip"
    correct_export(ip, source, version)
    subprocess.run([vivado, "-mode", "batch", "-source", str(FLOW / "package_ip.tcl"),
                    "-tclargs", str(ip / "component.xml")], cwd=dest, check=True)
    if not args.ip_only:
        subprocess.run([vivado, "-mode", "batch", "-source", str(FLOW / "board_build.tcl"),
                        "-tclargs", args.model, str(dest), str(args.jobs)], cwd=dest, check=True)
        description = dest / "xclbin_metadata"
        description.mkdir()
        for name in ("metadata.xml", "mem.json"):
            shutil.copyfile(FLOW / name, description / name)
        subprocess.run([xclbinutil, "--add-section", "EMBEDDED_METADATA:RAW:metadata.xml",
                        "--add-section", "MEM_TOPOLOGY:JSON:mem.json", "--skip-bank-grouping",
                        "--output", str(dest / "artifacts" / "cosmos_net.xclbin")],
                       cwd=description, check=True)


if __name__ == "__main__":
    main()
