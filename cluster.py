"""Install a Cosmos deployment or run one inference across a ZCU102 cluster."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parent
CONFIGURATIONS = {("2b", 1): 2, ("2b", 2): 2, ("14b", 3): 4}


@dataclass(frozen=True)
class Role:
    group: int
    rank: int
    host: str
    rx: tuple
    port: int


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    model, groups = config["model"], config["groups"]
    size = CONFIGURATIONS.get((model, groups))
    if not size:
        raise ValueError("Supported configurations: 2b/1 group, 2b/2 groups, 14b/3 groups")
    base = PurePosixPath(config["remote_root"])
    if not base.is_absolute() or base == PurePosixPath("/") or ".." in base.parts:
        raise ValueError("remote_root must be an absolute application directory")
    nodes = config["nodes"]
    expected = [(g, r) for g in range(groups) for r in range(size)]
    roles = []
    for node in nodes:
        rx = tuple(node["rx_polarities"])
        port = node.get("port", 22)
        if len(rx) != 4 or any(type(n) is not int or n not in (0, 1) for n in rx):
            raise ValueError("rx_polarities requires four polarity bits in SFP order")
        if not isinstance(node["host"], str) or not node["host"].strip() or not 1 <= port <= 65535:
            raise ValueError("Invalid SSH endpoint")
        roles.append(Role(node["group"], node["rank"], node["host"], rx, port))
    roles.sort(key=lambda role: (role.group, role.rank))
    if [(r.group, r.rank) for r in roles] != expected:
        raise ValueError("The inventory must contain exactly one board for every group/rank")
    if len({(r.host, r.port) for r in roles}) != len(roles):
        raise ValueError("Multiple roles assigned to the same SSH endpoint")
    return config, roles


def fingerprint(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def listing(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root)
        if "__pycache__" in name.parts or path.suffix == ".pyc" or name.name.startswith("."):
            continue
        if path.is_file():
            files[name.as_posix()] = fingerprint(path)
    if not files:
        raise ValueError(f"No installable files in {root}")
    return files


class Node:
    def __init__(self, config, role):
        import paramiko
        self.role = role
        self.base = PurePosixPath(config["remote_root"])
        self.model = config["model"]
        self.groups = config["groups"]
        self.python = config["board_python"]
        self.sudo = config.get("sudo", False)
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        if config.get("known_hosts"):
            self.client.load_host_keys(os.path.expanduser(config["known_hosts"]))
        try:
            self.client.connect(role.host, port=role.port, username=config.get("user"),
                                key_filename=os.path.expanduser(config["ssh_key"]) if config.get("ssh_key") else None,
                                timeout=15, banner_timeout=20, auth_timeout=20)
            self.sftp = self.client.open_sftp()
            self.sftp.get_channel().settimeout(90)
        except BaseException:
            self.client.close()
            raise
        self.channel = None
        self.job = None
        self.nonce = None

    def close(self):
        self.client.close()

    def mkdirs(self, path):
        try:
            self.sftp.stat(str(path))
        except FileNotFoundError:
            self.mkdirs(path.parent)
            try:
                self.sftp.mkdir(str(path))
            except OSError:
                self.sftp.stat(str(path))

    def read(self, path):
        try:
            with self.sftp.open(str(path), "rb") as stream:
                return json.load(stream)
        except OSError as error:
            if error.errno == errno.ENOENT:
                return None
            raise

    def write(self, path, value):
        temporary = str(path) + "." + uuid.uuid4().hex
        with self.sftp.open(temporary, "wb") as stream:
            stream.write(json.dumps(value).encode())
        self.sftp.rename(temporary, str(path))

    def install(self, tag, runtime, runtime_files, assets):
        prepared_files = listing(assets)
        record = dict(model=self.model, groups=self.groups, group=self.role.group, rank=self.role.rank,
                      runtime=runtime_files, assets=prepared_files)
        target = self.base / "releases" / tag
        existing = self.read(target / "install.json")
        if existing == record:
            return
        if existing is not None:
            raise ValueError(f"{self.role.host}: deployment {tag!r} already has different content")
        self.mkdirs(target.parent)
        staging = target.parent / ("." + tag + "-" + uuid.uuid4().hex)
        self.sftp.mkdir(str(staging))
        files, dirs = [], {staging}
        try:
            for subdir, local, items in (("runtime", runtime, runtime_files), ("assets", assets, prepared_files)):
                for name in items:
                    dest = staging / subdir / name
                    for folder in reversed(dest.parents):
                        if folder != staging and staging in folder.parents and folder not in dirs:
                            self.sftp.mkdir(str(folder))
                            dirs.add(folder)
                    files.append(dest)
                    self.sftp.put(str(local / name), str(dest))
            files.append(staging / "install.json")
            with self.sftp.open(str(files[-1]), "wb") as stream:
                stream.write(json.dumps(record).encode())
            self.sftp.rename(str(staging), str(target))
        except BaseException:
            # Roll back only files created by this upload.
            for file in reversed(files):
                try:
                    self.sftp.remove(str(file))
                except FileNotFoundError:
                    pass
            for folder in sorted(dirs, key=lambda d: len(d.parts), reverse=True):
                self.sftp.rmdir(str(folder))
            raise

    def start(self, release, cohort, args, groups):
        installed = self.read(release / "install.json")
        expected = (self.model, self.groups, self.role.group, self.role.rank)
        if not installed or tuple(installed.get(k) for k in ("model","groups","group","rank")) != expected:
            raise ValueError(f"{self.role.host}: missing deployment for this role")
        self.nonce = uuid.uuid4().hex
        job = self.base / "jobs" / cohort
        self.mkdirs(job.parent)
        self.sftp.mkdir(str(job))
        self.job = job
        cmd = [self.python, str(release / "runtime/run.py"), "--assets", str(release / "assets"),
               "--job", str(job), "--cohort", cohort, "--nonce", self.nonce, "--generation", str(args.generation),
               "--group", str(self.role.group), "--rank", str(self.role.rank), "--groups", str(groups),
               "--seed", str(args.seed), "--steps", str(args.steps), "--rx-polarities",
               ",".join(map(str, self.role.rx))]
        if self.sudo:
            cmd = ["sudo", "-n", "--"] + cmd
        self.channel = self.client.get_transport().open_session()
        self.channel.exec_command("umask 022; exec " + shlex.join(cmd) + " >" +
                                  shlex.quote(str(job / "worker.log")) + " 2>&1")

    def poll(self, name):
        # Once the shell exits, any final receipt has already been renamed into place.
        exited = self.channel.exit_status_ready()
        result = self.read(self.job / "result.json")
        if result and result.get("status") == "FAILED":
            raise RuntimeError(f"{self.role.host}: {result.get('error', 'Worker failed')}")
        data = result if name == "result.json" else self.read(self.job / name)
        if data and (name != "result.json" or data.get("status") == "COMPLETED"):
            return data
        if exited:
            raise RuntimeError(f"{self.role.host}: worker exited before {name}; see {self.job}/worker.log")
        return None

    def cancel(self):
        if self.job is not None and self.channel is not None and not self.channel.exit_status_ready():
            self.write(self.job / "cancel.json", {"nonce": self.nonce})

    def await_exit(self, deadline):
        if self.channel is None:
            return None
        while not self.channel.exit_status_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{self.role.host}: worker still owns the FPGA; see {self.job}/worker.log")
            time.sleep(.1)
        return self.channel.recv_exit_status()


def parallel(pool, items, fn):
    futures = [pool.submit(fn, item) for item in items]
    wait(futures)
    return [f.result() for f in futures]


def collect(pool, nodes, name, timeout):
    deadline = time.monotonic() + timeout
    receipts = {}
    pending = list(nodes)
    while pending:
        for n, value in zip(pending, parallel(pool, pending, lambda node: node.poll(name))):
            if value is not None:
                receipts[n] = value
        pending = [n for n in pending if n not in receipts]
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Missing {name} on {', '.join(n.role.host for n in pending)}")
            time.sleep(.1)
    return [receipts[n] for n in nodes]


def barrier(pool, nodes, config, args, cohort, phase, source, destination, step=None):
    docs = collect(pool, nodes, source, args.timeout)
    for node, doc in zip(nodes, docs):
        expected = (cohort, args.generation, args.seed, args.steps, node.role.group, node.role.rank)
        actual = tuple(doc.get(k) for k in ("cohort", "generation", "seed", "execution_steps", "group", "rank"))
        if actual != expected or (step is not None and doc.get("step") != step):
            raise ValueError(f"{node.role.host}: mismatched {phase} receipt")
    validator = ROOT / "runtime" / config["model"] / "make_permit.py"
    result = subprocess.run([sys.executable, str(validator), "--groups", str(config["groups"]), "--phase", phase],
                            input=json.dumps(docs), text=True, stdout=subprocess.PIPE, check=True)
    permit = json.loads(result.stdout)
    parallel(pool, nodes, lambda n: n.write(n.job / destination, permit))
    return permit


def execute(pool, nodes, config, args):
    cohort = uuid.uuid4().hex
    print(f"Run {cohort}", flush=True)
    try:
        # Settle every launch even if one fails, so cancellation reaches all started workers.
        parallel(pool, nodes, lambda n: n.start(n.base / "releases" / args.name, cohort, args, config["groups"]))
        for phase, source, dest in (("INITIALIZE", "prepared.json", "initialize.json"),
                                    ("RELEASE", "ready.json", "release.json")):
            barrier(pool, nodes, config, args, cohort, phase, source, dest)
        for step in range(args.steps):
            permit = barrier(pool, nodes, config, args, cohort, "RESUME", f"step_{step:02d}_done.json",
                             f"step_{step:02d}_resume.json", step)
            print(f"Update {step+1}/{args.steps}", flush=True)
        results = collect(pool, nodes, "result.json", args.timeout)
        for node, result in zip(nodes, results):
            expected = (node.nonce, cohort, args.generation, args.seed, args.steps, node.role.group, node.role.rank)
            keys = ("nonce", "cohort", "generation", "seed", "actual_completed_steps", "group", "rank")
            if tuple(result.get(key) for key in keys) != expected:
                raise ValueError(f"{node.role.host}: final receipt does not belong to this run")
        deadline = time.monotonic() + args.timeout
        if any(code != 0 for code in parallel(pool, nodes, lambda n: n.await_exit(deadline))):
            raise RuntimeError("A board worker returned a nonzero exit code")
        root = nodes[0]
        dest = args.output.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".latent-", delete=False) as tmp:
            temp = Path(tmp.name)
        try:
            root.sftp.get(str(root.job / "output/final_latent.npy"), str(temp))
            if fingerprint(temp) != permit["checkpoint"]["file_sha256"]:
                raise ValueError("Downloaded latent differs from the final committed checkpoint")
            os.replace(temp, dest)
        finally:
            temp.unlink(missing_ok=True)
        print(dest)
    except BaseException:
        # Cancellation goes through the owning worker; it retains its DMA buffers until drained.
        for node in nodes:
            try:
                node.cancel()
            except Exception as error:
                print(f"{node.role.host}: cancellation delivery failed: {error}", file=sys.stderr)
        try:
            deadline = time.monotonic() + min(60, args.timeout)
            parallel(pool, nodes, lambda n: n.await_exit(deadline))
        except Exception as error:
            print(error, file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("install", help="install the bitstream, runtime and per-role model files")
    run = commands.add_parser("run", help="execute a finite number of denoising updates")
    for sub in (upload, run):
        sub.add_argument("--name", required=True, help="immutable deployment name")
    upload.add_argument("--assets", required=True, type=Path)
    run.add_argument("--steps", required=True, type=int)
    run.add_argument("--seed", type=int, default=20261006)
    run.add_argument("--generation", type=int, required=True)
    run.add_argument("--output", type=Path, default=Path("latent.npy"))
    run.add_argument("--timeout", type=float, default=3600, help="per-phase deadline in seconds")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.name):
        parser.error("use a short alphanumeric deployment name")
    config, roles = read_config(args.config)
    if args.command == "run":
        supported_steps = (1, 10) if config["model"] == "2b" else (1, 5, 10)
        if not (1 <= args.generation <= 4095 and 0 <= args.seed < 2**32 and args.steps in supported_steps and args.timeout > 0):
            parser.error("generation, seed, steps or timeout out of range")
    runtime = ROOT / "runtime" / config["model"]
    if args.command == "install":
        runtime_files = listing(runtime)
        assets = {r: args.assets / f"node{i}" for i, r in enumerate(roles, 1)}
        for directory in assets.values():
            if not directory.is_dir():
                parser.error(f"missing role assets: {directory}")
    with ExitStack() as stack, ThreadPoolExecutor(max_workers=len(roles)) as pool:
        nodes = []
        for role in roles:
            node = Node(config, role)
            stack.callback(node.close)
            nodes.append(node)
        if args.command == "install":
            parallel(pool, nodes, lambda n: n.install(args.name, runtime, runtime_files, assets[n.role]))
            print(f"Installed {args.name} on {len(nodes)} boards")
        else:
            execute(pool, nodes, config, args)


if __name__ == "__main__":
    main()
