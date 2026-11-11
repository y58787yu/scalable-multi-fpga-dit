"""Bind locally prepared role assets to the matching model and FPGA overlay."""
import hashlib
import json
from pathlib import Path


PAYLOAD_HASHES = {
    ("Cosmos-Predict2-2B-Text2Image", 1, 10): "2118e4a7147f75fba6489ee7ee68e8eff8fba53f5e8f6abe7f80a735ea4ab2d9",
    ("Cosmos-Predict2-2B-Text2Image", 2, 10): "6874905a4ea692720803e9d3e1a4b158ee5cb4f748f7d430058c40d282be44a0",
    ("Cosmos-Predict2-14B-Text2Image", 3, 10): "5696489d5a1828c7935a13e756c712132fd7c6093695f6921896654137700fff",
    ("Cosmos-Predict2-14B-Text2Image", 3, 5): "e7f3abd583d4edc4366498ece82b1081a8d1cd0bb488bd10f9a261837c327cbf",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hardware(root):
    checksums = json.loads((root / "hardware" / "sha256.json").read_text())
    for name in ("cosmos_net.bit", "cosmos_net.hwh", "cosmos_net.xclbin"):
        if file_hash(root / "hardware" / name) != checksums[name]:
            raise ValueError(f"Overlay file does not match the model: {name}")


def fixed_task(folder, profile, original):
    folder = Path(folder)
    binding = json.loads((folder / "release_binding.json").read_text())
    updates = original["execution"]["updates"]
    key = (profile["model"], profile["sequence_parallel"], updates)
    expected = {
        "kind": "COSMOS_PUBLIC_ASSETS_V1",
        "model": profile["model"],
        "groups": profile["sequence_parallel"],
        "updates": updates,
        "profile_sha256": profile["profile_sha256"],
        "source_binding_sha256": original["source_binding_sha256"],
        "input_manifest_sha256": original["source_binding"]["input_manifest_sha256"],
        "asset_manifest_sha256": file_hash(folder / "tp4_assets.json"),
        "reuse_manifest_sha256": file_hash(folder / "sp3_assets.json"),
        "packed_manifest_sha256": file_hash(folder / "integrated_assets.json"),
    }
    for field, value in expected.items():
        if binding.get(field) != value:
            raise ValueError(f"Asset manifest binding mismatch: {field}")
    packed = json.loads((folder / "integrated_assets.json").read_text())
    table = json.dumps(packed["files"], sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(table).hexdigest()
    if digest != PAYLOAD_HASHES.get(key) or digest != binding.get("payload_sha256"):
        raise ValueError("Packed weights and fixed conditioning do not match the selected model")
    # The normal loader still verifies every local payload and operand layout.
    return dict(profile["fixed_task"], asset_updates=updates, preset=original["preset_id"],
                asset_manifest_sha256=expected["asset_manifest_sha256"],
                input_manifest_sha256=expected["input_manifest_sha256"])
