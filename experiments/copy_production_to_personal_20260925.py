"""Copy the production arrays between Modal workspaces; no local dataset payload."""

import json
from pathlib import Path

import modal

DATASET = "outputs/paper_dataset_full_equal_20260916/arrays"
VOLUME = "dno-fno-train-data"


def copy_dataset() -> dict:
    import os
    from time import perf_counter

    client = modal.Client.from_credentials(
        os.environ["DESTINATION_TOKEN_ID"], os.environ["DESTINATION_TOKEN_SECRET"],
    )
    destination = modal.Volume.from_name(VOLUME, environment_name="main", client=client)
    source = Path("/source") / DATASET
    files = sorted((p for p in source.rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    started = perf_counter()
    result = {"source_workspace": "sciml-at-ud", "destination_workspace": "zhaleon",
              "volume": VOLUME, "dataset": DATASET, "files": [],
              "verification": "Modal SDK SHA256/MD5 upload checks; destination file sizes checked"}
    print(f"Copying {len(files)} files, {sum(p.stat().st_size for p in files)} bytes", flush=True)
    for path in files:
        remote_path = f"/{DATASET}/{path.relative_to(source).as_posix()}"
        size = path.stat().st_size
        tick = perf_counter()
        print(f"Starting {path.name}: {size} bytes", flush=True)
        # The local path here is inside the Modal CPU container, not the laptop.
        # The SDK hashes and streams multipart uploads without reading the file into RAM.
        with destination.batch_upload(force=False) as batch:
            batch.put_file(path, remote_path)
        entries = destination.listdir(remote_path)
        assert len(entries) == 1 and entries[0].size == size, remote_path
        record = {"file": path.relative_to(source).as_posix(), "bytes": size,
                  "seconds": perf_counter() - tick}
        result["files"].append(record)
        print(json.dumps(record), flush=True)
    result["seconds"] = perf_counter() - started
    result["total_bytes"] = sum(row["bytes"] for row in result["files"])
    # Persist a completion record independently of the laptop's connection.
    manifest = Path("/tmp/transfer_manifest.json")
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    with destination.batch_upload(force=False) as batch:
        batch.put_file(manifest, f"/{DATASET}/transfer_manifest.json")
    return result


def main() -> None:
    import tomllib

    config = tomllib.loads((Path.home() / ".modal.toml").read_text())
    source_config, destination_config = config["sciml-at-ud"], config["zhaleon"]
    source_client = modal.Client.from_credentials(source_config["token_id"], source_config["token_secret"])
    destination_client = modal.Client.from_credentials(destination_config["token_id"], destination_config["token_secret"])
    # Refuse to overwrite an existing destination volume or any existing dataset.
    modal.Volume.objects.create(VOLUME, version=1, environment_name="main", client=destination_client)
    source = modal.Volume.from_name(VOLUME, environment_name="main", client=source_client)
    secret = modal.Secret.from_dict({
        "DESTINATION_TOKEN_ID": destination_config["token_id"],
        "DESTINATION_TOKEN_SECRET": destination_config["token_secret"],
    })
    app = modal.App("dno-production-dataset-transfer")
    worker = app.function(
        image=modal.Image.debian_slim(python_version="3.12"),
        volumes={"/source": source.with_mount_options(read_only=True)},
        secrets=[secret], cpu=4, memory=8192, timeout=3600, retries=0,
    )(copy_dataset)
    with modal.enable_output(), app.run(client=source_client, detach=True):
        call = worker.spawn()
        print(f"Transfer call: {call.object_id}", flush=True)
        result = call.get()
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
