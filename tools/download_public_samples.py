"""Download recorded official sources and verify bytes against the saved manifest."""
from pathlib import Path
import hashlib
import json
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "test_data/public_sources"
    records = json.loads((destination / "manifest.json").read_text())
    for record in records:
        if "sha256" not in record:
            continue
        path = destination / record["file"]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            with urlopen(record["url"], timeout=45) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            temporary.rename(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        print("VERIFIED", path.name, flush=True)


if __name__ == "__main__":
    main()
