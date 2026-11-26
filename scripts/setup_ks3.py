"""Copy KS3 executables and connection config into the container filesystem."""

import argparse
import os
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("/share/project/liuyou/humancoin/utils/ks3"))
    parser.add_argument("--destination", type=Path, default=Path("/opt/ks3"))
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    names = ("ks3util-linux-amd64", "ks3util-2.18", "ks3util-config")
    if source == destination:
        parser.error("Source and destination must differ")
    for name in names:
        if not (source / name).is_file():
            parser.error(f"Missing source file: {source / name}")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in names:
        target = destination / name
        if target.exists():
            if (source / name).read_bytes() != target.read_bytes():
                parser.error(f"Destination differs from source: {target}")
        else:
            descriptor, temporary = tempfile.mkstemp(dir=destination)
            os.close(descriptor)
            try:
                shutil.copyfile(source / name, temporary)
                os.chmod(temporary, 0o600 if name == "ks3util-config" else 0o755)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        target.chmod(0o600 if name == "ks3util-config" else 0o755)
        print(f"Ready: {target}")


if __name__ == "__main__":
    main()
