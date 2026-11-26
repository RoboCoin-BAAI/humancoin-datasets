"""Plan Jingdong instance directories into 100-instance download CSVs."""

import argparse
from pathlib import Path
import re
import subprocess

from humancoin_datasets.download_data.jianzhi.inventory import write_batches


SOURCE = "ks3://baai-eai-pwm/baai-eai-pwm-humancoin-waibu/jingdong/202611/"


def discover(source, tool, config):
    source = source.rstrip("/") + "/"
    result = subprocess.run(
        [str(tool), "-c", str(config), "ls", source, "-d", "-L", "EN",
         "--loglevel=off"],
        capture_output=True, encoding="utf-8", errors="strict", check=True)
    directories = set()
    for line in result.stdout.splitlines():
        uri = line.strip()
        if uri.startswith(source) and uri.endswith("/"):
            name = uri[len(source):-1]
            if re.fullmatch(r"instance-[A-Za-z0-9]+", name):
                directories.add(uri)
    if not directories:
        raise ValueError("No immediate instance-* directories found")
    return sorted(directories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--ks3-tool", type=Path,
                        default=Path("/opt/ks3/ks3util-2.18"))
    parser.add_argument("--ks3-config", type=Path,
                        default=Path("/opt/ks3/ks3util-config"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if list(args.output_dir.glob("batch_*.csv")) or (args.output_dir / "inventory.json").exists():
        parser.error("Output already contains an inventory; use a new directory")
    sessions = discover(args.source, args.ks3_tool, args.ks3_config)
    batches = write_batches(args.output_dir, sessions, 100,
                            {"source": args.source.rstrip("/") + "/",
                             "unit": "instance directory"})
    print(f"Saved {len(sessions)} instances in {len(batches)} batches -> {args.output_dir}")


if __name__ == "__main__":
    main()
