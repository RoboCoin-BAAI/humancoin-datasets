"""Download all L3 batches into output_dir/batch_name/session_name.

Run with no arguments on the server to download every L3 batch. Use
--batch-csv selects one batch and uses output_dir/session_name directly,
matching the L1 downloader. Use --dry-run to preview commands.
"""

from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humancoin_datasets.download_data.batch import download_batches


def main(argv: list[str] | None = None) -> int:
    return download_batches(
        argv,
        dataset='L3',
        session_is_valid=lambda source, name: name.startswith(('L3-001-', 'L3-002-', 'L3-003-')),
        description=__doc__,
    )


if __name__ == '__main__':
    raise SystemExit(main())

'''
Download all L3 batches (output: L3_lerobot/<batch_name>/<session_name>/):
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/humancoin_datasets/download_data/L3/run.py

Download a single batch (output: L3_lerobot/batch_0000001/<session_name>/):
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/humancoin_datasets/download_data/L3/run.py \
  --batch-csv /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001.csv \
  --output-dir /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001/

Preview all download commands without downloading:
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/roboego-hand-vis/humancoin_datasets/download_data/L3/run.py \
  --dry-run
'''

# scp "C:\Users\22563\Desktop\data\L3-lerobot\batch_0000001.csv" "C:\Users\22563\Desktop\data\L3-lerobot\batch_0000003.csv" baai-pwm2-job:/share/project/liuyou/humancoin/data/L3_lerobot/
# scp "C:\Users\22563\Desktop\data\L3-lerobot\*.csv" baai-pwm2-job:/share/project/liuyou/humancoin/data/L3_lerobot/

