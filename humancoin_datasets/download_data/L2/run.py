"""Download L2 batches into L2_lerobot/<batch_name>/<session_name>.

Run without arguments to process every batch CSV in the server directory.
Use --dry-run to preview, or pass --batch-csv and --output-dir to download
one batch into the specified directory.
"""

from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humancoin_datasets.download_data.batch import download_batches


def is_l2_session(source: str, _name: str) -> bool:
    path = urlsplit(source).path
    return re.fullmatch(
        r'/data/EGOL2/cp_\d+/(?P<date>\d{4}-\d{2}-\d{2})/'
        r'L2-\d+-(?P=date)-\d{2}-\d{2}-\d{2}/',
        path,
    ) is not None


def main(argv: list[str] | None = None) -> int:
    return download_batches(
        argv,
        dataset='L2',
        session_is_valid=is_l2_session,
        description=__doc__,
    )


if __name__ == '__main__':
    raise SystemExit(main())
