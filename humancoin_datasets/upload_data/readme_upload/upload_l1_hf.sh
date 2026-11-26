#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

guanglun_only=false
mode=''
for arg in "$@"; do
    case "$arg" in
        --guanglun) guanglun_only=true ;;
        --dry-run|--skip-uploaded)
            if [[ -n "$mode" ]]; then
                printf 'Choose only one of --dry-run and --skip-uploaded\n' >&2
                exit 2
            fi
            mode="$arg" ;;
        *)
            printf 'Usage: bash %s [--guanglun] [--dry-run | --skip-uploaded]\n' "$0" >&2
            exit 2 ;;
    esac
done

data_root=/share/project/liuyou/humancoin/data
python_bin=${HUMANCOIN_PYTHON:-/share/project/liuyou/humancoin/envs/ego/bin/python}
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

roots=("$data_root/guanglun_depth/publish")
if ! "$guanglun_only"; then
    roots=("$data_root/jianzhi/wait_for_publish/publish" "${roots[@]}")
fi
folders=()
for root in "${roots[@]}"; do
    groups=("$root"/l1_*/)
    if (( ${#groups[@]} == 0 )); then
        printf 'No L1 dataset directories in %s\n' "$root" >&2
        exit 1
    fi
    folders+=("${groups[@]}")
done

if [[ "$mode" != --dry-run ]]; then
    : "${HF_TOKEN:?Export HF_TOKEN before running this script}"
    : "${https_proxy:?Export https_proxy before running this script}"
    : "${http_proxy:?Export http_proxy before running this script}"
fi

options=()
if [[ "$mode" == --skip-uploaded ]]; then
    options=(--resume-hf-log-dir "$HOME/humancoin-logs")
elif [[ "$mode" == --dry-run ]]; then
    options=(--dry-run)
fi

exec "$python_bin" -m humancoin_datasets.upload_data.readme_upload.upload2hub \
    --hub hf --namespace HumanCOIN --create-repo --skip-errors \
    --folders "${folders[@]}" "${options[@]}"
