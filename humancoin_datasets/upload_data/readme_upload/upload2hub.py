#!/usr/bin/env python3
"""Upload local HumanCOIN datasets to Hugging Face or ModelScope."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import errno
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import shutil
import tempfile
import time
from urllib.parse import urlsplit

if __package__:
    from ._common import describe_error, get_token, plan_files, repo_id_for
else:
    from _common import describe_error, get_token, plan_files, repo_id_for

ALIASES = {'hf': 'huggingface', 'ms': 'modelscope'}
RELEASE_FILES = ('README.md', 'portal.yaml', 'preview.jpg')


def collection_slug(value):
    if value.startswith('https://'):
        parsed = urlsplit(value)
        if parsed.hostname != 'huggingface.co' or not parsed.path.startswith('/collections/'):
            raise ValueError('Expected a Hugging Face collection URL')
        value = parsed.path.removeprefix('/collections/')
    value = value.strip('/')
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', value):
        raise ValueError('Invalid collection slug')
    return value


def resolve_collection(client, value, namespace, token):
    requested = collection_slug(value)
    if requested.split('/')[0].lower() != namespace.lower():
        raise ValueError('Collection and uploaded repositories must belong to the same namespace')
    try:
        collection = client.get_collection(requested, token=token)
    except Exception as exc:
        if getattr(getattr(exc, 'response', None), 'status_code', None) != 404:
            raise
        matches = [c for c in client.list_collections(owner=namespace, token=token)
                   if re.sub(r'-[0-9a-f]{24}$', '', c.slug).lower() == requested.lower()]
        if len(matches) != 1:
            raise ValueError('Unable to uniquely resolve collection URL') from None
        collection = matches[0]
    return collection.slug


def add_to_collection(client, collection, repo, token, max_retries):
    for attempt in range(max_retries):
        try:
            client.add_collection_item(collection_slug=collection, item_id=repo,
                                       item_type='dataset', exists_ok=True, token=token)
            print(f'Collection added: {repo} -> {collection}', flush=True)
            return True
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in (400, 401, 403, 404) or attempt + 1 == max_retries:
                print(f'Collection add failed: {repo}: {describe_error(exc, token)}', file=sys.stderr, flush=True)
                return False
            time.sleep(min(2 ** attempt, 8))


def completed_repos(log_dir, hub):
    prefix = 'hf' if hub == 'huggingface' else 'ms'
    success = re.compile(rf'^Uploaded: {hub} (\S+)$')
    completed = set()
    paths = set(Path(log_dir).glob(f'{prefix}-upload-*.log'))
    legacy_log = Path(log_dir) / f'{prefix}.log'
    if legacy_log.is_file():
        paths.add(legacy_log)
    for path in sorted(paths):
        with path.open(encoding='utf-8', errors='replace') as log:
            for line in log:
                match = success.fullmatch(line.strip())
                if match:
                    completed.add(match[1])
    return completed


def hf_release_matches(client, folder, repo, token):
    from huggingface_hub.hf_api import RepoFolder

    try:
        recordings = {path.name for path in folder.iterdir()
                      if path.is_dir() and (path / 'meta/readme.json').is_file()}
        if not recordings:
            return False
        entries = list(client.list_repo_tree(repo, repo_type='dataset', revision='main', token=token))
        remote_recordings = {entry.path for entry in entries if isinstance(entry, RepoFolder)}
        if recordings != remote_recordings:
            return False
        remote_files = {entry.path: entry for entry in entries if not isinstance(entry, RepoFolder)}
        for name in ('README.md', 'portal.yaml'):
            entry = remote_files.get(name)
            if entry is None:
                return False
            data = (folder / name).read_bytes()
            if entry.lfs:
                if entry.lfs.sha256 != hashlib.sha256(data).hexdigest():
                    return False
            elif entry.blob_id != hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest():
                return False
        return True
    except Exception as exc:
        print(f'Resume check unavailable for {repo} ({type(exc).__name__}); keeping in queue',
              file=sys.stderr, flush=True)
        return False


def modelscope_release_matches(client, folder, repo):
    try:
        recordings = {path.name for path in folder.iterdir()
                      if path.is_dir() and (path / 'meta/readme.json').is_file()}
        if not recordings:
            return False
        legacy = client.legacy
        entries = {}
        page_size = 100
        for page in range(1, 1001):
            params = {'Revision': 'master', 'Recursive': 'False',
                      'PageSize': page_size, 'PageNumber': page}
            response = legacy._request_repo_tree(f'datasets/{repo}/repo/tree', params,
                                                 repo_id=repo, authorized=legacy._tree_reads_ok)
            data = legacy._json_data(response)
            rows = data if isinstance(data, list) else data.get('Files', data.get('files', []))
            previous_count = len(entries)
            entries.update((row['Path'], row) for row in rows)
            if len(entries) != previous_count + len(rows):
                return False
            if len(rows) < page_size:
                break
        else:
            return False
        if recordings != {path for path, row in entries.items() if row['Type'] == 'tree'}:
            return False
        for name in RELEASE_FILES:
            if name == 'preview.jpg' and not (folder / name).is_file():
                continue
            data = (folder / name).read_bytes()
            if name == 'README.md':
                data = modelscope_readme(data.decode('utf-8')).encode('utf-8')
            if entries.get(name, {}).get('Sha256') != hashlib.sha256(data).hexdigest():
                return False
        return True
    except Exception as exc:
        print(f'Resume check unavailable for {repo} ({type(exc).__name__}); keeping in queue',
              file=sys.stderr, flush=True)
        return False


def make_client(hub: str, token=None):
    try:
        if hub == 'huggingface':
            from huggingface_hub import HfApi
            return HfApi()
        if hub == 'modelscope':
            from modelscope_hub import HubApi
            return HubApi(token=token)
    except ImportError as exc:
        raise RuntimeError('Install dependencies with: pip install -r humancoin_datasets/upload_data/readme_upload/requirements.txt') from exc
    raise ValueError(f'Unsupported hub: {hub}')


def modelscope_readme(content):
    """Remove only HF gating keys from YAML front matter; preserve body verbatim."""
    match = re.match(r'\A(\ufeff?)---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)', content, re.S)
    if not match:
        return content
    import yaml
    metadata = yaml.safe_load(match[2])
    if not isinstance(metadata, dict):
        raise ValueError('README YAML front matter must be a mapping')
    keys = [key for key in metadata if isinstance(key, str)
            and (key == 'gated' or key.startswith('extra_gated_'))]
    if not keys:
        return content
    for key in keys:
        del metadata[key]
    header = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).rstrip()
    return match[1] + '---\n' + header + '\n---\n' + content[match.end():]


@contextmanager
def platform_folder(folder, files, hub):
    folder = Path(folder)
    if hub != 'modelscope' or 'README.md' not in files:
        yield folder
        return
    content = (folder / 'README.md').read_bytes().decode('utf-8')
    adapted = modelscope_readme(content)
    if content == adapted:
        yield folder
        return
    # Stage beside the source so large binary files can normally be hard-linked.
    # README is written separately and never hard-linked or modified in place.
    with tempfile.TemporaryDirectory(prefix='.humancoin-ms-upload-', dir=folder.parent) as directory:
        stage = Path(directory)
        for relative in files:
            source = folder / relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or source.is_symlink():
                raise ValueError('Invalid staged upload path')
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == 'README.md':
                target.write_bytes(adapted.encode('utf-8'))
            else:
                try:
                    os.link(source, target)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    shutil.copy2(source, target)
        yield stage


def upload_folder(client, *, hub, folder, repo_id, files, token=None,
                  create_repo=False, private=False, message='Upload HumanCOIN dataset',
                  max_workers=8, max_retries=3):
    with platform_folder(folder, files, hub) as prepared:
        return _upload_folder(client, hub=hub, folder=prepared, repo_id=repo_id,
                              files=files, token=token, create_repo=create_repo,
                              private=private, message=message, max_workers=max_workers,
                              max_retries=max_retries)


def _upload_folder(client, *, hub, folder, repo_id, files, token=None,
                  create_repo=False, private=False, message='Upload HumanCOIN dataset',
                  max_workers=8, max_retries=3):
    """Upload selected files without deleting remote-only files.

    Authentication and permission errors propagate; failed requests are never
    reported as successful. Creating a repository is explicitly opt-in.
    """
    if not files:
        raise ValueError('No files selected')
    if max_workers < 1 or max_retries < 1:
        raise ValueError('Worker and attempt counts must be positive')
    if hub == 'modelscope' and private:
        raise ValueError('--private is only supported for Hugging Face')
    if create_repo:
        # Do not turn permission/network errors into a missing-repository result.
        exists = client.repo_exists(repo_id=repo_id, repo_type='dataset', **({'token': token} if hub == 'huggingface' else {}))
        if not exists:
            kwargs = dict(repo_id=repo_id, repo_type='dataset', token=token)
            if hub == 'huggingface':
                kwargs.update(private=private, exist_ok=True)
            else:
                kwargs.pop('token')
                kwargs.update(visibility='public', license='cc-by-nc-4.0')
            client.create_repo(**kwargs)
    if hub == 'huggingface':
        client.update_repo_settings(repo_id=repo_id, repo_type='dataset', token=token, gated='auto')
    elif hub == 'modelscope':
        client.update_repo_settings(repo_id=repo_id, repo_type='dataset',
                                    visibility='public', license='cc-by-nc-4.0')
    if hub == 'huggingface':
        from huggingface_hub import CommitOperationAdd
        # The explicit allow-list is already computed by plan_files. Sending N
        # exact patterns through upload_folder triggers O(N^2) fnmatch checks.
        # Bounded commits also avoid one huge final commit for a large dataset.
        batch_size = 500
        total = (len(files) + batch_size - 1) // batch_size
        result = None
        for offset in range(0, len(files), batch_size):
            batch = files[offset:offset + batch_size]
            number = offset // batch_size + 1
            print(f'HF batch {number}/{total}: preparing {len(batch)} files for {repo_id}', flush=True)
            for attempt in range(max_retries):
                try:
                    # SDK mutates operations during preupload; recreate on retry.
                    operations = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=folder / p) for p in batch]
                    result = client.create_commit(repo_id=repo_id, repo_type='dataset', token=token,
                        operations=operations, commit_message=f'{message} (batch {number}/{total})')
                    break
                except Exception:
                    if attempt + 1 == max_retries:
                        raise
                    time.sleep(min(2 ** attempt, 8))
            print(f'HF batch {number}/{total}: committed for {repo_id}', flush=True)
        return result
    kwargs = dict(repo_id=repo_id, repo_type='dataset', folder_path=str(folder),
                  token=token, commit_message=message,
                  allow_patterns=[glob.escape(p) for p in files])
    # HF upload_folder manages its own concurrency; no worker argument.
    if hub == 'modelscope':
        kwargs.pop('token')  # New SDK authenticates the client, not each operation.
        kwargs['max_workers'] = max_workers
    for attempt in range(max_retries):
        try:
            result = client.upload_folder(**kwargs)
        except Exception:
            if attempt + 1 == max_retries:
                raise
            time.sleep(min(2 ** attempt, 8))
        else:
            return result


def parser_with_config(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args(argv)
    config = {}
    if known.config:
        import yaml
        with known.config.open() as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            raise ValueError('Config must be a YAML mapping')
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument('--hub', choices=['huggingface', 'hf', 'modelscope', 'ms'], default='huggingface')
    parser.add_argument('--folders', nargs='+', help='Dataset directories; one repository per directory')
    parser.add_argument('--namespace', default='HumanCOIN')
    parser.add_argument('--collection', help='HF collection URL or slug; add after upload, including resume-skipped repositories')
    parser.add_argument('--repo-id', help='Explicit owner/name (one folder only)')
    parser.add_argument('--token', help='Prefer HF_TOKEN or MODELSCOPE_TOKEN environment variables')
    parser.add_argument('--create-repo', action='store_true', help='Create repository if missing')
    parser.add_argument('--private', action='store_true', help='Create new repository as private')
    parser.add_argument('--readme-only', action='store_true')
    parser.add_argument('--allow', nargs='+', help='Include relative-path glob patterns')
    parser.add_argument('--ignore', nargs='+', help='Additional relative-path glob patterns')
    parser.add_argument('--message', default='Upload HumanCOIN dataset')
    parser.add_argument('--max-workers', type=int, default=8, help='ModelScope upload workers; HF uses SDK defaults')
    parser.add_argument('--max-retries', type=int, default=3, help='Total upload attempts')
    parser.add_argument('--skip-errors', action='store_true', help='Continue after a dataset fails')
    parser.add_argument('--dry-run', action='store_true', help='Print file plan without network or authentication')
    parser.add_argument('--resume-hf-log-dir', type=Path, help='Skip logged HF releases that still match the Hub')
    parser.add_argument('--resume-ms-log-dir', type=Path, help='Skip logged ModelScope releases that still match the Hub')
    keys = {a.dest for a in parser._actions} - {'help', 'config'}
    if set(config) - keys:
        raise ValueError(f'Unknown configuration keys: {sorted(set(config) - keys)}')
    if 'token' in config:
        raise ValueError('Use an environment variable for tokens, not YAML')
    # YAML has typed values, unlike argparse string defaults; validate explicitly.
    for key in ('create_repo', 'private', 'readme_only', 'skip_errors', 'dry_run'):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f'{key} must be a YAML boolean')
    for key in ('folders', 'allow', 'ignore'):
        if key in config and (not isinstance(config[key], list) or not all(isinstance(v, str) for v in config[key])):
            raise ValueError(f'{key} must be a list of strings')
    for key in ('max_workers', 'max_retries'):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError(f'{key} must be a positive integer')
    for key in ('hub', 'namespace', 'repo_id', 'message', 'collection'):
        if key in config and not isinstance(config[key], str):
            raise ValueError(f'{key} must be a string')
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if not args.folders:
        parser.error('Provide --folders or folders in YAML')
    if args.repo_id and len(args.folders) != 1:
        parser.error('--repo-id requires exactly one folder')
    if args.private and not args.create_repo:
        parser.error('--private requires --create-repo; it does not change existing visibility')
    if args.max_workers < 1 or args.max_retries < 1:
        parser.error('Worker and attempt counts must be positive')
    args.hub = ALIASES.get(args.hub, args.hub)
    if args.hub not in ('huggingface', 'modelscope'):
        parser.error('Unsupported hub')
    if args.hub == 'modelscope' and args.private:
        parser.error('--private is only supported for Hugging Face')
    if args.collection:
        if args.hub != 'huggingface':
            parser.error('--collection is only supported for Hugging Face')
        args.collection = collection_slug(args.collection)
        if args.collection.split('/')[0].lower() != args.namespace.lower():
            parser.error('Collection owner must match --namespace')
    if args.resume_hf_log_dir and (args.hub != 'huggingface' or args.dry_run or args.readme_only or args.allow or args.ignore):
        parser.error('--resume-hf-log-dir requires a full Hugging Face upload')
    if args.resume_ms_log_dir and (args.hub != 'modelscope' or args.dry_run or args.readme_only or args.allow or args.ignore):
        parser.error('--resume-ms-log-dir requires a full ModelScope upload')
    return args


def main(argv=None):
    try:
        args = parser_with_config(argv)
        plans = []
        repos = set()
        for name in args.folders:
            folder = Path(name).expanduser().resolve()
            repo = repo_id_for(folder, args.namespace, args.repo_id)
            if repo in repos:
                raise ValueError(f'Multiple folders target the same repository: {repo}')
            repos.add(repo)
            plans.append((folder, repo))
        token = None if args.dry_run else get_token(args.hub, args.token)
        client = None if args.dry_run else make_client(args.hub, token=token)
        collection = (resolve_collection(client, args.collection, args.namespace, token)
                      if args.collection and not args.dry_run else args.collection)
        failures = collection_failures = 0
        resume_dir = args.resume_hf_log_dir or args.resume_ms_log_dir
        if resume_dir:
            completed = completed_repos(resume_dir, args.hub)
            candidates = [(folder, repo) for folder, repo in plans if repo in completed]
            def check(plan):
                if args.hub == 'huggingface':
                    return hf_release_matches(client, *plan, token)
                return modelscope_release_matches(make_client(args.hub, token), *plan)
            with ThreadPoolExecutor(max_workers=4 if args.hub == 'huggingface' else 2) as pool:
                matches = pool.map(check, candidates)
                skipped = {repo for (_, repo), match in zip(candidates, matches) if match}
            plans = [(folder, repo) for folder, repo in plans if repo not in skipped]
            print(f'Resume: skipped {len(skipped)} matching releases; {len(plans)} queued', flush=True)
            if collection:
                for repo in sorted(skipped):
                    collection_failures += not add_to_collection(client, collection, repo, token, args.max_retries)
        for folder, repo in plans:
            try:
                print(f'Preparing: {args.hub} {repo}', file=sys.stderr, flush=True)
                files = plan_files(folder, args.readme_only, args.allow, args.ignore)
                if args.resume_hf_log_dir and 'preview.jpg' in files:
                    if client.repo_exists(repo_id=repo, repo_type='dataset', token=token) and client.file_exists(
                        repo_id=repo, filename='preview.jpg', repo_type='dataset', revision='main', token=token
                    ):
                        files.remove('preview.jpg')
                summary = dict(hub=args.hub, folder=str(folder), repo_id=repo,
                               files=len(files), create_repo=args.create_repo,
                               new_repo_private=args.private,
                               gated=False if args.hub == 'modelscope' else 'auto',
                               visibility='private' if args.private else 'public')
                if collection:
                    summary['collection'] = collection
                if args.dry_run:
                    summary['bytes'] = sum((folder / path).stat().st_size for path in files)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                if args.dry_run:
                    for path in files:
                        print(f'  {path}')
                    continue
                upload_folder(client, hub=args.hub, folder=folder, repo_id=repo,
                              files=files, token=token, create_repo=args.create_repo,
                              private=args.private, message=args.message,
                              max_workers=args.max_workers, max_retries=args.max_retries)
                print(f'Uploaded: {args.hub} {repo}', flush=True)
                if collection:
                    collection_failures += not add_to_collection(client, collection, repo, token, args.max_retries)
            except Exception as exc:
                print(f'Upload failed: {repo}: {describe_error(exc, token)}', file=sys.stderr)
                print('Check credentials and organization write permission for HTTP 401/403; check the reported validation details for HTTP 400/422.', file=sys.stderr)
                failures += 1
                if not args.skip_errors:
                    break
        return int(bool(failures or collection_failures))
    except Exception as exc:
        print(f'Error ({type(exc).__name__}): {exc}' if isinstance(exc, (ValueError, FileNotFoundError, RuntimeError)) else f'Error ({type(exc).__name__})', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
