"""Patch request fields in all organization dataset cards, preserving body and settings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import yaml
from huggingface_hub import HfApi, hf_hub_download

FIELDS=('Name','Email','Institution or organization','Intended use')


def patch_card(raw):
    text=raw.decode('utf-8-sig')
    match=re.match(r'\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)',text,re.S)
    if match:
        metadata=yaml.safe_load(match.group(1)) or {}
        body=text[match.end():]
    else:
        metadata={};body=text
    if not isinstance(metadata,dict):raise ValueError('Card metadata must be a mapping')
    existing=metadata.get('extra_gated_fields') or {}
    if not isinstance(existing,dict):raise ValueError('extra_gated_fields must be a mapping')
    if all(existing.get(k)=='text' for k in FIELDS):return raw
    fields=dict(existing)
    for name in FIELDS:fields[name]='text'
    metadata['extra_gated_fields']=fields
    metadata['extra_gated_prompt']='Please provide your name, email address, institution or organization, and intended use to request access to HumanCOIN datasets.'
    header=yaml.safe_dump(metadata,allow_unicode=True,sort_keys=False).rstrip('\n')
    result=('---\n'+header+'\n---\n'+body).encode('utf-8')
    check=re.match(r'\A---\n(.*?)\n---\n',result.decode('utf-8'),re.S)
    assert result.decode('utf-8')[check.end():]==body
    parsed=yaml.safe_load(check.group(1));assert all(parsed['extra_gated_fields'][k]=='text' for k in FIELDS)
    # Only request-form metadata is allowed to change.
    for key,value in metadata.items():
        if key not in ('extra_gated_fields','extra_gated_prompt'):assert parsed[key]==value
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',default='HumanCOIN')
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args();args.report_dir.mkdir(parents=True,exist_ok=True)
    api=HfApi();datasets=sorted({d.id for d in api.list_datasets(author=args.owner)})
    if any(d.split('/')[0].lower()!=args.owner.lower() for d in datasets):raise ValueError('Unexpected owner')
    (args.report_dir/'datasets.json').write_text(json.dumps(datasets,indent=2)+'\n')
    print(json.dumps(dict(datasets=len(datasets),apply=args.apply)),flush=True)
    records=[]
    with (args.report_dir/'progress.jsonl').open('a',encoding='utf-8') as log:
        for repo in datasets:
            result=dict(repo_id=repo,status='error')
            for attempt in range(3):
                try:
                    info=api.dataset_info(repo)
                    path=hf_hub_download(repo_id=repo,repo_type='dataset',filename='README.md',revision=info.sha)
                    before=Path(path).read_bytes();after=patch_card(before)
                    result.update(before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest())
                    if after==before:result['status']='already_current'
                    elif not args.apply:result['status']='needs_update'
                    else:
                        # A concurrent upload must not be overwritten using a stale card.
                        api.upload_file(repo_id=repo,repo_type='dataset',path_in_repo='README.md',
                            path_or_fileobj=after,parent_commit=info.sha,commit_message='Standardize HumanCOIN access request fields')
                        current=api.dataset_info(repo)
                        verify=Path(hf_hub_download(repo_id=repo,repo_type='dataset',filename='README.md',revision=current.sha)).read_bytes()
                        if patch_card(verify)!=verify:raise ValueError('Remote field verification failed')
                        result['status']='updated'
                    break
                except Exception as exc:
                    code=getattr(getattr(exc,'response',None),'status_code',None)
                    result.update(status='error',error_type=type(exc).__name__,http_status=code)
                    if code in (401,403,404) or attempt==2:break
                    time.sleep(2**attempt)
            records.append(result);log.write(json.dumps(result)+'\n');log.flush()
            print(json.dumps(dict(completed=len(records),total=len(datasets),**result)),flush=True)
            if result['status']=='error' and result.get('http_status')==401:break
            time.sleep(.2)
    from collections import Counter
    summary=dict(datasets=len(datasets),processed=len(records),status=dict(Counter(r['status'] for r in records)),
                 failures=[r for r in records if r['status']=='error'],settings_changed=False,apply=args.apply)
    (args.report_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if summary['failures'] or len(records)!=len(datasets):raise SystemExit(1)


if __name__=='__main__':main()
