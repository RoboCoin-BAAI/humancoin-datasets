"""Add organization datasets to an existing HF collection; never change gating settings."""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
from huggingface_hub import HfApi


def slug(value):
    if value.startswith('https://'):
        parsed=urlsplit(value)
        if parsed.hostname!='huggingface.co' or not parsed.path.startswith('/collections/'):
            raise ValueError('Expected a huggingface.co/collections URL')
        value=parsed.path.removeprefix('/collections/')
    value=value.strip('/')
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+',value):
        raise ValueError('Invalid collection slug')
    return value


def resolve(api, requested, owner):
    try:
        collection=api.get_collection(requested)
    except Exception as exc:
        if getattr(getattr(exc,'response',None),'status_code',None)!=404:
            raise
        # The UI may expose a friendly URL while older SDKs expect the ID suffix.
        matches=[c for c in api.list_collections(owner=owner)
                 if re.sub(r'-[0-9a-f]{24}$','',c.slug).lower()==requested.lower()]
        if len(matches)!=1:
            raise ValueError('Could not uniquely resolve the supplied collection URL') from None
        collection=api.get_collection(matches[0].slug)
    if collection.slug.split('/')[0].lower()!=owner.lower():
        raise ValueError('Collection must belong to the selected organization')
    return collection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',default='https://huggingface.co/collections/HumanCOIN/humancoin-all-datasets')
    p.add_argument('--owner',default='HumanCOIN')
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--apply',action='store_true',help='Without this option only list the plan')
    args=p.parse_args()
    api=HfApi()
    collection=resolve(api,slug(args.collection),args.owner)
    datasets=sorted({d.id for d in api.list_datasets(author=args.owner)})
    if any(d.split('/')[0].lower()!=args.owner.lower() for d in datasets):
        raise ValueError('Unexpected dataset owner')
    present={i.item_id for i in collection.items if i.item_type=='dataset'}
    pending=[d for d in datasets if d not in present]
    args.report_dir.mkdir(parents=True,exist_ok=True)
    plan=dict(collection=collection.slug,total=len(datasets),already_present=len(set(datasets)&present),pending=pending,apply=args.apply)
    (args.report_dir/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:v for k,v in plan.items() if k!='pending'}),flush=True)
    if not args.apply:return
    results=[]
    with (args.report_dir/'progress.jsonl').open('a') as out:
        for item in pending:
            row=dict(repo_id=item,status='error')
            for attempt in range(3):
                try:
                    api.add_collection_item(collection_slug=collection.slug,item_id=item,item_type='dataset',exists_ok=True)
                    row['status']='done';break
                except Exception as exc:
                    code=getattr(getattr(exc,'response',None),'status_code',None)
                    row.update(error_type=type(exc).__name__,http_status=code)
                    if code in (400,401,403,404) or attempt==2:break
                    time.sleep(2**attempt*2)
            results.append(row);out.write(json.dumps(row)+'\n');out.flush()
            print(json.dumps(dict(completed=len(results),total=len(pending),**row)),flush=True)
            if row.get('http_status') in (401,403) and row['status']=='error':break
            time.sleep(.25)
    current=api.get_collection(collection.slug)
    actual={i.item_id for i in current.items if i.item_type=='dataset'}
    missing=sorted(set(datasets)-actual)
    summary=dict(collection=collection.slug,total=len(datasets),present=len(set(datasets)&actual),
                 missing=missing,errors=sum(r['status']=='error' for r in results),gating_settings_changed=False)
    (args.report_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if missing:raise SystemExit(1)


if __name__=='__main__':main()
