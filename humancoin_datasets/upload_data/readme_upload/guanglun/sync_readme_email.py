"""Sync only README.md from a reviewed email-change manifest; preserve repo settings."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,'/share/project/liuyou/humancoin/code/roboego-hand-vis')
from humancoin_datasets.upload_data.readme_upload.upload2hub import make_client, modelscope_readme
from humancoin_datasets.upload_data.readme_upload._common import get_token, describe_error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hub',choices=['hf','ms'],required=True)
    p.add_argument('--report-dir',type=Path,default=Path('/share/project/liuyou/humancoin/data/readme_email_update_20261123'))
    args=p.parse_args();hub={'hf':'huggingface','ms':'modelscope'}[args.hub]
    manifest=json.loads((args.report_dir/'report.json').read_text());rows=manifest['records']
    logpath=args.report_dir/(args.hub+'_progress.jsonl');previous={}
    if logpath.exists():
        for line in logpath.read_text().splitlines():
            r=json.loads(line);previous[r['repo_id']]=r
    token=get_token(hub);client=make_client(hub,token=token);results=[]
    with logpath.open('a',encoding='utf-8') as out:
        for row in rows:
            data=Path(row['path']).read_text(encoding='utf-8')
            if 'humancoin.team@gmail.com' not in data or 'humancoin@163.com' in data:
                raise ValueError('Unexpected contact in '+row['repo_id'])
            payload=(modelscope_readme(data) if args.hub=='ms' else data).encode('utf-8')
            digest=hashlib.sha256(payload).hexdigest();old=previous.get(row['repo_id'])
            result=dict(repo_id=row['repo_id'],sha256=digest,status='error')
            if old and old['status']=='done' and old['sha256']==digest:
                results.append(old);continue
            for attempt in range(3):
                try:
                    kwargs=dict(repo_id=row['repo_id'],repo_type='dataset',path_in_repo='README.md',
                                path_or_fileobj=payload,commit_message='Update HumanCOIN contact email')
                    if args.hub=='hf':kwargs['token']=token
                    client.upload_file(**kwargs)
                    result['status']='done';break
                except Exception as exc:
                    result['error']=describe_error(exc,token)
                    code=getattr(getattr(exc,'response',None),'status_code',None)
                    if code in (400,401,403,404) or attempt==2:break
                    time.sleep(2**attempt)
            results.append(result);out.write(json.dumps(result)+'\n');out.flush()
            print(json.dumps(dict(completed=len(results),total=len(rows),**result)),flush=True)
    summary=dict(hub=hub,total=len(rows),successful=sum(r['status']=='done' for r in results),
                 failed=[r for r in results if r['status']!='done'],only_file='README.md',settings_changed=False)
    (args.report_dir/(args.hub+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if summary['failed']:raise SystemExit(1)


if __name__=='__main__':main()
