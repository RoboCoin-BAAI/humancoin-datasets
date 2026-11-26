"""Delete exactly five user-approved README-only MS repositories, never local data."""
import json
import os
from pathlib import Path
from modelscope_hub import HubApi

REPOS=['HumanCOIN/l1_retail_store_'+v+'_task_1600x1300_30fps_2views'
       for v in ('peel','pour','processing','service','store')]


def contents(api,repo):
    response=api.legacy._request_repo_tree('datasets/'+repo+'/repo/tree',
        {'Revision':'master','Recursive':'True','PageSize':100,'PageNumber':1},
        repo_id=repo,authorized=api.legacy._tree_reads_ok)
    d=api.legacy._json_data(response)
    return d if isinstance(d,list) else d.get('Files',[])


def main():
    token=next((os.environ[k] for k in ('MODELSCOPE_TOKEN','MS_TOKEN','MODELSCOPE_API_TOKEN') if os.environ.get(k)),None)
    if not token:raise ValueError('Set MS_TOKEN or MODELSCOPE_TOKEN in this terminal')
    api=HubApi(token=token);report=Path('/share/project/liuyou/humancoin/data/ms_empty_five_deleted_20261124.jsonl')
    # Preflight the entire scope before any deletion. OpenAPI existence checks omit these repos.
    selected=[]
    for repo in REPOS:
        try:rows=contents(api,repo)
        except Exception as exc:
            if getattr(getattr(exc,'response',None),'status_code',None)==404:continue
            raise
        if len(rows)>=100 or any(x.get('Type')!='blob' or x.get('Path') not in {'README.md','.gitattributes'} for x in rows):
            raise ValueError('Repository now contains data; refusing deletion: '+repo)
        selected.append(repo)
    with report.open('a') as out:
        for repo in selected:
            rows=contents(api,repo)
            if any(x.get('Type')!='blob' or x.get('Path') not in {'README.md','.gitattributes'} for x in rows):
                raise ValueError('Repository changed; refusing deletion: '+repo)
            api.delete_repo(repo_id=repo,repo_type='dataset')
            row=dict(repo_id=repo,status='delete_succeeded')
            out.write(json.dumps(row)+'\n');out.flush();print(json.dumps(row),flush=True)


if __name__=='__main__':main()
