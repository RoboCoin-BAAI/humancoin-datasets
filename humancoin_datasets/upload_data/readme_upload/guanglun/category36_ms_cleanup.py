"""Verify all replacement ModelScope repositories before optionally deleting old IDs."""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
BASE=Path('/share/project/liuyou/humancoin/data')
REPORT=BASE/'category36_ms_verify_delete_20261122'
sys.path.insert(0,'/share/project/liuyou/humancoin/code/roboego-hand-vis')
from humancoin_datasets.upload_data.readme_upload._common import get_token
from humancoin_datasets.upload_data.readme_upload.upload2hub import modelscope_readme
from modelscope_hub import HubApi

def tree(api,repo):
    result={};legacy=api.legacy
    for page in range(1,1001):
        response=legacy._request_repo_tree('datasets/'+repo+'/repo/tree',{'Revision':'master','Recursive':'True','PageSize':100,'PageNumber':page},repo_id=repo,authorized=legacy._tree_reads_ok)
        data=legacy._json_data(response);rows=data if isinstance(data,list) else data.get('Files',[])
        for row in rows:
            if row['Type']=='blob':
                if row['Path'] in result:raise ValueError('Repeated tree page')
                result[row['Path']]=row
        if len(rows)<100:return result
    raise ValueError('Tree pagination limit')

def verify(job):
    api=HubApi(token=get_token('modelscope'));repo='HumanCOIN/'+job['target'];folder=Path(job['folder'])
    result=dict(repo_id=repo,status='verified',old_repositories=job['sources'])
    try:
        remote=tree(api,repo)
        local={p.relative_to(folder).as_posix():p for p in folder.rglob('*') if p.is_file() and '.cache' not in p.relative_to(folder).parts}
        missing=sorted(set(local)-set(remote))
        if missing:raise ValueError('Missing target files: '+str(missing[:3]))
        expected={p.name for p in folder.iterdir() if p.is_dir() and (p/'meta/info.json').is_file()}
        found={name.split('/')[0] for name in remote if name.endswith('/meta/info.json')}
        if expected!=found:raise ValueError('Sample directories differ')
        for name,path in local.items():
            entry=remote[name]
            if name=='README.md':
                data=modelscope_readme(path.read_text()).encode('utf-8')
                if entry.get('Sha256')!=hashlib.sha256(data).hexdigest():raise ValueError('README hash differs')
            elif entry.get('Size')!=path.stat().st_size:raise ValueError('Target file size differs: '+name)
            elif name.endswith(('.json','.jsonl','.yaml','.yml')):
                if entry.get('Sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Metadata hash differs: '+name)
        fingerprints={}
        for old in job['sources']:
            files=tree(api,old);old_ids={name.split('/')[0] for name in files if name.endswith('/meta/info.json')}
            fingerprints[old]=manifest_hash(files)
            if not old_ids or not old_ids<=expected:raise ValueError('Old samples not covered: '+old)
            for name,entry in files.items():
                if name.split('/')[0] not in old_ids:continue
                if name not in remote:raise ValueError('Old file absent: '+name)
                if name.endswith(('/meta/readme.json','/annotation/episode_000000_structured.json')):continue
                newer=remote[name]
                if not entry.get('Sha256') or entry.get('Sha256')!=newer.get('Sha256') or entry.get('Size')!=newer.get('Size'):
                    raise ValueError('Unchanged sample file differs: '+name)
        result.update(samples=len(expected),files=len(local),old_fingerprints=fingerprints,target_fingerprint=manifest_hash(remote))
    except Exception as exc:
        result.update(status='error',error_type=type(exc).__name__,message=str(exc) if isinstance(exc,ValueError) else '')
    return result

def manifest_hash(files):
    payload=sorted((name,row.get('Size'),row.get('Sha256')) for name,row in files.items())
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--delete',action='store_true');args=parser.parse_args()
    plan=json.loads((BASE/'category_36_rename_merge_plan_20261121/plan.json').read_text())
    roots={'jianzhi':BASE/'jianzhi/wait_for_publish/publish','guanglun':BASE/'guanglun_depth/publish'}
    jobs=[dict(dataset=r['dataset'],target=r['target'],folder=str(roots[r['dataset']]/r['target']),sources=['HumanCOIN/'+r['source']]) for r in plan['rename']]
    jobs.extend(dict(dataset=r['dataset'],target=r['target'],folder=str(roots[r['dataset']]/r['target']),sources=['HumanCOIN/'+s for s in r['sources']]) for r in plan['merge'])
    assert len(jobs)==59
    REPORT.mkdir(exist_ok=True);results=[]
    with (REPORT/'verification_progress.jsonl').open('w') as log,ThreadPoolExecutor(4) as pool:
        for future in as_completed([pool.submit(verify,j) for j in jobs]):
            result=future.result();results.append(result);log.write(json.dumps(result)+'\n');log.flush()
            if len(results)%5==0 or len(results)==59:print(json.dumps({'checked':len(results),'total':59,'errors':sum(r['status']=='error' for r in results)}),flush=True)
    errors=[r for r in results if r['status']=='error'];deletions=[old for r in results if r['status']=='verified' for old in r['old_repositories']]
    summary={'targets':len(results),'verified_targets':len(results)-len(errors),'failed_targets':len(errors),'eligible_old_repositories':len(deletions),'deleted':0}
    (REPORT/'verification.json').write_text(json.dumps({'summary':summary,'records':results},indent=2));(REPORT/'deletion_candidates.json').write_text(json.dumps(deletions,indent=2))
    if args.delete:
        if errors:raise SystemExit('Verification incomplete: NO old repositories deleted; inspect verification.json')
        assert len(deletions)==len(set(deletions))==67
        targets={'HumanCOIN/'+j['target'] for j in jobs}
        if targets & set(deletions):raise ValueError('Deletion/target overlap')
        api=HubApi(token=get_token('modelscope'))
        with (REPORT/'deleted.jsonl').open('a') as log:
            for old in deletions:
                verified=next(r for r in results if old in r['old_repositories'])
                if manifest_hash(tree(api,old))!=verified['old_fingerprints'][old]:
                    raise ValueError('Old repository changed after verification; deletion stopped')
                if manifest_hash(tree(api,verified['repo_id']))!=verified['target_fingerprint']:
                    raise ValueError('Target repository changed after verification; deletion stopped')
                api.delete_repo(old,repo_type='dataset')
                if api.repo_exists(old,repo_type='dataset'):raise ValueError('Deletion not confirmed')
                summary['deleted']+=1;log.write(json.dumps({'repo_id':old})+'\n');log.flush()
                print(json.dumps({'deleted':summary['deleted'],'total':67}),flush=True)
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
