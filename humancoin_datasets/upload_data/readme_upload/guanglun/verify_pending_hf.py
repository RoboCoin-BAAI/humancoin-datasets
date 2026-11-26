"""Compare supplied local file-size manifests with pinned public HF revisions."""
import argparse,json,re,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=45) as response:
                return json.load(response),response.headers.get('Link','')
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)


def verify(row):
    repo=row['repo_id'];result=dict(repo_id=repo,status='done')
    try:
        info,_=get('https://huggingface.co/api/datasets/'+repo)
        revision=info['sha'];remote={};url='https://huggingface.co/api/datasets/'+repo+'/tree/'+revision+'?recursive=true&limit=1000'
        while url:
            entries,link=get(url)
            for e in entries:
                if e['type']=='file':remote[e['path']]=e['size']
            match=re.search(r'<([^>]+)>;\s*rel="next"',link)
            url=match.group(1) if match else None
        local=row['files'];missing=sorted(set(local)-set(remote))
        mismatch=[p for p in local.keys()&remote.keys() if local[p]!=remote[p]]
        remote_samples={p.split('/')[0] for p in remote if p.endswith('/meta/info.json')}
        result.update(revision=revision,local_samples=len(row['samples']),remote_samples=len(remote_samples),
            missing_samples=sorted(set(row['samples'])-remote_samples),missing_file_count=len(missing),
            size_mismatch_count=len(mismatch),missing_files=missing,size_mismatches=mismatch,
            missing_recording_files=sum('/' in p for p in missing),
            mismatched_recording_files=sum('/' in p for p in mismatch))
    except Exception as exc:result.update(status='error',error_type=type(exc).__name__)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads(a.manifest.read_text());results=[]
    with ThreadPoolExecutor(4) as pool:
        for f in as_completed([pool.submit(verify,r) for r in rows]):
            r=f.result();results.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ['missing_samples','missing_files','size_mismatches']}),flush=True)
    a.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
