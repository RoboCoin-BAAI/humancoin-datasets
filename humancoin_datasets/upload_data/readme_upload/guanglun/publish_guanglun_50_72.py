"""Preflight or move Guanglun 50-72, then regenerate affected dataset cards. No upload."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import json
from pathlib import Path
import sys
import move_guanglun_to_publish_by_dataset as mover
from reclassify_scenes_qwen import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true');args=p.parse_args()
    base=Path('/share/project/liuyou/humancoin/data')
    root=base/'guanglun_depth';dest=root/'publish';report=base/'guanglun_50_72_publish_20261122'
    if dest.is_symlink() or dest.resolve()!=dest or not dest.is_dir():raise ValueError('Unsafe publish root')
    report.mkdir(exist_ok=True)
    mover.ROOT=root;mover.DEST=dest
    frozen=report/'plan.json'
    if frozen.exists():plan=json.loads(frozen.read_text())['records']
    else:
        inventory=[json.loads(s) for s in (base/'release_v3_inventory_20261122/inventory.jsonl').read_text().splitlines()]
        rows=[dict(path=r['path']+'/annotation/episode_000000_structured.json') for r in inventory if r['dataset']=='guanglun']
        if len(rows)!=2142:raise ValueError('Unexpected scope')
        with ThreadPoolExecutor(12) as pool:plan=list(pool.map(mover.inspect,rows))
        if len({r['target'] for r in plan})!=2142:raise ValueError('Duplicate targets')
        # Retain the existing release policy: other FPS recordings go outside publish.
        badfps=root/'badfps'
        if badfps.is_symlink():raise ValueError('Unsafe badfps directory')
        badfps.mkdir(exist_ok=True)
        for r in plan:
            card=json.loads((Path(r['source'])/'meta/readme.json').read_text())
            r['fps']=card['master_fps'];r['release']=card['master_fps'] in (25,30)
            if not r['release']:
                r['target']=str(badfps/r['dataset_name']/r['unique_task_name'])
                if Path(r['target']).exists():raise ValueError('Badfps destination collision')
        save_json(frozen,dict(records=plan))
    for r in plan:
        source,target=Path(r['source']),Path(r['target'])
        allowed=dest if r['release'] else root/'badfps'
        if root not in source.parents or allowed not in target.parents or target.parent.is_symlink():raise ValueError('Unsafe path')
        if source.exists() and target.exists():raise ValueError('Collision')
        current=source if source.exists() else target
        st=current.stat()
        if current.is_symlink() or st.st_ino!=r['inode'] or st.st_dev!=r['device']:raise ValueError('Changed directory identity')
    groups=sorted({r['dataset_name'] for r in plan if r['release']})
    print(json.dumps(dict(stage='preflight',samples=len(plan),publish=sum(r['release'] for r in plan),badfps=sum(not r['release'] for r in plan),fps_counts=dict(Counter(r['fps'] for r in plan)),groups=len(groups),apply=args.apply)),flush=True)
    if not args.apply:return
    results=[]
    with (report/'move_progress.jsonl').open('a') as log,ThreadPoolExecutor(8) as pool:
        for f in as_completed([pool.submit(mover.move_one,r) for r in plan]):
            r=f.result();results.append(r);log.write(json.dumps(r)+'\n');log.flush()
            if len(results)%200==0:print(json.dumps(dict(stage='move',completed=len(results),total=len(plan))),flush=True)
    sys.path.insert(0,'/share/project/liuyou/humancoin/code/roboego-hand-vis')
    import generate_l1_dataset_cards as cards
    project=cards.read_json(cards.PROJECT_CONFIG)
    jobs=[]
    for group in groups:
        folder=dest/group
        samples=[p for p in folder.iterdir() if p.is_dir() and not p.name.startswith('.') and (p/'meta/readme.json').is_file()]
        jobs.append(dict(dataset='guanglun',path=str(folder),tasks=len(samples)))
    generated=[]
    with ThreadPoolExecutor(8) as pool:
        for result in pool.map(lambda j:cards.generate(j,project),jobs):
            generated.append(result)
    errors=[r for r in generated if r['status']!='done']
    for r in plan:
        target=Path(r['target'])
        if Path(r['source']).exists() or target.stat().st_ino!=r['inode']:raise ValueError('Final move check failed')
    summary=dict(samples_moved=len(results),published=sum(r['release'] for r in plan),badfps=sum(not r['release'] for r in plan),affected_datasets=len(groups),dataset_cards=len(generated),card_errors=len(errors),
                 tasks_in_affected_datasets=sum(r['tasks'] for r in jobs),destination=str(dest),uploaded=False)
    save_json(report/'result.json',dict(summary=summary,card_results=generated))
    save_json(report/'upload_directories.json',dict(folders=[str(dest/g) for g in groups]))
    print(json.dumps(summary),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
