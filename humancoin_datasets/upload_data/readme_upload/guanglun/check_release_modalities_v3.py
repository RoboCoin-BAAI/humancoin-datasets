"""Inventory modalities, downgrade only v3.1 metadata, audit/repair Jianzhi tail rows.

Uses inventory_release_v3's fixed batch selection; never changes annotations or video.
Readme generation is deferred. Independent sensor rows/statistics are preserved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import json
from pathlib import Path
import stat

import audit_jianzhi_video_rows as frames
import repair_jianzhi_video_rows as repairer
from reclassify_scenes_qwen import save_json
from apply_scene_classification import atomic_write

CAMERAS=set(frames.KEYS)
SCALARS={'timestamp','frame_index','episode_index','index','task_index'}


def inspect(job, args):
    row,root=job
    sample=root/row['path']
    result=dict(dataset=row['dataset'],path=row['path'],status='done',missing=[],retired_video_declarations=[],version_changed=False)
    try:
        if not sample.exists() and row['dataset']=='jianzhi' and (root/'private'/row['path']).is_dir():
            result['status']='private';return result
        if sample.is_symlink() or root.resolve() not in sample.resolve().parents:
            raise ValueError('Unsafe sample path')
        ip=sample/'meta/info.json'
        raw=ip.read_bytes();info=json.loads(raw)
        for feature,spec in info['features'].items():
            if feature in SCALARS:continue
            dtype=spec.get('dtype')
            if dtype=='video' and feature not in CAMERAS:
                result['retired_video_declarations'].append(feature)
                continue
            parquet=sample/'data/chunk-000/episode_000000'/f'{feature}.parquet'
            if not parquet.is_file() or parquet.stat().st_size==0:
                result['missing'].append(str(parquet.relative_to(sample)))
            if dtype=='video':
                video=sample/'videos/chunk-000'/feature/'episode_000000.mp4'
                if not video.is_file() or video.stat().st_size==0:
                    result['missing'].append(str(video.relative_to(sample)))
        # Required three camera streams even if info declarations are incomplete.
        for key in sorted(CAMERAS):
            if key not in info['features']:
                result['missing'].append('info.features.'+key)
        result['version_before']=info.get('codebase_version')
        if info.get('codebase_version') in ('v3.1','3.1'):
            revised=dict(info,codebase_version='v2.1' if info['codebase_version'].startswith('v') else '2.1')
            if args.apply_version:
                if ip.is_symlink() or ip.read_bytes()!=raw:raise ValueError('Concurrent info change or symlink')
                atomic_write(ip,(json.dumps(revised,ensure_ascii=False,indent=2)+'\n').encode(),stat.S_IMODE(ip.stat().st_mode))
                check=json.loads(ip.read_bytes());expected=dict(check,codebase_version=info['codebase_version'])
                if expected!=info:raise ValueError('Unexpected info changes')
                result['version_changed']=True
        if row['dataset']=='jianzhi':
            audit=frames.audit(dict(path=row['path']+'/annotation/episode_000000_structured.json'))
            result['frame_audit']=audit
            if audit['status']!='done':raise ValueError(audit['error'])
    except Exception as exc:result.update(status='error',error=str(exc))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--inventory-dir',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--apply-version',action='store_true')
    p.add_argument('--repair-jianzhi',action='store_true')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();args.report_dir.mkdir(parents=True,exist_ok=True)
    frames.ROOT=args.base/'jianzhi/wait_for_publish';repairer.ROOT=frames.ROOT
    inventory=[json.loads(s) for s in (args.inventory_dir/'inventory.jsonl').read_text().splitlines()]
    jobs=[(r,args.base/('jianzhi/wait_for_publish' if r['dataset']=='jianzhi' else 'guanglun_depth')) for r in inventory]
    results=[]
    with (args.report_dir/'audit_progress.jsonl').open('w') as out,ThreadPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(inspect,j,args) for j in jobs]):
            r=f.result();results.append(r);out.write(json.dumps(r)+'\n');out.flush()
            if len(results)%100==0:print(json.dumps(dict(stage='audit',completed=len(results),total=len(jobs))),flush=True)
    summary=dict(samples=dict(Counter(r['dataset'] for r in results)),errors=sum(r['status']=='error' for r in results),
                 missing_samples=sum(bool(r['missing']) for r in results),
                 versions_changed=dict(Counter(r['dataset'] for r in results if r['version_changed'])),
                 retired_video_declarations=sum(bool(r['retired_video_declarations']) for r in results),
                 jianzhi_frame_mismatches=sum(r.get('frame_audit',{}).get('mismatch',False) for r in results))
    save_json(args.report_dir/'audit.json',dict(summary=summary,records=results));print(json.dumps(summary),flush=True)
    if args.repair_jianzhi:
        selected=[]
        for r in results:
            if r['status']!='done' or r['dataset']!='jianzhi' or r['missing']:continue
            a=r['frame_audit'];left=a['streams'][frames.KEYS[0]]['videos'][0]['frames']
            if a['mismatch'] or a['info_frames']!=left or sum(x['rows'] for x in a['master'])!=left or any(e.get('length')!=left for e in a['episodes']):
                selected.append(a)
        fixed=[]
        with (args.report_dir/'repair_progress.jsonl').open('w') as out,ThreadPoolExecutor(args.workers) as pool:
            for f in as_completed([pool.submit(repairer.repair,a,update_readme=False) for a in selected]):
                r=f.result();fixed.append(r);out.write(json.dumps(r)+'\n');out.flush()
                if len(fixed)%50==0:print(json.dumps(dict(stage='repair',completed=len(fixed),total=len(selected))),flush=True)
        final=dict(selected=len(selected),updated=sum(bool(r['files']) for r in fixed),errors=sum(r['status']=='error' for r in fixed))
        save_json(args.report_dir/'repair.json',dict(summary=final,records=fixed));print(json.dumps(final),flush=True)
        if final['errors']:raise SystemExit(1)
    if summary['errors']:raise SystemExit(1)


if __name__=='__main__':main()
