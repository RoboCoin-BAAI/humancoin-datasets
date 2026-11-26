"""Apply completed V3 proposals or move reviewed private recordings. No API calls.

Default: preflight only. --apply executes the selected action. No full data backups.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import prepare_release_v3 as prepare
import reclassify_scenes_qwen as scene

ANNOTATION = 'annotation/episode_000000_structured.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sample_path(base, dataset, relative):
    if dataset not in ('jianzhi','guanglun') or not re.fullmatch(r'batch_[0-9]{7}/(?:[0-9a-f]{32}|[0-9a-f]{64})',relative):
        raise ValueError('Unsafe dataset/sample path')
    root = base/('jianzhi/wait_for_publish' if dataset=='jianzhi' else 'guanglun_depth')
    path = root/relative
    if root.resolve() not in path.resolve().parents or any(p.is_symlink() for p in (root,path.parent,path)):
        raise ValueError('Unsafe source or symlink: '+str(path))
    return path


def assert_scope(before, after):
    """Prove that masking the exact allowed fields yields identical documents."""
    masked = copy.deepcopy(after)
    for obj, old in zip(masked['objects'],before['objects']):
        obj['name'] = old['name']
    if len(masked['objects'])!=len(before['objects']):
        raise ValueError('Object count changed')
    masked['actions']['vocabulary'] = before['actions']['vocabulary']
    for key in ('category','area'):
        masked['scene_classification'][key] = before['scene_classification'][key]
    for key in ('task_name','common_task_name'):
        if key in before:
            masked[key] = before[key]
        else:
            masked.pop(key,None)
    if masked != before:
        raise ValueError('Modification outside the allowed fields')


def private_plan(args):
    report = prepare.read(args.report_dir/'privacy_decisions.json')
    if report.get('complete') is not True or report.get('errors')!=0:
        raise ValueError('Privacy stage not complete')
    plan = []
    private_root = args.base/'jianzhi/wait_for_publish/private'
    if private_root.is_symlink():
        raise ValueError('Private root is a symlink')
    for row in report['records']:
        if row.get('status')!='done' or row['decision']['private'] is not True:
            continue
        source = sample_path(args.base,row['dataset'],row['path'])
        if row['dataset']!='jianzhi':
            raise ValueError('Private rule is authorized for Jianzhi only')
        target = private_root/row['path']
        if private_root.resolve() not in target.resolve().parents or target.parent.is_symlink() or target.is_symlink():
            raise ValueError('Unsafe private destination')
        if source.exists() and target.exists():
            raise ValueError('Both source and target exist: '+row['path'])
        location = source if source.exists() else target
        if sha((location/ANNOTATION).read_bytes()) != row['source_sha256']:
            raise ValueError('Privacy evidence changed: '+row['path'])
        plan.append(dict(source=str(source),target=str(target),path=row['path'],
                         status='pending' if source.exists() else 'already_moved'))
    return plan


def apply_annotations(args):
    config = dict(actions=prepare.read(args.schema_dir/'humancoin_actions_v3.json'),
                  scenes=prepare.read(args.schema_dir/'humancoin_scenes_v2.json'),
                  categories=prepare.read(args.schema_dir/'humancoin_task_categories_v0_5_cover.json'),
                  standard=(args.schema_dir/'HUMANCOIN_ANNOTATION_STANDARD_V3.md').read_text(encoding='utf-8'))
    report = prepare.read(args.report_dir/'full_decisions.json')
    if report.get('stage')!='full' or report.get('complete') is not True or report.get('errors')!=0:
        raise ValueError('Complete full stage before writeback; incomplete reports cannot be applied')
    if report.get('policy_sha256')!=scene.digest(config):
        raise ValueError('V3 policy changed since preparation')
    identity = scene.digest(report)
    journal_path = args.report_dir/'annotation_write_journal.json'
    events_path = args.report_dir/'annotation_write_events.jsonl'
    journal = prepare.read(journal_path) if journal_path.exists() else dict(report_digest=identity,records={})
    if journal['report_digest']!=identity:
        raise ValueError('Writeback report changed; preserve its journal and use a separate report directory')
    if events_path.exists():
        for line in events_path.read_text(encoding='utf-8').splitlines():
            event=json.loads(line)
            if event['report_digest']!=identity:
                raise ValueError('Write journal identity mismatch')
            journal['records'][event['key']]=event['hashes']
    plans,skipped,seen = [],[],set()
    for row in report['records']:
        key = row['dataset']+'/'+row['path']
        if key in seen:
            raise ValueError('Duplicate sample: '+key)
        seen.add(key)
        if row.get('status')!='done':
            raise ValueError('Non-success record in completed report')
        path = sample_path(args.base,row['dataset'],row['path'])/ANNOTATION
        if path.is_symlink():
            raise ValueError('Annotation is a symlink')
        raw = path.read_bytes()
        old = json.loads(raw)
        prior = journal['records'].get(key)
        if prior and sha(raw)==prior['after_sha256']:
            skipped.append(dict(path=key,status='already_applied'))
            continue
        if sha(raw)!=row['source_sha256']:
            raise ValueError('Source changed; refusing stale writeback: '+key)
        decision = prepare.validate_pilot(row['decision'],old,config)
        if decision['task'] is None or row.get('needs_review'):
            skipped.append(dict(path=key,status='needs_review'))
            continue
        revised = prepare.proposal(old,decision)
        assert_scope(old,revised)
        encoded = (json.dumps(revised,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        plans.append((key,path,raw,encoded))
    print(json.dumps(dict(stage='writeback',selected=len(seen),ready=len(plans),skipped=len(skipped),apply=args.apply)),flush=True)
    if args.apply:
        if not journal_path.exists():
            scene.save_json(journal_path,dict(report_digest=identity,records={}))
        for key,path,before,after in plans:
            if path.read_bytes()!=before:
                raise ValueError('Source changed during writeback: '+key)
            journal['records'][key] = dict(before_sha256=sha(before),after_sha256=sha(after))
            # Append intended hashes durably before replacement; avoid rewriting the entire journal.
            with events_path.open('a',encoding='utf-8') as events:
                events.write(json.dumps(dict(report_digest=identity,key=key,hashes=journal['records'][key]))+'\n')
                events.flush()
                os.fsync(events.fileno())
            scene.atomic_write(path,after,stat.S_IMODE(path.stat().st_mode))
            if path.read_bytes()!=after:
                raise ValueError('Write verification failed: '+key)
            if len(journal['records'])%100==0:
                print(json.dumps(dict(stage='writeback',completed=len(journal['records']),total=len(seen))),flush=True)
        scene.save_json(args.report_dir/'annotation_write_summary.json',
                        dict(written=len(plans),skipped=skipped,report_digest=identity))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['private','annotations'])
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--schema-dir',type=Path)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    if args.stage=='annotations' and args.schema_dir is None:
        p.error('annotations requires --schema-dir')
    lock=args.report_dir/'writeback.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.write(fd,str(os.getpid()).encode('ascii'))
        if args.stage=='private':
            plan=private_plan(args)
            print(json.dumps(dict(stage='private',plan=plan,apply=args.apply)),flush=True)
            if args.apply:
                for row in plan:
                    if row['status']=='already_moved':
                        continue
                    source,target=Path(row['source']),Path(row['target'])
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():
                        raise ValueError('Destination appeared during move')
                    os.rename(source,target)  # Same-filesystem move, no copy/delete fallback.
                    row['status']='moved'
                scene.save_json(args.report_dir/'private_move_report.json',plan)
        else:
            apply_annotations(args)
    finally:
        os.close(fd)
        lock.unlink()


if __name__=='__main__':
    main()
