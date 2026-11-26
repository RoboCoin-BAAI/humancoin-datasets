"""Read-only source inventory and privacy candidates; writes reports only."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re

BODY = re.compile(r'\b(?:body|bodies|skin|scalp|hair|face|facial|head|neck|shoulders?|arms?|hands?|fingers?|nails?|back|abdomen|belly|waist|legs?|feet|foot|toes?|chest|breast|massage|massaging|massages|gua sha|cupping|manicure|pedicure)\b|人体|身体|皮肤|头发|头皮|面部|脸部|脖|颈部|肩部|手臂|手部|指甲|背部|腹部|腰部|腿部|足部|脚部|胸部|按摩|刮痧|拔罐|美甲', re.I)
NONBODY = re.compile(r'\b(?:chair back|back of (?:the )?(?:chair|cabinet|shelf|box)|robot(?:ic)? arm|clock hands?|door handles?|hair dryer|massage (?:chair|device|gun))\b', re.I)
CAMERAS = ('head_left_rectified', 'head_right_rectified', 'depth_head_left_rectified')


def inspect(item):
    dataset, root, sample = item
    row = dict(dataset=dataset, path=sample.relative_to(root).as_posix())
    required = ['annotation/episode_000000_structured.json', 'calibrations/dataset_manifest.json',
                'calibrations/rectify_params.yaml', 'data/chunk-000/episode_000000/master_index.parquet']
    required += ['meta/' + n for n in ('info.json', 'episodes.jsonl', 'episodes_stats.jsonl', 'tasks.jsonl')]
    required += [f'videos/chunk-000/observation.images.{c}/episode_000000.mp4' for c in CAMERAS]
    required += [f'data/chunk-000/episode_000000/observation.images.{c}.parquet' for c in CAMERAS]
    if dataset == 'guanglun':
        required += ['data/chunk-000/episode_000000/eye_gaze_predict.parquet']
    try:
        row['missing'] = [p for p in required if not (sample/p).is_file()]
        row['empty'] = [p for p in required if (sample/p).is_file() and (sample/p).stat().st_size == 0]
        data = json.loads((sample/required[0]).read_text(encoding='utf-8-sig'))
        scene = data.get('scene_classification') or {}
        row.update(task_name=data.get('task_name'), common_task_name=data.get('common_task_name'),
                   category=scene.get('category'), area=scene.get('area'),
                   object_names=[o.get('name') for o in data.get('objects', [])],
                   actions=(data.get('actions') or {}).get('vocabulary'),
                   task_description=data.get('task_description'),
                   original_description=scene.get('original_description', data.get('original_description')))
        texts = [str(row.get(k) or '') for k in ('task_name', 'task_description', 'original_description')]
        texts += [str(x or '') for x in row['object_names']]
        subs = data.get('subtask', data.get('subtasks', []))
        if isinstance(subs, list):
            texts += [str(s.get(k) or '') for s in subs if isinstance(s, dict)
                      for k in ('text', 'object', 'destination')]
        evidence = '\n'.join(texts)
        row['privacy_hits'] = sorted({m.group().lower() for m in BODY.finditer(evidence)}) if dataset == 'jianzhi' else []
        row['privacy_nonbody_context'] = sorted({m.group() for m in NONBODY.finditer(evidence)}) if dataset == 'jianzhi' else []
        row['privacy_status'] = 'candidate_requires_semantic_review' if row['privacy_hits'] else 'no_keyword_hit'
        row['video_parquets'] = sorted(p.name for p in (sample/'data/chunk-000/episode_000000').glob('observation.images.*.parquet'))
        row['status'] = 'done'
    except Exception as exc:
        row.update(status='error', error=str(exc))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    a = p.parse_args()
    a.report_dir.mkdir(parents=True, exist_ok=True)
    jobs, batches = [], []
    for dataset, root, start, end in [('jianzhi', a.base/'jianzhi/wait_for_publish', 72, 91),
                                      ('guanglun', a.base/'guanglun_depth', 50, 72)]:
        for i in range(start, end+1):
            batch = root/f'batch_{i:07d}'
            samples = sorted(x for x in batch.iterdir() if x.is_dir() and re.fullmatch(r'[0-9a-fA-F]{32}|[0-9a-fA-F]{64}', x.name)) if batch.is_dir() else []
            batches.append(dict(dataset=dataset, batch=batch.name, exists=batch.is_dir(), samples=len(samples)))
            jobs += [(dataset, root, s) for s in samples]
    rows = []
    with (a.report_dir/'inventory.jsonl').open('w', encoding='utf-8') as out, ThreadPoolExecutor(8) as pool:
        for row in pool.map(inspect, jobs):
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False)+'\n')
            if len(rows)%100 == 0:
                out.flush()
                print(json.dumps(dict(completed=len(rows), total=len(jobs))), flush=True)
    candidates = [r for r in rows if r.get('privacy_hits')]
    for name, selected in [('privacy_candidates', candidates), ('missing_files', [r for r in rows if r.get('missing') or r.get('empty') or r['status']=='error'])]:
        (a.report_dir/(name+'.json')).write_text(json.dumps(selected, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (a.report_dir/'privacy_candidates.csv').open('w', encoding='utf-8-sig', newline='') as out:
        keys = ['dataset', 'path', 'task_name', 'privacy_hits', 'privacy_nonbody_context', 'task_description', 'original_description']
        writer = csv.DictWriter(out, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(candidates)
    summary = dict(batches=batches, samples=dict(Counter(r['dataset'] for r in rows)),
                   privacy_candidates=len(candidates), errors=sum(r['status']=='error' for r in rows),
                   missing_samples=dict(Counter(r['dataset'] for r in rows if r.get('missing'))),
                   empty_samples=dict(Counter(r['dataset'] for r in rows if r.get('empty'))),
                   policy='Reports only. No sample moves or modifications. Privacy keyword hits need semantic review; no hit does not certify absence.')
    (a.report_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='batches'}), flush=True)


if __name__ == '__main__':
    main()
