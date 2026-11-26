"""Deterministic source conversion; translation and semantic cleaning belong to Argus."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def prepare_annotation(annotation, origin, out, *, source_bytes=None):
    rows=[]
    for index,segment in enumerate(annotation.get('annoDetail',{}).get('timeSegment',{}).get('segments',[])):
        start,end=segment['startTime'],segment['endTime']
        if type(start) is not int or type(end) is not int or end<=start:
            raise ValueError('Invalid annotation nanosecond interval')
        rows.append(dict(source_segment_index=index,start_s=(start-origin)/1e9,end_s=(end-origin)/1e9,
                         source_segment=deepcopy(segment)))
    if not rows:
        raise ValueError('No annotation segments')
    source=dict(file=str(Path(out)/'annotation/annotation.json'))
    if source_bytes is not None:
        source['sha256']=hashlib.sha256(source_bytes).hexdigest()
    return dict(format='archebase_annotation_source_v1',
        episode_dir=str(Path(out)/'data/chunk-000/episode_000000'),source=source,
        timestamp_origin_ns=origin,annotation_metadata={k:deepcopy(v) for k,v in annotation.items() if k!='annoDetail'},
        source_annotation=deepcopy(annotation),segments=rows,
        conversion_policy='Original fields preserved; integer nanosecond subtraction only; no translation, role inference or action splitting.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    args=parser.parse_args(argv)
    root=args.dataset.resolve()
    raw=(root/'annotation/annotation.json').read_bytes()
    meta=json.loads((root/'meta/archebase_source.json').read_text(encoding='utf-8'))
    value=prepare_annotation(json.loads(raw),meta['timestamp_origin_ns'],root,source_bytes=raw)
    path=root/'annotation/episode_000000_source.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=value:
        raise FileExistsError(f'Different intermediate annotation already exists: {path}')
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)
    print(path)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
