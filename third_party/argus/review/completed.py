"""Reuse fully completed annotations across model changes without paid calls."""
import hashlib
import json
from pathlib import Path


def completed_result(job, output_dir=None):
    job=Path(job)
    try:
        compact_path = job / 'structured_complete.json'
        if compact_path.exists():
            receipt = json.loads(compact_path.read_text(encoding='utf-8'))
            outputs = []
            if receipt.get('version') != 1 or receipt.get('status') != 'done' or not receipt.get('outputs'):
                return None
            for name, digest in receipt['outputs'].items():
                if Path(name).name != name or not name.endswith('_structured.json'):
                    return None
                location = receipt.get('output_location', 'job')
                if location not in ('job', 'annotation'):
                    return None
                if location == 'annotation' and job.parent.name != 'argus':
                    return None
                path = (job.parent.parent if location == 'annotation' else job) / name
                raw = path.read_bytes()
                data = json.loads(raw)
                if path.is_symlink() or hashlib.sha256(raw).hexdigest() != digest or not isinstance(data, dict) or 'extraction' in data:
                    return None
                outputs.append(str(path))
            return dict(status='done', reused=True, reason='previous_annotation_complete',
                        previous_model=receipt.get('previous_model'), outputs=outputs)
        info=json.loads((job/'run_final/run.json').read_text(encoding='utf-8'))
        if info.get('status')!='done':
            return None
        sources=sorted((job/'run_final/out').glob('episode_*.json'))
        if not sources:
            return None
        from review.extract_dimensions import prepare
        outputs=[]
        for source in sources:
            prepared=prepare(source)
            output=(Path(output_dir) if output_dir is not None else job)/(source.stem+'_structured.json')
            receipt=json.loads((job/('.'+source.stem+'_structured.claim.json')).read_text(encoding='utf-8'))
            raw=output.read_bytes()
            data=json.loads(raw.decode('utf-8'))
            if (receipt.get('status')!='complete' or
                receipt.get('source_sha256')!=prepared['source']['sha256'] or
                receipt.get('output_sha256')!=hashlib.sha256(raw).hexdigest() or
                not isinstance(data,dict) or 'extraction' in data):
                return None
            outputs.append(str(output))
        return dict(status='done',reused=True,reason='previous_annotation_complete',
                    previous_model=info.get('annotation_policy',{}).get('model'),outputs=outputs)
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return None
