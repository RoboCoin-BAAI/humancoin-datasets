from copy import deepcopy
import json
import pytest

from review.archebase.run import request_body,run,validate_response


def prepared():
    return dict(format='archebase_annotation_source_v1',timestamp_origin_ns=1000000000,
        episode_dir='/dataset/data/chunk-000/episode_000000',source=dict(file='/dataset/annotation/annotation.json'),
        source_annotation=dict(annoDetail=dict(desc=None)),segments=[
            dict(source_segment_index=0,start_s=-.1,end_s=0.,source_segment=dict(startTime=900000000,endTime=1000000000,verbs=['无效'])),
            dict(source_segment_index=1,start_s=0.,end_s=1.,source_segment=dict(startTime=1000000000,endTime=2000000000,
                verbs=['放入'],description='把书放入托盘',objects=['书'],referenceObjects=['桌子'],hand='both'))])


def reply():
    return dict(task_name='Organize books',task_description='Place the book in a tray.',objects=[],
        segments=[dict(source_segment_index=1,arm='both',action=['place'],object='book',destination='tray',
                       spatial_relation='in',notes='Source reference object: table.',text='Place the book in the tray.')],data_issues=[])


def test_source_intervals_fixed_and_invalid_preserved_as_issue():
    p=prepared()
    result=validate_response(reply(),p)
    assert result['subtask'][0]['start_s']==0.
    assert result['subtask'][0]['end_s']==1.
    assert result['subtask'][0]['destination']=='tray'
    assert result['subtask'][0]['progress'] is None
    assert result['data_issues'][0]['start_s']==-.1
    assert 'source_segment' not in result['subtask'][0]


def test_intermediate_changed_time_is_rejected():
    p=prepared();p['segments'][1]['end_s']=2
    with pytest.raises(ValueError,match='timestamp'):
        validate_response(reply(),p)


@pytest.mark.parametrize('mutation',['drop','time','arm','chinese','duplicate'])
def test_model_cannot_change_source_structure(mutation):
    r=reply()
    if mutation=='drop':r['segments']=[]
    if mutation=='time':r['segments'][0]['start_s']=3
    if mutation=='arm':r['segments'][0]['arm']='left'
    if mutation=='chinese':r['segments'][0]['object']='书'
    if mutation=='duplicate':r['segments']*=2
    with pytest.raises(ValueError):
        validate_response(r,prepared())


def test_dry_run_needs_no_key_and_calls_no_endpoint(tmp_path,monkeypatch):
    source=tmp_path/'source.json';source.write_text(json.dumps(prepared()),encoding='utf-8')
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:pytest.fail('Network call during dry-run'))
    out=tmp_path/'out.json'
    result=run(source,out,dry_run=True)
    assert result['status']=='prepared'
    body=json.loads((tmp_path/'out_request.json').read_text(encoding='utf-8'))
    assert len(json.loads(body['messages'][1]['content'])['segments'])==1
    assert not out.exists()


def test_saved_response_produces_validated_result_and_receipt(tmp_path):
    source=tmp_path/'source.json';source.write_text(json.dumps(prepared()),encoding='utf-8')
    response=tmp_path/'reply.json';response.write_text(json.dumps(reply()),encoding='utf-8')
    out=tmp_path/'out.json'
    assert run(source,out,response_file=response)['status']=='complete'
    assert run(source,out,response_file=response)['status']=='reused'
    assert json.loads(out.read_text())['subtask'][0]['action']==['place']


def test_pending_claim_blocks_automatic_paid_retry(tmp_path):
    source=tmp_path/'source.json';source.write_text(json.dumps(prepared()),encoding='utf-8')
    out=tmp_path/'out.json'
    (tmp_path/'.out.json.claim.json').write_text('{}')
    with pytest.raises(RuntimeError,match='billed'):
        run(source,out)
