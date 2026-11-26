"""Audited offline fixes from preserved responses; emits a plan, never edits sources."""
import argparse
import copy
import json
from pathlib import Path
import re
import prepare_release_v3 as m

ALIASES={'pick_up':'grasp','pick':'grasp','take_out':'remove','put_back':'place','put_down':'place',
 'put_away':'place','take_down':'remove','take_off':'remove','pull_out':'remove','peel_off':'peel',
 'tear_off':'tear','turn_over':'rotate','turn_around':'rotate','tidy_up':'organize',
 'clean_up':'clean','wring_out':'wring','stand_up':'stand','sit_down':'sit','squat_down':'squat',
 'hand_over':'hand','put_in':'insert','line_up':'align','turn_on':'switch','turn_off':'switch',
 'hold_down':'press','roll_up':'roll','step_forward':'step','reassemble':'assemble',
 'set_aside':'place','break_open':'open','pay_attention':'observe','touch_up':'adjust',
 'flip_through':'browse','push_back':'push','neaten':'organize','inspect':'check','verify':'check',
 'paste':'stick','fasten':'secure','pick_out':'collect'}
DEFS={'bag':'Put items into a bag.','bend':'Change posture or shape by bending.',
 'break':'Separate an object by causing a fracture.','catch':'Intercept and hold an object.',
 'crumple':'Compress flexible material into an irregular shape.','dip':'Immerse part of an object in a liquid.',
 'drain':'Allow liquid to flow out.','drop':'Release an object so that it falls.',
 'input':'Enter information into a device.','lock':'Secure a closure using its locking mechanism.',
 'mop':'Clean a surface with a mop.','rub':'Move in frictional contact with a surface.',
 'sit':'Lower the body to a seated position.','swipe':'Move a card or object across a reader or surface.',
 'tap':'Make brief light contact.','water':'Supply water to a plant or other target.',
 'step':'Move the body by taking a step.','browse':'Look through successive pages or items.'}
TASK_FIX={'62776ec4617341e09a322308a92c39e5':None,
 'f0bb1a4988de43bab9ef02154ac85a88':('collect','package','collect_task','Retrieve identified packages for customer pickup.'),
 '01eae5c25f3e45a38cddaccc31f46c4a':('remove','snack','remove_task','Explicit removal of snacks from shelf.'),
 '11e13a395481453fa7e009ae18d37f4a':('arrange','snack','organize_task','Return and arrange snacks in the display container.'),
 '193ae3dc991f4387896e1b479a6ba03b':('collect','slipper','collect_task','Select slippers and collect them in a shopping basket.'),
 '21117545bf924bdf8cbd914ce04cc061':('arrange','seasoning','organize_task','Place packaged seasonings onto the supermarket display shelf.')}
HTTP_FIX={'bf5110f36ff7ae87f98feb91acc41f40d4e4bb1b12097923c14c14d737f1eabb':('organize','magazine','organize_task'),
 '740cfdc2bd0f43cdaf8cc732fbdd1390':('arrange','medicine','organize_task'),
 '9167b40385a64a388e863bdc694f0963':('arrange','price tag','organize_task'),
 '01449f6f9ecf4f89ad9b244d6e30fee6':('hang','snack','store_task'),
 '120f8f3e7cd84fe3971a5e5fa73fefad':('check','canned food','check_task')}

# Explicit per-record adjudications from source descriptions, not global verb substitutions.
MORE_TASKS={
 '6075c88d56324ffdbcfc0aef30fb9a48':('clean','floor','clean_task'),
 '630a51a541c045849a11f4b51d6fdfc1':('arrange','price tag','organize_task'),
 '63c17f67be1242379ea910374f0a873d':('arrange','price tag','organize_task'),
 '6d2bc4773a1042df9cefdc00828e43a1':('arrange','price tag','organize_task'),
 '806b99e2da6a4eadaccd8680b83dc187':('insert','parcel','fill_task'),
 '882d869f3ba341c795f14ea7bd0d6233':('arrange','price tag','organize_task'),
 'aeef784b857d4e6a89b683698f255a4f':('arrange','price tag','organize_task'),
 'b0de3080c71f4318b8bff7cec05327fa':('collect','parcel','collect_task'),
 'eafe64550b2b41ed85764a7d7841f701':('arrange','label','organize_task'),
 '00d733e0342a4dc6aedb88830b06b35f':('collect','cleaning supply','collect_task'),
 '02245358ed9543ddb63afa1ab623461f':('collect','snack','collect_task'),
 '03860b35cd4149ab9cecbb099e8efeda':('arrange','toothpaste','organize_task'),
 '09b2eef401db4bb7b16f55e6caac7b84':('arrange','sweet potato','organize_task'),
 '0b91b09545b049e8b3b52b2aa6a2bf04':('collect','snack','collect_task'),
 '12c43f7e1df04bd78c44ea3a3fbc4193':('collect','coffee bottle','collect_task'),
 '13b9926b301643379139adc28b9008b3':('collect','goji berry','collect_task'),
 '13faec7ca16d4a09bd4be44a40c77326':('peel','edamame pod','peel_task'),
 '15cc426d56104d2d813ac93f6c0a83ae':('collect','snack','collect_task'),
 '17f638a162f3495c8d99e5e56aed5958':('collect','snack','collect_task'),
 '1c36ff423f6d4cce9f9b2ffd7d2179b9':('arrange','cooking oil bottle','organize_task'),
 '1ffd20a68685486eafafc374cf1f1a1c':('arrange','bagged food','organize_task'),
 '2128ee0048ae4cfcaeee690fc41cedea':('collect','cup noodle','collect_task'),
 '2292a8e6ec164a349db951ac20450033':('arrange','price tag','organize_task'),
 '23109e09fb90412c8f0fc1e85f9ef6a7':('cut','carrot','cut_task'),
 '258123d8a7214346ba990b0829baff9d':('collect','date','collect_task'),
 '2719af74951c47f68844a39f19779878':('collect','snack','collect_task'),
 'b17b388824c286e31387b8bde85c69e12dc1fba2635daa4e6ec1cd9b41c4e0cf':('arrange','box','organize_task')}
for key,(verb,noun,category) in MORE_TASKS.items():
 TASK_FIX[key]=(verb,noun,category,'Manual judgment of source purpose: '+verb+' '+noun)
for key in ['2a28bb6f180d0baa5bff118b3bbc26d410fbe12e8f6f80883be39c3f952b405d',
 '2260ec28fdd46a2698daf1b9ec356f3bc54b473f993a030629cf21c7111aedbb',
 '9155fbf104f6bbcfce6159e369f759f4ce9e921fdac0c32b579957905125a7d8',
 '5166f075b6f77f285cd1bee9cee7507d3938343f396df41aa7c6a709643ba593',
 'fa0783d27910efd20bcdeade7518469577ba96543d6b95868d15d2218dc1f5a0',
 'd75bd7eaac0c4aa4a3018f5974505be4','0e9cd7c8ba0b48329a51244b95b7cc25',
 '2119521764e247daaa5a1a31d366eb0a','26c8aad4bb4242fdb9c601ab5df9baeb']:
 TASK_FIX[key]=None


def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--schema-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=m.read(a.evidence);b=a.schema_dir
 cfg=dict(actions=m.read(b/'humancoin_actions_v3.json'),scenes=m.read(b/'humancoin_scenes_v2.json'),categories=m.read(b/'humancoin_task_categories_v0_5_cover.json'))
 ref={r['verb'] for r in cfg['actions']['entries']}
 pool={}
 for x in rows:
  for c in x['responses']:
   if isinstance(c,dict):
    for e in c.get('action_extensions',[]):
     if isinstance(e,dict) and isinstance(e.get('verb'),str) and e.get('definition'):pool[e['verb']]=e['definition']
 records=[];failed=[]
 for x in rows:
  s=x['source'];r=x['record'];sid=r['path'].split('/')[-1]
  candidates=[c for c in x['responses'] if isinstance(c,dict) and 'scene' in c]
  partial=next((c for c in x['responses'] if isinstance(c,dict) and set(c)=={'action_vocabulary','action_extensions'}),None)
  if sid in HTTP_FIX:
   verb,noun,common=HTTP_FIX[sid];sc=s['scene_classification']
   names=[o['name'] for o in s['objects']]
   names=[{'price tags':'price tag','snacks':'snack'}.get(n,n) for n in names]
   candidates=[dict(scene={k:sc[k] for k in ('category','area')},object_names=names,
    action_vocabulary=s['actions']['vocabulary'],action_extensions=[],task=dict(verb=verb,object_name=noun,reason=s['task_description']),common_task_name=common,reason='Text-only manual recovery from explicit source description; no new visual claim.')]
   candidates[0]['scene']['reason']='Retained valid existing scene; description: '+sc.get('original_description','')
  choices=([dict(candidates[0],**partial)] if candidates and partial else [])+candidates
  error='No parseable proposal'
  for original in choices:
   try:
    c=copy.deepcopy(original);changes=[]
    expected={'scene','object_names','action_vocabulary','action_extensions','task','common_task_name','reason'}
    if set(c)>expected:c={k:v for k,v in c.items() if k in expected};changes.append('Removed non-output response keys')
    if not s['objects'] and c.get('object_names'):
     c['object_names']=[];changes.append('Preserved original empty object list; task target may come from description')
    if c.get('scene',{}).get('area')=='other':
     c['scene']['category']='other';changes.append('Corrected uncertain scene pair to other/other')
    if sid in TASK_FIX:
     fix=TASK_FIX[sid]
     if fix is None:
      c['task']=None;c['common_task_name']='other_task';changes.append('Main purpose lacks a faithful closed-set main verb; retained source task for review')
     else:
      v,n,k,why=fix;c['task']=dict(verb=v,object_name=n,reason=why);c['common_task_name']=k;changes.append('Manual task adjudication: '+why)
    if c.get('task') is None:c['common_task_name']='other_task'
    old=c['action_vocabulary'];verbs=list(dict.fromkeys(ALIASES.get(v,v) for v in old))
    c['action_vocabulary']=verbs;c['action_extensions']=[]
    evidence=json.dumps(m.evidence(s),ensure_ascii=False)
    for v in verbs:
     if v in ref:continue
     definition=DEFS.get(v,pool.get(v))
     if not definition:raise ValueError('No reviewed extension definition: '+v)
     # Reuse existing proposed actions only, never synthesize a new action sequence.
     c['action_extensions'].append(dict(verb=v,definition=definition,reason='Retained action from cached model proposal/source action sequence; source task: '+str(s.get('task_description',''))))
    changes.append('Normalized explicit action aliases, removed duplicates; rebuilt exact reference-set extension metadata')
    m.validate_pilot(c,s,cfg)
    record=dict(r,status='done',decision=c,needs_review=c['task'] is None,before=m.evidence(s),after=m.evidence(m.proposal(s,c)),recovery='Offline cached-response adjudication',adjudication=changes)
    record.pop('error',None)
    records.append(dict(record=record,cache=x['cache']));break
   except (ValueError,KeyError,TypeError) as exc:error=str(exc)
  else:failed.append(dict(path=r['path'],error=error,original_error=r['error']))
 output=dict(repaired=len(records),remaining=len(failed),records=records,failed=failed,source_writes=0)
 a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in output.items() if k!='records'}))


if __name__=='__main__':main()
