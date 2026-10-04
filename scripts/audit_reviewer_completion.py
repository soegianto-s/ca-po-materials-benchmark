"""Audit fixed inputs, group separation, checkpoints and validation-only choices."""
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];D=R/'revision_ieee_access/reviewer_extension_v1'
m=json.loads((D/'experiment.json').read_text());s=json.loads((D/'summary.json').read_text());assert s['complete']
for p,h in m['inputs'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
counts=Counter()
for split_path in D.glob('partition*/split.json'):
 split=json.loads(split_path.read_text());assert split==json.loads((R/'revision_ieee_access/physics_hybrid_v1'/split_path.parent.name/'split.json').read_text())
 for a,b in [('fit','validation'),('fit','test'),('validation','test')]:
  assert not set(split[a])&set(split[b]);assert not set(split[a+'_groups'])&set(split[b+'_groups'])
 assert len(set(split['fit']+split['validation']+split['test']))==511
 for rp in split_path.parent.glob('seed*/*/result.json'):
  r=json.loads(rp.read_text());name=rp.parent.name;counts[name]+=1
  pred=json.loads((rp.parent/'predictions.json').read_text())
  for label in ['validation','test']:assert pred[label]['ids']==split[label]
  if name.startswith(('graph_','mlp_')):
   assert r['fingerprint']==m['fingerprint'];cp=torch.load(rp.parent/'checkpoint.pt',weights_only=False)
   assert cp['fingerprint']==m['fingerprint'];assert cp['best_epoch']==r['best_epoch'];assert cp['best']==r['validation_selection_objective']
   hist=json.loads((rp.parent/'loss_history.json').read_text());assert len(hist)==r['epochs_run'];assert abs(hist[r['best_epoch']-1]['validation_selection_objective']-r['validation_selection_objective'])<1e-9
   assert r['validation_selection_objective']<=min(x['validation_selection_objective'] for x in hist)+1.01e-5
for name,count in counts.items():assert count==45,(name,count)
for sel in s['selected']:
 scores=sel['mean_validation_objectives'];assert scores[sel['choice']]==min(scores.values())
 for name,score in scores.items():
  parent=R/'revision_ieee_access/physics_hybrid_v1' if name=='graph_baseline' else D
  vals=[json.loads((parent/f"partition{sel['partition']}_fold{sel['fold']}"/f'seed{seed}'/name/'result.json').read_text())['validation_selection_objective'] for seed in [42,2024,7]]
  assert abs(np.mean(vals)-score)<1e-12
out={'passed':True,'checks':['frozen input SHA256 hashes','identical original grouped splits','no fit/validation/test composition overlap','511 IDs per split','all candidate result counts','neural checkpoint fingerprint and best epoch consistency','selection equals minimum mean validation objective across seeds'],'candidate_counts':dict(counts),'selection_counts':{f:dict(Counter(x['choice'] for x in s['selected'] if x['family']==f)) for f in s['aggregate']},'scope':'Integrity checks; not an independent replication or estimate of retraining uncertainty.'}
(D/'completion_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
