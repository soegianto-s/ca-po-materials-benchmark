"""Select within outer folds using only validation; summarize complete families."""
from pathlib import Path
import json,hashlib
import numpy as np
from summarize_physics_hybrid import combine,measure
R=Path(__file__).resolve().parents[1];D=R/'revision_ieee_access/reviewer_extension_v1';OLD=R/'revision_ieee_access/physics_hybrid_v1'
FAMILIES={'searched_graph':['graph_baseline','graph_w64_l2','graph_w64_l3'],'composition_mlp':['mlp_w32','mlp_w64','mlp_w64_d30'],'xgboost':['xgb_d2_n200','xgb_d2_n500','xgb_d4_n200','xgb_d4_n500']}
def folder(p,f,s,name):return (OLD if name=='graph_baseline' else D)/f'partition{p}_fold{f}'/f'seed{s}'/name
def main():
 m=json.loads((D/'experiment.json').read_text())
 for p,h in m['inputs'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
 all_metrics={};selected=[];complete={};grid={};oof={};counts={}
 for family,names in FAMILIES.items():
  counts[family]=sum((folder(p,f,s,n)/'result.json').exists() for p in [42,2024,7] for f in range(1,6) for s in [42,2024,7] for n in names);complete[family]=counts[family]==45*len(names)
  if not complete[family]:continue
  choices={}
  for p in [42,2024,7]:
   for f in range(1,6):
    scores={n:float(np.mean([json.loads((folder(p,f,s,n)/'result.json').read_text())['validation_selection_objective'] for s in [42,2024,7]])) for n in names};choice=min(names,key=lambda n:(scores[n],names.index(n)));choices[p,f]=choice;selected.append({'family':family,'partition':p,'fold':f,'choice':choice,'mean_validation_objectives':scores})
   for seed in [42,2024,7]:
    rows=[]
    for f in range(1,6):
     n=choices[p,f];v=json.loads((folder(p,f,seed,n)/'predictions.json').read_text());split=json.loads((D/f'partition{p}_fold{f}'/'split.json').read_text());assert v['test']['ids']==split['test'];assert v['validation']['ids']==split['validation'];rows.append(v['test'])
    pooled=combine(rows);grid.setdefault(f'{p}/{seed}',{})[family]=measure(pooled);oof.setdefault(f'{p}/{seed}',{})[family]=pooled
  all_metrics[family]={k:float(np.mean([v[family][k] for v in grid.values()])) for k in next(iter(grid.values()))[family]}
 summary={'complete_families':complete,'candidate_fits_including_cached_baseline':counts,'complete':all(complete.values()),'aggregate':all_metrics,'selected':selected,'grid':grid,'selection':'Within each outer fold, minimum mean common validation objective across three seeds, then evaluate each seed on its original untouched outer test.'}
 (D/'summary.json').write_text(json.dumps(summary,indent=2));(D/'oof_predictions.json').write_text(json.dumps(oof,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['grid','selected']},indent=2))
if __name__=='__main__':main()
