"""Paired composition bootstrap and analytic class-weight offset diagnostic."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss
from pymatgen.core import Composition
R=Path(__file__).resolve().parents[1];D=R/'revision_ieee_access/reviewer_extension_v1';OLD=R/'revision_ieee_access/physics_hybrid_v1';CAP=R/'revision_ieee_access/capacity_control_v1'
def combine(folder,model,p,s):
 rows=[json.loads((folder/f'partition{p}_fold{f}'/f'seed{s}'/model/'predictions.json').read_text())['test'] for f in range(1,6)]
 return {k:sum([v[k] for v in rows],[]) for k in ['ids','truth','probability','calibrated_probability','band_gap']}
def run():
 df=pd.read_csv(R/'processed/15_ml_magpie_features.csv');ids=sorted(df.material_id);lookup=df.set_index('material_id');groups=[Composition(lookup.loc[i,'formula']).reduced_formula for i in ids];unique=sorted(set(groups));assert len(unique)==421
 groupidx=np.array([unique.index(g) for g in groups]);models=['rf','graph_capacity','graph_hybrid'];arrays={m:{'p':[],'gap':[]} for m in models};truth=None
 original=json.loads((OLD/'oof_predictions.json').read_text())
 for p in [42,2024,7]:
  for s in [42,2024,7]:
   for m in models:
    v=combine(CAP,m,p,s) if m=='graph_capacity' else original[f'{p}/{s}'][m];ix=[v['ids'].index(i) for i in ids];y=np.array(v['truth'])[ix]
    if truth is None:truth=y
    else:assert np.array_equal(y,truth)
    arrays[m]['p'].append(np.array(v['probability'])[ix]);arrays[m]['gap'].append(np.array(v['band_gap'])[ix])
 labels=truth[:,0];prepared={}
 for m in models:
  prepared[m]=[];arrays[m]['gap']=np.array(arrays[m]['gap'])
  for prob in arrays[m]['p']:
   order=np.argsort(-prob,kind='stable');ends=np.r_[np.flatnonzero(np.diff(prob[order])!=0),len(prob)-1];prepared[m].append((order,ends))
 def ap(m,w):
  total=np.dot(w,labels);out=[]
  for order,ends in prepared[m]:
   tp=np.cumsum((w*labels)[order])[ends];n=np.cumsum(w[order])[ends];dp=np.diff(np.r_[0,tp]);out.append(np.sum(dp*np.divide(tp,n,out=np.zeros_like(tp),where=n>0))/total)
  return np.mean(out)
 rng=np.random.default_rng(20261004);testw=rng.multinomial(len(unique),np.ones(len(unique))/len(unique))[groupidx]
 for m in models:assert np.isclose(ap(m,testw),np.mean([average_precision_score(labels,p,sample_weight=testw) for p in arrays[m]['p']]))
 errors={m:np.abs(arrays[m]['gap']-truth[:,2]).mean(0) for m in models};contrasts=[('graph_capacity','rf'),('graph_hybrid','graph_capacity')];samples={f'{a}_minus_{b}':{'ap':[],'gap_mae':[]} for a,b in contrasts}
 for j in range(10000):
  w=rng.multinomial(len(unique),np.ones(len(unique))/len(unique))[groupidx];metrics={m:(ap(m,w),float(np.dot(errors[m],w)/w.sum())) for m in models}
  for a,b in contrasts:
   for k,pos in [('ap',0),('gap_mae',1)]:samples[f'{a}_minus_{b}'][k].append(metrics[a][pos]-metrics[b][pos])
  if j%2000==0:print('bootstrap',j,flush=True)
 out={'resamples':10000,'clusters':421,'materials':511,'seed':20261004,'estimand':'Mean of nine OOF metric differences, retaining structure weighting and pairing all predictions within each resampled composition. Conditional on frozen models; no refit or hyperparameter-selection uncertainty.','contrasts':{}}
 for a,b in contrasts:
  key=f'{a}_minus_{b}';out['contrasts'][key]={}
  for k in ['ap','gap_mae']:
   arr=np.array(samples[key][k]);point=ap(a,np.ones(511))-ap(b,np.ones(511)) if k=='ap' else float(errors[a].mean()-errors[b].mean())
   out['contrasts'][key][k]={'difference':point,'percentile_95':np.quantile(arr,[.025,.975]).tolist(),'percentile_98_75_bonferroni_four_contrasts':np.quantile(arr,[.00625,.99375]).tolist()}
 (D/'bootstrap_summary.json').write_text(json.dumps(out,indent=2));np.savez_compressed(D/'bootstrap_replicates.npz',**{k+'_'+metric:np.array(v) for k,d in samples.items() for metric,v in d.items()})
 # Fixed analytic logit offset based only on fitting prevalence.
 rows=[];weights=[]
 for p in [42,2024,7]:
  for seed in [42,2024,7]:
   for model in ['graph_baseline','graph_physics','graph_hybrid']:
    ys=[];probs={k:[] for k in ['raw','offset','platt']}
    for fold in range(1,6):
     base=OLD/f'partition{p}_fold{fold}';split=json.loads((base/'split.json').read_text());fit=lookup.loc[split['fit'],'is_stable'].to_numpy(float);w=(len(fit)-fit.sum())/fit.sum();weights.append(w)
     v=json.loads((base/f'seed{seed}'/model/'predictions.json').read_text())['test'];y=np.array(v['truth'])[:,0];raw=np.array(v['probability']);corrected=expit(logit(np.clip(raw,1e-12,1-1e-12))-np.log(w))
     assert np.array_equal(np.argsort(raw,kind='stable'),np.argsort(corrected,kind='stable'))
     ys.extend(y.tolist());probs['raw'].extend(raw.tolist());probs['offset'].extend(corrected.tolist());probs['platt'].extend(v['calibrated_probability'])
    for mode,pr in probs.items():rows.append(dict(partition=p,seed=seed,model=model,mode=mode,brier=brier_score_loss(ys,pr),log_loss=log_loss(ys,np.clip(pr,1e-6,1-1e-6)),ap=average_precision_score(ys,pr)))
 agg={m:{mode:{k:float(np.mean([x[k] for x in rows if x['model']==m and x['mode']==mode])) for k in ['brier','log_loss','ap']} for mode in probs} for m in ['graph_baseline','graph_physics','graph_hybrid']}
 (D/'pos_weight_summary.json').write_text(json.dumps({'aggregate':agg,'rows':rows,'weight_range':[min(weights),max(weights)],'interpretation':'Analytic inverse of weighted-BCE population-optimum odds shift, conditional on same trained checkpoints. Does not identify causal weight effects or equal an unweighted retraining. Platt slope and offset also reflect model misspecification and finite-sample adaptation.'},indent=2))
 print(json.dumps(out,indent=2));print(json.dumps(agg,indent=2))
if __name__=='__main__':run()
