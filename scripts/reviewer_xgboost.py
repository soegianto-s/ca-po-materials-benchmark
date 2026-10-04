"""Four fixed XGBoost candidates using the frozen validation objective."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,time
from pathlib import Path
import numpy as np,torch
from scipy.special import logit
sys.path.insert(0,'/private/tmp/ieee-review-deps')
import xgboost as xgb
import physics_hybrid_experiment as ph
from train_periodic_benchmark import json_save,select_threshold
R=Path(__file__).resolve().parents[1];D=R/'revision_ieee_access/reviewer_extension_v1'
def main():
 ph.initialize('reviewer_extension_v1');G=ph.G;Y=G['Y'];X=G['X'];spec={'bounded':False,'density_loss':False}
 json_save({'xgboost':xgb.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selection':'same composite weighted BCE + standardized gap/energy MSE on inner validation; candidate selected by average over seeds per outer split','training_class_weight':'unweighted','feature_scaling':'none','n_jobs':1},D/'xgboost_manifest.json')
 for partition in [42,2024,7]:
  for fold in range(1,6):
   split=json.loads((D/f'partition{partition}_fold{fold}'/'split.json').read_text());fit,val,test=[np.array([G['index'][i] for i in split[k]]) for k in ['fit','validation','test']];mu=Y[fit,1:].mean(0);sd=np.maximum(Y[fit,1:].std(0),1e-8);weight=torch.tensor(float((len(fit)-Y[fit,0].sum())/Y[fit,0].sum()))
   for seed in [42,2024,7]:
    for c in G['manifest']['config']['xgboost_grid']:
     name=f"xgb_d{c['max_depth']}_n{c['n_estimators']}";folder=D/f'partition{partition}_fold{fold}'/f'seed{seed}'/name;folder.mkdir(exist_ok=True,parents=True)
     if (folder/'result.json').exists():continue
     start=time.monotonic();kwargs=dict(**c,n_jobs=1,tree_method='hist',subsample=.8,colsample_bytree=.8,reg_lambda=1.,random_state=seed)
     cl=xgb.XGBClassifier(**kwargs,objective='binary:logistic');cl.fit(X[fit],Y[fit,0]);cl.save_model(folder/'classifier.json')
     regs=[]
     for j in [2,3]:
      model=xgb.XGBRegressor(**kwargs,objective='reg:squarederror');model.fit(X[fit],Y[fit,j]);model.save_model(folder/f'regression_target{j}.json');regs.append(model)
     def output(ix):
      out=np.zeros((len(ix),4),dtype=np.float32);out[:,0]=logit(np.clip(cl.predict_proba(X[ix])[:,1],1e-6,1-1e-6))
      for j,reg in zip([2,3],regs):out[:,j]=(reg.predict(X[ix])-mu[j-1])/sd[j-1]
      return out
     vo=output(val);p=cl.predict_proba(X[val])[:,1];cal=ph.platt_fit(Y[val,0],p);th=select_threshold(Y[val,0],p);ct=select_threshold(Y[val,0],ph.calibrate(p,cal));records={}
     for label,ix in [('validation',val),('test',test)]:records[label]=ph.make_prediction_record(split[label],Y[ix],output(ix),torch.from_numpy(mu),torch.from_numpy(sd),spec,G['rho'][ix],th,cal,ct)
     score=float(ph.objective(torch.from_numpy(vo),torch.from_numpy(Y[val]),torch.from_numpy(mu),torch.from_numpy(sd),weight,spec,True));json_save(records,folder/'predictions.json');json_save({'variant':name,'partition_seed':partition,'fold':fold,'training_seed':seed,'validation_selection_objective':score,'config':kwargs,'training_seconds':time.monotonic()-start,'calibration':cal},folder/'result.json');print(partition,fold,seed,name,flush=True)
 print('ALL XGBOOST COMPLETE',flush=True)
if __name__=='__main__':main()
