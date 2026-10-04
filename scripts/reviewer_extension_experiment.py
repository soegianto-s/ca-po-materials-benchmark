"""Fixed post hoc graph/MLP/XGBoost search on original inner holdouts."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,copy,random,time,argparse,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,torch
from torch import nn
from scipy.special import expit
import physics_hybrid_experiment as ph
from train_periodic_benchmark import GraphModel,json_save,atomic_save,select_threshold
R=Path(__file__).resolve().parents[1];D=R/'revision_ieee_access/reviewer_extension_v1';OLD=R/'revision_ieee_access/physics_hybrid_v1'
SPECS={
 'graph_w64_l2':dict(kind='graph',width=64,layers=2,dropout=.1,lr=.001,bounded=False,density_loss=True,hybrid=False),
 'graph_w64_l3':dict(kind='graph',width=64,layers=3,dropout=.1,lr=.0005,bounded=False,density_loss=True,hybrid=False),
 'mlp_w32':dict(kind='mlp',width=32,layers=2,dropout=.1,lr=.001,bounded=True,density_loss=False,hybrid=False),
 'mlp_w64':dict(kind='mlp',width=64,layers=2,dropout=.1,lr=.001,bounded=True,density_loss=False,hybrid=False),
 'mlp_w64_d30':dict(kind='mlp',width=64,layers=2,dropout=.3,lr=.001,bounded=True,density_loss=False,hybrid=False)}
class AdaptGraph(GraphModel):
 def forward(self,*args):return super().forward(*args[:6])
class CompositionMLP(nn.Module):
 def __init__(self,width,dropout):
  super().__init__();self.net=nn.Sequential(nn.Linear(90,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Dropout(dropout),nn.Linear(width,4))
 def forward(self,x):return self.net(x)
def initialize():ph.initialize('reviewer_extension_v1')
def train(job):
 partition,fold,seed,name=job;spec=SPECS[name];G=ph.G;Y=G['Y'];folder=D/f'partition{partition}_fold{fold}'/f'seed{seed}'/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():return f'{job} cached'
 split=json.loads((folder.parent.parent/'split.json').read_text());fit,val,test=[np.array([G['index'][i] for i in split[k]]) for k in ['fit','validation','test']]
 X,xmu,xsd=ph.scale_fit(G['X'],fit);mu=torch.from_numpy(Y[fit,1:].mean(0));sd=torch.from_numpy(np.maximum(Y[fit,1:].std(0),1e-8));weight=torch.tensor(float((len(fit)-Y[fit,0].sum())/Y[fit,0].sum()))
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 model=AdaptGraph(spec['width'],spec['layers'],spec['dropout']) if spec['kind']=='graph' else CompositionMLP(spec['width'],spec['dropout'])
 def inputs(ix):return ph.inputs(X,ix) if spec['kind']=='graph' else (torch.from_numpy(X[ix]),)
 def predict(ix):
  model.eval();out=[]
  with torch.no_grad():
   for i in range(0,len(ix),16):out.append(model(*inputs(ix[i:i+16])).numpy())
  return np.concatenate(out)
 opt=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=1e-5);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=.5)
 checkpoint=folder/'checkpoint.pt';fingerprint=G['manifest']['fingerprint'];best=float('inf');best_state=None;best_epoch=0;since=0;history=[];start_epoch=1;elapsed_before=0
 if checkpoint.exists():
  cp=torch.load(checkpoint,weights_only=False);assert cp['fingerprint']==fingerprint
  model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);best=cp['best'];best_state=cp['best_state'];best_epoch=cp['best_epoch'];since=cp['since'];history=cp['history'];start_epoch=cp['epoch']+1;elapsed_before=cp['elapsed_seconds'];torch.set_rng_state(cp['torch_rng'])
 start=time.monotonic()
 for epoch in range(start_epoch,151):
  if since>=25:break
  model.train();total=0;order=np.random.default_rng(seed+epoch).permutation(fit)
  for offset in range(0,len(order),16):
   ix=order[offset:offset+16];opt.zero_grad(set_to_none=True);out=model(*inputs(ix));loss=ph.objective(out,torch.from_numpy(Y[ix]),mu,sd,weight,spec);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach())*len(ix)
  vo=predict(val);vl=float(ph.objective(torch.from_numpy(vo),torch.from_numpy(Y[val]),mu,sd,weight,spec,True));assert np.isfinite(vl);scheduler.step(vl)
  history.append(dict(epoch=epoch,train_objective=total/len(fit),validation_selection_objective=vl,lr=opt.param_groups[0]['lr']))
  if vl<best-1e-5:best=vl;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());since=0
  else:since+=1
  if epoch%10==0 or epoch==150 or since>=25:
   atomic_save(dict(fingerprint=fingerprint,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),best=best,best_state=best_state,best_epoch=best_epoch,since=since,history=history,torch_rng=torch.get_rng_state(),epoch=epoch,elapsed_seconds=elapsed_before+time.monotonic()-start,mu=mu,sd=sd,x_mu=xmu,x_sd=xsd),checkpoint)
 model.load_state_dict(best_state);vo=predict(val);vp=expit(vo[:,0]);threshold=select_threshold(Y[val,0],vp);cal=ph.platt_fit(Y[val,0],vp);ct=select_threshold(Y[val,0],ph.calibrate(vp,cal));records={}
 for label,ix in [('validation',val),('test',test)]:records[label]=ph.make_prediction_record(split[label],Y[ix],predict(ix),mu,sd,spec,G['rho'][ix],threshold,cal,ct)
 json_save(records,folder/'predictions.json');json_save(history,folder/'loss_history.json');json_save(dict(variant=name,partition_seed=partition,fold=fold,training_seed=seed,fingerprint=fingerprint,best_epoch=best_epoch,epochs_run=len(history),validation_selection_objective=best,calibration=cal,parameter_count=sum(p.numel() for p in model.parameters()),training_seconds=elapsed_before+time.monotonic()-start),folder/'result.json')
 return f'{partition}/{fold}/{seed}/{name} best={best_epoch} epochs={len(history)}'
def setup():
 D.mkdir(exist_ok=True)
 config=dict(device='cpu',partition_seeds=[42,2024,7],folds=[1,2,3,4,5],initialization_seeds=[42,2024,7],epochs=150,patience=25,specs=SPECS,graph_candidates=['original graph_baseline (32x2,lr=.001)','graph_w64_l2','graph_w64_l3'],mlp_candidates=['mlp_w32','mlp_w64','mlp_w64_d30'],selection='For each outer split choose lowest mean inner-validation common objective across three seeds; checkpoint independently validation-selected per seed. No outer-test selection.',xgboost_grid=[{'max_depth':d,'n_estimators':n,'learning_rate':.05} for d in [2,4] for n in [200,500]])
 inputs=json.loads((OLD/'experiment.json').read_text())['inputs'].copy()
 for p in [Path(__file__),R/'revision_ieee_access/periodic_benchmark_v1/graphs.pt']:
  inputs[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
 manifest={'config':config,'inputs':inputs};manifest['fingerprint']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
 if (D/'experiment.json').exists():assert json.loads((D/'experiment.json').read_text())==manifest
 else:json_save(manifest,D/'experiment.json')
 for p in OLD.glob('partition*/split.json'):
  target=D/p.parent.name/'split.json';target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
 (D/'PROTOCOL.md').write_text('Fixed post hoc extension requested 2026-10-04. Original 511 IDs and 15 grouped inner/outer splits, three initialization seeds, 150 epochs/patience25 retained. Two larger graph configurations supplement the cached 32x2 original baseline. Three composition-only MLP configurations and four XGBoost configurations. Within each outer split choose a family configuration by mean common inner-validation objective over seeds; report all three matched outer runs. No test-based configuration decisions. Raw, calibrated and threshold outputs retained. Bootstrap uses 10000 paired composition-cluster resamples, same resampling weights across all nine OOF estimates, averaging metrics across runs. Frozen-prediction intervals condition on fitted models and cannot include refitting/selection uncertainty; two model contrasts × AP and gap MAE, simultaneous 98.75% percentile intervals (Bonferroni family level95%) in addition to marginal95%. Pos-weight analysis compares raw p, fitting-prior offset correction sigmoid(logit(p)-log(w)), and validation-fitted Platt on the same original checkpoints. No claim that correction equals retraining without weights.\n')
 return config
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--kind',choices=['graph','mlp','all'],default='all');args=ap.parse_args();c=setup();jobs=[(p,f,s,n) for p in c['partition_seeds'] for f in c['folds'] for s in c['initialization_seeds'] for n in SPECS if args.kind=='all' or SPECS[n]['kind']==args.kind]
 with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as ex:
  futures=[ex.submit(train,j) for j in jobs]
  for f in as_completed(futures):print(f.result(),flush=True)
 print('COMPLETE',args.kind,flush=True)
if __name__=='__main__':main()
