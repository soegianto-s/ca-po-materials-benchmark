"""Controlled physics/composition ablation on frozen composition-disjoint splits.

Five graph variants x 15 frozen folds x three initialization seeds.
All model selection and calibration use inner validation only.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import argparse
import copy
import hashlib
import json
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss
from train_periodic_benchmark import GraphModel,batch,digest,json_save,atomic_save,select_threshold

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'revision_ieee_access/periodic_benchmark_v1'
VARIANTS={
    'graph_baseline':dict(bounded=False,density_loss=True,hybrid=False),
    'graph_gap':dict(bounded=True,density_loss=True,hybrid=False),
    'graph_nodensity':dict(bounded=False,density_loss=False,hybrid=False),
    'graph_physics':dict(bounded=True,density_loss=False,hybrid=False),
    'graph_hybrid':dict(bounded=True,density_loss=False,hybrid=True),
}
G={}

class HybridModel(GraphModel):
    def __init__(self,hybrid=False):
        super().__init__(32,2,.1)
        self.hybrid=hybrid
        if hybrid:
            self.composition=nn.Sequential(nn.Linear(90,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
            self.head=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,4))

    def forward(self,nodes,edges,src,dst,assignment,n_graphs,composition):
        h=self.embed(nodes)
        for conv,norm in zip(self.convs,self.norms): h=norm(conv(h,edges,src,dst))
        pooled=torch.zeros((n_graphs,h.shape[1]),dtype=h.dtype,device=h.device)
        pooled.index_add_(0,assignment,h)
        pooled=pooled/torch.bincount(assignment,minlength=n_graphs).to(h.dtype).unsqueeze(1)
        if self.hybrid: pooled=torch.cat([pooled,self.composition(composition)],dim=1)
        return self.head(pooled)

def decode(out,mu,sd,bounded):
    density=out[:,1]*sd[0]+mu[0]
    gap=out[:,2]*sd[1]+mu[1]
    if bounded: gap=torch.relu(gap)
    energy=out[:,3]*sd[2]+mu[2]
    return torch.stack([density,gap,energy],dim=1)

def objective(out,y,mu,sd,weight,spec,selection=False):
    pred=decode(out,mu,sd,spec['bounded'])
    mse=(((pred-y[:,1:])/sd)**2).mean(0)
    bce=nn.functional.binary_cross_entropy_with_logits(out[:,0],y[:,0],pos_weight=weight)
    # Common selection objective excludes density for every variant.
    return bce+.5*mse[1]+mse[2]+(0. if selection or not spec['density_loss'] else .5*mse[0])

def scale_fit(X,fit):
    mu=X[fit].mean(0);sd=X[fit].std(0);sd=np.where(sd<1e-8,1.,sd)
    return ((X-mu)/sd).astype(np.float32),mu,sd

def platt_fit(y,p):
    z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
    def fun(ab):
        a,b=ab; t=a*z+b; q=expit(t); e=q-y
        loss=np.mean(np.logaddexp(0,t)-y*t)+1e-4*((a-1)**2+b*b)
        grad=np.array([np.mean(e*z)+2e-4*(a-1),np.mean(e)+2e-4*b])
        return loss,grad
    r=minimize(fun,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(0.,20.),(-20.,20.)])
    if not r.success: raise RuntimeError('Calibration optimizer failed: '+str(r.message))
    return dict(a=float(r.x[0]),b=float(r.x[1]),objective=float(r.fun))

def calibrate(p,cal):
    p=np.clip(p,1e-6,1-1e-6)
    return expit(cal['a']*np.log(p/(1-p))+cal['b'])

def initialize(run_name):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run=ROOT/'revision_ieee_access'/run_name
    manifest=json.loads((run/'experiment.json').read_text())
    device=manifest['config']['device']
    torch.use_deterministic_algorithms(True,warn_only=(device=='mps'))
    if device=='mps':assert torch.backends.mps.is_available()
    df=pd.read_csv(ROOT/'processed/15_ml_magpie_features.csv')
    feat=[c for c in df if c.startswith(('frac_','wmean_','wstd_','min_','max_','range_'))]
    assert len(feat)==90
    rho=json.loads((OLD/'analytic_density_check.json').read_text())
    rd={r['material_id']:r['structure_density'] for r in rho['rows']}
    G.update(run=run,manifest=manifest,device=device,df=df,index={v:i for i,v in enumerate(df.material_id)},
        X=df[feat].to_numpy(dtype=np.float64),Y=df[['is_stable','density','band_gap','formation_energy_per_atom']].to_numpy(dtype=np.float32),
        rho=np.array([rd[x] for x in df.material_id]),graphs=torch.load(OLD/'graphs.pt',weights_only=False))

def inputs(X,indices):
    values=batch(G['graphs'],indices)
    return tuple(x.to(G['device']) if isinstance(x,torch.Tensor) else x for x in values)+(torch.from_numpy(X[indices]).to(G['device']),)

def prediction(model,X,indices):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(indices),16):
            ix=indices[start:start+16]
            out.append(model(*inputs(X,ix)).cpu().numpy())
    return np.concatenate(out)

def make_prediction_record(ids,y,out,mu,sd,spec,rho,threshold,cal,cal_threshold):
    reg=decode(torch.from_numpy(out),mu.cpu(),sd.cpu(),spec['bounded']).numpy()
    p=expit(out[:,0]);q=calibrate(p,cal)
    return dict(ids=ids,truth=y.tolist(),probability=p.tolist(),calibrated_probability=q.tolist(),
        band_gap=reg[:,1].tolist(),formation_energy=reg[:,2].tolist(),analytic_density=rho.tolist(),
        learned_density=reg[:,0].tolist() if spec['density_loss'] else None,
        threshold=threshold,calibrated_threshold=cal_threshold)

def graph_job(partition,fold,seed,variant):
    spec=VARIANTS[variant]; config=G['manifest']['config'];Y=G['Y']
    folder=G['run']/f'partition{partition}_fold{fold}'/f'seed{seed}'/variant
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists(): return str(folder.relative_to(G['run']))+' cached'
    split=json.loads((G['run']/f'partition{partition}_fold{fold}'/'split.json').read_text())
    fit,val,test=[np.array([G['index'][x] for x in split[k]]) for k in ['fit','validation','test']]
    X,xmu,xsd=scale_fit(G['X'],fit)
    mu=torch.from_numpy(Y[fit,1:].mean(0)).to(G['device']);sd=torch.from_numpy(np.maximum(Y[fit,1:].std(0),1e-8)).to(G['device'])
    pos=Y[fit,0].sum();weight=torch.tensor(float((len(fit)-pos)/pos),device=G['device'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if G['device']=='mps':torch.mps.manual_seed(seed)
    model=HybridModel(spec['hybrid']).to(G['device']);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=.5)
    checkpoint=folder/'checkpoint.pt';fingerprint=G['manifest']['fingerprint']
    best=float('inf');best_state=None;best_epoch=0;since=0;history=[];start_epoch=1;elapsed_before=0.
    if checkpoint.exists():
        cp=torch.load(checkpoint,weights_only=False)
        assert cp['fingerprint']==fingerprint and cp['variant']==variant and cp['seed']==seed
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        best=cp['best'];best_state=cp['best_state'];best_epoch=cp['best_epoch'];since=cp['since'];history=cp['history']
        start_epoch=cp['epoch']+1;elapsed_before=cp['elapsed_seconds'];torch.set_rng_state(cp['torch_rng'])
        if G['device']=='mps':torch.mps.set_rng_state(cp['mps_rng'])
    start=time.monotonic()
    for epoch in range(start_epoch,config['epochs']+1):
        if since>=config['patience']:break
        model.train();total=0.;order=np.random.default_rng(seed+epoch).permutation(fit)
        for offset in range(0,len(order),16):
            ix=order[offset:offset+16];opt.zero_grad(set_to_none=True)
            out=model(*inputs(X,ix))
            loss=objective(out,torch.from_numpy(Y[ix]).to(G['device']),mu,sd,weight,spec)
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach())*len(ix)
        vo=prediction(model,X,val)
        vl=float(objective(torch.from_numpy(vo),torch.from_numpy(Y[val]),mu.cpu(),sd.cpu(),weight.cpu(),spec,True))
        if not np.isfinite(vl): raise ValueError('Non-finite validation objective')
        scheduler.step(vl)
        history.append(dict(epoch=epoch,train_objective=total/len(fit),validation_selection_objective=vl,lr=opt.param_groups[0]['lr']))
        if vl<best-1e-5: best=vl;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());since=0
        else:since+=1
        if epoch%10==0 or epoch==config['epochs'] or since>=config['patience']:
            atomic_save(dict(fingerprint=fingerprint,variant=variant,seed=seed,epoch=epoch,model=model.state_dict(),
                optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),best=best,best_state=best_state,
                best_epoch=best_epoch,since=since,history=history,torch_rng=torch.get_rng_state(),
                elapsed_seconds=elapsed_before+time.monotonic()-start,mu=mu,sd=sd,x_mu=xmu,x_sd=xsd,
                mps_rng=torch.mps.get_rng_state() if G['device']=='mps' else None),checkpoint)
    model.load_state_dict(best_state)
    vo=prediction(model,X,val);to=prediction(model,X,test)
    vp=expit(vo[:,0]);threshold=select_threshold(Y[val,0],vp)
    cal=platt_fit(Y[val,0],vp);ct=select_threshold(Y[val,0],calibrate(vp,cal))
    records={}
    for name,ix,out in [('validation',val,vo),('test',test,to)]:
        records[name]=make_prediction_record(split[name],Y[ix],out,mu,sd,spec,G['rho'][ix],threshold,cal,ct)
    json_save(records,folder/'predictions.json');json_save(history,folder/'loss_history.json')
    report=dict(variant=variant,partition_seed=partition,fold=fold,training_seed=seed,fingerprint=fingerprint,
        best_epoch=best_epoch,epochs_run=len(history),validation_selection_objective=best,calibration=cal,
        training_seconds=elapsed_before+time.monotonic()-start,parameter_count=sum(p.numel() for p in model.parameters()),
        fit_n=len(fit),validation_n=len(val),test_n=len(test))
    json_save(report,folder/'result.json')
    return str(folder.relative_to(G['run']))+f' best_epoch={best_epoch} epochs={len(history)}'

def rf_job(partition,fold,seed):
    folder=G['run']/f'partition{partition}_fold{fold}'/f'seed{seed}'/'forest'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return str(folder.relative_to(G['run']))+' cached'
    split=json.loads((G['run']/f'partition{partition}_fold{fold}'/'split.json').read_text())
    fit,val,test=[np.array([G['index'][x] for x in split[k]]) for k in ['fit','validation','test']]
    X=G['X'];Y=G['Y'];reg={k:[] for k in ['validation','test']};start=time.monotonic()
    for j in [1,2,3]:
        model=RandomForestRegressor(n_estimators=400,max_depth=12,random_state=seed,n_jobs=1).fit(X[fit],Y[fit,j])
        for k,ix in [('validation',val),('test',test)]:reg[k].append(model.predict(X[ix]))
    records={};calibrators={}
    for name,weight in [('rf',None),('rf_balanced','balanced')]:
        model=RandomForestClassifier(n_estimators=400,max_depth=12,random_state=seed,n_jobs=1,class_weight=weight).fit(X[fit],Y[fit,0])
        vp=model.predict_proba(X[val])[:,1];cal=platt_fit(Y[val,0],vp);calibrators[name]=cal
        threshold=select_threshold(Y[val,0],vp);ct=select_threshold(Y[val,0],calibrate(vp,cal));records[name]={}
        for k,ix in [('validation',val),('test',test)]:
            p=model.predict_proba(X[ix])[:,1];rp=reg[k]
            records[name][k]=dict(ids=split[k],truth=Y[ix].tolist(),probability=p.tolist(),calibrated_probability=calibrate(p,cal).tolist(),
                band_gap=rp[1].tolist(),formation_energy=rp[2].tolist(),analytic_density=G['rho'][ix].tolist(),
                learned_density=rp[0].tolist(),threshold=threshold,calibrated_threshold=ct)
    json_save(records,folder/'predictions.json')
    json_save(dict(partition_seed=partition,fold=fold,training_seed=seed,calibration=calibrators,
        fingerprint=G['manifest']['fingerprint'],training_seconds=time.monotonic()-start),folder/'result.json')
    return str(folder.relative_to(G['run']))+' completed'

def prepare(run,epochs,patience,device):
    old=json.loads((OLD/'experiment.json').read_text())
    assert all(digest(ROOT/p)==h for p,h in old['inputs'].items())
    config=dict(partition_seeds=[42,2024,7],folds=[1,2,3,4,5],initialization_seeds=[42,2024,7],
        variants=VARIANTS,epochs=epochs,patience=patience,device=device,deterministic='strict' if device=='cpu' else 'best effort; warn on unsupported deterministic MPS kernels',threads_per_worker=1,selection_objective='weighted BCE + 0.5 standardized gap MSE + standardized formation-energy MSE; density excluded for all variants',
        calibration='validation-only monotone Platt; L2 1e-4; a in [0,20], b in [-20,20]',
        configuration_selection='per-fold/per-initialization minimum common validation objective; no outer-test selection')
    inputs={**old['inputs'],str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
        str((OLD/'graphs.pt').relative_to(ROOT)):digest(OLD/'graphs.pt'),
        str((OLD/'analytic_density_check.json').relative_to(ROOT)):digest(OLD/'analytic_density_check.json')}
    for partition in config['partition_seeds']:
        for fold in config['folds']:
            source=OLD/f'partition{partition}_fold{fold}'/'split.json';inputs[str(source.relative_to(ROOT))]=digest(source)
            dest=run/source.parent.name;dest.mkdir(parents=True,exist_ok=True)
            split=json.loads(source.read_text())
            for a,b in [('fit','validation'),('fit','test'),('validation','test')]:
                assert not(set(split[a])&set(split[b])) and not(set(split[a+'_groups'])&set(split[b+'_groups']))
            (dest/'split.json').write_text(source.read_text())
    fingerprint=hashlib.sha256(json.dumps(dict(config=config,inputs=inputs),sort_keys=True).encode()).hexdigest()
    manifest=run/'experiment.json'
    if manifest.exists():assert json.loads(manifest.read_text())['fingerprint']==fingerprint,'Changed protocol: use new run name.'
    else:json_save(dict(config=config,inputs=inputs,fingerprint=fingerprint,platform=platform.platform(),torch_version=torch.__version__),manifest)
    return config

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-name',default='physics_hybrid_v1');parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=150);parser.add_argument('--patience',type=int,default=25)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    run=ROOT/'revision_ieee_access'/args.run_name;run.mkdir(exist_ok=True)
    if args.device=='mps':assert args.jobs==1,'Use one worker on the shared GPU.'
    config=prepare(run,args.epochs,args.patience,args.device)
    jobs=[]
    for partition in config['partition_seeds']:
        for fold in config['folds']:
            for seed in config['initialization_seeds']:
                for variant in VARIANTS:jobs.append((graph_job,(partition,fold,seed,variant)))
                jobs.append((rf_job,(partition,fold,seed)))
    if args.pilot:jobs=jobs[:6]
    start=time.monotonic();completed=0
    print(f'START {len(jobs)} jobs; {args.jobs} workers; {run}',flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs,initializer=initialize,initargs=(args.run_name,)) as pool:
        futures=[pool.submit(fn,*values) for fn,values in jobs]
        for future in as_completed(futures):
            try:message=future.result()
            except Exception:
                for pending in futures:pending.cancel()
                raise
            completed+=1
            print(f'COMPLETE {completed}/{len(jobs)} {message}; wall_minutes={(time.monotonic()-start)/60:.1f}',flush=True)
    print('ALL JOBS COMPLETE',flush=True)

if __name__=='__main__':main()
