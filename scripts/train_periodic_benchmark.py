"""Matched composition-disjoint graph/RF benchmark with resumable checkpoints.

Outer five-fold grouping repeated over partition seeds. Inner grouped holdout
selects the graph epoch and F2 threshold without touching outer test labels.
RF and graph fit on identical inner-fit IDs; all models see identical tests.
This is a CGCNN-inspired model, not an exact original-CGCNN reproduction.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from pymatgen.core import Composition, Structure
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef,
    brier_score_loss, r2_score, mean_absolute_error, mean_squared_error)
from periodic_graph_v3 import periodic_edges, PeriodicGatedConv

ROOT=Path(__file__).resolve().parents[1]
TARGETS=['density','band_gap','formation_energy_per_atom']

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_save(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp); os.replace(tmp,path)

def json_save(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); os.replace(tmp,path)

def node_features(el):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        props=[el.X,el.atomic_radius_calculated,el.average_ionic_radius]
    missing=[v is None or not np.isfinite(float(v)) for v in props]
    vals=[float(v) if not m else 0. for v,m in zip(props,missing)]
    return [el.Z/100.,float(el.atomic_mass)/250.,vals[0]/4.,vals[1]/3.,vals[2]/3.]+[float(m) for m in missing]

def make_graphs(df,cutoff):
    graphs=[]
    for row in df.itertuples():
        path=ROOT/'raw/02_materials_project_ca_p_o/struktur'/f'{row.material_id}.json'
        st=Structure.from_dict(json.loads(path.read_text()))
        if not st.composition.fractional_composition.almost_equals(Composition(row.formula).fractional_composition):
            raise ValueError(f'Composition mismatch: {row.material_id}')
        src,dst,images,distance,edges=periodic_edges(st,cutoff,8)
        graphs.append(dict(nodes=torch.tensor([node_features(site.specie) for site in st],dtype=torch.float32),
            edges=torch.from_numpy(edges),src=torch.from_numpy(src),dst=torch.from_numpy(dst)))
    return graphs

def batch(graphs,indices):
    ns=[]; es=[]; ss=[]; ds=[]; assignment=[]; offset=0
    for j,i in enumerate(indices):
        g=graphs[int(i)]; n=len(g['nodes'])
        ns.append(g['nodes']); es.append(g['edges']); ss.append(g['src']+offset); ds.append(g['dst']+offset)
        assignment.append(torch.full((n,),j,dtype=torch.long)); offset+=n
    return torch.cat(ns),torch.cat(es),torch.cat(ss),torch.cat(ds),torch.cat(assignment),len(indices)

class GraphModel(nn.Module):
    def __init__(self,width=32,layers=2,dropout=.1):
        super().__init__()
        self.embed=nn.Linear(8,width)
        self.convs=nn.ModuleList([PeriodicGatedConv(width,8) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Dropout(dropout),nn.Linear(width,4))

    def forward(self,nodes,edges,src,dst,assignment,n_graphs):
        h=self.embed(nodes)
        for conv,norm in zip(self.convs,self.norms): h=norm(conv(h,edges,src,dst))
        pooled=torch.zeros((n_graphs,h.shape[1]),device=h.device,dtype=h.dtype)
        pooled.index_add_(0,assignment,h)
        counts=torch.bincount(assignment,minlength=n_graphs).to(h.dtype).unsqueeze(1)
        return self.head(pooled/counts)

def loss_fn(out,y,mu,sd,pos_weight):
    cls=nn.functional.binary_cross_entropy_with_logits(out[:,0],y[:,0],pos_weight=pos_weight)
    reg=((out[:,1:]-(y[:,1:]-mu)/sd)**2).mean(0)
    return cls+.5*reg[0]+.5*reg[1]+reg[2]

def predict(model,graphs,indices,batch_size):
    model.eval(); result=[]
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            result.append(model(*batch(graphs,indices[start:start+batch_size])).cpu().numpy())
    return np.concatenate(result)

def select_threshold(y,p):
    grid=np.arange(.05,.951,.025)
    scores=[fbeta_score(y,p>=t,beta=2,zero_division=0) for t in grid]
    # Tie break toward higher threshold; determined before outer-test scoring.
    return float(grid[np.flatnonzero(np.isclose(scores,np.max(scores)))[-1]])

def metrics(y,p,reg,threshold):
    pred=p>=threshold
    result=dict(roc_auc=float(roc_auc_score(y[:,0],p)),average_precision=float(average_precision_score(y[:,0],p)),
        balanced_accuracy=float(balanced_accuracy_score(y[:,0],pred)),precision=float(precision_score(y[:,0],pred,zero_division=0)),
        recall=float(recall_score(y[:,0],pred,zero_division=0)),f1=float(f1_score(y[:,0],pred,zero_division=0)),
        f2=float(fbeta_score(y[:,0],pred,beta=2,zero_division=0)),mcc=float(matthews_corrcoef(y[:,0],pred)),
        brier=float(brier_score_loss(y[:,0],p)),threshold=float(threshold))
    for j,name in enumerate(TARGETS):
        result[name]=dict(r2=float(r2_score(y[:,j+1],reg[:,j])),mae=float(mean_absolute_error(y[:,j+1],reg[:,j])),
                         rmse=float(np.sqrt(mean_squared_error(y[:,j+1],reg[:,j]))))
    return result

def train_fold(graphs,Y,fit,val,config,folder,fingerprint):
    checkpoint=folder/'checkpoint.pt'
    random.seed(config['training_seed']); np.random.seed(config['training_seed']); torch.manual_seed(config['training_seed'])
    model=GraphModel(config['width'],config['layers'],config['dropout'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,factor=.5)
    mu=torch.from_numpy(Y[fit,1:].mean(0)); sd=torch.from_numpy(np.maximum(Y[fit,1:].std(0),1e-8))
    positive=int(Y[fit,0].sum()); pos_weight=torch.tensor((len(fit)-positive)/positive)
    best=float('inf'); best_state=None; best_epoch=0; history=[]; since=0; start_epoch=1; elapsed_before=0.
    if checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
        assert saved['fingerprint']==fingerprint
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        best=saved['best']; best_state=saved['best_state']; best_epoch=saved['best_epoch']; history=saved['history']; since=saved['since']
        start_epoch=saved['epoch']+1; elapsed_before=saved['elapsed_seconds']; torch.set_rng_state(saved['torch_rng'])
    start=time.monotonic()
    for epoch in range(start_epoch,config['epochs']+1):
        if since>=config['patience']: break
        model.train(); running=0.; order=np.random.default_rng(config['training_seed']+epoch).permutation(fit)
        for offset in range(0,len(order),config['batch_size']):
            inds=order[offset:offset+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            output=model(*batch(graphs,inds))
            loss=loss_fn(output,torch.from_numpy(Y[inds]),mu,sd,pos_weight)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            running+=float(loss.detach())*len(inds)
        out=predict(model,graphs,val,config['batch_size'])
        vloss=float(loss_fn(torch.from_numpy(out),torch.from_numpy(Y[val]),mu,sd,pos_weight))
        scheduler.step(vloss); history.append(dict(epoch=epoch,train_loss=running/len(fit),val_loss=vloss,lr=optimizer.param_groups[0]['lr']))
        if vloss<best-1e-5:
            best=vloss; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; since=0
        else: since+=1
        elapsed=elapsed_before+time.monotonic()-start
        if epoch%5==0 or since>=config['patience'] or epoch==config['epochs']:
            atomic_save(dict(fingerprint=fingerprint,epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                best=best,best_state=best_state,best_epoch=best_epoch,history=history,since=since,
                elapsed_seconds=elapsed,torch_rng=torch.get_rng_state(),mu=mu,sd=sd),checkpoint)
        if epoch%10==0 or epoch==1: print(folder.name,'epoch',epoch,'val',round(vloss,4),'best_epoch',best_epoch,'seconds',round(elapsed),flush=True)
    model.load_state_dict(best_state)
    json_save(history,folder/'loss_history.json')
    return model,mu.numpy(),sd.numpy(),best_epoch,elapsed_before+time.monotonic()-start

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=150); parser.add_argument('--patience',type=int,default=25)
    parser.add_argument('--partition-seeds',default='42,2024,7'); parser.add_argument('--folds',default='1,2,3,4,5')
    parser.add_argument('--run-name',default='periodic_benchmark'); parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(); torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    config=dict(epochs=args.epochs,patience=args.patience,partition_seeds=[int(x) for x in args.partition_seeds.split(',')],
        folds=[int(x) for x in args.folds.split(',')],training_seed=42,batch_size=16,width=32,layers=2,dropout=.1,lr=.001,
        cutoff=5.,rbf=8,threads=args.threads,inner_splits=5,device='cpu',torch_version=torch.__version__)
    df=pd.read_csv(ROOT/'processed/15_ml_magpie_features.csv')
    feat=[c for c in df if c.startswith(('frac_','wmean_','wstd_','min_','max_','range_'))]
    X=df[feat].to_numpy(dtype=np.float64); Y=df[['is_stable']+TARGETS].to_numpy(dtype=np.float32)
    groups=np.array([Composition(f).reduced_formula for f in df.formula])
    assert np.isfinite(X).all() and np.isfinite(Y).all() and df.material_id.is_unique
    inputs={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'scripts/periodic_graph_v3.py',ROOT/'processed/15_ml_magpie_features.csv']}
    inputs.update({str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'raw/02_materials_project_ca_p_o/struktur').glob('*.json'))})
    fingerprint=hashlib.sha256(json.dumps(dict(config=config,inputs=inputs),sort_keys=True).encode()).hexdigest()
    folder=ROOT/'revision_ieee_access'/args.run_name; folder.mkdir(exist_ok=True)
    manifest=folder/'experiment.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())['fingerprint']==fingerprint,'Configuration/data/code changed: use a new run name.'
    else: json_save(dict(fingerprint=fingerprint,config=config,inputs=inputs,platform=platform.platform(),n=len(df),groups=len(set(groups))),manifest)
    cache=folder/'graphs.pt'
    if cache.exists(): graphs=torch.load(cache,weights_only=False)
    else:
        graphs=make_graphs(df,config['cutoff']); atomic_save(graphs,cache)
    print('RUN',folder,'nodes',sum(len(g['nodes']) for g in graphs),'edges',sum(len(g['src']) for g in graphs),flush=True)
    for partition_seed in config['partition_seeds']:
        outer=StratifiedGroupKFold(5,shuffle=True,random_state=partition_seed)
        for fold,(tr,test) in enumerate(outer.split(X,Y[:,0],groups),1):
            if fold not in config['folds']: continue
            sub=folder/f'partition{partition_seed}_fold{fold}'; sub.mkdir(exist_ok=True)
            if (sub/'result.json').exists(): print(sub.name,'already completed',flush=True); continue
            inner=StratifiedGroupKFold(5,shuffle=True,random_state=partition_seed+1000)
            fi,vi=next(inner.split(X[tr],Y[tr,0],groups[tr])); fit,val=tr[fi],tr[vi]
            assert not(set(groups[fit])&set(groups[val]) or set(groups[tr])&set(groups[test]))
            assert all(len(np.unique(Y[ix,0]))==2 for ix in [fit,val,test])
            json_save(dict(fit=df.material_id.iloc[fit].tolist(),validation=df.material_id.iloc[val].tolist(),test=df.material_id.iloc[test].tolist(),
                fit_groups=sorted(set(groups[fit])),validation_groups=sorted(set(groups[val])),test_groups=sorted(set(groups[test]))),sub/'split.json')
            model,mu,sd,best_epoch,seconds=train_fold(graphs,Y,fit,val,config,sub,fingerprint)
            va=predict(model,graphs,val,config['batch_size']); te=predict(model,graphs,test,config['batch_size'])
            sig=lambda x:1/(1+np.exp(-np.clip(x,-50,50)))
            gp=sig(te[:,0]); gr=te[:,1:]*sd+mu; gt=select_threshold(Y[val,0],sig(va[:,0]))
            models={'periodic_graph':(gp,gr,gt)}
            for name,weight in [('rf',None),('rf_balanced','balanced')]:
                clf=RandomForestClassifier(n_estimators=400,max_depth=12,class_weight=weight,random_state=42,n_jobs=args.threads)
                clf.fit(X[fit],Y[fit,0]); vp=clf.predict_proba(X[val])[:,1]; tp=clf.predict_proba(X[test])[:,1]
                if name=='rf':
                    rp=[]
                    for j in range(3):
                        reg=RandomForestRegressor(n_estimators=400,max_depth=12,random_state=42,n_jobs=args.threads)
                        reg.fit(X[fit],Y[fit,j+1]); rp.append(reg.predict(X[test]))
                    rf_reg=np.stack(rp,axis=1)
                models[name]=(tp,rf_reg,select_threshold(Y[val,0],vp))
            models['dummy']=(np.full(len(test),Y[fit,0].mean()),np.tile(Y[fit,1:].mean(0),(len(test),1)),.5)
            result=dict(partition_seed=partition_seed,fold=fold,fit_n=len(fit),validation_n=len(val),test_n=len(test),
                graph_best_epoch=best_epoch,graph_training_seconds=seconds,models={})
            predictions={}
            for name,(prob,reg,threshold) in models.items():
                result['models'][name]=dict(fixed=metrics(Y[test],prob,reg,.5),validation_selected=metrics(Y[test],prob,reg,threshold))
                predictions[name]=dict(ids=df.material_id.iloc[test].tolist(),truth=Y[test].tolist(),probability=prob.tolist(),regression=reg.tolist(),threshold=threshold)
            json_save(predictions,sub/'predictions.json'); json_save(result,sub/'result.json')
            print('COMPLETE',sub.name,'best_epoch',best_epoch,'graph AP',round(result['models']['periodic_graph']['fixed']['average_precision'],4),flush=True)
    print('ALL REQUESTED FOLDS COMPLETE',flush=True)

if __name__=='__main__': main()
