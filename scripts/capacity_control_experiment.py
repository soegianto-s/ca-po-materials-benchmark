"""Post hoc, parameter-count-matched graph-only control; old runs immutable."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import json,hashlib,shutil,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
import physics_hybrid_experiment as base
Original=base.HybridModel
ROOT=base.ROOT; RUN='capacity_control_v1'; VARIANT='graph_capacity'
class CapacityModel(Original):
    def __init__(self,hybrid=True):super().__init__(True)
    def forward(self,nodes,edges,src,dst,assignment,n_graphs,composition):
        h=self.embed(nodes)
        for conv,norm in zip(self.convs,self.norms):h=norm(conv(h,edges,src,dst))
        count=torch.bincount(assignment,minlength=n_graphs).to(h.dtype).unsqueeze(1)
        mean=torch.zeros((n_graphs,32),dtype=h.dtype,device=h.device).index_add_(0,assignment,h)/count
        second=torch.zeros_like(mean).index_add_(0,assignment,h.square())/count
        maximum=torch.stack([h[assignment==i].max(0).values for i in range(n_graphs)])
        # Exactly 90 graph-only features; no learned padding or unused parameters.
        graph_statistics=torch.cat([mean,second,maximum[:,:26]],dim=1)
        return self.head(torch.cat([mean,self.composition(graph_statistics)],dim=1))
def initialize():
    base.initialize(RUN)
    base.HybridModel=CapacityModel
    base.VARIANTS[VARIANT]=dict(bounded=True,density_loss=False,hybrid=True)
def prepare():
    old=ROOT/'revision_ieee_access/physics_hybrid_v1';run=ROOT/'revision_ieee_access'/RUN;run.mkdir(exist_ok=True)
    manifest=json.loads((old/'experiment.json').read_text())
    for p,h in manifest['inputs'].items():assert base.digest(ROOT/p)==h,p
    config=dict(manifest['config']);config['variants']={VARIANT:dict(bounded=True,density_loss=False,hybrid=True)}
    config['status']='post hoc extension after observing original results; no tuning on prior or new test scores'
    config['control']='Graph mean32, second moment32 and first26 max channels form90; identical 90-32-32 branch and64-32-4 head'
    inputs={**manifest['inputs'],str(Path(__file__).relative_to(ROOT)):base.digest(Path(__file__))}
    fingerprint=hashlib.sha256(json.dumps(dict(config=config,inputs=inputs),sort_keys=True).encode()).hexdigest()
    target=run/'experiment.json'
    if target.exists():assert json.loads(target.read_text())['fingerprint']==fingerprint
    else:base.json_save(dict(config=config,inputs=inputs,fingerprint=fingerprint,original_fingerprint=manifest['fingerprint']),target)
    for part in config['partition_seeds']:
        for fold in config['folds']:
            dst=run/f'partition{part}_fold{fold}';dst.mkdir(exist_ok=True)
            shutil.copyfile(old/dst.name/'split.json',dst/'split.json')
    return config
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=3);args=parser.parse_args()
    config=prepare();tasks=[(p,f,s,VARIANT) for p in config['partition_seeds'] for f in config['folds'] for s in config['initialization_seeds']]
    print('START 45 capacity controls',flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs,initializer=initialize) as pool:
        futures=[pool.submit(base.graph_job,*t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):print(f'COMPLETE {i}/45 {f.result()}',flush=True)
