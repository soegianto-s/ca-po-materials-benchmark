"""Fixed post hoc exclusion sensitivity; frozen training implementations reused."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import argparse,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
from pymatgen.core import Composition
import physics_hybrid_experiment as base
from capacity_control_experiment import CapacityModel,Original
ROOT=base.ROOT; RUN='descriptor_sensitivity_v1'; D=ROOT/'revision_ieee_access'/RUN
EXCLUDE={'mp-1214043','mp-1214415','mp-677016'}
VARIANTS=['graph_physics','graph_hybrid','graph_capacity']
def prepare():
 D.mkdir(exist_ok=True)
 old=ROOT/'revision_ieee_access/physics_hybrid_v1'; cap=ROOT/'revision_ieee_access/capacity_control_v1'
 m=json.loads((old/'experiment.json').read_text());cm=json.loads((cap/'experiment.json').read_text())
 inputs={**m['inputs'],**cm['inputs']}
 for rel,h in inputs.items():assert base.digest(ROOT/rel)==h,rel
 for p in [Path(__file__),D/'PROTOKOL.md',ROOT/'revision_ieee_access/submission_extension_v1/descriptor_coverage_audit.json']:
  inputs[str(p.relative_to(ROOT))]=base.digest(p)
 df=pd.read_csv(ROOT/'processed/15_ml_magpie_features.csv');idx=df.set_index('material_id');allids=set(idx.index)
 supported={x[5:] for x in df.columns if x.startswith('frac_')}
 uncovered={r.material_id for r in df.itertuples() if set(Composition(r.formula).as_dict())-supported}
 assert uncovered==EXCLUDE and len(allids-EXCLUDE)==508
 groups={x:Composition(idx.loc[x,'formula']).reduced_formula for x in allids}
 config=dict(m['config']);config['variants']={v:(dict(bounded=True,density_loss=False,hybrid=True) if v=='graph_capacity' else base.VARIANTS[v]) for v in VARIANTS}
 config.update(excluded_ids=sorted(EXCLUDE),unique_materials=508,unique_compositions=len({groups[x] for x in allids-EXCLUDE}),status='post hoc fixed exclusion sensitivity, no test selection',original_fingerprint=m['fingerprint'])
 splits={}
 for part in config['partition_seeds']:
  for fold in config['folds']:
   name=f'partition{part}_fold{fold}';original=json.loads((old/name/'split.json').read_text());new={}
   for k in ['fit','validation','test']:
    new[k]=[x for x in original[k] if x not in EXCLUDE]
    new[k+'_groups']=sorted({groups[x] for x in new[k]})
    assert set(new[k+'_groups'])<=set(original[k+'_groups'])
   assert set(sum([new[k] for k in ['fit','validation','test']],[]))==allids-EXCLUDE
   for a,b in [('fit','validation'),('fit','test'),('validation','test')]:assert not(set(new[a+'_groups'])&set(new[b+'_groups']))
   target=D/name/'split.json';target.parent.mkdir(exist_ok=True)
   if target.exists():assert json.loads(target.read_text())==new
   else:base.json_save(new,target)
   inputs[str(target.relative_to(ROOT))]=base.digest(target)
   splits[name]={k:len(new[k]) for k in ['fit','validation','test']}
 fingerprint=hashlib.sha256(json.dumps(dict(config=config,inputs=inputs),sort_keys=True).encode()).hexdigest()
 target=D/'experiment.json'
 if target.exists():assert json.loads(target.read_text())['fingerprint']==fingerprint,'Changed protocol; use a new run.'
 else:base.json_save(dict(config=config,inputs=inputs,fingerprint=fingerprint),target)
 base.json_save(dict(excluded=sorted(EXCLUDE),records=508,compositions=config['unique_compositions'],stable=int(idx.loc[sorted(allids-EXCLUDE),'is_stable'].sum()),splits=splits),D/'cohort_audit.json')
 return config

def initialize():
 base.initialize(RUN)
 base.VARIANTS['graph_capacity']=dict(bounded=True,density_loss=False,hybrid=True)
def job(part,fold,seed,variant):
 base.HybridModel=CapacityModel if variant=='graph_capacity' else Original
 return base.rf_job(part,fold,seed) if variant=='forest' else base.graph_job(part,fold,seed,variant)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=3);p.add_argument('--prepare-only',action='store_true');args=p.parse_args();c=prepare()
 if args.prepare_only:print('Protocol and 15 splits verified; 508 records,',c['unique_compositions'],'compositions');raise SystemExit
 tasks=[(part,fold,seed,v) for part in c['partition_seeds'] for fold in c['folds'] for seed in c['initialization_seeds'] for v in ['forest']+VARIANTS]
 print('START 135 graph fits + 45 forest groups',flush=True);start=time.monotonic()
 with ProcessPoolExecutor(max_workers=args.jobs,initializer=initialize) as pool:
  futures=[pool.submit(job,*t) for t in tasks]
  for n,f in enumerate(as_completed(futures),1):print(f'COMPLETE {n}/180 {f.result()}; wall_minutes={(time.monotonic()-start)/60:.1f}',flush=True)
 print('ALL SENSITIVITY FITS COMPLETE',flush=True)
