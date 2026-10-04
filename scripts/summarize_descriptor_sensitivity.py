"""Separate test-population sensitivity from full exclusion retraining effects."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from summarize_physics_hybrid import measure
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'revision_ieee_access';D=R/'descriptor_sensitivity_v1'
OLD=R/'physics_hybrid_v1';CAP=R/'capacity_control_v1'
EXCLUDE={'mp-1214043','mp-1214415','mp-677016'}
MODELS=['rf','rf_balanced','graph_baseline','graph_gap','graph_nodensity','graph_physics','graph_hybrid','validation_selected_graph','graph_capacity']
RETRAIN=['rf','rf_balanced','graph_physics','graph_hybrid','graph_capacity']
METRICS=['raw_ap','band_gap_mae','formation_energy_mae','calibrated_brier','negative_gap_count']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pool(rows,expected):
 keys=['ids','truth','probability','calibrated_probability','band_gap','formation_energy','analytic_density','threshold','calibrated_threshold'];out={k:[] for k in keys}
 for row in rows:
  ix=[i for i,x in enumerate(row['ids']) if x not in EXCLUDE]
  for k in keys:
   out[k].extend([row[k]]*len(ix) if k in ['threshold','calibrated_threshold'] and not isinstance(row[k],list) else [row[k][i] for i in ix])
 assert len(out['ids'])==expected and len(set(out['ids']))==expected
 assert not EXCLUDE&set(out['ids'])
 return out

def aggregate(grid,models):
 return {m:{k:dict(mean=float(np.mean([g[m][k] for g in grid.values()])),nine_values=[g[m][k] for g in grid.values()]) for k in next(iter(grid.values()))[m]} for m in models}
def contrasts(grid,models):
 out={}
 for a,b in [('graph_hybrid','rf'),('graph_hybrid','graph_physics'),('graph_hybrid','graph_capacity'),('graph_capacity','rf')]:
  if a not in models or b not in models:continue
  out[a+' minus '+b]={k:dict(mean=float(np.mean([g[a][k]-g[b][k] for g in grid.values()])),nine_values=[g[a][k]-g[b][k] for g in grid.values()]) for k in METRICS}
 return out

def main(require_complete=False):
 manifest=read(D/'experiment.json');c=manifest['config']
 for rel,h in manifest['inputs'].items():assert sha(ROOT/rel)==h,rel
 oldpooled=read(OLD/'oof_predictions.json');grid={};filtered={};oldgrid=read(OLD/'summary.json')['per_oof_run'];capgrid=read(CAP/'summary.json')['grid']
 for p in c['partition_seeds']:
  for s in c['initialization_seeds']:
   key=f'{p}/{s}';filtered[key]={};grid[key]={}
   for m in MODELS:
    rows=([read(CAP/f'partition{p}_fold{f}'/f'seed{s}'/m/'predictions.json')['test'] for f in c['folds']] if m=='graph_capacity' else [oldpooled[key][m]])
    row=pool(rows,508);assert sum(y[0] for y in row['truth'])==94
    filtered[key][m]=row;grid[key][m]=measure(row)
   assert all(filtered[key][m]['ids']==filtered[key]['rf']['ids'] for m in MODELS if m!='graph_capacity')
 original={k:{**oldgrid[k],'graph_capacity':capgrid[k]['graph_capacity']} for k in grid}
 changes={m:{metric:{'mean':float(np.mean([grid[k][m][metric]-original[k][m][metric] for k in grid])), 'nine_values':[grid[k][m][metric]-original[k][m][metric] for k in grid]} for metric in METRICS} for m in MODELS}
 initial=dict(status='complete',training_changed=False,unique_materials=508,stable=94,original_aggregate=aggregate(original,MODELS),aggregate=aggregate(grid,MODELS),per_oof_run=grid,change_from_511=changes,paired_contrasts=contrasts(grid,MODELS),interpretation='Only test population changed; original fit, validation selection and calibration still include excluded records when assigned there.')
 (D/'evaluation_only_summary.json').write_text(json.dumps(initial,indent=2))
 reports=list(D.glob('partition*/seed*/graph*/result.json'));forests=list(D.glob('partition*/seed*/forest/result.json'))
 status=dict(completed_graph_fits=len(reports),expected_graph_fits=135,completed_forest_groups=len(forests),expected_forest_groups=45,full_retraining_complete=len(reports)==135 and len(forests)==45)
 (D/'progress.json').write_text(json.dumps(status,indent=2))
 if not status['full_retraining_complete']:
  if require_complete:raise RuntimeError(status)
  print(json.dumps(status));return
 df=pd.read_csv(ROOT/'processed/15_ml_magpie_features.csv');index={v:i for i,v in enumerate(df.material_id)};feat=[x for x in df if x.startswith(('frac_','wmean_','wstd_','min_','max_','range_'))];X=df[feat].to_numpy(dtype=np.float64);Y=df[['is_stable','density','band_gap','formation_energy_per_atom']].to_numpy(dtype=np.float32)
 newgrid={};pooled={};stats=[]
 for p in c['partition_seeds']:
  for s in c['initialization_seeds']:
   key=f'{p}/{s}';rows={m:[] for m in RETRAIN}
   for f in c['folds']:
    fold=D/f'partition{p}_fold{f}';split=read(fold/'split.json');fit=np.array([index[x] for x in split['fit']]);folder=fold/f'seed{s}'
    for m in RETRAIN:
     md='forest' if m.startswith('rf') else m;result=read(folder/md/'result.json');pr=read(folder/md/'predictions.json');pr=pr[m] if md=='forest' else pr
     assert result['fingerprint']==manifest['fingerprint']
     assert (result['partition_seed'],result['fold'],result['training_seed'])==(p,f,s)
     if md!='forest':
      assert result['variant']==m
      assert result['parameter_count']==(10948 if m=='graph_physics' else 15940)
     for subset in ['validation','test']:
      assert pr[subset]['ids']==split[subset];assert not EXCLUDE&set(pr[subset]['ids'])
      np.testing.assert_array_equal(np.asarray(pr[subset]['truth'],dtype=np.float32),Y[[index[x] for x in split[subset]]])
      for field in ['truth','probability','calibrated_probability','band_gap','formation_energy']:assert np.isfinite(pr[subset][field]).all()
     if md!='forest':
      hist=read(folder/md/'loss_history.json');assert len(hist)==result['epochs_run']
      assert result['validation_selection_objective']<=min(h['validation_selection_objective'] for h in hist)+1.01e-5
      assert min(pr['test']['band_gap'])>=0
      cp=torch.load(folder/md/'checkpoint.pt',map_location='cpu',weights_only=False)
      np.testing.assert_allclose(cp['x_mu'],X[fit].mean(0),rtol=0,atol=1e-12)
      expected_sd=X[fit].std(0);expected_sd=np.where(expected_sd<1e-8,1.,expected_sd)
      np.testing.assert_allclose(cp['x_sd'],expected_sd,rtol=0,atol=1e-12)
      np.testing.assert_allclose(cp['mu'].numpy(),Y[fit,1:].mean(0),rtol=0,atol=1e-7)
      assert cp['fingerprint']==manifest['fingerprint'];stats.append(result)
     rows[m].append(pr['test'])
   pooled[key]={m:pool(v,508) for m,v in rows.items()}
   for m in RETRAIN:assert pooled[key][m]['ids']==filtered[key][m]['ids'] and pooled[key][m]['truth']==filtered[key][m]['truth']
   newgrid[key]={m:measure(v) for m,v in pooled[key].items()}
 effects={m:{metric:{'mean':float(np.mean([newgrid[k][m][metric]-grid[k][m][metric] for k in grid])), 'nine_values':[newgrid[k][m][metric]-grid[k][m][metric] for k in grid]} for metric in METRICS} for m in RETRAIN}
 result=dict(**status,fingerprint=manifest['fingerprint'],unique_materials=508,unique_compositions=418,stable=94,aggregate=aggregate(newgrid,RETRAIN),per_oof_run=newgrid,retraining_minus_original_on_same_508=effects,paired_contrasts=contrasts(newgrid,RETRAIN),training_diagnostics={m:dict(n_fits=sum(x['variant']==m for x in stats),budget_reached=sum(x['variant']==m and x['epochs_run']==150 for x in stats),median_best_epoch=float(np.median([x['best_epoch'] for x in stats if x['variant']==m]))) for m in RETRAIN if m.startswith('graph')},audits='hashes, same surviving split memberships, 508 common OOF identities, finite predictions, fit-only scaling, validation-only epoch selection, nonnegative bounded heads passed',external_results='remain original 511-trained models; not reevaluated as part of exclusion sensitivity')
 (D/'summary.json').write_text(json.dumps(result,indent=2));(D/'oof_predictions.json').write_text(json.dumps(pooled))
 lines=['# Hasil uji sensitivitas deskriptor','', '135 pelatihan graf dan 45 kelompok RF selesai. Tiga ID dengan Pd/Eu dikeluarkan dari fit, validasi, dan test tanpa membagi ulang sampel lain. Kohort: 508 material, 418 komposisi, 94 label stabil.','', '| Model | AP awal 511 | AP lama pada 508 | AP retrain 508 | MAE gap lama pada 508 | MAE gap retrain 508 |','|---|---:|---:|---:|---:|---:|']
 for m in RETRAIN:
  vals=[initial['original_aggregate'][m]['raw_ap']['mean'],initial['aggregate'][m]['raw_ap']['mean'],result['aggregate'][m]['raw_ap']['mean'],initial['aggregate'][m]['band_gap_mae']['mean'],result['aggregate'][m]['band_gap_mae']['mean']]
  lines.append('| '+m+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
 lines+=['','Hasil ini adalah sensitivitas post hoc pada domain unsur yang didukung, bukan pembuktian akurasi Pd/Eu atau signifikansi statistik. Model transfer eksternal tetap model utama 511. Nilai berpasangan lengkap, Brier, MAE energi, dan pelanggaran fisika tersedia di summary.json.']
 (D/'HASIL_SENSITIVITAS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({m:{k:result['aggregate'][m][k]['mean'] for k in METRICS} for m in RETRAIN},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args();main(a.require_complete)
