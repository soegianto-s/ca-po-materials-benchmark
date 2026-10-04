"""Audit all 45 controls and compare paired OOF metrics without selection."""
import json,hashlib
from pathlib import Path
import numpy as np
from summarize_physics_hybrid import combine,measure
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'revision_ieee_access/capacity_control_v1';OLD=D.parent/'physics_hybrid_v1'
def main():
 manifest=json.loads((D/'experiment.json').read_text());config=manifest['config'];grid={};reports=[]
 for path,h in manifest['inputs'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
 for p in config['partition_seeds']:
  for s in config['initialization_seeds']:
   rows={k:[] for k in ['graph_capacity','graph_hybrid','graph_physics']}
   for f in config['folds']:
    folder=D/f'partition{p}_fold{f}'/f'seed{s}'/'graph_capacity'
    rep=json.loads((folder/'result.json').read_text());split=json.loads((folder.parent.parent/'split.json').read_text());pred=json.loads((folder/'predictions.json').read_text());hist=json.loads((folder/'loss_history.json').read_text())
    assert rep['fingerprint']==manifest['fingerprint'] and rep['parameter_count']==15940
    assert pred['test']['ids']==split['test'] and pred['validation']['ids']==split['validation']
    assert rep['validation_selection_objective']<=min(x['validation_selection_objective'] for x in hist)+1.01e-5
    assert min(pred['test']['band_gap'])>=0
    assert (OLD/f'partition{p}_fold{f}'/'split.json').read_bytes()==(folder.parent.parent/'split.json').read_bytes()
    reports.append(rep);rows['graph_capacity'].append(pred['test'])
    for model in ['graph_hybrid','graph_physics']:rows[model].append(json.loads((OLD/f'partition{p}_fold{f}'/f'seed{s}'/model/'predictions.json').read_text())['test'])
   grid[f'{p}/{s}']={k:measure(combine(v)) for k,v in rows.items()}
 assert len(reports)==45
 aggregate={model:{metric:float(np.mean([g[model][metric] for g in grid.values()])) for metric in next(iter(grid.values()))[model]} for model in rows}
 contrasts={}
 for a,b in [('graph_hybrid','graph_capacity'),('graph_capacity','graph_physics')]:
  contrasts[a+' minus '+b]={k:{'mean':float(np.mean([g[a][k]-g[b][k] for g in grid.values()])), 'nine_paired_values':[g[a][k]-g[b][k] for g in grid.values()]} for k in ['raw_ap','band_gap_mae','formation_energy_mae','calibrated_brier']}
 out=dict(completed_fits=45,parameter_count=15940,aggregate=aggregate,paired_contrasts=contrasts,grid=grid,reached_epoch_budget=sum(r['epochs_run']==150 for r in reports),median_best_epoch=float(np.median([r['best_epoch'] for r in reports])),interpretation='Post hoc fixed protocol; descriptive paired effects, not independent-replicate significance; graph statistics pooling differs from composition branch.')
 (D/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['grid','paired_contrasts']},indent=2))
if __name__=='__main__':main()
