"""Audit all completed runs and compare physics ablations without test selection."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ieee-hybrid-mpl')
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import (average_precision_score,roc_auc_score,brier_score_loss,log_loss,
    precision_score,recall_score,f1_score,fbeta_score,matthews_corrcoef,balanced_accuracy_score,
    r2_score,mean_absolute_error,mean_squared_error)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'revision_ieee_access/physics_hybrid_v1'
VARIANTS=['graph_baseline','graph_gap','graph_nodensity','graph_physics','graph_hybrid']
MODELS=['rf','rf_balanced']+VARIANTS+['validation_selected_graph']
NAMES={'rf':'RF','rf_balanced':'Weighted RF','graph_baseline':'Graph baseline','graph_gap':'Nonnegative gap',
    'graph_nodensity':'No density loss','graph_physics':'Combined constraints','graph_hybrid':'Composition + graph',
    'validation_selected_graph':'Validation-selected graph'}
CONTRASTS=[('graph_gap','graph_baseline'),('graph_nodensity','graph_baseline'),
    ('graph_physics','graph_nodensity'),('graph_physics','graph_gap'),('graph_hybrid','graph_physics')]

def calibration_bins(y,p):
    rows=[]
    for j in range(10):
        ix=(p>=j/10)&(p<(j+1)/10 if j<9 else p<=1)
        rows.append(dict(lower=j/10,upper=(j+1)/10,n=int(ix.sum()),
            probability=float(p[ix].mean()) if ix.any() else None,
            observed=float(y[ix].mean()) if ix.any() else None))
    return rows

def measure(row):
    y=np.array(row['truth']);result={}
    for mode,key,tkey in [('raw','probability','threshold'),('calibrated','calibrated_probability','calibrated_threshold')]:
        p=np.array(row[key]);threshold=np.array(row[tkey]);bins=calibration_bins(y[:,0],p)
        d=dict(ap=float(average_precision_score(y[:,0],p)),roc_auc=float(roc_auc_score(y[:,0],p)),
            brier=float(brier_score_loss(y[:,0],p)),log_loss=float(log_loss(y[:,0],np.clip(p,1e-6,1-1e-6),labels=[0,1])),
            ece=sum(b['n']/len(y)*abs(b['probability']-b['observed']) for b in bins if b['n']))
        for name,pred in [('fixed',p>=.5),('selected',p>=threshold)]:
            for metric,value in dict(precision=precision_score(y[:,0],pred,zero_division=0),recall=recall_score(y[:,0],pred,zero_division=0),
                f1=f1_score(y[:,0],pred,zero_division=0),f2=fbeta_score(y[:,0],pred,beta=2,zero_division=0),
                mcc=matthews_corrcoef(y[:,0],pred),balanced_accuracy=balanced_accuracy_score(y[:,0],pred),selection_fraction=np.mean(pred)).items():
                d[name+'_'+metric]=float(value)
        result.update({mode+'_'+k:v for k,v in d.items()})
    for j,key in [(2,'band_gap'),(3,'formation_energy')]:
        p=np.array(row[key]);assert np.isfinite(p).all()
        result.update({key+'_mae':float(mean_absolute_error(y[:,j],p)),key+'_rmse':float(np.sqrt(mean_squared_error(y[:,j],p))),key+'_r2':float(r2_score(y[:,j],p))})
    gap=np.array(row['band_gap']);result.update(negative_gap_count=int((gap<0).sum()),negative_gap_rate=float(np.mean(gap<0)))
    return result

def combine(rows):
    out={k:[] for k in ['ids','truth','probability','calibrated_probability','band_gap','formation_energy','analytic_density','threshold','calibrated_threshold']}
    for row in rows:
        for k in out:
            out[k].extend([row[k]]*len(row['ids']) if k in ['threshold','calibrated_threshold'] else row[k])
    assert len(out['ids'])==511 and len(set(out['ids']))==511
    assert sum(x[0] for x in out['truth'])==96
    return out

def summarize_grid(grid,partitions,seeds):
    result={}
    for model in MODELS:
        result[model]={}
        for metric in grid[f'{partitions[0]}/{seeds[0]}'][model]:
            values=np.array([[grid[f'{p}/{s}'][model][metric] for s in seeds] for p in partitions])
            result[model][metric]=dict(mean=float(values.mean()),sd_all_nine_descriptive=float(values.std(ddof=1)),
                sd_partition_means=float(values.mean(1).std(ddof=1)),mean_within_partition_initialization_sd=float(values.std(1,ddof=1).mean()),
                by_partition_and_initialization=values.tolist())
    return result

def main():
    manifest=json.loads((RUN/'experiment.json').read_text());config=manifest['config']
    for path,h in manifest['inputs'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    partitions=config['partition_seeds'];seeds=config['initialization_seeds'];grid={};all_pooled={};reports=[];selected=[]
    for partition in partitions:
        for seed in seeds:
            rows={model:[] for model in MODELS}
            for fold in config['folds']:
                base=RUN/f'partition{partition}_fold{fold}'
                split=json.loads((base/'split.json').read_text())
                for a,b in [('fit','validation'),('fit','test'),('validation','test')]:
                    assert not(set(split[a])&set(split[b])) and not(set(split[a+'_groups'])&set(split[b+'_groups']))
                graph_rows={};val_scores={}
                for model in VARIANTS:
                    folder=base/f'seed{seed}'/model
                    report=json.loads((folder/'result.json').read_text());pred=json.loads((folder/'predictions.json').read_text())
                    assert report['fingerprint']==manifest['fingerprint']
                    assert (report['partition_seed'],report['fold'],report['training_seed'],report['variant'])==(partition,fold,seed,model)
                    history=json.loads((folder/'loss_history.json').read_text())
                    assert len(history)==report['epochs_run']
                    assert report['validation_selection_objective']<=min(h['validation_selection_objective'] for h in history)+1.01e-5
                    assert pred['test']['ids']==split['test'] and pred['validation']['ids']==split['validation']
                    assert np.isfinite(report['validation_selection_objective'])
                    if config['variants'][model]['bounded']:assert min(pred['test']['band_gap'])>=0
                    rows[model].append(pred['test']);graph_rows[model]=pred['test'];val_scores[model]=report['validation_selection_objective'];reports.append(report)
                choice=min(VARIANTS,key=lambda m:(val_scores[m],VARIANTS.index(m)))
                rows['validation_selected_graph'].append(graph_rows[choice])
                selected.append(dict(partition=partition,fold=fold,initialization_seed=seed,variant=choice,validation_objective=val_scores[choice]))
                forest=json.loads((base/f'seed{seed}'/'forest'/'predictions.json').read_text())
                rf_report=json.loads((base/f'seed{seed}'/'forest'/'result.json').read_text())
                assert rf_report['fingerprint']==manifest['fingerprint']
                for model in ['rf','rf_balanced']:
                    assert forest[model]['test']['ids']==split['test']
                    if seed==42:
                        old=json.loads((ROOT/'revision_ieee_access/periodic_benchmark_v1'/f'partition{partition}_fold{fold}'/'predictions.json').read_text())[model]
                        assert old['ids']==forest[model]['test']['ids']
                        assert np.max(np.abs(np.array(old['probability'])-np.array(forest[model]['test']['probability'])))<1e-8
                    rows[model].append(forest[model]['test'])
            key=f'{partition}/{seed}';grid[key]={};all_pooled[key]={}
            for model in MODELS:
                pooled=combine(rows[model]);all_pooled[key][model]=pooled;grid[key][model]=measure(pooled)
                assert pooled['truth']==all_pooled[key]['rf']['truth']
    assert len(reports)==225
    aggregate=summarize_grid(grid,partitions,seeds);contrasts={}
    for a,b in CONTRASTS:
        contrasts[a+' minus '+b]={}
        for metric in aggregate[a]:
            diffs=np.array(aggregate[a][metric]['by_partition_and_initialization'])-np.array(aggregate[b][metric]['by_partition_and_initialization'])
            contrasts[a+' minus '+b][metric]=dict(mean=float(diffs.mean()),sd_descriptive=float(diffs.std(ddof=1)),values=diffs.tolist())
    training_diagnostics={}
    for model in VARIANTS:
        rs=[r for r in reports if r['variant']==model]
        training_diagnostics[model]=dict(n_fits=len(rs),
            parameter_counts=sorted(set(r['parameter_count'] for r in rs)),
            reached_epoch_budget=sum(r['epochs_run']==config['epochs'] for r in rs),
            selected_in_final_ten_epochs=sum(r['best_epoch']>=config['epochs']-10 for r in rs),
            median_selected_epoch=float(np.median([r['best_epoch'] for r in rs])))
    summary=dict(completed_graph_fits=len(reports),completed_forest_groups=45,unique_materials=511,
        independent_material_count_is_not_4599=True,fingerprint=manifest['fingerprint'],aggregate=aggregate,per_oof_run=grid,
        paired_contrasts=contrasts,validation_selected_configurations=selected,
        graph_training_seconds_sum=sum(r['training_seconds'] for r in reports),graph_run_reports=reports,
        training_diagnostics=training_diagnostics)
    (RUN/'summary.json').write_text(json.dumps(summary,indent=2))
    (RUN/'oof_predictions.json').write_text(json.dumps(all_pooled))
    def mean(model,metric):return aggregate[model][metric]['mean']
    lines=['# Hasil eksperimen fisika–AI dan komposisi–struktur','',
        '225 pelatihan graf dan 45 kelompok RF selesai. Lima varian graf, 15 split identik, tiga seed inisialisasi.',
        'Setiap metrik berasal dari 511 prediksi OOF pada satu pasangan seed partisi/inisialisasi, lalu dirata-ratakan pada sembilan pasangan. SD deskriptif tidak dianggap confidence interval.',
        'Pemilihan epoch, kalibrasi, threshold, dan varian graph terpilih hanya menggunakan inner validation. Semua varian tetap dilaporkan.',
        '', '| Model | AP mentah | F2 threshold validasi | MAE gap | MAE energi | Band gap negatif (rerata jumlah/511) |',
        '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        lines.append('| '+NAMES[model]+' | '+' | '.join(f'{mean(model,k):.4f}' for k in ['raw_ap','raw_selected_f2','band_gap_mae','formation_energy_mae','negative_gap_count'])+' |')
    lines+=['','| Model | Brier mentah | Brier kalibrasi | Log-loss mentah | Log-loss kalibrasi | ECE mentah | ECE kalibrasi |','|---|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        lines.append('| '+NAMES[model]+' | '+' | '.join(f'{mean(model,k):.4f}' for k in ['raw_brier','calibrated_brier','raw_log_loss','calibrated_log_loss','raw_ece','calibrated_ece'])+' |')
    lines+=['','## Kontras berpasangan','',
        '| Perubahan | Δ AP (positif lebih baik) | Δ MAE gap (negatif lebih baik) | Δ MAE energi (negatif lebih baik) |','|---|---:|---:|---:|']
    for a,b in CONTRASTS:
        d=contrasts[a+' minus '+b]
        lines.append('| '+NAMES[a]+' − '+NAMES[b]+' | '+' | '.join(f'{d[k]["mean"]:+.4f}' for k in ['raw_ap','band_gap_mae','formation_energy_mae'])+' |')
    counts={m:sum(r['variant']==m for r in selected) for m in VARIANTS}
    lines+=['','Pemilihan varian oleh validasi (45 fold/seed): '+str(counts)+'.',
        '', 'Acuan selalu-positif: precision 0,188, recall 1,000, F2 0,536.',
        '', 'Densitas dihitung analitis pada seluruh model; tidak diklaim sebagai peningkatan akurasi AI. Head densitas tanpa loss tidak digunakan.',
        'Baseline baru tidak identik dengan skor lama karena kriteria seleksi epoch sekarang seragam tanpa densitas dan memakai tiga inisialisasi.',
        '', 'Batas: satu snapshot, tidak ada validasi eksternal/scaffold; cabang komposisi menambah parameter; ReLU memiliki wilayah gradien nol; penggunaan inner validation yang sama untuk beberapa keputusan dapat menambah ketidakstabilan. Tidak ada klaim kausal atau signifikansi dari sembilan estimasi OOF yang bergantung.']
    lines+=['', '## Diagnostik anggaran pelatihan', '',
        '| Varian | Parameter | Mencapai batas epoch / 45 | Epoch terpilih dalam 10 epoch terakhir / 45 | Median epoch terpilih |',
        '|---|---:|---:|---:|---:|']
    for model,d in training_diagnostics.items():
        lines.append(f"| {NAMES[model]} | {d['parameter_counts']} | {d['reached_epoch_budget']} | {d['selected_in_final_ten_epochs']} | {d['median_selected_epoch']:.0f} |")
    lines+=['', 'Mencapai batas epoch tidak membuktikan konvergensi. Pemilihan epoch mendekati batas dilaporkan sebagai keterbatasan anggaran optimasi; protokol tidak diubah berdasarkan hasil test.']
    (RUN/'HASIL_EKSPERIMEN.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    plot_models=['rf','rf_balanced']+VARIANTS
    labels=['RF','W. RF','Graph','Gap ≥ 0','No density','Physics','Hybrid']
    fig,axes=plt.subplots(1,3,figsize=(10.5,4.3),layout='constrained')
    for ax,metric,title in zip(axes,['raw_ap','band_gap_mae','formation_energy_mae'],['Average precision ↑','Band-gap MAE (eV) ↓','Energy MAE (eV/atom) ↓']):
        ax.bar(range(7),[mean(m,metric) for m in plot_models],yerr=[aggregate[m][metric]['sd_all_nine_descriptive'] for m in plot_models],capsize=3,color=['#7b8794']*2+['#527b9f']*4+['#b46648'])
        ax.set_xticks(range(7),labels,rotation=45,ha='right');ax.set_title(title)
    fig.savefig(RUN/'ablation_comparison.pdf');fig.savefig(RUN/'ablation_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
    # Reliability curves average per-run bin statistics, preserving repeated-data interpretation.
    reliability={}
    for mode,pkey in [('raw','probability'),('calibrated','calibrated_probability')]:
        reliability[mode]={}
        ax=axes[0 if mode=='raw' else 1];ax.plot([0,1],[0,1],'--',color='grey')
        for model in ['rf_balanced','graph_baseline','graph_physics','graph_hybrid']:
            bins=[calibration_bins(np.array(r[model]['truth'])[:,0],np.array(r[model][pkey])) for r in all_pooled.values()]
            curve=[]
            for j in range(10):
                valid=[x[j] for x in bins if x[j]['n']]
                if valid:curve.append(dict(probability=float(np.mean([b['probability'] for b in valid])),observed=float(np.mean([b['observed'] for b in valid])),n_runs=len(valid)))
            reliability[mode][model]=curve
            ax.plot([b['probability'] for b in curve],[b['observed'] for b in curve],marker='o',ms=3,label=NAMES[model])
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Mean observed fraction',title=mode.capitalize());ax.legend(fontsize=9)
    fig.savefig(RUN/'reliability.pdf');fig.savefig(RUN/'reliability.png',dpi=180);plt.close(fig)
    (RUN/'reliability_bins.json').write_text(json.dumps(reliability,indent=2))
    for partition in partitions:
        fig,axes=plt.subplots(5,3,figsize=(13,14),layout='constrained')
        for row,fold in enumerate(config['folds']):
            for col,seed in enumerate(seeds):
                ax=axes[row,col]
                for model in VARIANTS:
                    h=json.loads((RUN/f'partition{partition}_fold{fold}'/f'seed{seed}'/model/'loss_history.json').read_text())
                    ax.plot([x['epoch'] for x in h],[x['validation_selection_objective'] for x in h],label=NAMES[model],lw=1)
                ax.set_title(f'Fold {fold}, initialization {seed}',fontsize=10);ax.set_xlabel('Epoch',fontsize=9)
                ax.tick_params(labelsize=9)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=9)
        fig.supylabel('Common inner-validation selection objective')
        fig.savefig(RUN/f'validation_histories_partition{partition}.pdf');plt.close(fig)
    print('\n'.join(lines))

if __name__=='__main__':main()
