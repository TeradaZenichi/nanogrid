"""Publication figures from saved artifacts only; never imports simulation code.

Run: .venv/Scripts/python.exe scripts/make_results_figures.py
Outputs default to paper/figures/ (the explicit destination in the user request).
Missing figures are reported independently; mandatory failures yield exit code 1.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'data/Gulliver.otf').is_file())
DATA = ROOT / 'paper/artifacts/data/consolidated'
CASES = ['critical_50', 'full_100']
LABELS = ['Critical 50%', 'Full 100%']
COLORS = ['#215E83', '#A34B22', '#527846', '#785D89', '#333333', '#998027']
REPORT = []
DIVERGENCES = []
OUT = ROOT / 'paper/figures'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv(path, columns=()):
    frame = pd.read_csv(path)
    require(set(columns).issubset(frame.columns), f'{path}: missing {set(columns)-set(frame.columns)}')
    return frame


def source(path):
    return str(Path(path).relative_to(ROOT)).replace('\\', '/')


def check(label, actual, expected, tolerance):
    if abs(actual - expected) > tolerance:
        DIVERGENCES.append(f'{label}: artifact={actual:.8g}; expected={expected:.8g}')


def configure():
    path = ROOT / 'data/Gulliver.otf'
    font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({'font.family': [family], 'font.size': 8,
        'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5, 'legend.frameon': False, 'axes.linewidth': .6,
        'lines.linewidth': 1.4, 'lines.markersize': 3.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return family


def panels(axes, labels):
    for i, (ax, label) in enumerate(zip(np.ravel(axes), labels)):
        ax.text(0, 1.04, f'({chr(97+i)}) {label}', transform=ax.transAxes, fontsize=9)
        ax.grid(axis='y', linewidth=.4, color='.9')
        ax.set_axisbelow(True)


def save(fig, name, sources, n, values):
    fig.tight_layout(pad=.7, w_pad=1.8, h_pad=1.3)
    fig.canvas.draw()
    fontpath = (ROOT / 'data/Gulliver.otf').resolve()
    for item in fig.findobj(matplotlib.text.Text):
        resolved = Path(font_manager.findfont(item.get_fontproperties())).resolve()
        require(resolved == fontpath, f'Unexpected font {resolved} for {item.get_text()}')
    paths = [OUT / f'fig_results_{name}.{ext}' for ext in ['pdf', 'png']]
    for path in paths:
        fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=.04)
    pdf = paths[0].read_bytes()
    names = sorted({name.decode('ascii') for name in re.findall(rb'/BaseFont /([^\s/]+)', pdf)})
    require(names and all('Gulliver' in name for name in names), f'Unexpected PDF fonts: {names}')
    require(b'/FontFile2' in pdf and b'/Subtype /Image' not in pdf, 'PDF must embed font and contain vector graphics')
    plt.close(fig)
    REPORT.append({'figure': name, 'source_artifacts': [source(p) for p in sources],
        'observations': n, 'output_pdf': source(paths[0]), 'output_png': source(paths[1]),
        'status': 'OK', 'pdf_fonts': names, 'pdf_vector_and_font_check': 'PASS', 'important_values_checks': values})
    print(json.dumps(REPORT[-1], ensure_ascii=False))


def representative():
    base = ROOT / 'data/sizing'
    paths = [base / f for f in ['prototypes_load_dtw_all_train.csv', 'prototypes_pv_dtw_train.csv',
        'prob_load.csv', 'prob_pv.csv', 'prob_joint_load_pv.csv',
        'labels_load_dtw_all_train.csv', 'labels_pv_dtw_train.csv']]
    load, pv, pl, pp, joint, ll, lp = [csv(p) for p in paths]
    require(len(ll) == len(lp) == 603, 'Expected 603 fitting days')
    require(set(ll.date) == set(lp.date), 'Load/PV fitting days differ')
    merged = ll.merge(lp, on='date', suffixes=('_load', '_pv'), validate='one_to_one')
    counts = merged.groupby(['cluster_load', 'cluster_pv']).size()
    for row in joint.itertuples():
        require(counts.get((row.cluster_load, row.cluster_pv), 0) == row.n_days, 'Joint counts mismatch')
        require(np.isclose(row.probability, row.n_days / 603), 'Joint probability mismatch')
    require((joint.probability > 0).sum() == 22, 'Expected 22 positive joint combinations')
    for marginal, labels, key in [(pl, ll, 'cluster_load'), (pp, lp, 'cluster_pv')]:
        require(np.isclose(marginal.probability.sum(), 1), 'Marginal probability sum')
        for row in marginal.itertuples():
            require((labels.cluster == getattr(row, key)).sum() == row.n_days, 'Marginal count mismatch')
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    for ax, frame, count, prefix in zip(axes, [load, pv], [6, 4], ['L', 'P']):
        require(set(frame.cluster) == set(range(count)), 'Missing clusters')
        for k in range(count):
            d = frame[frame.cluster == k].sort_values('slot')
            require(d.slot.tolist() == list(range(48)), 'Expected 48 half-hour slots')
            require(np.isfinite(d.value).all() and (d.value >= 0).all(), 'Invalid profile')
            ax.plot(d.slot / 2, d.value, color=COLORS[k], linestyle=['-', '--', ':', '-.'][k % 4],
                    marker=['o', 's', '^', 'D', 'v', 'x'][k], markevery=8, label=f'{prefix}{k}')
        ax.set(xlim=(0, 24), xlabel='Hour of day [h]', ylabel='Normalized power [p.u.]')
        ax.set_xticks(range(0, 25, 4)); ax.legend(ncol=3 if count == 6 else 2, loc='upper left')
    panels(axes, ['Load profiles', 'PV profiles'])
    values = {'fitting_days': 603, 'positive_joint_combinations': 22,
        'load_probabilities': pl.probability.tolist(), 'pv_probabilities': pp.probability.tolist()}
    winner = joint.loc[joint.probability.idxmax()]
    values['most_frequent_joint'] = f'L{int(winner.cluster_load)}-P{int(winner.cluster_pv)}'
    values['most_frequent_joint_pct'] = float(winner.probability * 100)
    for label, actual, expected in [('L4', pl.set_index('cluster_load').loc[4, 'probability']*100, 34),
        ('P0', pp.set_index('cluster_pv').loc[0, 'probability']*100, 44.11),
        ('P3', pp.set_index('cluster_pv').loc[3, 'probability']*100, 41.96),
        ('L2-P0', joint.set_index(['cluster_load', 'cluster_pv']).loc[(2,0), 'probability']*100, 17.08)]:
        check(label+' probability (%)', actual, expected, .0051)
    if values['most_frequent_joint'] != 'L2-P0':
        DIVERGENCES.append('Most frequent joint: '+values['most_frequent_joint'])
    save(fig, 'representative_profiles', paths, len(load)+len(pv), values)


def sizing():
    path = DATA / 'sizing_degradation_comparison.csv'
    d = csv(path).set_index('sizing_case').loc[CASES]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7))
    vals = {}
    for ax, metric, unit, expected in zip(axes, ['pv_size_kw', 'bess_size_kwh'], ['kW', 'kWh'],
            [[[2.685,2.817],[2.389,2.574]], [[6.250,12.500],[6.525,13.012]]]):
        for j, (variant, label) in enumerate([('without_degradation','Without degradation'), ('with_degradation','With degradation')]):
            y = d[f'{metric}_{variant}'].to_numpy(float)
            require(np.isfinite(y).all(), 'Invalid sizing values')
            vals[f'{metric}_{variant}'] = y.tolist()
            bars = ax.bar(np.arange(2)+(j-.5)*.32, y, .32, color=COLORS[j],
                hatch='' if j == 0 else '//', label=label)
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=7.5)
            for i in range(2): check(f'{CASES[i]} {metric} {variant}', y[i], expected[j][i], .001)
        ax.set_xticks([0,1], LABELS); ax.set_ylabel(f'Capacity [{unit}]')
        ax.set_ylim(0, ax.get_ylim()[1]*1.17)
    axes[0].legend(loc='upper left', fontsize=7)
    panels(axes, ['PV capacity', 'BESS energy capacity'])
    vals['BESS_change_pct'] = d.bess_size_kwh_degradation_delta_pct.tolist()
    save(fig, 'sizing_degradation', [path], len(d), vals)


def autonomy():
    path = DATA / 'sizing_autonomy_processed.csv'
    data = csv(path); fig, axes = plt.subplots(1, 2, figsize=(7.16,2.8)); values = {}
    for k, case in enumerate(CASES):
        d = data[data.sizing_case == case].sort_values('outage_duration_h')
        require(d.outage_duration_h.tolist() == [1,2,4,6], 'Missing autonomy durations')
        good = d.with_degradation_status.eq('optimal')
        for ax, metric in zip(axes, ['bess_size_kwh', 'objective_usd']):
            y = d[f'{metric}_with_degradation'].where(good)
            require(np.isfinite(y[good]).all(), 'Missing feasible sizing')
            ax.plot(d.outage_duration_h, y, color=COLORS[k], marker=['o','s'][k],
                linestyle=['-','--'][k], label=LABELS[k])
        values[case] = d[['outage_duration_h','bess_size_kwh_with_degradation','objective_usd_with_degradation']].to_dict('records')
    cert = data[(data.sizing_case == 'full_100') & (data.outage_duration_h == 6)].iloc[0]
    require(bool(cert.capacity_limit_certified), 'Missing expanded-bound certificate')
    cap, minimum = float(cert.original_capacity_cap_kwh), float(cert.minimum_feasible_bess_kwh_with_degradation)
    require(minimum > cap and cert.with_degradation_status != 'optimal', 'Invalid infeasibility certificate')
    axes[0].axhline(cap, color='.45', ls='--', lw=.8)
    axes[0].scatter(6, minimum, marker='X', s=38, color=COLORS[1])
    axes[0].annotate(f'Minimum required: {minimum:.2f} kWh\nExpanded bound; original infeasible',
        (6,minimum), xytext=(-5,-29), textcoords='offset points', ha='right', fontsize=7)
    axes[0].text(1.1,cap+.5,f'{cap:g} kWh bound', fontsize=7, color='.35')
    axes[0].set_ylim(0,29); axes[0].set_ylabel('BESS capacity [kWh]')
    axes[1].set_ylabel('Life-cycle cost [USD]')
    axes[1].text(.98,.04,'Full 100%, 6 h: infeasible', transform=axes[1].transAxes,ha='right',fontsize=7)
    for ax in axes: ax.set_xlabel('Outage autonomy [h]'); ax.set_xticks([1,2,4,6]); ax.set_xlim(.8,6.2)
    axes[0].legend(loc='center left', bbox_to_anchor=(0,.61)); panels(axes,['BESS capacity','Life-cycle cost'])
    values.update(original_bound_kwh=cap, minimum_required_kwh=minimum)
    check('Expanded-bound minimum kWh', minimum,25.63,.01)
    save(fig,'autonomy_sensitivity',[path],8,values)


def mesh():
    fig, axes = plt.subplots(1,2,figsize=(7.16,3.1)); paths=[]; values={}
    for k,(ax,case) in enumerate(zip(axes,CASES)):
        path=ROOT/f'paper/artifacts/data/mesh/{case}/champion_mesh.csv'; paths.append(path)
        d=csv(path,['mean_solve_time_s','mean_regret','max_regret']); require(len(d)==27,'Expected all 27 meshes')
        require(d[['h','t1','t2']].drop_duplicates().shape[0]==27,'Duplicate meshes')
        z=d[['mean_solve_time_s','mean_regret','max_regret']].to_numpy(float)
        require(np.isfinite(z).all() and (z>=0).all(), 'Invalid mesh objectives')
        front=np.array([not np.any(np.all(z<=p,axis=1)&np.any(z<p,axis=1)) for p in z])
        require(np.array_equal(front,d.pareto.to_numpy(bool)), 'Stored three-objective Pareto mismatch')
        ax.scatter(z[~front,0],100*z[~front,1],s=13,c='.65',marker='o',label='Dominated')
        ax.scatter(z[front,0],100*z[front,1],s=30,facecolors='none',edgecolors=COLORS[0],marker='s',label='Pareto (3 objectives)')
        selected=d[(d.h==24)&(d.t1==5)&(d.t2==120)].iloc[0]
        official=json.loads(path.with_suffix('.json').read_text())['selected'][0]
        require(official['combo']==selected.combo and selected.selection_score==d[front].selection_score.min(), 'Selected mesh mismatch')
        ax.scatter(selected.mean_solve_time_s,100*selected.mean_regret,s=100,marker='*',color=COLORS[1],label='Selected',zorder=5)
        combos=[(24,5,120),(36,5,60),(24,5,60)] if k==0 else [(24,5,120),(24,5,60),(24,5,30)]
        vals=[]
        for j,(h,t1,t2) in enumerate(combos):
            row=d[(d.h==h)&(d.t1==t1)&(d.t2==t2)].iloc[0]
            offset = [(5,12),(12,4),(5,-15)][j] if k==0 else [(-50,5),(5,-10),(12,-8)][j]
            ax.annotate(f'{h}/{t1}/{t2}',(row.mean_solve_time_s,row.mean_regret*100),
                xytext=offset,textcoords='offset points',fontsize=7,
                arrowprops={'arrowstyle':'-', 'lw':.4,'color':'.4'})
            vals.append({'mesh':f'{h}/{t1}/{t2}','mean_solve_time_s':row.mean_solve_time_s,'mean_regret_pct':row.mean_regret*100})
        ax.set(xlabel='Mean solve time [s]',ylabel='Mean regret [%]',xlim=(0,z[:,0].max()*1.12))
        ax.set_ylim(0, z[:,1].max()*100*1.06)
        ax.legend(loc='upper right',fontsize=7)
        values[case]=vals
    values['note']='Official Pareto uses mean regret, maximum regret and time. No projected frontier connecting line.'
    panels(axes,LABELS); save(fig,'mesh_selection',paths,54,values)


def operation():
    candidates=[]; rejected=[]
    required=['Load_kw','PV_kw','P_bess_kw','P_bess_discharge_kw','P_bess_charge_mag_kw',
              'P_grid_in_kw','P_grid_out_kw','SoC_pct','Residual_kw']
    # Prefer critical_50 when it has any complete, physically audited day with
    # interior outage, daylight PV, and both charge/discharge >= 0.1 kWh.
    # Within that case, normalize each feature by its candidate maximum and use
    # equal weights: outage duration, SoC range, throughput, PV energy. Ties: date.
    for case in CASES:
        paths=sorted((ROOT/f'outputs/operation-sweep/with-degradation/{case}/03-forecast-operation').glob('*/prototype/operation_final.parquet'))
        for path in paths:
            met=json.loads(path.with_name('metrics.json').read_text())
            require([met[key] for key in ['horizon_hours','timestep_1_min','timestep_2_min']]==[24,5,120],f'Wrong mesh: {path}')
            require(met['status']=='ok' and met['n_solve_fail']==0,f'Unaudited run {path}')
            frame=pd.read_parquet(path); require(set(required+['timestamp','outage_active']).issubset(frame.columns),f'Missing operation columns: {path}')
            frame.timestamp=pd.to_datetime(frame.timestamp); frame=frame.sort_values('timestamp')
            for day,d in frame.groupby(frame.timestamp.dt.normalize()):
                d=d.reset_index(drop=True)
                if len(d)!=288 or not d.timestamp.equals(pd.Series(pd.date_range(day,periods=288,freq='5min'))):
                    rejected.append(f'{source(path)} {day.date()}: incomplete day'); continue
                if not np.isfinite(d[required].to_numpy(float)).all(): continue
                outage=d.outage_active.to_numpy(bool)
                if not outage.any() or outage[0] or outage[-1]: continue
                if d.Residual_kw.abs().max()>1e-7 or (d.SoC_pct<0).any() or (d.SoC_pct>100+1e-7).any(): continue
                if (d.loc[outage,['P_grid_in_kw','P_grid_out_kw']].abs().to_numpy()>1e-7).any(): continue
                require(np.allclose(d.P_bess_kw,d.P_bess_discharge_kw-d.P_bess_charge_mag_kw,atol=1e-7),'Realized BESS sign mismatch')
                charge=d.P_bess_charge_mag_kw.sum()/12; discharge=d.P_bess_discharge_kw.sum()/12; pv=d.PV_kw.sum()/12
                if min(charge,discharge)<.1 or pv<=0: continue
                candidates.append({'case':case,'path':path,'day':day,'data':d,
                    'features':[outage.sum()/12,float(d.SoC_pct.max()-d.SoC_pct.min()),charge+discharge,pv]})
        if candidates: break
    require(bool(candidates),'No complete audited 24-h prototype day with interior outage, PV and bidirectional BESS activity; require operation_final.parquet + metrics.json for mesh 24/5/120')
    features=np.array([c['features'] for c in candidates]); scores=(features/features.max(axis=0)).mean(axis=1)
    winner=sorted(zip(scores,candidates),key=lambda pair:(-pair[0],str(pair[1]['day']),str(pair[1]['path'])))[0]
    score,c=winner; d=c['data']; path=c['path']; day=c['day']; x=(d.timestamp-day).dt.total_seconds()/3600
    outage=d.outage_active.to_numpy(bool); changes=np.diff(np.r_[False,outage,False].astype(int))
    spans=list(zip(np.flatnonzero(changes==1)/12,np.flatnonzero(changes==-1)/12))
    params=json.loads(path.with_name('parameters_used.json').read_text())
    fine_h=float(params['EDS']['outage_duration_hours'])
    require(fine_h==2,'Expected 2-h fine region from stored parameters')
    edges=np.r_[np.arange(25)/12,np.arange(4,25,2)]
    require(len(edges)-1==35 and edges[-1]==24,'Invalid mesh timeline')
    fig,axes=plt.subplots(4,1,figsize=(7.16,6.2),gridspec_kw={'height_ratios':[.7,1,1,1]})
    for a,b in zip(edges[:-1],edges[1:]): axes[0].add_patch(Rectangle((a,.1),b-a,.32,fill=False,edgecolor=COLORS[0] if a<2 else '.4',lw=.5))
    axes[0].set(ylim=(0,1),xlim=(0,24)); axes[0].set_yticks([])
    axes[0].text(0,.62,'Fine: 24 x 5 min',fontsize=7.5)
    axes[0].text(13,.62,'Coarse: 11 x 120 min',ha='center',fontsize=7.5)
    axes[0].set_xticks([0,2,24],['0 h','2 h','24 h']); axes[0].grid(False)
    for spine in axes[0].spines.values(): spine.set_visible(False)
    axes[1].plot(x,d.Load_kw,color=COLORS[0],label='Load')
    axes[1].plot(x,d.PV_kw,color=COLORS[1],ls='--',label='Available PV')
    # Grid columns are nonnegative import/export magnitudes. Their difference
    # is positive import, negative export. P_bess_kw is realized discharge-charge.
    axes[2].plot(x,d.P_bess_kw,color=COLORS[0],label='BESS (+ discharge, - charge)')
    axes[2].plot(x,d.P_grid_in_kw-d.P_grid_out_kw,color=COLORS[1],ls='--',label='Grid (+ import, - export)')
    axes[2].axhline(0,color='.5',lw=.6)
    axes[3].plot(x,d.SoC_pct,color=COLORS[0]); axes[3].set_ylabel('SoC [%]')
    for ax in axes[1:3]: ax.set_ylabel('Power [kW]')
    for i,ax in enumerate(axes[1:]):
        for j,(a,b) in enumerate(spans): ax.axvspan(a,b,color='.8',alpha=.45,label='Grid outage' if i==0 and j==0 else None)
        ax.set_xlim(0,24); ax.set_xticks(range(0,25,4),[f'{h:02d}:00' for h in range(0,25,4)])
        if i<2: ax.tick_params(labelbottom=False)
    axes[1].legend(loc='upper left',ncol=3,fontsize=7)
    axes[2].legend(loc='upper left',ncol=2,fontsize=7)
    axes[3].set_xlabel('Time of day')
    panels(axes,['Selected temporal mesh','Load and PV','BESS and grid power','BESS state of charge'])
    metadata={'case':c['case'],'month':path.parent.parent.name,'date':str(day.date()),'parquet_source':source(path),
        'mesh':{'horizon_h':24,'fine_min':5,'coarse_min':120,'fine_region_h':fine_h,'intervals':35},
        'outage_interval':[{'start':str(day+pd.Timedelta(hours=a)),'end_exclusive':str(day+pd.Timedelta(hours=b))} for a,b in spans],
        'summary_statistics':{'daily_load_kwh':d.Load_kw.sum()/12,'daily_pv_kwh':d.PV_kw.sum()/12,
            'bess_throughput_kwh':c['features'][2],'soc_min_pct':d.SoC_pct.min(),'soc_max_pct':d.SoC_pct.max()},
        'selection_score':float(score),'eligible_days':len(candidates),'selection_rule':'Critical preferred; equal-weight maximum-normalized outage hours, SoC range, throughput and PV energy; date/path tie break',
        'soc_timestamp_convention':'Post-step SoC, stored at step timestamp; plotted without shifting',
        'grid_sign':'import minus export; positive import','bess_sign':'realized discharge minus charge; positive discharge',
        'rejected_incomplete_days':rejected}
    (OUT/'fig_results_selected_mesh_operation_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    save(fig,'selected_mesh_operation',[path,path.with_name('metrics.json'),path.with_name('parameters_used.json')],len(d),metadata)


def monthly():
    path=DATA/'controller_monthly_paired.csv'; d=csv(path)
    require(d.audit_pass.all(),'Unaudited controller window')
    require(not d.duplicated(['sizing_case','month','controller_name']).any(),'Duplicate controller window')
    return path,d


def costs(forecast=True):
    path,d=monthly()
    controllers=['ideal','lstm','prototype'] if forecast else ['prototype','reserve_only','load_shifting','peak_shaving','self_consumption']
    labels=['Perfect\ninformation','LSTM','Prototype'] if forecast else ['Prototype MPC','Reserve-only','Load shifting','Peak shaving','Self-consumption']
    metric='operation_total_cost' if forecast else 'operation_total_cost_inventory_adjusted'
    subset=d[d.sizing_case.isin(CASES)&d.controller_name.isin(controllers)]
    require(subset.groupby(['sizing_case','controller_name']).size().eq(12).all(),'Expected 12 monthly windows')
    require(subset.groupby(['sizing_case','month']).exogenous_sha256.nunique().eq(1).all(),'Unpaired exogenous series')
    values=subset.groupby(['sizing_case','controller_name'])[metric].mean().unstack().loc[CASES,controllers]
    require(np.isfinite(values.to_numpy()).all(),'Missing controller means')
    expected=[[10.674,10.795,10.906],[9.499,9.615,9.690]] if forecast else [[11.17,11.68,11.75,14.68,17.04],[10.65,11.16,11.14,13.51,17.04]]
    fig,ax=plt.subplots(figsize=(3.5,2.7) if forecast else (7.16,3))
    for i,case in enumerate(CASES):
        v=values.loc[case].to_numpy(); positions=np.arange(len(controllers))+(i-.5)*.34
        if forecast:
            require(v[0]<v[1]<v[2],'Raw cost ranking does not satisfy Perfect < LSTM < Prototype')
            bars=ax.bar(positions,v,.34,color=COLORS[i],hatch='' if i==0 else '//',label=LABELS[i])
        else: bars=ax.barh(positions,v,.34,color=COLORS[i],hatch='' if i==0 else '//',label=LABELS[i])
        ax.bar_label(bars,fmt='%.3f' if forecast else '%.2f',fontsize=7,padding=2)
        for j,vv in enumerate(v): check(f'{case} {controllers[j]} {metric}',vv,expected[i][j],.001 if forecast else .015)
    if forecast:
        ax.set_xticks(range(3),labels); ax.set_ylabel('Mean realized cost [USD/10 days]'); ax.set_ylim(0,14.2)
        ax.legend(ncol=2,loc='upper center',fontsize=7)
    else:
        ax.set_yticks(range(5),labels); ax.invert_yaxis(); ax.set_xlabel('Mean inventory-adjusted cost [USD/10 days]')
        ax.set_xlim(0,20.5); ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=2); ax.grid(axis='x',color='.9',lw=.4); ax.set_axisbelow(True)
    info={'metric':metric,'means':values.to_dict('index'),'monthly_windows_per_mean':12}
    if forecast: info['relative_to_own_perfect_pct']={case:((values.loc[case]/values.loc[case,'ideal']-1)*100).to_dict() for case in CASES}
    save(fig,'forecast_operational_value' if forecast else 'controller_comparison',[path],len(subset),info)


def robustness():
    path=DATA/'robustness_summary.csv'; d=csv(path)
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.8)); values={}
    expected=[[[11.094,11.291,11.474],[12.795,12.834,12.874]],[[9.883,10.185,10.488],[12.297,12.375,12.456]]]
    for i,(case,ax) in enumerate(zip(CASES,axes)):
        values[case]={}
        for j,ctrl in enumerate(['prototype','stochastic']):
            sub=d[(d.sizing_case==case)&(d.controller_name==ctrl)].set_index('variant').loc[['base','noise_005','noise_010']]
            require(len(sub)==3 and sub.seasonal_windows.eq(4).all(),'Expected four seasonal robustness windows')
            y=sub.operation_total_cost_mean.to_numpy(float); require(np.isfinite(y).all(),'Invalid robustness costs')
            ax.plot([0,5,10],y,color=COLORS[j],marker=['o','s'][j],ls=['-','--'][j],label=['Prototype MPC','Stochastic'][j])
            relative=(y/y[0]-1)*100
            require(np.allclose(relative,sub.operation_total_cost_delta_pct_vs_base,atol=1e-8),'Own-controller baseline mismatch')
            values[case][ctrl]={'cost_usd':y.tolist(),'own_baseline_change_pct':relative.tolist()}
            for n in range(3):check(f'{case} {ctrl} noise {[0,5,10][n]}',y[n],expected[i][j][n],.001)
        ax.set(xlabel='Actuator noise [%]',ylabel='Mean operating cost [USD/10 days]'); ax.set_xticks([0,5,10]); ax.legend(loc='center right')
    panels(axes,LABELS);save(fig,'robustness_noise',[path],12,values)


def validation():
    names=['llstm_vstf_load_5min_240min_96','lstm_vstf_pv_5min_60min_48','lstm_hourly_load_36h_8','lstm_hourly_pv_36h_8']
    paths=[ROOT/f'models/{name}_config.json' for name in names]
    require(all(p.is_file() for p in paths),'Need validation val_metrics MAE/MSE/R2 in '+', '.join(source(p) for p in paths if not p.is_file()))
    metrics=[json.loads(p.read_text())['val_metrics'] for p in paths]
    # RMSE is deterministically sqrt(saved MSE); no inference/training is run.
    a=np.array([[m['MAE'],np.sqrt(m['MSE']),m['R2']] for m in metrics])
    require(np.isfinite(a).all(),'Invalid validation metrics')
    expected=[[.0528,.0817,.6346],[.0334,.0707,.9348],[.0566,.0784,.3427],[.0723,.1217,.7999]]
    for i in range(4):
        for j,key in enumerate(['MAE','RMSE','R2']): check(f'{names[i]} {key}',a[i,j],expected[i][j],.000051)
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.6)); x=np.arange(4)
    for j,key in enumerate(['MAE','RMSE']):
        axes[0].bar(x+(j-.5)*.34,a[:,j],.34,color=COLORS[j],hatch='' if j==0 else '//',label=key)
    bars=axes[1].bar(x,a[:,2],.55,color=COLORS[0]);axes[1].bar_label(bars,fmt='%.3f',fontsize=7,padding=2)
    for ax in axes: ax.set_xticks(x,['Load VST\n4 h','PV VST\n1 h','Load ST\n36 h','PV ST\n36 h'])
    axes[0].set_ylabel('Validation error [p.u.]');axes[0].legend(loc='upper left');axes[1].set_ylabel('R2');axes[1].set_ylim(0,1.1)
    panels(axes,['MAE and RMSE','Coefficient of determination'])
    save(fig,'lstm_validation',paths,4,{'metrics':dict(zip(names,a.tolist())),'RMSE_calculation':'sqrt(saved MSE)'})


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
    family=configure(); failures=0
    tasks=[('representative_profiles',representative),('sizing_degradation',sizing),('autonomy_sensitivity',autonomy),
        ('mesh_selection',mesh),('selected_mesh_operation',operation),('forecast_operational_value',lambda:costs(True)),
        ('controller_comparison',lambda:costs(False)),('robustness_noise',robustness),('lstm_validation',validation)]
    for name,task in tasks:
        try: task()
        except (FileNotFoundError,ValueError,KeyError) as exc:
            status='SKIPPED' if name=='lstm_validation' else 'FAILED';failures+=status=='FAILED'
            REPORT.append({'figure':name,'status':status,'missing_or_invalid_data':str(exc)})
            print(f'{status}: {name}: {exc}');traceback.print_exc()
            plt.close('all')
    payload={'font':family,'figures':REPORT,'divergences':DIVERGENCES,
        'policy':'Existing artifacts only; no scientific results modified; no campaigns executed.'}
    # Infeasible values are explicit JSON null, never nonstandard NaN tokens.
    def clean(value):
        if isinstance(value,dict): return {key:clean(val) for key,val in value.items()}
        if isinstance(value,list): return [clean(val) for val in value]
        if isinstance(value,float) and not np.isfinite(value): return None
        return value
    (OUT/'results_figures_report.json').write_text(json.dumps(clean(payload),indent=2,allow_nan=False),encoding='utf-8')
    lines=['Figure | Source artifacts | Output PDF | Output PNG | Status | Important values/checks',
        *[' | '.join([r['figure'],', '.join(r.get('source_artifacts',[])),r.get('output_pdf',''),r.get('output_png',''),r['status'],json.dumps(r.get('important_values_checks',r.get('missing_or_invalid_data','')),ensure_ascii=False)]) for r in REPORT],
        '', 'Divergences:', *(DIVERGENCES or ['None beyond displayed rounding.'])]
    report='\n'.join(lines)+'\n';(OUT/'results_figures_report.txt').write_text(report,encoding='utf-8');print(report)
    return int(failures>0)


if __name__=='__main__':
    raise SystemExit(main())
