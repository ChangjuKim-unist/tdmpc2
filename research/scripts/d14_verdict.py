import json,math,statistics as st,glob,sys
T=2.571
def load(n):
    d=json.load(open(f'research/results/d14/{n}.json')); return {(r['speed'],r['seed']):r for r in d['robots']}
def iv(a,b,key='reward_per_step'):
    ks=sorted(a); diffs=[a[k][key]-b[k][key] for k in ks]
    m=st.mean(diffs); s=st.stdev(diffs)/math.sqrt(len(diffs)); return m,m-T*s,m+T*s,diffs
def lab(lo,hi,pos='이득',neg='해'): return pos if lo>0 else (neg if hi<0 else '미결')
N='go1_paper_restore_true'
nom=load('nominal_model_'+N); nomx=load('nominal_exact_'+N)
r=st.mean(v['reward_per_step'] for v in nomx.values())/st.mean(v['reward_per_step'] for v in nom.values())
print('V0 ratio exact/pre nominal', round(r,3), 'pass' if r>=0.9 else 'FAIL')
for c,ad in [('friction0.15','go1_paper_friction0.15_dyn_b45000_term'),('motor0.4','go1_paper_very_weak_motor_dyn_b45000_term')]:
    pre=load(f'{c}_model_{N}'); pre0=load(f'{c}_model_go1_paper_restore_false'); ex=load(f'{c}_exact_{N}'); co=load(f'{c}_consensus_{N}'); am=load(f'{c}_model_{ad}_restore_true')
    print('==',c)
    for nm,a,b in [('G exact-pre',ex,pre),('C con-exact',co,ex),('C con-pre',co,pre),('ref adapted-pre',am,pre),('ref exact-adapted',ex,am),('ref con-adapted',co,am),('ref restore: pre_true-pre_false',pre,pre0)]:
        m,lo,hi,d=iv(a,b); print(f'{nm:34s} {m*1e3:+.2f}e-3 [{lo*1e3:+.2f},{hi*1e3:+.2f}]',lab(lo,hi,'위' if 'C' in nm else '이득','아래' if 'C' in nm else '해'), [round(x*1e3,1) for x in d])
    for nm,a in [('pre',pre),('pre_norestore',pre0),('adapted',am),('exact',ex),('consensus',co)]:
        ks=sorted(a); f=lambda k2: st.mean(a[k][k2] for k in ks)
        print(f'  {nm:14s} rew {f("reward_per_step")*1e3:.2f}e-3 falls {f("falls_per_1000"):.2f} fallsN {sum(a[k]["falls"] for k in ks)} speed {f("forward_speed"):.3f} smooth {f("smooth"):.3f} effort {f("effort"):.2f} shake {f("shake"):.3f}', 'per-robot rew', [round(a[k]['reward_per_step']*1e3,1) for k in ks], 'falls',[a[k]['falls'] for k in ks])
    for nm,a in [('exact',ex),('consensus',co)]:
        ks=sorted(a)
        print('  diag',nm,{k2:round(st.mean(a[k][k2] for k in ks),3) for k2 in a[ks[0]] if k2 in('rho_model_exact','rho_reward_head_env','model_binds','cand_fall_frac')})
