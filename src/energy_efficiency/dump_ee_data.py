"""Diagnostic dump of the REAL cached EE data -- no plotting, no simulation,
and (deliberately) no reward computation. Run:

    python3 src/energy_efficiency/dump_ee_data.py

Only needs numpy + stdlib. The reward and lambda_ee are read from the training
metrics the model actually logged (training_error_learned_full.gzip) -- NOT
recomputed from rate/power, which would be fabrication. The sweep rate/EE and
the triplet operating points are read straight from their cached files.

Hard-codes the power-model constants used elsewhere (eta_PA=0.6,
N*P_circuit=16 W) only to report P_total alongside the measured radiated power.
"""
import gzip
import pickle
from pathlib import Path

import numpy as np

ETA_PA = 0.60
CIRCUIT_W = 16.0        # N=16 antennas * 1 W

TRAINING_NAME = 'EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow'
SWEEP = Path('outputs/metrics/EE_vs_transmit_power/ee_vs_power_sweep_sac_error0.gzip')
TRIPLET = Path('outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip')
TRAIN_METRICS = Path('outputs/metrics', TRAINING_NAME, 'base', 'training_error_learned_full.gzip')


def total_power(p_tx):
    return p_tx / ETA_PA + CIRCUIT_W


def last_valid(arr):
    """Last finite entry of a metrics array (they are pre-allocated with NaN/-inf)."""
    a = np.asarray(arr, dtype=float)
    finite = np.where(np.isfinite(a))[0]
    return (int(finite[-1]), float(a[finite[-1]])) if finite.size else (None, None)


print('=' * 72)
print('TRAINING METRICS (real logged reward & lambda):', TRAIN_METRICS)
print('  exists:', TRAIN_METRICS.exists())
if TRAIN_METRICS.exists():
    with gzip.open(TRAIN_METRICS, 'rb') as f:
        m = pickle.load(f)
    print('  keys:', list(m.keys()))
    for key in ['mean_reward_per_episode', 'lambda_ee_per_episode',
                'episode_mean_rate_dinkelbach', 'episode_mean_power_dinkelbach']:
        if key in m:
            idx, val = last_valid(m[key])
            print(f'  {key:32} last-valid[{idx}] = {val}')
    # short tail of the reward and lambda trajectories, if present
    for key in ['mean_reward_per_episode', 'lambda_ee_per_episode']:
        if key in m:
            a = np.asarray(m[key], dtype=float)
            fin = a[np.isfinite(a)]
            if fin.size:
                print(f'  {key} tail: {np.round(fin[-8:], 5)}')

print('=' * 72)
print('SWEEP (measured rate/EE vs forced power):', SWEEP, '  exists:', SWEEP.exists())
if SWEEP.exists():
    with gzip.open(SWEEP, 'rb') as f:
        s = pickle.load(f)
    print('  keys:', list(s.keys()), ' power_budget:', s.get('power_budget'))
    P = np.asarray(s['power_sweep_watt'])
    R = np.asarray(s['mean_rate'])
    Ptot = np.asarray(s['total_power_watt'])
    EE = np.asarray(s['ee'])
    print(f'  {"P_tx":>8} {"rate":>10} {"P_total":>10} {"EE":>12}')
    for i in range(len(P)):
        print(f'  {P[i]:8.2f} {R[i]:10.4f} {Ptot[i]:10.3f} {EE[i]:12.6f}')
    print(f'  -> EE argmax at P_tx = {P[int(np.argmax(EE))]:.2f} W (EE={EE.max():.6f})')
    print(f'  -> total_power check: file[0]={Ptot[0]:.3f} vs p/eta+circ={total_power(P[0]):.3f}')

print('=' * 72)
print('TRIPLET (operating points):', TRIPLET, '  exists:', TRIPLET.exists())
if TRIPLET.exists():
    with gzip.open(TRIPLET, 'rb') as f:
        t = pickle.load(f)
    print('  result keys:', list(t['results'].keys()))
    err = np.asarray(t['error_sweep_range'])
    i0 = int(np.argmin(np.abs(err - 0.0)))
    print(f'  error_sweep_range: {err}   (De=0 -> index {i0})')
    for k in ['sac_aod0.0', 'sac_aod0.0_fullpower', 'mmse_matched_aod0.0', 'mmse_nadir']:
        if k in t['results']:
            r = t['results'][k]
            p_tx = float(r['mean_power'][i0])
            rate = float(r['mean_rate'][i0])
            print(f'  {k:24} P_tx={p_tx:7.3f} W  P_total={total_power(p_tx):7.3f} W  rate={rate:8.4f}')
print('=' * 72)
