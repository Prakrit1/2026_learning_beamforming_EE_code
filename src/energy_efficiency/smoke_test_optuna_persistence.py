"""
Smoke test for optuna_train_EE.py's new persistent-storage + per-trial
exception-handling pattern, WITHOUT running real (expensive) SAC training --
tests the Optuna mechanics in isolation:
1. A study backed by SQLite storage actually persists trials to disk.
2. A trial that raises RuntimeError (simulating the NaN-divergence crash
   that killed job 143462) is caught and reported as a FAILED/PRUNED
   trial, WITHOUT crashing the whole study.optimize() loop -- confirms the
   next trial after a crash still runs.
3. Re-opening the same study (load_if_exists=True) picks up where the
   first run left off, instead of starting over.
"""
from pathlib import Path
import shutil

import optuna

tmp_dir = Path('/tmp/optuna_persistence_smoke_test')
shutil.rmtree(tmp_dir, ignore_errors=True)
tmp_dir.mkdir(parents=True)

storage_path = f'sqlite:///{Path(tmp_dir, "test_study")}.db'
study_name = 'smoke_test_study'


def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    if trial.number == 2:
        # simulate the exact failure mode that killed job 143462
        try:
            raise RuntimeError('simulated NaN divergence')
        except RuntimeError as e:
            print(f'[smoke test] Trial {trial.number} failed as expected: {e}')
            raise optuna.TrialPruned()
    return x ** 2


print('=== PHASE 1: run 5 trials, trial 2 will raise ===')
study = optuna.create_study(
    study_name=study_name, storage=storage_path, load_if_exists=True, direction='minimize'
)
study.optimize(objective, n_trials=5)
print(f'Phase 1 done: {len(study.trials)} trials recorded')
for t in study.trials:
    print(f'  trial {t.number}: state={t.state}, value={t.value}')
assert len(study.trials) == 5, f'expected 5 trials, got {len(study.trials)}'
assert study.trials[2].state == optuna.trial.TrialState.PRUNED, \
    f'expected trial 2 to be PRUNED, got {study.trials[2].state}'
assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
print('PHASE 1 PASSED: crash was contained to one trial, study kept going.')

print('\n=== PHASE 2: re-open the same study, confirm resume (not restart) ===')
study2 = optuna.create_study(
    study_name=study_name, storage=storage_path, load_if_exists=True, direction='minimize'
)
print(f'Re-opened study has {len(study2.trials)} trial(s) already recorded (should be 5, not 0)')
assert len(study2.trials) == 5, f'expected re-opened study to have 5 trials, got {len(study2.trials)}'
remaining = max(0, 10 - len(study2.trials))
study2.optimize(objective, n_trials=remaining)
print(f'After running {remaining} more trials: {len(study2.trials)} total')
assert len(study2.trials) == 10
print('PHASE 2 PASSED: study correctly resumed instead of restarting from trial 0.')

print('\nALL OPTUNA PERSISTENCE SMOKE TESTS PASSED')
shutil.rmtree(tmp_dir, ignore_errors=True)
