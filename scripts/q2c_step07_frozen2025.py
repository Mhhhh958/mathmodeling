from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scripts' / 'q2c_failure_driven_improvement.py'
spec = importlib.util.spec_from_file_location('q2c_base_frozen2025', SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Isolate this clean rerun from earlier mixed STEP07 evidence.
mod.OUT = ROOT / 'outputs' / 'q2c_step07_frozen2025'
mod.MODELS = mod.OUT / 'models'
mod.OUT.mkdir(parents=True, exist_ok=True)
mod.MODELS.mkdir(parents=True, exist_ok=True)

# STEP05/STEP06 frozen protocol guardrails.
assert mod.SEED == 2025
assert mod.RUNS == {
    'R1': {'train':['F3','F4'],'validation':['F2'],'test':['F1']},
    'R2': {'train':['F4','F1'],'validation':['F3'],'test':['F2']},
    'R3': {'train':['F1','F2'],'validation':['F4'],'test':['F3']},
    'R4': {'train':['F2','F3'],'validation':['F1'],'test':['F4']},
}

if __name__ == '__main__':
    mod.main()
