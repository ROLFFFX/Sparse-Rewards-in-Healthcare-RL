"""
Run every theory experiment. MIMIC experiments need the MDP data in mimic-sepsis/data and are skipped
without it.

    python run_all.py                    # everything
    python run_all.py --skip-mimic       # without MIMIC
    python run_all.py --theorems-only    # the two theorems only (main figure + their supplementary figures)
    python run_all.py --plot-only        # redraw every figure from the saved results
    python run_all.py --only mass_transfer terminal_defect
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MIMIC_READY = os.path.exists(os.path.join(HERE, '..', 'mimic-sepsis', 'data', 'traj_shifted_train.csv'))

# (job name, script and arguments relative to this directory)
# the two theorems: main figure (original setups) and their supplementary figures
THEOREMS = [
    ('original_setups', ['code/exp_original_setups.py']),
    ('terminal_defect', ['code/exp_terminal_defect.py', '--env', 'lifegate', '--K', '5000']),
    ('terminal_defect', ['code/exp_terminal_defect.py', '--env', 'mimic']),
    ('mass_transfer', ['code/exp_mass_transfer.py', '--env', 'lifegate']),
    ('mass_transfer', ['code/exp_mass_transfer.py', '--env', 'lifegate', '--gamma', '1']),
    ('mass_transfer', ['code/exp_mass_transfer.py', '--env', 'mimic']),
    ('mass_transfer', ['code/exp_mass_transfer.py', '--env', 'mimic', '--gamma', '1']),
    ('robustness', ['code/exp_mass_transfer.py', '--robustness']),
]
# propositions and corollaries (supplementary)
PROPOSITIONS = [
    ('invariance', ['code/supplementary/exp_equal_length_dag.py']),
    ('invariance', ['code/supplementary/exp_random_mdp_diagnostic.py']),
    ('certificate', ['code/supplementary/exp_certificate.py', '--env', 'lifegate']),
    ('certificate', ['code/supplementary/exp_certificate.py', '--env', 'mimic']),
    ('terminal_scale', ['code/supplementary/exp_terminal_scale.py', '--env', 'lifegate']),
    ('terminal_scale', ['code/supplementary/exp_terminal_scale.py', '--env', 'mimic']),
    ('mimic_sensitivity', ['code/supplementary/exp_mimic_sensitivity.py']),
]


def main():
    parser = argparse.ArgumentParser(description='Run the theory experiments.')
    parser.add_argument('--only', nargs='*', default=None, help='run only the jobs with these names')
    parser.add_argument('--skip-mimic', action='store_true', help='skip everything that needs MIMIC-III')
    parser.add_argument('--theorems-only', action='store_true', help='skip the supplementary propositions')
    parser.add_argument('--plot-only', action='store_true', help='redraw the figures from saved results')
    args = parser.parse_args()
    skip_mimic = args.skip_mimic or not MIMIC_READY
    jobs = THEOREMS + ([] if args.theorems_only else PROPOSITIONS)
    for name, cmd in jobs:
        if args.only and name not in args.only:
            continue
        if skip_mimic and ('mimic' in cmd or name == 'mimic_sensitivity'):
            print(f'-- skip {" ".join(cmd)} (MIMIC data not available or skipped)')
            continue
        if args.plot_only:
            cmd = cmd + ['--plot-only']
        # exp_original_setups includes MIMIC as its last setups
        if name == 'original_setups' and skip_mimic:
            cmd = cmd + ['--skip-mimic']
        print(f'== {" ".join(cmd)}', flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, *cmd], cwd=HERE, check=True)
        print(f'   done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
