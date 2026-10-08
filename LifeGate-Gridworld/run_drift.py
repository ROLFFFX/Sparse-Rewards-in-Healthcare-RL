"""Rerun the LifeGate experiments with the original 70/30 dead-end drift (gridworld_drift.py).

Experiments (positional argument, default `break`):
  break           break_assumptions.py with drifting dead-ends        -> plots_break_asms_drift/
  break_collapse  break_assumptions.py with the dead-ends of gridworld.py, plotted with the drift-style policy
                  plots                                               -> plots_break_asms_clean/
  scaling         gridworld_scaling.py with drifting dead-ends        -> plots_scaling_drift/
  asym            gridworld_asym.py with drifting dead-ends           -> plots_asym_drift/

The counts of dangerous states reported with these plots are computed by drift_danger_counts.py.

The experiment scripts import `get_gridworld` (and `plot_highlighted_policies`) from gridworld.py by name, so the
drift versions are swapped in by rebinding those names, and PLOT_DIR, in the imported experiment module;
gridworld.py itself is never modified.  The experiment modules are imported only when needed because each one
creates its default plot folder on import.

Usage:  python run_drift.py [break|break_collapse|scaling|asym]
"""
import argparse
import os

import gridworld_drift as gd


def configure(module, plot_dir, **overrides):
    """Point an experiment module at `plot_dir` and rebind the module-level names in `overrides`."""
    module.PLOT_DIR = plot_dir
    os.makedirs(plot_dir, exist_ok=True)
    for name, value in overrides.items():
        setattr(module, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('experiment', nargs='?', default='break',
                        choices=['break', 'break_collapse', 'scaling', 'asym'])
    which = parser.parse_args().experiment

    if which == 'break':
        import break_assumptions as ba
        configure(ba, 'plots_break_asms_drift', get_gridworld=gd.get_gridworld,
                  plot_highlighted_policies=gd.plot_highlighted_policies)
        ba.gridworld_break_asms()
    elif which == 'break_collapse':
        import break_assumptions as ba
        configure(ba, 'plots_break_asms_clean', plot_highlighted_policies=gd.plot_highlighted_policies)
        ba.gridworld_break_asms()
    elif which == 'scaling':
        import gridworld_scaling as gs
        configure(gs, 'plots_scaling_drift', get_gridworld=gd.get_gridworld)
        gs.gridworld_scaling()
    elif which == 'asym':
        import gridworld_asym as ga
        configure(ga, 'plots_asym_drift', get_gridworld=gd.get_gridworld)
        ga.run_all_asym_sweeps()


if __name__ == '__main__':
    main()
