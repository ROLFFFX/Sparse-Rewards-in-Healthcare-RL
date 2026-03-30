import argparse
import sys

# Import the main execution functions from your new modules
from break_assumptions import gridworld_break_asms
from gridworld_scaling import gridworld_scaling
from gridworld_asym import run_all_asym_sweeps

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Execution hub for Gridworld reinforcement learning experiments."
    )
    
    # Define the boolean flags
    parser.add_argument('-b', '--break-asms', action='store_true', 
                        help="Run the base assumption breaking experiments.")
    parser.add_argument('-s', '--scaling', action='store_true', 
                        help="Run the scaling parameter sweeps and heatmaps.")
    parser.add_argument('-a', '--asym', action='store_true', 
                        help="Run the asymmetric reward interpolation sweeps.")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Enforce the requirement that at least one flag must be present
    if not any([args.break_asms, args.scaling, args.asym]):
        parser.print_help()
        print("\nError: You must provide at least one execution flag (-b, -s, or -a).")
        sys.exit(1)

    # Execute the requested modules sequentially
    if args.break_asms:
        print("\n=== STARTING BASE ASSUMPTIONS SWEEP ===")
        gridworld_break_asms()
        
    if args.scaling:
        print("\n=== STARTING SCALING SWEEP ===")
        gridworld_scaling()
        
    if args.asym:
        print("\n=== STARTING ASYMMETRIC REWARD SWEEP ===")
        run_all_asym_sweeps()

    print("\nAll requested experiment suites finished successfully!")

if __name__ == "__main__":
    main()