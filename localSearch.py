# import
from argumentParser import parseArguments
from parameters import setupParameters, setupIndexes
from gurobipy import GRB
import math as m
import numpy as np
import random as rnd
import time as tm

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstanceName)
P, periods, meter_groups, substitution_squads, intervals = setupIndexes(J, K, SP, DH, T)

# Initialize the pseudo-random number generator
rnd.seed(args.argSeed)

# ----------------------------------------------------------------------------------------------------------------------
# Function(s) implementing the Random Variable MIP Neighborhood Descent used to improve the solution of the semi-greedy
# ----------------------------------------------------------------------------------------------------------------------
def variableMIPNeighborhoodDescent(solution, model, x, y, overline_y, z, S, R, X, D, metrics, neighborhood):
    """
        Performs Variable MIP Neighborhood Descent by fixing a subset of variables
        and re-optimizing the model.

        Args:
        - solution: current solution object
        - model: SMISP model
        - x, y, overline_y, z: operational variables
        - S, R, X, D: supporting variables
        - metrics: metrics object for tracking performance
        - neighborhood: neighborhood indicator (1-4)

        Returns:
            tuple: (solution, status, solution_count)
        """

    # Get 't2'
    t2 = tm.perf_counter()

    # --- DEBUG ONLY ---
    # Create a unique identifier for this iteration
    # iteration_id = f"Iteration_{int(metrics.NumIters)}_{int(t2)}"

    # ==================================================================================================================
    # LOG MARKER: Start of iteration
    # ==================================================================================================================
    # --- DEBUG ONLY ---
    print("=" * 114 + "\n")
    # print(f"NEIGHBORHOOD DESCENT - {iteration_id}")
    # print("=" * 114)

    # Store original bounds for restoration
    original_bounds = {}

    # Reset the model
    model.reset()

    # --- DEBUG ONLY ---
    # print("> Model reset completed...\n")

    # Get variables to fix based on 'neighborhood'
    y_fixed, overline_y_fixed, z_fixed = getVariablesToFix(neighborhood)

    # --- DEBUG ONLY ---
    # Message for the user
    # print(f"> Variables to fix: y={len(y_fixed)}, overline_y={len(overline_y_fixed)}, z={len(z_fixed)}")

    # Variables to store model status and solution count BEFORE any modifications
    status = GRB.LOADED
    solution_count = 0

    try:
        # ==============================================================================================================
        # PHASE 1: FIX VARIABLES BY SETTING BOUNDS
        # ==============================================================================================================
        # --- DEBUG ONLY ---
        # print("\n--- PHASE 1: FIXING VARIABLES ---")

        # fixed_to_one = 0
        # fixed_to_zero = 0

        # Store and fix y variables
        for j, t in y_fixed:
            original_bounds[('y', j, t)] = (y[j, t].LB, y[j, t].UB)
            if solution.y[j, t] == 1:
                y[j, t].LB = y[j, t].UB = 1.0
                # --- DEBUG ONLY ---
                # fixed_to_one += 1
            else:
                y[j, t].LB = y[j, t].UB = 0.0
                # --- DEBUG ONLY ---
                # fixed_to_zero += 1

        # Store and fix overline_y variables
        for j, t in overline_y_fixed:
            original_bounds[('overline_y', j, t)] = (overline_y[j, t].LB, overline_y[j, t].UB)
            if solution.overline_y[j, t] == 1:
                overline_y[j, t].LB = overline_y[j, t].UB = 1.0
                # --- DEBUG ONLY ---
                # fixed_to_one += 1
            else:
                overline_y[j, t].LB = overline_y[j, t].UB = 0.0
                # --- DEBUG ONLY ---
                # fixed_to_zero += 1

        # Store and fix z variables
        for j, t in z_fixed:
            original_bounds[('z', j, t)] = (z[j, t].LB, z[j, t].UB)
            if solution.z[j, t] == 1:
                z[j, t].LB = z[j, t].UB = 1.0
                # --- DEBUG ONLY ---
                # fixed_to_one += 1
            else:
                z[j, t].LB = z[j, t].UB = 0.0
                # --- DEBUG ONLY ---
                # fixed_to_zero += 1

        # --- DEBUG ONLY ---
        # print(f"> Fixed {fixed_to_one} variables to 1, {fixed_to_zero} variables to 0")
        # print(f"> Total fixed variables: {fixed_to_one + fixed_to_zero}")

        # ==============================================================================================================
        # PHASE 2: OPTIMIZE (no explicit update needed -- 'model.optimize()' calls it)
        # ==============================================================================================================
        # --- DEBUG ONLY ---
        # print("\n--- PHASE 2: OPTIMIZATION ---")

        # Message for the user
        print("> Solving the model with fixed variables...\n")

        # Solve the model
        model.optimize()

        # CRITICAL: Capture 'status' and 'solution_count' IMMEDIATELY after optimize
        # This must be done BEFORE any model modifications (including bound restoration)
        status = model.Status
        solution_count = model.SolCount

        # --- DEBUG ONLY ---
        # print(f"\n> Optimization completed:")
        # print(f"  - status: {status}")
        # print(f"  - solution count: {solution_count}\n")

        # ==============================================================================================================
        # PHASE 3: SOLUTION EXTRACTION WITH PROPER VALIDATION
        # ==============================================================================================================
        # --- DEBUG ONLY ---
        # print("\n--- PHASE 3: SOLUTION EXTRACTION ---")

        # Define acceptable statuses where solutions might be available
        acceptable_statuses = {
            GRB.OPTIMAL,         # 2  - Optimal solution found
            GRB.TIME_LIMIT,      # 9  - Time limit reached, solution may exist
            GRB.SUBOPTIMAL,      # 13 - Sub-optimal solution found
            GRB.SOLUTION_LIMIT,  # 10 - Solution limit reached
            GRB.INTERRUPTED,     # 11 - User interrupted, solution may exist
            GRB.USER_OBJ_LIMIT   # 15 - Objective limit reached
        }

        # Check if solution is available
        if status in acceptable_statuses and solution_count > 0:

            # Message for the user
            print(f"\n> Extracting the solution...\n")

            try:
                # Update 'NPV' attribute in the solution
                solution.NPV = model.ObjVal

                # Update 'F', 'S', 'R', 'X', and 'D' attributes in the solution
                for p in periods:
                    solution.F[p] = (1 - gamma) * (S[p].X + R[p].X) - X[p].X + gamma * D[p].X
                    solution.S[p] = S[p].X
                    solution.R[p] = R[p].X
                    solution.X[p] = X[p].X
                    solution.D[p] = D[p].X

                # Update 'x', 'y', 'overline_y', and 'z' attributes in the solution
                for j in meter_groups:
                    for t in intervals:
                        solution.x[j, t] = x[j, t].X
                        solution.y[j, t] = y[j, t].X
                        solution.overline_y[j, t] = overline_y[j, t].X
                        solution.z[j, t] = z[j, t].X

            except AttributeError as e:
                print(f"> ERROR: Failed to extract solution: {e}\n")
                print("  Keeping original solution values!\n")

        else:
            # Map status codes to names for better diagnostics
            status_names = {
                1: "LOADED",
                2: "OPTIMAL",
                3: "INFEASIBLE",
                4: "INF_OR_UNBD",
                5: "UNBOUNDED",
                6: "CUTOFF",
                7: "ITERATION_LIMIT",
                8: "NODE_LIMIT",
                9: "TIME_LIMIT",
                10: "SOLUTION_LIMIT",
                11: "INTERRUPTED",
                12: "NUMERIC",
                13: "SUBOPTIMAL",
                14: "INPROGRESS",
                15: "USER_OBJ_LIMIT",
                16: "WORK_LIMIT",
                17: "MEM_LIMIT"
            }

            # Get status name
            status_name = status_names.get(status, f"UNKNOWN({status})")
            print("> WARNING: No feasible solution available:")
            print(f"  - status: {status_name}\n")
            print("> Keeping original solution values!")

    except Exception as e:
        print(f"> EXCEPTION during optimization: {e}\n")
        import traceback
        traceback.print_exc()

    finally:
        # ==============================================================================================================
        # PHASE 4: RESTORE ORIGINAL BOUNDS (CRITICAL FOR NEXT ITERATION)
        # ==============================================================================================================
        # --- DEBUG ONLY ---
        # print("\n--- PHASE 4: BOUND RESTORATION ---")

        # Restore all bounds
        for j, t in y_fixed:
            orig_lb, orig_ub = original_bounds[('y', j, t)]
            y[j, t].LB, y[j, t].UB = orig_lb, orig_ub

        for j, t in overline_y_fixed:
            orig_lb, orig_ub = original_bounds[('overline_y', j, t)]
            overline_y[j, t].LB, overline_y[j, t].UB = orig_lb, orig_ub

        for j, t in z_fixed:
            orig_lb, orig_ub = original_bounds[('z', j, t)]
            z[j, t].LB, z[j, t].UB = orig_lb, orig_ub

        # --- DEBUG ONLY ---
        # print(f"> Restored bounds for {len(original_bounds)} variables")

        # CRITICAL: Update model to apply restored bounds
        # NOTE: This will change 'model.Status' to 1 (LOADED), but we've already captured the optimization status in 'status'
        model.update()

        # --- DEBUG ONLY ---
        # print("> Model updated with restored bounds!\n")
        # print(f"> Model Status is now: {model.Status} (LOADED - model modified)\n")

        # --- DEBUG ONLY ---
        print("=" * 114 + "\n")

    # Update 'metrics'
    eplapsed_tm = tm.perf_counter() - t2

    if neighborhood == 1:
        metrics.NumItersFirstNeighborhood += 1
        metrics.CumulativeRuntimeFirstNeighborhood += eplapsed_tm
    elif neighborhood == 2:
        metrics.NumItersSecondNeighborhood += 1
        metrics.CumulativeRuntimeSecondNeighborhood += eplapsed_tm
    elif neighborhood == 3:
        metrics.NumItersThirdNeighborhood += 1
        metrics.CumulativeRuntimeThirdNeighborhood += eplapsed_tm
    elif neighborhood == 4:
        metrics.NumItersFourthNeighborhood += 1
        metrics.CumulativeRuntimeFourthNeighborhood += eplapsed_tm

    return solution, status, solution_count


def getVariablesToFix(neighborhood):
    """ Separate function to determine which variables to fix. """

    # Define empty lists for fixed variables
    y_fixed = []
    overline_y_fixed = []
    z_fixed = []

    # Call the 'neighborhood'
    if neighborhood == 1:
        # Message to the user
        print(f"\n> Using \'neighborhood\' no. {neighborhood}...\n")

        """ > 1st 'neighborhood': fixes all variables for which a generated random value is greater than 'args.argBeta' """
        # Fix 'y' variables
        y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if rnd.random() > args.argBeta])

        # Fix 'overline_y' variables
        overline_y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if rnd.random() > args.argBeta])

        # Fix 'z' variables
        z_fixed = np.array([(j, t) for j in meter_groups for t in intervals if rnd.random() > args.argBeta])

    elif neighborhood == 2:
        # Message to the user
        print(f"\n> Using \'neighborhood\' no. {neighborhood}...\n")

        """ > 2nd 'neighborhood': randomly chooses a 'k'-size list of meter groups and fixes all variables for these meter groups """
        # Define 'k' and randomly choose a 'k'-size set of unique meter groups
        k = int(m.ceil(J * args.argChi))
        meter_group_fixed = set(rnd.sample(meter_groups, k=k))

        # Fix 'y' variables
        y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if j in meter_group_fixed])

        # Fix 'overline_y' variables
        overline_y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if j in meter_group_fixed])

        # Fix 'z' variables
        z_fixed = np.array([(j, t) for j in meter_groups for t in intervals if j in meter_group_fixed])

    elif neighborhood == 3:
        # Message to the user
        print(f"\n> Using \'neighborhood\' no. {neighborhood}...\n")

        """ > 3rd 'neighborhood': randomly chooses a starting interval and fixes all variables for 'k' intervals from this (i.e., re-starting from the beginning if necessary) """
        # Define 'k', randomly choose 'start_interval', and define a 'k'-size set of intervals from 'start_interval'
        k = int(m.ceil((SP * T) * args.argDelta))
        start_interval = rnd.choice(intervals)
        interval_fixed = set((start_interval + i) % len(intervals) for i in range(k))

        # Fix 'y' variables
        y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if t in interval_fixed])

        # Fix 'overline_y' variables
        overline_y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if t in interval_fixed])

        # Fix 'z' variables
        z_fixed = np.array([(j, t) for j in meter_groups for t in intervals if t in interval_fixed])

    elif neighborhood == 4:
        # Message to the user
        print(f"\n> Using \'neighborhood\' no. {neighborhood}...\n")

        """ > 4th 'neighborhood': randomly chooses a 'k'-size list of meter groups, a starting interval, and fixes all variables outside the 'k' intervals from this for meter groups that are not in the list """
        # Define 'k' and randomly choose a 'k'-size set of unique meter groups
        k_meter = int(m.ceil(J * args.argEpsilon))
        meter_group_fixed = set(rnd.sample(meter_groups, k=k_meter))

        # Define 'k', randomly choose 'start_interval', and define a 'k'-size set of intervals from 'start_interval'
        k_interval = int(m.ceil((SP * T) * args.argDelta))
        start_interval = rnd.choice(intervals)
        interval_fixed = set((start_interval + i) % len(intervals) for i in range(k_interval))

        # Fix 'y' variables
        y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if j not in meter_group_fixed or t not in interval_fixed])

        # Fix 'overline_y' variables
        overline_y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if j not in meter_group_fixed or t not in interval_fixed])

        # Fix 'z' variables
        z_fixed = np.array([(j, t) for j in meter_groups for t in intervals if j not in meter_group_fixed or t not in interval_fixed])

    return y_fixed, overline_y_fixed, z_fixed