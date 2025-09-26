# import
from argumentParser import parseArguments
from model import modelStatus
from parameters import setupParameters, setupIndexes
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
    # Get 't2'
    t2 = tm.perf_counter()

    # Store original bounds for restoration
    original_bounds = {}

    # Reset 'model'
    model.reset()

    # Get variables to fix based on 'neighborhood'
    y_fixed, overline_y_fixed, z_fixed = getVariablesToFix(neighborhood)

    try:
        # Store and fix y variables
        for j, t in y_fixed:
            original_bounds[('y', j, t)] = (y[j, t].LB, y[j, t].UB)
            if solution.y[j, t] == 1:
                y[j, t].LB = y[j, t].UB = 1.0
            else:
                y[j, t].LB = y[j, t].UB = 0.0

        # Store and fix overline_y variables
        for j, t in overline_y_fixed:
            original_bounds[('overline_y', j, t)] = (overline_y[j, t].LB, overline_y[j, t].UB)
            if solution.overline_y[j, t] == 1:
                overline_y[j, t].LB = overline_y[j, t].UB = 1.0
            else:
                overline_y[j, t].LB = overline_y[j, t].UB = 0.0

        # Store and fix z variables
        for j, t in z_fixed:
            original_bounds[('z', j, t)] = (z[j, t].LB, z[j, t].UB)
            if solution.z[j, t] == 1:
                z[j, t].LB = z[j, t].UB = 1.0
            else:
                z[j, t].LB = z[j, t].UB = 0.0

        # Update the model
        model.update()

        # Message for the user
        print("> Solving the model with fixed variables...\n")

        # Solve the model
        model.optimize()

        # Message for the user
        print("\n> Computing the results...\n")

        # If model status is 2 -- 'OPTIMAL' or 9 -- 'TIME_LIMIT'
        if modelStatus(model) in [2, 9]:

            # Update 'NPV' attribute in the solution
            solution.NPV = model.ObjVal

            # Update 'F', 'S', 'R', 'X', and 'D' attributes in the solution
            for p in periods:
                try:
                    solution.F[p] = (1 - gamma) * (S[p].X + R[p].X) - X[p].X + gamma * D[p].X
                    solution.S[p] = S[p].X
                    solution.R[p] = R[p].X
                    solution.X[p] = X[p].X
                    solution.D[p] = D[p].X
                except AttributeError:
                    print(f"Error: Unable to retrieve attribute 'X' for period {p}. Skipping...\n")
                    continue

            # Update 'x', 'y', 'overline_y', and 'z' attributes in the solution
            for j in meter_groups:
                for t in intervals:
                    try:
                        solution.x[j, t] = x[j, t].X
                        solution.y[j, t] = y[j, t].X
                        solution.overline_y[j, t] = overline_y[j, t].X
                        solution.z[j, t] = z[j, t].X
                    except AttributeError:
                        print(f"Error: Unable to retrieve attribute 'X' for meter group {j}, interval {t}. Skipping...\n")
                        continue
        else:
            print("> WARNING: Model did not return a feasible or optimal solution. Skipping update...\n")

    finally:
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

        # Update the model
        model.update()

    # Update 'metrics'
    if neighborhood == 1:
        metrics.NumItersFirstNeighborhood += 1
        metrics.CumulativeRuntimeFirstNeighborhood += tm.perf_counter() - t2
    elif neighborhood == 2:
        metrics.NumItersSecondNeighborhood += 1
        metrics.CumulativeRuntimeSecondNeighborhood += tm.perf_counter() - t2
    elif neighborhood == 3:
        metrics.NumItersThirdNeighborhood += 1
        metrics.CumulativeRuntimeThirdNeighborhood += tm.perf_counter() - t2
    elif neighborhood == 4:
        metrics.NumItersFourthNeighborhood += 1
        metrics.CumulativeRuntimeFourthNeighborhood += tm.perf_counter() - t2

    return solution

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