# import
from argumentParser import parseArguments
from parameters import setupParameters, setupIndexes
import math as m
import numpy as np
import random as rnd
import time as tm

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstanceName)
P, periods, meter_groups, substitution_squads, intervals = setupIndexes(J, K, SP, DH, T)

# Initialize the random number generator
rnd.seed(args.argSeed)

# ----------------------------------------------------------------------------------------------------------------------
# Function(s) implementing the large neighborhood search heuristic used to improve the semi-greedy solution
# ----------------------------------------------------------------------------------------------------------------------
def largeNeighborhoodSearch(solution, model, x, y, overline_y, z, S, R, X, D, metrics):
    # Get 't2'
    t2 = tm.perf_counter()

    # Define empty lists 'y_fixed', 'overline_y_fixed', 'z_fixed'
    y_fixed = []
    overline_y_fixed = []
    z_fixed = []

    # Reset 'model'
    model.reset()

    # Randomly choose a 'fix' method
    fix_method = rnd.choice([1, 2, 3])

    # Update current 'fix' method
    metrics.FixMethod = fix_method

    # Call the 'fix' method based on 'fix_method'
    if fix_method == 1:
        # Message to the user
        print("\n> Using \'fix\' method no. {}...\n".format(fix_method))

        # Update number of iterations for 1st 'fix' method
        metrics.NumItersFirstMethod += 1

        """ > 1st 'fix' method: fixes all variables for which a generated random value is greater than 'args.argBeta' """
        # Fix 'y' variables
        y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if solution.y[j, t] == 1 and rnd.random() > args.argBeta])

        for j, t in y_fixed:
            y[j, t].LB = 1.0  # Set lower bound to 1

        # Fix 'overline_y' variables
        overline_y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if solution.overline_y[j, t] == 1 and rnd.random() > args.argBeta])

        for j, t in overline_y_fixed:
            overline_y[j, t].LB = 1.0

        # Fix 'z' variables
        z_fixed = np.array([(j, t) for j in meter_groups for t in intervals if solution.z[j, t] == 1 and rnd.random() > args.argBeta])

        for j, t in z_fixed:
            z[j, t].LB = 1.0

    elif fix_method == 2:
        # Message to the user
        print("\n> Using \'fix\' method no. {}...\n".format(fix_method))

        # Update number of iterations for 2nd 'fix' method
        metrics.NumItersSecondMethod += 1

        """ > 2nd 'fix' method: randomly chooses a 'k' sized list of meter groups and fixes all variables for these meter groups """
        # Define 'k' and randomly choose a 'k'-sized set of unique meter groups
        k = int(m.ceil(J * args.argGamma))
        meter_group_fixed = set(rnd.sample(meter_groups, k=k))

        # Fix 'y' variables
        y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if solution.y[j, t] == 1 and j in meter_group_fixed])

        for j, t in y_fixed:
            y[j, t].LB = 1.0  # Set lower bound to 1

        # Fix 'overline_y' variables
        overline_y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if solution.overline_y[j, t] == 1 and j in meter_group_fixed])

        for j, t in overline_y_fixed:
            overline_y[j, t].LB = 1.0

        # Fix 'z' variables
        z_fixed = np.array([(j, t) for j in meter_groups for t in intervals if solution.z[j, t] == 1 and j in meter_group_fixed])

        for j, t in z_fixed:
            z[j, t].LB = 1.0

    elif fix_method == 3:
        # Message to the user
        print("\n> Using \'fix\' method no. {}...\n".format(fix_method))

        # Update number of iterations for 3rd 'fix' method
        metrics.NumItersThirdMethod += 1

        """ > 3rd 'fix' method: randomly chooses a starting interval and fixes the heuristic solution for 'k' intervals from this """
        # Define 'k', randomly choose 'start_interval', and define a 'k'-sized set of intervals from 'start_interval'
        k = int(m.ceil((SP * T) * args.argGamma))
        start_interval = rnd.choice(intervals)
        interval_fixed = set((start_interval + i) % len(intervals) for i in range(k))

        # Fix 'y' variables
        y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if solution.y[j, t] == 1 and t in interval_fixed])

        for j, t in y_fixed:
            y[j, t].LB = 1.0  # Set lower bound to 1

        # Fix 'overline_y' variables
        overline_y_fixed = np.array([(j, t) for j in meter_groups for t in intervals if solution.overline_y[j, t] == 1 and t in interval_fixed])

        for j, t in overline_y_fixed:
            overline_y[j, t].LB = 1.0

        # Fix 'z' variables
        z_fixed = np.array([(j, t) for j in meter_groups for t in intervals if solution.z[j, t] == 1 and t in interval_fixed])

        for j, t in z_fixed:
            z[j, t].LB = 1.0

    # Message for the user
    print('> Solving the model with fixed variables...\n')

    # Solve the model
    model.optimize()

    # Message for the user
    print('\n> Computing the results...\n')

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

    # Unfix variables
    for j, t in y_fixed:
        y[j, t].LB = 0.0

    for j, t in overline_y_fixed:
        overline_y[j, t].LB = 0.0

    for j, t in z_fixed:
        z[j, t].LB = 0.0

    # Update 'metrics'
    if fix_method == 1:
        # Update cumulative runtime for 1st 'fix' method
        metrics.CumulativeRuntimeFirstMethod += tm.perf_counter() - t2
    elif fix_method == 2:
        # Update cumulative runtime for 2nd 'fix' method
        metrics.CumulativeRuntimeSecondMethod += tm.perf_counter() - t2
    elif fix_method == 3:
        # Update cumulative runtime for 3rd 'fix' method
        metrics.CumulativeRuntimeThirdMethod += tm.perf_counter() - t2

    return solution