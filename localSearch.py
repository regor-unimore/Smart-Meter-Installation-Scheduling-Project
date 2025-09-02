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

# Initialize the random number generator
rnd.seed(args.argSeed)

# ----------------------------------------------------------------------------------------------------------------------
# Function(s) implementing the Variable MIP Neighborhood Descent used to improve the solution of the semi-greedy
# ----------------------------------------------------------------------------------------------------------------------
def variableFixing(var_dict, sol_dict, fixed_set):
    """ Fix variables in 'var_dict' based on current """
    for j, t in fixed_set:
        if sol_dict[j, t] == 1:
            var_dict[j, t].LB = 1.0 # Set lower bound to 1
        else:
            var_dict[j, t].UB = 0.0 # Set upper bound to 0

def variableUnfixing(var_dict, sol_dict, fixed_set):
    """ Restore original bounds for previously fixed variables """
    for j, t in fixed_set:
        if sol_dict[j, t] == 1:
            var_dict[j, t].LB = 0.0
        else:
            var_dict[j, t].UB = 1.0

def meterGroupNeighborhood(chi):
    """ Randomly chooses a 'k'-size set of meter groups and fixes all variables for these meter groups """
    # Define 'k' and randomly choose a 'k'-size set of unique meter groups
    k = int(m.ceil(len(meter_groups) * chi))
    meter_group_fixed = set(rnd.sample(meter_groups, k=k))

    return np.array([(j, t) for j in meter_groups for t in intervals if j in meter_group_fixed])

def intervalNeighborhood(delta):
    """ Randomly chooses a starting interval and fixes all variables for 'k' consecutive intervals (i.e., re-starting from the beginning if necessary) """
    # Define 'k', randomly choose 'start_interval', and define a 'k'-size set of intervals from 'start_interval'
    k = int(m.ceil(len(intervals) * delta))
    start_interval = rnd.choice(intervals)
    interval_fixed = set((start_interval + i) % len(intervals) for i in range(k))

    return np.array([(j, t) for j in meter_groups for t in intervals if t in interval_fixed])

def combinedNeighborhood(epsilon, delta):
    """ Randomly chooses a 'k'-size list of meter groups, a starting interval, and fixes all variables outside the 'k' consecutive intervals for meter groups that are not in the list """
    # Define 'k_meter' and randomly choose a 'k'-size set of unique meter groups
    k_meter = int(m.ceil(len(meter_groups) * epsilon))
    meter_group_fixed = set(rnd.sample(meter_groups, k=k_meter))

    # Define 'k_interval', randomly choose 'start_interval', and define a 'k'-size set of intervals from 'start_interval'
    k_interval = int(m.ceil(len(intervals) * delta))
    start_interval = rnd.choice(intervals)
    interval_fixed = set((start_interval + i) % len(intervals) for i in range(k_interval))

    return np.array([(j, t) for j in meter_groups for t in intervals if j not in meter_group_fixed or t not in interval_fixed])

def sparseNeighborhood(beta):
    """ Fixes all variables for which a generated random value is greater than 'beta' """
    return np.array([(j, t) for j in meter_groups for t in intervals if rnd.random() > beta])

# Map 'NEIGHBORHOOD' to the corresponding strategy
NEIGHBORHOOD = {
    1: lambda: meterGroupNeighborhood(args.argChi),
    2: lambda: intervalNeighborhood(args.argDelta),
    3: lambda: combinedNeighborhood(args.argEpsilon, args.argDelta),
    4: lambda: sparseNeighborhood(args.argBeta)
}

def variableMIPNeighborhoodDescent(solution, model, x, y, overline_y, z, S, R, X, D, metrics, neighborhood):
    # Get 't2'
    t2 = tm.perf_counter()

    # Reset 'model'
    model.reset()

    # Update current 'neighborhood'
    metrics.Neighborhood = neighborhood

    # Message to the user
    print("\n> Using \'fix\' method no. {}...\n".format(neighborhood))

    # Select the neighborhood based on 'neighborhood'
    fixed_set = NEIGHBORHOOD[neighborhood]()
    for var, sol in [(y, solution.y), (overline_y, solution.overline_y), (z, solution.z)]:
        variableFixing(var, sol, fixed_set)

    # Message for the user
    print('> Solving the model with fixed variables...\n')

    # Solve the model
    model.optimize()

    # Message for the user
    print('\n> Computing the results...\n')

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
                    print(
                        f"Error: Unable to retrieve attribute 'X' for meter group {j}, interval {t}. Skipping...\n")
                    continue
    else:
        print(f"> WARNING: Model status: {modelStatus(model)}. Skipping update...\n")

    # Unfix variables
    for var, sol in [(y, solution.y), (overline_y, solution.overline_y), (z, solution.z)]:
        variableUnfixing(var, sol, fixed_set)

    # Update 'metrics'
    tm_elapsed = tm.perf_counter() - t2
    if neighborhood == 1:
        metrics.NumItersFirstNeighborhood += 1
        metrics.CumulativeRuntimeFirstNeighborhood += tm_elapsed
    elif neighborhood == 2:
        metrics.NumItersSecondNeighborhood += 1
        metrics.CumulativeRuntimeSecondNeighborhood += tm_elapsed
    elif neighborhood == 3:
        metrics.NumItersThirdNeighborhood += 1
        metrics.CumulativeRuntimeThirdNeighborhood += tm_elapsed
    elif neighborhood == 4:
        metrics.NumItersFourthNeighborhood += 1
        metrics.CumulativeRuntimeFourthNeighborhood += tm_elapsed

    return solution