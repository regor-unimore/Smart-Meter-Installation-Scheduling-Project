# import
from argumentParser import parseArguments
from parameters import setupParameters, setupIndexes
import random as rnd

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstanceName)
P, periods, meter_groups, substitution_squads, intervals = setupIndexes(J, K, SP, DH, T)

# Initialize the random number generator
rnd.seed(args.argSeed)

# ----------------------------------------------------------------------------------------------------------------------
# Functions implementing the large neighborhood search used to improve the solution built through the semi-greedy
# ----------------------------------------------------------------------------------------------------------------------
def largeNeighborhoodSearch(solution, model, x, y, overline_y, z, S, R, X, D):
    # Define empty lists 'y_fixed', 'overline_y_fixed', 'z_fixed'
    y_fixed = []
    overline_y_fixed = []
    z_fixed = []

    # Reset 'model'
    model.reset()

    # Randomly choose a 'fix' method
    fix_id = rnd.choice([1, 2, 3, 4])

    # Call the 'fix' method based on 'fix_id'
    if fix_id == 1:
        # Message to the user
        print("\n> Using \'fix\' method no. {}...\n".format(fix_id))

        """ > 1st 'fix' method: fixes all variables for which a generated random value is greater than 'args.argBeta' """
        # Fix 'y' variables
        y_fixed = [(j, k, t) for j in meter_groups for k in substitution_squads for t in intervals if solution.y[(j, k, t)] == 1 and rnd.random() > args.argBeta]

        for j, k, t in y_fixed:
            y[j, k, t].LB = 1.0  # Set lower bound to 1

        # Fix 'overline_y' variables
        overline_y_fixed = [(j, t) for j in meter_groups for t in intervals if solution.overline_y[(j, t)] == 1 and rnd.random() > args.argBeta]

        for j, t in overline_y_fixed:
            overline_y[j, t].LB = 1.0

        # Fix 'z' variables
        z_fixed = [(j, t) for j in meter_groups for t in intervals if solution.z[(j, t)] == 1 and rnd.random() > args.argBeta]

        for j, t in z_fixed:
            z[j, t].LB = 1.0

    elif fix_id == 2:
        # Message to the user
        print("\n> Using \'fix\' method no. {}...\n".format(fix_id))

        """ > 2nd 'fix' method: randomly chooses a 'k' sized list of meter groups and fixes all variables for these meter groups """
        # Define 'k' and randomly choose a 'k'-sized set of unique meter groups
        k = int(round(J * args.argGamma, 0))
        meter_group_fixed = set(rnd.sample(meter_groups, k=k))

        # Fix 'y' variables
        y_fixed = [(j, k, t) for j in meter_groups for k in substitution_squads for t in intervals if solution.y[(j, k, t)] == 1 and j in meter_group_fixed]

        for j, k, t in y_fixed:
            y[(j, k, t)].LB = 1.0  # Set lower bound to 1

        # Fix 'overline_y' variables
        overline_y_fixed = [(j, t) for j in meter_groups for t in intervals if solution.overline_y[(j, t)] == 1 and j in meter_group_fixed]

        for j, t in overline_y_fixed:
            overline_y[(j, t)].LB = 1.0

        # Fix 'z' variables
        z_fixed = [(j, t) for j in meter_groups for t in intervals if solution.z[(j, t)] == 1 and j in meter_group_fixed]

        for j, t in z_fixed:
            z[(j, t)].LB = 1.0

    elif fix_id == 3:
        # Message to the user
        print("\n> Using \'fix\' method no. {}...\n".format(fix_id))

        """ > 3rd 'fix' method: randomly chooses a 'k' sized list of intervals and fixes the heuristic solution for these intervals """
        # Define 'k' and randomly choose a 'k'-sized set of unique intervals
        k = int(round((SP * T) * args.argGamma, 0))
        interval_fixed = set(rnd.sample(intervals, k=k))

        # Fix 'y' variables
        y_fixed = [(j, k, t) for j in meter_groups for k in substitution_squads for t in intervals if solution.y[(j, k, t)] == 1 and t in interval_fixed]

        for j, k, t in y_fixed:
            y[(j, k, t)].LB = 1.0  # Set lower bound to 1

        # Fix 'overline_y' variables
        overline_y_fixed = [(j, t) for j in meter_groups for t in intervals if solution.overline_y[(j, t)] == 1 and t in interval_fixed]

        for j, t in overline_y_fixed:
            overline_y[(j, t)].LB = 1.0

        # Fix 'z' variables
        z_fixed = [(j, t) for j in meter_groups for t in intervals if solution.z[(j, t)] == 1 and t in interval_fixed]

        for j, t in z_fixed:
            z[(j, t)].LB = 1.0

    elif fix_id == 4:
        # Message to the user
        print("\n> Using \'fix\' method no. {}...\n".format(fix_id))

        """ > 4th 'fix' method: randomly chooses a 'k' sized list of substitution squads and fixes the heuristic solution for these squads """
        # Define 'k' and randomly choose a 'k'-sized set of unique substitution squads
        k = int(round(K * args.argGamma, 0))
        squad_fixed = set(rnd.sample(substitution_squads, k=k))

        # Precompute solution 'y' values for efficiency and identify the 'y' variables to fix
        solution_y = solution.y
        y_fixed = [(j, k, t) for j in meter_groups for k in substitution_squads for t in intervals if solution_y[(j, k, t)] == 1 and k in squad_fixed]

        # Fix 'y' variables
        for j, k, t in y_fixed:
            y[(j, k, t)].LB = 1.0

        # Precompute 'y_sum_squad_interval' (i.e., how many times each meter group has been assigned to a 'fixed' squad)
        y_sum_squad_interval = {j: sum(solution_y[(j, k, t)] for t in intervals for k in squad_fixed) for j in meter_groups}

        # Define the maximum number of intervals in which 'fixed' substitution squads may have worked in each meter group 'j' (could tune this value) -- HARDCODED
        y_sum_squad_interval_max = int(round(0.05 * (SP * T), 0))

        # Fix 'overline_y' variables (only if 'fixed' substitution squads has been working in meter group 'j' for at least 'y_sum_squad_interval_max' intervals)
        overline_y_fixed = [(j, t) for j in meter_groups for t in intervals if solution.overline_y[(j, t)] == 1 and y_sum_squad_interval[j] >= y_sum_squad_interval_max]

        for j, t in overline_y_fixed:
            overline_y[(j, t)].LB = 1.0

        # Fix 'z' variables (only if 'fixed' substitution squads has been working in meter group 'j' for at least 'y_sum_squad_interval_max' intervals)
        z_fixed = [(j, t) for j in meter_groups for t in intervals if solution.z[(j, t)] == 1 and y_sum_squad_interval[j] >= y_sum_squad_interval_max]

        for j, t in z_fixed:
            z[(j, t)].LB = 1.0

    # Message for the user
    print('> Solving the model with fixed variables...\n')

    # Solve the model
    model.optimize()

    # Message for the user
    print('\n> Computing the results...\n')

    # Update 'NPV' attribute in the solution
    solution.updateSolution({'NPV': model.ObjVal})

    # Update 'F', 'S', 'R', 'X', and 'D' attributes in the solution
    for p in periods:
        solution.updateSolution({
            'F': {p: (1 - gamma) * (S[p].X + R[p].X) - X[p].X + gamma * D[p].X},
            'S': {p: S[p].X},
            'R': {p: R[p].X},
            'X': {p: X[p].X},
            'D': {p: D[p].X}
        })

    # Update 'overline_y' and 'z' attributes in the solution
    for j in meter_groups:
        for t in intervals:
            solution.updateSolution({
                'overline_y': {(j, t): overline_y[(j, t)].X},
                'z': {(j, t): z[(j, t)].X}
            })

    # Update the 'x' and 'y' attributes in the solution
    for j in meter_groups:
        for k in substitution_squads:
            for t in intervals:
                solution.updateSolution({
                    'x': {(j, k, t): x[(j, k, t)].X},
                    'y': {(j, k, t): y[(j, k, t)].X}
                })

    # Unfix variables
    for j, k, t in y_fixed:
        y[j, k, t].LB = 0.0

    for j, t in overline_y_fixed:
        overline_y[j, t].LB = 0.0

    for j, t in z_fixed:
        z[j, t].LB = 0.0

    return solution