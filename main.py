# import
from argumentParser import parseArguments
from classes import Solution
from localSearch import largeNeighborhoodSearch
from model import createModel
from parameters import setupParameters, setupIndexes
from semiGreedy import greedyRandomizedConstructiveHeuristics
from utils import writeOutputModelFile, writeOutputAlgorithmFile
import time as tm

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstance)
P, periods, meter_groups, substitution_squads, intervals = setupIndexes(J, K, SP, DH, T)

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

# Create the model
model, x, y, overline_y, z, S, R, X, D = createModel()

# Select the app based on 'argApp'

# Run the model
if args.argSolutionMethod == 'branch-and-cut':

    # Message for the user
    print('> Solving the model...\n')

    # Solve the model
    model.optimize()

    # Write the model solution to file
    writeOutputModelFile(args.argInstance, model)

# Run the algorithm
elif args.argSolutionMethod == 'grasp':

    # Message for the user
    print('\n> Running the algorithm...\n')

    # Define object 'best_solution' of class 'Solution'
    best_solution = Solution()

    # Get 't1'
    t1 = tm.perf_counter()

    # Initialise 'iteration'
    iteration = 0

    # Initialise 'improvements'
    improvements = 0

    # Main loop of the algorithm
    while iteration < args.argMaxIter:
        # Message for the user
        print('> Iteration: {}\n'.format(iteration))

        # Run the greedy randomized constructive heuristics to build an initial feasible solution
        solution = greedyRandomizedConstructiveHeuristics()

        # Message for the user
        print('  NPV_curr: {:.2f}\n'.format(solution.NPV))

        # Compute the gap between the current solution and the incumbent solution
        # gap = (best_solution.NPV - solution.NPV) / solution.NPV

        # Check whether the gap between the NPV of the current solution and the NPV of the incumbent solution is less than 5%. If so, try to improve it via local search
        # if gap < 0.05:

        # Message for the user
        print('> Improving the solution via local search...\n')

        # Run the local search and update the current solution
        solution.updateFromSolution(largeNeighborhoodSearch(solution, model, x, y, overline_y, z, S, R, X, D))

        # Message for the user
        print("  NPV_curr: {:.2f}\n".format(solution.NPV))

        # Check whether a new incumbent solution has been found
        if solution.NPV > best_solution.NPV:
            # Message for the user
            print('  New incumbent solution found!\n')

            # Update 'improvement'
            improvements += 1

            # Update the best solution
            best_solution.updateFromSolution(solution)

            # Message for the user
            print('  NPV_incumbent: {:.2f}\n'.format(best_solution.NPV))

        # Update 'iter_curr'
        iteration += 1

    # Get 't2'
    t2 = tm.perf_counter()

    # Compute 'computational_tm' for running the algorithm
    computational_tm = t2 - t1

    # Message for the user
    print('> Process finished! Found {} improvement(s).\n'.format(improvements))

    # Message for the user
    print('  NPV_best: {:.2f}'.format(best_solution.NPV))

    # Write the best solution to file
    writeOutputAlgorithmFile(args.argInstance, best_solution, computational_tm, periods, meter_groups, intervals, substitution_squads)