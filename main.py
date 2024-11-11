# import
from argumentParser import parseArguments
from classes import Solution, Metrics
from localSearch import largeNeighborhoodSearch
from model import createModel, modelStatus
from parameters import setupParameters, setupIndexes
from semiGreedy import greedyRandomizedConstructiveHeuristics
from utils import writeOutputModelFile, writeOutputAlgorithmFile
import time as tm

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstanceName)
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
    writeOutputModelFile(args.argInstanceName, model)

# Run the algorithm
elif args.argSolutionMethod == 'grasp':

    # Message for the user
    print('\n> Running the algorithm...\n')

    # Define object 'best_solution' of class 'Solution'
    best_solution = Solution()

    # Get 'start_tm'
    t1 = tm.perf_counter()

    # Define object 'algo_metrics' of class 'Metrics'
    metrics = Metrics()

    # Main loop of the algorithm
    while metrics.NumIters < args.argMaxIter:
        # Message for the user
        print('> Iteration: {}\n'.format(metrics.NumIters))

        # Run the greedy randomized constructive heuristics to build an initial feasible solution
        solution = greedyRandomizedConstructiveHeuristics()

        # Message for the user
        print('  NPV_curr: {:.2f}\n'.format(solution.NPV))

        # Message for the user
        print('> Improving the solution via local search...\n')

        # Run the local search and update the current solution
        solution.updateFromSolution(largeNeighborhoodSearch(solution, model, x, y, overline_y, z, S, R, X, D, metrics))

        # If model status is 2 -- 'OPTIMAL' or 9 -- 'TIME_LIMIT'
        if modelStatus(model) in [2, 9]:
            # Message for the user
            print("  NPV_curr: {:.2f}\n".format(solution.NPV))

            # Check whether a new incumbent solution has been found
            if solution.NPV > best_solution.NPV:
                # Message for the user
                print('  New incumbent solution found!\n')

                # Update 'BestIter'
                metrics.BestIter = metrics.NumIters

                # Update 'BestRuntime'
                metrics.BestRuntime = tm.perf_counter() - t1

                # Update 'NumImprovements'
                metrics.NumImprovements += 1

                # Update 'best_solution'
                best_solution.updateFromSolution(solution)

                # Message for the user
                print('  NPV_incumbent: {:.2f}\n'.format(best_solution.NPV))

        # Increment 'NumIters'
        metrics.NumIters += 1

        # Check whether the time limit has been reached
        if tm.perf_counter() - t1 >= 3600.00:
            break

    # Compute 'Runtime'
    metrics.Runtime = tm.perf_counter() - t1

    # Message for the user
    print('> Process finished! Found {} improvement(s).\n'.format(metrics.NumImprovements))

    # Message for the user
    print('  NPV_best: {:.2f}'.format(best_solution.NPV))

    # Write the best solution to file
    writeOutputAlgorithmFile(args.argInstanceName, best_solution, metrics, periods, meter_groups, intervals, substitution_squads, args.argAlpha, args.argBeta, args.argGamma, args.argMaxIter, args.argSeed)