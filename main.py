# import
from argumentParser import parseArguments
from classes import Solution, Metrics
from localSearch import MIPNeighborhoodSearch
from model import createModel, modelStatus, solutionCallback
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
m, x, y, overline_y, z, S, R, X, D = createModel()

# Select the app based on 'argApp'

# Run the model
if args.argSolutionMethod == 'branch-and-cut':

    # Message for the user
    print('> Solving the model...\n')

    # Clear the output file at the start
    with open('callbacks/results_bc_' + instanceName + '.txt', "w") as cbFile:
        cbFile.write("MIP solution callback(s):")
        cbFile.write("\n+-----------------+-----------------+-----------------+-----------------+")
        cbFile.write("\n|   Solution Node |       Incumbent |            Time |  Solution Count |")
        cbFile.write("\n+-----------------+-----------------+-----------------+-----------------+")

    # Solve the model
    m.optimize(lambda model, where: solutionCallback(model, where, instanceName))

    with open('callbacks/results_bc_' + instanceName + '.txt', "a") as cbFile:
        cbFile.write("\n+-----------------+-----------------+-----------------+-----------------+")

    # Write the model solution to file
    writeOutputModelFile(args.argInstanceName, m)

# Run the algorithm
elif args.argSolutionMethod == 'grasp':

    # Message for the user
    print('\n> Running the algorithm...\n')

    # Define object 'bestSolution' of class 'Solution'
    bestSolution = Solution()

    # Get 'start_tm'
    t1 = tm.perf_counter()

    # Define object 'metrics' of class 'Metrics'
    metrics = Metrics()

    # Main loop of the algorithm
    while metrics.NumIters < args.argMaxIter:
        # Message for the user
        print('> Iteration: {}\n'.format(metrics.NumIters))

        # Run the greedy randomized constructive heuristics to build an initial feasible solution
        solution = greedyRandomizedConstructiveHeuristics()

        # Define object 'localSearchSolution' of class 'Solution' and initialize it
        localSearchSolution = Solution()
        localSearchSolution.updateFromSolution(solution)

        # Message for the user
        print('  NPV_curr: {:.2f}\n'.format(solution.NPV))

        # Message for the user
        print('> Improving the solution via local search...\n')

        # Counter for the number of iterations without improvement in the local search
        noImprovementIters = 0

        for _ in range(args.argMaxLocalSearchIter):
            # Run the local search and update the current solution
            solution.updateFromSolution(MIPNeighborhoodSearch(localSearchSolution, m, x, y, overline_y, z, S, R, X, D, metrics))

            # If model status is 2 -- 'OPTIMAL' or 9 -- 'TIME_LIMIT'
            if modelStatus(m) in [2, 9]:
                # Message for the user
                print("  NPV_curr: {:.2f}\n".format(solution.NPV))

                # Check whether a new local search solution has been found
                if solution.NPV - localSearchSolution.NPV > 0.001:
                    localSearchSolution.updateFromSolution(solution)
                    noImprovementIters = 0  # Reset counter
                else:
                    noImprovementIters += 1 # Update counter

                # Check whether a new incumbent solution has been found
                if solution.NPV - bestSolution.NPV > 0.001:
                    # Message for the user
                    print('  New incumbent solution found!\n')

                    # Update 'metrics'
                    metrics.NumItersBest = metrics.NumIters
                    metrics.RuntimeBest = tm.perf_counter() - t1
                    metrics.FixMethodBest = metrics.FixMethod
                    metrics.NumImprovements += 1

                    if metrics.FixMethod == 1:
                        metrics.NumImprovementsFirstMethod += 1
                    elif metrics.FixMethod == 2:
                        metrics.NumImprovementsSecondMethod += 1
                    elif metrics.FixMethod == 3:
                        metrics.NumImprovementsThirdMethod += 1
                    elif metrics.FixMethod == 4:
                        metrics.NumImprovementsFourthMethod += 1

                    # Update 'best_solution'
                    bestSolution.updateFromSolution(solution)

                    # Message for the user
                    print('  NPV_incumbent: {:.2f}\n'.format(bestSolution.NPV))

                    # Breakout condition: 'break' if there is no improvement for 'args.argNoImprovementIter' iterations
                    if noImprovementIters >= args.argNoImprovementIter:
                        # Message for the user
                        print('  No improvements for {} iterations! BREAKING...\n'.format(noImprovementIters))
                        break

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
    print('  NPV_best: {:.2f}'.format(bestSolution.NPV))

    # Write the best solution to file
    writeOutputAlgorithmFile(args.argInstanceName, bestSolution, metrics, periods, meter_groups, intervals, args.argAlpha, args.argBeta, args.argGamma, args.argMaxIter, args.argSeed)