# import
from argumentParser import parseArguments
from classes import Solution, Metrics
from localSearch import variableMIPNeighborhoodDescent
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

    # Clear the output file at the start
    with open('output/info/results_grasp_' + instanceName + '_' + str(args.argAlpha) + '_' + str(args.argBeta).replace('.', '_') + '_' + str(args.argChi).replace('.', '_') + '_' + str(args.argDelta).replace('.', '_') + '_'+ str(args.argEpsilon).replace('.', '_') + '_' + str(args.argMaxIter) + '_' + str(args.argSeed) + '.txt', "w") as graspFile:
        graspFile.write("Solution improvement(s):")
        graspFile.write("\n+-------------------+-------------------+--------------------+")
        graspFile.write("\n|         Iteration | NPV (semi-greedy) | NPV (local search) |")
        graspFile.write("\n+-------------------+-------------------+--------------------+")

    # Main loop of the algorithm
    while metrics.NumIters < args.argMaxIter:
        # Message for the user
        print('> Iteration: {}\n'.format(metrics.NumIters))

        # SOLUTION CONSTRUCTION: run the semi-greedy (or randomized-greedy) algorithm
        currentSolution = greedyRandomizedConstructiveHeuristics()

        # Message for the user
        print('  NPV_curr: {:.2f}\n'.format(currentSolution.NPV))

        with open('output/info/results_grasp_' + instanceName + '_' + str(args.argAlpha) + '_' + str(args.argBeta).replace('.', '_') + '_' + str(args.argChi).replace('.', '_') + '_' + str(args.argDelta).replace('.', '_') + '_' + str(args.argEpsilon).replace('.', '_') + '_' + str(args.argMaxIter) + '_' + str(args.argSeed) + '.txt', "a") as graspFile:
            graspFile.write(f"\n| {metrics.NumIters:>17.0f} | {currentSolution.NPV:>17.2f} |")

        # LOCAL SEARCH: run the Variable MIP Neighborhood Descent

        # Message for the user
        print('> Improving the solution via local search...\n')

        # Define 'neighborhood' iterator
        neighborhood = 2

        while neighborhood <= 4:
            newSolution = currentSolution.copy()
            newSolution.updateFromSolution(variableMIPNeighborhoodDescent(newSolution, m, x, y, overline_y, z, S, R, X, D, metrics, neighborhood))

            # If model status is 2 -- 'OPTIMAL' or 9 -- 'TIME_LIMIT'
            if modelStatus(m) in [2, 9]:
                # Message for the user
                print("  NPV_new: {:.2f}\n".format(newSolution.NPV))

                # Check whether the solution of the model has been tied or improved
                if metrics.RuntimeTieModel is None and args.argSolutionValue is not None and (abs(newSolution.NPV - args.argSolutionValue) <= 0.01 or newSolution.NPV - args.argSolutionValue > 0.01):
                    metrics.RuntimeTieModel = tm.perf_counter() - t1

                # Check whether a new current solution has been found
                if newSolution.NPV - currentSolution.NPV > 0.001:
                    currentSolution.updateFromSolution(newSolution)
                    neighborhood = 2 # Reset iterator
                else:
                    neighborhood += 1 # Update iterator
            else:
                neighborhood += 1  # Update iterator

        with open('output/info/results_grasp_' + instanceName + '_' + str(args.argAlpha) + '_' + str(args.argBeta).replace('.', '_') + '_' + str(args.argChi).replace('.', '_') + '_' + str(args.argDelta).replace('.', '_') + '_' + str(args.argEpsilon).replace('.', '_') + '_' + str(args.argMaxIter) + '_' + str(args.argSeed) + '.txt', "a") as graspFile:
            graspFile.write(f"  {currentSolution.NPV:>17.2f} |")

        # Check whether a new incumbent solution has been found
        if currentSolution.NPV - bestSolution.NPV > 0.001:
            # Message for the user
            print('  New incumbent solution found!\n')

            # Update 'metrics'
            metrics.NumItersBest = metrics.NumIters
            metrics.RuntimeBest = tm.perf_counter() - t1

            # Update 'best_solution'
            bestSolution.updateFromSolution(currentSolution)

        # Increment 'NumIters'
        metrics.NumIters += 1

        # Check whether the time limit has been reached
        if tm.perf_counter() - t1 >= 3600.0:
            break

    with open('output/info/results_grasp_' + instanceName + '_' + str(args.argAlpha) + '_' + str(args.argBeta).replace('.', '_') + '_' + str(args.argChi).replace('.', '_') + '_' + str(args.argDelta).replace('.', '_') + '_' + str(args.argEpsilon).replace('.', '_') + '_' + str(args.argMaxIter) + '_' + str(args.argSeed) + '.txt', "a") as graspFile:
        graspFile.write("\n+-------------------+-------------------+--------------------+")

    # Compute 'Runtime'
    metrics.Runtime = tm.perf_counter() - t1

    # Message for the user
    print('> Process finished!\n')

    # Message for the user
    print('  NPV_best: {:.2f}'.format(bestSolution.NPV))

    # Write the best solution to file
    writeOutputAlgorithmFile(args.argInstanceName, bestSolution, metrics, periods, meter_groups, intervals, args.argAlpha, args.argBeta, args.argChi, args.argDelta, args.argEpsilon, args.argMaxIter, args.argSeed)