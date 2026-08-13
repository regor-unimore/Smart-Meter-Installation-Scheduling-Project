# import
from config import args, instanceName, periods, groups, intervals
from classes import Solution, Metrics
from gurobipy import GRB
from model import createModel, solutionCallback
from semiGreedy import greedyRandomizedConstructiveHeuristics
from localSearch import variableMIPNeighborhoodDescent
from utils import writeOutputModelFile, writeOutputAlgorithmFile
import time as tm

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

# Create the model
m, env, x, y, overline_y, z, S, R, X, D = createModel()

# Use context managers for proper resource management
with (env, m):

    # Run the model
    if args.argSolutionMethod == "branch-and-cut":

        # Message for the user
        print("> Solving the model...\n")

        # Clear the output file at the start
        with open("callbacks/results_bc_" + instanceName + '.txt', "w") as cbFile:
            cbFile.write("MIP solution callback(s):")
            cbFile.write("\n+-----------------+-----------------+-----------------+-----------------+")
            cbFile.write("\n|   Solution Node |       Incumbent |            Time |  Solution Count |")
            cbFile.write("\n+-----------------+-----------------+-----------------+-----------------+")

        # Solve the model
        try:
            m.optimize(lambda model, where: solutionCallback(model, where, instanceName))

            with open("callbacks/results_bc_" + instanceName + ".txt", "a") as cbFile:
                cbFile.write("\n+-----------------+-----------------+-----------------+-----------------+")

            # Write the model solution to file
            writeOutputModelFile(args.argInstanceName, m)

        except Exception as e:
            print(f"\n> Optimization failed: {e}\n")

    # Run the algorithm
    elif args.argSolutionMethod == 'grasp':

        # Message for the user
        print("\n> Running the algorithm...\n")

        # Define object 'bestSolution' of class 'Solution'
        bestSolution = Solution()

        # Get 'start_tm'
        t1 = tm.perf_counter()

        # Define object 'metrics' of class 'Metrics'
        metrics = Metrics()

        # Clear the output file at the start
        with open("output/info/results_grasp_" + instanceName + "_" + str(args.argAlpha) + "_" + str(args.argBeta).replace(".", "_") + "_" + str(args.argChi).replace(".", "_") + "_" + str(args.argDelta).replace(".", "_") + "_" + str(args.argMaxIter) + "_" + str(args.argSeed) + ".txt", "w") as graspFile:
            graspFile.write("Solution improvement(s):")
            graspFile.write("\n+-------------------+-------------------+--------------------+")
            graspFile.write("\n|         Iteration | NPV (semi-greedy) | NPV (local search) |")
            graspFile.write("\n+-------------------+-------------------+--------------------+")

        # Main loop of the algorithm
        while metrics.NumIters < args.argMaxIter:
            # Message for the user
            print(f"> Iteration: {metrics.NumIters}\n")

            # SOLUTION CONSTRUCTION: run the semi-greedy algorithm (or greedy randomized constructive heuristic)
            currentSolution = greedyRandomizedConstructiveHeuristics()

            # Message for the user
            print(f"--> NPV_curr: {currentSolution.NPV:.2f}\n")

            with open("output/info/results_grasp_" + instanceName + "_" + str(args.argAlpha) + "_" + str(args.argBeta).replace(".", "_") + "_" + str(args.argChi).replace(".", "_") + "_" + str(args.argDelta).replace(".", "_") + "_" + str(args.argMaxIter) + "_" + str(args.argSeed) + ".txt", "a") as graspFile:
                graspFile.write(f"\n| {metrics.NumIters:>17.0f} | {currentSolution.NPV:>17.2f} |")

            # LOCAL SEARCH: run the Variable MIP Neighborhood Descent

            # Message for the user
            print("> Improving the solution via variable MIP neighborhood descent...\n")

            # Define 'neighborhood' iterator
            neighborhood = 1

            while neighborhood <= 3:
                # Copy current solution
                newSolution = currentSolution.copy()

                # Apply neighborhood descent and CAPTURE the model status
                newSolution, status, solution_count = variableMIPNeighborhoodDescent(newSolution, m, x, y, overline_y, z, S, R, X, D, metrics, neighborhood)

                # Define acceptable statuses
                acceptable_statuses = {
                    GRB.OPTIMAL,
                    GRB.TIME_LIMIT,
                    GRB.SUBOPTIMAL,
                    GRB.SOLUTION_LIMIT,
                    GRB.INTERRUPTED,
                    GRB.USER_OBJ_LIMIT
                }

                # Use the CAPTURED model status and solution count, not 'model.Status'
                # (because 'model.Status' is now LOADED after update())
                if status in acceptable_statuses and solution_count > 0:
                    print(f"--> NPV_new: {newSolution.NPV:.2f}\n")

                    # Check whether the solution of the model has been tied or improved
                    if metrics.RuntimeTieModel is None and args.argSolutionValue is not None and (abs(newSolution.NPV - args.argSolutionValue) <= 0.01 or newSolution.NPV - args.argSolutionValue > 0.01):
                        metrics.RuntimeTieModel = tm.perf_counter() - t1

                    # Check whether a new current solution has been found
                    # Using tolerance of 0.001 to handle numerical precision
                    if newSolution.NPV - currentSolution.NPV > 0.001:

                        # Update current solution
                        currentSolution.updateFromSolution(newSolution)

                        # Reset 'neighborhood' iterator to explore the same neighborhood again
                        neighborhood = 1
                    else:
                        # Explore the next neighborhood
                        neighborhood += 1
                else:
                    # No feasible solution found in this neighborhood
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
                        15: "USER_OBJ_LIMIT"
                    }

                    # Get status name
                    status_name = status_names.get(status, f"UNKNOWN({status})")

                    # Explore the next neighborhood
                    neighborhood += 1

            with open("output/info/results_grasp_" + instanceName + "_" + str(args.argAlpha) + "_" + str(args.argBeta).replace(".", "_") + "_" + str(args.argChi).replace(".", "_") + "_" + str(args.argDelta).replace(".", "_") + "_" + str(args.argMaxIter) + "_" + str(args.argSeed) + ".txt", "a") as graspFile:
                graspFile.write(f"  {currentSolution.NPV:>17.2f} |")

            # Check whether a new incumbent solution has been found
            if currentSolution.NPV - bestSolution.NPV > 0.001:
                # Message for the user
                print("--> New incumbent solution found!\n")

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

        with open("output/info/results_grasp_" + instanceName + "_" + str(args.argAlpha) + "_" + str(args.argBeta).replace(".", "_") + "_" + str(args.argChi).replace(".", "_") + "_" + str(args.argDelta).replace(".", "_") + "_" + str(args.argMaxIter) + "_" + str(args.argSeed) + ".txt", "a") as graspFile:
            graspFile.write("\n+-------------------+-------------------+--------------------+")

        # Compute 'Runtime'
        metrics.Runtime = tm.perf_counter() - t1

        # Message for the user
        print("> Process finished!\n")

        # Message for the user
        print(f"--> NPV_best: {bestSolution.NPV:.2f}")

        # Write the best solution to file
        writeOutputAlgorithmFile(args.argInstanceName, bestSolution, metrics, periods, groups, intervals, args.argAlpha, args.argBeta, args.argChi, args.argDelta, args.argMaxIter, args.argSeed)