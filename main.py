# import
from utils import writeFile
from parameters import parseArguments, setupParameters, setupIndexes
from classes import Solution
from modelCreator import create_model
from semiGreedy import greedyRandomizedConstructiveHeuristics
from localSearch import large_neighborhood_search
import time as tm

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argFolder, args.argInstance)
P, periods, meter_groups, substitution_squads, intervals = setupIndexes(J, K, SP, DH, T)

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

# Create the model
model, x, y, overline_y, z, S, R, X, D = create_model()

# Run the algorithm

# Message for the user
print('\n> Running the algorithm...\n')

# Define a global object 'best_solution' of class 'Solution'
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
    gap = (best_solution.NPV - solution.NPV) / solution.NPV

    # Check whether the gap between the NPV of the current solution and the NPV of the incumbent solution is less than 5%. If so, try to improve it via local search
    if gap < 0.05:
        # Message for the user
        print('> Improving the solution via local search...\n')

        # Run the local search and update the current solution
        solution.update_from_solution(large_neighborhood_search(solution, model, x, y, overline_y, z, S, R, X, D))

        # Message for the user
        print("  Beta: {:.2f}, NPV_curr: {:.2f}\n".format(args.argBeta, solution.NPV))

        # Message for the user
        # print("> Local search completed!\n")

    # Check whether a new incumbent solution has been found
    if solution.NPV > best_solution.NPV:
        # Message for the user
        print('  New incumbent solution found!\n')

        # Update 'improvement'
        improvements += 1

        # Update the best solution
        best_solution.update_from_solution(solution)

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
writeFile(args.argInstance, best_solution, computational_tm, periods, meter_groups, intervals, substitution_squads)