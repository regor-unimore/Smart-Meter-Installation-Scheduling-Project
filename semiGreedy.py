# import
from argumentParser import parseArguments
from classes import Solution
from parameters import setupParameters, setupIndexes
import math as m
import numpy as np
import random as rnd

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstanceName)
P, periods, meter_groups, substitution_squads, intervals = setupIndexes(J, K, SP, DH, T)

# Initialize the random number generator
rnd.seed(args.argSeed)

# ----------------------------------------------------------------------------------------------------------------------
# Function(s) implementing the semi-greedy heuristics used to build an initial feasible solution
# ----------------------------------------------------------------------------------------------------------------------
def srptSort(interval, group_remaining_processing_time):
    # Define 'outlook' (i.e., a supporting variable that indicates the number of periods ahead to consider for the next reading -- HARDCODED
    OUTLOOK = 24

    # Initialise 'group_order'
    group_order = -1 * np.ones(len(meter_groups), dtype=int)

    # Initialise 'group_ordered'
    group_ordered = np.zeros(len(meter_groups), dtype=int)

    # First ordering loop based on the shortest remaining processing time
    for j in meter_groups:
        group_min_processing = max(group_remaining_processing_time) + 1
        group_min_index = -1
        for l in meter_groups:
            if group_ordered[l] == 0 and group_remaining_processing_time[l] < group_min_processing:
                group_min_processing = group_remaining_processing_time[l]
                group_min_index = l
        group_order[j] = group_min_index
        group_ordered[group_min_index] = 1

    # Initialise 'group_next_reading'
    group_next_reading = -1 * np.ones(len(meter_groups), dtype=int)

    # Initialize 'next reading' based on the predefined outlook
    for j in meter_groups:
        for t in range(interval, min(interval + OUTLOOK, SP * T)):
            if b[group_order[j]][t] == 0 and group_next_reading[group_order[j]] < 0:
                group_next_reading[group_order[j]] = t

    # Second ordering loop based on the farthest 'group next reading'
    for j in meter_groups:
        # Initialise the support variable and list
        group_index = j
        # Look for a meter group having the same remaining processing time but farther reading than 'j'
        for l in range(j + 1, len(meter_groups)):
            if group_remaining_processing_time[group_order[l]] == group_remaining_processing_time[group_order[group_index]] and group_next_reading[group_order[l]] > group_next_reading[group_order[group_index]]:
                group_index = l
        # Swap the elements (i.e., meter groups) if and only if 'j != group_index'
        group_order[j], group_order[group_index] = group_order[group_index], group_order[j]
    return group_order

def lrptSort(interval, group_remaining_processing_time):
    # Define 'outlook' (i.e., a supporting variable that indicates the number of periods ahead to consider for the next reading -- HARDCODED
    OUTLOOK = 24

    # Initialise 'group_order'
    group_order = -1 * np.ones(len(meter_groups), dtype=int)

    # Initialise 'group_ordered'
    group_ordered = np.zeros(len(meter_groups), dtype=int)

    # First ordering loop based on the longest remaining processing time
    for j in meter_groups:
        group_max_processing = -1
        group_max_index = -1
        for l in meter_groups:
            if group_ordered[l] == 0 and group_remaining_processing_time[l] > group_max_processing:
                group_max_processing = group_remaining_processing_time[l]
                group_max_index = l
        group_order[j] = group_max_index
        group_ordered[group_max_index] = 1

    # Initialise 'group_next_reading'
    group_next_reading = -1 * np.ones(len(meter_groups), dtype=int)

    # Initialize 'next reading' based on the predefined outlook
    for j in meter_groups:
        for t in range(interval, min(interval + OUTLOOK, SP * T)):
            if b[group_order[j]][t] == 0 and group_next_reading[group_order[j]] < 0:
                group_next_reading[group_order[j]] = t

    # Second ordering loop based on the farthest 'group next reading'
    for j in meter_groups:
        # Initialise the support variable and list
        group_index = j
        # Look for a meter group having the same remaining processing time but farther reading than 'j'
        for l in range(j + 1, J):
            if group_remaining_processing_time[group_order[l]] == group_remaining_processing_time[group_order[group_index]] and group_next_reading[group_order[l]] > group_next_reading[group_order[group_index]]:
                group_index = l
        # Swap the elements (i.e., meter groups) if and only if 'j != group_index'
        group_order[j], group_order[group_index] = group_order[group_index], group_order[j]
    return group_order

def greedyRandomizedConstructiveHeuristics():
    # Define a local object of 'Solution' class
    solution = Solution()

    # Estimated (rounded up) processing time for each meter group
    group_processing_time = np.array([m.ceil(N[j] / Q) for j in meter_groups])

    # Create a copy of 'group_processing_time'
    group_remaining_processing_time = group_processing_time.copy()

    # Create a copy of 'N'
    group_remaining_meters = N.copy()

    # Initialise 'completed'
    group_completed = np.zeros(len(meter_groups), dtype=int)

    # Initialise 'completion time'
    group_completion_time = -1 * np.ones(len(meter_groups), dtype=int)

    # Main loop
    interval = 0

    # While there are remaining intervals in the operational horizon
    while interval < len(intervals):
        # Condition to select 'srpt_sort' or 'lrpt_sort'
        if args.argSortingRule == "LongestRemainingProcessingTime":
            # Order or re-order meter groups based on the longest remaining processing time
            group_order = lrptSort(interval, group_remaining_processing_time)
        else:
            # Order or re-order meter groups based on the shortest remaining processing time (DEFAULT)
            group_order = srptSort(interval, group_remaining_processing_time)

        # Create a candidate list of all meter groups for which an activity can be scheduled

        # Initialise 'group_candidate_list'
        group_candidate_list = -1 * np.ones(len(meter_groups), dtype=int)

        # Initialise 'group_selected'
        group_selected = -1 * np.ones(len(meter_groups), dtype=int)

        # Append all the indexes of the meter groups that can be assigned to substitution squads to 'group_candidate_list' in ascending order based on the shortest remaining processing time, breaking ties by farthest next reading
        for j in meter_groups:
            if group_completed[group_order[j]] < 1 and b[group_order[j]][interval] > 0 and group_remaining_processing_time[group_order[j]] > 0:
                group_candidate_list[j] = group_order[j]

        # Initialise 'assignment_squad'
        squad_assignment = np.zeros(len(substitution_squads), dtype=int)

        # Assignment of meter groups (randomly chosen from the restricted candidate list)
        for k in substitution_squads:
            # Create a restricted candidate list of meter groups that can be assigned (i.e., if 'ALPHA' = 1, we are computing the constructive greedy solution)
            group_restricted_candidate_list = []

            # Scroll the list until you reach the given cardinality
            group_index = 0

            while len(group_restricted_candidate_list) < args.argAlpha and group_index < len(meter_groups):
                # If meter group 'group_index' in 'group_candidate_list' cannot be assigned in the current interval
                if group_candidate_list[group_index] == -1:
                    # Update the index
                    group_index += 1
                else:
                    # If meter group 'group_index' in 'group_candidate_list' can be assigned in the current interval and has not been assigned yet
                    if group_selected[group_candidate_list[group_index]] == -1:
                        # Append the element to the restricted candidate list of meter groups
                        group_restricted_candidate_list.append(group_candidate_list[group_index])

                        # Update the index
                        group_index += 1
                    else:
                        # Update the index
                        group_index += 1

            # If 'group_restricted_candidate_list' is not empty and substitution squad 'k' is idle
            if group_restricted_candidate_list and squad_assignment[k] == 0:
                # Randomly choose a meter group from the 'restricted_candidate_list'
                group_random = rnd.choice(group_restricted_candidate_list)

                # Select meter group 'j' in 'selected list'
                group_selected[group_random] = 1

                # Update the remaining processing time
                group_remaining_processing_time[group_random] -= 1

                # Set the value for the heuristic variable
                # solution.y[group_random, k, interval] = 1
                solution.y[group_random, interval] = 1

                # If the remaining number of meters is greater than or equal to the substitution capacity
                if group_remaining_meters[group_random] - Q >= 0:
                    # Update 'x'
                    # solution.x[group_random, k, interval] = Q
                    solution.x[group_random, interval] = Q

                    # Update 'group_remaining_meters'
                    group_remaining_meters[group_random] -= Q
                # If the remaining number of meters is less than to the substitution capacity
                else:
                    # Update 'x'
                    # solution.x[group_random, k, interval] = group_remaining_meters[group_random]
                    solution.x[group_random, interval] = group_remaining_meters[group_random]

                    # Update 'group_remaining_meters'
                    group_remaining_meters[group_random] -= group_remaining_meters[group_random]

                # Substitution squad 'k' has been used
                squad_assignment[k] = 1

        # Check if the selected meter groups have been completed and retrieve the completion time
        for j in meter_groups:
            # If meter group 'j' has been selected in the current 'interval' and its remaining processing time is equal to zero
            if group_selected[j] != -1 and group_remaining_processing_time[j] == 0:
                # Meter group 'j' has been completed
                group_completed[j] = 1

                # Retrieve its completion time
                group_completion_time[j] = interval

        # Update the interval
        interval += 1

    # Set the value for 'overline_y'
    for j in meter_groups:
        t = group_completion_time[j]
        solution.overline_y[j, t] = 1

    # Set the value for 'z'
    for j in meter_groups:
        completion_time = group_completion_time[j]  # Get the completion time for group 'j'
        for t in (t for t in intervals if t > completion_time):  # Loop only for 't > group_completion_time[j]'
            if t > group_completion_time[j]:
                solution.z[j, t] = 1

    # Update 'S(p)' within the operational horizon
    for p in range(SP):
        start_interval = T * p
        end_interval = T * (p + 1)
        solution.S[p] = sum(S1[j][t] * solution.z[j, t] for j in meter_groups for t in range(start_interval, end_interval))

    # Update 'S(p)' without the operational horizon
    for p in range(SP, P - 1):
        solution.S[p] = sum(S2[j] for j in meter_groups)

    # Update 'X(p)' within the operational horizon
    for p in range(SP):
        start_interval = T * p
        end_interval = T * (p + 1)
        solution.X[p] = sum(C * solution.x[j, t] for j in meter_groups for t in range(start_interval, end_interval))

    # Update 'D(p)'
    for p in range(1, P + 1):
        start_varphi = max(0, p - DH)
        end_varphi = (p - 1) + 1
        solution.D[p] = (1 / DH) * sum(solution.X[varphi] for varphi in range(start_varphi, end_varphi))

    # Update 'R(2)'
    solution.R[2] = r * solution.X[0]

    # Update 'R(p)'
    for p in range(3, P + 1):
        start_varphi = p - 2
        solution.R[p] = solution.D[p - 2] + r * sum(solution.X[varphi] - solution.D[varphi] for varphi in range(start_varphi + 1))

    # Update 'F(p)'
    for p in periods:
        solution.F[p] = (1 - gamma) * (solution.S[p] + solution.R[p]) - solution.X[p] + gamma * solution.D[p]

    # Update 'NPV'
    solution.NPV = sum(solution.d[p] * solution.F[p] for p in periods)

    return solution