# import
from argumentParser import parseArguments
from parameters import setupParameters, setupIndexes
import math as m
import numpy as np

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstanceName)
P, periods, meter_groups, substitution_squads, intervals = setupIndexes(J, K, SP, DH, T)

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
class Metrics:
    def __init__(self):
        # Total runtime
        self.Runtime = 0.0

        # Iteration counter used in the main loop of the algorithm
        self.NumIters = 0

        # Number of improvements found by the algorithm
        self.NumImprovements = 0

        # Number of iterations to find the best solution
        self.NumItersBest = 0

        # Runtime to find the best solution
        self.RuntimeBest = 0.0

        # Current 'fix' method
        self.FixMethod = 0

        # 'Fix' method that found the best solution
        self.FixMethodBest = 0

        # Number of iterations in which 1st 'fix' method was used
        self.NumItersFirstMethod = 0

        # Cumulative runtime for 1st 'fix' method was used
        self.CumulativeRuntimeFirstMethod = 0.0

        # Number of improvements found by the 1st 'fix' method
        self.NumImprovementsFirstMethod = 0

        # Number of iterations in which 2nd 'fix' method was used
        self.NumItersSecondMethod = 0

        # Cumulative runtime for 2nd 'fix' method was used
        self.CumulativeRuntimeSecondMethod = 0.0

        # Number of improvements found by the 2nd 'fix' method
        self.NumImprovementsSecondMethod = 0

        # Number of iterations in which 3rd 'fix' method was used
        self.NumItersThirdMethod = 0

        # Cumulative runtime for 3rd 'fix' method was used
        self.CumulativeRuntimeThirdMethod = 0.0

        # Number of improvements found by the 3rd 'fix' method
        self.NumImprovementsThirdMethod = 0

        # Number of iterations in which 4th 'fix' method was used
        self.NumItersFourthMethod = 0

        # Cumulative runtime for 4th 'fix' method was used
        self.CumulativeRuntimeFourthMethod = 0.0

        # Number of improvements found by the 4th 'fix' method
        self.NumImprovementsFourthMethod = 0

class Solution:
    def __init__(self):
        # Integer variable 'x' corresponding to the number of smart meters installed in meter group 'j' during time interval 't'
        self.x = np.zeros((len(meter_groups), len(intervals)), dtype=int)

        # Binary variable 'y' taking value 1 if meter group 'j' is served time interval 't' and 0 otherwise
        self.y = np.zeros((len(meter_groups), len(intervals)), dtype=int)

        # Binary variable 'overline_y' taking value 1 if installations in meter group 'j' are completed during time interval 't' and 0 otherwise
        self.overline_y = np.zeros((len(meter_groups), len(intervals)), dtype=int)

        # Binary variable 'z' taking value 1 if meter group 'j' is already smart during time interval 't' and 0 otherwise
        self.z = np.zeros((len(meter_groups), len(intervals)), dtype=int)

        # Continuous variable 'S' corresponding to the conditional cost savings
        self.S = np.zeros(len(periods), dtype=float)

        # Continuous variable 'R' corresponding to the additional revenues defined by the Authority
        self.R = np.zeros(len(periods), dtype=float)

        # Continuous variable 'X' corresponding to the capital expenditures
        self.X = np.zeros(len(periods), dtype=float)

        # Continuous variable 'D' corresponding to the depreciation charges
        self.D = np.zeros(len(periods), dtype=float)

        # Cash flows
        self.F = np.zeros(len(periods), dtype=float)

        # Discounting factors
        self.d = np.array([1 / pow(1 + r, p) for p in periods])

        # Continuous variable 'NPV' corresponding to the Net Present Value
        self.NPV = -m.inf

    def updateFromSolution(self, other_solution):
        """ Updates the current solution with the values from another solution object. """
        if not isinstance(other_solution, Solution):
            raise TypeError("The provided object must be an instance of the Solution class.")

        # Copy all attributes from other_solution
        self.x = other_solution.x.copy()
        self.y = other_solution.y.copy()
        self.overline_y = other_solution.overline_y.copy()
        self.z = other_solution.z.copy()
        self.S = other_solution.S.copy()
        self.R = other_solution.R.copy()
        self.X = other_solution.X.copy()
        self.D = other_solution.D.copy()
        self.F = other_solution.F.copy()
        self.d = other_solution.d.copy()
        self.NPV = other_solution.NPV

    def copy(self):
        """ Returns a deep copy of the current solution. """
        new_solution = Solution()
        new_solution.updateFromSolution(self)
        return new_solution
