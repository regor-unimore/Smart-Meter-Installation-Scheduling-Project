# import
from argumentParser import parseArguments
from parameters import setupParameters, setupIndexes

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstance)
P, periods, meter_groups, substitution_squads, intervals = setupIndexes(J, K, SP, DH, T)

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
class Solution:
    def __init__(self):
        # Integer variable 'x' corresponding to the number of smart meters installed in meter group 'j' by substitution squad 'k' during time interval 't' (in the best solution found)
        self.x = {(j, k, t): 0 for j in meter_groups for k in substitution_squads for t in intervals}

        # Binary variable 'y' taking value 1 if meter group 'j' is served by squad 'k' during time interval 't' and 0 otherwise
        self.y = {(j, k, t): 0 for j in meter_groups for k in substitution_squads for t in intervals}

        # Binary variable 'overline_y' taking value 1 if installations in meter group 'j' are completed during time interval 't' and 0 otherwise
        self.overline_y = {(j, t): 0 for j in meter_groups for t in intervals}

        # Binary variable 'z' taking value 1 if meter group 'j' is already smart during time interval 't' and 0 otherwise
        self.z = {(j, t): 0 for j in meter_groups for t in intervals}

        # Continuous variable 'NPV' corresponding to the Net Present Value
        self.NPV = 0.0

        # Continuous variable 'S' corresponding to the conditional cost savings
        self.S = {p: 0.0 for p in periods}

        # Continuous variable 'R' corresponding to the additional revenues defined by the Authority
        self.R = {p: 0.0 for p in periods}

        # Continuous variable 'X' corresponding to the capital expenditures
        self.X = {p: 0.0 for p in periods}

        # Continuous variable 'D' corresponding to the depreciation charges
        self.D = {p: 0.0 for p in periods}

        # Computed cash flows
        self.F = {p: 0.0 for p in periods}

        # Computed discounting factors
        self.d = {p: 1 / pow(1 + r, p) for p in periods}

    def updateSolution(self, updates):
        """
        Updates the solution with a dictionary of new values.
        The dictionary should contain keys matching the variable names (x, y, etc.)
        and the corresponding values to update.

        Example:
        updates = {
            'x': {(j, k, t): new_value, ...},
            'NPV': new_value,
            ...
        }
        """
        for key, value in updates.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, dict):
                    current_value.update(value)  # Update the dictionary
                else:
                    setattr(self, key, value)  # Set new value for non-dict variables
            else:
                raise AttributeError("{} is not a valid attribute of Solution".format(key))

    def updateFromSolution(self, other_solution):
        """Updates the current solution with the values from another solution object."""
        if not isinstance(other_solution, Solution):
            raise TypeError("The provided object must be an instance of the Solution class.")

        # Copy all attributes from other_solution
        self.x.update(other_solution.x)
        self.y.update(other_solution.y)
        self.overline_y.update(other_solution.overline_y)
        self.z.update(other_solution.z)
        self.NPV = other_solution.NPV
        self.S.update(other_solution.S)
        self.R.update(other_solution.R)
        self.X.update(other_solution.X)
        self.D.update(other_solution.D)
        self.F.update(other_solution.F)
        self.d.update(other_solution.d)