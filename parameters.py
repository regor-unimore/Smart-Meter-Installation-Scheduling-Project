# import
import argparse
import utils

# Cache to avoid multiple computations
_cached_args = None
_cached_params = None
_cached_indexes = None

# ----------------------------------------------------------------------------------------------------------------------
# Command line arguments
# ----------------------------------------------------------------------------------------------------------------------
def parseArguments():
    """
    Parses command line arguments and caches the results.

    Returns:
        Namespace: Parsed arguments as an argparse.Namespace object.
    """
    global _cached_args

    # If command line arguments are parsed for the first time
    if _cached_args is None:
        # Message for the user
        print("\n> Parsing command line arguments...\n")

        # Define 'parser'
        parser = argparse.ArgumentParser()

        # Folder where the instance is located
        parser.add_argument('-f', '--folder', action="store", type=str, required=True, help="Folder where the instance is located", dest="argFolder")

        # Name of the instance
        parser.add_argument('-i', '--instance', action="store", type=str, required=True, help="Name of the instance", dest="argInstance")

        # Cardinality of the restricted candidate list in the greedy randomized constructive heuristics
        parser.add_argument('-a', '--alpha', action="store", type=int, required=True, help="Cardinality of the restricted candidate list in the greedy randomized constructive heuristics", dest="argAlpha")

        # Parameter that guides the 'fix' method within the local search
        parser.add_argument('-b', '--beta', action="store", type=float, required=True, help="Parameter that guides the \'fix\' method within the local search", dest="argBeta")

        # Maximum number of iterations in the main loop of the algorithm
        parser.add_argument('-m', '--maxIter', action="store", type=int, required=True, help="Maximum number of iterations in the main loop of the algorithm", dest="argMaxIter")

        # Tolerance value for the solver
        parser.add_argument('-mg', '--mipGap', action="store", type=float, required=True, help="Tolerance value for the solver", dest="argMipGap")

        # Time limit for the solver (in seconds)
        parser.add_argument('-tl', '--timeLimit', action="store", type=float, required=True, help="Time limit for the solver (in seconds)", dest="argTimeLimit")

        # Define '_cached_args'
        _cached_args = parser.parse_args()

    return _cached_args

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
def setupParameters(folder, instance):
    """
    Reads parameters and structures from an input file and caches the results.

    Args:
        folder (str): The folder where the instance is located.
        instance (str): The name of the instance.

    Returns:
        tuple: All parameters and structures.
    """
    global _cached_params

    # If parameters and structures are read for the first time
    if _cached_params is None:
        # Message for the user
        print("> Reading input parameters and structures...\n")

        # Define '_cached_params'
        _cached_params = utils.readFile('input/' + folder + '/' + instance)

    return _cached_params

# ----------------------------------------------------------------------------------------------------------------------
# Indexes
# ----------------------------------------------------------------------------------------------------------------------
def setupIndexes(J, K, SP, DH, T):
    """
    Sets up the list of indexes for periods, meter groups, substitution squads, and time intervals.

    Args:
        J (int): Number of meter groups.
        K (int): Number of substitution squads.
        SP (int): Starting period.
        DH (int): Duration of horizon.
        T (int): Number of time intervals.

    Returns:
        tuple: All lists of periods, meter groups, substitution squads, and intervals
    """
    global _cached_indexes

    # If the lists of indexes are set up for the first time
    if _cached_indexes is None:
        # Message for the user
        print("> Estimating the number of periods and creating the lists of indexes...\n")

        # General formula for estimating the number of periods (i.e., years) 'P'
        P = SP + (DH + 1)

        # List of indexes for the periods
        periods = [p for p in range(P + 1)]

        # List of indexes for the meter groups
        meter_groups = [j for j in range(J)]

        # List of indexes for the substitution squads
        substitution_squads = [k for k in range(K)]

        # List of indexes for the time intervals within the operational horizon
        intervals = [t for t in range(SP * T)]

        # Define '_cached_indexes'
        _cached_indexes = (P, periods, meter_groups, substitution_squads, intervals)

    return _cached_indexes