# import
from utils import readInputFile

# Cache to avoid multiple computations
_cached_params = None
_cached_indexes = None

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
def setupParameters(pathToFolder, instanceName):
    """
    Reads parameters and structures from an input file and caches the results.

    Args:
        pathToFolder (str): The folder where the instance is located.
        instanceName (str): The name of the instance.

    Returns:
        tuple: All parameters and structures.
    """
    global _cached_params

    # If parameters and structures are read for the first time
    if _cached_params is None:
        # Message for the user
        print("> Reading input parameters and structures...\n")

        # Define '_cached_params'
        _cached_params = readInputFile(pathToFolder + '/' + instanceName)

    return _cached_params

# ----------------------------------------------------------------------------------------------------------------------
# Indexes
# ----------------------------------------------------------------------------------------------------------------------
def setupIndexes(J, K, SP, DH, T):
    """
    Sets up the list of indexes for periods, groups, teams, and time intervals.

    Args:
        J (int): Number of meter groups.
        K (int): Number of teams of technicians.
        SP (int): Operational horizon (i.e., number of years).
        DH (int): Depreciation horizon (i.e., number of years).
        T (int): Number of time intervals.

    Returns:
        tuple: All lists of periods, groups, teams, and time intervals
    """
    global _cached_indexes

    # If the lists of indexes are set up for the first time
    if _cached_indexes is None:
        # Message for the user
        print("> Estimating the number of periods and creating the lists of indexes...\n")

        # General formula for estimating the index of the last year of the project
        P = SP + (DH + 1)

        # List of indexes for periods
        periods = [p for p in range(P + 1)]

        # List of indexes for groups
        groups = [j for j in range(J)]

        # List of indexes for teams
        teams = [k for k in range(K)]

        # List of indexes for the time intervals within the operational horizon
        intervals = [t for t in range(SP * T)]

        # Define '_cached_indexes'
        _cached_indexes = (P, periods, groups, teams, intervals)

    return _cached_indexes