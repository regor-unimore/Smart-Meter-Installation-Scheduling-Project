# import
import utils

# Cache to avoid multiple computations
_cached_params = None
_cached_indexes = None

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