# import
import argparse

# Cache to avoid multiple computations
_cached_args = None

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