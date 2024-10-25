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
        parser.add_argument('--path', action="store", type=str, required=True, help="Path to folder where the instances are located", dest="argPath")

        # Name of the instance
        parser.add_argument('--instance', action="store", type=str, required=True, help="Name of the instance", dest="argInstance")

        # App (i.e., 'model' or 'algorithm')
        parser.add_argument('--solutionMethod', action="store", type=str, required=True, help="App (i.e., 'model' or 'algorithm')", dest="argSolutionMethod")

        # Sorting rule used within the greedy randomized constructive heuristics
        parser.add_argument('--sortingRule', action="store", type=str, required=False, help="Sorting rule used within the greedy randomized constructive heuristics (i.e., \'LongestRemainingProcessingTime\', or \'ShortestRemainingProcessingTime\'", dest="argSortingRule")

        # Cardinality of the restricted candidate list in the greedy randomized constructive heuristics
        parser.add_argument('--alpha', action="store", type=int, required=False, help="Cardinality of the restricted candidate list in the greedy randomized constructive heuristics", dest="argAlpha")

        # Parameter that guides the 1st 'fix' method within the local search
        parser.add_argument('--beta', action="store", type=float, required=False, help="Parameter that guides the 1st \'fix\' method within the local search", dest="argBeta")

        # Parameter that guides the 2nd, 3rd, and 4th 'fix' methods within the local search
        parser.add_argument('--gamma', action="store", type=float, required=False, help="Parameter that guides the 2nd, 3rd, and 4th \'fix\' method within the local search", dest="argGamma")

        # Maximum number of iterations in the main loop of the algorithm
        parser.add_argument('--maxIter', action="store", type=int, required=False, help="Maximum number of iterations in the main loop of the algorithm", dest="argMaxIter")

        # Tolerance value for the solver
        parser.add_argument('--mipGap', action="store", type=float, required=True, help="Tolerance value for the solver", dest="argMipGap")

        # Time limit for the solver (in seconds)
        parser.add_argument('--timeLimit', action="store", type=float, required=True, help="Time limit for the solver (in seconds)", dest="argTimeLimit")

        # Seed to initialize the random number generator
        parser.add_argument('--seed', action="store", type=int, required=False, help="Seed to initialize the random number generator", dest="argSeed")

        # Define '_cached_args'
        _cached_args = parser.parse_args()

    return _cached_args