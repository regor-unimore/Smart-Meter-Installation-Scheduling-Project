# import
from argumentParser import parseArguments
from parameters import setupParameters, setupIndexes

# Parse arguments
args = parseArguments()

# Setup parameters
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstanceName)

# Setup indexes
P, periods, groups, teams, intervals = setupIndexes(J, K, SP, DH, T)