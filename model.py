# import
from argumentParser import parseArguments
from parameters import setupParameters, setupIndexes
import gurobipy as gp
from gurobipy import GRB

# Access the parsed arguments, parameters, and indexes
args = parseArguments()
J, K, T, N, Q, b, sigma, S1, S2, SP, DH, r, gamma, C, instanceName = setupParameters(args.argPath, args.argInstanceName)
P, periods, meter_groups, substitution_squads, intervals = setupIndexes(J, K, SP, DH, T)

# ----------------------------------------------------------------------------------------------------------------------
# Function(s) for the MILP model
# ----------------------------------------------------------------------------------------------------------------------
def createModel():
    # Message for the user
    print("> Creating the model...\n")

    # Create explicit environment
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1) # Control logging explicitly
    env.start()

    # Define 'modelName'
    modelName = "SMISP_" + instanceName

    # Create the model
    model = gp.Model(modelName, env=env)

    # Default 'Seed' value of the solver for reproducibility control
    model.setParam("Seed", 0)

    # Tolerance value for the solver
    model.setParam("MIPGap", args.argMipGap)

    # Time limit for the solver (in seconds)
    # model.setParam("TimeLimit", args.argTimeLimit)

    # Work limit for the solver
    model.setParam("WorkLimit", args.argTimeLimit)

    # Thread count for the solver
    model.setParam("Threads", args.argThreads)

    """ Create the operational variables """

    # Integer variable corresponding to the number of smart meters installed in meter group 'j' by substitution squad 'k' during time interval 't'
    x = model.addVars(meter_groups, intervals, lb=0.0, ub=Q, obj=0.0, vtype=GRB.INTEGER, name="x")

    # Binary variable taking value 1 if meter group 'j' is served by squad 'k' during time interval 't' and 0 otherwise
    y = model.addVars(meter_groups, intervals, lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="y")

    # Binary variable taking value 1 if installations in meter group 'j' are completed during time interval 't' and 0 otherwise
    overline_y = model.addVars(meter_groups, intervals, lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="overline_y")

    # Binary variable taking value 1 if meter group 'j' is already smart during time interval 't' and 0 otherwise
    z = model.addVars(meter_groups, intervals, lb=0.0, ub=1.0, obj=0.0, vtype=GRB.BINARY, name="z")

    """ Create the supporting variables """

    # Conditional cost savings
    S = model.addVars(periods, lb=0.0, ub=float('inf'), obj=0.0, vtype=GRB.CONTINUOUS, name="S")

    # Additional revenues defined by the Authority
    R = model.addVars(periods, lb=0.0, ub=float('inf'), obj=0.0, vtype=GRB.CONTINUOUS, name="R")

    # Capital expenditures
    X = model.addVars(periods, lb=0.0, ub=float('inf'), obj=0.0, vtype=GRB.CONTINUOUS, name="X")

    # Depreciation charges
    D = model.addVars(periods, lb=0.0, ub=float('inf'), obj=0.0, vtype=GRB.CONTINUOUS, name="D")

    """ Add the objective function to the model """

    # Objective function (1)
    model.setObjective(gp.quicksum((1 / pow(1 + r, p)) * ((1 - gamma) * (S[p] + R[p]) - X[p] + gamma * D[p]) for p in periods), GRB.MAXIMIZE)

    """ Add the constraints to the model """

    # Constraints (2) impose that all the substitutions must be completed within 'SP' years
    # AGGREGATE SUM: FULL
    # model.addConstrs((gp.quicksum(overline_y[j, t] for t in intervals) == 1 for j in meter_groups), name="C2")
    # AGGREGATE SUM: SIMPLE
    model.addConstrs((overline_y.sum(j, "*") == 1 for j in meter_groups), name="C2")

    # Constraints (3) define the condition for which a meter group is considered smart (activation condition)
    model.addConstrs((z[j, t] <= gp.quicksum(overline_y[j, tau] for tau in range(t)) for j in meter_groups for t in intervals), name="C3")

    # Constraints (4) express the condition for completing the installations for each meter group 'j'
    model.addConstrs((N[j] * overline_y[j, t] <= gp.quicksum(x[j, tau] for tau in range(t + 1)) for j in meter_groups for t in intervals), name="C4")

    # Constraints (5) limit the capacity per time interval
    # AGGREGATE SUM: FULL
    # model.addConstrs((gp.quicksum(x[j, t] for j in meter_groups) <= K * Q for t in intervals), name="C5")
    # AGGREGATE SUM: SIMPLE
    model.addConstrs((x.sum("*", t) <= K * Q for t in intervals), name="C5")

    # Constraints (6) impose that 'K * sigma' meter groups can be served for each time interval 't' (by all substitution squads)
    # AGGREGATE SUM: FULL
    # model.addConstrs((gp.quicksum(y[j, t] for j in meter_groups) <= K * sigma for t in intervals), name="C6")
    # AGGREGATE SUM: SIMPLE
    model.addConstrs((y.sum("*", t) <= K * sigma for t in intervals), name="C6")

    # Constraints (7) establish the connection between variables 'x' and 'y'
    model.addConstrs((x[j, t] <= min(N[j], Q) * y[j, t] for j in meter_groups for t in intervals), name="C7")

    # Constraints (8) impose that substitutions can occur in meter group 'j' during time interval 't' only if readings are not performed
    # AGGREGATE SUM: FULL
    # model.addConstrs((y[j, t] <= b[j][t] * (1 - gp.quicksum(overline_y[j, tau] for tau in range(t))) for j in meter_groups for t in intervals), name="C8")
    # AGGREGATE SUM: SIMPLE
    model.addConstrs((y[j, t] <= b[j][t] * (1 - gp.quicksum(overline_y[j, tau] for tau in range(t))) for j in meter_groups for t in intervals), name="C8")

    # Constraints (8): original form (reinforced by valid inequality (1))
    # model.addConstrs((y[j, t] <= b[j][t] for j in meter_groups for t in intervals), name="C8")

    # Constraints (9)
    model.addConstrs((S[p] == gp.quicksum(S1[j][t] * z[j, t] for j in meter_groups for t in range(T * p, T * (p + 1))) for p in range(SP)), name="C9")

    # Constraints (10)
    model.addConstrs((S[p] == gp.quicksum(S2[j] for j in meter_groups) for p in range(SP, (P - 2) + 1)), name="C10")

    # Constraints (11)
    model.addConstrs((S[p] == 0 for p in [P - 1, P]), name="C11")

    # Constraints (12)
    model.addConstrs((X[p] == C * gp.quicksum(x[j, t] for j in meter_groups for t in range(T * p, T * (p + 1))) for p in range(SP)), name="C12")

    # Constraints (13)
    model.addConstrs((X[p] == 0 for p in range(SP, P + 1)), name="C13")

    # Constraints (14)
    model.addConstr(D[0] == 0, name="C14")

    # Constraints (15)
    model.addConstrs((D[p] == (1 / DH) * gp.quicksum(X[varphi] for varphi in range(max(0, p - DH), (p - 1) + 1)) for p in range(1, P + 1)), name="C15")

    # Constraints (16)
    model.addConstrs((R[p] == 0 for p in [0, 1]), name="C16")

    # Constraints (17)
    model.addConstr(R[2] == r * X[0], name="C17")

    # Constraints (18)
    model.addConstrs((R[p] == D[p - 2] + r * gp.quicksum((X[varphi] - D[varphi]) for varphi in range((p - 2) + 1)) for p in range(3, P + 1)), name="C18")

    # Update model to ensure fingerprint is available
    model.update()

    # Get and log fingerprint for verification
    fingerprint = model.getAttr("Fingerprint")
    print(f"> Model fingerprint: 0x{fingerprint:08x}")

    return model, env, x, y, overline_y, z, S, R, X, D

def modelStatus(model):
    if model.Status == GRB.LOADED:
        return 1
    elif model.Status == GRB.OPTIMAL:
        return 2
    elif model.Status == GRB.INFEASIBLE:
        return 3
    elif model.Status == GRB.INF_OR_UNBD:
        return 4
    elif model.Status == GRB.UNBOUNDED:
        return 5
    elif model.Status == GRB.CUTOFF:
        return 6
    elif model.Status == GRB.ITERATION_LIMIT:
        return 7
    elif model.Status == GRB.NODE_LIMIT:
        return 8
    elif model.Status == GRB.TIME_LIMIT:
        return 9
    elif model.Status == GRB.SOLUTION_LIMIT:
        return 10
    elif model.Status == GRB.INTERRUPTED:
        return 11
    elif model.Status == GRB.NUMERIC:
        return 12
    elif model.Status == GRB.SUBOPTIMAL:
        return 13
    elif model.Status == GRB.INPROGRESS:
        return 14
    elif model.Status == GRB.USER_OBJ_LIMIT:
        return 15
    elif model.Status == GRB.WORK_LIMIT:
        return 16
    elif model.Status == GRB.MEM_LIMIT:
        return 17

def solutionCallback(model, where, cbFilename):
    if where == GRB.Callback.MIPSOL:
        # MIP solution callback
        nodeCount = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        incumbent = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        solutionCount = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)

        with open('callbacks/results_bc_' + cbFilename + '.txt', 'a') as cbFile:
            cbFile.write(f"\n| {nodeCount:>15.0f} | {incumbent:>15.2f} | {runtime:>15.2f} | {solutionCount:>15.0f} |")