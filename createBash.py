import itertools

# Define your parameters and possible values
params = {
    "--path": ["instanceGenerator/instances"],
    # "--instance": ["instance_5_4_2021_3", "instance_5_4_2023_3"],
    # "--instance": ["instance_10_6_2022_2", "instance_10_6_2024_5"],
    "--instance": ["instance_20_8_2021_1", "instance_20_8_2021_4"],
    "--solutionMethod": ["grasp"],
    "--sortingRule": ["ShortestRemainingProcessingTime"],
    # "--alpha": [1, 2, 3],
    # "--alpha": [2, 4, 6],
    "--alpha": [4, 8, 12],
    "--beta": [0.00],
    "--chi": [0.80, 0.85, 0.90, 0.95],
    "--delta": [0.70, 0.75, 0.80, 0.85],
    "--epsilon": [0.05, 0.10, 0.15, 0.20],
    "--maxIter": [50],
    "--mipGap": [0.000001],
    "--timeLimit": [5],
    "--threads": [1],
    "--seed": [123, 321],
}

# Get parameter names and their corresponding lists
keys = list(params.keys())
values = list(params.values())

# Generate all combinations
combinations = list(itertools.product(*values))

# Write to file
with open("test4.sh", "w") as f:
    # f.write("# --instance = {instance_5_4_2021_4, instance_5_4_2023_3} | --solutionMethod = grasp | --alpha = {1, 2, 3} | --beta = {0.00} | --chi = {0.80, 0.85, 0.90, 0.95} | --delta = {0.70, 0.75, 0.80, 0.85} | --epsilon = {0.05, 0.10, 0.15, 0.20} | --maxIter = 50 | --mipGap = 0.000001 | --timeLimit = 5 | --threads = 1 | --seed = {123, 321}\n")
    # f.write("# --instance = {instance_10_6_2022_2, instance_10_6_2024_5} | --solutionMethod = grasp | --alpha = {2, 4, 6} | --beta = {0.00} | --chi = {0.80, 0.85, 0.90, 0.95} | --delta = {0.70, 0.75, 0.80, 0.85} | --epsilon = {0.05, 0.10, 0.15, 0.20} | --maxIter = 50 | --mipGap = 0.000001 | --timeLimit = 5 | --threads = 1 | --seed = {123, 321}\n")
    f.write("# --instance = {instance_20_8_2021_1, instance_20_8_2021_4} | --solutionMethod = grasp | --alpha = {4, 8, 12} | --beta = {0.00} | --chi = {0.80, 0.85, 0.90, 0.95} | --delta = {0.70, 0.75, 0.80, 0.85} | --epsilon = {0.05, 0.10, 0.15, 0.20} | --maxIter = 50 | --mipGap = 0.000001 | --timeLimit = 5 | --threads = 1 | --seed = {123, 321}\n")
    for combo in combinations:
        command = "python main.py " + " ".join(f"{k} {v}" for k, v in zip(keys, combo))
        f.write(command + "\n")

print(f"{len(combinations)} command combinations written to \'test4.sh\'")