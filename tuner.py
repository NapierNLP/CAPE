import subprocess
import os
import sys

logdir = 'dp_tuning'
data_size = 10_000
hidden_sizes = [128, 256, 512]
epsilons = [0.01, 0.1, 1.0, 10.0]
lambdas = [0.01, 0.1, 1.0, 10.0]
dropouts = [0.0, 0.2, 0.4]
learning_rate = [0.01, 0.001, 0.0001]

# for hidden_size in hidden_sizes:
#     for lr in learning_rate:
#         for dropout in dropouts:
#             subprocess.call([sys.executable,
#                              'main.py',
#                              "--log_dir",
#                              logdir,
#                              "--data_split",
#                              str(data_size),
#                              "--hidden_size",
#                              str(hidden_size),
#                              "--validate",
#                              "--dropout",
#                              str(dropout),
#                              "--tag",
#                              f"hidden_{hidden_size}_dropout_{dropout}_lr_{lr}"],
#                             env=os.environ.copy())


for epsilon in epsilons:
    subprocess.call([sys.executable,
                     'main.py',
                     "--log_dir",
                     logdir,
                     "--data_split",
                     str(data_size),
                     "--epsilon",
                     str(epsilon),
                     "--dp",
                     "--adversarial",
                     "--validate",
                     "--tag",
                     f"epsilon_{epsilon}"],
                    env=os.environ.copy())

for lbd in lambdas:
    subprocess.call([sys.executable,
                     'main.py',
                     "--log_dir",
                     logdir,
                     "--data_split",
                     str(data_size),
                     "--hplambda",
                     str(lbd),
                     "--dp",
                     "--adversarial",
                     "--validate",
                     "--tag",
                     f"lambda_{lbd}"],
                    env=os.environ.copy())
