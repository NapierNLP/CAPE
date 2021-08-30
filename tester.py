import subprocess
import os
import sys

logdir = 'logs'
data_size = 10_000
batch_size = 32

# subprocess.call([sys.executable,
#                  'main.py',
#                  "--log_dir",
#                  logdir,
#                  "--data_split",
#                  str(data_size),
#                  "--batch_size",
#                  str(batch_size),
#                  "--dropout",
#                  "0.2",
#                  "--validate",
#                  "--tag",
#                  f"baseline"],
#                 env=os.environ.copy())
#
# subprocess.call([sys.executable,
#                  'main.py',
#                  "--log_dir",
#                  logdir,
#                  "--data_split",
#                  str(data_size),
#                  "--batch_size",
#                  str(batch_size),
#                  "--dropout",
#                  "0.2",
#                  "--dp",
#                  "--validate",
#                  "--tag",
#                  f"dp"],
#                 env=os.environ.copy())
#
# subprocess.call([sys.executable,
#                  'main.py',
#                  "--log_dir",
#                  logdir,
#                  "--data_split",
#                  str(data_size),
#                  "--batch_size",
#                  "64",
#                  "--dropout",
#                  "0.2",
#                  "--adversarial",
#                  # "--validate",
#                  "--tag",
#                  f"adv"],
#                 env=os.environ.copy())

subprocess.call([sys.executable,
                 'main.py',
                 "--log_dir",
                 logdir,
                 "--data_split",
                 str(data_size),
                 "--batch_size",
                 str(batch_size),
                 "--dropout",
                 "0.2",
                 "--adversarial",
                 "--dp",
                 "--validate",
                 "--tag",
                 f"cae"],
                env=os.environ.copy())
