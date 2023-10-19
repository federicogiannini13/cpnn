# IMPORT
import os
from models.cpnn import *
from models.cgru import cGRULinear
from models.clstm import *
import numpy as np
import pandas as pd
import pickle
import argparse

# EDITABLE PARAMETERS
dataset = "sine_rw_10_1234"
iterations = 10

# OTHER PARAMETERS
batch_size = 128
hidden_size = 50
seq_len = 10
suffix = ""


if suffix != "" and suffix[0:1] != "_":
    suffix = "_" + suffix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="cpnn",
    help="Model to use: {'cpnn', 'single': cLSTM, 'multiple': mcLSTM}",
)
parser.add_argument(
    "--model_class",
    type=str,
    default="clstm",
    help="Base learner to use: {'clstm', 'cgru'}",
)
args = parser.parse_args()
if args.model_class == "clstm":
    model_class = cLSTMLinear
else:
    model_class = cGRULinear
if hidden_size is None:
    hidden_size = 50
device = torch.device("cpu")
df = pd.read_csv(os.path.join("datasets", f"{dataset}.csv"))
perf_test = {"accuracy": [], "kappa": [], "kappa_temporal": [], "loss": []}
perf_train = {"accuracy": [], "kappa": [], "kappa_temporal": [], "loss": []}
seq_str = ""

path = os.path.join(
    "performance",
    f"{dataset}/{args.model}_{args.model_class}{seq_str}{suffix}_{hidden_size}hs",
)
if not os.path.isdir(path):
    os.makedirs(path)


# UTILS
def create_cpnn():
    return cPNN(
        column_class=model_class,
        device=device,
        seq_len=seq_len,
        train_verbose=False,
        input_size=len(df.columns) - 2,
        hidden_size=hidden_size,
        batch_size=batch_size,
    )


# MAIN
if __name__ == "__main__":
    models = []
    print(dataset)
    for i in range(1, iterations + 1):
        models.append([])
        for k in perf_test:
            perf_test[k].append([])
        for k in perf_train:
            perf_train[k].append([])
        models[-1].append(create_cpnn())
        print(type(models[-1][-1].columns.columns[0]).__name__)
        print(f"{i}/{iterations} iteration of {args.model}")
        for task in range(1, df["task"].max() + 1):
            print("TASK:", task)
            if task > 1:
                if args.model == "cpnn":
                    models[-1][-1].add_new_column()
                elif args.model == "multiple":
                    models[-1].append(create_cpnn())
                elif args.model == "single":
                    models[-1].append(pickle.loads(pickle.dumps(models[-1][-1])))
            df_task = df[df["task"] == task]
            df_task = df_task.drop(columns="task")

            for k in perf_test:
                perf_test[k][-1].append([])
            for k in perf_train:
                perf_train[k][-1].append([])
            if len(df_task) % batch_size == 0:
                n_batches = int(len(df_task) / batch_size)
            else:
                n_batches = int(len(df_task) / batch_size) + 1
            for i in range(0, len(df_task), batch_size):
                x = df_task.iloc[i : i + batch_size, 0:-1].values.astype(np.float32)
                y = list(df_task.iloc[i : i + batch_size, -1])
                print(int(i / batch_size) + 1, "/", n_batches, " batch", end="\r")
                if len(y) >= seq_len:
                    batch_perf_test, batch_perf_train = models[-1][
                        -1
                    ].test_then_train(x, y)
                    for k in batch_perf_test:
                        perf_test[k][-1][-1].append(batch_perf_test[k])
                    for k in batch_perf_train:
                        perf_train[k][-1][-1].append(batch_perf_train[k])
            print()
            print(
                f"Average accuracy on TASK {task}: {np.mean(perf_test['accuracy'][-1][-1])}"
            )
            print()

            with open(
                os.path.join(path, "test_then_train.pkl"),
                "wb",
            ) as f:
                pickle.dump(perf_test, f)

            with open(
                os.path.join(path, "train.pkl"),
                "wb",
            ) as f:
                pickle.dump(perf_train, f)
            with open(os.path.join(path, "models.pkl"), "wb") as f:
                pickle.dump(models, f)
        print()
