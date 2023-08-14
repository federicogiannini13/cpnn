# IMPORT
import os
from models.cpnn import *
from models.cpnn_others import cPNNExp
from models.cpnn_seq import cPNNSeq
from models.cgru import cGRULinear
from models.clstm import *
import numpy as np
import pandas as pd
import pickle
import argparse

# EDITABLE PARAMETERS
dataset = "sine_rw_10_1234"

# OTHER PARAMETERS
batch_size = 128
hidden_size = 50
seq_len = 10
# TODO
iterations = 1
loss_on_seq = False
freeze_inputs_weights = False
pretraining_samples = 0
pretraining_epochs = 0
write_weights = False
combination = False
rembember_initial_states = False
suffix = ""

if freeze_inputs_weights:
    suffix += "_exp"
if combination:
    suffix = "_combination" + suffix
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
    if args.model_class == "clstm":
        hidden_size = 50
    else:
        hidden_size = 128
device = torch.device("cpu")
df = pd.read_csv(os.path.join("datasets", f"{dataset}.csv"))
perf_test = {"accuracy": [], "kappa": [], "kappa_temporal": [], "loss": []}
perf_train = {"accuracy": [], "kappa": [], "kappa_temporal": [], "loss": []}
perf_anytime = {"accuracy": [], "kappa": [], "kappa_temporal": []}
seq_str = "_seq" if loss_on_seq else ""

path = os.path.join(
    "performance",
    f"{dataset}/{args.model}_{args.model_class}{seq_str}{suffix}_{hidden_size}hs",
)
if not os.path.isdir(path):
    os.makedirs(path)

path_anytime = path + "_anytime"
if not os.path.isdir(path_anytime):
    os.makedirs(path_anytime)

# UTILS
def create_cpnn():
    if not loss_on_seq:
        if not freeze_inputs_weights:
            return cPNN(column_class=model_class, device=device, seq_len=seq_len, train_verbose=False,
                        combination=combination, input_size=len(df.columns) - 2, hidden_size=hidden_size, output_size=2,
                        batch_size=batch_size)
        else:
            return cPNNExp(
                column_class=model_class,
                input_size=len(df.columns) - 2,
                hidden_size=hidden_size,
                output_size=2,
                batch_size=batch_size,
                device=device,
                seq_len=seq_len,
                train_verbose=False,
                combination=combination,
                remember_initial_states=rembember_initial_states,
            )
    return cPNNSeq(
        column_class=model_class,
        input_size=len(df.columns) - 2,
        hidden_size=hidden_size,
        output_size=2,
        batch_size=batch_size,
        device=device,
        seq_len=seq_len,
    )


# MAIN
if __name__ == "__main__":
    if args.model == "cpnn" and write_weights:
        try:
            df_test = pd.read_csv(os.path.join("datasets", f"{dataset}_test.csv"))
        except:
            pass
    models = []
    params = []
    inputs = []
    hiddens = []
    print(dataset)
    for i in range(1, iterations + 1):
        models.append([])
        params.append([])
        inputs.append([])
        hiddens.append([])
        for k in perf_test:
            perf_test[k].append([])
        for k in perf_train:
            perf_train[k].append([])
        for k in perf_anytime:
            perf_anytime[k].append([])
        models[-1].append(create_cpnn())
        print(type(models[-1][-1].columns.columns[0]).__name__)
        print(f"{i}/{iterations} iteration of {args.model}")
        for task in range(1, df["task"].max() + 1):
            params[-1].append([])
            inputs[-1].append([])
            hiddens[-1].append([])
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

            if pretraining_samples > 0:
                df_pre = df_task.iloc[:pretraining_samples, 0:]
                df_task = df_task.iloc[pretraining_samples:, 0:]
                perf_pretraining = models[-1][-1].pretraining(
                    df_pre.iloc[0:, :-1].values.astype(np.float32),
                    list(df_pre.iloc[0:, -1]),
                    pretraining_epochs,
                )
                with open(
                    os.path.join(path, "pretraining.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(perf_pretraining, f)

            for k in perf_test:
                perf_test[k][-1].append([])
            for k in perf_train:
                perf_train[k][-1].append([])
            for k in perf_anytime:
                perf_anytime[k][-1].append([])
            if len(df_task) % batch_size == 0:
                n_batches = int(len(df_task) / batch_size)
            else:
                n_batches = int(len(df_task) / batch_size) + 1
            for i in range(0, len(df_task), batch_size):
                x = df_task.iloc[i : i + batch_size, 0:-1].values.astype(np.float32)
                y = list(df_task.iloc[i : i + batch_size, -1])
                print(int(i / batch_size) + 1, "/", n_batches, " batch", end="\r")
                if len(y) >= seq_len:
                    batch_perf_test, batch_perf_anytime, batch_perf_train = models[-1][-1].test_then_train(
                        x, y
                    )
                    for k in batch_perf_test:
                        perf_test[k][-1][-1].append(batch_perf_test[k])
                    for k in batch_perf_anytime:
                        perf_anytime[k][-1][-1].append(batch_perf_anytime[k])
                    for k in batch_perf_train:
                        perf_train[k][-1][-1].append(batch_perf_train[k])
                    if args.model == "cpnn" and write_weights:
                        try:
                            df_test_task = df_test[df_test["task"] == task].drop(
                                columns="task"
                            )
                            x_test = df_test_task.iloc[0:500, 0:-1].values.astype(
                                np.float32
                            )
                            inputs[-1][-1].append(
                                models[-1][-1]
                                .columns._convert_to_tensor_dataset(x_test)
                                .detach()
                                .numpy()
                            )
                            hiddens[-1][-1].append(models[-1][-1].get_hidden(x_test))
                        except:
                            pass
                        params[-1][-1].append(
                            pickle.loads(
                                pickle.dumps(
                                    models[-1][-1]
                                    .columns.columns[-1]
                                    .lstm.weight_ih_l0.data.detach()
                                    .numpy()
                                )
                            )
                        )
            print()
            print(
                f"Accuracy media sul task {task}: {np.mean(perf_test['accuracy'][-1][-1])}"
            )
            print()

            with open(
                os.path.join(path, "test_then_train.pkl"),
                "wb",
            ) as f:
                pickle.dump(perf_test, f)

            with open(
                os.path.join(path_anytime, "test_then_train.pkl"),
                "wb",
            ) as f:
                pickle.dump(perf_anytime, f)

            with open(
                os.path.join(path, "train.pkl"),
                "wb",
            ) as f:
                pickle.dump(perf_train, f)
            with open(os.path.join(path, "models.pkl"), "wb") as f:
                pickle.dump(models, f)

            if args.model == "cpnn" and write_weights:
                with open(
                    os.path.join(path, "cpnn_params.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(params, f)

                with open(
                    os.path.join(path, "inputs.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(inputs, f)

                with open(
                    os.path.join(path, "hiddens.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(hiddens, f)
        print()
