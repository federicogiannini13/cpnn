import os
import pickle
import numpy as np
from matplotlib import pyplot as plt


def colors_default(m):
    if "ensemble" in m:
        return "red"
    if "different_weights" in m:
        return "grey"
    elif "exp" in m or "combination" in m:
        return "maroon"
    if "pnn" in m:
        return "green"
    elif "multiple" in m:
        return "orange"
    elif "single" in m:
        return "blue"
    return None

def colors_confronto(x):
    if "ensemble" in x:
        return "red"
    elif "cgru" in x:
        return "blue"
    else: #clstm
        return "orange"


def model_dict_single (x):
    if "single" in x:
        return "cLSTM"
    elif "multiple" in x:
        return "mcLSTM"
    else: #cpnn
        return "cPNN"

def model_dict_confronto (x):
    if "ensemble" in x:
        return "ensemble"
    if "cgru" in x:
        return "cGRU"
    else: #clstm
        return "cLSTM"

def model_dict_ensemble_dim (x):
    return x.split("_")[-1]

def colors_dict_ensemble_dim (x):
    return None


colors = colors_default
model_dict = model_dict_single


def initialize_utils(models_):
    global models
    models = models_


def reset_colors():
    global colors
    colors = lambda x: None


def use_confronto():
    global model_dict
    global colors
    model_dict = model_dict_confronto
    colors = colors_confronto

def use_ensemble_dim():
    global model_dict
    global colors
    model_dict = model_dict_ensemble_dim
    colors = colors_dict_ensemble_dim


def single_plot(perf, metric, ax, colors, n_tasks, task_length, ax_title="", compressed=False, tasks_name=[]):
    min_value = 1000
    max_value = -1000
    for m in perf:
        if not compressed:
            x = list(range(1, len(perf[m][metric]) + 1))
        else:
            x = list(range(task_length+1, task_length*4+1))
        ax.plot(
            x,
            perf[m][metric],
            label=model_dict(m),
            color=colors(m),
            linestyle="solid"
        )
        min_ = np.min(perf[m][metric])
        if min_ < min_value:
            min_value = min_
        max_ = np.max(perf[m][metric])
        if max_ > max_value:
            max_value = max_
    set_yticks = False
    for f in [0.05, 0.1, 0.2, 0.25, 0.3, 0.5]:
        f_ = f * 100
        min_ = np.floor(min_value * 100 / f_) * f_ / 100
        max_ = np.ceil(max_value * 100 / f_) * f_ / 100
        if len(np.arange(min_, max_, f)) <= 30:
            set_yticks = True
            min_value = max(min_, np.floor(max_ / 2 * 100 / f_) * f_ / 100)
            max_value = max_
            break
    if not compressed:
        ax.set_xlim(0, len(perf[m][metric]))
        for i in range(1, n_tasks):
            ax.axvline(x=i * task_length + 1, color="grey")
    else:
        ax.set_xlim(left=task_length+1, right=task_length*4)
        for i in range(2, n_tasks+1):
            ax.axvline(x=i * task_length + 1, color="grey")

    if set_yticks:
        yticks = np.arange(min_value, max_value, f)
        ax.set_yticks(yticks)
        ax.set_ylim(min_value, max_value)
    else:
        yticks = ax.get_yticks()
    for y in yticks:
        ax.axhline(y=y, linestyle="dotted", color="gainsboro")

    if compressed:
        fontsize=7
        # ax.text(391/2, min_value+.1, tasks_name[0], ha='center', va="top")
        # ax.text(391+391/2, min_value+.1, tasks_name[1], ha='center', va="top")
        # ax.text(391*2+391/2, min_value+.1, tasks_name[2], ha='center', va="top")
        ax.text(task_length+1+10, min_value+.1, tasks_name[0], ha='left', va="top")
        ax.text(task_length*2+1+10, min_value+.1, tasks_name[1], ha='left', va="top")
        ax.text(task_length*3+1+15, min_value+.1, tasks_name[2], ha='left', va="top")
        labels = ax.get_yticks()
        new_labels = []
        if len(labels)%2==0:
            for i in range(0,len(labels),2):
                new_labels.append(str(np.round(labels[i],2)))
                new_labels.append("")
        else:
            for i in range(0,len(labels)-2,2):
                new_labels.append(str(np.round(labels[i],2)))
                new_labels.append("")
            new_labels.append(str(np.round(labels[-1],2)))
        ax.set_yticklabels(new_labels)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        plt.subplots_adjust(top=2.5)
    else:
        fontsize=20
        ax.tick_params(labelsize=18)
    ax.set_xlabel("batch", fontsize=fontsize)
    ax.set_ylabel(metric, fontsize=fontsize)
    ax.set_title(ax_title, fontsize=fontsize)
    ax.legend(prop={'size': 20})


def plot_metric(
    perf,
    perf_task,
    perf_task_ma,
    metric,
    n_tasks,
    task_length,
    performance,
    dataset,
    path,
    only_cpnn=False,
    only_clstm=False,
    dir_name="",
    plot_name=None,
    compressed=False
):
    if not compressed:
        idx = 1
        nrows=3
    else:
        idx = task_length
        nrows=2
        n_tasks -= 1
    cm = 1 / 2.54
    tasks_dict = {
        "1": "Sine1+",
        "2": "Sine2+",
        "3": "Sine1-",
        "4": "Sine2-"
    }
    tasks = dataset.replace("_validation", "").split("_")[-1]
    tasks_name = [tasks_dict[tasks[i:i + 1]] for i in range(0, len(tasks))]
    tasks_sfx = " ".join(tasks_name)
    if not compressed:
        fig, ax = plt.subplots(ncols=1, nrows=nrows, figsize=(45, 25), facecolor=(1, 1, 1))
    else:
        fig = plt.figure(figsize=(12.15*cm, 6*cm))
        gs = fig.add_gridspec(ncols=1, nrows=nrows)
        ax = [fig.add_subplot(gs[i, 0]) for i in range(0, nrows)]
    i=0
    if not compressed:
        single_plot(perf, metric, ax[i], colors, n_tasks, task_length, "From the first batch", compressed, tasks_name)
        i+=1
    else:
        plt.style.use(['science', 'ieee'])
        plt.rcParams.update({'font.size': 7})
    single_plot({m: {metric: perf_task[m][metric][idx:]} for m in perf_task}, metric, ax[i], colors, n_tasks,
                task_length, "Concept specific's accuracy", compressed, tasks_name)
    i+=1
    single_plot({m: {metric: perf_task_ma[m][metric][idx:]} for m in perf_task_ma}, metric, ax[i], colors, n_tasks,
                task_length, "Accuracy's moving average", compressed, tasks_name)
    dataset = dataset.replace("_"+tasks, "")

    if not compressed:
        title = dataset.upper() + " " + tasks_sfx + " , " + metric.upper() + ", " + performance.upper()
        sfx = ""
        fontweight = "bold"
        fontsize=25
    else:
        title = tasks_sfx
        sfx = "_compressed"
        fontweight = None
        fontsize = 7
    cpnn_suff = "__cpnn" if only_cpnn else ""
    clstm_suff = "__clstm" if only_clstm else ""
    if plot_name is None:
        plot_name = f"{performance}{cpnn_suff}{clstm_suff}_{metric}"

    plt.tight_layout()
    if not compressed:
        fig.suptitle(
            title,
            fontsize=fontsize,
            fontweight=fontweight,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(
            os.path.join(f"{path}{dir_name}", plot_name+sfx+".png"),
            dpi=100,
            transparent=False
        )
    else:
        plt.savefig(
            os.path.join(f"{path}{dir_name}", plot_name + sfx + ".eps"),
            dpi=300,
            format="eps"
        )
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('default')


def use_last_model(perf, model):
    n_models = len([k for k in perf if k.startswith(f"{model}_") and k != model])
    perf[model] = perf[f"{model}_{n_models}"]
    keys = list(perf.keys())
    for k in keys:
        if k != model and k.startswith(f"{model}_"):
            del perf[k]
    return perf


def use_correct_cpnn_model(perf, model):
    new_perf = {}
    for i, k in enumerate(perf[f"{model}_1"].keys()):
        new_perf[k] = []
        for it in range(0, len(perf[f"{model}_1"][k])):
            new_perf[k].append([])
            for task in range(0, len(perf[f"{model}_1"][k][it])):
                new_perf[k][-1].append(perf[f"{model}_{task+1}"][k][it][task])
    keys = list(perf.keys())
    for k in keys:
        if k != model and k.startswith(f"{model}_"):
            del perf[k]
    perf[model] = new_perf
    return perf


def i_last(i, last):
    return max(0, i - last)


def return_zero(i, last):
    return 0


def calculate_mean_perf(perf, last=-1):
    perf_mean = {}
    if last == -1 or last is None:
        calculate_last = return_zero
    else:
        calculate_last = i_last
    for m in perf:
        perf_mean[m] = {}
        for k in ["accuracy", "kappa", "kappa_temporal"]:
            perf_mean[m][k] = []
            for p in perf[m][k]:
                p = [x for xs in p for x in xs]
                perf_mean[m][k].append(
                    [
                        np.mean(p[calculate_last(i, last) : i + 1])
                        for i in range(0, len(p))
                    ]
                )

    final_perf_mean = {}
    for m in perf_mean:
        final_perf_mean[m] = {}
        for k in ["accuracy", "kappa", "kappa_temporal"]:
            final_perf_mean[m][k] = np.asarray(perf_mean[m][k])
            final_perf_mean[m][k] = final_perf_mean[m][k].mean(axis=0)

    return final_perf_mean


def separate_different_modules(perf, model):
    for i in range(1, len(perf[model]["accuracy"][0]) + 1):
        perf[f"{model}_{i}"] = {k: [] for k in perf[model].keys()}
    for k in perf[model]:
        for m in range(0, len(perf[model][k][0][0])):
            for t in range(0, len(perf[model][k])):
                perf[f"{model}_{m + 1}"][k].append(perf[model][k][t][m])
    del perf[model]
    return perf


def calculate_perf(
    path,
    performance,
    use_one_cpnn=True,
    use_one_clstm=True,
    only_cpnn=False,
    only_clstm=False,
):
    global models
    perf = {}
    models_ = (
        [m for m in models if "ensemble" not in m]
        if "second" in performance
        else [m for m in models]
    )
    for m in models_:
        with open(os.path.join(path, m, f"{performance}.pkl"), "rb") as f:
            perf[m] = pickle.load(f)
    if performance != "test_then_train":
        perf = separate_different_modules(perf, "clstm")
        perf = separate_different_modules(perf, "cpnn")

        if use_one_cpnn:
            perf = use_correct_cpnn_model(perf, "cpnn")
        if use_one_clstm:
            perf = use_last_model(perf, "clstm")

        if only_cpnn:
            keys = list(perf.keys())
            for k in keys:
                if not (k.startswith("cpnn")):
                    del perf[k]
        elif only_clstm:
            keys = list(perf.keys())
            for k in keys:
                if not (k.startswith("clstm_")):
                    del perf[k]

    final_perf_mean = calculate_mean_perf(perf)
    final_perf_ma = calculate_mean_perf(perf, 10)

    perf_mean_task = {}
    for m in perf:
        perf_mean_task[m] = {}
        for k in ["accuracy", "kappa", "kappa_temporal"]:
            perf_mean_task[m][k] = []
            for p_iter in perf[m][k]:
                perf_mean_task[m][k].append([])
                for p_task in p_iter:
                    perf_mean_task[m][k][-1] += [
                        np.mean(p_task[: i + 1]) for i in range(0, len(p_task))
                    ]

    final_perf_mean_task = {}
    for m in perf_mean_task:
        final_perf_mean_task[m] = {}
        for k in perf_mean_task[m]:
            final_perf_mean_task[m][k] = np.asarray(perf_mean_task[m][k])
            final_perf_mean_task[m][k] = final_perf_mean_task[m][k].mean(axis=0)

    return perf, final_perf_mean, final_perf_ma, perf_mean_task, final_perf_mean_task
