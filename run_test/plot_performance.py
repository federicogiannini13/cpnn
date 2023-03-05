from run_test.plot_utils import *
import os

# EDITABLE PARAMETERS
datasets = [
    # "sine_rw_10_1234",
    # "sine_rw_10_1432",
    # "sine_rw_10_2143",
    # "sine_rw_10_2341",
    "sine_rw_10_2143_validation"
]

metrics = [
    "accuracy",
    "kappa",
    "kappa_temporal"
]

# OTHER PARAMETERS
performances = [
    {"performance": "test_then_train"},
    # {"performance": "second_test_on_train", "models":["all", "cpnn", "clstm"]},
    # {"performance": "second_test_on_test", "models":["all", "cpnn", "clstm"]}
]

# TEST THEN TRAIN ENSEMBLE clstm
# plot_name = "ensemble"
# dir_name = "_plot_clstm_50hs"
# models = ["ensemble_clstm_50hs_5ws", "ensemble_clstm_50hs_10ws", "ensemble_clstm_50hs_20ws"]

# TEST THEN TRAIN ENSEMBLE mixed
plot_name = "ensemble_mixed"
dir_name = "_plot_clstm_50hs_vs_cgru_128hs"
models = ["ensemble_mixed_clstm_128hs_cgru_50hs_5ws", "cpnn_clstm_50hs", "cpnn_cgru_128hs"]
use_confronto()

# TEST THEN TRAIN clstm
# plot_name = "clstm"
# dir_name = "_plot_clstm_50hs"
# models = ["single_clstm_50hs", "multiple_clstm_50hs", "cpnn_clstm_50hs"]

# TEST THEN TRAIN cgru
# plot_name = "cgru"
# dir_name = "_plot_cgru_128hs"
# models = ["single_cgru_128hs", "multiple_cgru_128hs", "cpnn_cgru_128hs"]

# # TEST THEN TRAIN WITH ENSEMBLE clstm
# plot_name = "with_ensemble"
# dir_name = "_plot_clstm_50hs"
# models = [""single_clstm_50hs", "cpnn_clstm_50hs", "ensemble_50hs_5ws"]

# TEST THEN TRAIN cpnn COMBINATION
# plot_name = "combination"
# dir_name = "_plot_clstm_50hs"
# models = ["cpnn_clstm_50hs", "cpnn_clstm_combination_50hs", "cpnn_clstm_combination_different_weights_50hs"]

# SECOND TEST
# plot_name = None
# dir_name = "_plot_clstm_50hs"
# models = ["single_clstm_50hs", "multiple_clstm_50hs", "cpnn_clstm_50hs"]

# clstm vs cGRU
# plot_name = "clstm_vs_cgru"
# dir_name = "_plot_clstm_vs_cgru"
# models = ["cpnn_cgru_128hs", "cpnn_clstm_50hs"]
# use_confronto()

# ENSEMBLE dim
plot_name = "ensemble_dim"
dir_name = "_plot_cgru"
models = [
    "ensemble_mixed_clstm_128hs_cgru_50hs_5ws",
    "ensemble_mixed_clstm_128hs_cgru_50hs_10ws",
    "ensemble_mixed_clstm_128hs_cgru_50hs_20ws"
]
use_ensemble_dim()

# ALTRO
# plot_name = "_clstm"
# dir_name = "_plot_remember"
# models = [
#     "single_clstm_initial_state_50hs",
#     "cpnn_clstm_initial_state_50hs",
#     "multiple_clstm_initial_state_50hs",
#     "single_clstm_50hs",
#     "cpnn_clstm_50hs",
#     "multiple_clstm_50hs"
# ]
# reset_colors()

if dir_name is not None and dir_name != "" and not dir_name.startswith("/"):
    dir_name = "/" + dir_name
elif dir_name is None:
    dir_name = ""

initialize_utils(models)

# MAIN
def plot(performance, dataset, use_one_cpnn, use_one_clstm, only_cpnn, only_clstm):
    path = f"performance/{dataset}"
    (
        perf,
        final_perf_mean,
        final_perf_ma,
        perf_mean_task,
        final_perf_mean_task,
    ) = calculate_perf(
        path, performance, use_one_cpnn, use_one_clstm, only_cpnn, only_clstm
    )

    if not os.path.isdir(path+dir_name):
        os.makedirs(path+dir_name)

    for metric in metrics:
        if plot_name is not None:
            plot_name_ = f"{performance}__{plot_name}_{metric}"
        else:
            plot_name_ = None
        plot_metric(final_perf_mean, final_perf_mean_task, final_perf_ma, metric,
                    len(perf[list(perf.keys())[0]]["accuracy"][0]), len(perf[list(perf.keys())[0]]["accuracy"][0][0]),
                    performance, dataset, path, only_cpnn, only_clstm, dir_name=dir_name, plot_name=plot_name_)
        plot_metric(final_perf_mean, final_perf_mean_task, final_perf_ma, metric,
                    len(perf[list(perf.keys())[0]]["accuracy"][0]), len(perf[list(perf.keys())[0]]["accuracy"][0][0]),
                    performance, dataset, path, only_cpnn, only_clstm, dir_name=dir_name, plot_name=plot_name_, compressed=True)


if __name__ == "__main__":
    if type(datasets) != list:
        datasets = [datasets]
    initialize_utils(models)
    for dataset in datasets:
        for p in performances:
            performance = p["performance"]
            if performance != "test_then_train":
                models_list = p["models"]
                if type(models_list) != list:
                    models_list = [models_list]
            else:
                models_list = ["all"]
            for models in models_list:
                print(dataset, performance, models)
                if models == "cpnn":
                    only_cpnn = True
                    only_clstm = False
                    use_one_cpnn = False
                    use_one_clstm = False
                elif models == "clstm":
                    only_cpnn = False
                    only_clstm = True
                    use_one_cpnn = False
                    use_one_clstm = False
                else:
                    only_cpnn = False
                    only_clstm = False
                    use_one_cpnn = True
                    use_one_clstm = True
                plot(
                    performance,
                    dataset,
                    use_one_cpnn,
                    use_one_clstm,
                    only_cpnn,
                    only_clstm,
                )
