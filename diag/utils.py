import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import pprint

pp = pprint.PrettyPrinter(indent=2)

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": 14,
    }
)

# Colors associated to Kernel names in plots
KCOLORS = [
    ("BasicRange2D", "C0"),
    ("BasicRange1D", "C1"),
    ("NDRange", "C4"),
    ("Hierarchical", "C2"),
    ("Scoped", "C3"),
]

#items per second peak with STSREAM benchmark
peak_xeon = 9.6875#9.8173
peak_a100 = 87.625#86.1848
peak_mi250 = 80.8125#59.5168
peak_epyc = 9.625#9.6437
peak_genoa = 29.125#36.5901

#PEAK IN GIGA FLOP/S
# peak_xeon = 62        / 30
# peak_a100 = 9700      / 30
# peak_mi250 = 47900    / 2 / 30
# peak_epyc = 130.383   / 30
# peak_genoa = 559.698  / 30


# peak_xeon = 192
# peak_a100 = 1330
# peak_mi250 = 1056
# peak_epyc = 160
# peak_genoa = 727

# Hw list in the same order as peak
hw_peak_list = [peak_mi250, peak_a100, peak_epyc, peak_genoa, peak_xeon]
__HW_LIST = ["mi250", "a100", "epyc", "genoa", "xeon"]

__CPUS_ONLY = [None, None, "epyc", "genoa", "xeon"]
__GPUS_ONLY = ["mi250", "a100", None, None, None]

kernel_id = {
    -1: "FakeAdvector",
    0: "BasicRange3D",
    1: "Hierarchical",
    2: "NDRange",
    3: "Scoped",
    4: "StreamY",
    5: "Straddled",
    6: "Revidx",
    7: "TwoDimWG",
    8: "Seq_TwoDimWG",
    9: "HierLDG",
}

################################################################################
import json
def get_cleaned_df(path):
  with open(path, "r") as f:
    data = json.load(f)
    df = pd.DataFrame(data["benchmarks"])
    return clean_raw_df(df)

################################################################################
################################################################################
def clean_raw_df(
    df: pd.DataFrame, perf_name: str = "items_per_second", magnitude: int = 1e9
):
    """Cleans the raw data from google benchmark' json into a dataframe ready to
    be used with the plot functions.

    Args:
        df (pd.DataFrame): pd.df created with the JSON input of google benchmark
        perf_name (str): id of the interested perf in the dataframe (i.e.
        bytes_per_seconds or items_per_seconds)
        magnitude (int): order of magnitude, default Giga 1e9

    Returns:
        pd.Dataframe : cleaned dataframe ready to be inputed in
        @create_list_from_df
    """
    # ===========================================================
    # FILTERS
    # ===========================================================
    try:
        df = df.drop(df[df.error_occurred == True].index)
    except Exception:
        print("No error occured in Dataframe.")

    df = df.drop(df[df["name"].str.endswith("_mean")].index)
    df = df.drop(df[df["name"].str.startswith("BM_WgSize")].index)

    df["kernel_id"] = df["kernel_id"].map(kernel_id)

    # setting up things to work with old utils.py
    df["global_size"] = df["n0"] * df["n1"] * df["n2"]

    grouped_rows = [df.iloc[i : i + 3] for i in range(0, len(df), 3)]

    cleaned_df = pd.DataFrame(
        columns=["kernel",
                 "global_size",
                 "perf_median",
                 "perf_std",
                 "real_time",
                 "gpu"]
    )
    # Display the selected groups of rows
    for group in grouped_rows:
        df_median = group.iloc[0]
        df_stdev = group.iloc[1]
        df_cvar = group.iloc[2]

        # 1st should be mean, then stddev, then cv.
        #If assert fails here, bench probably ran without --benchmark_repetitions
        assert df_median["name"].endswith("_median")
        assert df_stdev["name"].endswith("_stddev")
        assert df_cvar["name"].endswith("_cv")
        # the id of the benchmark should be the same for all 3
        start_substring = group["name"].str.split("_").str[0].iloc[0]
        assert all(group["name"].str.startswith(start_substring))

        global_size = df_median["global_size"]
        kernel_name = df_median["kernel_id"]

        perf_median = df_median[perf_name] / magnitude
        perf_std = df_stdev[perf_name] / magnitude

        new_row = {
            "kernel": kernel_name,
            "global_size": global_size,
            "perf_median": perf_median,
            "perf_std": perf_std,
            "real_time": df_median['real_time'],
            "gpu": df_median["gpu"],
        }

        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([new_row])], ignore_index=True)

    return cleaned_df


################################################################################
################################################################################
def create_dict_from_df(df: pd.DataFrame):
    """Creates a python list from a pandas DataFrame keeping only size and
    throughput.
    """
    kernels_name = df["kernel"].unique()

    values_all_kernels = {
        #'kernel_name': ([size1,size2,size3], [value1, value2, value3])
    }

    for kernel_type in kernels_name:
        val_one_kernel = df[df["kernel"] == kernel_type]
        values_all_kernels[kernel_type] = (
            val_one_kernel["global_size"],
            val_one_kernel["perf_median"],
            val_one_kernel["perf_std"],
        )

    return values_all_kernels


################################################################################
################################################################################
def plot_values(values: dict, title: str, do_show=False, log_base=2, show_peak=False):
    """Generates a plot with errorbar for a dict of values obtained with
    the `create_dict_from_df` function.

    Args:
        values (dict): Values to plot
        title (str): Title of the matplotlib plot
        do_show (bool, optional): Show the plot. Defaults to False.
    """
    fig = plt.figure()  # figsize=(12,12)
    ax = fig.add_subplot(111)

    maxs = []
    for key, data in values.items():
        sizes, perf, std = data

        ax.errorbar(sizes, perf, capsize=1, yerr=std, label=key)
        maxs.append(max(perf))

    if show_peak:
        max_perf = max(maxs)
        ax.axhline(
            y=max_perf, color="red", linestyle="--", label=f"{round(max_perf, 4)}"
        )

        ax.axhline(
            y=max_perf,
            alpha=0.01,
            linestyle="--",
            color="red",
            label=f"Estimation *8*2 = {round(max_perf*8*2,4)}",
        )

    ax.set_title(title)
    ax.set_ylabel("Items processed (G/s)")
    ax.set_xlabel("Global size ($n_x \\times n_{y}$ with $n_x = 2^{10} = 1024 $)")
    ax.set_xscale("log", base=log_base)

    ax.legend()
    ax.grid()

    # fig.savefig(f"plot{title}.pdf")
    if do_show:
        plt.show()
    else:
        plt.close()


################################################################################
################################################################################
def plot_all_general_perf(values: list):
    """Generates the main performance benchmark plot for the paper. Saves the
    plot as `multiplot.pdf`

    Args:
        values (list): A list containing the 5 tuples, each tuple has the form
        (val_acpp, val_dpcpp, title), with val_acpp and val_dpcpp the
        dictionnaries created from the `create_dict_from_df` func.
    """
    # values = [(val_acpp, val_dpcpp, title), ...]
    fig = plt.figure(figsize=(15, 10))  # , constrained_layout=True)
    ax = fig.add_subplot(111)  # The big subplot for legend
    # gpu
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    # cpu
    ax3 = fig.add_subplot(234)
    ax4 = fig.add_subplot(235)
    ax5 = fig.add_subplot(236)

    subfigs = [ax1, ax2, ax3, ax4, ax5]

    # empty plot to fix legend
    for kernel_name, kernel_color in KCOLORS:
        ax.plot(0, 0, label=kernel_name, color=kernel_color)
    ax.plot(0, 0, "-", color="k", label="\\texttt{acpp}")
    ax.plot(0, 0, "--", color="k", label="\\texttt{dpc++}")

    # plot actual values
    for i, vi in enumerate(values):
        for compiler_id, data_compiler in enumerate(
            vi[:-1]
        ):  #:-1 to get rid of the title
            if data_compiler is not None:
                for key, data in data_compiler.items():
                    sizes, perf, std = data

                    current_color = next((c[1] for c in KCOLORS if c[0] == key), "")
                    if current_color != "":
                        plot_style = "x-"

                        if compiler_id == 1:  # it's dpcpp
                            plot_style = "x--"

                        subfigs[i].plot(
                            sizes, perf, plot_style, color=current_color, label=key
                        )

    box = ax.get_position()

    ax.legend(
        fontsize=18,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(box.width / 2 + 0.57, box.width / 2 + 0.56),
    )

    # set common labels (i.e. the labels of the large subplot)
    ax.set_ylabel("Items processed (G/s)")
    ax.set_xlabel("Global size ($n_x \\times n_{y}$ with $n_x = 1024 $)")

    # Turn off axis for the large subplot
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False)

    for i, subfig in enumerate(subfigs):
        subfig.set_title(values[i][2])
        subfig.grid(True)
        subfig.set_xscale("log", base=2)

    # plt.show()
    fig.savefig("multiplot.pdf")


################################################################################
################################################################################
## PP STUDY
################################################################################
################################################################################
# careful run with df list in the right order (same as __HW_LIST)
def create_pp_values(dfs_list: list, ny_size: int, best_runtimes):
    """_summary_

    Args:
        dfs_list (list): List of DataFrames used to compute perf-port,
        same order as hw list
        ny_size (int): Which size of n0 to keep for the perf-port. All other
        sizes will be removed from DataFrames.

    Returns:
        dict: A 3D python dictionnary pp_values[implem][hardware][app/arch]
    """
    pp_val: dict[str, dict[str, dict[str, float]]] = {
        "BasicRange2D": {},
        "BasicRange1D": {},
        "NDRange": {},
        "Hierarchical": {},
        "Scoped": {},
    }

    # calculating perf_port for arch and application efficiencies
    eff_list = ["arch", "app"]

    for key in pp_val:
        for hw in __HW_LIST:
            pp_val[key][hw] = {}
            for eff in eff_list:
                pp_val[key][hw][eff] = 0

    # we have general structure of pp_val dict of dict
    m_list_df = []  # list of dataframes we will use, same order as hw list
    for df in dfs_list:
        # we only keep the rows with targeted n0 size. n0 = global_size/n1 and n1 = 1024
        m_list_df.append(
            df.drop(df[(df["global_size"]/1024 != ny_size)].index) if df is not None else None
        )

    # for i_hw, m_df in enumerate(m_list_df):
    #     print(f'HW IS : {__HW_LIST[i_hw]}')
    #     display(m_df)

    m_lists_impl_rt = {}  # dict of lists for each impl for each hw
    for impl_name in pp_val:
        m_lists_impl_rt[impl_name] = []
        for m_df in m_list_df:
            if m_df is not None:
                values_rt = m_df[m_df["kernel"] == impl_name]["real_time"].values
                m_lists_impl_rt[impl_name].append(
                    values_rt[0] if len(values_rt) > 0 else -1
                )
                # print(m_lists_impl_rt[impl_name][-1])
            else:
                m_lists_impl_rt[impl_name].append(-1)

    # now we have the dropped dfs and the pp_val data template
    for key in pp_val:
        for i_hw, hw in enumerate(__HW_LIST):
            pp_val[key][hw]["arch"] = -1
            pp_val[key][hw]["app"] = -1

            current_df = m_list_df[i_hw]
            if current_df is not None:
                val = current_df[current_df["kernel"] == key]
                if val is not None:
                    pd_series_mem = val["perf_median"].values
                    pd_series_rt = val["real_time"].values

                    perf_mem = pd_series_mem[0] if len(pd_series_mem > 0) else -1
                    perf_rt = pd_series_rt[0] if len(pd_series_rt > 0) else -1
                else:
                    perf_mem = -1
                    perf_rt = -1

                if perf_rt != -1:
                    pp_val[key][hw]["arch"] = perf_mem / hw_peak_list[i_hw]
                    pp_val[key][hw]["app"] = best_runtimes[hw] / perf_rt

            else:
                # if is None, the application does not run on this hw, we set pp to 0
                # print(__HW_LIST[i_hw] + " is none")
                pp_val[key][hw]["arch"] = -1
                pp_val[key][hw]["app"] = -1

    return pp_val


################################################################################
################################################################################
def compute_pp(pp_values: dict, hw_subset: list, do_print=False):
    """Computes the performance portability for arch and app efficiency on a
    given hardware subset.

    Args:
        pp_values (dict): Perf-port values in the form given by the
        `create_pp_values` function.
        hw_subset (list): List containing names of the hardwares or None if the
        hw at that position is not included in the tested subset
        do_print (bool, optional): Print debug infos. Defaults to False.

    Returns:
        tuple[dict[str, float], dict[str, float]]: The pp values for arch and
        app efficiencies in the tuple (arch, app). The pp values are a dict with
        entries each implementation.
    """
    if do_print:
        pp.pprint(pp_values)

    nb_hw_in_subset = sum(x is not None for x in hw_subset)
    pp_arch: dict[str, float] = {}
    pp_app: dict[str, float] = {}
    for impl in pp_values:
        # pp_arch[impl] = 0
        # pp_app[impl]  = 0
        sum_arch = 0
        sum_app = 0
        for i_hw, hw in enumerate(hw_subset):
            if hw is not None:  # if hw is none, we just skip, it's not in the H subset
                if (
                    pp_values[impl][hw]["arch"] == -1
                ):  # impl did not run on this hw, pp is 0
                    pp_arch[impl] = -1
                    pp_app[impl] = -1
                    sum_arch = -1
                    sum_app = -1
                    break
                else:
                    sum_arch += 1 / pp_values[impl][hw]["arch"]
                    sum_app += 1 / pp_values[impl][hw]["app"]

        pp_arch[impl] = nb_hw_in_subset / sum_arch if sum_arch > 0 else 0
        pp_app[impl] = nb_hw_in_subset / sum_app if sum_app > 0 else 0

        if do_print:
            print(f"------- impl: {impl}")
            print(f"        pp arch : {pp_arch[impl]}")
            print(f"        pp app  : {pp_app[impl]}")

    return (pp_arch, pp_app)


################################################################################
################################################################################
# def plot_pp(data, data_mean_cpu, data_mean_gpu, data_mean_allsubset):
#     """Generates a performance portability histogram for the paper.

#     Args:
#         data (_type_): Values in the form created by
#         `create_pp_values` function
#         data_mean_cpu (_type_): Perf-port values (harmonic mean) for the CPU
#         subset
#         data_mean_gpu (_type_): Perf-port values (harmonic mean) for the GPU
#         subset
#         data_mean_allsubset (_type_): Perf-port values (harmonic mean) for the
#         complete subset
#     """
#     implems = data.keys()

#     num_implementations = len(implems)
#     num_hardware = len(__HW_LIST)

#     fig, axs = plt.subplots(1, num_implementations, figsize=(13, 2.8))  # , sharey=True)
#     x = np.arange(num_hardware)
#     bar_width = 0.4  # Adjust the width to control the spacing

#     __COLOR_APP = "C0"
#     __COLOR_ARCH = "C1"

#     __LINESTYLE_GPU = "--"
#     __LINESTYLE_CPU = ":"
#     __LINESTYLE_ALL = "-"

#     for i, implem in enumerate(implems):
#         app_efficiency = [
#             data[implem][hw]["app"] if data[implem][hw]["app"] != -1 else 0
#             for hw in __HW_LIST
#         ]
#         arch_efficiency = [
#             data[implem][hw]["arch"] if data[implem][hw]["arch"] != -1 else 0
#             for hw in __HW_LIST
#         ]

#         # plot bars
#         axs[i].bar(
#             x - bar_width / 2, app_efficiency, bar_width, color="C0"
#         )  # label='App Efficiency',
#         axs[i].bar(
#             x + bar_width / 2, arch_efficiency, bar_width, color="C1"
#         )  # label='Arch Efficiency',

#         # plot harmonix mean for each implem
#         # line '--' for CPU, '---' for GPUs, and '-' for all subset ?
#         axs[i].plot(
#             x,
#             [data_mean_cpu["app"][implem] for _ in range(5)],
#             linestyle=__LINESTYLE_CPU,
#             color=__COLOR_APP,
#         )  # label='CPU APP PerfPort',
#         axs[i].plot(
#             x,
#             [data_mean_cpu["arch"][implem] for _ in range(5)],
#             linestyle=__LINESTYLE_CPU,
#             color=__COLOR_ARCH,
#         )  # label='CPU ARCH PerfPort',

#         axs[i].plot(
#             x,
#             [data_mean_gpu["app"][implem] for _ in range(5)],
#             linestyle=__LINESTYLE_GPU,
#             color=__COLOR_APP,
#         )  # label='Harmonic Mean',
#         axs[i].plot(
#             x,
#             [data_mean_gpu["arch"][implem] for _ in range(5)],
#             linestyle=__LINESTYLE_GPU,
#             color=__COLOR_ARCH,
#         )  # label='Harmonic Mean',

#         axs[i].plot(
#             x,
#             [data_mean_allsubset["app"][implem] for _ in range(5)],
#             linestyle=__LINESTYLE_ALL,
#             color=__COLOR_APP,
#         )
#         axs[i].plot(
#             x,
#             [data_mean_allsubset["arch"][implem] for _ in range(5)],
#             linestyle=__LINESTYLE_ALL,
#             color=__COLOR_ARCH,
#         )

#         axs[i].xaxis.set_label_position("top")
#         axs[i].set_xlabel(implem)
#         axs[i].grid()
#         axs[i].set_ylim(0, 1.1)  # set ylim max to 1.1 so we can see
#         axs[i].set_xticks(x)
#         axs[i].set_xticklabels([str(j + 1) for j in range(num_hardware)])
#         axs[i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
#         if i != 0:
#             axs[i].set_yticklabels([])
#         # if i != 0: axs[i].yaxis.set_ticks_position('none')

#     # Legend for all the fig
#     axs[0].bar(0, 0, color=__COLOR_APP, label="App Efficiency")
#     axs[0].bar(0, 0, color=__COLOR_ARCH, label="Arch Efficiency")
#     axs[0].plot(0, 0, __LINESTYLE_CPU, color="k", label="H = CPUs")
#     axs[0].plot(0, 0, __LINESTYLE_GPU, color="k", label="H = GPUs")
#     axs[0].plot(0, 0, __LINESTYLE_ALL, color="k", label="H = CPUs $\\cup$ GPUs")
#     # axs[0].plot(0, 0, color='k', label="GPUs = 1,2 CPUs = 3,4,5")

#     plt.figlegend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))

#     axs[0].set_ylabel("Efficiency")

#     # axs[0].legend(loc='upper right', ncol=5, borderaxespad=-2.2)
#     plt.tight_layout()
#     plt.savefig("pp.pdf", bbox_inches="tight")
#     plt.show()


################################################################################
################################################################################
def plot_perfport_full(array_of_data):
    """Generates the performance portability histogram for the paper.
    Args:
        data (_type_): Values in the form created by
        `create_pp_values` function
        data_mean_cpu (_type_): Perf-port values (harmonic mean) for the CPU
        subset
        data_mean_gpu (_type_): Perf-port values (harmonic mean) for the GPU
        subset
        data_mean_allsubset (_type_): Perf-port values (harmonic mean) for the
        complete subset
    """

    fig, axs = plt.subplots(2, 5, figsize=(12, 5.2))  
    x = np.arange(len(__HW_LIST))
    bar_width = 0.4  # Adjust the width to control the spacing

    for idx_subplot, d in enumerate(array_of_data):
        sycl_impl, data, data_mean_cpu, data_mean_gpu, data_mean_allsubset = d

        implems = data.keys()

        # num_implementations = len(implems)
        num_hardware = len(__HW_LIST)


        __COLOR_APP = "C0"
        __COLOR_ARCH = "C1"

        __LINESTYLE_GPU = "--"
        __LINESTYLE_CPU = ":"
        __LINESTYLE_ALL = "-"

        for i, implem in enumerate(implems):
            if f"{implem}-{sycl_impl}".startswith('Scoped-dpcpp') : continue
            
            app_efficiency = [
                data[implem][hw]["app"] if data[implem][hw]["app"] != -1 else 0
                for hw in __HW_LIST
            ]
            arch_efficiency = [
                data[implem][hw]["arch"] if data[implem][hw]["arch"] != -1 else 0
                for hw in __HW_LIST
            ]

            # plot bars
            axs[idx_subplot][i].bar(x - bar_width / 2, app_efficiency, bar_width, color="C0"
            )  # label='App Efficiency',
            axs[idx_subplot][i].bar(
                x + bar_width / 2, arch_efficiency, bar_width, color="C1"
            )  # label='Arch Efficiency',

            # plot harmonix mean for each implem
            # line '--' for CPU, '---' for GPUs, and '-' for all subset ?
            axs[idx_subplot][i].plot(
                x,
                [data_mean_cpu["app"][implem] for _ in range(5)],
                linestyle=__LINESTYLE_CPU,
                color=__COLOR_APP,
            )  # label='CPU APP PerfPort',
            axs[idx_subplot][i].plot(
                x,
                [data_mean_cpu["arch"][implem] for _ in range(5)],
                linestyle=__LINESTYLE_CPU,
                color=__COLOR_ARCH,
            )  # label='CPU ARCH PerfPort',

            axs[idx_subplot][i].plot(
                x,
                [data_mean_gpu["app"][implem] for _ in range(5)],
                linestyle=__LINESTYLE_GPU,
                color=__COLOR_APP,
            )  # label='Harmonic Mean',
            axs[idx_subplot][i].plot(
                x,
                [data_mean_gpu["arch"][implem] for _ in range(5)],
                linestyle=__LINESTYLE_GPU,
                color=__COLOR_ARCH,
            )  # label='Harmonic Mean',

            axs[idx_subplot][i].plot(
                x,
                [data_mean_allsubset["app"][implem] for _ in range(5)],
                linestyle=__LINESTYLE_ALL,
                color=__COLOR_APP,
            )
            axs[idx_subplot][i].plot(
                x,
                [data_mean_allsubset["arch"][implem] for _ in range(5)],
                linestyle=__LINESTYLE_ALL,
                color=__COLOR_ARCH,
            )

            axs[idx_subplot][i].xaxis.set_label_position("top")
            axs[idx_subplot][i].set_xlabel(f"{implem}-{sycl_impl}")
            # axs[idx_subplot][i].grid(axis = 'y')
            axs[idx_subplot][i].grid()
            axs[idx_subplot][i].set_ylim(0, 1.1)  # set ylim max to 1.1 so we can see
            axs[idx_subplot][i].set_xticks(x)
            axs[idx_subplot][i].set_xticklabels([str(j + 1) for j in range(num_hardware)])
            axs[idx_subplot][i].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            if i != 0:
                axs[idx_subplot][i].set_yticklabels([])
            # if i != 0: axs[i].yaxis.set_ticks_position('none')

    # Legend for all the fig
    axs[1][0].bar(0, 0, color=__COLOR_APP, label="$e_{app}$")
    axs[1][0].bar(0, 0, color=__COLOR_ARCH, label="$e_{arch}$")
    axs[1][0].plot(0, 0, __LINESTYLE_CPU, color="k", label="{CPUs}")
    axs[1][0].plot(0, 0, __LINESTYLE_GPU, color="k", label="{GPUs}")
    axs[1][0].plot(0, 0, __LINESTYLE_ALL, color="k", label="{CPUs $\\cup$ GPUs}")
    # axs[0].plot(0, 0, color='k', label="GPUs = 1,2 CPUs = 3,4,5")

    # plt.figlegend(loc=(0.818, 0.117), ncol=1)
    plt.figlegend(loc=(0.8221, 0.1529), ncol=1)

    axs[-1][-1].axis('off') #remove scoped dpcpp

    # axs[0].legend(loc='upper right', ncol=5, borderaxespad=-2.2)
    plt.tight_layout()
    plt.savefig("pp.pdf", bbox_inches="tight")
    plt.show()
