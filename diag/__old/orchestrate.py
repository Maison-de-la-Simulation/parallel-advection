#!/usr/bin/env python3

import warnings

# remove pandas warning for depreciated .append
warnings.simplefilter(action="ignore", category=FutureWarning)

from shutil import copyfile
import subprocess
from optparse import OptionParser
import pandas as pd
import os.path as path
import bench_params as p

run_with_container = True if p.CONTAINER_ROOTDIR != "" else False

executable = path.join(
    p.CONTAINER_ROOTDIR if run_with_container else p.HOST_ROOTDIR, p.EXECUTABLE
)

# base file .ini to edit and use for the runtime
host_base_inifile = path.join(p.HOST_ROOTDIR, p.INIFILE_DIR, "advection.ini")

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--run",
        action="store_true",
        dest="run_mode",
        default=False,
        help="""Use the script to RUN the binary and store logs with unique
 names based on parameters of simulation""",
    )
    parser.add_option(
        "--parse",
        action="store_true",
        dest="parse_mode",
        default=False,
        help="""Use the script in PARSE mode to parse the previously generated
 logs""",
    )
    parser.add_option(
        "--prof",
        action="store_true",
        dest="profile_mode",
        default=False,
        help="""Use the script in PROFILE mode to get GPU informations""",
    )

    (options, args) = parser.parse_args()

    RUN_MODE = options.__dict__["run_mode"]
    PARSE_MODE = options.__dict__["parse_mode"]
    PROFILE_MODE = options.__dict__["profile_mode"]

    if not (RUN_MODE or PARSE_MODE or PROFILE_MODE):
        exit("Error: missing argument.\nUsage: orchestrate.py --help")

    if RUN_MODE:
        print("Running script in RUN mode.")
    if PARSE_MODE:
        print("Running script in PARSE mode.")
    if PROFILE_MODE:
        print("Running script in NCU/ROCM PROFILE mode.")

    # init a list that we will use to create a pandas DataFrame
    global_data_as_list = []

    for kernelImpl in p.SETS["kernelImpl"]:
        for use_gpu in p.SETS["use_gpu"]:
            for sizes in p.SETS["(n1,nvx)"]:
                n1 = sizes[0]
                nvx = sizes[1]

                use_gpu_str = "gpu" if use_gpu else "cpu"

                unique_prefix = f"adv_{kernelImpl}_{n1}_{nvx}_{use_gpu_str}"

                # append with "--" so we can send it as an arg for other script
                use_gpu_str = "--" + use_gpu_str

                out_filename = "perfs" + f"{unique_prefix}"

                if RUN_MODE or PROFILE_MODE:
                    # We copy the advection file and edit the new advection file
                    uid_inifile = f"{unique_prefix}.ini"
                    new_inifile_host = path.join(p.HOST_ROOTDIR,
                                                p.INIFILE_DIR,
                                                uid_inifile)

                    copyfile(host_base_inifile, new_inifile_host)

                    # update copied advection.ini on host
                    subprocess.run(
                        [
                            "python3",
                            "script/edit_ini_file.py",
                            f"--inifile={new_inifile_host}",
                            f"--kernel={kernelImpl}",
                            use_gpu_str,
                            f"--n1={n1}",
                            f"--nvx={nvx}",
                        ]
                    )

                    # if container, run with mounted inifile
                    run_inifile = (
                        new_inifile_host
                        if not run_with_container
                        else path.join(
                            p.CONTAINER_ROOTDIR, p.INIFILE_DIR, f"{unique_prefix}.ini"
                        )
                    )

                    if PROFILE_MODE : out_filename = "prof" + f"{unique_prefix}_{p.IMPL}"

                    launch_script = "launch.sh" if not PROFILE_MODE else "launch_ncu_profiling.sh"
                    # run the advection binary with recently modified .ini
                    subprocess.run(
                        [
                            "sbatch",      # scheduler/runner
                            launch_script, # shell script
                            executable,    # arg #1
                            p.LOG_PATH,    # arg #2
                            out_filename,  # arg #3
                            run_inifile,   # arg #4
                            unique_prefix, # arg #5
                        ]
                    )


                if PARSE_MODE:
                    parsed_file = p.LOG_PATH + "/" + out_filename + ".csv"
                    # average results and store into nice csv
                    try:
                        df = pd.read_csv(parsed_file, sep=";")

                        global_data_as_list.append(
                            [
                                n1 * nvx,
                                n1,
                                nvx,
                                kernelImpl,
                                df["error"].mean(),
                                df["error"].std(),
                                df["duration"].mean(),
                                df["duration"].std(),
                                df["cellspersec"].mean(),
                                df["cellspersec"].std(),
                                df["throughput"].mean(),
                                df["throughput"].std(),
                                df["gpu"][0],
                            ]  # any value should be the same so we use [0]
                        )
                    except:
                        print(f"{parsed_file} not found, skipping it.")

    if PARSE_MODE:
        global_dataframe = pd.DataFrame(
            global_data_as_list,
            columns=[
                "global_size",
                "n1",
                "nvx",
                "kernel",
                "error_mean",
                "error_std",
                "runtime_mean",
                "runtime_std",
                "cellspersec_mean",
                "cellspersec_std",
                "throughput_mean",
                "throughput_std",
                "gpu",
            ],
        )
        global_dataframe.to_csv(p.GLOBAL_CSV_FILE, index=False)
        print(f"Write file: {p.GLOBAL_CSV_FILE}")
