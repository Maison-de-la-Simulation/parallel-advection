#!/usr/bin/env python3

import warnings
#remove pandas warning for depreciated .append
warnings.simplefilter(action='ignore', category=FutureWarning)

from shutil import copyfile
import subprocess
from optparse import OptionParser
import pandas as pd

#path to save tmp logs
LOG_PATH="/gpfs/workdir/millana/ADVECTION_LOGS"

#file .ini to update and use for the runtime
INIFILE_ROOTDIR = "/gpfs/users/millana/source/parallel-advection/benchmark/script"
BASE_INIFILE = INIFILE_ROOTDIR + "/advection.ini"

#the file used to store the mean, std, and all infos for each run
GLOBAL_CSV_FILE="/gpfs/workdir/millana/ADVECTION_LOGS/describe_all.csv"

#The configurations we want to bench
SETS={
    'kernelImpl':["Hierarchical"],
    'use_gpu':[True, False],
    '(nx,nvx)':[(128,64), (256,64), (512,64)],
}
# SETS={
#     'kernelImpl':["BasicRange2D", "BasicRange1D", "Hierarchical" , "Scoped", "NDRange"],
#     'use_gpu':[True, False],
#     '(nx,nvx)':[(128,64), (256,64), (512,64)],
# }

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("--run", action="store_true", dest="run_mode", default=False, help="Use the script to RUN the binary and store logs with unique names based on parameters of simulation")
    parser.add_option("--parse", action="store_true", dest="parse_mode", default=False, help="Use the script in PARSE mode to parse the previously generated logs")

    (options, args) = parser.parse_args()

    RUN_MODE      = options.__dict__['run_mode']
    PARSE_MODE    = options.__dict__['parse_mode']

    if not (RUN_MODE or PARSE_MODE) :
        exit("Error: missing argument.\nUsage: orchestrate.py --help")

    if RUN_MODE   : print("Running script in RUN mode.")
    if PARSE_MODE : print("Running script in PARSE mode.")

    
    global_data_as_list = []


    for kernelImpl in SETS['kernelImpl'] :
        for use_gpu in SETS['use_gpu'] :
            for sizes in SETS['(nx,nvx)'] :
                nx = sizes[0]
                nvx = sizes[1]

                use_gpu_str = "gpu" if use_gpu else "cpu"
                
                unique_prefix = f"adv_{kernelImpl}_{nx}_{nvx}_{use_gpu_str}"
                #We copy the advection file and edit the new advection file
                new_inifile = f"{unique_prefix}.ini"
                new_inifile = INIFILE_ROOTDIR+"/"+new_inifile
                copyfile(BASE_INIFILE, new_inifile)

                #append with "--" so we can send it as an arg for other script
                use_gpu_str = "--" + use_gpu_str

                OUT_FILENAME = "perfs" + f"{unique_prefix}"

                if RUN_MODE :
                    #update advection.ini
                    subprocess.run(["python3", "script/edit_ini_file.py",
                                            f"--inifile={new_inifile}",
                                            f"--kernel={kernelImpl}",
                                            use_gpu_str,
                                            f"--nx={nx}",
                                            f"--nvx={nvx}"])


                    #run the advection binary with recently modified .ini
                    # subprocess.run(["./launch.sh", LOG_PATH, OUT_FILENAME, new_inifile, unique_prefix])
                    subprocess.run(["sbatch", "launch.sh", LOG_PATH, OUT_FILENAME, new_inifile, unique_prefix])

                if PARSE_MODE :
                    parsed_file = LOG_PATH+"/"+OUT_FILENAME+".csv"
                    #average results and store into nice csv
                    df = pd.read_csv(parsed_file, sep=";")

                    global_data_as_list.append(
                        [nx*nvx,
                        nx,
                        nvx,
                        kernelImpl,
                        df['error'].mean(),
                        df['error'].std(),
                        df['duration'].mean(),
                        df['duration'].std(),
                        df['cellspersec'].mean(),
                        df['cellspersec'].std(),
                        df['throughput'].mean(),
                        df['throughput'].std(),
                        df['gpu'][0]]#any value should be the same
                        )

    if PARSE_MODE :
        global_dataframe = pd.DataFrame(global_data_as_list,
            columns=["global_size",
                    "nx",
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
                    "gpu"]
        )
        global_dataframe.to_csv(GLOBAL_CSV_FILE, index=False)