import warnings
#remove pandas warning for depreciated .append
warnings.simplefilter(action='ignore', category=FutureWarning)

from shutil import copyfile
import subprocess
import pandas as pd

__SCRIPT__ = "PARSE" # "RUN" or "PARSE" depending on what we want to do


#path to save tmp logs
LOG_PATH="/local/home/am273028/source/advection/benchmark/log"
#name of the outputted csv file
OUT_FILENAME="perfs"
#file .ini to update and use for the runtime
INIFILE_ROOTDIR = "/local/home/am273028/source/advection/benchmark/script"
BASE_INIFILE = INIFILE_ROOTDIR + "/advection.ini"

#the file used to store the mean, std, and all infos for each run
GLOBAL_CSV_FILE="/local/home/am273028/source/advection/benchmark/log/describe_all.csv"

#The configurations we want to bench
SETS={
    'kernelImpl':["BasicRange2D", "BasicRange1D", "Hierarchical" , "Scoped", "NDRange"],
    # 'kernelImpl':["BasicRange2D"],#, "BasicRange1D", "Hierarchical" , "Scoped", "NDRange"],
    # 'use_gpu':[True, False],
    'use_gpu':[False],
    '(nx,nvx)':[(128,64), (256,64), (512,64)],
}

if __name__ == "__main__":

    print("Running script in " + __SCRIPT__ +  " mode.")

    global_dataframe = pd.DataFrame(
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

                if "RUN" in __SCRIPT__ :
                    #update advection.ini
                    subprocess.run(["python3", "script/edit_ini_file.py",
                                            f"--inifile={new_inifile}",
                                            f"--kernel={kernelImpl}",
                                            use_gpu_str,
                                            f"--nx={nx}",
                                            f"--nvx={nvx}"])


                    #run the advection binary with recently modified .ini
                    subprocess.run(["./launch.sh", LOG_PATH, OUT_FILENAME, new_inifile, unique_prefix])

                if "PARSE" in __SCRIPT__:
                    parsed_file = LOG_PATH+"/"+OUT_FILENAME+".csv"
                    #average results and store into nice csv
                    df = pd.read_csv(parsed_file, sep=";")


                    global_dataframe = global_dataframe.append(
                        {
                        "global_size":nx*nvx,
                        "nx":nx,
                        "nvx":nvx,
                        "kernel":kernelImpl,
                        "error_mean":df['error'].mean(),
                        "error_std":df['error'].std(),
                        "runtime_mean":df['duration'].mean(),
                        "runtime_std":df['duration'].std(),
                        "cellspersec_mean":df['cellspersec'].mean(),
                        "cellspersec_std":df['cellspersec'].std(),
                        "throughput_mean":df['throughput'].mean(),
                        "throughput_std":df['throughput'].std(),
                        "gpu":df['gpu'][0]},#any value of the gpu should be the same so we take [0]
                        ignore_index=True)

    if "PARSE" in __SCRIPT__:
        global_dataframe.to_csv(GLOBAL_CSV_FILE, index=False)