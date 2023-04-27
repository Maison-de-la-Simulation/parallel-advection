import subprocess


if __name__ == "__main__":
    inifile = "script/advection.ini"

    sets={
        'kernelImpl':["BasicRange2D", "BasicRange1D", "Hierarchical" , "Scoped", "NDRange"],
        'use_gpu':[True, False],
        '(nx,nvx)':[(1,2), (10,20), (100,200)],
    }

    for kernelImpl in sets['kernelImpl'] :
        for use_gpu in sets['use_gpu'] :
            for sizes in sets['(nx,nvx)'] :
                nx = sizes[0]
                nvx = sizes[1]

                use_gpu_str = "--gpu" if use_gpu else "--cpu"
                
                #update advection.ini
                subprocess.run(["python3", "script/edit_ini_file.py",
                                        f"--inifile={inifile}",
                                        f"--kernel={kernelImpl}",
                                        use_gpu_str,
                                        f"--nx={nx}",
                                        f"--nvx={nvx}"])
                
                #run the advection binary with recently modified .ini

                #parse results
