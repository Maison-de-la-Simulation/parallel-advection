import configparser
from optparse import OptionParser
import sys

#python3 edit_ini_file.py --inifile=advection.ini --kernel=BasicRange2D --gpu --nx=512 --nvx=65000

def edit_ini_file(
    inifile : str,
    kernelImpl : str,
    use_gpu : bool,
    nx : int,
    nvx : int):

    config = configparser.ConfigParser()
    config.read(inifile)
    if(use_gpu):
        config['run']['gpu'] = "true"
    else:
        config['run']['gpu'] = "false"

    config['run']['kernelImpl'] = kernelImpl
    config['geometry']['nx'] = str(nx)
    config['geometry']['nvx'] = str(nvx)

    with open(inifile, 'w') as configfile:
        config.write(configfile)
    
    # print(inifile, kernelImpl, use_gpu, nx, nvx)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--inifile", dest="inifile", action="store", type=str,
                      help="Path to .ini file")#, metavar="FILE")
    parser.add_option("--kernel", dest="kernelImpl", action="store", type=str,
                      help="Kernel implementation")
    
    parser.add_option("--gpu", action="store_true", dest="use_gpu", help="Use the GPU")
    parser.add_option("--cpu", action="store_false", dest="use_gpu", help="Use the CPU")

    parser.add_option("--nx", dest="nx", action="store", type=int,
                      help="Number of point for x")
    parser.add_option("--nvx", dest="nvx", action="store", type=int,
                      help="Number of point for vx")

    (options, args) = parser.parse_args()

    inifile    = options.__dict__['inifile']
    kernelImpl = options.__dict__['kernelImpl']
    use_gpu    = options.__dict__['use_gpu']
    nx         = options.__dict__['nx']
    nvx        = options.__dict__['nvx']

    edit_ini_file(inifile, kernelImpl, use_gpu, nx, nvx)