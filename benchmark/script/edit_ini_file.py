import configparser
from optparse import OptionParser
import sys

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", action="store", type="string",
                      help="Path to .ini file")#, metavar="FILE")
    parser.add_option("-k", "--kernel", dest="kernelImpl", action="store", type="string",
                      help="Kernel implementation")
    
    parser.add_option("--gpu", action="store_true", dest="use_gpu", help="Use the GPU")
    parser.add_option("--cpu", action="store_false", dest="use_gpu", help="Use the CPU")

    parser.add_option("--nx", dest="nx", action="store", type="string",
                      help="Number of point for x")
    parser.add_option("--nvx", dest="nvx", action="store", type="string",
                      help="Number of point for vx")

    (options, args) = parser.parse_args()

    print(options.__dict__['filename'])
    # print(filename)

    # if len(sys.argv) < 6 :
    #     print ('usage: summarize.py path/to/file.ini kernelImpl use_gpu nvx nx')
    #     sys.exit()
    # else:
    #     file_path = sys.argv[1]

    # config = configparser.ConfigParser()


    # config.read(file_path)
    # config['run']['kernelImpl'] = 


    # nIter = int(config['run']['maxIter'])
    # dt    = float(config['discretization']['dt'])
    # dVx   = float(config['discretization']['dVx'])
    # minX  = float(config['discretization']['minRealx'])
    # maxX  = float(config['discretization']['maxRealx'])
    # minVx = float(config['discretization']['minRealVx'])

    # print ('Ini file is ', file_path)
    #    df = pd.read_csv(file, sep=";")