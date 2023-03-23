import pandas as pd
import matplotlib.pyplot as plt
import sys, getopt

if __name__ == "__main__":
   if len(sys.argv) < 2 :
      print ('usage: summarize.py -f path/to/file.csv')
      sys.exit()
   else:
      file = sys.argv[1]

   print ('Input file is ', file)
   df = pd.read_csv(file, sep=";")