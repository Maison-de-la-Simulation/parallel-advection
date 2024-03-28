import pandas as pd
import json
from utils import *

# with open('STREAM_A100.json', 'r') as f:
def get_cleaned_df(path):
  with open(path, "r") as f:
    data = json.load(f)
    df = pd.DataFrame(data["benchmarks"])
    return clean_raw_df(df)

def on_pick(event):
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind
    print ('Artist picked:', event.artist)
    print ('{} vertices picked'.format(len(ind)))
    print ('Pick between vertices {} and {}'.format(min(ind), max(ind)+1))
    print ('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
    print ('Data point:', x[ind[0]], y[ind[0]])
    print()

with open("LOGS/A100/a100_acpp_with_wgsize.json", "r") as f:
# with open("LOGS/MI250/mi250_dpcpp.json", "r") as f:
# with open("LOGS/IntelPVC/FULL_BENCH_SYCL_INTEL.json", "r") as f:
    data = json.load(f)
    df = pd.DataFrame(data["benchmarks"])
    df["kernel_id"] = df["kernel_id"].map(kernel_id)

    #Filter the right benchmarks
    df = df.drop(df[df["name"].str.startswith("BM_Ad")].index)
    # df = df.drop(df[df.error_occurred == True].index)

    #TODO: HERE ONLY ONE SIZE, COULD DRAW DIFFERENT SIZES WITH DIFFERENT COLORS
    df = df.drop(df[df["ny"] != 16384].index)
    
    df = df[df["gpu"] == 1]
    # df = df[df["gpu"] == 0]

    fig, ax = plt.subplots()
    ax.plot(df.wg_size, df.bytes_per_second, 'x', picker=10)

    fig.canvas.callbacks.connect('pick_event', on_pick)

    plt.xlabel("Wg Size")
    plt.ylabel("Bytes processed per sec")
    plt.title("Work group size impact on ... hardware")
    plt.xticks([64, 128, 256, 512])

    plt.grid()
    plt.show()