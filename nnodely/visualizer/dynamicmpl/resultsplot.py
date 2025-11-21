import json
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from mplplots import plots

line = sys.stdin.readline().strip()
key, A, B, sample_time = None, None, None, None
name_data = None
sample_time = None

if line:
    try:
        # Convert to float and append to buffer
        data_point = json.loads(line)
        name_data = data_point['name_data']
        key = data_point['key']
        A = data_point['prediction_A']
        B = data_point['prediction_B']
        data_idxs = data_point['data_idxs']
        sample_time = data_point['sample_time']

        fig, ax = plt.subplots()
        ax.cla()
        plots.plot_results(ax, name_data, key, A, B, data_idxs, sample_time)
        plt.show()

    except ValueError:
        pass