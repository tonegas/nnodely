import sys
import json

import matplotlib.pyplot as plt

from mplplots import plots

# Plot data
line = sys.stdin.readline().strip()
name, x, x0, x1, params, output = None, None, None, None, None, None
data_point = {}
input_names = []
if line:
    try:
        # Convert to float and append to buffer
        data_point = json.loads(line)
        name = data_point["name"]
        if "x1" in data_point.keys():
            x0 = data_point["x0"]
            x1 = data_point["x1"]
        else:
            x = data_point["x0"]
        params = data_point["params"]
        input_names = data_point["input_names"]
        output = data_point["output"]

        if "x1" in data_point.keys():
            plots.plot_3d_function(plt, name, x0, x1, params, output, input_names)
        else:
            plots.plot_2d_function(plt, name, x, params, output, input_names)
        plt.show()

    except ValueError:
        pass
