import sys, json

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mplplots import plots

# Plot data
line = sys.stdin.readline().strip()
name, x, y = None, None, []
chan_centers = []

if line:
    try:
        # Convert to float and append to buffer
        data_point = json.loads(line)
        name = data_point["name"]
        x = data_point["x"]
        chan_centers = data_point["chan_centers"]
        tableau_colors = mcolors.TABLEAU_COLORS
        num_of_colors = len(list(tableau_colors.keys()))
        for ind, key in enumerate(data_point["y"].keys()):
            y.append(data_point["y"][key])

        fig, ax = plt.subplots()
        ax.cla()
        plots.plot_fuzzy(ax, name, x, y, chan_centers)
        plt.show()

    except ValueError:
        pass
