import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import json

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

from mplplots import plots

# Buffer to hold the data points
data_train = deque(maxlen=2000)
data_val = deque(maxlen=2000)
last = 1
epoch = 0
# Set up the plot
fig, ax = plt.subplots()

def update_graph(frame):
    global last, title, epoch
    if last > 0:
        # Read data from stdin
        line = sys.stdin.readline().strip()
        if line:
            try:
                # Convert to float and append to buffer
                data = json.loads(line)
                data_train.append(data['train_losses'])
                if data['val_losses']:
                    data_val.append(data['val_losses'])
                title = data['title']
                key = data['key']
                last = data['last']
                epoch = data['epoch']
                # Clear the current plot
                ax.cla()
                # Clear the current plot
                plots.plot_training(ax, title, key, data_train, data_val, last)
            except ValueError:
                pass
    else:
        pass

# Use FuncAnimation to update the plot dynamically
ani = animation.FuncAnimation(fig, update_graph, interval=10, save_count=20)

# Show the plot
plt.show(block=True)
