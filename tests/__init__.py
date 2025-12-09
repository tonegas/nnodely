import sys

# Imposta un backend non-GUI solo su Windows
if sys.platform.startswith("win"):
    import matplotlib
    matplotlib.use("Agg")