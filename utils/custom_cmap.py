import matplotlib.pyplot as plt

# Define the colors for the custom colormap
colors = [(0, 0, 0),        # black
        (0.4, 0, 0.5),      # purple
        (0, 0, 1),          # blue
        (0.2, 1, 0.7),      # neon green
        (1, 1, 0),          # yellow
        (1, 0.4, 0),        # orange
        (1, 0, 0),          # red
        (0.5, 0, 0)]        # dark red

# Create the colormap using the colors
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', colors)

