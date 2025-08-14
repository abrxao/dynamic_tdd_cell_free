import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import numpy as np
import time

from positioning import Positioning as ps
from entities import SquareGrid
from cycler import cycler
from utils import *
from kpi_calculus import getSINR, spectralEfficiency
from channel import LOSprobability
from channel import channelTDLCombined, channelUMa, ChannelConfig
from tqdm import tqdm


plot_colors = ["sienna", "royalblue", "cadetblue", "olivedrab", "gold", "orangered"]
# Algumas configurações estéticas para os gráficos
mpl.rcParams["lines.linewidth"] = 0.8
mpl.rcParams["axes.prop_cycle"] = cycler(color=plot_colors)
mpl.rcParams["lines.linestyle"] = (0, (10, 5))
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["font.family"] = "Jetbrains Mono"
plt.rcParams["axes.facecolor"] = "seashell"
plt.rcParams["axes.labelcolor"] = "#2d2d2d"
plt.rcParams["xtick.color"] = "#3f3f3f"
plt.rcParams["ytick.color"] = "#3f3f3f"
plt.rcParams["legend.facecolor"] = "white"


def plotPositions(users, bs_positions, my_grid, bs_n):
    plt.scatter(
        users[0, :, 0], users[0, :, 1], label="Users"
    )  # Scatter plot of users with color based on path loss
    plt.scatter(
        bs_positions[0, :, 0],
        bs_positions[0, :, 1],
        marker="1",
        label="BSs",
        color="red",
        linewidths=2,
        s=100,
    )  # Scatter plot of BS positions

    plt.xticks(np.arange(0, my_grid.width + 1, my_grid.width / np.sqrt(bs_n) / 2))
    plt.yticks(np.arange(0, my_grid.height + 1, my_grid.height / np.sqrt(bs_n) / 2))
    plt.grid()
    plt.legend()
    plt.xlim(0, my_grid.width)
    plt.ylim(0, my_grid.height)
    plt.show()


def calcCDF(data):
    """
    Calculate the Cumulative Density Probability of the given data.
    """
    data_sorted = np.sort(data)
    cdf_values = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    cdf_values = np.clip(cdf_values, 0, 1)  # Ensure CDF values are between 0 and 1
    return data_sorted, cdf_values


def main():
    # Set timer to see how long this code runs
    tf.random.set_seed(1337)  # Set random seed for reproducibility
    np.random.seed(1337)  # Set random seed for reproducibility
    my_grid = SquareGrid(1000, 1000)  # Create a grid of 100x100 meters
    users_n_arr = np.array([2, 10, 50])  # Array of possible number of users
    bs_n_arr = np.array([4])  # Array of possible number of base stations
    config_pairs = ["AP-AP", "UE-UE", "AP-UE"]  # Different link pairs to test

    # Instantiate ChannelConfig with desired parameters
    channel_config = ChannelConfig(
        n_bs=bs_n_arr[0],  # Use the first value from the array
        n_user=users_n_arr[0],  # Use the first value from the array
        batch_size=1,
        carrier_frequency=3.5e9,
        o2i_model="low",
        bs_num_rows_per_panel=2,
        bs_num_cols_per_panel=1,
        ut_num_rows_per_panel=1,
        ut_num_cols_per_panel=1,
        num_ofdm_symbols=1,
        fft_size=1,
        subcarrier_spacing=15e3,
        direction="downlink",
        link_pair="AP-UE",
    )

    bs_positions = ps.equal_distribution(
        channel_config.n_bs,
        my_grid.width,
        my_grid.height,
        z_min_max=(30.0, 70.0),
        batch_size=channel_config.batch_size,
    )
    users = ps.uniform_distribution(
        channel_config.n_user,
        my_grid.width,
        my_grid.height,
        z_value=1.4,
        batch_size=channel_config.batch_size,
    )
    # plot positions
    # Pass only ut_loc, bs_loc, and config
    channel = channelUMa(ap_loc=bs_positions, ue_loc=users, config=channel_config)
    g_APUE = channel(channel_config.batch_size)
    H_APAP = g_APUE
    h_UEUE = g_APUE


if __name__ == "__main__":
    main()
