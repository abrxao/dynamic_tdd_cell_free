def SINR_CDF():
    """for idx_bs, n_bs in enumerate(bs_n):
        for idx_ue, n_user in enumerate(users_n):
            n_monte_carlo = min_samples // (n_bs * n_user)
            sinr_values = np.array([])  # Initialize an empty array to store SINR values
            print(
                f"\nCalculating SINR for {n_user} users and {n_bs} BSs with {n_monte_carlo} Monte Carlo iterations"
            )
            for _ in tqdm(range(n_monte_carlo)):
                bs_positions = ps.equal_distribution(
                    n_bs, my_grid.width, my_grid.height
                )  # Place 16 entities
                users = ps.uniform_distribution(
                    n_user, my_grid.width, my_grid.height
                )  # Place 16 users uniformly
                distances = ps.get_distances(bs_positions, users)

                channel, _ = channelTDLCombined(
                    my_grid, n_bs, n_user, distances
                )  # Get the channel matrix
                sinr = getSINR(channel, dBmWToLinear(60), 1e-9)  # Calculate the SINR
                sinr_values = np.concatenate(
                    (sinr_values, linearToDB(sinr).numpy().flatten())
                )
            # Flatten the SINR tensor to a 1D array
            sinr_values = tf.reshape(sinr_values, [-1])
            # Calculate the CDF of the SINR values
            sinr_sorted, cdf_values = calcCDF(sinr_values.numpy())
            plt.plot(
                sinr_sorted,
                cdf_values,
                label=f"{n_user} users, {n_bs} BSs",
                color=plot_colors[idx_ue],
                ls="solid" if idx_bs == 0 else "dashed",
            )
    plt.title("CDF of SINR for Different User and BS Configurations")
    plt.xlabel("SINR (dB)")
    plt.ylabel("Cumulative Density Probability")
    plt.grid()
    plt.legend()
    plt.show()"""
    return


def sirn_per_devices():
    """tf.random.set_seed(42)  # Set random seed for reproducibility
    np.random.seed(42)  # Set random seed for reproducibility
    my_grid = SquareGrid(1000, 1000)  # Create a grid of 100x100 meters
    users_n = np.array([2, 10, 50])  # Number of users
    bs_n = np.array([64])  # Number of base stations
    min_samples = 10000  # Minimum number of samples for each SINR calculation
    plt.figure(figsize=(12, 4.5))  # Set the figure size
    dl_pow = dBmWToLinear(60)  # Downlink power in linear scale
    ul_pow = dBmWToLinear(10)  # Uplink power in linear scale

    for idx_bs, n_bs in enumerate(bs_n):
        for idx_ue, n_user in enumerate(users_n):
            n_monte_carlo = min_samples // (n_bs * n_user)
            spc_efficience = np.array(
                []
            )  # Initialize an empty array to store SINR values
            print(
                f"\nCalculating SINR for {n_user} users and {n_bs} BSs with {n_monte_carlo} Monte Carlo iterations"
            )
            for _ in tqdm(range(n_monte_carlo)):
                bs_positions = ps.equal_distribution(
                    n_bs, my_grid.width, my_grid.height
                )  # Place 16 entities
                users = ps.uniform_distribution(
                    n_user, my_grid.width, my_grid.height
                )  # Place 16 users uniformly
                distances = ps.get_distances(bs_positions, users)

                channel, _ = channelTDLCombined(
                    my_grid, n_bs, n_user, distances
                )  # Get the channel matrix
                sinr = getSINR(channel, dBmWToLinear(60), 1e-9)  # Calculate the SINR
                spc_efficience = np.concatenate(
                    (
                        spc_efficience,
                        spectralEfficiency(sinr).numpy().flatten(),
                    )
                )
            # Flatten the SINR tensor to a 1D array
            spc_efficience = tf.reshape(spc_efficience, [-1])
            # Calculate the CDF of the SINR values
            spc_sorted, cdf_values = calcCDF(spc_efficience.numpy())
            plt.plot(
                spc_sorted,
                cdf_values,
                label=f"{n_user} users, {n_bs} BSs",
                color=plot_colors[idx_ue],
                ls="solid" if idx_bs == 0 else "dashed",
            )
    plt.title("CDF of Spectral Efficiency for Different User and BS Configurations")
    plt.xlabel("Spectral Efficiency (bits/s/Hz)")
    plt.ylabel("Cumulative Density Probability")
    plt.grid()
    plt.legend()
    plt.show()"""
    return
