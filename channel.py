import tensorflow as tf
import positioning as ps
from sionna.phy.channel.tr38901 import TDL, UMa, PanelArray
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import GenerateOFDMChannel
from lossEffects import calculatePathLoss
from entities import SquareGrid
from typing import Optional


class ChannelConfig:
    """
    A class to store and manage all configuration parameters for the UMa channel.
    """

    def __init__(
        self,
        n_bs: int = 4,  # Moved from channelUMa
        n_user: int = 2,  # Moved from channelUMa
        batch_size: int = 1,  # Moved from channelUMa
        carrier_frequency: float = 3.5e9,
        o2i_model: str = "low",
        bs_num_rows_per_panel: int = 2,
        bs_num_cols_per_panel: int = 2,
        bs_polarization: str = "dual",
        bs_polarization_type: str = "cross",
        bs_antenna_pattern: str = "38.901",
        ut_num_rows_per_panel: int = 1,
        ut_num_cols_per_panel: int = 1,
        ut_polarization: str = "single",
        ut_polarization_type: str = "V",
        ut_antenna_pattern: str = "omni",
        direction: str = "downlink",  # or "uplink"
        link_pair: str = "AP-UE",
        num_ofdm_symbols: int = 15,
        fft_size: int = 52,
        subcarrier_spacing: float = 15e3,
    ):
        self.n_bs = n_bs
        self.n_user = n_user
        self.batch_size = batch_size
        self.carrier_frequency = carrier_frequency
        self.o2i_model = o2i_model
        self.bs_num_rows_per_panel = bs_num_rows_per_panel
        self.bs_num_cols_per_panel = bs_num_cols_per_panel
        self.bs_polarization = bs_polarization
        self.bs_polarization_type = bs_polarization_type
        self.bs_antenna_pattern = bs_antenna_pattern
        self.ut_num_rows_per_panel = ut_num_rows_per_panel
        self.ut_num_cols_per_panel = ut_num_cols_per_panel
        self.ut_polarization = ut_polarization
        self.ut_polarization_type = ut_polarization_type
        self.ut_antenna_pattern = ut_antenna_pattern
        self.direction = direction
        self.link_pair = link_pair  # AP-UE, AP-AP, or UE-UE
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.subcarrier_spacing = subcarrier_spacing

    def get_config(self) -> dict:
        """Returns a dictionary containing all stored configuration parameters."""
        return self.__dict__


def channelUMa(
    ap_loc: tf.Tensor,
    ue_loc: tf.Tensor,
    config: Optional[ChannelConfig] = None,
) -> GenerateOFDMChannel:
    """
    Generates a UMa channel model.

    Parameters:
    - ap_loc (tf.Tensor): Base station locations (batch_size, n_bs, 3).
    - ue_loc (tf.Tensor): User terminal locations (batch_size, n_user, 3).
    - config (Optional[ChannelConfig]): An instance of ChannelConfig containing channel parameters.
                                        If None, default parameters will be used.

    Returns:
    - GenerateOFDMChannel: A Sionna GenerateOFDMChannel object.
    """
    if config is None:
        config = ChannelConfig()  # Use default configuration if none provided
    # Create the channel model with the provided configuration
    ap_array = PanelArray(
        num_rows_per_panel=config.bs_num_rows_per_panel,
        num_cols_per_panel=config.bs_num_cols_per_panel,
        polarization=config.bs_polarization,
        polarization_type=config.bs_polarization_type,
        antenna_pattern=config.bs_antenna_pattern,
        carrier_frequency=config.carrier_frequency,
    )
    ue_array = PanelArray(
        num_rows_per_panel=config.ut_num_rows_per_panel,
        num_cols_per_panel=config.ut_num_cols_per_panel,
        polarization=config.ut_polarization,
        polarization_type=config.ut_polarization_type,
        antenna_pattern=config.ut_antenna_pattern,
        carrier_frequency=config.carrier_frequency,
    )
    # By default, the link is AP-UE
    node_array1 = ap_array  # Use base station array for the first node
    node_array2 = ue_array  # Use user terminal array for the second node
    node_loc1 = ap_loc  # Use base station locations for the first node
    node_loc2 = ue_loc  # Use user terminal locations for the second node
    # If the link is AP-AP or UE-UE, adjust the node arrays and locations accordingly
    if config.link_pair == "AP-AP":
        node_array2 = ap_array
        node_loc2 = ap_loc  # Use the same locations for both base stations
        config.n_user = config.n_bs  # Use the same number of base stations as users
    if config.link_pair == "UE-UE":
        node_array1 = ue_array
        config.n_bs = config.n_user  # Use the same number of users as base stations
        node_loc1 = ue_loc  # Use user terminal locations for both

    channel_model = UMa(
        carrier_frequency=config.carrier_frequency,
        o2i_model=config.o2i_model,
        bs_array=node_array1,
        ut_array=node_array2,
        direction=config.direction,
    )
    # Getting tensor of indoor/outdoor state
    in_state = tf.ones(
        (config.batch_size, config.n_user), dtype=tf.bool
    )  # Assuming all are outdoor for simplicity
    node_orientations1 = tf.zeros(
        (config.batch_size, config.n_bs, 3), dtype=tf.float32
    )  # No orientation
    node_orientations2 = tf.zeros(
        (config.batch_size, config.n_user, 3), dtype=tf.float32
    )  # No orientation
    ut_velocities = tf.zeros(
        (config.batch_size, config.n_user, 3), dtype=tf.float32
    )  # No velocity
    channel_model.set_topology(
        bs_loc=node_loc1,
        ut_loc=node_loc2,
        in_state=in_state,
        bs_orientations=node_orientations1,
        ut_orientations=node_orientations2,
        ut_velocities=ut_velocities,
    )
    rg = ResourceGrid(
        num_ofdm_symbols=config.num_ofdm_symbols,
        fft_size=config.fft_size,
        subcarrier_spacing=config.subcarrier_spacing,
        num_tx=config.n_bs,  # Use n_bs from config
    )
    channel = GenerateOFDMChannel(channel_model=channel_model, resource_grid=rg)
    return channel


def channelTDLCombined(
    grid: SquareGrid, bs_n: int, users_n: int, distances
) -> tf.Tensor:
    # Generate a TDL model(Small Scale Model) with 100 ns delay spread
    tdl_A = TDL(
        model="A",  # NLOS
        delay_spread=100e-9,  # 100 ns delay spread
        carrier_frequency=3.5e9,
        los_angle_of_arrival=50,
        num_tx_ant=bs_n,
        num_rx_ant=users_n,
    )
    a_A, _ = tdl_A(
        batch_size=1,
        num_time_steps=1,
        sampling_frequency=1,
    )

    tdl_D = TDL(
        model="D",  # NLOS
        delay_spread=100e-9,  # 100 ns delay spread
        carrier_frequency=3.5e9,
        los_angle_of_arrival=50,
        num_tx_ant=bs_n,
        num_rx_ant=users_n,
    )
    a_D, _ = tdl_D(
        batch_size=1,
        num_time_steps=1,
        sampling_frequency=1,
    )

    distances = tf.reshape(distances, [1, 1, users_n, 1, bs_n, 1, 1])

    # sinr = getSINR(a_A, 1, 1e-9)
    LOS_prob = LOSprobability(distances)
    random_LOS_prob = tf.random.uniform(
        shape=distances.shape, minval=0, maxval=1, dtype=tf.float32
    )
    is_los = tf.cast(random_LOS_prob < LOS_prob, tf.float32)
    is_los = tf.broadcast_to(is_los, shape=a_D.shape)
    is_nlos = 1 - is_los
    is_nlos = tf.broadcast_to(is_nlos[:, :, :, :, :, :1, :], shape=a_A.shape)

    pl_linear = calculatePathLoss(3.5e9, distances, a_D.shape)
    a_D *= pl_linear
    pl_linear = tf.broadcast_to(pl_linear[:, :, :, :, :, :1, :], shape=a_A.shape)
    a_A *= pl_linear
    a_A = tf.abs(a_A**2)
    a_D = tf.abs(a_D**2)

    a_A = tf.reduce_sum(tf.squeeze(a_A * is_nlos), axis=-1)  # Sum
    a_D = tf.reduce_sum(tf.squeeze(a_D * is_los), axis=-1)  # Sum
    channel = a_A + a_D
    is_los = tf.transpose(tf.squeeze(is_los)[:, :, :1])  # Remove the last dimension
    return channel, is_los


def LOSprobability(dist_matrix: tf.Tensor) -> tf.Tensor:
    """
    Calcula a probabilidade de haver linha de visada (LOS - Line of Sight) entre pares de pontos (por exemplo, UEs ou APs)
    com base em uma matriz de distâncias.

    A probabilidade de LOS entre dois pontos é modelada usando a equação:
    \[
    P_{LOS} = \frac{18}{d} + \exp\left(-\frac{d}{36}\right) \cdot \left(1 - \frac{18}{d}\right)
    \]
    Onde `d` é a distância entre os pontos.

    Se a distância for menor que 18 unidades, a probabilidade de LOS é igual a 1.
    Se a distância for igual a 0, indicando que o link é entre o mesmo ponto, a probabilidade de LOS é 0.

    Parâmetros:
    -----------
    dist_matrix : tf.Tensor
        Matriz de distâncias entre os pontos (UEs ou APs), onde cada elemento \( d_{ij} \) representa a distância
        entre os pontos \( i \) e \( j \).

    Retorno:
    --------
    tf.Tensor
        Matriz de probabilidades de LOS, onde cada elemento \( P_{LOS}(i, j) \) é a probabilidade de haver LOS
        entre os pontos \( i \) e \( j \), calculada com base na distância correspondente na `dist_matrix`.

    Método:
    -------
    - Se a distância entre dois pontos for igual a 0, o link é entre o mesmo ponto (UE ou AP), então a probabilidade de LOS é 0.
    - Se a distância for menor que 18, a probabilidade de LOS é 1.
    - Para distâncias maiores que 18, a probabilidade de LOS é calculada com a equação dada.

    Exemplo:
    --------
    >>> dist_matrix = np.array([[0, 10, 20], [10, 0, 30], [20, 30, 0]])
    >>> LOSprobability(dist_matrix)
    array([[0.        , 0.8959269 , 0.66432157],
           [0.8959269 , 0.        , 0.45055952],
           [0.66432157, 0.45055952, 0.        ]])
    """
    # Calcula a probabilidade de LOS usando a equação fornecida
    prob = 18 / dist_matrix + tf.math.exp(-dist_matrix / 36) * (1 - 18 / dist_matrix)
    prob_matrix = tf.where(dist_matrix != 0, tf.where(dist_matrix < 18, 1, prob), 0)

    return prob_matrix
