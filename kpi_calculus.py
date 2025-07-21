import tensorflow as tf


# to uplink and downlink
def getSINR(channel: tf.Tensor, power: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Signal-to-Interference-plus-Noise Ratio (SINR).

    Args:
        channel (tf.Tensor 2D (Users, Acess Points)): The channel gain.
        power (tf.Tensor): The transmitted power.
        noise (tf.Tensor): The noise power.

    Returns:
        tf.Tensor: The SINR value.
    """
    sinr = tf.Variable(tf.zeros_like(channel, dtype=tf.float32))
    # Calculating Norm of the channel

    for idx, ue_channel in enumerate(channel):
        # Removing values of index idx from the channel
        channel_without_idx = tf.concat([channel[:idx], channel[idx + 1 :]], axis=0)
        # Calculating the interference for the current user
        interference = tf.reduce_sum(channel_without_idx * power, axis=0)
        sinr[idx].assign(
            power * ue_channel / (interference + noise * tf.norm(channel) ** 2)
        )
    return sinr


def spectralEfficiency(sinr: tf.Tensor) -> tf.Tensor:
    """
    Calculate the spectral efficiency based on SINR.

    Args:
        sinr (tf.Tensor): The SINR value.

    Returns:
        tf.Tensor: The spectral efficiency in bits/s/Hz.
    """
    return tf.math.log1p(sinr) / tf.math.log(2.0)  # log base 2
