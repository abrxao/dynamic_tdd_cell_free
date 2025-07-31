import tensorflow as tf
import numpy as np

# to uplink and downlink
def getSINR(main_dir_channel: tf.Tensor, other_dir_channel: tf.Tensor, main_power: tf.Tensor, other_power: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Signal-to-Interference-plus-Noise Ratio (SINR).

    Args:
        channel (tf.Tensor 2D (Users, Acess Points)): The channel gain.
        power (tf.Tensor): The transmitted power.
        noise (tf.Tensor): The noise power.

    Returns:
        tf.Tensor: The SINR value.
    """
    num = np.zeros((main_dir_channel.shape[0], main_dir_channel.shape[2]))
    den = np.zeros((main_dir_channel.shape[0], main_dir_channel.shape[2]))
    # Calculating Norm of the channel
    for k in range(main_dir_channel.shape[0]):
        for j in range(main_dir_channel.shape[0]):
            for t in range(main_dir_channel.shape[2]):
                #for a in range(main_dir_channel.shape[2]):
                # var auxiliar para checar index do usuario
                num[k, t] += np.sum(np.abs(main_dir_channel[k, :, t])*main_power)**2
                den[k, t] += np.sum(k!=j*np.abs(main_dir_channel[j, :, t])*main_power)**2 + np.sum(np.abs(other_dir_channel[k, :, t])*other_power)**2 + noise**2
                
    return  num, den


def spectralEfficiency(sinr: tf.Tensor) -> tf.Tensor:
    """
    Calculate the spectral efficiency based on SINR.

    Args:
        sinr (tf.Tensor): The SINR value.

    Returns:
        tf.Tensor: The spectral efficiency in bits/s/Hz.
    """
    return tf.math.log1p(sinr) / tf.math.log(2.0)  # log base 2
