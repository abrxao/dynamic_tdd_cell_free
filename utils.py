import tensorflow as tf

# Constantes para conversão entre linear e dB
_10 = tf.constant(10, dtype=tf.float32)
BASE_10 = tf.math.log(_10)


def linearToDB(x):
    """
    Converte um valor linear para dB.

    Parâmetros:
    - x (TensorFlow tensor): Valor em escala linear.

    Retorna:
    - TensorFlow tensor: Valor convertido para dB.
    """
    return _10 / BASE_10 * tf.math.log(x)


def linearToDBmW(x):
    """
    Converte um valor linear para dBm (decibel-milliwatts).

    Parâmetros:
    - x (TensorFlow tensor): Valor em escala linear.

    Retorna:
    - TensorFlow tensor: Valor convertido para dBm.
    """
    return _10 * tf.math.log(x) + 30  # 30 é a conversão de W para mW


def dBmWToLinear(x):
    """
    Converte um valor em dBm (decibel-milliwatts) para escala linear.

    Parâmetros:
    - x (TensorFlow tensor): Valor em dBm.

    Retorna:
    - TensorFlow tensor: Valor convertido para escala linear.
    """
    return _10 ** ((x - 30) / _10)  # 30 é a conversão de mW para W


def dBToLinear(x):
    """
    Converte um valor em dB para escala linear.

    Parâmetros:
    - x (TensorFlow tensor): Valor em dB.

    Retorna:
    - TensorFlow tensor: Valor convertido para escala linear.
    """
    return _10 ** (x / _10)
