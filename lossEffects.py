import tensorflow as tf
from utils import *


def calculatePathLoss(fc, distances, shape):
    """
    Calcula a perda de percurso (Path Loss) em dB e converte para escala linear.

    A função recebe parâmetros de frequência, distâncias entre usuários,
    parâmetros do cenário e o formato do tensor de taps (amplitudes retornadas
    pelo canal). Retorna um tensor com o mesmo formato, contendo os valores
    de path loss em escala linear para multiplicação ponto a ponto.

    Parâmetros:
    - fc (float ou TensorFlow tensor): Frequência de operação em Hz.
    - distances (TensorFlow tensor): Distâncias entre usuários em metros.
    - sc_params (dict): Parâmetros do cenário, incluindo `clutter_loss` e `shadow_fading`.
    - shape (tuple): Formato do tensor de taps.

    Retorna:
    - TensorFlow tensor: Path loss convertido para escala linear e pronto para uso.
    """
    # Transformando a tupla de shapes das distancias para uma lista
    distances_shape_list = list(distances.shape)
    # Muda o ultimo elemento da lista para 1
    distances_shape_list[-1] = 1
    # Converte a lista de volta para uma tupla
    path_loss = (
        # Cálculo da perda de percurso em espaço livre (Free Space Path Loss)
        32.45  # Constante da equação de FSPL
        + 36.7 * tf.math.log(distances) / tf.math.log(10.0)  # Conversão da distância
    )
    # Converte a perda de percurso de dB para linear
    pl_linear = tf.sqrt(tf.pow(10.0, -path_loss / 10.0))

    # Ajusta o tensor para ter o formato correto e retorna em número complexo
    return tf.broadcast_to(
        tf.complex(pl_linear, tf.zeros_like(pl_linear, dtype=tf.float32)), shape=shape
    )
