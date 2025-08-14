import numpy as np
from docplex.mp.model import Model

mdl = Model("otimizacao")


# to uplink and downlink
def getSINR_DL_cplex(
    g,
    h,
    v,
    rho_DL,
    rho_UL,
    noise,
):
    """
    Calculate the Signal-to-Interference-plus-Noise Ratio (SINR).

    Args:
        channel (tf.Tensor 2D (Users, Acess Points)): The channel gain.
        power (tf.Tensor): The transmitted power.
        noise (tf.Tensor): The noise power.

    Returns:
        tf.Tensor: The SINR value.
    """
    a_DL = np.zeros((g.shape[0], g.shape[2]))
    b_DL = np.zeros((g.shape[0], g.shape[2]))
    g_hat = g.conj()
    """ 
    for k in range(g.shape[0]):
        for a in range(g.shape[1]):
            for t in range(g.shape[2]):
                # for a in range(main_dir_channel.shape[2]):
                # var auxiliar para checar index do usuario
                g_hat[k, a, t] = g[k, a, t].conj() """
    # Broadcasting Rho_DL to match the shape of g_hat

    # Calculating Norm of the channel
    for k in range(g.shape[0]):
        for t in range(g.shape[2]):

            for a in range(g.shape[1]):
                # for a in range(main_dir_channel.shape[2]):
                # var auxiliar para checar index do usuario
                a_DL[k, t] += np.sum(g_hat[k, a, t] * v[a, k, t]) * 1  # rho_DL[a][k][t]

            a_DL[k, t] = mdl.abs(a_DL[k, t]) ** 2

            aux_b1 = 0
            aux_b2 = 0
            aux_b3 = 0
            for j in range(g.shape[0]):
                for a in range(g.shape[1]):
                    aux_b1 += np.sum(g_hat[k, a, t] * v[a, j, t]) * rho_DL[a][j][t]

                aux_b2 += (j != k) * mdl.abs(aux_b1) ** 2

            for i in range(h.shape[1]):
                aux_b3 += np.abs(h[k, i, t]) ** 2 * rho_UL[i][t]
            b_DL[k, t] = aux_b2 + aux_b3 + noise**2

    return a_DL, b_DL


def getSINR_DL(
    g,
    h,
    v,
    rho_DL,
    rho_UL,
    noise,
):
    """
    Calculate the Signal-to-Interference-plus-Noise Ratio (SINR).

    Args:
        channel (tf.Tensor 2D (Users, Acess Points)): The channel gain.
        power (tf.Tensor): The transmitted power.
        noise (tf.Tensor): The noise power.

    Returns:
        tf.Tensor: The SINR value.
    """
    a_DL = np.zeros((g.shape[0], g.shape[2]))
    b_DL = np.zeros((g.shape[0], g.shape[2]))
    g_hat = g.conj()
    """ 
    for k in range(g.shape[0]):
        for a in range(g.shape[1]):
            for t in range(g.shape[2]):
                # for a in range(main_dir_channel.shape[2]):
                # var auxiliar para checar index do usuario
                g_hat[k, a, t] = g[k, a, t].conj() """
    # Broadcasting Rho_DL to match the shape of g_hat

    # Calculating Norm of the channel
    for k in range(g.shape[0]):
        for t in range(g.shape[2]):

            for a in range(g.shape[1]):
                # for a in range(main_dir_channel.shape[2]):
                # var auxiliar para checar index do usuario
                a_DL[k, t] += np.sum(g_hat[k, a, t] * v[a, k, t]) * np.sqrt(
                    rho_DL[a][k][t]
                )

            a_DL[k, t] = np.abs(a_DL[k, t]) ** 2

            aux_b1 = 0
            aux_b2 = 0
            aux_b3 = 0
            for j in range(g.shape[0]):
                for a in range(g.shape[1]):
                    aux_b1 += np.sum(g_hat[k, a, t] * v[a, j, t]) * np.sqrt(
                        rho_DL[a][j][t]
                    )
                aux_b2 += (j != k) * np.abs(aux_b1) ** 2

            for i in range(h.shape[1]):
                aux_b3 += np.abs(h[k, i, t]) ** 2 * rho_UL[i][t]
            b_DL[k, t] = aux_b2 + aux_b3 + noise**2

    return a_DL, b_DL


def getSINR_UL(
    g,
    g_est,
    x_mtx,
    H_ap,
    v,
    rho_DL,
    rho_UL,
    noise,
):
    """
    Calculate the Signal-to-Interference-plus-Noise Ratio (SINR).
    """
    a_UL = np.zeros((g.shape[0], g.shape[2]))
    b_UL = np.zeros((g.shape[0], g.shape[2]))
    g_est_herm = g_est.conj()

    for k in range(g.shape[0]):
        for t in range(g.shape[2]):
            for a in range(g.shape[1]):
                a_UL[k, t] += np.sum(g_est_herm[k, a, t] * g[k, a, t])
            a_UL[k, t] = np.abs(a_UL[k, t]) ** 2 * rho_UL[k, t]

            aux_b1 = 0
            aux_b2 = 0
            aux_b3 = 0
            aux_b4 = 0
            aux_b5 = 0
            for j in range(g.shape[0]):
                for a in range(g.shape[1]):
                    aux_b1 += np.sum(g_est_herm[k, a, t] * g[j, a, t])
                aux_b2 += np.abs(aux_b1) ** 2 * rho_UL[j, t] * (j != k)

            for k_lin in range(g.shape[0]):
                for a in range(g.shape[1]):
                    for a_lin in range(g.shape[1]):
                        aux_b3 += np.sqrt(rho_DL[a_lin, k_lin, t]) * np.sum(
                            g_est_herm[k, a, t] @ H_ap[a_lin, a, t] * v[a_lin, k_lin, t]
                        )
                aux_b4 += np.abs(aux_b3) ** 2

            for a in range(g.shape[1]):
                aux_b5 += (
                    x_mtx[a, k, t]
                    * np.abs(np.sum(g_est[k, a, t] * g_est[k, a, t])) ** 2
                )
            b_UL[k, t] = aux_b2 + aux_b4 + aux_b5 * noise**2
    return a_UL, b_UL


# def spectralEfficiency(sinr: tf.Tensor) -> tf.Tensor:
#     """
#     Calculate the spectral efficiency based on SINR.
#
#     Args:
#         sinr (tf.Tensor): The SINR value.
#
#     Returns:
#         tf.Tensor: The spectral efficiency in bits/s/Hz.
#     """
#     return tf.math.log1p(sinr) / tf.math.log(2.0)  # log base 2
