from docplex.mp.model import Model
import numpy as np
from kpi_calculus import getSINR_DL, getSINR_UL, getSINR_DL_cplex

# Suponha que esses sejam os tamanhos
A = 2  # Numero de APs
N = 2  # Numero de antenas do APs
P_max_ap = 100  # Potência máxima APs
P_max_ue = 100  # Potência máxima UEs
K = 2  # Numero de UEs
T = 4  # Numero de slots de tempo


x_mtx_UL = np.random.randint(0, 2, size=(A, K, T))  # Matriz de associação
x_mtx_DL = np.abs(x_mtx_UL - 1)
g = np.random.rand(K, A, T, N) + 1j * np.random.rand(K, A, T, N)  # Canal de downlink
h = np.random.rand(K, A, T) + 1j * np.random.rand(K, A, T)  # Canal de uplink
H_ap = np.random.rand(A, A, T, N, N) + 1j * np.random.rand(A, A, T, N, N)
g_hat = g.conj()
g_est = g
max_kt = np.argmax(h, axis=1)  # Máximo de KAT por slot de tempo

x_mtx = np.zeros_like(x_mtx_UL)  # Inicialização da matriz de associação para uplink
x_mtx_UL = np.zeros_like(x_mtx_UL)
x_mtx_DL = np.zeros_like(x_mtx_DL)

for k in range(K):
    for t in range(T):
        a_idx = max_kt[k, t]
        x_mtx[a_idx, k, t] = 1
        x_mtx_UL[a_idx, k, t] = 1 * np.random.randint(0, 2)
        x_mtx_DL[a_idx, k, t] = np.abs(x_mtx_UL[a_idx, k, t] - 1)
UE_in_UL = np.ceil(np.sum(x_mtx_UL, axis=0) / np.max(np.sum(x_mtx_UL, axis=0)))
UE_in_DL = np.ceil(np.sum(x_mtx_DL, axis=0) / np.max(np.sum(x_mtx_DL, axis=0)))
""" print("UE_in_UL:", UE_in_UL)
print("UE_in_DL:", UE_in_DL)
print("x_mtx_UL:", x_mtx_UL)
print("x_mtx_DL:", x_mtx_DL)
UE_in_UL = np.random.randint(0, 2, size=(K, T))  # Matriz de associação para uplink
UE_in_DL = np.abs(x_mtx_UL - 1)  # Matriz de associação para uplink """

# Trocando dimensões do g para compatibilidade
v = np.transpose(g, (1, 0, 2, 3))  # (A, K, T, N)
noise = 1e-9  # Ruído

# UE_in_DL = np.ceil(np.sum(x_mtx_DL, axis=0) / np.max(np.sum(x_mtx_DL, axis=0)))

rho_DL_initial = np.random.rand(A, K, T) * x_mtx_DL  # Inicialização de rho_DL
rho_UL_initial = np.random.rand(K, T) * UE_in_UL  # Inicialização de rho_UL

a_n_DL, b_n_DL = getSINR_DL(g, h, v, rho_DL_initial, rho_UL_initial, noise)

a_n_UL, b_n_UL = getSINR_UL(
    g, g_est, x_mtx_UL, H_ap, v, rho_DL_initial, rho_UL_initial, noise
)

mdl = Model("otimizacao")
# \rho^{DL}: variável contínua >= 0, dimensão A x K x T
rho_DL = [
    [
        [mdl.continuous_var(lb=0, name=f"rho_DL_{a}_{k}_{t}") for t in range(T)]
        for k in range(K)
    ]
    for a in range(A)
]

# \rho^{UL}: variável contínua >= 0, dimensão K x T
rho_UL = [
    [mdl.continuous_var(lb=0, name=f"rho_UL_{k}_{t}") for t in range(T)]
    for k in range(K)
]

# s^{DL}: variável binária, dimensão (A+K) x T
s_DL_A = [[mdl.binary_var(name=f"s_DL_A_{i}_{t}") for t in range(T)] for i in range(A)]
s_DL_K = [[mdl.binary_var(name=f"s_DL_K_{i}_{t}") for t in range(T)] for i in range(K)]

# s^{UL}: variável binária, dimensão (A+K) x T
s_UL_A = [[mdl.binary_var(name=f"s_UL_A_{i}_{t}") for t in range(T)] for i in range(A)]
s_UL_K = [[mdl.binary_var(name=f"s_UL_K_{i}_{t}") for t in range(T)] for i in range(K)]

# T^{DL}: variável inteira entre 0 e T, dimensão K. Quantos slots de tempo cada usuário k está em downlink
# T^{UL}: variável inteira entre 0 e T, dimensão K. Quantos slots de tempo cada usuário k está em uplink
""" T_DL_K = [mdl.integer_var(lb=0, ub=T, name=f"T_DL_K_{k}") for k in range(K)]
T_UL_K = [mdl.integer_var(lb=0, ub=T, name=f"T_UL_K_{k}") for k in range(K)] """

# x^{DL}: variável contínua >= 0, dimensão A x K x T
x_DL = [
    [
        [mdl.continuous_var(lb=0, name=f"x_DL_{a}_{k}_{t}") for t in range(T)]
        for k in range(K)
    ]
    for a in range(A)
]
# x_UL: variável contínua >= 0, dimensão A x K x T
x_UL = [
    [
        [mdl.continuous_var(lb=0, name=f"x_UL_{a}_{k}_{t}") for t in range(T)]
        for k in range(K)
    ]
    for a in range(A)
]
a_DL = [
    [mdl.continuous_var(lb=0, name=f"a_DL_{k}_{t}") for t in range(T)] for k in range(K)
]
b_DL = [
    [mdl.continuous_var(lb=0, name=f"b_DL_{k}_{t}") for t in range(T)] for k in range(K)
]
# Broadcasting Rho_DL to match the shape of g_hat

for k in range(g.shape[0]):
    for t in range(g.shape[2]):

        for a in range(g.shape[1]):
            # for a in range(main_dir_channel.shape[2]):
            # var auxiliar para checar index do usuario
            a_DL[k][t] += np.sum(g_hat[k, a, t] * v[a, k, t]) * rho_DL[a][k][t]

        a_DL[k][t] = mdl.abs(a_DL[k][t]) ** 2

        aux_b1 = 0
        aux_b2 = 0
        aux_b3 = 0
        for j in range(g.shape[0]):
            for a in range(g.shape[1]):
                aux_b1 += np.sum(g_hat[k, a, t] * v[a, j, t]) * rho_DL[a][j][t]
            if j != k:
                aux_b2 += mdl.abs(aux_b1) ** 2

        for i in range(h.shape[1]):
            aux_b3 += np.abs(h[k, i, t]) ** 2 * rho_UL[i][t]
        b_DL[k][t] = aux_b2 + aux_b3 + noise**2


a_UL = [
    [mdl.continuous_var(lb=0, name=f"a_UL_{k}_{t}") for t in range(T)] for k in range(K)
]
b_UL = [
    [mdl.continuous_var(lb=0, name=f"b_UL_{k}_{t}") for t in range(T)] for k in range(K)
]
g_est_herm = g_est.conj()

for k in range(g.shape[0]):
    for t in range(g.shape[2]):
        aux_a1 = 0
        for a in range(g.shape[1]):
            aux_a1 += np.sum(g_est_herm[k, a, t] * g[k, a, t])
        a_UL[k][t] = mdl.abs(aux_a1) ** 2 * rho_UL[k][t]

        aux_b1 = 0
        aux_b2 = 0
        aux_b3 = 0
        aux_b4 = 0
        aux_b5 = 0
        for j in range(g.shape[0]):
            for a in range(g.shape[1]):
                aux_b1 += np.sum(g_est_herm[k, a, t] * g[j, a, t])
            if j != k:
                aux_b2 += np.abs(aux_b1) ** 2 * rho_UL[j][t]

        for k_lin in range(g.shape[0]):
            for a in range(g.shape[1]):
                for a_lin in range(g.shape[1]):
                    aux_b3 += rho_DL[a_lin][k_lin][t] * np.sum(
                        g_est_herm[k, a, t] @ H_ap[a_lin, a, t] * v[a_lin, k_lin, t]
                    )
            aux_b4 += mdl.abs(aux_b3) ** 2

        for a in range(g.shape[1]):
            aux_b5 += (
                x_mtx_UL[a, k, t] * np.abs(np.sum(g_est[k, a, t] * g_est[k, a, t])) ** 2
            )
        b_UL[k][t] = aux_b2 + aux_b4 + aux_b5 * noise**2
""" a_DL = [
    [
        (j != k)
        * np.abs(np.sum(g_hat[k, a, t] * v[a, j, t]) * np.sqrt(rho_DL[a, j, t])) ** 2 + np.abs(h[k, i, t]) ** 2 * rho_UL[i, t]
        for t in range(T)
    ]
    for a in range(A)
    for j in range(K)
    for k in range(K)
]
b_DL = [
    [np.sum(g_hat[k, a, t] * v[a, j, t]) * np.sqrt(rho_DL[a, j, t]) for t in range(T)]
    for a in range(A)
    for k in range(K)
] """
""" for k in range(K):
    mdl.add_constraint(T_DL_K[k] == mdl.sum(s_DL_K[k][t] for t in range(T)))
    mdl.add_constraint(T_UL_K[k] == mdl.sum(s_UL_K[k][t] for t in range(T)))
    mdl.add_constraint(T_DL_K[k] + T_UL_K[k] == T) """

for t in range(T):
    # Expression 25.a
    for a in range(A):
        mdl.add_constraint(s_DL_A[a][t] + s_UL_A[a][t] <= 1)  # Constraint 29.b

    for k in range(K):
        mdl.add_constraint(s_DL_K[k][t] + s_UL_K[k][t] <= 1)  # Constraint 29.b

    for a in range(A):
        for k in range(K):
            mdl.add_constraint(x_DL[a][k][t] <= s_DL_A[a][t])  # Constraint 29.c
            mdl.add_constraint(x_UL[a][k][t] <= s_UL_A[a][t])  # Constraint 29.d
            mdl.add_constraint(x_DL[a][k][t] <= s_DL_K[k][t])  # Constraint 29.e
            mdl.add_constraint(x_UL[a][k][t] <= s_UL_K[k][t])  # Constraint 29.f

    for a in range(A):
        mdl.add_constraint(
            mdl.sum(x_DL[a][k][t] for k in range(K)) <= N
        )  # Constraint 29.g
        mdl.add_constraint(
            mdl.sum(x_UL[a][k][t] for k in range(K)) <= N
        )  # Constraint 29.h

    for a in range(A):
        for k in range(K):
            mdl.add_constraint(
                rho_DL[a][k][t] <= x_DL[a][k][t] * P_max_ap
            )  # Constraint 29.i

    for a in range(A):
        mdl.add_constraint(
            mdl.sum(rho_DL[a][k][t] for k in range(K)) <= P_max_ap
        )  # Constraint 29.j

    for k in range(K):
        mdl.add_constraint(rho_UL[k][t] <= s_UL_K[k][t] * P_max_ue)  # Constraint 29.k

    for k in range(K):
        mdl.add

    for a in range(A):
        for k in range(K):
            mdl.add_constraint(
                x_mtx_UL[a][k][t] == x_DL[a][k][t] + x_UL[a][k][t]
            )  # Constraint 29.n
E_DL_bar_k = [mdl.continuous_var(lb=0, name=f"E_DL_bar_{k}") for k in range(K)]
E_UL_bar_k = [mdl.continuous_var(lb=0, name=f"E_UL_bar_{k}") for k in range(K)]
for k in range(K):
    for t in range(T):
        if t == 1:
            E_DL_bar_k[k] = 0
            E_UL_bar_k[k] = 0

        if UE_in_DL[k, t] == 1:
            E_DL_bar_k[k] += np.log(1 + a_n_DL[k][t] / b_n_DL[k][t])
            +a_n_DL[k][t] / b_n_DL[k][t] * (
                2 * a_DL[k][t] / a_n_DL[k][t]
                - (a_DL[k][t] + b_DL[k][t]) / (a_n_DL[k][t] + b_n_DL[k][t])
                - 1
            )
        else:
            E_UL_bar_k[k] += np.log(1 + a_n_UL[k][t] / b_n_UL[k][t])
            +a_n_UL[k][t] / b_n_UL[k][t] * (
                2 * a_UL[k][t] / a_n_UL[k][t]
                - (a_UL[k][t] + b_UL[k][t]) / (a_n_UL[k][t] + b_n_UL[k][t])
                - 1
            )
    T_k_DL = np.sum(UE_in_DL[k, :])
    T_k_UL = np.sum(UE_in_UL[k, :])
    if T_k_DL > 0:
        E_DL_bar_k[k] = 1 / (T_k_DL * np.log(2)) * E_DL_bar_k[k]
    if T_k_UL > 0:
        E_UL_bar_k[k] = 1 / (T_k_UL * np.log(2)) * E_UL_bar_k[k]


E_bar_k = E_DL_bar_k + E_UL_bar_k
mdl.set_objective("max", E_bar_k)
mdl.print_information()
mdl.solve()
