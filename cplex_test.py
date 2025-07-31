from docplex.mp.model import Model
import numpy as np
from kpi_calculus import getSINR

# Suponha que esses sejam os tamanhos
A = 4  # Numero de APs
N = 2  # Numero de antenas do APs
K = 2  # Numero de UEs
T = 4  # Numero de slots de tempo
P_max_ap = 1  # Potência máxima APs
P_max_ue = 0.1  # Potência máxima UEs
x_mtx = np.random.randint(0, 1, size=(A, K, T))  # Matriz de alocação de recursos

g = np.random.rand(
    K, A, T
)  # g_k,a,t é um vetor com dimensão Nx1 que representa o canal entre AP a, UE k no instante t
g_hat = g  # g_hat é uma estimativa de g
v = g_hat
g_up = g * P_max_ue / 3

#  Criação do modelo
mdl = Model("power_control")
# \rho^{DL}: variável contínua >= 0, dimensão A x K x T

# Craiação das variáveis de decisão

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
    [mdl.continuous_var(name=f"a_DL_K_{i}_{t}") for t in range(T)] for i in range(K)
]
b_DL = [
    [mdl.continuous_var(name=f"b_DL_K_{i}_{t}") for t in range(T)] for i in range(K)
]

# Adicionando as restrições ao modelo
for t in range(T):
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
        mdl.add_constraint(rho_UL[k][t] <= P_max_ue)  # Constraint 29.k

    for a in range(A):
        for k in range(K):
            mdl.add_constraint(
                x_mtx[a][k][t] == x_DL[a][k][t] + x_UL[a][k][t]
            )  # Constraint 29.n

    for k in range(K):
        for a in range(A):
            mdl.add_constraint(
                a_DL[k][t]
                == mdl.sum(
                    rho_DL[a][k][t] * g_hat[k][a][t] * v[k][a][t] for k in range(K)
                )
                ** 2
            )  # Constraint 25.a
            mdl.add_constraint(
                b_DL[k][t]
                == mdl.sum(
                    int(j != k) * rho_DL[a][j][t] * g_hat[j][a][t] * v[j][a][t]
                    for a in range(A)
                    for j in range(K)
                )
                ** 2
                + mdl.sum(rho_UL[i][t] * g_up[i][a][t] for i in range(K)) ** 2
            )  # Constraint 25.b
a_n, b_n = getSINR(g, g_up, 1, 0.1, 1e-6)  # Example call to getSINR
ratio_ab = a_n / b_n  # Calculate the ratio of SINR
obj_fn = mdl.sum(
    np.log(1 + ratio_ab[k][t] ** 2)
    + ratio_ab[k][t] ** 2
    * (
        2 * (a_DL[k][t] / a_n[k][t])
        - ((a_DL[k][t] + b_DL[k][t]) / (a_n[k][t] + b_n[k][t]))
        - 1
    )
    for k in range(K)
    for t in range(T)
)
mdl.set_objective("max", obj_fn)  # Set the objective function
mdl.print_information()
mdl.solve()
