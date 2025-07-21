from docplex.mp.model import Model
import numpy as np

# Suponha que esses sejam os tamanhos
A = 3  # Numero de APs
N = 4  # Numero de antenas do APs
P_max_ap = 10  # Potência máxima APs
P_max_ue = 10  # Potência máxima UEs
K = 5  # Numero de UEs
T = 10  # Numero de slots de tempo
x_mtx = np.random.randint(0, 1, size=(A, K, T))  # Matriz de alocação de recursos


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
