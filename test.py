import numpy as np
from kpi_calculus import getSINR_DL

A = 3  # Numero de APs
N = 4  # Numero de antenas do APs
P_max_ap = 10  # Potência máxima APs
P_max_ue = 10  # Potência máxima UEs
K = 5  # Numero de UEs
T = 10  # Numero de slots de tempo
g = np.random.rand(K, A, T, N) + 1j * np.random.rand(K, A, T, N)  # Canal de downlink
h = np.random.rand(K, K, T) + 1j * np.random.rand(K, K, T)  # Canal de uplink
# Trocando dimensões do g para compatibilidade
v = np.transpose(g, (1, 0, 2, 3))  # (A, K, T, N)
rho_DL = np.random.rand(A, K, T)  # Variável contínua >= 0, dimensão A x K x T
rho_UL = np.random.rand(K, T)  # Variável contínua >= 0
noise = 1e-9  # Potência do ruído

a, b = getSINR_DL(
    g,
    h,
    v,
    rho_DL,
    rho_UL,
    noise,
)
print((a / b).shape)


x = np.array([1, 2])
y = np.array([[1, 2], [1, 2]])
print(x @ y)
