import numpy as np
import cvxpy as cp
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. PARÂMETROS
# ==============================================================================
print("1. Definindo parâmetros...")
A, K, T, N = 2, 3, 4, 2
P_max_ap, P_max_ue = 1.0, 0.1
MAX_ITER, EPSILON = 20, 1e-3

# ==============================================================================
# 2. DADOS DE EXEMPLO
# ==============================================================================
print("2. Gerando dados de exemplo...")
sigma_sq = 1e-13
g = (np.random.randn(K, A, T, N) + 1j * np.random.randn(K, A, T, N)) * np.sqrt(0.5)
h = (np.random.randn(K, K, T) + 1j * np.random.randn(K, K, T)) * np.sqrt(0.5)
g_hat_norm = np.linalg.norm(g, axis=3, keepdims=True)
v = g / np.maximum(g_hat_norm, 1e-9) # Normaliza o beamforming

# ==============================================================================
# 3. ASSOCIAÇÃO UE-AP
# ==============================================================================
print("3. Executando associação UE-AP...")
s_DL_A = np.zeros((A, T)); s_UL_A = np.zeros((A, T))
s_DL_K = np.zeros((K, T)); s_UL_K = np.zeros((K, T))
x_DL = np.zeros((A, K, T)); x_UL = np.zeros((A, K, T))

for t in range(T):
    mode = 'DL' if t % 2 == 0 else 'UL'
    s_DL_A[:, t], s_UL_A[:, t] = (1, 0) if mode == 'DL' else (0, 1)
    s_DL_K[:, t], s_UL_K[:, t] = (1, 0) if mode == 'DL' else (0, 1)

channel_strength = np.linalg.norm(g, axis=3)**2
for t in range(T):
    ap_counts = np.zeros(A)
    for k in range(K):
        if s_DL_K[k, t]:
            best_ap = np.argmax([channel_strength[k, a, t] * s_DL_A[a, t] for a in range(A)])
            if ap_counts[best_ap] < N:
                x_DL[best_ap, k, t] = 1
                ap_counts[best_ap] += 1

# ==============================================================================
# 4. INICIALIZAÇÃO
# ==============================================================================
print("4. Inicializando variáveis de potência e SINR...")
rho_DL_n = np.random.uniform(0, 0.1 * P_max_ap, size=(A, K, T)) * x_DL
rho_UL_n = np.random.uniform(0, 0.1 * P_max_ue, size=(K, T)) * s_UL_K
gamma_n = np.ones((K, T)) # Inicializa SINR

# ==============================================================================
# 5. ALGORITMO SCO COM CONVEXIFICAÇÃO DE RESTRIÇÃO
# ==============================================================================
print("\n5. Iniciando o laço de Otimização Convexa Sucessiva (SCO)...")

for n_iter in range(MAX_ITER):
    print(f"\n--- Iteração {n_iter + 1}/{MAX_ITER} ---")

    # --- 5.1. CALCULAR VALORES DA ITERAÇÃO ANTERIOR (S^n, I^n) ---
    print("   Calculando S^n e I^n da iteração anterior...")
    S_n = np.zeros((K, T))
    I_n = np.zeros((K, T))

    for k in range(K):
        for t in range(T):
            if s_DL_K[k, t]:
                # Cálculo do Sinal S^n
                signal_term = np.sum([np.sqrt(rho_DL_n[a, k, t]) * np.abs(np.vdot(g[k, a, t], v[k, a, t])) for a in range(A) if x_DL[a, k, t]])
                S_n[k, t] = signal_term**2
                
                # Cálculo da Interferência I^n
                interf_dl = np.sum([np.sum([np.sqrt(rho_DL_n[a, j, t]) * np.abs(np.vdot(g[k, a, t], v[j, a, t])) for a in range(A) if x_DL[a, j, t]])**2 for j in range(K) if j != k])
                interf_ul = np.sum([rho_UL_n[i, t] * np.abs(h[k, i, t])**2 for i in range(K) if s_UL_K[i, t]])
                I_n[k, t] = interf_dl + interf_ul + sigma_sq
            else:
                S_n[k, t], I_n[k, t] = 1e-12, 1.0 # Valores placeholder para evitar NaN/Inf

    # --- 5.2. RESOLVER O PROBLEMA CONVEXO ---
    print("   Configurando e resolvendo o problema CVXPY...")

    # Novas variáveis: y_DL, rho_UL, e a SINR target gamma
    y_DL = cp.Variable(shape=(A, K, T), nonneg=True)
    rho_UL = cp.Variable(shape=(K, T), nonneg=True)
    gamma = cp.Variable(shape=(K, T), nonneg=True)

    # Objetivo: Maximizar a soma da utilidade (log) da SINR. É CÔNCAVO.
    objective = cp.Maximize(cp.sum(cp.log(1 + gamma)))

    constraints = [
        cp.power(y_DL, 2) <= x_DL * P_max_ap,
        cp.sum(cp.power(y_DL, 2), axis=1) <= P_max_ap,
        rho_UL <= s_UL_K * P_max_ue
    ]

    # Adicionar a restrição SINR convexificada
    for k in range(K):
        for t in range(T):
            if s_DL_K[k, t] == 1:
                # Expressão CVXPY para o Sinal S(p)
                signal_S_expr = cp.square(cp.sum([y_DL[a, k, t] * np.abs(np.vdot(g[k, a, t], v[k, a, t])) for a in range(A) if x_DL[a, k, t]]))
                
                # Expressão CVXPY para a Interferência I(p)
                interf_dl_expr = cp.sum([cp.square(cp.sum([y_DL[a, j, t] * np.abs(np.vdot(g[k, a, t], v[j, a, t])) for a in range(A) if x_DL[a, j, t]])) for j in range(K) if j != k])
                interf_ul_expr = cp.sum([rho_UL[i, t] * np.abs(h[k, i, t])**2 for i in range(K) if s_UL_K[i, t]])
                interf_I_expr = interf_dl_expr + interf_ul_expr + sigma_sq
                
                # Aproximação de Taylor para γ*I: S - (γ^n*I + I^n*γ - γ^n*I^n) >= 0
                # Esta restrição é CONVEXA: convex - affine >= 0
                constraints.append(signal_S_expr - (gamma_n[k, t] * interf_I_expr + I_n[k, t] * gamma[k, t] - gamma_n[k, t] * I_n[k, t]) >= 0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"   ERRO: O solver falhou na iteração {n_iter + 1} com o status: {problem.status}")
        break
        
    # --- 5.3. ATUALIZAR VARIÁVEIS E CHECAR CONVERGÊNCIA ---
    rho_DL_n_plus_1 = np.square(y_DL.value)
    rho_UL_n_plus_1 = rho_UL.value
    gamma_n_plus_1 = gamma.value
    
    # Limpeza de valores inválidos
    for arr in [rho_DL_n_plus_1, rho_UL_n_plus_1, gamma_n_plus_1]:
        if arr is not None: arr[np.isnan(arr)] = 0
    
    diff = np.max(np.abs(gamma_n_plus_1 - gamma_n))
    print(f"   Status do Solver: {problem.status}")
    print(f"   Mudança Máxima na SINR (gamma): {diff:.4f}")

    if diff < EPSILON:
        print(f"\nConvergência atingida na iteração {n_iter + 1}!")
        rho_DL_n, rho_UL_n, gamma_n = rho_DL_n_plus_1, rho_UL_n_plus_1, gamma_n_plus_1
        break
    
    rho_DL_n, rho_UL_n, gamma_n = rho_DL_n_plus_1, rho_UL_n_plus_1, gamma_n_plus_1

# ==============================================================================
# 6. RESULTADOS FINAIS
# ==============================================================================
print("\n" + "="*60)
print("6. RESULTADOS FINAIS DA OTIMIZAÇÃO")
print("="*60)
if problem.status in ["optimal", "optimal_inaccurate"]:
    print(f"\nSINR final para UE 0 nos slots de tempo:\n{gamma_n[0, :]}")
    print(f"\nPotência final DL (rho_DL) para UE 0, AP 0:\n{rho_DL_n[0, 0, :]}")
else:
    print("\nA otimização não encontrou uma solução ótima.")