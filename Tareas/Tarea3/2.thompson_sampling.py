import numpy as np
import matplotlib.pyplot as plt
import random

# Configuración de los parámetros
N = 10000
d = 9

# Creación de la simulación
conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
X = np.zeros([N, d])
for i in range(N):
    for j in range(d):
        if np.random.rand() <= conversion_rates[j]:
            X[i, j] = 1

# Regret of Thompson Sampling
strategies_selected_ts = []
total_reward_ts = 0
total_reward_bs = 0
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
rewards_strategies = [0] * d
regret = []

for n in range(0, N):
    # Thompson Sampling: elegir la estrategia
    strategy_ts = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,
                                         numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            strategy_ts = i

    # Recompensa de la estrategia elegida (FUERA del bucle interno)
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] += 1
    else:
        numbers_of_rewards_0[strategy_ts] += 1
    strategies_selected_ts.append(strategy_ts)
    total_reward_ts += reward_ts

    # Mejor estrategia (con conocimiento perfecto a posteriori)
    for i in range(0, d):
        rewards_strategies[i] += X[n, i]
    total_reward_bs = max(rewards_strategies)

    # Regret acumulado
    regret.append(total_reward_bs - total_reward_ts)

# Retorno absoluto y relativo
absolute_return = (total_reward_ts - total_reward_bs) * 100
relative_return = (total_reward_ts - total_reward_bs) / total_reward_bs * 100
print("Rendimiento Absoluto: {:.0f} $".format(absolute_return))
print("Rendimiento Relativo: {:.0f} %".format(relative_return))

# Histograma de selecciones
plt.hist(strategies_selected_ts, bins=np.arange(d + 1) - 0.5, rwidth=0.8)
plt.title("Histograma de Selecciones")
plt.xlabel("Estrategia")
plt.ylabel("Número de veces que se ha seleccionado la estrategia")
plt.xticks(range(d))
plt.show()

# Curva de arrepentimiento
plt.plot(regret)
plt.title("Curva de Arrepentimiento (Thompson)")
plt.xlabel("Ronda")
plt.ylabel("Regret acumulado")
plt.grid(True, alpha=0.3)
plt.show()