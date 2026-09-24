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

# Regret of the Random Strategy
strategies_selected_rs = []
total_reward_rs = 0
total_reward_bs = 0
rewards_strategies = [0] * d
regret = []

for n in range(0, N):
    # Estrategia aleatoria
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs]
    total_reward_rs += reward_rs

    # Mejor estrategia (conocimiento perfecto a posteriori)
    for i in range(0, d):
        rewards_strategies[i] += X[n, i]
    total_reward_bs = max(rewards_strategies)

    # Regret acumulado
    regret.append(total_reward_bs - total_reward_rs)

# Retorno absoluto y relativo (respecto a la mejor estrategia)
absolute_return = (total_reward_rs - total_reward_bs) * 100
relative_return = (total_reward_rs - total_reward_bs) / total_reward_bs * 100
print("Rendimiento Absoluto: {:.0f} $".format(absolute_return))
print("Rendimiento Relativo: {:.0f} %".format(relative_return))

# Histograma de selecciones
plt.hist(strategies_selected_rs, bins=np.arange(d + 1) - 0.5, rwidth=0.8)
plt.title("Histograma de Selecciones (Estrategia Aleatoria)")
plt.xlabel("Estrategia")
plt.ylabel("Número de veces que se ha seleccionado la estrategia")
plt.xticks(range(d))
plt.show()

# Curva de arrepentimiento
plt.plot(regret)
plt.title("Curva de Arrepentimiento (Estrategia Aleatoria)")
plt.xlabel("Ronda" )
plt.ylabel("Regret acumulado")
plt.grid(True, alpha=0.3)
plt.show()