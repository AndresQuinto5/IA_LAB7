import matplotlib.pyplot as plt

# Resultados contra Alpha-Beta Pruning
algorithms = ['Q-Learning', 'Alpha-Beta Pruning', 'Ties']
wins_alpha_beta = [19, 55, 1]
win_rates_alpha_beta = [0.25, 0.73, 0.01]

fig, ax = plt.subplots()
ax.bar(algorithms, wins_alpha_beta)
ax.set_title('Resultados contra Alpha-Beta Pruning')
ax.set_xlabel('Algoritmo')
ax.set_ylabel('Victorias')

fig2, ax2 = plt.subplots()
ax2.bar(algorithms, win_rates_alpha_beta)
ax2.set_title('Tasas de victoria contra Alpha-Beta Pruning')
ax2.set_xlabel('Algoritmo')
ax2.set_ylabel('Tasa de victoria')

# Resultados contra Minimax
wins_minimax = [42, 0, 33]
win_rates_minimax = [0.56, 0.0, 0.44]

fig3, ax3 = plt.subplots()
ax3.bar(algorithms, wins_minimax)
ax3.set_title('Resultados contra Minimax')
ax3.set_xlabel('Algoritmo')
ax3.set_ylabel('Victorias')

fig4, ax4 = plt.subplots()
ax4.bar(algorithms, win_rates_minimax)
ax4.set_title('Tasas de victoria contra Minimax')
ax4.set_xlabel('Algoritmo')
ax4.set_ylabel('Tasa de victoria')

plt.show()