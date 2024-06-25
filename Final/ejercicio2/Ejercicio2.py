import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random

# Definición del entorno del rompecabezas 4x5
class PuzzleEnv:
    def __init__(self):
        self.n_rows = 4
        self.n_cols = 5
        self.state = self.reset()
        self.action_space = 4  # 4 acciones posibles: arriba, abajo, izquierda, derecha
        self.observation_space = self.n_rows * self.n_cols  # Número total de estados

    def reset(self):
        # Estado inicial aleatorio
        self.state = random.randint(0, self.n_rows * self.n_cols - 1)
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.n_cols)
        if action == 0 and row > 0:  # Arriba
            row -= 1
        elif action == 1 and row < self.n_rows - 1:  # Abajo
            row += 1
        elif action == 2 and col > 0:  # Izquierda
            col -= 1
        elif action == 3 and col < self.n_cols - 1:  # Derecha
            col += 1
        new_state = row * self.n_cols + col
        reward = -1  # Penalización por movimiento
        if new_state == self.n_rows * self.n_cols - 1:  # Estado objetivo (última celda)
            reward = 0
            done = True
        else:
            done = False
        self.state = new_state
        return new_state, reward, done, {}

    def render(self):
        # Renderizar el entorno
        grid = np.arange(self.n_rows * self.n_cols).reshape(self.n_rows, self.n_cols)
        row, col = divmod(self.state, self.n_cols)
        grid[row, col] = -1  # Representar el agente con un -1
        print(grid)

# Función para guardar la tabla Q en un archivo de texto
def save_q_table(q, filename):
    with open(filename, 'w') as f:
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                f.write(f"state({i},{j}): {q[i, j]}\n")

# Función principal para ejecutar el Q-learning
def run(episodes, is_training=True, render=False):
    env = PuzzleEnv()

    # Directorio donde se guardará y cargará la tabla Q
    q_table_path = r'C:\Users\Manuel\Desktop\Septimo Semestre\Inteligencia Artificial\Final\puzzle_q_table.pkl'

    # Inicializar la tabla Q
    if is_training:
        q = np.zeros((env.observation_space, env.action_space))
    else:
        if os.path.exists(q_table_path):
            with open(q_table_path, 'rb') as f:
                q = pickle.load(f)
        else:
            print(f"Archivo '{q_table_path}' no encontrado. Por favor, entrena el modelo primero.")
            return

    learning_rate = 0.9  # Tasa de aprendizaje
    discount_factor = 0.9  # Factor de descuento
    epsilon = 1.0  # Probabilidad de exploración inicial
    epsilon_decay = 0.995  # Tasa de decaimiento de epsilon
    epsilon_min = 0.1  # Valor mínimo de epsilon

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()

        done = False
        rewards = 0

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = random.randint(0, env.action_space - 1)  # Exploración: acción aleatoria
            else:
                action = np.argmax(q[state, :])  # Explotación: acción con el valor Q más alto

            new_state, reward, done, _ = env.step(action)
            rewards += reward

            if is_training:
                # Actualización de Q usando la fórmula de Q-learning
                q[state, action] += learning_rate * (
                    reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

            if render:
                env.render()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)  # Reducir epsilon gradualmente
        rewards_per_episode[i] = rewards

        if (i + 1) % 50 == 0:
            print(f'Episodio: {i + 1} - Recompensa: {rewards_per_episode[i]}')

    # Guardar la tabla Q en un archivo de texto
    save_q_table(q, r'C:\Users\Manuel\Desktop\Septimo Semestre\Inteligencia Artificial\Final\q_table_puzzle.txt')

    # Guardar la tabla Q en un archivo pickle si se está entrenando
    if is_training:
        with open(q_table_path, 'wb') as f:
            pickle.dump(q, f)

    # Calcular y mostrar la suma de recompensas acumuladas en bloques de 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Media')
    plt.title('Recompensa Media por Episodio')
    plt.grid(True)
    plt.savefig(r'C:\Users\Manuel\Desktop\Septimo Semestre\Inteligencia Artificial\Final\puzzle.png')
    plt.show()

if __name__ == '__main__':
    run(15000, is_training=True, render=False)  # Primero entrena el modelo
    run(10, is_training=False, render=True)  # Luego usa el modelo entrenado con renderización

