import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def save_q_table(q, filename):
    with open(filename, 'w') as f:
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                f.write(f"state({i},{j}): {q[i, j]}\n")

def save_rewards(rewards, filename):
    with open(filename, 'w') as f:
        for episode, reward in enumerate(rewards):
            f.write(f"Episode {episode}: {reward}\n")

def run(episodes, is_training=True, render=False, epsilon=0.1):
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001 #tasa de decaimiento de Epsilon
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]

        terminated = False
        truncated = False
        rewards = 0

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Exploración: seleccionar una acción aleatoria
            else:
                action = np.argmax(q[state, :]) # Explotación: seleccionar la mejor acción conocida

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if is_training:
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)  #El valor de 𝜖 se reduce gradualmente después de cada episodio para disminuir la cantidad de exploración a lo largo del tiempo
        rewards_per_episode[i] = rewards

        if (i + 1) % 50 == 0:
            print(f'Episodio: {i + 1} - Recompensa: {rewards_per_episode[i]}')

    env.close()

    # Guardar la tabla Q final en un archivo de texto
    save_q_table(q, 'C:/Users/Manuel/Desktop/Septimo Semestre/Inteligencia Artificial/Laboratorios/Lab7/Juego_Taxi🕹️🚗/1-MetodoAccionporvalor/q_table.txt')

    # Calcular y mostrar la suma de recompensas acumuladas en bloques de 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig('C:/Users/Manuel/Desktop/Septimo Semestre/Inteligencia Artificial/Laboratorios/Lab7/Juego_Taxi🕹️🚗/1-MetodoAccionporvalor/taxi.png')
    plt.show()

    # Guardar la tabla Q y las recompensas si se está entrenando
    if is_training:
        with open("taxi.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(15000, is_training=True, render=False, epsilon=0.1)  # Primero entrena el modelo    is_training: Indicador de si el modelo está en modo entrenamiento.
    run(15, is_training=False, render=True, epsilon=0.1)  # Luego usa el modelo entrenado con renderización
