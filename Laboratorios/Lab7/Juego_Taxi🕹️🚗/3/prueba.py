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

# Función principal para ejecutar el juego
def run(episodes, is_training=True, render=False, epsilon=1):
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        # Inicializar la tabla Q con valores optimistas (por ejemplo, 10.0)
        q = np.full((env.observation_space.n, env.action_space.n), 10.0)
    else:
        with open(os.path.join(script_dir, 'taxi.pkl'), 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    states_visited = np.zeros(env.observation_space.n)  # Seguimiento de estados visitados

    for i in range(episodes):
        state = env.reset()[0]

        terminated = False
        truncated = False
        rewards = 0

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
            states_visited[state] += 1  # Incrementar el contador de visitas al estado actual

            if is_training:
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = rewards

        if (i + 1) % 50 == 0:
            print(f'Episodio: {i + 1} - Recompensa: {rewards_per_episode[i]}')

    env.close()

    # Guardar la tabla Q final en un archivo de texto
    save_q_table(q, os.path.join(script_dir, 'q_table.txt'))

    # Guardar la lista de estados visitados
    with open(os.path.join(script_dir, 'states_visited.txt'), 'w') as f:
        for state, count in enumerate(states_visited):
            f.write(f"State {state}: {count} visits\n")

    if is_training:
        with open(os.path.join(script_dir, "taxi.pkl"), "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Obtener el directorio del script actual
    run(15000, is_training=True, render=False, epsilon=1)  # Primero entrena el modelo
    run(15, is_training=False, render=True, epsilon=0.1)  # Luego usa el modelo entrenado con renderización
