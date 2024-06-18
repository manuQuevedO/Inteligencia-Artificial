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

def run(episodes, is_training=True, render=False, epsilon=0.1):
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
        visit_count = np.zeros_like(q)  # Contador de visitas para la implementación incremental
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)
        visit_count = np.zeros_like(q)

    discount_factor_g = 0.9
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]

        terminated = False
        truncated = False
        rewards = 0

        while not terminated and not truncated:
            if is_training:
                # Selección de acción utilizando epsilon-greedy
                if rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state, :])
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if is_training:
                visit_count[state, action] += 1  # Incrementar el contador de visitas
                alpha = 1 / visit_count[state, action]  # Tasa de aprendizaje incremental

                # Actualización de Q usando la fórmula de acción-valor incremental
                q[state, action] += alpha * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - 0.00001, 0)  # Reduce epsilon más lentamente
        rewards_per_episode[i] = rewards

        if (i + 1) % 50 == 0:
            print(f'Episodio: {i + 1} - Recompensa: {rewards_per_episode[i]}')

        # Guardar progreso periódicamente
        if (i + 1) % 1000 == 0:
            with open(os.path.join(script_dir, f"taxi_{i+1}.pkl"), "wb") as f:
                pickle.dump(q, f)

        # Renderizar periódicamente para reducir la carga
        if (i % 1000 == 0) and render:
            env.render()

    env.close()

    # Guardar la tabla Q final en un archivo de texto
    save_q_table(q, os.path.join(script_dir, 'q_table.txt'))

    if is_training:
        with open(os.path.join(script_dir, "taxi.pkl"), "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Obtener el directorio del script actual
    run(15000, is_training=True, render=False, epsilon=0.1)  # Primero entrena el modelo
    run(6, is_training=False, render=True, epsilon=0.1)  # Luego usa el modelo entrenado con renderización
