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

def run(episodes, is_training=True, render=False, c=1.0):
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
        visit_count = np.zeros_like(q)  # Contador de visitas para UCB
    else:
        with open(os.path.join(script_dir, 'taxi.pkl'), 'rb') as f:
            q = pickle.load(f)
        visit_count = np.zeros_like(q)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    c = 2.0                      # Parámetro de exploración UCB
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]

        terminated = False
        truncated = False
        rewards = 0

        while not terminated and not truncated:
            if is_training:
                total_visits = np.sum(visit_count[state, :]) + 1  # +1 para evitar división por cero
                ucb_values = q[state, :] + c * np.sqrt(np.log(total_visits) / (visit_count[state, :] + 1)) # formula de UCB
                action = np.argmax(ucb_values)
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if is_training:                             # Actualizar la tabla Q si se está entrenando
                visit_count[state, action] += 1         # Incrementar el contador de visitas
                alpha = 1 / visit_count[state, action]  # Tasa de aprendizaje incremental

                # Actualización de Q usando la fórmula de acción-valor incremental
                q[state, action] += alpha * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        rewards_per_episode[i] = rewards

        if (i + 1) % 50 == 0:
            print(f'Episodio: {i + 1} - Recompensa: {rewards_per_episode[i]}')

    env.close()

    # Guardar la tabla Q final en un archivo de texto
    save_q_table(q, os.path.join(script_dir, 'q_table.txt'))

    if is_training:
        with open(os.path.join(script_dir, "taxi.pkl"), "wb") as f:
            pickle.dump(q, f)


    # Calcular y mostrar la suma de recompensas acumuladas en bloques de 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig(os.path.join(script_dir, 'taxi.png'))
    plt.show()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Obtener el directorio del script actual
    run(15000, is_training=True, render=False, c=2.0)  # Primero entrena el modelo
    run(8, is_training=False, render=True)  # Luego usa el modelo entrenado con renderización
