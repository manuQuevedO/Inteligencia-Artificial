import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Función para guardar la tabla Q en un archivo de texto
def save_q_table(q, filename):
    with open(filename, 'w') as f:
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                f.write(f"state({i},{j}): {q[i, j]}\n")

# Función para guardar las recompensas en un archivo de texto
def save_rewards(rewards, filename):
    with open(filename, 'w') as f:
        for episode, reward in enumerate(rewards):
            f.write(f"Episode {episode}: {reward}\n")

# Función principal para ejecutar el juego
def run(episodes, is_training=True, render=False):
    # Crear el entorno del juego Taxi
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    # Inicializar la tabla Q con valores optimistas o cargar una tabla Q entrenada
    if is_training:
        q = np.full((env.observation_space.n, env.action_space.n), 10.0)  # Valores iniciales optimistas
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)  # Cargar la tabla Q previamente entrenada

    learning_rate_a = 0.9  # Tasa de aprendizaje
    discount_factor_g = 0.9  # Factor de descuento

    rewards_per_episode = np.zeros(episodes)  # Arreglo para almacenar las recompensas por episodio

    for i in range(episodes):
        state = env.reset()[0]  # Reiniciar el entorno y obtener el estado inicial

        terminated = False  # Indica si el episodio ha terminado
        truncated = False  # Indica si el episodio ha sido truncado
        rewards = 0  # Recompensa acumulada

        while not terminated and not truncated:
            action = np.argmax(q[state, :])  # Seleccionar la acción con el valor Q más alto

            new_state, reward, terminated, truncated, _ = env.step(action)  # Ejecutar la acción
            rewards += reward  # Acumular la recompensa

            if is_training:
                # Actualización de Q usando la fórmula de acción-valor
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state  # Actualizar el estado actual

        rewards_per_episode[i] = rewards  # Almacenar la recompensa del episodio

        if (i + 1) % 50 == 0:
            print(f'Episodio: {i + 1} - Recompensa: {rewards_per_episode[i]}')

    env.close()

    # Guardar la tabla Q final en un archivo de texto
    save_q_table(q, os.path.join(script_dir, 'q_table.txt'))

    # Calcular y mostrar la suma de recompensas acumuladas en bloques de 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig(os.path.join(script_dir, 'taxi.png'))
    plt.show()

    # Guardar la tabla Q y las recompensas si se está entrenando
    if is_training:
        with open(os.path.join(script_dir, "taxi.pkl"), "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Obtener el directorio del script actual
    run(20000, is_training=True, render=False)  # Primero entrena el modelo
    run(7, is_training=False, render=True)  # Luego usa el modelo entrenado con renderización
