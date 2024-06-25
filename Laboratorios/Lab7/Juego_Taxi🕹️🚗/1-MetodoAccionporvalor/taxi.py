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
def run(episodes, is_training=True, render=False, epsilon=0.1):
    # Crear el entorno del juego Taxi
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    # Inicializar la tabla Q
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # Tabla Q inicializada en ceros
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)  # Cargar la tabla Q previamente entrenada

    # Parámetros de aprendizaje
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1                 # Probabilidad de exploración inicial
    epsilon_decay_rate = 0.0001 # Tasa de decaimiento de Epsilon
    rng = np.random.default_rng()

    # Arreglo para almacenar las recompensas por episodio
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # Reiniciar el entorno y obtener el estado inicial

        terminated = False  # Indica si el episodio ha terminado
        truncated = False  # Indica si el episodio ha sido truncado
        rewards = 0  # Recompensa acumulada

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:  # Número aleatorio entre 0 y 1
                action = env.action_space.sample()  # Exploración: seleccionar una acción aleatoria
            else:
                action = np.argmax(q[state, :])     # Explotación: seleccionar la mejor acción conocida

            new_state, reward, terminated, truncated, _ = env.step(action)  # Ejecutar la acción
            rewards += reward  # Acumular la recompensa

            if is_training:
                # Actualización de la tabla Q usando la fórmula de Q-learning
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state  # Actualizar el estado actual

        # Reducir gradualmente epsilon para disminuir la exploración con el tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)
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
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Media')
    plt.title('Recompensa Media por Episodio')
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, 'taxi.png'))
    plt.show()

   

    # Guardar la tabla Q y las recompensas si se está entrenando
    if is_training:
        with open("taxi.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    # Obtener el directorio del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Primero entrena el modelo
    run(15000, is_training=True, render=False)
    
    # Luego usa el modelo entrenado con renderización
    run(6, is_training=False, render=True)