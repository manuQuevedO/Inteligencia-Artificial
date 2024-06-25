import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Es una forma de actualizar lo que el taxi sabe sobre el mundo (sus valores Q) de manera gradual, 
# cada vez que toma una acción y recibe una recompensa.

# Función para guardar la tabla Q en un archivo de texto
def save_q_table(q, filename):
    with open(filename, 'w') as f:
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                f.write(f"state({i},{j}): {q[i, j]}\n")

# Función principal para ejecutar el juego
def run(episodes, is_training=True, render=False, epsilon=1):
    # Crear el entorno del juego Taxi
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    # Inicializar la tabla Q y el contador de visitas
    if is_training:        
        q = np.zeros((env.observation_space.n, env.action_space.n))  # Tabla Q inicializada en ceros
        visit_count = np.zeros_like(q)  # Contador de visitas para la implementación incremental
    else:   
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)  # Cargar la tabla Q previamente entrenada
        visit_count = np.zeros_like(q)

    discount_factor_g = 0.9  # Factor de descuento
    rng = np.random.default_rng()  # Generador de números aleatorios

    rewards_per_episode = np.zeros(episodes)  # Arreglo para almacenar las recompensas por episodio

    for i in range(episodes):
        state = env.reset()[0]  # Reiniciar el entorno y obtener el estado inicial

        terminated = False  # Indica si el episodio ha terminado
        truncated = False  # Indica si el episodio ha sido truncado
        rewards = 0  # Recompensa acumulada

        while not terminated and not truncated:
            if is_training:
                # Selección de acción utilizando epsilon-greedy
                if rng.random() < epsilon:  # Número aleatorio entre 0 y 1
                    action = env.action_space.sample()  # Exploración: seleccionar una acción aleatoria
                else:
                    action = np.argmax(q[state, :])  # Explotación: seleccionar la mejor acción conocida
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)  # Ejecutar la acción
            rewards += reward  # Acumular la recompensa

            if is_training:
                visit_count[state, action] += 1  # Incrementar el contador de visitas
                alpha = 1 / visit_count[state, action]  # Tasa de aprendizaje incremental

                # Actualización de Q usando la fórmula de acción-valor incremental
                q[state, action] += alpha * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state  # Actualizar el estado actual

        epsilon = max(epsilon - 0.00001, 0)  # Reducir epsilon más lentamente
        rewards_per_episode[i] = rewards  # Almacenar la recompensa del episodio

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

    # Graficar las recompensas medias
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Media')
    plt.title('Recompensa Media por Episodio')
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, 'taxi.png'))
    plt.show()

if __name__ == '__main__':
    # Obtener el directorio del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Primero entrena el modelo
    run(30000, is_training=True, render=False)
    
    # Luego usa el modelo entrenado con renderización
    run(6, is_training=False, render=True)
