import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def save_q_table(q, filename):
    with open(filename, 'w') as f:
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                f.write(f"state({i},{j}): {q[i, j, :]}\n")

def save_rewards(rewards, filename):
    with open(filename, 'w') as f:
        for episode, reward in enumerate(rewards):
            f.write(f"Episode {episode}: {reward}\n")

def run(episodes, is_training=True, render=False):
    # Crear el entorno
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Dividir la posición y la velocidad en segmentos
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    # Inicializar o cargar la Q-table
    if is_training:
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        with open('mountain_car.pkl', 'rb') as f:
            q = pickle.load(f)

    # Parámetros de Q-Learning
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 2 / episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):    #comienzo de cada episodio
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        terminated = False
        rewards = 0
        

        while not terminated and rewards > -1000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = rewards

    env.close()

    # Guardar la Q-table generada del entrenamiento
    if is_training:
        with open('mountain_car.pkl', 'wb') as f:
            pickle.dump(q, f)
        # Guardar la Q-table en un archivo de texto
        save_q_table(q, 'q_table.txt')
        # Guardar las recompensas por episodio en un archivo de texto
        save_rewards(rewards_per_episode, 'rewards_per_episode.txt')
        print("Recompensas por episodio guardadas en rewards_per_episode.txt")

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car.png')

if __name__ == '__main__':
    run(4000, is_training=True, render=False)  # Primero entrena el modelo
    run(3, is_training=False, render=True)  # Luego usa el modelo entrenado
