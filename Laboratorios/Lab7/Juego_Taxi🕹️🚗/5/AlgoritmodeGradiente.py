import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def save_preferences(pref, filename):
    with open(filename, 'w') as f:
        for i in range(pref.shape[0]):
            for j in range(pref.shape[1]):
                f.write(f"state({i},{j}): {pref[i, j]}\n")

def save_rewards(rewards, filename):
    with open(filename, 'w') as f:
        for episode, reward in enumerate(rewards):
            f.write(f"Episode {episode}: {reward}\n")

def softmax(preferences):
    exp_preferences = np.exp(preferences - np.max(preferences))
    return exp_preferences / exp_preferences.sum()

def run(episodes, is_training=True, render=False, alpha=0.01):
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        preferences = np.random.randn(env.observation_space.n, env.action_space.n) * 0.1
    else:
        with open(os.path.join(script_dir, 'taxi_prefs.pkl'), 'rb') as f:
            preferences = pickle.load(f)

    rewards_per_episode = np.zeros(episodes)
    baseline = 0  # Recompensa promedio inicial

    for i in range(episodes):
        state = env.reset()[0]

        terminated = False
        truncated = False
        rewards = 0
        t = 0  # Step counter

        while not terminated and not truncated:
            # Calculate action probabilities using softmax
            action_probabilities = softmax(preferences[state, :])
            action = np.random.choice(np.arange(env.action_space.n), p=action_probabilities)

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if is_training:
                t += 1
                # Update preferences
                avg_reward = rewards / t
                for a in range(env.action_space.n):
                    if a == action:
                        preferences[state, a] += alpha * (reward - baseline) * (1 - action_probabilities[a])
                    else:
                        preferences[state, a] -= alpha * (reward - baseline) * action_probabilities[a]

                baseline += (reward - baseline) / t  # Actualizar el baseline

            state = new_state

        rewards_per_episode[i] = rewards
        baseline = (baseline * i + rewards) / (i + 1)  # Actualizar el baseline global

        if (i + 1) % 50 == 0:
            print(f'Episodio: {i + 1} - Recompensa: {rewards_per_episode[i]}')

    env.close()

    # Guardar las preferencias finales en un archivo de texto
    save_preferences(preferences, os.path.join(script_dir, 'preferences.txt'))

    # Calcular y mostrar la suma de recompensas acumuladas en bloques de 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig(os.path.join(script_dir, 'taxi.png'))
    plt.show()

    if is_training:
        with open(os.path.join(script_dir, "taxi_prefs.pkl"), "wb") as f:
            pickle.dump(preferences, f)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Obtener el directorio del script actual
    run(15000, is_training=True, render=False)  # Primero entrena el modelo
    run(8, is_training=False, render=True)  # Luego usa el modelo entrenado con renderización
