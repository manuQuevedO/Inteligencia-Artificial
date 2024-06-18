import gym
import warnings
import time
import numpy as np

# Suprimir la advertencia de depreciación
warnings.filterwarnings("ignore", category=DeprecationWarning, module='gym.utils.passive_env_checker')

# Crear el entorno con render_mode especificado
env = gym.make('CartPole-v1', render_mode="human")

# Reiniciar el entorno y obtener el estado inicial
state = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()  # Acción aleatoria
    
    # Ejecutar la acción en el entorno y obtener los resultados
    result = env.step(action)
    
    if len(result) == 4:
        next_state, reward, done, info = result
    else:
        next_state, reward, done, truncated, info = result
        done = done or truncated

    # Actualizar el estado
    state = next_state
    
    # Añadir un pequeño retardo
    time.sleep(0.01)

# Cerrar el entorno
env.close()