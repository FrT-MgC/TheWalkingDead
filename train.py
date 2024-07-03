import numpy as np
import random
import pickle
import sys
from environment_generator import GridGenerate

training_quantity = 10000
size = 15
#carregar ultimo grid
load_last_grid = False
#executar o treinamento
train = True

def save_q_table(q_table, filename='training_robot.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)

def load_q_table(filename='training_robot.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

env = GridGenerate(size, load_last_grid)

q_table_file = 'training_robot.pkl'
if train:
    q_table = {}
else:
    q_table = load_q_table(q_table_file)
    if q_table is None:
        print("Erro: Arquivo de treinamento nao encontrado. Execute o treinamento primeiro.")
        sys.exit()

if train:
    num_episodes = training_quantity
    max_steps_per_episode = 100
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay_rate = 0.001

    def epsilon_greedy_policy(state, collected_supplies):
        supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
        state_action = (state, supply_index)
        if random.uniform(0, 1) < epsilon:
            #exploration
            return random.randint(0, 3)
        else:
            #explotation
            if state_action in q_table:
                return np.argmax(q_table[state_action])
            else:
                return random.randint(0, 3)
        

    # executa o treinamento
    for episode in range(num_episodes):
        state, collected_supplies = env.reset()
        done = False
        t = 0

        while not done and t < max_steps_per_episode:
            action = epsilon_greedy_policy(state, collected_supplies)
            
            next_state, next_collected_supplies, reward, done, reason = env.step(action)
            
            supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
            next_supply_index = int(''.join(['1' if (i, j) in next_collected_supplies else '0' for (i, j) in env.supply_states]), 2)
            
            state_action = (state, supply_index)
            next_state_action = (next_state, next_supply_index)
            
            if state_action not in q_table:
                q_table[state_action] = np.zeros(4)
            if next_state_action not in q_table:
                q_table[next_state_action] = np.zeros(4)
            
            q_table[state_action][action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state_action]) - 
                q_table[state_action][action]
            )
            
            state, collected_supplies = next_state, next_collected_supplies
            t += 1
            
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay_rate))

        if episode % 1000 == 0:
            print(f'Treinando: {episode}')

    save_q_table(q_table, q_table_file)


import pygame
pygame.init()

if size <= 12:
    cell_size = 60
else:
    cell_size = 100 - (size * 4)
    if cell_size <= 0:
       cell_size = 20

screen = pygame.display.set_mode((env.size * cell_size, env.size * cell_size))
pygame.display.set_caption('The Walking Dead')

state, collected_supplies = env.reset()
done = False
move_count = 0

if not train:
    print('Executando com o arquivo de treinamento')

while not done:
    supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
    state_action = (state, supply_index)
    
    if state_action in q_table:
        action = np.argmax(q_table[state_action])
    else:
        action = random.randint(0, 3)
    
    next_state, next_collected_supplies, reward, done, reason = env.step(action)

    env.render(screen, cell_size)
    pygame.time.wait(350)
    
    state, collected_supplies = next_state, next_collected_supplies
    move_count += 1

print(f"Estado Final: {state}")
print(f"Presentes Coletados: {len(collected_supplies)} de {len(env.supply_states)}")
print(f"Movimentos Necessarios: {move_count}")
print(f"Status: {reason}")

pygame.quit()
sys.exit()
