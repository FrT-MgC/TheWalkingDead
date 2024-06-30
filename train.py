import numpy as np
import random
import pickle
import sys
from environment_generator import GridGenerate

training_quantity = 10000
size = 10
# Utilizar o último grid gerado
load_last_grid = True
# Fazer o treino ou utilizar o arquivo
train = True

def save_q_table(q_table, filename='training_robot.pkl'):
    """Salva a Q-table em um arquivo"""
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)

def load_q_table(filename='training_robot.pkl'):
    """Carrega a Q-table de um arquivo"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Cria o ambiente
env = GridGenerate(size, load_last_grid)  # Define load_last_grid=True para usar o último grid salvo

# Inicializa a Q-table
q_table_file = 'training_robot.pkl'
if train:
    q_table = {}  # Usa um dicionário esparso para a Q-table durante o treinamento
else:
    q_table = load_q_table(q_table_file)
    if q_table is None:
        print("Erro: Arquivo de treinamento não encontrado. Execute o treinamento primeiro.")
        sys.exit()

if train:
    # Define hiperparâmetros
    num_episodes = training_quantity
    max_steps_per_episode = 100
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay_rate = 0.001

    # Define a política epsilon-greedy
    def epsilon_greedy_policy(state, collected_supplies):
        # Converte os suprimentos coletados para um índice binário
        supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
        state_action = (state, supply_index)
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)
        else:
            if state_action in q_table:
                return np.argmax(q_table[state_action])
            else:
                return random.randint(0, 3)

    # Treina o agente
    for episode in range(num_episodes):
        state, collected_supplies = env.reset()
        done = False
        t = 0

        while not done and t < max_steps_per_episode:
            # Seleciona uma ação utilizando a política epsilon-greedy
            action = epsilon_greedy_policy(state, collected_supplies)
            
            # Executa a ação no ambiente
            next_state, next_collected_supplies, reward, done, reason = env.step(action)
            
            # Converte os estados de suprimentos coletados para índices
            supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
            next_supply_index = int(''.join(['1' if (i, j) in next_collected_supplies else '0' for (i, j) in env.supply_states]), 2)
            
            state_action = (state, supply_index)
            next_state_action = (next_state, next_supply_index)
            
            if state_action not in q_table:
                q_table[state_action] = np.zeros(4)  # Inicializa a entrada na Q-table se não existir
            if next_state_action not in q_table:
                q_table[next_state_action] = np.zeros(4)  # Inicializa a entrada na Q-table se não existir
            
            # Atualiza a Q-table
            q_table[state_action][action] += learning_rate * (
                reward + discount_factor * np.max(q_table[next_state_action]) - 
                q_table[state_action][action]
            )
            
            # Atualiza o estado atual e os suprimentos coletados
            state, collected_supplies = next_state, next_collected_supplies
            t += 1

        # Decaimento do epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay_rate))

        # Mostra progresso a cada 1000 episódios
        if episode % 1000 == 0:
            print(f'Episode: {episode}')

    # Salva a Q-table treinada
    save_q_table(q_table, q_table_file)

# Inicializa o Pygame após o treinamento
import pygame
pygame.init()
cell_size = 60
screen = pygame.display.set_mode((env.size * cell_size, env.size * cell_size))
pygame.display.set_caption('The Walking Dead')

# Testa o agente com a Q-table carregada
state, collected_supplies = env.reset()
done = False
move_count = 0

if not train:
    print('Executando com o arquivo de treinamento')

while not done:
    # Converte os suprimentos coletados para um índice binário
    supply_index = int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in env.supply_states]), 2)
    state_action = (state, supply_index)
    
    # Seleciona a melhor ação com base na Q-table
    if state_action in q_table:
        action = np.argmax(q_table[state_action])
    else:
        action = random.randint(0, 3)
    
    # Executa a ação e obtém o novo estado e informações adicionais
    next_state, next_collected_supplies, reward, done, reason = env.step(action)
    
    # Renderiza o ambiente na tela
    env.render(screen, cell_size)
    
    # Aguarda meio segundo antes da próxima ação
    pygame.time.wait(350)
    
    # Atualiza o estado atual e os suprimentos coletados
    state, collected_supplies = next_state, next_collected_supplies
    move_count += 1

# Mostra o estado final e a razão da finalização
print(f"Estado Final: {state}")
print(f"Presentes Coletados: {len(collected_supplies)} de {len(env.supply_states)}")
print(f"Movimentos Necessários: {move_count}")
print(f"Status: {reason}")

# Fecha a janela do Pygame corretamente
pygame.quit()
sys.exit()
