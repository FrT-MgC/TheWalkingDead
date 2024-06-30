import numpy as np
import random
import pickle
import pygame

class GridGenerate:
    def __init__(self, size, load_last_grid):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_state = (0, 0)
        self.goal_state = (size-1, size-1)
        
        if load_last_grid and self._load_grid():
            self.zombie_states, self.supply_states, self.obstacle_states = self._load_grid()
        else:
            self._generate_random_grid()
            self._save_grid()

        self.supplies_collected = set()
        
        # Atualiza o grid com zumbis, suprimentos e obstáculos
        for i, j in self.zombie_states:
            self.grid[i][j] = 1
        for i, j in self.supply_states:
            self.grid[i][j] = 2
        for i, j in self.obstacle_states:
            self.grid[i][j] = 3

    def _generate_random_grid(self):
        #Gera posições aleatórias para zumbis, suprimentos e obstáculos
        num_zombies = random.randint(1, self.size * self.size // 4)
        num_supplies = random.randint(1, self.size * self.size // 4)
        num_obstacles = random.randint(1, self.size * self.size // 8)

        self.zombie_states = self._place_random_items(num_zombies)
        self.supply_states = self._place_random_items(num_supplies, exclude=self.zombie_states)
        self.obstacle_states = self._place_random_items(num_obstacles, exclude=self.zombie_states + self.supply_states)

    def _place_random_items(self, num_items, exclude=[]):
        #Coloca itens aleatórios no grid, excluindo certas posições
        items = []
        while len(items) < num_items:
            i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (i, j) not in items and (i, j) not in exclude and (i, j) != self.start_state and (i, j) != self.goal_state:
                items.append((i, j))
        return items

    def _save_grid(self):
        #Salva o grid gerado em um arquivo
        grid_data = {
            'zombie_states': self.zombie_states,
            'supply_states': self.supply_states,
            'obstacle_states': self.obstacle_states
        }
        with open('last_grid.pkl', 'wb') as f:
            pickle.dump(grid_data, f)

    def _load_grid(self):
        #Carrega o grid de um arquivo
        try:
            with open('last_grid.pkl', 'rb') as f:
                grid_data = pickle.load(f)
                return grid_data['zombie_states'], grid_data['supply_states'], grid_data['obstacle_states']
        except FileNotFoundError:
            return None

    def reset(self):
        #Reseta o estado atual e os suprimentos coletados
        self.current_state = self.start_state
        self.supplies_collected = set()
        return self.current_state, tuple(self.supplies_collected)

    def step(self, action):
        #Executa uma ação e retorna o novo estado, suprimentos coletados, recompensa, se terminou e a razão
        i, j = self.current_state
        if action == 0:  # move up
            i = max(i - 1, 0)
        elif action == 1:  # move down
            i = min(i + 1, self.size - 1)
        elif action == 2:  # move left
            j = max(j - 1, 0)
        elif action == 3:  # move right
            j = min(j + 1, self.size - 1)
        
        # Verifica se a movimentação é válida (não em um obstáculo)
        if (i, j) in self.obstacle_states:
            i, j = self.current_state
        
        self.current_state = (i, j)
        
        # Avalia as condições de término e recompensa
        if self.current_state == self.goal_state:
            if len(self.supplies_collected) == len(self.supply_states):
                reward, done, reason = 1, True, "Chegou ao objetivo apos coletar todos os presentes"
            else:
                reward, done, reason = -0.1, True, "Chegou ao objetivo sem coletar todos os presentes"
        elif self.current_state in self.zombie_states:
            reward, done, reason = -0.5, True, "Encontrou um zumbi"
        elif self.current_state in self.supply_states and self.current_state not in self.supplies_collected:
            self.supplies_collected.add(self.current_state)
            reward, done, reason = 0.5, False, ""
        else:
            reward, done, reason = -0.01, False, ""
        
        return self.current_state, tuple(self.supplies_collected), reward, done, reason

    def render(self, screen, cell_size=60):
        #Renderiza o grid utilizando pygame
        colors = {
            "background": (80, 167, 56),
            "agent": (30, 146, 212),
            "goal": (0, 255, 35),
            "zombie": (184, 124, 18),
            "supply": (190, 108, 200),
            "obstacle": (128, 139, 150),
            "empty": (138, 197, 131)
        }

        screen.fill(colors["background"])

        for i in range(self.size):
            for j in range(self.size):
                color = colors["empty"]
                if (i, j) == self.current_state:
                    color = colors["agent"]
                elif (i, j) == self.goal_state:
                    color = colors["goal"]
                elif self.grid[i][j] == 1:
                    color = colors["zombie"]
                elif self.grid[i][j] == 2:
                    if (i, j) not in self.supplies_collected:
                        color = colors["supply"]
                elif self.grid[i][j] == 3:
                    color = colors["obstacle"]
                pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size))

        pygame.display.flip()
