import random
from pickletools import bytes4

import numpy as np


class SnakeField:
    def __init__(self, x_size=72, y_size=48):
        self.x_size = x_size
        self.y_size = y_size

        self.observation_space = self.x_size * self.y_size
        self.head_position, self.snake_body, self.fruit_positions = self.init_positions()
        self.field = None

        self.actions = ['LEFT', 'RIGHT', 'STRAIGHT']
        self.action_space = len(self.actions)

        self.directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.direction = 0

        self.spawn_time_limit = 25
        self.fruit_timer = 0

    def init_positions(self):
        head_position = self.y_size // 2 * self.x_size + self.x_size // 2
        snake_body = [head_position + 1, head_position + 2, head_position + 3]

        fruit_positions = [random.randrange(0, self.x_size * self.y_size)]
        return head_position, snake_body, fruit_positions

    def get_state(self, terminal=False):
        self.field = np.zeros(self.observation_space, dtype=np.int8)

        for fruit in self.fruit_positions:
            self.field[fruit] = 1

        for part in self.snake_body:
            self.field[part] = 2
        if not terminal:
            self.field[self.head_position] = 3

        return self.field

    def reset(self):
        self.head_position, self.snake_body, self.fruit_positions = self.init_positions()
        self.fruit_timer = 0
        return self.get_state()

    def spawn_fruit(self):
        pos = random.randrange(0, self.x_size * self.y_size)
        self.fruit_positions.append(pos)
        self.fruit_timer = 0

    def step(self, action):
        if self.actions[action] == 'LEFT':
            self.direction = (self.direction - 1) % len(self.directions)
        elif self.actions[action] == 'RIGHT':
            self.direction = (self.direction + 1) % len(self.directions)

        self.snake_body.insert(0, self.head_position)

        terminal = False
        reword = 0

        match self.directions[self.direction]:
            case 'UP':
                self.head_position -= self.x_size
                terminal = self.head_position < 0
            case 'RIGHT':
                self.head_position += 1
                terminal = self.head_position % self.x_size == 0
            case 'DOWN':
                self.head_position += self.x_size
                terminal = self.head_position >= self.observation_space
            case 'LEFT':
                terminal = self.head_position % self.x_size == 0
                self.head_position -= 1

        if terminal:
            return self.get_state(terminal), reword, terminal, []

        if self.head_position in self.fruit_positions:
            reword = 1
            self.fruit_positions.remove(self.head_position)
        else:
            self.snake_body.pop()

        terminal = self.head_position in self.snake_body

        if self.spawn_time_limit <= self.fruit_timer and not terminal:
            self.spawn_fruit()

        return self.get_state(terminal), reword, terminal, []
