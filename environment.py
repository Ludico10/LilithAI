import random

import numpy as np


class SnakeField:
    def __init__(self, x_size=72, y_size=48):
        self.x_size = x_size
        self.y_size = y_size
        self.observation_space = self.x_size * self.y_size * 2
        self.head_position, self.snake_body, self.fruit_positions = self.init_positions()
        self.field = None

        self.actions = ['LEFT', 'RIGHT', 'STRAIGHT']
        self.action_space = len(self.actions)

        self.directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.direction = 3

        self.spawn_time_limit = 1000
        self.fruit_timer = 0

    def init_positions(self):
        head_position = (self.y_size * self.x_size) // 2
        snake_body = [head_position + 1, head_position + 2, head_position + 3, head_position + 4]

        fruit_positions = [random.randrange(0, self.x_size * self.y_size)]
        return head_position, snake_body, fruit_positions

    def get_state(self, border_collision=False):
        old_field = self.field.copy()
        self.field = np.zeros(self.x_size * self.y_size, dtype=np.int8)

        for fruit in self.fruit_positions:
            self.field[fruit] = 3

        for part in self.snake_body:
            self.field[part] = 2
        if not border_collision:
            self.field[self.head_position] = 1

        double_field = np.concatenate([self.field, old_field], dtype=np.int8)
        return np.reshape(double_field, (self.x_size, self.y_size, 2))

    def reset(self):
        self.head_position, self.snake_body, self.fruit_positions = self.init_positions()
        self.field = np.zeros(self.x_size * self.y_size, dtype=np.int8)
        self.fruit_timer = 0
        self.direction = 3
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

        match self.directions[self.direction]:
            case 'UP':
                self.head_position = (self.head_position - self.x_size) % (self.x_size * self.y_size)
            case 'RIGHT':
                self.head_position = self.head_position - self.head_position % self.x_size + \
                                     (self.head_position + 1) % self.x_size
            case 'DOWN':
                self.head_position = (self.head_position + self.x_size) % (self.x_size * self.y_size)
            case 'LEFT':
                self.head_position = self.head_position - self.head_position % self.x_size + \
                                     (self.head_position - 1) % self.x_size

        reword = 0
        if self.head_position in self.fruit_positions:
            reword = 10
            self.fruit_positions.remove(self.head_position)
            self.spawn_fruit()
        else:
            self.snake_body.pop()

        terminal = False
        if self.head_position not in self.snake_body:
            self.fruit_timer += 1
            if self.spawn_time_limit <= self.fruit_timer:
                self.spawn_fruit()
        else:
            terminal = True
            reword = -1

        return self.get_state(), reword, terminal, []
