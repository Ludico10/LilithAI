import pygame
import sys

import tensorflow as tf

from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses

from agent.DuelingDDQN import DuelingDDQN
from environment import SnakeField


print("Доступные устройства:")
print(tf.config.list_physical_devices('GPU'))

x_size = 10
y_size = 10
grid_step = 45

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

environment = SnakeField(x_size, y_size)

lilith = DuelingDDQN(environment.action_space)
lilith.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=3e-2), loss_fn=losses.MeanSquaredError())
lilith.fit(environment)

history, _ = lilith.predict(environment)
buttons = []

pygame.init()

pygame.display.set_caption('Lilith game')
game_window = pygame.display.set_mode((x_size * grid_step, y_size * grid_step))

fps = pygame.time.Clock()
snake_speed = 10


def show_score(score_, step, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score: ' + str(score_) + '   Step: ' + str(step), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)


def game_over(score_):
    score_font = pygame.font.SysFont('times new roman', 50)
    game_over_surface = score_font.render(
        'Your Score is : ' + str(score_), True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (x_size * grid_step / 2, y_size * grid_step / 4)
    game_window.blit(game_over_surface, game_over_rect)

    btn_font = pygame.font.SysFont(None, 20)
    restart_btn_rect = pygame.Rect(x_size * grid_step / 8, y_size * grid_step / 4 * 3, grid_step * 2, grid_step)
    btn_text = btn_font.render("Снова", True, white)
    pygame.draw.rect(game_window, blue, restart_btn_rect)
    text_rect = btn_text.get_rect(center=restart_btn_rect.center)
    game_window.blit(btn_text, text_rect)
    buttons.append({'rect': restart_btn_rect, 'action': lambda: restart()})

    exit_btn_rect = pygame.Rect(x_size * grid_step / 8 * 5, y_size * grid_step / 4 * 3, grid_step * 2, grid_step)
    btn_text = btn_font.render("Выход", True, white)
    pygame.draw.rect(game_window, red, exit_btn_rect)
    text_rect = btn_text.get_rect(center=exit_btn_rect.center)
    game_window.blit(btn_text, text_rect)
    buttons.append({'rect': exit_btn_rect, 'action': lambda: game_exit()})

    pygame.display.flip()


def game_exit():
    pygame.quit()
    sys.exit()


def restart():
    h = lilith.predict(environment)
    buttons.clear()
    return h


def draw_field(field_state):
    game_window.fill(black)
    colors_list = [black, white, green, red]
    for pos in range(len(field_state)):
        if pos > 0:
            pygame.draw.rect(game_window, colors_list[field_state[pos]], pygame.Rect(pos % x_size * grid_step,
                                                                                     pos // x_size * grid_step,
                                                                                     grid_step, grid_step))


score = 0
moment = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if moment < len(history):
                    state, score = history[moment]
                    moment += 1
                    draw_field(state[0])
                    show_score(score, moment, white, 'times new roman', 20)
                else:
                    game_over(score)

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            for btn in buttons:
                if btn['rect'].collidepoint(mouse_pos):
                    score, moment = 0, 0
                    history = btn['action']()
                    game_window.fill(black)
                    break

    pygame.display.update()
    fps.tick(snake_speed)
