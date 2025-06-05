import pygame

from agent.Lilith import Lilith
from environment import SnakeField


x_size = 9
y_size = 9
grid_step = 45

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

environment = SnakeField(x_size, y_size)
lilith = Lilith()
lilith.train(environment)
history = lilith.demonstration(environment)

pygame.init()

pygame.display.set_caption('Lilith game')
game_window = pygame.display.set_mode((x_size * grid_step, y_size * grid_step))

fps = pygame.time.Clock()
snake_speed = 10


def show_score(score, step, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score: ' + str(score) + '   Step: ' + str(step), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)


def game_over(score):
    my_font = pygame.font.SysFont('times new roman', 50)
    game_over_surface = my_font.render(
        'Your Score is : ' + str(score), True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (x_size * grid_step / 2, y_size * grid_step / 4)
    game_window.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    pygame.quit()
    quit()


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
                    draw_field(state)
                    show_score(score, moment, white, 'times new roman', 20)
                else:
                    game_over(score)
    pygame.display.update()
    fps.tick(snake_speed)
