# Task 1: 2048_game.py - A Python module for the 2048 game that humans can play using Pygame.
# This includes the game logic in a class and a function for human play.

import pygame
import random

class Game2048:
    def __init__(self):
        self.grid = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.add_tile()
        self.add_tile()

    def add_tile(self):
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.grid[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = random.choice([2, 4])

    def compress(self, row):
        return [x for x in row if x != 0] + [0] * (4 - len([x for x in row if x != 0]))

    def merge(self, row):
        for i in range(3):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                self.score += row[i]
                row[i + 1] = 0
        return row

    def move_left(self):
        moved = False
        for i in range(4):
            old_row = self.grid[i][:]
            self.grid[i] = self.compress(self.grid[i])
            self.grid[i] = self.merge(self.grid[i])
            self.grid[i] = self.compress(self.grid[i])
            if old_row != self.grid[i]:
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(4):
            old_row = self.grid[i][:]
            self.grid[i] = list(reversed(self.grid[i]))
            self.grid[i] = self.compress(self.grid[i])
            self.grid[i] = self.merge(self.grid[i])
            self.grid[i] = self.compress(self.grid[i])
            self.grid[i] = list(reversed(self.grid[i]))
            if old_row != self.grid[i]:
                moved = True
        return moved

    def transpose(self):
        self.grid = [list(row) for row in zip(*self.grid)]

    def move_up(self):
        self.transpose()
        moved = self.move_left()
        self.transpose()
        return moved

    def move_down(self):
        self.transpose()
        moved = self.move_right()
        self.transpose()
        return moved

    def is_game_over(self):
        if any(0 in row for row in self.grid):
            return False
        for i in range(4):
            for j in range(3):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return False
                if self.grid[j][i] == self.grid[j + 1][i]:
                    return False
        return True

def play_human():
    pygame.init()
    screen = pygame.display.set_mode((400, 450))
    pygame.display.set_caption("2048")
    clock = pygame.time.Clock()

    game = Game2048()

    tile_colors = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
    }

    font = pygame.font.Font(None, 50)
    small_font = pygame.font.Font(None, 30)
    score_font = pygame.font.Font(None, 36)

    running = True
    while running:
        screen.fill((187, 173, 160))

        # Draw score
        score_text = score_font.render(f"Score: {game.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 410))

        # Draw grid
        for i in range(4):
            for j in range(4):
                value = game.grid[i][j]
                color = tile_colors.get(value, (0, 0, 0))
                pygame.draw.rect(screen, color, (j * 100 + 10, i * 100 + 10, 80, 80), border_radius=5)
                if value != 0:
                    if value <= 4:
                        text_color = (119, 110, 101)
                    else:
                        text_color = (249, 246, 242)
                    if value >= 1024:
                        text = small_font.render(str(value), True, text_color)
                    else:
                        text = font.render(str(value), True, text_color)
                    screen.blit(text, (j * 100 + 50 - text.get_width() // 2, i * 100 + 50 - text.get_height() // 2))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                moved = False
                if event.key == pygame.K_LEFT:
                    moved = game.move_left()
                elif event.key == pygame.K_RIGHT:
                    moved = game.move_right()
                elif event.key == pygame.K_UP:
                    moved = game.move_up()
                elif event.key == pygame.K_DOWN:
                    moved = game.move_down()
                if moved:
                    game.add_tile()
                if game.is_game_over():
                    print(f"Game Over! Final Score: {game.score}")
                    running = False

        clock.tick(60)

    pygame.quit()

# To play: call play_human()
# 添加入口点
if __name__ == "__main__":
    play_human()
