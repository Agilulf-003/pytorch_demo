# Task 3: visualize_ai.py - Visualize the trained AI playing the 2048 game.
# Requires Game2048 from Task 1 and Agent/DQN from Task 2.

import pygame
import torch
import time

# Assume imports:
from game_2048 import Game2048
from ml_player import Agent, DQN

def play_ai(model_path="2048_dqn_model.pth"):
    agent = Agent()
    agent.model = DQN()
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0  # Greedy policy for demonstration

    pygame.init()
    screen = pygame.display.set_mode((400, 450))
    pygame.display.set_caption("AI Playing 2048")
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
    done = False
    while running and not done:
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

        # AI move
        state = agent.get_state(game)
        action = agent.act(state)
        print(action)
        moved = False
        if action == 0:
            moved = game.move_left()
        elif action == 1:
            moved = game.move_right()
        elif action == 2:
            moved = game.move_up()
        elif action == 3:
            moved = game.move_down()
        if moved:
            game.add_tile()
        done = game.is_game_over()

        # Handle events (allow quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        time.sleep(0.3)  # Delay for visualization (adjust as needed)
        clock.tick(60)

    print(f"AI Game Over! Final Score: {game.score}")
    pygame.quit()

# To visualize: call play_ai("2048_dqn_model.pth")
if __name__ == "__main__":
    play_ai(model_path="2048_dqn_model.pth")
