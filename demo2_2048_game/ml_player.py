# Task 2: ml_player.py - A machine learning module using PyTorch DQN to play the 2048 game and aim for high scores.
# Requires the Game2048 class from Task 1.

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random

# Assume Game2048 is imported from the previous module
from game_2048 import Game2048

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_every = 200
        self.steps = 0
        self.last_invalid_action = None  # 记录上一次无效动作

    def get_state(self, game):
        grid = np.array(game.grid).flatten()
        # Normalize by log2 to handle large numbers
        # 把棋盘数字转成连续的 0~1 之间的数
        grid = np.log2(grid + 1) / np.log2(2048 + 1)
        return torch.tensor(grid, dtype=torch.float32)

    def act(self, state):
        valid_actions = list(range(4))
        # 如果上一步动作无效，就避免继续同方向
        if self.last_invalid_action is not None:
            valid_actions.remove(self.last_invalid_action)

    #epsilon 是一个 0~1 之间的超参数，表示 随机探索的概率。
    #比如 epsilon = 0.1，就有 10% 的概率走随机动作，而不是按照当前模型预测的最优动作。
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
#表示下面的计算不需要记录梯度（不做反向传播），这样会加快推理速度并节省内存。
#因为这里是在“玩游戏”阶段选动作，不是在训练阶段更新参数。
                q_values = self.model(state.unsqueeze(0)).squeeze(0)
                # 屏蔽无效方向
                mask = torch.full_like(q_values, -float('inf'))
                mask[valid_actions] = 0
                q_values = q_values + mask
                return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=1000):
        scores = []
        for episode in range(episodes):
            game = Game2048()
            state = self.get_state(game)
            done = False
            self.last_invalid_action = None  # 每局重置

            while not done:
                action = self.act(state)
                prev_grid = np.array(game.grid)
                prev_score = game.score

                moved = False
                if action == 0:
                    moved = game.move_left()
                elif action == 1:
                    moved = game.move_right()
                elif action == 2:
                    moved = game.move_up()
                elif action == 3:
                    moved = game.move_down()

                reward = game.score - prev_score

                if moved:
                    game.add_tile()
                    self.last_invalid_action = None
                else:
                    reward -= 10  # 惩罚无效动作
                    self.last_invalid_action = action

                if reward == 0 and moved:
                    reward = 1  # 小奖励

                next_state = self.get_state(game)
                done = game.is_game_over()

                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.steps += 1

                if self.steps % self.update_target_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                self.replay()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            scores.append(game.score)
            print(f"Episode {episode + 1}/{episodes}, Score: {game.score}, Epsilon: {self.epsilon:.3f}")

        torch.save(self.model.state_dict(), "2048_dqn_model.pth")
        return scores

# To train: agent = Agent(); agent.train(episodes=5000)  # Adjust episodes as needed for better performance
# 添加入口点
if __name__ == "__main__":
    agent = Agent()
    agent.train(episodes=100)  # 可以调整 episodes 数量
