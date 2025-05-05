import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pygame
import sys
import time
import matplotlib.pyplot as plt

# ======================== Game Settings ========================
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
GRAVITY = 0.5
JUMP_STRENGTH = -12 
PLAYER_SPEED = 6

# Colors
SKY_BLUE = (135, 206, 235)
GROUND_GREEN = (34, 139, 34)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)
WHITE = (255, 255, 255)

# Action map
ACTION_MAP = {
    0: "left",
    1: "right",
    2: "jump",
    3: "none"
}

# ======================== Neural Network ========================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ======================== Experience Memory ========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

# ======================== DQN Agent ========================
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.97, epsilon=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        
        self.memory = ReplayBuffer(100000)
        self.batch_size = 128
        self.update_target_steps = 500
        self.steps_done = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.steps_done += 1
        if self.steps_done % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# ======================== Mario Character ========================
class Mario:
    def __init__(self):
        self.reset_position()
        self.width = 40
        self.height = 60
        self.score = 0
        self.lives = 3
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.load_image()
    
    def load_image(self):
        try:
            self.image = pygame.image.load(os.path.join('images', 'Super.png')).convert_alpha()
            self.image = pygame.transform.scale(self.image, (self.width, self.height))
        except:
            self.image = pygame.Surface((self.width, self.height))
            self.image.fill(RED)
    
    def reset_position(self):
        self.x = 50
        self.y = SCREEN_HEIGHT - 100
        self.x_vel = 0
        self.y_vel = 0
        self.is_jumping = False
    
    def move(self, action):
        if action == "left":
            self.x_vel = -PLAYER_SPEED
        elif action == "right":
            self.x_vel = PLAYER_SPEED
        elif action == "jump" and not self.is_jumping:
            self.y_vel = JUMP_STRENGTH
            self.is_jumping = True
        else:
            self.x_vel = 0
        
        self.x += self.x_vel
        self.y += self.y_vel
        
        self.y_vel += GRAVITY
        
        if self.x < 0:
            self.x = 0
        if self.x > SCREEN_WIDTH - self.width:
            self.x = SCREEN_WIDTH - self.width
        if self.y > SCREEN_HEIGHT - self.height:
            self.y = SCREEN_HEIGHT - self.height
            self.is_jumping = False
            self.y_vel = 0
        
        self.rect.x = self.x
        self.rect.y = self.y
    
    def get_state(self, enemies, coins):
        # Find nearest enemy and coin
        nearest_enemy = min(enemies, key=lambda e: abs(e.x - self.x)) if enemies else None
        nearest_coin = min(coins, key=lambda c: abs(c.x - self.x)) if coins else None
        
        # Calculate distances and directions
        enemy_dist = abs(nearest_enemy.x - self.x) / SCREEN_WIDTH if nearest_enemy else 1.0
        coin_dist = abs(nearest_coin.x - self.x) / SCREEN_WIDTH if nearest_coin else 1.0
        
        # Enemy direction (-1 for left, 1 for right, 0 if no enemy)
        enemy_dir = 0
        if nearest_enemy:
            enemy_dir = 1 if nearest_enemy.x > self.x else -1
        
        # Coin direction (-1 for left, 1 for right, 0 if no coin)
        coin_dir = 0
        if nearest_coin:
            coin_dir = 1 if nearest_coin.x > self.x else -1
        
        return np.array([
            self.x / SCREEN_WIDTH,  # Normalized x position
            self.y / SCREEN_HEIGHT,  # Normalized y position
            self.x_vel / PLAYER_SPEED,  # Normalized x velocity
            self.y_vel / 10,  # Normalized y velocity
            enemy_dist,  # Distance to nearest enemy
            enemy_dir,  # Direction to nearest enemy
            coin_dist,  # Distance to nearest coin
            coin_dir,  # Direction to nearest coin
            int(self.is_jumping),  # Is jumping flag
            self.x_vel > 0  # Moving right flag
        ])

# ======================== Enemy ========================
class Enemy:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 40
        self.speed = 3
        self.direction = 1
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.load_image()
    
    def load_image(self):
        try:
            self.image = pygame.image.load(os.path.join('images', 'obstical.png')).convert_alpha()
            self.image = pygame.transform.scale(self.image, (self.width, self.height))
        except:
            self.image = pygame.Surface((self.width, self.height))
            self.image.fill(BLACK)
    
    def update(self):
        self.x += self.speed * self.direction
        if self.x <= 0 or self.x >= SCREEN_WIDTH - self.width:
            self.direction *= -1
        self.rect.x = self.x
    
    def draw(self, screen):
        screen.blit(self.image, (self.rect.x, self.rect.y))

# ======================== Coin ========================
class Coin:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 30
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.load_image()
    
    def load_image(self):
        try:
            self.image = pygame.image.load(os.path.join('images', 'coin.png')).convert_alpha()
            self.image = pygame.transform.scale(self.image, (self.width, self.height))
        except:
            self.image = pygame.Surface((self.width, self.height))
            self.image.fill(GOLD)
    
    def draw(self, screen):
        screen.blit(self.image, (self.rect.x, self.rect.y))

# ======================== Platform ========================
class Platform:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.load_image()
    
    def load_image(self):
        try:
            self.image = pygame.image.load(os.path.join('images', 'brickwork.png')).convert_alpha()
            self.image = pygame.transform.scale(self.image, (self.rect.width, self.rect.height))
        except:
            self.image = None
    
    def draw(self, screen):
        if self.image:
            screen.blit(self.image, (self.rect.x, self.rect.y))
        else:
            pygame.draw.rect(screen, GROUND_GREEN, self.rect)

# ======================== Game Environment ========================
class MarioGame:
    def __init__(self, render=False):
        self.render = render
        if self.render:
            self.init_pygame()
            self.load_images()
        
        self.reset()
    
    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Mario RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.big_font = pygame.font.SysFont('Arial', 48)
    
    def load_images(self):
        try:
            self.background = pygame.image.load(os.path.join('images', 'bg.png')).convert()
            self.background = pygame.transform.scale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))
        except:
            self.background = None
    
    def reset(self):
        self.mario = Mario()
        self.platforms = [
            Platform(0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 40),
            Platform(200, 400, 200, 40),
            Platform(500, 300, 200, 40),
            Platform(300, 200, 200, 40)
        ]
        
        self.enemies = [
            Enemy(300, SCREEN_HEIGHT - 80),
            Enemy(600, 380),
            Enemy(400, 280)
        ]
        
        self.coins = [
            Coin(250, 370),
            Coin(550, 270),
            Coin(350, 170),
            Coin(700, SCREEN_HEIGHT - 70),
            Coin(150, SCREEN_HEIGHT - 70)
        ]
        
        self.done = False
        return self.mario.get_state(self.enemies, self.coins)
    
    def check_collision_with_platforms(self):
        for platform in self.platforms:
            if (self.mario.rect.colliderect(platform.rect) and 
                self.mario.y_vel > 0 and 
                self.mario.rect.bottom > platform.rect.top + 5):
                
                self.mario.y = platform.rect.top - self.mario.height
                self.mario.is_jumping = False
                self.mario.y_vel = 0
                return True
        return False
    
    def step(self, action_idx):
        if self.done:
            return self.reset(), 0, True
        
        prev_x = self.mario.x
        prev_y = self.mario.y
        action = ACTION_MAP[action_idx]
        self.mario.move(action)
        
        for enemy in self.enemies:
            enemy.update()
        
        self.check_collision_with_platforms()
        
        # Calculate rewards
        reward = 0
        
        # Progress reward (moving right) - only reward forward movement
        progress = (self.mario.x - prev_x) / SCREEN_WIDTH
        if progress > 0:
            reward += progress * 10  # Reward for moving right
        
        # No penalty for moving left/backward
        
        # Vertical movement penalty (to discourage unnecessary jumping)
        if abs(self.mario.y - prev_y) > 5:
            reward -= 0.1
        
        # Survival reward
        reward += 0.1
        
        done = False
        
        # Collision with enemies
        for enemy in self.enemies:
            if self.mario.rect.colliderect(enemy.rect):
                reward = -20  # Large penalty for hitting enemy
                self.mario.lives -= 1
                if self.mario.lives <= 0:
                    done = True
                else:
                    self.mario.reset_position()
        
        # Collecting coins
        for coin in self.coins[:]:
            if self.mario.rect.colliderect(coin.rect):
                self.coins.remove(coin)
                reward += 20  # Big reward for collecting coins
                self.mario.score += 10
        
        # No penalty for falling off screen - just reset position
        if self.mario.y > SCREEN_HEIGHT:
            self.mario.reset_position()
        
        # If collected all coins
        if len(self.coins) == 0:
            reward += 100  # Large reward for collecting all coins
            done = True
        
        # Reward for being near coins
        nearest_coin = min(self.coins, key=lambda c: abs(c.x - self.mario.x)) if self.coins else None
        if nearest_coin:
            coin_dist = abs(nearest_coin.x - self.mario.x) / SCREEN_WIDTH
            reward += (1 - coin_dist) * 0.5  # Small reward for getting closer to coins
        
        # Penalty for being near enemies
        nearest_enemy = min(self.enemies, key=lambda e: abs(e.x - self.mario.x)) if self.enemies else None
        if nearest_enemy:
            enemy_dist = abs(nearest_enemy.x - self.mario.x) / SCREEN_WIDTH
            if enemy_dist < 0.2:  # Very close to enemy
                reward -= (1 - enemy_dist) * 5  # Significant penalty
        
        next_state = self.mario.get_state(self.enemies, self.coins)
        
        if self.render:
            self._render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    sys.exit()
        
        self.done = done
        return next_state, reward, done
    
    def _render(self):
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill(SKY_BLUE)

        for platform in self.platforms:
            platform.draw(self.screen)

        for coin in self.coins:
            coin.draw(self.screen)

        for enemy in self.enemies:
            enemy.draw(self.screen)

        self.screen.blit(self.mario.image, (self.mario.rect.x, self.mario.rect.y))

        # Display score and lives
        score_text = self.font.render(f"Score: {self.mario.score}", True, WHITE)
        lives_text = self.font.render(f"Lives: {self.mario.lives}", True, WHITE)
        coins_text = self.font.render(f"Coins Left: {len(self.coins)}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 40))
        self.screen.blit(coins_text, (10, 70))

        # Game over or success messages
        if self.done:
            if self.mario.lives <= 0:
                game_over_text = self.big_font.render("GAME OVER!", True, RED)
                text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
                self.screen.blit(game_over_text, text_rect)
                
                # Display final score
                final_score_text = self.font.render(f"Final Score: {self.mario.score}", True, WHITE)
                score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
                self.screen.blit(final_score_text, score_rect)
                
                pygame.display.flip()
                pygame.time.delay(2000)  # Pause for 2 seconds
            elif len(self.coins) == 0:
                success_text = self.big_font.render("LEVEL COMPLETE!", True, GOLD)
                text_rect = success_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
                self.screen.blit(success_text, text_rect)
                
                # Display final score
                final_score_text = self.font.render(f"Final Score: {self.mario.score}", True, WHITE)
                score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
                self.screen.blit(final_score_text, score_rect)
                
                pygame.display.flip()
                pygame.time.delay(2000)  # Pause for 2 seconds

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

# ======================== Training and Testing ========================
def train_dqn_agent(episodes=1000, render_every=100):
    env = MarioGame(render=False)
    state_dim = len(env.reset())
    action_dim = len(ACTION_MAP)
    agent = DQNAgent(state_dim, action_dim)
    
    rewards_history = []
    coins_collected_history = []
    survival_time_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        if episode % render_every == 0:
            env.close()
            env = MarioGame(render=True)
        else:
            env.close()
            env = MarioGame(render=False)
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            steps += 1
        
        rewards_history.append(total_reward)
        coins_collected_history.append(10 - len(env.coins))  # Track how many coins collected
        survival_time_history.append(steps)  # Track how long the agent survived
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_coins = np.mean(coins_collected_history[-10:])
            avg_survival = np.mean(survival_time_history[-10:])
            print(f"Episode {episode} | Reward: {total_reward:.1f} | Avg Reward: {avg_reward:.1f} | "
                  f"Coins: {10 - len(env.coins)} | Avg Coins: {avg_coins:.1f} | "
                  f"Steps: {steps} | Avg Steps: {avg_survival:.1f} | Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Reward Evolution')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(coins_collected_history)
    plt.title('Coins Collected')
    plt.xlabel('Episode')
    plt.ylabel('Coins Collected')
    
    plt.subplot(1, 3, 3)
    plt.plot(survival_time_history)
    plt.title('Survival Time')
    plt.xlabel('Episode')
    plt.ylabel('Steps Survived')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    
    return agent, rewards_history

def test_dqn_agent(agent, episodes=5):
    env = MarioGame(render=True)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            time.sleep(0.02)
        
        print(f"Test Episode {episode} | Final Score: {env.mario.score} | "
              f"Coins Collected: {10 - len(env.coins)} | Reward: {total_reward:.1f}")
    
    env.close()

# ======================== Main Execution ========================
if __name__ == "__main__":
    print("Starting Mario RL Training...")
    print(f"PyGame {pygame.__version__}, Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Training
    trained_agent, rewards_history = train_dqn_agent(episodes=100)
    
    # Save model
    torch.save(trained_agent.policy_net.state_dict(), 'mario_dqn.pth')
    
    # Testing
    test_dqn_agent(trained_agent, episodes=3)
    
    print("Training Completed!")