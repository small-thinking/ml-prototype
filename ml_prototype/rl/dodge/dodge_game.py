import pygame
from numpy import array
import random


class DodgeGame:
    def __init__(self):
        self.player_pos = 2  # initial player position (0-4)
        self.obstacle_pos = None
        self.paused = False
        self.game_over = False
        self.cell_size = 100
        self.width = 5 * self.cell_size
        self.height = self.cell_size
        self.screen = None

    def init_display(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dodge Game")

    def render(self):
        if self.screen is None:
            self.init_display()

        self.screen.fill((255, 255, 255))  # White background
        
        # Draw grid lines
        for x in range(0, self.width + 1, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.height))

        # Draw player (blue)
        player_x = self.player_pos * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, (0, 0, 255), (player_x, self.height // 2), 30)

        # Draw obstacle (red)
        obstacle_x = self.obstacle_pos * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, (255, 0, 0), (obstacle_x, self.height // 2), 30)

        pygame.display.flip()
        
    def close(self):
        if self.screen is not None:
            pygame.quit()

    def reset(self):
        self.player_pos = 2
        self.obstacle_pos = random.randint(0, 4)
        self.paused = False
        self.game_over = False
        return self.get_state()

    def get_state(self):
        dist = abs(self.player_pos - self.obstacle_pos)
        direction = (
            1
            if self.player_pos < self.obstacle_pos
            else -1
            if self.player_pos > self.obstacle_pos
            else 0
        )
        return array([self.player_pos, self.obstacle_pos, int(self.paused), dist, direction])

    def step(self, action):
        reward = 0
        old_dist = abs(self.player_pos - self.obstacle_pos)

        if action == 3:  # pause
            self.paused = True
            reward = -1
            return self.get_state(), reward, self.game_over

        if self.paused:
            self.obstacle_pos = (self.obstacle_pos + 1) % 5
            return self.get_state(), reward, self.game_over

        if action == 0 and self.player_pos > 0:  # left
            self.player_pos -= 1
        elif action == 1 and self.player_pos < 4:  # right
            self.player_pos += 1
        elif action == 2 and old_dist >= 2:  # stay when safe
            reward += 2

        new_dist = abs(self.player_pos - self.obstacle_pos)

        if new_dist > old_dist:
            reward += 5
        elif new_dist < old_dist:
            reward -= 3 if new_dist < 2 else 1

        if self.player_pos == self.obstacle_pos:
            self.game_over = True
            reward = -20
        else:
            if new_dist >= 3:
                reward += 4
            elif new_dist == 2:
                reward += 2
            elif new_dist == 1:
                reward += 1

        self.obstacle_pos = (
            (self.obstacle_pos + 1) % 5
            if self.obstacle_pos < self.player_pos
            else (self.obstacle_pos - 1) % 5
        )

        return self.get_state(), reward, self.game_over
