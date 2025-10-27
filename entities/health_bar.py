"""
health_bar.py - Health bar visualization component
Task: T2.3
"""

import pygame
from config import *

class HealthBar:
    """Visual health bar for players"""
    
    def __init__(self, x, y, max_health, is_player_1=True):
        """
        Initialize health bar
        
        Args:
            x, y: Position on screen
            max_health: Maximum health value
            is_player_1: True for player 1 (left-aligned), False for player 2 (right-aligned)
        """
        self.x = x
        self.y = y
        self.max_health = max_health
        self.current_health = max_health
        self.is_player_1 = is_player_1
        
        # Bar dimensions
        self.width = HEALTH_BAR_WIDTH
        self.height = HEALTH_BAR_HEIGHT
        self.border = HEALTH_BAR_BORDER
        
    def update(self, current_health):
        """
        Update current health value
        
        Args:
            current_health: Current health to display
        """
        self.current_health = max(0, min(current_health, self.max_health))
    
    def render(self, screen):
        """
        Render health bar
        
        Args:
            screen: Pygame surface to draw on
        """
        # Calculate health ratio
        health_ratio = self.current_health / self.max_health
        
        # Choose color based on health level
        if health_ratio > 0.6:
            health_color = GREEN
        elif health_ratio > 0.3:
            health_color = YELLOW
        else:
            health_color = RED
        
        # Draw border (background)
        border_rect = pygame.Rect(
            self.x - self.border,
            self.y - self.border,
            self.width + 2 * self.border,
            self.height + 2 * self.border
        )
        pygame.draw.rect(screen, WHITE, border_rect)
        
        # Draw background (max health)
        background_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, DARK_GRAY, background_rect)
        
        # Draw current health
        health_width = int(self.width * health_ratio)
        health_rect = pygame.Rect(self.x, self.y, health_width, self.height)
        pygame.draw.rect(screen, health_color, health_rect)
        
        # Draw health text
        font = pygame.font.Font(None, 24)
        health_text = f"{int(self.current_health)}/{self.max_health}"
        text_surface = font.render(health_text, True, WHITE)
        text_rect = text_surface.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(text_surface, text_rect)

