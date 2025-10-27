"""
hud.py - Heads-up display for game information
Task: T2.3
"""

import pygame
from config import *
from entities.health_bar import HealthBar

class HUD:
    """Heads-up display showing health bars and game info"""
    
    def __init__(self, player1, player2):
        """
        Initialize HUD
        
        Args:
            player1: Player 1 instance
            player2: Player 2 instance
        """
        self.player1 = player1
        self.player2 = player2
        
        # Create health bars (T2.3)
        self.health_bar_1 = HealthBar(
            x=50,
            y=HEALTH_BAR_OFFSET_Y,
            max_health=PLAYER_MAX_HEALTH,
            is_player_1=True
        )
        
        self.health_bar_2 = HealthBar(
            x=SCREEN_WIDTH - HEALTH_BAR_WIDTH - 50,
            y=HEALTH_BAR_OFFSET_Y,
            max_health=PLAYER_MAX_HEALTH,
            is_player_1=False
        )
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
    def update(self, dt):
        """
        Update HUD elements
        
        Args:
            dt: Delta time in seconds
        """
        # Update health bars with current player health
        self.health_bar_1.update(self.player1.health)
        self.health_bar_2.update(self.player2.health)
    
    def render(self, screen):
        """
        Render HUD elements
        
        Args:
            screen: Pygame surface to draw on
        """
        # Render health bars
        self.health_bar_1.render(screen)
        self.health_bar_2.render(screen)
        
        # Render player labels
        p1_label = self.font_large.render("PLAYER 1", True, PLAYER_1_COLOR)
        p2_label = self.font_large.render("PLAYER 2", True, PLAYER_2_COLOR)
        
        screen.blit(p1_label, (50, HEALTH_BAR_OFFSET_Y - 40))
        p2_label_rect = p2_label.get_rect(right=SCREEN_WIDTH - 50, y=HEALTH_BAR_OFFSET_Y - 40)
        screen.blit(p2_label, p2_label_rect)
        
        # Show shield indicator if active
        if self.player1.shield_active:
            shield_text = self.font_small.render("SHIELD", True, SHIELD_COLOR)
            screen.blit(shield_text, (50, HEALTH_BAR_OFFSET_Y + HEALTH_BAR_HEIGHT + 10))
        
        if self.player2.shield_active:
            shield_text = self.font_small.render("SHIELD", True, SHIELD_COLOR)
            shield_rect = shield_text.get_rect(right=SCREEN_WIDTH - 50, y=HEALTH_BAR_OFFSET_Y + HEALTH_BAR_HEIGHT + 10)
            screen.blit(shield_text, shield_rect)

