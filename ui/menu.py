"""
menu.py - Main menu UI
Task: T2.2
"""

import pygame
from config import *

class Menu:
    """Main menu screen"""
    
    def __init__(self):
        """Initialize menu"""
        self.title_font = pygame.font.Font(None, 96)
        self.menu_font = pygame.font.Font(None, 48)
        self.instruction_font = pygame.font.Font(None, 32)
        
        # Animation
        self.pulse_time = 0
        
    def update(self, dt):
        """
        Update menu animations
        
        Args:
            dt: Delta time in seconds
        """
        self.pulse_time += dt
    
    def render(self, screen):
        """
        Render menu screen
        
        Args:
            screen: Pygame surface to draw on
        """
        # Title
        title = self.title_font.render("POSE FIGHTERS", True, YELLOW)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 150))
        screen.blit(title, title_rect)
        
        # Subtitle
        subtitle = self.menu_font.render("Arena Battle", True, WHITE)
        subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 230))
        screen.blit(subtitle, subtitle_rect)
        
        # Pulsing "Press SPACE" text
        alpha = int((pygame.math.Vector2(1, 0).rotate(self.pulse_time * 200).y + 1) * 127.5)
        start_text = self.instruction_font.render("Press SPACE to Start", True, GREEN)
        start_text.set_alpha(alpha)
        start_rect = start_text.get_rect(center=(SCREEN_WIDTH // 2, 400))
        screen.blit(start_text, start_rect)
        
        # Instructions
        instructions = [
            "Player 1: Left side of screen",
            "Player 2: Right side of screen",
            "",
            "Perform poses to attack!",
            "Fireball - ??",
            "Lightning - ??",
            "Shield - Arms crossed"
        ]
        
        y_offset = 500
        for instruction in instructions:
            text = self.instruction_font.render(instruction, True, LIGHT_GRAY)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            screen.blit(text, text_rect)
            y_offset += 35

