"""
hitbox.py - Hitbox and collision detection
Task: T2.7
"""

import pygame
from config import *

class AttackHitbox:
    """Hitbox for attack collision detection"""
    
    def __init__(self, x, y, width, height, damage):
        """
        Initialize attack hitbox
        
        Args:
            x, y: Top-left position
            width, height: Dimensions
            damage: Damage amount on hit
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.damage = damage
        
    def update_position(self, x, y):
        """
        Update hitbox position (centered)
        
        Args:
            x, y: Center position
        """
        self.rect.center = (x, y)
    
    def check_collision(self, target_rect):
        """
        Check collision with another rect
        
        Args:
            target_rect: pygame.Rect to check against
            
        Returns:
            bool: True if collision detected
        """
        return self.rect.colliderect(target_rect)
    
    def render_debug(self, screen):
        """
        Render hitbox for debugging
        
        Args:
            screen: Pygame surface
        """
        pygame.draw.rect(screen, RED, self.rect, 2)

