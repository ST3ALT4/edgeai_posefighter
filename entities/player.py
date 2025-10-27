"""
player.py - Player entity class
Task: T2.4
"""

import pygame
from config import *
from entities.health_bar import HealthBar

class Player(pygame.sprite.Sprite):
    """Player character controlled by pose detection"""
    
    def __init__(self, player_id, x, y, color):
        """
        Initialize player
        
        Args:
            player_id: 1 or 2
            x, y: Starting position
            color: Player color
        """
        super().__init__()
        
        self.player_id = player_id
        self.color = color
        
        # Position and movement
        self.x = x
        self.y = y
        self.pose_x_offset = 0  # Offset from pose data
        
        # Create sprite surface
        self.image = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT))
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(x, y))
        
        # Hitbox (smaller than visual sprite for better gameplay)
        self.hitbox = pygame.Rect(
            x - PLAYER_WIDTH // 3,
            y - PLAYER_HEIGHT // 3,
            PLAYER_WIDTH // 1.5,
            PLAYER_HEIGHT // 1.5
        )
        
        # Health system
        self.max_health = PLAYER_MAX_HEALTH
        self.health = self.max_health
        
        # Shield status
        self.shield_active = False
        self.shield_timer = 0
        
    def update_from_pose(self, pose_data):
        """
        Update player position from pose landmarks
        
        Args:
            pose_data: Dictionary with normalized pose landmarks from T1
                      Expected format: {'hip_x': float, 'hip_y': float, ...}
        """
        if not pose_data:
            return
        
        # Use hip position as main tracking point
        if 'hip_x' in pose_data:
            # Normalize hip_x is typically 0-1, map to horizontal movement range
            # Allow player to move within their half of the screen
            if self.player_id == 1:
                movement_range = SCREEN_WIDTH // 2 - 100
                base_x = 50
            else:
                movement_range = SCREEN_WIDTH // 2 - 100
                base_x = SCREEN_WIDTH // 2 + 50
            
            self.pose_x_offset = pose_data['hip_x'] * movement_range
            self.x = base_x + self.pose_x_offset
    
    def update(self, dt):
        """
        Update player state
        
        Args:
            dt: Delta time in seconds
        """
        # Update sprite position
        self.rect.center = (self.x, self.y)
        
        # Update hitbox
        self.hitbox.center = self.rect.center
        
        # Update shield timer
        if self.shield_active:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.shield_active = False
    
    def take_damage(self, amount):
        """
        Apply damage to player
        
        Args:
            amount: Damage amount
        """
        self.health = max(0, self.health - amount)
        print(f"Player {self.player_id} took {amount} damage! Health: {self.health}")
    
    def activate_shield(self):
        """Activate defensive shield"""
        self.shield_active = True
        self.shield_timer = SHIELD_DURATION
    
    def render(self, screen):
        """
        Render player sprite
        
        Args:
            screen: Pygame surface to draw on
        """
        # Draw main sprite
        screen.blit(self.image, self.rect)
        
        # Draw shield if active
        if self.shield_active:
            shield_surface = pygame.Surface((PLAYER_WIDTH + 20, PLAYER_HEIGHT + 20), pygame.SRCALPHA)
            shield_color = (*SHIELD_COLOR, SHIELD_ALPHA)
            pygame.draw.ellipse(shield_surface, shield_color, shield_surface.get_rect(), 3)
            shield_rect = shield_surface.get_rect(center=self.rect.center)
            screen.blit(shield_surface, shield_rect)
        
        # Debug: Draw hitbox (optional, comment out for production)
        # pygame.draw.rect(screen, GREEN, self.hitbox, 2)

