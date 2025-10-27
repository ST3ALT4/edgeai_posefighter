"""
sprite_manager.py - Sprite loading and management
Task: T2.8
"""

import pygame
import os
from config import *

class SpriteManager:
    """Manages sprite loading and caching"""
    
    def __init__(self):
        """Initialize sprite manager"""
        self.sprites = {}
        self.sprite_sheets = {}
        
    def load_sprite(self, name, filepath):
        """
        Load a sprite image
        
        Args:
            name: Identifier for the sprite
            filepath: Path to image file
            
        Returns:
            pygame.Surface: Loaded sprite
        """
        if name in self.sprites:
            return self.sprites[name]
        
        try:
            sprite = pygame.image.load(filepath).convert_alpha()
            self.sprites[name] = sprite
            return sprite
        except pygame.error as e:
            print(f"Could not load sprite {filepath}: {e}")
            # Return placeholder surface
            placeholder = pygame.Surface((64, 64))
            placeholder.fill(PURPLE)
            return placeholder
    
    def load_sprite_sheet(self, name, filepath, frame_width, frame_height):
        """
        Load sprite sheet and split into frames
        
        Args:
            name: Identifier for sprite sheet
            filepath: Path to sprite sheet image
            frame_width: Width of each frame
            frame_height: Height of each frame
            
        Returns:
            list: List of frame surfaces
        """
        if name in self.sprite_sheets:
            return self.sprite_sheets[name]
        
        try:
            sheet = pygame.image.load(filepath).convert_alpha()
            frames = []
            
            sheet_width, sheet_height = sheet.get_size()
            cols = sheet_width // frame_width
            rows = sheet_height // frame_height
            
            for row in range(rows):
                for col in range(cols):
                    x = col * frame_width
                    y = row * frame_height
                    frame = sheet.subsurface((x, y, frame_width, frame_height))
                    frames.append(frame)
            
            self.sprite_sheets[name] = frames
            return frames
            
        except pygame.error as e:
            print(f"Could not load sprite sheet {filepath}: {e}")
            return []
    
    def get_sprite(self, name):
        """
        Get cached sprite
        
        Args:
            name: Sprite identifier
            
        Returns:
            pygame.Surface: Sprite or None
        """
        return self.sprites.get(name)
    
    def clear_cache(self):
        """Clear all cached sprites"""
        self.sprites.clear()
        self.sprite_sheets.clear()


# Global sprite manager instance
sprite_manager = SpriteManager()

