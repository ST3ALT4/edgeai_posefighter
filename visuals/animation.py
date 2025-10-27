"""
animations.py - Animation system for sprites
Task: T2.8
"""

import pygame
from config import *

class Animation:
    """Sprite animation controller"""
    
    def __init__(self, frames, frame_duration=0.1, loop=True):
        """
        Initialize animation
        
        Args:
            frames: List of pygame.Surface frames
            frame_duration: Time per frame in seconds
            loop: Whether to loop the animation
        """
        self.frames = frames
        self.frame_duration = frame_duration
        self.loop = loop
        
        self.current_frame = 0
        self.time_accumulator = 0
        self.is_playing = False
        self.finished = False
        
    def play(self):
        """Start playing animation"""
        self.is_playing = True
        self.current_frame = 0
        self.finished = False
        
    def stop(self):
        """Stop animation"""
        self.is_playing = False
        
    def reset(self):
        """Reset animation to first frame"""
        self.current_frame = 0
        self.time_accumulator = 0
        self.finished = False
        
    def update(self, dt):
        """
        Update animation
        
        Args:
            dt: Delta time in seconds
        """
        if not self.is_playing or self.finished:
            return
        
        self.time_accumulator += dt
        
        # Advance frames
        while self.time_accumulator >= self.frame_duration:
            self.time_accumulator -= self.frame_duration
            self.current_frame += 1
            
            # Check for animation end
            if self.current_frame >= len(self.frames):
                if self.loop:
                    self.current_frame = 0
                else:
                    self.current_frame = len(self.frames) - 1
                    self.finished = True
                    self.is_playing = False
    
    def get_current_frame(self):
        """
        Get current animation frame
        
        Returns:
            pygame.Surface: Current frame
        """
        if not self.frames:
            return None
        return self.frames[self.current_frame]


class AnimationController:
    """Manages multiple animations for an entity"""
    
    def __init__(self):
        """Initialize animation controller"""
        self.animations = {}
        self.current_animation = None
        
    def add_animation(self, name, animation):
        """
        Add animation
        
        Args:
            name: Animation identifier
            animation: Animation instance
        """
        self.animations[name] = animation
        
    def play(self, name):
        """
        Play animation by name
        
        Args:
            name: Animation identifier
        """
        if name in self.animations:
            # Stop current animation
            if self.current_animation:
                self.current_animation.stop()
            
            # Play new animation
            self.current_animation = self.animations[name]
            self.current_animation.play()
    
    def update(self, dt):
        """Update current animation"""
        if self.current_animation:
            self.current_animation.update(dt)
    
    def get_current_frame(self):
        """Get current frame from active animation"""
        if self.current_animation:
            return self.current_animation.get_current_frame()
        return None

