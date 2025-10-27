"""
particle_effects.py - Particle effects system
Task: T2.8
"""

import pygame
import random
import math
from config import *

class Particle:
    """Single particle instance"""
    
    def __init__(self, x, y, color):
        """
        Initialize particle
        
        Args:
            x, y: Starting position
            color: Particle color (R, G, B)
        """
        self.x = x
        self.y = y
        self.color = color
        
        # Random velocity
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(PARTICLE_SPEED_MIN, PARTICLE_SPEED_MAX)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        
        # Random size
        self.size = random.randint(PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX)
        
        # Lifetime
        self.lifetime = PARTICLE_LIFETIME
        self.age = 0
        self.alive = True
        
    def update(self, dt):
        """Update particle physics"""
        self.x += self.vx
        self.y += self.vy
        
        # Gravity effect
        self.vy += 0.2
        
        # Age and fade
        self.age += 1
        if self.age >= self.lifetime:
            self.alive = False
    
    def render(self, screen):
        """Render particle"""
        # Calculate alpha based on age
        alpha = int(255 * (1 - self.age / self.lifetime))
        
        # Draw particle with transparency
        surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        color_with_alpha = (*self.color, alpha)
        pygame.draw.circle(surf, color_with_alpha, (self.size, self.size), self.size)
        screen.blit(surf, (int(self.x - self.size), int(self.y - self.size)))


class ParticleEffect:
    """Particle effect system (T2.8)"""
    
    def __init__(self, x, y, color, count=20):
        """
        Initialize particle effect
        
        Args:
            x, y: Origin position
            color: Effect color
            count: Number of particles
        """
        self.particles = []
        self.alive = True
        
        # Create particles
        for _ in range(count):
            self.particles.append(Particle(x, y, color))
    
    def update(self, dt):
        """Update all particles"""
        # Update particles
        for particle in self.particles[:]:
            particle.update(dt)
            if not particle.alive:
                self.particles.remove(particle)
        
        # Check if effect is done
        if len(self.particles) == 0:
            self.alive = False
    
    def render(self, screen):
        """Render all particles"""
        for particle in self.particles:
            particle.render(screen)


class ExplosionEffect(ParticleEffect):
    """Explosion particle effect"""
    
    def __init__(self, x, y, color, radius=50):
        """
        Initialize explosion
        
        Args:
            x, y: Center position
            color: Explosion color
            radius: Effect radius
        """
        super().__init__(x, y, color, count=30)
        
        # Override particle velocities for radial explosion
        for particle in self.particles:
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3, 8)
            particle.vx = math.cos(angle) * speed
            particle.vy = math.sin(angle) * speed


class TrailEffect:
    """Trail effect following moving object"""
    
    def __init__(self, color):
        """
        Initialize trail
        
        Args:
            color: Trail color
        """
        self.color = color
        self.points = []
        self.max_points = 10
        
    def add_point(self, x, y):
        """Add new point to trail"""
        self.points.append((x, y))
        if len(self.points) > self.max_points:
            self.points.pop(0)
    
    def update(self, dt):
        """Update trail"""
        pass
    
    def render(self, screen):
        """Render trail"""
        if len(self.points) < 2:
            return
        
        for i in range(len(self.points) - 1):
            alpha = int(255 * (i / len(self.points)))
            start = self.points[i]
            end = self.points[i + 1]
            
            # Draw line segment
            pygame.draw.line(screen, self.color, start, end, 3)

