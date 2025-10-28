"""
attack_system.py - Attack spawning and management
Tasks: T2.5, T2.6
"""

import pygame
import math
from config import *
from visuals.particle_effects import ParticleEffect
from combat.hitbox import AttackHitbox

class Attack(pygame.sprite.Sprite):
    """Base attack class"""
    
    def __init__(self, attacker, attack_type):
        """
        Initialize attack
        
        Args:
            attacker: Player who triggered the attack
            attack_type: Type of attack (from SUPERPOWERS keys)
        """
        super().__init__()
        self.attacker = attacker
        self.attack_type = attack_type
        self.alive = True
        
    def update(self, dt):
        """Update attack state"""
        pass
    
    def check_hit(self, target):
        """Check if attack hits target"""
        return False
    
    def render(self, screen):
        """Render attack visual"""
        pass


class Fireball(Attack):
    """Fireball projectile attack"""
    
    def __init__(self, attacker):
        super().__init__(attacker, "fireball")
        
        # Starting position (from player center)
        self.x = attacker.x
        self.y = attacker.y
        
        # Direction (toward opponent side)
        self.direction = 1 if attacker.player_id == 1 else -1
        self.speed = FIREBALL_SPEED
        
        # Visual properties
        self.radius = FIREBALL_RADIUS
        self.color = FIREBALL_COLOR
        
        # Create hitbox
        self.hitbox = AttackHitbox(
            self.x, self.y,
            self.radius * 2, self.radius * 2,
            FIREBALL_DAMAGE
        )
        
    def update(self, dt):
        """Update fireball position"""
        self.x += self.direction * self.speed
        self.hitbox.update_position(self.x, self.y)
        
        # Remove if off screen
        if self.x < 0 or self.x > SCREEN_WIDTH:
            self.alive = False
    
    def check_hit(self, target):
        """Check collision with target"""
        return self.hitbox.check_collision(target.hitbox)
    
    def render(self, screen):
        """Render fireball"""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        # Add glow effect
        glow_surface = pygame.Surface((self.radius * 4, self.radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.color, 64), 
                          (self.radius * 2, self.radius * 2), self.radius * 2)
        screen.blit(glow_surface, (int(self.x - self.radius * 2), int(self.y - self.radius * 2)))


class LightningStrike(Attack):
    """Lightning strike area attack"""
    
    def __init__(self, attacker, target_x):
        super().__init__(attacker, "lightning")
        
        self.x = target_x
        self.y = 0
        self.duration = LIGHTNING_DURATION
        self.timer = self.duration
        
        # Visual properties
        self.width = LIGHTNING_WIDTH
        self.color = LIGHTNING_COLOR
        
        # Hitbox spans from top to bottom
        self.hitbox = AttackHitbox(
            self.x - self.width // 2,
            0,
            self.width,
            SCREEN_HEIGHT,
            LIGHTNING_DAMAGE
        )
        self.has_hit = False
        
    def update(self, dt):
        """Update lightning duration"""
        self.timer -= 1
        if self.timer <= 0:
            self.alive = False
    
    def check_hit(self, target):
        """Check collision with target (only hits once)"""
        if self.has_hit:
            return False
        
        if self.hitbox.check_collision(target.hitbox):
            self.has_hit = True
            return True
        return False
    
    def render(self, screen):
        """Render lightning bolt"""
        # Draw jagged lightning bolt
        segments = 8
        points = []
        for i in range(segments + 1):
            y = (SCREEN_HEIGHT // segments) * i
            x_offset = (i % 2 - 0.5) * 30
            points.append((self.x + x_offset, y))
        
        if len(points) > 1:
            pygame.draw.lines(screen, self.color, False, points, 5)
        
        # Flash effect
        alpha = int((self.timer / self.duration) * 200)
        flash_surf = pygame.Surface((self.width * 2, SCREEN_HEIGHT), pygame.SRCALPHA)
        flash_surf.fill((*self.color, alpha))
        screen.blit(flash_surf, (self.x - self.width, 0))


class AttackSystem:
    """Manages all active attacks (T2.5, T2.6)"""
    
    def __init__(self):
        """Initialize attack system"""
        self.active_attacks = []
        self.particle_effects = []
        
    def trigger_attack(self, player, move_name):
        """
        Trigger an attack from pose classification (T2.5)
        
        Args:
            player: Player triggering the attack
            move_name: Name of the superpower move
        """
        print(f"Player {player.player_id} triggered {move_name}!")
        
        # Create attack based on type (T2.6)
        if move_name == "fireball":
            attack = Fireball(player)
            self.active_attacks.append(attack)
            
        elif move_name == "lightning":
            # Target opponent's position
            target_x = SCREEN_WIDTH - player.x
            attack = LightningStrike(player, target_x)
            self.active_attacks.append(attack)
            
        elif move_name == "shield":
            player.activate_shield()
            
        # Spawn particle effect
        self._spawn_particles(player.x, player.y, move_name)
    
    def _spawn_particles(self, x, y, attack_type):
        """
        Spawn particle effect for attack
        
        Args:
            x, y: Position to spawn particles
            attack_type: Type of attack
        """
        # Map attack type to particle color
        color_map = {
            "fireball": FIREBALL_COLOR,
            "lightning": LIGHTNING_COLOR,
            "shield": SHIELD_COLOR,
        }
        
        color = color_map.get(attack_type, WHITE)
        effect = ParticleEffect(x, y, color, PARTICLES_PER_EFFECT)
        self.particle_effects.append(effect)
    
    def update(self, dt):
        """
        Update all active attacks
        
        Args:
            dt: Delta time in seconds
        """
        # Update attacks
        for attack in self.active_attacks[:]:
            attack.update(dt)
            if not attack.alive:
                self.active_attacks.remove(attack)
        
        # Update particle effects
        for effect in self.particle_effects[:]:
            effect.update(dt)
            if not effect.alive:
                self.particle_effects.remove(effect)
    
    def check_collisions(self, player1, player2):
        """
        Check for attack collisions with players (T2.7)
        
        Args:
            player1, player2: Player instances
            
        Returns:
            List of hit information dictionaries
        """
        hits = []
        
        for attack in self.active_attacks:
            # Don't hit own player
            target = player2 if attack.attacker.player_id == 1 else player1
            
            if attack.check_hit(target):
                hit_info = {
                    'attacker': attack.attacker,
                    'target': target,
                    'damage': attack.hitbox.damage,
                    'attack_type': attack.attack_type
                }
                hits.append(hit_info)
                
                # Spawn hit particles
                self._spawn_particles(target.x, target.y, attack.attack_type)
        
        return hits
    
    def render(self, screen):
        """
        Render all attacks and effects (T2.8)
        
        Args:
            screen: Pygame surface to draw on
        """
        # Render attacks
        for attack in self.active_attacks:
            attack.render(screen)
        
        # Render particle effects
        for effect in self.particle_effects:
            effect.render(screen)

