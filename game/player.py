import pygame
import time

MAX_HEALTH = 10
BLOCK_REDUCTION = 0.5
ATTACK_COOLDOWN = 1.0
class Player:
    """Fighting game player character"""
    
    def __init__(self, player_id: int, x: int, y: int):
        """
        Initialize player
        
        Args:
            player_id: 0 or 1 (Player 1 or Player 2)
            x, y: Screen position
        """
        self.player_id = player_id
        self.x = x
        self.y = y
        self.health = MAX_HEALTH
        self.color = PLAYER_COLORS[player_id]
        
        # State
        self.current_move = 'idle'
        self.is_blocking = False
        self.last_attack_time = 0
        
        # Combo tracking
        self.combo_count = 0
        self.last_hit_time = 0
        
        # Character dimensions
        self.width = 60
        self.height = 100
        
        # Animation state
        self.is_attacking = False
        self.attack_frame = 0
        
    def take_damage(self, damage: int):
        """Apply damage to player"""
        if self.is_blocking:
            damage = int(damage * BLOCK_REDUCTION)
        
        self.health = max(0, self.health - damage)
        
        if self.health == 0:
            return True  # KO
        return False
    
    def execute_move(self, move_name: str, opponent: 'Player') -> bool:
        """
        Execute a move
        
        Args:
            move_name: Name of the move
            opponent: Opponent player
            
        Returns:
            True if move was executed successfully
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_attack_time < ATTACK_COOLDOWN:
            return False
        
        self.current_move = move_name
        
        # Update blocking state
        if move_name == 'block':
            self.is_blocking = True
            return True
        else:
            self.is_blocking = False
        
        # Check if it's an attack move
        if move_name in MOVE_DAMAGE and MOVE_DAMAGE[move_name] > 0:
            # Deal damage to opponent
            damage = MOVE_DAMAGE[move_name]
            ko = opponent.take_damage(damage)
            
            # Update combo
            if current_time - self.last_hit_time < COMBO_WINDOW:
                self.combo_count += 1
            else:
                self.combo_count = 1
            
            self.last_hit_time = current_time
            self.last_attack_time = current_time
            self.is_attacking = True
            self.attack_frame = 0
            
            return True
        
        return False
    
    def update(self):
        """Update player state"""
        # Update attack animation
        if self.is_attacking:
            self.attack_frame += 1
            if self.attack_frame > 15:  # Attack animation duration
                self.is_attacking = False
                self.attack_frame = 0
    
    def draw(self, screen: pygame.Surface):
        """Draw player on screen"""
        # Draw character body
        body_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
        # Add attack animation effect
        if self.is_attacking:
            # Pulsing effect during attack
            pulse = abs(self.attack_frame - 7.5) / 7.5
            color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in self.color)
        else:
            color = self.color
        
        # Draw body
        pygame.draw.rect(screen, color, body_rect, border_radius=5)
        
        # Draw head
        head_radius = 20
        head_x = self.x + self.width // 2
        head_y = self.y - 10
        pygame.draw.circle(screen, color, (head_x, head_y), head_radius)
        
        # Draw arms (simple rectangles)
        arm_width = 15
        arm_height = 40
        # Left arm
        left_arm = pygame.Rect(self.x - arm_width, self.y + 20, arm_width, arm_height)
        pygame.draw.rect(screen, color, left_arm, border_radius=3)
        # Right arm
        right_arm = pygame.Rect(self.x + self.width, self.y + 20, arm_width, arm_height)
        pygame.draw.rect(screen, color, right_arm, border_radius=3)
        
        # Draw blocking indicator
        if self.is_blocking:
            shield_rect = pygame.Rect(self.x - 10, self.y + 20, self.width + 20, self.height - 20)
            pygame.draw.rect(screen, (100, 100, 255), shield_rect, 3, border_radius=5)
        
        # Draw health bar above character
        self._draw_health_bar(screen)
        
        # Draw current move text
        font = pygame.font.Font(None, 24)
        move_color = YELLOW if self.is_attacking else WHITE
        text = font.render(self.current_move.upper(), True, move_color)
        screen.blit(text, (self.x - 10, self.y - 45))
        
        # Draw combo counter
        if self.combo_count > 1:
            combo_font = pygame.font.Font(None, 28)
            combo_text = combo_font.render(f"x{self.combo_count} COMBO!", True, ORANGE)
            screen.blit(combo_text, (self.x - 10, self.y - 70))
    
    def _draw_health_bar(self, screen: pygame.Surface):
        """Draw health bar above character"""
        bar_width = 80
        bar_height = 10
        bar_x = self.x - 10
        bar_y = self.y - 130
        
        # Background (dark red)
        pygame.draw.rect(screen, (100, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        
        # Health (green to red gradient based on health)
        health_width = int(bar_width * (self.health / MAX_HEALTH))
        if self.health > 60:
            health_color = GREEN
        elif self.health > 30:
            health_color = YELLOW
        else:
            health_color = RED
        
        pygame.draw.rect(screen, health_color, (bar_x, bar_y, health_width, bar_height))
        
        # Border
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Health text
        font = pygame.font.Font(None, 20)
        health_text = font.render(f"{int(self.health)}", True, WHITE)
        text_rect = health_text.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
        screen.blit(health_text, text_rect)
    
    def reset(self):
        """Reset player state"""
        self.health = MAX_HEALTH
        self.current_move = 'idle'
        self.is_blocking = False
        self.combo_count = 0
        self.is_attacking = False
        self.attack_frame = 0
