"""
states.py - Game state classes implementing state machine pattern
Task: T2.2
"""

import pygame
from config import *
from ui.hud import HUD
from ui.menu import Menu
from entities.player import Player
from communication.pose_receiver import PoseReceiver
from combat.attack_system import AttackSystem

class State:
    """Base state class"""
    
    def __init__(self, game_engine):
        """
        Initialize state
        
        Args:
            game_engine: Reference to the main game engine
        """
        self.game = game_engine
        
    def enter(self):
        """Called when entering this state"""
        pass
        
    def exit(self):
        """Called when exiting this state"""
        pass
        
    def handle_event(self, event):
        """Handle pygame events"""
        pass
        
    def update(self, dt):
        """Update state logic"""
        pass
        
    def render(self, screen):
        """Render state visuals"""
        pass


class MenuState(State):
    """Main menu state"""
    
    def __init__(self, game_engine):
        super().__init__(game_engine)
        self.menu = Menu()
        
    def enter(self):
        """Initialize menu"""
        print("Entering Menu State")
        
    def handle_event(self, event):
        """Handle menu input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Start battle
                self.game.change_state(STATE_BATTLE)
    
    def update(self, dt):
        """Update menu"""
        self.menu.update(dt)
    
    def render(self, screen):
        """Render menu"""
        self.menu.render(screen)


class BattleState(State):
    """Main battle gameplay state"""
    
    def __init__(self, game_engine):
        super().__init__(game_engine)
        
        # Initialize components (will be created on enter)
        self.player1 = None
        self.player2 = None
        self.hud = None
        self.pose_receiver = None
        self.attack_system = None
        
    def enter(self):
        """Initialize battle"""
        print("Entering Battle State")
    
        # Create players (T2.4)
        self.player1 = Player(
            player_id=1,
            x=PLAYER_1_START_X,
            y=PLAYER_START_Y,
            color=PLAYER_1_COLOR
        )
    
        self.player2 = Player(
            player_id=2,
            x=PLAYER_2_START_X,
            y=PLAYER_START_Y,
            color=PLAYER_2_COLOR
        )
    
        # Create HUD (T2.3)
        self.hud = HUD(self.player1, self.player2)
    
        # Initialize pose receiver with game's pose queue (T2.4)
        self.pose_receiver = PoseReceiver(self.game.pose_queue)
        self.pose_receiver.start()
    
        # Initialize attack system (T2.5, T2.6)
        self.attack_system = AttackSystem()

    def exit(self):
        """Cleanup battle resources"""
        if self.pose_receiver:
            self.pose_receiver.stop()
    
    def handle_event(self, event):
        """Handle battle input"""
        # For testing: manual attack triggers with keyboard
        if event.type == pygame.KEYDOWN:
            # Player 1 controls
            if event.key == pygame.K_q:
                self.attack_system.trigger_attack(self.player1, "fireball")
            elif event.key == pygame.K_w:
                self.attack_system.trigger_attack(self.player1, "lightning")
            elif event.key == pygame.K_e:
                self.attack_system.trigger_attack(self.player1, "shield")
            
            # Player 2 controls
            elif event.key == pygame.K_u:
                self.attack_system.trigger_attack(self.player2, "fireball")
            elif event.key == pygame.K_i:
                self.attack_system.trigger_attack(self.player2, "lightning")
            elif event.key == pygame.K_o:
                self.attack_system.trigger_attack(self.player2, "shield")
    
    def update(self, dt):
        """Update battle logic"""
        # Receive pose data from T1 (T2.4)
        pose_data = self.pose_receiver.get_latest_poses()
        
        # Update player positions based on pose data
        if pose_data:
            self.player1.update_from_pose(pose_data.get('player1'))
            self.player2.update_from_pose(pose_data.get('player2'))
        
        # Get attack predictions from T3 (T2.5)
        predictions = self.pose_receiver.get_predictions()
        for player_id, move_name in predictions:
            player = self.player1 if player_id == 1 else self.player2
            self.attack_system.trigger_attack(player, move_name)
        
        # Update players
        self.player1.update(dt)
        self.player2.update(dt)
        
        # Update attack system (T2.6)
        self.attack_system.update(dt)
        
        # Collision detection and damage (T2.7)
        hits = self.attack_system.check_collisions(self.player1, self.player2)
        for hit_info in hits:
            self._apply_damage(hit_info)
        
        # Update HUD
        self.hud.update(dt)
        
        # Check win condition
        if self.player1.health <= 0 or self.player2.health <= 0:
            self.game.change_state(STATE_GAME_OVER)
    
    def _apply_damage(self, hit_info):
        """
        Apply damage from attack hits
        
        Args:
            hit_info: Dictionary with 'attacker', 'target', 'damage', 'attack_type'
        """
        target = hit_info['target']
        damage = hit_info['damage']
        
        # Apply shield reduction if active
        if target.shield_active:
            damage *= SHIELD_REDUCTION
        
        target.take_damage(damage)
    
    def render(self, screen):
        """Render battle scene"""
        # Draw background
        screen.fill(DARK_GRAY)
        
        # Draw arena floor
        pygame.draw.rect(screen, LIGHT_GRAY, 
                        (0, SCREEN_HEIGHT - 150, SCREEN_WIDTH, 150))
        
        # Draw players
        self.player1.render(screen)
        self.player2.render(screen)
        
        # Draw attacks and effects (T2.8)
        self.attack_system.render(screen)
        
        # Draw HUD (T2.3)
        self.hud.render(screen)


class GameOverState(State):
    """Game over state"""
    
    def __init__(self, game_engine):
        super().__init__(game_engine)
        self.font = None
        
    def enter(self):
        """Initialize game over screen"""
        print("Entering Game Over State")
        self.font = pygame.font.Font(None, 74)
        
    def handle_event(self, event):
        """Handle game over input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Restart battle
                self.game.change_state(STATE_BATTLE)
            elif event.key == pygame.K_m:
                # Return to menu
                self.game.change_state(STATE_MENU)
    
    def render(self, screen):
        """Render game over screen"""
        # Game over text
        text = self.font.render("GAME OVER", True, WHITE)
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        screen.blit(text, text_rect)
        
        # Instructions
        small_font = pygame.font.Font(None, 36)
        restart_text = small_font.render("Press SPACE to restart", True, WHITE)
        menu_text = small_font.render("Press M for menu", True, WHITE)
        
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        menu_rect = menu_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 90))
        
        screen.blit(restart_text, restart_rect)
        screen.blit(menu_text, menu_rect)

