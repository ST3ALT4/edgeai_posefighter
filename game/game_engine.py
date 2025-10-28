"""
game_engine.py - UPDATED to accept pose queue
"""

import pygame
from config import *
from game.states import MenuState, BattleState, GameOverState


class GameEngine:
    """Main game engine that manages the game loop and state transitions"""
    
    def __init__(self, pose_queue=None):
        """
        Initialize the game engine
        
        Args:
            pose_queue: Multiprocessing Queue from T1 (optional)
        """
        # Setup display (T2.1)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(GAME_TITLE)
        
        # Frame rate control (T2.1)
        self.clock = pygame.time.Clock()
        self.fps = FPS
        
        # Store pose queue
        self.pose_queue = pose_queue
        
        # Game state management (T2.2)
        self.current_state = None
        self.states = {
            STATE_MENU: MenuState(self),
            STATE_BATTLE: BattleState(self),
            STATE_GAME_OVER: GameOverState(self)
        }
        
        # Start with menu state
        self.change_state(STATE_MENU)
        
        # Game running flag
        self.running = True
        
    def change_state(self, state_name):
        """Change the current game state"""
        if self.current_state:
            self.current_state.exit()
            
        self.current_state = self.states[state_name]
        self.current_state.enter()
        
    def handle_events(self):
        """Handle global events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            
            # Pass events to current state
            if self.current_state:
                self.current_state.handle_event(event)
    
    def update(self, dt):
        """Update the current game state"""
        if self.current_state:
            self.current_state.update(dt)
    
    def render(self):
        """Render the current game state"""
        self.screen.fill(BLACK)
        
        if self.current_state:
            self.current_state.render(self.screen)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop (T2.1)"""
        while self.running:
            # Calculate delta time
            dt = self.clock.tick(self.fps) / 1000.0  # Convert to seconds
            
            # Game loop phases
            self.handle_events()
            self.update(dt)
            self.render()

