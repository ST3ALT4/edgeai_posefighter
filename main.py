"""
main.py - Main entry point for Pose Fighters game
Team Member 2: Game Engine & Mechanics Specialist
"""

import pygame
import sys
from game.game_engine import GameEngine

def main():
    """Initialize Pygame and start the game"""
    pygame.init()
    
    # Create and run the game engine
    game = GameEngine()
    game.run()
    
    # Cleanup
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

