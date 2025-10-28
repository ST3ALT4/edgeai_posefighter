"""
test_t2.py - Test T2 (Game Engine) Standalone
Run this to test game with keyboard controls only
"""

import pygame
from game.game_engine import GameEngine


def test_t2():
    print("\n" + "=" * 60)
    print("TESTING T2: GAME ENGINE")
    print("=" * 60)
    print("\nStarting game with keyboard controls...")
    print("\nWhat to check:")
    print("  ✓ Game window opens (1280x720)")
    print("  ✓ Menu screen appears")
    print("  ✓ Press SPACE → Battle starts")
    print("  ✓ 2 players visible")
    print("  ✓ Health bars at top")
    print("\nKeyboard Controls:")
    print("  Player 1:")
    print("    Q - Block")
    print("    W - Fireball")
    print("    E - Lightning")
    print("\n  Player 2:")
    print("    U - Block")
    print("    I - Fireball")
    print("    O - Lightning")
    print("\n  ESC - Quit")
    print("=" * 60)
    
    # Initialize Pygame
    pygame.init()
    
    # Create game engine (no pose detection)
    game = GameEngine()
    
    # Run game
    try:
        game.run()
        print("\n✅ SUCCESS: Game ran without errors!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()


if __name__ == "__main__":
    test_t2()
