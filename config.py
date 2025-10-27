"""
config.py - Game Configuration Constants
Centralized configuration for the Pose Fighters game
"""

# Screen settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 30  # Match T1's pose detection frame rate

# Game title
GAME_TITLE = "Pose Fighters - Arena Battle"

# Colors (R, G, B)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 20, 60)
GREEN = (50, 205, 50)
BLUE = (30, 144, 255)
YELLOW = (255, 215, 0)
ORANGE = (255, 140, 0)
PURPLE = (138, 43, 226)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)

# Player settings
PLAYER_MAX_HEALTH = 100
PLAYER_WIDTH = 80
PLAYER_HEIGHT = 120
PLAYER_1_COLOR = BLUE
PLAYER_2_COLOR = RED
PLAYER_1_START_X = SCREEN_WIDTH // 4
PLAYER_2_START_X = 3 * SCREEN_WIDTH // 4
PLAYER_START_Y = SCREEN_HEIGHT - 200

# Health bar settings
HEALTH_BAR_WIDTH = 300
HEALTH_BAR_HEIGHT = 30
HEALTH_BAR_OFFSET_Y = 20
HEALTH_BAR_BORDER = 3

# Attack settings
FIREBALL_SPEED = 8
FIREBALL_DAMAGE = 15
FIREBALL_RADIUS = 20
FIREBALL_COLOR = ORANGE

LIGHTNING_DAMAGE = 25
LIGHTNING_DURATION = 15  # frames
LIGHTNING_WIDTH = 80
LIGHTNING_COLOR = YELLOW

SHIELD_DURATION = 60  # frames (2 seconds at 30fps)
SHIELD_REDUCTION = 0.5  # Damage reduction factor
SHIELD_COLOR = BLUE
SHIELD_ALPHA = 128

GROUND_POUND_DAMAGE = 20
GROUND_POUND_RADIUS = 150
GROUND_POUND_COLOR = PURPLE

ENERGY_BEAM_DAMAGE = 18
ENERGY_BEAM_SPEED = 10
ENERGY_BEAM_LENGTH = 100
ENERGY_BEAM_WIDTH = 15
ENERGY_BEAM_COLOR = GREEN

# Particle settings
PARTICLE_LIFETIME = 30  # frames
PARTICLE_SPEED_MIN = 2
PARTICLE_SPEED_MAX = 6
PARTICLE_SIZE_MIN = 2
PARTICLE_SIZE_MAX = 6
PARTICLES_PER_EFFECT = 20

# Superpower names (must match T3's classification output)
SUPERPOWERS = {
    "fireball": {"damage": FIREBALL_DAMAGE, "name": "Fireball"},
    "lightning": {"damage": LIGHTNING_DAMAGE, "name": "Lightning Strike"},
    "shield": {"damage": 0, "name": "Force Shield"},
    "ground_pound": {"damage": GROUND_POUND_DAMAGE, "name": "Ground Pound"},
    "energy_beam": {"damage": ENERGY_BEAM_DAMAGE, "name": "Energy Beam"},
}

# Communication settings (for receiving data from T1 and T3)
POSE_QUEUE_MAX_SIZE = 10
PREDICTION_QUEUE_MAX_SIZE = 5

# Game states
STATE_MENU = "menu"
STATE_BATTLE = "battle"
STATE_GAME_OVER = "game_over"

