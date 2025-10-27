# Pose Fighters - Team Member 2 Quick Start Guide

## 🚀 Quick Setup (5 Minutes)

### 1. Install Python Packages
```bash
pip install pygame opencv-python numpy
```

### 2. Create Project Structure
```bash
git clone <whaterver the repo name is>
```
### 4. Run
```bash
python main.py
```

---

## 🎮 Controls Reference

### Menu Screen
- **SPACE**: Start game
- **ESC**: Quit

### In-Game (Testing Mode)
**Player 1 (Blue - Left Side):**
- **Q**: Fireball
- **W**: Lightning Strike
- **E**: Force Shield

**Player 2 (Red - Right Side):**
- **U**: Fireball
- **I**: Lightning Strike
- **O**: Force Shield

**Global:**
- **ESC**: Quit to desktop

### Game Over Screen
- **SPACE**: Restart battle
- **M**: Return to menu
- **ESC**: Quit

---

## 📋 Task Completion Summary

| Task | Status | Module | Description |
|------|--------|--------|-------------|
| **T2.1** | ✅ | `game_engine.py` | Game window, FPS control, main loop |
| **T2.2** | ✅ | `states.py` | State machine (Menu/Battle/GameOver) |
| **T2.3** | ✅ | `hud.py`, `health_bar.py` | Health bars and UI overlay |
| **T2.4** | ✅ | `player.py`, `pose_receiver.py` | Player sprites + pose integration |
| **T2.5** | ✅ | `attack_system.py` | Classification listener |
| **T2.6** | ✅ | `attack_system.py` | Attack spawning (5 types) |
| **T2.7** | ✅ | `hitbox.py`, `damage_calculator.py` | Collision detection + damage |
| **T2.8** | ✅ | `particle_effects.py`, `animations.py` | Visual effects + polish |

**All 8 tasks completed!** ✨

---

## 🔗 Integration with T1 (Pose Detection)

### Data Format Expected from T1
```python
pose_data = {
    'player1': {
        'hip_x': 0.45,  # Normalized 0-1 (horizontal position)
        'hip_y': 0.60,  # Normalized 0-1 (vertical position)
        # Add more landmarks as needed
    },
    'player2': {
        'hip_x': 0.72,
        'hip_y': 0.58,
    }
}
```

### Integration Code
```python
# In T1's pose detection process:
from multiprocessing import Queue

# Create shared queue
pose_queue = Queue(maxsize=10)

# Send pose data every frame
pose_queue.put(pose_data)

# In T2's main.py, pass queue to game:
from game.game_engine import GameEngine

game = GameEngine(pose_queue=pose_queue)
game.run()
```

---

## 🔗 Integration with T3 (Pose Classification)

### Data Format Expected from T3
```python
# T3 sends predictions as (player_id, move_name) tuples
prediction = (1, "fireball")  # Player 1 detected fireball pose
prediction = (2, "shield")     # Player 2 detected shield pose
```

### Valid Move Names
- `"fireball"` - Projectile attack
- `"lightning"` - Vertical strike
- `"shield"` - Defensive buff
- `"ground_pound"` - Area attack
- `"energy_beam"` - Laser beam

### Integration Code
```python
# In T3's classification process:
from multiprocessing import Queue

# Create shared queue
prediction_queue = Queue(maxsize=5)

# Send prediction when pose detected
if confidence > threshold:
    prediction_queue.put((player_id, move_name))

# In T2's main.py:
game = GameEngine(
    pose_queue=pose_queue,
    prediction_queue=prediction_queue
)
```

---

## 🎯 Superpower Mechanics

| Superpower | Type | Damage | Special Effect |
|------------|------|--------|----------------|
| **Fireball** | Projectile | 15 | Horizontal movement |
| **Lightning Strike** | Area | 25 | Instant vertical strike |
| **Force Shield** | Buff | 0 | 50% damage reduction |
| **Ground Pound** | Area | 20 | Expanding shockwave |
| **Energy Beam** | Projectile | 18 | Fast laser beam |

---

## 🐛 Troubleshooting

### Game won't start
```bash
# Check Python version (need 3.8+)
python --version

# Reinstall dependencies
pip install --upgrade pygame opencv-python numpy
```

### Import errors
```bash
# Make sure __init__.py files exist
touch game/__init__.py entities/__init__.py combat/__init__.py visuals/__init__.py ui/__init__.py communication/__init__.py

# Run from project root
cd pose_fighters_team2
python main.py
```

### Low FPS
```python
# In config.py, reduce particle count:
PARTICLES_PER_EFFECT = 10  # Default is 20
```

### No attacks spawning
```bash
# Check if keyboard controls work (testing mode)
# Press Q, W, E for Player 1
# Press U, I, O for Player 2

# In production, verify T3 predictions are being sent
```

---

## 📊 Performance Benchmarks

**Target Performance:**
- **FPS**: Stable 30 FPS
- **Frame Time**: ~33ms per frame
- **Memory**: < 200 MB
- **CPU**: Single core, ~30-40% usage

**Stress Test:**
- 10+ active attacks
- 200+ particles on screen
- Should maintain 30 FPS

---

## 🔧 Configuration Tweaks

### Adjust Game Balance
Edit `config.py`:
```python
# Make game faster
FIREBALL_SPEED = 12  # Default: 8

# Increase damage
FIREBALL_DAMAGE = 20  # Default: 15

# Longer shield duration
SHIELD_DURATION = 90  # Default: 60 (frames)
```

### Visual Quality
```python
# More particles
PARTICLES_PER_EFFECT = 30  # Default: 20

# Particle lifetime
PARTICLE_LIFETIME = 45  # Default: 30 (frames)
```

### Screen Resolution
```python
# Larger window
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Adjust UI positions accordingly
```

---

## 📝 Code Structure at a Glance

```
main.py
  └─> GameEngine (game_engine.py)
       ├─> States (states.py)
       │    ├─> MenuState
       │    ├─> BattleState
       │    │    ├─> Player (player.py) x2
       │    │    ├─> HUD (hud.py)
       │    │    │    └─> HealthBar (health_bar.py) x2
       │    │    ├─> AttackSystem (attack_system.py)
       │    │    │    ├─> Attack classes
       │    │    │    └─> ParticleEffect (particle_effects.py)
       │    │    └─> PoseReceiver (pose_receiver.py)
       │    │         ├─> From T1: Pose data
       │    │         └─> From T3: Predictions
       │    └─> GameOverState
       └─> Clock (30 FPS)
```

---

## ✅ Pre-Integration Checklist

Before integrating with T1 and T3:

- [ ] Game runs standalone without errors
- [ ] All 5 attack types spawn correctly (test with keyboard)
- [ ] Health bars update when damage is taken
- [ ] State transitions work (Menu → Battle → GameOver)
- [ ] Particle effects render properly
- [ ] FPS stays stable at 30
- [ ] No memory leaks after 5+ minutes of gameplay

---

## 🎓 Learning Outcomes

By completing this module, you've implemented:

1. **Game Loop Architecture** - Frame-rate independent updates
2. **State Machine Pattern** - Clean state management
3. **Entity-Component System** - Modular game objects
4. **Collision Detection** - Hitbox-based physics
5. **Particle Systems** - Real-time visual effects
6. **Event-Driven Programming** - Attack triggers from external input
7. **Inter-Process Communication** - Queue-based data exchange
8. **Performance Optimization** - 30 FPS with multiple effects

---

## 📚 Additional Resources

### Pygame Documentation
- https://www.pygame.org/docs/

### Multiprocessing in Python
- https://docs.python.org/3/library/multiprocessing.html

### Game Design Patterns
- State Pattern
- Component Pattern
- Object Pool Pattern

---

**Status**: ✅ All tasks complete and ready for integration!  
**Next Steps**: Integrate with T1 (Pose Detection) and T3 (Classification)  
**Contact**: Team Member 2 - Game Engine Specialist
