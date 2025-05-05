# 🕹️ Super Mario Reinforcement Learning Agent

A reinforcement learning (RL) agent that learns to play **Super Mario** using the `super-mario-game 0.0.1` package. The agent learns to **move**, **jump**, **avoid enemies**, and **collect coins** through trial and error, using **rewards for progress and penalties for mistakes**.

---
<p align="center">
  <img src="https://github.com/dohaaymann/Super-Mario-With-Reinforcement-Learning/raw/main/super%20mario.gif" alt="Training Preview" width="600"/>
</p>

## 🎮 Features

- Custom environment built with `super-mario-game`
- Screen capture and OCR-based game state detection (using `mss` and `pytesseract`)
- Keyboard simulation with `pydirectinput`
- Reward-based learning for:

  - Progress through levels
  - Coin collection
  - Avoiding enemies
- Compatible with **Windows only**

---

## 🧠 Reinforcement Learning Setup

- **Environment**: `SuperMarioGame`
- **Agent**: Deep Q-Network (DQN)
- **State**: Processed screen frames
- **Actions**: Simulated keyboard presses (e.g., left, right, jump)
- **Rewards**:
  - ✅ +1 for moving forward
  - 💰 +5 for collecting a coin
  - ❌ -10 for hitting an enemy

---

