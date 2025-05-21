# Ai_Pacman_agent

A Deep Q-Learning AI agent that learns to play Pac-Man using PyTorch and the Gymnasium environment.


## Full Setup Instructions (Ubuntu)

These steps assume you're starting with a fresh Ubuntu installation.


### Step 1: Install Required System Packages

Open a terminal and run:

```bash
sudo apt update
sudo apt upgrade -y

# Install Python, pip, venv, Git, and basic build tools
sudo apt install -y python3 python3-pip python3-venv git cmake zlib1g-dev libopenmpi-dev ffmpeg

mkdir -p ~/pacman_ws/src
cd ~/pacman_ws/src

# Clone this repository
git clone https://github.com/Jesse-G0nzalez/Ai_Pacman_agent.git

cd ~/pacman_ws/src
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

# Core ML libraries
pip install torch torchvision

# RL and environment libraries
pip install gymnasium

# Utility libraries
pip install numpy matplotlib pandas

pip install "stable-baselines3[extra]"


# Install stable-retro
cd ~/pacman_ws/src
git clone https://github.com/Farama-Foundation/stable-retro.git
cd stable-retro
pip install -e .

cd ~/pacman_ws/src/Ai_Pacman_agent/roms
python3 -m retro.import .

cd ~/pacman_ws/src/Ai_Pacman_agent
python cnn_dqn_train.py

