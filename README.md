
Ai_Pacman_agent
A Deep Q-Learning AI agent that learns to play Ms. Pac-Man using PyTorch, Gymnasium, and Stable-Retro.

Full Setup Guide (Ubuntu)
This guide assumes a fresh Ubuntu system with no pre-installed Python environment or packages.

Step 1: Install Required System Packages
Open a terminal and run the following commands to install Python, pip, Git, and essential build tools:

sudo apt update
sudo apt upgrade -y

sudo apt install -y python3 python3-pip python3-venv git \
                    cmake zlib1g-dev libopenmpi-dev ffmpeg

Checkpoint: You should now have Python, pip, and Git available. Run python3 --version to confirm.

Step 2: Set Up the Project Workspace
Create a workspace and clone this repository:

mkdir -p ~/pacman_ws/src
cd ~/pacman_ws/src

# Clone the AI Pacman repository
git clone https://github.com/Jesse-G0nzalez/Ai_Pacman_agent.git


Step 3: Create and Activate a Virtual Environment
We’ll isolate your dependencies with a virtual environment inside the src folder:

cd ~/pacman_ws/src
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

Checkpoint: Your shell prompt should now start with (venv).

Step 4: Install Python Dependencies
Install core libraries and reinforcement learning tools:

# Machine learning and RL
pip install torch torchvision gymnasium "stable-baselines3[extra]"

# Utility libraries
pip install numpy matplotlib pandas

Checkpoint: You should now be ready to run environments and models.

Step 5: Install Stable-Retro (for ROM-based Environments)
Stable-Retro is a maintained fork of OpenAI's Gym Retro.


cd ~/pacman_ws/src
git clone https://github.com/Farama-Foundation/stable-retro.git
cd stable-retro

pip install -e .

Checkpoint: Test that it works with:
python -c "import retro; print(retro.__version__)"

Step 6: Import Your Game ROM
Place your Ms. Pac-Man NES ROM (e.g., MsPacMan.nes) into:

~/pacman_ws/src/Ai_Pacman_agent/src/roms/ (This has already been done for you)


Run the import script:
cd ~/pacman_ws/src/Ai_Pacman_agent/src/roms
python3 -m retro.import .



Step 7: Train the Agent
Navigate to the training script and start training:

cd ~/pacman_ws/src/Ai_Pacman_agent/src
python cnn_dqn_train.py

Checkpoint: If everything is set up, training should begin and logs will be written to the logs/ folder.

