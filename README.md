# Ai_Pacman_agent

A Deep Q-Learning AI agent that learns to play **Ms. Pac-Man** using **PyTorch**, **Gymnasium**, and **Stable-Retro**.

## Scope

This project demonstrates how to train an AI agent using **Deep Q-Learning** to play the classic game *Ms. Pac-Man (NES)* through a modern reinforcement learning pipeline.

The project supports:

- Training from scratch using a custom reward wrapper
- Saving and resuming models
- Running pretrained agents in the environment
- Parallelised training using vectorised environments (8 environments by default)

---

## Requirements

This project has been tested on:

- **Ubuntu 22.04+**
- **Python 3.10+**
- **VMware, WSL2, or native Ubuntu (not Windows)**
- **NVIDIA GPU (optional but recommended for faster training)**

The following must be installed:

- Python, pip, and venv
- `torch`, `torchvision`, `gymnasium`
- `stable-baselines3[extra]`
- `numpy`, `matplotlib`, `pandas`
- `cmake`, `ffmpeg`, `zlib1g-dev`, and other system dependencies
- The `stable-retro` environment (installed from source)
- A valid Ms. Pac-Man NES ROM

---

 **If your system is not set up yet**, use the dropdown below to install everything:
 <details>

  <summary><strong>Full Setup Guide (Ubuntu)</strong></summary> 
  
  This guide assumes a **fresh Ubuntu system** with no pre-installed Python environment or packages.

  ### Step 1: Install Required System Packages

  Open a terminal and run the following commands to install Python, pip, Git, and essential build tools:

  ```bash
  sudo apt update
  sudo apt upgrade -y

  sudo apt install -y python3 python3-pip python3-venv git \
                    cmake zlib1g-dev libopenmpi-dev ffmpeg

  ```
  **Checkpoint:** You should now have Python, pip, and Git available. Confirm With:
  ```bash
  python3 --version
  ```
  ---

  ### Step 2: Set Up the Project Workspace
  Create a workspace and clone this repository:

  ```bash
  mkdir -p ~/pacman_ws/src
  cd ~/pacman_ws/src

  # Clone the AI Pacman repository
  git clone https://github.com/Jesse-G0nzalez/Ai_Pacman_agent.git

  ```
  ---

  ### Step 3: Create and Activate a Virtual Environment
  We’ll isolate your dependencies with a virtual environment inside the src folder:

  ```bash
  cd ~/pacman_ws/src
  python3 -m venv venv
  source venv/bin/activate

  # Upgrade pip
  pip install --upgrade pip
  ```

  **Checkpoint:** Your shell prompt should now start with (venv).

  ---

  ### Step 4: Install Python Dependencies
  Install core libraries and reinforcement learning tools:

  ```bash
  # Machine learning and RL
  pip install torch torchvision gymnasium "stable-baselines3[extra]"

  # Utility libraries
  pip install numpy matplotlib pandas
  ```

  ---

  ### Step 5: Install Stable-Retro (for ROM-based Environments)
  Stable-Retro is a maintained fork of OpenAI's Gym Retro.

  ```bash
  cd ~/pacman_ws/src
  git clone https://github.com/Farama-Foundation/stable-retro.git
  cd stable-retro

  pip install -e .

  ```

  **Checkpoint:** Test that it works with:
  ```bash
  python -c "import retro; print(retro.__version__)"
  ```
  ---

  ### Step 6: Import Your Game ROM
  Place your ROMS (e.g., MsPacMan.nes) into:

  ~/pacman_ws/src/Ai_Pacman_agent/src/roms/ **(This has already been done for you)**
  
  Run the import script:

  ```bash
  cd ~/pacman_ws/src/Ai_Pacman_agent/src/roms
  python3 -m retro.import .
  ```
</details>


 **To learn how to train or run the agent**, open these dropdowns:



<details> 

  <summary><strong>Running and Training the Agent</strong></summary> 

   ### Step 1: Activate the Virtual Environment
  ```bash
  source ~/pacman_ws/src/venv/bin/activate
  ```
  ---
  
  ### Step 2: Navigate to the Training Script 
  ```bash
  cd ~/pacman_ws/src/Ai_Pacman_agent/src 
  ```
  ---

  ### Step 3: Start Training 
  ```bash 
  python cnn_dqn_train_old.py 
  ``` 
  The agent will start training across multiple environments. Training progress will be logged to the `logs/` folder. 
  
  **Checkpoint:** If everything is configured correctly, you should start seeing episode rewards and saved model checkpoints in the `models/` folder. 

  ---

   ### Step 4: Running the Agent 

  **You do not need to train the agent every time before running the agent You can run this if you only wish to run old models that you have saved**
  
  ```bash 
  python run_saved.py 
  ``` 
  
  **Checkpoint:** You should see the agent within the environment playing Pac-Man. 
  
</details>



