import os
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torchvision import transforms
import retro
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from pacman_wrapper import PacManRewardWrapper

# --- Parameters ---
NUM_ENVS = 8
SAVE_FREQ = 50          # episodes
TARGET_UPDATE_FREQ = 1000  # steps
SAVE_PATH = "./models/cnn_dqn_mspacman.pth"
LOG_PATH = "./logs/training_log.csv"

# Hyperparameters
GAMMA = 0.99
LR = 5e-5 
BATCH_SIZE = 32
MEMORY_SIZE = 100_000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 200_000  # steps for decay

# Preprocessing transform (84x84 grayscale)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

# --- Network definitions ---
class CNN_DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(args)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    def __len__(self):
        return len(self.buffer)

# Preprocess batch of observations
def preprocess(obs_batch, device):
    tensors = [transform(o).unsqueeze(0) for o in obs_batch]
    return torch.cat(tensors, dim=0).to(device)

# Epsilon-greedy action selection
def select_actions(states, steps_done, policy_net, n_actions, device):
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
    actions = []
    for s in states:
        if random.random() < eps:
            actions.append(random.randrange(n_actions))
        else:
            with torch.no_grad():
                q = policy_net(s.unsqueeze(0).to(device))
                actions.append(q.argmax(1).item())
    return actions

# Environment factory without FrameSkip
def make_env_fn():
    def _thunk():
        env = retro.make("MsPacMan-Nes", render_mode="none")
        env = PacManRewardWrapper(env)
        return env
    return _thunk

# Main training loop
def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Vectorized environments
    envs = AsyncVectorEnv([make_env_fn() for _ in range(NUM_ENVS)])
    n_actions = envs.single_action_space.n
    num_buttons = retro.make("MsPacMan-Nes").num_buttons

    # Models, optimizer, replay memory
    policy_net = CNN_DQN(n_actions).to(device)
    target_net = CNN_DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    # Resume training if checkpoint exists
    episode = 0
    steps_done = 0
    last_loss = None
    rewards_buffer = deque(maxlen=100)

    if os.path.exists(SAVE_PATH):
        chk = torch.load(SAVE_PATH, map_location=device)
        policy_net.load_state_dict(chk['model'])
        optimizer.load_state_dict(chk['optimizer'])
        print("Resumed training from saved checkpoint.")
    else:
        print("Starting fresh training.")

    # CSV log setup
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Episode', 'TotalReward'])
    else:
        with open(LOG_PATH, newline='') as f:
            rows = list(csv.reader(f))
            episode = int(rows[-1][0]) if len(rows) > 1 else 0

    start_time = time.time()
    obs, _ = envs.reset()
    states = preprocess(obs, device)
    total_rewards = [0.0] * NUM_ENVS

    try:
        while True:
            actions = select_actions(states, steps_done, policy_net, n_actions, device)
            onehots = np.zeros((NUM_ENVS, num_buttons), dtype=np.int8)
            for i, a in enumerate(actions): onehots[i][a] = 1

            next_obs, rewards, terminated, truncated, _ = envs.step(onehots)
            next_states = preprocess(next_obs, device)
            dones = np.logical_or(terminated, truncated)

            # Store transitions
            for i in range(NUM_ENVS):
                memory.push(
                    states[i].cpu().numpy(),
                    actions[i],
                    rewards[i],
                    next_states[i].cpu().numpy(),
                    dones[i]
                )
                total_rewards[i] += rewards[i]

            states = next_states
            steps_done += NUM_ENVS

            # Training step
            if len(memory) >= BATCH_SIZE:
                s, a, r, s2, d = memory.sample(BATCH_SIZE)
                s  = torch.tensor(np.stack(s)).to(device)
                a  = torch.tensor(a).long().unsqueeze(1).to(device)
                r  = torch.tensor(r, dtype=torch.float32).to(device)
                s2 = torch.tensor(np.stack(s2)).to(device)
                d  = torch.tensor(d, dtype=torch.float32).to(device)

                q     = policy_net(s).gather(1, a)
                max_q = target_net(s2).max(1)[0].detach()
                target= r + GAMMA * max_q * (1 - d)

                loss = nn.functional.mse_loss(q.squeeze(), target)
                last_loss = loss.item()
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Update target network
            if steps_done % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Episode termination handling
            for i in range(NUM_ENVS):
                if dones[i]:
                    episode += 1
                    with open(LOG_PATH, 'a', newline='') as f:
                        csv.writer(f).writerow([episode, total_rewards[i]])
                    print(f"Episode {episode} - Total Reward: {total_rewards[i]}")
                    rewards_buffer.append(total_rewards[i])
                    total_rewards[i] = 0.0

                    if episode % SAVE_FREQ == 0:
                        torch.save({'model': policy_net.state_dict(), 'optimizer': optimizer.state_dict()}, SAVE_PATH)
                        print(f"💾 Saved model at episode {episode}")

                    if episode % 500 == 0:
                        elapsed = time.time() - start_time
                        fps = int(steps_done / elapsed)
                        avg_reward = np.mean(rewards_buffer) if rewards_buffer else 0
                        print("-"*42)
                        print(f"| Episode            | {episode}")
                        print(f"| Total Steps        | {steps_done}")
                        print(f"| Time Elapsed (s)   | {int(elapsed)}")
                        print(f"| Steps/sec (FPS)    | {fps}")
                        print(f"| Avg Reward (100 ep)| {avg_reward:.2f}")
                        print(f"| Last Loss          | {last_loss:.4f}" if last_loss else "| Last Loss          | N/A")
                        print("-"*42)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        torch.save({'model': policy_net.state_dict(), 'optimizer': optimizer.state_dict()}, SAVE_PATH)
        print("Saved latest model before exit.")

if __name__ == "__main__":
    main()