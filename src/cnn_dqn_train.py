import retro  # Import Stable Retro for environments
import gymnasium as gym  # Import Gymnasium API
import numpy as np  # Numerical operations
import random  # Random sampling
from collections import deque  # Efficient buffer for replay
import torch  # PyTorch core
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional API for activations
import torch.optim as optim  # Optimizers
from gymnasium.vector import AsyncVectorEnv  # Async vectorized envs
from pacman_wrapper import PacManRewardWrapper  # Custom reward shaping wrapper
import os
import time
import csv

class QNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_shape, n_actions, device):
        self.device = device
        self.policy_net = QNetwork(input_shape, n_actions).to(device)
        self.target_net = QNetwork(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 1e-6

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(n_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        expected_q = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    def make_env():
        def _init():
            env = retro.make(game='MsPacMan-Nes', use_restricted_actions=retro.Actions.DISCRETE, render_mode=None)
            env = PacManRewardWrapper(env)
            return env
        return _init

    num_envs = 8
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])

    input_shape = (1, 84, 84)
    dummy_env = make_env()()
    n_actions = dummy_env.action_space.n
    dummy_env.close()

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    SAVE_PATH = "./models/dqn_checkpoint.pth"
    LOG_PATH = "./logs/episode_rewards.csv"

    agent = DQNAgent(input_shape, n_actions, device)
    steps_done = 0
    start_time = time.time()
    rewards_buffer = deque(maxlen=100)

    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        agent.policy_net.load_state_dict(checkpoint['model'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        print("✅ Resumed training from checkpoint")
        resumed = True
    else:
        print("🚀 Starting fresh training.")
        resumed = False

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Episode', 'TotalReward'])

    num_episodes = 10000
    target_update = 10

    for episode in range(num_episodes):
        obs_batch, _ = envs.reset()
        obs_batch = np.mean(obs_batch, axis=3, keepdims=True)
        obs_batch = np.array([np.resize(obs, input_shape) for obs in obs_batch])

        total_rewards = [0.0 for _ in range(num_envs)]
        dones = [False for _ in range(num_envs)]

        while not all(dones):
            for i in range(num_envs):
                if not dones[i]:
                    action = agent.select_action(obs_batch[i])
                    next_obs, reward, terminated, truncated, _ = envs.step([action]*num_envs)
                    next_obs = np.mean(next_obs, axis=3, keepdims=True)
                    next_obs = np.array([np.resize(o, input_shape) for o in next_obs])
                    agent.replay_buffer.push(obs_batch[i], action, reward[i], next_obs[i], terminated[i])
                    obs_batch[i] = next_obs[i]
                    total_rewards[i] += reward[i]
                    dones[i] = terminated[i] or truncated[i]
                    agent.optimize_model()

        if episode % target_update == 0:
            agent.update_target_network()

        avg_reward = np.mean(total_rewards)
        rewards_buffer.append(avg_reward)

        with open(LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, r in enumerate(total_rewards):
                ep_num = episode * num_envs + i + 1
                writer.writerow([ep_num, r])

        for i, r in enumerate(total_rewards):
            ep_num = episode * num_envs + i + 1
            print(f"Episode {ep_num} - Total Reward: {r:.2f}")

        if episode % 500 == 0:
            elapsed = time.time() - start_time
            fps = int(steps_done / elapsed)
            print("-" * 42)
            print(f"| Episode            | {episode}")
            print(f"| Total Steps        | {steps_done}")
            print(f"| Time Elapsed (s)   | {int(elapsed)}")
            print(f"| Steps/sec (FPS)    | {fps}")
            print(f"| Avg Reward (100 ep)| {np.mean(rewards_buffer):.2f}")
            print("-" * 42)

        if episode % 50 == 0:
            torch.save({'model': agent.policy_net.state_dict(), 'optimizer': agent.optimizer.state_dict()}, SAVE_PATH)
            print(f"💾 Saved model at episode {episode}")

    envs.close()