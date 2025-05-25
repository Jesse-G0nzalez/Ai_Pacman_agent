import os
import time
import numpy as np
import torch
from torchvision import transforms
import retro
from pacman_wrapper import PacManRewardWrapper
from cnn_dqn_train_old import CNN_DQN  # your model class

# —— Model selection prompt ——
print("Select model to run:")
print("  [1] DQN         (cnn_dqn_mspacman.pth)")
print("  [2] Double DQN  (double_dqn_mspacman.pth)")
choice = input("Enter 1 or 2 (default=1): ").strip()

if choice == "2":
    CHECKPOINT_PATH = os.path.join("models", "double_dqn_mspacman.pth")
else:
    CHECKPOINT_PATH = os.path.join("models", "cnn_dqn_mspacman.pth")

print(f"→ Loading checkpoint: {CHECKPOINT_PATH}")

# —— rest of your code unchanged —— 
N_EPISODES     = 5
FRAME_DELAY    = 0.02  # seconds between frames

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

def make_render_env():
    base = retro.make("MsPacMan-Nes", render_mode="human")
    return PacManRewardWrapper(base)

def onehot_action(a, n_buttons):
    vec = np.zeros(n_buttons, dtype=np.int8)
    vec[a] = 1
    return vec

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    env = make_render_env()
    n_buttons = env.unwrapped.num_buttons
    n_actions = env.action_space.n

    model = CNN_DQN(n_actions).to(device)
    chk = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(chk["model"])
    model.eval()

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        state = transform(obs).unsqueeze(0).to(device)

        done = False
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                q = model(state)
                action = int(q.argmax(1).item())

            obs, reward, terminated, truncated, _ = env.step(onehot_action(action, n_buttons))
            done = terminated or truncated
            total_reward += reward

            env.render()
            if FRAME_DELAY:
                time.sleep(FRAME_DELAY)

            state = transform(obs).unsqueeze(0).to(device)

        print(f"Episode {ep} → Reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    run()
