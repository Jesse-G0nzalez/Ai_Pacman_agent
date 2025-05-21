import time
import numpy as np
import torch
from torchvision import transforms
import retro
from pacman_wrapper import PacManRewardWrapper
from cnn_dqn_train_old import CNN_DQN  # your model class

# —– Settings —–
CHECKPOINT_PATH = "./cnn_dqn_mspacman.pth"
N_EPISODES     = 5
FRAME_DELAY    = 0.02  # seconds between frames (None or 0 → no delay)

# same preprocessing as training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

def make_render_env():
    # render_mode='human' opens a window
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

    # load your model
    model = CNN_DQN(n_actions).to(device)
    chk = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(chk["model"])
    model.eval()

    for ep in range(1, N_EPISODES + 1):
        # Gymnasium reset returns (obs, infos)
        obs, _ = env.reset()
        state = transform(obs).unsqueeze(0).to(device)

        done = False
        total_reward = 0.0

        while not done:
            # choose action
            with torch.no_grad():
                q = model(state)                  # [1, n_actions]
                action = int(q.argmax(1).item())

            # Gymnasium step returns (obs, rew, term, trunc, infos)
            obs, reward, terminated, truncated, _ = env.step(onehot_action(action, n_buttons))
            done = terminated or truncated
            total_reward += reward

            # render & optional delay
            env.render()
            if FRAME_DELAY:
                time.sleep(FRAME_DELAY)

            # preprocess next state
            state = transform(obs).unsqueeze(0).to(device)

        print(f"Episode {ep} → Reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    run()
