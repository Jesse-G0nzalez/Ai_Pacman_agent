import numpy as np
import gymnasium as gym
import retro

class PacManRewardWrapper(gym.Wrapper):
    """
    Aggressive pellet‑collector wrapper:
      • big pellet bonus
      • small survival reward each step
      • heavy idle penalty on NOOP
      • standard time penalty & death penalty
      • extra stagnation penalty when Pac‑Man doesn't move
    """
    # raw Atari scores
    PELLET_SCORE       = 10
    POWER_PELLET_SCORE = 50
    FRUIT_SCORES       = {100, 300, 500, 700, 1000, 2000}
    GHOST_SCORES       = {200, 400, 800, 1600}

    def __init__(
        self,
        env: gym.Env,
        pellet_reward:       float = +5.0,
        power_reward:        float = +5.0,
        fruit_reward:        float = +10.0,
        ghost_reward:        float = +20.0,
        time_penalty:        float = -0.01,
        survival_reward:     float = +0.0,    # changed: removed free per‑step bonus
        idle_penalty:        float = -1.0,    # changed: heavier NOOP penalty
        death_penalty:       float = -10.0,
        level_clear_bonus:   float = +50.0,
        noop_action:         int   = 0
    ):
        super().__init__(env)
        self.pellet_reward       = pellet_reward
        self.power_reward        = power_reward
        self.fruit_reward        = fruit_reward
        self.ghost_reward        = ghost_reward
        self.time_penalty        = time_penalty
        self.survival_reward     = survival_reward
        self.idle_penalty        = idle_penalty
        self.death_penalty       = death_penalty
        self.level_clear_bonus   = level_clear_bonus
        self.noop_action         = noop_action

        # for oscillation penalty
        self.reverse_action = {
            2: 3, 3: 2,
            4: 5, 5: 4,
            6: 7, 7: 6,
            8: 9, 9: 8,
        }
        self.last_int_action = None
        self.last_pos = None  # track position for stagnation penalty

    def reset(self, **kwargs):
        self.last_int_action = None
        self.last_pos = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if isinstance(action, np.ndarray):
            int_action = int(action.argmax())
        else:
            int_action = action

        obs, raw_rew, terminated, truncated, info = self.env.step(action)
        r = 0.0

        # 1) Pellet & fruit & ghost shaping
        if raw_rew == self.PELLET_SCORE:
            r += self.pellet_reward
        elif raw_rew == self.POWER_PELLET_SCORE:
            r += self.power_reward
        elif raw_rew in self.FRUIT_SCORES:
            r += self.fruit_reward
        elif raw_rew in self.GHOST_SCORES:
            r += self.ghost_reward

        # 2) Time penalty + survival bonus
        r += self.time_penalty
        if not terminated and not truncated:
            r += self.survival_reward

        # 3) Idle penalty on NOOP
        if int_action == self.noop_action:
            r += self.idle_penalty

        # 4) Death & clear
        if terminated:
            r += self.death_penalty
        if info.get("level_complete", False):
            r += self.level_clear_bonus

        # 5) Extra stagnation penalty
        pos = info.get("pacman_position")
        if self.last_pos is not None and pos == self.last_pos:
            r += -0.2  # penalty when Pac‑Man doesn’t move
        self.last_pos = pos

        # 6) Oscillation penalty
        if (self.last_int_action is not None and
            int_action == self.reverse_action.get(self.last_int_action)):
            r += -0.05

        self.last_int_action = int_action
        return obs, r, terminated, truncated, info