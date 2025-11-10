# ppo.py
"""
PPO (Proximal Policy Optimization) with Centralized Critic
Implements CTDE (Centralized Training, Decentralized Execution)
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RolloutBuffer:
    """
    On-policy trajectory storage for 2 agents + centralized critic.
    Stores:
      - per-agent obs, actions, log_probs, rewards
      - joint obs for the centralized critic
      - centralized values and dones
    """

    def __init__(self):
        self.clear()

    def add(self, obs_pair, joint_obs, actions, log_probs, rewards, value, done):
        # obs_pair: [obs0, obs1] (np arrays)
        for i in range(2):
            self.observations[i].append(np.asarray(obs_pair[i], dtype=np.float32))
            self.actions[i].append(int(actions[i]))
            self.log_probs[i].append(float(log_probs[i]))
            self.rewards[i].append(float(rewards[i]))
        self.joint_observations.append(np.asarray(joint_obs, dtype=np.float32))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def get(self):
        return {
            "observations": [np.stack(self.observations[i], axis=0) for i in range(2)],
            "actions": [np.asarray(self.actions[i], dtype=np.int64) for i in range(2)],
            "log_probs": [
                np.asarray(self.log_probs[i], dtype=np.float32) for i in range(2)
            ],
            "rewards": [
                np.asarray(self.rewards[i], dtype=np.float32) for i in range(2)
            ],
            "joint_observations": np.stack(self.joint_observations, axis=0).astype(
                np.float32
            ),
            "values": np.asarray(self.values, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
        }

    def clear(self):
        self.observations = [[], []]
        self.actions = [[], []]
        self.log_probs = [[], []]
        self.rewards = [[], []]
        self.joint_observations = []
        self.values = []
        self.dones = []

    def __len__(self):
        return len(self.dones)


@dataclass
class PPO:
    actors: list
    critic: nn.Module
    hp: object
    device: str = "cpu"

    def __post_init__(self):
        actor_params = list(self.actors[0].parameters()) + list(
            self.actors[1].parameters()
        )
        self.actor_optimizer = optim.Adam(actor_params, lr=self.hp.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp.lr)
        self.buffer = RolloutBuffer()
        self.update_step = 0
        self._team_adv_checked = False

    # ------------- Acting -------------

    @torch.no_grad()
    def select_actions(self, observations):
        """
        observations: [obs0, obs1] each (96,)
        Returns: actions [a0,a1], log_probs [lp0,lp1], entropies [H0,H1], value
        """
        actions, logps, ents = [], [], []
        obs_tensors = []
        for i, actor in enumerate(self.actors):
            obs_tensor = torch.as_tensor(
                observations[i], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action, logp, ent = actor.get_action(obs_tensor)
            actions.append(int(action.item()))
            logps.append(float(logp.item()))
            ents.append(float(ent.item()))
            obs_tensors.append(obs_tensor)

        joint_obs = np.concatenate(observations, axis=0)
        joint_obs_t = torch.as_tensor(
            joint_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        value = float(self.critic(joint_obs_t).item())
        return actions, logps, ents, value, joint_obs

    # ------------- Learning -------------

    @staticmethod
    def _compute_gae(rewards, values, dones, next_value, gamma, lam):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * nv * (1.0 - dones[t]) - values[t]
            last = delta + gamma * lam * (1.0 - dones[t]) * last
            adv[t] = last
        returns = adv + values
        return adv, returns

    def update(self, next_obs_pair):
        """
        Perform PPO updates using a single team advantage shared by both actors,
        and a centralized critic trained on team returns.
        """
        data = self.buffer.get()

        # Next value for GAE
        with torch.no_grad():
            joint_next = np.concatenate(next_obs_pair, axis=0)
            joint_next_t = torch.as_tensor(
                joint_next, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            next_value = float(self.critic(joint_next_t).item())

        # Team rewards and single GAE
        team_rewards = data["rewards"][0] + data["rewards"][1]

        # One-time assertions to guard CTDE wiring
        if not self._team_adv_checked:
            assert (
                data["rewards"][0].shape == data["rewards"][1].shape
            ), "Per-agent rewards must align over time"
            assert (
                data["values"].ndim == 1
                and data["dones"].ndim == 1
                and data["values"].shape[0] == len(team_rewards)
            )
            self._team_adv_checked = True

        adv, rets = self._compute_gae(
            rewards=team_rewards,
            values=data["values"],
            dones=data["dones"],
            next_value=next_value,
            gamma=self.hp.gamma,
            lam=self.hp.gae_lambda,
        )
        # Normalize single team advantage once
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # To tensors
        obs_t = [
            torch.as_tensor(
                data["observations"][i], dtype=torch.float32, device=self.device
            )
            for i in range(2)
        ]
        act_t = [
            torch.as_tensor(data["actions"][i], dtype=torch.int64, device=self.device)
            for i in range(2)
        ]
        oldlp_t = [
            torch.as_tensor(
                data["log_probs"][i], dtype=torch.float32, device=self.device
            )
            for i in range(2)
        ]
        joint_t = torch.as_tensor(
            data["joint_observations"], dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(rets, dtype=torch.float32, device=self.device)

        B = len(data["dones"])
        idx = np.arange(B)

        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(self.hp.ppo_epochs):
            np.random.shuffle(idx)
            for s in range(0, B, self.hp.minibatch_size):
                mb = idx[s : s + self.hp.minibatch_size]
                # Actor updates (both share the same team advantage)
                total_actor_loss = 0.0
                total_ent = 0.0
                for i in range(2):
                    mb_obs = obs_t[i][mb]
                    mb_act = act_t[i][mb]
                    mb_oldlp = oldlp_t[i][mb]
                    new_lp, ent = self.actors[i].evaluate_actions(mb_obs, mb_act)
                    ratio = torch.exp(new_lp - mb_oldlp)
                    mb_adv = adv_t[mb].detach()
                    surr1 = ratio * mb_adv
                    surr2 = (
                        torch.clamp(
                            ratio, 1 - self.hp.clip_epsilon, 1 + self.hp.clip_epsilon
                        )
                        * mb_adv
                    )
                    loss_pi = -torch.min(surr1, surr2).mean()
                    total_actor_loss = total_actor_loss + loss_pi
                    total_ent = total_ent + ent.mean()

                total_actor_loss = total_actor_loss - self.hp.entropy_coef * total_ent

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actors[0].parameters())
                    + list(self.actors[1].parameters()),
                    self.hp.max_grad_norm,
                )
                self.actor_optimizer.step()

                # Critic update on team returns
                v_pred = self.critic(joint_t[mb]).squeeze(-1)
                loss_v = nn.MSELoss()(v_pred, ret_t[mb].detach())

                self.critic_optimizer.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.hp.max_grad_norm
                )
                self.critic_optimizer.step()

                actor_losses.append(float(total_actor_loss.item()))
                critic_losses.append(float(loss_v.item()))
                entropies.append(float((total_ent / 2.0).item()))

        self.buffer.clear()
        self.update_step += 1
        return {
            "update_step": int(self.update_step),
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }
