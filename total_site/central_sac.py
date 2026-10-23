"""Single-agent Soft Actor-Critic for a centralized total-site design problem.

The module intentionally has no knowledge of HEN topology or process simulation.
An environment presents one flat global state and one flat action vector; the
action can then be split by the environment and evaluated by several independent
HEN simulators.  A transition always carries one scalar total-site reward.

Only NumPy and PyTorch are required.  Checkpoints contain the complete training
state, including replay contents and random-number-generator states, so an
interrupted long run can continue reproducibly.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SACConfig:
    """Hyperparameters for :class:`CentralSAC`.

    ``target_entropy=None`` uses the conventional ``-action_dim`` target.  The
    entropy is computed in normalized ``[-1, 1]`` action coordinates, making it
    independent of the physical units used for exchanger areas or loop variables.
    """

    hidden_dims: tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    initial_alpha: float = 0.2
    target_entropy: float | None = None
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    gradient_clip_norm: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(int(width) <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if min(self.actor_lr, self.critic_lr, self.alpha_lr) <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive")
        if self.batch_size <= 0 or self.replay_capacity <= 0:
            raise ValueError("batch_size and replay_capacity must be positive")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when provided")


class ReplayBuffer:
    """Fixed-capacity replay memory with its own reproducible NumPy RNG."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, seed: int) -> None:
        if obs_dim <= 0 or action_dim <= 0 or capacity <= 0:
            raise ValueError("obs_dim, action_dim, and capacity must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _vector(value: Any, expected: int, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (expected,):
            raise ValueError(f"{name} must have shape ({expected},), got {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} contains a non-finite value")
        return result

    def add(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool | float,
    ) -> None:
        reward_array = np.asarray(reward)
        if reward_array.ndim != 0:
            raise ValueError("CentralSAC requires one scalar total-site reward")
        reward_value = float(reward_array)
        done_value = float(done)
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite")
        if done_value not in (0.0, 1.0):
            raise ValueError("done must be boolean or 0/1")

        index = self.position
        self.observations[index] = self._vector(observation, self.obs_dim, "observation")
        self.actions[index] = self._vector(action, self.action_dim, "action")
        self.rewards[index, 0] = reward_value
        self.next_observations[index] = self._vector(
            next_observation, self.obs_dim, "next_observation"
        )
        self.dones[index, 0] = done_value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from replay size {self.size}"
            )
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_observations": torch.as_tensor(
                self.next_observations[indices], device=device
            ),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a compact but lossless replay snapshot."""

        stored = self.capacity if self.size == self.capacity else self.size
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "position": self.position,
            "size": self.size,
            "observations": self.observations[:stored].copy(),
            "actions": self.actions[:stored].copy(),
            "rewards": self.rewards[:stored].copy(),
            "next_observations": self.next_observations[:stored].copy(),
            "dones": self.dones[:stored].copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = (self.obs_dim, self.action_dim, self.capacity)
        actual = (
            int(state["obs_dim"]),
            int(state["action_dim"]),
            int(state["capacity"]),
        )
        if actual != expected:
            raise ValueError(f"replay geometry mismatch: expected {expected}, got {actual}")

        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity:
            raise ValueError("invalid replay size in checkpoint")
        if not 0 <= position < self.capacity:
            raise ValueError("invalid replay position in checkpoint")
        stored = self.capacity if size == self.capacity else size
        fields = (
            ("observations", self.observations, (stored, self.obs_dim)),
            ("actions", self.actions, (stored, self.action_dim)),
            ("rewards", self.rewards, (stored, 1)),
            ("next_observations", self.next_observations, (stored, self.obs_dim)),
            ("dones", self.dones, (stored, 1)),
        )
        for name, destination, shape in fields:
            values = np.asarray(state[name], dtype=np.float32)
            if values.shape != shape:
                raise ValueError(
                    f"invalid replay {name} shape: expected {shape}, got {values.shape}"
                )
            destination[:stored] = values
        self.size = size
        self.position = position
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


def _mlp(input_dim: int, hidden_dims: Sequence[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
        previous = int(width)
    return nn.Sequential(*layers), previous


class SquashedGaussianActor(nn.Module):
    """The single stochastic actor used for the entire global action vector."""

    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: Sequence[int],
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.backbone, output_width = _mlp(obs_dim, hidden_dims)
        action_dim = int(action_low.size)
        self.mean_head = nn.Linear(output_width, action_dim)
        self.log_std_head = nn.Linear(output_width, action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.backbone(observations)
        mean = self.mean_head(latent)
        log_std = self.log_std_head(latent).clamp(
            min=self.log_std_min, max=self.log_std_max
        )
        return mean, log_std

    def sample(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self(observations)
        std = log_std.exp()
        pre_tanh = mean + std * torch.randn_like(mean)
        normalized_action = torch.tanh(pre_tanh)
        action = normalized_action * self.action_scale + self.action_bias

        # Log probability in normalized action coordinates.  Omitting the fixed
        # physical-unit scale keeps alpha comparable across heterogeneous design
        # variables (m2, degC, kg/s, ...).
        normal_log_prob = -0.5 * (
            ((pre_tanh - mean) / std).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_prob = normal_log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - normalized_action.pow(2) + 1.0e-6).sum(
            dim=-1, keepdim=True
        )

        deterministic = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic

    def deterministic(self, observations: Tensor) -> Tensor:
        """Return the transformed policy mean without advancing an RNG."""

        mean, _ = self(observations)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.backbone, output_width = _mlp(obs_dim + action_dim, hidden_dims)
        self.output = nn.Linear(output_width, 1)

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        return self.output(self.backbone(torch.cat((observations, actions), dim=-1)))


class TwinQCritic(nn.Module):
    """Two independent Q functions; this is one centralized critic module."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)

    def forward(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(observations, actions), self.q2(observations, actions)


class CentralSAC:
    """True single-actor SAC operating on a flat global design vector.

    Parameters
    ----------
    obs_dim:
        Length of the global state vector.
    action_low, action_high:
        Per-variable physical bounds for the one global action vector.
    config:
        SAC and replay hyperparameters.
    device:
        Any PyTorch device string.  Checkpoints are device-portable.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        obs_dim: int,
        action_low: Sequence[float] | np.ndarray,
        action_high: Sequence[float] | np.ndarray,
        config: SACConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or SACConfig()
        self.obs_dim = int(obs_dim)
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        if self.action_low.ndim != 1 or self.action_high.shape != self.action_low.shape:
            raise ValueError("action_low and action_high must be equal-length vectors")
        if self.action_low.size == 0:
            raise ValueError("the global action vector cannot be empty")
        if not np.all(np.isfinite(self.action_low)) or not np.all(
            np.isfinite(self.action_high)
        ):
            raise ValueError("action bounds must be finite")
        if not np.all(self.action_high > self.action_low):
            raise ValueError("every action_high value must exceed action_low")
        self.action_dim = int(self.action_low.size)
        self.device = torch.device(device)

        # Seed all stochastic components whose states are included in checkpoints.
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        self.actor = SquashedGaussianActor(
            self.obs_dim,
            self.action_low,
            self.action_high,
            self.config.hidden_dims,
            self.config.log_std_min,
            self.config.log_std_max,
        ).to(self.device)
        self.critic = TwinQCritic(
            self.obs_dim, self.action_dim, self.config.hidden_dims
        ).to(self.device)
        self.target_critic = TwinQCritic(
            self.obs_dim, self.action_dim, self.config.hidden_dims
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(self.config.initial_alpha), device=self.device)
        )
        self.alpha_optimizer = torch.optim.Adam(
            (self.log_alpha,), lr=self.config.alpha_lr
        )
        self.target_entropy = float(
            -self.action_dim
            if self.config.target_entropy is None
            else self.config.target_entropy
        )
        self.replay = ReplayBuffer(
            self.obs_dim,
            self.action_dim,
            self.config.replay_capacity,
            self.config.seed + 1,
        )
        self.environment_steps = 0
        self.gradient_updates = 0

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def act(self, observation: Any, deterministic: bool = False) -> np.ndarray:
        """Return one bounded physical action for a single global observation."""

        obs = ReplayBuffer._vector(observation, self.obs_dim, "observation")
        obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                selected = self.actor.deterministic(obs_tensor)
            else:
                selected, _, _ = self.actor.sample(obs_tensor)
        return selected.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    def remember(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool | float,
    ) -> None:
        """Store one centralized transition with one total-site reward."""

        self.replay.add(observation, action, reward, next_observation, done)
        self.environment_steps += 1

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    def _clip(self, parameters: Any) -> None:
        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip_norm)

    def update(self, batch_size: int | None = None) -> dict[str, float] | None:
        """Perform one SAC gradient update, or return ``None`` if replay is short."""

        actual_batch_size = int(batch_size or self.config.batch_size)
        if len(self.replay) < actual_batch_size:
            return None
        batch = self.replay.sample(actual_batch_size, self.device)
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_observations)
            next_q1, next_q2 = self.target_critic(next_observations, next_actions)
            next_value = torch.minimum(next_q1, next_q2) - self.alpha.detach() * next_log_prob
            target_q = rewards + self.config.gamma * (1.0 - dones) * next_value

        q1, q2 = self.critic(observations, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(self.critic.parameters())
        self.critic_optimizer.step()

        self._set_trainable(self.critic, False)
        policy_actions, log_prob, _ = self.actor.sample(observations)
        policy_q1, policy_q2 = self.critic(observations, policy_actions)
        actor_loss = (self.alpha.detach() * log_prob - torch.minimum(policy_q1, policy_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip(self.actor.parameters())
        self.actor_optimizer.step()
        self._set_trainable(self.critic, True)

        alpha_loss = -(
            self.log_alpha * (log_prob.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target, source in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target.mul_(1.0 - self.config.tau).add_(source, alpha=self.config.tau)

        self.gradient_updates += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "mean_q1": float(q1.detach().mean().cpu()),
            "mean_q2": float(q2.detach().mean().cpu()),
            "mean_log_prob": float(log_prob.detach().mean().cpu()),
            "gradient_updates": float(self.gradient_updates),
        }

    def checkpoint(self, include_replay: bool = True) -> dict[str, Any]:
        """Build a complete, device-portable checkpoint dictionary."""

        rng_state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "obs_dim": self.obs_dim,
            "action_low": self.action_low.copy(),
            "action_high": self.action_high.copy(),
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "gradient_updates": self.gradient_updates,
            "rng_state": rng_state,
            "replay": self.replay.state_dict() if include_replay else None,
        }

    def save(self, path: str | Path, include_replay: bool = True) -> Path:
        """Atomically save training state, including replay by default."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self.checkpoint(include_replay=include_replay), temporary)
        temporary.replace(destination)
        return destination

    @staticmethod
    def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, Tensor):
                    values[key] = value.to(device)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> "CentralSAC":
        """Recreate an agent and its replay memory from ``path``."""

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        version = int(checkpoint.get("checkpoint_version", -1))
        if version != cls.CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported CentralSAC checkpoint version {version}; "
                f"expected {cls.CHECKPOINT_VERSION}"
            )
        config_values = dict(checkpoint["config"])
        config_values["hidden_dims"] = tuple(config_values["hidden_dims"])
        agent = cls(
            int(checkpoint["obs_dim"]),
            np.asarray(checkpoint["action_low"], dtype=np.float32),
            np.asarray(checkpoint["action_high"], dtype=np.float32),
            config=SACConfig(**config_values),
            device=device,
        )
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.target_critic.load_state_dict(checkpoint["target_critic"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        with torch.no_grad():
            agent.log_alpha.copy_(checkpoint["log_alpha"].to(agent.device))
        agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        cls._optimizer_to(agent.actor_optimizer, agent.device)
        cls._optimizer_to(agent.critic_optimizer, agent.device)
        cls._optimizer_to(agent.alpha_optimizer, agent.device)
        agent.environment_steps = int(checkpoint["environment_steps"])
        agent.gradient_updates = int(checkpoint["gradient_updates"])
        if checkpoint.get("replay") is not None:
            agent.replay.load_state_dict(checkpoint["replay"])

        if restore_rng:
            rng_state = checkpoint["rng_state"]
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy_global"])
            torch.set_rng_state(rng_state["torch_cpu"])
            if torch.cuda.is_available() and "torch_cuda" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        return agent


__all__ = [
    "CentralSAC",
    "QNetwork",
    "ReplayBuffer",
    "SACConfig",
    "SquashedGaussianActor",
    "TwinQCritic",
]
