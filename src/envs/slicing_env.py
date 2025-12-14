# -*- coding: utf-8 -*-
"""
Optimized Gymnasium environment for 5G network slicing with DRL.

Features:
- Configurable observation/action spaces
- Multi-objective reward function with SLA constraints
- Support for both real network (via NIL) and simulation modes
- Proper episode management and info logging
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class SlicingEnv(gym.Env):
    """
    RL Environment for dynamic bandwidth allocation between URLLC and mMTC slices.

    Action Space:
        Box([0, 1]) - beta_urllc: fraction of bandwidth allocated to URLLC
        beta_mmtc = 1 - beta_urllc is implicit

    Observation Space (8-dimensional):
        [0] d1_norm: URLLC delay normalized by SLA threshold
        [1] p1_norm: URLLC loss normalized by SLA threshold
        [2] d2_norm: mMTC delay normalized by SLA threshold
        [3] p2_norm: mMTC loss normalized by SLA threshold
        [4] lambda1_norm: URLLC traffic intensity (normalized)
        [5] lambda2_norm: mMTC traffic intensity (normalized)
        [6] beta_prev: Previous URLLC allocation
        [7] utilization: Overall link utilization

    Reward Function:
        R = SLA_margin_term - violation_penalty - smoothness_penalty

        Where:
        - SLA_margin_term: Weighted sum of normalized margins from SLA thresholds
        - violation_penalty: Penalty for exceeding SLA thresholds
        - smoothness_penalty: Penalty for large allocation changes
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config=None,
        nil=None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize SlicingEnv.

        Args:
            config: EnvConfig dataclass or dict with environment parameters
            nil: NILInterface for network communication (None = simulation mode)
            render_mode: Rendering mode ('human', 'ansi', or None)
        """
        super().__init__()

        self.render_mode = render_mode
        self._parse_config(config)
        self._setup_nil(nil)
        self._setup_spaces()
        self._reset_state()

    def _parse_config(self, config) -> None:
        """Parse configuration from dataclass or dict."""
        # Default values
        defaults = {
            # SLA thresholds
            "d1_max_ms": 1.0,      # URLLC max delay (ms)
            "p1_max": 1e-5,        # URLLC max loss rate
            "d2_max_ms": 1000.0,   # mMTC max delay (ms)
            "p2_max": 0.01,        # mMTC max loss rate

            # Traffic reference values (for normalization)
            "lambda1_ref": 2000.0,  # URLLC reference rate (pkt/s)
            "lambda2_ref": 5000.0,  # mMTC reference rate (pkt/s)

            # Episode parameters
            "max_steps": 1000,
            "dt_s": 0.1,  # Timestep duration (seconds)

            # Reward weights (positive contributions)
            "w_urllc": 10.0,       # URLLC SLA weight (prioritized)
            "w_mmtc": 2.0,         # mMTC SLA weight
            "w_balance": 0.5,      # Balance bonus weight

            # Penalty coefficients
            "alpha_urllc": 50.0,   # URLLC violation penalty
            "alpha_mmtc": 5.0,     # mMTC violation penalty
            "alpha_smooth": 0.1,   # Action smoothness penalty

            # Observation space bounds
            "obs_high": 2.0,       # Allow values up to 2x SLA (captures violations)

            # Simulation mode traffic params
            "sim_urllc_lambda": 2000.0,
            "sim_mmtc_lambda": 3000.0,
        }

        # Extract values from config
        if config is None:
            cfg = defaults
        elif hasattr(config, "__dataclass_fields__"):
            # Dataclass config
            cfg = defaults.copy()
            for key in defaults:
                if hasattr(config, key):
                    cfg[key] = getattr(config, key)
            # Handle nested SLA config
            if hasattr(config, "sla"):
                sla = config.sla
                cfg["d1_max_ms"] = getattr(sla, "urllc_max_delay_ms", cfg["d1_max_ms"])
                cfg["p1_max"] = getattr(sla, "urllc_max_loss", cfg["p1_max"])
                cfg["d2_max_ms"] = getattr(sla, "mmtc_max_delay_ms", cfg["d2_max_ms"])
                cfg["p2_max"] = getattr(sla, "mmtc_max_loss", cfg["p2_max"])
        else:
            # Dict config
            cfg = {**defaults, **config}

        # Store as attributes
        for key, value in cfg.items():
            setattr(self, f"_{key}", value)

    def _setup_nil(self, nil) -> None:
        """Setup network interface layer."""
        self._nil = nil
        self._simulation_mode = nil is None

        if self._simulation_mode:
            # Import traffic generators for simulation
            from ..traffic.urllc import URLLCTraffic
            from ..traffic.mmtc import MMTCTraffic

            self._urllc_gen = URLLCTraffic(lambda_rate=self._sim_urllc_lambda)
            self._mmtc_gen = MMTCTraffic(lambda_on=self._sim_mmtc_lambda)

    def _setup_spaces(self) -> None:
        """Setup action and observation spaces."""
        # Action: beta_urllc in [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: 8-dimensional normalized vector
        self.observation_space = spaces.Box(
            low=0.0,
            high=self._obs_high,
            shape=(8,),
            dtype=np.float32,
        )

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self._step_count = 0
        self._prev_beta = 0.5
        self._episode_reward = 0.0
        self._episode_violations = {"urllc": 0, "mmtc": 0}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        self._reset_state()

        # Reset network interface if available
        if not self._simulation_mode and self._nil is not None:
            self._nil.reset()

        # Initial observation (neutral state)
        obs = np.zeros(8, dtype=np.float32)
        obs[6] = self._prev_beta  # Previous allocation

        info = {
            "episode_start": True,
            "simulation_mode": self._simulation_mode,
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Array with beta_urllc in [0, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 1. Parse and clip action
        beta_urllc = float(np.clip(action[0], 0.0, 1.0))
        beta_mmtc = 1.0 - beta_urllc

        # 2. Get network metrics (real or simulated)
        if self._simulation_mode:
            metrics = self._simulate_step(beta_urllc)
        else:
            metrics = self._real_step(beta_urllc)

        # 3. Compute normalized observations
        obs = self._compute_observation(metrics, beta_urllc)

        # 4. Compute reward
        reward, reward_info = self._compute_reward(metrics, beta_urllc)

        # 5. Update episode tracking
        self._step_count += 1
        self._prev_beta = beta_urllc
        self._episode_reward += reward

        # 6. Check termination
        terminated = False  # No early termination condition
        truncated = self._step_count >= self._max_steps

        # 7. Build info dict
        info = {
            "beta_urllc": beta_urllc,
            "beta_mmtc": beta_mmtc,
            "step": self._step_count,
            **metrics,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def _simulate_step(self, beta_urllc: float) -> Dict[str, float]:
        """Simulate network step without real network."""
        # Simple M/M/1 queueing model simulation
        capacity_mbps = 100.0

        # Allocated capacities
        c1 = beta_urllc * capacity_mbps
        c2 = (1 - beta_urllc) * capacity_mbps

        # Traffic rates (convert to Mbps)
        lambda1_mbps = (self._sim_urllc_lambda * 200 * 8) / 1e6  # 200 bytes/packet
        lambda2_mbps = (self._sim_mmtc_lambda * 50 * 8) / 1e6   # 50 bytes/packet

        # Utilization
        rho1 = lambda1_mbps / max(c1, 0.01)
        rho2 = lambda2_mbps / max(c2, 0.01)

        # M/M/1 latency estimation (ms)
        if rho1 < 1.0:
            d1 = 1.0 / (c1 - lambda1_mbps + 0.001) * 1000  # Convert to ms
        else:
            d1 = 100.0  # Congestion

        if rho2 < 1.0:
            d2 = 1.0 / (c2 - lambda2_mbps + 0.001) * 1000
        else:
            d2 = 100.0

        # Clip to reasonable ranges
        d1 = np.clip(d1, 0.01, 100.0)
        d2 = np.clip(d2, 0.01, 1000.0)

        # Loss estimation (simple threshold model)
        p1 = 0.0 if rho1 < 0.9 else min(0.1, (rho1 - 0.9) * 0.5)
        p2 = 0.0 if rho2 < 0.9 else min(0.1, (rho2 - 0.9) * 0.5)

        return {
            "d1_ms": d1,
            "p1": p1,
            "d2_ms": d2,
            "p2": p2,
            "lambda1": self._sim_urllc_lambda,
            "lambda2": self._sim_mmtc_lambda,
            "rho1": rho1,
            "rho2": rho2,
            "throughput_urllc": min(lambda1_mbps, c1),
            "throughput_mmtc": min(lambda2_mbps, c2),
        }

    def _real_step(self, beta_urllc: float) -> Dict[str, float]:
        """Execute step with real network via NIL."""
        # Apply action to network
        self._nil.apply_action(beta_urllc)

        # Get metrics from network
        metrics = self._nil.get_metrics()

        return {
            "d1_ms": metrics.delay_urllc,
            "p1": metrics.loss_urllc,
            "d2_ms": metrics.delay_mmtc,
            "p2": metrics.loss_mmtc,
            "lambda1": self._lambda1_ref,  # Estimated from config
            "lambda2": self._lambda2_ref,
            "rho1": 0.0,  # Not directly available
            "rho2": 0.0,
            "throughput_urllc": metrics.throughput_urllc,
            "throughput_mmtc": metrics.throughput_mmtc,
        }

    def _compute_observation(
        self,
        metrics: Dict[str, float],
        beta_urllc: float,
    ) -> np.ndarray:
        """Compute normalized observation vector."""
        # Normalized ratios (can exceed 1.0 on violation)
        d1_norm = metrics["d1_ms"] / self._d1_max_ms
        p1_norm = metrics["p1"] / self._p1_max if self._p1_max > 0 else 0.0
        d2_norm = metrics["d2_ms"] / self._d2_max_ms
        p2_norm = metrics["p2"] / self._p2_max if self._p2_max > 0 else 0.0

        # Traffic intensity (normalized)
        lambda1_norm = metrics["lambda1"] / self._lambda1_ref
        lambda2_norm = metrics["lambda2"] / self._lambda2_ref

        # Overall utilization
        utilization = (metrics.get("rho1", 0) + metrics.get("rho2", 0)) / 2

        obs = np.array([
            np.clip(d1_norm, 0.0, self._obs_high),
            np.clip(p1_norm, 0.0, self._obs_high),
            np.clip(d2_norm, 0.0, self._obs_high),
            np.clip(p2_norm, 0.0, self._obs_high),
            np.clip(lambda1_norm, 0.0, self._obs_high),
            np.clip(lambda2_norm, 0.0, self._obs_high),
            beta_urllc,
            np.clip(utilization, 0.0, 1.0),
        ], dtype=np.float32)

        return obs

    def _compute_reward(
        self,
        metrics: Dict[str, float],
        beta_urllc: float,
    ) -> Tuple[float, Dict]:
        """
        Compute multi-objective reward.

        Returns:
            reward: Scalar reward value
            info: Dictionary with reward components for debugging
        """
        # Normalized ratios
        d1_ratio = metrics["d1_ms"] / self._d1_max_ms
        p1_ratio = metrics["p1"] / self._p1_max if self._p1_max > 0 else 0.0
        d2_ratio = metrics["d2_ms"] / self._d2_max_ms
        p2_ratio = metrics["p2"] / self._p2_max if self._p2_max > 0 else 0.0

        # SLA margins (positive = meeting SLA, negative = violating)
        margin_d1 = max(0.0, 1.0 - d1_ratio)
        margin_p1 = max(0.0, 1.0 - p1_ratio)
        margin_d2 = max(0.0, 1.0 - d2_ratio)
        margin_p2 = max(0.0, 1.0 - p2_ratio)

        # Combined SLA scores per slice
        sla_urllc = 0.5 * (margin_d1 + margin_p1)
        sla_mmtc = 0.5 * (margin_d2 + margin_p2)

        # Violations (excess beyond SLA)
        viol_d1 = max(0.0, d1_ratio - 1.0)
        viol_p1 = max(0.0, p1_ratio - 1.0)
        viol_d2 = max(0.0, d2_ratio - 1.0)
        viol_p2 = max(0.0, p2_ratio - 1.0)

        # Track violations
        if viol_d1 > 0 or viol_p1 > 0:
            self._episode_violations["urllc"] += 1
        if viol_d2 > 0 or viol_p2 > 0:
            self._episode_violations["mmtc"] += 1

        # Action smoothness
        delta_beta = abs(beta_urllc - self._prev_beta)

        # === Reward computation ===

        # 1. SLA margin term (positive reward for meeting SLA)
        sla_term = (
            self._w_urllc * sla_urllc +
            self._w_mmtc * sla_mmtc +
            self._w_balance * min(sla_urllc, sla_mmtc)
        )

        # 2. Violation penalty (negative for exceeding SLA)
        violation_penalty = (
            self._alpha_urllc * (viol_d1 + viol_p1) +
            self._alpha_mmtc * (viol_d2 + viol_p2)
        )

        # 3. Smoothness penalty (discourage oscillations)
        smooth_penalty = self._alpha_smooth * delta_beta

        # Total reward
        reward = sla_term - violation_penalty - smooth_penalty

        # Build info dict
        info = {
            "sla_urllc": sla_urllc,
            "sla_mmtc": sla_mmtc,
            "sla_term": sla_term,
            "viol_d1": viol_d1,
            "viol_p1": viol_p1,
            "viol_d2": viol_d2,
            "viol_p2": viol_p2,
            "violation_penalty": violation_penalty,
            "smooth_penalty": smooth_penalty,
            "delta_beta": delta_beta,
        }

        return float(reward), info

    def render(self) -> Optional[str]:
        """Render environment state."""
        if self.render_mode == "ansi":
            return (
                f"Step {self._step_count}/{self._max_steps} | "
                f"Beta: {self._prev_beta:.2f} | "
                f"Reward: {self._episode_reward:.2f} | "
                f"Violations: URLLC={self._episode_violations['urllc']}, "
                f"mMTC={self._episode_violations['mmtc']}"
            )
        elif self.render_mode == "human":
            print(self.render())
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    @property
    def unwrapped(self):
        """Return unwrapped environment."""
        return self
