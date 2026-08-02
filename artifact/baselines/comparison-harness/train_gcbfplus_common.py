#!/usr/bin/env python3
"""Train public GCBF+ under the common bounded-unicycle protocol.

The public DubinsCar environment represents a 16 m workspace at 1:4 scale.
This adapter retains the public GCBF+ model and trainer while making the
executed speed, yaw-rate, time step, body radius, and arrival threshold match
the comparison protocol.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import gcbfplus.env as env_module
from gcbfplus.env.dubins_car import DubinsCar


SCALE = 4.0
DT = 0.03
SPEED_LIMIT_PHYSICAL = 0.8
YAW_LIMIT = np.pi / 2.0
ARRIVAL_RADIUS_PHYSICAL = 0.22
BODY_RADIUS_PHYSICAL = 0.2


class CommonBoundedDubinsCar(DubinsCar):
    """Public DubinsCar with the common physical input and goal bounds."""

    PARAMS = dict(DubinsCar.PARAMS)
    PARAMS.update(
        {
            "car_radius": BODY_RADIUS_PHYSICAL / SCALE,
            "comm_radius": 3.0 / SCALE,
        }
    )

    @property
    def scaled_speed_limit(self) -> float:
        return SPEED_LIMIT_PHYSICAL / SCALE

    @property
    def normalized_yaw_limit(self) -> float:
        # DubinsCar.agent_xdot maps action[:, 0] to theta_dot by a factor of 20.
        return YAW_LIMIT / 20.0

    def state_lim(self, state=None):
        del state
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -self.scaled_speed_limit])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, self.scaled_speed_limit])
        return lower, upper

    def action_lim(self):
        lower = jnp.array([-self.normalized_yaw_limit, -3.0])
        upper = jnp.array([self.normalized_yaw_limit, 3.0])
        return lower, upper

    def control_affine_dyn(self, state):
        """Match the affine model to the factor used by agent_xdot."""
        assert state.ndim == 2
        f = jnp.concatenate(
            [
                (jnp.cos(state[:, 2]) * state[:, 3])[:, None],
                (jnp.sin(state[:, 2]) * state[:, 3])[:, None],
                jnp.zeros((state.shape[0], 2)),
            ],
            axis=1,
        )
        g_single = jnp.concatenate(
            [jnp.zeros((2, 2)), jnp.array([[20.0, 0.0], [0.0, 1.0]])],
            axis=0,
        )
        g = jnp.broadcast_to(g_single, (state.shape[0], 4, 2))
        return f, g

    def u_ref(self, graph):
        # The nominal controller is part of the learned residual policy. Keeping
        # it inside the feasible set avoids training against unattainable yaw.
        return self.clip_action(super().u_ref(graph))

    def finish_mask(self, graph):
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        return jnp.linalg.norm(agent_pos - goal_pos, axis=1) < (
            ARRIVAL_RADIUS_PHYSICAL / SCALE
        )

    def stop_mask(self, graph):
        return self.finish_mask(graph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--obs", type=int, default=4)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--area-size", type=float, default=4.0)
    parser.add_argument("--episode-steps", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-env-train", type=int, default=12)
    parser.add_argument("--n-env-test", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-dir", default="/work/gcbfplus-common")
    parser.add_argument("--name", default="gcbfplus_common_unicycle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # make_env reads these module globals at call time.
    env_module.ENV["DubinsCar"] = CommonBoundedDubinsCar
    env_module.DEFAULT_MAX_STEP = args.episode_steps

    # Training metrics are printed by the public trainer. A tiny local stub
    # disables the optional online experiment tracker without changing the
    # learning or checkpoint code.
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.config = SimpleNamespace(update=lambda *unused_args, **unused_kwargs: None)
    wandb_stub.login = lambda *unused_args, **unused_kwargs: None
    wandb_stub.init = lambda *unused_args, **unused_kwargs: None
    wandb_stub.log = lambda *unused_args, **unused_kwargs: None
    sys.modules["wandb"] = wandb_stub

    # The upstream trainer imports ipdb only for optional interactive
    # breakpoints.  Stubbing that hook avoids pulling a modern IPython stack
    # into the pinned TensorFlow Probability environment.
    ipdb_stub = types.ModuleType("ipdb")
    ipdb_stub.set_trace = lambda *unused_args, **unused_kwargs: None
    sys.modules["ipdb"] = ipdb_stub

    # Import after registering the environment so the public entry point uses
    # the bounded class without modifying its trainer or learning algorithm.
    sys.path.insert(0, "/opt/gcbfplus")
    import train as public_train

    train_args = argparse.Namespace(
        num_agents=args.num_agents,
        algo="gcbf+",
        env="DubinsCar",
        seed=args.seed,
        steps=args.steps,
        name=args.name,
        debug=False,
        obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        gnn_layers=1,
        alpha=1.0,
        horizon=args.horizon,
        lr_actor=2e-5,
        lr_cbf=2e-5,
        loss_action_coef=1e-5,
        loss_unsafe_coef=1.0,
        loss_safe_coef=1.0,
        loss_h_dot_coef=0.01,
        buffer_size=args.buffer_size,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        log_dir=args.log_dir,
        eval_interval=args.eval_interval,
        eval_epi=args.eval_epi,
        save_interval=args.save_interval,
        episode_steps=args.episode_steps,
        physical_dt=DT,
        physical_speed_limit=SPEED_LIMIT_PHYSICAL,
        physical_yaw_limit=YAW_LIMIT,
        physical_body_radius=BODY_RADIUS_PHYSICAL,
        physical_arrival_radius=ARRIVAL_RADIUS_PHYSICAL,
        spatial_scale=SCALE,
    )
    public_train.train(train_args)


if __name__ == "__main__":
    main()
