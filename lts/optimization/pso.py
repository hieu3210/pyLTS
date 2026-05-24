"""Generic Particle Swarm Optimization (PSO).

Implements the standard PSO velocity update:
    V_i^{t+1} = ω·V_i^t + c1·r1·(P_i - X_i) + c2·r2·(P_g - X_i)
    X_i^{t+1} = X_i^t + V_i^{t+1}

Used by LTSPSOModel and COLTSModel.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PSOConfig:
    n_particles: int = 300
    max_iter: int = 1000
    omega: float = 0.4
    c1: float = 2.0
    c2: float = 2.0
    bounds: list[tuple[float, float]] = field(default_factory=list)
    seed: int | None = None


class PSO:
    """Minimizes `objective(position) -> float` over a bounded search space."""

    def __init__(
        self,
        objective: Callable[[list[float]], float],
        config: PSOConfig,
    ) -> None:
        self._obj = objective
        self._cfg = config
        self._dim = len(config.bounds)

    def run(self) -> tuple[list[float], float]:
        """Return (best_position, best_value)."""
        cfg = self._cfg
        rng = random.Random(cfg.seed)

        lo = [b[0] for b in cfg.bounds]
        hi = [b[1] for b in cfg.bounds]

        # Initialise positions and velocities
        pos = [
            [rng.uniform(lo[d], hi[d]) for d in range(self._dim)]
            for _ in range(cfg.n_particles)
        ]
        vel = [
            [rng.uniform(-(hi[d] - lo[d]), hi[d] - lo[d]) for d in range(self._dim)]
            for _ in range(cfg.n_particles)
        ]

        pbest = [p[:] for p in pos]
        pbest_val = [self._obj(p) for p in pbest]

        gbest_idx = min(range(cfg.n_particles), key=lambda i: pbest_val[i])
        gbest = pbest[gbest_idx][:]
        gbest_val = pbest_val[gbest_idx]

        for _ in range(cfg.max_iter):
            for i in range(cfg.n_particles):
                for d in range(self._dim):
                    r1 = rng.random()
                    r2 = rng.random()
                    vel[i][d] = (
                        cfg.omega * vel[i][d]
                        + cfg.c1 * r1 * (pbest[i][d] - pos[i][d])
                        + cfg.c2 * r2 * (gbest[d] - pos[i][d])
                    )
                    pos[i][d] = max(lo[d], min(hi[d], pos[i][d] + vel[i][d]))

                val = self._obj(pos[i])
                if val < pbest_val[i]:
                    pbest[i] = pos[i][:]
                    pbest_val[i] = val
                    if val < gbest_val:
                        gbest = pos[i][:]
                        gbest_val = val

        return gbest, gbest_val
