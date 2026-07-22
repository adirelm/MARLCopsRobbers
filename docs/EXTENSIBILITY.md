# Extensibility — the seams, and how to extend each (V3 §12)

Every extension point below is a **real interface with ≥2 live implementations** (or an
implementation plus a test double), so adding a third is a subclass, not a refactor. Each
section states the contract, the existing implementations, and a worked example.

| Seam | Interface | Live implementations | Add a new one by |
|---|---|---|---|
| Value-decomposition mixer | `BaseMixer` (ABC) | `VdnMixer`, `QmixMixer` | subclassing + registering the name |
| Learner arm | `LearnerBase` template method | `CopLearner` (CTDE), `ThiefLearner` (DDQN), `IqlLearner` | subclassing the template |
| Acting policy | duck-typed `act()` + `reset()` | `RecurrentPolicy`, `ObsFleePolicy`, `AdaptiveThiefPolicy`, heuristics | any object with the two methods |
| Email transport | `EmailSender` Protocol | `GmailMailer` (App-Password), OAuth path, test doubles | implementing `send(subject, body, to)` |
| Spectator feed | `StateClient` trio | `InProcStateClient`, `ReplayStateClient`, `HttpStateClient` | implementing `reset()` / `step()` |
| MCP auth | `build_verifier(cfg, role, …)` seam | `StaticTokenVerifier` (Stage-1), `RevocableJWTVerifier` (Stage-2) | returning any FastMCP verifier |
| Egress channel | `ApiGatekeeper.execute(channel, call)` | `gmail`, `peer_mcp` | adding a channel to `config/rate_limits.json` |
| Curriculum stage | `env.curriculum.stages` config | 2×2 → 3×3 → 4×4 → 5×5 | appending a `[h, w]` pair |

---

## 1. Add a new mixer (worked example — ~20 lines)

The mixer is the value-decomposition seam. `BaseMixer` fixes the contract:
`forward(q_agents, state) -> Q_tot`, and the learner never knows which mixer it
holds. To add, say, a mean-pooling mixer:

```python
# src/marl/mixers/mean_mixer.py
"""Mean-pooling mixer — Q_tot = mean_i Q_i (a scale-invariant VDN variant)."""

from __future__ import annotations

from torch import Tensor

from src.marl.mixers.base_mixer import BaseMixer


class MeanMixer(BaseMixer):
    """Average the per-agent Q-values (monotone: ∂Q_tot/∂Q_i = 1/N ≥ 0, so IGM holds)."""

    def forward(self, q_agents: Tensor, state: Tensor) -> Tensor:
        """In: per-agent Q ``[B, T, N]`` + global state. Out: ``Q_tot`` ``[B, T, 1]``."""
        return q_agents.mean(dim=-1, keepdim=True)
```

Then register the name so `algo.name: mean` selects it (`src/sdk/_train_helpers.py`
`cfg_for_algo` maps the config name to the mixer/learner pair). Nothing else changes: the
buffer, the trainer, the figures and the MCP layer are all mixer-agnostic. Monotonicity is
the only constraint — a non-monotone mixer breaks the IGM guarantee (README §7.2), which is
exactly the QPLEX/Weighted-QMIX discussion.

## 2. Add an acting policy (the §9 lineup used this seam)

`MarlSDK.build_policy(role, net_or_policy)` returns a ready policy: pass a **net** and it
wraps it in `RecurrentPolicy`; pass an object that already has `act()` + `reset()` and it is
passed through untouched. That is how the bonus lineup ships a scripted flee policy and an
auto-switching wrapper (`src/services/bonus_policies.py`) into the same MCP server that
normally serves a trained net — no server change, no protocol change.

```python
class AlwaysUpPolicy:
    def reset(self) -> None: ...
    def act(self, obs_list, legal_masks, epsilon, rng, state=None):
        return [Action.UP]

server = make_thief_server(cfg, AlwaysUpPolicy())   # serves immediately
```

## 3. Add an egress channel

`ApiGatekeeper` is config-driven: add the channel to `config/rate_limits.json`

```json
"slack_notify": { "per_minute": 10, "burst": 2, "_use": "build notifications" }
```

then call `gate.execute("slack_notify", thunk)`. Rate limiting, the FIFO overflow queue,
backpressure and logging come for free — and the architecture test asserts the Gmail and
peer-MCP channels route this way. (The §9 wire-match client is the one documented exception:
queueing is incompatible with its synchronous 10-second move deadline.)

## 4. Add an email transport

`EmailSender` is a Protocol (`send(subject, body, to)`). `GmailMailer` implements it over
smtplib+STARTTLS; the OAuth path is a drop-in; tests inject a recording double. Swapping
transports never touches the report assembly, the redaction, or the idempotency sentinel.

## 5. Add a spectator feed

`StateClient` is `reset()` / `step()` returning a frozen `SpectatorFrame`. The three
implementations (in-process, recorded replay, HTTP) prove the GUI is transport-agnostic:
the renderer only consumes frames, so a new feed (e.g. a websocket stream) is one class.

## 6. Add a curriculum stage

`env.curriculum.stages` is a config list of `[h, w]` pairs with a parallel
`num_cops_by_stage`. The env, the observation encoder (fixed padded footprint) and the
figures are all size-generic, so extending the ladder to 6×6 is a config edit — no code.

---

**Why these are the right seams.** Each one sits on a boundary the assignment itself moves
across: algorithms (§5.2), transports (§5.3 localhost→cloud), roles/policies (§9 bonus),
and board sizes (§5.1 Table 2). The ADRs in `docs/PLAN.md` §6 record why each boundary was
drawn where it is.
