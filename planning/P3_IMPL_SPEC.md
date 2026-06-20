# P3 implementation spec — frozen interfaces (readiness-audit-approved)

> TDD, ≤150 LOC/file, ruff(+D), no-hardcode (config via config_loader). Builds on P2:
> CopsRobbersEnv.reset(seed)/step(joint_a,eval_mode)/state(); RewardModel; Scorer;
> resolve_joint_action; build_observation; observation_encoder.encode_image; agent-keys cop_0../thief.
> Decision (user-signed-off): P3 "train" = a THROWAWAY TABULAR Q-learner on the 2×2 state space.

## Pinned decisions (from the codex+claude readiness audit)
- **obs_encoder.py is a thin BATCHING ADAPTER** over the P2 `env.observation_encoder.encode_image`
  (import + stack + to-array), NOT a 2nd geometry impl (DRY). Canonical training tensor =
  **channels-first** `(B, C, 5, 5)` to match the env `Observation`.
- **TransitionDTO** = a single-step interchange/dataset record; the **replay buffer** stores padded
  whole EPISODES (different shapes). Don't conflate.
- **global_state stored = the stage-invariant 77-float** `encode_state()` (3·5·5 + 2, masked) so P4's
  centralized critic gets a fixed input. `encode_state()` is a pure data encoder (NOT a net) → built here.
- **next_legal_mask** IS stored per step (PRD:486 — P4 Double-DQN needs it; don't recompute from cropped obs).
- **source_tag** = a provenance label `SourceTag(StrEnum){EXPERT, SELF_PLAY, RANDOM, LIVE_CTDE}`.
- **"first GUI shot" → P7**; P3 emits a headless god-view JSON via `render_state()`. **"export"** = the
  NPZ dataset + a minimal sub-game JSON, NOT actor weights (no net at P3).
- **Docs:** reconcile PRD/PLAN P3 milestone rows (still say "BC pretrain val-acc≥0.9 / train→eval") to
  match the post-#27 ordering (BC training is P4/T4.4); fix T3.3 coverage target (src/marl/replay +
  src/marl/data, not src/marl/env).

## src/marl/data/schemas.py  (T3.2)
```python
class SourceTag(StrEnum): EXPERT="expert"; SELF_PLAY="self_play"; RANDOM="random"; LIVE_CTDE="live_ctde"
@dataclass(frozen=True)
class TransitionDTO:          # single-step interchange/dataset record
    obs: np.ndarray          # local obs image (C,5,5) f32 for one agent
    scalars: np.ndarray      # (obs_scalars,) f32
    global_state: np.ndarray # (77,) f32 stage-invariant encoded state
    action: int              # the agent's action index
    reward: float
    done: bool
    next_legal_mask: np.ndarray  # (a_cop,) bool
@dataclass(frozen=True)
class DatasetManifest:        # provenance for a saved BC/replay artifact
    grid: tuple[int, int]; n_pairs: int; seed: int; source: str
    obs_channels: int; obs_scalars: int; w_v: int; schema_version: str
```

## src/marl/data/obs_encoder.py  (T3.2 — adapter + encode_state)
```python
def encode_obs_batch(obs_list: list[Observation]) -> tuple[np.ndarray, np.ndarray]   # stack -> (B,C,5,5) f32 images + (B,obs_scalars) f32 scalars
def encode_state(state: GlobalState, cfg: dict) -> np.ndarray   # stage-invariant 77-float: 3 planes (cop, thief, barrier) on a top-left-anchored W=2*view_radius_max+1=5 grid, off-board cells masked 0 + [step_norm, barriers_left_norm]; shape (3*5*5+2,)=(77,)
```
- `encode_obs_batch` imports + reuses `env.observation_encoder.encode_image` results (already (C,5,5));
  it ONLY stacks/concatenates — no geometry. `encode_state` places each cop/thief/barrier on a 5×5
  plane **top-left anchored** to the max grid (off-board cells masked 0), flattens, appends the 2 scalars.

## src/marl/data/heuristics.py  (T3.2 — train-time ORACLES, consume GlobalState)
```python
def cop_expert(state: GlobalState, cfg: dict, idx: int = 0, rng: Random | None = None, epsilon: float = 0.0) -> Action
def thief_expert(state: GlobalState, cfg: dict, rng: Random | None = None, epsilon: float = 0.0) -> Action
```
- cop: barrier-aware BFS shortest path from `cop_pos[idx]` toward `thief_pos`; pick the first step on a
  shortest path; **deterministic tie-break = lowest `Action` enum index** among equal-cost moves; with
  `epsilon>0` (+ rng) take a random legal move with prob ε (BC diversity). Honors `can_enter`.
- thief: greedy 1-step — among legal moves maximize `manhattan(thief, nearest cop)`; tie-break lowest
  Action index; ε-greedy with rng. Both honor barriers + grid.
- Tests on a known 3×3: cop path is shortest + barrier-respecting + deterministic; cop captures
  optimally on 2×2; thief increases distance.

## src/marl/data/bc_dataset.py  (T3.2 — npz IO)
```python
def build_bc_dataset(cfg, grid, n_pairs, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetManifest]  # (obs, scalars, actions, manifest) BC pairs via experts
def save_npz(path, obs, scalars, actions, manifest) -> None
def load_npz(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetManifest]
```
- Generates `n_pairs` (obs, expert-action) pairs by rolling out experts (seeded). Round-trips via npz.

## src/marl/replay/episode_buffer.py  (T3.1)
```python
class CentralizedReplayBuffer:
    def __init__(self, capacity, t_max, n_agents, obs_channels, w_v, obs_scalars, state_dim, n_actions, seed): ...
    def add_episode(self, episode: dict, source_tag: SourceTag) -> None
    def sample(self, batch_size: int) -> dict   # batched arrays + per-item source_tag
    def __len__(self) -> int
```
- Pre-allocated numpy ring (capacity from `replay.buffer_episodes`); seeded `np.random.default_rng`.
- Episode dict keys (padded to `t_max = game.max_moves`): `obs (T+1,N,C,5,5) f32`,
  `scalars (T+1,N,obs_scalars) f32`, `global_state (T+1, state_dim=77) f32`, `actions (T,N) i64`,
  `reward (T,N) f32`, `done (T,) bool`, `filled (T,) bool` (1 for real steps, 0 for pad),
  `next_legal_mask (T,N,n_actions) bool`, `hidden_seed (i64 scalar; written, unused until P4)`.
- **N convention:** cop buffer `n_agents = max(num_cops_by_stage) = 2` (N=1 stages padded; the agent
  axis beyond active cops is `filled=0`/zero); thief buffer `n_agents = 1`. TWO separate buffers.
- post-done masking: `filled` zeroes padded steps; sampling returns `filled` so BPTT ignores pad.
- Tests: add/sample shapes `(B,T,N,*)`; `filled` masking; seeded reproducibility; **one episode from
  EACH SourceTag lands with the right tag + a mixed sample draws across sources**.

## src/marl/replay/tabular_smoke.py  (T3.3 — the THROWAWAY 2×2 learner; deleted in P4)
```python
class TabularQLearner:        # dict Q[(state_key, agent_key)][action]; throwaway, 2×2-scale only
    def __init__(self, cfg): ...   # alpha, gamma=algo.gamma, epsilon from config (add p3_smoke block)
    def update(self, s_key, agent_key, action, reward, next_s_key, next_legal) -> None   # Q += alpha*(r + gamma*max_legal Q' - Q)
    def greedy_action(self, s_key, agent_key, legal_mask) -> Action
def state_key(state: GlobalState) -> tuple   # exact discrete key (cop_pos, thief_pos, barriers, step)
def run_smoke(cfg, seeds) -> dict   # collect (cop=learner vs thief=heuristic) → tabular train → assert 2×2 optimal → return metrics
```
- "Reaches optimal" (D3): the thief FLEES, so naive min-Manhattan is the wrong target. After training,
  the greedy (eval) cop achieves **pursuit-optimal capture from EVERY 2×2 start over all
  `training.seeds`** (5), NaN-free; a zeroed/untrained Q-table must FAIL this gate (non-vacuous). Feeds
  episodes into the CentralizedReplayBuffer (exercises the buffer end-to-end).

## src/sdk/sdk.py + src/sdk/_helpers.py  (T0.7 minimal facade — single-entry, §4)
```python
class MarlSDK:
    def __init__(self, cfg): ...
    def build_env(self, h=None, w=None, num_cops=None) -> CopsRobbersEnv
    def collect_episode(self, env, cop_policy, thief_policy, seed) -> dict   # one padded episode dict
    def run_p3_smoke(self, seeds) -> dict       # the 2×2 collect→train→export (delegates to tabular_smoke)
    def write_subgame_json(self, result, path) -> None   # minimal sub-game record
```
- The P3 integration smoke routes through MarlSDK (no business logic in scripts). Keep ≤150 LOC via
  the `_helpers.py` split.

## sub-game JSON (minimal, T3.3) + docs/schema/subgame.schema.json
- Minimal record: `{game_id, grid:[h,w], winner:"cop"|"thief", capture:bool, steps:int, scores:{cop,thief}, seed}`.
  A draft-2020-12 schema in `docs/schema/subgame.schema.json` (reused/extended by the P9 report `sub_games[]`).

## config additions (config/config.yaml)
- `p3_smoke: {alpha: 0.5, epsilon: 0.1, episodes: 4000}` (tabular learner hyperparams — no-hardcode;
  4000 exploring-starts episodes converge ALL `training.seeds` to the 2×2 pursuit-optimal capture gate).
- `paths.bc_dataset_dir: "results/datasets"`, `paths.subgames_dir: "results/subgames"`.

## Docs to update (§1.4 spec-first)
- PRD.md + PLAN.md P3 milestone rows: remove "BC pretrain val-acc≥0.9 / train→eval"; state P3 =
  pipeline + tabular 2×2 smoke + dataset/replay machinery; BC training is P4/T4.4.
- TODO.md: T3.1 add next_legal_mask + SourceTag enum + N convention (note: the full N=2 agent-padding
  convention — zero-filled phantom slot + per-agent active mask — is FINALIZED in P4 when num_cops=2
  episodes exist; P3 only adds the fail-loud N-mismatch guard); T3.2 obs_encoder = adapter +
  encode_state; T3.3 = tabular smoke (define "optimal" = **pursuit-optimal capture from every 2×2 start
  over all `training.seeds`**, since the thief flees min-Manhattan is wrong),
  coverage target = src/marl/replay + src/marl/data, GUI shot → P7 (headless render_state JSON at P3).
