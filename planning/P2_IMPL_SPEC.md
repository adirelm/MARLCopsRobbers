# P2 implementation spec — frozen interfaces (readiness-audit-approved)

> TDD (RED→GREEN→REFACTOR), every file ≤150 LOC, ruff `D` docstrings, NO hardcoded values
> (read from config via config_loader). Gates: `uv run pytest -q --no-cov <paths>` +
> `uv run ruff check <files>` (ignore the VIRTUAL_ENV warning). Builds on P1: GlobalState,
> Observation, grid.{in_grid,manhattan,can_enter,resolve_move,sample_spawn}, actions.{Action,
> DELTAS,action_mask}, reward.RewardModel, scorer.Scorer. Decisions below pinned from the
> codex+claude readiness audit (low-stakes; graded path is 1-cop; 2-cop cases are 4×4 training).

## Agent-key convention (used by joint_a, obs_dict, reward_dict, action masks)
`"cop_0".."cop_{num_cops-1}"` + `"thief"` — matches RewardModel.compute keys. The env builds
all per-agent dicts with these keys.

## src/marl/env/transition.py  (T2.1)
```python
@dataclass(frozen=True)
class TransitionResult:
    next_state: GlobalState          # the new frozen state (terminal flag set)
    winner: str | None               # "cop" (capture) | "thief" (timeout) | None (ongoing)
    capture: bool                    # True iff a cop caught the thief this step

def resolve_joint_action(state: GlobalState, joint_a: dict[str, Action], cfg: dict) -> TransitionResult
```
- SIMULTANEOUS (default `env.move_resolution`): compute every agent's intended next cell from the
  SAME pre-move `state` (positions + `state.barriers`), via `grid.resolve_move` (OOB/barrier ⇒ stay).
  Order-independent: no agent sees another's move applied first.
- Barrier placement (cop PLACE_BARRIER): the barrier is placed on the cop's CURRENT cell; the cop
  stays on that cell (placement IS the move). **TEAM budget cap**: iterate cops by index; a PLACE is
  honored only while `barriers_used < game.max_barriers`; once the team cap is hit, later cops'
  PLACE drops to a stay (deterministic by cop index — codex F1). Increment `barriers_used` per
  honored placement; add each placed cell to `barriers`.
- Capture (check BEFORE timeout): `capture = any(cop ends on thief's cell) OR swap` where swap =
  a cop and the thief exchange cells in one tick (`env.capture_on_swap`). `capture ⇒ winner="cop"`,
  `terminal=True`.
- Timeout: if NOT captured and `(state.step + 1) >= game.max_moves` ⇒ `winner="thief"`, `terminal=True`.
- 2-cop double-occupancy ALLOWED (no inter-cop collision; cooperative, 4×4 training only).
- `next_state` = a NEW frozen GlobalState with updated cop_pos/thief_pos/barriers/barriers_used,
  `step=state.step+1`, same h/w, `terminal` per above.
- PURE (no RNG) ⇒ replay-deterministic. Tests: same-cell capture, swap=capture, both-stay on
  barrier/edge, single-cop budget cap, **2-cop simultaneous over-budget (team cap, deterministic)**,
  timeout exact tie `(step+1)==max_moves`, capture-on-final-move beats timeout, step increments.

## src/marl/env/observation.py  (T2.2, may split observation_encoder.py)
```python
def view_radius(h: int, w: int, cfg: dict) -> int     # env.view_radius_by_grid.get(min(h,w)) else max(0, ceil(min(h,w)/2)-1)
def build_observation(state: GlobalState, role: str, memory: dict, cfg: dict, idx: int = 0) -> Observation
```
- `W_v = 2*cfg.env.view_radius_max + 1` (=5). image shape `(obs_channels, W_v, W_v)` float32,
  channel order PINNED: `0 self, 1 other_visible, 2 barrier, 3 out_of_bounds, 4 time_norm`.
- Egocentric: crop a (2r+1) window centered on the agent (cop_pos[idx] or thief_pos), then pad to
  W_v=5 with the `out_of_bounds` channel = 1 on padded/off-board cells (so ONE encoder spans
  2×2→5×5). `self`=1 at center. `barrier`=1 on visible barrier cells. `time_norm` = `state.step/max_moves`
  broadcast. `other_visible`=1 at the opponent's cell ONLY if `manhattan(self,opp) <= view_radius(h,w)`
  (0 beyond — partial observability). NO absolute opponent position leaks beyond the local window.
- scalars `(obs_scalars=6,)` float32: `own_r_norm=r/(h-1 or 1), own_c_norm=c/(w-1 or 1),
  step_norm=step/max_moves, barriers_left_norm=(max_barriers-barriers_used)/max_barriers,
  other_seen_now=1.0 if opp in view else 0.0, steps_since_seen_norm=min(memory[role].steps_since_seen, max_moves)/max_moves`.
- PURE given `memory` (env owns + updates it). Tests: shape==(5,5,5) at 2×2/3×3/4×4/5×5;
  other_visible=0 beyond radius; out_of_bounds padding correct on a 2×2; scalars in [0,1].

## src/marl/env/cops_robbers_env.py  (T2.3) + render_state.py
```python
class CopsRobbersEnv:
    def __init__(self, cfg: dict, h: int|None=None, w: int|None=None, num_cops: int|None=None): ...
    def reset(self, seed: int | None = None) -> tuple[dict[str, Observation], dict]   # (obs_dict LOCAL, info)
    def step(self, joint_a: dict[str, Action]) -> tuple[dict[str, Observation], dict[str, float], bool, dict]
    def state(self) -> GlobalState                # TRAIN-ONLY global accessor
```
- `reset(seed)`: seed a `random.Random` (P1 `grid.sample_spawn` calls `rng.choice` on a tuple
  list, so a numpy Generator would coerce tuples to arrays — use stdlib `random.Random`),
  `grid.sample_spawn` cop+thief with
  `min_dist = view_radius(h,w,cfg)` (spawn dist>radius); init `barriers=frozenset()`,
  step=0; reset per-agent visibility memory (steps_since_seen=large/0 per chosen convention);
  return per-agent LOCAL obs_dict + info `{action_mask: {agent: mask}, started_at: ISO-ts}`.
- `step(joint_a)`: `resolve_joint_action(self._state, joint_a, cfg)` → update `self._state`; update
  visibility memory per agent (reset counter to 0 when opp in view, else +1); build new obs_dict;
  `reward_dict = RewardModel.compute(prev, next_state, winner)`; `terminated = next_state.terminal`;
  info `{action_mask, winner, capture, scores: Scorer.score(winner) if terminated else None}`.
- `state()` returns the full GlobalState (train-only). `render_state()` (in render_state.py) returns a
  plain serializable dict for GUI/MCP — board dims + positions for DRAWING are allowed there (it is the
  god-view spectator path, NOT the agent obs path) but it returns NO GlobalState instance.
- env + render_state each ≤150 LOC (factor reset/step helpers into a `_helpers` module if needed).
- obs_dict + info from reset()/step() carry ONLY local Observations + action_mask/ts/winner —
  NEVER a GlobalState, full barrier set, or absolute opponent position.

## src/marl/env/curriculum.py  (T2.4)
```python
class Curriculum:
    def __init__(self, cfg: dict): ...            # stages, num_cops_by_stage, promotion_threshold, promotion_window
    def current(self) -> tuple[int, int, int]     # (h, w, num_cops) for the active stage
    def maybe_promote(self, capture_rate: float) -> bool   # advance if rate >= promotion_threshold; True if promoted
    def make_env(self, cfg: dict) -> CopsRobbersEnv        # env for the current stage (h,w,num_cops)
```
- Stages from `env.curriculum.stages`; `num_cops_by_stage` parallel list. Promotion advances the
  stage index (clamped at the last). Cop-count switch between stages = a NEW env (the GlobalState
  cop_pos tuple length differs per stage) via make_env. non-square (h≠w) supported by the obs pad.
- Tests: promotes on threshold over window; clamps at final stage; current() returns the right
  (h,w,num_cops) per stage; a full ≤max_moves episode runs at each stage size; winner∈{cop,thief}.

## tests/architecture/test_step_no_leak.py  (T2.3 exit-gate, the most-graded requirement)
- Run a FULL episode (reset → step until terminated, capped at max_moves) on a small grid.
- Recursively scan every value in the reset() and step() obs_dict + info: assert NO `GlobalState`
  instance appears anywhere; assert no dict key equals any `GlobalState` field name
  (`{f.name for f in dataclasses.fields(GlobalState)}`); assert each Observation value's keys ==
  `{"image","scalars"}`; assert `info` has no `state`/`global_state`/`barriers`/board-dim keys.
- Assert `env.state()` IS a GlobalState (the sanctioned train-only accessor) — i.e. the global state
  exists but only via state(), never via step(). Give the test teeth: a helper that would FAIL if any
  global field were added to an obs/info payload.

## Docs to update (§1.4 spec-first, minimal)
- docs/PRD.md FR-ENV-*: add the TransitionResult contract + the edge-case rules (capture>timeout,
  any-cop-captures, double-occupancy allowed, deterministic over-budget) + the env-owned
  visibility-memory ownership.
- docs/PLAN.md: note transition.py returns TransitionResult; render_state.py is the only other global
  accessor; ADR-0004 capture rule extended (capture beats timeout on the final move).
- docs/TODO.md: T2.x already cover most; add the visibility-memory-ownership + leak-test-recursive-scan notes.
