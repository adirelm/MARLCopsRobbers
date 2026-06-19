# P1 implementation spec — frozen interfaces (architect-approved)

> TDD (RED→GREEN→REFACTOR), every file ≤150 LOC, ruff `D` docstrings on every module/class/fn,
> NO hardcoded algorithm values (read from config via the loader). All values come from
> `config/config.yaml`. Run gates with `uv --directory "<A6>" run pytest -q <paths>` and
> `uv --directory "<A6>" run ruff check <paths>` (a harmless `VIRTUAL_ENV` warning prints — ignore).
> Architect decisions (user-signed-off this turn): (1) GlobalState carries `h,w`; (2)
> `d_max=(H-1)+(W-1)` derived per live grid (NOT a config key); (3) P1 split test = type-level
> (Observation exposes only local fields). Adopted: Φ uses `−min` over the cop team;
> `load_config()→dict` with `${VAR}` interpolation + ValueError; `action_mask` never all-False.

## src/utils/config_loader.py  (prereq T0.3)
```python
DEFAULT_PATH = <repo>/config/config.yaml   # resolve relative to this file, package-relative
def load_config(path: str | Path | None = None) -> dict   # yaml.safe_load → _interpolate_env → _validate_config → dict
def _interpolate_env(obj)                                  # recurse; os.path.expandvars on str scalars (${VAR} from os.environ)
def _validate_config(cfg: dict) -> None                    # raise ValueError if cfg["version"]!="1.0.0" OR any required top-level section missing
```
- Required top-level sections: `version, project, game, env, algo, nets, olora, bc, replay, selfplay, training, reward, mcp, cloud, gmail, gui, paths, logging`.
- Return the raw nested dict (no dataclass). Module-level `_CACHE` optional (load once). Tests: round-trips real config; raises ValueError on bad version + on a missing section; `${VAR}` expands from env (use a tmp yaml with a `${TEST_VAR}` token).

## src/marl/env/types.py  (T1.3)
```python
Pos = tuple[int, int]                       # (row, col), 0-indexed
@dataclass(frozen=True)
class GlobalState:                          # TRAIN-TIME global state (CTDE). NEVER sent to exec/MCP.
    cop_pos: tuple[Pos, ...]                # 1 cop (graded 5x5) or 2 (4x4 stage); tuple for hashable/frozen
    thief_pos: Pos
    barriers: frozenset[Pos]                # all placed barriers (full board knowledge → train-only)
    barriers_used: int
    step: int
    h: int                                  # board rows  (architect decision #1)
    w: int                                  # board cols
    terminal: bool = False
class Observation(TypedDict):               # EXEC-TIME LOCAL obs — ONLY local fields (architect decision #3)
    image: np.ndarray                       # (obs_channels, Wv, Wv) float32; Wv=2*view_radius_max+1=5; channels: self,other_visible,barrier,out_of_bounds,time_norm
    scalars: np.ndarray                     # (obs_scalars,) float32 — own_r_norm,own_c_norm,step_norm,barriers_left_norm,other_seen_now,steps_since_seen_norm
```
- `Observation` MUST NOT contain `barriers` (full set), absolute `thief_pos`, `cop_pos`, `h/w`, or `step` int — those are global. (The split test asserts this.)

## src/marl/env/grid.py  (T1.3)
```python
def in_grid(pos: Pos, h: int, w: int) -> bool
def manhattan(a: Pos, b: Pos) -> int
def can_enter(target: Pos, actor: Pos, barriers, h: int, w: int) -> bool   # in_grid(target) AND (target not in barriers OR target == actor)
def resolve_move(pos: Pos, delta: Pos, actor: Pos, barriers, h, w) -> Pos   # target=pos+delta; return target if can_enter else pos (edge==wall & barrier-blocked ⇒ STAY)
def sample_spawn(rng, h, w, min_dist: int) -> tuple[Pos, Pos]               # uniform cop,thief with manhattan>min_dist; ValueError if min_dist >= (h-1)+(w-1) (infeasible)
```
- Tests: `in_grid` corners/OOB; `manhattan` symmetry; `can_enter` self-cell exception + barrier block + OOB; `resolve_move` edge==wall stays + barrier stays + free move; `sample_spawn` respects `dist>min_dist` (seeded) + raises on infeasible.

## src/marl/env/actions.py  (T1.2)
```python
class Action(IntEnum): UP=0; DOWN=1; LEFT=2; RIGHT=3; PLACE_BARRIER=4; STAY=5   # STAY reserved, config-gated OFF
DELTAS: dict[Action, Pos] = {UP:(-1,0),DOWN:(1,0),LEFT:(0,-1),RIGHT:(0,1),PLACE_BARRIER:(0,0),STAY:(0,0)}
def action_mask(state: GlobalState, role: str, cfg: dict, idx: int = 0) -> np.ndarray   # bool, length env.actions.a_cop (=5)
```
- `A_max = cfg["env"]["actions"]["a_cop"]` (=5). Mask indices 0..4 = UP,DOWN,LEFT,RIGHT,PLACE_BARRIER.
- cop (role="cop", position = state.cop_pos[idx]): move legal iff `can_enter(pos+DELTA, pos, barriers, h, w)`; PLACE_BARRIER legal iff `state.barriers_used < cfg["game"]["max_barriers"]` AND `cfg["env"]["actions"]["enable_barrier"]`.
- thief (role="thief", position = state.thief_pos): moves as above; PLACE_BARRIER (index 4) ALWAYS False.
- STAY (index 5) excluded from the mask whenever `cfg["env"]["actions"]["enable_stay"] is False` (default) — mask length stays a_cop=5.
- NEVER all-False: if no move is legal (boxed in), return all-True over the move indices (transition no-ops blocked moves to stay) — add a RED test for the boxed-in case.
- Tests: thief PLACE masked; OOB move masked; barrier-blocked move masked; `barriers_used==max_barriers` masks PLACE for cop; boxed-in ⇒ not all-False; IntEnum integer values pinned (UP..RIGHT contiguous 0..3, PLACE_BARRIER=4).

## src/marl/env/reward.py  (T1.4 — RL training signal ONLY)
```python
def potential(state: GlobalState) -> float        # Φ = -min_i manhattan(cop_i, thief)/d_max ; d_max=(h-1)+(w-1); returns 0.0 if state.terminal (phi_terminal_zero)
class RewardModel:
    def __init__(self, cfg: dict)                  # reads reward.*, algo.gamma, env.reward_mode
    def compute(self, prev: GlobalState, nxt: GlobalState, winner: str|None=None, eval_mode: bool=False) -> dict[str, float]
```
- d_max derived: `(prev.h - 1) + (prev.w - 1)` (architect decision #2). NOT a config literal.
- Shaping `F = distance_weight * (gamma*Φ(nxt) - Φ(prev))`, included only if `reward.shaping_enabled` AND not (`eval_mode` AND not `reward.shaping_eval_enabled`). Φ(terminal)=0 when `reward.phi_terminal_zero` (so terminal F = -Φ(prev)).
- Cop-team per-step: `F - reward.step_penalty` (+ `-reward.barrier_cost` if a barrier was placed — derivable from barriers_used delta).
- Terminal: capture (winner=="cop") ⇒ `+reward.capture_bonus` to cop team; timeout (winner=="thief") ⇒ `-reward.timeout_penalty` to cop team.
- `dec_pomdp` (default `env.reward_mode`): return per-agent dict with ALL cops equal (`cop_i`→team reward) + `thief` = `-team_reward` (zero-sum proxy). For N=1: `{"cop": r, "thief": -r}` (use keys `cop_0`/`thief` or `cop`/`thief` consistently — pick `cop_{i}` for i in range(num cops) + `thief`).
- `posg`: cop team as above; thief role-specific = `+reward.evade_step` per non-terminal step, `+reward.timeout_bonus` on timeout, mirror-shaping `-F` optional (keep simple: evade_step + timeout_bonus, no shaping). Document the choice.
- Tests: shaping identity `F=gamma*Φ(nxt)-Φ(prev)` (numeric, e.g. 3x3 cop(0,0) thief(2,2): manhattan=4, d_max=4, Φ=-1.0); Φ(terminal)=0; eval_mode zeroes shaping; dec_pomdp cop entries equal; capture/timeout terminal signs; `-min` over 2 cops picks the nearer cop.

## src/marl/env/scorer.py  (T1.4 — REPORT-ONLY, never training)
```python
class Scorer:
    def __init__(self, cfg: dict)                  # reads game.scoring.{cop_win,thief_win,cop_loss,thief_loss}
    def score(self, winner: str) -> dict[str, int] # capture ⇒ {"cop":cop_win(20),"thief":thief_loss(5)}; timeout ⇒ {"cop":cop_loss(5),"thief":thief_win(10)}
```
- Tests: capture totals 20/5; timeout totals 5/10; assert Scorer is a DISTINCT module from RewardModel and its output is never imported by reward.py (architecture-style import check).

## tests/conftest.py
- `cfg` fixture → `load_config()` (real config).
- `make_state` factory fixture → build a `GlobalState` from kwargs with sensible 5x5 defaults (cop_pos, thief_pos, barriers=frozenset(), barriers_used=0, step=0, h=5, w=5, terminal=False).

## tests/architecture/test_train_exec_split.py  (P1 exit-gate, architect decision #3)
- Assert `Observation.__annotations__` keys are a subset of {"image","scalars"} (NO global fields: no barriers/cop_pos/thief_pos/h/w/step).
- Assert `GlobalState` HAS the global fields (barriers, cop_pos, thief_pos) — i.e. the global state is a distinct, richer type.
- Docstring: this is the P1 type-level SEED of the full P2 runtime leak test (T2.3).

## Docs to encode the decisions (§1.4 spec-first)
- PRD FR-ENV-6: add the normative line `d_max=(H-1)+(W-1)` per live grid (derived, not a config key).
- PRD GlobalState row + a new Observation field table: reflect `h,w` on GlobalState and `{image,scalars}` on Observation.
- ADR-0005 (PLAN ADR table) + config reward comment + TODO T1.4: Φ = `−min_i manhattan(cop_i,thief)/d_max`.
- docs/THEORY.md (T1.1): the Dec-POMDP tuple + POSG caveat (eq1/eq3), N=cop-team.
- README §7 outline (T1.5): §7.1/§7.2/§7.3 skeleton naming F1-F6 + the IQL/VDN/QMIX × seeds × ladder experiment.
