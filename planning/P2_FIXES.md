# P2 validate-after fixes (codex + Claude adversarial review, all triaged REAL)

> Both models converged on the 2-cop visibility flaw + the leak-test teeth gap. Graded path is
> 1-cop (unaffected), but the 4×4 2-cop stage is the K1 credit-assignment evidence (§7.2), so fix.
> TDD: strengthen/add the failing test, then fix. Keep ≤150 LOC/file, ruff(+D), no-hardcode.

## OWNER A — transition.py + test_transition.py
- **H (co-located double-PLACE, Claude L1 major):** in `_resolve_cops`, honor+charge a PLACE only if
  the cop's cell is NOT already a barrier this tick: `if action==PLACE_BARRIER and used<max_barriers
  and cop not in barriers: barriers |= {cop}; used += 1` (else degrade to stay, no increment). Keeps
  `barriers_used == len(distinct placed cells)` so reward.py:93 charges once. Test: two co-located
  cops `((2,2),(2,2))` both PLACE → barriers_used delta==1, len(barriers)==1.
- **F (dead move_resolution, codex #6 / Claude L9):** read `cfg["env"]["move_resolution"]`; only
  `"simultaneous"` is supported → `raise NotImplementedError(...)` for cop_first/thief_first (live the
  config key + honest limitation). Test: a non-simultaneous cfg raises.
- **Test gaps:** capture_on_swap=False (cfg override) → swap yields capture=False/winner=None;
  2-cop swap-capture (one of two cops swaps) → capture True; 2-cop double-occupancy (both cops move
  to one cell) → both there, no collision, no crash.

## OWNER B — observation.py + observation_encoder.py + _env_helpers.py + cops_robbers_env.py + test_observation.py + test_env.py
- **A (2-cop visibility — THE big one, codex #2/#3, Claude L3/L4/L6/L7/L9):** redesign so the THIEF
  sees the whole cop TEAM and per-agent memory is correct.
  1. Visibility memory keyed by AGENT KEY (`cop_0`..`cop_{n-1}`, `thief`) — NOT role. Each agent has
     its own `VisibilityMemory.steps_since_seen`.
  2. `build_observation(state, agent_key, memory, cfg)` (agent_key replaces role+idx): for `cop_{i}`,
     center=`cop_pos[i]`, opponents=`[thief_pos]`; for `thief`, center=`thief_pos`,
     opponents=`list(cop_pos)` (ALL cops).
  3. `encode_image` takes `opponents: list[Pos]` (not a single opp): mark `other_visible=1` at EACH
     opponent cell that is on-board AND `manhattan(center, opp) <= radius`. `other_seen_now` scalar =
     1.0 if ANY opponent in view. `steps_since_seen` resets to 0 when ANY opponent in view.
  4. `_env_helpers.update_memory`: for EACH agent_key, recompute "any opponent in view" from that
     agent's own center (cop_i uses cop_pos[i]; thief uses all cops) → reset/inc its own counter.
  5. `build_obs_dict` builds `cop_{i}` obs with the per-cop memory + the thief obs seeing all cops.
- **B (partial-obs fog beyond radius, codex #1, Claude L2):** in `encode_image`, an ON-BOARD cell with
  `manhattan(center, cell) > radius` is NOT visible → set `out_of_bounds=1` (fog) and DO NOT set
  barrier/other there. Visible region = the Manhattan disk of `radius` (so 2×2 radius-0 is truly
  near-blind: only the self cell visible). Update the docstring: out_of_bounds = off-board OR beyond
  the view radius (not-visible). Fix the 2×2 test expectation accordingly.
- **C (terminal-step guard, Claude L4/L8 major):** `step()` raises `RuntimeError("step() on a
  terminated episode; call reset()")` if `self._state.terminal`. Test: stepping after termination raises.
- **D (eval_mode, Claude L4 minor):** `step(self, joint_a, eval_mode: bool = False)` → pass
  `eval_mode` to `RewardModel.compute(...)`. Test: eval_mode=True zeroes shaping in the reward_dict.
- **DRY (Claude L9 nit):** extract one `opponent_in_view(center, opp, radius)` helper (in
  observation.py) used by encoder + _env_helpers (remove the 3 copies).
- **Test gaps:** non-square obs (h=3,w=5) → shape (5,5,5), self centered, correct OOB; opponent
  in-window-but-out-of-radius (cop(2,2),thief(0,0) on 5×5, manhattan 4>r2) → other_visible.sum()==0;
  off-window on-board barrier not marked; a CAPTURE driven through env.step (pick a seed/policy that
  captures — assert winner=="cop", capture True, scores present); opponent-in-view memory reset
  (steps_since_seen→0 when in view); 2-cop: thief sees BOTH cops when both in radius; strengthen the
  vacuous `test_visibility_memory_resets_on_reset` (assert the actual before/after counter, not just [0,1]).

## OWNER C — test_step_no_leak.py + curriculum.py + test_curriculum.py
- **E (leak-test teeth — MOST-GRADED, codex #5, Claude L5 major×2 / L10):** make the recursive
  scanner descend into `set`/`frozenset` AND `np.ndarray` (if `arr.dtype == object`, iterate
  `arr.flat`; numeric arrays need no scan). Strengthen the negative-control: plant a `GlobalState`
  inside (i) an `info` value, (ii) an obs `image` object-array, and (iii) a `set` — assert each is
  caught. Run the full-episode leak scan on a **2-cop** stage AND on a **capture** episode (not only
  the 1-cop timeout). Fold the hardcoded `FORBIDDEN_INFO_KEYS` into the dataclass-derived
  `GLOBAL_FIELD_NAMES` (single source).
- **Curriculum (Claude L6 minor):** `__init__` validates `len(stages)==len(num_cops_by_stage)` →
  `ValueError`. `make_env` uses ONE cfg consistently (use `self._cfg` for h/w/num_cops AND rules — drop
  the split-brain, or document it reads dims from current() + rules from the same cfg). Add a test
  covering `promotion_window` (the lone uncovered line) + a non-square stage if config allows, else a
  unit test of current() on a synthetic non-square stages cfg.

## DOC-ONLY (no code change)
- Update `planning/P2_IMPL_SPEC.md` line 69: "seed a np.random.Generator" → "seed a `random.Random`
  (matches P1 grid.sample_spawn which calls rng.choice on a tuple list)". (Claude L4/L7/L9 nit — the
  code is correct; the spec text was wrong.)

## NOT FIXED (validated NOT defects)
- own_r_norm/own_c_norm "leak" (Claude L3 nit): an agent knowing its OWN position is standard and not
  a partial-obs violation. Keep.
- started_at UTC vs Asia/Jerusalem (Claude L4 nit): spec only requires an ISO timestamp; UTC is fine.
