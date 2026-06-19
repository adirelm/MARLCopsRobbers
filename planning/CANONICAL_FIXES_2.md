# A6 plan — round-2 triaged fixes (post 90-agent super-deep review)

> 90-agent review (Claude 40 + Codex 16 real; Gemini auth-failed). 226 findings (14 blocker /
> 135 material / 77 minor). Clustered + triaged below. **FIX = genuine plan-level error /
> contradiction / missing artifact → correct now.** **DEFER = real but an implementation
> detail that the build + its tests will resolve (note it, don't over-spec the plan).** Apply
> all FIX items; keep config.yaml authoritative; preserve good content.

## FIX — modeling / algorithm correctness
1. **Dec-POMDP N (L4, Codex C01/C11).** Disambiguate, do NOT leave |N| contradictory: the FULL
   game is a **POSG with agents {cop, thief}** (matches L10 eq1 / source [1]); the **cooperative
   Dec-POMDP the COP TEAM solves** has `N = the cop team` (|N|=`env.num_cops`: 1 graded 5×5, 2 on
   4×4). Fix FR-ENV-1 AC (drop the bare "|N|=2"; test |N|==env.num_cops for the Dec-POMDP and
   note the POSG game has 2 role-agents). In the eq-1 tuple, **A = the cop-team JOINT action**
   (thief action ∈ POSG/`T` dynamics, NOT the Dec-POMDP A). State S/R for the multi-cop case:
   make `cop_pos` a list + `R` shared over the team (not singular `Manhattan(cop,thief)`).
2. **IQL correctness (L2/L3, blocker).** (a) The "Mixer ABC is the ONLY variation point" claim is
   false — IQL also has **no centralized critic, no global state, independent per-agent target**.
   Reframe ADR-0008/FR-ALG-4: the shared `RecurrentQNet` is reused, but IQL differs in target
   operator AND drops the mixer+global-state. (b) Reconcile the IQL target equation: ex06 eq2 ≡
   L10 eq4 is the plain `y=r+γ max Q(o',a')`; the plan's Double-DQN target-net is an
   implementation refinement — state it as "Double-DQN realization of the eq2/eq4 independent
   target" rather than claiming byte-equality. (c) Drop the fictional `iql_mixer.py` "identity
   module"; IQL has no mixer (learner branches instead). (d) `IqlLearner` must NOT inherit the
   QMIX 2-param-group (lr_mixer) optimizer.
3. **QMIX mixer (L1/L5/L15).** (a) Commit the hypernetwork weight constraint to **softplus**
   (strictly >0; abs admits zero-weight IGM degeneracy) — only V(s) bias is sign-free per [3];
   update config comment + ADR + TODO. (b) **Global-state padding:** the centralized state MUST
   be encoded to a **stage-invariant footprint (3·5·5+2=77, out-of-board masked)** for ALL
   curriculum stages, mirroring the obs pad-to-5×5 — add an FR + a shape test. (c) **Drop
   `include_opponent_utility`** (it risks a monotone dependence on the adversary's utility =
   the forbidden cop/thief mixer boundary); rely solely on the 4×4 2-cop scenario for the K1
   non-trivial-mixer evidence. (d) IGM/monotonicity unit tests run on an explicit **N=2 mixer
   fixture** (not the default N=1 where they are vacuous) + a non-monotone negative-control that
   must FAIL.
4. **Reward potential-shaping (L10, blocker).** (a) Pin **Φ(terminal)=0** so Ng-1999 invariance
   holds (state it in FR + ADR-0005 + a test); note the swap-capture terminal case. (b) The §9
   sensitivity sweep must NOT sweep `reward.distance_weight` (a policy-invariant shaping coef →
   no effect by construction) — sweep a **non-invariant** param (`env.view_radius_by_grid`, or
   `olora.rank` only when OLoRA enabled, or `selfplay.window_k`). (c) For 2-cop, Φ = −min over
   cops of manhattan(cop_i,thief)/d_max (not singular).
5. **Encoder architecture (L14).** **Drop the conv encoder** — it has no basis in L10 (which
   prescribes a feed-forward/MLP recurrent net, eq8), is degenerate on a ≤5×5 patch,
   over-parameterizes (~150k params for a tiny state), and is **incompatible with OLoRA** (wraps
   `nn.Linear` only). Use an **MLP encoder** (flatten obs → Linear trunk → GRU) per L10 eq8.
   Remove `nets.conv_channels`; reconcile `encoder.py`/`recurrent_q_net.py` to ONE definition.
6. **Action masking (L12/L33).** (a) Masking is **additive −inf** before softmax/argmax (NOT
   multiplicative `mask·Q`, which is wrong for negative Q). Fix the exec formula in PLAN/PRD/TODO.
   (b) The shared net has a **role-conditioned output head** (cop 5 / thief 4) — clarify it is
   not literally one head emitting both widths. (c) Store the **next-state legal mask** for the
   Double-DQN bootstrap. (d) Resolve invalid-action: masking is authoritative (no illegal action
   reaches the env), so `invalid_penalty` is a defensive guard only — say so (don't double-spec
   invalid→NOOP + masking).
7. **Self-play (L7, blocker).** Pick **ONE** regime: alternating best-response with a
   frozen-opponent window + opponent pool. Define coherently: `selfplay.rounds` (best-response
   rounds/stage) × episodes-per-round = `training.episodes_per_stage`; remove the triple-overload
   of `window_k`; reset/anneal ε **per best-response round** (not once per stage); specify the
   pool sampling rule + snapshot cadence; reconcile `pool_size` vs `rounds`.

## FIX — egress / MCP / cloud / auth
8. **ApiGatekeeper egress (L22/L23, blocker).** Route **ALL** egress through
   `ApiGatekeeper.execute(channel=...)`: the **peer-MCP** client (highest volume) AND **Gmail**
   AND Prefect — none may import `httpx`/`smtplib` directly outside `src/api/`. Fix the egress
   boundary + make `test_egress_via_gatekeeper` actually catch the peer-MCP path. Add **retry on
   transient failure** to `execute` (V3 §5.1). Define overflow at `max_queue` (reject-with-error
   when full, logged — "no crash" must be enforceable).
9. **Single rate-limit source (L22, Codex C16).** Remove `cloud.rate_limits` from config.yaml;
   `config/rate_limits.json` is the SINGLE source (already has peer_mcp/gmail/prefect_deploy).
10. **Report recipient (Codex C03/L23).** Remove `REPORT_RECIPIENT_OVERRIDE` from the prod path
    (it permits a "successful" send to the WRONG address); recipient is fixed `gmail.to`
    (`rmisegal+marl@gmail.com`). A test-only override may exist but must be inert in prod.
11. **MCP contract (L17/L18).** (a) ONE tool-name set — pick `new_sub_game` (drop the
    `start_sub_game`/`end_sub_game` variant) and define **`query_opponent`** as a real tool. (b)
    Define **`session_id`**: a wire field on every tool request (the keying primitive for the
    per-session GRU state); add it to the Observation/request schema. (c) Specify session
    lifecycle incl. **eviction** (TTL / on game end). DEFER exact concurrency-lock mechanics to
    the build, but state the invariant (one in-flight request_move per session).
12. **Cloud recurrent state (L18/L21/L31, blocker).** Server-side in-memory GRU `z_t` is unsafe
    on a multi-worker serverless host. Pin **single-worker** deploy (`fastmcp run --workers 1` /
    document the constraint) as the chosen resolution; note the client-carried-hidden-state
    alternative as the fallback. State it in PLAN + ADR.
13. **Cloud model delivery (L21).** Checkpoints are git-ignored (`*.pt/*.pth`) but the cloud host
    builds from the GitHub repo → the deployed container would have NO model. Resolve: commit a
    **small final actor-only checkpoint** (inference weights only, well under size limits) OR
    fetch it at deploy from a release asset. State the chosen path in PLAN/TODO + adjust
    `.gitignore` to allow the one committed inference checkpoint if that path is chosen.
14. **Auth deps (L20).** `pyproject` mcp group must use **`pyjwt[crypto]`** (RS256 needs the
    cryptography backend). The **jti deny-list is NOT a built-in JWTVerifier feature** — state it
    as a custom verifier wrapper. Reconcile the spectator-token env-var names (one name).

## FIX — V3 / docs / process / PII
15. **CLAUDE.md is MISSING (L25).** Create `CLAUDE.md` at the A6 root — the §1.4 human-architect/
    AI-implementer contract + the canonical values + hard constraints (it is cited across
    PRD/PLAN/TODO/README but the file does not exist). Adapt from A5's CLAUDE.md.
16. **Prompt Log MISSING (L25/L40).** Create `docs/PROMPTS.md` (or `docs/shared/PROMPTS.md`) +
    add a TODO task to maintain it (V3 §8 requires a prompt log; it is the §1.4 evidence).
17. **Self-grade NOT in tracked repo (L38, A1 lesson).** Remove the P11 "self-grade written"
    from any tracked doc — the numeric self-grade goes ONLY on the git-ignored Moodle cover
    sheet (the A1–A5 locked rule). Fix TODO P11 + any PRD/PLAN reference.
18. **Docstring gate (Codex C14).** Add a docstring checker (ruff `D` rules or an AST script) to
    the hard-gate sweep + give every package `__init__.py` a one-line module docstring.
19. **Per-mechanism PRDs (L26).** `docs/prd/PRD-{ENV,CTDE,MCP,REPORT}.md` are named but ungated;
    add a real TODO task (one per PRD) + a `test_required_docs_present` entry.
20. **§16 modular building blocks (L32).** Add a requirement: components expose Input/Output/Setup,
    `_validate_config()→ValueError` + `_validate_input()→TypeError`, DI for testability — not
    just the config loader.
21. **§13 ISO 25010 (L30).** Assign a concrete A6 mechanism to ALL 8 characteristics (not just
    Security/Maintainability) in the QUALITY.md plan.
22. **LICENSE owner slug (L35).** Genericize the copyright holder to `adrl-001` (consistency with
    the "owner slug lives only in players.local.yaml" design; avoids any deny-list doubt).
23. **PII checker (L35, A4 lesson).** `check_pii.py` must NOT hardcode the literal deny-list
    tokens in tracked source — load them from a git-ignored pattern file or derive generically
    (listing the literal tokens in the checker is itself a leak).
24. **SDK single-entry leaks (L27, Codex C15).** (a) Fix the README Usage block — the `src.cli`
    entry points must actually exist (add a `src/cli/` SDK-only surface) or be corrected. (b)
    `referee.py` business logic belongs behind the SDK (the MCP controller delegates). (c) Remove
    the **duplicate action/mask** (one source in `src/marl/env/actions.py`; the mcp layer imports
    it, not a second copy).

## FIX — schedule / figures / reproducibility
25. **Schedule / feasibility (L37, blocker).** Add a TIME dimension: per-phase **effort estimate**
    (days) across the ~50-day window, a **critical path**, and a **solo cut/defer plan** (what to
    drop if behind — cloud, OLoRA-ablation, bonus are the deferral candidates). Quantify the
    **training-compute envelope** (rounds × episodes × stages × wall-clock) UP FRONT, not at P10.
26. **Figures from JSONL (L36, Codex C12).** Correct the "all 6 figures regenerate from
    results/runs/*.jsonl" claim — F3 (GUI screenshots) + F4 (MCP-comms screenshots) are captured,
    not plotted. State the loss curves cover BOTH cop+thief nets across stages. Resolve the
    `results/runs/*.jsonl` tracked-vs-gitignored contradiction (track the small JSONL run records
    needed for figure regen; git-ignore only heavy checkpoints/histories).
27. **Build-order (Codex C10).** Move RecurrentQNet + mixers out of P1 into P4 (per BRIEF §6 "step
    4 = network/training"); P1 = theory/env/reward/scorer only. De-overload P3 (BC→P4, GUI→P7).

## DEFER to the build (note in a "deferred-to-implementation" doc section; do NOT over-spec now)
- Exact conv/MLP layer geometry beyond "MLP per eq8"; BPTT initial-hidden convention; conftest
  fixture enumeration; session eviction TTL value; canonical-JSON sort/format details (will be
  pinned in the serializer); replay ring exact dtype; max-replay-attempts cap number; game
  edge-case precedence (max_moves-tie, double-occupancy, thief-trapped) — capture as a short
  "rules edge-case table" stub in PRD to be finalized in P1/P2 with tests, but the exact
  resolutions are implementation+test decisions.
- Non-square grids (3×2/4×3): keep the SQUARE ladder as the default (Table-2 "/" = alternatives);
  state the env is generic over H≠W but the graded curriculum is square. (Already mostly done.)
