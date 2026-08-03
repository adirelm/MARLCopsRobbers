# UX — Pygame God-View Spectator (Nielsen's 10 Heuristics)

The §5.4 GUI is a **god-view spectator**: it renders the referee's ground-truth
board (cop, thief, barriers, capture) plus the match HUD, and replays/streams a
match frame-by-frame. It reads ONLY `SDK.spectator_session()` → `SpectatorFrame`
(the GUI imports only `src.sdk`, `src.gui`, and `pygame`; never the env / MCP /
referee internals — enforced by `tests/architecture/test_gui_purity.py`).

> **Screenshots.** Each heuristic below references a capture under
> `results/screenshots/` produced by `scripts/capture_screens.py` (headless
> pygame-ce, `SDL_VIDEODRIVER=dummy`). Beyond the §7.3c grid-size matrix
> (`grid_{2x2,3x3,4x4,5x5}.png` — the running board), it captures the distinct GUI
> **states** §10.2 asks for: the agent-view overlay (`state_view_radius.png`), the
> terminal winner-banner (`state_terminal.png`), and barrier rendering
> (`state_barriers.png`). The §9.3 wire-match replay adds 18 more under
> `results/screenshots/bonus/`.

## 1. Visibility of system status
The HUD always shows **sub-game `i/6`**, **move `k/25`**, live **scores** and
**totals**, the **barrier budget `placed/5`** (§3.3 — the cop's scarcest resource,
previously inferable only by counting grey cells), the **last joint action** (e.g.
`cop_0: UP, thief: LEFT`), and a **winner banner** at terminal, all on a panel backdrop
that separates it from the board. On the board itself, status is also **spatial**: each
sprite **carries its own heading** (see §2), and a capture is marked by concentric
**shockwave rings** on the cop that actually closed the distance. → `grid_5x5.png`
(heading); the shockwave needs a capture, so see a cop-win frame such as
`bonus/bonus_sg1_final.png`.

An earlier revision also drew a fading motion tail behind each agent. It was **removed**:
heading is already stated twice — by the sprite itself and by the HUD's `Last` line — so
the tail only added 2-3 cells of history at the cost of small circles that read as game
entities. Redundant context that costs legibility is worse than no context.

Heading is shown *only* for the four movement actions. `PLACE_BARRIER` consumes the cop's
move without moving it, so its pupils stay centred rather than claiming travel that never
happened; a thief with no heading yet (the spawn tick) is drawn as a closed disc.

Both frame sources that exist — the live spectator session and the §9.3 wire replay —
carry the budget, so every rendered screenshot shows it. The field nonetheless defaults
to "unknown", and the HUD then **omits the line rather than zeroing it**: a hand-built or
future frame source that lacks the budget would otherwise render `Barriers 0/0`, asserting
a fact it cannot support. Silence is the honest rendering.

## 2. Match between system and the real world
The board is a literal grid with Manhattan movement and a plain-language HUD — no
internal jargon (no `z_t`, no Q-values, no tensor shapes are ever shown). → `grid_3x3.png`.

The two sprites borrow the **arcade maze-chase idiom**, whose silhouettes happen to map
exactly onto this game's roles — which is why they were chosen over abstract discs:

| Role | Sprite | How it reads |
|---|---|---|
| **Thief** (pursued) | red open-mouth wedge | the **mouth faces its direction of travel** |
| **Cop** (pursuer) | blue ghost body | the **pupils look where it is heading** |
| Barrier | grey block | an impassable wall |

Because each character encodes its own heading, the separate arrow marker an earlier
revision drew beside every token was **removed** rather than kept alongside — two
indicators of one fact is clutter, and the redundant one had to go. Shapes are original
geometry built from primitives (`src/gui/sprites.py`), not third-party artwork: no image
assets ship with this repo, so the GUI has no binary dependencies to license or vendor.

## 3. User control and freedom
The spectator is fully controllable: **space** pauses/resumes, **+/-** change
playback speed, **n**/**return** advance to the next sub-game, **r** resets, **v** toggles
the agent view (§8a), **esc** quits.
Pausing then stepping lets the user inspect any position; nothing auto-commits.

## 4. Consistency and standards
One palette + one `GridView` geometry across all board sizes (2×2..5×5): cells are
always square and letterboxed, colours are fixed (`src/gui/palette.py`), and the
key bindings (`src/gui/input_map.py`) are stable. Standard window controls apply.

## 5. Error prevention
The GUI cannot drive the game into an illegal state: it only *renders* frames the
SDK produces (the referee enforces legality). `GridView.cell_rect` bounds-checks
every cell; an unbound key is a no-op (`command_for` → `None`); a finished
sub-game is idempotent (stepping past terminal repeats the final frame).

## 6. Recognition rather than recall
A persistent **help/legend line** lists the active key bindings and the token
legend, so the user never has to recall controls. The HUD restates the full match
context every frame (no hidden modes). → `grid_4x4.png`.

## 7. Flexibility and efficiency of use
The state source is **transport-agnostic** (`src/gui/state_client.py`): the same
renderer drives a local in-proc session, a recorded **replay**, or a Stage-2
**cloud HTTP** stream — all yielding identical `SpectatorFrame`s. Speed controls
let an expert skim or a newcomer step slowly.

## 8. Aesthetic and minimalist design
A dark, minimal board: background, a subtle checkerboard, thin neon gridlines, three
sprite types, and a compact HUD panel. The added layers (halo, rings) are
all **low-alpha context drawn behind or around the tokens** — nothing that competes with
the two things the viewer is actually tracking. The **agent-view overlay** (key **v**) is
off by default → `state_view_radius.png`, so the default board stays uncluttered.

### 8a. Agent view — rendering what the cops KNOW
This is the one visual that carries modelling meaning rather than styling. Pressing **v**
switches the board from the referee's god view to the cop team's epistemic state:

| | god view (default) | agent view (**v**) |
|---|---|---|
| Cells | uniform board | the cops' **Manhattan knowledge halo** is lit |
| Thief inside the halo | solid | solid — it is genuinely observed |
| Thief outside the halo | solid | **ghosted** — the whole sprite, mouth included |

The halo is the UNION over cops (under CTDE the team is one decision-maker at training
time), and it is the true Manhattan **diamond**, not a bounding square — a square would
overstate what the agents observe by the corners. The ghosting is the honest part: the
spectator still sees where the thief IS, but is shown plainly that the cops do not.
This makes the Dec-POMDP partial observability of §2.1/§4 legible in one keypress
instead of only as a claim in prose. Logic + tests: `src/gui/effects.py`,
`src/gui/sprites.py`, `tests/unit/test_gui_effects.py`, `tests/unit/test_gui_sprites.py`,
`tests/unit/test_draw_board.py`.

## 9. Help users recognize, diagnose, and recover from errors
A terminal sub-game shows an explicit **winner banner** (cop capture vs thief
timeout) rather than silently freezing; the move counter hitting `25/25` makes a
timeout self-explanatory (→ `state_terminal.png`). Replay + reset let the user re-watch any contested call.

## 10. Help and documentation
This `docs/UX.md` is the GUI's reference; `scripts/play.py --help` documents the
launch flags (`--seed --grid`; config comes from `config/config.yaml` via the SDK); the
in-window legend line documents the controls. The README §7.3c embeds the
screenshot matrix.
