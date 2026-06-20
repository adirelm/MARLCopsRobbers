# P4c implementation spec — OLoRA + BC pretrain + attach (frozen, audit-approved)

> P4c = T4.3 (OLoRA) + T4.4 (BC pretrain) + T4.5 ATTACH (bundle + learner-injection seam + smoke).
> SCOPE NOTE (forced by deps): T4.5's full curriculum-finetune LOOP reuses the P4d CTDE trainer
> (T4.6) — so the finetune *orchestration* lands in P4d; P4c builds OLoRA, BC, the attach bundle, the
> injection seam, and a smoke proving an OLoRA-wrapped learner.update runs. TDD, ≤150 LOC/file,
> ruff(+D), no-hardcode. OLoRA math numerically validated at seeded float32, rank=4 (function-
> preserving residual ~4.5e-6, orthonormal BᵀB≈I ~1e-6 — both well under the 1e-5 test tolerance).

## Pinned contracts (codex+Claude P4c readiness)
- **OLoRA math** ([7] eqs 3-5): `W_pre = QR(thin, mode="reduced")`; `B=Q[:, :r]`, `A=R[:r, :]`; frozen
  `W0 = W_pre - scale*(B@A)`; forward `F.linear(x, W0 + scale*(B@A), bias)`. Tests assert against the
  SAVED `W_pre`: `‖W_eff - W_pre‖ < 1e-5` (function-preserving) + `BᵀB ≈ I_r` (orthonormal).
- **Freeze W0 AND bias** (LoRA convention — bias ∈ frozen base); only `{A, B, head, mixer}` train.
  Assert `rank <= min(m,n)//2` per wrapped Linear; `scale=8`.
- **Attach scope:** wrap ONLY the `nn.Linear` children of `RecurrentQNet.encoder` (the nn.Sequential);
  NEVER `gru`/`head`/QMIX hypernet. `OLoRALinear` duck-types `nn.Linear` so the Sequential forward works.
- **BC gate (user-signed-off per-grid realistic):** full-run `bc.val_acc_gate_by_grid: {2: 0.50, 3: 0.78}`
  (random floors cop 0.20 / thief 0.25; Bayes ceilings cop ~0.635 @2×2, ~0.823 @3×3 — 0.78 keeps
  headroom). Required for BOTH roles. The CI test does NOT assert the full gate (a tiny vendored set
  under-trains) — it asserts **beats-random by a margin + loss decreases** (overfit smoke).
- **CopNet pretrains with `n_agents=2`** (fixed encoder input width across the ladder via the agent-id
  one-hot); thief `n_agents=1`.
- **Bundle:** `{base_sha, adapters(A,B per wrapped Linear), head}`; `base_sha = sha256(base state_dict,
  sorted keys, fixed dtype)`; load rejects a mismatched base; reconstruct the rebase from the INITIAL
  QR (recompute W0 from W_pre), NOT from trained adapters (else learned deltas are erased).
- **Learner injection seam (re-opens P4b QmixLearner — backward-compatible):** add an OPTIONAL
  `net: RecurrentQNet | None = None` param (default = build internally, unchanged) so finetune can pass
  an OLoRA-wrapped net; the optimizer's net group uses only `p for p in net.parameters() if
  p.requires_grad` (so frozen W0/bias are excluded). Existing P4b behavior unchanged when net is None.

## src/marl/nets/olora_init.py  (T4.3)
```python
def olora_init(weight: Tensor, rank: int, scale: float) -> tuple[Tensor, Tensor, Tensor]
    # thin-QR of the PRETRAINED weight -> (W0_rebased, B, A); W0 = weight - scale*(B@A)
    # asserts rank <= min(m,n)//2; B=(m,r) orthonormal cols, A=(r,n)
```

## src/marl/nets/olora_linear.py  (T4.3)
```python
class OLoRALinear(nn.Module):           # duck-types nn.Linear (in/out_features, forward(x))
    def __init__(self, pretrained: nn.Linear, rank: int, scale: float): ...
    # registers W0 (frozen buffer/param requires_grad=False) + bias (frozen) + trainable A,B (nn.Parameter)
    def forward(self, x): return F.linear(x, self.W0 + self.scale * (self.B @ self.A), self.bias)
def wrap_encoder(net: RecurrentQNet, cfg: dict) -> RecurrentQNet
    # replace each nn.Linear in net.encoder with OLoRALinear(rank=olora.rank, scale=olora.scale);
    # leave gru/head untouched; return the net (mutated in place)
```
- Tests (test_olora_linear.py + test_olora_scope.py): function-preserving (‖forward(x)-pretrained(x)‖
  <1e-5 at init); orthonormal BᵀB≈I; rank-guard raises when rank>min(m,n)//2; ONLY encoder Linears
  become OLoRALinear (head/gru stay plain); trainable set == {A,B}(+head/mixer downstream); ~8×-fewer
  trainable params per wrapped Linear (relational assert, not exact).

## src/marl/data/bc_dataset.py  (T4.4 — extend to role-parameterized)
- Parameterize: `build_bc_dataset(cfg, grid, n_pairs, seed, role="cop")` records `obs[role_key]` with the
  role's greedy (ε=0) expert label (`cop_expert` / `thief_expert`). Add a deterministic EPISODE-level
  train/val split (`bc.val_fraction` + split seed) — split by episode/seed, never adjacent records.

## src/marl/nets/bc_train.py (or scripts/pretrain_bc.py)  (T4.4)
```python
def bc_train(cfg, grid, role, dataset) -> tuple[RecurrentQNet, float]   # CE on the role head; returns (net, val_acc)
def gate_for(cfg, grid) -> float                                        # bc.val_acc_gate_by_grid[min(h,w)]
```
- BC = supervised single-step: `RecurrentQNet.forward(obs, h0=zeros)` → logits → `CrossEntropy(logits,
  a*)`; eval val-acc on the held-out split. CopNet built `n_agents=2`. Base checkpoints →
  `paths.base_checkpoint_dir`. Tests (CI): a TINY vendored set (tests/fixtures/bc_mini_*.npz) →
  overfit drives loss down + val-acc beats random by a margin (NOT the full gate); `gate_for` reads
  the per-grid config.

## src/marl/olora_bundle.py  (T4.5 attach)
```python
def base_sha(net: RecurrentQNet) -> str                                # sha256 of sorted state_dict
def save_bundle(path, net, base_sha) -> None                           # {base_sha, adapters:{name:(A,B)}, head}
def load_bundle(path, base_net) -> RecurrentQNet                       # guard base_sha match; wrap + load A,B + head
```
- Tests: bundle round-trips (save→load reproduces the wrapped net's forward); a TAMPERED base_sha is
  REJECTED (ValueError); a smoke that a QmixLearner built with the OLoRA-wrapped net runs one update
  and only {A,B,head,mixer} have grad (W0/bias frozen).

## config + docs
- config.yaml: replace `bc.val_acc_gate: 0.9` → `bc.val_acc_gate_by_grid: {2: 0.50, 3: 0.78}`; add
  `bc.val_fraction: 0.2`, `bc.split_seed: 7`; `paths.base_checkpoint_dir: results/checkpoints/base`,
  `paths.olora_bundle_dir: results/checkpoints/olora`. Update config_loader REQUIRED keys if needed.
- docs/ANALYSIS.md §0: derive the random floors (cop 1/5, thief 1/4) + per-grid Bayes ceilings (the
  privileged-expert-vs-local-obs imitation gap) justifying the per-grid gates (the honest §7.2 point).
- Reconcile stale `0.9` (docs/TODO.md:181-182, docs/PRD.md:205,469) → per-grid; OLoRA eq-numbering:
  cite ONE scheme ([7]-internal eq3/4/5) in the olora module docstrings.
- pyproject: register a `slow` pytest marker; mark the full-BC/finetune long runs (kept as scripts /
  marked slow) so the default `uv run pytest` stays fast.

## DEFERRED to P4d (with the trainer)
- scripts/finetune_ctde.py — the full OLoRA-wrapped CTDE curriculum-finetune LOOP (3×3→4×4→5×5) reuses
  the P4d SelfPlayTrainer + rollout + centralized replay. P4c builds the injection seam + the attach
  bundle + the one-update smoke; P4d wires the loop. (Forced by the T4.5→T4.6 dependency.)
