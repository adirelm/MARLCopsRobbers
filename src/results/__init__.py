"""Results & figures (§7) — aggregate ``results/runs/*.jsonl`` into the F1-F6 figures.

Pure analysis layer on top of the SDK's training output: ``run_log`` writes per-round
records, ``aggregate`` reduces them to per-method mean±SE curves + final stats, ``plots``
renders the matplotlib figures, and ``make_figures`` is the single regen entry. No
business logic lives here — it only reads the append-only run log the SDK produced.
"""
