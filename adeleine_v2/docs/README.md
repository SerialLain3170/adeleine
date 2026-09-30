# Adeleine v2 Engineering Documentation

These documents describe the current FLUX.2 Klein implementation, not the older GAN-based projects in the repository.

## Documents

- [Architecture](architecture.md): executed model and data path, condition token layout, trainable boundary, and implementation constraints.
- [Training methodology](training_methodology.md): dataset construction, task mixture, augmentation, objective, optimization, DDP, and checkpoint semantics.
- [Validation](validation.md): fixed holdout sampling, different-image reference tests, metrics, and current results.
- [Takeover runbook](takeover.md): dated live-run state, paths, monitoring, recovery, and next actions.

## Source-of-truth order

When documentation and code disagree, use this order:

1. The active process command and checkpoint `adapter_config.json` for the currently running experiment.
2. `smoke_train_flux_klein.py`, `openniji.py`, and `reference_conditioning.py` for executed behavior.
3. These documents for intent and operational context.
4. `adeleine_v2/README.md` for broader usage examples.

The repository was dirty when these documents were written. Read [takeover.md](takeover.md) before changing or committing files.
