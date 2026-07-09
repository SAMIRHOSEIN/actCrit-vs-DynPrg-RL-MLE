# Contributing

Thanks for your interest in this research code. It accompanies work on
life-cycle-cost-optimal maintenance planning for deteriorating bridge elements.
Contributions, questions, and reproductions are welcome.

## Getting set up

```bash
git clone https://github.com/SAMIRHOSEIN/actCrit-vs-DynPrg-RL-MLE.git
cd actCrit-vs-DynPrg-RL-MLE
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The `graphviz` Python package also needs the Graphviz **system** binaries
> (`apt install graphviz`, `brew install graphviz`, or the Windows installer)
> to render oblique-tree diagrams.

## How to contribute

1. **Open an issue first** for anything beyond a small fix, so we can agree on
   scope before you invest time.
2. **Branch** from `main` using a descriptive name (e.g. `fix/critic-layers`,
   `docs/readme`).
3. **Keep changes focused.** One logical change per pull request is much easier
   to review.
4. **Preserve reproducibility.** If a change affects numerical results, note the
   before/after in the PR description and update any committed result summaries.
5. **Open a pull request** against `main` with a clear description of *what* and
   *why*.

## Style

- Python code targets 3.10+.
- Match the surrounding style; keep the explanatory comments that document the
  domain assumptions and derivations — they are a deliberate feature of this
  codebase.
- Do not commit generated artifacts (caches, logs, model checkpoints, CSV
  outputs). These are covered by `.gitignore`.

## Reporting problems

Please include your OS, Python version, and the exact versions of `torch`,
`torchrl`, and `tensordict` (these are tightly coupled), plus the command you
ran and the full traceback.
