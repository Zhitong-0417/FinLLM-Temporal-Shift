# FinGPT Temporal Generalization

A research extension of [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) investigating how financial sentiment classifiers generalize to future data—and how much apparent temporal degradation reflects insufficient historical optimization.

## Research focus

Using Meta-Llama-3-8B with LoRA, this study builds a chronological benchmark of **39,400 financial-news observations**, including **20,002 future observations** across six evaluation windows. Model tuning uses historical data only.

The main contributions are a leakage-controlled temporal benchmark, a historical-only optimization adequacy control, and a layered diagnosis of residual errors by time window, sentiment class, source, and lexical content.

## Key findings

- Historical-only optimization improved pooled future accuracy from **77.59% to 84.25%** and Macro-F1 from **0.7350 to 0.8142**, substantially reducing aggregate temporal degradation.
- A localized failure remained in **F3 (March 2026)**, particularly positive-to-negative errors. Source and lexical analyses identified associations, rather than causal explanations.
- The final out-of-time test **did not support** the hypothesis that exposure to historically unstable lexical features predicts higher future error.

These results show why historical optimization adequacy matters when interpreting future performance drops. Scores measure agreement with Dataset B's LLM-generated sentiment labels.

## Repository structure

```text
├── notebooks/   # Experiments and analysis
├── data/        # Formatted datasets from the earlier reproduction workflow
├── models/      # Baseline adapters and training artifacts
├── results/     # Benchmark data, predictions, metrics, figures, and selected adapters
├── scripts/     # Notebook maintenance and audit tools
└── README.md
```

**Main notebook order:** Day14–17 → Day22–24 → Day25–27: benchmark construction and baseline evaluation → historical optimization and future re-evaluation → residual-failure diagnosis and lexical-instability testing.

Start with the numbered main-line notebooks in [notebooks/](notebooks/). Saved results can be inspected without rerunning training. 

Built on the AI4Finance Foundation's FinGPT framework and [Instruct-FinGPT](https://arxiv.org/abs/2306.12659).
