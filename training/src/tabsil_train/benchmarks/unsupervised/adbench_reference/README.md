# ADBench comparison results (copied)

Copied unchanged from fm-agent-bench (the anomaly track, `docs/anomaly.md`) at commit
`e4235ac8e6c42ac6b781ac41f8d2102fe5ba1e33` (2026-11-25, `results/`), read by
`tabsil_train.benchmarks.unsupervised.adbench`:

| File | Content |
|---|---|
| `anomaly.csv` | per (table, split) test AUROC (`score`) and AUPRC (`aupr`) of pyod 3.6.8's 10 default methods, `iforest+clean`, `tabpfn_unsup` (tabpfn-extensions 0.6.3 `TabPFNUnsupervisedModel.outliers`, TabPFN v2 weights, 10 orders) and `tabpfn_unsup+clean`. PCA misses cardio s1/s3 and CBLOF Waveform s1 (failed there; the Elo imputes them with IsolationForest). Some rows repeat with the same score |
| `anomaly_reference.csv` | ADBench paper (arXiv:2206.09426) Table D4/D5: seed 1-3 mean AUROC and AUPRC of 14 unsupervised methods (pyod 1.0.0) on the 12 tables |
| `anomaly_elo.csv` | the fm-agent-bench Elo (IsolationForest = 1000); `tabsil_train.benchmarks.unsupervised.adbench.elo` reproduces its `fm` pool |
