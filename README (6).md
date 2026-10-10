# MD17 physics-constraint study: data-only vs soft penalty vs hard projection

Compares three matched graph neural networks on aspirin (MD17) that predict energy and per-atom forces:

1. **data_only**: trained on energy and force error only.
2. **soft**: same network plus a penalty on net force and energy-force inconsistency (λ = 1.0).
3. **hard**: forces projected exactly onto zero net force and zero net torque. Energy-force consistency is *not* enforced, so it is the check the hard model fails by design.

Status: **preliminary**. One completed comparison at 300 epochs, 3 seeds. Longer training was started but is not part of these results.

## Contents

- `main.py`: the full experiment as a marimo notebook (data, splits, models, training, evaluation, figures, zip export).
- `results/Results.zip`: outputs of the run, including `config.json`, frozen `splits.npz`, `results_long.csv`, six figures, and model weights.
- `requirements.txt`, `.gitignore`.

## Reproduce

```bash
pip install -r requirements.txt
marimo edit main.py
```

Press **Run full experiment**. The notebook downloads MD17 through PyTorch Geometric. The frozen split in `runs/md17_three_way/splits.npz` is checked against the saved one if present. The full run is 12 trainings (9 main plus 3 ablation, since λ = 1.0 is shared with the main runs), each taking about 50 to 100 seconds on the GPU used.

## Setup

| Item | Value |
|---|---|
| Dataset | MD17, aspirin, via `torch_geometric.datasets.MD17` |
| Splits | train 1000, val 1000, test 1000 (in-distribution); shift 1000 from top 5% of energies |
| Split seed | 12345 (fixed; saved to `splits.npz`) |
| Model | 4-layer message-passing network, hidden 128, 50 radial basis functions, cutoff 5 Å, 347,138 parameters |
| Training | Adam, lr 5e-4, weight decay 1e-6, cosine schedule, batch 32, 300 epochs, best checkpoint by validation loss |
| Seeds | 0, 1, 2 |
| Soft λ | 1.0 (fixed; not tuned) |
| Tolerances (relative) | net force 1e-2, torque 1e-2, consistency 5e-2 |

Tolerances are relative (residual divided by the magnitude of the forces), so they do not depend on the dataset's units.

## Results

Test split, mean over 3 seeds. Lower is better for errors; violation rates are fractions of test molecules over tolerance.

| Model | Energy MAE | Force MAE | Net-force viol. | Torque viol. | Consistency viol. |
|---|---|---|---|---|---|
| data_only | 2.07 | 2.24 | 0.86 | 0.71 | 1.00 |
| hard | 2.07 | 2.24 | **0.00** | **0.00** | 1.00 (by design) |
| soft (λ = 1.0) | **1.37** | 2.66 | 0.01 | 0.17 | 1.00 |

Shift split (top 5% energies), mean over 3 seeds:

| Model | Energy MAE | Force MAE |
|---|---|---|
| data_only | 3.89 | 2.99 |
| hard | 3.89 | 3.01 |
| soft (λ = 1.0) | 2.58 | 3.50 |

Paired tests on test split, soft or hard minus data_only (3 seeds, indicative only):

| Metric | Soft − data_only | p | Hard − data_only | p |
|---|---|---|---|---|
| Force MAE | +0.42 | 0.003 | +0.004 | 0.58 |
| Energy MAE | −0.69 | 0.001 | +0.002 | 0.73 |

Compute (training wall time per 300-epoch run, one GPU): data_only about 48 s, hard about 56 s, soft about 99 s. Inference is about 0.01 ms per molecule for all three models.

### λ ablation (seed 0, test split)

Note: this ablation was reported on the test split. It is descriptive only and was not used to choose λ.

| λ | Energy MAE | Force MAE | Consistency viol. | Mean consistency residual |
|---|---|---|---|---|
| 0.01 | 1.94 | 2.20 | 1.00 | 0.38 |
| 0.1 | 1.73 | 2.25 | 1.00 | 0.29 |
| 1.0 | 1.37 | 2.66 | 1.00 | 0.15 |
| 10.0 | 1.51 | 3.75 | 0.98 | 0.08 |

Increasing λ lowers the consistency residual, but force error rises with it. No λ tested gets the consistency violation rate below the tolerance target.

## Findings

- **Hard projection removes net-force and torque violations** without a measurable change in force or energy error (p > 0.5).
- **Soft penalty trades force accuracy for energy accuracy** at λ = 1.0: energy MAE drops by about a third, while force MAE rises by about 19%. Both differences are significant on 3 seeds.
- **Neither method enforces energy-force consistency.** The hard model is not designed to; the soft penalty does not bring the violation rate down at any λ tested.
- **The hard model's errors are the same as data-only**, so the projection does not hurt accuracy here.

## Limitations

- **Undertrained.** The best validation epoch falls between 278 and 289 of 300 for every model, so validation loss was still improving at the end. Accuracy numbers will probably change with longer training. The test energy MAE of about 2 kcal/mol is well above published MD17 aspirin numbers, which are typically under 1 kcal/mol, but I have not checked this against the literature.
- **Three seeds only.** P-values are rough.
- **Shift split is defined here.** The top-5%-energy split is not a standard MD17 benchmark and has not been agreed with the project lead.
- **λ ablation was reported on the test split** and only for seed 0.
- **Energy-tail shift is one kind of shift.** Results may not carry over to other shifts such as temperature or new molecules.

## Next steps

- Train longer and check whether the force-error trade-off stays.
- Tune λ on the validation split and report test results only once.
- Agree the shift definition with the project lead.
- Add more seeds if compute allows.
