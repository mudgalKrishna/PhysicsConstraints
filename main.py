# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "torch_geometric",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
# ]
# ///


@app.cell
def _(mo):
    mo.md(
        """
    # MD17 three-way physics-constraint experiment

    Compares three matched models on molecular force/energy prediction (aspirin):

    1. **data_only**: no physics terms.
    2. **soft**: penalty on net force and energy-force consistency, weight λ = 1.0.
    3. **hard**: forces projected to exactly zero net force and zero net torque.
       Energy-force consistency is **not** enforced, so it is the unenforced check.

    Run top to bottom. Training starts only when you press the run button.
    """
    )
    return


@app.cell
def _():
    import json
    import math
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from scipy import stats
    

    return Path, json, math, mo, nn, np, pd, plt, stats, time, torch


@app.cell
def _(Path, json, torch):
    CONFIG = {
        "molecule": "aspirin",
        "data_root": "data/md17",
        "run_dir": "runs/md17_three_way",
        "cutoff": 5.0,
        "n_rbf": 50,
        "hidden": 128,
        "n_interactions": 4,
        "n_train": 1000,
        "n_val": 1000,
        "n_test": 1000,
        "n_shift": 1000,
        "shift_energy_quantile": 0.95,
        "split_seed": 12345,
        "epochs": 300,
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 1e-6,
        "rho_force": 10.0,
        "lam_phys": 1.0,
        "seeds": [0, 1, 2],
        "ablation_lams": [0.01, 0.1, 1.0, 10.0],
        "ablation_seeds": [0],
        "tol_net_force": 1e-2,
        "tol_torque": 1e-2,
        "tol_consistency": 5e-2,
        "eval_chunk": 256,
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUN_DIR = Path(CONFIG["run_dir"])
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

    print("device:", DEVICE, "| cuda available:", torch.cuda.is_available())
    if DEVICE.type == "cuda":
        _p = torch.cuda.get_device_properties(0)
        print("GPU:", _p.name, "| memory GB:", round(_p.total_memory / 1e9, 1))
    return CONFIG, DEVICE, RUN_DIR


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run full experiment (9 main + 3 ablation trainings)")
    run_btn
    return (run_btn,)


@app.cell
def _(CONFIG, DEVICE, torch):
    from torch_geometric.datasets import MD17 as _MD17

    _ds = _MD17(root=CONFIG["data_root"], name=CONFIG["molecule"])
    print("dataset object:", _ds)
    print("num frames:", len(_ds))

    _g0 = _ds[0]
    print("\nfirst frame:")
    print(_g0)
    for _attr in ["pos", "z", "energy", "force"]:
        assert hasattr(_g0, _attr), f"missing attribute '{_attr}' in MD17 sample"

    N_ATOMS = int(_g0.pos.shape[0])
    assert _g0.pos.shape[1] == 3, "positions must be (N, 3)"
    assert _g0.force.shape == _g0.pos.shape, "forces must match positions shape"
    assert _g0.z.shape[0] == N_ATOMS, "z must have one entry per atom"
    print("atoms per molecule:", N_ATOMS)
    print("pos dtype:", _g0.pos.dtype, "| force dtype:", _g0.force.dtype, "| z dtype:", _g0.z.dtype)
    print("energy raw shape:", tuple(_g0.energy.shape))

    _pos, _frc, _eng, _zs = [], [], [], []
    for _d in _ds:
        assert _d.pos.shape[0] == N_ATOMS, "frames have different atom counts"
        _pos.append(_d.pos.float())
        _frc.append(_d.force.float())
        _eng.append(_d.energy.reshape(-1)[0].float())
        _zs.append(_d.z.long())

    _z_all = torch.stack(_zs)
    assert torch.equal(_z_all, _z_all[:1].expand_as(_z_all)), "atom types change between frames"

    DATA = {
        "P": torch.stack(_pos).to(DEVICE),
        "F": torch.stack(_frc).to(DEVICE),
        "E": torch.stack(_eng).to(DEVICE),
        "Z": _z_all[0].to(DEVICE),
    }
    assert int(DATA["Z"].max()) < 10, "embedding table assumes atomic numbers < 10"

    print("\nstacked shapes:")
    print("  positions:", tuple(DATA["P"].shape))
    print("  forces:   ", tuple(DATA["F"].shape))
    print("  energies: ", tuple(DATA["E"].shape))
    print("  atom types (unique Z):", torch.unique(DATA["Z"]).tolist())
    print("energy min/mean/max:", float(DATA["E"].min()), float(DATA["E"].mean()), float(DATA["E"].max()))
    print("force std:", float(DATA["F"].std()))
    print("NOTE: units are whatever PyG stores. All tolerances in this notebook are relative.")
    return DATA, N_ATOMS


@app.cell
def _(CONFIG, DATA, DEVICE, RUN_DIR, np, torch):
    _rng = np.random.default_rng(CONFIG["split_seed"])
    _e = DATA["E"].cpu().double().numpy()
    _thr = float(np.quantile(_e, CONFIG["shift_energy_quantile"]))
    _tail = np.where(_e >= _thr)[0]
    _main = np.where(_e < _thr)[0]
    _perm = _rng.permutation(_main)

    _nt, _nv, _ne, _ns = CONFIG["n_train"], CONFIG["n_val"], CONFIG["n_test"], CONFIG["n_shift"]
    assert len(_perm) >= _nt + _nv + _ne, "not enough in-distribution frames"
    assert len(_tail) >= _ns, "not enough tail frames for the shift split"

    _splits = {
        "train": _perm[:_nt],
        "val": _perm[_nt:_nt + _nv],
        "test": _perm[_nt + _nv:_nt + _nv + _ne],
        "shift": _rng.choice(_tail, size=_ns, replace=False),
    }
    _all = np.concatenate(list(_splits.values()))
    assert len(np.unique(_all)) == len(_all), "splits overlap"

    _file = RUN_DIR / "splits.npz"
    if _file.exists():
        _old = np.load(_file)
        for _k in _splits:
            assert np.array_equal(_old[_k], _splits[_k]), f"frozen split '{_k}' has changed"
        print("splits identical to frozen file:", _file)
    else:
        np.savez(_file, **_splits)
        print("splits written and frozen:", _file)

    for _k, _v in _splits.items():
        print(f"{_k:>6}: n={len(_v):5d}  mean E={_e[_v].mean():.4f}  max E={_e[_v].max():.4f}")
    print(f"shift threshold (energy quantile {CONFIG['shift_energy_quantile']}): {_thr:.4f}")

    SPLITS_T = {k: torch.as_tensor(v, dtype=torch.long, device=DEVICE) for k, v in _splits.items()}
    _tr = SPLITS_T["train"]
    MU_E = float(DATA["E"][_tr].mean())
    SIG_E = float(DATA["E"][_tr].std())
    SIG_F = float(DATA["F"][_tr].std())
    print(f"train stats: mu_E={MU_E:.4f} sig_E={SIG_E:.4f} sig_F={SIG_F:.4f}")
    return MU_E, SIG_E, SIG_F, SPLITS_T


@app.cell
def _(CONFIG, DEVICE, DATA, N_ATOMS, math, nn, torch):
    _a, _b = torch.meshgrid(torch.arange(N_ATOMS), torch.arange(N_ATOMS), indexing="ij")
    _keep = _a != _b
    PAIR_I = _a[_keep].to(DEVICE)
    PAIR_J = _b[_keep].to(DEVICE)
    CUTOFF = CONFIG["cutoff"]

    def make_batch(idx, data=DATA):
        B = int(idx.shape[0])
        pos = data["P"][idx].reshape(B * N_ATOMS, 3).clone()
        z = data["Z"].repeat(B)
        batch = torch.arange(B, device=DEVICE).repeat_interleave(N_ATOMS)
        base = (torch.arange(B, device=DEVICE) * N_ATOMS)[:, None]
        I = (base + PAIR_I[None, :]).reshape(-1)
        J = (base + PAIR_J[None, :]).reshape(-1)
        dist = (pos[I] - pos[J]).norm(dim=-1).detach()
        keep = dist < CUTOFF
        return {
            "pos": pos, "z": z, "batch": batch,
            "I": I[keep], "J": J[keep], "B": B,
            "E": data["E"][idx], "F": data["F"][idx].reshape(-1, 3),
        }

    class Interaction(nn.Module):
        def __init__(self, hidden, n_rbf):
            super().__init__()
            self.filt = nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.in2f = nn.Linear(hidden, hidden, bias=False)
            self.upd = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

        def forward(self, h, rbf, env, I, J):
            W = self.filt(rbf) * env[:, None]
            msg = self.in2f(h)[J] * W
            agg = torch.zeros_like(h).index_add(0, I, msg)
            return h + self.upd(agg)

    class Net(nn.Module):
        def __init__(self, hidden, n_rbf, n_int, cutoff):
            super().__init__()
            self.cutoff = cutoff
            centers = torch.linspace(0.0, cutoff, n_rbf)
            self.register_buffer("centers", centers)
            spacing = float(centers[1] - centers[0])
            self.gamma = 0.5 / spacing ** 2
            self.emb = nn.Embedding(10, hidden)
            self.inter = nn.ModuleList([Interaction(hidden, n_rbf) for _ in range(n_int)])
            self.atom_out = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            self.edge_force = nn.Sequential(
                nn.Linear(2 * hidden + n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )

        def forward(self, pos, z, batch, I, J, B):
            d_vec = pos[I] - pos[J]
            dist = d_vec.norm(dim=-1).clamp(min=1e-8)
            u = d_vec / dist[:, None]
            env = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)
            rbf = torch.exp(-self.gamma * (dist[:, None] - self.centers[None, :]) ** 2)
            h = self.emb(z)
            for layer in self.inter:
                h = layer(h, rbf, env, I, J)
            e_atom = self.atom_out(h).squeeze(-1)
            E_raw = torch.zeros(B, device=pos.device, dtype=pos.dtype).index_add(0, batch, e_atom)
            f_scalar = self.edge_force(torch.cat([h[I], h[J], rbf], dim=-1)).squeeze(-1) * env
            F_raw = torch.zeros_like(pos).index_add(0, I, f_scalar[:, None] * u)
            return E_raw, F_raw

    def _com_r(pos, batch, B):
        cnt = torch.zeros(B, device=pos.device, dtype=pos.dtype).index_add(
            0, batch, torch.ones_like(pos[:, 0])
        ).clamp(min=1.0)
        com = torch.zeros(B, 3, device=pos.device, dtype=pos.dtype).index_add(0, batch, pos)
        return pos - (com / cnt[:, None])[batch]

    def project_forces(F, pos, batch, B):
        """Exact projection onto zero net force and zero net torque (per molecule)."""
        dt = torch.float64
        Fd = F.to(dt)
        r = _com_r(pos.to(dt), batch, B)
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        zero = torch.zeros_like(x)
        skew = torch.stack([
            torch.stack([zero, -z, y], -1),
            torch.stack([z, zero, -x], -1),
            torch.stack([-y, x, zero], -1),
        ], -2)
        eye = torch.eye(3, device=F.device, dtype=dt).expand(r.shape[0], 3, 3)
        G = torch.cat([eye, skew], dim=1)
        w = torch.zeros(B, 6, device=F.device, dtype=dt).index_add(
            0, batch, torch.einsum("mkj,mj->mk", G, Fd)
        )
        Mmat = torch.zeros(B, 6, 6, device=F.device, dtype=dt).index_add(
            0, batch, G @ G.transpose(1, 2)
        )
        Mmat = Mmat + 1e-9 * torch.eye(6, device=F.device, dtype=dt)
        mu = torch.linalg.solve(Mmat, w.unsqueeze(-1)).squeeze(-1)
        corr = torch.einsum("mkj,mk->mj", G, mu[batch])
        return (Fd - corr).to(F.dtype)

    def _gsum(x, batch, B):
        out = torch.zeros((B,) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype)
        return out.index_add(0, batch, x)

    def graph_mean(x, batch, B):
        cnt = _gsum(torch.ones_like(x), batch, B).clamp(min=1.0)
        return _gsum(x, batch, B) / cnt

    def phys_metrics(F, pos, gradE, batch, B):
        """Relative violation residuals per molecule. gradE = dE/dpos, so the
        energy-force consistency residual is F + gradE."""
        r = _com_r(pos, batch, B)
        netF = _gsum(F, batch, B).norm(dim=-1)
        absF = _gsum(F.norm(dim=-1), batch, B)
        netT = _gsum(torch.cross(r, F, dim=-1), batch, B).norm(dim=-1)
        absT = _gsum(r.norm(dim=-1) * F.norm(dim=-1), batch, B)
        cons_num = _gsum((F + gradE).pow(2).sum(-1), batch, B).sqrt()
        cons_den = _gsum(F.pow(2).sum(-1), batch, B).sqrt()
        return {
            "net_force": netF / (absF + 1e-12),
            "torque": netT / (absT + 1e-12),
            "consistency": cons_num / (cons_den + 1e-12),
        }

    return Net, graph_mean, make_batch, phys_metrics, project_forces


@app.cell
def _(
    CONFIG, DEVICE, DATA, MU_E, SIG_E, SIG_F, SPLITS_T,
    make_batch, Net, project_forces, phys_metrics, graph_mean, np, time, torch,
):
    def build_model():
        return Net(CONFIG["hidden"], CONFIG["n_rbf"], CONFIG["n_interactions"], CONFIG["cutoff"]).to(DEVICE)

    def forward_model(model, b, mode, need_grad):
        pos = b["pos"]
        if need_grad:
            pos.requires_grad_(True)
        E_raw, F_raw = model(pos, b["z"], b["batch"], b["I"], b["J"], b["B"])
        E_pred = SIG_E * E_raw + MU_E
        F_dir = SIG_F * F_raw
        if mode == "hard":
            F_used = project_forces(F_dir, pos, b["batch"], b["B"])
        else:
            F_used = F_dir
        gradE = None
        if need_grad:
            gradE = torch.autograd.grad(E_pred.sum(), pos, create_graph=model.training)[0]
        return E_pred, F_used, gradE

    def data_loss(E_pred, F_used, E_true, F_true):
        le = (((E_pred - E_true) / SIG_E) ** 2).mean()
        lf = (((F_used - F_true) / SIG_F) ** 2).mean()
        return le + CONFIG["rho_force"] * lf

    def phys_penalty(F_used, gradE, b):
        sumF = torch.zeros(b["B"], 3, device=DEVICE).index_add(0, b["batch"], F_used)
        net = sumF.pow(2).sum(-1).mean() / SIG_F ** 2
        cons = (F_used + gradE).pow(2).sum(-1).mean() / SIG_F ** 2
        return net + cons

    def val_loss(model, mode, ids):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(ids), CONFIG["eval_chunk"]):
                b = make_batch(ids[s:s + CONFIG["eval_chunk"]])
                E_p, F_u, _ = forward_model(model, b, mode, False)
                tot += float(data_loss(E_p, F_u, b["E"], b["F"])) * b["B"]
                n += b["B"]
        return tot / n

    def train_run(mode, seed, lam=0.0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model()
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])
        tr_ids, va_ids = SPLITS_T["train"], SPLITS_T["val"]
        bs = CONFIG["batch_size"]
        best = {"val": float("inf"), "epoch": -1, "state": None}

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for ep in range(CONFIG["epochs"]):
            model.train()
            perm = tr_ids[torch.randperm(len(tr_ids), device=DEVICE)]
            for s in range(0, len(perm), bs):
                b = make_batch(perm[s:s + bs])
                E_p, F_u, gE = forward_model(model, b, mode, need_grad=(mode == "soft"))
                loss = data_loss(E_p, F_u, b["E"], b["F"])
                if mode == "soft":
                    loss = loss + lam * phys_penalty(F_u, gE, b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            v = val_loss(model, mode, va_ids)
            if v < best["val"]:
                best = {
                    "val": v, "epoch": ep,
                    "state": {k: t.detach().clone() for k, t in model.state_dict().items()},
                }
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        model.load_state_dict(best["state"])
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else float("nan")
        info = {
            "n_params": n_params,
            "best_epoch": best["epoch"],
            "best_val_loss": best["val"],
            "train_time_s": train_time,
            "train_peak_mem_MB": peak,
        }
        return model, info

    def evaluate(model, mode, ids):
        model.eval()
        acc = {k: [] for k in ["abs_E", "mae_F", "net_force", "torque", "consistency"]}
        for s in range(0, len(ids), CONFIG["eval_chunk"]):
            b = make_batch(ids[s:s + CONFIG["eval_chunk"]])
            E_p, F_u, gE = forward_model(model, b, mode, True)
            with torch.no_grad():
                acc["abs_E"].append((E_p - b["E"]).abs().detach())
                acc["mae_F"].append(graph_mean((F_u - b["F"]).abs().mean(-1).detach(), b["batch"], b["B"]))
                pm = phys_metrics(F_u.detach(), b["pos"].detach(), gE.detach(), b["batch"], b["B"])
                for k in pm:
                    acc[k].append(pm[k])
        out = {k: torch.cat(v).cpu().numpy() for k, v in acc.items()}

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, len(ids), CONFIG["eval_chunk"]):
                b = make_batch(ids[s:s + CONFIG["eval_chunk"]])
                forward_model(model, b, mode, False)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        out["ms_per_molecule"] = 1000 * (time.time() - t0) / len(ids)
        return out

    return build_model, data_loss, evaluate, forward_model, phys_penalty, train_run, val_loss


@app.cell
def _(
    CONFIG, DEVICE, SPLITS_T,
    build_model, forward_model, make_batch, phys_metrics, torch,
):
    _ids = SPLITS_T["test"][:8]
    _b = make_batch(_ids)
    _report = {}
    for _mode in ["data_only", "soft", "hard"]:
        _m = build_model()
        _E, _F, _g = forward_model(_m, _b, _mode, True)
        assert _E.shape == (8,), f"energy shape {tuple(_E.shape)}"
        assert _F.shape == (_b["pos"].shape[0], 3), f"force shape {tuple(_F.shape)}"
        assert torch.isfinite(_E).all() and torch.isfinite(_F).all() and torch.isfinite(_g).all()
        _pm = phys_metrics(_F.detach(), _b["pos"].detach(), _g.detach(), _b["batch"], 8)
        _report[_mode] = {k: float(v.max()) for k, v in _pm.items()}
        _report[_mode]["params"] = sum(p.numel() for p in _m.parameters())

    for _mode, _vals in _report.items():
        print(_mode, {k: (f"{v:.3e}" if k != "params" else v) for k, v in _vals.items()})
    assert _report["hard"]["net_force"] < 1e-5, "hard projection: net force not zero"
    assert _report["hard"]["torque"] < 1e-5, "hard projection: torque not zero"
    assert len({v["params"] for v in _report.values()}) == 1, "parameter counts differ across models"
    print("smoke test passed: shapes finite, projection exact, parameter counts matched")
    return


@app.cell
def _(CONFIG, RUN_DIR, SPLITS_T, DATA, evaluate, mo, pd, run_btn, torch, train_run):
    mo.stop(
        not run_btn.value,
        mo.md("Press **Run full experiment** above to start. Setup and smoke tests have already run."),
    )

    def to_rows(model_name, lam, seed, split, res):
        rows = []

        def add(metric, value):
            rows.append({"model": model_name, "lam": float(lam), "seed": seed,
                         "split": split, "metric": metric, "value": float(value)})

        add("MAE_E", res["abs_E"].mean())
        add("MAE_F", res["mae_F"].mean())
        for k, tol in [("net_force", "tol_net_force"), ("torque", "tol_torque"),
                       ("consistency", "tol_consistency")]:
            add(f"mean_rel_{k}", res[k].mean())
            add(f"viol_rate_{k}", (res[k] > CONFIG[tol]).mean())
        add("ms_per_molecule", res["ms_per_molecule"])
        return rows

    jobs = []
    for _s in CONFIG["seeds"]:
        jobs += [("data_only", _s, 0.0), ("hard", _s, 0.0), ("soft", _s, CONFIG["lam_phys"])]
    for _lam in CONFIG["ablation_lams"]:
        if _lam == CONFIG["lam_phys"]:
            continue
        for _s in CONFIG["ablation_seeds"]:
            jobs.append(("soft", _s, _lam))

    rows, PER_MOL = [], {}
    for mode, seed, lam in jobs:
        print(f"training {mode:9s} lam={lam:<5} seed={seed}")
        model, info = train_run(mode, seed, lam)
        for k, v in info.items():
            rows.append({"model": mode, "lam": float(lam), "seed": seed,
                         "split": "run", "metric": k, "value": float(v)})
        for split in ["test", "shift"]:
            res = evaluate(model, mode, SPLITS_T[split])
            PER_MOL[(mode, float(lam), seed, split)] = res
            rows += to_rows(mode, lam, seed, split, res)
        torch.save(model.state_dict(), RUN_DIR / f"{mode}_lam{lam}_seed{seed}.pt")

    RESULTS = pd.DataFrame(rows)
    RESULTS.to_csv(RUN_DIR / "results_long.csv", index=False)
    print("saved", len(RESULTS), "rows to", RUN_DIR / "results_long.csv")
    return PER_MOL, RESULTS


@app.cell
def _(CONFIG, RESULTS, mo, pd, stats):
    _keep = (RESULTS["model"] != "soft") | (RESULTS["lam"] == CONFIG["lam_phys"])
    _main = RESULTS[_keep & RESULTS["split"].isin(["test", "shift"])]
    SUMMARY = (
        _main.groupby(["model", "split", "metric"])["value"]
        .agg(["mean", "std", "count"]).reset_index()
    )

    _rows = []
    for _metric in ["MAE_E", "MAE_F", "viol_rate_net_force", "viol_rate_torque", "viol_rate_consistency"]:
        _sub = _main[(_main["split"] == "test") & (_main["metric"] == _metric)]
        _piv = _sub.pivot_table(index="seed", columns="model", values="value")
        for _m in ["soft", "hard"]:
            if _m in _piv.columns and "data_only" in _piv.columns:
                _t, _p = stats.ttest_rel(_piv[_m], _piv["data_only"])
                _rows.append({
                    "metric": _metric,
                    "comparison": f"{_m} - data_only",
                    "mean_diff": float((_piv[_m] - _piv["data_only"]).mean()),
                    "p_value": float(_p),
                })
    PAIRED = pd.DataFrame(_rows)

    mo.vstack([
        mo.md("## Summary (mean and std across seeds)"),
        mo.ui.table(SUMMARY),
        mo.md("## Paired tests on test split (3 seeds, low power; treat p-values as rough)"),
        mo.ui.table(PAIRED),
    ])
    return


@app.cell
def _(CONFIG, PER_MOL, RESULTS, RUN_DIR, np, plt):
    _figdir = RUN_DIR / "figs"
    _figdir.mkdir(parents=True, exist_ok=True)
    _col = {"data_only": "tab:gray", "soft": "tab:orange", "hard": "tab:blue"}
    _keep = (RESULTS["model"] != "soft") | (RESULTS["lam"] == CONFIG["lam_phys"])

    def _vals(model, metric, split="test"):
        _d = RESULTS[_keep & (RESULTS["model"] == model) & (RESULTS["split"] == split) & (RESULTS["metric"] == metric)]
        return _d.sort_values("seed")["value"].to_numpy()

    # Fig 1: error vs violation
    _f, _a = plt.subplots(figsize=(6, 4))
    for _m in _col:
        _a.scatter(_vals(_m, "MAE_F"), _vals(_m, "viol_rate_consistency"), c=_col[_m], label=_m, s=60)
    _a.set_xlabel("test force MAE")
    _a.set_ylabel("consistency violation rate")
    _a.set_title("Error vs physics violation (one point per seed)")
    _a.legend()
    _f.savefig(_figdir / "fig1_error_vs_violation.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Fig 2: in-distribution vs shift
    _f, _a = plt.subplots(figsize=(7, 4))
    _x, _w = np.arange(3), 0.35
    for _i, _split in enumerate(["test", "shift"]):
        _means = [_vals(m, "MAE_F", _split).mean() for m in _col]
        _errs = [_vals(m, "MAE_F", _split).std() for m in _col]
        _a.bar(_x + (_i - 0.5) * _w, _means, _w, yerr=_errs, label=_split)
    _a.set_xticks(_x, list(_col))
    _a.set_ylabel("force MAE")
    _a.set_title("In-distribution vs energy-tail shift")
    _a.legend()
    _f.savefig(_figdir / "fig2_shift_error.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Fig 3: violation rate per constraint
    _f, _a = plt.subplots(figsize=(8, 4))
    _cons = ["net_force", "torque", "consistency"]
    _x, _w = np.arange(len(_cons)), 0.25
    for _i, _m in enumerate(_col):
        _rates = [_vals(_m, f"viol_rate_{c}").mean() for c in _cons]
        _a.bar(_x + (_i - 1) * _w, _rates, _w, color=_col[_m], label=_m)
    _a.set_xticks(_x, _cons)
    _a.set_ylabel("violation rate (test)")
    _a.set_title("Violation rate per constraint")
    _a.legend()
    _f.savefig(_figdir / "fig3_violation_per_constraint.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Fig 4: compute cost
    _f, _a = plt.subplots(figsize=(7, 4))
    _t = [_vals(m, "train_time_s", "run").mean() for m in _col]
    _a.bar(list(_col), _t, color=list(_col.values()))
    _a.set_ylabel("training wall time (s)")
    _a.set_title("Training cost (mean over seeds)")
    _f.savefig(_figdir / "fig4_training_time.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Fig 5: residual distribution on test, seed 0
    _f, _a = plt.subplots(figsize=(6, 4))
    for _m in _col:
        _lam = CONFIG["lam_phys"] if _m == "soft" else 0.0
        _arr = PER_MOL[(_m, float(_lam), CONFIG["seeds"][0], "test")]["consistency"]
        _a.hist(np.log10(_arr + 1e-12), bins=40, alpha=0.5, color=_col[_m], label=_m)
    _a.axvline(np.log10(CONFIG["tol_consistency"]), ls="--", color="k", label="tolerance")
    _a.set_xlabel("log10 relative consistency residual (test, seed 0)")
    _a.set_title("Residual distribution")
    _a.legend()
    _f.savefig(_figdir / "fig5_residual_hist.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Fig 6: ablation over lambda (seed 0), test split
    _ab = RESULTS[(RESULTS["model"] == "soft") & (RESULTS["split"] == "test")
                  & (RESULTS["seed"] == CONFIG["ablation_seeds"][0])]
    _mae = _ab[_ab["metric"] == "MAE_F"].sort_values("lam")
    _vc = _ab[_ab["metric"] == "viol_rate_consistency"].sort_values("lam")
    _f, _a = plt.subplots(figsize=(6, 4))
    _a.semilogx(_mae["lam"], _mae["value"], "o-", label="test force MAE")
    _a.set_xlabel("lambda (soft penalty weight)")
    _a.set_ylabel("force MAE")
    _a2 = _a.twinx()
    _a2.semilogx(_vc["lam"], _vc["value"], "s--", color="tab:red", label="consistency viol. rate")
    _a2.set_ylabel("consistency violation rate")
    _a.set_title("Ablation: soft-penalty weight (seed 0)")
    _f.savefig(_figdir / "fig6_lambda_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(CONFIG, PER_MOL, SPLITS_T, mo, pd):
    _ids = SPLITS_T["test"].cpu().numpy()
    _seed = CONFIG["seeds"][0]
    _tn, _tc = CONFIG["tol_net_force"], CONFIG["tol_consistency"]
    _do = PER_MOL[("data_only", 0.0, _seed, "test")]
    _hd = PER_MOL[("hard", 0.0, _seed, "test")]
    _so = PER_MOL[("soft", float(CONFIG["lam_phys"]), _seed, "test")]

    _df = pd.DataFrame({
        "test_index": range(len(_ids)),
        "data_only_net_rel": _do["net_force"],
        "hard_net_rel": _hd["net_force"],
        "data_only_cons_rel": _do["consistency"],
        "soft_cons_rel": _so["consistency"],
        "hard_cons_rel": _hd["consistency"],
    })

    _caught_mask = (_df.data_only_net_rel > _tn) & (_df.hard_net_rel <= _tn)
    _caught = _df[_caught_mask].sort_values("data_only_net_rel", ascending=False).head(5)
    _missed = _df[_df.hard_cons_rel > _tc].sort_values("hard_cons_rel", ascending=False).head(5)

    mo.vstack([
        mo.md(f"**Caught by hard:** data-only violates net force, hard does not. "
              f"Count: {int(_caught_mask.sum())} (seed 0, test split)"),
        mo.ui.table(_caught),
        mo.md(f"**Missed by hard:** hard violates energy-force consistency. "
              f"Count: {int((_df.hard_cons_rel > _tc).sum())} (seed 0, test split)"),
        mo.ui.table(_missed),
    ])
    return


@app.cell
def _(RUN_DIR, mo):
    import zipfile

    _zip_path = RUN_DIR.parent / f"{RUN_DIR.name}_results.zip"
    _files = [p for p in RUN_DIR.rglob("*") if p.is_file()]

    with zipfile.ZipFile(_zip_path, "w", zipfile.ZIP_DEFLATED) as _zf:
        for _p in _files:
            _zf.write(_p, arcname=_p.relative_to(RUN_DIR.parent))

    print(f"Zipped {len(_files)} files to {_zip_path}")
    print(f"Size: {_zip_path.stat().st_size / 1e6:.1f} MB")

    mo.download(
        data=_zip_path.read_bytes(),
        filename=_zip_path.name,
        label="Download all results (zip)",
    )
    return


if __name__ == "__main__":
    app.run()
