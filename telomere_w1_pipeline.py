#!/usr/bin/env python3
"""Fixed-a Wasserstein fits for synthetic and experimental telomere lengths.

Run the fit stage once per fixed a, then the summarize stage after all fits.
The lower boundary is max(0, L-a); geometric elongations are
normalized over the allowed states (no accumulated upper tail at Lmax).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import wasserstein_distance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["fit", "summarize"], required=True)
    parser.add_argument("--mode", choices=["lab", "natural", "synthetic"], required=True)
    parser.add_argument("--data", type=Path, help="Input CSV for lab or natural data")
    parser.add_argument("--group", help="Lab: WT, RIF1, KU70, or an exact replicate")
    parser.add_argument("--strain", help="Natural isolate name")
    parser.add_argument("--extremity", help="Natural chromosome end, e.g. chrI_Left")
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument("--N", type=int, default=3000, help="Synthetic sample size")
    parser.add_argument("--true-a", type=int, default=4)
    parser.add_argument("--true-beta", type=float, default=0.045)
    parser.add_argument("--true-Ls", type=int, default=90)
    parser.add_argument("--true-p", type=float, default=0.026)

    parser.add_argument("--Lmax", type=int, default=1400)
    parser.add_argument("--a-min", type=int, default=2)
    parser.add_argument("--a-max", type=int, default=20)
    parser.add_argument("--fixed-a", type=int, help="Required in the fit stage")
    parser.add_argument("--beta-min", type=float, default=1e-4)
    parser.add_argument("--beta-max", type=float, default=0.20)
    parser.add_argument("--Ls-min", type=int, default=0)
    parser.add_argument("--Ls-max", type=int, default=400)
    parser.add_argument("--p-min", type=float, default=0.003)
    parser.add_argument("--p-max", type=float, default=0.20)
    parser.add_argument("--Ls-alpha", type=float, help="Optional upper bound: data quantile")

    parser.add_argument("--outlier-filter", choices=["none", "upper_iqr", "two_sided_iqr"],
                        default="upper_iqr")
    parser.add_argument("--iqr-k", type=float, default=2.0)
    parser.add_argument("--overflow-data", choices=["error", "drop"], default="error",
                        help="What to do when an observation exceeds Lmax")
    parser.add_argument("--min-n", type=int, default=1001)
    parser.add_argument("--restarts", type=int, default=15)
    parser.add_argument("--cma-maxiter", type=int, default=80)
    parser.add_argument("--cma-popsize", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--solver-maxiter", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args()


def wasserstein_1(probability_a, probability_b):
    """Wasserstein distance between discrete telomere-length distributions.

    Both distributions have support 0, 1, ..., Lmax and contain
    probability weights (not lists of sampled observations).
    """
    lengths = np.arange(len(probability_a))
    return float(wasserstein_distance(
        lengths, lengths,
        u_weights=probability_a,
        v_weights=probability_b,
    ))


def stationary_distribution(a, beta, Ls, p, Lmax, tol, maxiter):
    """Stationary law of the finite chain, computed by power iteration.

    Shortening: b = max(0, i-a).
    When telomerase acts, the geometric gain is conditioned on G <= Lmax-b.
    The geometric probabilities are normalized for each starting state.
    No probability tail is accumulated in state Lmax.
    """
    states = np.arange(Lmax + 1)
    baseline = np.maximum(states - int(a), 0)

    recruitment = np.ones(Lmax + 1)
    above = states > Ls
    recruitment[above] = 1.0 / (1.0 + beta * (states[above] - Ls))

    q = 1.0 - p
    # P(G <= K) = 1 - (1-p)^K; use expm1 for numerical stability.
    # K is at least 1 because a >= 1.
    K = Lmax - baseline
    normalizer = -np.expm1(K * np.log1p(-p))

    pi = np.full(Lmax + 1, 1.0 / (Lmax + 1))
    for _ in range(maxiter):
        recruited = pi * recruitment

        # Cells that do not recruit telomerase lose a base pairs.
        new = np.bincount(
            baseline, weights=pi * (1.0 - recruitment), minlength=Lmax + 1
        )

        # A source at baseline b contributes
        # p*(1-p)^(j-b-1) / (1-(1-p)^(Lmax-b)),  b < j <= Lmax.
        source = np.bincount(
            baseline, weights=recruited / normalizer, minlength=Lmax + 1
        )
        weighted_sum = lfilter([1.0], [1.0, -q], source[:-1])
        new[1:] += p * weighted_sum

        # Small floating-point round-off is corrected; no tail is folded
        # into the final state.
        new = np.maximum(new, 0.0)
        total = new.sum()
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Invalid Markov-chain iteration")
        new /= total

        difference = np.abs(new - pi).sum()
        pi = new
        if difference < tol:
            return pi

    raise RuntimeError("Stationary distribution did not converge")


def read_sample(args):
    """Return integer telomere lengths (one selected biological group)."""
    if args.mode == "synthetic":
        true_pi = stationary_distribution(
            args.true_a, args.true_beta, args.true_Ls, args.true_p,
            args.Lmax, args.tol, args.solver_maxiter,
        )
        rng = np.random.default_rng(args.seed)
        sample = rng.choice(args.Lmax + 1, size=args.N, p=true_pi)
        return sample, true_pi

    if args.data is None:
        raise ValueError("--data is required for real data")
    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()

    if args.mode == "lab":
        if not args.group or "strain" not in df:
            raise ValueError("Lab data need --group and a 'strain' column")
        names = df["strain"].astype(str).str.strip()
        if args.group in ("WT", "RIF1", "KU70"):
            selected = names.eq(args.group) | names.str.fullmatch(args.group + r"_rep\d+")
        else:
            selected = names.eq(args.group)
        df = df.loc[selected]
    else:
        if not args.strain or not args.extremity:
            raise ValueError("Natural data need --strain and --extremity")
        if "strain" not in df:
            raise ValueError("Missing 'strain' column")
        if "extremity" in df:
            ends = df["extremity"].astype(str).str.strip()
        elif "side" in df and ("chrom" in df or "chromosome" in df):
            chrom = "chrom" if "chrom" in df else "chromosome"
            ends = df[chrom].astype(str).str.strip() + "_" + df["side"].astype(str).str.strip().str.title()
        elif "chrom" in df:
            ends = df["chrom"].astype(str).str.strip()
            if not ends.str.contains(r"_(?:Left|Right)$", regex=True).any():
                raise ValueError("The natural dataset needs a Left/Right side column")
        else:
            raise ValueError("Missing extremity or chromosome/side columns")
        selected = df["strain"].astype(str).str.strip().eq(args.strain) & ends.eq(args.extremity)
        df = df.loc[selected]

    length_col = "len" if "len" in df else "length" if "length" in df else None
    if length_col is None:
        raise ValueError("Expected a 'len' or 'length' column")
    values = pd.to_numeric(df[length_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("No observations for this group")
    if np.any(values < 0):
        raise ValueError("Negative telomere lengths in the input")

    if args.outlier_filter != "none":
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        keep = values <= q3 + args.iqr_k * iqr
        if args.outlier_filter == "two_sided_iqr":
            keep &= values >= q1 - args.iqr_k * iqr
        values = values[keep]

    # Do not clip large observations to Lmax: that would create a false spike.
    lengths = np.rint(values).astype(int)
    too_long = lengths > args.Lmax
    if np.any(too_long):
        if args.overflow_data == "error":
            raise ValueError(
                f"{too_long.sum()} observations exceed Lmax={args.Lmax}. "
                "Increase --Lmax or explicitly choose --overflow-data drop."
            )
        print(f"Dropped {too_long.sum()} observations above Lmax={args.Lmax}", flush=True)
        lengths = lengths[~too_long]
    return lengths, None


def prepare_data(args):
    lengths, true_pi = read_sample(args)
    if len(lengths) < args.min_n:
        raise ValueError(f"Only N={len(lengths)} observations (minimum: {args.min_n})")
    counts = np.bincount(lengths, minlength=args.Lmax + 1)
    empirical = counts / counts.sum()
    Ls_max = min(args.Ls_max, args.Lmax)
    if args.Ls_alpha is not None:
        if not 0 < args.Ls_alpha < 1:
            raise ValueError("--Ls-alpha must be between 0 and 1")
        Ls_max = min(Ls_max, int(np.floor(np.quantile(lengths, args.Ls_alpha))))
    if Ls_max < args.Ls_min:
        raise ValueError("The allowed Ls range is empty")
    return counts, empirical, true_pi, Ls_max


def fit_fixed_a(args, empirical, Ls_max, a):
    import cma  # only fitting workers need the CMA-ES package

    bounds_low = np.array([args.beta_min, args.Ls_min, args.p_min], dtype=float)
    bounds_high = np.array([args.beta_max, Ls_max, args.p_max], dtype=float)
    if np.any(bounds_high <= bounds_low):
        raise ValueError("Check beta, Ls and p bounds (each needs a nonzero interval)")

    def decode(x):
        # Search in [0,1]^3; logarithmic coordinates for beta and p.
        beta = np.exp(np.log(bounds_low[0]) + x[0] * np.log(bounds_high[0] / bounds_low[0]))
        Ls = int(np.rint(bounds_low[1] + x[1] * (bounds_high[1] - bounds_low[1])))
        p = np.exp(np.log(bounds_low[2]) + x[2] * np.log(bounds_high[2] / bounds_low[2]))
        return float(beta), Ls, float(p)

    def objective(x):
        beta, Ls, p = decode(x)
        try:
            pi = stationary_distribution(a, beta, Ls, p, args.Lmax, args.tol, args.solver_maxiter)
            return wasserstein_1(empirical, pi)
        except (RuntimeError, FloatingPointError, ValueError):
            return 1e9

    rows = []
    for restart in range(1, args.restarts + 1):
        run_seed = args.seed + 10000 * a + restart
        rng = np.random.default_rng(run_seed)
        start = rng.uniform(0.1, 0.9, size=3)
        initial_beta, initial_Ls, initial_p = decode(start)
        initial_W1 = objective(start)

        result = cma.fmin(
            objective, start.tolist(), 0.22,
            options={"bounds": [[0, 0, 0], [1, 1, 1]],
                     "maxiter": args.cma_maxiter, "popsize": args.cma_popsize,
                     "seed": run_seed, "verbose": -9, "verb_disp": 0},
            restarts=0,
        )
        beta, Ls, p = decode(np.clip(result[0], 0, 1))
        W1 = objective(np.clip(result[0], 0, 1))
        success = bool(np.isfinite(W1) and W1 < 1e9)
        rows.append({
            "a": a, "restart": restart, "seed": run_seed,
            "initial_beta": initial_beta, "initial_Ls": initial_Ls,
            "initial_p": initial_p, "initial_W1": initial_W1,
            "beta": beta, "Ls": Ls, "p": p, "W1": W1,
            "success": success,
        })
        print(f"a={a:2d}  start={restart:2d}/{args.restarts}  W1={W1:.5f}", flush=True)
    return pd.DataFrame(rows)


def summarize(args, counts, empirical, true_pi):
    pieces = args.out / "_parts"
    files = [pieces / f"a_{a:02d}.csv" for a in range(args.a_min, args.a_max + 1)]
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError("Missing fixed-a results:\n" + "\n".join(missing))
    runs = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    # Keep failed attempts in the run table for diagnosing optimization.
    runs["success"] = runs["success"].astype(str).str.lower().eq("true")
    if not runs["success"].any():
        raise RuntimeError("No successful stationary fit")

    # Bootstrap sampling variability at the observed sample size.
    # Synthetic: reference is the known stationary law; real: the empirical law.
    reference = true_pi if true_pi is not None else empirical
    rng = np.random.default_rng(args.seed + 123456)
    bootstrap_counts = rng.multinomial(int(counts.sum()), reference, size=args.bootstrap)
    # Apply the same SciPy Wasserstein distance to every bootstrap sample.
    distances = np.array([
        wasserstein_1(sample_counts / counts.sum(), reference)
        for sample_counts in bootstrap_counts
    ])
    mean_W1 = float(distances.mean())
    sd_W1 = float(distances.std(ddof=1))
    threshold = mean_W1 + 3.0 * sd_W1

    runs["accepted"] = runs["success"] & (runs["W1"] <= threshold)
    runs["rank_accepted"] = pd.Series(pd.NA, index=runs.index, dtype="Int64")
    accepted_indices = runs.loc[runs["accepted"]].sort_values("W1").index
    runs.loc[accepted_indices, "rank_accepted"] = range(1, len(accepted_indices) + 1)
    runs["bootstrap_mean_W1"] = mean_W1
    runs["bootstrap_sd_W1"] = sd_W1
    runs["W_threshold"] = threshold
    runs["N"] = int(counts.sum())
    runs["global_best"] = False
    best_index = runs.loc[runs["success"], "W1"].idxmin()
    runs.loc[best_index, "global_best"] = True
    runs = runs.sort_values(["a", "restart"])

    best = runs.loc[best_index]
    best_pi = stationary_distribution(
        int(best.a), float(best.beta), int(best.Ls), float(best.p),
        args.Lmax, args.tol, args.solver_maxiter,
    )
    distribution = pd.DataFrame({
        "length": np.arange(args.Lmax + 1), "count": counts,
        "empirical": empirical, "best_fit": best_pi,
    })
    if true_pi is not None:
        distribution["true_law"] = true_pi

    profiles = []
    for a, group in runs.groupby("a", sort=True):
        valid = group.loc[group["success"]]
        row = valid.loc[valid["W1"].idxmin()] if not valid.empty else None
        profiles.append({
            "a": int(a), "best_W1": float(row.W1) if row is not None else np.nan,
            "beta": float(row.beta) if row is not None else np.nan,
            "Ls": int(row.Ls) if row is not None else np.nan,
            "p": float(row.p) if row is not None else np.nan,
            "n_runs": len(group), "n_success": len(valid),
            "n_accepted": int(group.accepted.sum()),
            "W_threshold": threshold,
            "bootstrap_mean_W1": mean_W1, "bootstrap_sd_W1": sd_W1,
        })

    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    runs.to_csv(tables / "fit_summary.csv", index=False)
    pd.DataFrame(profiles).to_csv(tables / "profiles.csv", index=False)
    distribution.to_csv(tables / "distributions.csv", index=False)

    print(f"N={counts.sum()} | bootstrap mean={mean_W1:.4f}, SD={sd_W1:.4f}")
    print(f"W_threshold={threshold:.4f} | accepted={len(accepted_indices)}/{len(runs)}")
    print(f"Best: a={int(best.a)}, beta={best.beta:.6g}, Ls={int(best.Ls)}, "
          f"p={best.p:.6g}, W1={best.W1:.4f}")
    print(f"Tables: {tables}")
    if not args.keep_parts:
        shutil.rmtree(pieces)


def main():
    args = parse_args()
    if args.Lmax < 2 or args.a_min < 1 or args.a_min > args.a_max:
        raise ValueError("Check Lmax and the a range")
    if args.bootstrap < 2 or args.restarts < 1 or args.N < 1:
        raise ValueError("Bootstrap needs B>=2 and restarts/N must be positive")
    if args.cma_maxiter < 1 or args.cma_popsize < 2 or args.solver_maxiter < 1:
        raise ValueError("Invalid CMA-ES or stationary-solver settings")
    if not (0 < args.beta_min < args.beta_max and 0 < args.p_min < args.p_max < 1):
        raise ValueError("Invalid beta or p parameter bounds")
    if args.mode == "synthetic" and not (
        0 < args.true_p < 1 and args.true_beta > 0
        and args.true_a >= 1 and 0 <= args.true_Ls <= args.Lmax
    ):
        raise ValueError("Invalid synthetic parameters")

    counts, empirical, true_pi, Ls_max = prepare_data(args)
    if args.stage == "fit":
        if args.fixed_a is None or not args.a_min <= args.fixed_a <= args.a_max:
            raise ValueError("Provide --fixed-a within the configured a range")
        parts = args.out / "_parts"
        parts.mkdir(parents=True, exist_ok=True)
        runs = fit_fixed_a(args, empirical, Ls_max, args.fixed_a)
        path = parts / f"a_{args.fixed_a:02d}.csv"
        runs.to_csv(path, index=False)
        print(f"Saved {path}")
    else:
        summarize(args, counts, empirical, true_pi)


if __name__ == "__main__":
    main()
