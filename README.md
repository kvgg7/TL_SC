# Telomere length inference (fixed-a, W1)

This pipeline fits the stationary distribution of a finite-state stochastic telomere model.
It uses a **normalized truncated geometric law** for elongations: excess
geometric gain is not concentrated at `Lmax`.
For each fixed shortening parameter `a`, it optimizes `(beta, Ls, p)` with independent
CMA-ES starts. A bootstrap threshold based on sampling variability is used to mark
acceptable fits, and fits are ranked by the one-dimensional Wasserstein distance:

The objective uses `scipy.stats.wasserstein_distance` with the telomere lengths
`0, 1, ..., Lmax` as the support and each distribution as probability weights:

```python
from scipy.stats import wasserstein_distance

W1 = wasserstein_distance(
    lengths, lengths,
    u_weights=empirical,
    v_weights=stationary,
)
```

Here one unit of distance is **one base pair**. The same SciPy function is
used for the bootstrap threshold.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put experimental CSV files under `data/` or set `DATA` to an absolute path. Do not
commit confidential datasets or large result folders to a public repository.

### Pooled WT, RIF1, or KU70

Edit the configuration block in `run_telomere_w1.sh` (or override variables):

```bash
MODE=lab GROUP=WT LABEL=WT_pooled bash run_telomere_w1.sh submit
MODE=lab GROUP=KU70 LABEL=KU70_pooled bash run_telomere_w1.sh submit
MODE=lab GROUP=RIF1 LABEL=RIF1_pooled bash run_telomere_w1.sh submit
```

The lab CSV must contain `strain,len`, with strain names such as `WT_rep1` and
`KU70_rep3`. Set `GROUP=WT_rep1` to analyze an individual replicate.

### One natural strain chromosome end

```bash
MODE=natural DATA=data/Telo_100_strains.csv STRAIN=BGP_1a \
EXTREMITY=chrI_Left LABEL=BGP_1a_chrI_Left bash run_telomere_w1.sh submit
```

The natural CSV must contain `strain`, a length column (`length` or `len`), and
an end identifier (`extremity` or `chrom` + `side`). Chromosome ends are kept
separate; `side` must say `Left` or `Right`.

### Synthetic data

```bash
MODE=synthetic LABEL=synthetic_B SYNTHETIC_N=3000 \
TRUE_A=4 TRUE_BETA=0.045 TRUE_LS=90 TRUE_P=0.026 \
bash run_telomere_w1.sh submit
```

Use different `LABEL` values for different data selections or settings. The command
submits one Slurm array task per fixed `a` and a dependent summary job. `PARTITION`,
`WALLTIME`, `PARALLEL`, `A_MIN`, `A_MAX`, `RESTARTS`, `MAXITER`, `POPSIZE`,
`LS_ALPHA`, `MIN_N`, `BOOTSTRAP`, `VENV`, and other settings can be changed
in the shell configuration or passed as environment variables.

## Outputs

After the summary job completes, the selected output folder has three tables:

- `tables/fit_summary.csv`: all CMA-ES starts, their initial/final
  parameters, final W1, acceptance flag and rank among accepted fits.
- `tables/profiles.csv`: best W1 and fitted parameters for each fixed `a`, plus
  the number of accepted starts.
- `tables/distributions.csv`: empirical histogram and the globally best
  stationary distribution (also the true law for synthetic tests).

The threshold is `mean(T_boot) + 3 * SD(T_boot)`, using `BOOTSTRAP` multinomial
samples of the same size as the observations. For synthetic data, the known
stationary distribution is the bootstrap reference. For real data, the empirical
distribution is the reference. It is a **descriptive sampling threshold**, not a
refitted-bootstrap p-value or a formal goodness-of-fit test.

The input is filtered by an upper-IQR rule (`IQR_K=2`) unless you select
`OUTLIER_FILTER=none` or `OUTLIER_FILTER=two_sided_iqr`. After filtering,
the default minimum sample size is `N > 1000`. Observations beyond `LMAX` **are not capped** (which would create an
artificial spike). By default, the program raises an error and asks you to
increase `LMAX`. Set `OVERFLOW_DATA=drop` if you explicitly prefer to exclude
those observations, and note that this changes the analyzed sample. By default, the temporary per-`a` files are deleted
only after all three final tables have been written. Set `KEEP_PARTS=1` to keep them.

### Model boundary and geometric normalization

The shortening baseline is `b = max(0, L-a)`. Recruitment occurs with probability
`f(L)`. Given recruitment, the increment is geometric with parameter `p`,
**conditioned on fitting within the finite state space**:

```text
K = Lmax - b
P(G = k | G <= K) = p*(1-p)^(k-1) / (1-(1-p)^K),  k=1,...,K
```

This makes every transition row sum to one **without** putting the unobserved
geometric tail into state `Lmax`. This is a different boundary convention from
the former *capped* model; fitted parameters should not be directly compared
across these two conventions. Increasing `LMAX` until fitted mass near the
boundary is negligible is recommended.

### Verify locally

```bash
python -m unittest discover -s tests -v
```

