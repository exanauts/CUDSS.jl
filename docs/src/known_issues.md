# Known issues

This page lists known bugs and limitations that affect CUDSS.jl in practice.

Unless stated otherwise, the issues below originate in the underlying NVIDIA **cuDSS** library, not in CUDSS.jl: CUDSS.jl forwards the documented arguments to `libcudss` unchanged, and each issue has been reproduced by a standalone C program that calls `libcudss` directly with the same API sequence. Exact thresholds may depend on the GPU and the cuDSS version — see the [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/release_notes.html).

## Matching changes the reported diagonal

Since cuDSS 1.0, matching with scaling is enabled by default for general, symmetric and Hermitian matrices. In that case `cudss_get(solver, "diag")` returns the diagonal of the scaled matrix rather than of the input matrix. Set `cudss_set(solver, "matching_alg", "none")` before the analysis to obtain the unscaled diagonal.

## Rare nondeterministic `NaN` in large uniform-batch solves

With cuDSS **0.8.0**, for large uniform batches (batch sizes in the hundreds) the solve occasionally produced `NaN` for a small number of systems even though the input was finite, most often under device-memory pressure with concurrent GPU work on another stream. The failure was **nondeterministic**, and every affected system solved correctly on its own (`nbatch = 1`).

With cuDSS **1.0.0**, neither the dumped application case nor the memory-pressure reproducer produced a `NaN` on an NVIDIA RTX 4080. Because the failure is nondeterministic, this is not yet a confirmation that the issue is fixed.

!!! note
    If you observe this with cuDSS 1.0, enabling iterative refinement (`cudss_set(solver, "ir_n_steps", k)` with `k ≥ 1`) reduced the frequency of the issue with cuDSS 0.8.

## Resolved in cuDSS 1.0

The following issues were observed with `CUDSS_jll` v0.8.0 on an NVIDIA RTX 4080 and no longer reproduce with cuDSS 1.0.0:

- **Multiple right-hand sides in a uniform batch returned incorrect results.** With `nrhs > 1` and `ubatch_size ≥ 16`, only the first four right-hand-side columns of each system were correct. With cuDSS 1.0 all columns are correct.
- **`deterministic_mode` crashed with uniform batches.** Enabling `deterministic_mode` with `ubatch_size > 1` caused an illegal memory access during the solve. cuDSS 1.0 rejects this configuration during the analysis with `CUDSS_STATUS_NOT_SUPPORTED`; deterministic mode remains unsupported for uniform batches.
- **Inertia was misreported with matching.** cuDSS 0.8 returned `(0, 0)` as inertia when matching was enabled. cuDSS 1.0 returns the correct inertia with every matching algorithm.
