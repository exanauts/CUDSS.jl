# Known issues

This page lists known bugs and limitations that affect CUDSS.jl in practice.

Unless stated otherwise, the issues below originate in the underlying NVIDIA **cuDSS** library, not in CUDSS.jl: CUDSS.jl forwards the documented arguments to `libcudss` unchanged, and each issue has been reproduced by a standalone C program that calls `libcudss` directly with the same API sequence. They were observed with `CUDSS_jll` **v0.8.0** on an NVIDIA RTX 4080. Exact thresholds may depend on the GPU and the cuDSS version, and a given issue may be fixed in a later cuDSS release — see the [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/release_notes.html).

All three issues below are confined to the **uniform batch** code path (`ubatch_size > 1`).

## Multiple right-hand sides in a uniform batch return incorrect results

When a uniform batch is solved with more than one right-hand side per system (`nrhs > 1`) and the batch size is large enough, the **solve** phase returns incorrect values for all but the first few right-hand-side columns. On the tested setup the results are correct for `nbatch ≤ 15` and become wrong for `nbatch ≥ 16` (only the first four columns of each system remain correct). The batch factorization itself is correct; only the multi-RHS solve is affected.

- A single right-hand side (`nrhs = 1`) is **always** correct, at any batch size.
- This affects the `cudss("solve", …)` interface as well as `ldiv!` and `\` when the right-hand side is a 3-D `CuArray` of shape `(n, nrhs, nbatch)`.

!!! warning
    Until this is fixed in cuDSS, either solve the right-hand sides one column at a time — each as a separate single-RHS uniform-batch solve reusing the same factorization — or use the [non-uniform batch](nonuniform_batch.md) interface (`CudssBatchedSolver`), which solves multiple right-hand sides correctly.

## `deterministic_mode` is not compatible with uniform batches

Enabling `cudss_set(solver, "deterministic_mode", 1)` on a uniform batch (`ubatch_size > 1`) triggers an illegal memory access inside cuDSS during the **solve** phase, crashing the process, instead of returning a clean status. cuDSS documents deterministic mode as supported only for a single GPU, a single right-hand side, without iterative refinement, and with the hybrid memory/execute modes disabled; a uniform batch falls outside this supported set.

!!! warning
    Do not enable `deterministic_mode` together with `ubatch_size > 1`. Note that if iterative refinement is also enabled (`ir_n_steps > 0`), cuDSS instead rejects the configuration cleanly with `CUDSS_STATUS_NOT_SUPPORTED` during the analysis phase, so the crash is only observed when iterative refinement is off.

## Rare nondeterministic `NaN` in large uniform-batch solves

For large uniform batches (batch sizes in the hundreds) of ill-conditioned systems, the solve occasionally produces `NaN` for a small number of systems even though the input is finite. The failure is **nondeterministic** — the same input may succeed on one run and fail on another — and every affected system solves correctly on its own (`nbatch = 1`). This behavior is consistent with a race or an uninitialized-memory read inside the cuDSS batch factorization/solve, and it is much more likely to surface inside a long-running application than in an isolated reproducer.

!!! note
    Enabling iterative refinement (`cudss_set(solver, "ir_n_steps", k)` with `k ≥ 1`) reduces the frequency of this issue but does not eliminate it.
