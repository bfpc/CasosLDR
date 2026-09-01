# CasosLDR

Repositório de casos de estudo para LDR em Otimização Multiestágio

## Running the case studies

`runall.jl` loads every case study and offers two entry points:

```julia
include("runall.jl")

test_default()                      # every case study, with its default solver
test_solvers(HydroToy2)             # one case study, with every solver of `all_solvers`
```

`test_default(; kwargs...)` runs `main()` of each module of `problems`
(`Newsvendor`, `HydroToy2`, `HydroToy3`, `HydroQuad`, `HydroQuadPw`), each with
the solver it declares as `default_solver`.

`test_solvers(problem::Module; kwargs...)` takes one such module -- a required
positional argument -- and runs it once per entry of `all_solvers` (`Ipopt`,
`HiGHS (ASM)`, `HiGHS (IPM)`, all capped at 60 s).

### Keyword arguments

Both functions forward their keyword arguments unchanged, through each case's `main` and `compare_methods`, down to `Compare.run_comparison`.
The arguments that survive that trip are:

| Argument | Default | Meaning |
| --- | --- | --- |
| `n_scenarios` | `10_000` (`1_000` in `HydroToy3`) | Out-of-sample sample size: how many inflows each solution's recourse is averaged over. Ignored by `Newsvendor`, whose policy is evaluated analytically. |
| `replications` | `3` | How many times each SAA (and, in `HydroToy3`, each SDDP) sample size is re-solved, on fresh samples. Sample sizes are always `4, 10, 30, 100`. |
| `summarize` | `:append` | The `mean ± std` line given to each group of replications: `:append` adds it under the rows, `:only` replaces them by it, `:none` prints the rows alone. |
| `verbose` | `true` | Print `Solving ... evaluating policy ... done.` progress while the table is being built. |

Anything else raises a `MethodError` -- `run_comparison`, at the end of the
chain, has no catch-all `kwargs...`.

**Examples**:
- A quick smoke test of the whole package, cutting both the out-of-sample sample size and the replications, and keeping only the summary lines:

```julia
test_default(n_scenarios = 100, replications = 2, summarize = :only)
```

- One case study across all solvers, with short evaluation:

```julia
test_solvers(HydroQuad; n_scenarios = 100, replications = 1)
```

### Reading the table

Each row is one solution method: the first-stage decisions it commits to, its
out-of-sample assessment (`Perf`, `Std`, and the average stage-2 quantities),
and two timings -- `Opt time`, spent inside the optimizer, and `Time (ms)`, the
whole call, model building included.

The first rows of every table are `(warmup)` runs, which exist only to pay for
first-call compilation so that the timings below them measure solving rather
than compiling. They are not evaluated out of sample, so their evaluation
columns read `NaN`.
