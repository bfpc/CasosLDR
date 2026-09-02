# Decomposing the basis-function LDR model

`hydro_scenario.jl` builds the toy hydrothermal problem as a *linear decision
rule* over a basis of functions of the inflows: every decision of stage $t$ is

$$
d_t(\xi) = \sum_{j=1}^{K} x^{d}_{t,j}\, \varphi_j(\xi),
\qquad \varphi_1 \equiv 1,
$$

with one coefficient $x^d_{t,j}$ per decision, stage and basis function, and
non-anticipativity fixing $x^d_{t,j} = 0$ whenever $\varphi_j$ depends on inflows
revealed after stage $t$. `basis_function_ldr` writes this as a monolithic LP: one
copy of the dispatch problem per sampled scenario, every copy tied to the same
coefficients. With $\Phi_{kj} = \varphi_j(\xi^k)$ the basis matrix of the sample
(`basis_values`) and $x$ the coefficients flattened (`ldr_coefficients`), the
model is

$$
\min_x \; \sum_k p_k\, Q_k(x),
\qquad
Q_k(x) = \begin{cases}
\text{cost of scenario } k \text{ under the rule } x, \\
+\infty & \text{if the rule is infeasible in scenario } k.
\end{cases}
$$

The coefficients are the only variables the scenarios share, so they are the
natural first stage. Two decompositions of that split are implemented, and they
are worth reading together because the second one exists as an answer to what
measuring the first one showed.

Both reproduce the monolithic optimum exactly — verified by
`compare_basis_function_ldr`, which runs all of them on the same scenarios.

## What the split implies

Fixing $x$ fixes *every* scenario variable: volume, hydro, spill, thermal and
future cost are all values of the rule. The scenario subproblems therefore have
**no recourse**. Two consequences drive everything below.

* A scenario's cost is an **affine function of the rule**, so pricing it is easy
  and its optimality cuts are exact almost immediately.
* A scenario's *feasibility* is the hard part: the rule has to satisfy that
  scenario's bounds, its water balance, its demand and its future-cost rows.
  Those include equalities, so the set of implementable rules is a polyhedron
  living in a proper affine subspace of the coefficient space.

The work of decomposing this model is describing that polyhedron.

## 1. Benders decomposition (`basis_function_ldr_benders`)

Master: the coefficients $x$, plus one epigraph variable $\alpha_k \ge 0$ per
scenario, minimizing $\sum_k p_k \alpha_k$.

Subproblem (`basis_ldr_subproblem`), one per scenario, built once and re-solved:
scenario $k$'s dispatch constraints, a **fixed copy** of $x$, and the coupling
`decision == Σⱼ xⱼ Φₖⱼ` relaxed by non-negative slacks $s^\pm$. It is solved in
two passes:

| pass | slacks | objective | cut |
| --- | --- | --- | --- |
| feasibility | free | $\min \sum s^\pm$ | $0 \ge v + g'(x - \bar x)$ |
| optimality | pinned to 0 | $\min \text{cost}$ | $\alpha_k \ge q + g'(x - \bar x)$ |

The feasibility pass measures the $L_1$ distance from the policy the rule
prescribes to scenario $k$'s feasible set, so it is always feasible and $v > 0$
proves the rule is not implementable there. Both subgradients $g$ are read off
`reduced_cost` of the fixed copy of $x$ (`subproblem_cut`), which lets JuMP do
the multiplication by $\Phi_{k\cdot}$ and keeps the cuts sparse by dropping the
zero components.

Validity: $v$ and $q$ are values of LPs parameterized by $x$ in their right-hand
sides, so their optimal duals give affine minorants of the convex value
functions — valid for every $x$, and tight at $\bar x$.

**Convergence.** The master objective is a lower bound; the cost of a rule
feasible in every scenario is an upper bound; the loop stops when they meet or
when no cut is violated. Finite, because both value functions are piecewise
linear with finitely many pieces.

**Measured behaviour** (toy problem, `T=3`):

| case | iterations | feasibility cuts | optimality cuts |
| --- | --- | --- | --- |
| `bp2`, 8 scenarios | 113 | 773 | 20 |
| `bp2`, 30 scenarios | 69 | 1736 | 48 |
| `pw3`, 100 scenarios | 54 | 4607 | 156 |

The ratio is the whole story: **thousands of feasibility cuts against a hundred
optimality cuts**. Optimality cuts are cheap because the cost is affine;
feasibility cuts are many because they have to rebuild the polyhedron of
implementable rules one supporting hyperplane at a time, and a hyperplane is a
poor way to describe an equality. The master's objective is also degenerate for
the first many iterations — with no optimality cut yet, any rule with
$\alpha = 0$ is optimal — so early iterates wander.

Benders was slower than simply solving the monolith at every size tested
(4.6 s against 65 ms at 100 scenarios).

Two numerical guards are worth knowing about. The optimality pass pins the
slacks to zero, which a solver can call infeasible when the feasibility pass
accepted the rule only to within tolerance; the pass is then retried allowing
the violation already accepted, and if that still fails the scenario is cut off
as infeasible using the feasibility cut. And `INFEASIBLE` from the master is a
proof that no rule is implementable, which the loop reports rather than retries.

## 2. Scenario generation (`basis_function_ldr_scenario_generation`)

Rather than approximate a scenario's feasible region with cuts, add the
scenario's own rows, which describe it exactly.
Since the subproblems have no recourse, Benders decomposition needs to build several cuts per scenario to describe the feasible region.

Two features of the model make this work particularly well.

**The objective does not depend on the working set.** Substituting the rule into
the monolith's objective,

$$
\sum_{i,t,k} p_k c_i\, \text{termos}_{i,t,k} + \sum_k p_k \theta_k
 = \sum_{i,t,j} c_i\, x^{\text{termos}}_{i,t,j} \bar\varphi_j
 + \sum_j x^{\theta}_j \bar\varphi_j,
\qquad
\bar\varphi_j = \sum_k p_k \Phi_{kj},
$$

so it depends on the sample **only through the mean basis vector**
$\bar\varphi$ — a $K$-vector of the whole sample, computable up front
(`mean_basis`, `set_ldr_objective!`). The master carries the exact full-sample
objective from the first round: adding scenarios never changes it, and
generation is about feasibility alone. At termination the master objective *is*
the expected cost of the rule over the whole sample, not an estimate of it.

**A scenario's rows are linear in the coefficients.** Substituting the rule
eliminates the scenario's variables, so a scenario contributes rows and no
variables (`add_scenario_rows!`): bounds on each of the $3 + $ `termo_count`
decisions per stage, two balances per stage, and $1 + $ `cuts_count`
future-cost rows — about 45 rows for `T=3`.

### The aggregate rows

A master holding a subset of the scenarios' rows can be unbounded: nothing yet
pins the coefficients the objective pays for. The fix is to add the **sample
average of the scenarios' rows** up front. It is a convex combination of rows of
the full model, hence implied by it, and — because every row is affine in
$(\Phi_{k\cdot}, a_k)$ — it is exactly `add_scenario_rows!` evaluated at the mean
basis vector and the mean inflow. It bounds the objective immediately, since it
makes expected thermal generation and expected future cost non-negative.

This one line is worth an order of magnitude. Before it, the loop had to widen
the working set repeatedly against an unbounded master, overshooting to hundreds
or thousands of scenarios; the `pw3` row at 10 000 scenarios in
`compare_methods` took **42 s**, of which 37 s was building rows. After it, no
master is ever unbounded and the same row takes **0.45 s**.

### The loop

1. Solve the master; its objective is a lower bound on the full-sample optimum,
   monotone across rounds.
2. Separate over **all** scenarios at once: `rule_values` computes each
   decision's value per scenario as a matrix product $\Phi X'$, and
   `scenario_violation` returns the largest violation per scenario. No solver is
   involved — this is why the sample can be enormous.
3. Stop if no scenario is violated by more than `tol`: the rule is feasible for
   the whole sample, so it is optimal for it. Otherwise add the `max_add` most
   violated scenarios and repeat.

Finite, since every round adds at least one scenario.

### Equality modes

The `equalities` keyword chooses how the water balance and the demand are
handled.

* `:per_scenario` (default) generates them with each scenario, reproducing
  `basis_function_ldr` exactly.
* `:coefficientwise` (`add_coefficientwise_equalities!`) imposes them once per
  basis coefficient, so they hold for *every* realization and are never
  generated. This needs each stage's inflow to be a combination of the basis
  functions **that stage knows**, $a_t(\xi) = \sum_j r_{t,j}\varphi_j(\xi)$ with
  $r_{t,j} = 0$ for later-stage functions: non-anticipativity zeroes the
  coefficients such a row would talk about, so a representation using them is
  infeasible rather than merely wrong. The representation is fitted by least
  squares over the admissible columns and its residual checked. It is exact for
  `bp2_functions` and `pw3_functions`, whose pieces sum back to the inflow of
  their stage, $\text{pwl}(a,0,40) + \text{pwl}(a,40,80) + \text{pwl}(a,80,120) = a$
  on $[0,120]$.

The two modes are the same model whenever $\Phi$ has full column rank, and
`:coefficientwise` is stronger when it does not — `pw3` at any sample size is
rank-deficient by one, because
$(\varphi_2+\varphi_3+\varphi_4) + (\varphi_5+\varphi_6+\varphi_7) = \varphi_8 + \dots + \varphi_{13}$.
In that case the master is no longer a relaxation, so no bound is reported
(`Bound = -Inf`); the objective still came out equal to the monolith's in every
case tested.

### Measured behaviour

Toy problem, `T=3`, generation against the monolith on identical scenarios:

| basis | scenarios | monolith | generation | rounds | active scenarios |
| --- | --- | --- | --- | --- | --- |
| `bp2` (K=7) | 1 000 | 1.28 s | 0.63 s | 5 | 91 |
| `bp2` | 10 000 | 82.4 s | 0.17 s | 6 | 130 |
| `bp2` | 100 000 | — | 1.73 s | 10 | 239 |
| `pw3` (K=13) | 1 000 | 2.03 s | 0.12 s | 6 | 128 |
| `pw3` | 10 000 | 144.5 s | 0.72 s | 8 | 184 |
| `pw3` | 100 000 | — | 1.52 s | 10 | 251 |

Objectives agree to the last printed digit wherever the monolith could be built.
The point of the table is the last column: **a few hundred scenarios carry the
solution out of a hundred thousand**, and the cost of the method follows the
active set, not the sample. Most of the remaining time is sampling, building
$\Phi$ and separating; the master solves take 20-100 ms.

`max_add` trades rounds against master size, and the master is what costs:

| `max_add` | rounds | active | total |
| --- | --- | --- | --- |
| 1 | 16 | 41 | 0.12 s |
| 5 | 11 | 75 | 0.32 s |
| 25 | 8 | 184 | 0.26 s |
| 100 | 5 | 334 | 0.59 s |
| 500 | 3 | 1026 | 4.19 s |

(`pw3`, 10 000 scenarios.) The default is 10. Adding many scenarios per round
buys few rounds and pays for a large master.

## Which to use

Scenario generation, for this model. Benders is the textbook split of the same
first stage, and it is instructive precisely because it is the wrong shape here:
a decomposition whose subproblems have no recourse spends everything on
feasibility cuts, and rows do that job better than cuts. Benders would come into
its own if the scenario subproblems gained genuine recourse — decisions that the
rule does not determine.

## Reference

| Function | |
| --- | --- |
| `basis_function_ldr` | the monolithic model, one scenario block per sample point |
| `basis_function_ldr_benders` | decomposition 1 |
| `basis_ldr_subproblem`, `feasibility_mode!`, `optimality_mode!`, `subproblem_cut` | its subproblems and cuts |
| `basis_function_ldr_scenario_generation` | decomposition 2 |
| `mean_basis`, `set_ldr_objective!`, `add_scenario_rows!`, `add_coefficientwise_equalities!`, `rule_values`, `scenario_violation` | its master rows and separation |
| `basis_ldr_generation_solution` | a `run_comparison` row, used by `compare_methods` |
| `compare_basis_function_ldr` | all of them on the same scenarios |
| `sample_scenarios`, `inflow_matrix`, `basis_values`, `add_ldr_coefficients!`, `ldr_coefficients`, `reshape_coefficients`, `fix_nonanticipativity!` | shared by every formulation |

Both decompositions return the `(; Obj, Bound, H₁, V₁, T₁, dt)` fields the other
solution methods of `hydro3.jl` return, followed by diagnostics; only the six
are safe to hand to `run_comparison`, since every other field becomes a column
of its table.

```julia
julia> problem = HydroToy3.build_toy_problem();

julia> HydroToy3.compare_basis_function_ldr(problem, HydroToy3.bp2_functions,
           HydroToy3.bp2_stages; T = 3, n_scen = 30)

julia> HydroToy3.basis_function_ldr_scenario_generation(problem,
           HydroToy3.pw3_functions, HydroToy3.pw3_stages;
           T = 3, n_scen = 100_000, verbose = true)
```
