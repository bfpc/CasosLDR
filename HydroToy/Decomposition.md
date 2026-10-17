# Decomposing the basis-function LDR model

`hydro_scenario.jl` builds the toy hydrothermal problem as a *linear decision
rule* over a basis of functions of the inflows: every decision of stage $t$ is

$$
x^d_t(\xi) = \sum_{j=1}^{K} X^{d}_{t,j}\, \varphi_j(\xi),
\qquad \varphi_1 \equiv 1,
$$

with one coefficient $X^{d}_{t,j}$ per decision, stage and basis function, and non-anticipativity fixing $X^{d}_{t,j} = 0$ whenever $\varphi_j$ depends on inflows revealed after stage $t$.
`primal_ldr` writes this as a monolithic LP: one copy of the dispatch problem per sampled scenario, every copy tied to the same coefficients.
With $\Phi_{kj} = \varphi_j(\xi^k)$ the basis matrix of the sample (`basis_values`) and $x$ the coefficients flattened (`ldr_coefficients`), the model is

$$
\min_x \; \sum_k p_k\, Q_k(X),
\qquad
Q_k(X) = \begin{cases}
\text{cost of scenario $k$ under the rule $X$}, \\
\text{$+\infty$ if the rule is infeasible in scenario $k$}.
\end{cases}
$$

The coefficients are the only variables the scenarios share, so they are the
natural first stage. Two decompositions of that split are implemented, and they
are worth reading together because the second one improves the first one.

Both reproduce the monolithic optimum exactly — verified by
`compare_primal_ldr`, which runs all of them on the same scenarios.

## What the split implies

Fixing $X$ fixes *every* scenario variable:
volume, hydro, spill, thermal and future cost are all values of the rule,
so the scenario subproblems have **no recourse**.
This has two consequences:

* A scenario's cost is an **affine function of the rule**, so pricing it is easy
  and its optimality cuts are exact almost immediately.
* A scenario's *feasibility* is the hard part: the rule has to satisfy that
  scenario's bounds, its water balance, its demand and its future-cost rows.
  Those include equalities, so the set of implementable rules is a polyhedron
  living in a proper affine subspace of the coefficient space.

The main difficulty of decomposing this model is describing that polyhedron.

## 1. Benders decomposition (`primal_ldr_benders`)

Master: the coefficients $X$, plus one epigraph variable $\alpha_k \ge 0$ per
scenario, minimizing $\sum_k p_k \alpha_k$.

Subproblem (`primal_ldr_subproblem`), one per scenario, built once and re-solved:
scenario $k$'s dispatch constraints, a **fixed copy** of $X$, and the coupling
`decision == Σⱼ xⱼ Φₖⱼ` relaxed by non-negative slacks $s^\pm$. It is solved in
two passes:

| pass | slacks | objective | cut |
| --- | --- | --- | --- |
| feasibility | free | $\min \sum s^\pm$ | $0 \ge v + g'(X - \bar X)$ |
| optimality | pinned to 0 | $\min \text{cost}$ | $\alpha_k \ge q + g'(X - \bar X)$ |

The feasibility pass measures the $L^1$ distance from the policy the rule
prescribes to scenario $k$'s feasible set, so it is always feasible and $v > 0$
proves the rule is not implementable there. Both subgradients $g$ are read off
`reduced_cost` of the fixed copy of $X$ (`subproblem_cut`), which lets JuMP do
the multiplication by $\Phi_{k\cdot}$ and keeps the cuts sparse by dropping the
zero components.

Validity: $v$ and $q$ are values of LPs parameterized by $X$ in their right-hand
sides, so their optimal duals give affine minorants of the convex value
functions — valid for every $X$, and tight at $\bar X$.

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

## 2. Scenario generation (`primal_ldr_scenario_generation`)

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
  `primal_ldr` exactly.
* `:coefficientwise` (`add_coefficientwise_equalities!`) imposes them once per
  basis coefficient, so they hold for *every* realization and are never
  generated. This needs each stage's inflow to be a combination of the basis
  functions **that stage knows**, $a_t(\xi) = \sum_j r_{t,j}\varphi_j(\xi)$ with
  $r_{t,j} = 0$ for later-stage functions: non-anticipativity zeroes the
  coefficients such a row would talk about, so a representation using them is
  infeasible rather than merely wrong. The representation is fitted by least
  squares over the admissible columns and its residual checked. It is exact for
  `BP2` and `PW3`, whose pieces sum back to the inflow of
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

# Decomposing the *dual* LDR

`Dual_scenarios.tex` restricts the Lagrange multiplier of the canonical form to
a rule, $z(\xi) = Z L(\xi)$, instead of the decisions. Each row of
$A x(\xi) + s(\xi) = b(\xi)$ then only has to hold on average against every
basis function, which gives a **lower** bound on the sampled problem —
dual to the upper bound above. In the sample it is one LP with every scenario's
variables and only $m \ell$ coupling rows:

$$
\min_{x,\,s \ge 0} \; \frac{1}{K}\sum_k c^\top x(\xi_k)
\quad\text{s.t.}\quad
\sum_k \big(A x(\xi_k) + s(\xi_k) - b(\xi_k)\big) L(\xi_k)^\top = 0 .
$$

## 3. What does *not* work: cutting the moments

The obvious split puts the moment matrices $M_x, M_s$ in the master (the only
thing the coupling rows and the objective see) and asks a subproblem whether
they are achievable by actual scenario variables. It does not converge, and not
for want of tuning: the achievable moments of a single family of decisions,
say $0 \le h(\xi_k) \le \bar H$, are

$$
\Big\{\, \tfrac1K \textstyle\sum_k h_k \Phi[k,:] \;:\; h_k \in [0,\bar H] \,\Big\} \subset \mathbb{R}^{\ell},
$$

a Minkowski sum of $K$ segments — a **zonotope** with $K$ generators, whose
facets number $O\!\big(\binom{K-1}{\ell-1}\big)$, about $10^9$ for $K=100$,
$\ell=7$. `add_coupling_moment_bounds!` is exactly its bounding box, and one
aggregated L1 feasibility cut per iteration never gets appreciably past that
box: the bound sits at the expected-value solution however long the loop runs.
`dual_ldr_benders` is kept, in both `slacks` modes, because that is worth
knowing, not because either one works.

A zonotope has no usable facet description — but its *support function* is
closed form, which says to work on the dual side instead.

## 4. Lagrangian cutting planes (`dual_ldr_lagrangian`)

Dualizing the coupling rows with $Z$ — where $Z$ came from in the first place —
separates the sample completely:

$$
\text{value} = \max_Z \; \frac1K \sum_k \varphi_k\big(Z L(\xi_k)\big),
\qquad
\varphi_k(z) = \min_{y_k \in Y_k}\Big[c^\top x(\xi_k) + z^\top\big(A y_k - b(\xi_k)\big)\Big],
$$

with $Y_k$ the box of `dual_ldr(s; slacks = :coupling)`. Because $Y_k$ *is* a
box, `scenario_lagrangian` evaluates $\varphi_k$ and its minimizer by
arithmetic: every variable goes to the bound its reduced cost asks for, and a
negative reduced cost on one of the variables unbounded above (`spill`, `θ`,
`cuts_slack`) means $\varphi_k = -\infty$, i.e. $z$ is dual-infeasible for that
scenario. Separation over the whole sample is one matrix product; no solver is
called per scenario.

Two things make this the right shape.

* The supergradient is $\frac1K\sum_k (A y_k^\star - b(\xi_k)) L(\xi_k)^\top$ —
  the moment residual. What decomposition 3 carried as *variables* is here a
  gradient, and the moment matrices never need representing.
* Every $Z$ is a feasible dual rule, so the Lagrangian is a **valid lower bound
  at every iterate**, not only at convergence. The master's value is the upper
  bound, and the two meet.

The master is a Kelley model of that concave maximum: one epigraph variable per
scenario (`cut_mode = :multi`), the dual-feasibility rows (`ray_mode`), and an
L∞ trust region around the incumbent, scaled per basis function, which is what
makes the early iterations — where the cut model is still unbounded above —
mean anything. Two knobs matter at scale: `max_cuts`, because the epigraph of
an outer model is above the Lagrangian nearly everywhere so "add the violated
cuts" caps nothing on its own; and `ray_mode = :lazy`, because stating
$K(T + \text{cuts} + 1)$ dual-feasibility rows up front is itself the expensive
part past a few thousand scenarios.

## 5. Scenario column generation (`dual_ldr_column_generation`)

The primal face of the same decomposition, and the mirror of decomposition 2:
where that one starts from a few scenarios' *rows*, this starts from a few
scenarios' *columns*. Requiring a scenario to satisfy $A y_k = b(\xi_k)$
pointwise makes its contribution to every coupling row vanish, so for a working
set $S$,

$$
\min \; \frac1K\sum_{k \in S} c^\top x(\xi_k) + \frac1K \sum_{k \notin S} v_k
\quad\text{s.t.}\quad
\sum_{k \in S}\big(A y_k - b(\xi_k)\big)L(\xi_k)^\top = 0,
$$

with $v_k$ the scenario's own deterministic optimum (`wait_and_see_values`), is
a **restriction**: complete it with pointwise-feasible decisions outside $S$ and
it is feasible for the full model at the same cost. Its value therefore comes
*down* to the answer, and with $|S| = 1$ it starts exactly at the wait-and-see
value. Pricing is the Lagrangian again: scenario $k \notin S$ has gap
$v_k - \varphi_k(\bar Z L(\xi_k)) \ge 0$ for $\bar Z$ the coupling rows' duals
(infinite when $\bar Z$ is dual-infeasible for it), and all gaps vanishing is
the optimality certificate. The sign convention of those duals is settled by
strong duality rather than guessed — see `coupling_multipliers`.

Unlike the moment masters this one is always feasible and bounded, the constant
alone bounding it below. But its iterates are *upper* bounds on a quantity
whose point is to be a lower bound, so every round also reports the Lagrangian
of its own multipliers, which is valid throughout.

## Measured behaviour

Toy problem, `T=3`, `bp2` basis, 100 scenarios, seed 7. The monolithic dual LDR
is 93 163.31, and both decompositions reproduce it exactly:

| method | value | time | work |
| --- | --- | --- | --- |
| monolithic | 93 163.31 | 0.03 s | — |
| lagrangian, `:multi` + `:upfront` | 93 163.31 | 1.06 s | 16 iterations, 1 322 cuts, 900 rays |
| lagrangian, `:multi` + `:lazy` | 93 163.31 | 0.44 s | 18 iterations, 1 193 cuts, 524 rays |
| lagrangian, `:single` + `:upfront` | 90 911.23 | 0.59 s | 171 iterations, **not converged** |
| lagrangian, `:single` + `:lazy` | 91 934.46 | 0.73 s | 286 iterations, **not converged** |
| column generation, `max_add=1` | 93 163.31 | 1.90 s | 90 rounds, 90 of 100 scenarios |
| column generation, `max_add=5` | 93 163.31 | 0.34 s | 20 rounds, 93 of 100 scenarios |
| column generation, `max_add=25` | 93 163.31 | 0.07 s | 5 rounds, 100 of 100 scenarios |

Every value that converged is the monolithic optimum to the last digit, from
both sides: the Lagrangian's two bounds meet, and the column generation
descends to it from the wait-and-see value (95 607.78 → 95 584 → 95 396 → … →
93 163.31 over its 20 rounds, the last with no scenario mispriced).

Three readings. The aggregated cut (`:single`) is far too weak — one cut per
iteration for a sum of $K$ piecewise-linear terms — so `:multi` is the default,
and `:single`'s value, while a perfectly valid lower bound, is simply the one it
got to.

The master LP is the *entire* cost of the Lagrangian method: with the trace on,
scenario evaluation measures 0.000 s an iteration and cut building 0.005 s,
against 0.05–0.2 s for the master. Which is also where the one sharp edge is:
a cutting-plane master is the worst case for a dual simplex, and HiGHS's default
choice stalls outright on the later, fatter ones — 30 s for a 2 000-row LP the
primal simplex does in 0.05 s, with its interior-point and PDLP options not
finishing at all. Hence `master_attributes` asking for the primal simplex,
`prune_cuts!` dropping the rows priced at zero (644 of 900 at one stroke, on the
`pw3` basis), and `master_time_limit` to bound what a stall can cost. None of
that is needed for `bp2`; all of it is for `pw3`, and even then some of `pw3`'s
masters still hit the limit. The consequence is worth being precise about,
because it is asymmetric: `pw3`'s **bound is exact** — the Lagrangian reaches
95 142.83, the monolithic optimum, at iteration 15 — but its **certificate is
not**, since `ObjBound` only updates on a master solved to optimality, so the
run ends on the stall limit with `converged = false` and `ObjBound = Inf`. A
valid bound that cannot prove it is still a valid bound; take the certificate
from `dual_ldr_column_generation`, whose master is a small,
well-behaved LP, and pair the two.

The working set of the column generation, meanwhile, grows to **nearly the whole
sample** — 93 of 100 here — which is the opposite of what happens in the primal
(a few hundred scenarios out of a hundred thousand). That is structural, not a
tuning failure:
the dual LDR value is an average of every scenario's $\varphi_k$, and leaving
scenario $k$ out replaces $\varphi_k(\bar Z L(\xi_k))$ by the strictly larger
$v_k$ unless that scenario's pointwise problem happens to be optimal at
$\bar Z$. Few are — though a richer basis prices more of them exactly, which is
why `pw3` converges on 596 of 1 000 scenarios where `bp2` needs 997.

## Scaling: what actually wins

`bp2`, `T=3`, seed 7, against the monolith on the same samples:

| scenarios | monolith (columns) | lagrangian | column generation |
| --- | --- | --- | --- |
| 100 | **0.37 s** (2 400) | 0.69 s, exact | 1.34 s, exact, 94 active |
| 1 000 | **0.46 s** (24 000) | 18.8 s, exact | 22.2 s, 997 active, not conv. |
| 10 000 | 78.9 s (240 000) | 267 s, 66 745 (not conv.) | **23.0 s**, bound 91 473.57 |
| 100 000 | >18 min, abandoned (2.4 M) | — | **31.7 s**, bound 91 532.72 |

Read this honestly, because it is not the result one expects. Up to a thousand
scenarios **the monolith wins outright** and neither decomposition has any
reason to exist: its rows are only the $m\ell$ coupling rows however many
scenarios the columns come from, and HiGHS eats it. The dual LDR simply is not
the kind of model that needs decomposing at those sizes.

The crossover is around ten thousand, where the monolith's quarter-million
columns cost 79 s. There the useful object is not either method's *certificate*
but the column generation's **by-product**: the Lagrangian of its own
multipliers, `ObjBound`, which is valid at every round. Truncated at 100 rounds
it returns 91 473.57 in 23 s against the monolith's exact 91 535.30 in 79 s —
within 0.07%, a third of the time — and at 100 000 scenarios, where the
monolith's 2.4 M columns were abandoned after 18 minutes without an answer, it
still returns 91 532.72 in 32 s. Stopping early is free here precisely because
that bound needs no convergence to be valid.

The `:multi` cutting plane is the one to avoid at scale: `max_cuts` caps it at
1 000 cuts a round, so with 10 000 epigraph variables most scenarios never get a
cut and the model of the Lagrangian is too poor to climb (66 745 after 267 s).
It is the better method at a hundred scenarios and the worse one at ten
thousand.

## The bracket, and the ceiling

`runner.jl` prints where every method falls (100 scenarios, `T=3`, seed 7):

| | `bp2` ($\ell=7$) | `pw3` ($\ell=13$) |
| --- | --- | --- |
| EV, sample (dual LDR, constant basis) | 91 817.90 | 91 817.90 |
| **dual LDR** | **93 163.31** | **95 142.83** |
| WS, sample (perfect foresight) | 95 607.78 | 95 607.78 |
| **primal LDR** | **104 382.87** | **97 613.60** |

Both decompositions reproduce the dual LDR column exactly, on both bases.

The wait-and-see value is the dual side's ceiling *by construction*: the
canonical form of `Dual_scenarios.tex` puts no non-anticipativity on $x(\xi)$,
so a basis rich enough to price every row scenario by scenario recovers the
perfect-foresight problem exactly, and none can do better. With the constant
basis alone the rows only have to hold on average and the value is the
expected-value problem. So

$$
\text{EV} \;\le\; \text{dual LDR}(L) \;\le\; \text{WS} \;\le\; \text{true optimum} \;\le\; \text{primal LDR}(L),
$$

and the dual LDR interpolates between the two classical bounds — `pw3` already
recovers three quarters of the EV–WS gap. Worth measuring before investing in
any of this: if WS − EV is small, so is everything the dual side can win.
Getting above WS needs measurability back among the dualized rows; see the last
section of `Dual_scenarios.tex`.

**This ceiling is a property of the formulation, not of dual decision rules.**
It holds because the scenario expansion here lets *every* variable depend on the
scenario. `LinearDecisionRules.jl` instead gives first-stage variables only
their constant coefficient, so what its dual bound brackets is the true
two-stage optimum, not WS — and it does overtake wait-and-see on problems with
a large EVPI. The criterion is `gap(L) < EVPI`, and the worked examples (a
two-point inflow, and a newsvendor where WS and EV coincide) are in
`Dual_scenarios.tex`, §"Non-anticipativity, and when the ceiling does not
apply". Lifting the ceiling *here* needs only the stage-1 variables tied across
scenarios — `volume[1,k] == volume[1,1]` and friends, imposed pointwise rather
than against the basis.

## Which to use, on the dual side

Up to a few thousand scenarios, **the monolith** (`dual_ldr(s; slacks = :coupling)`).
It is the fastest and it is exact, and the decompositions earn their keep only as
an independent check on it — which they pass, from both sides, on both bases.

Past that, **`dual_ldr_column_generation`, stopped early**, and
read `ObjBound` rather than `Obj`. Its restricted primal is a small
well-behaved LP, its pricing is the closed-form Lagrangian, and that Lagrangian
is a valid lower bound every round, so truncating it costs tightness and
nothing else. This is the opposite of the primal side, where generation
converges on a genuinely sparse active set; here the working set grows with the
sample and the right move is to stop caring about convergence and take the
bound.

`dual_ldr_lagrangian` is the most elegant of the three — no
moment matrices, no LP per scenario, separation in $O(K)$ arithmetic — and the
best at a hundred scenarios. It is also the one whose master is a degenerate
cutting-plane LP, and that is what limits it. Worth reading for the structure;
not what to run at scale.

`dual_ldr_benders`, in either `slacks` mode, is worth reading for the zonotope
argument and nothing else.

## Reference

| Function | |
| --- | --- |
| `primal_ldr` | the monolithic model, one scenario block per sample point |
| `primal_ldr_benders` | decomposition 1 |
| `primal_ldr_subproblem`, `feasibility_mode!`, `optimality_mode!`, `subproblem_cut` | its subproblems and cuts |
| `primal_ldr_scenario_generation` | decomposition 2 |
| `mean_basis`, `set_ldr_objective!`, `add_scenario_rows!`, `add_coefficientwise_equalities!`, `rule_values`, `scenario_violation` | its master rows and separation |
| `primal_ldr_generation_solution` | a `run_comparison` row, used by `compare_methods` |
| `compare_primal_ldr` | all of them on the same scenarios |
| `dual_ldr` | the monolithic dual LDR; `slacks = :all` for a slack per row, `:coupling` for only the genuine ones |
| `dual_ldr_benders` | decomposition 3, the moment masters that do not converge, in both `slacks` modes |
| `dual_ldr_benders_loop!` | the feasibility-cut loop shared by both modes |
| `add_ms_moments!`, `dual_ldr_ms_subproblem` | `:all`'s moments and achievability subproblem |
| `add_coupling_moments!`, `add_coupling_moment_bounds!`, `dual_ldr_coupling_subproblem`, `dual_ldr_moments` | `:coupling`'s moments and achievability subproblem |
| `dual_ldr_lagrangian` | decomposition 4 |
| `scenario_lagrangian`, `lagrangian`, `lagrangian_moment_residual`, `coupling_row_count`, `balance_row`, `demand_row`, `cut_row` | its closed-form evaluation and row layout |
| `prune_cuts!`, `restricting` (both local to it) | the cut budget, and when the master's value is a bound |
| `dual_ldr_column_generation` | decomposition 5 |
| `wait_and_see_values`, `dual_ldr_restricted_primal`, `coupling_multipliers` | its ceiling, master and pricing multipliers |
| `dual_ldr_row` | the basis-weighted average shared by every dual formulation |
| `Basis`, `BP2`, `PW3`, `CONSTANT_BASIS`, `linear_basis` | the basis functions of a rule, with the stage each one belongs to |
| `Sample`, `subsample` | a sample of scenarios with its probabilities, basis matrix and mean basis vector, built once and passed to every formulation |
| `sample_scenarios`, `inflow_matrix`, `basis_values`, `add_ldr_coefficients!`, `ldr_coefficients`, `reshape_coefficients`, `fix_nonanticipativity!` | shared by every formulation |

Both decompositions return the `(; Obj, Bound, H₁, V₁, T₁, dt)` fields the other
solution methods of `hydro3.jl` return, followed by diagnostics; only the six
are safe to hand to `run_comparison`, since every other field becomes a column
of its table.

```julia
julia> problem = HydroToy3.build_toy_problem();

julia> HydroToy3.compare_primal_ldr(
           HydroToy3.Sample(problem, HydroToy3.BP2; scenarios = 30))

julia> HydroToy3.primal_ldr_scenario_generation(
           HydroToy3.Sample(problem, HydroToy3.PW3; scenarios = 100_000);
           verbose = true)
```
