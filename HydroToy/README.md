# Toy Hydro model

This directory contains variations on a toy hydrothermal dispatch problem.
It is formulated as

$$
\begin{array}{rll}
\min & \mathbb{E}\left[\sum_{t=1}^{T} \sum_{i=1}^{N} c_i p_{i,t} + V(v_t) \right] \\
\text{s.t.} & v_{t} = v_{t-1} + a_t(\xi_t) - gh_t - s_t, & \text{water balance} \\
& gh_t + \sum_{i=1}^{N} p_{i,t} = D_t, & \text{demand satisfaction} \\
& 0 \leq p_{i,t} \leq p_{i,\max}, & \text{bounds} \\
& 0 \leq v_t \leq v_{\max}, \\
& 0 \leq gh_t \leq gh_{\max}, \\
& 0 \leq s_t.
\end{array}
$$

There, $v_t$ represents stored volume, $gh_t$ is the hydro generation, $s_t$ is spillage, and $p_{i,t}$ is the thermal generation of unit $i$; all for each time step $t$.
The inflow $a_t(\xi_t)$ is a random variable that depends on the scenario $\xi_t$, and $D_t$ is the demand at time $t$. The objective is to minimize the expected cost of thermal generation over the planning horizon, plus a terminal value function $V$ for the final stored volume.

## hydro2.jl

This is a 2-stage model, comparing the performance of a deterministic model with several stochastic models. The deterministic model uses the expected inflow for each time step.
One stochastic model uses the Sample Average Approximation (SAA) method, and another uses a Linear Decision Rule (LDR) approximation for the second-stage decisions.

The SAA model is parameterized by the number of scenarios (more scenarios provide a better approximation of the true distribution, and therefore both constraints and objective function), and the LDR model is parameterized by the number of breakpoints along the inflow distribution (more breakpoints allow for more flexible approximations).

## hydro3.jl

This repeats the same problem, with identical parameters, except that the model is now 3-stage.
We take inflows to be independent across time, and the SAA and LDR models are parameterized by the number of scenarios and breakpoints, respectively, for each stage.

Besides the SAA and LDR models, a third stochastic model is included that uses a Stochastic Dual Dynamic Programming (SDDP) approximation to decide the policy.
The SDDP model is parameterized by the number of scenarios (to build the SAA scenario tree) and a time limit.

Design choices:
* The SAA model uses a tree of $N \times 30$ scenarios, which could be too small in the 3rd stage;
* The SDDP model uses a tree of $N \times 100$ scenarios;
* SDDP has a fixed limit of 400 iterations;
* The LDR model uses the same number of breakpoints in each stage.

Limitations:
* The `evaluate_first_stage` uses a deterministic SAA model wth 100 uniformly chosen scenarios to evaluate the costs; this is only a Monte-Carlo approximation;
* The LDR model does not use "cross-stage" breakpoints, which could improve the approximation of the second-stage decisions.

## hydro_scenario.jl

Scenario-wise expansions of the LDR model, written out explicitly rather than through `LinearDecisionRules`:
- over the extreme points of the uncertainty set (`extreme_point_ldr`), compatible with uniform, independent probabilities (cf `generate_extreme_points`) for each uncertainty;
- over a sample of the scenarios, with decision rules spanned by a _set_ of piecewise-linear functions of the inflows (`primal_ldr`), trying to mitigate the last limitation (the current polyhedral implementation of the LDR package does not support breakpoints over a linear function of the uncertainties).

That last model can also be solved by two **decompositions** of its decision-rule coefficients:
- a Benders decomposition; and
- a scenario (constraint) generation

The details on the decompositions, and how they compare to each other and to a single LP are in [Decomposition.md](Decomposition.md).

The `compare_methods` uses scenario generation for its `Basis-LDR` rows: it reaches 100 000 scenarios in about a second and a half, where the monolithic model needs 82 s for 10 000.
