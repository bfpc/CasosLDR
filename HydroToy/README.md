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

Current limitations:
* The `evaluate_first_stage` uses a deterministic SAA model to choose the second-stage decisions, which is only an approximation;
* The SAA model uses a tree of $N \times 30$ scenarios, which could be too small in the 3rd stage;
* The SDDP model uses a tree of $N \times 100$ scenarios;
* SDDP has a fixed time limit of 5 seconds;
* The LDR model uses the same number of breakpoints in each stage;
* The LDR model does not use "cross-stage" breakpoints, which could improve the approximation of the second-stage decisions.