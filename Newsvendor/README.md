# The newsvendor problem

This directory contains a simple implementation of the newsvendor problem, which is a classic problem in operations research and inventory management. The problem involves determining the optimal order quantity for a product that has uncertain demand, with the goal of maximizing expected profit, and is a basic example of a stochastic optimization problem.

It is formulated as
$$ \begin{array}{rll}
\max & \mathbb{E}\left[ p \cdot \mathrm{sell} + r \cdot \mathrm{ret} - c \cdot \mathrm{buy} \right] \\
\text{s.t.}
& \mathrm{sell} + \mathrm{ret} \leq \mathrm{buy}, \\
& \mathrm{sell} \leq d, \\
& \mathrm{buy}, \mathrm{sell}, \mathrm{ret} \geq 0;
\end{array} $$
where $\mathrm{buy}$ is the order quantity, $\mathrm{sell}$ is the quantity sold and $\mathrm{ret}$ is the quantity returned.
The parameters for the problem are:
* $d$, the random demand,
* $p$, the selling price,
* $c$, the purchase cost, and
* $r$, the salvage value.


## newsvendor.jl

This file implements three models for the newsvendor problem, comparing:
1. the expected value (EV) solution (a deterministic formulation), with
2. the Sample Average Approximation (SAA) solution for different numbers of scenarios, and
3. the Linear Decision Rule (LDR) solution with different number of breakpoints.

We consider two parameter choices for $(p, r, c)$, which yield different _a posteriori_ relative performances of the EV and LDR solutions. The SAA solution is expected to converge to the true optimal solution as the number of scenarios increases, while the LDR solution is expected to improve as the number of breakpoints increases.
