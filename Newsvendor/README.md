# The newsvendor problem

This directory contains a simple implementation of the newsvendor problem, which is a classic problem in operations research and inventory management. The problem involves determining the optimal order quantity for a product that has uncertain demand, with the goal of maximizing expected profit, and is a basic example of a stochastic optimization problem.

It is formulated as
$$ \begin{array}{rll}
\max & \mathbb{E}\left[ p \min(x, d) + r \max(0, x - d) - c x \right] \\
\text{s.t.} & x \geq 0,
\end{array} $$
where $x$ is the order quantity, $d$ is the random demand, $p$ is the selling price, $c$ is the purchase cost, and $r$ is the salvage value.

## newsvendor.jl

This file implements the newsvendor problem, and compares the expected value (EV) solution (a deterministic problem) with the Sample Average Approximation (SAA) solution for different numbers of scenarios and the Linear Decision Rules solution with different number of breakpoints.
We consider two parameter choices for $(p, r, c)$, which yield different _a posteriori_ relative performances of the EV and LDR solutions. The SAA solution is expected to converge to the true optimal solution as the number of scenarios increases, while the LDR solution is expected to improve as the number of breakpoints increases.
