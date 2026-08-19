# Toy Hydro model

This directory contains the `HydroQuad` model, a variation on the toy hydrothermal dispatch problem from `HydroToy`, but with quadratic costs.
It is formulated as

$$
\begin{array}{rll}
\min & \mathbb{E}\left[\sum_{t=1}^{T} c(p_t) + V(v_t) \right] \\
\text{s.t.} & v_{t} = v_{t-1} + a_t(\xi_t) - gh_t - s_t, & \text{water balance} \\
& gh_t + p_t = D_t, & \text{demand satisfaction} \\
& 0 \leq p_t \leq p_{\max}, & \text{bounds} \\
& 0 \leq v_t \leq v_{\max}, \\
& 0 \leq gh_t \leq gh_{\max}, \\
& 0 \leq s_t.
\end{array}
$$

The immediate costs $c(p_t)$ are (piecewise-)quadratic in the total thermal generation $p_t$.
This makes the problem a bit more sensitive to the scenario sampling, since 1st and 2nd stage decisions are not piecewise constant anymore.

## hydroquad.jl

This is a 2-stage model, with a simple quadratic cost function for the thermal generation.

## hydroquad_pw.jl

This is a 2-stage model, with a piecewise-quadratic cost function for the thermal generation.
The pieces correspond approximately to the thermal generation costs in the Brazilian national system.
