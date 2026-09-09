function generate_extreme_points(problem, T)
    # Generate extreme points for the uncertainty set.
    # We assume there are not too many stages / uncertainties, so enumerating the cartesian product is feasible.
    initial = problem.initial_inflow
    min_inflow = Distributions.minimum(problem.inflow_distr)
    max_inflow = Distributions.maximum(problem.inflow_distr)

    # Recursion
    scenarios = [initial]
    probs = [1.0]
    for t in 2:T
        new_scenarios = Vector{Float64}[]
        new_probs = Float64[]
        for (p,s) in zip(probs, scenarios)
            push!(new_scenarios, [s; min_inflow])
            push!(new_probs, p * 0.5)
            push!(new_scenarios, [s; max_inflow])
            push!(new_probs, p * 0.5)
        end
        scenarios = new_scenarios
        probs = new_probs
    end
    return scenarios, probs
end


function extreme_point_ldr(problem::HydroProblem, T=2)
    ldr = Model(Ipopt.Optimizer)
    set_silent(ldr)

    # Uncertainties: inflow
    scenarios, probs = generate_extreme_points(problem, T)
    inflow = inflow_matrix(scenarios) # each column is a scenario
    n_vertices = size(inflow, 2)
    # Decision variables
    @variables(ldr, begin
        0 <= volume[0:T,1:n_vertices] <= problem.hydro_capacity
        0 <= hydro[1:T,1:n_vertices] <= problem.hydro_generation
        0 <= spill[1:T,1:n_vertices]
        0 <= termos[i=1:problem.termo_count, 1:T, 1:n_vertices] <= problem.termo_generations[i]
    end)
    # Future cost θ
    @variable(ldr, θ[1:n_vertices] >= 0)

    # Initial conditions
    fix.(volume[0,1:n_vertices], problem.initial_volume; force=true)
    # Stage decisions
    for t in 1:T, k in 1:n_vertices
        @constraint(ldr, volume[t,k] == volume[t-1,k] + inflow[t,k] - hydro[t,k] - spill[t,k])
        @constraint(ldr, hydro[t,k] + sum(termos[:,t,k]) == problem.stage_demands[t])
    end

    # Future-cost function
    for k in 1:n_vertices
        @constraint(ldr, [i=1:problem.cuts_count], θ[k] >= problem.cuts_coeff[i] * volume[T,k] + problem.cuts_inter[i])
    end
    @objective(ldr, Min, sum(problem.termo_costs[i] * termos[i,t,k] * probs[k] for i in 1:problem.termo_count, t in 1:T, k in 1:n_vertices) + sum(θ[k] * probs[k] for k in 1:n_vertices))

    # LDR decisions
    @variables(ldr, begin
        volume_ldr[1:T,1:2]
        hydro_ldr[1:T,1:2]
        spill_ldr[1:T,1:2]
        termos_ldr[1:problem.termo_count, 1:T, 1:2]
    end)
    # NA constraints
    for t in 1:T-1
        fix.(volume_ldr[t,t+1:T], 0)
        fix.(hydro_ldr[t,t+1:T], 0)
        fix.(spill_ldr[t,t+1:T], 0)
        for i in 1:problem.termo_count
            fix.(termos_ldr[i,t,t+1:T], 0)
        end
    end
    # LDR constraints
    for t in 1:T
        @constraint(ldr, [k=1:n_vertices], volume[t,k] == volume_ldr[t,1] + volume_ldr[t,2] * inflow[t,k])
        @constraint(ldr, [k=1:n_vertices], hydro[t,k] == hydro_ldr[t,1] + hydro_ldr[t,2] * inflow[t,k])
        @constraint(ldr, [k=1:n_vertices], spill[t,k] == spill_ldr[t,1] + spill_ldr[t,2] * inflow[t,k])
        for i in 1:problem.termo_count
            @constraint(ldr, [k=1:n_vertices], termos[i,t,k] == termos_ldr[i,t,1] + termos_ldr[i,t,2] * inflow[t,k])
        end
    end
    @variable(ldr, θ_ldr[1:2])
    @constraint(ldr, [k=1:n_vertices], θ[k] == θ_ldr[1] + θ_ldr[2] * inflow[T,k])

    return ldr
end

simple_functions = [
    x -> x,   # linear
]

function pwl(x, xmin, xmax)
    return min(max(x - xmin, 0), xmax - xmin)
end

# PWlinear on each inflow, 2 breakpoints per inflow
bp2_functions = [
    x -> pwl(x[1], 0, 40),
    x -> pwl(x[1], 40, 80),
    x -> pwl(x[1], 80, 120),
    x -> pwl(x[2], 0, 40),
    x -> pwl(x[2], 40, 80),
    x -> pwl(x[2], 80, 120)
]
bp2_stages = [repeat([2], 3); repeat([3], 3)]

# PWlinear on each inflow, and on the total inflow
pw3_functions = [
    bp2_functions...,
    x -> pwl(x[1] + x[2], 0, 40),
    x -> pwl(x[1] + x[2], 40, 80),
    x -> pwl(x[1] + x[2], 80, 120),
    x -> pwl(x[1] + x[2], 120, 160),
    x -> pwl(x[1] + x[2], 160, 200),
    x -> pwl(x[1] + x[2], 200, 240),
]
pw3_stages = [bp2_stages;  repeat([3], 6)]

"""
    sample_scenarios(problem, T, n_scen)

`n_scen` inflow scenarios of `T` stages: a deterministic first stage followed by
independent draws of `problem.inflow_distr`.
"""
sample_scenarios(problem::HydroProblem, T, n_scen) =
    [[problem.initial_inflow; [rand(problem.inflow_distr) for t in 2:T]] for s in 1:n_scen]

"""
    inflow_matrix(scenarios)

The scenarios as a `T x n_scen` matrix, one column per scenario.
"""
inflow_matrix(scenarios) = reduce(hcat, scenarios)

"""
    basis_values(functions, stage_of_func, inflow)

The `n_scen x (1 + length(functions))` matrix `Φ` of basis values, where
`Φ[k,1] = 1` (the constant term of the decision rule) and `Φ[k,1+j]` is the
`j`-th basis function evaluated on the inflows scenario `k` reveals up to its
stage `stage_of_func[j]`.  A decision rule with coefficients `c[1:K]` then takes
the value `sum(c[j] * Φ[k,j] for j in 1:K)` in scenario `k`.
"""
function basis_values(functions, stage_of_func, inflow)
    n_scen = size(inflow, 2)
    Φ = ones(n_scen, 1 + length(functions))
    for (idx, (f, stage)) in enumerate(zip(functions, stage_of_func))
        for k in 1:n_scen
            Φ[k, 1+idx] = f(inflow[2:stage, k])
        end
    end
    return Φ
end

"""
    add_ldr_coefficients!(model, problem, T, K)

Declare the coefficients of the decision rules -- `K` per decision and stage --
and return them as a flat vector. The Benders decomposition orders them the
same way in the master problem and in every subproblem.
"""
function add_ldr_coefficients!(model, problem::HydroProblem, T, K)
    @variables(model, begin
        volume_ldr[1:T,1:K]
        hydro_ldr[1:T,1:K]
        spill_ldr[1:T,1:K]
        termos_ldr[1:problem.termo_count, 1:T, 1:K]
        θ_ldr[1:K]
    end)
    return ldr_coefficients(model)
end

"The decision-rule coefficients of `model` as a flat vector."
ldr_coefficients(model) = [
    vec(model[:volume_ldr])
    vec(model[:hydro_ldr])
    vec(model[:spill_ldr])
    vec(model[:termos_ldr])
    vec(model[:θ_ldr])
]

"Undo the flattening of `ldr_coefficients` on a vector of coefficient values."
function reshape_coefficients(problem::HydroProblem, T, K, x)
    offset = 0
    next(dims...) = begin
        block = reshape(x[offset .+ (1:prod(dims))], dims...)
        offset += prod(dims)
        block
    end
    return (;
        volume_ldr = next(T, K),
        hydro_ldr = next(T, K),
        spill_ldr = next(T, K),
        termos_ldr = next(problem.termo_count, T, K),
        θ_ldr = next(K),
    )
end

"""
    fix_nonanticipativity!(model, problem, stage_of_func, T)

Zero the coefficients of the basis functions that a stage-`t` decision may not
use, because they depend on inflows revealed only later.
"""
function fix_nonanticipativity!(model, problem::HydroProblem, stage_of_func, T)
    for t in 1:T-1
        for (idx, stage) in enumerate(stage_of_func)
            if t < stage
                fix(model[:volume_ldr][t,1+idx], 0)
                fix(model[:hydro_ldr][t,1+idx], 0)
                fix(model[:spill_ldr][t,1+idx], 0)
                for i in 1:problem.termo_count
                    fix(model[:termos_ldr][i,t,1+idx], 0)
                end
            end
        end
    end
    return model
end

function basis_function_ldr(problem::HydroProblem, functions, stage_of_func; T=2, solver=HiGHS.Optimizer, n_scen=100, scenarios=nothing)
    ldr = Model(solver)
    set_silent(ldr)

    # Uncertainties: inflow
    scenarios = something(scenarios, sample_scenarios(problem, T, n_scen))
    n_scen = length(scenarios)
    probs = fill(1/n_scen, n_scen)
    inflow = inflow_matrix(scenarios) # each column is a scenario
    # Decision variables
    @variables(ldr, begin
        0 <= volume[0:T,1:n_scen] <= problem.hydro_capacity
        0 <= hydro[1:T,1:n_scen] <= problem.hydro_generation
        0 <= spill[1:T,1:n_scen]
        0 <= termos[i=1:problem.termo_count, 1:T, 1:n_scen] <= problem.termo_generations[i]
    end)
    # Future cost θ
    @variable(ldr, θ[1:n_scen] >= 0)

    # Initial conditions
    fix.(volume[0,1:n_scen], problem.initial_volume; force=true)
    # Stage decisions
    for t in 1:T, k in 1:n_scen
        @constraint(ldr, volume[t,k] == volume[t-1,k] + inflow[t,k] - hydro[t,k] - spill[t,k])
        @constraint(ldr, hydro[t,k] + sum(termos[:,t,k]) == problem.stage_demands[t])
    end

    # Future-cost function
    for k in 1:n_scen
        @constraint(ldr, [i=1:problem.cuts_count], θ[k] >= problem.cuts_coeff[i] * volume[T,k] + problem.cuts_inter[i])
    end
    @objective(ldr, Min, sum(problem.termo_costs[i] * termos[i,t,k] * probs[k] for i in 1:problem.termo_count, t in 1:T, k in 1:n_scen) + sum(θ[k] * probs[k] for k in 1:n_scen))

    # Basis LDR decisions
    Φ = basis_values(functions, stage_of_func, inflow)
    K = size(Φ, 2)
    add_ldr_coefficients!(ldr, problem, T, K)
    volume_ldr, hydro_ldr, spill_ldr, termos_ldr, θ_ldr =
        ldr[:volume_ldr], ldr[:hydro_ldr], ldr[:spill_ldr], ldr[:termos_ldr], ldr[:θ_ldr]
    # NA constraints
    fix_nonanticipativity!(ldr, problem, stage_of_func, T)
    # LDR constraints
    for t in 1:T
        @constraint(ldr, [k=1:n_scen], volume[t,k] == sum(volume_ldr[t,j] * Φ[k,j] for j in 1:K))
        @constraint(ldr, [k=1:n_scen], hydro[t,k] == sum(hydro_ldr[t,j] * Φ[k,j] for j in 1:K))
        @constraint(ldr, [k=1:n_scen], spill[t,k] == sum(spill_ldr[t,j] * Φ[k,j] for j in 1:K))
        for i in 1:problem.termo_count
            @constraint(ldr, [k=1:n_scen], termos[i,t,k] == sum(termos_ldr[i,t,j] * Φ[k,j] for j in 1:K))
        end
    end
    @constraint(ldr, [k=1:n_scen], θ[k] == sum(θ_ldr[j] * Φ[k,j] for j in 1:K))

    return ldr
end

"""
    dual_ldr_row(probs, Φ, row)

The basis-weighted average of a canonical-form row over the sample: a `K`-vector,
one entry per basis function, of `Σ_k probs[k] * row[k] * Φ[k,j]` -- what the
scenario expansion of the dual LDR (`Dual_scenarios.tex`, eq. 93) sets to zero
for every row of the canonical form `A x(ξ) + s(ξ) = b(ξ)`.
"""
dual_ldr_row(probs, Φ, row) = Φ' * (probs .* row)

"""
    basis_function_dual_ldr(problem, functions, stage_of_func; T=2, solver=HiGHS.Optimizer, n_scen=100, scenarios=nothing)

Monolithic scenario expansion of the dual linear decision rule of `problem`
(`Dual_scenarios.tex`, "Scenario expansion"). Where `basis_function_ldr`
restricts the *primal* decisions to a rule and keeps every dispatch constraint
exact, scenario by scenario, this restricts the *dual* multiplier instead,
`z(ξ) = Z·L(ξ)`, and leaves the primal scenario variables `x(ξ_k)`, `s(ξ_k)`
free. Eliminating `Z` -- the sup over it is `0` where the multiplier's
coefficient vanishes, `+∞` otherwise -- turns every row of the canonical form
`A x(ξ) + s(ξ) = b(ξ), s(ξ) ≥ 0` from a pointwise constraint into its
basis-weighted average over the sample being zero:
`(1/K) Σ_k (A x(ξ_k) + s(ξ_k) - b(ξ_k)) L(ξ_k)^T = 0`. By weak duality this is
a *lower* bound on the value of `basis_function_ldr`'s sampled model -- dual to
that upper bound -- and it can be loose, or even unbounded below, for a basis
that is too poor, since nothing here forces `x(ξ_k)` itself to be
non-anticipative.

The canonical form is built explicitly: `volume`, `hydro`, `spill`, `termos`
and `θ` are free, and every one of their original bounds -- along with the
future-cost cuts -- becomes an equality against a slack `≥ 0`; the water
balance and the demand balance, already equalities, get none. Accepts the same
arguments as `basis_function_ldr`.
"""
function basis_function_dual_ldr(problem::HydroProblem, functions, stage_of_func; T=2, solver=HiGHS.Optimizer, n_scen=100, scenarios=nothing)
    ldr = Model(solver)
    set_silent(ldr)

    # Uncertainties: inflow
    scenarios = something(scenarios, sample_scenarios(problem, T, n_scen))
    n_scen = length(scenarios)
    probs = fill(1/n_scen, n_scen)
    inflow = inflow_matrix(scenarios) # each column is a scenario

    # Basis functions
    Φ = basis_values(functions, stage_of_func, inflow)

    # Primal scenario variables x(ξ_k): free, their bounds become slack rows below
    @variables(ldr, begin
        volume[1:T,1:n_scen]
        hydro[1:T,1:n_scen]
        spill[1:T,1:n_scen]
        termos[1:problem.termo_count,1:T,1:n_scen]
        θ[1:n_scen]
    end)

    # Slacks s(ξ_k) >= 0, one per bound / inequality of basis_function_ldr
    @variables(ldr, begin
        volume_hi[1:T,1:n_scen] >= 0
        volume_lo[1:T,1:n_scen] >= 0
        hydro_hi[1:T,1:n_scen] >= 0
        hydro_lo[1:T,1:n_scen] >= 0
        spill_lo[1:T,1:n_scen] >= 0
        termos_hi[1:problem.termo_count,1:T,1:n_scen] >= 0
        termos_lo[1:problem.termo_count,1:T,1:n_scen] >= 0
        θ_lo[1:n_scen] >= 0
        cuts_slack[1:problem.cuts_count,1:n_scen] >= 0
    end)

    @objective(ldr, Min,
        sum(problem.termo_costs[i] * termos[i,t,k] * probs[k]
            for i in 1:problem.termo_count, t in 1:T, k in 1:n_scen) +
        sum(θ[k] * probs[k] for k in 1:n_scen))

    # Water balance and demand balance: already equalities, no slack
    for t in 1:T
        previous = t == 1 ? fill(problem.initial_volume, n_scen) : volume[t-1,:]
        @constraint(ldr, dual_ldr_row(probs, Φ,
            volume[t,:] .- previous .- inflow[t,:] .+ hydro[t,:] .+ spill[t,:]) .== 0)
        @constraint(ldr, dual_ldr_row(probs, Φ,
            hydro[t,:] .+ sum(termos[i,t,:] for i in 1:problem.termo_count) .-
            problem.stage_demands[t]) .== 0)
    end

    # Bounds, turned into slack rows
    for t in 1:T
        @constraint(ldr, dual_ldr_row(probs, Φ, volume[t,:] .+ volume_hi[t,:] .- problem.hydro_capacity) .== 0)
        @constraint(ldr, dual_ldr_row(probs, Φ, volume_lo[t,:] .- volume[t,:]) .== 0)
        @constraint(ldr, dual_ldr_row(probs, Φ, hydro[t,:] .+ hydro_hi[t,:] .- problem.hydro_generation) .== 0)
        @constraint(ldr, dual_ldr_row(probs, Φ, hydro_lo[t,:] .- hydro[t,:]) .== 0)
        @constraint(ldr, dual_ldr_row(probs, Φ, spill_lo[t,:] .- spill[t,:]) .== 0)
        for i in 1:problem.termo_count
            @constraint(ldr, dual_ldr_row(probs, Φ,
                termos[i,t,:] .+ termos_hi[i,t,:] .- problem.termo_generations[i]) .== 0)
            @constraint(ldr, dual_ldr_row(probs, Φ, termos_lo[i,t,:] .- termos[i,t,:]) .== 0)
        end
    end
    @constraint(ldr, dual_ldr_row(probs, Φ, θ_lo .- θ) .== 0)
    for i in 1:problem.cuts_count
        @constraint(ldr, dual_ldr_row(probs, Φ,
            cuts_slack[i,:] .+ problem.cuts_coeff[i] .* volume[T,:] .- θ .+ problem.cuts_inter[i]) .== 0)
    end

    return ldr
end

"""
    basis_function_dual_ldr_compact(problem, functions, stage_of_func; T=2, solver=HiGHS.Optimizer, n_scen=100, scenarios=nothing)

A leaner scenario expansion of the dual LDR than `basis_function_dual_ldr`: a
canonical-form row only needs a slack -- hence a dual multiplier for
`dual_ldr_row` to relax -- when it carries information beyond a bound already
on `x` itself. Every bound of `basis_function_ldr` (`0<=volume<=capacity`,
`0<=hydro<=generation`, `spill>=0`, `0<=termos<=termo_generations[i]`,
`θ>=0`) has exactly this shape: its "slack" is just `x` (the lower bound) or
`bound - x` (the upper one), tied to `x` by a trivial identity rather than a
genuine coupling. Relaxing *that* identity's multiplier to a basis-restricted
rule throws away information for nothing, since substituting the identity back
in changes nothing about what `x` is allowed to be -- it only lets `x` and its
own bound-slack disagree scenario by scenario, which the model in
`basis_function_dual_ldr`'s docstring found HiGHS is only too willing to
exploit, at the cost of the bound (see there for how it got stuck at the
deterministic mean-scenario value).

So here `volume`, `hydro`, `spill`, `termos` and `θ` keep their ORIGINAL,
pointwise bounds -- exactly as in `basis_function_ldr` -- and only the rows
that genuinely couple different components -- water balance, demand balance,
and the future-cost cuts (the one bound that mixes `θ` and `volume`, so it
keeps its own slack `cuts_slack`) -- are relaxed to their basis-weighted
average, following the same derivation as `basis_function_dual_ldr`.

Accepts the same arguments as `basis_function_ldr`. Every row this model
enforces is also enforced (in one of the two forms) by `basis_function_dual_ldr`,
so solving it gives a lower bound at least as tight; see
`dual_ldr_compact_moments` for reading `E[x(ξ)L(ξ)^T]` off the solution.
"""
function basis_function_dual_ldr_compact(problem::HydroProblem, functions, stage_of_func; T=2, solver=HiGHS.Optimizer, n_scen=100, scenarios=nothing)
    ldr = Model(solver)
    set_silent(ldr)

    # Uncertainties: inflow
    scenarios = something(scenarios, sample_scenarios(problem, T, n_scen))
    n_scen = length(scenarios)
    probs = fill(1/n_scen, n_scen)
    inflow = inflow_matrix(scenarios) # each column is a scenario

    # Basis functions
    Φ = basis_values(functions, stage_of_func, inflow)

    # Primal scenario variables x(ξ_k), bounded exactly as in basis_function_ldr:
    # no slack for a bound that x already carries directly.
    @variables(ldr, begin
        0 <= volume[1:T,1:n_scen] <= problem.hydro_capacity
        0 <= hydro[1:T,1:n_scen] <= problem.hydro_generation
        0 <= spill[1:T,1:n_scen]
        0 <= termos[i=1:problem.termo_count,1:T,1:n_scen] <= problem.termo_generations[i]
        θ[1:n_scen] >= 0
    end)
    # The one genuine coupling inequality still needs its own slack s(ξ_k) >= 0
    @variable(ldr, cuts_slack[1:problem.cuts_count,1:n_scen] >= 0)

    @objective(ldr, Min,
        sum(problem.termo_costs[i] * termos[i,t,k] * probs[k]
            for i in 1:problem.termo_count, t in 1:T, k in 1:n_scen) +
        sum(θ[k] * probs[k] for k in 1:n_scen))

    # Water balance and demand balance: already equalities, no slack
    for t in 1:T
        previous = t == 1 ? fill(problem.initial_volume, n_scen) : volume[t-1,:]
        @constraint(ldr, dual_ldr_row(probs, Φ,
            volume[t,:] .- previous .- inflow[t,:] .+ hydro[t,:] .+ spill[t,:]) .== 0)
        @constraint(ldr, dual_ldr_row(probs, Φ,
            hydro[t,:] .+ sum(termos[i,t,:] for i in 1:problem.termo_count) .-
            problem.stage_demands[t]) .== 0)
    end
    # The future-cost cuts, the one bound genuinely coupling θ and volume
    for i in 1:problem.cuts_count
        @constraint(ldr, dual_ldr_row(probs, Φ,
            cuts_slack[i,:] .+ problem.cuts_coeff[i] .* volume[T,:] .- θ .+ problem.cuts_inter[i]) .== 0)
    end

    return ldr
end

"""
    dual_ldr_compact_moments(model, Φ, probs)

`M_x = E[x(ξ)L(ξ)^T]` of a solved `basis_function_dual_ldr_compact`: the
basis-weighted average of every scenario's `volume`, `hydro`, `spill`,
`termos` and `θ`, one `... x size(Φ,2)` matrix per decision. Unlike
`basis_function_ldr`'s rule, `x(ξ_k)` here is never forced to equal this
combination -- individual scenarios are free to deviate from it, which is the
source of the model's extra slack (see `basis_function_dual_ldr`'s docstring)
-- so `M_x` is only the closest rule-shaped *summary* of the solution, not a
policy the model guarantees is itself feasible.
"""
function dual_ldr_compact_moments(model, Φ, probs)
    moment(row) = dual_ldr_row(probs, Φ, row)
    volume, hydro, spill, termos, θ = model[:volume], model[:hydro], model[:spill], model[:termos], model[:θ]
    T = size(volume, 1)
    termo_count = size(termos, 1)
    return (;
        volume_mx = reduce(vcat, moment(value.(volume[t,:]))' for t in 1:T),
        hydro_mx = reduce(vcat, moment(value.(hydro[t,:]))' for t in 1:T),
        spill_mx = reduce(vcat, moment(value.(spill[t,:]))' for t in 1:T),
        termos_mx = cat(
            (reduce(vcat, moment(value.(termos[i,t,:]))' for t in 1:T) for i in 1:termo_count)...,
            dims = 3,
        ),
        θ_mx = moment(value.(θ)),
    )
end

# ---------------------------------------------------------------------------
# Two-stage (Benders) decomposition of the basis-function LDR model
#
# `basis_function_ldr` builds one copy of the dispatch problem per scenario and
# ties every copy to the same decision-rule coefficients.  Those coefficients
# are the only variables shared by the scenarios, so they are a natural first
# stage:
#
#   min_x  Σ_k p_k Q_k(x),   x = (volume_ldr, hydro_ldr, spill_ldr, termos_ldr, θ_ldr)
#
# where Q_k(x) is the cost of scenario k under the rule x, and Q_k(x) = +∞ when
# the rule is infeasible in that scenario.  Each Q_k is evaluated by a small LP
# (`basis_ldr_subproblem`) holding scenario k's dispatch constraints, coupled to
# a fixed copy of x.  Solving it in two passes gives the two kinds of cut:
#
#   * feasibility pass: minimize the L1 distance from the policy the rule
#     prescribes to scenario k's feasible set.  A positive optimum v means the
#     rule is infeasible here, and its subgradient g yields the feasibility cut
#     `0 >= v + g'(x - x̄)`, which every feasible rule satisfies;
#   * optimality pass: with that distance pinned to zero, minimize scenario k's
#     cost.  The optimum q and its subgradient g yield the optimality cut
#     `α_k >= q + g'(x - x̄)` on the master's epigraph variable α_k.
#
# Both subgradients are read off the reduced costs of the fixed copy of x.
#
# Note that a rule leaves the subproblem no recourse: fixing x fixes every
# scenario variable.  A scenario's cost is therefore an affine function of the
# rule, and a handful of optimality cuts describe it exactly; the work of this
# decomposition is in the feasibility cuts, which have to reconstruct the
# polyhedron of implementable rules -- expect thousands of them over tens of
# iterations, and the monolithic model to be faster on problems this size.
# ---------------------------------------------------------------------------

"""
    basis_ldr_subproblem(problem, T, K, inflow, basis, solver)

The subproblem of one scenario: its dispatch constraints, a fixed copy of the
decision-rule coefficients, and the slacks measuring the violation of the
coupling `decision == Σ_j coefficient_j * basis_j`.

`inflow` is the scenario's inflow per stage and `basis` its row of the basis
matrix (see `basis_values`).  `feasibility_mode!` and `optimality_mode!` switch
the model between the two passes.
"""
function basis_ldr_subproblem(problem::HydroProblem, T, K, inflow, basis, solver)
    sub = Model(solver)
    set_silent(sub)

    # Scenario decisions, exactly as in the monolithic model
    @variables(sub, begin
        0 <= volume[0:T] <= problem.hydro_capacity
        0 <= hydro[1:T] <= problem.hydro_generation
        0 <= spill[1:T]
        0 <= termos[i=1:problem.termo_count, 1:T] <= problem.termo_generations[i]
        θ >= 0
    end)
    fix(volume[0], problem.initial_volume; force=true)
    for t in 1:T
        @constraint(sub, volume[t] == volume[t-1] + inflow[t] - hydro[t] - spill[t])
        @constraint(sub, hydro[t] + sum(termos[:,t]) == problem.stage_demands[t])
    end
    @constraint(sub, [i=1:problem.cuts_count], θ >= problem.cuts_coeff[i] * volume[T] + problem.cuts_inter[i])

    # Copy of the first-stage variables, fixed to the master solution each iteration
    params = add_ldr_coefficients!(sub, problem, T, K)
    fix.(params, 0.0)

    # Coupling to the decision rule, relaxed by slacks (positive and negative
    # part of each violation) so that the feasibility pass is always feasible
    @variables(sub, begin
        volume_gap[1:T,1:2] >= 0
        hydro_gap[1:T,1:2] >= 0
        spill_gap[1:T,1:2] >= 0
        termos_gap[1:problem.termo_count, 1:T, 1:2] >= 0
        θ_gap[1:2] >= 0
    end)
    rule(coeffs) = sum(coeffs[j] * basis[j] for j in 1:K)
    for t in 1:T
        @constraint(sub, volume[t] - rule(sub[:volume_ldr][t,:]) == volume_gap[t,1] - volume_gap[t,2])
        @constraint(sub, hydro[t] - rule(sub[:hydro_ldr][t,:]) == hydro_gap[t,1] - hydro_gap[t,2])
        @constraint(sub, spill[t] - rule(sub[:spill_ldr][t,:]) == spill_gap[t,1] - spill_gap[t,2])
        for i in 1:problem.termo_count
            @constraint(sub, termos[i,t] - rule(sub[:termos_ldr][i,t,:]) == termos_gap[i,t,1] - termos_gap[i,t,2])
        end
    end
    @constraint(sub, θ - rule(sub[:θ_ldr]) == θ_gap[1] - θ_gap[2])

    @expression(sub, infeasibility,
        sum(volume_gap) + sum(hydro_gap) + sum(spill_gap) + sum(termos_gap) + sum(θ_gap))
    @expression(sub, cost,
        sum(problem.termo_costs[i] * termos[i,t] for i in 1:problem.termo_count, t in 1:T) + θ)
    sub.ext[:gaps] = [vec(volume_gap); vec(hydro_gap); vec(spill_gap); vec(termos_gap); vec(θ_gap)]
    sub.ext[:params] = params
    return sub
end

"Let the coupling slacks free and minimize them: how infeasible is the rule here?"
function feasibility_mode!(sub)
    for gap in sub.ext[:gaps]
        has_upper_bound(gap) && delete_upper_bound(gap)
    end
    @objective(sub, Min, sub[:infeasibility])
    return sub
end

"""
    optimality_mode!(sub, slack = 0.0)

Pin the coupling slacks to zero and minimize the scenario cost.  `slack` is the
violation left admissible: the feasibility pass accepts a rule whose violation is
within a tolerance, and the cost of such a rule has to be priced by a subproblem
that is not infeasible by that same tolerance.  Cuts stay valid because a larger
`slack` only relaxes the subproblem, hence lowers the value they minorize.
"""
function optimality_mode!(sub, slack = 0.0)
    set_upper_bound.(sub.ext[:gaps], slack)
    @objective(sub, Min, sub[:cost])
    return sub
end

"""
    subproblem_cut(value, sub, x, x̄)

The affine minorant `value + g'(x - x̄)` of the subproblem's optimal value, whose
subgradient `g` with respect to the first-stage variables is the reduced cost of
their fixed copy inside the subproblem.  Coefficients that came out zero are
dropped, which keeps the cuts sparse.
"""
function subproblem_cut(value, sub, x, x̄)
    g = reduced_cost.(sub.ext[:params])
    cut = AffExpr(value)
    for j in eachindex(x)
        iszero(g[j]) && continue
        add_to_expression!(cut, g[j], x[j])
        add_to_expression!(cut, -g[j] * x̄[j])
    end
    return cut
end

"""
    basis_function_ldr_benders(problem, functions, stage_of_func; kwargs...)

Solve the model of `basis_function_ldr` by Benders decomposition: the
decision-rule coefficients stay in a master problem and each scenario is solved
separately, contributing a feasibility or an optimality cut (see the comment
above).  Accepts the same `T`, `solver`, `n_scen` and `scenarios` as
`basis_function_ldr`, plus

  * `max_iterations`, `tol`: stop when the gap between the master bound and the
    best rule found is within `tol` (relative), or after so many iterations;
  * `feasibility_tol`: coupling violation above which a scenario counts as
    infeasible;
  * `verbose`: print the bounds and the cuts of every iteration.

`Obj`, `Bound`, `H₁`, `V₁`, `T₁` and `dt` are the fields the other solution
methods return -- first-stage decisions are the constant terms of the stage-1
rules, which non-anticipativity leaves deterministic -- so
`sol -> (; sol.Obj, sol.Bound, sol.H₁, sol.V₁, sol.T₁, sol.dt)` makes a
`run_comparison` row out of the result.  The remaining fields are for
inspection: whether the loop `converged`, its iteration count, the number of
cuts of each kind, the fitted `coefficients`, and the `master` and `subproblems`
models.  A rule found infeasible has `Obj = Inf` and `NaN` coefficients.
"""
function basis_function_ldr_benders(
    problem::HydroProblem,
    functions,
    stage_of_func;
    T = 2,
    solver = HiGHS.Optimizer,
    n_scen = 100,
    scenarios = nothing,
    max_iterations = 500,
    tol = 1e-6,
    feasibility_tol = 1e-6,
    verbose = false,
)
    # Uncertainties: inflow
    scenarios = something(scenarios, sample_scenarios(problem, T, n_scen))
    n_scen = length(scenarios)
    probs = fill(1/n_scen, n_scen)
    inflow = inflow_matrix(scenarios) # each column is a scenario
    Φ = basis_values(functions, stage_of_func, inflow)
    K = size(Φ, 2)

    # First stage: the decision-rule coefficients, and one epigraph variable per
    # scenario (costs are non-negative, hence the lower bound of zero).
    master = Model(solver)
    set_silent(master)
    x = add_ldr_coefficients!(master, problem, T, K)
    fix_nonanticipativity!(master, problem, stage_of_func, T)
    @variable(master, α[1:n_scen] >= 0)
    @objective(master, Min, sum(probs[k] * α[k] for k in 1:n_scen))

    # Second stage: one subproblem per scenario, built once and re-solved
    subs = [basis_ldr_subproblem(problem, T, K, inflow[:,k], Φ[k,:], solver) for k in 1:n_scen]

    lower, upper = -Inf, Inf
    best, converged = nothing, false
    feasibility_cuts, optimality_cuts, iterations, dt = 0, 0, 0, 0.0
    for iteration in 1:max_iterations
        iterations = iteration
        dt += @elapsed optimize!(master)
        if termination_status(master) != OPTIMAL
            # Feasibility cuts have exhausted the rules: none is implementable.
            verbose && println("master problem is $(termination_status(master))")
            termination_status(master) == INFEASIBLE && ((lower, converged) = (Inf, true))
            break
        end
        lower = objective_value(master)
        # All of the master solution is read before any cut is added to it.
        x̄, ᾱ = value.(x), value.(α)

        cost, cuts, implementable, total_violation = 0.0, 0, true, 0.0
        for k in 1:n_scen
            sub = subs[k]
            fix.(sub.ext[:params], x̄)

            feasibility_mode!(sub)
            dt += @elapsed optimize!(sub)
            termination_status(sub) == OPTIMAL ||
                error("feasibility pass of scenario $k is $(termination_status(sub))")
            violation = objective_value(sub)
            total_violation += probs[k] * violation
            # Read the feasibility cut before re-solving: it is also the
            # fallback if the cost of a borderline rule cannot be priced.
            feasibility_cut = subproblem_cut(violation, sub, x, x̄)

            infeasible = violation > feasibility_tol
            if !infeasible
                optimality_mode!(sub)
                dt += @elapsed optimize!(sub)
                if termination_status(sub) != OPTIMAL
                    # The rule only just fits this scenario: price it allowing
                    # the violation the feasibility pass already accepted.
                    optimality_mode!(sub, feasibility_tol)
                    dt += @elapsed optimize!(sub)
                end
                # Very unlikely, but it is numerically possible that a "re-solve"
                # with fixed bounds returns infeasible from the solver.
                infeasible = termination_status(sub) != OPTIMAL
            end
            if infeasible
                @constraint(master, 0 >= feasibility_cut)
                feasibility_cuts += 1
                cuts += 1
                implementable = false
                continue
            end
            q = objective_value(sub)
            cost += probs[k] * q
            if q > ᾱ[k] + tol * (1 + abs(q))
                @constraint(master, α[k] >= subproblem_cut(q, sub, x, x̄))
                optimality_cuts += 1
                cuts += 1
            end
        end
        if implementable && cost < upper
            upper, best = cost, x̄
        end

        verbose && println("iteration $iteration: bound $(round(lower; digits=2)), " *
            "cost $(implementable ? round(cost; digits=2) : Inf), " *
            "violation $(round(total_violation; digits=4)), " *
            "$feasibility_cuts feasibility and $optimality_cuts optimality cuts")
        # No cut left to add means the master already models every scenario
        # exactly at x̄, hence its bound is attained.
        if cuts == 0 || (implementable && upper - lower <= tol * (1 + abs(upper)))
            converged = true
            break
        end
    end

    coefficients = reshape_coefficients(problem, T, K, something(best, fill(NaN, length(x))))
    return (;
        Obj = upper,
        ObjBound = lower,
        Bound = -Inf,
        H₁ = coefficients.hydro_ldr[1,1],
        V₁ = coefficients.volume_ldr[1,1],
        T₁ = coefficients.termos_ldr[:,1,1],
        dt,
        converged,
        iterations,
        feasibility_cuts,
        optimality_cuts,
        coefficients,
        master,
        subproblems = subs,
    )
end

# ---------------------------------------------------------------------------
# Scenario (constraint) generation for the basis-function LDR model
#
# The Benders decomposition above generates feasibility cuts for each scenario, but the
# scenarios' own rows describe it exactly, because the rule leaves its subproblems with
# no recourse.  Generating the rows instead attacks that directly, and two features
# of the model make it a good fit.
#
# First, substituting the rule into the objective of `basis_function_ldr` gives
#
#   Σ_i,t,k p_k c_i termos[i,t,k] + Σ_k p_k θ[k]
#     = Σ_i,t,j c_i termos_ldr[i,t,j] φ̄_j + Σ_j θ_ldr[j] φ̄_j,   φ̄_j = Σ_k p_k Φ[k,j]
#
# so the objective depends on the sample only through the mean basis vector φ̄, a
# K-vector of the whole sample, computable up front.  The master therefore carries
# the exact objective of the full sample (no need for optimality cuts!), and adding
# scenarios is about feasibility alone.
#
# Second, substituting the rule into a scenario's constraints eliminates the
# scenario's variables (no recourse!), so a scenario contributes rows in the
# LDR coefficients alone -- (3 + termo_count) bounded decisions and two balances per
# stage, plus the future-cost rows.
#
# The master is then a relaxation of the sampled model: its objective is a
# monotone lower bound, and as soon as no scenario is violated, the rule is
# feasible for the whole sample, hence optimal for it.  Separation needs no
# solver, only the matrix product Φ * coefficients, which is what makes scenario
# counts far past the reach of the monolithic model workable.
# ---------------------------------------------------------------------------

"The mean basis vector `φ̄`, the only thing the objective needs of the sample."
mean_basis(Φ, probs) = Φ' * probs

"""
    set_ldr_objective!(model, problem, T, K, φ̄)

The expected cost of a rule, exactly: linear in the coefficients, with the mean
basis vector `φ̄` of `mean_basis` carrying the whole sample.
"""
function set_ldr_objective!(model, problem::HydroProblem, T, K, φ̄)
    termos_ldr, θ_ldr = model[:termos_ldr], model[:θ_ldr]
    @objective(model, Min,
        sum(problem.termo_costs[i] * termos_ldr[i,t,j] * φ̄[j]
            for i in 1:problem.termo_count, t in 1:T, j in 1:K) +
        sum(θ_ldr[j] * φ̄[j] for j in 1:K))
    return model
end

"""
    rule_values(problem, T, K, x, Φ)

The decisions the rule `x` -- coefficient values, flattened as
`ldr_coefficients` orders them -- prescribes in every scenario of the basis
matrix `Φ`, as one `n_scen x T` matrix per decision (a vector of them per
thermal plant) and one `n_scen` vector of future costs.
"""
function rule_values(problem::HydroProblem, T, K, x, Φ)
    c = reshape_coefficients(problem, T, K, x)
    return (;
        volume = Φ * c.volume_ldr',
        hydro = Φ * c.hydro_ldr',
        spill = Φ * c.spill_ldr',
        termos = [Φ * c.termos_ldr[i,:,:]' for i in 1:problem.termo_count],
        θ = Φ * c.θ_ldr,
    )
end

"""
    scenario_violation(problem, T, decisions, inflow)

How far the decisions of `rule_values` are from feasible, scenario by scenario:
the largest violation among the bounds, the water balance, the demand and the
future-cost rows.  This is the separation step, and it is "just" arithmetic --
the whole sample is checked without calling a solver.
"""
function scenario_violation(problem::HydroProblem, T, decisions, inflow)
    n_scen = size(inflow, 2)
    violation = zeros(n_scen)
    worst!(v) = violation .= max.(violation, v)
    above(value, bound) = max.(value .- bound, 0.0)
    below(value, bound) = max.(bound .- value, 0.0)
    for t in 1:T
        worst!(below(decisions.volume[:,t], 0.0))
        worst!(above(decisions.volume[:,t], problem.hydro_capacity))
        worst!(below(decisions.hydro[:,t], 0.0))
        worst!(above(decisions.hydro[:,t], problem.hydro_generation))
        worst!(below(decisions.spill[:,t], 0.0))
        for i in 1:problem.termo_count
            worst!(below(decisions.termos[i][:,t], 0.0))
            worst!(above(decisions.termos[i][:,t], problem.termo_generations[i]))
        end
        previous = t == 1 ? fill(problem.initial_volume, n_scen) : decisions.volume[:,t-1]
        worst!(abs.(decisions.volume[:,t] .- previous .- inflow[t,:] .+
                    decisions.hydro[:,t] .+ decisions.spill[:,t]))
        worst!(abs.(decisions.hydro[:,t] .+
                    sum(decisions.termos[i][:,t] for i in 1:problem.termo_count) .-
                    problem.stage_demands[t]))
    end
    worst!(below(decisions.θ, 0.0))
    for i in 1:problem.cuts_count
        worst!(above(problem.cuts_coeff[i] .* decisions.volume[:,T] .+ problem.cuts_inter[i],
                     decisions.θ))
    end
    return violation
end

"""
    add_scenario_rows!(master, problem, T, K, inflow, basis; equalities)

Add one scenario's constraints, written in the coefficients alone: the bounds of
every decision and the future-cost rows, plus -- when `equalities` is
`:per_scenario` -- its water balance and demand rows.
"""
function add_scenario_rows!(master, problem::HydroProblem, T, K, inflow, basis;
                            equalities = :per_scenario)
    volume_ldr, hydro_ldr, spill_ldr, termos_ldr, θ_ldr =
        master[:volume_ldr], master[:hydro_ldr], master[:spill_ldr],
        master[:termos_ldr], master[:θ_ldr]
    rule(coeffs) = sum(coeffs[j] * basis[j] for j in 1:K)
    for t in 1:T
        @constraint(master, 0 <= rule(volume_ldr[t,:]) <= problem.hydro_capacity)
        @constraint(master, 0 <= rule(hydro_ldr[t,:]) <= problem.hydro_generation)
        @constraint(master, rule(spill_ldr[t,:]) >= 0)
        for i in 1:problem.termo_count
            @constraint(master, 0 <= rule(termos_ldr[i,t,:]) <= problem.termo_generations[i])
        end
        if equalities === :per_scenario
            previous = t == 1 ? AffExpr(problem.initial_volume) : rule(volume_ldr[t-1,:])
            @constraint(master, rule(volume_ldr[t,:]) ==
                previous + inflow[t] - rule(hydro_ldr[t,:]) - rule(spill_ldr[t,:]))
            @constraint(master, rule(hydro_ldr[t,:]) +
                sum(rule(termos_ldr[i,t,:]) for i in 1:problem.termo_count) ==
                problem.stage_demands[t])
        end
    end
    @constraint(master, rule(θ_ldr) >= 0)
    @constraint(master, [i=1:problem.cuts_count],
        rule(θ_ldr) >= problem.cuts_coeff[i] * rule(volume_ldr[T,:]) + problem.cuts_inter[i])
    return master
end

"""
    add_coefficientwise_equalities!(master, problem, T, K, Φ, inflow, stage_of_func)

Impose the water balance and the demand once per basis coefficient, so that they
hold for every realization of the inflow and never have to be generated.

This needs the inflow of each stage to be a combination of the basis functions
that stage knows, `inflow[t,k] == Σ_j r[t,j] Φ[k,j]` with `r[t,j] = 0` for the
functions of later stages -- non-anticipativity zeroes the coefficients such a
row would talk about, so a representation using them would be infeasible rather
than merely wrong.  The representation is fitted by least squares over the
admissible columns and its residual checked.  It holds for `bp2_functions` and
`pw3_functions`, whose pieces sum back to the inflow of their stage.
"""
function add_coefficientwise_equalities!(master, problem::HydroProblem, T, K, Φ, inflow,
                                         stage_of_func)
    r = zeros(T, K)
    for t in 1:T
        available = [1; 1 .+ findall(<=(t), stage_of_func)]
        r[t,available] = Φ[:,available] \ inflow[t,:]
        residual = maximum(abs, Φ * r[t,:] .- inflow[t,:])
        residual <= 1e-6 * max(1.0, maximum(abs, inflow[t,:])) || error(
            "the stage-$t inflow is not in the span of the basis functions of " *
            "stages up to $t (residual $residual), so equalities = " *
            ":coefficientwise cannot hold for every realization; use " *
            ":per_scenario, or a basis that represents the inflow as " *
            "bp2_functions does")
    end
    volume_ldr, hydro_ldr, spill_ldr, termos_ldr =
        master[:volume_ldr], master[:hydro_ldr], master[:spill_ldr], master[:termos_ldr]
    for t in 1:T, j in 1:K
        # The initial volume is deterministic, hence carried by the constant term
        previous = t == 1 ? AffExpr(j == 1 ? problem.initial_volume : 0.0) : volume_ldr[t-1,j]
        @constraint(master, volume_ldr[t,j] ==
            previous + r[t,j] - hydro_ldr[t,j] - spill_ldr[t,j])
        @constraint(master, hydro_ldr[t,j] +
            sum(termos_ldr[i,t,j] for i in 1:problem.termo_count) ==
            (j == 1 ? problem.stage_demands[t] : 0.0))
    end
    return master
end

"""
    basis_function_ldr_scenario_generation(problem, functions, stage_of_func; kwargs...)

Solve the model of `basis_function_ldr` by scenario generation: a master problem
over the decision-rule coefficients carries the exact objective of the whole
sample and the rows of a working set of scenarios, which grows with the
scenarios the current rule violates (see the comment above).  Accepts the same
`T`, `solver`, `n_scen` and `scenarios` as `basis_function_ldr`, plus

  * `equalities`: `:per_scenario` generates the water balance and the demand
    with each scenario, and reproduces `basis_function_ldr` exactly;
    `:coefficientwise` imposes them for every realization up front, which is the
    same model whenever `Φ` has full column rank and a stronger one when it does
    not -- in which case the master is no longer a relaxation and no bound is
    reported;
  * `n_init`: size of the initial working set, `2K` by default -- a head start,
    not a necessity, since the aggregate rows keep the master bounded on their
    own.  A master the solver still fails on is widened by `n_init` scenarios at
    a time;
  * `max_add`: how many of the most violated scenarios to add per round.  Fewer
    means more rounds but a smaller master, and the master is what costs: 5 to
    10 beat both extremes on the toy problem;
  * `max_rounds`, `tol`: stop when no scenario is violated by more than `tol`,
    or after so many rounds;
  * `verbose`: print the objective, the working set and the worst violation of
    every round.

Returns the `(; Obj, Bound, H₁, V₁, T₁, dt)` fields of the other solution
methods -- see `basis_ldr_generation_solution` for a `run_comparison` row -- plus
whether the loop `converged`, its number of `rounds`, the `active` scenarios,
the fitted `coefficients` and the `master` model.  `Obj` is exact rather than a
sample estimate: the cost of a rule is affine, so the master objective at
termination *is* the expected cost of the rule over the whole sample.
"""
function basis_function_ldr_scenario_generation(
    problem::HydroProblem,
    functions,
    stage_of_func;
    T = 2,
    solver = HiGHS.Optimizer,
    n_scen = 100,
    scenarios = nothing,
    equalities = :per_scenario,
    n_init = nothing,
    max_add = 10,
    max_rounds = 100,
    tol = 1e-6,
    verbose = false,
)
    equalities in (:per_scenario, :coefficientwise) ||
        throw(ArgumentError("equalities must be :per_scenario or :coefficientwise"))

    # Uncertainties: inflow
    scenarios = something(scenarios, sample_scenarios(problem, T, n_scen))
    n_scen = length(scenarios)
    probs = fill(1/n_scen, n_scen)
    inflow = inflow_matrix(scenarios) # each column is a scenario
    Φ = basis_values(functions, stage_of_func, inflow)
    K = size(Φ, 2)

    master = Model(solver)
    set_silent(master)
    x = add_ldr_coefficients!(master, problem, T, K)
    fix_nonanticipativity!(master, problem, stage_of_func, T)
    φ̄ = mean_basis(Φ, probs)
    set_ldr_objective!(master, problem, T, K, φ̄)

    # With the equalities imposed for every realization the master is a
    # relaxation of the sampled model only if the sample cannot tell the two
    # formulations apart, which is what full column rank of Φ says.
    bounding = true
    if equalities === :coefficientwise
        add_coefficientwise_equalities!(master, problem, T, K, Φ, inflow, stage_of_func)
        bounding = rank(Φ) == K
    end

    # The sample average of the scenarios' rows: a convex combination of rows of
    # the full model, hence implied by it, and the same rows evaluated at the
    # mean basis vector and the mean inflow.  Adding it up front keeps the master
    # bounded (thermal generation and future cost are non-negative),
    # which no working set of scenarios is guaranteed to do on its own.
    add_scenario_rows!(master, problem, T, K, inflow * probs, φ̄; equalities)

    added = falses(n_scen)
    function activate!(candidates)
        activated = 0
        for k in candidates
            added[k] && continue
            add_scenario_rows!(master, problem, T, K, inflow[:,k], Φ[k,:]; equalities)
            added[k] = true
            activated += 1
        end
        return activated
    end
    n_init = something(n_init, min(n_scen, 2K))
    activate!(1:n_init)

    lower, upper = -Inf, Inf
    best, converged = nothing, false
    rounds, dt = 0, 0.0
    for iteration in 1:max_rounds
        rounds = iteration
        dt += @elapsed optimize!(master)
        # The aggregate rows bound the master, so this is a numerical fallback
        # rather than the rule: a master the solver will not solve is one too
        # badly determined by its rows, and more rows are the answer.
        while termination_status(master) ∉ (OPTIMAL, INFEASIBLE) && !all(added)
            verbose && println("round $iteration: master is " *
                "$(termination_status(master)), widening the working set")
            activate!(Iterators.take(findall(.!added), n_init))
            dt += @elapsed optimize!(master)
        end
        if termination_status(master) == INFEASIBLE
            # A subset of the rows is already infeasible, so the whole sample is.
            verbose && println("round $iteration: master problem is INFEASIBLE")
            lower, converged = Inf, true
            break
        end
        termination_status(master) == OPTIMAL ||
            error("master problem is $(termination_status(master))")
        objective = objective_value(master)
        lower = bounding ? objective : -Inf
        x̄ = value.(x)

        violation = scenario_violation(problem, T, rule_values(problem, T, K, x̄, Φ), inflow)
        worst = maximum(violation)
        verbose && println("round $iteration: objective $(round(objective; digits=2)), " *
            "$(count(added)) of $n_scen scenarios, " *
            "worst violation $(round(worst; digits=6))")
        if worst <= tol
            # The rule is feasible for the whole sample, and the objective it was
            # minimized against is that sample's expected cost.
            upper, best, converged = objective, x̄, true
            break
        end
        candidates = Iterators.filter(k -> violation[k] > tol, sortperm(violation; rev = true))
        if activate!(Iterators.take(candidates, max_add)) == 0
            # Only scenarios already in the working set are violated, by less
            # than the solver's own feasibility tolerance: no row left to add.
            verbose && println("round $iteration: no scenario left to add")
            break
        end
    end

    coefficients = reshape_coefficients(problem, T, K, something(best, fill(NaN, length(x))))
    return (;
        Obj = upper,
        ObjBound = lower,
        Bound = -Inf,
        H₁ = coefficients.hydro_ldr[1,1],
        V₁ = coefficients.volume_ldr[1,1],
        T₁ = coefficients.termos_ldr[:,1,1],
        dt,
        converged,
        rounds,
        active = findall(added),
        coefficients,
        master,
    )
end

"""
    basis_ldr_generation_solution(problem, functions, stage_of_func; kwargs...)

`basis_function_ldr_scenario_generation` reduced to the fields `run_comparison`
tabulates, since every other field of the full result would become a column of
the table -- a JuMP model among them.

Meant for a 3-stage `problem`, hence `T = 3` and a basis of the same horizon:
`evaluate_policy` scores the stage-2 and stage-3 recourse of the first-stage
decisions, which a shorter horizon would leave out of the rule.
"""
function basis_ldr_generation_solution(
    problem::HydroProblem,
    functions = bp2_functions,
    stage_of_func = bp2_stages;
    T = 3,
    kwargs...,
)
    sol = basis_function_ldr_scenario_generation(
        problem, functions, stage_of_func; T, kwargs...
    )
    return (; sol.Obj, sol.Bound, sol.H₁, sol.V₁, sol.T₁, sol.dt)
end

"""
    compare_basis_function_ldr(problem, functions, stage_of_func; kwargs...)

Solve the monolithic basis-function LDR model, its Benders decomposition and its
scenario generation in both equality modes, all on the same scenarios.  Keyword
arguments of the two decompositions go in the `benders` and `generation` named
tuples.
"""
function compare_basis_function_ldr(
    problem::HydroProblem,
    functions,
    stage_of_func;
    T = 2,
    n_scen = 20,
    solver = HiGHS.Optimizer,
    verbose = false,
    benders = (;),
    generation = (;),
)
    scenarios = sample_scenarios(problem, T, n_scen)

    full = basis_function_ldr(problem, functions, stage_of_func; T, solver, scenarios)
    dt_full = @elapsed optimize!(full)

    bend = basis_function_ldr_benders(
        problem, functions, stage_of_func; T, solver, scenarios, verbose, benders...
    )
    gen = basis_function_ldr_scenario_generation(
        problem, functions, stage_of_func; T, solver, scenarios, verbose, generation...
    )
    gen_cw = basis_function_ldr_scenario_generation(
        problem, functions, stage_of_func;
        T, solver, scenarios, verbose, generation..., equalities = :coefficientwise
    )

    return (;
        monolithic = termination_status(full) == OPTIMAL ? objective_value(full) : Inf,
        benders = bend.Obj,
        generation = gen.Obj,
        generation_cw = gen_cw.Obj,
        objbound_benders = bend.ObjBound,
        objbound_generation = gen.ObjBound,
        objbound_generation_cw = gen_cw.ObjBound,
        bound_benders = bend.Bound,
        bound_generation = gen.Bound,
        bound_generation_cw = gen_cw.Bound,
        benders_iterations = bend.iterations,
        benders_cuts = bend.feasibility_cuts + bend.optimality_cuts,
        generation_rounds = gen.rounds,
        generation_active = length(gen.active),
        generation_cw_rounds = gen_cw.rounds,
        generation_cw_active = length(gen_cw.active),
        converged = bend.converged && gen.converged && gen_cw.converged,
        dt_monolithic = dt_full,
        dt_benders = bend.dt,
        dt_generation = gen.dt,
        dt_generation_cw = gen_cw.dt,
    )
end

function extreme_point_ldr_X(problem::HydroProblem)
    ldr = Model(Ipopt.Optimizer)
    # set_silent(ldr)

    # Uncertainties: inflow (1+1)
    n = 1
    # Decision variables: (volume, hydro, spill, termos) * 2 stages
    m = 2*(3 + problem.termo_count)
    # Future cost θ (1+1)
    @variable(ldr, X[1:m, 1:n+1])
    @variable(ldr, θ[1:n+1])

    # First-stage
    m_first = 3 + problem.termo_count
    fix.(X[1:m_first, 1+1:1+n], 0) # First-stage decisions do not dependent on uncertainty
    @constraint(ldr, X[1,1] == problem.initial_volume + problem.initial_inflow - X[2,1] - X[3,1])
    @constraint(ldr, X[1,1] >= 0)
    @constraint(ldr, X[2,1] >= 0)
    @constraint(ldr, X[3,1] >= 0)
    @constraint(ldr, [i=1:problem.termo_count], X[3+i,1] >= 0)
    @constraint(ldr, X[1,1] <= problem.hydro_capacity)
    @constraint(ldr, X[2,1] <= problem.hydro_generation)
    @constraint(ldr, [i=1:problem.termo_count], X[3+i,1] <= problem.termo_generations[i])
    @constraint(ldr, X[2,1] + sum(X[3+i,1] for i in 1:problem.termo_count) == problem.stage_demands[1])

    # Second-stage
    min_inflow, max_inflow = Distributions.minimum(problem.inflow_distr), Distributions.maximum(problem.inflow_distr)
    for ξ in [min_inflow, max_inflow]
        @constraint(ldr, X[m_first+1,1] + X[m_first+1,2] * ξ == X[1,1] + ξ - X[m_first+2,1] - X[m_first+2,2] * ξ - X[m_first+3,1] - X[m_first+3,2] * ξ)
        @constraint(ldr, X[m_first+1,1] + X[m_first+1,2] * ξ >= 0)
        @constraint(ldr, X[m_first+2,1] + X[m_first+2,2] * ξ >= 0)
        @constraint(ldr, X[m_first+3,1] + X[m_first+3,2] * ξ >= 0)
        @constraint(ldr, [i=1:problem.termo_count], X[m_first+3+i,1] + X[m_first+3+i,2] * ξ >= 0)
        @constraint(ldr, X[m_first+1,1] + X[m_first+1,2] * ξ <= problem.hydro_capacity)
        @constraint(ldr, X[m_first+2,1] + X[m_first+2,2] * ξ <= problem.hydro_generation)
        @constraint(ldr, [i=1:problem.termo_count], X[m_first+3+i,1] + X[m_first+3+i,2] * ξ <= problem.termo_generations[i])
        @constraint(ldr, X[m_first+2,1] + X[m_first+2,2] * ξ + sum(X[m_first+3+i,1] + X[m_first+3+i,2] * ξ for i in 1:problem.termo_count) == problem.stage_demands[2])
    end

    # Future-cost function
    @constraint(ldr, [i=1:problem.cuts_count], θ[1] + θ[2] * min_inflow >= problem.cuts_coeff[i] * (X[m_first+1,1] + X[m_first+1,2] * min_inflow) + problem.cuts_inter[i])
    @constraint(ldr, [i=1:problem.cuts_count], θ[1] + θ[2] * max_inflow >= problem.cuts_coeff[i] * (X[m_first+1,1] + X[m_first+1,2] * max_inflow) + problem.cuts_inter[i])

    @objective(ldr, Min, sum(problem.termo_costs[i] * (X[3+i,1] + X[m_first+3+i,1] + X[m_first+3+i,2] * mean(problem.inflow_distr)) for i in 1:problem.termo_count) + θ[1] + θ[2] * mean(problem.inflow_distr))

    return ldr
end
