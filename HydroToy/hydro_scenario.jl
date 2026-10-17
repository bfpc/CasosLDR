function generate_extreme_points(problem, T)
    # Generate extreme points for the uncertainty set.
    # We assume there are not too many stages / uncertainties, so enumerating the cartesian product is feasible.
    initial = problem.initial_inflow
    min_inflow = Distributions.minimum(problem.inflow_distr)
    max_inflow = Distributions.maximum(problem.inflow_distr)

    # Recursion. Each scenario is itself a vector of per-stage inflows,
    # so that `scenarios` is always a `Vector{Vector{Float64}}`.
    scenarios = Vector{Float64}[[initial]]
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


"""
    extreme_point_ldr(problem; T = stage_count(problem), solver = HiGHS.Optimizer)

The linear decision rule fitted on the extreme points of the uncertainty set:
`basis_function_ldr` on the `2^(T-1)` scenarios of `generate_extreme_points`,
weighted by their probabilities, with the basis `linear_basis(T)`.
"""
function extreme_point_ldr(problem::HydroProblem; T = stage_count(problem), solver = HiGHS.Optimizer)
    scenarios, probs = generate_extreme_points(problem, T)
    return basis_function_ldr(Sample(problem, linear_basis(T); T, scenarios, probs); solver)
end

simple_functions = [
    x -> x,   # linear
]

function pwl(x, xmin, xmax)
    return min(max(x - xmin, 0), xmax - xmin)
end

"""
    Basis(functions, stages)

The basis functions of a decision rule, besides the constant term that every
rule has.  `functions[j]` is evaluated on the inflows of stages `2:stages[j]`
(so its argument `x[1]` is the stage-2 inflow), and only the decisions of stage
`stages[j]` or later may use it (see `fix_nonanticipativity!`).
"""
struct Basis
    functions::Vector{Function}
    stages::Vector{Int}
    function Basis(functions, stages)
        length(functions) == length(stages) ||
            throw(ArgumentError("$(length(functions)) basis functions but $(length(stages)) stages"))
        return new(functions, stages)
    end
end

"The constant term alone: rules that do not depend on the inflow."
const CONSTANT_BASIS = Basis(Function[], Int[])

"""
    linear_basis(T)

The inflow of every stage `2:T`, each as a basis function of its own stage: a
rule linear in the inflow history.
"""
linear_basis(T) = Basis([x -> x[t-1] for t in 2:T], collect(2:T))

"PWlinear on each inflow, 2 breakpoints per inflow"
const BP2 = Basis(
    [
        x -> pwl(x[1], 0, 40),
        x -> pwl(x[1], 40, 80),
        x -> pwl(x[1], 80, 120),
        x -> pwl(x[2], 0, 40),
        x -> pwl(x[2], 40, 80),
        x -> pwl(x[2], 80, 120),
    ],
    [repeat([2], 3); repeat([3], 3)],
)

"PWlinear on each inflow, and on the total inflow"
const PW3 = Basis(
    [
        BP2.functions...,
        x -> pwl(x[1] + x[2], 0, 40),
        x -> pwl(x[1] + x[2], 40, 80),
        x -> pwl(x[1] + x[2], 80, 120),
        x -> pwl(x[1] + x[2], 120, 160),
        x -> pwl(x[1] + x[2], 160, 200),
        x -> pwl(x[1] + x[2], 200, 240),
    ],
    [BP2.stages; repeat([3], 6)],
)

"The number of stages of `problem`."
stage_count(problem::HydroProblem) = length(problem.stage_demands)

"""
    sample_scenarios(problem, T, n_scen; rng = Random.default_rng())

`n_scen` inflow scenarios of `T` stages: a deterministic first stage followed by
independent draws of `problem.inflow_distr`.
"""
sample_scenarios(problem::HydroProblem, T, n_scen; rng = Random.default_rng()) =
    [[problem.initial_inflow; [rand(rng, problem.inflow_distr) for t in 2:T]] for s in 1:n_scen]

"""
    inflow_matrix(scenarios)

The scenarios as a `T x n_scen` matrix, one column per scenario.
"""
inflow_matrix(scenarios) = reduce(hcat, scenarios)

"""
    basis_values(basis, inflow)

The `n_scen x (1 + length(basis.functions))` matrix `Φ` of basis values, where
`Φ[k,1] = 1` (the constant term of the decision rule) and `Φ[k,1+j]` is the
`j`-th basis function evaluated on the inflows scenario `k` reveals up to its
stage `basis.stages[j]`.  A decision rule with coefficients `c[1:K]` then takes
the value `sum(c[j] * Φ[k,j] for j in 1:K)` in scenario `k`.
"""
function basis_values(basis::Basis, inflow)
    n_scen = size(inflow, 2)
    Φ = ones(n_scen, 1 + length(basis.functions))
    for (idx, (f, stage)) in enumerate(zip(basis.functions, basis.stages))
        for k in 1:n_scen
            Φ[k, 1+idx] = f(inflow[2:stage, k])
        end
    end
    return Φ
end

"The mean basis vector `φ̄`, the only thing a rule's expected cost needs of the sample."
mean_basis(Φ, probs) = Φ' * probs

"""
    active_rank(Φ, probs)

Effective `rank` of basis functions matrix `Φ`, considering only scenarios of positive probability.
"""
active_rank(Φ, probs) = rank(Φ[probs .> 0, :])

"""
    Sample(problem, basis; T = nothing, scenarios = 100, probs = nothing,
           rng = Random.default_rng())

Everything the scenario formulations of this file need of a sample, computed once:
- the `T x n_scen` matrix `inflow` (one column per scenario),
- the scenario probabilities `probs`,
- the `n_scen x K` basis matrix `Φ` of `basis_values`, and
- the mean basis vector `φ̄` of `mean_basis`,
along with the `problem`, `basis` and horizon `T` they came from.

`scenarios` is either a number of scenarios to draw from `rng` with
`sample_scenarios`, or the scenarios themselves.
`T` defaults to the length of the given scenarios, or, when `scenarios` is a number,
to every stage of `problem`; either way it equals the number of stages `problem` actually has.
`probs` defaults to uniform.
"""
struct Sample
    problem::HydroProblem
    basis::Basis
    T::Int
    inflow::Matrix{Float64}
    probs::Vector{Float64}
    Φ::Matrix{Float64}
    φ̄::Vector{Float64}
end

function Sample(
    problem::HydroProblem,
    basis::Basis;
    T = nothing,
    scenarios = 100,
    probs = nothing,
    rng = Random.default_rng(),
)
    if T === nothing
        if scenarios isa Integer
            T = stage_count(problem)
        else
            isempty(scenarios) && throw(ArgumentError("need at least one scenario"))
            T = length(first(scenarios))
        end
    end
    1 <= T <= stage_count(problem) ||
        throw(ArgumentError("T = $T, but the problem has $(stage_count(problem)) stages"))
    maximum(basis.stages; init = 1) <= T ||
        throw(ArgumentError("the basis uses inflows up to stage $(maximum(basis.stages)), past T = $T"))
    scenarios isa Integer && (scenarios = sample_scenarios(problem, T, scenarios; rng))
    all(scenario -> length(scenario) == T, scenarios) ||
        throw(ArgumentError("every scenario must have T = $T inflows"))

    n_scen = length(scenarios)
    probs = something(probs, fill(1/n_scen, n_scen))
    length(probs) == n_scen && all(>=(0), probs) && sum(probs) ≈ 1 ||
        throw(ArgumentError("probs must be $n_scen nonnegative weights summing to one"))
    inflow = inflow_matrix(scenarios) # each column is a scenario
    Φ = basis_values(basis, inflow)
    return Sample(problem, basis, T, inflow, probs, Φ, mean_basis(Φ, probs))
end

"The number of scenarios of `s`."
scenario_count(s::Sample) = length(s.probs)
"The number of coefficients `K` of every decision rule over `s`, the constant term included."
basis_count(s::Sample) = size(s.Φ, 2)

"""
    subsample(s, scenarios)

The scenarios of `s` indexed by `scenarios` -- a vector of indices or a
`BitVector`, never a single scenario, which this would silently drop a
dimension from.  They keep their probabilities in the whole sample, which
therefore no longer sum to one.
"""
function subsample(s::Sample, scenarios)
    scenarios isa Integer &&
        throw(ArgumentError("subsample needs a collection of scenario indices, not a single one"))
    return Sample(s.problem, s.basis, s.T, s.inflow[:,scenarios], s.probs[scenarios],
           s.Φ[scenarios,:], mean_basis(s.Φ[scenarios,:], s.probs[scenarios]))
end

"""
    add_ldr_coefficients!(model, s)

Declare the coefficients of the decision rules -- `basis_count(s)` per decision
and stage -- and return them as a flat vector. The Benders decomposition orders
them the same way in the master problem and in every subproblem.
"""
function add_ldr_coefficients!(model, s::Sample)
    (; problem, T) = s
    K = basis_count(s)
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

"Undo the flattening of `ldr_coefficients` on a vector `x` of coefficient values."
function reshape_coefficients(s::Sample, x)
    (; problem, T) = s
    K = basis_count(s)
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
    fix_nonanticipativity!(model, s)

Zero the coefficients of the basis functions that a stage-`t` decision may not
use, because they depend on inflows revealed only later.
"""
function fix_nonanticipativity!(model, s::Sample)
    (; problem, T) = s
    for t in 1:T-1
        for (idx, stage) in enumerate(s.basis.stages)
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

"""
    basis_function_ldr(s; solver = HiGHS.Optimizer)

The decision rule spanned by the basis of the sample `s`, fitted on its
scenarios: one copy of the dispatch problem per scenario, every copy tied to the
same rule coefficients.  Returns the model, unsolved.
"""
function basis_function_ldr(s::Sample; solver = HiGHS.Optimizer)
    ldr = Model(solver)
    set_silent(ldr)

    (; problem, T, inflow, probs, Φ) = s
    n_scen = scenario_count(s)
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
    K = basis_count(s)
    add_ldr_coefficients!(ldr, s)
    volume_ldr, hydro_ldr, spill_ldr, termos_ldr, θ_ldr =
        ldr[:volume_ldr], ldr[:hydro_ldr], ldr[:spill_ldr], ldr[:termos_ldr], ldr[:θ_ldr]
    # NA constraints
    fix_nonanticipativity!(ldr, s)
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
dual_ldr_row(s::Sample, row) = dual_ldr_row(s.probs, s.Φ, row)

"""
    basis_function_dual_ldr(s; solver = HiGHS.Optimizer)

Monolithic scenario expansion of the dual linear decision rule over the sample
`s` (`Dual_scenarios.tex`, "Scenario expansion"). Where `basis_function_ldr`
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
function basis_function_dual_ldr(s::Sample; solver = HiGHS.Optimizer)
    ldr = Model(solver)
    set_silent(ldr)

    (; problem, T, inflow, probs, Φ) = s
    n_scen = scenario_count(s)

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
    basis_function_dual_ldr_compact(s; solver = HiGHS.Optimizer)

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
function basis_function_dual_ldr_compact(s::Sample; solver = HiGHS.Optimizer)
    ldr = Model(solver)
    set_silent(ldr)

    (; problem, T, inflow, probs, Φ) = s
    n_scen = scenario_count(s)

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
    dual_ldr_compact_moments(model, s)

`M_x = E[x(ξ)L(ξ)^T]` of a `basis_function_dual_ldr_compact` model built on
the sample `s` and solved: the basis-weighted average of every scenario's
`volume`, `hydro`, `spill`, `termos` and `θ`, one `... x basis_count(s)` matrix
per decision. Unlike
`basis_function_ldr`'s rule, `x(ξ_k)` here is never forced to equal this
combination -- individual scenarios are free to deviate from it, which is the
source of the model's extra slack (see `basis_function_dual_ldr`'s docstring)
-- so `M_x` is only the closest rule-shaped *summary* of the solution, not a
policy the model guarantees is itself feasible.
"""
function dual_ldr_compact_moments(model, s::Sample)
    moment(row) = dual_ldr_row(s, row)
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
    basis_ldr_subproblem(s, k; solver = HiGHS.Optimizer)

The subproblem of scenario `k` of the sample `s`: its dispatch constraints, a
fixed copy of the decision-rule coefficients, and the slacks measuring the
violation of the coupling `decision == Σ_j coefficient_j * Φ[k,j]`.

`feasibility_mode!` and `optimality_mode!` switch the model between the two passes.
"""
function basis_ldr_subproblem(s::Sample, k; solver = HiGHS.Optimizer)
    sub = Model(solver)
    set_silent(sub)
    (; problem, T) = s
    inflow, basis = s.inflow[:,k], s.Φ[k,:]
    K = basis_count(s)

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
    params = add_ldr_coefficients!(sub, s)
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
    basis_function_ldr_benders(s; kwargs...)

Solve the model of `basis_function_ldr` by Benders decomposition: the
decision-rule coefficients stay in a master problem and each scenario is solved
separately, contributing a feasibility or an optimality cut (see the comment
above).  Accepts the same `solver` as `basis_function_ldr`, plus

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
    s::Sample;
    solver = HiGHS.Optimizer,
    max_iterations = 500,
    tol = 1e-6,
    feasibility_tol = 1e-6,
    verbose = false,
)
    probs = s.probs
    n_scen = scenario_count(s)

    # First stage: the decision-rule coefficients, and one epigraph variable per
    # scenario (costs are non-negative, hence the lower bound of zero).
    master = Model(solver)
    set_silent(master)
    x = add_ldr_coefficients!(master, s)
    fix_nonanticipativity!(master, s)
    @variable(master, α[1:n_scen] >= 0)
    @objective(master, Min, sum(probs[k] * α[k] for k in 1:n_scen))

    # Second stage: one subproblem per scenario, built once and re-solved
    subs = [basis_ldr_subproblem(s, k; solver) for k in 1:n_scen]

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

    coefficients = reshape_coefficients(s, something(best, fill(NaN, length(x))))
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
# Two-stage (Benders) decomposition of the dual-LDR model
#
# `basis_function_dual_ldr` keeps every scenario's primal variables x(ξ_k),
# s(ξ_k) and enforces the moment matrices
#
#   M_x = (1/K) Σ_k x(ξ_k) L(ξ_k)^T,   M_s = (1/K) Σ_k s(ξ_k) L(ξ_k)^T
#
# only implicitly, through the aggregate row `A x(ξ_k) + s(ξ_k) - b(ξ_k)`
# weighted by every basis function.  Making M_x and M_s themselves the first
# stage (`Dual_scenarios.tex`, eq. 103-112) turns that around: the balance
# `A M_x + M_s = (1/K) Σ_k b(ξ_k) L(ξ_k)^T` becomes a first-stage constraint on
# the moments directly, and two further things fall out of the structure of
# this model:
#
#   * the cost c(ξ) does not depend on the scenario, so `(1/K) Σ_k c^T x(ξ_k)
#     = c^T M_x[:,1]` -- the first column of M_x, since the constant basis
#     function L₁(ξ) ≡ 1 makes it exactly the mean of x(ξ_k) (`basis_values`).
#     The objective is therefore a fixed linear function of the first-stage
#     variables, and the master needs no epigraph and no optimality cuts;
#   * M_x needs no feasibility subproblem either: with the basis matrix Φ of
#     full column rank on its positive-probability scenarios (checked below,
#     `active_rank`), M_x = X diag(probs) Φ can realize any target matrix from
#     some free X, because right-multiplication by diag(probs) Φ is then
#     surjective. A rank-deficient Φ would need the extra null-space rows
#     Dual_scenarios.tex describes for M_x (not implemented here).
#
# M_s is the one moment that needs a subproblem: not every matrix is the
# moment of *nonnegative* s(ξ_k) (`rows of M_s ∈ cone L(ξ)`), and unlike M_x,
# checking membership of that cone is exactly what `dual_ldr_ms_subproblem`
# does, covering every scenario and every slack of `basis_function_dual_ldr`
# at once, since none of them are otherwise coupled once M_x is set aside.
# `feasibility_mode!` and `subproblem_cut`, already defined above for the
# primal decomposition, apply unchanged: the subproblem's gaps and fixed
# parameters are laid out the same way, just for M_s's coupling instead of the
# LDR coefficients'.
# ---------------------------------------------------------------------------

"""
    add_ms_moments!(model, s)

Declare the second-moment matrix `M_s` of `basis_function_dual_ldr` -- one
`... x K` matrix per slack, laid out exactly like that model's slacks -- and
return them flattened, in the order every other function of this decomposition
uses.

Every basis in this file (`BP2`, `PW3`, `simple_functions`,
and the constant term) evaluates to a nonnegative quantity, so `M_s`, a moment
of the nonnegative slacks against those functions, is itself entrywise
nonnegative -- a cheap necessary condition of `rows of M_s ∈ cone L(ξ)`
(`Dual_scenarios.tex`, eq. 69) declared directly as a bound rather than left to
the feasibility subproblem, which keeps the master from being unbounded before
any cut has narrowed it down. A basis with a negative value somewhere would
need this bound dropped, and the master would then rely on the subproblem
alone (far more cuts, if it converges in reasonable time at all -- the master
is otherwise unbounded in every direction this bound would have blocked).
"""
function add_ms_moments!(model, s::Sample)
    (; problem, T) = s
    K = basis_count(s)
    @variables(model, begin
        volume_hi_ms[1:T,1:K] >= 0
        volume_lo_ms[1:T,1:K] >= 0
        hydro_hi_ms[1:T,1:K] >= 0
        hydro_lo_ms[1:T,1:K] >= 0
        spill_lo_ms[1:T,1:K] >= 0
        termos_hi_ms[1:problem.termo_count,1:T,1:K] >= 0
        termos_lo_ms[1:problem.termo_count,1:T,1:K] >= 0
        θ_lo_ms[1:K] >= 0
        cuts_ms[1:problem.cuts_count,1:K] >= 0
    end)
    return ms_moments(model)
end

"The `M_s` moments of `model` as a flat vector."
ms_moments(model) = [
    vec(model[:volume_hi_ms]); vec(model[:volume_lo_ms])
    vec(model[:hydro_hi_ms]); vec(model[:hydro_lo_ms])
    vec(model[:spill_lo_ms])
    vec(model[:termos_hi_ms]); vec(model[:termos_lo_ms])
    vec(model[:θ_lo_ms])
    vec(model[:cuts_ms])
]

"""
    dual_ldr_ms_subproblem(s; solver = HiGHS.Optimizer)

The feasibility subproblem of the dual-LDR decomposition: every scenario's
slacks `s(ξ_k)`, and the gaps measuring how far the basis-weighted moment of
each one is from a fixed copy of the master's `M_s`. It needs neither `x(ξ_k)`
nor the inflow -- every slack's defining row (a bound or a future-cost cut of
`basis_function_dual_ldr`) is already accounted for at the moment level by the
master's `A M_x + M_s = ...` balance, so all that is left to check here is
whether `M_s` is the moment of *some* nonnegative choice of `s(ξ_k)`.

`feasibility_mode!` (shared with `basis_ldr_subproblem`) is the only mode this
needs: its optimum is the L1 distance from the master's `M_s` to the set of
achievable moments, `0` meaning some nonnegative `s(ξ_k)` realizes it exactly.
"""
function dual_ldr_ms_subproblem(s::Sample; solver = HiGHS.Optimizer)
    sub = Model(solver)
    set_silent(sub)
    (; problem, T, probs, Φ) = s
    n_scen = scenario_count(s)
    K = basis_count(s)

    # Every scenario's slacks, exactly as in the monolithic model
    @variables(sub, begin
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

    # Fixed copy of the master's M_s, refreshed every iteration
    params = add_ms_moments!(sub, s)
    fix.(params, 0.0; force = true)

    # Gap between each M_s entry and the moment its scenario slacks realize,
    # split into positive and negative parts so the feasibility pass is always
    # feasible
    @variables(sub, begin
        volume_hi_gap[1:T,1:K,1:2] >= 0
        volume_lo_gap[1:T,1:K,1:2] >= 0
        hydro_hi_gap[1:T,1:K,1:2] >= 0
        hydro_lo_gap[1:T,1:K,1:2] >= 0
        spill_lo_gap[1:T,1:K,1:2] >= 0
        termos_hi_gap[1:problem.termo_count,1:T,1:K,1:2] >= 0
        termos_lo_gap[1:problem.termo_count,1:T,1:K,1:2] >= 0
        θ_lo_gap[1:K,1:2] >= 0
        cuts_gap[1:problem.cuts_count,1:K,1:2] >= 0
    end)
    for t in 1:T
        @constraint(sub, dual_ldr_row(probs, Φ, volume_hi[t,:]) .- sub[:volume_hi_ms][t,:] .==
            volume_hi_gap[t,:,1] .- volume_hi_gap[t,:,2])
        @constraint(sub, dual_ldr_row(probs, Φ, volume_lo[t,:]) .- sub[:volume_lo_ms][t,:] .==
            volume_lo_gap[t,:,1] .- volume_lo_gap[t,:,2])
        @constraint(sub, dual_ldr_row(probs, Φ, hydro_hi[t,:]) .- sub[:hydro_hi_ms][t,:] .==
            hydro_hi_gap[t,:,1] .- hydro_hi_gap[t,:,2])
        @constraint(sub, dual_ldr_row(probs, Φ, hydro_lo[t,:]) .- sub[:hydro_lo_ms][t,:] .==
            hydro_lo_gap[t,:,1] .- hydro_lo_gap[t,:,2])
        @constraint(sub, dual_ldr_row(probs, Φ, spill_lo[t,:]) .- sub[:spill_lo_ms][t,:] .==
            spill_lo_gap[t,:,1] .- spill_lo_gap[t,:,2])
        for i in 1:problem.termo_count
            @constraint(sub, dual_ldr_row(probs, Φ, termos_hi[i,t,:]) .- sub[:termos_hi_ms][i,t,:] .==
                termos_hi_gap[i,t,:,1] .- termos_hi_gap[i,t,:,2])
            @constraint(sub, dual_ldr_row(probs, Φ, termos_lo[i,t,:]) .- sub[:termos_lo_ms][i,t,:] .==
                termos_lo_gap[i,t,:,1] .- termos_lo_gap[i,t,:,2])
        end
    end
    @constraint(sub, dual_ldr_row(probs, Φ, θ_lo) .- sub[:θ_lo_ms] .== θ_lo_gap[:,1] .- θ_lo_gap[:,2])
    for i in 1:problem.cuts_count
        @constraint(sub, dual_ldr_row(probs, Φ, cuts_slack[i,:]) .- sub[:cuts_ms][i,:] .==
            cuts_gap[i,:,1] .- cuts_gap[i,:,2])
    end

    @expression(sub, infeasibility,
        sum(volume_hi_gap) + sum(volume_lo_gap) + sum(hydro_hi_gap) + sum(hydro_lo_gap) +
        sum(spill_lo_gap) + sum(termos_hi_gap) + sum(termos_lo_gap) + sum(θ_lo_gap) + sum(cuts_gap))
    sub.ext[:gaps] = [
        vec(volume_hi_gap); vec(volume_lo_gap); vec(hydro_hi_gap); vec(hydro_lo_gap);
        vec(spill_lo_gap); vec(termos_hi_gap); vec(termos_lo_gap); vec(θ_lo_gap); vec(cuts_gap)
    ]
    sub.ext[:params] = params
    return sub
end

"""
    basis_function_dual_ldr_benders(s; kwargs...)

Solve the model of `basis_function_dual_ldr` by Benders decomposition: the
moment matrices `M_x` and `M_s` stay in a master problem, and a single
feasibility subproblem (`dual_ldr_ms_subproblem`) checks whether the master's
`M_s` is achievable by nonnegative scenario slacks, contributing a feasibility
cut when it is not (see the comment above). Accepts the same `solver` as
`basis_function_ldr`, plus

  * `max_iterations`, `feasibility_tol`: stop when the subproblem's violation
    is within tolerance, or after so many iterations;
  * `verbose`: print the bound and the violation of every iteration.

Unlike the primal decomposition, this one never produces a rule -- `M_x` is a
*moment*, not a decision, so there is no first-stage policy to report -- and
its master's objective at every iteration is already a valid lower bound on
`basis_function_dual_ldr`'s value, exact as soon as the subproblem reports no
violation. Returns `(; Obj, dt)`, matching the lower bound to the other
methods' `Obj` field, plus whether the loop `converged`, its `iterations`, the
number of feasibility `cuts`, and the `master` and `subproblem` models.

Before enough cuts pin down `M_s`, the master is only the balance equation on
an otherwise free `M_s` and is typically unbounded (`DUAL_INFEASIBLE`) rather
than merely loose -- unlike the monolithic model, where every scenario's own
slacks are individually sign-restricted. Since the achievable-`M_s` cone is
positively homogeneous, the unbounded ray HiGHS reports in that case is itself
a valid point to feed the subproblem, and a positive violation there yields a
homogeneous cut that removes the whole ray, not just one point on it; a zero
violation instead means the ray is genuine, and the dual LDR really is
unbounded below for this basis (see `basis_function_dual_ldr`'s docstring).
"""
function basis_function_dual_ldr_benders(
    s::Sample;
    solver = HiGHS.Optimizer,
    max_iterations = 500,
    feasibility_tol = 1e-6,
    verbose = false,
)
    (; problem, T, inflow, probs, Φ, φ̄) = s
    K = basis_count(s)
    r = active_rank(Φ, probs)
    r == K || @warn "the basis matrix Φ is rank-deficient on the positive-probability " *
        "scenarios ($r < $K): M_x is then under-determined by the balance constraints " *
        "alone, and would need the extra null-space rows Dual_scenarios.tex describes " *
        "(not implemented here)."

    # First stage: the moment matrices. Any M_x is achievable by some free
    # x(ξ_k) (see the comment above), so it needs no subproblem; M_s does, and
    # is the only coupling the subproblem below is given.
    master = Model(solver)
    set_silent(master)
    @variables(master, begin
        volume_mx[1:T,1:K]
        hydro_mx[1:T,1:K]
        spill_mx[1:T,1:K]
        termos_mx[1:problem.termo_count,1:T,1:K]
        θ_mx[1:K]
    end)
    Ms = add_ms_moments!(master, s)
    volume_hi_ms, volume_lo_ms, hydro_hi_ms, hydro_lo_ms, spill_lo_ms, termos_hi_ms, termos_lo_ms, θ_lo_ms, cuts_ms =
        master[:volume_hi_ms], master[:volume_lo_ms], master[:hydro_hi_ms], master[:hydro_lo_ms],
        master[:spill_lo_ms], master[:termos_hi_ms], master[:termos_lo_ms], master[:θ_lo_ms], master[:cuts_ms]

    for t in 1:T
        previous = t == 1 ? problem.initial_volume .* φ̄ : volume_mx[t-1,:]
        @constraint(master, volume_mx[t,:] .- previous .- dual_ldr_row(probs, Φ, inflow[t,:]) .+
            hydro_mx[t,:] .+ spill_mx[t,:] .== 0)
        @constraint(master, hydro_mx[t,:] .+ sum(termos_mx[i,t,:] for i in 1:problem.termo_count) .-
            problem.stage_demands[t] .* φ̄ .== 0)
        @constraint(master, volume_mx[t,:] .+ volume_hi_ms[t,:] .- problem.hydro_capacity .* φ̄ .== 0)
        @constraint(master, volume_lo_ms[t,:] .- volume_mx[t,:] .== 0)
        @constraint(master, hydro_mx[t,:] .+ hydro_hi_ms[t,:] .- problem.hydro_generation .* φ̄ .== 0)
        @constraint(master, hydro_lo_ms[t,:] .- hydro_mx[t,:] .== 0)
        @constraint(master, spill_lo_ms[t,:] .- spill_mx[t,:] .== 0)
        for i in 1:problem.termo_count
            @constraint(master, termos_mx[i,t,:] .+ termos_hi_ms[i,t,:] .-
                problem.termo_generations[i] .* φ̄ .== 0)
            @constraint(master, termos_lo_ms[i,t,:] .- termos_mx[i,t,:] .== 0)
        end
    end
    @constraint(master, θ_lo_ms .- θ_mx .== 0)
    for i in 1:problem.cuts_count
        @constraint(master, cuts_ms[i,:] .+ problem.cuts_coeff[i] .* volume_mx[T,:] .- θ_mx .+
            problem.cuts_inter[i] .* φ̄ .== 0)
    end
    # Costs do not depend on the scenario, so the objective only needs the
    # mean of x(ξ_k) -- the first column of M_x, since L₁(ξ) ≡ 1 (see the
    # comment above and `basis_values`).
    @objective(master, Min,
        sum(problem.termo_costs[i] * termos_mx[i,t,1] for i in 1:problem.termo_count, t in 1:T) +
        θ_mx[1])

    # Second stage: one feasibility subproblem, covering every scenario
    sub = dual_ldr_ms_subproblem(s; solver)

    lower = -Inf
    converged, iterations, cuts, dt = false, 0, 0, 0.0
    for iteration in 1:max_iterations
        iterations = iteration
        dt += @elapsed optimize!(master)
        status = termination_status(master)
        if status == INFEASIBLE
            # Feasibility cuts have exhausted every M_s: the basis cannot
            # represent a feasible dual multiplier at all.
            verbose && println("master problem is INFEASIBLE")
            lower, converged = Inf, true
            break
        elseif status ∉ (OPTIMAL, DUAL_INFEASIBLE)
            verbose && println("master problem is $status")
            break
        end
        # Before enough cuts restrict M_s, the master is only the balance
        # equation on a totally free M_s and is typically unbounded: HiGHS then
        # reports an unbounded *ray* instead of an optimum. Since the cone of
        # achievable M_s is positively homogeneous, checking that ray against
        # the very same subproblem is meaningful, and `subproblem_cut` turns a
        # positive violation into a homogeneous cut that rules out the whole
        # ray, not just this one point.
        M̄s = value.(Ms)

        fix.(sub.ext[:params], M̄s; force = true)
        feasibility_mode!(sub)
        dt += @elapsed optimize!(sub)
        termination_status(sub) == OPTIMAL ||
            error("feasibility subproblem is $(termination_status(sub))")
        violation = objective_value(sub)

        if status == OPTIMAL
            lower = objective_value(master)
        end
        verbose && println("iteration $iteration: " *
            (status == OPTIMAL ? "bound $(round(lower; digits=2))" : "master unbounded") *
            ", violation $(round(violation; digits=6)), $cuts cuts so far")
        if violation <= feasibility_tol
            converged = true
            if status == DUAL_INFEASIBLE
                # A feasible ray: this basis leaves the dual LDR genuinely
                # unbounded below (see `basis_function_dual_ldr`'s docstring),
                # not merely under-cut.
                lower = -Inf
            end
            break
        end
        @constraint(master, 0 >= subproblem_cut(violation, sub, Ms, M̄s))
        cuts += 1
    end

    return (;
        Obj = lower,
        dt,
        converged,
        iterations,
        cuts,
        master,
        subproblem = sub,
    )
end

# ---------------------------------------------------------------------------
# Two-stage (Benders) decomposition of the compact dual-LDR model
#
# `basis_function_dual_ldr_compact` drops the slack (and its relaxed
# multiplier) that `basis_function_dual_ldr` used to add for every bound `x`
# already carries as its own sign restriction, keeping `volume`, `hydro`,
# `spill`, `termos` and `θ` bounded exactly as in `basis_function_ldr`. That
# changes what a decomposition of it needs to check:
#
#   * water balance, demand balance and the future-cost cuts are still pure
#     algebraic relations *between* moments (`A M_x + M_s = ...`), so they stay
#     exact linear constraints on the master's moments, no subproblem needed;
#   * but now `M_x`'s own rows are *not* trivially achievable the way they
#     were when `x` was free -- `volume`, say, is bounded pointwise, so its
#     moment is restricted to the (nontrivial) polytope of moments of
#     bounded sequences, not all of R^K. Checking that membership is the
#     subproblem's job here, one family at a time (they never interact with
#     each other inside it, only through the master's own equations) -- and it
#     covers `cuts_slack`'s cone-membership too, the same check
#     `dual_ldr_ms_subproblem` made, since that is still a genuine slack.
#
# Because every basis here is nonnegative-valued, the same trick as
# `add_ms_moments!` gives cheap, necessary bounds on every M_x row directly --
# and here they cut both ways (`0 <= M_x <= bound * φ̄`) for the once-bounded
# decisions, so the master's objective (a sum of two such bounded terms) can
# never be unbounded below, unlike the full-slack decomposition's master.
# ---------------------------------------------------------------------------

"""
    add_compact_moments!(model, s)

Declare the moment matrices of `basis_function_dual_ldr_compact`'s primal
scenario variables -- `volume`, `hydro`, `spill`, `termos`, `θ`, and the one
genuine slack `cuts_slack` -- each `... x K`, and return them as a named
tuple.
"""
function add_compact_moments!(model, s::Sample)
    (; problem, T) = s
    K = basis_count(s)
    @variables(model, begin
        volume_mx[1:T,1:K]
        hydro_mx[1:T,1:K]
        spill_mx[1:T,1:K]
        termos_mx[1:problem.termo_count,1:T,1:K]
        θ_mx[1:K]
        cuts_mx[1:problem.cuts_count,1:K]
    end)
    return (;
        volume_mx = model[:volume_mx], hydro_mx = model[:hydro_mx], spill_mx = model[:spill_mx],
        termos_mx = model[:termos_mx], θ_mx = model[:θ_mx], cuts_mx = model[:cuts_mx],
    )
end

"The moments of `add_compact_moments!` as a flat vector, in the order the feasibility subproblem below matches them."
compact_moments_flat(mts) =
    [vec(mts.volume_mx); vec(mts.hydro_mx); vec(mts.spill_mx); vec(mts.termos_mx); vec(mts.θ_mx); vec(mts.cuts_mx)]

"""
    add_compact_moment_bounds!(model, mts, s)

The cheap, necessary bounds every row of `mts` obeys because every basis
function in this file is nonnegative-valued: the moment of a variable that is
pointwise within `[lo,hi]` is itself within `[lo,hi] * φ̄` entrywise (a
nonnegative combination of numbers in `[lo,hi]`, weighted by the nonnegative
`Φ[k,:]`, stays in `[lo,hi] * (weights' sum)`). `volume`, `hydro` and `termos`
get both bounds this way; `spill`, `θ` and `cuts_mx` only the lower one, since
those are unbounded above already in `basis_function_dual_ldr_compact`.

Declared directly rather than left to the feasibility subproblem, these are
what let the master's objective (`Σ cost·termos_mx[:,:,1] + θ_mx[1]`, both
terms now bounded below) never be unbounded, unlike the full-slack
decomposition's master before any cut narrowed it down.
"""
function add_compact_moment_bounds!(model, mts, s::Sample)
    (; problem, φ̄) = s
    @constraint(model, mts.volume_mx .>= 0)
    @constraint(model, mts.volume_mx .<= problem.hydro_capacity .* φ̄')
    @constraint(model, mts.hydro_mx .>= 0)
    @constraint(model, mts.hydro_mx .<= problem.hydro_generation .* φ̄')
    @constraint(model, mts.spill_mx .>= 0)
    for i in 1:problem.termo_count
        @constraint(model, mts.termos_mx[i,:,:] .>= 0)
        @constraint(model, mts.termos_mx[i,:,:] .<= problem.termo_generations[i] .* φ̄')
    end
    @constraint(model, mts.θ_mx .>= 0)
    @constraint(model, mts.cuts_mx .>= 0)
    return model
end

"""
    compact_moments_subproblem(s; solver = HiGHS.Optimizer)

The feasibility subproblem of the compact dual-LDR decomposition: every
scenario's primal variables -- `volume`, `hydro`, `spill`, `termos` and `θ`,
bounded exactly as in `basis_function_ldr`, plus the slack `cuts_slack >= 0`
-- and the gaps measuring how far each one's basis-weighted moment is from a
fixed copy of the master's moments.

Unlike `dual_ldr_ms_subproblem`, this has to check `volume`, `hydro`, `spill`,
`termos` and `θ` too, since they now carry their own bounds directly (see
`basis_function_dual_ldr_compact`'s docstring) instead of delegating to a
separately-relaxed slack -- but water balance, demand balance and the
future-cost cuts stay out of it entirely: those are exact algebraic relations
between moments in the master already, needing no realization of their own.
Each family here is independent of the others, so nothing couples them beyond
sharing the same L1 objective; `feasibility_mode!` (shared with the two other
decompositions) is the only mode this needs.
"""
function compact_moments_subproblem(s::Sample; solver = HiGHS.Optimizer)
    sub = Model(solver)
    set_silent(sub)
    (; problem, T, probs, Φ) = s
    n_scen = scenario_count(s)
    K = basis_count(s)

    # Every scenario's primal variables, exactly as in basis_function_ldr / _compact
    @variables(sub, begin
        0 <= volume[1:T,1:n_scen] <= problem.hydro_capacity
        0 <= hydro[1:T,1:n_scen] <= problem.hydro_generation
        0 <= spill[1:T,1:n_scen]
        0 <= termos[i=1:problem.termo_count,1:T,1:n_scen] <= problem.termo_generations[i]
        θ[1:n_scen] >= 0
        cuts_slack[1:problem.cuts_count,1:n_scen] >= 0
    end)

    # Fixed copy of the master's moments, refreshed every iteration
    mts = add_compact_moments!(sub, s)
    params = compact_moments_flat(mts)
    fix.(params, 0.0; force = true)

    # Gap between each moment and the target, split into positive and negative
    # parts so the feasibility pass is always feasible
    @variables(sub, begin
        volume_gap[1:T,1:K,1:2] >= 0
        hydro_gap[1:T,1:K,1:2] >= 0
        spill_gap[1:T,1:K,1:2] >= 0
        termos_gap[1:problem.termo_count,1:T,1:K,1:2] >= 0
        θ_gap[1:K,1:2] >= 0
        cuts_gap[1:problem.cuts_count,1:K,1:2] >= 0
    end)
    for t in 1:T
        @constraint(sub, dual_ldr_row(probs, Φ, volume[t,:]) .- mts.volume_mx[t,:] .==
            volume_gap[t,:,1] .- volume_gap[t,:,2])
        @constraint(sub, dual_ldr_row(probs, Φ, hydro[t,:]) .- mts.hydro_mx[t,:] .==
            hydro_gap[t,:,1] .- hydro_gap[t,:,2])
        @constraint(sub, dual_ldr_row(probs, Φ, spill[t,:]) .- mts.spill_mx[t,:] .==
            spill_gap[t,:,1] .- spill_gap[t,:,2])
        for i in 1:problem.termo_count
            @constraint(sub, dual_ldr_row(probs, Φ, termos[i,t,:]) .- mts.termos_mx[i,t,:] .==
                termos_gap[i,t,:,1] .- termos_gap[i,t,:,2])
        end
    end
    @constraint(sub, dual_ldr_row(probs, Φ, θ) .- mts.θ_mx .== θ_gap[:,1] .- θ_gap[:,2])
    for i in 1:problem.cuts_count
        @constraint(sub, dual_ldr_row(probs, Φ, cuts_slack[i,:]) .- mts.cuts_mx[i,:] .==
            cuts_gap[i,:,1] .- cuts_gap[i,:,2])
    end

    @expression(sub, infeasibility,
        sum(volume_gap) + sum(hydro_gap) + sum(spill_gap) + sum(termos_gap) + sum(θ_gap) + sum(cuts_gap))
    sub.ext[:gaps] = [
        vec(volume_gap); vec(hydro_gap); vec(spill_gap); vec(termos_gap); vec(θ_gap); vec(cuts_gap)
    ]
    sub.ext[:params] = params
    return sub
end

"""
    basis_function_dual_ldr_compact_benders(s; kwargs...)

Solve the model of `basis_function_dual_ldr_compact` by Benders decomposition:
the moment matrices stay in a master problem (linked by the exact algebraic
relations of water balance, demand balance and the future-cost cuts), and a
single feasibility subproblem (`compact_moments_subproblem`) checks whether
they are jointly achievable by the primal scenario variables, contributing a
feasibility cut when they are not. Accepts the same `solver` as
`basis_function_ldr`, plus

  * `max_iterations`, `feasibility_tol`: stop when the subproblem's violation
    is within tolerance, or after so many iterations;
  * `verbose`: print the bound and the violation of every iteration.

As with `basis_function_dual_ldr_benders`, `M_x` is a moment, not a decision,
so there is no first-stage policy to report; the master's objective at every
iteration is already a valid lower bound on `basis_function_dual_ldr_compact`'s
value, exact as soon as the subproblem reports no violation. Unlike that
decomposition, though, this master's objective is bounded below from the
start (see the comment above), so it should not need unbounded-ray handling.
Returns `(; Obj, dt)`, plus whether the loop `converged`, its `iterations`,
the number of feasibility `cuts`, and the `master` and `subproblem` models.
"""
function basis_function_dual_ldr_compact_benders(
    s::Sample;
    solver = HiGHS.Optimizer,
    max_iterations = 500,
    feasibility_tol = 1e-6,
    verbose = false,
)
    (; problem, T, inflow, probs, Φ, φ̄) = s

    # First stage: the moment matrices, bounded as described above
    master = Model(solver)
    set_silent(master)
    mts = add_compact_moments!(master, s)
    add_compact_moment_bounds!(master, mts, s)
    Mflat = compact_moments_flat(mts)
    volume_mx, hydro_mx, spill_mx, termos_mx, θ_mx, cuts_mx =
        mts.volume_mx, mts.hydro_mx, mts.spill_mx, mts.termos_mx, mts.θ_mx, mts.cuts_mx

    for t in 1:T
        previous = t == 1 ? problem.initial_volume .* φ̄ : volume_mx[t-1,:]
        @constraint(master, volume_mx[t,:] .- previous .- dual_ldr_row(probs, Φ, inflow[t,:]) .+
            hydro_mx[t,:] .+ spill_mx[t,:] .== 0)
        @constraint(master, hydro_mx[t,:] .+ sum(termos_mx[i,t,:] for i in 1:problem.termo_count) .-
            problem.stage_demands[t] .* φ̄ .== 0)
    end
    for i in 1:problem.cuts_count
        @constraint(master, cuts_mx[i,:] .+ problem.cuts_coeff[i] .* volume_mx[T,:] .- θ_mx .+
            problem.cuts_inter[i] .* φ̄ .== 0)
    end
    # Costs do not depend on the scenario, so the objective only needs the
    # mean of x(ξ_k) -- the first column of each moment, since L₁(ξ) ≡ 1.
    @objective(master, Min,
        sum(problem.termo_costs[i] * termos_mx[i,t,1] for i in 1:problem.termo_count, t in 1:T) +
        θ_mx[1])

    # Second stage: one feasibility subproblem, covering every scenario
    sub = compact_moments_subproblem(s; solver)

    lower = -Inf
    converged, iterations, cuts, dt = false, 0, 0, 0.0
    for iteration in 1:max_iterations
        iterations = iteration
        dt += @elapsed optimize!(master)
        status = termination_status(master)
        if status == INFEASIBLE
            verbose && println("master problem is INFEASIBLE")
            lower, converged = Inf, true
            break
        elseif status ∉ (OPTIMAL, DUAL_INFEASIBLE)
            verbose && println("master problem is $status")
            break
        end
        M̄ = value.(Mflat)

        fix.(sub.ext[:params], M̄; force = true)
        feasibility_mode!(sub)
        dt += @elapsed optimize!(sub)
        termination_status(sub) == OPTIMAL ||
            error("feasibility subproblem is $(termination_status(sub))")
        violation = objective_value(sub)

        if status == OPTIMAL
            lower = objective_value(master)
        end
        verbose && println("iteration $iteration: " *
            (status == OPTIMAL ? "bound $(round(lower; digits=2))" : "master unbounded") *
            ", violation $(round(violation; digits=6)), $cuts cuts so far")
        if violation <= feasibility_tol
            converged = true
            status == DUAL_INFEASIBLE && (lower = -Inf)
            break
        end
        @constraint(master, 0 >= subproblem_cut(violation, sub, Mflat, M̄))
        cuts += 1
    end

    return (;
        Obj = lower,
        dt,
        converged,
        iterations,
        cuts,
        master,
        subproblem = sub,
    )
end

# ---------------------------------------------------------------------------
# Dual-side (Lagrangian) solution of the compact dual LDR
#
# Both Benders decompositions above keep the moment matrices in the master and
# ask a subproblem whether they are achievable.  That question is hopeless as a
# cutting-plane problem: the achievable moments of a single family -- say
# `hydro`, with `0 <= hydro(ξ_k) <= hydro_generation` -- are the zonotope
#
#   { Σ_k p_k h_k Φ[k,:] : h_k ∈ [0, hydro_generation] } ⊂ R^K,
#
# a Minkowski sum of `n_scen` segments, whose facets number
# O(binomial(n_scen-1, K-1)) -- around 10^9 of them for `n_scen = 100`, `K = 7`.
# `add_compact_moment_bounds!` is exactly its bounding box, and one aggregated
# L1 cut per iteration never gets appreciably past that box, which is why the
# bound of those experiments sits at the expected-value solution however long
# they run.
#
# A zonotope has no usable facet description, but it does have a closed-form
# support function, so the same information is trivial to exploit from the
# *dual* side.  Dualizing the only coupling of the scenario expansion -- the
# aggregate rows of `dual_ldr_row` -- with the multiplier matrix `Z` of the
# decision rule itself (`Dual_scenarios.tex`, eq. 35) gives
#
#   dual LDR value = max_Z Σ_k p_k φ_k(Z L(ξ_k)),
#   φ_k(z) = min_{y_k ∈ Y_k} [ c' x(ξ_k) + z' (A x(ξ_k) + s(ξ_k) - b(ξ_k)) ],
#
# with `Y_k` the box of `basis_function_dual_ldr_compact` (every decision at its
# original bounds, plus `cuts_slack >= 0`).  Each `φ_k` is concave and piecewise
# linear in `z`, and because `Y_k` is a *box* its value and its minimizer are
# pure arithmetic (`dual_ldr_scenario_lagrangian`): every variable goes to the
# bound its reduced cost asks for, and a negative reduced cost on one of the
# variables that is unbounded above -- `spill`, `θ`, `cuts_slack` -- means
# φ_k = -∞, i.e. `z` is dual-infeasible for that scenario.  No LP is solved per
# scenario, the moment matrices never appear as variables, and
#
#   ∂/∂Z Σ_k p_k φ_k = Σ_k p_k (A x(ξ_k) + s(ξ_k) - b(ξ_k)) L(ξ_k)',
#
# the moment residual: what the Benders master carried as first-stage variables
# is here just a supergradient.  Two practical consequences:
#
#   * every `Z` is a feasible dual decision rule, so Σ_k p_k φ_k(Z L(ξ_k)) is a
#     valid lower bound at *every* iterate -- unlike the moment master, whose
#     value is only the relaxation's until its cuts have converged;
#   * separation is a single matrix product plus O(n_scen · T) arithmetic, which
#     is what makes sample sizes far past the monolithic model workable.
#
# The master is a Kelley cutting-plane model of that concave maximum: one
# epigraph variable per scenario (`cut_mode = :multi`, so every iteration
# contributes `n_scen` cuts) or a single aggregated one, plus an L∞ trust region
# around the best `Z` so far, which is what keeps the early iterations -- where
# the cut model is still unbounded above -- meaningful.
# ---------------------------------------------------------------------------

"The number of rows the compact dual LDR couples across scenarios: water balance and demand per stage, then the future-cost cuts."
coupling_row_count(problem::HydroProblem, T) = 2T + problem.cuts_count
coupling_row_count(s::Sample) = coupling_row_count(s.problem, s.T)

"Row index of the stage-`t` water balance among the coupling rows."
balance_row(T, t) = t
"Row index of the stage-`t` demand balance among the coupling rows."
demand_row(T, t) = T + t
"Row index of the `i`-th future-cost cut among the coupling rows."
cut_row(T, i) = 2T + i

"""
    dual_ldr_scenario_lagrangian(s, k, z; ray_tol = 1e-9)

`φ_k(z)`, the Lagrangian of the dispatch problem of scenario `k` of the sample
`s` at the multiplier `z = Z L(ξ_k)` (ordered as `balance_row`, `demand_row`,
`cut_row`), and the minimizer attaining it.

Since the scenario's variables are only box-constrained once the coupling rows
are priced out, this needs no solver: each variable's reduced cost sends it to
one of its own bounds, and every lower bound here is zero, so a variable stays
at zero unless its reduced cost is negative.  The variables that are unbounded
above -- `spill`, `θ` and `cuts_slack` -- make `φ_k = -∞` when their reduced
cost is negative, which is precisely dual infeasibility of `z` for this
scenario; the `rays` field then reports which condition failed (`w(ξ_k) >= 0`,
`Σ_i μ_i(ξ_k) <= 1`, `μ_i(ξ_k) >= 0`), each a valid linear restriction on `Z`
since any `Z` violating it has Lagrangian `-∞` and can never be optimal.

Those conditions are tested up to `ray_tol` (relative to the size of `z`), and
that tolerance is not cosmetic: a cutting-plane master maximizing over `Z`
*pushes* against them, so its iterates land on the boundary and land there with
rounding error.  Testing them exactly declares every such iterate
dual-infeasible and spends the whole run re-cutting a ray it is already on --
at the price that the value reported is only a lower bound up to the same
tolerance, exactly as any LP solver's dual feasibility is.

Returns `(; value, cost, residual, rays, ...)` where `residual` is the
scenario's row residual `A x + s - b` at the minimizer -- the supergradient
this scenario contributes -- and `value = cost + z'residual`.
"""
function dual_ldr_scenario_lagrangian(s::Sample, k, z; ray_tol = 1e-9)
    (; problem, T) = s
    inflow = view(s.inflow, :, k)
    nc = problem.cuts_count
    w = z[1:T]                      # water balance
    y = z[T+1:2T]                   # demand balance
    μ = z[2T+1:2T+nc]               # future-cost cuts
    Σμ = sum(μ; init = 0.0)

    # Dual feasibility of the variables that are unbounded above
    slack = ray_tol * max(1.0, maximum(abs, z))
    rays = Tuple{Symbol,Int}[]
    for t in 1:T
        w[t] < -slack && push!(rays, (:spill, t))
    end
    for i in 1:nc
        μ[i] < -slack && push!(rays, (:cuts_slack, i))
    end
    Σμ > 1 + slack && push!(rays, (:θ, 0))
    isempty(rays) || return (; value = -Inf, cost = NaN, residual = Float64[], rays)

    # Every lower bound is zero, so a variable moves only if its reduced cost is
    # negative, and then all the way to its upper bound
    at_bound(coefficient, upper) = coefficient < 0 ? upper : 0.0
    volume = [at_bound(
                  t < T ? w[t] - w[t+1] :
                          w[T] + sum(μ[i] * problem.cuts_coeff[i] for i in 1:nc; init = 0.0),
                  problem.hydro_capacity) for t in 1:T]
    hydro = [at_bound(w[t] + y[t], problem.hydro_generation) for t in 1:T]
    spill = zeros(T)        # w >= 0 here, so spill stays at its lower bound
    termos = [at_bound(problem.termo_costs[i] + y[t], problem.termo_generations[i])
              for i in 1:problem.termo_count, t in 1:T]
    θ = 0.0                 # Σ_i μ_i <= 1 here
    cuts_slack = zeros(nc)  # μ >= 0 here

    residual = zeros(coupling_row_count(problem, T))
    for t in 1:T
        previous = t == 1 ? problem.initial_volume : volume[t-1]
        residual[balance_row(T,t)] =
            volume[t] - previous - inflow[t] + hydro[t] + spill[t]
        residual[demand_row(T,t)] =
            hydro[t] + sum(termos[:,t]) - problem.stage_demands[t]
    end
    for i in 1:nc
        residual[cut_row(T,i)] =
            cuts_slack[i] + problem.cuts_coeff[i] * volume[T] - θ + problem.cuts_inter[i]
    end
    cost = sum(problem.termo_costs[i] * termos[i,t]
               for i in 1:problem.termo_count, t in 1:T) + θ
    return (; value = cost + dot(z, residual), cost, residual, rays,
              volume, hydro, spill, termos, θ, cuts_slack)
end

"""
    dual_ldr_lagrangian(s, Z; ray_tol = 1e-9)

The Lagrangian `Σ_k p_k φ_k(Z L(ξ_k))` of the whole sample `s` at the dual
decision rule `Z` -- a valid lower bound on the dual LDR value, hence on the
sampled primal, for *any* `Z` -- together with every scenario's evaluation (see
`dual_ldr_scenario_lagrangian`).  `-Inf` as soon as one scenario finds `Z`
dual-infeasible.
"""
function dual_ldr_lagrangian(s::Sample, Z; ray_tol = 1e-9)
    probs = s.probs
    n_scen = scenario_count(s)
    zs = Z * s.Φ'   # column k is z(ξ_k) = Z L(ξ_k)
    scenarios = [dual_ldr_scenario_lagrangian(s, k, view(zs,:,k); ray_tol)
                 for k in 1:n_scen]
    value = all(scenario -> isfinite(scenario.value), scenarios) ?
        sum(probs[k] * scenarios[k].value for k in 1:n_scen) : -Inf
    return (; value, scenarios, zs)
end

"""
    lagrangian_moment_residual(s, scenarios)

`Σ_k p_k (A x(ξ_k) + s(ξ_k) - b(ξ_k)) L(ξ_k)'` at the Lagrangian minimizers
`scenarios` (the field of the same name of `dual_ldr_lagrangian`): the
supergradient of `dual_ldr_lagrangian` with respect to `Z`, which is exactly the
moment residual the Benders masters carried as variables.  Requires every
scenario to be dual-feasible.
"""
lagrangian_moment_residual(s::Sample, scenarios) =
    sum(s.probs[k] .* scenarios[k].residual * s.Φ[k,:]' for k in eachindex(s.probs))

"""
    basis_function_dual_ldr_lagrangian(s; kwargs...)

Solve the model of `basis_function_dual_ldr_compact` from the dual side, by
cutting planes on the dual decision rule `Z` itself (see the comment above).
Accepts the same `solver` as `basis_function_ldr`, plus

  * `cut_mode`: `:multi` keeps one epigraph variable per scenario, so every
    iteration adds `n_scen` cuts and the model of the Lagrangian is sharp;
    `:single` keeps one, aggregating each iteration into a single cut -- a much
    smaller master, at the cost of many more iterations.  A `:single` iteration
    where some scenario is dual-infeasible contributes only ray constraints,
    since the aggregate cut needs a finite value from every scenario;
  * `radius`, `min_radius`, `max_radius`: the L∞ trust region around the best
    `Z` so far, scaled per basis function by `maximum(abs, Φ[:,j])` so that it
    bounds each basis function's *contribution* to `z(ξ)`.  It doubles on an
    improving iterate and halves on a null one; the loop stops once it is below
    `min_radius`, having run out of progress to make;
  * `ray_mode`: `:upfront` states every scenario's dual-feasibility conditions
    from the start -- only `n_scen · (T + cuts_count + 1)` rows, and worth it,
    since then *every* iterate has a finite Lagrangian and so reports a bound;
    `:lazy` generates them as scenarios reject the iterate, for samples large
    enough that stating them all is the expensive part;
  * `max_cuts`: how many cuts to add per iteration under `:multi`, keeping the
    scenarios that account for most of the master's optimism.  The epigraph of
    an outer model is above the Lagrangian almost everywhere, so without this
    cap "only the violated ones" is no cap at all;
  * `max_total_cuts`: how many cuts the master may hold before the ones priced
    at zero are dropped again (`prune_cuts!`).  This is what keeps the late
    master solves from degenerating; `Inf` to keep every cut;
  * `max_rays`: how many dual-infeasibility constraints to add per iteration
    under `:lazy` (rounding still lets a few through under `:upfront`);
  * `master_attributes`: solver attributes for the master, applied
    best-effort.  The default asks HiGHS for the *primal* simplex, which on
    this shape of LP is the difference between 0.05 s and 30 s a solve (see the
    code);
  * `master_time_limit`: seconds per master solve, `nothing` for none.  A
    stalled master still yields a point to evaluate, and the Lagrangian there
    is a valid bound regardless; only `ObjBound` needs the master solved;
  * `stall_limit`: give up after this many iterations without an improving
    bound.  This is the exit that matters when masters keep timing out, since
    `ObjBound` then never updates and the gap can never close;
  * `max_iterations`, `tol`, `ray_tol`, `verbose`.

Returns `Obj`, the best Lagrangian value found -- a *certified* lower bound on
the sampled primal, valid at every iteration rather than only at convergence --
and `ObjBound`, the cutting-plane master's value, an upper bound on the dual LDR
value whenever the trust region did not restrict it (see `restricting`).  The
two coincide at convergence.  Also returns the multiplier
matrix `Z` attaining `Obj`, whether the loop `converged`, its `iterations`, the
number of `cuts` and `rays` generated, `dt`, and the `master` model.
"""
function basis_function_dual_ldr_lagrangian(
    s::Sample;
    solver = HiGHS.Optimizer,
    cut_mode = :multi,
    ray_mode = :upfront,
    max_cuts = 1_000,
    max_total_cuts = nothing,
    max_iterations = 200,
    radius = nothing,
    min_radius = 1e-3,
    max_radius = nothing,
    max_rays = 100,
    master_attributes = ("simplex_strategy" => 4,),
    master_time_limit = 10.0,
    stall_limit = 20,
    tol = 1e-6,
    ray_tol = 1e-9,
    verbose = false,
)
    cut_mode in (:multi, :single) ||
        throw(ArgumentError("cut_mode must be :multi or :single"))
    ray_mode in (:upfront, :lazy) ||
        throw(ArgumentError("ray_mode must be :upfront or :lazy"))

    (; problem, T, probs, Φ) = s
    n_scen = scenario_count(s)
    K = basis_count(s)
    nrows = coupling_row_count(s)

    # The dual decision rule, one multiplier per coupling row and basis function
    master = Model(solver)
    set_silent(master)
    @variable(master, Z[1:nrows, 1:K])
    n_epi = cut_mode === :multi ? n_scen : 1
    @variable(master, epi[1:n_epi])
    # Twice the one cut per epigraph the master needs to stay bounded, plus room
    # to work in (see `prune_cuts!`)
    max_total_cuts = something(max_total_cuts, 2 * n_epi + 500)
    @objective(master, Max, cut_mode === :multi ?
        sum(probs[k] * epi[k] for k in 1:n_scen) : epi[1])

    # A basis function with large values needs proportionally smaller
    # multipliers to move z(ξ) by the same amount
    scale = [max(1.0, maximum(abs, Φ[:,j])) for j in 1:K]

    # Every cut in the master, with the scenario and the vertex it came from, so
    # that the ones doing no work can be dropped again (`prune_cuts!`)
    cut_rows = Tuple{Any,Int,UInt64}[]

    "The cut a scenario's Lagrangian minimizer gives: φ_k(z) <= cost + z'residual for every z."
    function scenario_cut(k, scenario, key)
        row = @constraint(master, epi[k] <=
            scenario.cost + sum(scenario.residual[r] * Φ[k,j] * Z[r,j]
                                for r in 1:nrows, j in 1:K if scenario.residual[r] != 0))
        push!(cut_rows, (row, k, key))
        return row
    end

    # The cuts already in the master, per scenario.  A box minimizer is a
    # vertex, and the same vertex comes back again and again over the
    # iterations -- re-adding its row each time leaves the master with dozens of
    # duplicates of the same constraint, which is how a two-thousand-row LP
    # degenerates into one the simplex stalls on for minutes.  The values here
    # are exact bound arithmetic, so identical vertices hash identically.
    seen = [Set{UInt64}() for _ in 1:(cut_mode === :multi ? n_scen : 1)]
    cut_key(cost, residual) = hash((cost, residual))
    function new_cut(k, key)
        key in seen[k] && return false
        push!(seen[k], key)
        return true
    end

    cuts, rays = 0, 0
    function add_cuts!(lag, epigraph = nothing)
        if cut_mode === :multi
            # A scenario whose epigraph the master already models tightly has
            # nothing to add; of the rest, the ones worth a row are those
            # accounting for most of the master's optimism.  At 10^4 scenarios a
            # cut for every one of them every iteration is what makes the master
            # unsolvable, and most of those rows are never binding.
            candidates = Tuple{Int,UInt64}[]
            for k in 1:n_scen
                scenario = lag.scenarios[k]
                isfinite(scenario.value) || continue
                epigraph === nothing ||
                    epigraph[k] > scenario.value + tol * (1 + abs(scenario.value)) || continue
                key = cut_key(scenario.cost, scenario.residual)
                new_cut(k, key) || continue
                push!(candidates, (k, key))
            end
            if epigraph !== nothing && length(candidates) > max_cuts
                sort!(candidates; rev = true,
                    by = ((k, _),) -> probs[k] * (epigraph[k] - lag.scenarios[k].value))
                candidates = candidates[1:max_cuts]
            end
            for (k, key) in candidates
                scenario_cut(k, lag.scenarios[k], key)
                cuts += 1
            end
        elseif isfinite(lag.value)
            R = lagrangian_moment_residual(s, lag.scenarios)
            cost = sum(probs[k] * lag.scenarios[k].cost for k in 1:n_scen)
            key = cut_key(cost, vec(R))
            if new_cut(1, key)
                row = @constraint(master, epi[1] <=
                    cost + sum(R[r,j] * Z[r,j] for r in 1:nrows, j in 1:K if R[r,j] != 0))
                push!(cut_rows, (row, 1, key))
                cuts += 1
            end
        end
        return cuts
    end

    """
    Drop the cuts that are doing no work, once the master holds more than
    `max_total_cuts` of them.

    A cutting-plane master accumulates rows for ever while only a handful bind
    at any one time, and it is the sheer count of near-parallel slack rows that
    makes the LP degenerate -- the master solves that stall here are the late,
    fat ones, not the early ones.  A cut priced at zero is not supporting the
    current optimum, so removing it cannot lower the master's value below the
    Lagrangian at the points still cut; it only costs the iterations it takes to
    rediscover, which is why its key leaves `seen` as well.  Each scenario keeps
    at least one cut, without which its epigraph -- and so the master -- is
    unbounded above.

    Must be called while the master's solution is still valid, since it reads
    the duals, and before any row is added.
    """
    function prune_cuts!()
        length(cut_rows) > max_total_cuts || return 0
        # Every price is read before the first deletion, since deleting a row
        # is a modification and leaves the rest of them unqueryable
        prices = [dual(row) for (row, _, _) in cut_rows]
        remaining = zeros(Int, length(seen))
        for (_, k, _) in cut_rows
            remaining[k] += 1
        end
        kept, dropped = empty(cut_rows), 0
        for (index, (row, k, key)) in enumerate(cut_rows)
            if remaining[k] > 1 && isapprox(prices[index], 0.0; atol = 1e-12)
                delete(master, row)
                delete!(seen[k], key)
                remaining[k] -= 1
                dropped += 1
            else
                push!(kept, (row, k, key))
            end
        end
        cut_rows = kept
        return dropped
    end

    # A Z some scenario finds dual-infeasible has Lagrangian -∞, so excluding it
    # is no restriction at all on the maximization
    "The multiplier `z(ξ_k)[r]` of coupling row `r` in scenario `k`, as an expression in Z."
    multiplier(k, r) = sum(Φ[k,j] * Z[r,j] for j in 1:K)
    function add_rays!(lag)
        added = 0
        for k in 1:n_scen, (kind, idx) in lag.scenarios[k].rays
            if kind === :spill
                @constraint(master, multiplier(k, balance_row(T, idx)) >= 0)
            elseif kind === :cuts_slack
                @constraint(master, multiplier(k, cut_row(T, idx)) >= 0)
            else
                @constraint(master,
                    sum(multiplier(k, cut_row(T, i)) for i in 1:problem.cuts_count) <= 1)
            end
            added += 1
            rays += 1
            added >= max_rays && return added
        end
        return added
    end

    # Stating every scenario's dual feasibility up front is cheap and keeps the
    # Lagrangian finite -- hence the bound usable -- at every iterate
    if ray_mode === :upfront
        for k in 1:n_scen
            @constraint(master, [t=1:T], multiplier(k, balance_row(T, t)) >= 0)
            @constraint(master, [i=1:problem.cuts_count], multiplier(k, cut_row(T, i)) >= 0)
            @constraint(master,
                sum(multiplier(k, cut_row(T, i)) for i in 1:problem.cuts_count) <= 1)
            rays += T + problem.cuts_count + 1
        end
    end

    # A cutting-plane master is exactly the shape a *dual* simplex handles
    # worst: thousands of near-parallel rows on a handful of columns, most of
    # them slack at the optimum. HiGHS's default choice stalls here -- 10 to 30
    # seconds a solve on a 2000-row LP, against 0.05 s for the primal simplex on
    # the same model, and its interior-point option does not finish at all. So
    # the master asks for the primal simplex, best-effort: a solver that does
    # not know the attribute keeps its own default, which costs time but no
    # correctness.
    for (name, value) in master_attributes
        try
            set_attribute(master, name, value)
        catch
            @warn "the master's solver does not accept $name = $value; " *
                "expect slower master solves"
        end
    end
    # A master the solver stalls on is still useful: its incumbent is a point
    # to evaluate, and the Lagrangian there is a valid bound whatever the LP
    # thinks. Only the master's own value needs it solved to optimality.
    master_time_limit === nothing || set_time_limit_sec(master, master_time_limit)
    function solve_master!(centre, radius)
        for i in 1:nrows, j in 1:K
            set_lower_bound(Z[i,j], centre[i,j] - radius / scale[j])
            set_upper_bound(Z[i,j], centre[i,j] + radius / scale[j])
        end
        dt += @elapsed optimize!(master)
        return termination_status(master)
    end

    """
    Does the trust region restrict the master's *value*, and not merely its
    solution?  Only then is `objective_value(master)` not an upper bound on the
    dual LDR value.

    Testing instead whether the solution sits on the trust-region boundary --
    the obvious thing -- reports `Inf` forever on any basis rich enough to
    matter: the master's objective sees `Z` only through the cuts, so in the
    many directions no cut has yet mentioned the LP is indifferent and happily
    returns a vertex against the boundary.  Those bounds carry a zero reduced
    cost, and a trust region whose bounds are all priced at zero is one the same
    primal-dual pair certifies optimal without it, so the master's value is its
    maximum over all of `Z` -- which is above the value of every dual rule.
    """
    restricting(radius) = any(eachindex(Z)) do i
        !isapprox(reduced_cost(Z[i]), 0.0; atol = 1e-9 * max(1.0, radius))
    end

    # Z = 0 prices nothing out, so every scenario is dual-feasible there and
    # every epigraph variable starts with a cut -- without which the master
    # would be unbounded above
    dt = 0.0
    center = zeros(nrows, K)
    lag = dual_ldr_lagrangian(s, center; ray_tol)
    add_cuts!(lag)
    best, best_Z = lag.value, copy(center)

    # The steepest cost in the problem is already a generous step for a
    # multiplier, and letting the trust region grow past it buys nothing while
    # making the master LP badly scaled, enormous bounds on variables the cuts
    # barely constrain.  It is not, on its own, what decides whether a master
    # stalls -- with the cut budget in place they stall at radius 2 as readily
    # as at 30 000, so the degeneracy of the cut model is the real culprit and
    # `master_time_limit` the real safeguard -- but there is nothing to gain
    # out there either.
    r = something(radius,
        max(maximum(problem.termo_costs), maximum(abs, problem.cuts_coeff)))
    r_max = something(max_radius, r)
    upper, converged, iterations, stalled = Inf, false, 0, 0
    for iteration in 1:max_iterations
        iterations = iteration
        before = dt
        status = solve_master!(center, r)
        master_dt = dt - before
        if status != OPTIMAL && !has_values(master)
            # Either the rays have cut off the whole trust region, or the solver
            # stopped with nothing usable at all -- a time limit reached before
            # it had a feasible point, which does happen on the degenerate
            # masters.  Either way the best point so far satisfies every ray, so
            # fall back on it and tighten rather than give up.
            verbose && (println("iteration $iteration: master $status with no " *
                "solution, recentering"); flush(stdout))
            center, r = copy(best_Z), max(r/2, min_radius)
            r <= min_radius && break
            continue
        end
        # All of the master solution is read before any cut or ray is added to it
        Z̄ = value.(Z)
        epigraph = cut_mode === :multi ? value.(epi) : nothing
        status == OPTIMAL && !restricting(r) &&
            (upper = min(upper, objective_value(master)))
        pruned = status == OPTIMAL ? prune_cuts!() : 0

        eval_dt = @elapsed lag = dual_ldr_lagrangian(s, Z̄; ray_tol)
        cut_dt = @elapsed begin
            n_new_rays = add_rays!(lag)
            add_cuts!(lag, epigraph)
        end

        improved = lag.value > best + tol * (1 + abs(best))
        improved && ((best, best_Z) = (lag.value, Z̄))
        stalled = improved ? 0 : stalled + 1
        if improved
            center, r = Z̄, min(2r, r_max)
        elseif n_new_rays == 0
            # A dual-feasible iterate that did not improve: the cut model was
            # too optimistic this far out, so look closer to the best point.
            center, r = copy(best_Z), r/2
        end

        # Flushed, since this loop is long enough that a buffered trace is no trace
        verbose && (println("iteration $iteration: lower $(round(best; digits=2)), " *
            "upper $(round(upper; digits=2)), radius $(round(r; digits=4)), " *
            "$cuts cuts ($(length(cut_rows)) kept, $pruned just dropped) and " *
            "$rays rays, master $(round(master_dt; digits=2))s, " *
            "evaluation $(round(eval_dt; digits=3))s, " *
            "cuts $(round(cut_dt; digits=3))s"); flush(stdout))
        if upper - best <= tol * (1 + abs(best))
            converged = true
            break
        end
        # Out of progress: the bound has not moved in `stall_limit` iterations.
        # This is the exit that matters when a master keeps timing out, since
        # `upper` then never updates and the gap test can never fire.
        stalled >= stall_limit && break
        r <= min_radius && break
    end

    return (;
        Obj = best,
        ObjBound = upper,
        Z = best_Z,
        dt,
        converged,
        iterations,
        cuts,
        rays,
        master,
    )
end

# ---------------------------------------------------------------------------
# Scenario (column) generation for the compact dual LDR
#
# The primal face of the same decomposition, and the exact mirror of the
# constraint generation that works for the primal LDR: where that one starts
# from a few scenarios' *rows* and adds the ones a rule violates, this starts
# from a few scenarios' *columns* and adds the ones that would lower the
# objective.
#
# Requiring a scenario to satisfy `A x(ξ_k) + s(ξ_k) = b(ξ_k)` pointwise makes
# its contribution to every aggregate row vanish, so for a working set S,
#
#   min  (1/K) Σ_{k∈S} c'x(ξ_k) + (1/K) Σ_{k∉S} v_k
#   s.t. Σ_{k∈S} p_k (A x(ξ_k) + s(ξ_k) - b(ξ_k)) L(ξ_k)' = 0,  y_k ∈ Y_k,
#
# with `v_k` the scenario's own deterministic optimum (`wait_and_see_values`),
# is a *restriction* of the scenario expansion: any of its solutions, completed
# by the pointwise-feasible decisions of the scenarios outside S, is feasible
# there at the same cost.  Its value is therefore an upper bound on the dual LDR
# value, decreasing monotonically as S grows, and only |S| scenario blocks are
# ever variables.  With |S| = 1 the single scenario is forced to be pointwise
# feasible too, so the first bound is exactly the wait-and-see value.
#
# Pricing is the Lagrangian of the dual-side method: with Z̄ the multipliers of
# the aggregate rows, scenario k's gap is `v_k - φ_k(Z̄ L(ξ_k)) >= 0` (infinite
# when Z̄ is dual-infeasible for it, the analogue of a violated scenario in the
# primal generation), and all gaps vanishing is exactly the optimality
# certificate: the restricted value then equals Σ_k p_k φ_k(Z̄ L(ξ_k)), which is
# the full Lagrangian, hence the dual LDR value.
#
# Unlike the moment masters, this master is always feasible and bounded -- the
# constant alone bounds it below -- so it needs neither the cheap moment bounds
# of `add_compact_moment_bounds!` nor any unbounded-ray handling.  Its
# iterates, though, are upper bounds on the dual LDR value and so are *not*
# valid lower bounds on the sampled primal until it converges, which is why
# every round also reports the Lagrangian of its own multipliers: that one is
# always valid.
# ---------------------------------------------------------------------------

"""
    wait_and_see_values(s; solver = HiGHS.Optimizer)

`v_k`, the optimal cost of every scenario of `s` solved on its own with perfect
foresight, i.e. with its dispatch constraints imposed pointwise.  Their sample
average is the wait-and-see value, which bounds the dual LDR value from above:
the canonical form of `Dual_scenarios.tex` puts no non-anticipativity on
`x(ξ)`, so a dual rule rich enough to enforce every row scenario by scenario
recovers exactly the perfect-foresight problem, and no basis can do better.
"""
function wait_and_see_values(s::Sample; solver = HiGHS.Optimizer)
    (; problem, T, inflow) = s
    n_scen = scenario_count(s)
    model = Model(solver)
    set_silent(model)
    @variables(model, begin
        0 <= volume[1:T] <= problem.hydro_capacity
        0 <= hydro[1:T] <= problem.hydro_generation
        0 <= spill[1:T]
        0 <= termos[i=1:problem.termo_count,1:T] <= problem.termo_generations[i]
        θ >= 0
        inflow_k[1:T]   # the scenario, refreshed below
    end)
    fix.(inflow_k, 0.0; force = true)
    for t in 1:T
        previous = t == 1 ? AffExpr(problem.initial_volume) : volume[t-1] + 0
        @constraint(model, volume[t] == previous + inflow_k[t] - hydro[t] - spill[t])
        @constraint(model, hydro[t] + sum(termos[:,t]) == problem.stage_demands[t])
    end
    @constraint(model, [i=1:problem.cuts_count],
        θ >= problem.cuts_coeff[i] * volume[T] + problem.cuts_inter[i])
    @objective(model, Min,
        sum(problem.termo_costs[i] * termos[i,t]
            for i in 1:problem.termo_count, t in 1:T) + θ)

    values = zeros(n_scen)
    for k in 1:n_scen
        fix.(inflow_k, inflow[:,k]; force = true)
        optimize!(model)
        termination_status(model) == OPTIMAL ||
            error("scenario $k is $(termination_status(model))")
        values[k] = objective_value(model)
    end
    return values
end

"""
    dual_ldr_restricted_primal(s, active, constant; solver = HiGHS.Optimizer)

The restricted primal of the scenario generation: the compact dual LDR of the
scenarios of `s` in `active` alone -- their aggregate rows, weighted by the
*whole* sample's `probs` so that the objective keeps its scale (see
`subsample`) -- plus the `constant` that the scenarios left out contribute
through their own deterministic optima.

Returns the model and the aggregate rows as a `coupling_row_count x K` matrix,
whose duals are the dual decision rule (see `coupling_multipliers`).
"""
function dual_ldr_restricted_primal(s::Sample, active, constant; solver = HiGHS.Optimizer)
    model = Model(solver)
    set_silent(model)
    (; problem, T) = s
    n_active = length(active)
    K = basis_count(s)
    sub = subsample(s, active)
    p, Ψ, I = sub.probs, sub.Φ, sub.inflow

    @variables(model, begin
        0 <= volume[1:T,1:n_active] <= problem.hydro_capacity
        0 <= hydro[1:T,1:n_active] <= problem.hydro_generation
        0 <= spill[1:T,1:n_active]
        0 <= termos[i=1:problem.termo_count,1:T,1:n_active] <= problem.termo_generations[i]
        θ[1:n_active] >= 0
        cuts_slack[1:problem.cuts_count,1:n_active] >= 0
    end)

    rows = Matrix{Any}(undef, coupling_row_count(problem, T), K)
    for t in 1:T
        previous = t == 1 ? fill(problem.initial_volume, n_active) : volume[t-1,:]
        rows[balance_row(T,t),:] = @constraint(model, dual_ldr_row(p, Ψ,
            volume[t,:] .- previous .- I[t,:] .+ hydro[t,:] .+ spill[t,:]) .== 0)
        rows[demand_row(T,t),:] = @constraint(model, dual_ldr_row(p, Ψ,
            hydro[t,:] .+ sum(termos[i,t,:] for i in 1:problem.termo_count) .-
            problem.stage_demands[t]) .== 0)
    end
    for i in 1:problem.cuts_count
        rows[cut_row(T,i),:] = @constraint(model, dual_ldr_row(p, Ψ,
            cuts_slack[i,:] .+ problem.cuts_coeff[i] .* volume[T,:] .- θ .+
            problem.cuts_inter[i]) .== 0)
    end
    @objective(model, Min, constant +
        sum(p[k] * (sum(problem.termo_costs[i] * termos[i,t,k]
                        for i in 1:problem.termo_count, t in 1:T) + θ[k])
            for k in 1:n_active))
    return (; model, rows)
end

"""
    coupling_multipliers(rows, target, lagrangian; tol)

The dual decision rule `Z` read off the duals of the aggregate `rows`.  Which
sign convention a solver reports for an equality row is not worth guessing, so
strong duality settles it: the restricted primal's value minus its constant *is*
the Lagrangian of its own multipliers, so both signs are evaluated by
`lagrangian` and the one reproducing `target` is returned.  A mismatch is a
warning rather than an error, since a badly conditioned master can miss it while
still giving a usable pricing direction.
"""
function coupling_multipliers(rows, target, lagrangian; tol = 1e-6)
    raw = Float64[dual(rows[r,j]) for r in axes(rows,1), j in axes(rows,2)]
    for Z in (-raw, raw)
        v = lagrangian(Z)
        isfinite(v) && abs(v - target) <= tol * (1 + abs(target)) && return Z
    end
    @warn "neither sign of the aggregate rows' duals reproduces the restricted " *
        "master's value $target; pricing may be unreliable"
    return -raw
end

"""
    basis_function_dual_ldr_column_generation(s; kwargs...)

Solve the model of `basis_function_dual_ldr_compact` by generating scenarios:
a working set keeps its scenario blocks as variables and every other scenario is
pinned to its own deterministic optimum, which grows the working set with the
scenarios whose Lagrangian gap says they would lower the objective (see the
comment above).  Accepts the same `solver` as `basis_function_ldr`, plus

  * `initial`: the starting working set, one scenario by default -- which makes
    the first bound the wait-and-see value;
  * `max_add`: how many of the largest-gap scenarios to add per round;
  * `max_rounds`, `tol`, `verbose`.

Returns `Obj`, the restricted primal's value: an upper bound on the dual LDR
value, decreasing every round and exact once no scenario is mispriced.  Since
that is *not* a valid lower bound on the sampled primal before convergence,
`ObjBound` reports the best Lagrangian of the multipliers priced along the way,
which is (see `dual_ldr_lagrangian`).  Also returns the `wait_and_see` value the
first round starts from, whether the loop `converged`, its `rounds`, the
`active` working set, the worst remaining `gap`, the multipliers `Z`, `dt` and
the last restricted `master`.
"""
function basis_function_dual_ldr_column_generation(
    s::Sample;
    solver = HiGHS.Optimizer,
    initial = nothing,
    max_add = 5,
    max_rounds = 100,
    tol = 1e-6,
    verbose = false,
)
    probs = s.probs
    n_scen = scenario_count(s)

    dt = @elapsed v = wait_and_see_values(s; solver)
    wait_and_see = dot(probs, v)

    active = falses(n_scen)
    active[something(initial, [1])] .= true

    upper, lower, converged, rounds, gap = Inf, -Inf, false, 0, Inf
    master, Z = nothing, zeros(coupling_row_count(s), basis_count(s))
    for iteration in 1:max_rounds
        rounds = iteration
        constant = sum(probs[k] * v[k] for k in 1:n_scen if !active[k]; init = 0.0)
        restricted = dual_ldr_restricted_primal(s, findall(active), constant; solver)
        master = restricted.model
        dt += @elapsed optimize!(master)
        termination_status(master) == OPTIMAL ||
            error("restricted master is $(termination_status(master))")
        upper = objective_value(master)

        # Strong duality on the restricted primal: its multipliers are a dual
        # decision rule, and the Lagrangian of the active scenarios alone is
        # what identifies their sign.
        Z = coupling_multipliers(restricted.rows, upper - constant,
            Z -> dual_ldr_lagrangian(subsample(s, active), Z).value)

        # Pricing: the same Lagrangian, now over the whole sample -- a valid
        # lower bound as a by-product
        lag = dual_ldr_lagrangian(s, Z)
        lower = max(lower, lag.value)
        gaps = [active[k] ? -Inf : v[k] - lag.scenarios[k].value for k in 1:n_scen]
        gap = maximum(gaps)

        verbose && (println("round $iteration: value $(round(upper; digits=2)), " *
            "lagrangian $(round(lag.value; digits=2)), " *
            "$(count(active)) of $n_scen scenarios, " *
            "worst gap $(round(gap; digits=6))"); flush(stdout))
        if gap <= tol
            # Every scenario outside the working set is priced exactly at its
            # own optimum, so the restricted value is the full Lagrangian.
            converged = true
            break
        end
        added = 0
        for k in sortperm(gaps; rev = true)
            gaps[k] > tol || break
            active[k] = true
            added += 1
            added >= max_add && break
        end
        added == 0 && break
    end

    return (;
        Obj = upper,
        ObjBound = lower,
        wait_and_see,
        gap,
        Z,
        dt,
        converged,
        rounds,
        active = findall(active),
        master,
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

"""
    set_ldr_objective!(model, s)

The expected cost of a rule over the sample `s`, exactly: linear in the
coefficients, with the mean basis vector `s.φ̄` (see `mean_basis`) carrying the
whole sample.
"""
function set_ldr_objective!(model, s::Sample)
    (; problem, T, φ̄) = s
    K = basis_count(s)
    termos_ldr, θ_ldr = model[:termos_ldr], model[:θ_ldr]
    @objective(model, Min,
        sum(problem.termo_costs[i] * termos_ldr[i,t,j] * φ̄[j]
            for i in 1:problem.termo_count, t in 1:T, j in 1:K) +
        sum(θ_ldr[j] * φ̄[j] for j in 1:K))
    return model
end

"""
    rule_values(s, x)

The decisions the rule `x` -- coefficient values, flattened as
`ldr_coefficients` orders them -- prescribes in every scenario of the sample
`s`, as one `n_scen x T` matrix per decision (a vector of them per thermal
plant) and one `n_scen` vector of future costs.
"""
function rule_values(s::Sample, x)
    (; problem, Φ) = s
    c = reshape_coefficients(s, x)
    return (;
        volume = Φ * c.volume_ldr',
        hydro = Φ * c.hydro_ldr',
        spill = Φ * c.spill_ldr',
        termos = [Φ * c.termos_ldr[i,:,:]' for i in 1:problem.termo_count],
        θ = Φ * c.θ_ldr,
    )
end

"""
    scenario_violation(s, decisions)

How far the decisions of `rule_values` are from feasible, scenario by scenario
of the sample `s`: the largest violation among the bounds, the water balance,
the demand and the future-cost rows.  This is the separation step, and it is
"just" arithmetic -- the whole sample is checked without calling a solver.
"""
function scenario_violation(s::Sample, decisions)
    (; problem, T, inflow) = s
    n_scen = scenario_count(s)
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
    add_scenario_rows!(master, s, inflow, basis; equalities)

Add one scenario's constraints, written in the coefficients alone: the bounds of
every decision and the future-cost rows, plus -- when `equalities` is
`:per_scenario` -- its water balance and demand rows.  `inflow` is the
scenario's inflow per stage and `basis` its row of the basis matrix, which need
not come from the sample `s` itself (see the aggregate rows of
`basis_function_ldr_scenario_generation`).
"""
function add_scenario_rows!(master, s::Sample, inflow, basis; equalities = :per_scenario)
    (; problem, T) = s
    K = length(basis)
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
    add_coefficientwise_equalities!(master, s)

Impose the water balance and the demand once per basis coefficient, so that they
hold for every realization of the inflow and never have to be generated.

This needs the inflow of each stage to be a combination of the basis functions
that stage knows, `inflow[t,k] == Σ_j r[t,j] Φ[k,j]` with `r[t,j] = 0` for the
functions of later stages -- non-anticipativity zeroes the coefficients such a
row would talk about, so a representation using them would be infeasible rather
than merely wrong.  The representation is fitted by least squares over the
admissible columns and its residual checked.  It holds for `BP2` and
`PW3`, whose pieces sum back to the inflow of their stage.
"""
function add_coefficientwise_equalities!(master, s::Sample)
    (; problem, T, inflow, Φ) = s
    K = basis_count(s)
    r = zeros(T, K)
    for t in 1:T
        available = [1; 1 .+ findall(<=(t), s.basis.stages)]
        r[t,available] = Φ[:,available] \ inflow[t,:]
        residual = maximum(abs, Φ * r[t,:] .- inflow[t,:])
        residual <= 1e-6 * max(1.0, maximum(abs, inflow[t,:])) || error(
            "the stage-$t inflow is not in the span of the basis functions of " *
            "stages up to $t (residual $residual), so equalities = " *
            ":coefficientwise cannot hold for every realization; use " *
            ":per_scenario, or a basis that represents the inflow as " *
            "BP2 does")
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
    basis_function_ldr_scenario_generation(s; kwargs...)

Solve the model of `basis_function_ldr` by scenario generation: a master problem
over the decision-rule coefficients carries the exact objective of the whole
sample and the rows of a working set of scenarios, which grows with the
scenarios the current rule violates (see the comment above).  Accepts the same
`solver` as `basis_function_ldr`, plus

  * `equalities`: `:per_scenario` generates the water balance and the demand
    with each scenario, and reproduces `basis_function_ldr` exactly;
    `:coefficientwise` imposes them for every realization up front, which is the
    same model whenever `Φ` has full column rank on its positive-probability
    scenarios (`active_rank`) and a stronger one when it does not -- in which
    case the master is no longer a relaxation and no bound is reported;
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
    s::Sample;
    solver = HiGHS.Optimizer,
    equalities = :per_scenario,
    n_init = nothing,
    max_add = 10,
    max_rounds = 100,
    tol = 1e-6,
    verbose = false,
)
    equalities in (:per_scenario, :coefficientwise) ||
        throw(ArgumentError("equalities must be :per_scenario or :coefficientwise"))

    (; inflow, probs, Φ, φ̄) = s
    n_scen = scenario_count(s)
    K = basis_count(s)

    master = Model(solver)
    set_silent(master)
    x = add_ldr_coefficients!(master, s)
    fix_nonanticipativity!(master, s)
    set_ldr_objective!(master, s)

    # With the equalities imposed for every realization the master is a
    # relaxation of the sampled model only if the sample cannot tell the two
    # formulations apart, which is what full column rank of Φ on its
    # positive-probability scenarios says (`active_rank`).
    bounding = true
    if equalities === :coefficientwise
        add_coefficientwise_equalities!(master, s)
        bounding = active_rank(Φ, probs) == K
    end

    # The sample average of the scenarios' rows: a convex combination of rows of
    # the full model, hence implied by it, and the same rows evaluated at the
    # mean basis vector and the mean inflow.  Adding it up front keeps the master
    # bounded (thermal generation and future cost are non-negative),
    # which no working set of scenarios is guaranteed to do on its own.
    add_scenario_rows!(master, s, inflow * probs, φ̄; equalities)

    added = falses(n_scen)
    function activate!(candidates)
        activated = 0
        for k in candidates
            added[k] && continue
            add_scenario_rows!(master, s, inflow[:,k], Φ[k,:]; equalities)
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

        violation = scenario_violation(s, rule_values(s, x̄))
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

    coefficients = reshape_coefficients(s, something(best, fill(NaN, length(x))))
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
    basis_ldr_generation_solution(problem, basis = BP2; T, scenarios = 100, rng, kwargs...)

`basis_function_ldr_scenario_generation` on a fresh `Sample(problem, basis; T,
scenarios, rng)`, reduced to the fields `run_comparison` tabulates, since every
other field of the full result would become a column of the table -- a JuMP
model among them.

Meant for the whole horizon of `problem`, hence the default `T =
stage_count(problem)`: `evaluate_policy` scores the stage-2 and stage-3 recourse
of the first-stage decisions, which a shorter horizon would leave out of the
rule.
"""
function basis_ldr_generation_solution(
    problem::HydroProblem,
    basis = BP2;
    T = stage_count(problem),
    scenarios = 100,
    rng = Random.default_rng(),
    kwargs...,
)
    s = Sample(problem, basis; T, scenarios, rng)
    sol = basis_function_ldr_scenario_generation(s; kwargs...)
    return (; sol.Obj, sol.Bound, sol.H₁, sol.V₁, sol.T₁, sol.dt)
end

"""
    compare_basis_function_ldr(s; kwargs...)

Solve the monolithic basis-function LDR model, its Benders decomposition and its
scenario generation in both equality modes, all on the sample `s`.  Keyword
arguments of the two decompositions go in the `benders` and `generation` named
tuples.
"""
function compare_basis_function_ldr(
    s::Sample;
    solver = HiGHS.Optimizer,
    verbose = false,
    benders = (;),
    generation = (;),
)
    full = basis_function_ldr(s; solver)
    dt_full = @elapsed optimize!(full)

    bend = basis_function_ldr_benders(s; solver, verbose, benders...)
    gen = basis_function_ldr_scenario_generation(s; solver, verbose, generation...)
    gen_cw = basis_function_ldr_scenario_generation(
        s; solver, verbose, generation..., equalities = :coefficientwise
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
