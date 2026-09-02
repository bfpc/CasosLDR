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
    inflow = hcat(scenarios...) # each column is a scenario
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
and return them as a flat vector.
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
    inflow = hcat(scenarios...) # each column is a scenario
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
