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
    inflow = hcat(scenarios...) # each column is a scenario
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
        Bound = lower,
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

"""
    compare_basis_function_ldr(problem, functions, stage_of_func; T, n_scen, solver, kwargs...)

Solve the monolithic basis-function LDR model and its Benders decomposition on
the same scenarios.
"""
function compare_basis_function_ldr(
    problem::HydroProblem,
    functions,
    stage_of_func;
    T = 2,
    n_scen = 20,
    solver = HiGHS.Optimizer,
    kwargs...,
)
    scenarios = sample_scenarios(problem, T, n_scen)

    full = basis_function_ldr(problem, functions, stage_of_func; T, solver, scenarios)
    dt_full = @elapsed optimize!(full)

    bend = basis_function_ldr_benders(
        problem, functions, stage_of_func; T, solver, scenarios, kwargs...
    )

    return (;
        monolithic = termination_status(full) == OPTIMAL ? objective_value(full) : Inf,
        decomposed = bend.Obj,
        bound = bend.Bound,
        bend.converged,
        bend.iterations,
        bend.feasibility_cuts,
        bend.optimality_cuts,
        dt_monolithic = dt_full,
        dt_benders = bend.dt,
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
