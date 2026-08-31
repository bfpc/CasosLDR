isdefined(Main, :Compare) || include(joinpath(@__DIR__, "..", "src", "Compare.jl"))

module HydroQuadPw

using Random

using JuMP
using HiGHS, Ipopt
using Distributions
using LinearDecisionRules

using Main.Compare

# Two-stage hydro-thermal model with a single thermal plant whose cost is
# piecewise quadratic: one quadratic piece below `termo_cost_breakpoint` and
# another above it, modelled as two generation variables.
#
# Every solution method returns a NamedTuple `(; Obj, Bound, H₁, V₁, T₁, dt)`,
# which `Compare.run_comparison` times, evaluates and tabulates.
default_solver = optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm")

struct HydroProblem
    hydro_generation::Float64
    hydro_capacity::Float64
    termo_generation::Float64
    termo_cost_coeffs::Matrix{Float64}
    termo_cost_breakpoint::Float64
    stage_demands::Vector{Float64}
    initial_volume::Float64
    initial_inflow::Float64
    inflow_distr::Distribution
    cuts_count::Int
    cuts_coeff::Vector{Float64}
    cuts_inter::Vector{Float64}
    solver::Any
    second_stage::Ref{Union{Nothing, Model}}
    ext::Dict{Symbol, Any}

    function HydroProblem(
        hydro_generation::Float64,
        hydro_capacity::Float64,
        termo_generation::Float64, #assume que inicia em 0
        termo_cost_coeffs::Matrix{Float64},
        termo_cost_breakpoint::Float64,
        stage_demands::Vector{Float64},
        initial_volume::Float64,
        initial_inflow::Float64,
        inflow_distr::Distribution,
        cuts_coeff::Vector{Float64},
        cuts_inter::Vector{Float64},
        solver = default_solver
    )
        if hydro_generation <= 0 || hydro_capacity <= 0
            error("Hydro parameters must be positive values")
        end
        if size(termo_cost_coeffs, 1) != 2 || size(termo_cost_coeffs, 2) != 2
            error("Must have 2 coefficients by piece for the two-pieces quadratic cost function for termos")
        end
        if termo_generation <= 0
            error("Generation of thermos must be greater than 0")
        end
        if termo_cost_breakpoint >= termo_generation || termo_cost_breakpoint <= 0
            error("The breakpoint for the cost function must be inside the generation bounds")
        end
        if length(stage_demands) != 2
            error("stage_demands should have only two values; only two stages problems are supported")
        end
        if any(stage_demands .<= 0)
            error("Demand parameters must be positive values")
        end
        if initial_volume <= 0 || initial_inflow <= 0
            error("Initial volume and inflow must be positive values")
        end
        if length(cuts_coeff) != length(cuts_inter)
            error("Mismatch between quantity of cuts coefficients and intercepts")
        end

        cuts_count = length(cuts_coeff)

        return new(
            hydro_generation,
            hydro_capacity,
            termo_generation,
            termo_cost_coeffs,
            termo_cost_breakpoint,
            stage_demands,
            initial_volume,
            initial_inflow,
            inflow_distr,
            cuts_count,
            cuts_coeff,
            cuts_inter,
            solver,
            Ref{Union{Nothing, Model}}(nothing),
            Dict{Symbol, Any}(),
        )
    end
end

"Piecewise-quadratic thermal cost of generating `x`."
function quad_cost(x, problem::HydroProblem)
    coeffs = problem.termo_cost_coeffs
    bp = problem.termo_cost_breakpoint
    if x <= bp
        return coeffs[1,1]*x^2 + coeffs[1,2]*x
    else
        base = coeffs[1,1]*bp^2 + coeffs[1,2]*bp
        return base + coeffs[2,1]*(x - bp)^2 + coeffs[2,2]*(x - bp)
    end
end

"Marginal thermal cost at generation `x`."
function quad_marg_cost(x, problem::HydroProblem)
    coeffs = problem.termo_cost_coeffs
    bp = problem.termo_cost_breakpoint
    if x <= bp
        return 2*coeffs[1,1]*x + coeffs[1,2]
    else
        return 2*coeffs[2,1]*(x - bp) + coeffs[2,2]
    end
end

"""
Evaluate second-stage recourse cost for out-of-sample policy assessment.
"""
function recourse_solution(
    problem::HydroProblem,
    volume_stage2_in::Real,
    inflow_stage2::Real,
)
    fs = cached_model(problem) do
        cost_coeffs = problem.termo_cost_coeffs
        cuts_count = problem.cuts_count

        fs = Model(HiGHS.Optimizer)
        set_silent(fs)

        @variables(fs, begin
            0 <= hydro <= problem.hydro_generation
            0 <= termo_generation₁ <= problem.termo_cost_breakpoint
            0 <= termo_generation₂ <= problem.termo_generation - problem.termo_cost_breakpoint
            volume₂ >= 0
            spill >= 0
            θ >= 0
            volume₁
            inflow₂
        end)

        @constraints(fs, begin
            volume₂ == volume₁ + inflow₂ - hydro - spill
            volume₂ <= problem.hydro_capacity
            hydro + termo_generation₁ + termo_generation₂ == problem.stage_demands[2]
            [i in 1:cuts_count], θ >= problem.cuts_coeff[i]*volume₂ + problem.cuts_inter[i]
        end)

        @objective(fs, Min,
            cost_coeffs[1,1]*termo_generation₁^2 + cost_coeffs[1,2]*termo_generation₁
            + cost_coeffs[2,1]*termo_generation₂^2 + cost_coeffs[2,2]*termo_generation₂
            + θ)

        return fs
    end

    fix(fs[:volume₁], volume_stage2_in)
    fix(fs[:inflow₂], inflow_stage2)
    optimize!(fs)
    return (
        status = termination_status(fs),
        objective = objective_value(fs),
        hydro = value(fs[:hydro]),
        volume_final = value(fs[:volume₂]),
        thermal = value(fs[:termo_generation₁]) + value(fs[:termo_generation₂]),
        spill = value(fs[:spill]),
        future_cost = value(fs[:θ]),
    )
end

"First-stage thermal cost of a solution."
firststage_cost(problem::HydroProblem, sol) = quad_cost(sol.T₁, problem)

"""
Out-of-sample assessment of a solution: average the stage-2 recourse over
`n_scenarios` sampled inflows and add the first-stage cost it commits to.
"""
function evaluate_policy(problem::HydroProblem, sol; n_scenarios = 10_000, seed = 0)
    inflows = rand(MersenneTwister(seed), problem.inflow_distr, n_scenarios)

    average, stderror =
        average_scenarios(inflow -> recourse_solution(problem, sol.V₁, inflow), inflows)

    return (;
        H₂ = average.hydro,
        V₂ = average.volume_final,
        T₂ = average.thermal,
        Perf = average.objective + firststage_cost(problem, sol),
        Std = stderror.objective,
    )
end

function expected_value_solution(problem::HydroProblem)
    inflows = [problem.initial_inflow, mean(problem.inflow_distr)]
    cost_coeffs = problem.termo_cost_coeffs

    ev = Model(problem.solver)
    set_silent(ev)

    @variables(ev, begin
        0 <= hydro[1:2] <= problem.hydro_generation
        0 <= termo_generation₁[1:2] <= problem.termo_cost_breakpoint
        0 <= termo_generation₂[1:2] <= problem.termo_generation - problem.termo_cost_breakpoint
        0 <= volume[1:2] <= problem.hydro_capacity
        spill[1:2] >= 0
        θ >= 0
    end)


    @constraints(ev, begin
        volume[1] == problem.initial_volume + inflows[1] - hydro[1] - spill[1]
        volume[2] == volume[1] + inflows[2] - hydro[2] - spill[2]
        hydro[1] <= problem.initial_volume
        hydro[2] <= volume[1]
        [t in 1:2], hydro[t] + termo_generation₁[t] + termo_generation₂[t] == problem.stage_demands[t]
        [i in 1:problem.cuts_count], θ >= problem.cuts_coeff[i]*volume[2] + problem.cuts_inter[i]
    end)

    @objective(ev, Min,
        cost_coeffs[1,1]*sum(termo_generation₁ .^ 2) + cost_coeffs[1,2]*sum(termo_generation₁)
        + cost_coeffs[2,1]*sum(termo_generation₂ .^ 2) + cost_coeffs[2,2]*sum(termo_generation₂)
        + θ)

    dt = @elapsed optimize!(ev)

    return (;
        Obj = objective_value(ev),
        Bound = -Inf,
        H₁ = value(hydro[1]),
        V₁ = value(volume[1]),
        T₁ = value(termo_generation₁[1] + termo_generation₂[1]),
        dt,
    )
end

function ldr_solution(problem::HydroProblem, n_breakpoints=0)
    cost_coeffs = problem.termo_cost_coeffs

    ldr = LinearDecisionRules.LDRModel(problem.solver)
    set_silent(ldr)

    @variable(ldr, inflow in
             LinearDecisionRules.Uncertainty(; distribution = problem.inflow_distr))
    @variables(ldr, begin
        0 <= hydro₁ <= problem.hydro_generation, LinearDecisionRules.FirstStage
        0 <= termo₁_generation₁ <= problem.termo_cost_breakpoint, LinearDecisionRules.FirstStage
        0 <= termo₁_generation₂ <= problem.termo_generation - problem.termo_cost_breakpoint, LinearDecisionRules.FirstStage
        0 <= volume₁ <= problem.hydro_capacity, LinearDecisionRules.FirstStage
        spill₁ >= 0, LinearDecisionRules.FirstStage

        0 <= hydro₂ <= problem.hydro_generation
        0 <= termo₂_generation₁ <= problem.termo_cost_breakpoint
        0 <= termo₂_generation₂ <= problem.termo_generation - problem.termo_cost_breakpoint
        0 <= volume₂ <= problem.hydro_capacity
        spill₂ >= 0
        θ >= 0
    end)

    @constraints(ldr, begin
        volume₁ == problem.initial_volume + problem.initial_inflow - hydro₁ - spill₁
        volume₂ == volume₁ + inflow - hydro₂ - spill₂
        hydro₁ <= problem.initial_volume
        hydro₂ <= volume₁
        hydro₁ + termo₁_generation₁ + termo₁_generation₂ == problem.stage_demands[1]
        hydro₂ + termo₂_generation₁ + termo₂_generation₂ == problem.stage_demands[2]
        [i in 1:problem.cuts_count], θ >= problem.cuts_coeff[i]*volume₂ + problem.cuts_inter[i]
    end)

    @objective(ldr, Min,
        cost_coeffs[1,1]*(termo₁_generation₁^2 + termo₂_generation₁^2)
        + cost_coeffs[1,2]*(termo₁_generation₁ + termo₂_generation₁)
        + cost_coeffs[2,1]*(termo₁_generation₂^2 + termo₂_generation₂^2)
        + cost_coeffs[2,2]*(termo₁_generation₂ + termo₂_generation₂)
        + θ)

    # Set breakpoints for uncertainty approximation.
    if n_breakpoints > 0
        set_attribute(inflow, LinearDecisionRules.BreakPoints(), n_breakpoints)
    end

    dt = @elapsed optimize!(ldr)

    return (;
        Obj = objective_value(ldr),
        Bound = objective_value(ldr; dual = true),
        H₁ = LinearDecisionRules.get_decision(ldr, hydro₁),
        V₁ = LinearDecisionRules.get_decision(ldr, volume₁),
        T₁ = LinearDecisionRules.get_decision(ldr, termo₁_generation₁) +
             LinearDecisionRules.get_decision(ldr, termo₁_generation₂),
        dt,
    )
end

function saa_solution(problem::HydroProblem, n_samples::Int)
    if n_samples <= 0
        error("n_samples must be positive")
    end

    cuts_count = problem.cuts_count
    inflows = rand(problem.inflow_distr, n_samples)
    cost_coeffs = problem.termo_cost_coeffs

    saa = Model(problem.solver)
    set_silent(saa)

    @variables(saa, begin
        0 <= hydro₁ <= problem.hydro_generation
        0 <= termo₁_generation₁ <= problem.termo_cost_breakpoint
        0 <= termo₁_generation₂ <= problem.termo_generation - problem.termo_cost_breakpoint
        0 <= volume₁ <= problem.hydro_capacity
        spill₁ >= 0

        0 <= hydro₂[1:n_samples] <= problem.hydro_generation
        0 <= termo₂_generation₁[1:n_samples] <= problem.termo_cost_breakpoint
        0 <= termo₂_generation₂[1:n_samples] <= problem.termo_generation - problem.termo_cost_breakpoint
        0 <= volume₂[1:n_samples] <= problem.hydro_capacity
        spill₂[1:n_samples] >= 0
        θ[1:n_samples] >= 0
    end)


    @constraints(saa, begin
        volume₁ == problem.initial_volume + problem.initial_inflow - hydro₁ - spill₁
        hydro₁ <= problem.initial_volume
        hydro₁ + termo₁_generation₁ + termo₁_generation₂ == problem.stage_demands[1]
    end)

    for s in 1:n_samples
        @constraints(saa, begin
            volume₂[s] == volume₁ + inflows[s] - hydro₂[s] - spill₂[s]
            hydro₂[s] <= volume₁
            hydro₂[s] + termo₂_generation₁[s] + termo₂_generation₂[s] == problem.stage_demands[2]
            [i in 1:cuts_count], θ[s] >= problem.cuts_coeff[i] * volume₂[s] + problem.cuts_inter[i]
        end)
    end

    @objective(saa, Min,
        cost_coeffs[1,1]*(termo₁_generation₁^2 + (sum(termo₂_generation₁)/n_samples)^2)
        + cost_coeffs[1,2]*(termo₁_generation₁ + (sum(termo₂_generation₁)/n_samples))
        + cost_coeffs[2,1]*(termo₁_generation₂^2 + (sum(termo₂_generation₂)/n_samples)^2)
        + cost_coeffs[2,2]*(termo₁_generation₂ + (sum(termo₂_generation₂)/n_samples))
        + sum(θ[s] for s in 1:n_samples) / n_samples,
    )

    dt = @elapsed optimize!(saa)

    return (;
        Obj = objective_value(saa),
        Bound = -Inf,
        H₁ = value(hydro₁),
        V₁ = value(volume₁),
        T₁ = value(termo₁_generation₁ + termo₁_generation₂),
        dt,
    )
end

# Remaining keyword arguments, `summarize` in particular, go to `run_comparison`.
function compare_methods(
    problem::HydroProblem;
    n_scenarios = 10_000,
    replications = 3,
    kwargs...,
)
    solvers = [
        "EV" => expected_value_solution
        ["LDR ($n breakpoints)" => p -> ldr_solution(p, n) for n in 0:3]
        ["SAA ($n samples)" => p -> saa_solution(p, n) for
         n in repeat([4, 10, 30, 100]; inner = replications)]
    ]

    return run_comparison(
        problem,
        with_warmup(solvers);
        evaluate = (p, sol) -> evaluate_policy(p, sol; n_scenarios),
        kwargs...,
    )
end

function build_toy_problem(solver = default_solver)
    h_generation = 100.
    h_capacity = 300.
    t_generation = 20.
    stage_demands = [70., 70.]
    initial_vol = 100.
    initial_inflow = 60.
    inflow_distr = truncated(Normal(60., 20.), 0., 120)
    cuts_coeff = [-300., -600., -1200., -2500., -7800.]
    cuts_inter = [75000., 135000., 210000., 310000., 390000.]
    t_breakpoint = 10.
    t_cost_coeffs = [
        113.1/2.   0.;
        3118.6/2.  1131.;
    ]

    return HydroProblem(
        h_generation,
        h_capacity,
        t_generation,
        t_cost_coeffs,
        t_breakpoint,
        stage_demands,
        initial_vol,
        initial_inflow,
        inflow_distr,
        cuts_coeff,
        cuts_inter,
        solver
    )
end

function main(solver = default_solver; kwargs...)
    toy = build_toy_problem(solver)
    compare_methods(toy; kwargs...)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    HydroQuadPw.main()
end
