isdefined(Main, :Compare) || include(joinpath(@__DIR__, "..", "src", "Compare.jl"))

module HydroQuad

using Random

using JuMP
using HiGHS, Ipopt
using Distributions
using LinearDecisionRules

using Main.Compare

# Two-stage hydro-thermal model with a single thermal plant whose cost is a
# quadratic function of its generation.
#
# Every solution method returns a NamedTuple `(; Obj, Bound, H₁, V₁, T₁, dt)`,
# which `Compare.run_comparison` times, evaluates and tabulates.

struct HydroProblem
    hydro_generation::Float64
    hydro_capacity::Float64
    termo_generation::Float64
    termo_cost_coeffs::Vector{Float64}
    stage_demands::Vector{Float64}
    initial_volume::Float64
    initial_inflow::Float64
    inflow_distr::Distribution
    cuts_count::Int
    cuts_coeff::Vector{Float64}
    cuts_inter::Vector{Float64}
    second_stage::Ref{Union{Nothing, Model}}
    ext::Dict{Symbol, Any}

    function HydroProblem(
        hydro_generation::Float64,
        hydro_capacity::Float64,
        termo_generation::Float64, #assume que inicia em 0
        termo_cost_coeffs::Vector{Float64},
        stage_demands::Vector{Float64},
        initial_volume::Float64,
        initial_inflow::Float64,
        inflow_distr::Distribution,
        cuts_coeff::Vector{Float64},
        cuts_inter::Vector{Float64}
    )
        if hydro_generation <= 0 || hydro_capacity <= 0
            error("Hydro parameters must be positive values")
        end
        if length(termo_cost_coeffs) != 3
            error("Must have 3 coefficients for the quadratic cost function for termos")
        end
        if termo_generation <= 0
            error("Generation of thermos must be greater than 0")
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
            stage_demands,
            initial_volume,
            initial_inflow,
            inflow_distr,
            cuts_count,
            cuts_coeff,
            cuts_inter,
            Ref{Union{Nothing, Model}}(nothing),
            Dict{Symbol, Any}(),
        )
    end
end

"Quadratic thermal cost of generating `x`."
function quad_cost(x, problem::HydroProblem)
    coeffs = problem.termo_cost_coeffs
    return coeffs[1]*x^2 + coeffs[2]*x + coeffs[3]
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
        cuts_count = problem.cuts_count

        fs = Model(HiGHS.Optimizer)
        set_silent(fs)

        @variables(fs, begin
            0 <= hydro <= problem.hydro_generation
            0 <= termo <= problem.termo_generation
            volume₂ >= 0
            spill >= 0
            θ >= 0
            volume₁
            inflow₂
        end)

        @constraints(fs, begin
            volume₂ == volume₁ + inflow₂ - hydro - spill
            volume₂ <= problem.hydro_capacity
            hydro + termo == problem.stage_demands[2]
            [i in 1:cuts_count], θ >= problem.cuts_coeff[i]*volume₂ + problem.cuts_inter[i]
        end)

        @objective(fs, Min, problem.termo_cost_coeffs[1]*termo^2 + problem.termo_cost_coeffs[2]*termo + problem.termo_cost_coeffs[3] + θ)

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
        thermal = value(fs[:termo]),
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

    ev = Model(HiGHS.Optimizer)
    set_silent(ev)
    set_attribute(ev, "solver", "ipm")
    set_attribute(ev, "time_limit", 60.)
    # set_attribute(ev, "max_wall_time", 60.)

    @variables(ev, begin
        0 <= hydro[1:2] <= problem.hydro_generation
        0 <= termo[1:2] <= problem.termo_generation
        0 <= volume[1:2] <= problem.hydro_capacity
        spill[1:2] >= 0
        θ >= 0
    end)


    @constraints(ev, begin
        volume[1] == problem.initial_volume + inflows[1] - hydro[1] - spill[1]
        volume[2] == volume[1] + inflows[2] - hydro[2] - spill[2]
        hydro[1] <= problem.initial_volume
        hydro[2] <= volume[1]
        [t in 1:2], hydro[t] + termo[t] == problem.stage_demands[t]
        [i in 1:problem.cuts_count], θ >= problem.cuts_coeff[i]*volume[2] + problem.cuts_inter[i]
    end)

    @objective(ev, Min,
        problem.termo_cost_coeffs[1]*(termo[1]^2 + termo[2]^2)
        + problem.termo_cost_coeffs[2]*(termo[1]+termo[2])
        + 2*problem.termo_cost_coeffs[3]
        + θ)

    dt = @elapsed optimize!(ev)

    return (;
        Obj = objective_value(ev),
        Bound = -Inf,
        H₁ = value(hydro[1]),
        V₁ = value(volume[1]),
        T₁ = value(termo[1]),
        dt,
    )
end

function ldr_solution(problem::HydroProblem, n_breakpoints=0)
    ldr = LinearDecisionRules.LDRModel(optimizer_with_attributes(
        HiGHS.Optimizer,
        "solver" => "ipm",
        "time_limit" => 60.,
        # "max_wall_time" => 60.,
    ))
    set_silent(ldr)

    @variable(ldr, inflow in
             LinearDecisionRules.Uncertainty(; distribution = problem.inflow_distr))
    @variables(ldr, begin
        0 <= hydro₁ <= problem.hydro_generation, LinearDecisionRules.FirstStage
        0 <= termo₁ <= problem.termo_generation, LinearDecisionRules.FirstStage
        0 <= volume₁ <= problem.hydro_capacity, LinearDecisionRules.FirstStage
        spill₁ >= 0, LinearDecisionRules.FirstStage

        0 <= hydro₂ <= problem.hydro_generation
        0 <= termo₂ <= problem.termo_generation
        0 <= volume₂ <= problem.hydro_capacity
        spill₂ >= 0
        θ >= 0
    end)

    @constraints(ldr, begin
        volume₁ == problem.initial_volume + problem.initial_inflow - hydro₁ - spill₁
        volume₂ == volume₁ + inflow - hydro₂ - spill₂
        hydro₁ <= problem.initial_volume
        hydro₂ <= volume₁
        hydro₁ + termo₁ == problem.stage_demands[1]
        hydro₂ + termo₂ == problem.stage_demands[2]
        [i in 1:problem.cuts_count], θ >= problem.cuts_coeff[i]*volume₂ + problem.cuts_inter[i]
    end)

    @objective(ldr,
               Min,
               problem.termo_cost_coeffs[1]*(termo₁^2 + termo₂^2)
               + problem.termo_cost_coeffs[2]*(termo₁+termo₂)
               + 2*problem.termo_cost_coeffs[3]
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
        T₁ = LinearDecisionRules.get_decision(ldr, termo₁),
        dt,
    )
end

function saa_solution(problem::HydroProblem, n_samples::Int)
    if n_samples <= 0
        error("n_samples must be positive")
    end

    cuts_count = problem.cuts_count
    inflows = rand(problem.inflow_distr, n_samples)

    saa = Model(HiGHS.Optimizer)
    set_silent(saa)
    set_attribute(saa, "solver", "ipm")
    set_attribute(saa, "time_limit", 60.)
    # set_attribute(saa, "max_wall_time", 60.)

    @variables(saa, begin
        0 <= hydro₁ <= problem.hydro_generation
        0 <= termo₁ <= problem.termo_generation
        0 <= volume₁ <= problem.hydro_capacity
        spill₁ >= 0

        0 <= hydro₂[1:n_samples] <= problem.hydro_generation
        0 <= termo₂[1:n_samples] <= problem.termo_generation
        0 <= volume₂[1:n_samples] <= problem.hydro_capacity
        spill₂[1:n_samples] >= 0
        θ[1:n_samples] >= 0
    end)


    @constraints(saa, begin
        volume₁ == problem.initial_volume + problem.initial_inflow - hydro₁ - spill₁
        hydro₁ <= problem.initial_volume
        hydro₁ + termo₁ == problem.stage_demands[1]
    end)

    for s in 1:n_samples
        @constraints(saa, begin
            volume₂[s] == volume₁ + inflows[s] - hydro₂[s] - spill₂[s]
            hydro₂[s] <= volume₁
            hydro₂[s] + termo₂[s] == problem.stage_demands[2]
            [i in 1:cuts_count], θ[s] >= problem.cuts_coeff[i] * volume₂[s] + problem.cuts_inter[i]
        end)
    end

    @objective(
        saa,
        Min,
        problem.termo_cost_coeffs[1]*(termo₁^2 + (sum(termo₂)/n_samples)^2)
        + problem.termo_cost_coeffs[2]*(termo₁+sum(termo₂)/n_samples)
        + 2*problem.termo_cost_coeffs[3]
        + sum(θ[s] for s in 1:n_samples) / n_samples,
    )

    dt = @elapsed optimize!(saa)

    return (;
        Obj = objective_value(saa),
        Bound = -Inf,
        H₁ = value(hydro₁),
        V₁ = value(volume₁),
        T₁ = value(termo₁),
        dt,
    )
end

function compare_methods(problem::HydroProblem)
    solvers = [
        "EV" => expected_value_solution
        ["LDR ($n breakpoints)" => p -> ldr_solution(p, n) for n in 0:3]
        ["SAA ($n samples)" => p -> saa_solution(p, n) for
         n in repeat([4, 10, 30, 100]; inner = 3)]
    ]

    return run_comparison(problem, solvers; evaluate = evaluate_policy)
end

function build_toy_problem()
    h_generation = 100.
    h_capacity = 300.
    t_generation = 20.
    stage_demands = [70., 70.]
    initial_vol = 100.
    initial_inflow = 60.
    inflow_distr = truncated(Normal(60., 20.), 0., 120)
    cuts_coeff = [-300., -600., -1200., -2500., -7800.]
    cuts_inter = [75000., 135000., 210000., 310000., 390000.]
    t_cost_coeffs = [121.30*0.51^2/2, 0., 0.]

    return HydroProblem(
        h_generation,
        h_capacity,
        t_generation,
        t_cost_coeffs,
        stage_demands,
        initial_vol,
        initial_inflow,
        inflow_distr,
        cuts_coeff,
        cuts_inter
    )
end

function main()
    toy = build_toy_problem()
    compare_methods(toy)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    HydroQuad.main()
end
