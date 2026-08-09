using Random
using Printf

using JuMP
using HiGHS
using Distributions
using LinearDecisionRules

# Two-stage hydro-thermal toy model:
# - Stage 1 inflow is deterministic (initial_inflow).
# - Stage 2 inflow is stochastic (inflow_distr).
# - Demand must be fully met at each stage.
# - Future cost is represented by piecewise-linear cuts on final storage.
#
# This file is intentionally organized in the same style as ev_ldr_saa.jl:
# evaluate policy -> expected value model -> LDR model -> (next commits) SAA + comparison.

struct HydroProblem
    hydro_generation::Float64
    hydro_capacity::Float64
    termo_count::Int
    termo_generations::Vector{Float64}
    termo_costs::Vector{Float64}
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
        termo_generations::Vector{Float64},
        termo_costs::Vector{Float64},
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
        if length(termo_generations) != length(termo_costs)
            error("Mismatch between quantity of parameters for the termos")
        end
        if any(termo_generations .<= 0) || any(termo_costs .<= 0.)
            error("Termo parameters must be positive values")
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

        termo_count = length(termo_costs)
        cuts_count = length(cuts_coeff)

        return new(
            hydro_generation,
            hydro_capacity,
            termo_count,
            termo_generations,
            termo_costs,
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

"""
Evaluate second-stage recourse cost for out-of-sample policy assessment.

Inputs:
- volume_stage2_in: storage entering stage 2
- inflow_stage2: sampled stage-2 inflow

Returns:
- NamedTuple with solver status and all relevant stage-2 outputs.
"""
function evaluate_firststage(
    problem::HydroProblem,
    volume_stage2_in::Real,
    inflow_stage2::Real,
)

    if problem.second_stage[] !== nothing
        fs = problem.second_stage[]
    else
        termo_count = length(problem.termo_costs)
        cuts_count = length(problem.cuts_coeff)

        fs = Model(HiGHS.Optimizer)
        set_silent(fs)

        @variable(fs, hydro >= 0)
        @variable(fs, termos[1:termo_count] >= 0)
        @variable(fs, volume₂ >= 0)
        @variable(fs, spill >= 0)
        @variable(fs, θ >= 0)

        @variable(fs, volume₁)
        @variable(fs, inflow₂)

        @constraint(fs, volume₂ == volume₁ + inflow₂ - hydro - spill)
        @constraint(fs, volume₂ <= problem.hydro_capacity)
        @constraint(fs, termos .<= problem.termo_generations)
        @constraint(fs, hydro <= volume₁)
        @constraint(fs, hydro <= problem.hydro_generation)
        @constraint(fs, hydro + sum(termos[i] for i in 1:termo_count) == problem.stage_demands[2])
        @constraint(fs, [i in 1:cuts_count], θ >= problem.cuts_coeff[i]*volume₂ + problem.cuts_inter[i])

        @objective(fs, Min, sum(problem.termo_costs[i] * termos[i] for i in 1:termo_count) + θ)

        problem.second_stage[] = fs
    end

    fix(fs[:volume₁], volume_stage2_in)
    fix(fs[:inflow₂], inflow_stage2)
    optimize!(fs)
    return (
        status = termination_status(fs),
        objective = objective_value(fs),
        hydro = value(fs[:hydro]),
        volume_final = value(fs[:volume₂]),
        thermal = value.(fs[:termos]),
        spill = value(fs[:spill]),
        future_cost = value(fs[:θ]),
    )
end

function evaluate_firststage(problem::HydroProblem, volume₁::Float64, N_samples::Int; seed=0)
    rng = MersenneTwister(seed)
    inflows = rand(rng, problem.inflow_distr, N_samples)

    costs = zeros(N_samples)
    hydro₂ = zeros(N_samples)
    volume₂ = zeros(N_samples)
    thermal₂ = zeros(N_samples, problem.termo_count)
    for s in 1:N_samples
        res = evaluate_firststage(problem, volume₁, inflows[s])
        costs[s] = res.objective
        hydro₂[s] = res.hydro
        volume₂[s] = res.volume_final
        thermal₂[s,:] = res.thermal
    end

    fs_obj = mean(costs)
    fs_std = std(costs)/sqrt(N_samples)
    fs_hydro = sum(hydro₂)/N_samples
    fs_vol = sum(volume₂)/N_samples
    fs_termos = sum(thermal₂, dims=1)/N_samples
    return fs_obj, fs_std, fs_hydro, fs_vol, fs_termos
end

"""
Deterministic expected-value approximation:
- Replaces stochastic stage-2 inflow by its mean.
- Solves a two-stage deterministic JuMP model.
"""
function expected_value_solution(problem::HydroProblem)
    inflows = [problem.initial_inflow, mean(problem.inflow_distr)]

    ev = Model(HiGHS.Optimizer)
    set_silent(ev)

    @variable(ev, hydro[1:2] >= 0)
    @variable(ev, termos[1:2, 1:problem.termo_count] >= 0)
    @variable(ev, volume[1:2] >= 0)
    @variable(ev, spill[1:2] >= 0)
    @variable(ev, θ >= 0)

    @constraint(ev, [i in 1:2], volume[i] <= problem.hydro_capacity)
    @constraint(ev, volume[1] == problem.initial_volume + inflows[1] - hydro[1] - spill[1])
    @constraint(ev, volume[2] == volume[1] + inflows[2] - hydro[2] - spill[2])
    @constraint(ev, hydro[1] <= problem.initial_volume)
    @constraint(ev, hydro[2] <= volume[1])
    @constraint(ev, [i in 1:2], hydro[i] <= problem.hydro_generation)
    @constraint(ev, [i in 1:2], termos[i,:] .<= problem.termo_generations)
    @constraint(ev, [i in 1:2], hydro[i] + sum(termos[i,j] for j in 1:problem.termo_count) == problem.stage_demands[i])
    @constraint(ev, [i in 1:problem.cuts_count], θ >= problem.cuts_coeff[i]*volume[2] + problem.cuts_inter[i])

    @objective(ev,
               Min,
               sum(problem.termo_costs[i] * (termos[1,i] + termos[2,i]) for i in 1:problem.termo_count) + θ)

    dt = @elapsed optimize!(ev)
    ev_obj = objective_value(ev)
    ev_hydro₁ = value(hydro[1])
    ev_vol₁ = value(volume[1])
    ev_termos₁ = value(termos[1, :])

    return dt, ev_obj, ev_hydro₁, ev_vol₁, ev_termos₁
end

"""
LDR model where stage-2 decisions depend on stochastic inflow.
"""
function ldr_solution(problem::HydroProblem, n_breakpoints=0)
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, hydro₁ >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, termos₁[1:problem.termo_count] >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, volume₁ >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, spill₁ >= 0, LinearDecisionRules.FirstStage)
    @variable(
        ldr,
        inflow in
        LinearDecisionRules.Uncertainty(; distribution = problem.inflow_distr)
    )
    @variable(ldr, hydro₂ >= 0)
    @variable(ldr, termos₂[1:problem.termo_count] >= 0)
    @variable(ldr, volume₂ >= 0)
    @variable(ldr, spill₂ >= 0)
    @variable(ldr, θ >= 0)


    @constraint(ldr, volume₁ <= problem.hydro_capacity)
    @constraint(ldr, volume₂ <= problem.hydro_capacity)
    @constraint(ldr, volume₁ == problem.initial_volume + problem.initial_inflow - hydro₁ - spill₁)
    @constraint(ldr, volume₂ == volume₁ + inflow - hydro₂ - spill₂)
    @constraint(ldr, hydro₁ <= problem.initial_volume)
    @constraint(ldr, hydro₂ <= volume₁)
    @constraint(ldr, hydro₁ <= problem.hydro_generation)
    @constraint(ldr, hydro₂ <= problem.hydro_generation)
    @constraint(ldr, termos₁ .<= problem.termo_generations)
    @constraint(ldr, termos₂ .<= problem.termo_generations)
    @constraint(ldr, hydro₁ + sum(termos₁[j] for j in 1:problem.termo_count) == problem.stage_demands[1])
    @constraint(ldr, hydro₂ + sum(termos₂[j] for j in 1:problem.termo_count) == problem.stage_demands[2])
    @constraint(ldr, [i in 1:problem.cuts_count], θ >= problem.cuts_coeff[i]*volume₂ + problem.cuts_inter[i])

    @objective(ldr,
               Min,
               sum(problem.termo_costs[i] * (termos₁[i] + termos₂[i]) for i in 1:problem.termo_count) + θ)

    # Set breakpoints for uncertainty approximation.
    if n_breakpoints > 0
        set_attribute(inflow, LinearDecisionRules.BreakPoints(), n_breakpoints)
    end

    dt = @elapsed optimize!(ldr)
    ldr_p_obj = objective_value(ldr)
    ldr_d_obj = objective_value(ldr; dual = true)
    ldr_hydro₁ = LinearDecisionRules.get_decision(ldr, hydro₁)
    ldr_vol₁ = LinearDecisionRules.get_decision(ldr, volume₁)

    ldr_termos₁ = [
        LinearDecisionRules.get_decision(ldr, termos₁[i]) for i in 1:problem.termo_count
    ]


    return dt, ldr_p_obj, ldr_d_obj, ldr_hydro₁, ldr_vol₁, ldr_termos₁
end


"""
Sample Average Approximation (SAA) for the two-stage hydro-thermal model.

- Stage-2 inflow is sampled from problem.inflow_distr.
- Objective uses the sample average of second-stage thermal cost and future cost.

Returns:
- objective value and first-stage decisions.
"""
function saa_solution(problem::HydroProblem, n_samples::Int)
    if n_samples <= 0
        error("n_samples must be positive")
    end

    termo_count = problem.termo_count
    cuts_count = problem.cuts_count
    inflows = rand(problem.inflow_distr, n_samples)

    saa = Model(HiGHS.Optimizer)
    set_silent(saa)

    @variable(saa, hydro₁ >= 0)
    @variable(saa, termos₁[1:termo_count] >= 0)
    @variable(saa, volume₁ >= 0)
    @variable(saa, spill₁ >= 0)

    @variable(saa, hydro₂[1:n_samples] >= 0)
    @variable(saa, termos₂[1:n_samples, 1:termo_count] >= 0)
    @variable(saa, volume₂[1:n_samples] >= 0)
    @variable(saa, spill₂[1:n_samples] >= 0)
    @variable(saa, θ[1:n_samples] >= 0)

    @constraint(saa, volume₁ <= problem.hydro_capacity)
    @constraint(saa, volume₁ == problem.initial_volume + problem.initial_inflow - hydro₁ - spill₁)
    @constraint(saa, hydro₁ <= problem.initial_volume)
    @constraint(saa, hydro₁ <= problem.hydro_generation)
    @constraint(saa, termos₁ .<= problem.termo_generations)
    @constraint(saa, hydro₁ + sum(termos₁[j] for j in 1:termo_count) == problem.stage_demands[1])

    for s in 1:n_samples
        @constraint(saa, volume₂[s] <= problem.hydro_capacity)
        @constraint(saa, volume₂[s] == volume₁ + inflows[s] - hydro₂[s] - spill₂[s])
        @constraint(saa, hydro₂[s] <= volume₁)
        @constraint(saa, hydro₂[s] <= problem.hydro_generation)
        @constraint(saa, termos₂[s, :] .<= problem.termo_generations)
        @constraint(saa, hydro₂[s] + sum(termos₂[s, j] for j in 1:termo_count) == problem.stage_demands[2])
        @constraint(saa, [i in 1:cuts_count], θ[s] >= problem.cuts_coeff[i] * volume₂[s] + problem.cuts_inter[i])
    end

    @objective(
        saa,
        Min,
        sum(problem.termo_costs[j] * termos₁[j] for j in 1:termo_count) +
        sum(
            sum(problem.termo_costs[j] * termos₂[s, j] for j in 1:termo_count) + θ[s]
            for s in 1:n_samples
        ) / n_samples,
    )

    dt = @elapsed optimize!(saa)
    saa_obj = objective_value(saa)
    saa_hydro₁ = value(hydro₁)
    saa_vol₁ = value(volume₁)
    saa_termos₁ = value.(termos₁)

    return dt, saa_obj, saa_hydro₁, saa_vol₁, saa_termos₁
end


struct TestResult
    method::String
    obj::Float64
    hydro::Float64
    volume::Float64
    termos::Vector{Float64}
    hydro₂::Float64
    volume₂::Float64
    termos₂::Vector{Float64}
    perf::Float64
    std_perf::Float64
    bound::Float64
    time::Float64
end


function compare_methods(problem::HydroProblem)
    results = TestResult[]
    scenarios_count = 10000

    # --- EV ---
    ev_dt, ev_obj, ev_hydro₁, ev_vol₁, ev_termos₁ = expected_value_solution(problem)
    ev_fst, ev_std, ev_hydro₂, ev_vol₂, ev_termos₂ = evaluate_firststage(problem, ev_vol₁, scenarios_count)
    ev_perf = ev_fst + sum(ev_termos₁[i]*problem.termo_costs[i] for i in 1:problem.termo_count)
    push!(results,
          TestResult(
              "EV",
              ev_obj,
              ev_hydro₁,
              ev_vol₁,
              ev_termos₁,
              ev_hydro₂,
              ev_vol₂,
              vec(ev_termos₂),
              ev_perf,
              ev_std,
              +Inf,
              ev_dt
          )
    )

    # --- LDR with different numbers of breakpoints ---
    for n_breakpoints in 0:3
        ldr_dt, ldr_p_obj, ldr_d_obj, ldr_hydro₁, ldr_vol₁, ldr_termos₁ =
            ldr_solution(problem, n_breakpoints)
        ldr_fst, ldr_std, ldr_hydro₂, ldr_vol₂, ldr_termos₂ = evaluate_firststage(problem, ldr_vol₁, scenarios_count)
        ldr_perf = ldr_fst + sum(ldr_termos₁[i]*problem.termo_costs[i] for i in 1:problem.termo_count)
        push!(
            results,
            TestResult(
                "LDR ($n_breakpoints breakpoints)",
                ldr_p_obj,
                ldr_hydro₁,
                ldr_vol₁,
                ldr_termos₁,
                ldr_hydro₂,
                ldr_vol₂,
                vec(ldr_termos₂),
                ldr_perf,
                ldr_std,
                ldr_d_obj,
                ldr_dt,
            ),
        )
    end

    # --- SAA with multiple sample sizes ---
    for N_samples in repeat([4, 10, 30, 100]; inner = 3)
        saa_dt, saa_obj, saa_hydro₁, saa_vol₁, saa_termos₁ = saa_solution(problem, N_samples)
        saa_fst, saa_std, saa_hydro₂, saa_vol₂, saa_termos₂ = evaluate_firststage(problem, saa_vol₁, scenarios_count)
        saa_perf = saa_fst + sum(saa_termos₁[i]*problem.termo_costs[i] for i in 1:problem.termo_count)
        push!(
            results,
            TestResult(
                "SAA ($N_samples samples)",
                saa_obj,
                saa_hydro₁,
                saa_vol₁,
                saa_termos₁,
                saa_hydro₂,
                saa_vol₂,
                vec(saa_termos₂),
                saa_perf,
                saa_std,
                +Inf,
                saa_dt,
            ),
        )
    end

    # --- Print table ---
    n = problem.termo_count
    t1_headers = join(["   T$(i)₁" for i in 1:n], "  ")
    t2_headers = join(["   T$(i)₂" for i in 1:n], "  ")
    header = @sprintf(
        "%-25s  %9s  %7s  %7s  %-s  %7s  %7s  %-s  %9s %9s  %9s  %11s",
        "Method", "Obj", "H₁", "V₁", t1_headers, "H₂", "V₂", t2_headers, "Perf", "Std", "Bound", "Time (ms)"
    )
    println(header)
    println(repeat("-", length(header)))
    for r in results
        t1_str = join([@sprintf("%6.2f", t) for t in r.termos],  "  ")
        t2_str = join([@sprintf("%6.2f", t) for t in r.termos₂], "  ")
        @printf(
            "%-25s  %9.2f  %7.2f  %7.2f  %-s  %7.2f  %7.2f  %-s  %9.2f %9.2f  %9.2f  %11.2f\n",
            r.method, r.obj, r.hydro, r.volume, t1_str,
            r.hydro₂, r.volume₂, t2_str,
            r.perf, r.std_perf, isinf(r.bound) ? Inf : r.bound, 1000 * r.time
        )
    end

    return results
end

"""
Create a reproducible baseline case for method-to-method comparisons.

The baseline is intentionally small and stable so that EV/LDR/SAA outputs can
be compared across commits without changing input data.
"""
function build_baseline_problem()
    hydro_generation = 5.0
    hydro_capacity = 100.0
    termo_generations = [20.0, 20.0, 20.0]
    termo_costs = [10.0, 20.0, 30.0]
    stage_demands = [30.0, 30.0]
    initial_volume = 50.0
    initial_inflow = 10.0
    inflow_distr = Uniform(5.0, 15.0)
    cuts_coeff = [0.0, 0.1]
    cuts_inter = [0.0, 1.0]

    return HydroProblem(
        hydro_generation,
        hydro_capacity,
        termo_generations,
        termo_costs,
        stage_demands,
        initial_volume,
        initial_inflow,
        inflow_distr,
        cuts_coeff,
        cuts_inter,
    )
end

function build_toy_problem()
    h_generation = 100.
    h_capacity = 300.
    t_generations = [10., 5., 5.]
    t_costs = [250., 750., 1750.]
    stage_demands = [70., 70.]
    initial_vol = 100.
    initial_inflow = 60.
    #inflow_distr = Normal(60., 20.)
    inflow_distr = truncated(Normal(60., 20.), 0., 120)
    cuts_coeff = [-300., -600., -1200., -2500., -7800.]
    cuts_inter = [75000., 135000., 210000., 310000., 390000.]

    return HydroProblem(
        h_generation,
        h_capacity,
        t_generations,
        t_costs,
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

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
