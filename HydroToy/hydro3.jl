using Random
using Printf

using JuMP
using HiGHS
using Distributions
using LinearDecisionRules
import SDDP

# 3-stage hydro-thermal toy model:
# - Stage 1 inflow is deterministic (initial_inflow).
# - Stages 2 and 3 inflow is stochastic (inflow_distr).
# - Demand must be fully met at each stage.
# - Future cost is represented by piecewise-linear cuts on final storage.

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
            error("Mismatch between quantity of parameters for the termal plants")
        end
        if any(termo_generations .<= 0) || any(termo_costs .<= 0.)
            error("Termo parameters must be positive values")
        end
        if length(stage_demands) != 3
            error("stage_demands should have only three values; only three stages problems are supported")
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
- problem: HydroProblem instance
- volume_stage2_in: storage entering stage 2
- inflow_stage2: sampled stage-2 inflow
- stage3_inflows: sampled stage-3 inflows (optional, defaults to 100 samples from problem.inflow_distr)

Returns:
- NamedTuple with solver status and all relevant stage-2 outputs.
"""
function evaluate_firststage(
    problem::HydroProblem,
    volume_stage2_in::Real,
    inflow_stage2::Real;
    stage3_inflows::Vector{Float64} = rand(problem.inflow_distr, 100)
)

    Nscen = length(stage3_inflows)
    if problem.second_stage[] !== nothing
        recourse = problem.second_stage[]
        Nscen = recourse.ext[:Nscen]
    else
        termo_count = length(problem.termo_costs)
        cuts_count = length(problem.cuts_coeff)

        recourse = Model(HiGHS.Optimizer)
        set_silent(recourse)

        @variable(recourse, 0 <= volume₂ <= problem.hydro_capacity)
        @variable(recourse, 0 <= hydro₂ <= problem.hydro_generation)
        @variable(recourse, spill₂ >= 0)
        @variable(recourse, 0 <= termos₂[i=1:termo_count] <= problem.termo_generations[i])

        @variable(recourse, 0 <= volume₃[1:Nscen] <= problem.hydro_capacity)
        @variable(recourse, 0 <= hydro₃[1:Nscen] <= problem.hydro_generation)
        @variable(recourse, 0 <= spill₃[1:Nscen])
        @variable(recourse, 0 <= termos₃[i=1:termo_count, 1:Nscen] <= problem.termo_generations[i])

        @variable(recourse, θ[1:Nscen] >= 0)

        # Fake varibles to allow faster solve
        @variable(recourse, volume₁)
        @variable(recourse, inflow₂)

        # Second stage
        @constraint(recourse, volume₂ == volume₁ + inflow₂ - hydro₂ - spill₂)
        @constraint(recourse, hydro₂ + sum(termos₂) == problem.stage_demands[2])

        # Third stage
        @constraint(recourse, [j=1:Nscen], volume₃[j] == volume₂ + stage3_inflows[j] - hydro₃[j] - spill₃[j])
        @constraint(recourse, [j=1:Nscen], hydro₃[j] + sum(termos₃[:, j]) == problem.stage_demands[3])

        @constraint(recourse, [j=1:Nscen, k in 1:cuts_count], θ[j] >= problem.cuts_coeff[k]*volume₃[j] + problem.cuts_inter[k])

        @expression(recourse, cost_to_go[j=1:Nscen],
            sum(problem.termo_costs[i] * termos₃[i, j] for i in 1:termo_count) + θ[j]
        )
        @expression(recourse, future_cost, sum(cost_to_go) / Nscen)
        @objective(recourse, Min,
            sum(problem.termo_costs[i] * termos₂[i] for i in 1:termo_count) + future_cost
        )

        recourse.ext[:Nscen] = Nscen
        problem.second_stage[] = recourse
    end

    fix(recourse[:volume₁], volume_stage2_in)
    fix(recourse[:inflow₂], inflow_stage2)
    optimize!(recourse)
    return (
        status = termination_status(recourse),
        objective = objective_value(recourse),
        hydro = value(recourse[:hydro₂]),
        volume_final = value(recourse[:volume₂]),
        thermal = value.(recourse[:termos₂]),
        spill = value(recourse[:spill₂]),
        future_cost = value(recourse[:future_cost]),
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
    inflows = [problem.initial_inflow, mean(problem.inflow_distr), mean(problem.inflow_distr)]

    ev = Model(HiGHS.Optimizer)
    set_silent(ev)

    @variable(ev, 0 <= volume[1:3] <= problem.hydro_capacity)
    @variable(ev, 0 <= hydro[1:3] <= problem.hydro_generation)
    @variable(ev, spill[1:3] >= 0)
    @variable(ev, 0 <= termos[t=1:3, i=1:problem.termo_count] <= problem.termo_generations[i])

    @variable(ev, θ >= 0)

    @constraint(ev, volume[1] == problem.initial_volume + inflows[1] - hydro[1] - spill[1])
    @constraint(ev, volume[2] == volume[1] + inflows[2] - hydro[2] - spill[2])
    @constraint(ev, volume[3] == volume[2] + inflows[3] - hydro[3] - spill[3])
    @constraint(ev, [t in 1:3], hydro[t] + sum(termos[t,j] for j in 1:problem.termo_count) == problem.stage_demands[t])
    @constraint(ev, [k in 1:problem.cuts_count], θ >= problem.cuts_coeff[k]*volume[3] + problem.cuts_inter[k])

    @objective(ev,
               Min,
               sum(problem.termo_costs[i] * sum(termos[:,i]) for i in 1:problem.termo_count) + θ)

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

    # Uncertainties
    @variable(
        ldr,
        inflow2 in
        LinearDecisionRules.Uncertainty(; distribution = problem.inflow_distr, stage = 2)
    )
    @variable(
        ldr,
        inflow3 in
        LinearDecisionRules.Uncertainty(; distribution = problem.inflow_distr, stage = 3)
    )
    inflows = [problem.initial_inflow, inflow2, inflow3]

    # Decision variables / constraints / objective (almost equal to deterministic EV model)
    @variable(ldr, 0 <= volume[1:3] <= problem.hydro_capacity)
    @variable(ldr, 0 <= hydro[1:3] <= problem.hydro_generation)
    @variable(ldr, spill[1:3] >= 0)
    @variable(ldr, 0 <= termos[t=1:3, i=1:problem.termo_count] <= problem.termo_generations[i])

    @variable(ldr, θ >= 0)

    @constraint(ldr, volume[1] == problem.initial_volume + inflows[1] - hydro[1] - spill[1])
    @constraint(ldr, volume[2] == volume[1] + inflows[2] - hydro[2] - spill[2])
    @constraint(ldr, volume[3] == volume[2] + inflows[3] - hydro[3] - spill[3])
    @constraint(ldr, [t in 1:3], hydro[t] + sum(termos[t,j] for j in 1:problem.termo_count) == problem.stage_demands[t])
    @constraint(ldr, [k in 1:problem.cuts_count], θ >= problem.cuts_coeff[k]*volume[3] + problem.cuts_inter[k])

    @objective(ldr,
               Min,
               sum(problem.termo_costs[i] * sum(termos[:,i]) for i in 1:problem.termo_count) + θ
    )

    # Stages
    for t in 1:3
        set_attribute(volume[t], LinearDecisionRules.Stage(t))
        set_attribute(hydro[t], LinearDecisionRules.Stage(t))
        set_attribute(spill[t], LinearDecisionRules.Stage(t))
        set_attribute.(termos[t, :], [LinearDecisionRules.Stage(t)])
    end
    set_attribute(θ, LinearDecisionRules.Stage(3))


    # Set breakpoints for uncertainty approximation.
    if n_breakpoints > 0
        set_attribute(inflow2, LinearDecisionRules.BreakPoints(), n_breakpoints)
        set_attribute(inflow3, LinearDecisionRules.BreakPoints(), n_breakpoints)
    end

    dt = @elapsed optimize!(ldr)
    ldr_p_obj = objective_value(ldr)
    ldr_d_obj = objective_value(ldr; dual = true)
    ldr_hydro₁ = LinearDecisionRules.get_decision(ldr, hydro[1])
    ldr_vol₁ = LinearDecisionRules.get_decision(ldr, volume[1])

    ldr_termos₁ = [
        LinearDecisionRules.get_decision(ldr, termos[1,i]) for i in 1:problem.termo_count
    ]

    return dt, ldr_p_obj, ldr_d_obj, ldr_hydro₁, ldr_vol₁, ldr_termos₁
end


"""
Sample Average Approximation (SAA) for a 3-stage hydro-thermal model.

- Inflows for stages 2 and 3 are sampled from problem.inflow_distr.
- Objective uses the sample average of  thermal costs and future cost.

Returns:
- objective value and first-stage decisions.
"""
function saa_solution(problem::HydroProblem, n_samples2::Int, n_samples3::Int = 30)
    if n_samples2 <= 0 || n_samples3 <= 0
        error("n_samples2 and n_samples3 must be positive")
    end

    termo_count = problem.termo_count
    cuts_count = problem.cuts_count
    inflows2 = rand(problem.inflow_distr, n_samples2)
    inflows3 = rand(problem.inflow_distr, n_samples3)

    saa = Model(HiGHS.Optimizer)
    set_silent(saa)

    @variable(saa, 0 <= volume₁ <= problem.hydro_capacity)
    @variable(saa, 0 <= hydro₁ <= problem.hydro_generation)
    @variable(saa, spill₁ >= 0)
    @variable(saa, 0 <= termos₁[i=1:termo_count] <= problem.termo_generations[i])

    @variable(saa, 0 <= volume₂[1:n_samples2] <= problem.hydro_capacity)
    @variable(saa, 0 <= hydro₂[1:n_samples2] <= problem.hydro_generation)
    @variable(saa, spill₂[1:n_samples2] >= 0)
    @variable(saa, 0 <= termos₂[1:n_samples2, i=1:termo_count] <= problem.termo_generations[i])
    
    @variable(saa, 0 <= volume₃[1:n_samples2, 1:n_samples3] <= problem.hydro_capacity)
    @variable(saa, 0 <= hydro₃[1:n_samples2, 1:n_samples3] <= problem.hydro_generation)
    @variable(saa, spill₃[1:n_samples2, 1:n_samples3] >= 0)
    @variable(saa, 0 <= termos₃[1:n_samples2, 1:n_samples3, i=1:termo_count] <= problem.termo_generations[i])
    
    @variable(saa, θ[1:n_samples2, 1:n_samples3] >= 0)

    @constraint(saa, volume₁ == problem.initial_volume + problem.initial_inflow - hydro₁ - spill₁)
    @constraint(saa, hydro₁ + sum(termos₁) == problem.stage_demands[1])

    for s in 1:n_samples2
        @constraint(saa, volume₂[s] == volume₁ + inflows2[s] - hydro₂[s] - spill₂[s])
        @constraint(saa, hydro₂[s] + sum(termos₂[s, j] for j in 1:termo_count) == problem.stage_demands[2])
    end
    for s in 1:n_samples2, t in 1:n_samples3
        @constraint(saa, volume₃[s, t] == volume₂[s] + inflows3[t] - hydro₃[s, t] - spill₃[s, t])
        @constraint(saa, hydro₃[s, t] + sum(termos₃[s, t, j] for j in 1:termo_count) == problem.stage_demands[3])
        @constraint(saa, [k in 1:cuts_count], θ[s,t] >= problem.cuts_coeff[k] * volume₃[s, t] + problem.cuts_inter[k])
    end

    @objective(
        saa,
        Min,
        sum(problem.termo_costs[j] * termos₁[j] for j in 1:termo_count) +
        sum(
            sum(problem.termo_costs[j] * termos₂[s, j] for j in 1:termo_count)
            for s in 1:n_samples2
        ) / n_samples2 +
        sum(
            sum(problem.termo_costs[j] * termos₃[s, t, j] for j in 1:termo_count) + θ[s, t]
            for s in 1:n_samples2, t in 1:n_samples3
        ) / (n_samples2 * n_samples3)
    )

    dt = @elapsed optimize!(saa)
    saa_obj = objective_value(saa)
    saa_hydro₁ = value(hydro₁)
    saa_vol₁ = value(volume₁)
    saa_termos₁ = value.(termos₁)

    return dt, saa_obj, saa_hydro₁, saa_vol₁, saa_termos₁
end

"""
SDDP solution for hydro-thermal model.

- Inflows are sampled from problem.inflow_distr.

Returns:
- objective value and first-stage decisions.
"""
function sddp_solution(problem::HydroProblem, n_samples::Vector{Int})
    T = length(n_samples)
    if any(n_samples .<= 0)
        error("All elements of n_samples must be positive")
    end

    termo_count = problem.termo_count
    cuts_count = problem.cuts_count
    inflows = [rand(problem.inflow_distr, n_samples[t]) for t in 1:T]
    inflows[1] = [problem.initial_inflow]

    pb = SDDP.LinearPolicyGraph(;
        optimizer = HiGHS.Optimizer,
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
    ) do sp, t
        @variable(sp, 0 <= volume <= problem.hydro_capacity, SDDP.State, initial_value = problem.initial_volume)
        @variable(sp, 0 <= hydro <= problem.hydro_generation)
        @variable(sp, spill >= 0)
        @variable(sp, deficit >= 0)
        @variable(sp, 0 <= termos[i=1:termo_count] <= problem.termo_generations[i])

        # Aux variable for inflow modeling
        @variable(sp, inflow)
        SDDP.parameterize(sp, inflows[t]) do ω
            JuMP.fix(inflow, ω)
        end

        @constraint(sp, volume.out == volume.in + inflow - hydro - spill)
        @constraint(sp, hydro + sum(termos) + deficit == problem.stage_demands[t])

        # Last-stage value function
        @variable(sp, θ >= 0)
        if t == T
            @constraint(sp, [k in 1:cuts_count], θ >= problem.cuts_coeff[k] * volume.out + problem.cuts_inter[k]) 
        end

        SDDP.@stageobjective(
            sp,
            sum(problem.termo_costs[j] * termos[j] for j in 1:termo_count) + 8000 * deficit + θ
        )
    end

    dt = @elapsed SDDP.train(pb;
            time_limit = 5,
            print_level = 0,
            cut_deletion_minimum = 50,
    )
    sp1 = pb[1].subproblem
    pb_bound = SDDP.calculate_bound(pb)
    pb_hydro₁ = value(sp1[:hydro])
    pb_vol₁ = value(sp1[:volume].out)
    pb_termos₁ = value.(sp1[:termos])

    return dt, pb_bound, pb_hydro₁, pb_vol₁, pb_termos₁
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
    scenarios_count = 1000

    # --- EV ---
    print("Solving EV model...")
    ev_dt, ev_obj, ev_hydro₁, ev_vol₁, ev_termos₁ = expected_value_solution(problem)
    print(" evaluating policy...")
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
              -Inf,
              ev_dt
          )
    )
    println(" done.")

    # --- LDR with different numbers of breakpoints ---
    print("Solving LDR models:")
    for n_breakpoints in 0:3
        print(" $n_breakpoints,")
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
    println(" done.")

    # --- SAA with multiple sample sizes ---
    print("Solving SAA models:")
    for N_samples in repeat([4, 10, 30, 100]; inner = 3)
        print(" $N_samples,")
        saa_dt, saa_obj, saa_hydro₁, saa_vol₁, saa_termos₁ = saa_solution(problem, N_samples)
        saa_fst, saa_std, saa_hydro₂, saa_vol₂, saa_termos₂ = evaluate_firststage(problem, saa_vol₁, scenarios_count)
        saa_perf = saa_fst + sum(saa_termos₁[i]*problem.termo_costs[i] for i in 1:problem.termo_count)
        push!(
            results,
            TestResult(
                "SAA ($N_samples x 30 samples)",
                saa_obj,
                saa_hydro₁,
                saa_vol₁,
                saa_termos₁,
                saa_hydro₂,
                saa_vol₂,
                vec(saa_termos₂),
                saa_perf,
                saa_std,
                -Inf,
                saa_dt,
            ),
        )
    end
    println(" done.")

    # --- SDDP model with multiple sample sizes ---
    print("Solving SDDP models:")
    for N_samples in repeat([4, 10, 30, 100]; inner = 3)
        print(" $N_samples,")
        sddp_dt, sddp_bound, sddp_hydro₁, sddp_vol₁, sddp_termos₁ = sddp_solution(problem, [1, N_samples, 100])
        sddp_fst, sddp_std, sddp_hydro₂, sddp_vol₂, sddp_termos₂ = evaluate_firststage(problem, sddp_vol₁, scenarios_count)
        sddp_perf = sddp_fst + sum(sddp_termos₁[i]*problem.termo_costs[i] for i in 1:problem.termo_count)
        push!(
            results,
            TestResult(
                "SDDP ($N_samples x 100 samples)",
                +Inf,
                sddp_hydro₁,
                sddp_vol₁,
                sddp_termos₁,
                sddp_hydro₂,
                sddp_vol₂,
                vec(sddp_termos₂),
                sddp_perf,
                sddp_std,
                sddp_bound,
                sddp_dt,
            ),
        )
    end
    println(" done.")

    # --- Print table ---
    n = problem.termo_count
    t1_headers = join(["   T$(i)₁" for i in 1:n], "  ")
    t2_headers = join(["   T$(i)₂" for i in 1:n], "  ")
    header = @sprintf(
        "%-25s  %9s  %9s  %11s | %7s  %7s  %-s | %7s  %7s  %-s | %9s %9s",
        "Method", "Obj", "Bound", "Time (ms)", "H₁", "V₁", t1_headers, "H₂", "V₂", t2_headers, "Perf", "Std"
    )
    println(header)
    println(repeat("-", length(header)))
    for r in results
        t1_str = join([@sprintf("%6.2f", t) for t in r.termos],  "  ")
        t2_str = join([@sprintf("%6.2f", t) for t in r.termos₂], "  ")
        @printf(
            "%-25s  %9.2f  %9.2f  %11.2f | %7.2f  %7.2f  %-s | %7.2f  %7.2f  %-s | %9.2f %9.2f\n",
            r.method, r.obj, r.bound, 1000 * r.time,
            r.hydro, r.volume, t1_str,
            r.hydro₂, r.volume₂, t2_str,
            r.perf, r.std_perf
        )
    end

    return results
end

"""
A small hydro thermal problem roughly in the Brazilian scale.
"""
function build_toy_problem()
    h_generation = 100.
    h_capacity = 300.
    t_generations = [10., 5., 5.]
    t_costs = [250., 750., 1750.]
    stage_demands = [70., 70., 70.]
    initial_vol = 100.
    initial_inflow = 60.
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
