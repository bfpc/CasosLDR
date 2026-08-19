module HydroQuad

using Random
using Printf

using JuMP
using HiGHS, Ipopt
using Distributions
using LinearDecisionRules

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

function evaluate_firststage(
    problem::HydroProblem,
    volume_stage2_in::Real,
    inflow_stage2::Real,
)

    if problem.second_stage[] !== nothing
        fs = problem.second_stage[]
    else
        cost_coeffs = problem.termo_cost_coeffs
        cuts_count = length(problem.cuts_coeff)

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
	thermal = value(fs[:termo]),
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
    thermal₂ = zeros(N_samples)
    for s in 1:N_samples
        res = evaluate_firststage(problem, volume₁, inflows[s])
        costs[s] = res.objective
        hydro₂[s] = res.hydro
        volume₂[s] = res.volume_final
        thermal₂[s] = res.thermal
    end

    fs_obj = mean(costs)
    fs_std = std(costs)/sqrt(N_samples)
    fs_hydro = sum(hydro₂)/N_samples
    fs_vol = sum(volume₂)/N_samples
    fs_termo = sum(thermal₂)/N_samples
    return fs_obj, fs_std, fs_hydro, fs_vol, fs_termo
end

function expected_value_solution(problem::HydroProblem)
    inflows = [problem.initial_inflow, mean(problem.inflow_distr)]
    cost_coeffs = problem.termo_cost_coeffs
    
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
    ev_obj = objective_value(ev)
    ev_hydro₁ = value(hydro[1])
    ev_vol₁ = value(volume[1])
    ev_termo₁ = value(termo[1])

    return dt, ev_obj, ev_hydro₁, ev_vol₁, ev_termo₁
end

function ldr_solution(problem::HydroProblem, n_breakpoints=0)
    cost_coeffs = problem.termo_cost_coeffs
    
    ldr = LinearDecisionRules.LDRModel(optimizer_with_attributes(
        HiGHS.Optimizer,
        "solver" => "ipm",
        "time_limit" => 60.,
        # "max_wall_time" => 60.,
    ))
    #set_silent(ldr)
    
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
    ldr_p_obj = objective_value(ldr)
    ldr_d_obj = objective_value(ldr; dual = true)
    ldr_hydro₁ = LinearDecisionRules.get_decision(ldr, hydro₁)
    ldr_vol₁ = LinearDecisionRules.get_decision(ldr, volume₁)
    ldr_termo₁ = LinearDecisionRules.get_decision(ldr, termo₁)

    return dt, ldr_p_obj, ldr_d_obj, ldr_hydro₁, ldr_vol₁, ldr_termo₁
end

function saa_solution(problem::HydroProblem, n_samples::Int)
    if n_samples <= 0
        error("n_samples must be positive")
    end

    cuts_count = problem.cuts_count
    inflows = rand(problem.inflow_distr, n_samples)
    cost_coeffs = problem.termo_cost_coeffs

    saa = Model(HiGHS.Optimizer)
    #set_silent(saa)
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
    saa_obj = objective_value(saa)
    saa_hydro₁ = value(hydro₁)
    saa_vol₁ = value(volume₁)
    saa_termo₁ = value(termo₁)

    return dt, saa_obj, saa_hydro₁, saa_vol₁, saa_termo₁
end

struct TestResult
    method::String
    obj::Float64
    hydro::Float64
    volume::Float64
    termo::Float64
    hydro₂::Float64
    volume₂::Float64
    termo₂::Float64
    perf::Float64
    std_perf::Float64
    bound::Float64
    time::Float64
end


function compare_methods(problem::HydroProblem)
    results = TestResult[]
    scenarios_count = 10000
    function quad_cost(x)
   	 return problem.termo_cost_coeffs[1]*x^2 
	 + problem.termo_cost_coeffs[2]*x 
	 + problem.termo_cost_coeffs[3]
    end

    # --- EV ---
    print("Solving EV model...")
    ev_dt, ev_obj, ev_hydro₁, ev_vol₁, ev_termo₁ = expected_value_solution(problem)
    print(" evaluating policy...")
    ev_fst, ev_std, ev_hydro₂, ev_vol₂, ev_termo₂ = evaluate_firststage(problem, ev_vol₁, scenarios_count)
    ev_perf = ev_fst + quad_cost(ev_termo₁)
    push!(results,
          TestResult(
              "EV",
              ev_obj,
              ev_hydro₁,
              ev_vol₁,
              ev_termo₁,
              ev_hydro₂,
              ev_vol₂,
              ev_termo₂,
              ev_perf,
              ev_std,
              +Inf,
              ev_dt
          )
    )
    println(" done.")

    # --- LDR with different numbers of breakpoints ---
    print("Solving LDR models...")
    for n_breakpoints in 0:3
        print(" $n_breakpoints ...")
        ldr_dt, ldr_p_obj, ldr_d_obj, ldr_hydro₁, ldr_vol₁, ldr_termo₁ =
            ldr_solution(problem, n_breakpoints)
        print(" evaluating policy...")
        ldr_fst, ldr_std, ldr_hydro₂, ldr_vol₂, ldr_termo₂ = evaluate_firststage(problem, ldr_vol₁, scenarios_count)
        ldr_perf = ldr_fst + quad_cost(ldr_termo₁)
        push!(
            results,
            TestResult(
                "LDR ($n_breakpoints breakpoints)",
                ldr_p_obj,
                ldr_hydro₁,
                ldr_vol₁,
                ldr_termo₁,
                ldr_hydro₂,
                ldr_vol₂,
                ldr_termo₂,
                ldr_perf,
                ldr_std,
                ldr_d_obj,
                ldr_dt,
            ),
        )
    end
    println(" done.")

    # --- SAA with multiple sample sizes ---
    print("Solving SAA models...")
    for N_samples in repeat([4, 10, 30, 100]; inner = 3)
        print(" $N_samples,")
        saa_dt, saa_obj, saa_hydro₁, saa_vol₁, saa_termo₁ = saa_solution(problem, N_samples)
        saa_fst, saa_std, saa_hydro₂, saa_vol₂, saa_termo₂ = evaluate_firststage(problem, saa_vol₁, scenarios_count)
        saa_perf = saa_fst +  quad_cost(saa_termo₁)
        push!(
            results,
            TestResult(
                "SAA ($N_samples samples)",
                saa_obj,
                saa_hydro₁,
                saa_vol₁,
                saa_termo₁,
                saa_hydro₂,
                saa_vol₂,
                saa_termo₂,
                saa_perf,
                saa_std,
                +Inf,
                saa_dt,
            ),
        )
    end
    println(" done.")

    # --- Print table ---
    header = @sprintf(
        "%-25s  %9s  %7s  %7s  %7s  %7s  %7s  %7s  %9s %9s  %9s  %11s",
        "Method", "Obj", "H₁", "V₁", "T₁", "H₂", "V₂", "T₂", "Perf", "Std", "Bound", "Time (ms)"
    )
    println(header)
    println(repeat("-", length(header)))
    for r in results
        @printf(
            "%-25s  %9.2f  %7.2f  %7.2f  %7.2f  %7.2f  %7.2f  %7.2f  %9.2f %9.2f  %9.2f  %11.2f\n",
            r.method, r.obj, r.hydro, r.volume, r.termo,
            r.hydro₂, r.volume₂, r.termo₂,
            r.perf, r.std_perf, isinf(r.bound) ? Inf : r.bound, 1000 * r.time
        )
    end

    return results
end

function build_toy_problem()
    h_generation = 100.
    h_capacity = 300.
    t_generation = 20.
    t_costs = [250., 750., 1750.]
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
