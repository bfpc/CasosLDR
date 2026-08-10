module Newsvendor

using Random
using Printf

using JuMP
using HiGHS
using Distributions
using LinearDecisionRules

struct NewsvendorProblem
    buy_cost::Float64
    return_value::Float64
    sell_value::Float64
    demand_distr::Distribution

    function NewsvendorProblem(
        buy_cost::Float64,
        return_value::Float64,
        sell_value::Float64,
        demand_distr::Distribution,
    )
        if sell_value < buy_cost
            error("Sell value must be at least as large as buy cost.")
        end
        if return_value > buy_cost
            error("Return value cannot exceed buy cost.")
        end
        return new(buy_cost, return_value, sell_value, demand_distr)
    end
end

struct TestResult
    method::String
    obj::Float64
    buy::Float64
    perf::Float64
    bound::Float64
    time::Float64
end

# Analytical expected profit for a given first-stage order quantity `x`
function evaluate_firststage(problem::NewsvendorProblem, x)
    distr = problem.demand_distr
    truc_distr = Truncated(distr, minimum(distr), x)
    delta = problem.sell_value - problem.return_value

    return x * (problem.return_value - problem.buy_cost) +
           delta * (x * ccdf(distr, x) + mean(truc_distr) * cdf(distr, x))
end

# Monte Carlo estimate of expected profit for a given first-stage order quantity `x`
function evaluate_firststage(problem::NewsvendorProblem, x, N_samples)
    demands = rand(problem.demand_distr, N_samples)

    total_profit = 0.0
    for d in demands
        sell = min(x, d)
        ret = max(0, x - sell)
        profit =
            -problem.buy_cost * x +
            problem.return_value * ret +
            problem.sell_value * sell
        total_profit += profit
    end

    expected_profit = total_profit / N_samples
    return expected_profit
end

# Deterministic problem, using expected demand
function expected_value_solution(problem::NewsvendorProblem)
    E_demand = mean(problem.demand_distr)

    ev = Model(HiGHS.Optimizer)
    set_silent(ev)

    @variable(ev, buy >= 0)
    @variable(ev, sell >= 0)
    @variable(ev, ret >= 0)

    @constraint(ev, sell + ret <= buy)
    @constraint(ev, sell <= E_demand)

    @objective(
        ev,
        Max,
        -problem.buy_cost * buy +
        problem.return_value * ret +
        problem.sell_value * sell
    )

    dt = @elapsed optimize!(ev)
    ev_obj = objective_value(ev)
    ev_buy = value(buy)

    return dt, ev_obj, ev_buy
end

# LDR solution, with optional breakpoints for piecewise linear decision rules
function ldr_solution(problem::NewsvendorProblem; n_breakpoints = 0)
    ldr = LinearDecisionRules.LDRModel(HiGHS.Optimizer)
    set_silent(ldr)

    @variable(ldr, buy >= 0, LinearDecisionRules.FirstStage)
    @variable(ldr, sell >= 0)
    @variable(ldr, ret >= 0)
    @variable(
        ldr,
        demand in
        LinearDecisionRules.Uncertainty(; distribution = problem.demand_distr)
    )

    @constraint(ldr, sell + ret <= buy)
    @constraint(ldr, sell <= demand)

    @objective(
        ldr,
        Max,
        -problem.buy_cost * buy +
        problem.return_value * ret +
        problem.sell_value * sell
    )

    # Set breakpoints
    if n_breakpoints > 0
        set_attribute(demand, LinearDecisionRules.BreakPoints(), n_breakpoints)
    end

    dt = @elapsed optimize!(ldr)
    ldr_p_obj = objective_value(ldr)
    ldr_d_obj = objective_value(ldr; dual = true)
    ldr_buy = LinearDecisionRules.get_decision(ldr, buy)

    return dt, ldr_p_obj, ldr_buy, ldr_d_obj
end

# SAA solution with `N_samples` demand scenarios
function saa_solution(problem::NewsvendorProblem, N_samples)
    saa = Model(HiGHS.Optimizer)
    set_silent(saa)

    @variable(saa, buy >= 0)
    @variable(saa, sell[i = 1:N_samples] >= 0)
    @variable(saa, ret[i = 1:N_samples] >= 0)

    # Sample demand scenarios
    demands = rand(problem.demand_distr, N_samples)

    for i in 1:N_samples
        @constraint(saa, sell[i] + ret[i] <= buy)
        @constraint(saa, sell[i] <= demands[i])
    end

    @objective(
        saa,
        Max,
        -problem.buy_cost * buy +
        (problem.return_value * sum(ret) + problem.sell_value * sum(sell)) /
        N_samples
    )

    dt = @elapsed optimize!(saa)
    saa_obj = objective_value(saa)
    saa_buy = value(buy)

    return dt, saa_obj, saa_buy
end


function compare_methods(problem::NewsvendorProblem)
    results = TestResult[]
    # Solve expected value SAA problem
    ev_dt, ev_obj, ev_buy = expected_value_solution(problem)
    ev_perf = evaluate_firststage(problem, ev_buy)
    push!(results, TestResult("EV", ev_obj, ev_buy, ev_perf, +Inf, ev_dt))

    # LDR / PWLDR with different numbers of breakpoints
    for n_breakpoints in 0:3
        ldr_dt, ldr_p_obj, ldr_buy, ldr_d_obj =
            ldr_solution(problem; n_breakpoints = n_breakpoints)
        ldr_perf = evaluate_firststage(problem, ldr_buy)
        push!(
            results,
            TestResult(
                "LDR ($n_breakpoints breakpoints)",
                ldr_p_obj,
                ldr_buy,
                ldr_perf,
                ldr_d_obj,
                ldr_dt,
            ),
        )
    end

    # SAA problems
    for N_samples in repeat([4, 10, 30, 100]; inner = 3)
        saa_dt, saa_obj, saa_buy = saa_solution(problem, N_samples)
        saa_perf = evaluate_firststage(problem, saa_buy)
        push!(
            results,
            TestResult(
                "SAA ($N_samples samples)",
                saa_obj,
                saa_buy,
                saa_perf,
                +Inf,
                saa_dt,
            ),
        )
    end

    # Print results
    # Use Printf for decent formatting
    @printf(
        "%-25s & %10s & %10s & %10s & %10s & %10s\n",
        "Method",
        "Obj",
        "Buy",
        "Perf",
        "Bound",
        "Time (ms)"
    )
    for res in results
        @printf(
            "%-25s & %10.2f & %10.2f & %10.2f & %10.2f & %10.2f\n",
            res.method,
            res.obj,
            res.buy,
            res.perf,
            res.bound,
            1000 * res.time
        )
    end
    return
end

function main()
    # LDR loses to EV in re-optimization
    problem1 = NewsvendorProblem(
        10.0, # buy_cost
        8.0,  # return_value
        15.0, # sell_value
        Uniform(80, 120), # demand_distr
    )
    compare_methods(problem1)

    println(repeat("=", 90))

    # LDR wins over EV in re-optimization
    problem2 = NewsvendorProblem(
        10.0, # buy_cost
        8.0,  # return_value
        20.0, # sell_value
        Uniform(80, 120), # demand_distr
    )
    compare_methods(problem2)

    return
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Newsvendor.main()
end
