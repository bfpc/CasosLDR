isdefined(Main, :Compare) || include(joinpath(@__DIR__, "..", "src", "Compare.jl"))

module Newsvendor

using Random

using JuMP
using HiGHS
using Distributions
using LinearDecisionRules

using Main.Compare
default_solver = HiGHS.Optimizer

struct NewsvendorProblem
    buy_cost::Float64
    return_value::Float64
    sell_value::Float64
    demand_distr::Distribution
    solver::Any

    function NewsvendorProblem(
        buy_cost::Float64,
        return_value::Float64,
        sell_value::Float64,
        demand_distr::Distribution,
        solver = default_solver
    )
        if sell_value < buy_cost
            error("Sell value must be at least as large as buy cost.")
        end
        if return_value > buy_cost
            error("Return value cannot exceed buy cost.")
        end
        return new(buy_cost, return_value, sell_value, demand_distr, solver)
    end
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

    average, _ = average_scenarios(demands) do d
        sell = min(x, d)
        ret = max(0, x - sell)
        profit =
            -problem.buy_cost * x +
            problem.return_value * ret +
            problem.sell_value * sell
        return (; profit)
    end

    return average.profit
end

# Out-of-sample assessment of the first-stage order quantity of a solution
function evaluate_policy(problem::NewsvendorProblem, sol)
    return (; Perf = evaluate_firststage(problem, sol.Buy))
end

# Deterministic problem, using expected demand
function expected_value_solution(problem::NewsvendorProblem)
    E_demand = mean(problem.demand_distr)

    ev = Model(problem.solver)
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

    return (; Obj = objective_value(ev), Bound = +Inf, Buy = value(buy), dt)
end

# LDR solution, with optional breakpoints for piecewise linear decision rules
function ldr_solution(problem::NewsvendorProblem; n_breakpoints = 0)
    ldr = LinearDecisionRules.LDRModel(problem.solver)
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

    return (;
        Obj = objective_value(ldr),
        Bound = objective_value(ldr; dual = true),
        Buy = LinearDecisionRules.get_decision(ldr, buy),
        dt,
    )
end

# SAA solution with `N_samples` demand scenarios
function saa_solution(problem::NewsvendorProblem, N_samples)
    saa = Model(problem.solver)
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

    return (; Obj = objective_value(saa), Bound = +Inf, Buy = value(buy), dt)
end

function compare_methods(problem::NewsvendorProblem)
    solvers = [
        "EV" => expected_value_solution
        ["LDR ($n breakpoints)" => p -> ldr_solution(p; n_breakpoints = n) for n in 0:3]
        ["SAA ($n samples)" => p -> saa_solution(p, n) for
         n in repeat([4, 10, 30, 100]; inner = 3)]
    ]

    return run_comparison(problem, solvers; evaluate = evaluate_policy, verbose = true)
end

function main(solver = default_solver)
    # LDR loses to EV in re-optimization
    problem1 = NewsvendorProblem(
        10.0, # buy_cost
        8.0,  # return_value
        15.0, # sell_value
        Uniform(80, 120), # demand_distr
        solver,
    )
    compare_methods(problem1)

    println(repeat("=", 90))

    # LDR wins over EV in re-optimization
    problem2 = NewsvendorProblem(
        10.0, # buy_cost
        8.0,  # return_value
        20.0, # sell_value
        Uniform(80, 120), # demand_distr
        solver,
    )
    compare_methods(problem2)

    return
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Newsvendor.main()
end
