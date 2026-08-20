module SolverList

using JuMP
using Ipopt, HiGHS

ipopt = Ipopt.Optimizer
highs_asm = HiGHS.Optimizer
highs_ipm = optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm")

end
