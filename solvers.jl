module SolverList

using JuMP
using Ipopt, HiGHS, NLPModelsJuMP, NLPModelsAlgencan, Clarabel

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time" => 60.0)
highs_asm = optimizer_with_attributes(
    HiGHS.Optimizer,
    "solver" => "asm",
    "time_limit" => 60.0)
highs_ipm = optimizer_with_attributes(
	HiGHS.Optimizer,
	"solver" => "ipm",
	"time_limit" => 60.0,
)
algencan = optimizer_with_attributes(
	NLPModelsJuMP.Optimizer,
	"solver" => NLPModelsAlgencan.AlgencanSolver
)
clarabel = optimizer_with_attributes(Clarabel.Optimizer, "time_limit" => 60.0)

end
