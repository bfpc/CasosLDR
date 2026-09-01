module SolverList

using JuMP
using Ipopt, HiGHS

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "max_cpu_time" => 60.0)
highs_asm = optimizer_with_attributes(
    HiGHS.Optimizer,
    "solver" => "qpasm",
    "time_limit" => 60.0)
highs_ipm = optimizer_with_attributes(
	HiGHS.Optimizer,
	"solver" => "ipm",
	"time_limit" => 60.0,
)

end
