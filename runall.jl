include("src/Compare.jl")

include("Newsvendor/newsvendor.jl")
include("HydroToy/hydro2.jl")
include("HydroToy/hydro3.jl")
include("HydroQuad/hydroquad.jl")
include("HydroQuad/hydroquad_pw.jl")

include("solvers.jl")

problems = [Newsvendor, HydroToy2, HydroToy3, HydroQuad, HydroQuadPw]
all_solvers = [
    "Ipopt" => SolverList.ipopt,
    "HiGHS (ASM)" => SolverList.highs_asm,
    "HiGHS (IPM)" => SolverList.highs_ipm,
]

# Extra keyword arguments go to each case study's `main`; in particular
# `n_scenarios` sets the out-of-sample sample size of the hydro-thermal cases,
# so `test_default(n_scenarios = 100)` is a quick smoke test of the package.
function test_default(; kwargs...)
    for m in problems
        println(repeat("=", 90))
        println("Running $(m)")
        m.main(; kwargs...)
        println()
    end
end

function test_solvers(problem::Module; kwargs...)
    for (name, solver) in all_solvers
        println(repeat("=", 90))
        println("Testing solver: $(name)")
        problem.main(solver; kwargs...)
    end
end
