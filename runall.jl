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

function test_default()
    for m in problems
        println(repeat("=", 90))
        println("Running $(m)")
        m.main()
        println()
    end
end

function test_solvers(problem::Module)
    for (name, solver) in all_solvers
        println(repeat("=", 90))
        println("Testing solver: $(name)")
        problem.main(solver)
    end
end
