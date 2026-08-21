include("src/Compare.jl")

include("Newsvendor/newsvendor.jl")
include("HydroToy/hydro2.jl")
include("HydroToy/hydro3.jl")
include("HydroQuad/hydroquad.jl")
include("HydroQuad/hydroquad_pw.jl")

include("solvers.jl")

problems = [Newsvendor, HydroToy2, HydroToy3, HydroQuad, HydroQuadPw]
all_solvers = [SolverList.ipopt, SolverList.highs_asm, SolverList.highs_ipm]

function test_default()
    for m in problems
        println(repeat("=", 90))
        println("Running $(m)")
        m.main()
        println()
    end
end

function test_solvers(problem::Module)
    for s in all_solvers
        println(repeat("=", 90))
        println("Testing solver: $(s)")
        problem.main(s)
    end
end
