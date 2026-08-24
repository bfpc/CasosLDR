include("src/Compare.jl")

include("Newsvendor/newsvendor.jl")
include("HydroToy/hydro2.jl")
include("HydroToy/hydro3.jl")
include("HydroQuad/hydroquad.jl")
include("HydroQuad/hydroquad_pw.jl")

modules = [Newsvendor, HydroToy2, HydroToy3, HydroQuad, HydroQuadPw]

for m in modules
    println(repeat("=", 90))
    println("Running $(m)")
    m.main()
    println()
end
