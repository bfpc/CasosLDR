"""
Shared harness for the LDR case studies.

Each case study defines its own optimization models; this module holds the parts
that every case repeats: running a list of solution methods, timing them,
evaluating each solution out of sample and printing the comparison table.
"""
module Compare

using Printf
using Statistics

export run_comparison, average_scenarios, cached_model, with_warmup

"""
    run_comparison(problem, solvers; evaluate, verbose = true)

Run every `name => solve` pair of `solvers` on `problem`, print the comparison
table and return the rows.

`solve(problem)` must return a NamedTuple of results, e.g.
`(; Obj, Bound, H₁, V₁, dt)`.  Its `dt` field is the time spent inside the
optimizer (`dt = @elapsed optimize!(model)`) and is reported as `Opt time`;
`run_comparison` separately times the whole call, model building included, and
reports it as `Time (ms)`.  Both are printed in milliseconds.

`evaluate(problem, sol)` returns a NamedTuple assessing that solution, e.g.
`(; H₂, V₂, Perf, Std)`.

All solvers must return the same field names, and likewise all `evaluate` calls:
the field names are the columns of the table, in the order they appear.
Vector-valued fields are expanded into one column per entry.
"""
function run_comparison(problem, solvers; evaluate, verbose = true)
    rows = NamedTuple[]
    for (name, solve) in solvers
        verbose && print("Solving $name ...")
        t = @elapsed sol = solve(problem)
        verbose && print(" evaluating policy ...")
        perf = evaluate(problem, sol)
        verbose && println(" done.")
        push!(
            rows,
            (;
                Method = name,
                drop(sol, :dt)...,
                perf...,
                var"Opt time" = 1000 * get(sol, :dt, NaN),
                var"Time (ms)" = 1000 * t,
            ),
        )
    end
    print_table(rows)
    return rows
end

"""
    with_warmup(solvers, count = 3; suffix = " (warmup)")

Prepend a throw-away run of the first `count` entries of `solvers`, renamed with
`suffix`.  Those rows pay for the first-call compilation of the solution methods
and of the evaluation code, so the timings of the rest of the table measure
solving rather than compiling.  The default `count` covers the `EV`, `LDR (0)`
and `LDR (1)` entries the case studies list first.
"""
function with_warmup(solvers, count = 3; suffix = " (warmup)")
    warmup = [name * suffix => solve for (name, solve) in solvers[1:min(count, end)]]
    return [warmup; collect(solvers)]
end

"The NamedTuple `nt` without its `key` field."
drop(nt::NamedTuple, key::Symbol) = NamedTuple{filter(!=(key), keys(nt))}(nt)

const NumberOrArray = Union{Real, AbstractArray{<:Real}}

"""
    average_scenarios(f, scenarios)

Apply `f` to every scenario and return `(mean, stderr)`: two NamedTuples holding
the field-by-field average of the results and the corresponding standard error.

Fields that are not numbers (a solver status, say) are left out of both, and the
standard error of a vector-valued field is `NaN`.
"""
function average_scenarios(f, scenarios)
    results = [f(s) for s in scenarios]
    column(key) = [getproperty(r, key) for r in results]
    numeric = filter(key -> eltype(column(key)) <: NumberOrArray, keys(first(results)))
    return (
        NamedTuple{numeric}(map(key -> mean(column(key)), numeric)),
        NamedTuple{numeric}(map(key -> stderror(column(key)), numeric)),
    )
end

stderror(values::AbstractVector{<:Real}) = std(values) / sqrt(length(values))
stderror(values) = NaN

"""
    cached_model(build, problem)

Return `problem.second_stage[]`, calling `build()` to create it on first use, so
that the recourse model is built once and only re-solved for each scenario.
"""
function cached_model(build, problem)
    if problem.second_stage[] === nothing
        problem.second_stage[] = build()
    end
    return problem.second_stage[]
end

format(x::Real) = isfinite(x) ? @sprintf("%.2f", x) : string(x)
format(x) = string(x)

"Row as `header => text` pairs; vector fields expand into one column per entry."
function cells(row)
    columns = Pair{String, String}[]
    for (key, value) in pairs(row)
        if value isa AbstractVector
            append!(columns, ["$key[$i]" => format(v) for (i, v) in enumerate(value)])
        else
            push!(columns, string(key) => format(value))
        end
    end
    return columns
end

"Print `rows` as a table, sizing each column to its widest entry."
function print_table(rows; io = stdout)
    isempty(rows) && return
    table = map(cells, rows)
    headers = first.(first(table))
    widths = [
        max(textwidth(header), maximum(textwidth(row[j].second) for row in table)) for
        (j, header) in enumerate(headers)
    ]
    # The method name is left-aligned, every number right-aligned under its header.
    line(values) = join(
        (j == 1 ? rpad(v, widths[j]) : lpad(v, widths[j]) for (j, v) in enumerate(values)),
        "  ",
    )
    println(io, line(headers))
    println(io, repeat("-", sum(widths) + 2 * (length(widths) - 1)))
    for row in table
        println(io, line(last.(row)))
    end
    return
end

end # module
