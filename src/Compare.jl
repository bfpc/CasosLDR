"""
Shared harness for the LDR case studies.

Each case study defines its own optimization models; this module holds the parts
that every case repeats: running a list of solution methods, timing them,
evaluating each solution out of sample and printing the comparison table.
"""
module Compare

using Printf
using Statistics

export run_comparison, average_scenarios, cached_model, with_warmup, Trial

"""
    run_comparison(problem, solvers; evaluate, verbose = true, summarize = :append)

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

An entry of `solvers` may also be a `Trial`, which additionally says whether the
solution is to be evaluated; the evaluation columns of an unevaluated trial are
filled with `NaN`.

`summarize` controls the `mean ± std` line that consecutive entries sharing a
name -- the replications of one method, e.g. three runs of `"SAA (10 samples)"` --
get in the printed table:

  * `:append` adds the summary line under the rows it summarizes,
  * `:only` replaces them by it, which is what a run with many replications
    wants,
  * `:none` prints the rows alone.

The returned rows are always the raw ones, one per entry of `solvers`.
"""
function run_comparison(problem, solvers; evaluate, verbose = true, summarize = :append)
    summarize in (:append, :only, :none) ||
        throw(ArgumentError("summarize must be :append, :only or :none"))

    records = NamedTuple[]
    for trial in map(Trial, solvers)
        verbose && print("Solving $(trial.name) ...")
        t = @elapsed sol = trial.solve(problem)
        perf = nothing
        if trial.evaluated
            verbose && print(" evaluating policy ...")
            perf = evaluate(problem, sol)
        end
        verbose && println(" done.")
        push!(records, (; trial.name, sol, perf, t))
    end

    rows = comparison_rows(records)
    print_table(
        summarize === :none ? rows : with_summaries(rows; only = summarize === :only),
    )
    return rows
end

"""
    Trial(name, solve; evaluated = true)

One line of the comparison table: `solve(problem)`, reported under `name` and,
unless `evaluated` is unset, assessed out of sample.  `run_comparison` accepts a
`Trial` wherever it accepts a `name => solve` pair.
"""
struct Trial
    name::String
    solve::Any
    evaluated::Bool
end

Trial(name, solve; evaluated = true) = Trial(name, solve, evaluated)
Trial(trial::Trial) = trial
Trial(entry::Pair) = Trial(first(entry), last(entry))

"""
    with_warmup(solvers, count = 3; suffix = " (warmup)", families = true)

Prepend a throw-away run of some entries of `solvers`, renamed with `suffix`.
Those rows pay for the first-call compilation of the solution methods, so the
timings of the rest of the table measure solving rather than compiling -- which
also keeps compilation out of the `mean ± std` of a group of replications.

Warmed up are the first `count` entries, which covers the `EV`, `LDR (0)` and
`LDR (1)` entries the case studies list first, and, unless `families` is unset,
the first entry of every other method family (`SAA`, `SDDP`, ...), a family being
everything up to the first space of a name.

Warmup solutions are not evaluated out of sample -- their timings are the point,
not their quality -- so their evaluation columns are `NaN`.
"""
function with_warmup(solvers, count = 3; suffix = " (warmup)", families = true)
    trials = map(Trial, collect(solvers))
    chosen = collect(1:min(count, length(trials)))
    if families
        warmed = Set(family(trials[i].name) for i in chosen)
        for (i, trial) in enumerate(trials)
            family(trial.name) in warmed && continue
            push!(warmed, family(trial.name))
            push!(chosen, i)
        end
    end
    warmup = [
        Trial(trials[i].name * suffix, trials[i].solve; evaluated = false) for
        i in sort(chosen)
    ]
    return [warmup; trials]
end

"The method family of a table line: its name up to the first space."
family(name::AbstractString) = first(split(name, ' '))

"""
    comparison_rows(records)

One table row per record, with the columns of the solution, of its evaluation and
of the two timings.  Records of unevaluated trials get the evaluation columns of
the others, filled with `NaN`.
"""
function comparison_rows(records)
    evaluated = findfirst(record -> record.perf !== nothing, records)
    blank = evaluated === nothing ? (;) : blank_evaluation(records[evaluated].perf)
    return NamedTuple[
        (;
            Method = record.name,
            drop(record.sol, :dt)...,
            (record.perf === nothing ? blank : record.perf)...,
            var"Opt time" = 1000 * get(record.sol, :dt, NaN),
            var"Time (ms)" = 1000 * record.t,
        ) for record in records
    ]
end

"The columns of the evaluation `perf`, filled with `NaN`."
blank_evaluation(perf::NamedTuple) =
    map(value -> value isa AbstractVector ? fill(NaN, length(value)) : NaN, perf)

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

"""
    with_summaries(rows; only = false)

`rows` with a `mean ± std` line inserted after each run of consecutive rows that
share a `Method` name, i.e. after each group of replications of one method, or
with each such group replaced by that line if `only` is set.  Constant columns (a
`-Inf` bound, say) are reported as the value alone.
"""
function with_summaries(rows; only = false)
    table = NamedTuple[]
    group = NamedTuple[]
    flush!() = begin
        if length(group) > 1
            only || append!(table, group)
            push!(table, summary_row(group, only ? first(group).Method : "  ↳ mean ± std"))
        else
            append!(table, group)
        end
        empty!(group)
    end
    for row in rows
        isempty(group) || row.Method == first(group).Method || flush!()
        push!(group, row)
    end
    flush!()
    return table
end

"The `mean ± std` line of a group of replications of one method, named `label`."
function summary_row(group, label)
    columns = keys(first(group))
    summary(key) =
        key === :Method ? label : mean_std([getproperty(row, key) for row in group])
    return NamedTuple{columns}(map(summary, columns))
end

"`mean ± std` of the replications of one column, formatted for the table."
function mean_std(values::AbstractVector{<:Real})
    all(isequal(first(values)), values) && return format(first(values))
    return string(format(mean(values)), " ± ", format(std(values)))
end

mean_std(values::AbstractVector{<:AbstractVector{<:Real}}) =
    [mean_std([value[i] for value in values]) for i in eachindex(first(values))]

# Non-numeric columns (a solver status, say) have no meaningful summary.
mean_std(values) = ""

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
