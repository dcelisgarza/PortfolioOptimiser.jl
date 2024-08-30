using Documenter, DocumenterTools, DocumenterCitations, Literate, PortfolioOptimiser

# utility function from https://github.com/JuliaOpt/Convex.jl/blob/master/docs/make.jl
fix_math_md(content) = replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```")
fix_suffix(filename) = replace(filename, ".jl" => ".md")
function postprocess(cont)
    return """
           The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).
           """ * cont
end

example_path = joinpath(@__DIR__, "../examples/")
build_path = joinpath(@__DIR__, "src", "examples/")
files = readdir(example_path)
code_files = filter(x -> endswith(x, ".jl"), files)
data_files = filter(x -> endswith(x, ".csv"), files)
examples_nav = fix_suffix.("./examples/" .* code_files)

for file ∈ data_files
    cp(joinpath(@__DIR__, "../examples/" * file),
       joinpath(@__DIR__, "src/examples/" * file); force = true)
end

for file ∈ code_files
    Literate.markdown(example_path * file, build_path; preprocess = fix_math_md,
                      postprocess = postprocess, documenter = true, credit = true)
end

makedocs(;
         #modules = [PortfolioOptimiser],
         #             Base.get_extension(PortfolioOptimiser, :PortfolioOptimiserPlotExt)],
         authors = "Daniel Celis Garza",
         repo = "https://github.com/dcelisgarza/PortfolioOptimiser.jl/blob/{commit}{path}#{line}",
         sitename = "PortfolioOptimiser.jl",
         format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://dcelisgarza.github.io/PortfolioOptimiser.jl",
                                  assets = String[]),
         pages = ["Home" => "index.md", "Examples" => examples_nav,
                  "API" => ["Risk Measures" => "RiskMeasures.md",
                            "Portfolio Optimisation" => "PortfolioOptim.md",
                            "Portfolio Types" => "PortfolioTypes.md",
                            "References" => "References.md"]],
         plugins = [CitationBibliography(joinpath(@__DIR__, "src", "refs.bib");
                                         style = :numeric)])

deploydocs(; repo = "github.com/dcelisgarza/PortfolioOptimiser.jl.git", push_preview = true,
           devbranch = "main")
