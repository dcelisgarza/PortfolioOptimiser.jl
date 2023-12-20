using Documenter, DocumenterTools, Literate, PortfolioOptimiser

# DBHTs internals.
import PortfolioOptimiser.distance_wei,
    PortfolioOptimiser.clique3,
    PortfolioOptimiser.breadth,
    PortfolioOptimiser.FindDisjoint,
    PortfolioOptimiser.CliqHierarchyTree2s,
    PortfolioOptimiser.DBHTRootMethods,
    PortfolioOptimiser.BubbleCluster8s,
    PortfolioOptimiser.BuildHierarchy,
    PortfolioOptimiser.AdjCliq,
    PortfolioOptimiser.BubbleHierarchy,
    PortfolioOptimiser.DirectHb,
    PortfolioOptimiser.HierarchyConstruct4s,
    PortfolioOptimiser.LinkageFunction,
    PortfolioOptimiser._build_link_and_dendro,
    PortfolioOptimiser.DendroConstruct,
    PortfolioOptimiser.BubbleMember,
    PortfolioOptimiser.turn_into_Hclust_merges,
    # OWA
    PortfolioOptimiser._optimize_owa,
    PortfolioOptimiser._crra_method,
    PortfolioOptimiser.OWAMethods,
    # Portfolio
    PortfolioOptimiser.AbstractPortfolio,
    PortfolioOptimiser.RiskMeasures,
    PortfolioOptimiser.HRRiskMeasures,
    PortfolioOptimiser.TrackingErrKinds,
    # Asset Statistics
    PortfolioOptimiser.BinTypes

# utility function from https://github.com/JuliaOpt/Convex.jl/blob/master/docs/make.jl
fix_math_md(content) = replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```")
fix_suffix(filename) = replace(filename, ".jl" => ".md")
function postprocess(cont)
    """
    The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).
    """ * cont
end

example_path = joinpath(@__DIR__, "../examples/")
build_path = joinpath(@__DIR__, "src", "examples/")
files = readdir(example_path)
code_files = filter(x -> endswith(x, ".jl"), files)
data_files = filter(x -> endswith(x, ".csv"), files)
examples_nav = fix_suffix.("./examples/" .* code_files)

for file in data_files
    cp(
        joinpath(@__DIR__, "../examples/" * file),
        joinpath(@__DIR__, "src/examples/" * file);
        force = true,
    )
end

for file in code_files
    Literate.markdown(
        example_path * file,
        build_path;
        preprocess = fix_math_md,
        postprocess = postprocess,
        documenter = true,
        credit = true,
    )
end

makedocs(;
    # modules = [PortfolioOptimiser],
    authors = "Daniel Celis Garza",
    repo = "https://github.com/dcelisgarza/PortfolioOptimiser.jl/blob/{commit}{path}#{line}",
    sitename = "PortfolioOptimiser.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://dcelisgarza.github.io/PortfolioOptimiser.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "DBHT" => "DBHT.md",
        "Constraint Functions" => "Constraint_functions.md",
        "OWA" => "OWA.md",
        "Risk Measures" => "Risk_measures.md",
        "Portfolio Optimisation" => "Portfolio.md",
        "Examples" => examples_nav,
        "API" => ["Definitions" => "Definitions.md", "Statistics" => "Statistics.md"],
        "Index" => "idx.md",
    ],
)

deploydocs(;
    repo = "github.com/dcelisgarza/PortfolioOptimiser.jl.git",
    push_preview = true,
    devbranch = "main",
)
