using Documenter
using PortfolioOptimiser

DocMeta.setdocmeta!(
    PortfolioOptimiser,
    :DocTestSetup,
    :(using PortfolioOptimiser);
    recursive = true,
)

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
        "Examples" => "Examples.md",
        "API" => [
            "Definitions" => "Definitions.md",
            "Types" => "Types.md",
            "Risk Measures" => "Risk_measures.md",
            "Constraint Functions" => "Constraint_functions.md",
            "Asset Statistics" => "Asset_statistics.md",
            "DBHT" => "DBHT.md",
            "OWA" => "OWA.md",
        ],
        "Index" => "idx.md",
    ],
)

deploydocs(;
    repo = "github.com/dcelisgarza/PortfolioOptimiser.jl.git",
    push_preview = true,
    devbranch = "main",
)
