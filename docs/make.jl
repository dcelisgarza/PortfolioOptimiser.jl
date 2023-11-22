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
        # "API" => [
        #     "obj_func_api.md",
        #     "exp_ret_api.md",
        #     "risk_models_api.md",
        #     "Optimisers" => [
        #         "base_optimiser_api.md",
        #         "efficient_frontier_api.md",
        #         "hrpopt_api.md",
        #         "black_litterman_api.md",
        #         "cla_api.md",
        #     ],
        #     "asset_allocation_api.md",
        # ],
        "Index" => "idx.md",
    ],
)

deploydocs(;
    repo = "github.com/dcelisgarza/PortfolioOptimiser.jl.git",
    push_preview = true,
    devbranch = "main",
)
