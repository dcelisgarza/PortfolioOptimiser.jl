# using PortfolioOptimiser
# using Documenter

# DocMeta.setdocmeta!(
#     PortfolioOptimiser,
#     :DocTestSetup,
#     :(using PortfolioOptimiser);
#     recursive = true,
# )

# makedocs(;
#     modules = [PortfolioOptimiser],
#     authors = "Daniel Celis Garza",
#     repo = "https://github.com/dcelisgarza/PortfolioOptimiser.jl/blob/{commit}{path}#{line}",
#     sitename = "PortfolioOptimiser.jl",
#     format = Documenter.HTML(;
#         prettyurls = get(ENV, "CI", "false") == "true",
#         canonical = "https://dcelisgarza.github.io/PortfolioOptimiser.jl",
#         assets = String[],
#     ),
#     pages = ["Home" => "index.md", "API" => "api.md", "Index" => "idx.md"],
# )

# deploydocs(; repo = "github.com/dcelisgarza/PortfolioOptimiser.jl", devbranch = "main")

using PortfolioOptimiser
using Documenter

DocMeta.setdocmeta!(
    PortfolioOptimiser,
    :DocTestSetup,
    :(using PortfolioOptimiser);
    recursive = true,
)

makedocs(;
    modules = [PortfolioOptimiser],
    authors = "Daniel Celis Garza",
    repo = "https://github.com/dcelisgarza/PortfolioOptimiser.jl/blob/{commit}{path}#{line}",
    sitename = "PortfolioOptimiser.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://dcelisgarza.github.io/PortfolioOptimiser.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/dcelisgarza/PortfolioOptimiser.jl", devbranch = "main")
