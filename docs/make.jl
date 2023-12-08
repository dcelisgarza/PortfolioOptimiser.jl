using Documenter
using PortfolioOptimiser

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
    PortfolioOptimiser.turn_into_Hclust_merges

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
        # "Examples" => "Examples.md",
        "DBHT" => "DBHT.md",
        "Constraint Functions" => "Constraint_functions.md",
        "OWA" => "OWA.md",
        "Risk Measures" => "Risk_measures.md",
        "API" => [
            "Definitions" => "Definitions.md",
            "Types" => "Types.md",
            "Asset Statistics" => "Asset_statistics.md",
        ],
        "Index" => "idx.md",
    ],
)

deploydocs(;
    repo = "github.com/dcelisgarza/PortfolioOptimiser.jl.git",
    push_preview = true,
    devbranch = "main",
)
