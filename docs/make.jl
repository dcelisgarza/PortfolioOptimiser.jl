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
        #size_threshold = 500 * 2^10,
        #size_threshold_ignore = ["Examples.md"],
        #example_size_threshold = 0,
    ),
    pages = [
        "Home" => "index.md",
        "DBHT" => "DBHT.md",
        "Constraint Functions" => "Constraint_functions.md",
        "OWA" => "OWA.md",
        "Risk Measures" => "Risk_measures.md",
        "Portfolio Optimisation" => "Portfolio.md",
        "Examples" => "Examples.md",
        "API" => ["Definitions" => "Definitions.md", "Statistics" => "Statistics.md"],
        "Index" => "idx.md",
    ],
)

deploydocs(;
    repo = "github.com/dcelisgarza/PortfolioOptimiser.jl.git",
    push_preview = true,
    devbranch = "main",
)
