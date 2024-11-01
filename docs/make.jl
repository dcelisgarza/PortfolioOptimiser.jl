using Documenter, DocumenterTools, DocumenterCitations, Literate, PortfolioOptimiser,
      StatsPlots, GraphRecipes

# utility function from https://github.com/JuliaOpt/Convex.jl/blob/master/docs/make.jl
function pre_process_content_md(content)
    return replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```",
                   r"^#note # (.*)$"m => s"""
 # !!! note
 #     \1""", r"^#warning # (.*)$"m => s"""
             # !!! warning
             #     \1""", r"^#tip # (.*)$"m => s"""
             # !!! tip
             #     \1""", r"^#info # (.*)$"m => s"""
             # !!! info
             #     \1""")
end
function pre_process_content_nb(content)
    return replace(content, r"\$\$(.*?)\$\$"s => s"```math\1```",
                   r"^#note # (.*)$"m => s"""
# > *note*
# > \1""", r"^#warning # (.*)$"m => s"""
     # > *warning*
     # > \1""", r"^#tip # (.*)$"m => s"""
     # > *tip*
     # > \1""", r"^#info # (.*)$"m => s"""
     # > *info*
     # > \1""")
end

fix_suffix_md(filename) = replace(filename, ".jl" => ".md")
function postprocess(cont)
    return """
           The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).
           """ * cont
end

example_path = joinpath(@__DIR__, "../examples/")
build_path_md = joinpath(@__DIR__, "src", "examples/")
files = readdir(example_path)
code_files = filter(x -> endswith(x, ".jl"), files)
data_files = filter(x -> endswith(x, ".csv"), files)
examples_nav = fix_suffix_md.("./examples/" .* code_files)

for file ∈ data_files
    cp(joinpath(@__DIR__, "../examples/" * file),
       joinpath(@__DIR__, "src/examples/" * file); force = true)
end

for file ∈ code_files
    Literate.markdown(example_path * file, build_path_md;
                      preprocess = pre_process_content_md, postprocess = postprocess,
                      documenter = true, credit = true)
    Literate.notebook(example_path * file, example_path;
                      preprocess = pre_process_content_nb, documenter = true, credit = true)
end

makedocs(;
         modules = [PortfolioOptimiser,
                    Base.get_extension(PortfolioOptimiser, :PortfolioOptimiserPlotsExt)],
         authors = "Daniel Celis Garza",
         repo = "https://github.com/dcelisgarza/PortfolioOptimiser.jl/blob/{commit}{path}#{line}",
         sitename = "PortfolioOptimiser.jl",
         format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://dcelisgarza.github.io/PortfolioOptimiser.jl",
                                  assets = String[],
                                  #   size_threshold_ignore = ["RiskMeasures/RiskMeasures.md"]
                                  ),
         pages = ["Home" => "index.md", "Examples" => examples_nav,
                  "API" => ["Constraints" => ["Constraints/WeightConstraints.md",
                                              "Constraints/ViewsConstraints.md",
                                              "Constraints/NetworkConstraints.md"],
                            "Parameter Estimation" => ["ParameterEstimation/PosdefFix.md",
                                                       "ParameterEstimation/MatrixDenoising.md",
                                                       "ParameterEstimation/DistanceMatrices.md",
                                                       "ParameterEstimation/Clustering.md",
                                                       "ParameterEstimation/CovCorKurtSkew.md",
                                                       "ParameterEstimation/MeanEstimators.md",
                                                       "ParameterEstimation/WorstCase.md",
                                                       "ParameterEstimation/Regression.md",
                                                       "ParameterEstimation/BlackLitterman.md",
                                                       "ParameterEstimation/OWA.md",
                                                       "ParameterEstimation/ParameterEstimationMisc.md"],
                            "(HC)Portfolio" => ["Portfolio/PortfolioTypes.md",
                                                "Portfolio/PortfolioStatistics.md",
                                                "Portfolio/PortfolioRisk.md",
                                                "Portfolio/PortfolioClustering.md",
                                                "Portfolio/PortfolioNetwork.md"],
                            "Risk Measures" => ["RiskMeasures/RiskMeasures.md",
                                                "RiskMeasures/RiskMeasureStatistics.md",
                                                "RiskMeasures/MiscRiskMeasureFunctions.md"],
                            "Optimisation" => ["Optimisation/OptimisationKinds.md",
                                               "Optimisation/ObjectiveFunctions.md",
                                               "Optimisation/PortfolioClasses.md",
                                               "Optimisation/AssetAllocation.md",
                                               "Optimisation/OptimisationConstraints.md",
                                               "Optimisation/OptimisationSetup.md",
                                               "Optimisation/OptimisationRiskMeasures.md",
                                               "Optimisation/OptimisationGetZ.md",
                                               "Optimisation/OptimisationFinalisation.md"],
                            "Plots Extension" => "PlotsExtension.md",
                            "References" => "References.md", "Index" => "Contents.md"]],
         plugins = [CitationBibliography(joinpath(@__DIR__, "src", "refs.bib");
                                         style = :numeric)])

deploydocs(; repo = "github.com/dcelisgarza/PortfolioOptimiser.jl.git", push_preview = true,
           devbranch = "main")
