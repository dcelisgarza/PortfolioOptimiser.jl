using COSMO, CSV, Clarabel, DataFrames, HiGHS, LinearAlgebra, OrderedCollections,
      PortfolioOptimiser, Statistics, Test, TimeSeries, Clustering

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Deepcopy portfolios" begin
    returns = dropmissing!(DataFrame(percentchange(prices)))

    portfolio = Portfolio(; prices = prices,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))),
                          alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                              :params => Dict("log_to_console" => false))))
    asset_statistics!(portfolio)
    optimise!(portfolio)
    allocate!(portfolio)
    portfolio2 = deepcopy(portfolio)

    hcportfolio = HCPortfolio(; prices = prices,
                              solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                      :params => Dict("verbose" => false)),
                                                    :COSMO => Dict(:solver => COSMO.Optimizer,
                                                                   :params => Dict("verbose" => false))),
                              alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                                                  :params => Dict("log_to_console" => false))))
    asset_statistics!(hcportfolio)
    optimise!(hcportfolio)
    allocate!(hcportfolio)
    hcportfolio2 = deepcopy(hcportfolio)

    for name ∈ propertynames(portfolio)
        if name ∈ (:model, :alloc_model)
            continue
        end
        @test isequal(getfield(portfolio, name), getfield(portfolio2, name))
    end

    for name ∈ propertynames(hcportfolio)
        if name ∈ (:alloc_model, :clusters, :opt_params)
            continue
        end
        @test isequal(getfield(hcportfolio, name), getfield(hcportfolio2, name))
    end
end
