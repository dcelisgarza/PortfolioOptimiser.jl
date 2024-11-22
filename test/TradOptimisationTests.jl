using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      HiGHS, PortfolioOptimiser
path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

include("TradOptimisationTests_4.jl")
include("TradOptimisationTests_1.jl")
include("TradOptimisationTests_2.jl")
include("TradOptimisationTests_3.jl")

# #################
# #################
# portfolio = OmniPortfolio(; prices = prices,
#                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
#                                                            :check_sol => (allow_local = true,
#                                                                           allow_almost = true),
#                                                            :params => Dict("verbose" => false))))
# asset_statistics!(portfolio)
# rm = SLPM(; target = rf)
# obj = Sharpe(; rf = rf)
# wt1 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj, str_names = true)
# #################
# portfolio2 = Portfolio(; prices = prices,
#                        solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
#                                                         :check_sol => (allow_local = true,
#                                                                        allow_almost = true),
#                                                         :params => Dict("verbose" => false))))
# asset_statistics!(portfolio2)
# rm2 = SLPM(; target = rf)
# obj2 = Sharpe(; rf = rf)
# wt2 = optimise!(portfolio2; rm = rm2, kelly = AKelly(), obj = obj2, str_names = true)
# #################
# #################
