using Test, PortfolioOptimiser, DataFrames, TimeSeries, CSV, Dates, Clarabel, HiGHS, Logging

Logging.disable_logging(Logging.Warn)

A = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Allocation" begin
    portfolio = Portfolio(; returns = returns,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :params => Dict("verbose" => false))),
                          alloc_solvers = Dict(:HiGHS => Dict(:solver => (HiGHS.Optimizer),
                                                              :params => Dict("log_to_console" => false))),
                          latest_prices = Vector(dropmissing(DataFrame(A))[end, 2:end]))
    asset_statistics!(portfolio)

    opt_port!(portfolio)
    alloc_type = :LP
    lp_alloc = allocate_port!(portfolio; alloc_type = alloc_type)

    alloc_type = :Greedy
    gr_alloc = allocate_port!(portfolio; alloc_type = alloc_type)

    lp_alloct = DataFrame(; tickers = ["AMZN", "AMD", "BBY", "MA", "JPM"],
                          shares = [3, 57, 20, 13, 10],
                          weights = [0.44567597303213385, 0.05827001169763145,
                                     0.147637358865155, 0.23325910345545967,
                                     0.11515755294961992])

    gr_alloct = DataFrame(;
                          tickers = ["AMZN", "MA", "BBY", "JPM", "AMD", "WMT", "FB", "PFE",
                                     "T", "SBUX", "SHLD"],
                          shares = [3.0, 13.0, 20.0, 10.0, 57.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0],
                          weights = [0.4284594841283738, 0.2242482906467279,
                                     0.141934119057567, 0.11070900994414283,
                                     0.056019037737808156, 0.008597912881214939,
                                     0.01664538312195933, 0.003581879831091563,
                                     0.003527836281590984, 0.005946780845289751,
                                     0.00033026552423404956])

    lp_leftovert = 394.02972999999974
    gr_leftovert = 8.03971999999931

    @test isapprox(lp_leftovert, portfolio.alloc_params[:LP_Trad][:leftover])
    @test isapprox(gr_leftovert, portfolio.alloc_params[:Greedy_Trad][:leftover])

    lp_allocjoin = outerjoin(lp_alloct, lp_alloc; on = :tickers, makeunique = true)
    lp_allocjoin.shares[ismissing.(lp_allocjoin.shares)] .= 0
    lp_allocjoin.weights[ismissing.(lp_allocjoin.weights)] .= 0

    gr_allocjoin = outerjoin(gr_alloct, gr_alloc; on = :tickers, makeunique = true)
    gr_allocjoin.shares[ismissing.(gr_allocjoin.shares)] .= 0
    gr_allocjoin.weights[ismissing.(gr_allocjoin.weights)] .= 0

    @test isapprox(lp_allocjoin.shares, lp_allocjoin.shares_1)
    @test isapprox(lp_allocjoin.weights, lp_allocjoin.weights_1)

    @test isapprox(gr_allocjoin.shares, gr_allocjoin.shares_1)
    @test isapprox(gr_allocjoin.weights, gr_allocjoin.weights_1)

    portfolio = Portfolio(; returns = returns,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :params => Dict("verbose" => false))),
                          alloc_solvers = Dict(:HiGHS => Dict(:solver => (HiGHS.Optimizer),
                                                              :params => Dict("log_to_console" => false))),
                          latest_prices = Vector(dropmissing(DataFrame(A))[end, 2:end]),
                          short = true)
    asset_statistics!(portfolio)

    opt_port!(portfolio)
    alloc_type = :LP
    lp_alloc = allocate_port!(portfolio; alloc_type = alloc_type)

    alloc_type = :Greedy
    gr_alloc = allocate_port!(portfolio; alloc_type = alloc_type)

    lp_alloct = DataFrame(;
                          tickers = ["AMZN", "AMD", "WMT", "T", "BBY", "MA", "PFE", "JPM",
                                     "GE", "UAA", "SHLD", "RRC"],
                          shares = [2, 79, 5, 18, 19, 11, 2, 18, -21, -33, -101, -56],
                          weights = [0.28541228564902454, 0.07757861860414689,
                                     0.04295534452162168, 0.06345050594798939,
                                     0.13473008192420027, 0.1895975129316084,
                                     0.007158057277632926, 0.19911759314377597,
                                     -0.1363533964446091, -0.27655154115332437,
                                     -0.16685606724304514, -0.4202389951590214])

    gr_alloct = DataFrame(;
                          tickers = ["AMZN", "JPM", "MA", "BBY", "T", "AMD", "WMT", "PFE",
                                     "FB", "SBUX", "BABA", "GM", "BAC", "RRC", "UAA",
                                     "SHLD", "GE"],
                          shares = [2.0, 17.0, 11.0, 19.0, 18.0, 46.0, 5.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, -56.0, -33.0, -101.0, -21.0],
                          weights = [0.28542570070684303, 0.18806434369281091,
                                     0.18960642446669676, 0.1347364145592331,
                                     0.0634534882730283, 0.04517448340849857,
                                     0.04295736352521524, 0.0035791968616945163,
                                     0.016632915072883347, 0.005942326471672758,
                                     0.0175369641718073, 0.003900214409216869,
                                     0.0029901643803995994, -0.42023899515902136,
                                     -0.2765515411533243, -0.1668560672430451,
                                     -0.1363533964446091])

    lp_leftovert = 2.5495485778292277
    gr_leftovert = 3.0197298793932283

    @test isapprox(lp_leftovert, portfolio.alloc_params[:LP_Trad][:leftover])
    @test isapprox(gr_leftovert, portfolio.alloc_params[:Greedy_Trad][:leftover])

    lp_allocjoin = outerjoin(lp_alloct, lp_alloc; on = :tickers, makeunique = true)
    lp_allocjoin.shares[ismissing.(lp_allocjoin.shares)] .= 0
    lp_allocjoin.weights[ismissing.(lp_allocjoin.weights)] .= 0

    gr_allocjoin = outerjoin(gr_alloct, gr_alloc; on = :tickers, makeunique = true)
    gr_allocjoin.shares[ismissing.(gr_allocjoin.shares)] .= 0
    gr_allocjoin.weights[ismissing.(gr_allocjoin.weights)] .= 0

    @test isapprox(lp_allocjoin.shares, lp_allocjoin.shares_1)
    @test isapprox(lp_allocjoin.weights, lp_allocjoin.weights_1)

    @test isapprox(gr_allocjoin.shares, gr_allocjoin.shares_1)
    @test isapprox(gr_allocjoin.weights, gr_allocjoin.weights_1)

    portfolio = HCPortfolio(; returns = returns,
                            solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                             :params => Dict("verbose" => false))),
                            alloc_solvers = Dict(:HiGHS => Dict(:solver => (HiGHS.Optimizer),
                                                                :params => Dict("log_to_console" => false))),
                            latest_prices = Vector(dropmissing(DataFrame(A))[end, 2:end]))
    asset_statistics!(portfolio)

    opt_port!(portfolio)
    alloc_type = :LP
    lp_alloc = allocate_port!(portfolio; alloc_type = alloc_type)

    alloc_type = :Greedy
    gr_alloc = allocate_port!(portfolio; alloc_type = alloc_type)

    lp_alloct = DataFrame(;
                          tickers = ["AAPL", "FB", "BABA", "GE", "AMD", "WMT", "BAC", "GM",
                                     "T", "UAA", "SHLD", "XOM", "RRC", "BBY", "MA", "PFE",
                                     "JPM", "SBUX"],
                          shares = [5, 4, 2, 51, 17, 10, 11, 15, 27, 24, 45, 6, 23, 7, 4,
                                    18, 6, 12],
                          weights = [0.08623931753776092, 0.06654290758616725,
                                     0.03507985765312407, 0.06616181942714512,
                                     0.01669774008672745, 0.08592925108672071,
                                     0.032897368602639615, 0.05851310621022857,
                                     0.09519632279587187, 0.040185000941916973,
                                     0.014853326961058021, 0.04646840834726153,
                                     0.03448472415059915, 0.0496481233639968,
                                     0.06895944646091011, 0.06443643476237498,
                                     0.06638687163604927, 0.07131997238944758])

    gr_alloct = DataFrame(;
                          tickers = ["T", "WMT", "AAPL", "SBUX", "MA", "GE", "JPM", "PFE",
                                     "GM", "FB", "BBY", "XOM", "UAA", "RRC", "BABA", "GOOG",
                                     "BAC", "AMD", "SHLD"],
                          shares = [26.0, 9.0, 4.0, 12.0, 3.0, 51.0, 5.0, 17.0, 14.0, 3.0,
                                    6.0, 5.0, 23.0, 23.0, 2.0, 1.0, 10.0, 16.0, 45.0],
                          weights = [0.09169364550589719, 0.07735582440772054,
                                     0.06900884850812526, 0.0713379539453109,
                                     0.05173262465168604, 0.06617850048312691,
                                     0.05533634119445326, 0.06087197631636115,
                                     0.0546260015779813, 0.0499197635430322,
                                     0.04256626363075451, 0.0387334368331785,
                                     0.03852033539845121, 0.034493418615459,
                                     0.03508870216608353, 0.10204557004640942,
                                     0.029914238959370714, 0.01571948235885059,
                                     0.014857071857747662])

    lp_leftovert = 2.239876000001459
    gr_leftovert = 4.759927000000322

    @test isapprox(lp_leftovert, portfolio.alloc_params[:LP_HRP][:leftover])
    @test isapprox(gr_leftovert, portfolio.alloc_params[:Greedy_HRP][:leftover])

    lp_allocjoin = outerjoin(lp_alloct, lp_alloc; on = :tickers, makeunique = true)
    lp_allocjoin.shares[ismissing.(lp_allocjoin.shares)] .= 0
    lp_allocjoin.weights[ismissing.(lp_allocjoin.weights)] .= 0

    gr_allocjoin = outerjoin(gr_alloct, gr_alloc; on = :tickers, makeunique = true)
    gr_allocjoin.shares[ismissing.(gr_allocjoin.shares)] .= 0
    gr_allocjoin.weights[ismissing.(gr_allocjoin.weights)] .= 0

    @test isapprox(lp_allocjoin.shares, lp_allocjoin.shares_1)
    @test isapprox(lp_allocjoin.weights, lp_allocjoin.weights_1)

    @test isapprox(gr_allocjoin.shares, gr_allocjoin.shares_1)
    @test isapprox(gr_allocjoin.weights, gr_allocjoin.weights_1)

    investment = 69420
    opt_port!(portfolio; linkage = :complete, type = :HERC)
    alloc_type = :LP
    lp_alloc = allocate_port!(portfolio; port_type = :HERC, alloc_type = alloc_type,
                              investment = investment,)

    alloc_type = :Greedy
    gr_alloc = allocate_port!(portfolio; port_type = :HERC, alloc_type = alloc_type,
                              investment = investment,)

    lp_alloct = DataFrame(;
                          tickers = ["GOOG", "AAPL", "FB", "BABA", "GE", "AMD", "WMT",
                                     "BAC", "GM", "T", "UAA", "SHLD", "XOM", "RRC", "BBY",
                                     "MA", "PFE", "JPM", "SBUX"],
                          shares = [1, 4, 4, 3, 47, 303, 335, 16, 13, 22, 21, 2533, 9, 256,
                                    231, 4, 21, 5, 13],
                          weights = [0.014693069123740864, 0.009936264561200548,
                                     0.009583620808428936, 0.007578389627606793,
                                     0.008781384022864722, 0.04286267311582058,
                                     0.41458509282878303, 0.006891540406729905,
                                     0.007303534670175715, 0.01117138291266522,
                                     0.00506407214586503, 0.12041324645864256,
                                     0.01003868679606578, 0.055279834486625434,
                                     0.23596338932588753, 0.009931654777554406,
                                     0.010826950248780262, 0.00796762354167102,
                                     0.01112759014089158])

    gr_alloct = DataFrame(;
                          tickers = ["WMT", "BBY", "SHLD", "RRC", "AMD", "T", "SBUX", "PFE",
                                     "XOM", "GOOG", "AAPL", "MA", "GE", "JPM", "FB", "GM",
                                     "BAC", "BABA", "UAA"],
                          shares = [335.0, 231.0, 2531.0, 255.0, 303.0, 23.0, 13.0, 21.0,
                                    9.0, 1.0, 4.0, 4.0, 46.0, 5.0, 4.0, 13.0, 16.0, 3.0,
                                    21.0],
                          weights = [0.41458097200928257, 0.2359610439387693,
                                     0.1203169749421343, 0.05506335031878181,
                                     0.04286224707701147, 0.01167905695848274,
                                     0.011127479536841315, 0.010826842632977167,
                                     0.010038587015304185, 0.014692923080173899,
                                     0.009936165798477349, 0.009931556060650716,
                                     0.008594460638384179, 0.007967544346497265,
                                     0.00958352555085167, 0.007303462075794942,
                                     0.006891471907416766, 0.007578314301271379,
                                     0.005064021810897344])

    lp_leftovert = 1.5577120000156128
    gr_leftovert = 0.8677120000015961

    @test isapprox(lp_leftovert, portfolio.alloc_params[:LP_HERC][:leftover])
    @test isapprox(gr_leftovert, portfolio.alloc_params[:Greedy_HERC][:leftover])

    lp_allocjoin = outerjoin(lp_alloct, lp_alloc; on = :tickers, makeunique = true)
    lp_allocjoin.shares[ismissing.(lp_allocjoin.shares)] .= 0
    lp_allocjoin.weights[ismissing.(lp_allocjoin.weights)] .= 0

    gr_allocjoin = outerjoin(gr_alloct, gr_alloc; on = :tickers, makeunique = true)
    gr_allocjoin.shares[ismissing.(gr_allocjoin.shares)] .= 0
    gr_allocjoin.weights[ismissing.(gr_allocjoin.weights)] .= 0

    @test isapprox(lp_allocjoin.shares, lp_allocjoin.shares_1)
    @test isapprox(lp_allocjoin.weights, lp_allocjoin.weights_1)

    @test isapprox(gr_allocjoin.shares, gr_allocjoin.shares_1)
    @test isapprox(gr_allocjoin.weights, gr_allocjoin.weights_1)
end
