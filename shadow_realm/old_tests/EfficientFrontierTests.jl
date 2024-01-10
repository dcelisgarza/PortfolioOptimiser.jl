using Test
using PortfolioOptimiser, DataFrames, CSV, Statistics, StatsBase, JuMP

@testset "Mean Variance" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])
    tickers = names(df)[2:end]

    mean_ret = ret_model(MRet(), Matrix(returns))
    S = cov(Cov(), Matrix(returns))

    ef = EffMeanVar(tickers, mean_ret, S)
    sectors = ["Tech", "Medical", "RealEstate", "Oil"]
    sector_map = Dict(tickers[1:4:20] .=> sectors[1])
    merge!(sector_map, Dict(tickers[2:4:20] .=> sectors[2]))
    merge!(sector_map, Dict(tickers[3:4:20] .=> sectors[3]))
    merge!(sector_map, Dict(tickers[4:4:20] .=> sectors[4]))
    sector_lower = Dict([(sectors[1], 0.2), (sectors[2], 0.1), (sectors[3], 0.15),
                         (sectors[4], 0.05)])
    sector_upper = Dict([(sectors[1], 0.4), (sectors[2], 0.5), (sectors[3], 0.2),
                         (sectors[4], 0.2)])

    # Do it twice for the coverage.
    add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)
    add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)

    max_sharpe!(ef)

    @test 0.2 - 1e-9 <= sum(ef.weights[1:4:20]) <= 0.4 + 1e-9
    @test 0.1 - 1e-9 <= sum(ef.weights[2:4:20]) <= 0.5 + 1e-9
    @test 0.15 - 1e-9 <= sum(ef.weights[3:4:20]) <= 0.2 + 1e-9
    @test 0.05 - 1e-9 <= sum(ef.weights[4:4:20]) <= 0.2 + 1e-9

    ef = EffMeanVar(tickers, mean_ret, S; market_neutral = true)
    add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)

    ef = EffMeanVar(tickers, mean_ret, S)
    efficient_risk!(ef, 0.001 / 252; optimiser_attributes = "tol" => 1e-5)
    @test termination_status(ef.model) == MOI.LOCALLY_INFEASIBLE

    efficient_risk!(ef,
                    0.01 / 252;
                    optimiser_attributes = ("tol" => 1e-3, "max_iter" => 20))
    @test termination_status(ef.model) == MOI.ITERATION_LIMIT

    ef = EffMeanVar(tickers, mean_ret, S)
    max_sharpe!(ef)
    sr1 = sharpe_ratio(ef.weights, ef.mean_ret, ef.cov_mtx)
    mu, sigma, sr = portfolio_performance(ef)
    @test sr ≈ sr1
    @test L2_reg(ef.weights, 0.69) ≈ 0.24823854411002114
    @test transaction_cost(ef.weights, fill(1 / 20, 20), 0.005) ≈ 0.007930322585937595
    @test ex_ante_tracking_error(ef.weights, S, fill(1 / 20, 20)) ≈ 7.395319914901144e-5
    @test ex_post_tracking_error(ef.weights, Matrix(returns), fill(1 / 895, 895)) ≈
          0.14663925084244206

    spy_prices = CSV.read("./assets/spy_prices.csv", DataFrame)
    delta = market_implied_risk_aversion(spy_prices[!, 2])
    # In the order of the dataframes, the
    mcapsdf = DataFrame(; ticker = tickers,
                        mcap = [927e9,
                                1.19e12,
                                574e9,
                                533e9,
                                867e9,
                                96e9,
                                43e9,
                                339e9,
                                301e9,
                                51e9,
                                61e9,
                                78e9,
                                0,
                                295e9,
                                1e9,
                                22e9,
                                288e9,
                                212e9,
                                422e9,
                                102e9])

    prior = market_implied_prior_returns(mcapsdf[!, 2], S, delta)

    # 1. SBUX drop by 20%
    # 2. GOOG outperforms FB by 10%
    # 3. BAC and JPM will outperform T and GE by 15%
    views = [-0.20, 0.10, 0.15] / 252
    picking = hcat([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0])

    bl = BlackLitterman(mcapsdf[!, 1],
                        S;
                        rf = 0,
                        tau = 0.01,
                        pi = prior, # either a vector, `nothing`, `:Equal`, or `:market`
                        Q = views,
                        P = picking,)

    ef = EffMeanVar(tickers, bl.post_ret, S)
    @test (0.0, 0.0, 0.0) == portfolio_performance(ef)

    max_sharpe!(ef)
    testweights = [0.22329034927489064,
                   2.4224808852378336e-12,
                   -1.9334026626835167e-12,
                   0.0682172439762337,
                   -6.15750578752061e-13,
                   -1.9512827548956932e-12,
                   0.01858647109227584,
                   -1.32108065115569e-12,
                   0.6899059356628131,
                   -1.4763299137373267e-12,
                   -1.6513999979195792e-12,
                   -1.9558196968692412e-12,
                   -2.075565125265058e-12,
                   1.3610943503346779e-12,
                   1.605276801539429e-12,
                   -1.6167690648494224e-12,
                   -1.1535074463509815e-12,
                   -2.1800818976719744e-13,
                   6.586370331891721e-12,
                   -2.2195384364506567e-12]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef; verbose = true)
    mutest, sigmatest, srtest = 0.0003239536955011413, 0.013804988027600572,
                                0.017773920051645197
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2, 4, 19, 231]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [230, 2, 4, 19, 1, 1]
    @test gAlloc.shares == testshares

    ef = EffMeanVar(tickers, bl.post_ret, S)
    max_sharpe!(ef, 1.03^(1 / 252) - 1)
    testweights = [0.15569648321556417,
                   -3.85124869697848e-13,
                   -1.8619632802904204e-12,
                   0.04899658633079129,
                   -1.1306309796120543e-12,
                   -1.8890430863400524e-12,
                   0.01899368641871192,
                   -1.6388501072065458e-12,
                   0.7763132440561539,
                   -1.642142316879662e-12,
                   -1.7658605375650975e-12,
                   -1.8769370122231116e-12,
                   -1.9536517164487535e-12,
                   -8.773285376956962e-13,
                   -7.679398936120036e-13,
                   -1.7223407966975508e-12,
                   -1.474296917451195e-12,
                   -1.244627865289495e-12,
                   1.0457390825685952e-12,
                   -2.0362115943436763e-12]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef; rf = 1.03^(1 / 252) - 1)
    mutest, sigmatest, srtest = 0.0003351394082199234, 0.014493158187285502,
                                0.015030243344516612
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2, 1, 4, 259]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [259, 1, 3, 19, 1]
    @test gAlloc.shares == testshares

    ef = EffMeanVar(tickers, bl.post_ret, S; weight_bounds = (-1, 1))
    max_sharpe!(ef)
    testweights = [0.7627027226891953,
                   0.29663630542640523,
                   -0.485817375133461,
                   0.17697919839038476,
                   0.12246502971889195,
                   -0.46243301978298923,
                   0.012679175675503675,
                   0.19028865257468236,
                   0.5511892088246185,
                   -0.004799311874420769,
                   -0.10615917324273837,
                   -0.0393723254341099,
                   -0.02402389567952595,
                   0.30370331429613306,
                   0.015735492715989955,
                   -0.0012282511001722196,
                   -0.03832567177612073,
                   0.3111057808720458,
                   0.4186741428364029,
                   -0.9999999999967155]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0009255811731360878, 0.021865546273422756,
                                0.03873656850645206
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [3, 5, 3, 2, 7, 58, 12, 3, 25, 12, -29, -357, -2, -30, -24, -73, -2, -169]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [3, 58, 11, 27, 12, 5, 7, 3, 4, 4, -168, -29, -356, -30, -24, -2, -73, -2,
                  -1]
    @test gAlloc.shares == testshares

    ef = EffMeanVar(tickers, bl.post_ret, S; weight_bounds = (-1, 1))
    max_sharpe!(ef, 1.03^(1 / 252) - 1)
    testweights = [0.856599312704922,
                   0.3242467090518291,
                   -0.5662220500219136,
                   0.19712987463815562,
                   0.11766010098352891,
                   -0.5520558713205291,
                   0.019059295241710694,
                   0.17772647718776885,
                   0.6336420676275687,
                   -0.016718794516616484,
                   -0.17337826531558917,
                   -0.047121084633332706,
                   -0.028145132760726683,
                   0.32907034902645027,
                   0.02061313704506288,
                   -0.013526119454701061,
                   -0.07075628744040385,
                   0.31816098047835784,
                   0.4740153014777353,
                   -0.9999999999992776]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef; rf = 1.03^(1 / 252) - 1)
    mutest, sigmatest, srtest = 0.0009981936254663442, 0.023784920767394704,
                                0.037035646250268506
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [3, 5, 3, 3, 6, 61, 12, 2, 25, 12, -34, -426, -5, -49, -28, -86, -2, -4,
                  -168]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [3, 61, 12, 12, 5, 26, 3, 5, 4, 5, -168, -34, -426, -49, -4, -28, -86, -5,
                  -2]
    @test gAlloc.shares == testshares

    ef = EffMeanVar(tickers, bl.post_ret, S)
    min_risk!(ef)
    testweights = [0.009535838694577005,
                   0.029453596657794606,
                   0.010454652764939246,
                   0.0265745002589632,
                   0.01164931947605233,
                   0.032140343768240584,
                   0.0001587791025930307,
                   0.13959556062683245,
                   0.000185191651506179,
                   0.0032750031678137758,
                   0.2870320198637697,
                   0.00011207588732677174,
                   0.00011294608582064199,
                   0.12284408366979002,
                   0.00104620378089814,
                   0.014540676852556738,
                   0.004490601346799808,
                   0.1916573861932764,
                   0.0002130013490054621,
                   0.11492821880144395]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 4.606750780387605e-5, 0.007706990600830219,
                                -0.004219212902280001
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2, 1, 2, 24, 16, 1, 82, 1, 16, 1, 2, 54, 20]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [81, 53, 16, 16, 19, 24, 2, 2, 2, 1, 1, 1]
    @test gAlloc.shares == testshares

    ef = EffMeanVar(tickers, bl.post_ret, S; weight_bounds = (-1, 1))
    min_risk!(ef)
    testweights = [0.003591664289605136,
                   0.037555083787440845,
                   0.017679255319435046,
                   0.033085636841628664,
                   0.012486831789267554,
                   0.053797724809782635,
                   -0.009710707766541071,
                   0.14121969035694254,
                   -0.010906762052986565,
                   0.018593992542797456,
                   0.2836715883430379,
                   -0.021141263921778567,
                   -0.00913258593982454,
                   0.1458819637802722,
                   0.0007739819968842805,
                   0.025432345790577262,
                   0.014570358392349714,
                   0.20358237133825047,
                   -0.06419234061997808,
                   0.12316117092283699]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 2.9585195977930233e-5, 0.007629678335342931,
                                -0.006422255808575575
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2, 1, 2, 37, 15, 4, 72, 17, 1, 3, 1, 51, 19, -10, -3, -12, -29, -6]
    @test rmsd(lpAlloc.shares, testshares) < 0.5

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [72, 51, 17, 15, 19, 37, 2, 2, 3, 4, 1, 1, 1, -6, -12, -3, -10, -28]
    @test gAlloc.shares == testshares

    ef = EffMeanVar(tickers, bl.post_ret, S)
    max_utility!(ef)
    testweights = [0.14260776745587128,
                   6.901942798740376e-5,
                   1.2766740891143295e-5,
                   0.045365294651479045,
                   4.194726182971812e-5,
                   1.1586565884664498e-5,
                   0.01914977971359382,
                   2.1132339523096494e-5,
                   0.7923412006480633,
                   2.1348584753668067e-5,
                   1.62933401627684e-5,
                   1.2154283282090457e-5,
                   9.035471131186343e-6,
                   4.967739391075272e-5,
                   6.0162807165563335e-5,
                   1.8139068087034055e-5,
                   2.798735680942031e-5,
                   3.6106233336921185e-5,
                   0.00012287135360935662,
                   5.729302627776753e-6]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0003371825953919165, 0.014633614785020623,
                                0.01767148153113711
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [1, 3, 20, 265, 1, 1, 1, 1, 1, 1, 1, 1]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [265, 1, 3, 20, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    max_utility!(ef, 2)
    testweights = [0.2609247351898595,
                   0.02638569233827588,
                   5.293812775164639e-6,
                   0.07878185546973393,
                   2.2036593955614676e-5,
                   5.770092524586731e-6,
                   0.014049832647449077,
                   3.272860283270112e-5,
                   0.47139844483683696,
                   1.292529732820289e-5,
                   1.3878046561076705e-5,
                   5.038948142316934e-6,
                   4.0665879965649306e-6,
                   0.09198265513247973,
                   0.0009304853004471541,
                   1.1968387047010711e-5,
                   1.7414533236457654e-5,
                   0.007950896824424044,
                   0.04746187219539807,
                   2.409162695955628e-6]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0002882829151052814, 0.012164116719399255,
                                0.017239062889474483
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [3, 1, 4, 14, 155, 11, 4]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [158, 2, 12, 5, 5, 2, 15, 3, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    efficient_return!(ef, 1.07^(1 / 252) - 1)
    testweights = [0.23899650932857186,
                   0.041407560268281914,
                   6.323875850913023e-6,
                   0.07382822905925486,
                   2.8545615591783717e-5,
                   7.073227346013539e-6,
                   0.011115627818180302,
                   8.240849476809082e-5,
                   0.38629720530881145,
                   1.6494775379471162e-5,
                   2.0573643340761348e-5,
                   5.9440748818651055e-6,
                   4.940650138885412e-6,
                   0.12127644442853346,
                   0.0018656653784090377,
                   1.6397171620540606e-5,
                   2.179362992947602e-5,
                   0.06517919126425785,
                   0.059820166575856246,
                   2.9054109952427134e-6]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.00026851381290565214, 0.011386791802521324,
                                0.016679752665623295
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2, 3, 5, 11, 129, 16, 1, 19, 6]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [129, 2, 16, 5, 19, 6, 3, 11, 1]
    @test gAlloc.shares == testshares

    efficient_risk!(ef, 0.2 / sqrt(252))
    testweights = [0.26888056712255604,
                   0.013622519100757985,
                   5.23637356927458e-6,
                   0.08088075743695511,
                   2.1215803141394203e-5,
                   5.6013945950326856e-6,
                   0.015688170229284676,
                   2.4686239840777793e-5,
                   0.5287420102967382,
                   1.2251188363527316e-5,
                   1.2302071841782674e-5,
                   5.005910806375696e-6,
                   3.976273978896604e-6,
                   0.06319624731747847,
                   0.0008023555396722106,
                   1.1020598368861438e-5,
                   1.66169454668449e-5,
                   0.0001932853653112542,
                   0.027873804753578674,
                   2.3700376946530716e-6]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.00029859789435197085, 0.01259921112342683,
                                0.017462438736202433
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [3, 4, 13, 176, 1, 8, 2]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [176, 2, 5, 9, 3, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    ef = EffMeanVar(tickers, bl.post_ret, S; market_neutral = true)
    max_utility!(ef)
    testweights = [0.9999043225428015,
                   0.42776988871978605,
                   -0.8984306105122349,
                   0.2768202903574887,
                   0.1366784686461898,
                   -0.9976181765552804,
                   0.05609616594146933,
                   -0.03387849468911428,
                   0.9973844528679411,
                   -0.09065360899458808,
                   -0.7727773037032114,
                   -0.06685887648263143,
                   -0.04133521621113476,
                   0.3094913932374971,
                   0.03316796490220134,
                   -0.10417104695856348,
                   -0.2149514066469411,
                   0.16777735904100596,
                   0.8155735504367417,
                   -0.9999891159394229]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.001224907340431768, 0.031040456412191674,
                                0.03692994662272042
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2,
                  6,
                  4,
                  14,
                  93,
                  10,
                  7,
                  12,
                  18,
                  -54,
                  -769,
                  -4,
                  -23,
                  -219,
                  -40,
                  -122,
                  -14,
                  -13,
                  -168]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [3, 79, 17, 5, 9, 3, 11, 14, 5, -168, -769, -54, -219, -13, -14, -23, -39,
                  -126, -4]
    @test gAlloc.shares == testshares

    max_utility!(ef, 2)
    testweights = [0.672753971405211,
                   0.22975742514551503,
                   -0.4456204757306331,
                   0.127448546384999,
                   0.09807865798088214,
                   -0.45665369585280474,
                   0.019723922861696427,
                   0.04410285869932412,
                   0.4977227151473107,
                   -0.02043310088121996,
                   -0.3448672170063536,
                   -0.01597618504875624,
                   -0.013122117363312412,
                   0.1396744582107124,
                   0.013159748606672853,
                   -0.023330065450981523,
                   -0.04601999211232336,
                   0.09549617046395437,
                   0.4280976232683553,
                   -0.9999932487282485]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0007953388785726502, 0.018178001504681237,
                                0.039429743495368326
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [3, 6, 3, 9, 2, 71, 8, 4, 12, 17, -27, -352, -5, -97, -9, -38, -3, -3,
                  -168]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [3, 70, 17, 6, 8, 3, 11, 3, 9, 3, -168, -352, -26, -98, -3, -4, -6, -10,
                  -39]
    @test gAlloc.shares == testshares

    efficient_return!(ef, 1.07^(1 / 252) - 1)
    testweights = [0.18912603081908547,
                   0.06980477778357273,
                   -0.10396232360018451,
                   0.033062727069433154,
                   0.049214392256857586,
                   -0.09807758604812665,
                   0.0012997574842149632,
                   0.034303740581922766,
                   0.12472151350121413,
                   0.004867747362026868,
                   -0.07461221263171136,
                   0.001967813769023307,
                   -0.0010090398129964078,
                   0.0323923448784401,
                   0.00014068402672721863,
                   0.004023108561416976,
                   0.017499074740317373,
                   0.03418715445503901,
                   0.125505354212989,
                   -0.44445505940926183]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0002685162237981721, 0.005936258827985843,
                                0.031995114653376004
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [3, 6, 3, 2, 6, 58, 2, 2, 6, 1, 1, 1, 14, 16, -6, -76, -21, -4, -75]
    @test rmsd(lpAlloc.shares, testshares) < 0.5

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [2, 15, 57, 6, 1, 5, 13, 3, 6, 1, 1, 1, 2, -74, -6, -75, -21, -4]
    @test gAlloc.shares == testshares

    efficient_risk!(ef, 0.2 / sqrt(252))
    testweights = [0.40141858123412566,
                   0.14815791086894223,
                   -0.2206717713903187,
                   0.07017770435741666,
                   0.10444566346157438,
                   -0.2081871219561556,
                   0.0027611928420925034,
                   0.07279686118401488,
                   0.26473189123676255,
                   0.010325816468801678,
                   -0.15837700939809643,
                   0.0041728894574808145,
                   -0.002143317000434125,
                   0.06875700538237957,
                   0.00030060578424630566,
                   0.008532954019674478,
                   0.03712478090928557,
                   0.07255825985094812,
                   0.2663810451394503,
                   -0.9432639424521909]
    @test ef.weights ≈ testweights
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0005699007357381988, 0.012599157816898571,
                                0.03899592344930975
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [3, 6, 3, 2, 6, 58, 2, 2, 6, 1, 1, 1, 14, 16, -13, -161, -45, -12, -159]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [2, 15, 57, 6, 1, 5, 13, 3, 6, 1, 1, 1, 2, -158, -13, -160, -45, -7]
    @test gAlloc.shares == testshares

    # L1 Regularisation
    mu = vec(ret_model(MRet(), Matrix(dropmissing(returns))))
    S = cov(Cov(), Matrix(dropmissing(returns)))

    n = length(tickers)
    prev_weights = fill(1 / n, n)

    k = 0.001 / 252
    ef = EffMeanVar(tickers,
                    mu,
                    S;
                    extra_vars = [:(0 <= l1)],
                    extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                           MOI.NormOneCone($(n + 1)))],
                    extra_obj_terms = [quote
                                           $k * model[:l1]
                                       end],)
    min_risk!(ef)
    testweights = [0.01965515635745056,
                   0.04101978347901424,
                   0.014559050480109794,
                   0.028065301778287706,
                   0.015559350769150472,
                   0.04975286766349492,
                   4.7542317751952435e-5,
                   0.1352879163594813,
                   7.195275622894228e-5,
                   0.009576616550054791,
                   0.2749406687911405,
                   4.262875608625715e-5,
                   5.167209111243921e-5,
                   0.10530595610426109,
                   0.001534764310509473,
                   0.0209617128187514,
                   0.019482731905338255,
                   0.17476349291929663,
                   0.00010366089048120374,
                   0.08921717290199815]
    @test ef.weights ≈ testweights

    max_utility!(ef)
    testweights = [3.106817914699003e-6,
                   3.617299899510345e-6,
                   4.478154493500536e-6,
                   3.6380382235777383e-6,
                   0.999938427005369,
                   1.26633278054637e-6,
                   7.852142433025727e-6,
                   2.3534863220499526e-6,
                   3.840056263484496e-6,
                   2.503303175604993e-6,
                   2.312592969464391e-6,
                   1.1454438364235091e-6,
                   6.929005361283515e-7,
                   1.7564772322601408e-6,
                   8.089676765297588e-7,
                   6.172023403945758e-6,
                   5.802380121012501e-6,
                   2.560034446838593e-6,
                   4.475234988484612e-6,
                   3.191307914029851e-6]
    @test ef.weights ≈ testweights

    efficient_return!(ef, 1.05^(1 / 252) - 1)
    testweights = [0.019612904885579697,
                   0.04088065227126394,
                   0.014604884523189027,
                   0.028043823317505035,
                   0.015633580410044916,
                   0.04980290300139978,
                   6.928116739059165e-5,
                   0.13529193113161117,
                   8.384568088539031e-5,
                   0.009652065495971432,
                   0.274919199207126,
                   4.921627654823037e-5,
                   5.185122368143861e-5,
                   0.10524241832747007,
                   0.0015429505803039058,
                   0.020958590980923092,
                   0.019517949605639014,
                   0.1747417432488495,
                   0.0001181995050675282,
                   0.08918200915955049]
    @test ef.weights ≈ testweights

    efficient_risk!(ef, 0.15 / sqrt(252))
    testweights = [9.535085714095306e-6,
                   0.0090171328246857,
                   0.0004393977860515455,
                   6.0087436682392404e-5,
                   0.2597826661132842,
                   2.5780102745638448e-6,
                   0.0006648152142365959,
                   0.07236592423509898,
                   1.1549174269124644e-5,
                   1.2976368720191418e-5,
                   0.1632614766898498,
                   1.4438498036850809e-6,
                   8.834015601879667e-7,
                   6.615355050227938e-6,
                   1.1107366259770485e-6,
                   0.070094986698512,
                   0.2008167514122795,
                   0.1027193044103887,
                   0.08724461627047166,
                   0.033486148926440695]
    @test ef.weights ≈ testweights

    k = 0.001 / 252
    ef = EffMeanVar(tickers,
                    mu,
                    S;
                    extra_vars = [:(l1 >= 0)],
                    extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                           MOI.NormOneCone($(n + 1)))],
                    extra_obj_terms = [quote
                                           $k * model[:l1]
                                       end],)
    max_sharpe!(ef)
    testweights = [2.4089378278495e-11,
                   9.505987670039489e-11,
                   1.0030867036936811e-10,
                   5.028523293690665e-11,
                   0.5154468741375988,
                   -8.032618137567141e-13,
                   0.00696791590700499,
                   6.926345552027405e-11,
                   5.858786490286024e-11,
                   2.865765135365044e-11,
                   6.728064683208615e-11,
                   -4.3241871956828704e-12,
                   -7.33715616249326e-12,
                   9.09552363681336e-12,
                   -6.562975868165075e-12,
                   0.1063640822060386,
                   0.268739876516344,
                   7.775616138856622e-11,
                   0.10248125059695108,
                   7.47055097880935e-11]

    @test ef.weights ≈ testweights

    k = 0.01 / 252
    ef = EffMeanVar(tickers,
                    mu,
                    S;
                    extra_vars = [:(0 <= l1)],
                    extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                           MOI.NormOneCone($(n + 1))),
                                         :(model[:w][6] == 0.2)],
                    extra_obj_terms = [quote
                                           $k * model[:l1]
                                       end],)
    max_sharpe!(ef)
    testweights = [4.642431527000067e-12,
                   1.9498213614020918e-11,
                   2.20520580710831e-11,
                   1.2318737094772983e-11,
                   0.6061115514502201,
                   0.20000000000000004,
                   0.004735238374447829,
                   7.947257894608764e-12,
                   2.053849937561063e-11,
                   4.1763168565873735e-12,
                   6.7742075836079495e-12,
                   -6.0546557114461266e-12,
                   -7.457614383025367e-12,
                   -1.5835953329105947e-12,
                   -7.099639056485276e-12,
                   0.08686140010036032,
                   0.10229180988019464,
                   9.950541232084451e-12,
                   9.650146803651886e-11,
                   1.2572773543575703e-11]
    @test ef.weights ≈ testweights

    k = 0.0001 / 252
    ef = EffMeanVar(tickers,
                    mu,
                    S;
                    extra_vars = [:(0 <= l1)],
                    extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                           MOI.NormOneCone($(n + 1))),
                                         :(model[:w][6] == 0.2),
                                         :(model[:w][1] >= 0.01),
                                         :(model[:w][16] <= 0.03)],
                    extra_obj_terms = [quote
                                           $k * model[:l1]
                                       end],)
    constraint = max_sharpe!(ef)
    testweights = [0.0099999999953802,
                   1.1905073891006014e-12,
                   1.6863059162792628e-12,
                   -1.6171141312020976e-12,
                   0.6225796026029272,
                   0.19999999999999998,
                   0.007396553459131181,
                   -3.1520985812153166e-12,
                   2.357223572828574e-12,
                   -4.499157712400907e-12,
                   -3.680038796214735e-12,
                   -8.387386229371857e-12,
                   -8.895989902613227e-12,
                   -6.77727163398283e-12,
                   -8.775008980638848e-12,
                   0.02999999999453233,
                   0.13002384395029154,
                   -2.5015740293427294e-12,
                   4.2126192889422335e-11,
                   -1.336793371276376e-12]
    @test ef.weights ≈ testweights
end

@testset "Mean Semivariance" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])
    tickers = names(df)[2:end]

    mean_ret = vec(ret_model(MRet(), Matrix(returns)))
    S = cov(Cov(), Matrix(returns))

    ef = EffMeanSemivar(tickers, mean_ret, S; market_neutral = true)
    sectors = ["Tech", "Medical", "RealEstate", "Oil"]
    sector_map = Dict(tickers[1:4:20] .=> sectors[1])
    merge!(sector_map, Dict(tickers[2:4:20] .=> sectors[2]))
    merge!(sector_map, Dict(tickers[3:4:20] .=> sectors[3]))
    merge!(sector_map, Dict(tickers[4:4:20] .=> sectors[4]))
    sector_lower = Dict([(sectors[1], 0.2), (sectors[2], 0.1), (sectors[3], 0.15),
                         (sectors[4], 0.05)])
    sector_upper = Dict([(sectors[1], 0.4), (sectors[2], 0.5), (sectors[3], 0.2),
                         (sectors[4], 0.2)])

    # Do it for the warning.
    add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)

    spy_prices = CSV.read("./assets/spy_prices.csv", DataFrame)
    delta = market_implied_risk_aversion(spy_prices[!, 2])
    # In the order of the dataframes, the
    mcapsdf = DataFrame(; ticker = tickers,
                        mcap = [927e9,
                                1.19e12,
                                574e9,
                                533e9,
                                867e9,
                                96e9,
                                43e9,
                                339e9,
                                301e9,
                                51e9,
                                61e9,
                                78e9,
                                0,
                                295e9,
                                1e9,
                                22e9,
                                288e9,
                                212e9,
                                422e9,
                                102e9])

    prior = market_implied_prior_returns(mcapsdf[!, 2], S, delta)

    # 1. SBUX drop by 20%
    # 2. GOOG outperforms FB by 10%
    # 3. BAC and JPM will outperform T and GE by 15%
    views = [-0.20, 0.10, 0.15] / 252
    picking = hcat([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0])

    bl = BlackLitterman(mcapsdf[!, 1],
                        S;
                        rf = 0,
                        tau = 0.01,
                        pi = prior, # either a vector, `nothing`, `:Equal`, or `:market`
                        Q = views,
                        P = picking,)

    ef = EffMeanSemivar(tickers, bl.post_ret, Matrix(returns))
    @test (0.0, 0.0, 0.0) == portfolio_performance(ef)

    max_sharpe!(ef)
    mumax, varmax, smax = portfolio_performance(ef)

    ef = EffMeanSemivar(tickers, bl.post_ret, Matrix(returns))
    efficient_risk!(ef, varmax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 5e-2)
    @test isapprox(sigma, varmax, rtol = 5e-2)
    @test isapprox(sr, smax, rtol = 5e-4)

    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 5e-5)
    @test isapprox(sigma, varmax, rtol = 5e-4)
    @test isapprox(sr, smax, rtol = 5e-4)

    ef = EffMeanSemivar(tickers, bl.post_ret, Matrix(returns))
    max_sharpe!(ef, 1.03^(1 / 252) - 1)
    mumax, varmax, smax = portfolio_performance(ef)

    ef = EffMeanSemivar(tickers, bl.post_ret, Matrix(returns))
    efficient_risk!(ef, varmax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 5e-2)
    @test isapprox(sigma, varmax, rtol = 5e-2)
    @test isapprox(sr, smax, rtol = 5e-3)

    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 5e-5)
    @test isapprox(sigma, varmax, rtol = 5e-5)
    @test isapprox(sr, smax, rtol = 5e-4)

    ef = EffMeanSemivar(tickers, bl.post_ret, Matrix(returns); weight_bounds = (-1, 1))
    max_sharpe!(ef)
    mumax, varmax, smax = portfolio_performance(ef)

    ef = EffMeanSemivar(tickers, bl.post_ret, Matrix(returns); weight_bounds = (-1, 1))
    efficient_risk!(ef, varmax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 5e-3)
    @test isapprox(sigma, varmax, rtol = 5e-3)
    @test isapprox(sr, smax, rtol = 1e-5)

    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 5e-5)
    @test isapprox(sigma, varmax, rtol = 1e-5)
    @test isapprox(sr, smax, rtol = 1e-5)

    ef = EffMeanSemivar(tickers, bl.post_ret, Matrix(returns); weight_bounds = (-1, 1))
    max_sharpe!(ef, 1.03^(1 / 252) - 1)
    mumax, varmax, smax = portfolio_performance(ef)

    ef = EffMeanSemivar(tickers, bl.post_ret, Matrix(returns); weight_bounds = (-1, 1))
    efficient_risk!(ef, varmax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 5e-3)
    @test isapprox(sigma, varmax, rtol = 5e-3)
    @test isapprox(sr, smax, rtol = 1e-4)

    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mu, mumax, rtol = 1e-5)
    @test isapprox(sigma, varmax, rtol = 1e-4)
    @test isapprox(sr, smax, rtol = 1e-4)

    ef = EffMeanSemivar(tickers, bl.post_ret, Matrix(returns); target = 0)
    min_risk!(ef)
    testweights = [-8.8369599e-09,
                   0.0521130667306068,
                   -5.54696556e-08,
                   0.0032731356678013,
                   0.0324650564450746,
                   6.3744722e-09,
                   3.37125568e-07,
                   0.1255659243075475,
                   1.633273845e-07,
                   4.5898985e-08,
                   0.3061337299353124,
                   -6.68458646e-08,
                   -1.4187474e-08,
                   0.0967467542201713,
                   0.004653018274916,
                   0.0198838224019178,
                   0.0018092577849403,
                   0.2315968397352792,
                   1.032235721e-07,
                   0.1257588839643445]
    @test isapprox(ef.weights, testweights, rtol = 1e-1)
    mu, sigma, sr = portfolio_performance(ef; verbose = true)
    mutest, sigmatest, srtest = 4.8745206589257286e-5, 0.005355195554829349,
                                -0.005572109382351622
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [3, 1, 1, 2, 15, 87, 13, 4, 3, 65, 22]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [86, 64, 21, 15, 13, 3, 3, 3, 1, 1, 1]
    @test gAlloc.shares == testshares

    max_utility!(ef)
    testweights = [1.100000000000000e-15,
                   1.200000000000000e-15,
                   1.100000000000000e-15,
                   1.000000000000000e-15,
                   1.200000000000000e-15,
                   1.000000000000000e-15,
                   5.000000000000000e-16,
                   1.500000000000000e-15,
                   9.999999999999784e-01,
                   9.000000000000000e-16,
                   1.500000000000000e-15,
                   8.000000000000000e-16,
                   7.000000000000000e-16,
                   1.200000000000000e-15,
                   9.000000000000000e-16,
                   1.100000000000000e-15,
                   1.000000000000000e-15,
                   1.300000000000000e-15,
                   5.000000000000000e-16,
                   1.300000000000000e-15]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.00036247930698906614, 0.01152281097796569,
                                0.02463759629028793
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [334]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [334, 1]
    @test gAlloc.shares == testshares

    max_utility!(ef, 2)
    testweights = [0.1462022850123406,
                   1.72696369e-08,
                   2.8627388e-09,
                   2.761140791e-07,
                   1.01336466e-08,
                   2.513371e-09,
                   0.0066277509841478,
                   4.751995e-09,
                   0.8471695614090771,
                   4.6036865e-09,
                   3.7028104e-09,
                   2.755922e-09,
                   1.9606178e-09,
                   1.04848137e-08,
                   1.19804731e-08,
                   4.6723839e-09,
                   6.2853476e-09,
                   8.8590514e-09,
                   3.23131513e-08,
                   1.3307094e-09]
    @test isapprox(ef.weights, testweights, rtol = 1e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0003429692295918099, 0.010512337102958619,
                                0.025149905774306962
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [1, 1, 7, 284, 1, 1, 1, 1, 1]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [283, 1, 7, 1, 1, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    efficient_return!(ef, 0.09 / 252)
    testweights = [3.904746815385050e-02,
                   5.923397958900000e-06,
                   9.829962419900001e-06,
                   4.485521037400000e-06,
                   6.398473463000000e-06,
                   1.058079920190000e-05,
                   8.609310912354300e-03,
                   9.421584972200000e-06,
                   9.522038516156038e-01,
                   7.426588225800000e-06,
                   1.019932942990000e-05,
                   9.315528047900000e-06,
                   1.118486159000000e-05,
                   6.553049888500000e-06,
                   5.498591175600000e-06,
                   8.239081503600000e-06,
                   6.940805469200000e-06,
                   7.319651581700000e-06,
                   3.430088268500000e-06,
                   1.661143469400000e-05]
    @test isapprox(ef.weights, testweights, rtol = 1e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0003571336094192716, 0.011229212482586095,
                                0.024805717040850942
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [1, 9, 319, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [318, 9, 1, 1, 1, 1, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    efficient_risk!(ef, 0.13 / sqrt(252))
    testweights = [2.874750375415112e-01,
                   6.195275902493500e-02,
                   4.152863104000000e-07,
                   3.754883121164440e-02,
                   2.335468505800000e-06,
                   4.139595347000000e-07,
                   5.180351073718700e-03,
                   2.282885531500000e-06,
                   4.040036784043188e-01,
                   8.569500938000000e-07,
                   1.073950326300000e-06,
                   4.148599173000000e-07,
                   3.056287327000000e-07,
                   5.362159168381390e-02,
                   3.496258672140000e-05,
                   1.238539893500000e-06,
                   1.317666202400000e-06,
                   6.146402530123640e-02,
                   8.870790761308990e-02,
                   2.002169627000000e-07]
    @test isapprox(ef.weights, testweights, rtol = 5e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.00027396309257425724, 0.008136621606917017,
                                0.024012195727956046
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [3, 3, 2, 135, 7, 17, 8]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [135, 3, 8, 3, 17, 6, 2, 6, 1, 1]
    @test gAlloc.shares == testshares

    ef = EffMeanSemivar(tickers,
                        bl.post_ret,
                        Matrix(returns);
                        market_neutral = true,
                        target = 0)
    min_risk!(ef)
    testweights = [-6.87746970453e-05,
                   -1.40007211953e-05,
                   2.38011464746e-05,
                   4.6974224158e-06,
                   0.000127662765229,
                   -0.0001817372873211,
                   1.84999231954e-05,
                   -2.16770346062e-05,
                   -5.9067519062e-05,
                   1.08932533038e-05,
                   1.38594330145e-05,
                   -4.04318857464e-05,
                   -1.23208757829e-05,
                   -7.53221096413e-05,
                   -3.7905535553e-05,
                   3.82318960547e-05,
                   7.14342416822e-05,
                   -1.0279450203e-05,
                   0.0002096363297839,
                   2.800705002e-06]
    @test isapprox(ef.weights, testweights, atol = 1e-1)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 9.68355572016792e-06, 3.830973475511849e-05,
                                -521.8077486586869
    @test isapprox(mu, mutest, atol = 1e-3)
    @test isapprox(sigma, sigmatest, atol = 1e-2)

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2, 2, 34, 5, 6, 10, 8, 36, -1]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [36, 2, 7, 10, 2, 37, 8, 5, 1, -1]
    @test gAlloc.shares == testshares

    max_utility!(ef)
    testweights = [0.9999999987191048,
                   0.8025061567310201,
                   -0.9999999983660633,
                   0.6315188600428658,
                   0.4922749743437155,
                   -0.9999999990605686,
                   0.1762382499326327,
                   -0.733252470272586,
                   0.9999999993131146,
                   -0.2585591997727487,
                   -0.9999999983706372,
                   -0.2591719865317014,
                   -0.1281535712856913,
                   0.4788874233441203,
                   -0.0263096469774795,
                   -0.0874576034104335,
                   -0.4552360118252822,
                   0.3667148242725146,
                   0.9999999989095066,
                   -0.9999999997354032]
    @test isapprox(ef.weights, testweights, rtol = 1e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0014675481684477804, 0.027106278640627462,
                                0.05124138377229412
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [1,
                  8,
                  6,
                  1,
                  30,
                  57,
                  11,
                  17,
                  15,
                  -60,
                  -771,
                  -85,
                  -66,
                  -284,
                  -155,
                  -387,
                  -17,
                  -12,
                  -27,
                  -168]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [57,
                  15,
                  1,
                  8,
                  6,
                  1,
                  11,
                  17,
                  30,
                  -168,
                  -771,
                  -284,
                  -60,
                  -85,
                  -27,
                  -154,
                  -66,
                  -389,
                  -12,
                  -17]
    @test gAlloc.shares == testshares

    max_utility!(ef, 2)
    testweights = [0.9999999857360712,
                   0.4685479704127186,
                   -0.8592431647111499,
                   0.2920727402492227,
                   0.3445846893910677,
                   -0.999999984329561,
                   0.0734859850410818,
                   -0.172920245654445,
                   0.9999999347257726,
                   -0.2143822719074016,
                   -0.937463177415737,
                   -0.1008400899077779,
                   -0.0510006884730285,
                   0.2396024358477434,
                   -0.0269876564283418,
                   -0.038065236125291,
                   -0.314010116118947,
                   0.2966189911453665,
                   0.9999998963472027,
                   -0.9999999978245666]
    @test isapprox(ef.weights, testweights, rtol = 1e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0012893686628308249, 0.021861187259986887,
                                0.05538508528593527
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2,
                  5,
                  2,
                  1,
                  12,
                  69,
                  6,
                  16,
                  19,
                  -52,
                  -772,
                  -20,
                  -55,
                  -266,
                  -61,
                  -156,
                  -18,
                  -5,
                  -18,
                  -168]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [2,
                  70,
                  19,
                  6,
                  17,
                  4,
                  7,
                  15,
                  -168,
                  -771,
                  -265,
                  -52,
                  -18,
                  -55,
                  -20,
                  -60,
                  -154,
                  -6,
                  -18]
    @test gAlloc.shares == testshares

    efficient_return!(ef, 0.09 / 252)
    testweights = [0.26587342118737,
                   0.0965371129134666,
                   -0.1083440586709533,
                   0.0472631405569012,
                   0.0925397377541989,
                   -0.16577616699613,
                   0.0072119213407655,
                   0.0227400114825395,
                   0.10350347319794,
                   -0.0071334231129069,
                   -0.0883015612045468,
                   0.0134635120313752,
                   -0.0009731371433595,
                   0.010494163524052,
                   -0.0066909561294794,
                   0.0173324674445928,
                   -0.0122390919128145,
                   0.0763576474308947,
                   0.232051643617103,
                   -0.5959098573110091]
    @test isapprox(ef.weights, testweights, rtol = 5e-2)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.0003571501978227821, 0.005139043300749973,
                                0.05420566427169036
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2, 6, 3, 1, 7, 3, 35, 7, 2, 3, 22, 21, -6, -128, -2, -26, -3, -5, -1,
                  -100]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [2, 21, 35, 6, 1, 21, 3, 3, 3, 8, 2, 8, -100, -128, -6, -25, -1, -2, -5,
                  -3]
    @test gAlloc.shares == testshares

    efficient_risk!(ef, 0.13 / sqrt(252))
    testweights = [0.4238401998849985,
                   0.1538686236477102,
                   -0.1727359058771851,
                   0.0753461438850074,
                   0.1475612389187346,
                   -0.2642402283037698,
                   0.0115008067529429,
                   0.0362772030456925,
                   0.1649564844571324,
                   -0.0113530132083892,
                   -0.1407414688712099,
                   0.0214516748483527,
                   -0.0015593312594264,
                   0.0166505514359901,
                   -0.0106664370188864,
                   0.0276341390914952,
                   -0.0195065353158665,
                   0.1216951812979697,
                   0.3699322405966654,
                   -0.9499115680157854]
    @test isapprox(ef.weights, testweights, rtol = 5e-3)
    mu, sigma, sr = portfolio_performance(ef)
    mutest, sigmatest, srtest = 0.00056832498954627, 0.008176190287484111,
                                0.05989831820711182
    @test mu ≈ mutest
    @test sigma ≈ sigmatest
    @test sr ≈ srtest

    ef.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]);
                                    investment = 10000)
    testshares = [2, 6, 3, 1, 7, 3, 35, 7, 2, 3, 22, 21, -10, -204, -3, -40, -8, -11, -1,
                  -160]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), ef, Vector(df[end, ef.tickers]);
                                   investment = 10000)
    testshares = [2, 21, 35, 6, 1, 21, 3, 3, 3, 8, 2, 8, -159, -204, -11, -39, -1, -3, -7,
                  -4]
    @test gAlloc.shares == testshares

    # L1 Regularisation
    mean_ret = ret_model(MRet(), Matrix(returns))
    S = cov(Cov(), Matrix(returns))
    n = length(tickers)
    prev_weights = fill(1 / n, n)

    k = 0.00001
    ef = EffMeanSemivar(tickers,
                        mean_ret,
                        Matrix(returns);
                        target = 0,
                        extra_vars = [:(0 <= l1)],
                        extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                               MOI.NormOneCone($(n + 1)))
                                             # :(model[:w][6] == 0.2),
                                             # :(model[:w][1] >= 0.01),
                                             # :(model[:w][16] <= 0.03),
                                             ],
                        extra_obj_terms = [quote
                                               $k * model[:l1]
                                           end],)
    min_risk!(ef)
    testweights = [0.05000003815106,
                   0.0500000323406222,
                   0.0305470536068529,
                   0.0119165898801848,
                   0.0499999897938178,
                   0.0500000778586521,
                   -9.70902283e-08,
                   0.1050006178939432,
                   -1.074573985e-07,
                   0.0217993269186788,
                   0.2267216752364789,
                   4.54544546e-08,
                   2.475203931e-07,
                   0.0500000608483738,
                   0.0195085389944512,
                   0.0500000224593989,
                   0.050000026895504,
                   0.1345057919377536,
                   0.0499999927807392,
                   0.0500000757003792]
    @test isapprox(ef.weights, testweights, rtol = 5e-2)

    max_utility!(ef)
    testweights = [-1.926600572e-07,
                   -4.32976813e-08,
                   -2.100198937e-07,
                   -2.227540243e-07,
                   0.9999993287957882,
                   1.339119054e-07,
                   -1.90544883e-07,
                   2.103890502e-07,
                   2.02309627e-08,
                   4.75400441e-08,
                   2.628872504e-07,
                   -6.86900634e-08,
                   1.741756607e-07,
                   1.917324164e-07,
                   1.195934128e-07,
                   1.58547084e-07,
                   -1.05257917e-08,
                   2.052601162e-07,
                   9.02686384e-08,
                   -5.2371344e-09]
    @test isapprox(ef.weights, testweights, rtol = 1e-4)

    efficient_return!(ef, 1.17^(1 / 252) - 1)
    testweights = [0.048243135588022394,
                   0.05005015448888129,
                   0.04950738917530671,
                   0.013508969423008067,
                   0.14612074950023246,
                   5.161083914557875e-5,
                   0.00020386952047784184,
                   0.0794989456777301,
                   0.014025026462612935,
                   0.018565248696993213,
                   0.21031304840072806,
                   2.3540968703639644e-5,
                   1.1619378307027638e-5,
                   0.049896168124027404,
                   2.0172531262396392e-5,
                   0.050739534329305475,
                   0.050232893750889594,
                   0.118803721194177,
                   0.05000934982112427,
                   0.05017485212906453]
    @test isapprox(ef.weights, testweights, rtol = 5e-2)

    efficient_risk!(ef, 0.13 / sqrt(252))
    testweights = [6.446548926626026e-6,
                   2.3069734505128152e-5,
                   1.1209416248366952e-5,
                   6.328326555291172e-6,
                   0.554876423425052,
                   2.1111052484695074e-6,
                   9.774524119877544e-6,
                   3.489779052250414e-5,
                   1.041313825931083e-5,
                   6.479023479848044e-6,
                   0.03673192054364961,
                   1.4210013701492194e-6,
                   8.529296844084627e-7,
                   4.8646790787061755e-6,
                   1.0498564580430158e-6,
                   0.13235577066930504,
                   0.19466813411537295,
                   0.00013420077214587974,
                   0.08109324747504564,
                   2.1384924972134433e-5]
    @test ef.weights ≈ testweights

    k = 0.00001
    ef = EffMeanSemivar(tickers,
                        mean_ret,
                        Matrix(returns);
                        target = 0,
                        extra_vars = [:(0 <= l1)],
                        extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                               MOI.NormOneCone($(n + 1)))
                                             # :(model[:w][6] == 0.2),
                                             # :(model[:w][1] >= 0.01),
                                             # :(model[:w][16] <= 0.03),
                                             ],
                        extra_obj_terms = [quote
                                               $k * model[:l1]
                                           end],)
    max_sharpe!(ef)
    mumax, sigmamax, srmax = portfolio_performance(ef; verbose = true)

    k = 0.00001
    ef = EffMeanSemivar(tickers,
                        mean_ret,
                        Matrix(returns);
                        target = 0,
                        extra_vars = [:(0 <= l1)],
                        extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                               MOI.NormOneCone($(n + 1)))
                                             # :(model[:w][6] == 0.2),
                                             # :(model[:w][1] >= 0.01),
                                             # :(model[:w][16] <= 0.03),
                                             ],
                        extra_obj_terms = [quote
                                               $k * model[:l1]
                                           end],)
    efficient_return!(ef, mumax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mumax, mu, rtol = 1e-4)
    @test isapprox(sigmamax, sigma, rtol = 1e-2)
    @test isapprox(srmax, sr, rtol = 1e-2)

    efficient_risk!(ef, sigmamax)
    mu, sigma, sr = portfolio_performance(ef)
    @test isapprox(mumax, mu, rtol = 5e-3)
    @test isapprox(sigmamax, sigma, rtol = 5e-3)
    @test isapprox(srmax, sr, rtol = 1e-4)

    k = 0.00001
    ef = EffMeanSemivar(tickers,
                        mean_ret,
                        Matrix(returns);
                        target = 0,
                        extra_vars = [:(0 <= l1)],
                        extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                               MOI.NormOneCone($(n + 1))),
                                             :(model[:w][6] == 0.2),
                                             :(model[:w][1] >= 0.01),
                                             :(model[:w][16] <= 0.03)],
                        extra_obj_terms = [quote
                                               $k * model[:l1]
                                           end],)
    max_sharpe!(ef)

    @test ef.weights[6] ≈ 0.2
    @test ef.weights[1] >= 0.01
    @test ef.weights[16] <= 0.03
end

@testset "Efficient CVaR" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])
    tickers = names(df)[2:end]

    mu = vec(ret_model(MRet(), Matrix(returns)))
    S = cov(Cov(), Matrix(returns))

    spy_prices = CSV.read("./assets/spy_prices.csv", DataFrame)
    delta = market_implied_risk_aversion(spy_prices[!, 2])
    # In the order of the dataframes, the
    mcapsdf = DataFrame(; ticker = tickers,
                        mcap = [927e9,
                                1.19e12,
                                574e9,
                                533e9,
                                867e9,
                                96e9,
                                43e9,
                                339e9,
                                301e9,
                                51e9,
                                61e9,
                                78e9,
                                0,
                                295e9,
                                1e9,
                                22e9,
                                288e9,
                                212e9,
                                422e9,
                                102e9])

    prior = market_implied_prior_returns(mcapsdf[!, 2], S, delta)

    # 1. SBUX drop by 20%
    # 2. GOOG outperforms FB by 10%
    # 3. BAC and JPM will outperform T and GE by 15%
    views = [-0.20, 0.10, 0.15] / 252
    picking = hcat([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0])

    bl = BlackLitterman(mcapsdf[!, 1],
                        S;
                        rf = 0,
                        tau = 0.01,
                        pi = prior, # either a vector, `nothing`, `:Equal`, or `:market`
                        Q = views,
                        P = picking,)

    cv = EffCVaR(tickers, bl.post_ret, Matrix(returns))
    min_risk!(cv)
    testweights = [1.196700000000000e-12,
                   4.242033345496910e-02,
                   1.247300000000000e-12,
                   3.092200000000000e-12,
                   7.574027808866600e-03,
                   1.305400000000000e-12,
                   1.735000000000000e-13,
                   9.464947870552300e-02,
                   5.519000000000000e-13,
                   7.337000000000000e-13,
                   3.040110655916818e-01,
                   6.947000000000000e-13,
                   3.879000000000000e-13,
                   6.564167130410940e-02,
                   1.189600000000000e-12,
                   2.937161110264830e-02,
                   1.548900000000000e-12,
                   3.663101128889494e-01,
                   7.108000000000000e-13,
                   9.002169913041960e-02]
    @test isapprox(cv.weights, testweights, rtol = 1e-3)
    mu, sigma = portfolio_performance(cv; verbose = true)
    mutest, sigmatest = (1 + 0.014253439792781208)^(1 / 252) - 1, 0.017049502122532846
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)

    cv.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]);
                                    investment = 10000)
    testshares = [3, 11, 86, 8, 4, 103, 15]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cv, Vector(df[end, cv.tickers]);
                                   investment = 10000)
    testshares = [102, 86, 11, 16, 8, 3, 4]
    @test gAlloc.shares == testshares

    efficient_return!(cv, 0.07 / 252)
    testweights = [0.3849302546640336,
                   0.03565202448702544,
                   1.1373985731736585e-7,
                   3.127659454203103e-6,
                   5.062290484467805e-7,
                   1.305592902496062e-7,
                   5.127224047979289e-7,
                   6.693652992428642e-7,
                   0.4467982488922247,
                   2.2687320422429358e-7,
                   3.345255435369585e-7,
                   1.519167044873033e-7,
                   9.816565968243053e-8,
                   0.10832173182377393,
                   3.437269744381875e-6,
                   3.764555886872321e-7,
                   3.062970067942558e-7,
                   0.02428383324651445,
                   3.861479324887282e-6,
                   5.3628296751594037e-8]
    @test cv.weights ≈ testweights
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 0.07 / 252, 0.028255572821056847
    @test isapprox(mu, mutest, rtol = 1e-4)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)

    lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]);
                                    investment = 10000)
    testshares = [4, 2, 148, 13, 4]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cv, Vector(df[end, cv.tickers]);
                                   investment = 10000)
    testshares = [149, 3, 14, 3, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    efficient_risk!(cv, 0.14813)
    testweights = [2.058627316294932e-5,
                   1.4626530158677794e-5,
                   7.72406444506308e-6,
                   2.2013645997865593e-5,
                   1.3213683123887514e-5,
                   7.060298688294134e-6,
                   5.498815200839621e-5,
                   8.133727891459036e-6,
                   0.9997191958106376,
                   1.0810430264358776e-5,
                   7.385708983996071e-6,
                   8.263317937127186e-6,
                   6.616339694798414e-6,
                   1.2742435494251256e-5,
                   1.594969804407447e-5,
                   9.568337529768926e-6,
                   1.1828385177918504e-5,
                   1.1076762385040825e-5,
                   3.3977017923655226e-5,
                   4.239380450804566e-6]
    @test isapprox(cv.weights, testweights, rtol = 1e-6)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 0.00036251807194545517, 0.148136422458718
    @test isapprox(mu, mutest, atol = 1e-6)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)

    cv.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]);
                                    investment = 10000)
    testshares = [1, 334, 1]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cv, Vector(df[end, cv.tickers]);
                                   investment = 10000)
    testshares = [334.0, 1.0, 1.0]
    @test gAlloc.shares == testshares

    cv = EffCVaR(tickers, bl.post_ret, Matrix(returns); market_neutral = true)
    min_risk!(cv)
    testweights = [-2.2800e-12,
                   -5.5010e-13,
                   5.6220e-13,
                   -3.1620e-13,
                   5.1780e-12,
                   -4.6641e-12,
                   8.1920e-13,
                   -1.2637e-12,
                   -2.9427e-12,
                   2.0640e-13,
                   3.8680e-13,
                   -1.8054e-12,
                   -5.7220e-13,
                   -3.0342e-12,
                   -1.6556e-12,
                   1.6296e-12,
                   2.5777e-12,
                   -4.7750e-13,
                   8.4177e-12,
                   -2.1610e-13]
    @test isapprox(cv.weights, testweights, atol = 1e-4)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 3.279006735579045e-13, 5.19535240389637e-13
    @test isapprox(mu, mutest, atol = 1e-1)
    @test isapprox(sigma, sigmatest, atol = 1e-1)

    cv.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]);
                                    investment = 10000)
    testshares = [2, 2, 41, 2, 4, 11, 7, 38]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cv, Vector(df[end, cv.tickers]);
                                   investment = 10000)
    testshares = [38, 2, 7, 12, 43, 1, 5, 3]
    @test gAlloc.shares == testshares

    efficient_return!(cv, 0.07 / 252)
    testweights = [0.119904777207769,
                   0.105260613527609,
                   -0.059118345889513,
                   0.047994829008171,
                   0.080535064283592,
                   -0.07038968929214,
                   -0.000695786742806,
                   0.054006019393697,
                   0.088529728836899,
                   -0.01637772952951,
                   -0.063653950435258,
                   0.012986548220159,
                   0.019165300733247,
                   0.021582076971202,
                   -0.00296786688154,
                   0.018096675589053,
                   -0.024160465458667,
                   0.072693909074444,
                   0.137412159501253,
                   -0.540803868117662]
    @test isapprox(cv.weights, testweights, rtol = 1e-3)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 0.07000000000000023 / 252, 0.01208285385909055
    @test isapprox(mu, mutest, rtol = 5e-5)
    @test isapprox(sigma, sigmatest, rtol = 1e-3)

    cv.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]);
                                    investment = 10000)
    testshares = [1, 8, 4, 1, 8, 38, 10, 75, 4, 3, 26, 16, -4, -54, -1, -4, -18, -2, -1,
                  -91]
    @test rmsd(lpAlloc.shares, testshares) < 0.5

    gAlloc, remaining = Allocation(Greedy(), cv, Vector(df[end, cv.tickers]);
                                   investment = 10000)
    testshares = [16, 1, 8, 38, 1, 26, 8, 4, 4, 75, 3, 10, -91, -54, -18, -4, -1, -4, -2,
                  -1]
    @test gAlloc.shares == testshares

    efficient_risk!(cv, 0.18)
    testweights = [0.9999729203499247,
                   0.9999319425509923,
                   -0.9999737229065027,
                   0.9999628640920663,
                   0.9997549276200383,
                   -0.9999804996804014,
                   0.9999552198145188,
                   -0.9999471759430825,
                   0.9999881442037077,
                   -0.9988195839710775,
                   -0.9999750046344874,
                   -0.9999407986629681,
                   -0.8070940330922357,
                   0.9998502766728347,
                   0.7225903515496652,
                   -0.1865227675280002,
                   -0.32553204533840924,
                   -0.4042099896714802,
                   0.9999817760685096,
                   -0.9999928014936116]
    @test isapprox(cv.weights, testweights, rtol = 5e-3)
    mu, sigma = portfolio_performance(cv)
    mutest, sigmatest = 0.0019440206128798242, 0.17999920000461947
    @test isapprox(mu, mutest, rtol = 5e-6)
    @test sigma ≈ sigmatest

    lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]);
                                    investment = 10000)
    testshares = [1,
                  7,
                  6,
                  1,
                  116,
                  38,
                  14,
                  55,
                  10,
                  -60,
                  -771,
                  -116,
                  -256,
                  -284,
                  -597,
                  -2445,
                  -27,
                  -19,
                  -113,
                  -168]
    @test rmsd(lpAlloc.shares, testshares) < 3

    gAlloc, remaining = Allocation(Greedy(), cv, Vector(df[end, cv.tickers]);
                                   investment = 10000)
    testshares = [38,
                  10,
                  1,
                  6,
                  116,
                  7,
                  14,
                  1,
                  55,
                  -168,
                  -771,
                  -283,
                  -60,
                  -117,
                  -598,
                  -256,
                  -2446,
                  -113,
                  -19,
                  -26]
    @test rmsd(Int.(gAlloc.shares), testshares) < 3.2

    cv = EffCVaR(tickers, bl.post_ret, Matrix(returns); beta = 0.2, market_neutral = false)
    min_risk!(cv)
    testweights = [1.2591e-12,
                   0.0239999111487575,
                   0.0650098845478511,
                   0.0531407521736819,
                   0.0990996575024338,
                   3.624e-13,
                   0.0106099759402764,
                   0.1031880979043503,
                   8.525e-13,
                   0.0238096685816571,
                   0.2741982381970403,
                   8.64e-14,
                   6.49e-14,
                   0.0021107467292905,
                   1.032e-13,
                   0.0302164465597255,
                   0.053838138362372,
                   0.1363179609151427,
                   0.0210516374364299,
                   0.1034088839982625]
    @test isapprox(cv.weights, testweights, rtol = 1e-2)
    mu, cvar = portfolio_performance(cv)
    mutest, cvartest = 6.0049160399632115e-5, 0.002059389329978222
    @test mu ≈ mutest
    @test cvar ≈ cvartest

    efficient_return!(cv, 0.09 / 252)
    testweights = [5.398e-13,
                   1.53e-14,
                   2.9e-15,
                   0.0208931899503421,
                   2.98e-14,
                   1.4e-15,
                   0.0561037656427129,
                   3.6e-15,
                   0.9230030444062806,
                   4.5e-15,
                   2.6e-15,
                   1.9e-15,
                   7e-16,
                   7e-15,
                   3.9e-15,
                   4.1e-15,
                   8e-15,
                   6.5e-15,
                   3.2e-14,
                   2e-16]
    @test isapprox(cv.weights, testweights, rtol = 5e-3)
    mu, cvar = portfolio_performance(cv)
    mutest, cvartest = 0.09 / 252, 0.004545680961201856
    @test isapprox(mu, mutest, rtol = 5e-5)
    @test isapprox(cvar, cvartest, rtol = 5e-3)

    efficient_risk!(cv, 0.1438)
    testweights = [2.1e-14,
                   1.09e-14,
                   -2.19e-14,
                   5.82e-14,
                   -1.67e-14,
                   -1.52e-14,
                   4.488e-12,
                   -3.38e-14,
                   0.9999999999954172,
                   1.4e-15,
                   -3.02e-14,
                   -4.3e-15,
                   -1.04e-14,
                   6.9e-15,
                   9.49e-14,
                   -1.49e-14,
                   -5.5e-15,
                   -1.65e-14,
                   8.64e-14,
                   -1.56e-14]
    @test isapprox(cv.weights, testweights, rtol = 1e-3)
    mu, cvar = portfolio_performance(cv)
    mutest, cvartest = 0.00036251815516132356, 0.14377337204585958
    @test isapprox(mu, mutest, rtol = 1e-6)
    @test isapprox(cvar, cvartest, rtol = 1e-3)

    cv = EffCVaR(tickers, bl.post_ret, Matrix(returns); beta = 1, market_neutral = false)
    @test (0, 0) == portfolio_performance(cv)
    @test cv.beta == 0.95
    cv = EffCVaR(tickers, bl.post_ret, Matrix(returns); beta = -0.1, market_neutral = false)
    @test cv.beta == 0.95

    mean_ret = ret_model(MRet(), Matrix(returns))
    S = cov(Cov(), Matrix(returns))
    n = length(tickers)
    prev_weights = fill(1 / n, n)

    k = 0.0001
    ef = EffCVaR(tickers,
                 mean_ret,
                 Matrix(returns);
                 extra_vars = [:(0 <= l1)],
                 extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                        MOI.NormOneCone($(n + 1)))
                                      # :(model[:w][6] == 0.2),
                                      # :(model[:w][1] >= 0.01),
                                      # :(model[:w][16] <= 0.03),
                                      ],
                 extra_obj_terms = [quote
                                        $k * model[:l1]
                                    end],)
    min_risk!(ef)
    testweights = [1.0892e-12,
                   0.0500000000076794,
                   1.0671e-12,
                   3.4685e-12,
                   0.0071462860549324,
                   1.4497e-12,
                   1.053e-13,
                   0.09689425274277,
                   5.118e-13,
                   6.801e-13,
                   0.306535765681518,
                   6.189e-13,
                   3.304e-13,
                   0.0681247368437245,
                   1.7277e-12,
                   0.0285905010302666,
                   1.3816e-12,
                   0.3642755366704354,
                   6.405e-13,
                   0.078432920955603]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)

    efficient_return!(ef, 1.11^(1 / 252) - 1)
    testweights = [8.800201332344973e-7,
                   0.030673364610425576,
                   8.36182875646316e-7,
                   3.6411050627223824e-6,
                   0.07740095379315007,
                   6.830131276098486e-7,
                   1.7996554931382363e-7,
                   0.10565147924517973,
                   5.26941503365685e-7,
                   4.1671947615525764e-7,
                   0.2612563236366075,
                   3.802544849952017e-7,
                   1.6199989489683066e-7,
                   0.04999245471772181,
                   3.832804471651682e-7,
                   0.03174152450870135,
                   2.3669464039639354e-6,
                   0.35507401802720256,
                   8.211028208796623e-7,
                   0.08819860392923153]
    @test ef.weights ≈ testweights

    efficient_risk!(ef, 0.03)
    testweights = [1.0316445622564217e-5,
                   3.508481983225964e-5,
                   1.7293056061197458e-5,
                   7.992067161053934e-6,
                   0.6106069506532417,
                   2.2340883251691354e-6,
                   9.608274729155412e-6,
                   0.0003945682844916167,
                   1.2531231668581377e-5,
                   5.870320925594187e-6,
                   0.001267762836642832,
                   1.6631314551487166e-6,
                   8.876088464764226e-7,
                   5.300493167887579e-6,
                   1.1334922586819853e-6,
                   0.11308474477935936,
                   0.21757392433718226,
                   0.006812853688116319,
                   0.049997871129612646,
                   0.00015140926129948918]
    @test ef.weights ≈ testweights

    cv = EffCVaR(tickers, mean_ret, Matrix(returns))

    max_utility!(cv, 1e8)
    mu, cvar = portfolio_performance(cv)

    min_risk!(cv)
    muinf, cvarinf = portfolio_performance(cv)
    @test isapprox(mu, muinf, rtol = 1e-3)
    @test isapprox(cvar, cvarinf, rtol = 1e-3)

    max_utility!(cv, 0.25)
    mu1, cvar1 = portfolio_performance(cv)
    max_utility!(cv, 0.5)
    mu2, cvar2 = portfolio_performance(cv)
    max_utility!(cv, 1)
    mu3, cvar3 = portfolio_performance(cv)
    max_utility!(cv, 2)
    mu4, cvar4 = portfolio_performance(cv)
    max_utility!(cv, 4)
    mu5, cvar5 = portfolio_performance(cv)
    max_utility!(cv, 8)
    mu6, cvar6 = portfolio_performance(cv)
    max_utility!(cv, 16)
    mu7, cvar7 = portfolio_performance(cv)

    @test cvar1 > cvar2 > cvar3 > cvar4 > cvar5 > cvar6 > cvar7

    mean_ret = ret_model(MRet(), Matrix(returns))
    S = cov(Cov(), Matrix(returns))
    n = length(tickers)
    prev_weights = fill(1 / n, n)
    k = 0.001
    cv = EffCVaR(tickers,
                 mean_ret,
                 Matrix(returns);
                 extra_vars = [:(0 <= l1)],
                 extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                        MOI.NormOneCone($(n + 1)))
                                      # :(model[:w][6] == 0.2),
                                      # :(model[:w][1] >= 0.01),
                                      # :(model[:w][16] <= 0.03),
                                      ],
                 extra_obj_terms = [quote
                                        $k * model[:l1]
                                    end],)
    max_utility!(cv)
    testweights = [1.3067478510726676e-5,
                   0.04999923472956355,
                   3.036154765862227e-6,
                   0.03579663469941745,
                   0.05000081536946797,
                   2.912191110509021e-6,
                   6.172655136884889e-7,
                   0.09765007053693836,
                   1.7164643259092006e-6,
                   4.678174287355049e-6,
                   0.23717697196145363,
                   1.1806603885355381e-6,
                   6.231322456560854e-7,
                   0.0500008746083107,
                   1.0381944107747894e-6,
                   0.045989498652916815,
                   0.04999542446759715,
                   0.3333523715350914,
                   3.0857363685624538e-6,
                   0.050006147987315334]
    @test isapprox(cv.weights, testweights)

    mean_ret = ret_model(MRet(), Matrix(returns))

    cvar = EffCVaR(tickers, mean_ret, Matrix(returns))
    max_sharpe!(cvar)
    mu, risk = portfolio_performance(cvar; verbose = true)

    function nl_cvar_sharpe(w...)
        n = obj_params[1]
        mean_ret = obj_params[2]
        beta = obj_params[3]
        rf = obj_params[4]

        weights = w[1:n]
        alpha = w[n + 1]
        u = w[(n + 2):end]

        samples = length(u)

        ret = PortfolioOptimiser.port_return(weights, mean_ret) - rf
        CVaR = PortfolioOptimiser.cvar(alpha, u, samples, beta)

        return -ret / CVaR
    end

    # nl_cvar = EffCVaR(tickers, mean_ret, Matrix(returns))
    # model = nl_cvar.model
    # alpha = model[:alpha]
    # u = model[:u]
    # w = model[:w]
    # mean_ret = nl_cvar.mean_ret
    # beta = nl_cvar.beta
    # rf = nl_cvar.rf

    # extra_vars = [(alpha, nothing), (u, 1 / length(u))]
    # obj_params = [length(w), mean_ret, beta, rf]

    # custom_nloptimiser!(
    #     nl_cvar,
    #     nl_cvar_sharpe,
    #     obj_params,
    #     extra_vars,
    #     # optimiser_attributes = ("max_iter" => 200),
    # )
    # @test isapprox(cvar.weights, nl_cvar.weights, rtol = 1e-3)
    # nl_mu, nl_risk = portfolio_performance(nl_cvar, verbose = true)
end

@testset "Efficient CDaR" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])
    tickers = names(df)[2:end]

    mu = vec(ret_model(MRet(), Matrix(returns)))
    S = cov(Cov(), Matrix(returns))

    spy_prices = CSV.read("./assets/spy_prices.csv", DataFrame)
    delta = market_implied_risk_aversion(spy_prices[!, 2])
    # In the order of the dataframes, the
    mcapsdf = DataFrame(; ticker = tickers,
                        mcap = [927e9,
                                1.19e12,
                                574e9,
                                533e9,
                                867e9,
                                96e9,
                                43e9,
                                339e9,
                                301e9,
                                51e9,
                                61e9,
                                78e9,
                                0,
                                295e9,
                                1e9,
                                22e9,
                                288e9,
                                212e9,
                                422e9,
                                102e9])

    prior = market_implied_prior_returns(mcapsdf[!, 2], S, delta)

    # 1. SBUX drop by 20%
    # 2. GOOG outperforms FB by 10%
    # 3. BAC and JPM will outperform T and GE by 15%
    views = [-0.20, 0.10, 0.15] / 252
    picking = hcat([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0])

    bl = BlackLitterman(mcapsdf[!, 1],
                        S;
                        rf = 0,
                        tau = 0.01,
                        pi = prior, # either a vector, `nothing`, `:Equal`, or `:market`
                        Q = views,
                        P = picking,)

    cdar = EffCDaR(tickers, bl.post_ret, Matrix(returns))
    @test (0, 0) == portfolio_performance(cdar)
    min_risk!(cdar)
    testweights = [5.500000000000000e-15,
                   4.210000000000000e-13,
                   1.010000000000000e-14,
                   4.000000000000000e-16,
                   3.409901158173600e-03,
                   -3.300000000000000e-15,
                   -1.070000000000000e-14,
                   7.904282392519640e-02,
                   1.250000000000000e-14,
                   -2.900000000000000e-15,
                   3.875931701996790e-01,
                   4.740000000000000e-14,
                   6.370000000000000e-14,
                   1.063000000000000e-13,
                   5.545152701592000e-04,
                   9.598828930536250e-02,
                   2.679088825496902e-01,
                   1.995000000000000e-13,
                   6.560171960161000e-04,
                   1.648464003948736e-01]
    @test isapprox(cdar.weights, testweights, rtol = 1e-3)
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = 1.807731800440685e-5, 0.05643561307593433
    @test mu ≈ mutest
    @test sigma ≈ sigmatest

    cdar.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), cdar, Vector(df[end, cdar.tickers]);
                                    investment = 10000)
    testshares = [9, 110, 13, 16, 28]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]);
                                   investment = 10000)
    testshares = [109, 16, 27, 14, 9, 1, 1]
    @test gAlloc.shares == testshares

    efficient_return!(cdar, 0.071 / 252)
    testweights = [0.1799721301554317,
                   -1.4019535336406e-9,
                   -5.850040068171784e-9,
                   -1.7839963580325484e-9,
                   -3.8878246339297776e-9,
                   -6.814874851859814e-9,
                   0.037504274107288404,
                   -2.2935597642958482e-9,
                   0.14662929983180042,
                   -5.396452299776588e-9,
                   -4.477707672775698e-9,
                   -5.568602697606249e-9,
                   -7.430667288502608e-9,
                   1.466325633507462e-8,
                   0.03034143594018504,
                   -3.220046410031206e-9,
                   -3.856034496435162e-9,
                   -3.0513464303166673e-9,
                   0.6055529087308935,
                   -8.39574885736102e-9]
    @test cdar.weights ≈ testweights
    mu, sigma = portfolio_performance(cdar; verbose = true)
    mutest, sigmatest = 0.00028173603243845854, 0.14949408627549254
    @test mu ≈ mutest
    @test sigma ≈ sigmatest

    lpAlloc, remaining = Allocation(LP(), cdar, Vector(df[end, cdar.tickers]);
                                    investment = 10000)
    testshares = [2, 24, 49, 19, 54]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]);
                                   investment = 10000)
    testshares = [54, 1, 49, 38, 20, 1]
    @test gAlloc.shares == testshares

    efficient_risk!(cdar, 0.11)
    testweights = [0.3376745828284745,
                   3.6144088525005345e-5,
                   1.9278176065958248e-5,
                   1.3886556521888776e-5,
                   0.00016345618709142073,
                   1.269744226599325e-5,
                   6.318599613888324e-5,
                   0.0378297101041481,
                   4.476968445737409e-5,
                   1.4926930106820333e-5,
                   0.008965228837675048,
                   0.0003721840473962978,
                   2.4417106125616396e-5,
                   0.06327521354384988,
                   0.014598708214163435,
                   0.04017468324628745,
                   7.638967103935877e-5,
                   5.0357368073576195e-5,
                   0.49658176817550176,
                   8.411796091652408e-6]
    @test cdar.weights ≈ testweights
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = 0.00023557493204037468, 0.10999887481985175
    @test mu ≈ mutest
    @test sigma ≈ sigmatest

    lpAlloc, remaining = Allocation(LP(), cdar, Vector(df[end, cdar.tickers]);
                                    investment = 10000)
    testshares = [3, 1, 1, 5, 1, 1, 3, 1, 1, 9, 10, 6, 1, 45]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]);
                                   investment = 10000)
    testshares = [45, 3, 8, 6, 5, 10, 3, 1, 1, 1, 1]
    @test gAlloc.shares == testshares

    cdar = EffCDaR(tickers, bl.post_ret, Matrix(returns); market_neutral = true)
    min_risk!(cdar)
    testweights = [-7.8660e-13,
                   -5.1690e-13,
                   5.5350e-13,
                   2.5550e-13,
                   1.0952e-12,
                   -1.6305e-12,
                   2.7030e-13,
                   -2.7810e-13,
                   -1.7863e-12,
                   -1.7250e-13,
                   3.4360e-13,
                   -4.9670e-13,
                   -1.1400e-13,
                   -6.5110e-13,
                   -2.9790e-13,
                   5.5730e-13,
                   4.9150e-13,
                   -6.2820e-13,
                   2.9461e-12,
                   8.4570e-13]
    @test isapprox(cdar.weights, testweights, atol = 1e-3)
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = -5.45658046908877e-9, 8.539475955542743e-6
    @test mu ≈ mutest
    @test sigma ≈ sigmatest

    cdar.weights .= testweights
    lpAlloc, remaining = Allocation(LP(), cdar, Vector(df[end, cdar.tickers]);
                                    investment = 10000)
    testshares = [5, 2, 1, 35, 13, 11, 4, 36, 19]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]);
                                   investment = 10000)
    testshares = [36, 1, 19, 10, 5, 4, 14, 38, 2]
    @test gAlloc.shares == testshares

    efficient_return!(cdar, 0.071 / 252)
    testweights = [0.025106330418334377,
                   0.057610563293440345,
                   -0.029090641886840286,
                   -0.05538569699320718,
                   0.27701184780279303,
                   -0.21803788569186527,
                   0.01760322037191931,
                   0.0333895501246084,
                   -0.361160497662433,
                   -0.08849295135335004,
                   -0.18456677846312178,
                   -0.023883989980442714,
                   -0.006314009747286087,
                   0.2775835100256476,
                   -0.06968430837501745,
                   0.08448010553301848,
                   0.11755531043570804,
                   -0.008076589681676323,
                   0.6901451943047429,
                   -0.5357922824749722]
    @test cdar.weights ≈ testweights
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = 0.0002817360420399082, 0.0685929138123518
    @test mu ≈ mutest
    @test sigma ≈ sigmatest

    lpAlloc, remaining = Allocation(LP(), cdar, Vector(df[end, cdar.tickers]);
                                    investment = 10000)
    testshares = [3, 1, 16, 3, 23, 8, 5, 40, -2, -3, -168, -121, -23, -52, -14, -18, -47,
                  -2, -90]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]);
                                   investment = 10000)
    testshares = [39, 23, 1, 5, 8, 3, 3, 12, -90, -121, -168, -52, -23, -46, -3, -2, -14,
                  -2, -20]
    @test gAlloc.shares == testshares

    efficient_risk!(cdar, 0.11)
    testweights = [0.033796952655170374,
                   0.13337831909364453,
                   -0.03096567374865451,
                   -0.10120870803834797,
                   0.43866749992421994,
                   -0.3527415350463746,
                   0.023891988502132925,
                   0.06884620316354743,
                   -0.5135004814787475,
                   -0.12375252474101363,
                   -0.24985550974393078,
                   -0.021442572281340888,
                   -0.01896559923948774,
                   0.42046615090062534,
                   -0.1204292112344196,
                   0.129469039894361,
                   0.1915525244958119,
                   -0.03295505801780674,
                   0.9998850102594589,
                   -0.8741368153188485]
    @test cdar.weights ≈ testweights
    mu, sigma = portfolio_performance(cdar)
    mutest, sigmatest = 0.0004468872981757747, 0.109999374983889
    @test mu ≈ mutest
    @test sigma ≈ sigmatest

    lpAlloc, remaining = Allocation(LP(), cdar, Vector(df[end, cdar.tickers]);
                                    investment = 10000)
    testshares = [4, 1, 10, 4, 23, 8, 5, 38, -2, -6, -272, -171, -31, -71, -13, -57, -80,
                  -9, -147]
    @test lpAlloc.shares == testshares

    gAlloc, remaining = Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]);
                                   investment = 10000)
    testshares = [37, 1, 22, 5, 4, 8, 4, 10, -147, -171, -271, -70, -32, -80, -6, -9, -2,
                  -13, -58]
    @test gAlloc.shares == testshares

    cdar = EffCDaR(tickers, bl.post_ret, Matrix(returns); beta = 0.2)
    min_risk!(cdar)
    testweights = [1.229e-13,
                   0.0444141271754544,
                   0.1587818250703651,
                   1.98e-14,
                   0.0678958478066876,
                   6.64e-14,
                   2.56e-14,
                   0.061989718949941,
                   3.84e-14,
                   6.16e-14,
                   0.2102532269364574,
                   2.378e-13,
                   0.0012132758140646,
                   0.01721619579299,
                   2.7e-15,
                   0.0690692111656633,
                   0.1014609071930568,
                   0.0224777909586333,
                   0.1303610329941893,
                   0.1148668401419223]
    @test isapprox(cdar.weights, testweights, rtol = 1e-2)
    mu, cvar = portfolio_performance(cdar)
    mutest, cvartest = 6.54502826667942e-5, 0.017697217976780904
    @test isapprox(mu, mutest, rtol = 1e-3)
    @test isapprox(cvar, cvartest, rtol = 1e-3)

    efficient_return!(cdar, 0.08 / 252)
    testweights = [0.09841109785601676,
                   0.04861547667149016,
                   2.2503284910414088e-8,
                   4.5135695959511086e-7,
                   0.06130973384499447,
                   1.1042814497596773e-8,
                   0.08936604343657141,
                   3.2730102871065444e-8,
                   0.6635393991525804,
                   2.7282992332008832e-8,
                   1.9906200022204973e-8,
                   2.4621106035598856e-8,
                   3.206141230725002e-8,
                   5.979684761702935e-8,
                   0.03875402861997502,
                   1.1190090189193411e-7,
                   8.030154619503404e-8,
                   5.320488266424174e-8,
                   3.2910684320679113e-6,
                   2.640889000144384e-9]
    @test cdar.weights ≈ testweights
    mu, cvar = portfolio_performance(cdar)
    mutest, cvartest = 0.00031745032252068734, 0.05680617951538816
    @test mu ≈ mutest
    @test cvar ≈ cvartest

    efficient_risk!(cdar, 0.06)
    testweights = [0.07862395940435725,
                   0.05148503130314269,
                   1.625378197945849e-5,
                   0.0003845443446575024,
                   0.06164780959656015,
                   1.0342704395242032e-5,
                   0.09572829669616902,
                   2.1293054400689415e-5,
                   0.6753743779820783,
                   1.834564671543546e-5,
                   1.4658173601525717e-5,
                   1.702908715143999e-5,
                   2.0562023047838758e-5,
                   3.4952872612792646e-5,
                   0.03624266312816516,
                   6.722523645909205e-5,
                   4.500865992658763e-5,
                   3.1342186040534566e-5,
                   0.00021004488171723215,
                   6.259236821976496e-6]
    @test cdar.weights ≈ testweights
    mu, cvar = portfolio_performance(cdar)
    mutest, cvartest = 0.0003193768010386029, 0.05999879944247901
    @test mu ≈ mutest
    @test cvar ≈ cvartest

    cdar = EffCDaR(tickers, bl.post_ret, Matrix(returns); beta = 1)
    @test (0, 0) == portfolio_performance(cdar)
    @test cdar.beta == 0.95
    cdar = EffCDaR(tickers, bl.post_ret, Matrix(returns); beta = -0.1,
                   market_neutral = false)
    @test cdar.beta == 0.95

    mean_ret = ret_model(MRet(), Matrix(returns))
    S = cov(Cov(), Matrix(returns))
    n = length(tickers)
    prev_weights = fill(1 / n, n)

    k = 0.0001
    ef = EffCDaR(tickers,
                 mean_ret,
                 Matrix(returns);
                 extra_vars = [:(0 <= l1)],
                 extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                        MOI.NormOneCone($(n + 1)))
                                      # :(model[:w][6] == 0.2),
                                      # :(model[:w][1] >= 0.01),
                                      # :(model[:w][16] <= 0.03),
                                      ],
                 extra_obj_terms = [quote
                                        $k * model[:l1]
                                    end],)
    min_risk!(ef)
    testweights = [5.8e-15,
                   5.127e-13,
                   1.02e-14,
                   5e-16,
                   0.0035594792687726,
                   -3e-15,
                   -1.04e-14,
                   0.079973068549514,
                   1.26e-14,
                   -2.8e-15,
                   0.3871598884148665,
                   5.02e-14,
                   6.05e-14,
                   1.133e-13,
                   0.0002218291232121,
                   0.0965257689615006,
                   0.2659397480643742,
                   2.058e-13,
                   0.0019205861934667,
                   0.164699631423338]
    @test isapprox(ef.weights, testweights, rtol = 1e-3)

    efficient_return!(ef, 0.17 / 252)
    testweights = [6.054270289787292e-8,
                   1.5416757975021074e-7,
                   0.010016041902866472,
                   3.0972585641439495e-8,
                   0.14289651879038948,
                   1.2568547268139337e-8,
                   8.822539519397757e-9,
                   0.11589787570574933,
                   4.163911590729622e-8,
                   3.0468147764414474e-8,
                   0.29498390916737266,
                   1.2744174590680609e-8,
                   2.9628763960843103e-9,
                   6.609621536392699e-8,
                   8.824437534130712e-9,
                   0.140653954541203,
                   0.13537316955172882,
                   9.714720567795575e-8,
                   2.389135864289537e-7,
                   0.16017776447097531]
    @test ef.weights ≈ testweights

    efficient_risk!(ef, 0.09)
    testweights = [1.0472483657291897e-5,
                   2.441071448365272e-5,
                   0.11739868113077082,
                   3.512304683860334e-6,
                   0.349616059518583,
                   1.6664841100299865e-6,
                   1.7158749883783415e-6,
                   0.19807028552489095,
                   2.8761681271073235e-6,
                   3.4170708766616095e-6,
                   0.05005736670644256,
                   1.769227013587645e-6,
                   7.644655388064644e-7,
                   7.098018280629737e-6,
                   1.4028069223216178e-6,
                   0.18474330701378933,
                   0.049975786894972014,
                   1.0922322071827534e-5,
                   1.2738736807743145e-5,
                   0.050055746532989484]
    @test ef.weights ≈ testweights

    cd = EffCDaR(tickers, mean_ret, Matrix(returns))

    max_utility!(cd, 1e8)
    mu, cdar = portfolio_performance(cd)
    min_risk!(cd)
    muinf, cdarinf = portfolio_performance(cd)
    @test isapprox(mu, muinf, rtol = 1e-3)
    @test isapprox(cdar, cdarinf, rtol = 1e-3)

    max_utility!(cd, 0.25)
    mu1, cdar1 = portfolio_performance(cd)
    max_utility!(cd, 0.5)
    mu2, cdar2 = portfolio_performance(cd)
    max_utility!(cd, 1)
    mu3, cdar3 = portfolio_performance(cd)
    max_utility!(cd, 2)
    mu4, cdar4 = portfolio_performance(cd)
    max_utility!(cd, 4)
    mu5, cdar5 = portfolio_performance(cd)
    max_utility!(cd, 8)
    mu6, cdar6 = portfolio_performance(cd)
    max_utility!(cd, 16)
    mu7, cdar7 = portfolio_performance(cd)

    @test cdar1 > cdar2 > cdar3 > cdar4 > cdar5 > cdar6 > cdar7

    k = 0.001
    cd = EffCDaR(tickers,
                 mean_ret,
                 Matrix(returns);
                 extra_vars = [:(0 <= l1)],
                 extra_constraints = [:([model[:l1]; (model[:w] - $prev_weights)] in
                                        MOI.NormOneCone($(n + 1)))
                                      # :(model[:w][6] == 0.2),
                                      # :(model[:w][1] >= 0.01),
                                      # :(model[:w][16] <= 0.03),
                                      ],
                 extra_obj_terms = [quote
                                        $k * model[:l1]
                                    end],)
    max_utility!(cd)
    testweights = [2.652664907331096e-7,
                   0.009698024789270218,
                   3.497805253655786e-7,
                   1.5328243075030825e-7,
                   0.006713602877664739,
                   1.2727509135493578e-7,
                   2.8271998703581564e-8,
                   0.07678104477298273,
                   3.2774155374262747e-7,
                   1.280825742940062e-7,
                   0.39012427244459796,
                   4.781052243610445e-7,
                   2.72291494046745e-7,
                   6.454848400019041e-6,
                   7.885989554008511e-7,
                   0.09339034386630214,
                   0.25266685169581937,
                   0.009036454738919517,
                   0.0006823190426043853,
                   0.16089771222710028]
    @test isapprox(cd.weights, testweights)

    cdar = EffCDaR(tickers, mean_ret, Matrix(returns))
    max_sharpe!(cdar)
    mu, risk = portfolio_performance(cdar; verbose = true)

    # function nl_cdar_sharpe(w...)
    #     n = obj_params[1]
    #     mean_ret = obj_params[2]
    #     beta = obj_params[3]
    #     rf = obj_params[4]

    #     weights = w[1:n]
    #     alpha = w[n + 1]
    #     z = w[(n + 2):end]

    #     samples = length(z)

    #     ret = PortfolioOptimiser.port_return(weights, mean_ret) - rf
    #     CDaR = PortfolioOptimiser.cdar(alpha, z, samples, beta)

    #     return -ret / CDaR
    # end

    # nl_cdar = EffCDaR(tickers, mean_ret, Matrix(returns))
    # model = nl_cdar.model
    # alpha = model[:alpha]
    # z = model[:z]
    # w = model[:w]
    # mean_ret = nl_cdar.mean_ret
    # beta = nl_cdar.beta
    # rf = nl_cdar.rf

    # extra_vars = [(alpha, nothing), (z, 1 / length(z))]
    # obj_params = [length(w), mean_ret, beta, rf]

    # custom_nloptimiser!(nl_cdar, nl_cdar_sharpe, obj_params, extra_vars)
    # @test isapprox(cdar.weights, nl_cdar.weights, rtol = 1e-4)
    # nl_mu, nl_risk = portfolio_performance(nl_cdar, verbose = true)
end
