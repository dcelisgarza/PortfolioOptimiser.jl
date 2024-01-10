using Test,
      PortfolioOptimiser,
      DataFrames,
      TimeSeries,
      CSV,
      Dates,
      ECOS,
      SCS,
      Clarabel,
      COSMO,
      OrderedCollections,
      LinearAlgebra,
      StatsBase,
      Logging

Logging.disable_logging(Logging.Warn)

A = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Risk parity" begin
    portfolio = Portfolio(; returns = returns,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))
                                                # :COSMO => Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
                                                # :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
                                                # :ECOS => Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
                                                ))
    asset_statistics!(portfolio)
    rm = :SD
    type = :RP
    kelly = :None
    portfolio.risk_budget = Float64[]

    w1 = opt_port!(portfolio; type = type, rm = rm, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio; type = type, rm = rm, rf = rf)
    m1 = dot(portfolio.mu, w1.weights)
    w1t = [0.050632255832484985,
           0.051247934672526446,
           0.04690525804803236,
           0.043689918290747254,
           0.04571517573464733,
           0.05615049134992348,
           0.02763275610012436,
           0.07705790240307762,
           0.039495768009814226,
           0.04723331078528691,
           0.08435263410648593,
           0.033855585237659115,
           0.02754587849147457,
           0.062067311889335994,
           0.03563640213891282,
           0.044130936457772975,
           0.050852684605541744,
           0.07142022083826376,
           0.045296393183736884,
           0.059081181824151126]
    @test isapprox(w1t, w1.weights, rtol = 7e-5)

    portfolio.risk_budget = collect(1:20.0)
    w2 = opt_port!(portfolio; type = type, rm = rm, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio; type = type, rm = rm, rf = rf)
    m2 = dot(portfolio.mu, w2.weights)
    w2t = [0.005639420548623995,
           0.011008510767523629,
           0.01558186826516206,
           0.01936958294612637,
           0.025437216420414747,
           0.03265698847059332,
           0.020371048887667463,
           0.05957161481564274,
           0.03298215667659502,
           0.04482943317736114,
           0.08741189235568801,
           0.038228907615732076,
           0.031437874906039845,
           0.0796101103451106,
           0.04614263726618927,
           0.060975144598721644,
           0.08329766572186004,
           0.11639315319856354,
           0.0792820130418486,
           0.109772759974536]
    @test isapprox(w2t, w2.weights, rtol = 8e-5)
end

@testset "Relaxed risk parity" begin
    portfolio = Portfolio(; returns = returns,
                          solvers = OrderedDict(:Clarabel => Dict(:solver => (Clarabel.Optimizer),
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))
                                                # :COSMO => Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
                                                # :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
                                                # :ECOS => Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
                                                ))
    asset_statistics!(portfolio)
    type = :RRP
    kelly = :None
    rrp_ver = :None
    rrp_penalty = 20

    portfolio.risk_budget = Float64[]
    w1 = opt_port!(portfolio; type = type, rrp_penalty = rrp_penalty, rrp_ver = rrp_ver)
    m1 = dot(portfolio.mu, w1.weights)
    w1t = [0.05082999293061849,
           0.0514568396566574,
           0.04704855196476608,
           0.04378169859516516,
           0.04583551710538893,
           0.05642846261305365,
           0.027668688388458793,
           0.07655128325664473,
           0.039575525892706416,
           0.0474134958359509,
           0.08261584825468093,
           0.03390673011825231,
           0.027590224293893392,
           0.06227796063499927,
           0.03570102482953629,
           0.0442584256631949,
           0.05106943796512795,
           0.07123177261938061,
           0.045460450815143105,
           0.05929806856638062]
    @test isapprox(w1t, w1.weights, rtol = 9e-3)

    portfolio.risk_budget = collect(1:20.0)
    w2 = opt_port!(portfolio; type = type, rrp_penalty = rrp_penalty, rrp_ver = rrp_ver)
    m2 = dot(portfolio.mu, w2.weights)
    w2t = [0.005642983858876163,
           0.011014001842975046,
           0.015593406797743485,
           0.019379777959651744,
           0.025463414443957937,
           0.03270427797491965,
           0.02037819266176675,
           0.05981525898555451,
           0.03299529856538065,
           0.04489764806150711,
           0.08701972169249905,
           0.03825924971516721,
           0.031453909634545106,
           0.07986750769873659,
           0.04616433496548614,
           0.061077860328168615,
           0.08366198738759104,
           0.11564215186730005,
           0.07965877637360663,
           0.10931023918456664]
    @test isapprox(w2t, w2.weights, rtol = 5e-3)

    rrp_ver = :Reg
    portfolio.risk_budget = Float64[]
    w3 = opt_port!(portfolio; type = type, rrp_penalty = rrp_penalty, rrp_ver = rrp_ver)
    m3 = dot(portfolio.mu, w3.weights)
    w3t = [0.05063705964078256,
           0.051252914047950814,
           0.04690965831089903,
           0.04369310441935011,
           0.04571899068159735,
           0.056154189338723424,
           0.02763402949508363,
           0.0770274709915493,
           0.03949868849299765,
           0.04723835365825583,
           0.0843407527136848,
           0.03385746808172211,
           0.027547547736053123,
           0.06206642980338261,
           0.035638303548570706,
           0.04413478233206673,
           0.050857890940679024,
           0.07140807705050851,
           0.04530123790673084,
           0.05908305080941188]
    @test isapprox(w3t, w3.weights, rtol = 2e-4)

    portfolio.risk_budget = collect(1:20.0)
    w4 = opt_port!(portfolio; type = type, rrp_penalty = rrp_penalty, rrp_ver = rrp_ver)
    m4 = dot(portfolio.mu, w4.weights)
    w4t = [0.005646140830247419,
           0.011018347551640148,
           0.015604022777639589,
           0.019391195736302255,
           0.025491532218706515,
           0.032722342696139085,
           0.02037862895580644,
           0.05987109995691716,
           0.03302560491602334,
           0.04494612905590547,
           0.08796454538144627,
           0.03829424590602345,
           0.03145813069875428,
           0.07969874564524285,
           0.046188609944801076,
           0.06116937892997849,
           0.08341465123158819,
           0.11551336801659909,
           0.07973071620398466,
           0.10847256334625408]
    @test isapprox(w4t, w4.weights, rtol = 7e-3)

    rrp_ver = :Reg_Pen
    portfolio.risk_budget = Float64[]
    w5 = opt_port!(portfolio; type = type, rrp_penalty = rrp_penalty, rrp_ver = rrp_ver)
    m5 = dot(portfolio.mu, w5.weights)
    w5t = [0.049269594505827144,
           0.04995841515639617,
           0.04566382307264653,
           0.04251930548435496,
           0.044416405546686376,
           0.05482060079752967,
           0.0270228592016435,
           0.07264560366189014,
           0.03813485926752143,
           0.04601038619706809,
           0.08855259946035533,
           0.03304800143521492,
           0.027078388030667526,
           0.0663525475128018,
           0.03452785668215129,
           0.042720932925557245,
           0.055633940194681375,
           0.07415486554285952,
           0.049299887110731765,
           0.058169128213415466]
    @test isapprox(w5t, w5.weights, rtol = 1e-2)

    portfolio.risk_budget = collect(1:20.0)
    w6 = opt_port!(portfolio; type = type, rrp_penalty = rrp_penalty, rrp_ver = rrp_ver)
    m6 = dot(portfolio.mu, w6.weights)
    w6t = [0.041058783415684905,
           0.043527757584266524,
           0.03444679955941352,
           0.02384846912632278,
           0.026779955569221556,
           0.054407168050064,
           0.016732000105393625,
           0.06020058646645128,
           0.03524121747423443,
           0.04080167804207209,
           0.09280252513444463,
           0.03136100299182605,
           0.027138226386426993,
           0.06980771073843914,
           0.03950108966042306,
           0.05119635844300681,
           0.0664688843766782,
           0.09371551015661385,
           0.06465373164363557,
           0.08631054507538109]
    @test isapprox(w6t, w6.weights, rtol = 8e-3)
end
