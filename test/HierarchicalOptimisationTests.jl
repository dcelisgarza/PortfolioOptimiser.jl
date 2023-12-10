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

A = TimeArray(CSV.File("./assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "HRP" begin
    portfolio = HCPortfolio(returns = returns)
    asset_statistics!(portfolio, calc_kurt = false)
    portfolio.w_max = Float64[]
    portfolio.w_min = Float64[]
    w1 = opt_port!(portfolio, linkage = :DBHT)
    portfolio.w_max = 1
    portfolio.w_min = 0
    w2 = opt_port!(portfolio, linkage = :DBHT)
    @test isapprox(w1.weights, w2.weights)

    portfolio = HCPortfolio(
        ret = Matrix(returns[!, 2:end]),
        assets = names(returns[!, 2:end]),
        timestamps = returns[!, 1],
    )
    @test portfolio.returns == Matrix(returns[!, 2:end])
    @test portfolio.assets == names(returns[!, 2:end])
    @test portfolio.timestamps == returns[!, 1]

    portfolio = HCPortfolio(
        returns = returns,
        solvers = OrderedDict(
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
        ),
    )
    asset_statistics!(portfolio)

    type = :HRP
    rm = :Variance
    obj = :Min_Risk
    kelly = :None
    linkage = :DBHT
    branchorder = :default

    w1 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
    )

    w1t = [
        0.05508878886665077,
        0.07921980553056827,
        0.030128391561595795,
        0.028872972320853722,
        0.033334402615699414,
        0.06358468477746226,
        0.010166844125574337,
        0.06322418256310877,
        0.03881462357698402,
        0.06553219418190419,
        0.10808979041234959,
        0.015170184483397027,
        0.008202518714882668,
        0.07891942014031887,
        0.01571166898114577,
        0.027326705729654836,
        0.08348923079358755,
        0.08473133008827005,
        0.06130631600043508,
        0.04908594453555704,
    ]

    @test isapprox(w1t, w1.weights)

    dbht_method = :Equal
    w2 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
        dbht_method = dbht_method,
    )
    w2t = [
        0.04806115269446613,
        0.07984550882154572,
        0.04427600432815406,
        0.030782684773087033,
        0.030478840318032118,
        0.048159603578557586,
        0.010247145103159693,
        0.09048341196725802,
        0.04114841956937171,
        0.05401479654353932,
        0.08186816487911966,
        0.018826570882692822,
        0.005374668097010057,
        0.0791253386565669,
        0.011181036283665477,
        0.0260021206015489,
        0.07279798859546274,
        0.10106400417562743,
        0.060089516797333674,
        0.06617302333380098,
    ]
    @test isapprox(w2.weights, w2t)
end

@testset "HERC" begin
    portfolio = HCPortfolio(
        returns = returns,
        solvers = OrderedDict(
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
        ),
    )
    asset_statistics!(portfolio)

    type = :HERC
    rm = :Variance
    obj = :Min_Risk
    kelly = :None
    linkage = :DBHT
    branchorder = :default
    w1 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
    )

    w1t = [
        0.049141829952957225,
        0.08890554096278969,
        0.04011249526834231,
        0.025756069886785627,
        0.03116417114657615,
        0.04954824538953943,
        0.011409883813963236,
        0.06309701295416763,
        0.03212704206115593,
        0.07940635883011783,
        0.08422876480710824,
        0.012556431315299158,
        0.009939116986938707,
        0.06177790325183918,
        0.008929720036096727,
        0.03822533548129327,
        0.11577046171718577,
        0.084560900836747,
        0.047990414170144664,
        0.06535230113095213,
    ]

    @test isapprox(w1t, w1.weights)

    dbht_method = :Equal
    w2 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
        dbht_method = dbht_method,
    )
    w2t = [
        0.060515970737320735,
        0.1554699725614562,
        0.04939674798809242,
        0.031717450755683774,
        0.038377286132088115,
        0.020034960459139613,
        0.019952573307311736,
        0.07770115588292159,
        0.012990652086749049,
        0.132456395584715,
        0.03405811768238419,
        0.010720860388684838,
        0.016579271871680158,
        0.024980054069895605,
        0.0036107552634377493,
        0.06376303148090225,
        0.022088424062064575,
        0.10413297602992737,
        0.040974901020356155,
        0.08047844263518882,
    ]
    @test isapprox(w2.weights, w2t)
end

@testset "HERC2" begin
    portfolio = HCPortfolio(
        returns = returns,
        solvers = OrderedDict(
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
        ),
    )
    asset_statistics!(portfolio)

    type = :HERC
    rm = :Variance
    obj = :Min_Risk
    kelly = :None
    linkage = :DBHT
    branchorder = :default
    w1 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        rm_i = :Equal,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
    )

    w1t = [
        0.05131211159664687,
        0.0720286288313129,
        0.05131211159664687,
        0.05131211159664687,
        0.05131211159664687,
        0.04245121729016905,
        0.0720286288313129,
        0.05131211159664687,
        0.04245121729016905,
        0.042523603766116594,
        0.04245121729016905,
        0.04245121729016905,
        0.042523603766116594,
        0.04245121729016905,
        0.04245121729016905,
        0.042523603766116594,
        0.0720286288313129,
        0.05131211159664687,
        0.04245121729016905,
        0.05131211159664687,
    ]

    @test isapprox(w1t, w1.weights)

    dbht_method = :Equal
    w2 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        rm_i = :Equal,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
        dbht_method = dbht_method,
    )
    w2t = [
        0.06318857573731755,
        0.08771127293438398,
        0.06318857573731755,
        0.06318857573731755,
        0.06318857573731755,
        0.01962716060394513,
        0.08771127293438398,
        0.06318857573731755,
        0.01962716060394513,
        0.0709328996457658,
        0.01962716060394513,
        0.025847880704520494,
        0.0709328996457658,
        0.01962716060394513,
        0.01962716060394513,
        0.0709328996457658,
        0.01962716060394513,
        0.06318857573731755,
        0.025847880704520494,
        0.06318857573731755,
    ]
    @test isapprox(w2.weights, w2t)

    type = :HERC
    rm = :Equal
    kelly = :None
    linkage = :DBHT
    branchorder = :default
    w1 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        kelly = kelly,
        linkage = linkage,
        branchorder = branchorder,
    )
    w1t = [
        0.03571428571428571,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.03571428571428571,
        0.03571428571428571,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.08333333333333333,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.03571428571428571,
    ]
    @test isapprox(w1.weights, w1t)

    type = :HERC
    rm = :Equal
    kelly = :None
    linkage = :DBHT
    branchorder = :default
    w2 = opt_port!(
        portfolio;
        type = type,
        rm_i = :Equal,
        rm = rm,
        kelly = kelly,
        linkage = linkage,
        branchorder = branchorder,
    )
    w2t = [
        0.03571428571428571,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.03571428571428571,
        0.03571428571428571,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.08333333333333333,
        0.08333333333333333,
        0.03571428571428571,
        0.03571428571428571,
        0.03571428571428571,
    ]
    @test isapprox(w2.weights, w2t)
end

@testset "NCO" begin
    portfolio = HCPortfolio(
        returns = returns,
        solvers = OrderedDict(
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    portfolio.codep_type = :Pearson
    asset_statistics!(portfolio)

    type = :NCO
    rm = :SD
    obj = :Min_Risk
    kelly = :None
    linkage = :DBHT
    branchorder = :default
    w1 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
    )
    w1t = [
        0.025117587653493957,
        0.0024048985731171847,
        0.022135371734711015,
        0.02429059700839297,
        0.011430669574765646,
        0.06565629672888273,
        2.6271145081525728e-5,
        0.13511327499330353,
        2.1745072038914858e-8,
        3.7616017207683504e-7,
        0.24203124072378313,
        0.0061739548023942836,
        1.0693420061805877e-8,
        0.1154519756683228,
        1.5660071684882708e-9,
        1.3060602681031305e-7,
        0.004094640899102485,
        0.1963562319205421,
        0.041569633327894866,
        0.10814681447551365,
    ]
    @test isapprox(w1t, w1.weights, rtol = 3e-4)

    type = :NCO
    rm = :SD
    obj = :Equal
    kelly = :None
    linkage = :DBHT
    branchorder = :default
    w2 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
    )
    w2t = [
        0.03686989551346506,
        0.0885648202981644,
        0.03550338070885995,
        0.03365193555569722,
        0.032410817940959694,
        0.04373193503852158,
        0.035009184770548325,
        0.05892155114612448,
        0.03280418878206287,
        0.09589560971220468,
        0.06533686857891337,
        0.02703273246131023,
        0.03466128468840722,
        0.04661028146874129,
        0.022775178622336942,
        0.06580197558444718,
        0.09949545752431102,
        0.06027877962414397,
        0.037339275204482505,
        0.04730484677629799,
    ]
    @test isapprox(w2t, w2.weights, rtol = 2.0e-5)
end

@testset "Weight bounds" begin
    portfolio = HCPortfolio(
        returns = returns,
        solvers = OrderedDict(
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
        ),
    )
    asset_statistics!(portfolio)

    type = :HRP
    rm = :Variance
    obj = :Min_Risk
    kelly = :None
    linkage = :DBHT
    branchorder = :default

    asset_classes = DataFrame(Assets = portfolio.assets)
    constraints = DataFrame(
        Enabled = fill(true, 20),
        Type = fill("Assets", 20),
        Position = [
            "GOOG",
            "AAPL",
            "FB",
            "BABA",
            "AMZN",
            "GE",
            "AMD",
            "WMT",
            "BAC",
            "GM",
            "T",
            "UAA",
            "SHLD",
            "XOM",
            "RRC",
            "BBY",
            "MA",
            "PFE",
            "JPM",
            "SBUX",
        ],
        Sign = [
            "<=",
            "<=",
            ">=",
            ">=",
            ">=",
            "<=",
            ">=",
            "<=",
            "<=",
            "<=",
            "<=",
            ">=",
            ">=",
            "<=",
            ">=",
            ">=",
            ">=",
            "<=",
            "<=",
            ">=",
        ],
        Weight = [
            0.025,
            0.05,
            0.06,
            0.045,
            0.06,
            0.03,
            0.02,
            0.04,
            0.03,
            0.02,
            0.06,
            0.03,
            0.01,
            0.05,
            0.07,
            0.01,
            0.07,
            0.1,
            0.001,
            0.02,
        ],
    )
    w_min, w_max = hrp_constraints(constraints, asset_classes)

    portfolio.w_min = w_min
    portfolio.w_max = w_max
    w1 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
    )
    w1t = [
        0.025,
        0.05,
        0.20557894736842108,
        0.045,
        0.06,
        0.03,
        0.07342105263157891,
        0.04,
        0.03,
        0.02,
        0.06,
        0.03,
        0.01,
        0.05,
        0.07,
        0.01,
        0.07,
        0.1,
        0.001,
        0.02,
    ]
    w1t2 = [
        0.025,
        0.05,
        0.19266666666666657,
        0.045,
        0.06,
        0.03,
        0.06880952380952378,
        0.04,
        0.03,
        0.02,
        0.06,
        0.03,
        0.027523809523809516,
        0.05,
        0.07,
        0.01,
        0.07,
        0.1,
        0.001,
        0.02,
    ]

    if isapprox(w1.weights, w1t)
        @test isapprox(w1.weights, w1t)
    else
        @test isapprox(w1.weights, w1t2)
    end

    type = :HERC
    w2 = opt_port!(
        portfolio;
        type = type,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
    )
    w2t = [
        0.025,
        0.05,
        0.06,
        0.045,
        0.06,
        0.03,
        0.02,
        0.04,
        0.03,
        0.02,
        0.06,
        0.03,
        0.01,
        0.05,
        0.07,
        0.07285539025375766,
        0.07,
        0.1,
        0.001,
        0.15614460974624225,
    ]
    @test isapprox(w2.weights, w2t)

    type = :HERC
    w3 = opt_port!(
        portfolio;
        type = type,
        rm_i = :Equal,
        rm = rm,
        obj = obj,
        kelly = kelly,
        rf = rf,
        l = l,
        linkage = linkage,
        branchorder = branchorder,
    )
    w3t = [
        0.025,
        0.048189367669678886,
        0.07439403001797913,
        0.07439403001797913,
        0.07439403001797913,
        0.03,
        0.048189367669678886,
        0.04,
        0.03,
        0.02,
        0.049096735632513294,
        0.049096735632513294,
        0.04918045383660326,
        0.049096735632513294,
        0.07,
        0.04918045383660326,
        0.07,
        0.07439403001797913,
        0.001,
        0.07439403001797913,
    ]
    @test isapprox(w3.weights, w3t)

    portfolio = HCPortfolio(
        returns = returns,
        solvers = Dict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
        ),
    )
    asset_statistics!(portfolio)
    portfolio.owa_w = Float64[]
    w = opt_port!(portfolio; type = :NCO, owa_w_i = owa_gmd(size(returns, 1)))
    @test isempty(portfolio.owa_w)
    w = opt_port!(portfolio; type = :HERC, owa_w_i = owa_gmd(size(returns, 1)))
    @test isempty(portfolio.owa_w)
end
