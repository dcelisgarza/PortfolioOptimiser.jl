using COSMO,
    CSV,
    Clarabel,
    HiGHS,
    OrderedCollections,
    PortfolioOptimiser,
    Statistics,
    Test,
    TimeSeries

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

classes = PortfolioOptimiser.PortClasses
types = PortfolioOptimiser.PortTypes
rrpvs = PortfolioOptimiser.RRPVersions
rms = PortfolioOptimiser.RiskMeasures
objs = PortfolioOptimiser.ObjFuncs
krets = PortfolioOptimiser.KellyRet
umus = PortfolioOptimiser.UncertaintyTypes
ucovs = PortfolioOptimiser.UncertaintyTypes

@testset "$(classes[1]), $(types[1]), $(rms[1])" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = Dict(
            :Clarabel => OrderedDict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[1],
        kelly = krets[1],
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[1],
        kelly = krets[2],
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[1],
        kelly = krets[3],
    )
    w4 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[2],
        kelly = krets[1],
    )
    w5 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[2],
        kelly = krets[2],
    )
    w6 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[2],
        kelly = krets[3],
    )
    w7 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[3],
        kelly = krets[1],
    )
    risk7 = calc_risk(portfolio; type = types[1], rm = rms[1], rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[3],
        kelly = krets[2],
    )
    risk8 = calc_risk(portfolio; type = types[1], rm = rms[1], rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[3],
        kelly = krets[3],
    )
    risk9 = calc_risk(portfolio; type = types[1], rm = rms[1], rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[4],
        kelly = krets[1],
    )
    w11 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[4],
        kelly = krets[2],
    )
    w12 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[4],
        kelly = krets[3],
    )
    setproperty!(portfolio, :mu_l, ret7)
    w13 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[1],
        kelly = krets[1],
    )
    setproperty!(portfolio, :mu_l, ret8)
    w14 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[1],
        kelly = krets[1],
    )
    setproperty!(portfolio, :mu_l, ret9)
    w15 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[1],
        kelly = krets[1],
    )
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rms[1])) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[4],
        kelly = krets[1],
    )
    setproperty!(portfolio, rmf, risk8)
    w17 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[4],
        kelly = krets[1],
    )
    setproperty!(portfolio, rmf, risk9)
    w18 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[1],
        obj = objs[4],
        kelly = krets[1],
    )
    setproperty!(portfolio, rmf, Inf)

    w1t = [
        0.007909165129892893,
        0.03068795992113529,
        0.010505462324663135,
        0.027486145235593888,
        0.012276018553961078,
        0.033412161578499563,
        3.2321179119962583e-6,
        0.13985058398522618,
        4.5159126054365523e-7,
        6.679888687319533e-7,
        0.287814733702039,
        2.03553442446771e-6,
        5.907756693357551e-6,
        0.12527904140848548,
        3.4627255624635605e-6,
        0.015085226726492243,
        1.6440747053585462e-6,
        0.1931211902084604,
        9.21705659146357e-7,
        0.11655398773046471,
    ]
    w2t = [
        0.007909165129892893,
        0.03068795992113529,
        0.010505462324663135,
        0.027486145235593888,
        0.012276018553961078,
        0.033412161578499563,
        3.2321179119962583e-6,
        0.13985058398522618,
        4.5159126054365523e-7,
        6.679888687319533e-7,
        0.287814733702039,
        2.03553442446771e-6,
        5.907756693357551e-6,
        0.12527904140848548,
        3.4627255624635605e-6,
        0.015085226726492243,
        1.6440747053585462e-6,
        0.1931211902084604,
        9.21705659146357e-7,
        0.11655398773046471,
    ]
    w3t = [
        0.007910118680211848,
        0.030689246848362748,
        0.010506699313824301,
        0.027486415024679866,
        0.012277365375248477,
        0.033410990287855345,
        8.232782489688113e-8,
        0.13984815414419116,
        1.4352765201023874e-7,
        2.2500717252294503e-6,
        0.2878217656228225,
        9.619117118726138e-8,
        7.392169944225015e-8,
        0.1252822312874345,
        3.7523797951028816e-7,
        0.015085204052465634,
        3.4862429888958257e-6,
        0.19312270470448398,
        1.817151944134266e-7,
        0.11655241542218392,
    ]
    w4t = [
        2.3609747119111256e-6,
        2.3707353366244665e-7,
        5.718149540667744e-7,
        4.507210785199077e-6,
        0.7741585474739995,
        2.8839561220812933e-6,
        0.10998297045867082,
        2.3612276504874214e-6,
        7.716049671758139e-7,
        1.7550994655485284e-6,
        1.6429037560060136e-6,
        4.615175855163581e-6,
        2.691930003441115e-6,
        2.101553580562451e-6,
        6.708958961005669e-7,
        0.11582704950858472,
        2.1692204029307075e-6,
        8.48415979648811e-7,
        1.1086069756041419e-6,
        1.348941053730444e-7,
    ]
    w5t = [
        1.2549817039720941e-7,
        4.604336082269598e-8,
        1.5823129936477105e-8,
        1.1224973222554158e-7,
        0.7186292792203343,
        7.681501768247084e-8,
        0.09861460083278183,
        1.960963817550673e-7,
        4.313399353879921e-8,
        3.0578845672505957e-8,
        9.261287000575881e-8,
        7.907217950738718e-8,
        3.4710779925247866e-7,
        8.070100985103079e-8,
        5.1355405606660706e-8,
        0.15517358820342603,
        0.02758105181169084,
        9.98328984064895e-8,
        2.0147170947265042e-8,
        6.286380135976993e-8,
    ]
    w6t = [
        2.597026161649993e-7,
        1.3725460234237335e-6,
        1.1400043528465821e-6,
        1.0422228674703206e-7,
        0.72071260106369,
        2.5164487578560885e-6,
        0.09835331633497986,
        7.001280561997808e-6,
        2.727854579201843e-7,
        9.598913180748353e-7,
        4.678730491230836e-6,
        1.203661631313765e-6,
        1.8863788837906832e-5,
        2.512168367742872e-6,
        4.9710930693425485e-6,
        0.15503789676391527,
        0.02584631339750962,
        1.201218125333396e-6,
        1.1125498420546555e-6,
        1.7023481651513507e-6,
    ]
    w7t = [
        5.637515029624511e-8,
        1.9584441127122506e-8,
        1.480703402176879e-9,
        4.446858242112429e-8,
        0.5180367665642354,
        2.0868214072244138e-7,
        0.0636421912822041,
        6.657198335263341e-8,
        1.3522013729790282e-8,
        3.910144986145967e-8,
        1.1189676888767621e-7,
        1.7716955120464698e-7,
        3.7406907230970524e-7,
        1.6385621333084264e-7,
        2.54316123673522e-7,
        0.1432494516115838,
        0.19653839643726287,
        6.422873354861721e-8,
        0.0785315392658621,
        5.9515924035764746e-8,
    ]
    w8t = [
        1.100571398035122e-7,
        9.701022323123346e-8,
        2.7419765380115724e-7,
        7.924753436261306e-7,
        0.45216471686351906,
        3.555820324995833e-7,
        0.05193297740698236,
        2.476185393856072e-7,
        2.8639523259250403e-7,
        2.214388301461168e-8,
        8.60424179342337e-7,
        4.7548456829323383e-7,
        1.142842076634402e-7,
        2.1196263289049632e-7,
        4.0552461917333266e-7,
        0.13651684716868484,
        0.2351976577161838,
        4.9215235452735755e-8,
        0.12418345354574106,
        4.492339796172148e-8,
    ]
    w9t = [
        2.387216613685937e-8,
        7.828771156369864e-8,
        1.0229870690822666e-7,
        5.6299119725801325e-8,
        0.4894148286726404,
        7.444140272727556e-9,
        0.05833657453105973,
        8.106665070266756e-8,
        5.276461180379594e-8,
        2.7180318459157044e-8,
        5.86669235243501e-8,
        6.0358107251716945e-9,
        4.40402551166592e-9,
        1.3582073904098907e-8,
        4.27712094853996e-9,
        0.14032584660655578,
        0.2133774892572834,
        6.897142501094063e-8,
        0.09854461196095957,
        6.382069575305657e-8,
    ]
    w10t = [
        2.394474061468988e-10,
        9.691865713845658e-10,
        7.036190321490679e-10,
        1.7150390470839232e-9,
        3.189372166523254e-9,
        4.121721292463664e-10,
        0.9999999795140174,
        2.859736565383054e-10,
        7.12912870374418e-11,
        1.6015208291372517e-9,
        5.195036766277017e-10,
        3.6529643920589647e-9,
        1.2052256286507646e-9,
        1.068935971975755e-9,
        2.222851512944649e-10,
        1.202547990151027e-9,
        4.35029508556139e-10,
        2.1542761734425178e-10,
        2.03016923006831e-9,
        7.462709945112771e-10,
    ]
    w11t = [
        1.9155759157067284e-8,
        3.5382932361721944e-8,
        7.018550777468444e-8,
        7.807532206649825e-8,
        0.8503236241944822,
        5.652263701461687e-8,
        0.14967530302544793,
        6.104050462597333e-8,
        6.759109669561753e-8,
        4.1057301864427604e-8,
        1.0975543048809675e-8,
        3.240775394100855e-9,
        2.37685936114733e-7,
        3.3246265465548764e-8,
        8.127693632030379e-8,
        7.518036258848118e-9,
        1.0340426179878576e-7,
        7.34912523035038e-8,
        6.923866272826978e-8,
        2.369133880403176e-8,
    ]
    w12t = [
        2.7645873503234275e-9,
        2.973259803593348e-9,
        3.6878114973229714e-9,
        3.321370576117124e-9,
        0.8533950505923042,
        1.2742245971629924e-9,
        0.14660490317164734,
        1.9962609910135675e-9,
        3.1697472598073793e-9,
        2.1019702300844745e-9,
        1.8844964419676663e-9,
        1.2186378467195185e-9,
        8.840525159658307e-10,
        1.5657402848907941e-9,
        8.942385792282134e-10,
        6.363468927866918e-9,
        4.103435435690254e-9,
        2.1221723839725065e-9,
        3.387040113458199e-9,
        2.5235336172152343e-9,
    ]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-4)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-2)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-2)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-3)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-3)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-2)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-3)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-2)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-2)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-2)
end

@testset "$(classes[1]), $(types[1]), $(rms[2])" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = Dict(
            :Clarabel => OrderedDict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[1],
        kelly = krets[1],
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[1],
        kelly = krets[2],
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[1],
        kelly = krets[3],
    )
    w4 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[2],
        kelly = krets[1],
    )
    w5 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[2],
        kelly = krets[2],
    )
    w6 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[2],
        kelly = krets[3],
    )
    w7 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[3],
        kelly = krets[1],
    )
    risk7 = calc_risk(portfolio; type = types[1], rm = rms[2], rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[3],
        kelly = krets[2],
    )
    risk8 = calc_risk(portfolio; type = types[1], rm = rms[2], rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[3],
        kelly = krets[3],
    )
    risk9 = calc_risk(portfolio; type = types[1], rm = rms[2], rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[4],
        kelly = krets[1],
    )
    w11 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[4],
        kelly = krets[2],
    )
    w12 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[4],
        kelly = krets[3],
    )
    setproperty!(portfolio, :mu_l, ret7)
    w13 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[1],
        kelly = krets[1],
    )
    setproperty!(portfolio, :mu_l, ret8)
    w14 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[1],
        kelly = krets[1],
    )
    setproperty!(portfolio, :mu_l, ret9)
    w15 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[1],
        kelly = krets[1],
    )
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rms[2])) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[4],
        kelly = krets[1],
    )
    setproperty!(portfolio, rmf, risk8)
    w17 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[4],
        kelly = krets[1],
    )
    setproperty!(portfolio, rmf, risk9)
    w18 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[2],
        obj = objs[4],
        kelly = krets[1],
    )
    setproperty!(portfolio, rmf, Inf)

    w1t = [
        0.011536592369934385,
        0.04265236363572172,
        0.017016440367161566,
        0.002301300959666355,
        0.019037309743096083,
        0.05416089816562454,
        3.586145291624363e-6,
        0.15820244690119933,
        8.568553896855049e-7,
        2.114744760841588e-7,
        0.2371128231071386,
        8.834869803706911e-7,
        1.5049858277752496e-6,
        0.1276104639951137,
        0.0008982488700572819,
        0.0003054829474777324,
        0.043863513942966406,
        0.18364159672306285,
        3.9522858751453925e-7,
        0.10165308009522649,
    ]
    w2t = [
        0.014061752856837131,
        0.04237578891874663,
        0.016863690055886074,
        0.002020864183965532,
        0.017683588216924864,
        0.054224227833375285,
        2.429437994037918e-9,
        0.1582163369456717,
        3.1632003770504287e-9,
        1.0927844041886865e-8,
        0.2368971535868154,
        7.068169136095108e-10,
        5.920404944790139e-10,
        0.1278319758420608,
        0.00035110130472155846,
        0.0009122170031755927,
        0.04394981566186466,
        0.1827244653614497,
        5.119296838076666e-9,
        0.10188699928986843,
    ]
    w3t = [
        0.014061741691207703,
        0.04237665253779464,
        0.01686404443813141,
        0.0020208820443605355,
        0.017683153251940243,
        0.054224330485324025,
        4.397994305150137e-9,
        0.15821609411876122,
        4.404947340260583e-9,
        1.0898810833273423e-8,
        0.23689704820843152,
        1.4984607516738583e-9,
        1.353482739509153e-9,
        0.12783203738023824,
        0.00035127661577836615,
        0.0009121596326355965,
        0.04395015994905309,
        0.1827246337623197,
        5.722172879917316e-9,
        0.10188575760815485,
    ]
    w4t = [
        0.009424831248463862,
        0.05292040782870765,
        0.01007809982587373,
        0.011214429193060732,
        0.07065531587864068,
        5.561552303853725e-8,
        9.828222666334032e-8,
        0.14347563164249233,
        2.508498441610754e-7,
        0.013053054372599574,
        0.2019938186616052,
        6.368988253552106e-7,
        9.100271596157463e-7,
        0.07707890589904803,
        3.230379775191112e-7,
        0.01817335408461812,
        0.08411767104429861,
        0.16951093331165137,
        0.036537432944001094,
        0.10176383935338264,
    ]
    w5t = [
        0.00954001462989987,
        0.05608884354388105,
        0.009099080956435662,
        0.010229893132460477,
        0.07101511634469561,
        2.565766175478752e-9,
        0.00017197124742815046,
        0.14184884914672502,
        2.926730409653829e-9,
        0.011612816167616218,
        0.2033085543007606,
        1.2765261822267684e-9,
        1.2968716833134146e-9,
        0.07784157797062317,
        1.0738275904388476e-11,
        0.018210092020721925,
        0.0851246938503036,
        0.17133671710044868,
        0.03464895781294488,
        0.09992281369842247,
    ]
    w6t = [
        0.0096541945646638,
        0.05668225606733838,
        0.008734348582497732,
        0.0100326150536412,
        0.07088276444739741,
        4.276962566683111e-9,
        0.00027417427393046617,
        0.14173946581018104,
        5.338070235053351e-9,
        0.011687234549490623,
        0.20349109883888222,
        5.528641142269042e-10,
        5.939594758177769e-10,
        0.0779325440019444,
        1.5506089092558476e-9,
        0.018197720512202992,
        0.08532495894732768,
        0.17148151873340392,
        0.03409940792900322,
        0.09978568537562955,
    ]
    w7t = [
        9.259127402925153e-9,
        2.2980515040591824e-8,
        2.3804308460720392e-8,
        1.318154641226635e-8,
        0.6622515657528877,
        2.830715027119353e-9,
        0.04259149391317827,
        1.3386835891152789e-8,
        2.732658446888359e-8,
        1.1778099607993433e-8,
        2.0403898522298196e-8,
        2.0621159463481253e-9,
        1.463960975170087e-9,
        5.1526331089249376e-9,
        1.5995178505492494e-9,
        0.1343663509523453,
        0.08738455173733144,
        1.893470517579942e-8,
        0.07340584371041851,
        1.9769274963389775e-8,
    ]
    w8t = [
        3.5729505605139967e-7,
        6.239565838692029e-7,
        3.5078003910597426e-7,
        2.0831973428064871e-7,
        0.5063070440752794,
        3.252902217903201e-7,
        0.031761023566904856,
        4.3400913845851675e-8,
        7.294403188048272e-7,
        6.57634077594246e-7,
        0.0030706330178306416,
        1.5026411633230058e-7,
        1.853745075022494e-6,
        6.528267330645546e-7,
        1.51232191624419e-6,
        0.1380975650848352,
        0.1892916585839196,
        4.6312241493412136e-8,
        0.13146414784703317,
        4.1623716996717496e-7,
    ]
    w9t = [
        5.354967548546116e-7,
        2.536749371286361e-7,
        4.270459951872576e-7,
        2.5088206016360975e-7,
        0.5800950986627982,
        1.5338010448384115e-7,
        0.039106932990123314,
        6.269395568774156e-8,
        3.9963234884605826e-7,
        7.105551019307873e-7,
        3.9073546175044934e-8,
        6.017376674699383e-7,
        4.4341744761623017e-7,
        1.6735534185288507e-7,
        2.619167006747716e-7,
        0.13601059306412489,
        0.13843247234827671,
        1.7187266670437531e-7,
        0.106350033453632,
        3.9074641624738195e-7,
    ]
    w10t = [
        5.44181397962296e-9,
        1.0687288292329156e-9,
        9.520715756524373e-9,
        4.839663135772991e-9,
        1.5281220388616556e-9,
        4.738242328009774e-9,
        0.999999913507594,
        2.4150936628563626e-9,
        4.2940973987241036e-9,
        4.482686660969904e-9,
        7.451776463976159e-10,
        3.403983200407404e-9,
        1.58285538939002e-8,
        3.0421490797214415e-9,
        1.545882894442648e-9,
        7.322022338646908e-9,
        3.736148300844746e-9,
        4.29678468460637e-9,
        4.2702181465654555e-9,
        3.9723221485475396e-9,
    ]
    w11t = [
        4.975929017837501e-7,
        9.207699078031047e-7,
        7.189502012669678e-7,
        1.3509359257711308e-6,
        0.850286523838444,
        2.8122997890812475e-6,
        0.1496700485027153,
        2.6648551802267483e-6,
        3.2142651963942666e-6,
        2.5625240706718635e-6,
        2.0247008867018078e-6,
        2.101307235426345e-6,
        9.005047099801414e-6,
        2.4574851189660917e-6,
        4.698532482700366e-6,
        2.0693132858080525e-6,
        1.717807606307787e-6,
        2.503759328557567e-6,
        9.564871075703666e-7,
        1.1510255158048172e-6,
    ]
    w12t = [
        2.5742355639191003e-9,
        2.7707095338095783e-9,
        3.4398915027060286e-9,
        3.0946403611644163e-9,
        0.853395112703566,
        1.1885293695688552e-9,
        0.14660484419495157,
        1.8615532784981483e-9,
        2.9537909540856124e-9,
        1.9597683145029672e-9,
        1.7573815678039295e-9,
        1.1370240886156044e-9,
        8.260556319078895e-10,
        1.4588631844041763e-9,
        8.344564898483304e-10,
        5.929979283721413e-9,
        3.8282649737436145e-9,
        1.9764014752773826e-9,
        3.157429415745761e-9,
        2.3525073356166437e-9,
    ]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-5)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-2)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w8.weights, w9.weights, atol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-2)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-2)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-3)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-2)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
end

@testset "$(classes[1]), $(types[1]), $(rms[3])" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = Dict(
            :Clarabel => OrderedDict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[1],
        kelly = krets[1],
    )
    w2 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[1],
        kelly = krets[2],
    )
    w3 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[1],
        kelly = krets[3],
    )
    w4 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[2],
        kelly = krets[1],
    )
    w5 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[2],
        kelly = krets[2],
    )
    w6 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[2],
        kelly = krets[3],
    )
    w7 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[3],
        kelly = krets[1],
    )
    risk7 = calc_risk(portfolio; type = types[1], rm = rms[3], rf = rf)
    ret7 = dot(portfolio.mu, w7.weights)
    w8 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[3],
        kelly = krets[2],
    )
    risk8 = calc_risk(portfolio; type = types[1], rm = rms[3], rf = rf)
    ret8 = dot(portfolio.mu, w8.weights)
    w9 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[3],
        kelly = krets[3],
    )
    risk9 = calc_risk(portfolio; type = types[1], rm = rms[3], rf = rf)
    ret9 = dot(portfolio.mu, w9.weights)
    w10 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[4],
        kelly = krets[1],
    )
    w11 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[4],
        kelly = krets[2],
    )
    w12 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[4],
        kelly = krets[3],
    )
    setproperty!(portfolio, :mu_l, ret7)
    w13 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[1],
        kelly = krets[1],
    )
    setproperty!(portfolio, :mu_l, ret8)
    w14 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[1],
        kelly = krets[1],
    )
    setproperty!(portfolio, :mu_l, ret9)
    w15 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[1],
        kelly = krets[1],
    )
    setproperty!(portfolio, :mu_l, Inf)
    rmf = Symbol(lowercase(string(rms[3])) * "_u")
    setproperty!(portfolio, rmf, risk7)
    w16 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[4],
        kelly = krets[1],
    )
    setproperty!(portfolio, rmf, risk8)
    w17 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[4],
        kelly = krets[1],
    )
    setproperty!(portfolio, rmf, risk9)
    w18 = opt_port!(
        portfolio;
        rf = rf,
        l = l,
        class = classes[1],
        type = types[1],
        rrp_ver = rrpvs[1],
        u_mu = umus[1],
        u_cov = ucovs[1],
        rm = rms[3],
        obj = objs[4],
        kelly = krets[1],
    )
    setproperty!(portfolio, rmf, Inf)

    w1t = [
        2.294658710340585e-6,
        0.049763668733591046,
        7.961055145934363e-7,
        0.0029795274939585148,
        0.0025372274527977307,
        0.020117939536826016,
        6.112368887136918e-7,
        0.1280743214296101,
        1.2315795130739444e-6,
        1.0248408000678491e-6,
        0.2997374343048944,
        6.018022897851476e-7,
        9.834308301609585e-7,
        0.12048440118622397,
        0.01225795271427003,
        0.009665001430579332,
        5.073199110987412e-7,
        0.2293387216180222,
        1.6806888414369342e-6,
        0.1250340724359274,
    ]
    w2t = [
        3.5891892291328865e-7,
        0.049731356237034154,
        2.213226916274102e-7,
        0.0029825065561024646,
        0.0025432574731308485,
        0.020118197789344202,
        2.898922262258069e-7,
        0.12808851493900056,
        6.681707201415276e-8,
        1.9691849936506614e-7,
        0.2995836058391698,
        1.5265911064220314e-7,
        9.212890528396579e-7,
        0.1207162542191395,
        0.012266345340963,
        0.009660949761070908,
        1.5657719209783955e-7,
        0.22926764950411901,
        6.221965812244979e-7,
        0.1250383757495765,
    ]
    w3t = [
        3.261092110963764e-7,
        0.0497302741559704,
        3.8616546848731017e-7,
        0.0029688309235350306,
        0.0025478707753447075,
        0.020118844943269976,
        2.020537932003107e-5,
        0.12809658956131603,
        1.6323976783456562e-6,
        2.1647571886537855e-6,
        0.2995826841776134,
        1.527261284111934e-6,
        1.1148535418785614e-5,
        0.12069781641170096,
        0.012261439117636186,
        0.009654029349936099,
        1.0893005197319773e-6,
        0.22926132983834108,
        6.969933570607719e-7,
        0.12504111384588984,
    ]
    w4t = [
        8.795798553587976e-8,
        0.05530723320342701,
        2.661371854293347e-7,
        0.004276596642855529,
        0.03390755546528191,
        4.072810477698677e-8,
        1.0097284008755572e-7,
        0.12521191184886815,
        4.251090338025609e-7,
        3.1883478974400824e-8,
        0.29835536767268245,
        4.934555505539811e-7,
        2.0558559012340991e-7,
        0.10726741524009341,
        0.002761557199782584,
        0.02112759655991647,
        0.005349793283007991,
        0.22898532821276255,
        3.8065598654825226e-7,
        0.11744761218556614,
    ]
    w5t = [
        5.383385901867174e-8,
        0.05518230862115867,
        2.2640384612262514e-7,
        0.004441741060125254,
        0.033776800408247626,
        3.8625917218474385e-7,
        1.5543518201708106e-6,
        0.12529479174483446,
        1.5494110396011125e-7,
        5.425256861678395e-7,
        0.29840505644709947,
        1.7771177648805527e-8,
        1.2235336183402192e-6,
        0.10740711092062147,
        0.002730286843218186,
        0.021068859649230408,
        0.005382452264649192,
        0.22881294382965503,
        2.2756382672294833e-7,
        0.11749326102704992,
    ]
    w6t = [
        3.6963998014182935e-9,
        0.05518903048141686,
        3.4599618567368905e-9,
        0.0044374647398944455,
        0.03376640323328681,
        3.2510090714661645e-9,
        6.035569216006214e-10,
        0.12529118190304275,
        1.2618185082003576e-9,
        1.426408649803653e-9,
        0.2984002799241621,
        7.20856652878251e-10,
        4.411560345980172e-10,
        0.10741594769521565,
        0.002730634438652209,
        0.02106510515327547,
        0.005352551419669141,
        0.22884014402081754,
        1.8378569753810924e-9,
        0.11751124029154254,
    ]
    w7t = [
        1.4099948393469208e-6,
        1.848073486223839e-6,
        1.836305859010906e-6,
        1.2408355756967585e-6,
        0.6666600814590865,
        2.1853511841389627e-7,
        0.037933921155361566,
        1.3408356261538182e-6,
        1.9461665850559206e-6,
        7.839373970054235e-7,
        1.5739816797058624e-6,
        5.485142226847579e-7,
        2.74507795063819e-6,
        2.971997352499675e-7,
        3.998897913208504e-6,
        0.17183560186717134,
        0.10253717646722338,
        1.5552386800160526e-6,
        0.021010320905006923,
        1.5545514819998097e-6,
    ]
    w8t = [
        8.560027870991375e-8,
        2.3317000990770938e-7,
        3.1465442739030865e-7,
        1.7136681955990386e-6,
        0.5670154407602958,
        8.849315742580662e-7,
        0.029172701633831413,
        2.0139490146136813e-6,
        2.9347646123533755e-7,
        1.0678395120653644e-6,
        3.6965721903141065e-6,
        1.579543564549904e-6,
        1.7960500541146092e-6,
        8.399877013976244e-8,
        3.055554099629842e-6,
        0.1643290401884484,
        0.1621557316368289,
        3.930164985219438e-7,
        0.07730923368427274,
        6.400716718664505e-7,
    ]
    w9t = [
        1.691732873281931e-7,
        4.1902811252473786e-7,
        3.2433701622935056e-7,
        2.0739127262420673e-7,
        0.6079810913616143,
        5.8253204203366666e-8,
        0.03580115851913769,
        3.8006144372105117e-7,
        3.4322252514798573e-7,
        1.6628893025466347e-7,
        5.199176790478948e-7,
        4.788628324839719e-8,
        3.4435993172487995e-8,
        1.1183475431767843e-7,
        3.5668106101310785e-8,
        0.16670483358415644,
        0.13731644935712478,
        5.09438279984467e-7,
        0.05219279055456652,
        3.4968651256721096e-7,
    ]
    w10t = [
        6.014141670649078e-9,
        5.870574562212074e-8,
        2.18492172478836e-8,
        7.991539298696463e-8,
        1.777587612010077e-8,
        4.2421683722163386e-8,
        0.9999991371758413,
        6.497830999812479e-8,
        2.2749769728338092e-8,
        7.511606233744479e-9,
        3.1864586965607655e-8,
        1.3602018376945587e-7,
        3.1824994744856706e-8,
        4.940533827138055e-8,
        9.414508872397167e-8,
        7.803278451954132e-8,
        4.228922861914537e-8,
        3.067771701573662e-8,
        2.094725355514916e-8,
        2.5695239289355947e-8,
    ]
    w11t = [
        4.514978711322309e-6,
        1.204541599443082e-7,
        1.7531564347518097e-6,
        1.0596755631376526e-6,
        0.850283342045059,
        1.8384975852922195e-6,
        0.14966738818032477,
        1.424676077207898e-6,
        3.9766647941042735e-6,
        9.187773679287797e-7,
        5.73752611722372e-7,
        2.040653144129611e-6,
        5.685186891843195e-6,
        4.01256911727474e-6,
        6.481547887023812e-6,
        3.897979773189292e-6,
        1.6281761087168786e-6,
        5.7680545302252395e-6,
        4.128998172612027e-7,
        3.162074041158838e-6,
    ]
    w12t = [
        4.5988084643901325e-6,
        3.7508737433618526e-6,
        1.383806196251782e-6,
        4.677750639680115e-6,
        0.8532885749379749,
        4.334317773252942e-6,
        0.1466001460795411,
        3.376371265671292e-6,
        3.322327129380064e-6,
        1.822292606284569e-6,
        5.369512963696114e-6,
        1.2172832024799194e-5,
        1.8801899088377826e-5,
        8.494869325940076e-7,
        3.333728180742772e-5,
        2.2902137545084894e-6,
        1.9352926456574743e-7,
        6.97952067977199e-6,
        3.204567906516933e-6,
        8.135902435246453e-7,
    ]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w2.weights, w3.weights, rtol = 1e-4)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w5.weights, w6.weights, rtol = 1e-3)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w8.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test isapprox(w11.weights, w12.weights, rtol = 1e-2)
    @test isapprox(w13.weights, w7.weights, rtol = 1e-3)
    @test isapprox(w14.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w15.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w16.weights, w7.weights, rtol = 1e-4)
    @test isapprox(w17.weights, w8.weights, rtol = 1e-1)
    @test isapprox(w18.weights, w9.weights, rtol = 1e-1)
    @test isapprox(w13.weights, w16.weights, rtol = 1e-3)
    @test isapprox(w14.weights, w17.weights, rtol = 1e-3)
    @test isapprox(w15.weights, w18.weights, rtol = 1e-3)
end