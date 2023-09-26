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
    HiGHS

A = TimeArray(CSV.File("./assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))
rf = 1.0329^(1 / 252) - 1
l = 2.0
type = :trad

@testset "SD" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SD

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.007908005080901599,
        0.030690249796045278,
        0.01050698801014926,
        0.027487038641603477,
        0.012278036322950098,
        0.033411704997684165,
        4.470549367699486e-10,
        0.13984847338907136,
        8.073120621710569e-10,
        3.005906213628074e-8,
        0.2878225727483246,
        4.926242422793698e-10,
        3.591040640033909e-10,
        0.1252836399316465,
        1.4665856176546734e-9,
        0.015085379086518916,
        6.69139552775205e-8,
        0.19312407728704264,
        1.0288534507044424e-9,
        0.11655373313351022,
    ]
    w2t = [
        1.8504378837346683e-10,
        2.859054220733439e-10,
        3.629887313382879e-10,
        2.9730232445529207e-10,
        0.7741880999723328,
        6.136538564429944e-11,
        0.10999292898269301,
        1.7848007318393276e-10,
        3.686085195361869e-10,
        1.7323067191690648e-10,
        1.648706748831337e-10,
        5.62073356474089e-11,
        3.667228998653034e-11,
        1.0012924513259964e-10,
        3.594249486077627e-11,
        0.11581896676141921,
        9.563142603215974e-10,
        1.919988222655805e-10,
        5.899098778416517e-10,
        2.385849352545192e-10,
    ]
    w3t = [
        7.179506307916457e-12,
        1.914107597851177e-11,
        2.442828242865303e-11,
        1.5415316473711545e-11,
        0.5180589873168485,
        2.2248163559762217e-12,
        0.06365089570166141,
        1.6083198093728474e-11,
        1.5767461160799445e-11,
        7.937026546069197e-12,
        1.3137860840363715e-11,
        1.7427427416358682e-12,
        1.2271864288937697e-12,
        4.0970513336755e-12,
        1.2119635713300584e-12,
        0.14326820230127735,
        0.19649614109499838,
        1.5441931656159965e-11,
        0.07852577342436559,
        1.5813088271938218e-11,
    ]
    w4t = [
        6.580527355161422e-12,
        6.592533527352555e-12,
        6.837381947934612e-12,
        6.835652585151393e-12,
        1.1249845694181488e-9,
        4.931585765770242e-12,
        0.9999999987672188,
        6.071877703469958e-12,
        7.0594318275308255e-12,
        6.243463219649402e-12,
        6.00799701800439e-12,
        4.4585463812496784e-12,
        3.3128937905786473e-12,
        5.601022741433536e-12,
        3.851608127252456e-12,
        7.31617567849035e-12,
        6.6444228969708886e-12,
        6.167997935289061e-12,
        6.817498571215114e-12,
        6.46609443105899e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-4)
    @test isapprox(w2t, w2.weights, rtol = 3e-5)
    @test isapprox(w3t, w3.weights, rtol = 4e-5)
    @test isapprox(w4t, w4.weights, rtol = 3e-8)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.007908005080901599,
        0.030690249796045278,
        0.01050698801014926,
        0.027487038641603477,
        0.012278036322950098,
        0.033411704997684165,
        4.470549367699486e-10,
        0.13984847338907136,
        8.073120621710569e-10,
        3.005906213628074e-8,
        0.2878225727483246,
        4.926242422793698e-10,
        3.591040640033909e-10,
        0.1252836399316465,
        1.4665856176546734e-9,
        0.015085379086518916,
        6.69139552775205e-8,
        0.19312407728704264,
        1.0288534507044424e-9,
        0.11655373313351022,
    ]
    w2t = [
        1.8084767726475723e-9,
        3.15598787930466e-9,
        3.993045017470242e-9,
        3.0384397270199916e-9,
        0.7186299565717855,
        6.129692334645302e-10,
        0.09860145533248676,
        1.9497340297073745e-9,
        5.395233170821649e-9,
        1.7484893608991342e-9,
        1.8787220297108975e-9,
        5.228938242435098e-10,
        3.4287840078386283e-10,
        1.0428972048971869e-9,
        3.534927700681722e-10,
        0.15517802927199634,
        0.027590510040471142,
        2.2348240297174565e-9,
        1.8089277922242984e-8,
        2.6158990418725226e-9,
    ]
    w3t = [
        2.994788155563856e-8,
        1.1689379051337314e-7,
        1.360745386554711e-7,
        7.76443556091882e-8,
        0.4523917992390467,
        8.724346784224794e-9,
        0.051975236544563985,
        1.0369708072475508e-7,
        5.1270683674142364e-8,
        3.5518177851359255e-8,
        1.4718464947184667e-7,
        6.395018328265321e-9,
        4.642995437880929e-9,
        1.734097617571041e-8,
        4.588332289519737e-9,
        0.13644166802105195,
        0.23517690632056756,
        1.4967279522829946e-7,
        0.12401339045432143,
        1.0982482590303471e-7,
    ]
    w4t = [
        1.492384419146211e-11,
        1.6739191162167344e-11,
        2.0836013764845944e-11,
        1.8751749610844453e-11,
        0.8503252807188676,
        5.7961013000671885e-12,
        0.14967471902985832,
        1.0927909565621052e-11,
        1.7808151191717344e-11,
        1.2061312430023853e-11,
        1.0494612386411761e-11,
        5.9927999064637444e-12,
        3.7216962076482586e-12,
        8.205374317845782e-12,
        3.877654760300829e-12,
        3.231530012167001e-11,
        2.320519555824282e-11,
        1.1674082362668371e-11,
        1.9169856390225634e-11,
        1.4773199024696087e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-4)
    @test isapprox(w2t, w2.weights, rtol = 7e-5)
    @test isapprox(w3t, w3.weights, rtol = 3e-4)
    @test isapprox(w4t, w4.weights, rtol = 3e-5)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.007908005080901599,
        0.030690249796045278,
        0.01050698801014926,
        0.027487038641603477,
        0.012278036322950098,
        0.033411704997684165,
        4.470549367699486e-10,
        0.13984847338907136,
        8.073120621710569e-10,
        3.005906213628074e-8,
        0.2878225727483246,
        4.926242422793698e-10,
        3.591040640033909e-10,
        0.1252836399316465,
        1.4665856176546734e-9,
        0.015085379086518916,
        6.69139552775205e-8,
        0.19312407728704264,
        1.0288534507044424e-9,
        0.11655373313351022,
    ]
    w2t = [
        5.916686728668315e-9,
        1.0836209607054554e-8,
        1.3804374676405447e-8,
        1.0502197573449387e-8,
        0.720711229502948,
        2.0983733777480523e-9,
        0.09835466117687634,
        6.8516422322943975e-9,
        1.4605175159013772e-8,
        5.9328852690414205e-9,
        6.202692618703326e-9,
        1.813891574018169e-9,
        1.3175546850946821e-9,
        3.3711341934478463e-9,
        1.2990714899169543e-9,
        0.1550518109880465,
        0.025882145861629416,
        7.363060369189841e-9,
        5.175843124132503e-8,
        8.79711887272667e-9,
    ]
    w3t = [
        1.2366247299308615e-8,
        3.8760594982905365e-8,
        5.01769357793134e-8,
        2.90490208031484e-8,
        0.48941958484418036,
        3.6849251993879394e-9,
        0.05833763120208744,
        4.096965328704624e-8,
        2.402361391896188e-8,
        1.4149174480171825e-8,
        3.0381982896359125e-8,
        2.8440017698383022e-9,
        2.0683637060318e-9,
        7.018436258480784e-9,
        2.001421994532382e-9,
        0.140327354862629,
        0.2133795775140911,
        3.455573050831223e-8,
        0.0985355268380842,
        3.268882511261784e-8,
    ]
    w4t = [
        2.697552976386826e-12,
        3.020953795609185e-12,
        3.753620425993217e-12,
        3.384880662539435e-12,
        0.8533951447716036,
        1.1879262268291211e-12,
        0.1466048551815731,
        2.057259241821614e-12,
        3.271989728932198e-12,
        2.2255474150587952e-12,
        1.9770094329998858e-12,
        1.2319437947926673e-12,
        8.781818466077055e-13,
        1.5791159073970072e-12,
        8.925281327890656e-13,
        6.117590249323203e-12,
        4.204128068671476e-12,
        2.165843521220285e-12,
        3.502429128038115e-12,
        2.674823455426402e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 7e-5)
    @test isapprox(w2t, w2.weights, rtol = 3e-3)
    @test isapprox(w3t, w3.weights, rtol = 4e-5)
    @test isapprox(w4t, w4.weights, rtol = 5e-6)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SD
    kelly = :none
    portfolio.dev_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = :SD)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = :SD)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = :SD)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = :SD)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.dev_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = :SD)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 2e-3)
    @test isapprox(r5, r1, rtol = 2e-7)
    @test isapprox(m5, m1, rtol = 4e-3)

    portfolio.dev_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = :SD)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 7e-6)
    @test isapprox(r6, r2, rtol = 9e-8)
    @test isapprox(m6, m2, rtol = 4e-7)

    portfolio.dev_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = :SD)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 8e-6)
    @test isapprox(r7, r3, rtol = 5e-8)
    @test isapprox(m7, m3, rtol = 5e-7)

    portfolio.dev_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = :SD)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 8e-6)
    @test isapprox(r8, r4, rtol = 5e-6)
    @test isapprox(m8, m4, rtol = 4e-7)

    portfolio.dev_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.dev_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = :SD)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 2e-3)
    @test isapprox(r13, r1)
    @test isapprox(m13, m1, rtol = 4e-3)

    obj = :min_risk
    portfolio.dev_u = Inf
    portfolio.mu_l = m1
    w9 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r9 = calc_risk(portfolio, rm = :SD)
    m9 = dot(portfolio.mu, w9.weights)
    @test isapprox(w9.weights, w1.weights, rtol = 2e-3)
    @test isapprox(r9, r1, rtol = 8e-7)
    @test isapprox(m9, m1, rtol = 2e-3)

    portfolio.mu_l = m2
    w10 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r10 = calc_risk(portfolio, rm = :SD)
    m10 = dot(portfolio.mu, w10.weights)
    @test isapprox(w10.weights, w2.weights, rtol = 7e-6)
    @test isapprox(r10, r2, rtol = 9e-7)
    @test isapprox(m10, m2, rtol = 3e-8)

    portfolio.mu_l = m3
    w11 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r11 = calc_risk(portfolio, rm = :SD)
    m11 = dot(portfolio.mu, w11.weights)
    @test isapprox(w11.weights, w3.weights, rtol = 1e-5)
    @test isapprox(r11, r3, rtol = 1e-6)
    @test isapprox(m11, m3, rtol = 6e-8)

    portfolio.mu_l = m4
    w12 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r12 = calc_risk(portfolio, rm = :SD)
    m12 = dot(portfolio.mu, w12.weights)
    @test isapprox(w12.weights, w4.weights, rtol = 2e-5)
    @test isapprox(r12, r4, rtol = 2e-5)
    @test isapprox(m12, m4, rtol = 2e-10)
end

@testset "MAD" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :MAD

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.014061880636502702,
        0.042374911769897106,
        0.01686333849846529,
        0.002020812321804824,
        0.01768388643597061,
        0.054224067212201486,
        2.5285705216797175e-12,
        0.1582165546075748,
        2.4580789142464603e-12,
        6.635484861618362e-12,
        0.23689727662799842,
        2.860639084255924e-14,
        8.536213543127721e-14,
        0.1278320022564028,
        0.0003509457152103411,
        0.0009122889812524929,
        0.04394937939355737,
        0.18272427854703233,
        3.4419533445581458e-12,
        0.10188837698095136,
    ]
    w2t = [
        0.009581435615911975,
        0.0562750193218485,
        0.008927351132833645,
        0.010190537205604415,
        0.07101666352123231,
        3.418178924505084e-9,
        0.00017802961625240025,
        0.14177251633039242,
        3.967780956372942e-9,
        0.011581823277704503,
        0.2033448930886587,
        4.0308877598657896e-10,
        4.477979680374143e-10,
        0.07778000907383353,
        1.228782967395642e-9,
        0.018217317664500226,
        0.08521738874972037,
        0.17150733209884145,
        0.03456659785543433,
        0.09984307598160146,
    ]
    w3t = [
        3.6505434757561137e-11,
        1.1760119951123315e-10,
        1.2484360660470936e-10,
        6.187461102716296e-11,
        0.6622657503882446,
        3.163239689478531e-11,
        0.04259470199789395,
        5.236792200043181e-11,
        1.294473002321869e-10,
        4.625165845002466e-11,
        9.65101902925865e-11,
        3.0215304348004426e-11,
        4.614294245962137e-11,
        4.394867706953358e-12,
        4.9837601080195894e-11,
        0.13436659129606182,
        0.08736657849319988,
        8.867633096743114e-11,
        0.07340637681124025,
        9.705810174060336e-11,
    ]
    w4t = [
        9.173670341082928e-11,
        1.018721492422151e-10,
        1.4792929705934207e-10,
        1.2195080913664372e-10,
        4.002156884972695e-9,
        3.383705282558462e-11,
        0.9999999944089604,
        6.266311240046398e-11,
        1.0591250169076717e-10,
        6.489443800628426e-11,
        6.04540156207843e-11,
        3.295512734478604e-11,
        2.2582689785453977e-11,
        4.587953990398343e-11,
        2.4076677157718124e-11,
        2.388673459501035e-10,
        1.627041098591065e-10,
        6.611088689517032e-11,
        1.171673646256256e-10,
        8.728903651047666e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-4)
    @test isapprox(w2t, w2.weights, rtol = 2e-3)
    @test isapprox(w3t, w3.weights, rtol = 4e-4)
    @test isapprox(w4t, w4.weights, rtol = 4e-6)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.014061880636502702,
        0.042374911769897106,
        0.01686333849846529,
        0.002020812321804824,
        0.01768388643597061,
        0.054224067212201486,
        2.5285705216797175e-12,
        0.1582165546075748,
        2.4580789142464603e-12,
        6.635484861618362e-12,
        0.23689727662799842,
        2.860639084255924e-14,
        8.536213543127721e-14,
        0.1278320022564028,
        0.0003509457152103411,
        0.0009122889812524929,
        0.04394937939355737,
        0.18272427854703233,
        3.4419533445581458e-12,
        0.10188837698095136,
    ]
    w2t = [
        0.009637860191031553,
        0.05653151173009478,
        0.008693290402406864,
        0.010136510624477449,
        0.07101741008352966,
        5.1479344876975744e-11,
        0.00018718971750499648,
        0.14166816109955463,
        6.522615587318691e-11,
        0.011541247378874771,
        0.20339667013168472,
        2.0158611466839134e-12,
        2.9518233673619816e-12,
        0.07769848088109295,
        1.6085256375784812e-11,
        0.018226691891045345,
        0.08534287177184482,
        0.17173813508569383,
        0.034449430137292646,
        0.0997345387361127,
    ]
    w3t = [
        2.7544992311002863e-9,
        8.572433683762577e-9,
        1.1799978140364881e-8,
        3.457175111615613e-9,
        0.5086311805686491,
        7.715521271140319e-10,
        0.03158984132973597,
        1.7294761139855175e-8,
        4.847936638736765e-9,
        2.989579035311913e-9,
        0.002772419361011961,
        4.92450371190903e-10,
        3.675908754817044e-10,
        1.508568609557153e-9,
        4.0467654405943263e-10,
        0.13813036059569692,
        0.18544670962184126,
        2.2100655329838375e-8,
        0.13342939943929194,
        1.172191615720413e-8,
    ]
    w4t = [
        6.247830609842199e-10,
        9.366365031334395e-10,
        1.642821189402799e-9,
        1.283408037974675e-9,
        0.8503450577367491,
        8.327211967116204e-10,
        0.14965492407464284,
        1.8834240112637242e-11,
        1.171489008158045e-9,
        1.7599100926276864e-10,
        5.734286258047603e-11,
        7.855146385568771e-10,
        1.1091100526954552e-9,
        4.4569921209670714e-10,
        1.1055541094160778e-9,
        3.7905448736999e-9,
        2.0750033199629922e-9,
        1.207009644535358e-10,
        1.401800396224821e-9,
        6.106532670780226e-10,
    ]

    @test isapprox(w1t, w1.weights, rtol = 6e-7)
    @test isapprox(w2t, w2.weights, rtol = 3e-3)
    @test isapprox(w3t, w3.weights, rtol = 2e-3)
    @test isapprox(w4t, w4.weights, rtol = 1e-9)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.014061880636502702,
        0.042374911769897106,
        0.01686333849846529,
        0.002020812321804824,
        0.01768388643597061,
        0.054224067212201486,
        2.5285705216797175e-12,
        0.1582165546075748,
        2.4580789142464603e-12,
        6.635484861618362e-12,
        0.23689727662799842,
        2.860639084255924e-14,
        8.536213543127721e-14,
        0.1278320022564028,
        0.0003509457152103411,
        0.0009122889812524929,
        0.04394937939355737,
        0.18272427854703233,
        3.4419533445581458e-12,
        0.10188837698095136,
    ]
    w2t = [
        0.009649764014266284,
        0.05665188270056131,
        0.008738135114762506,
        0.010048707088017599,
        0.07090074538875636,
        3.963408467729263e-9,
        0.00026214661248251836,
        0.14173403815265792,
        5.343514559766271e-9,
        0.01166927995877412,
        0.20347642922907602,
        4.639935804015694e-10,
        5.296982899287488e-10,
        0.07790441819403413,
        1.5266283158812933e-9,
        0.018201229128942096,
        0.0853223695866153,
        0.1715067419961666,
        0.03415094730100343,
        0.09978315370664052,
    ]
    w3t = [
        1.0235133677049314e-9,
        2.717916196062684e-9,
        3.30900102511276e-9,
        1.4753774416959354e-9,
        0.5791916399450482,
        3.1372674735154114e-10,
        0.03881583481996154,
        2.254341895237212e-9,
        2.3262224232308805e-9,
        1.2098174915175847e-9,
        3.855882837300716e-9,
        2.1521451002041826e-10,
        1.5854231209497452e-10,
        5.90101969090206e-10,
        1.7164569895001376e-10,
        0.13597945491561356,
        0.13969680435134357,
        2.9848878059830347e-9,
        0.10631624075541607,
        2.6064253550456307e-9,
    ]
    w4t = [
        2.7462995127650345e-12,
        3.4547354561297505e-12,
        4.638228249512704e-12,
        4.027450273303031e-12,
        0.8533951447671612,
        1.2491743001999807e-12,
        0.14660485517629157,
        2.2606929911136244e-12,
        4.074536117285144e-12,
        2.5896147160714515e-12,
        2.1791822915117813e-12,
        1.3835188712694755e-12,
        1.1710980464498196e-12,
        1.6965332185628418e-12,
        1.1455634621293855e-12,
        8.511463740765157e-12,
        5.491779287346602e-12,
        2.368415950460791e-12,
        4.427973683207736e-12,
        3.130826651748219e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 2e-6)
    @test isapprox(w2t, w2.weights, rtol = 3e-4)
    @test isapprox(w3t, w3.weights, rtol = 3e-4)
    @test isapprox(w4t, w4.weights, rtol = 3e-5)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)

    kelly = :none
    portfolio.mad_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.mad_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 2e-3)
    @test isapprox(r5, r1, rtol = 2e-7)
    @test isapprox(m5, m1, rtol = 2e-4)

    portfolio.mad_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 3e-4)
    @test isapprox(r6, r2, rtol = 5e-7)
    @test isapprox(m6, m2, rtol = 5e-6)

    portfolio.mad_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 5e-6)
    @test isapprox(r7, r3, rtol = 2e-6)
    @test isapprox(m7, m3, rtol = 2e-6)

    portfolio.mad_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 5e-5)
    @test isapprox(r8, r4, rtol = 3e-5)
    @test isapprox(m8, m4, rtol = 4e-7)

    portfolio.mad_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.mad_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 2e-4)
    @test isapprox(r13, r1)
    @test isapprox(m13, m1, rtol = 7e-6)
end

@testset "SSD" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SSD

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        3.1129089757995387e-10,
        0.04973782521677862,
        1.429378265771766e-10,
        0.002977871127532673,
        0.0025509272401636114,
        0.02013279896189975,
        2.1549243386052154e-11,
        0.1280947541046824,
        1.3166390364848502e-12,
        9.65489638554896e-12,
        0.2995770810993405,
        1.1068810958359105e-11,
        1.326131764207242e-11,
        0.12069610574950249,
        0.012266324879126218,
        0.009663104378288215,
        1.845494805500937e-10,
        0.22927598326596532,
        6.75204453413519e-12,
        0.12502722327433904,
    ]
    w2t = [
        1.55193312992023e-8,
        0.05530890633965519,
        1.4750240918886373e-8,
        0.004276236563089458,
        0.033907283846895724,
        1.4185530061826787e-8,
        1.727561372509235e-9,
        0.1252160791387278,
        4.629555278520507e-9,
        5.246627923854521e-9,
        0.29837137763130545,
        2.688585337667067e-9,
        1.5548185551870198e-9,
        0.10724288240746908,
        0.0027606550332788912,
        0.021120810842555595,
        0.005355965799278624,
        0.22898979636608655,
        6.8011104839264295e-9,
        0.11744993892829635,
    ]
    w3t = [
        1.4020058698069193e-10,
        7.701036227657359e-10,
        6.6980461881224e-10,
        3.1655858990984467e-10,
        0.6665958991853101,
        7.963765287254749e-10,
        0.03791811841238643,
        3.596848608407397e-10,
        8.147121133955132e-10,
        3.827948638087292e-11,
        5.598067335418763e-10,
        6.948718740849434e-10,
        9.413099814872317e-10,
        3.5533893627365026e-10,
        1.0546331722059122e-9,
        0.17184698735591036,
        0.10259045728298342,
        6.620599480047146e-10,
        0.02104852902992978,
        5.597385928493645e-10,
    ]
    w4t = [
        9.173670341082928e-11,
        1.018721492422151e-10,
        1.4792929705934207e-10,
        1.2195080913664372e-10,
        4.002156884972695e-9,
        3.383705282558462e-11,
        0.9999999944089604,
        6.266311240046398e-11,
        1.0591250169076717e-10,
        6.489443800628426e-11,
        6.04540156207843e-11,
        3.295512734478604e-11,
        2.2582689785453977e-11,
        4.587953990398343e-11,
        2.4076677157718124e-11,
        2.388673459501035e-10,
        1.627041098591065e-10,
        6.611088689517032e-11,
        1.171673646256256e-10,
        8.728903651047666e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 8e-6)
    @test isapprox(w2t, w2.weights, rtol = 6e-6)
    @test isapprox(w3t, w3.weights, rtol = 5e-5)
    @test isapprox(w4t, w4.weights, rtol = 2e-6)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        3.1129089757995387e-10,
        0.04973782521677862,
        1.429378265771766e-10,
        0.002977871127532673,
        0.0025509272401636114,
        0.02013279896189975,
        2.1549243386052154e-11,
        0.1280947541046824,
        1.3166390364848502e-12,
        9.65489638554896e-12,
        0.2995770810993405,
        1.1068810958359105e-11,
        1.326131764207242e-11,
        0.12069610574950249,
        0.012266324879126218,
        0.009663104378288215,
        1.845494805500937e-10,
        0.22927598326596532,
        6.75204453413519e-12,
        0.12502722327433904,
    ]
    w2t = [
        2.987690503920611e-8,
        0.05530508224435074,
        2.8388871816456116e-8,
        0.004274714298079788,
        0.03390394380455495,
        2.7297988861851484e-8,
        3.3412061748632256e-9,
        0.1252163515267396,
        8.920959970044592e-9,
        1.0106778322532886e-8,
        0.29838292553812706,
        5.189087901339316e-9,
        3.009049853934386e-9,
        0.1072410425878912,
        0.002761286726878307,
        0.02111931898148037,
        0.00535402295117604,
        0.2289859822647217,
        1.3097569374507497e-8,
        0.1174551998475829,
    ]
    w3t = [
        2.2561439129349693e-7,
        6.817798453900856e-7,
        4.4357565870073027e-7,
        2.7387202330454094e-7,
        0.5658856354128571,
        7.377489571222419e-8,
        0.02902206465175713,
        9.678928970756973e-7,
        3.628430675283585e-7,
        2.3321487767535117e-7,
        2.51986554125595e-6,
        5.5625356929041854e-8,
        4.060610610285758e-8,
        1.5520525293369017e-7,
        4.2628087881141595e-8,
        0.16420032748319247,
        0.16305408530857535,
        1.3491200911944634e-6,
        0.07782984575857536,
        6.157669496191041e-7,
    ]
    w4t = [
        1.530418161094495e-10,
        2.861231416908675e-10,
        5.50565719114187e-10,
        4.5496566639282957e-10,
        0.8503475681594903,
        3.8305109783050716e-10,
        0.14965242607382143,
        7.250695003305623e-11,
        3.8623708593737487e-10,
        1.7288368272702123e-11,
        9.18170076454288e-11,
        3.4046685216501507e-10,
        3.8768506683038414e-10,
        2.347497794488437e-10,
        4.1508405016915417e-10,
        6.184903567113634e-10,
        7.002855763348704e-10,
        3.106332338153356e-11,
        4.812278883700028e-10,
        1.6203871132884816e-10,
    ]

    @test isapprox(w1t, w1.weights, rtol = 5e-6)
    @test isapprox(w2t, w2.weights, rtol = 9e-4)
    @test isapprox(w3t, w3.weights, rtol = 5e-4)
    @test isapprox(w4t, w4.weights, rtol = 1e-9)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        3.1129089757995387e-10,
        0.04973782521677862,
        1.429378265771766e-10,
        0.002977871127532673,
        0.0025509272401636114,
        0.02013279896189975,
        2.1549243386052154e-11,
        0.1280947541046824,
        1.3166390364848502e-12,
        9.65489638554896e-12,
        0.2995770810993405,
        1.1068810958359105e-11,
        1.326131764207242e-11,
        0.12069610574950249,
        0.012266324879126218,
        0.009663104378288215,
        1.845494805500937e-10,
        0.22927598326596532,
        6.75204453413519e-12,
        0.12502722327433904,
    ]
    w2t = [
        5.004945744409125e-9,
        0.05518898998457758,
        4.773201482097686e-9,
        0.004437473075963702,
        0.0337663950316292,
        4.615138862710002e-9,
        5.566056795652831e-10,
        0.12529118681581383,
        1.4810222579928231e-9,
        1.6899301509549966e-9,
        0.29840030505067183,
        8.624328911487197e-10,
        5.011533509356351e-10,
        0.10741597225906216,
        0.0027306427297041015,
        0.021065100035719056,
        0.0053525308820444786,
        0.22884012160245676,
        2.171300548362501e-9,
        0.11751126087662636,
    ]
    w3t = [
        2.2138825065410003e-8,
        5.808945242308328e-8,
        4.313646644994959e-8,
        2.6728878536988503e-8,
        0.6079823928128232,
        7.362759171084387e-9,
        0.035790143378752935,
        5.449772392386681e-8,
        4.3824965313017236e-8,
        2.2074615770094526e-8,
        7.510962539406988e-8,
        5.81142260331465e-9,
        4.186060219236698e-9,
        1.4904465901926345e-8,
        4.355898268392594e-9,
        0.16672170400750502,
        0.1373135283178017,
        7.242810575019228e-8,
        0.05219172790132909,
        4.8932523319133615e-8,
    ]
    w4t = [
        2.7462995127650345e-12,
        3.4547354561297505e-12,
        4.638228249512704e-12,
        4.027450273303031e-12,
        0.8533951447671612,
        1.2491743001999807e-12,
        0.14660485517629157,
        2.2606929911136244e-12,
        4.074536117285144e-12,
        2.5896147160714515e-12,
        2.1791822915117813e-12,
        1.3835188712694755e-12,
        1.1710980464498196e-12,
        1.6965332185628418e-12,
        1.1455634621293855e-12,
        8.511463740765157e-12,
        5.491779287346602e-12,
        2.368415950460791e-12,
        4.427973683207736e-12,
        3.130826651748219e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 6e-5)
    @test isapprox(w2t, w2.weights, rtol = 3e-6)
    @test isapprox(w3t, w3.weights, rtol = 7e-4)
    @test isapprox(w4t, w4.weights, rtol = 5e-6)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SSD
    kelly = :none
    portfolio.sdev_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.sdev_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 4e-4)
    @test isapprox(r5, r1, rtol = 4e-8)
    @test isapprox(m5, m1, rtol = 1e-3)

    portfolio.sdev_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 3e-4)
    @test isapprox(r6, r2, rtol = 5e-7)
    @test isapprox(m6, m2, rtol = 5e-6)

    portfolio.sdev_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 3e-6)
    @test isapprox(r7, r3, rtol = 1e-6)
    @test isapprox(m7, m3, rtol = 1e-6)

    portfolio.sdev_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 7e-5)
    @test isapprox(r8, r4, rtol = 4e-5)
    @test isapprox(m8, m4, rtol = 3e-7)

    portfolio.sdev_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.sdev_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 2e-3)
    @test isapprox(r13, r1, rtol = 2e-6)
    @test isapprox(m13, m1, rtol = 6e-3)
end

@testset "FLPM" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :FLPM

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.004266104054352623,
        0.04362179139799261,
        0.01996045918300812,
        0.007822769449990673,
        0.06052525732257479,
        2.2211049042813212e-11,
        0.0003959617775312277,
        0.1308926806342499,
        8.0214707797462e-12,
        0.011878930331335047,
        0.20660964132624798,
        4.553198556527535e-13,
        4.833024630837389e-13,
        0.08329480532205666,
        8.342668739909475e-13,
        0.01388809686095509,
        0.08734726339950173,
        0.1921003427001766,
        0.03721320875129444,
        0.10018268745672719,
    ]
    w2t = [
        1.8594027754958927e-8,
        0.0421384952420599,
        0.012527612299503596,
        0.007205887495437495,
        0.09868068752245247,
        1.8752600092747134e-9,
        0.0027731712941033936,
        0.10920980934701956,
        5.253410669088289e-9,
        1.700563384913051e-8,
        0.20154772848944386,
        5.172974436424006e-10,
        5.009390254440632e-10,
        0.026543608941042865,
        8.590302773990872e-10,
        0.04173039782244224,
        0.11080146520430383,
        0.17232528722022764,
        0.06800153745456954,
        0.10651426706179459,
    ]
    w3t = [
        1.1587206453251757e-11,
        3.9709992104292095e-11,
        4.209755573399237e-11,
        2.317979270933299e-11,
        0.6999124722828386,
        1.2205682223251806e-11,
        0.029295829804598526,
        2.4104366228902963e-11,
        5.2853046482197944e-11,
        1.7619474604794833e-11,
        3.5550313985279626e-11,
        1.1880907260469357e-11,
        1.817698021845065e-11,
        1.2427166037830573e-12,
        1.956975879233484e-11,
        0.15181164397586505,
        0.042264600174880755,
        3.3653272525668416e-11,
        0.07671545337385213,
        4.4533874225390245e-11,
    ]
    w4t = [
        9.28763383151679e-11,
        1.0307866727010052e-10,
        1.4940823317834682e-10,
        1.2336714694005373e-10,
        4.0216604451367195e-9,
        3.433828145899989e-11,
        0.9999999943713728,
        6.346479442018532e-11,
        1.0727008867463966e-10,
        6.582256693437888e-11,
        6.12288906107524e-11,
        3.351272578637522e-11,
        2.2962634788969947e-11,
        4.653371362021598e-11,
        2.446049868951821e-11,
        2.40636195065083e-10,
        1.641952253901428e-10,
        6.696099079697773e-11,
        1.1850678442534715e-10,
        8.834304768468476e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 8e-6)
    @test isapprox(w2t, w2.weights, rtol = 7e-6)
    @test isapprox(w3t, w3.weights, rtol = 5e-5)
    @test isapprox(w4t, w4.weights, rtol = 4e-6)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.004266104054352623,
        0.04362179139799261,
        0.01996045918300812,
        0.007822769449990673,
        0.06052525732257479,
        2.2211049042813212e-11,
        0.0003959617775312277,
        0.1308926806342499,
        8.0214707797462e-12,
        0.011878930331335047,
        0.20660964132624798,
        4.553198556527535e-13,
        4.833024630837389e-13,
        0.08329480532205666,
        8.342668739909475e-13,
        0.01388809686095509,
        0.08734726339950173,
        0.1921003427001766,
        0.03721320875129444,
        0.10018268745672719,
    ]
    w2t = [
        1.5020949813578004e-10,
        0.0421391305273706,
        0.012527563032353606,
        0.007205814934079214,
        0.09868075566577757,
        9.741638973652938e-12,
        0.0027730094648556166,
        0.10921155052626846,
        3.445736320155924e-11,
        1.164203915903924e-10,
        0.201547016707104,
        6.763729502913016e-13,
        9.270558047499105e-13,
        0.026542577483190048,
        3.5490963354488068e-12,
        0.041731332118682844,
        0.11080124312220684,
        0.17232503241334168,
        0.06800110479895408,
        0.10651386888983401,
    ]
    w3t = [
        1.606486830952879e-10,
        4.33425286216922e-10,
        4.914167760908183e-10,
        2.1900355929256663e-10,
        0.5637275589383445,
        5.2801628372560704e-11,
        0.026029803921426675,
        5.430798568631471e-10,
        3.5555587683254253e-10,
        2.033319218096134e-10,
        3.080027580361297e-9,
        3.3405424117920664e-11,
        2.608752717458878e-11,
        1.0050071792868831e-10,
        2.896816064266277e-11,
        0.1491771576737989,
        0.1297876428029518,
        1.0810198479858125e-9,
        0.13127782917679792,
        6.774072933539338e-10,
    ]
    w4t = [
        6.184817805823537e-10,
        9.270209378838167e-10,
        1.6257277125312362e-9,
        1.2702030965791546e-9,
        0.8503448667003264,
        8.23423005783669e-10,
        0.1496551152996521,
        1.881202141055004e-11,
        1.15945257112404e-9,
        1.743614404976479e-10,
        5.6593301833950176e-11,
        7.768388557303726e-10,
        1.097225242553106e-9,
        4.407276997444489e-10,
        1.0935450089784614e-9,
        3.753052330518356e-9,
        2.0533520728096807e-9,
        1.196379064761205e-10,
        1.387191534215606e-9,
        6.043752858965341e-10,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-7)
    @test isapprox(w2t, w2.weights, rtol = 5e-2)
    @test isapprox(w3t, w3.weights, rtol = 7e-6)
    @test isapprox(w4t, w4.weights, rtol = 1e-9)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.004266104054352623,
        0.04362179139799261,
        0.01996045918300812,
        0.007822769449990673,
        0.06052525732257479,
        2.2211049042813212e-11,
        0.0003959617775312277,
        0.1308926806342499,
        8.0214707797462e-12,
        0.011878930331335047,
        0.20660964132624798,
        4.553198556527535e-13,
        4.833024630837389e-13,
        0.08329480532205666,
        8.342668739909475e-13,
        0.01388809686095509,
        0.08734726339950173,
        0.1921003427001766,
        0.03721320875129444,
        0.10018268745672719,
    ]
    w2t = [
        3.9758945638828316e-8,
        0.0376560628920269,
        0.015519756718506642,
        0.008042529572223543,
        0.09912648486302245,
        3.127619052692243e-9,
        0.0025437761299397415,
        0.10498232103852406,
        8.032922297643308e-9,
        2.660571486085021e-8,
        0.20306948036978265,
        8.527079874190427e-10,
        9.311212738559772e-10,
        0.03560798206661041,
        1.562947699552061e-9,
        0.037395429550472245,
        0.10532378281471647,
        0.17844869316327228,
        0.0626002838636233,
        0.10968333608530062,
    ]
    w3t = [
        7.490867782216147e-10,
        1.8855025502105734e-9,
        1.9987250238986213e-9,
        1.0870097993270504e-9,
        0.6065756467399331,
        2.518655757482574e-10,
        0.028286974436350563,
        1.6661437017315185e-9,
        1.9332212274649863e-9,
        8.876017091609831e-10,
        2.782038906889727e-9,
        1.693733743881779e-10,
        1.285023286419838e-10,
        4.575371776460016e-10,
        1.4137608514991102e-10,
        0.14977668747247325,
        0.10824989463168377,
        2.449254990648767e-9,
        0.10711077778574643,
        2.34657370714424e-9,
    ]
    w4t = [
        3.1468223975377797e-12,
        3.5250641086111997e-12,
        4.381800454354387e-12,
        3.950305656122614e-12,
        0.8533951450752805,
        1.3837049369962544e-12,
        0.14660485487008543,
        2.399398496082109e-12,
        3.818717774652294e-12,
        2.5962792710869566e-12,
        2.3056714649405635e-12,
        1.4357889919420828e-12,
        1.0232182665762994e-12,
        1.84091013461997e-12,
        1.0396824675443122e-12,
        7.143569876525336e-12,
        4.908030010553304e-12,
        2.5261705356126976e-12,
        4.087883179992093e-12,
        3.120898690286033e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 7e-6)
    @test isapprox(w2t, w2.weights, rtol = 2e-3)
    @test isapprox(w3t, w3.weights, rtol = 3e-4)
    @test isapprox(w4t, w4.weights, rtol = 4e-6)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :FLPM
    kelly = :none
    portfolio.flpm_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.flpm_u = r1 + 0.011 * r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 2e-1)
    @test isapprox(r5, r1, rtol = 5e-3)
    @test isapprox(m5, m1, rtol = 2e-1)

    portfolio.flpm_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 2e-1)
    @test isapprox(r6, r2, rtol = 9e-3)
    @test isapprox(m6, m2, rtol = 8e-2)

    portfolio.flpm_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 5e-2)
    @test isapprox(r7, r3, rtol = 2e-2)
    @test isapprox(m7, m3, rtol = 2e-2)

    portfolio.flpm_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 9e-3)
    @test isapprox(r8, r4, rtol = 6e-3)
    @test isapprox(m8, m4, rtol = 2e-4)

    obj = :sharpe
    portfolio.flpm_u = r4
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w3.weights, rtol = 3e-5)
    @test isapprox(r13, r3, rtol = 9e-6)
    @test isapprox(m13, m3, rtol = 8e-6)
end

@testset "SLPM" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SLPM

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        9.138109505189105e-11,
        0.055245353085412084,
        8.548445384624783e-11,
        0.004319553592236582,
        0.03359642416337583,
        8.378425560032282e-11,
        1.6036761190861498e-11,
        0.12478083268457807,
        9.065337635653628e-12,
        1.3791914346482854e-11,
        0.30056466866131276,
        6.700098222018098e-12,
        1.687155349931867e-11,
        0.1066185631354502,
        0.0031241562747808993,
        0.02139160661593031,
        0.0035938916972850965,
        0.22965140938674222,
        2.688873117174822e-11,
        0.11711354035289176,
    ]
    w2t = [
        7.589286936547091e-9,
        0.055105438811854056,
        9.27689322946418e-9,
        0.0004241273095529266,
        0.06590361772912803,
        4.0406427775343055e-9,
        2.0963884922493578e-9,
        0.1192860647617436,
        5.404873478915994e-9,
        5.15467517747711e-9,
        0.2937398188433721,
        1.7390886190266594e-9,
        1.0538977745183755e-9,
        0.07604687597489113,
        5.676372034531971e-9,
        0.034524956954358185,
        0.028234972794452463,
        0.22320701828445685,
        1.1102429255195782e-8,
        0.10352705540164307,
    ]
    w3t = [
        4.921967445976419e-11,
        2.733103469410024e-10,
        2.3233491647702964e-10,
        1.088557862068793e-10,
        0.6654357285499267,
        2.8856578236406863e-10,
        0.03807168696789054,
        1.3112053419429828e-10,
        2.8464451395291793e-10,
        9.617648092829944e-12,
        1.9987083132997106e-10,
        2.5117687235818757e-10,
        3.40504925789857e-10,
        1.2859129757062226e-10,
        3.816496562095631e-10,
        0.1745910033744489,
        0.10411381498296386,
        2.3739824514392557e-10,
        0.01778776301178969,
        1.9611942365722475e-10,
    ]
    w4t = [
        9.28763383151679e-11,
        1.0307866727010052e-10,
        1.4940823317834682e-10,
        1.2336714694005373e-10,
        4.0216604451367195e-9,
        3.433828145899989e-11,
        0.9999999943713728,
        6.346479442018532e-11,
        1.0727008867463966e-10,
        6.582256693437888e-11,
        6.12288906107524e-11,
        3.351272578637522e-11,
        2.2962634788969947e-11,
        4.653371362021598e-11,
        2.446049868951821e-11,
        2.40636195065083e-10,
        1.641952253901428e-10,
        6.696099079697773e-11,
        1.1850678442534715e-10,
        8.834304768468476e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 9e-6)
    @test isapprox(w2t, w2.weights, rtol = 2e-5)
    @test isapprox(w3t, w3.weights, rtol = 3e-5)
    @test isapprox(w4t, w4.weights, rtol = 2e-6)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        9.138109505189105e-11,
        0.055245353085412084,
        8.548445384624783e-11,
        0.004319553592236582,
        0.03359642416337583,
        8.378425560032282e-11,
        1.6036761190861498e-11,
        0.12478083268457807,
        9.065337635653628e-12,
        1.3791914346482854e-11,
        0.30056466866131276,
        6.700098222018098e-12,
        1.687155349931867e-11,
        0.1066185631354502,
        0.0031241562747808993,
        0.02139160661593031,
        0.0035938916972850965,
        0.22965140938674222,
        2.688873117174822e-11,
        0.11711354035289176,
    ]
    w2t = [
        1.1217469492804112e-8,
        0.05510500183386468,
        1.3708966499934479e-8,
        0.0004258041903650659,
        0.06590308899755908,
        5.982135388290026e-9,
        3.111166205066013e-9,
        0.11928779945373989,
        7.993028202512215e-9,
        7.619399766205237e-9,
        0.2937376390526622,
        2.5836272573005613e-9,
        1.571380287287184e-9,
        0.07604715013694095,
        8.405638190032012e-9,
        0.03452463124376386,
        0.028234980907126332,
        0.2232067763383854,
        1.6400904899004892e-8,
        0.10352704925187628,
    ]
    w3t = [
        2.7748995900742376e-7,
        8.563034031076425e-7,
        5.264979855794242e-7,
        3.2340184210029766e-7,
        0.5740477192048896,
        9.095441735504873e-8,
        0.02956314052921492,
        1.0983809357005622e-6,
        4.7369722473197326e-7,
        2.7631008447176374e-7,
        2.410281671954967e-6,
        7.045567027969547e-8,
        5.1055153184943326e-8,
        1.9557701676457745e-7,
        5.301975869191055e-8,
        0.1672097816722272,
        0.15901175060124068,
        1.580487156894848e-6,
        0.07015860723412552,
        7.168460221700648e-7,
    ]
    w4t = [
        7.210672121923212e-11,
        1.1766571471107745e-10,
        2.1258896130845384e-10,
        1.6536735519800757e-10,
        0.8503141906370602,
        1.2734686395995448e-10,
        0.14968580698966108,
        1.9862227077997186e-12,
        1.512850995708011e-10,
        1.851547657596133e-11,
        1.2072113150982461e-11,
        1.1803803828728927e-10,
        1.5679984702030818e-10,
        6.956618814182257e-11,
        1.6100350800794473e-10,
        4.425511340711267e-10,
        2.740530752335572e-10,
        9.565221142562228e-12,
        1.8545932133088304e-10,
        7.730789424008123e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 6e-6)
    @test isapprox(w2t, w2.weights, rtol = 2e-3)
    @test isapprox(w3t, w3.weights, rtol = 2e-3)
    @test isapprox(w4t, w4.weights, rtol = 1e-9)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        9.138109505189105e-11,
        0.055245353085412084,
        8.548445384624783e-11,
        0.004319553592236582,
        0.03359642416337583,
        8.378425560032282e-11,
        1.6036761190861498e-11,
        0.12478083268457807,
        9.065337635653628e-12,
        1.3791914346482854e-11,
        0.30056466866131276,
        6.700098222018098e-12,
        1.687155349931867e-11,
        0.1066185631354502,
        0.0031241562747808993,
        0.02139160661593031,
        0.0035938916972850965,
        0.22965140938674222,
        2.688873117174822e-11,
        0.11711354035289176,
    ]
    w2t = [
        1.594968866042584e-8,
        0.05498026161283858,
        1.954828168483134e-8,
        0.0006322184839422863,
        0.06548858547308255,
        8.51603223790841e-9,
        4.396272436354527e-9,
        0.11941660503313882,
        1.1249354716013038e-8,
        1.0818609014274188e-8,
        0.29378536236689384,
        3.650086968537561e-9,
        2.225636123849807e-9,
        0.07641741132563451,
        1.1883135204773722e-8,
        0.034350159607906515,
        0.028114075779567618,
        0.22310115719229656,
        2.2967061368063704e-8,
        0.10371405192054035,
    ]
    w3t = [
        1.1423434001363644e-8,
        2.9635101343562188e-8,
        2.156966461996186e-8,
        1.3491131662901221e-8,
        0.6119473289581964,
        3.8201388400754895e-9,
        0.036135852079736255,
        2.804149446416773e-8,
        2.2483883777542024e-8,
        1.1226355242731472e-8,
        3.778832322484641e-8,
        3.057918368127965e-9,
        2.1962671981757643e-9,
        7.772231450997719e-9,
        2.2766291293994816e-9,
        0.1695316393366309,
        0.13611115441123306,
        3.668545348543196e-8,
        0.046273769055923876,
        2.4690252865072435e-8,
    ]
    w4t = [
        3.1468223975377797e-12,
        3.5250641086111997e-12,
        4.381800454354387e-12,
        3.950305656122614e-12,
        0.8533951450752805,
        1.3837049369962544e-12,
        0.14660485487008543,
        2.399398496082109e-12,
        3.818717774652294e-12,
        2.5962792710869566e-12,
        2.3056714649405635e-12,
        1.4357889919420828e-12,
        1.0232182665762994e-12,
        1.84091013461997e-12,
        1.0396824675443122e-12,
        7.143569876525336e-12,
        4.908030010553304e-12,
        2.5261705356126976e-12,
        4.087883179992093e-12,
        3.120898690286033e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 2e-5)
    @test isapprox(w2t, w2.weights, rtol = 3e-6)
    @test isapprox(w3t, w3.weights, rtol = 6e-4)
    @test isapprox(w4t, w4.weights, rtol = 9e-6)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SLPM
    kelly = :none
    portfolio.slpm_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.slpm_u = r1 + 0.01 * r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 2e-1)
    @test isapprox(r5, r1, rtol = 4e-3)
    @test isapprox(m5, m1, rtol = 2e-1)

    portfolio.slpm_u = r2 + 0.01 * r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 4e-2)
    @test isapprox(r6, r2, rtol = 2e-3)
    @test isapprox(m6, m2, rtol = 5e-2)

    portfolio.slpm_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 2e-2)
    @test isapprox(r7, r3, rtol = 7e-3)
    @test isapprox(m7, m3, rtol = 7e-3)

    portfolio.slpm_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 5e-3)
    @test isapprox(r8, r4, rtol = 3e-3)
    @test isapprox(m8, m4, rtol = 8e-5)

    portfolio.slpm_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.slpm_u = r4
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w3.weights, rtol = 5e-6)
    @test isapprox(r13, r3, rtol = 2e-6)
    @test isapprox(m13, m3, rtol = 2e-6)
end

@testset "WR" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :WR

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        2.299169995048075e-10,
        0.22119717233703096,
        3.304631314138549e-10,
        2.49134522827939e-10,
        1.1517350974415458e-9,
        3.3894185964710075e-10,
        0.029185393977558677,
        3.739104955455042e-10,
        3.619404544899459e-10,
        4.3638800884468217e-10,
        0.5455165175366683,
        8.865127664121923e-11,
        1.9098648024985617e-10,
        1.379961037937106e-10,
        7.01835468857822e-8,
        3.1328574613792696e-10,
        7.170765419194737e-10,
        1.5517617050844686e-10,
        4.14727081385081e-10,
        0.20410084047486526,
    ]
    w2t = [
        9.583446094743587e-11,
        0.22119704440016452,
        1.0508679002135356e-10,
        2.411423032450491e-10,
        2.13651278639432e-10,
        1.221965556272005e-10,
        0.02918541297756664,
        1.3073579560699885e-10,
        8.744115195743876e-11,
        1.2280214770511386e-10,
        0.545516550798052,
        9.486473395012712e-11,
        8.533250495352074e-11,
        6.389397004843065e-11,
        3.0842919579045e-9,
        1.223857887793584e-10,
        2.3512554460145316e-10,
        7.860583135621488e-11,
        9.217986647972617e-11,
        0.2041009868486462,
    ]
    w3t = [
        4.016098737938145e-11,
        1.206768412418525e-10,
        4.9718834104504366e-11,
        7.73010176971938e-11,
        0.37976617145554137,
        1.132829547754181e-11,
        0.1766051383842627,
        2.6928158238052707e-11,
        7.07849614616034e-11,
        0.04075090918204076,
        0.056382275053874234,
        6.1698267294760204e-12,
        2.725918710060152e-11,
        5.701183122805996e-11,
        1.9136183795945957e-11,
        0.15854734462125905,
        0.18794816049028398,
        4.7003010124319816e-11,
        2.2468680514695185e-10,
        3.4572028879869585e-11,
    ]
    w4t = [
        9.827108130590075e-11,
        1.0964043837068321e-10,
        1.646184654607458e-10,
        1.3873014579697693e-10,
        3.621896966144375e-9,
        2.953467429355499e-11,
        0.9999999946948224,
        5.83093587574998e-11,
        1.2099593448934678e-10,
        6.767104891221812e-11,
        5.5517404346776356e-11,
        3.204490613863917e-11,
        2.0075249357988037e-11,
        4.2192124071032316e-11,
        2.1049678873656126e-11,
        2.651315806553456e-10,
        1.778659135468265e-10,
        6.321540734680977e-11,
        1.2913908989560217e-10,
        8.927802357923575e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 4e-7)
    @test isapprox(w2t, w2.weights, rtol = 2e-8)
    @test isapprox(w3t, w3.weights, rtol = 7e-7)
    @test isapprox(w4t, w4.weights, rtol = 3e-7)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        2.299169995048075e-10,
        0.22119717233703096,
        3.304631314138549e-10,
        2.49134522827939e-10,
        1.1517350974415458e-9,
        3.3894185964710075e-10,
        0.029185393977558677,
        3.739104955455042e-10,
        3.619404544899459e-10,
        4.3638800884468217e-10,
        0.5455165175366683,
        8.865127664121923e-11,
        1.9098648024985617e-10,
        1.379961037937106e-10,
        7.01835468857822e-8,
        3.1328574613792696e-10,
        7.170765419194737e-10,
        1.5517617050844686e-10,
        4.14727081385081e-10,
        0.20410084047486526,
    ]
    w2t = [
        1.355691953015e-10,
        0.22119704213263816,
        1.1753592552855614e-10,
        7.727908935776536e-11,
        3.2444994156241663e-10,
        1.4937596347927887e-10,
        0.029185413371580444,
        1.8866572760513463e-10,
        1.043632265310085e-10,
        1.3970761410073908e-10,
        0.5455165508744271,
        6.43595863702571e-11,
        6.935253338001921e-11,
        1.0285879024240579e-10,
        9.024966816609079e-10,
        2.6788577758359053e-10,
        3.1719897939129475e-10,
        1.6478074245443607e-10,
        1.2302706287591485e-10,
        0.20410099037244753,
    ]
    w3t = [
        7.78415326735601e-7,
        1.8814283118157248e-5,
        6.793043874332238e-7,
        1.4855073745030008e-6,
        0.3798107606922456,
        2.2482932861929203e-7,
        0.17658660312616717,
        1.4236909836945459e-6,
        1.1351771686542962e-6,
        0.040740770760175044,
        0.05635307827963364,
        2.773084267066403e-7,
        1.4573315353892842e-7,
        2.448847108791597e-6,
        2.466546696982123e-7,
        0.158554771361081,
        0.18791776402596735,
        1.30854168915656e-6,
        6.457300057083737e-6,
        8.261619375345363e-7,
    ]
    w4t = [
        3.544336258608111e-11,
        5.235085056521497e-11,
        9.066920017322703e-11,
        7.120910429336545e-11,
        0.8503259395353118,
        4.355473003341134e-11,
        0.14967405946760146,
        2.429037333055176e-12,
        6.509066875201268e-11,
        1.0998030377886223e-11,
        1.7019461020403904e-12,
        4.105726674831167e-11,
        5.880128168567957e-11,
        2.270000322030009e-11,
        5.835103767552056e-11,
        2.0849804861679554e-10,
        1.1412264698687525e-10,
        8.007518238368203e-12,
        7.75188383036564e-11,
        3.4582992051815595e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 4e-7)
    @test isapprox(w2t, w2.weights, rtol = 2e-8)
    @test isapprox(w3t, w3.weights, rtol = 2e-4)
    @test isapprox(w4t, w4.weights, rtol = 1e-9)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        2.299169995048075e-10,
        0.22119717233703096,
        3.304631314138549e-10,
        2.49134522827939e-10,
        1.1517350974415458e-9,
        3.3894185964710075e-10,
        0.029185393977558677,
        3.739104955455042e-10,
        3.619404544899459e-10,
        4.3638800884468217e-10,
        0.5455165175366683,
        8.865127664121923e-11,
        1.9098648024985617e-10,
        1.379961037937106e-10,
        7.01835468857822e-8,
        3.1328574613792696e-10,
        7.170765419194737e-10,
        1.5517617050844686e-10,
        4.14727081385081e-10,
        0.20410084047486526,
    ]
    w2t = [
        1.3939979503239753e-10,
        0.22119705346356366,
        1.7822205517215336e-10,
        5.139516639740984e-10,
        3.595661533249619e-9,
        2.2806269648873606e-10,
        0.029185410211065705,
        4.049592390101357e-10,
        1.7842133788103036e-10,
        3.2013307132328874e-10,
        0.5455165550141776,
        6.251295783219357e-11,
        1.4706419593984068e-10,
        9.906637296394084e-11,
        2.452814195501395e-9,
        3.1907535178879246e-10,
        9.235452017857123e-10,
        1.668489818891145e-10,
        2.624700832228736e-10,
        0.2041009713189841,
    ]
    w3t = [
        1.9829182115492817e-10,
        6.620856343474973e-10,
        1.5786279568215212e-10,
        3.1883061547143337e-10,
        0.37976617050877287,
        5.563110615358895e-11,
        0.17660513696675353,
        3.1906755895221523e-10,
        2.8586380930272815e-10,
        0.040750907805173445,
        0.056382272517820534,
        5.867118305299877e-11,
        3.5930256458744575e-11,
        2.642012788113992e-10,
        5.80493968369723e-11,
        0.15854734516583013,
        0.1879481629272116,
        3.586456944045736e-10,
        1.199359003210698e-9,
        1.3594766555055647e-10,
    ]
    w4t = [
        1.8383768279104635e-12,
        2.4106928595561774e-12,
        3.3203632301806884e-12,
        2.8471058943700408e-12,
        0.8533951445404722,
        8.371730308870157e-13,
        0.1466048554193493,
        1.5516609593137177e-12,
        2.92278040604675e-12,
        1.8141944304401937e-12,
        1.4970743063295334e-12,
        9.529909394924647e-13,
        8.477064774612736e-13,
        1.1519987691355922e-12,
        8.186986956350651e-13,
        6.355895171362846e-12,
        3.997573434355696e-12,
        1.6236415696028493e-12,
        3.1920674638044346e-12,
        2.198590587365655e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 4e-7)
    @test isapprox(w2t, w2.weights, rtol = 5e-8)
    @test isapprox(w3t, w3.weights, rtol = 6e-8)
    @test isapprox(w4t, w4.weights, rtol = 2e-6)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :WR
    kelly = :none
    portfolio.wr_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.wr_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 1e-8)
    @test isapprox(r5, r1, rtol = 3e-10)
    @test isapprox(m5, m1, rtol = 6e-9)

    portfolio.wr_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 5e-9)
    @test isapprox(r6, r2, rtol = 2e-10)
    @test isapprox(m6, m2, rtol = 2e-9)

    portfolio.wr_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 5e-7)
    @test isapprox(r7, r3, rtol = 2e-8)
    @test isapprox(m7, m3, rtol = 9e-8)

    portfolio.wr_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 2e-6)
    @test isapprox(r8, r4, rtol = 9e-7)
    @test isapprox(m8, m4, rtol = 5e-8)

    portfolio.wr_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.wr_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 3e-8)
    @test isapprox(r13, r1, rtol = 5e-10)
    @test isapprox(m13, m1, rtol = 4e-8)
end

@testset "CVaR" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :CVaR

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        4.984670308683193e-10,
        0.04242034175167885,
        5.443812201102558e-10,
        1.857542231587736e-9,
        0.007574034145325163,
        5.725600230623837e-10,
        8.050538716410491e-11,
        0.09464947444629536,
        2.453320660179745e-10,
        3.3387780248201525e-10,
        0.30401099066477505,
        2.874880359257968e-10,
        1.1824085877613384e-10,
        0.06564165184237246,
        5.653590393764593e-10,
        0.02937161385076713,
        7.187315887659594e-10,
        0.3663101417565558,
        3.2675445166618107e-10,
        0.0900217453929905,
    ]
    w2t = [
        3.151157276285723e-11,
        0.03635918161683666,
        3.115629703338205e-11,
        1.0285374918082521e-10,
        0.017177338405401418,
        2.9719391228045804e-11,
        4.913679379539862e-12,
        0.09164254968541638,
        1.6417788787625058e-11,
        1.8892533770526228e-11,
        0.32258424651449746,
        1.528632516851912e-11,
        8.602362714552705e-12,
        0.03955073264831513,
        2.9347804668494516e-11,
        0.030919378629983334,
        3.8665403214182784e-11,
        0.3733298485763679,
        2.0793933648151133e-11,
        0.08843672357502094,
    ]
    w3t = [
        2.3870744145601207e-11,
        3.952709896636753e-11,
        5.4587782836445974e-11,
        2.431903187468614e-11,
        0.5628454331006074,
        4.61636266665396e-11,
        0.04434184495213232,
        7.356595545081174e-11,
        4.612575829495798e-11,
        4.851253596890643e-12,
        5.3750532440735297e-11,
        3.967725270224657e-11,
        6.025247571792753e-11,
        1.732309707791013e-11,
        6.55468426207033e-11,
        0.20959173574034917,
        0.18322098528904368,
        8.504438762386204e-11,
        2.224424232528151e-10,
        6.081888290358941e-11,
    ]
    w4t = [
        9.208673838854866e-11,
        1.0225882038030698e-10,
        1.484825670449222e-10,
        1.2241022434064553e-10,
        4.01658369493568e-9,
        3.397775270719105e-11,
        0.9999999943884567,
        6.290776796228114e-11,
        1.0631402203731418e-10,
        6.51475129101419e-11,
        6.069064893027897e-11,
        3.309315425607115e-11,
        2.268357399224573e-11,
        4.606368574296622e-11,
        2.418259317356735e-11,
        2.397495327413968e-10,
        1.6331073458482453e-10,
        6.636805651527622e-11,
        1.1760944015943576e-10,
        8.762288424618493e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 2e-7)
    @test isapprox(w2t, w2.weights, rtol = 4e-7)
    @test isapprox(w3t, w3.weights, rtol = 4e-7)
    @test isapprox(w4t, w4.weights, rtol = 6e-7)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        4.984670308683193e-10,
        0.04242034175167885,
        5.443812201102558e-10,
        1.857542231587736e-9,
        0.007574034145325163,
        5.725600230623837e-10,
        8.050538716410491e-11,
        0.09464947444629536,
        2.453320660179745e-10,
        3.3387780248201525e-10,
        0.30401099066477505,
        2.874880359257968e-10,
        1.1824085877613384e-10,
        0.06564165184237246,
        5.653590393764593e-10,
        0.02937161385076713,
        7.187315887659594e-10,
        0.3663101417565558,
        3.2675445166618107e-10,
        0.0900217453929905,
    ]
    w2t = [
        2.765840162726268e-10,
        0.03635927452310441,
        2.7975037934076965e-10,
        9.153610605825448e-10,
        0.017177253736804083,
        2.596833592363724e-10,
        4.808853894415055e-11,
        0.09164244062560582,
        1.5017881023528405e-10,
        1.68995171508869e-10,
        0.32258413959224763,
        1.4597455130549495e-10,
        8.127205027947128e-11,
        0.03955087830861623,
        2.420061355882148e-10,
        0.03091938054077641,
        3.3744822453428145e-10,
        0.37332988807423734,
        1.8380286326115488e-10,
        0.08843674150946294,
    ]
    w3t = [
        9.646219575580272e-8,
        1.447093444158025e-7,
        1.3672159800596605e-7,
        9.302895439017001e-8,
        0.5593106889014595,
        3.04426969294529e-8,
        0.029475704190340596,
        2.775029811546195e-6,
        1.1167258344398405e-7,
        6.714959232474424e-8,
        3.706824867365048e-7,
        2.6720209265567412e-8,
        1.7977821893175844e-8,
        5.726533316779283e-8,
        1.8608547203924263e-8,
        0.20296785636972903,
        0.20823925419479142,
        1.3578830519249017e-6,
        8.953961484003773e-7,
        2.9659330408223086e-7,
    ]
    w4t = [
        1.622469183226498e-11,
        2.4305238304407725e-11,
        4.2602092235373706e-11,
        3.3294448859143716e-11,
        0.8503255339883252,
        2.1529386962576576e-11,
        0.14967446554010286,
        5.218848519661671e-13,
        3.039576597225923e-11,
        4.597138866373511e-12,
        1.4530600157725738e-12,
        2.0309475635126493e-11,
        2.8702447567459835e-11,
        1.1510239158196856e-11,
        2.8604440244539228e-11,
        9.834466751452002e-11,
        5.3801352073183567e-11,
        3.1622247932315463e-12,
        3.635793611109776e-11,
        1.585529029013495e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 2e-7)
    @test isapprox(w2t, w2.weights, rtol = 4e-7)
    @test isapprox(w3t, w3.weights, rtol = 1e-5)
    @test isapprox(w4t, w4.weights, rtol = 1e-9)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        4.984670308683193e-10,
        0.04242034175167885,
        5.443812201102558e-10,
        1.857542231587736e-9,
        0.007574034145325163,
        5.725600230623837e-10,
        8.050538716410491e-11,
        0.09464947444629536,
        2.453320660179745e-10,
        3.3387780248201525e-10,
        0.30401099066477505,
        2.874880359257968e-10,
        1.1824085877613384e-10,
        0.06564165184237246,
        5.653590393764593e-10,
        0.02937161385076713,
        7.187315887659594e-10,
        0.3663101417565558,
        3.2675445166618107e-10,
        0.0900217453929905,
    ]
    w2t = [
        8.701438286385476e-7,
        0.036603502005786506,
        8.567744395406258e-7,
        3.2289747664115315e-6,
        0.017151739856636265,
        8.05359221215493e-7,
        1.9198945804122145e-7,
        0.09180307686521497,
        4.7350546219051085e-7,
        5.355664881847761e-7,
        0.3213914840431771,
        4.4643125484367346e-7,
        2.5671890885756744e-7,
        0.03990409036260171,
        8.451048176932434e-7,
        0.03090760907912184,
        1.1244017660466777e-6,
        0.37339037474712355,
        5.941774835543455e-7,
        0.0888378938924428,
    ]
    w3t = [
        5.009021802911202e-9,
        7.163024851234713e-9,
        7.860035615080352e-9,
        4.3742892261680865e-9,
        0.5622703152305417,
        1.588186604728419e-9,
        0.041922070149699175,
        1.526373703472945e-8,
        7.098394296517561e-9,
        3.608847313990078e-9,
        1.4236339167641993e-8,
        1.4407130200768537e-9,
        9.497834157815786e-10,
        2.966825564448344e-9,
        9.790714111161174e-10,
        0.20851379425859945,
        0.18729367998762764,
        1.7287695700216645e-8,
        4.017714449302826e-8,
        1.0370422615951058e-8,
    ]
    w4t = [
        3.3525846906015926e-12,
        3.765134432028318e-12,
        4.6888802898788194e-12,
        4.223260903338083e-12,
        0.8533951448214468,
        1.4752202021949855e-12,
        0.14660485512012023,
        2.560490622393456e-12,
        4.087315521649065e-12,
        2.7742087070501858e-12,
        2.460628614316493e-12,
        1.5329438266958805e-12,
        1.0971020465480176e-12,
        1.9635370929512457e-12,
        1.1135945478799249e-12,
        7.670499898544006e-12,
        5.259725392892443e-12,
        2.6954849063410302e-12,
        4.3771342569957706e-12,
        3.3352177778610072e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-7)
    @test isapprox(w2t, w2.weights, rtol = 3e-3)
    @test isapprox(w3t, w3.weights, rtol = 6e-6)
    @test isapprox(w4t, w4.weights, rtol = 3e-5)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :CVaR
    kelly = :none
    portfolio.cvar_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.cvar_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 4e-5)
    @test isapprox(r5, r1, rtol = 5e-8)
    @test isapprox(m5, m1, rtol = 2e-5)

    portfolio.cvar_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 8e-7)
    @test isapprox(r6, r2, rtol = 6e-9)
    @test isapprox(m6, m2, rtol = 5e-7)

    portfolio.cvar_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 3e-6)
    @test isapprox(r7, r3, rtol = 2e-7)
    @test isapprox(m7, m3, rtol = 2e-7)

    portfolio.cvar_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 7e-6)
    @test isapprox(r8, r4, rtol = 4e-6)
    @test isapprox(m8, m4, rtol = 5e-7)

    portfolio.cvar_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.cvar_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 9e-6)
    @test isapprox(r13, r1, rtol = 2e-8)
    @test isapprox(m13, m1, rtol = 3e-6)
end

@testset "EVaR" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :EVaR

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        1.24622464025655e-8,
        0.15679715527362653,
        1.048497386794946e-8,
        0.01670907448717411,
        1.4798667019328108e-8,
        6.543545672295415e-8,
        0.014445817649664383,
        0.15569630453710406,
        9.919544705109654e-9,
        1.0560185519062545e-8,
        0.45245260118328584,
        8.724513956508136e-9,
        1.272408040948579e-8,
        2.1717204037686402e-8,
        0.004799777735798809,
        1.840649500315675e-7,
        0.018416730009352917,
        1.4767587691891797e-7,
        1.2506773526718167e-8,
        0.18068202804952035,
    ]
    w2t = [
        1.4200763617433353e-9,
        0.15138860715914101,
        1.2258560991627827e-9,
        0.02042215518628802,
        1.7997599555287695e-9,
        4.342999730611357e-9,
        0.017225267642953592,
        0.1532371269121663,
        1.1602841168769887e-9,
        1.2713377079843552e-9,
        0.4464526048296655,
        9.576133382331068e-10,
        1.2583330934504498e-9,
        2.2939825329376776e-9,
        1.9162359183712346e-8,
        7.117911652189184e-8,
        0.0276206017079171,
        1.15820579947238e-8,
        1.4786306382879078e-9,
        0.1836535174294612,
    ]
    w3t = [
        1.111455978741029e-10,
        3.5016439007373604e-10,
        1.1120182492250033e-10,
        1.3649993997505915e-10,
        0.5351913584129233,
        2.570400454761057e-11,
        0.13906619419367272,
        1.5745353965685233e-10,
        1.0301816454469565e-10,
        7.742549502931409e-11,
        1.9569223636375934e-10,
        2.5011561885936336e-11,
        1.723772650785039e-11,
        6.132535857953737e-11,
        1.8392918911116602e-11,
        0.18358898402496415,
        0.14215346134309303,
        1.9178179111467891e-10,
        3.4009027745553895e-10,
        1.0320182033601481e-10,
    ]
    w4t = [
        6.781090243430038e-11,
        6.903761187447166e-11,
        7.075324914559998e-11,
        7.061715005174002e-11,
        1.872179644806397e-8,
        4.1916637342359715e-11,
        0.9999999802149482,
        5.847041149716215e-11,
        6.988776330570124e-11,
        6.16240862194193e-11,
        5.736656378517266e-11,
        4.391342070284974e-11,
        3.3073894254734484e-11,
        5.078318556259974e-11,
        3.40878519601141e-11,
        6.638716140689984e-11,
        7.051380411808158e-11,
        6.022278719187515e-11,
        7.029379214728053e-11,
        6.649506650084859e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 4e-4)
    @test isapprox(w2t, w2.weights, rtol = 5e-5)
    @test isapprox(w3t, w3.weights, rtol = 5e-5)
    @test isapprox(w4t, w4.weights, rtol = 1e-6)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        1.24622464025655e-8,
        0.15679715527362653,
        1.048497386794946e-8,
        0.01670907448717411,
        1.4798667019328108e-8,
        6.543545672295415e-8,
        0.014445817649664383,
        0.15569630453710406,
        9.919544705109654e-9,
        1.0560185519062545e-8,
        0.45245260118328584,
        8.724513956508136e-9,
        1.272408040948579e-8,
        2.1717204037686402e-8,
        0.004799777735798809,
        1.840649500315675e-7,
        0.018416730009352917,
        1.4767587691891797e-7,
        1.2506773526718167e-8,
        0.18068202804952035,
    ]
    w2t = [
        5.652972514016703e-10,
        0.15139566407687802,
        5.019543903815286e-10,
        0.020421820696616636,
        7.797387396437127e-10,
        1.2886675812241988e-9,
        0.017223019275779787,
        0.1532364242223127,
        4.4985116085203827e-10,
        4.643582554800252e-10,
        0.44645043767906595,
        3.251049781101099e-10,
        4.022664893866583e-10,
        7.204281890280923e-10,
        5.082810441871677e-9,
        2.0977781344707037e-8,
        0.027621308398247273,
        3.018153743519964e-9,
        5.605143085870899e-10,
        0.18365129051417287,
    ]
    w3t = [
        1.0432508999053578e-6,
        6.7974377687905275e-6,
        9.465168545291885e-7,
        1.3394815651954694e-6,
        0.4887375087227115,
        2.3612894129115242e-7,
        0.1106830264909854,
        3.6219720445749258e-6,
        9.562943610067765e-7,
        8.596773610501789e-7,
        6.607828963588489e-6,
        2.231238782253199e-7,
        1.4870667952843975e-7,
        6.498990619584535e-7,
        1.6848458324307785e-7,
        0.1800152442889253,
        0.2205315750101543,
        3.0747114525595593e-6,
        4.973007433517361e-6,
        9.989653745916951e-7,
    ]
    w4t = [
        4.7727209935228895e-12,
        6.181171969908455e-12,
        9.344480443560326e-12,
        7.728357535987228e-12,
        0.8503259982576274,
        1.7626689635978848e-12,
        0.1496740016447349,
        2.000481938674273e-12,
        7.217384819265894e-12,
        2.7047244092352033e-12,
        1.639183488014504e-12,
        1.609322819630582e-12,
        3.1218349506358897e-12,
        8.931000716375926e-14,
        3.054215463493856e-12,
        1.9703783678003294e-11,
        1.133383705530084e-11,
        2.469187982058449e-12,
        8.231886091736689e-12,
        4.673108903566455e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-4)
    @test isapprox(w2t, w2.weights, rtol = 4e-3)
    @test isapprox(w3t, w3.weights, rtol = 4e-2)
    @test isapprox(w4t, w4.weights, rtol = 1e-5)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        1.24622464025655e-8,
        0.15679715527362653,
        1.048497386794946e-8,
        0.01670907448717411,
        1.4798667019328108e-8,
        6.543545672295415e-8,
        0.014445817649664383,
        0.15569630453710406,
        9.919544705109654e-9,
        1.0560185519062545e-8,
        0.45245260118328584,
        8.724513956508136e-9,
        1.272408040948579e-8,
        2.1717204037686402e-8,
        0.004799777735798809,
        1.840649500315675e-7,
        0.018416730009352917,
        1.4767587691891797e-7,
        1.2506773526718167e-8,
        0.18068202804952035,
    ]
    w2t = [
        1.3198539848010798e-10,
        0.15111275907875793,
        1.137317803921747e-10,
        0.0205870601186122,
        1.661027750387836e-10,
        4.0551441010721075e-10,
        0.017132169991558758,
        0.1532498865018977,
        1.076602782951486e-10,
        1.1801248774456978e-10,
        0.44642647035574057,
        8.905362522494031e-11,
        1.171936705916202e-10,
        2.1410859742865612e-10,
        1.792007131575009e-9,
        6.843758743685425e-9,
        0.027856098899668173,
        1.0826243841652004e-9,
        1.3720644578588537e-10,
        0.18363554373480495,
    ]
    w3t = [
        8.266331433964576e-8,
        3.728699560161654e-7,
        9.269866763349194e-8,
        1.1210657741506878e-7,
        0.5137969965134378,
        1.954341280065966e-8,
        0.11863859546203073,
        1.5287285610914292e-7,
        8.662208998177539e-8,
        6.058504207628976e-8,
        2.0303472094261e-7,
        1.858191729611256e-8,
        1.2842153650699011e-8,
        4.6096211294738755e-8,
        1.3554788074336607e-8,
        0.18154566943344225,
        0.18601670327754108,
        1.785124332178018e-7,
        4.995919815355082e-7,
        8.313742580354917e-8,
    ]
    w4t = [
        2.3421861988228217e-10,
        2.899044590983677e-10,
        4.1922498587769095e-10,
        3.3955841108953745e-10,
        0.8533951283102168,
        7.64424711901825e-11,
        0.1466048672084869,
        1.684001052347552e-10,
        3.1231169466034897e-10,
        1.9193854803114692e-10,
        1.6064323198267341e-10,
        9.374237662728358e-11,
        8.444611594182334e-11,
        1.1738835220686122e-10,
        8.009353439455104e-11,
        7.010195052361501e-10,
        4.539682881066326e-10,
        1.7774061028560572e-10,
        3.3266729454610433e-10,
        2.4758763753935936e-10,
    ]

    @test isapprox(w1t, w1.weights, rtol = 4e-5)
    @test isapprox(w2t, w2.weights, rtol = 8e-4)
    @test isapprox(w3t, w3.weights, rtol = 1e-4)
    @test isapprox(w4t, w4.weights, rtol = 9e-7)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :EVaR
    kelly = :none
    portfolio.evar_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.evar_u = r1 + 0.000001 * r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 2.1e-2)
    @test isapprox(r5, r1, rtol = 2e-4)
    @test isapprox(m5, m1, rtol = 4e-2)

    portfolio.evar_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 2e-3)
    @test isapprox(r6, r2, rtol = 2e-5)
    @test isapprox(m6, m2, rtol = 2e-3)

    portfolio.evar_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 3e-5)
    @test isapprox(r7, r3, rtol = 3e-6)
    @test isapprox(m7, m3, rtol = 3e-6)

    portfolio.evar_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 4e-5)
    @test isapprox(r8, r4, rtol = 3e-5)
    @test isapprox(m8, m4, rtol = 7e-7)

    portfolio.evar_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.evar_u = r1 + 0.00001 * r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 6e-3)
    @test isapprox(r13, r1, rtol = 8e-6)
    @test isapprox(m13, m1, rtol = 1e-2)
end

@testset "RVaR" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :RVaR

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        5.110313791920887e-9,
        0.21106973276046728,
        5.816885761794202e-9,
        3.1619885314374784e-8,
        4.233897593091962e-8,
        7.832504135318656e-9,
        0.03667155885745096,
        0.06103394794774874,
        5.1409103867146504e-9,
        4.405837887175969e-9,
        0.4935366974899362,
        2.277873418892271e-9,
        3.927678260007868e-9,
        5.7771732297537545e-9,
        1.104365450810092e-8,
        2.1455221596219762e-8,
        2.3270841902788066e-8,
        1.0297208082916483e-8,
        6.324159838243276e-9,
        0.1976878763052727,
    ]
    w2t = [
        1.3231106960801037e-10,
        0.21292412092919938,
        1.233013705946509e-10,
        2.6649837634539984e-9,
        1.7981557540928736e-10,
        3.277764842692053e-10,
        0.03942243700238616,
        0.05653735578093796,
        9.918328664458085e-11,
        1.2661526273726566e-10,
        0.4901433680101727,
        7.454910034292223e-11,
        1.728263440313629e-10,
        1.3137573410867778e-10,
        3.9528182551563536e-10,
        3.847597662835751e-10,
        1.5232844811457161e-9,
        2.7563523177094984e-10,
        1.237245575497077e-10,
        0.20097271154187996,
    ]
    w3t = [
        6.358499728781665e-11,
        1.841235347236196e-10,
        4.966788385238307e-11,
        9.461030598649592e-11,
        0.5059939228656858,
        1.573466302230962e-11,
        0.1723392795697184,
        1.2688782045196753e-10,
        8.116848640262134e-11,
        2.2210023830740553e-10,
        1.9928452153102552e-8,
        1.579616056461963e-11,
        1.067074593030458e-11,
        6.117113479773812e-11,
        1.5193863328547213e-11,
        0.18247702917137404,
        0.13918974614759666,
        1.585190456635428e-10,
        1.1793093447767731e-9,
        3.863460204643069e-11,
    ]
    w4t = [
        1.0512920811765491e-11,
        1.0399565426893284e-11,
        8.527947992900331e-12,
        9.62788965703129e-12,
        6.330598862038141e-10,
        6.044868532484011e-12,
        0.9999999992199721,
        9.329321873596821e-12,
        1.0168877346399065e-11,
        9.885034779353401e-12,
        9.124284524092252e-12,
        6.436490879244218e-12,
        4.423674823327669e-12,
        7.82094612411599e-12,
        4.6060471079715825e-12,
        2.1948875745183155e-12,
        7.810359590424589e-12,
        9.646291310016587e-12,
        9.939211877739414e-12,
        1.0469429042886405e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 6e-5)
    @test isapprox(w2t, w2.weights, rtol = 6e-4)
    @test isapprox(w3t, w3.weights, rtol = 5e-5)
    @test isapprox(w4t, w4.weights, rtol = 3e-6)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        2.1218320503470277e-7,
        0.21210518906492756,
        3.0428492447646423e-7,
        2.2443040146483448e-7,
        5.566710810421073e-7,
        7.020746213564105e-8,
        0.036666107577511785,
        0.06268827694184259,
        2.2143718108409463e-8,
        2.2923576487353292e-7,
        0.4928081498007071,
        1.0407561286420424e-7,
        1.3531310491866263e-7,
        3.2694536739725996e-7,
        1.5286494919074166e-6,
        5.0839047316380346e-8,
        2.009658915722553e-7,
        4.675954756917541e-8,
        2.645933908596813e-7,
        0.19572799931699936,
    ]
    w2t = [
        1.2125711073092181e-10,
        0.21292730496635828,
        1.1532027812437289e-10,
        2.419219110508641e-9,
        1.8281897628667242e-10,
        3.2671671264636963e-10,
        0.03942303597672992,
        0.05653821913832925,
        9.584075865091072e-11,
        1.1840652443454612e-10,
        0.4901420035126832,
        7.475470296631096e-11,
        1.7349365782305275e-10,
        1.3396643843956138e-10,
        4.129063353295834e-10,
        4.3236383514829846e-10,
        1.6183546670516787e-9,
        2.909552829135583e-10,
        1.1600053207070512e-10,
        0.2009694297735244,
    ]
    w3t = [
        4.4157625742525866e-10,
        5.802061044357666e-9,
        5.649475720757318e-9,
        1.4605343648389533e-10,
        0.4466470126345901,
        2.695715122775934e-9,
        0.14745856420509573,
        8.131980908534112e-9,
        5.105172200929205e-9,
        2.226403914145264e-9,
        0.0549578990270768,
        1.6420957832438588e-9,
        8.891748888942573e-9,
        8.119201625926738e-9,
        7.131433457158359e-10,
        0.1686398996672221,
        0.18229656628774565,
        5.464826639119419e-9,
        9.75661142779142e-10,
        2.1731537703537534e-9,
    ]
    w4t = [
        1.8627122645236273e-11,
        2.408694808141871e-11,
        3.6424621073439797e-11,
        3.0174191690909147e-11,
        0.8503271551723861,
        6.9268523006275565e-12,
        0.1496728444480554,
        7.85567861957879e-12,
        2.8160503255372478e-11,
        1.0658665145342347e-11,
        6.471150254909595e-12,
        6.2160861933501414e-12,
        1.2202245892621786e-11,
        2.988215511968398e-13,
        1.1957133510750498e-11,
        7.544005357222848e-11,
        4.398875603911654e-11,
        9.676675293326075e-12,
        3.2107546650158034e-11,
        1.8285626807746916e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 1e-3)
    @test isapprox(w2t, w2.weights, rtol = 7e-4)
    @test isapprox(w3t, w3.weights, rtol = 1e-7)
    @test isapprox(w4t, w4.weights, rtol = 1e-6)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        1.2822539467571446e-10,
        0.2110469190307352,
        1.3851715384583383e-10,
        8.168665705342977e-10,
        8.494596518852085e-10,
        2.065673661712813e-10,
        0.03667015667962741,
        0.06104556477212127,
        1.0478058935859193e-10,
        1.1206142903654103e-10,
        0.49353434919586164,
        5.33250081734951e-11,
        1.1340882813294869e-10,
        9.575258598843666e-11,
        2.949511773043986e-10,
        4.1290437139256246e-10,
        6.793964363415394e-10,
        2.0311526943057728e-10,
        1.4490103350802484e-10,
        0.1977030059674216,
    ]
    w2t = [
        1.668014543773979e-11,
        0.21281366183578446,
        1.633409236904392e-11,
        4.175757459688923e-10,
        2.7991676196620482e-11,
        4.9243340757382226e-11,
        0.03930178268982399,
        0.05687391441587713,
        1.3882657273908015e-11,
        1.718736777939302e-11,
        0.4901773026808702,
        1.0612873232775134e-11,
        2.797719410212253e-11,
        1.5379008146178975e-11,
        6.671586432121067e-11,
        6.191097963391492e-11,
        2.476282785673929e-10,
        2.7729751445912837e-11,
        1.6846039626009635e-11,
        0.20083333734394923,
    ]
    w3t = [
        0.00020951098404609422,
        0.00026109773624657945,
        0.0003151223550578498,
        0.00027246322955495834,
        0.8060082183534146,
        8.925802345168651e-5,
        0.18941720354558209,
        0.00016691580392084898,
        0.0002840396261436436,
        0.0001728267369028469,
        0.0001617855626528449,
        8.87502485210866e-5,
        6.456853217838862e-5,
        0.0001220414806425407,
        6.599182538793858e-5,
        0.0011634618625357355,
        0.0004182999348165407,
        0.0001798399793284494,
        0.00032822084598869245,
        0.0002103833336264928,
    ]
    w4t = [
        1.3239407421779054e-12,
        1.914626404820604e-12,
        2.7764843961251617e-12,
        2.3242220518644206e-12,
        0.8533951448020407,
        6.193924782972888e-13,
        0.14660485516481803,
        1.1945399584719348e-12,
        2.4565307752464815e-12,
        1.458090210876948e-12,
        1.1552399435439264e-12,
        7.454079492808247e-13,
        7.384998250709401e-13,
        8.697675306376061e-13,
        6.980825386661236e-13,
        5.686843287735052e-12,
        3.4527041414512346e-12,
        1.2452361137006632e-12,
        2.7073478535669685e-12,
        1.7742907192908375e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 1e-5)
    @test isapprox(w2t, w2.weights, rtol = 5e-4)
    @test isapprox(w3t, w3.weights, rtol = 1e-7)
    @test isapprox(w4t, w4.weights, rtol = 4e-6)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :RVaR
    kelly = :none
    portfolio.rvar_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.rvar_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 0.015)
    @test isapprox(r5, r1, rtol = 7e-5)
    @test isapprox(m5, m1, rtol = 0.013)

    portfolio.rvar_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 5e-3)
    @test isapprox(r6, r2, rtol = 6e-5)
    @test isapprox(m6, m2, rtol = 5e-3)

    portfolio.rvar_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 4e-5)
    @test isapprox(r7, r3, rtol = 1e-6)
    @test isapprox(m7, m3, rtol = 8e-7)

    portfolio.rvar_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 4e-4)
    @test isapprox(r8, r4, rtol = 3e-4)
    @test isapprox(m8, m4, rtol = 6e-6)

    portfolio.rvar_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.rvar_u = r1 + 0.000001 * r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 2e-3)
    @test isapprox(r13, r1, rtol = 7e-7)
    @test isapprox(m13, m1, rtol = 2e-3)
end

@testset "MDD" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :MDD

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.03773810030715867,
        3.2851220912549452e-12,
        1.4919168695997023e-12,
        4.834424850506721e-12,
        0.06767028684278731,
        4.4864909341300076e-12,
        3.0033652609202115e-12,
        5.58254256448653e-10,
        2.5108728630585757e-12,
        1.2802047576437716e-11,
        0.49113432463080997,
        3.5523392071870465e-12,
        0.02869987004806063,
        5.466053501940173e-12,
        5.8594194688973876e-12,
        0.1454387863473454,
        0.09520044745137303,
        0.10299334289506569,
        4.345250434894088e-12,
        0.03112484086750765,
    ]
    w2t = [
        0.03773810207553227,
        2.2205566734044313e-12,
        5.915614379456541e-12,
        1.5639518326830707e-12,
        0.06767028634083613,
        7.053018487964507e-12,
        4.855978387873715e-12,
        1.559001904145987e-10,
        3.0700649747458483e-12,
        1.1889614535807893e-11,
        0.49113432400181195,
        6.781392323922151e-12,
        0.02869987011564124,
        1.2980500199289623e-12,
        1.2342144908924804e-12,
        0.14543878696174237,
        0.09520044612049887,
        0.10299334172710228,
        7.330569020466936e-12,
        0.03112484244772168,
    ]
    w3t = [
        3.427812371858652e-13,
        9.817999458992407e-13,
        3.5544666671638125e-11,
        9.117019494481572e-13,
        0.2853954619732876,
        1.217306822762418e-11,
        3.407494722569436e-12,
        0.07964282376792359,
        1.5897999642873108e-12,
        4.217713133415332e-12,
        0.3337054960315391,
        1.0682908999961373e-11,
        1.45919549335211e-11,
        7.349914446019587e-12,
        1.5532345004435624e-11,
        0.30125621811227704,
        2.381441256393713e-12,
        2.9371381492190852e-12,
        3.406533678225089e-13,
        1.9870439982974457e-12,
    ]
    w4t = [
        8.54176069443864e-11,
        9.504756259972373e-11,
        1.1625053653142306e-10,
        1.1032209427965938e-10,
        1.5573659387364335e-7,
        3.19180963159945e-11,
        0.9999998428865456,
        6.174691035525801e-11,
        1.0189446495814366e-10,
        6.782972985917808e-11,
        5.832527564484337e-11,
        3.379349409163981e-11,
        1.8915961238950584e-11,
        4.6144860801606284e-11,
        2.1260164937446047e-11,
        1.547194112580361e-10,
        1.241740334586059e-10,
        6.398640788276505e-11,
        1.0485907533953896e-10,
        8.025491256881099e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 6e-9)
    @test isapprox(w2t, w2.weights, rtol = 6e-5)
    @test isapprox(w3t, w3.weights, rtol = 2e-5)
    @test isapprox(w4t, w4.weights, rtol = 2e-4)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.03773810030715867,
        3.2851220912549452e-12,
        1.4919168695997023e-12,
        4.834424850506721e-12,
        0.06767028684278731,
        4.4864909341300076e-12,
        3.0033652609202115e-12,
        5.58254256448653e-10,
        2.5108728630585757e-12,
        1.2802047576437716e-11,
        0.49113432463080997,
        3.5523392071870465e-12,
        0.02869987004806063,
        5.466053501940173e-12,
        5.8594194688973876e-12,
        0.1454387863473454,
        0.09520044745137303,
        0.10299334289506569,
        4.345250434894088e-12,
        0.03112484086750765,
    ]
    w2t = [
        0.03773810521057835,
        1.8535780158748973e-11,
        2.8814661200070773e-11,
        1.3786323405973329e-11,
        0.06767028541794959,
        5.552299773332654e-11,
        3.6127779234136535e-11,
        1.5038680596091834e-9,
        2.5006409877437125e-11,
        8.786973394354901e-11,
        0.4911343208674391,
        5.209796435434728e-11,
        0.028699869173201276,
        1.169540846710002e-11,
        1.1506458612308341e-11,
        0.1454387886447474,
        0.09520044894837335,
        0.10299333893178049,
        5.278598500276358e-11,
        0.031124840908312686,
    ]
    w3t = [
        8.945474003259725e-6,
        2.4635274273498562e-6,
        0.007136314641981204,
        1.2616116279965748e-6,
        0.2874210754928745,
        7.061707516116819e-7,
        1.3329829580537536e-6,
        0.0684007154368847,
        9.540705069590038e-7,
        1.129626258183107e-6,
        0.34644945460663884,
        5.630481416688294e-7,
        3.845419256704969e-7,
        1.3552104184297183e-6,
        5.143375237706606e-7,
        0.2905569643887298,
        5.731720515404184e-6,
        2.9869355946494527e-6,
        2.1182738265178033e-6,
        5.0279014114979045e-6,
    ]
    w4t = [
        6.478984080411506e-11,
        5.992173958582291e-11,
        5.350580922293195e-11,
        5.160796429988034e-11,
        0.850325056058688,
        8.512532288212798e-11,
        0.14967494278628427,
        7.399010733627036e-11,
        4.986307336663039e-11,
        6.85769538036335e-11,
        7.424407406277766e-11,
        8.247802812528716e-11,
        8.584557881654812e-11,
        7.908063983037005e-11,
        8.90109435470697e-11,
        1.2282989856159104e-11,
        4.0483629444010345e-11,
        6.958419039143785e-11,
        5.146013324576591e-11,
        6.317655172958214e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 5e-9)
    @test isapprox(w2t, w2.weights, rtol = 2e-8)
    @test isapprox(w3t, w3.weights, rtol = 2e-2)
    @test isapprox(w4t, w4.weights, rtol = 9e-6)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.03773810030715867,
        3.2851220912549452e-12,
        1.4919168695997023e-12,
        4.834424850506721e-12,
        0.06767028684278731,
        4.4864909341300076e-12,
        3.0033652609202115e-12,
        5.58254256448653e-10,
        2.5108728630585757e-12,
        1.2802047576437716e-11,
        0.49113432463080997,
        3.5523392071870465e-12,
        0.02869987004806063,
        5.466053501940173e-12,
        5.8594194688973876e-12,
        0.1454387863473454,
        0.09520044745137303,
        0.10299334289506569,
        4.345250434894088e-12,
        0.03112484086750765,
    ]
    w2t = [
        0.03773810516216719,
        2.3133165576865112e-11,
        9.16691635428826e-11,
        1.7102727510595636e-11,
        0.06767028605476284,
        5.7318223311748983e-11,
        7.11146500177375e-11,
        7.24069712380935e-10,
        2.8755175411437956e-11,
        8.909786731439296e-11,
        0.49113432057595974,
        5.3647997448853285e-11,
        0.028699869065732502,
        1.4479459326508248e-11,
        1.2786269068274312e-11,
        0.14543878835225577,
        0.09520044931068246,
        0.10299333951358529,
        6.313578020602817e-11,
        0.031124840718543833,
    ]
    w3t = [
        0.0003124111051024796,
        0.00026567143267758455,
        0.3394109545084788,
        0.00016553087078775058,
        0.38425742527089146,
        7.168831026077838e-5,
        0.023041036206364315,
        0.08931884154081252,
        0.000130138399695877,
        0.00012197975506665112,
        0.00828840515898142,
        6.478098091567688e-5,
        3.735694076453984e-5,
        0.00015074881597069132,
        6.057152264073392e-5,
        0.15313775054984702,
        0.00038562920309542857,
        0.00017596316153319636,
        0.0002885476417541668,
        0.0003145686243588022,
    ]
    w4t = [
        9.271465712891691e-10,
        1.038198544811814e-9,
        1.2900026614212648e-9,
        1.1632183149055187e-9,
        0.8533952372875252,
        4.0800134823591043e-10,
        0.14660474662375575,
        7.065979944750803e-10,
        1.1240629667161824e-9,
        7.645996215776372e-10,
        6.793515260354585e-10,
        4.233129447696972e-10,
        3.025599873545369e-10,
        5.425566272768719e-10,
        3.064781515880674e-10,
        2.101596083975705e-9,
        1.4448155229767694e-9,
        7.440457659819019e-10,
        1.2032198565194631e-9,
        9.189546833512431e-10,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-8)
    @test isapprox(w2t, w2.weights, rtol = 2e-8)
    @test isapprox(w3t, w3.weights, rtol = 6e-4)
    @test isapprox(w4t, w4.weights, rtol = 2e-6)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :MDD
    kelly = :none
    portfolio.mdd_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.mdd_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 2e-6)
    @test isapprox(r5, r1, rtol = 4e-9)
    @test isapprox(m5, m1, rtol = 5e-7)

    portfolio.mdd_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 7e-6)
    @test isapprox(r6, r2, rtol = 6e-9)
    @test isapprox(m6, m2, rtol = 3e-6)

    portfolio.mdd_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 3e-6)
    @test isapprox(r7, r3, rtol = 5e-9)
    @test isapprox(m7, m3, rtol = 2e-7)

    portfolio.mdd_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 2e-5)
    @test isapprox(r8, r4, rtol = 3e-5)
    @test isapprox(m8, m4, rtol = 2e-7)

    portfolio.mdd_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.mdd_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 8e-6)
    @test isapprox(r13, r1, rtol = 2e-7)
    @test isapprox(m13, m1, rtol = 3e-6)
end

@testset "ADD" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :ADD

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        9.988041100720739e-12,
        0.04441408949417829,
        0.15878174563263356,
        7.928413270127653e-13,
        0.0678958156553574,
        4.102152531541772e-12,
        4.1485393274035035e-13,
        0.061989663697334954,
        1.1409739286294732e-12,
        3.6000093244289955e-12,
        0.2102533239868234,
        2.1455611338703486e-11,
        0.001213272005842165,
        0.0172161924319746,
        2.624471211208985e-12,
        0.06906921936777355,
        0.10146092361909866,
        0.02247789861930498,
        0.13036103306680463,
        0.1148668223787549,
    ]
    w2t = [
        1.5446099892212711e-10,
        0.042412180900042416,
        0.15457804760833926,
        5.5913846826882047e-11,
        0.06853006015017853,
        9.235177974336182e-11,
        5.857451045239888e-11,
        0.06477893358550368,
        7.177793482528217e-11,
        9.711810062104678e-11,
        0.207649058874031,
        1.614364122388604e-10,
        0.0007314206795904687,
        0.01852440595336185,
        3.343942075652099e-11,
        0.07144518110838553,
        0.10791275631790125,
        0.016304925913306397,
        0.13059136045738123,
        0.11654166772690559,
    ]
    w3t = [
        6.238491592821614e-13,
        2.277630519322571e-11,
        0.2230288186138038,
        3.948253593243766e-12,
        0.2313132569360856,
        8.475482928270558e-12,
        0.002413064848618368,
        0.06938282616413582,
        2.5142873035083564e-12,
        1.607363098012637e-14,
        0.09196993154792613,
        7.250264751537828e-12,
        1.0766945988821766e-11,
        1.5863047806990716e-12,
        1.1377276065974522e-11,
        0.14134964304156383,
        0.14559334574952526,
        4.113240719038557e-12,
        0.09494911298861203,
        3.628096586460391e-11,
    ]
    w4t = [
        8.672282010741108e-11,
        9.644043194477562e-11,
        1.180159068835407e-10,
        1.1192707419294832e-10,
        1.5647357102567998e-7,
        3.241770015144735e-11,
        0.9999998421289982,
        6.265705897336817e-11,
        1.0338818358775372e-10,
        6.884471506342884e-11,
        5.920221620402874e-11,
        3.433220861626752e-11,
        1.923304002244607e-11,
        4.6845029077112315e-11,
        2.1603869155768523e-11,
        1.5694603370822014e-10,
        1.2601965843741132e-10,
        6.495440053411419e-11,
        1.0640550644049826e-10,
        8.147499254184257e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 2e-3)
    @test isapprox(w2t, w2.weights, rtol = 3e-6)
    @test isapprox(w3t, w3.weights, rtol = 1e-6)
    @test isapprox(w4t, w4.weights, rtol = 9e-5)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        9.988041100720739e-12,
        0.04441408949417829,
        0.15878174563263356,
        7.928413270127653e-13,
        0.0678958156553574,
        4.102152531541772e-12,
        4.1485393274035035e-13,
        0.061989663697334954,
        1.1409739286294732e-12,
        3.6000093244289955e-12,
        0.2102533239868234,
        2.1455611338703486e-11,
        0.001213272005842165,
        0.0172161924319746,
        2.624471211208985e-12,
        0.06906921936777355,
        0.10146092361909866,
        0.02247789861930498,
        0.13036103306680463,
        0.1148668223787549,
    ]
    w2t = [
        1.8195655962152877e-11,
        0.042412179657305274,
        0.15457804887042095,
        3.873081729454078e-12,
        0.06853006378706016,
        8.877111639661211e-12,
        3.973300269265778e-12,
        0.06477894076776304,
        6.413808905657474e-12,
        1.07069777315437e-11,
        0.20764904657850702,
        2.1290165338402962e-11,
        0.0007314272425457178,
        0.01852440950127287,
        1.0591944928681595e-12,
        0.07144518476891717,
        0.10791276302941676,
        0.016304911536672118,
        0.13059135865623986,
        0.11654166552948982,
    ]
    w3t = [
        1.5357839424761593e-11,
        1.7593453333599103e-10,
        0.23659138365642227,
        1.8165164022556647e-11,
        0.20617017151968062,
        6.456258003318886e-12,
        0.0015585627311090185,
        0.07372212145839394,
        1.643234943112264e-11,
        1.7524780015086923e-11,
        0.13344049053543225,
        5.84410624420094e-12,
        3.5893919929794046e-12,
        2.5053019703859442e-11,
        4.690183022753835e-12,
        0.13268931284856028,
        0.14391004798585,
        3.694432369620535e-11,
        0.07191790877376798,
        1.6479165475081457e-10,
    ]
    w4t = [
        1.4931842447566132e-10,
        1.3636793476969533e-10,
        1.1570576737029381e-10,
        1.171103156847248e-10,
        0.8503180435480688,
        1.9404927456591877e-10,
        0.14968195383559865,
        1.5836233386798634e-10,
        1.2996888849801928e-10,
        1.6075215168415495e-10,
        1.613403842778533e-10,
        1.8951573102298767e-10,
        1.9530890387574737e-10,
        1.7614908237448046e-10,
        2.0134010899821042e-10,
        2.6598510712382002e-11,
        8.403671952460737e-11,
        1.5833373293071834e-10,
        1.1634394601272993e-10,
        1.457304146007259e-10,
    ]

    @test isapprox(w1t, w1.weights, rtol = 2e-5)
    @test isapprox(w2t, w2.weights, rtol = 4e-8)
    @test isapprox(w3t, w3.weights, rtol = 8e-8)
    @test isapprox(w4t, w4.weights, rtol = 1e-9)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        9.988041100720739e-12,
        0.04441408949417829,
        0.15878174563263356,
        7.928413270127653e-13,
        0.0678958156553574,
        4.102152531541772e-12,
        4.1485393274035035e-13,
        0.061989663697334954,
        1.1409739286294732e-12,
        3.6000093244289955e-12,
        0.2102533239868234,
        2.1455611338703486e-11,
        0.001213272005842165,
        0.0172161924319746,
        2.624471211208985e-12,
        0.06906921936777355,
        0.10146092361909866,
        0.02247789861930498,
        0.13036103306680463,
        0.1148668223787549,
    ]
    w2t = [
        3.194797273904451e-10,
        0.04241217998134255,
        0.1545780364555072,
        1.1597266832547837e-10,
        0.06853006169496553,
        1.8771719813259987e-10,
        1.2928104100908042e-10,
        0.06477893293313822,
        1.491269425842249e-10,
        2.0513502516416386e-10,
        0.20764905660281938,
        3.3922864289609965e-10,
        0.0007314176413779752,
        0.01852440449239082,
        7.170259357328038e-11,
        0.07144516379819858,
        0.10791275308784107,
        0.016304949431299714,
        0.13059138300377174,
        0.11654165935970348,
    ]
    w3t = [
        1.752388363154156e-10,
        1.0675666231423542e-9,
        0.21694956557791933,
        2.6599872671476823e-10,
        0.23650509743956757,
        7.442988780904606e-11,
        0.00258698957621783,
        0.06889373020445291,
        2.1120244940263085e-10,
        2.1112812517024935e-10,
        0.0923960027474717,
        7.036543346526426e-11,
        4.2268683073394135e-11,
        2.3582248892490674e-10,
        5.610576707082707e-11,
        0.1408946650150609,
        0.14605097652517482,
        4.306514342284344e-10,
        0.09572296739157563,
        2.6817809643726806e-9,
    ]
    w4t = [
        2.310126084138472e-12,
        2.586678918492548e-12,
        3.214330757417172e-12,
        2.8982505552249603e-12,
        0.8533951447999877,
        1.0156678448788097e-12,
        0.14660485515993127,
        1.7609342271077702e-12,
        2.8011697012624134e-12,
        1.9050005196521198e-12,
        1.692126266065065e-12,
        1.0536410096604785e-12,
        7.503287564215962e-13,
        1.3511646949874367e-12,
        7.625337619609635e-13,
        5.2374507013493405e-12,
        3.5994660020952347e-12,
        1.8540071121672065e-12,
        2.9984171011666522e-12,
        2.2898864456396156e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 6e-6)
    @test isapprox(w2t, w2.weights, rtol = 3e-7)
    @test isapprox(w3t, w3.weights, rtol = 4e-5)
    @test isapprox(w4t, w4.weights, rtol = 4e-6)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :ADD
    kelly = :none
    portfolio.add_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.add_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 3e-4)
    @test isapprox(r5, r1, rtol = 2e-7)
    @test isapprox(m5, m1, rtol = 5e-5)

    portfolio.add_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 2e-6)
    @test isapprox(r6, r2, rtol = 5e-9)
    @test isapprox(m6, m2, rtol = 4e-7)

    portfolio.add_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 1e-5)
    @test isapprox(r7, r3, rtol = 3e-6)
    @test isapprox(m7, m3, rtol = 3e-6)

    portfolio.add_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 1e-4)
    @test isapprox(r8, r4, rtol = 2e-4)
    @test isapprox(m8, m4, rtol = 2e-6)

    portfolio.add_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.add_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 2e-4)
    @test isapprox(r13, r1, rtol = 2e-8)
    @test isapprox(m13, m1, rtol = 4e-5)
end

@testset "CDaR" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :CDaR

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        1.50806703376587e-11,
        6.249671237502534e-10,
        8.774547100425806e-12,
        2.31356551572488e-11,
        0.003409904106577405,
        2.8099845639145353e-11,
        4.153231006171802e-11,
        0.07904282952853041,
        4.7648487407066395e-12,
        2.809653009354099e-11,
        0.3875931702930109,
        4.95192077349802e-11,
        7.173223640179149e-11,
        1.4232647649802934e-10,
        0.000554507936023411,
        0.09598828916948485,
        0.2679088570260291,
        2.353669954994726e-10,
        0.0006560272186424142,
        0.164846413448305,
    ]
    w2t = [
        1.4787989960845795e-11,
        3.707522133790865e-10,
        1.8732534938987102e-11,
        1.076439926383524e-11,
        0.00355947812678088,
        7.52603262060471e-12,
        1.4733123984448708e-12,
        0.0799730575587584,
        2.028989104127757e-11,
        7.791305541528433e-12,
        0.3871598970222953,
        4.806436503153469e-11,
        5.0344201774631185e-11,
        9.220045750590425e-11,
        0.00022182814062198487,
        0.09652576149025685,
        0.2659397609725724,
        1.3554243438655823e-10,
        0.0019205742866616102,
        0.1646996416237835,
    ]
    w3t = [
        4.94790014376243e-12,
        9.239453101653306e-13,
        0.07233521355163774,
        3.898023756513953e-12,
        0.3107248862034117,
        3.06520928537622e-11,
        8.090232056009105e-12,
        0.12861272446784397,
        4.4646493731161895e-12,
        1.12323293906573e-11,
        0.1643840413360557,
        2.584523587815502e-11,
        3.5236378159450484e-11,
        1.8035149709806896e-11,
        3.810717984437754e-11,
        0.26288265752799483,
        1.476464041020182e-11,
        9.339724498020173e-12,
        3.431249932364517e-13,
        0.06106047670717545,
    ]
    w4t = [
        3.803976066853783e-12,
        4.1481468584872144e-12,
        5.303664037467956e-12,
        4.908331094915101e-12,
        4.467138211245663e-9,
        1.4269783328920572e-12,
        0.9999999954687258,
        2.7275761091770803e-12,
        4.4301788241778626e-12,
        2.9717098349061466e-12,
        2.586304360084648e-12,
        1.5257626207357174e-12,
        8.928781388456967e-13,
        2.047639752198374e-12,
        9.873358139044833e-13,
        9.780073534471294e-12,
        5.59859035593447e-12,
        2.8300750838071842e-12,
        4.621480076961893e-12,
        3.545183964839339e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 7e-8)
    @test isapprox(w2t, w2.weights, rtol = 6e-8)
    @test isapprox(w3t, w3.weights, rtol = 3e-6)
    @test isapprox(w4t, w4.weights, rtol = 6e-5)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        1.50806703376587e-11,
        6.249671237502534e-10,
        8.774547100425806e-12,
        2.31356551572488e-11,
        0.003409904106577405,
        2.8099845639145353e-11,
        4.153231006171802e-11,
        0.07904282952853041,
        4.7648487407066395e-12,
        2.809653009354099e-11,
        0.3875931702930109,
        4.95192077349802e-11,
        7.173223640179149e-11,
        1.4232647649802934e-10,
        0.000554507936023411,
        0.09598828916948485,
        0.2679088570260291,
        2.353669954994726e-10,
        0.0006560272186424142,
        0.164846413448305,
    ]
    w2t = [
        3.537482310072326e-11,
        6.984881581761563e-10,
        4.505579538130518e-11,
        2.624276050766228e-11,
        0.0035594813469905424,
        1.9112360404146435e-11,
        5.624158097569954e-12,
        0.07997303603432825,
        4.8186372066370727e-11,
        1.9662945482369227e-11,
        0.3871599102619969,
        1.0511730598657314e-10,
        1.1656197806576236e-10,
        2.1058934965251036e-10,
        0.00022183284679132606,
        0.09652574148014607,
        0.26593980316756555,
        2.7721174742291405e-10,
        0.0019205365410174472,
        0.16469965671393635,
    ]
    w3t = [
        7.357076012669321e-8,
        1.5114949311879244e-7,
        0.061886375570835955,
        4.2308082444337063e-8,
        0.256284942817227,
        2.1021828352932678e-8,
        2.7273239753746537e-8,
        0.13588992824499888,
        4.0012742482194575e-8,
        3.844287878953507e-8,
        0.18249676581541938,
        2.6994301560673048e-8,
        1.4523459717499217e-8,
        6.357033689275406e-8,
        2.1459944861297988e-8,
        0.17539098156297359,
        1.5718091634633675e-6,
        1.0929025982960656e-7,
        1.3309738233270562e-7,
        0.1880486714646715,
    ]
    w4t = [
        5.833171246691649e-12,
        5.667558619733636e-12,
        4.997831533751737e-12,
        5.332153543737134e-12,
        0.8503241084323161,
        7.222487462442484e-12,
        0.14967589146340424,
        6.336895024879878e-12,
        5.360827789317116e-12,
        6.261559221617383e-12,
        6.352353549059081e-12,
        7.064084010039971e-12,
        7.295107905810181e-12,
        6.7437496821312885e-12,
        7.437013886756752e-12,
        1.0412289964781843e-12,
        4.2955754225997876e-12,
        6.213011115808583e-12,
        5.013449353989785e-12,
        5.81158599282837e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 6e-8)
    @test isapprox(w2t, w2.weights, rtol = 2e-7)
    @test isapprox(w3t, w3.weights, rtol = 7e-4)
    @test isapprox(w4t, w4.weights, rtol = 1e-9)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        1.50806703376587e-11,
        6.249671237502534e-10,
        8.774547100425806e-12,
        2.31356551572488e-11,
        0.003409904106577405,
        2.8099845639145353e-11,
        4.153231006171802e-11,
        0.07904282952853041,
        4.7648487407066395e-12,
        2.809653009354099e-11,
        0.3875931702930109,
        4.95192077349802e-11,
        7.173223640179149e-11,
        1.4232647649802934e-10,
        0.000554507936023411,
        0.09598828916948485,
        0.2679088570260291,
        2.353669954994726e-10,
        0.0006560272186424142,
        0.164846413448305,
    ]
    w2t = [
        2.2129715442476234e-11,
        5.110823156575394e-10,
        2.721033478958096e-11,
        1.6545279481354886e-11,
        0.003559478287238249,
        1.200082689222971e-11,
        3.5732787440471967e-12,
        0.07997305875543002,
        3.0060021704515915e-11,
        1.235972617897965e-11,
        0.3871598988289338,
        6.823662167198845e-11,
        7.269026757550347e-11,
        1.3144645502323245e-10,
        0.00022182446405750458,
        0.09652576209108102,
        0.2659397509704526,
        1.9073525186716622e-10,
        0.001920579073591718,
        0.16469964643114504,
    ]
    w3t = [
        1.8289495349378173e-8,
        3.1517902998623714e-8,
        0.07723933391984794,
        9.91822424444531e-9,
        0.2834086188728828,
        5.213883597289644e-9,
        7.191133818700399e-9,
        0.12091322119030708,
        8.71321260024212e-9,
        9.125712283951855e-9,
        0.16258519523696466,
        6.801859241589251e-9,
        3.504747943301885e-9,
        1.5616939520472865e-8,
        5.496276732313215e-9,
        0.237195825814522,
        8.138743754719605e-8,
        2.168601034732469e-8,
        2.3663154668147574e-8,
        0.11865755683948462,
    ]
    w4t = [
        3.703735708778353e-12,
        4.147126707189452e-12,
        5.153399221038753e-12,
        4.6466556347094806e-12,
        0.8533951436642115,
        1.6284108113907754e-12,
        0.14660485627152742,
        2.823248074650451e-12,
        4.491022194262484e-12,
        3.054227617109781e-12,
        2.7129323237514746e-12,
        1.689286128556467e-12,
        1.203015725952695e-12,
        2.1662918633740794e-12,
        1.2225775819736924e-12,
        8.397016578654984e-12,
        5.770896453180728e-12,
        2.9724681385502103e-12,
        4.807257536894347e-12,
        3.671290233929755e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 2e-7)
    @test isapprox(w2t, w2.weights, rtol = 2e-4)
    @test isapprox(w3t, w3.weights, rtol = 7e-4)
    @test isapprox(w4t, w4.weights, rtol = 8e-7)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :CDaR
    kelly = :none
    portfolio.cdar_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.cdar_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 2e-4)
    @test isapprox(r5, r1, rtol = 7e-8)
    @test isapprox(m5, m1, rtol = 2e-5)

    portfolio.cdar_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 4e-6)
    @test isapprox(r6, r2, rtol = 2e-8)
    @test isapprox(m6, m2, rtol = 2e-6)

    portfolio.cdar_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 6e-6)
    @test isapprox(r7, r3, rtol = 3e-7)
    @test isapprox(m7, m3, rtol = 2e-7)

    portfolio.cdar_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 5e-4)
    @test isapprox(r8, r4, rtol = 6e-4)
    @test isapprox(m8, m4, rtol = 9e-6)

    portfolio.cdar_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.cdar_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 8e-5)
    @test isapprox(r13, r1, rtol = 5e-8)
    @test isapprox(m13, m1, rtol = 2e-5)
end

@testset "UCI" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :UCI

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        2.8573856133640373e-11,
        0.025012517716847604,
        0.09201957767601905,
        3.397212605030151e-11,
        0.019404855563209034,
        2.2192356878822477e-11,
        5.1293200887220306e-11,
        0.06508432959496258,
        2.454567525313381e-12,
        1.6616452289722766e-11,
        0.2853653890219146,
        5.163944847058319e-11,
        2.5129434218651528e-11,
        0.00837124575457371,
        2.8931196539044938e-11,
        0.09716508345237464,
        0.17512886474536019,
        0.024589945044680345,
        0.057283684697387816,
        0.15057450647186765,
    ]
    w2t = [
        3.31255149102299e-10,
        0.023995156491853242,
        0.09614562814573017,
        1.1318869705495483e-10,
        0.019578365504499408,
        1.4539421227405334e-10,
        5.911560701845206e-11,
        0.06625970142543393,
        2.2290867688024977e-10,
        1.7213187805698612e-10,
        0.28380831638597565,
        3.998480882136187e-10,
        2.7718113337040545e-10,
        0.0015417376110757842,
        1.2876165333179256e-10,
        0.10123940156201022,
        0.17626350121483378,
        0.020477940699002876,
        0.060870857857993385,
        0.14981939125180643,
    ]
    w3t = [
        2.1543660581513774e-13,
        1.0226585618359557e-13,
        0.20538866456491212,
        4.7531322759925855e-14,
        0.2857955775007091,
        1.1588019746725433e-12,
        4.4331957523261757e-13,
        0.14358960326848028,
        7.852224526829135e-14,
        3.6512812402469434e-13,
        0.075755866586437,
        9.883666129010734e-13,
        1.3108076134131662e-12,
        6.142328083142264e-13,
        1.4422232670321696e-12,
        0.18477327890483056,
        0.10469700916060591,
        2.327776333344939e-13,
        3.027985066240413e-12,
        3.9976565597126604e-12,
    ]
    w4t = [
        8.672282010741108e-11,
        9.644043194477562e-11,
        1.180159068835407e-10,
        1.1192707419294832e-10,
        1.5647357102567998e-7,
        3.241770015144735e-11,
        0.9999998421289982,
        6.265705897336817e-11,
        1.0338818358775372e-10,
        6.884471506342884e-11,
        5.920221620402874e-11,
        3.433220861626752e-11,
        1.923304002244607e-11,
        4.6845029077112315e-11,
        2.1603869155768523e-11,
        1.5694603370822014e-10,
        1.2601965843741132e-10,
        6.495440053411419e-11,
        1.0640550644049826e-10,
        8.147499254184257e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-6)
    @test isapprox(w2t, w2.weights, rtol = 9e-6)
    @test isapprox(w3t, w3.weights, rtol = 3e-4)
    @test isapprox(w4t, w4.weights, rtol = 2e-5)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        2.8573856133640373e-11,
        0.025012517716847604,
        0.09201957767601905,
        3.397212605030151e-11,
        0.019404855563209034,
        2.2192356878822477e-11,
        5.1293200887220306e-11,
        0.06508432959496258,
        2.454567525313381e-12,
        1.6616452289722766e-11,
        0.2853653890219146,
        5.163944847058319e-11,
        2.5129434218651528e-11,
        0.00837124575457371,
        2.8931196539044938e-11,
        0.09716508345237464,
        0.17512886474536019,
        0.024589945044680345,
        0.057283684697387816,
        0.15057450647186765,
    ]
    w2t = [
        7.381664717806015e-11,
        0.023994975528831865,
        0.09614605651806482,
        1.358539126044996e-10,
        0.0195782061703337,
        1.0493108633498114e-10,
        1.8780854578320663e-10,
        0.0662597760400171,
        3.016745526947596e-11,
        7.921002771643784e-11,
        0.28380816390048436,
        1.4082568933009985e-10,
        2.2223393136027205e-11,
        0.0015413528083968624,
        1.206312764541068e-10,
        0.10123970026436047,
        0.17626331050274827,
        0.020477673157072482,
        0.060871408815968996,
        0.14981937539825313,
    ]
    w3t = [
        9.470177046367355e-8,
        2.3318418183851794e-7,
        0.19301330540331357,
        8.494366855818314e-8,
        0.2307780806699567,
        3.2773034710306665e-8,
        1.1020862802894329e-7,
        0.13725124449232973,
        9.128218847850525e-8,
        7.927389149067948e-8,
        0.11256924657844981,
        3.2764660464593575e-8,
        2.227672007302793e-8,
        1.2554166055285175e-7,
        2.4215592420699135e-8,
        0.15377397204773313,
        0.12946959110498163,
        1.9037761973188245e-7,
        0.013808763179287609,
        0.029334674980330877,
    ]
    w4t = [
        5.94428603277797e-11,
        5.79592995051968e-11,
        5.468349703086591e-11,
        5.6280631572055565e-11,
        0.850323083422901,
        6.60772106918945e-11,
        0.14967691550168977,
        6.219101325722699e-11,
        5.671880120889565e-11,
        6.144651652571907e-11,
        6.257694738422385e-11,
        6.587449201678317e-11,
        6.740943518343593e-11,
        6.435535915064993e-11,
        6.746015441685499e-11,
        4.350521718923856e-11,
        5.258104027975796e-11,
        6.175025766565361e-11,
        5.5630613187843894e-11,
        5.946609144083753e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 8e-6)
    @test isapprox(w2t, w2.weights, rtol = 2e-3)
    @test isapprox(w3t, w3.weights, rtol = 4e-4)
    @test isapprox(w4t, w4.weights, rtol = 1e-9)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        2.8573856133640373e-11,
        0.025012517716847604,
        0.09201957767601905,
        3.397212605030151e-11,
        0.019404855563209034,
        2.2192356878822477e-11,
        5.1293200887220306e-11,
        0.06508432959496258,
        2.454567525313381e-12,
        1.6616452289722766e-11,
        0.2853653890219146,
        5.163944847058319e-11,
        2.5129434218651528e-11,
        0.00837124575457371,
        2.8931196539044938e-11,
        0.09716508345237464,
        0.17512886474536019,
        0.024589945044680345,
        0.057283684697387816,
        0.15057450647186765,
    ]
    w2t = [
        6.902410401873416e-11,
        0.024111036317818545,
        0.09587710149003388,
        2.4394995103859166e-11,
        0.019666485160494962,
        3.10054947560329e-11,
        1.3349257142124168e-11,
        0.0661884594209506,
        4.691890960113134e-11,
        3.647843119985767e-11,
        0.2838959581699876,
        8.307261179674661e-11,
        5.831680988460277e-11,
        0.0019256330438164524,
        2.764791564834839e-11,
        0.10102226159655256,
        0.17613350867969116,
        0.020642250728953728,
        0.060701881315712525,
        0.14983542368577935,
    ]
    w3t = [
        1.5247842677676474e-10,
        3.524679927962467e-10,
        0.19373303257981378,
        1.614531556540307e-10,
        0.28311920349066666,
        4.819509378664574e-11,
        1.9391470274470558e-10,
        0.1383602056894536,
        1.4816992497199137e-10,
        1.286990163811923e-10,
        0.09651848273011439,
        4.776979697126245e-11,
        2.7803933557949598e-11,
        1.9552049973783585e-10,
        3.46991968536065e-11,
        0.17232429931743645,
        0.11594476527708376,
        3.1429651922060757e-10,
        3.846928430419556e-9,
        5.263034661559541e-9,
    ]
    w4t = [
        2.310126084138472e-12,
        2.586678918492548e-12,
        3.214330757417172e-12,
        2.8982505552249603e-12,
        0.8533951447999877,
        1.0156678448788097e-12,
        0.14660485515993127,
        1.7609342271077702e-12,
        2.8011697012624134e-12,
        1.9050005196521198e-12,
        1.692126266065065e-12,
        1.0536410096604785e-12,
        7.503287564215962e-13,
        1.3511646949874367e-12,
        7.625337619609635e-13,
        5.2374507013493405e-12,
        3.5994660020952347e-12,
        1.8540071121672065e-12,
        2.9984171011666522e-12,
        2.2898864456396156e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 4e-6)
    @test isapprox(w2t, w2.weights, rtol = 7e-7)
    @test isapprox(w3t, w3.weights, rtol = 9e-4)
    @test isapprox(w4t, w4.weights, rtol = 6e-6)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :UCI
    kelly = :none
    portfolio.uci_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.uci_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 3e-3)
    @test isapprox(r5, r1, rtol = 2e-6)
    @test isapprox(m5, m1, rtol = 8e-4)

    portfolio.uci_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 9e-6)
    @test isapprox(r6, r2, rtol = 7e-8)
    @test isapprox(m6, m2, rtol = 6e-6)

    portfolio.uci_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 2e-5)
    @test isapprox(r7, r3, rtol = 7e-6)
    @test isapprox(m7, m3, rtol = 6e-6)

    portfolio.uci_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 8e-4)
    @test isapprox(r8, r4, rtol = 2e-3)
    @test isapprox(m8, m4, rtol = 2e-5)

    portfolio.uci_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.uci_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 3e-3)
    @test isapprox(r13, r1, rtol = 7e-7)
    @test isapprox(m13, m1, rtol = 2e-3)
end

@testset "EDaR" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :EDaR

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.03767505471711524,
        6.9862592972923115e-9,
        1.8380856776444667e-9,
        8.596028875936026e-10,
        0.0150744932739616,
        1.2129067932956508e-9,
        2.2833544554017917e-10,
        0.03750004065958383,
        1.484618283318785e-9,
        1.5980408579265315e-9,
        0.4198566216597687,
        2.210652182094716e-9,
        0.013791407674534949,
        2.650354373480663e-9,
        1.4321446310486007e-9,
        0.13380993934095334,
        0.20614910645041068,
        0.03733620657653665,
        3.292948441991316e-9,
        0.09880710585318606,
    ]
    w2t = [
        0.03711706902498548,
        8.729581371387324e-10,
        2.9141427613003625e-10,
        1.254991315009368e-10,
        0.016054886679969346,
        1.684601513210898e-10,
        4.839276626284541e-11,
        0.03761163318751976,
        2.0440093247581053e-10,
        2.186145228571287e-10,
        0.4197912802940466,
        2.8763767665258153e-10,
        0.013865394848671102,
        3.5193416563858027e-10,
        1.9257962104050944e-10,
        0.13411499052539214,
        0.20507901191934055,
        0.03787532977971588,
        4.4193001034604603e-10,
        0.09849040053653775,
    ]
    w3t = [
        1.992999360146986e-11,
        2.3620076858116867e-11,
        0.10585599929554294,
        7.519270263484781e-12,
        0.2712801917707634,
        4.143430270848442e-12,
        5.9450432965012895e-12,
        0.0885275827410627,
        6.4466141800088215e-12,
        6.991269467187078e-12,
        0.2612188914959996,
        4.081432761846942e-12,
        2.3852067660328066e-12,
        1.1566918522828015e-11,
        3.690170268347856e-12,
        0.2731173344356761,
        4.6026253810506296e-11,
        1.7264331707364347e-11,
        1.554062960825502e-11,
        8.580443969504943e-11,
    ]
    w4t = [
        1.7958311155299647e-11,
        1.9092819780808933e-11,
        2.313540801807031e-11,
        2.187595681405776e-11,
        7.542365215298968e-8,
        8.379795587042162e-12,
        0.9999999242803798,
        1.3401698979702848e-11,
        2.0461703487665662e-11,
        1.4635011949397094e-11,
        1.2946202500574078e-11,
        8.893956465038269e-12,
        6.3420022252644e-12,
        1.0875736115337036e-11,
        6.664422514194507e-12,
        3.523123246029849e-11,
        2.4342735223240856e-11,
        1.3977899121830702e-11,
        2.097202424720089e-11,
        1.678106090306329e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 1e-4)
    @test isapprox(w2t, w2.weights, rtol = 3e-5)
    @test isapprox(w3t, w3.weights, rtol = 3e-4)
    @test isapprox(w4t, w4.weights, rtol = 7e-5)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.037678852063241154,
        1.2320790677839736e-8,
        3.2716224350174573e-9,
        1.5370846682019213e-9,
        0.015072179921953983,
        2.1678813431969566e-9,
        4.125655657843223e-10,
        0.03749819033657992,
        2.6503044946243245e-9,
        2.8703176128524173e-9,
        0.4198580308127915,
        4.000608804550504e-9,
        0.013791459013869755,
        4.7371859429333805e-9,
        2.5583384653651057e-9,
        0.1338105183392617,
        0.20615226897743133,
        0.03733171050603101,
        5.8402462452203755e-9,
        0.0988067476618934,
    ]
    w2t = [
        0.03711675220834487,
        2.3882085015889046e-10,
        1.0244991604634996e-10,
        3.4857646338074e-11,
        0.016054896135949032,
        4.668322225838249e-11,
        1.3621821287350694e-11,
        0.037611623026377425,
        5.652557709531356e-11,
        6.021566445715158e-11,
        0.41979135889782204,
        7.798133199795387e-11,
        0.013865445354519716,
        9.627085038506303e-11,
        5.2726299305248456e-11,
        0.1341147157652597,
        0.2050789401649467,
        0.037875815165511695,
        1.3264722162040313e-10,
        0.09849045236846839,
    ]
    w3t = [
        1.0175852924596092e-5,
        1.1337255755070881e-5,
        0.0910230357956426,
        3.293389111467496e-6,
        0.2684095225982562,
        1.7677176689326612e-6,
        3.084990211228876e-6,
        0.08490109186940797,
        3.019431614204926e-6,
        3.193618898000002e-6,
        0.30981348433208755,
        1.594395066942501e-6,
        1.02137066369318e-6,
        5.004456920393623e-6,
        1.4165256599124134e-6,
        0.24562365301783817,
        4.978183335760592e-5,
        9.48547573294788e-6,
        8.316776384819347e-6,
        0.00011671929679781688,
    ]
    w4t = [
        9.98293166041218e-11,
        9.613273712096868e-11,
        8.775785511030848e-11,
        9.196270133505501e-11,
        0.8503243447043849,
        1.1685940335492614e-10,
        0.14967565348063677,
        1.0701208491934121e-10,
        9.331122266055903e-11,
        1.0514309899415045e-10,
        1.0795729760261243e-10,
        1.1637976028843092e-10,
        1.203459557362761e-10,
        1.1246106093736785e-10,
        1.2019267103864063e-10,
        6.05174062036615e-11,
        8.262874373969107e-11,
        1.058229582416426e-10,
        9.061512642385994e-11,
        1.0004896141490282e-10,
    ]

    @test isapprox(w1t, w1.weights, rtol = 1e-4)
    @test isapprox(w2t, w2.weights, rtol = 2e-4)
    @test isapprox(w3t, w3.weights, rtol = 9e-3)
    @test isapprox(w4t, w4.weights, rtol = 1e-7)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.033608708036103455,
        0.0004537950176223421,
        0.00015245065545274833,
        5.682640269197295e-5,
        0.02356867147678142,
        7.455658472709149e-5,
        2.0332331864825853e-5,
        0.046177161064675794,
        9.435503830477678e-5,
        8.257440656712476e-5,
        0.415600362151364,
        8.318411211694279e-5,
        0.009945132617318324,
        0.00016427329923060117,
        7.844734228485846e-5,
        0.13568869177135937,
        0.21231804702323134,
        0.0177030495458141,
        0.00030039401158856176,
        0.10382898711090033,
    ]
    w2t = [
        0.02718627740740722,
        1.1210459472559174e-9,
        1.2204698998768534e-10,
        2.896876272879051e-9,
        0.01788586136107961,
        2.3960001655146344e-9,
        7.399990004977621e-9,
        0.04181340541303176,
        1.5741182903537986e-9,
        1.7101520968056247e-9,
        0.4152231659598705,
        2.1269689676892835e-10,
        0.012177281289746194,
        6.40835196830973e-10,
        3.32688412178124e-9,
        0.13168289623273985,
        0.21183337034255523,
        0.036197126554739084,
        2.9555476253781347e-10,
        0.10600059374262981,
    ]
    w3t = [
        0.026296449874543715,
        0.02218818594589978,
        0.015558231983073293,
        0.025081675782865086,
        0.24318325268623073,
        0.025100251061381605,
        0.008184829383686705,
        0.022913551365973256,
        0.03042992198463704,
        0.005190175748776066,
        0.2802605509148086,
        0.014906245647211052,
        0.006508719331879848,
        0.008978699116553062,
        0.004175852422303486,
        0.13821095611897674,
        0.03222477838574424,
        0.017023948958441663,
        0.011885251816368889,
        0.06169847147064508,
    ]
    w4t = [
        5.249174838247834e-12,
        5.87759619915689e-12,
        7.303483696772372e-12,
        6.5855674183277305e-12,
        0.8533951445485147,
        2.3081402247758664e-12,
        0.14660485536041026,
        4.001402225119689e-12,
        6.36501178085105e-12,
        4.328739150897184e-12,
        3.84505095333338e-12,
        2.394297277926787e-12,
        1.7052103817089558e-12,
        3.070371303699421e-12,
        1.732914367099223e-12,
        1.1900388129382948e-11,
        8.17872506699709e-12,
        4.2128658820167e-12,
        6.813123718972056e-12,
        5.2031643643685795e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 6e-2)
    @test isapprox(w2t, w2.weights, rtol = 4e-2)
    @test isapprox(w3t, w3.weights, rtol = 5e-1)
    @test isapprox(w4t, w4.weights, rtol = 4e-7)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :EDaR
    kelly = :none
    portfolio.edar_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.edar_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 7e-2)
    @test isapprox(r5, r1, rtol = 3e-3)
    @test isapprox(m5, m1, rtol = 6e-2)

    portfolio.edar_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 1e-3)
    @test isapprox(r6, r2, rtol = 2e-6)
    @test isapprox(m6, m2, rtol = 3e-4)

    portfolio.edar_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 5e-5)
    @test isapprox(r7, r3, rtol = 2e-6)
    @test isapprox(m7, m3, rtol = 2e-6)

    portfolio.edar_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 7e-4)
    @test isapprox(r8, r4, rtol = 9e-4)
    @test isapprox(m8, m4, rtol = 2e-5)

    portfolio.edar_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.edar_u = r1 + 0.0001 * r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 2e-2)
    @test isapprox(r13, r1, rtol = 1e-4)
    @test isapprox(m13, m1, rtol = 2e-2)
end

@testset "RDaR" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :RDaR

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.05579002972141407,
        4.559085421181959e-9,
        1.8596335544931795e-9,
        2.9802281464342953e-9,
        0.02857882770931755,
        5.415987709744306e-10,
        5.200639256109981e-8,
        0.02016234404770185,
        5.185391359718703e-9,
        7.612518150324201e-9,
        0.4407875773011993,
        6.004924418499805e-9,
        0.02204964864890511,
        1.5605628094681774e-8,
        3.473464920508876e-9,
        0.1434081587814339,
        0.16474438796286034,
        0.054676628485664056,
        2.768572294301908e-9,
        0.06980229474406624,
    ]
    w2t = [
        0.056636687497673845,
        6.27865593424031e-11,
        8.106634585423627e-11,
        4.1983104927634675e-11,
        0.030740881403189306,
        8.501078569071155e-11,
        6.396713373455335e-11,
        0.0209369197102892,
        5.888614447820761e-11,
        1.0861052698545209e-10,
        0.4409670232854981,
        9.619172232751722e-11,
        0.02151848059277469,
        3.689356310482286e-11,
        3.7464482532395707e-11,
        0.14456209509135773,
        0.1584578969581559,
        0.06207652527903462,
        1.278200151884165e-10,
        0.0641034893813464,
    ]
    w3t = [
        2.133617860347163e-11,
        1.5407574130269098e-11,
        0.042683076065995834,
        6.107979175751156e-12,
        0.27528407175039904,
        3.449201714106315e-12,
        5.3121493727321016e-12,
        0.09150577569148877,
        4.873346073301156e-12,
        5.581807301267693e-12,
        0.3004567557117296,
        2.8758453543444498e-12,
        1.8360136009206755e-12,
        8.538123056663443e-12,
        2.7026139061102472e-12,
        0.29007032062533095,
        2.4539651394109665e-11,
        1.3445342665761049e-11,
        1.0816941559252514e-11,
        2.8233199682213427e-11,
    ]
    w4t = [
        1.798015468632578e-10,
        1.9095052335950088e-10,
        2.2425626010891484e-10,
        2.1589752389465577e-10,
        5.963145008355697e-8,
        8.466712945379815e-11,
        0.9999999375658778,
        1.3332158275445916e-10,
        2.0247795110882202e-10,
        1.4634238071456172e-10,
        1.2939020402283305e-10,
        8.959191577263325e-11,
        6.416785986738725e-11,
        1.0868291560385538e-10,
        6.669930910819947e-11,
        2.0710190751632579e-10,
        2.414811424026195e-10,
        1.4005732048979146e-10,
        2.0793324864769145e-10,
        1.6985143604918585e-10,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-2)
    @test isapprox(w2t, w2.weights, rtol = 1e-4)
    @test isapprox(w3t, w3.weights, rtol = 6e-5)
    @test isapprox(w4t, w4.weights, rtol = 9e-5)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.05579002972141407,
        4.559085421181959e-9,
        1.8596335544931795e-9,
        2.9802281464342953e-9,
        0.02857882770931755,
        5.415987709744306e-10,
        5.200639256109981e-8,
        0.02016234404770185,
        5.185391359718703e-9,
        7.612518150324201e-9,
        0.4407875773011993,
        6.004924418499805e-9,
        0.02204964864890511,
        1.5605628094681774e-8,
        3.473464920508876e-9,
        0.1434081587814339,
        0.16474438796286034,
        0.054676628485664056,
        2.768572294301908e-9,
        0.06980229474406624,
    ]
    w2t = [
        0.05663699285036402,
        2.4750728979525563e-11,
        2.8314379441226932e-11,
        1.6667197953981526e-11,
        0.03074130377126806,
        3.376082184741563e-11,
        2.5574937741583603e-11,
        0.020936866071767556,
        2.3361253788537988e-11,
        4.351266062522461e-11,
        0.440967145366926,
        3.862200114650797e-11,
        0.0215183572573781,
        1.4595250612845393e-11,
        1.4986427605500122e-11,
        0.14456274738879232,
        0.15845664262723463,
        0.062076512755033904,
        4.9452505921584474e-11,
        0.06410343159763718,
    ]
    w3t = [
        1.966635607553692e-8,
        8.98137497417884e-9,
        0.04795059013392795,
        4.015089705038871e-9,
        0.2753757331344829,
        2.3018052961878973e-9,
        4.5329749415613195e-9,
        0.08982851026510076,
        3.34918634426132e-9,
        3.8114307159920624e-9,
        0.3141337264899054,
        1.894101088865196e-9,
        1.3042663617960842e-9,
        4.706489295373536e-9,
        1.6515259314724798e-9,
        0.2727113183694448,
        2.4337613441536403e-8,
        1.133394704881852e-8,
        8.040791306651472e-9,
        2.1680185762548424e-8,
    ]
    w4t = [
        3.850474101113577e-11,
        3.6864796739240165e-11,
        3.082984501180563e-11,
        3.3386350740925036e-11,
        0.850324448678227,
        4.647698083923578e-11,
        0.14967555062721782,
        4.363785713064446e-11,
        3.5297900515825345e-11,
        4.2818554495939095e-11,
        4.4289213050586806e-11,
        4.7435666727898976e-11,
        4.975147709837622e-11,
        4.6359236365246014e-11,
        4.941058628909707e-11,
        8.909263488788948e-12,
        2.5767932694497204e-11,
        4.2882248525504654e-11,
        3.280349733406062e-11,
        3.91292532510967e-11,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-2)
    @test isapprox(w2t, w2.weights, rtol = 2e-4)
    @test isapprox(w3t, w3.weights, rtol = 2e-5)
    @test isapprox(w4t, w4.weights, rtol = 1e-6)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.05579002972141407,
        4.559085421181959e-9,
        1.8596335544931795e-9,
        2.9802281464342953e-9,
        0.02857882770931755,
        5.415987709744306e-10,
        5.200639256109981e-8,
        0.02016234404770185,
        5.185391359718703e-9,
        7.612518150324201e-9,
        0.4407875773011993,
        6.004924418499805e-9,
        0.02204964864890511,
        1.5605628094681774e-8,
        3.473464920508876e-9,
        0.1434081587814339,
        0.16474438796286034,
        0.054676628485664056,
        2.768572294301908e-9,
        0.06980229474406624,
    ]
    w2t = [
        0.05649067259157338,
        5.8103104337953924e-8,
        1.6567766029586955e-8,
        1.2555013064205857e-8,
        0.030203079328841493,
        4.306105278573005e-9,
        3.999830320763112e-8,
        0.02062554463709978,
        2.3459203001733976e-8,
        8.139661264558102e-8,
        0.44108997104854836,
        3.932121966900731e-8,
        0.02160030140233836,
        9.391008794990029e-9,
        2.7692448497900765e-7,
        0.14442531147020343,
        0.15998848112783523,
        0.05948903612064255,
        1.7701631906438088e-8,
        0.06608702254846442,
    ]
    w3t = [
        5.8973218767681847e-8,
        5.5819599676523714e-8,
        0.06301891418194162,
        1.8522676986776382e-8,
        0.2783746368026414,
        9.912669260857452e-9,
        1.1305185318610609e-7,
        0.08134233436726744,
        1.4641931938424231e-8,
        1.6766258117017237e-8,
        0.3030318756113728,
        8.258727038022735e-9,
        5.291572574523198e-9,
        2.6795364248549612e-8,
        8.222186143635196e-9,
        0.27423166835897655,
        8.13396035348626e-8,
        4.062121304066539e-8,
        3.3943879457652875e-8,
        7.851704625915233e-8,
    ]
    w4t = [
        1.2718789854626197e-12,
        1.4241364119812152e-12,
        1.7696696606354671e-12,
        1.5956776132738882e-12,
        0.8533951288587733,
        5.592598472428562e-13,
        0.14660487111915968,
        9.69505231633587e-13,
        1.542233137899069e-12,
        1.0488347529526164e-12,
        9.316260474081235e-13,
        5.800904780031275e-13,
        4.1312460217273386e-13,
        7.439209976535365e-13,
        4.1979215294748297e-13,
        2.8834175649419184e-12,
        1.9817154590077295e-12,
        1.0207489951197946e-12,
        1.6508092678608276e-12,
        1.2607089038440672e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-2)
    @test isapprox(w2t, w2.weights, rtol = 8e-3)
    @test isapprox(w3t, w3.weights, rtol = 6e-2)
    @test isapprox(w4t, w4.weights, rtol = 1e-4)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :RDaR
    kelly = :none
    portfolio.rdar_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.rdar_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 4e-3)
    @test isapprox(r5, r1, rtol = 4e-6)
    @test isapprox(m5, m1, rtol = 3e-3)

    portfolio.rdar_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 2e-3)
    @test isapprox(r6, r2, rtol = 7e-6)
    @test isapprox(m6, m2, rtol = 2e-3)

    portfolio.rdar_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 2e-5)
    @test isapprox(r7, r3, rtol = 3e-6)
    @test isapprox(m7, m3, rtol = 2e-6)

    portfolio.rdar_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 3e-3)
    @test isapprox(r8, r4, rtol = 3e-3)
    @test isapprox(m8, m4, rtol = 5e-5)

    portfolio.rdar_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.rdar_u = r1 + 0.01 * r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 2e-1)
    @test isapprox(r13, r1, rtol = 1e-2)
    @test isapprox(m13, m1, rtol = 2e-1)
end

@testset "Kurt full" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :Kurt

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        9.37887062985676e-10,
        0.040595827124190915,
        7.33779394635287e-10,
        0.07615519909349815,
        7.195445799737569e-10,
        0.0002488568040515834,
        1.316547801225105e-9,
        0.12868741827629487,
        1.3303094094344748e-9,
        3.5220831961377146e-10,
        0.37188983319168933,
        1.3408325442203143e-9,
        2.313796015165089e-9,
        0.040993353331281894,
        5.36075521118924e-10,
        0.02035251368640191,
        2.5540952978598503e-10,
        0.18817927618180288,
        1.0415864199419324e-9,
        0.13289771143281173,
    ]
    w2t = [
        4.501364323484029e-9,
        2.5872700783289023e-8,
        0.012845866543269004,
        0.020852379650490353,
        0.35356857190266605,
        1.382254209203807e-9,
        0.08634900860358363,
        3.6109971512077456e-7,
        7.546454166461182e-9,
        6.5748370197303755e-9,
        1.3034546165372696e-8,
        1.2637061958449874e-9,
        8.03707187049686e-10,
        2.598527802014657e-9,
        7.670367908485034e-10,
        0.13914989262996097,
        0.30506681780848616,
        1.4322338460401846e-8,
        0.08216701023746346,
        1.285689225262228e-8,
    ]
    w3t = [
        3.6419200497455516e-8,
        0.007710268469310552,
        5.603055391937799e-7,
        0.06255617249283361,
        0.21592044154782297,
        1.3279242601118099e-8,
        0.04812424024349563,
        0.12658728887646772,
        6.716224325183401e-8,
        6.774248418845593e-8,
        0.19710244454253203,
        1.1195182575417427e-8,
        6.662928692587823e-9,
        2.9693096124252296e-8,
        7.142596763644542e-9,
        0.09193977277632265,
        0.2084653570226398,
        0.033822859313448396,
        0.007768542644696255,
        1.8124679163199437e-6,
    ]
    w4t = [
        1.4787352452290254e-8,
        1.490924726238489e-8,
        1.5271106865978208e-8,
        1.5131823081412687e-8,
        4.1477661002483844e-7,
        1.2453355384674588e-8,
        0.9999993320298523,
        1.406175010490928e-8,
        1.5009974980910758e-8,
        1.4303559762991834e-8,
        1.3975491719132134e-8,
        1.2701349531971336e-8,
        1.1047669990477897e-8,
        1.3415788743833218e-8,
        1.124740653460713e-8,
        1.560383024320098e-8,
        1.5331292122357093e-8,
        1.4196994336520328e-8,
        1.5069820362795756e-8,
        1.4675724302947847e-8,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-2)
    @test isapprox(w2t, w2.weights, rtol = 8e-5)
    @test isapprox(w3t, w3.weights, rtol = 5e-4)
    @test isapprox(w4t, w4.weights, rtol = 1e-6)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        9.37887062985676e-10,
        0.040595827124190915,
        7.33779394635287e-10,
        0.07615519909349815,
        7.195445799737569e-10,
        0.0002488568040515834,
        1.316547801225105e-9,
        0.12868741827629487,
        1.3303094094344748e-9,
        3.5220831961377146e-10,
        0.37188983319168933,
        1.3408325442203143e-9,
        2.313796015165089e-9,
        0.040993353331281894,
        5.36075521118924e-10,
        0.02035251368640191,
        2.5540952978598503e-10,
        0.18817927618180288,
        1.0415864199419324e-9,
        0.13289771143281173,
    ]
    w2t = [
        9.130816441898655e-9,
        9.62385489164749e-8,
        0.010916986871984021,
        0.022522577569575795,
        0.3366156058213832,
        2.874116400773798e-9,
        0.07880406865774194,
        0.044380443644904885,
        1.364525905005014e-8,
        1.2106235799411881e-8,
        4.432298032592873e-8,
        2.581364755239325e-9,
        1.6242108254739377e-9,
        5.370115406552764e-9,
        1.598214898241713e-9,
        0.13239299087136405,
        0.290902461295891,
        3.840345361409899e-8,
        0.08346460726037884,
        3.0111459750895364e-8,
    ]
    w3t = [
        7.555217050276665e-9,
        0.011974197546645455,
        5.846243988513614e-8,
        0.06099298999108579,
        0.19300292887899212,
        2.909252089434819e-9,
        0.041495724721781355,
        0.12397038722751569,
        1.3603251090955495e-8,
        1.3803792369297707e-8,
        0.2297830046492663,
        2.3778432327235743e-9,
        1.4167521916547024e-9,
        6.742236787559794e-9,
        1.5726718069566978e-9,
        0.0832087790891664,
        0.18057810026069063,
        0.05910275955257812,
        2.2201618735363026e-7,
        0.01589079762263438,
    ]
    w4t = [
        7.873276750814923e-13,
        1.0504023549108063e-12,
        1.6462979216469935e-12,
        1.3488712592635367e-12,
        0.8503243196105421,
        4.3617858595866253e-13,
        0.1496756803721629,
        2.698115398872899e-13,
        1.251829389886749e-12,
        4.058133216744164e-13,
        2.043190230711752e-13,
        3.9969738979188116e-13,
        6.829488267087261e-13,
        1.1855316020535182e-13,
        6.722112085296718e-13,
        3.4435075095701036e-12,
        2.010812476744877e-12,
        3.569537732759504e-13,
        1.439845891611167e-12,
        7.69880921433547e-13,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-2)
    @test isapprox(w2t, w2.weights, rtol = 9e-5)
    @test isapprox(w3t, w3.weights, rtol = 2e-5)
    @test isapprox(w4t, w4.weights, rtol = 1e-6)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        9.37887062985676e-10,
        0.040595827124190915,
        7.33779394635287e-10,
        0.07615519909349815,
        7.195445799737569e-10,
        0.0002488568040515834,
        1.316547801225105e-9,
        0.12868741827629487,
        1.3303094094344748e-9,
        3.5220831961377146e-10,
        0.37188983319168933,
        1.3408325442203143e-9,
        2.313796015165089e-9,
        0.040993353331281894,
        5.36075521118924e-10,
        0.02035251368640191,
        2.5540952978598503e-10,
        0.18817927618180288,
        1.0415864199419324e-9,
        0.13289771143281173,
    ]
    w2t = [
        5.5212857821809355e-8,
        4.2299647901007313e-7,
        0.010716621963845957,
        0.022459463930099486,
        0.33659793185430215,
        1.2712374637258332e-6,
        0.07875143270490408,
        0.04488100466158959,
        2.688840488681268e-6,
        3.5064334565861675e-7,
        2.059403600308199e-6,
        5.133981038057419e-7,
        1.2244044525655954e-6,
        6.817230749460916e-7,
        2.632852514131345e-6,
        0.13242516959274409,
        0.2908358047137628,
        4.4506346217756045e-7,
        0.08331994115834425,
        2.836445647452957e-7,
    ]
    w4t = [
        1.7725244472635677e-6,
        2.2332808134257328e-6,
        1.766747131519019e-6,
        1.9168836115916535e-6,
        0.852898775315055,
        2.4592805272459505e-6,
        0.14705134293690694,
        2.7147347511090495e-6,
        2.752944064807937e-6,
        2.758527429213058e-6,
        2.5781240439607573e-6,
        2.7595865722110965e-6,
        4.289346875090677e-6,
        2.483041231169163e-6,
        2.572683315759476e-6,
        6.719705735051469e-6,
        2.3869389493995773e-6,
        2.546815524413547e-6,
        2.6846797205762933e-6,
        2.485903294297768e-6,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-2)
    @test isapprox(w2t, w2.weights, rtol = 1e-2)
    @test isapprox(w4t, w4.weights, rtol = 1e-3)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :Kurt
    kelly = :none
    portfolio.krt_u = Inf
    portfolio.mu_l = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.krt_u = r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 7e-3)
    @test isapprox(r5, r1, rtol = 2e-5)
    @test isapprox(m5, m1, rtol = 9e-3)

    portfolio.krt_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 2e-5)
    @test isapprox(r6, r2, rtol = 4e-8)
    @test isapprox(m6, m2, rtol = 3e-8)

    portfolio.krt_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 4e-5)
    @test isapprox(r7, r3, rtol = 4e-8)
    @test isapprox(m7, m3, rtol = 2e-7)

    portfolio.krt_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 2e-6)
    @test isapprox(r8, r4, rtol = 3e-6)
    @test isapprox(m8, m4, rtol = 4e-8)

    portfolio.krt_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.krt_u = r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 5e-4)
    @test isapprox(r13, r1, rtol = 2e-9)
    @test isapprox(m13, m1, rtol = 7e-4)
end

@testset "Kurt relaxed" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    portfolio.max_num_assets_kurt = 1
    rm = :Kurt

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.05000006003933561,
        0.05000011653200388,
        0.0499999811494444,
        0.050000050665165635,
        0.04999995500833027,
        0.05000014736153201,
        0.04999933385945339,
        0.050000406481719405,
        0.049999839817117114,
        0.05000004649195019,
        0.05000037997924674,
        0.04999979503089235,
        0.04999949397470346,
        0.05000017341069554,
        0.04999970578355134,
        0.050000007173188775,
        0.05000007895565978,
        0.0500002821038347,
        0.049999963645046906,
        0.050000182537128575,
    ]
    w2t = [
        2.9642239447710655e-6,
        2.961944973825627e-6,
        2.9629758860954194e-6,
        2.9621198647043782e-6,
        0.4774596756646753,
        2.962111196491014e-6,
        0.522487009827231,
        2.961955390189544e-6,
        2.9610715351766874e-6,
        2.9623022912176586e-6,
        2.961664092172224e-6,
        2.9618533306684965e-6,
        2.9616551480715517e-6,
        2.9615877017937714e-6,
        2.958169191445656e-6,
        2.9613709715270997e-6,
        2.961884591018069e-6,
        2.961907348912632e-6,
        2.9615680496225393e-6,
        2.9641425861038096e-6,
    ]
    w3t = [
        0.018900022212015612,
        0.01928031532443446,
        0.023212048738661308,
        0.02206152234086144,
        0.21978018415254422,
        0.03599951609081468,
        0.2979971301026944,
        0.022175076567082907,
        0.020490675641144932,
        0.02084549856026369,
        0.022652315303891103,
        0.03219517175696326,
        0.05774684918202847,
        0.025739910401496997,
        0.054727368306260016,
        0.020762758185093346,
        0.023417365983190525,
        0.021432409038411905,
        0.02130108302456876,
        0.019282779087577927,
    ]
    w4t = [
        1.4742833133591588e-8,
        1.4861223252106373e-8,
        1.5225818305400315e-8,
        1.508493337786444e-8,
        4.1168522411894644e-7,
        1.2393323304878398e-8,
        0.9999993360743196,
        1.4009612732380198e-8,
        1.495965236761461e-8,
        1.4252294581887943e-8,
        1.3922507622838417e-8,
        1.2641000288249372e-8,
        1.0977725644197108e-8,
        1.335927687065135e-8,
        1.1174231589507641e-8,
        1.5555053465019993e-8,
        1.528453349126005e-8,
        1.4145162345449358e-8,
        1.5020870114054354e-8,
        1.4630403986901223e-8,
    ]

    @test isapprox(w1t, w1.weights, rtol = 6e-6)
    @test isapprox(w2t, w2.weights, rtol = 4e-3)
    @test isapprox(w3t, w3.weights, rtol = 3e-6)
    @test isapprox(w4t, w4.weights, rtol = 1e-5)

    rm = :Kurt
    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.05000006003933561,
        0.05000011653200388,
        0.0499999811494444,
        0.050000050665165635,
        0.04999995500833027,
        0.05000014736153201,
        0.04999933385945339,
        0.050000406481719405,
        0.049999839817117114,
        0.05000004649195019,
        0.05000037997924674,
        0.04999979503089235,
        0.04999949397470346,
        0.05000017341069554,
        0.04999970578355134,
        0.050000007173188775,
        0.05000007895565978,
        0.0500002821038347,
        0.049999963645046906,
        0.050000182537128575,
    ]
    w2t = [
        6.893398815874922e-5,
        6.855371428270107e-5,
        6.628236079821025e-5,
        6.835574840310483e-5,
        0.585672003438326,
        6.855718940774994e-5,
        0.3311306613035955,
        6.851564645943585e-5,
        6.826372362303255e-5,
        6.836791695653822e-5,
        6.853766219596829e-5,
        6.84725791337386e-5,
        6.767382951969928e-5,
        6.855528332333018e-5,
        6.82056945037667e-5,
        0.08205262353969407,
        5.2035718048637346e-5,
        6.855167372100891e-5,
        6.820774965042043e-5,
        6.864124019823871e-5,
    ]
    w4t = [
        3.3536502225529586e-6,
        4.0465937584935305e-6,
        2.1324486856943454e-5,
        2.747736243355171e-6,
        0.849447892318292,
        1.6937087965363221e-6,
        0.1503459647845561,
        1.4719091785845583e-6,
        2.311833554836331e-6,
        2.1544130265853426e-6,
        1.5151843011462864e-6,
        2.544614364489471e-6,
        1.3898188370338073e-6,
        1.6909006046248497e-6,
        2.298592389029479e-6,
        0.00013988306053585418,
        1.0553928506869986e-5,
        1.7570328907198234e-6,
        2.9398113539230116e-6,
        2.4656217303691158e-6,
    ]

    @test isapprox(w1t, w1.weights, rtol = 8e-2)
    @test isapprox(w2t, w2.weights, rtol = 3e-3)
    @test isapprox(w4t, w4.weights, rtol = 1e-4)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.05514023675290348,
        0.0416866820129796,
        0.015295307078387419,
        0.03845084467756283,
        0.107027161655376,
        0.006219390653204759,
        0.12539243400976255,
        0.06516138518589928,
        0.016994458968288797,
        0.05222021336611838,
        0.06471330154047447,
        0.0380060957230634,
        0.08332857729501973,
        0.05698827996620065,
        0.09277062357848598,
        0.02617070350792744,
        0.003210039999103272,
        0.06606992504871273,
        0.0025604867285477007,
        0.04259385225198163,
    ]
    w2t = [
        5.673389778842182e-5,
        5.4854601841120865e-5,
        5.6154244341095505e-5,
        5.534371040533454e-5,
        0.7368072959160795,
        5.9329307865146104e-5,
        0.2621521676883429,
        6.0108445585543555e-5,
        5.750232926295368e-5,
        5.7282054543322466e-5,
        5.8289628048143017e-5,
        6.0154371260665074e-5,
        7.345347667351535e-5,
        5.794173233515267e-5,
        5.5006866798384556e-5,
        4.5715621921449315e-5,
        5.647068499265451e-5,
        5.933969815701808e-5,
        5.9583600766359974e-5,
        5.7272122991445e-5,
    ]
    w4t = [
        1.4491957102956583e-6,
        1.963870612586247e-6,
        1.4666951569689761e-6,
        1.6788669179629582e-6,
        0.8533637070347461,
        2.1997720840140544e-6,
        0.14659121953304186,
        2.4312300157359173e-6,
        2.5125579689854973e-6,
        2.5151401746488557e-6,
        2.307600715733255e-6,
        2.4639804674260457e-6,
        4.058928277080703e-6,
        2.2203788526429885e-6,
        2.3824013904162583e-6,
        6.427099136388782e-6,
        2.118021812974547e-6,
        2.2764138106314845e-6,
        2.4277289650486426e-6,
        2.1735501425682977e-6,
    ]

    @test isapprox(w1t, w1.weights, rtol = 4e-4)
    @test isapprox(w2t, w2.weights, rtol = 3e-5)
    @test isapprox(w4t, w4.weights, rtol = 2e-4)
end

@testset "SKurt full" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 500),
            ),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SKurt

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        2.2200282140039283e-8,
        0.09964215574656404,
        1.6300824166522265e-8,
        4.63165816832639e-8,
        1.6779220406258903e-8,
        3.7735433420674933e-7,
        7.3180321770793506e-9,
        0.13536485860194186,
        1.3804583483815892e-8,
        1.965997835978975e-8,
        0.3900883700227835,
        2.247620162911149e-8,
        1.0290625389271409e-8,
        0.021330256430485865,
        3.849217435440389e-8,
        6.835155419338246e-8,
        1.6854945252646005e-7,
        0.19923237893580617,
        2.0936679689028987e-8,
        0.154341131431894,
    ]
    w2t = [
        2.7232968673423247e-9,
        5.718811368285733e-9,
        4.000102920298186e-9,
        2.7793775236333037e-9,
        0.638391527373887,
        6.285044187076252e-10,
        0.10348437499242895,
        2.5968189248342456e-9,
        2.9257470716553796e-9,
        1.5589348226859494e-9,
        2.601730364367239e-9,
        6.067601306689852e-10,
        3.7279334663615195e-10,
        1.1587589332786675e-9,
        3.931119759151813e-10,
        0.19207088993095656,
        0.0660531646900066,
        3.05297884688996e-9,
        8.65557924056538e-9,
        3.2394139419755e-9,
    ]
    w3t = [
        2.6258147841594324e-8,
        1.0510196750611228e-5,
        2.6591502613927026e-8,
        3.280385059657239e-8,
        0.32096951889858966,
        7.09825387162415e-9,
        0.004686133095413608,
        0.052737886206915795,
        2.134706624026086e-8,
        1.6907115727115198e-8,
        0.2447652925005578,
        6.322861232706945e-9,
        3.623490216467676e-9,
        1.8761795782112842e-8,
        4.040201399835902e-9,
        0.1358579526676256,
        0.24097154261984027,
        6.118713796332662e-7,
        8.050859425435343e-8,
        3.076800472898512e-7,
    ]
    w4t = [
        1.4746529194477868e-8,
        1.4864769608682732e-8,
        1.5229388429089284e-8,
        1.50888378601317e-8,
        4.1187243279095537e-7,
        1.2397646778384592e-8,
        0.9999993358150661,
        1.4012890214782413e-8,
        1.4963956846385612e-8,
        1.425685113917346e-8,
        1.3926054761890902e-8,
        1.2645807001881997e-8,
        1.0983683336761459e-8,
        1.336316426439152e-8,
        1.1179588853266343e-8,
        1.555837142452553e-8,
        1.5287788048913906e-8,
        1.4148691515576842e-8,
        1.5024741036182204e-8,
        1.4633740910885559e-8,
    ]

    @test isapprox(w1t, w1.weights, rtol = 2e-6)
    @test isapprox(w2t, w2.weights, rtol = 1e-6)
    @test isapprox(w3t, w3.weights, rtol = 5e-5)
    @test isapprox(w4t, w4.weights, rtol = 7e-7)

    rm = :SKurt
    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        2.2676362560691623e-8,
        0.0996421342535889,
        1.6695669124200388e-8,
        4.749490577435628e-8,
        1.7203585683887377e-8,
        3.8260732441694137e-7,
        7.595845849781463e-9,
        0.13536486814578497,
        1.4226827665765992e-8,
        2.010966034666888e-8,
        0.39008837268314783,
        2.330269305553652e-8,
        1.0706070210219683e-8,
        0.021330336321834082,
        3.976626665781445e-8,
        7.003385772301568e-8,
        1.717030599282148e-7,
        0.19923229872829892,
        2.145537034494752e-8,
        0.15434112428984598,
    ]
    w2t = [
        2.279737045244205e-9,
        5.771901218332141e-9,
        3.387884002743629e-9,
        2.5115264821642705e-9,
        0.5728216819925475,
        5.516564446309139e-10,
        0.08065971646272137,
        2.9387706277841195e-9,
        2.622948202536878e-9,
        1.4499409268216012e-9,
        2.9615486661760628e-9,
        5.058797079302989e-10,
        3.156087793387414e-10,
        1.0579032508200423e-9,
        3.311456033581071e-10,
        0.18372057739658174,
        0.16279797975600413,
        3.3632751799109867e-9,
        1.1190688664602479e-8,
        3.1517306354315127e-9,
    ]
    w3t = [
        1.2120617167886154e-8,
        0.016330390099980604,
        1.2745897585256626e-8,
        1.794897445072473e-8,
        0.2826492742576598,
        3.4153975952566197e-9,
        0.0006818265793849898,
        0.0709965121072632,
        1.0415781986044248e-8,
        8.368136834503685e-9,
        0.26894596566170376,
        3.025150956284808e-9,
        1.689658951600264e-9,
        9.4762119506344e-9,
        1.9094516696914915e-9,
        0.12089000718268687,
        0.21860821616946585,
        0.013240813534020115,
        3.686887958728598e-8,
        0.007656876423676218,
    ]
    w4t = [
        3.874261234059564e-12,
        5.104956991524285e-12,
        8.520577125817582e-12,
        7.291031371558718e-12,
        0.8503243999792303,
        2.3379624973096368e-12,
        0.1496755999366918,
        1.0705863587596301e-12,
        6.6612372661883604e-12,
        2.305336904457556e-12,
        6.777137172765885e-13,
        1.6338005023536322e-12,
        2.932461678879475e-12,
        7.923913410180556e-13,
        3.029721269616135e-12,
        1.5850611948318243e-11,
        9.708412524329252e-12,
        1.32142894413712e-12,
        7.161964000657596e-12,
        3.803398337169116e-12,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-6)
    @test isapprox(w2t, w2.weights, rtol = 1e-6)
    @test isapprox(w3t, w3.weights, rtol = 1e-5)
    @test isapprox(w4t, w4.weights, rtol = 1e-6)

    rm = :SKurt
    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        1.203645897370344e-7,
        0.0996403093189641,
        8.430275439205838e-8,
        2.1638785988942243e-7,
        8.425851677667726e-8,
        1.6851309910892206e-6,
        3.706926778985433e-8,
        0.13536464636126616,
        7.16600314312669e-8,
        1.0146813659867939e-7,
        0.390088782216095,
        1.188198111208054e-7,
        5.718546139992823e-8,
        0.021332017452715842,
        1.952145720030393e-7,
        3.0971972460165e-7,
        7.074450387949375e-7,
        0.19923010614357767,
        1.0635858784228959e-7,
        0.15434024312203776,
    ]
    w2t = [
        5.488045918341213e-7,
        4.0010429889635506e-7,
        3.649251037669618e-8,
        3.962071866824743e-6,
        0.5734888526554968,
        2.077214283207515e-6,
        0.0804750397320627,
        3.0723829186774463e-7,
        1.925712860713409e-6,
        3.0693977739681104e-6,
        1.9525722487537864e-7,
        1.4167487057569416e-6,
        5.101887518690194e-6,
        1.0667825881309905e-6,
        2.1027147138411767e-6,
        0.18383487935961787,
        0.16217431488824227,
        4.258293845173537e-6,
        2.856637102197894e-7,
        1.5897979598757086e-7,
    ]
    w4t = [
        5.196060109192332e-6,
        4.879700870790502e-6,
        5.102905879124217e-6,
        4.8337029112352186e-6,
        0.8550851947842012,
        4.784224808907169e-6,
        0.14482930798250065,
        4.829823898619645e-6,
        4.645031868839119e-6,
        4.668191729144458e-6,
        4.801087711908069e-6,
        4.861791985256761e-6,
        4.257838322291474e-6,
        4.790696295182476e-6,
        4.458511821550166e-6,
        4.003965429434134e-6,
        4.8446058170060465e-6,
        4.81102708811913e-6,
        4.727049801657758e-6,
        5.001016950102919e-6,
    ]

    @test isapprox(w1t, w1.weights, rtol = 1e-6)
    @test isapprox(w2t, w2.weights, rtol = 8e-2)
    @test isapprox(w4t, w4.weights, rtol = 3e-3)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SKurt
    kelly = :none
    portfolio.skrt_u = Inf
    portfolio.mu_l = Inf
    # portfolio.max_num_assets_kurt = 1

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r1 = calc_risk(portfolio, rm = rm)
    m1 = dot(portfolio.mu, w1.weights)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r2 = calc_risk(portfolio, rm = rm)
    m2 = dot(portfolio.mu, w2.weights)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r3 = calc_risk(portfolio, rm = rm)
    m3 = dot(portfolio.mu, w3.weights)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r4 = calc_risk(portfolio, rm = rm)
    m4 = dot(portfolio.mu, w4.weights)

    obj = :max_ret
    portfolio.skrt_u = r1 + 0.034375 * r1
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r5 = calc_risk(portfolio, rm = rm)
    m5 = dot(portfolio.mu, w5.weights)
    @test isapprox(w5.weights, w1.weights, rtol = 6e-2)
    @test isapprox(r5, r1, rtol = 2e-3)
    @test isapprox(m5, m1, rtol = 6e-2)

    portfolio.skrt_u = r2
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r6 = calc_risk(portfolio, rm = rm)
    m6 = dot(portfolio.mu, w6.weights)
    @test isapprox(w6.weights, w2.weights, rtol = 2e-1)
    @test isapprox(r6, r2, rtol = 7e-2)
    @test isapprox(m6, m2, rtol = 3e-2)

    portfolio.skrt_u = r3
    w7 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r7 = calc_risk(portfolio, rm = rm)
    m7 = dot(portfolio.mu, w7.weights)
    @test isapprox(w7.weights, w3.weights, rtol = 9e-2)
    @test isapprox(r7, r3, rtol = 6e-2)
    @test isapprox(m7, m3, rtol = 6e-2)

    portfolio.skrt_u = r4
    w8 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r8 = calc_risk(portfolio, rm = rm)
    m8 = dot(portfolio.mu, w8.weights)
    @test isapprox(w8.weights, w4.weights, rtol = 7e-7)
    @test isapprox(r8, r4, rtol = 9e-7)
    @test isapprox(m8, m4, rtol = 5e-9)

    portfolio.skrt_u = Inf
    portfolio.mu_l = Inf
    obj = :sharpe
    portfolio.skrt_u = r1 + 0.034375 * r1
    w13 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    r13 = calc_risk(portfolio, rm = rm)
    m13 = dot(portfolio.mu, w13.weights)
    @test isapprox(w13.weights, w1.weights, rtol = 6e-2)
    @test isapprox(r13, r1, rtol = 2e-3)
    @test isapprox(m13, m1, rtol = 6e-2)
end

@testset "SKurt relaxed" begin
    portfolio = Portfolio(
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
    portfolio.max_num_assets_kurt = 5
    rm = :SKurt

    kelly = :none
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :sharpe
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.049999999999999094,
        0.04999999999999947,
        0.05000000000000004,
        0.05000000000000004,
        0.05000000000000004,
        0.05000000000000019,
        0.05000000000000004,
        0.04999999999999997,
        0.05000000000000004,
        0.04999999999999997,
        0.05000000000000004,
        0.04999999999999997,
        0.05000000000000059,
        0.05000000000000004,
        0.04999999999999997,
        0.05000000000000004,
        0.05000000000000004,
        0.05000000000000019,
        0.05000000000000019,
        0.04999999999999997,
    ]
    w2t = [
        6.455953960568686e-7,
        6.455954089302699e-7,
        6.457332512076015e-7,
        6.459447514669266e-7,
        0.47081241480500097,
        6.455528508128865e-7,
        0.5291759634810956,
        6.452097254335057e-7,
        6.459710137599309e-7,
        6.458974454891232e-7,
        6.454286560507449e-7,
        6.456349842090027e-7,
        6.461553944769725e-7,
        6.454265996659646e-7,
        6.458188425856627e-7,
        6.45658545517431e-7,
        6.456318799574831e-7,
        6.453540962414621e-7,
        6.457530303041166e-7,
        6.453520312001723e-7,
    ]
    w3t = [
        0.018900022212015612,
        0.01928031532443446,
        0.023212048738661308,
        0.02206152234086144,
        0.21978018415254422,
        0.03599951609081468,
        0.2979971301026944,
        0.022175076567082907,
        0.020490675641144932,
        0.02084549856026369,
        0.022652315303891103,
        0.03219517175696326,
        0.05774684918202847,
        0.025739910401496997,
        0.054727368306260016,
        0.020762758185093346,
        0.023417365983190525,
        0.021432409038411905,
        0.02130108302456876,
        0.019282779087577927,
    ]
    w4t = [
        1.4742833286822958e-8,
        1.486122376700924e-8,
        1.522581841340201e-8,
        1.5084933116815453e-8,
        4.1168522004155456e-7,
        1.239332335129735e-8,
        0.999999336074322,
        1.4009613028332303e-8,
        1.4959652189675798e-8,
        1.425229516609218e-8,
        1.3922507598209601e-8,
        1.2641000163056434e-8,
        1.0977725787126606e-8,
        1.3359277114810495e-8,
        1.1174232119672775e-8,
        1.5555053571997216e-8,
        1.5284533759579546e-8,
        1.4145161845104134e-8,
        1.502086998953666e-8,
        1.46304039079864e-8,
    ]

    @test isapprox(w1t, w1.weights, rtol = 1e-6)
    @test isapprox(w2t, w2.weights, rtol = 2e-2)
    @test isapprox(w3t, w3.weights, rtol = 3e-6)
    @test isapprox(w4t, w4.weights, rtol = 1e-5)

    kelly = :approx
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.05017226262057069,
        0.05016351197056789,
        0.05007513173567151,
        0.04995252155033251,
        0.05004993916733474,
        0.05024184814141873,
        0.04903691714667429,
        0.050510868437165284,
        0.049846780236468755,
        0.05005729669421008,
        0.05055302249437607,
        0.049584318239207635,
        0.048970085535307066,
        0.050322335708133915,
        0.0495851406027916,
        0.04994840699223913,
        0.05015837445875571,
        0.050447019587693175,
        0.05002851978169167,
        0.05029569889938957,
    ]
    w2t = [
        6.893398815940391e-5,
        6.855371428308346e-5,
        6.628236079904856e-5,
        6.835574840408902e-5,
        0.5856720034383224,
        6.855718940835826e-5,
        0.3311306613035934,
        6.851564646012384e-5,
        6.826372362362984e-5,
        6.836791695700935e-5,
        6.853766219630029e-5,
        6.847257913390434e-5,
        6.767382952014367e-5,
        6.855528332382559e-5,
        6.820569450339143e-5,
        0.0820526235396916,
        5.2035718049200346e-5,
        6.855167372145803e-5,
        6.820774965097384e-5,
        6.864124019858442e-5,
    ]
    w4t = [
        3.35365022139209e-6,
        4.046593757194653e-6,
        2.132448685664194e-5,
        2.7477362414985734e-6,
        0.8494478923182281,
        1.693708795170189e-6,
        0.1503459647846432,
        1.4719091776181591e-6,
        2.311833554550305e-6,
        2.154413025264905e-6,
        1.5151842994922282e-6,
        2.5446143629983105e-6,
        1.3898188353187887e-6,
        1.690900602645287e-6,
        2.29859238735903e-6,
        0.0001398830605340488,
        1.055392850694175e-5,
        1.757032889116033e-6,
        2.9398113520313215e-6,
        2.4656217292334974e-6,
    ]

    @test isapprox(w1t, w1.weights, rtol = 3e-4)
    @test isapprox(w2t, w2.weights, rtol = 2e-2)
    @test isapprox(w4t, w4.weights, rtol = 1e-5)

    kelly = :exact
    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :utility
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    obj = :max_ret
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)

    w1t = [
        0.05514023675290348,
        0.0416866820129796,
        0.015295307078387419,
        0.03845084467756283,
        0.107027161655376,
        0.006219390653204759,
        0.12539243400976255,
        0.06516138518589928,
        0.016994458968288797,
        0.05222021336611838,
        0.06471330154047447,
        0.0380060957230634,
        0.08332857729501973,
        0.05698827996620065,
        0.09277062357848598,
        0.02617070350792744,
        0.003210039999103272,
        0.06606992504871273,
        0.0025604867285477007,
        0.04259385225198163,
    ]
    w2t = [
        5.673389778842182e-5,
        5.4854601841120865e-5,
        5.6154244341095505e-5,
        5.534371040533454e-5,
        0.7368072959160795,
        5.9329307865146104e-5,
        0.2621521676883429,
        6.0108445585543555e-5,
        5.750232926295368e-5,
        5.7282054543322466e-5,
        5.8289628048143017e-5,
        6.0154371260665074e-5,
        7.345347667351535e-5,
        5.794173233515267e-5,
        5.5006866798384556e-5,
        4.5715621921449315e-5,
        5.647068499265451e-5,
        5.933969815701808e-5,
        5.9583600766359974e-5,
        5.7272122991445e-5,
    ]
    w4t = [
        1.4491942108529963e-6,
        1.9638692122871565e-6,
        1.4666937030072718e-6,
        1.6788655942688532e-6,
        0.8533637068598828,
        2.1997706888034523e-6,
        0.14659121973352748,
        2.431228564690077e-6,
        2.5125565897768887e-6,
        2.5151387976043706e-6,
        2.307599297654686e-6,
        2.4639789857238116e-6,
        4.0589268710223e-6,
        2.220377451522181e-6,
        2.382400139290716e-6,
        6.42709749883426e-6,
        2.11802039333695e-6,
        2.276412387686396e-6,
        2.4277275648964346e-6,
        2.173548638545329e-6,
    ]

    @test isapprox(w1t, w1.weights, rtol = 4e-4)
    @test isapprox(w2t, w2.weights, rtol = 3e-5)
    @test isapprox(w4t, w4.weights, rtol = 2e-4)
end

@testset "OWA optimisations" begin
    portfolio = Portfolio(
        returns = returns[1:100, :],
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ), # "max_step_fraction" => 0.75
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            # :ECOS =>
            #     Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            # :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            # :HiGHS => Dict(
            #     :solver => HiGHS.Optimizer,
            #     :params => Dict("log_to_console" => false),
            # ),
        ),
    )
    asset_statistics!(portfolio)

    portfolio.owa_w = owa_wr(size(portfolio.returns, 1))
    portfolio.owa_u = Inf
    portfolio.wr_u = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; type = :trad, rm = :WR, obj = obj, rf = rf, l = l)
    r1 = calc_risk(portfolio; rm = :WR)
    w2 = opt_port!(portfolio; type = :trad, rm = :OWA, obj = obj, rf = rf, l = l)
    r2 = calc_risk(portfolio; rm = :WR)
    @test isapprox(w1.weights, w2.weights)

    obj = :sharpe
    w3 = opt_port!(portfolio; type = :trad, rm = :WR, obj = obj, rf = rf, l = l)
    r3 = calc_risk(portfolio; rm = :WR)
    w4 = opt_port!(portfolio; type = :trad, rm = :OWA, obj = obj, rf = rf, l = l)
    r4 = calc_risk(portfolio; rm = :WR)
    @test isapprox(w3.weights, w4.weights, rtol = 3e-5)

    obj = :max_ret
    portfolio.owa_u = r4
    w5 = opt_port!(portfolio; type = :trad, rm = :OWA, obj = obj, rf = rf, l = l)
    @test isapprox(w5.weights, w4.weights, rtol = 7e-8)

    obj = :sharpe
    portfolio.owa_u = r2
    w6 = opt_port!(portfolio; type = :trad, rm = :OWA, obj = obj, rf = rf, l = l)
    @test isapprox(w2.weights, w6.weights, rtol = 1e-9)

    portfolio = Portfolio(
        returns = returns[1:100, :],
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ), # "max_step_fraction" => 0.75
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            # :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            # :HiGHS => Dict(
            #     :solver => HiGHS.Optimizer,
            #     :params => Dict("log_to_console" => false),
            # ),
        ),
    )
    asset_statistics!(portfolio)

    portfolio.owa_w = owa_gmd(size(portfolio.returns, 1))
    rm = :GMD
    portfolio.gmd_u = Inf
    portfolio.owa_u = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r1 = calc_risk(portfolio; rm = rm)
    w2 = opt_port!(portfolio; type = :trad, rm = :OWA, obj = obj, rf = rf, l = l)
    r2 = calc_risk(portfolio; rm = rm)
    @test isapprox(w1.weights, w2.weights, rtol = 1e-3)

    obj = :sharpe
    w3 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r3 = calc_risk(portfolio; rm = rm)
    w4 = opt_port!(portfolio; type = :trad, rm = :OWA, obj = obj, rf = rf, l = l)
    r4 = calc_risk(portfolio; rm = rm)
    @test isapprox(w3.weights, w4.weights, rtol = 6e-5)

    obj = :max_ret
    portfolio.gmd_u = r3
    w5 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r5 = calc_risk(portfolio; rm = rm)
    portfolio.gmd_u = Inf
    portfolio.owa_u = r4
    w6 = opt_port!(portfolio; type = :trad, rm = :OWA, obj = obj, rf = rf, l = l)
    r6 = calc_risk(portfolio; rm = rm)
    @test isapprox(w5.weights, w6.weights, rtol = 4e-5)

    portfolio.gmd_u = Inf
    portfolio.owa_u = Inf
    obj = :sharpe
    portfolio.gmd_u = r1
    w7 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r7 = calc_risk(portfolio; rm = rm)
    portfolio.gmd_u = Inf
    portfolio.owa_u = r2
    w8 = opt_port!(portfolio; type = :trad, rm = :OWA, obj = obj, rf = rf, l = l)
    r8 = calc_risk(portfolio; rm = rm)
    @test isapprox(w7.weights, w8.weights, rtol = 2.9e-3)

    portfolio = Portfolio(
        returns = returns[1:100, :],
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ), # "max_step_fraction" => 0.75
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            # :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            # :HiGHS => Dict(
            #     :solver => HiGHS.Optimizer,
            #     :params => Dict("log_to_console" => false),
            # ),
        ),
    )
    asset_statistics!(portfolio)

    rm = :RG
    portfolio.rg_u = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r1 = calc_risk(portfolio; rm = rm)
    w1t = [
        1.0065872796863453e-11,
        2.1172409094512587e-12,
        4.656469039402821e-11,
        0.005579310299754643,
        0.10757009230887618,
        0.17194950789532434,
        0.028498664407906254,
        0.2758340964489913,
        5.86034619543313e-12,
        8.20789757894973e-12,
        0.1944512388249204,
        2.296762782961449e-12,
        3.236306403625521e-11,
        0.015606557975161818,
        1.9802846057251476e-12,
        2.256669267968185e-12,
        9.87317609550903e-12,
        9.774694044687755e-12,
        2.3565804667776783e-11,
        0.2005105316841385,
    ]
    @test isapprox(w1.weights, w1t, rtol = 1e-8)

    obj = :sharpe
    w2 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r2 = calc_risk(portfolio; rm = rm)
    w2t = [
        4.8040690556815e-11,
        0.1363058983570171,
        2.589189601257243e-11,
        3.0730462766521974e-11,
        0.21282385997488296,
        2.296010906040605e-11,
        5.069310961114895e-11,
        3.5887534233031304e-10,
        8.44325403872794e-12,
        2.1864060494758833e-11,
        2.231454700456383e-11,
        3.7326744964614845e-11,
        0.051052004528936296,
        1.9879352428374655e-12,
        8.762386015951151e-11,
        3.429891470282633e-11,
        2.6966289651644305e-9,
        0.2592901895134874,
        2.088484713846883e-11,
        0.3405280441571115,
    ]
    @test isapprox(w2.weights, w2t, rtol = 5e-8)

    obj = :max_ret
    portfolio.rg_u = r2
    w3 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r3 = calc_risk(portfolio; rm = rm)
    @test isapprox(w3.weights, w2.weights, rtol = 8e-8)

    obj = :sharpe
    portfolio.rg_u = r1
    w4 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r4 = calc_risk(portfolio; rm = rm)
    @test isapprox(w4.weights, w1.weights, rtol = 3e-6)

    portfolio = Portfolio(
        returns = returns[1:100, :],
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ), # "max_step_fraction" => 0.75
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            # :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            # :HiGHS => Dict(
            #     :solver => HiGHS.Optimizer,
            #     :params => Dict("log_to_console" => false),
            # ),
        ),
    )
    asset_statistics!(portfolio)

    rm = :RCVaR
    portfolio.rcvar_u = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r1 = calc_risk(portfolio; rm = rm)
    w1t = [
        0.04631605770586672,
        2.439474068922396e-12,
        2.7825313317216076e-11,
        0.009382704469318921,
        0.06381054054465991,
        0.1183875645301668,
        5.019947727736155e-12,
        0.2257900349650555,
        8.702889875205234e-12,
        5.2449517836383804e-12,
        0.34102324405814416,
        1.4189672404350314e-11,
        1.9365220911621207e-11,
        1.7980956445223487e-9,
        2.444259484631097e-12,
        1.7822926621285815e-11,
        3.762012963255535e-11,
        0.045856691705033044,
        1.8934623365967657e-11,
        0.1494331600640499,
    ]
    @test isapprox(w1.weights, w1t, rtol = 3e-7)

    obj = :sharpe
    w2 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r2 = calc_risk(portfolio; rm = rm)
    w2t = [
        2.344861032727787e-11,
        0.34443801449432965,
        1.2576648160903799e-11,
        1.666939063677421e-11,
        0.03627867821942035,
        1.456685809365143e-11,
        2.664961126288338e-11,
        5.156646589690019e-10,
        1.0612812094460367e-11,
        3.910100694108692e-11,
        3.667704780486475e-14,
        2.4261883811051277e-12,
        0.02002758605086437,
        1.2111762715383566e-11,
        4.67053785267986e-11,
        1.1489072256406304e-11,
        7.414396737077622e-11,
        0.3127803065776162,
        1.2082411621514808e-11,
        0.28647541383948427,
    ]
    @test isapprox(w2.weights, w2t, rtol = 2e-6)

    obj = :max_ret
    portfolio.rcvar_u = r2
    w3 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r3 = calc_risk(portfolio; rm = rm)
    @test isapprox(w3.weights, w2.weights, rtol = 3e-4)

    obj = :sharpe
    portfolio.rcvar_u = r1
    w4 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r4 = calc_risk(portfolio; rm = rm)
    @test isapprox(w4.weights, w1.weights, rtol = 5e-6)

    portfolio = Portfolio(
        returns = returns[1:100, :],
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ), # "max_step_fraction" => 0.75
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            # :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            # :HiGHS => Dict(
            #     :solver => HiGHS.Optimizer,
            #     :params => Dict("log_to_console" => false),
            # ),
        ),
    )
    asset_statistics!(portfolio)

    rm = :TG
    portfolio.tg_u = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r1 = calc_risk(portfolio; rm = rm)
    w1t = [
        2.9629604999600752e-12,
        1.3172830824827146e-12,
        0.05607738995137136,
        0.05649863708151018,
        0.018854132355071794,
        7.074417451706739e-12,
        2.4054872616101202e-14,
        0.2972001759702997,
        1.6316785013199786e-12,
        4.338487661044525e-13,
        0.37575766068547595,
        2.1104229834455135e-13,
        0.0019134826765316978,
        0.06033991076180079,
        6.938347228404816e-13,
        3.643319918681142e-13,
        1.415563364180365e-12,
        0.13335861049810332,
        1.3607104480014456e-12,
        2.345483649462695e-12,
    ]
    @test isapprox(w1.weights, w1t, rtol = 5e-8)

    obj = :sharpe
    w2 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r2 = calc_risk(portfolio; rm = rm)
    w2t = [
        2.653130304349078e-10,
        0.18585547050128823,
        1.5031499112610813e-11,
        2.6864417032965195e-11,
        0.044607198059599375,
        1.0617121196363914e-10,
        4.867342571782538e-10,
        0.2666938887315908,
        8.178071147385676e-11,
        4.0438182027182395e-10,
        2.6432147383329667e-10,
        1.9820872246245402e-10,
        0.07200195552437774,
        1.1952515770258174e-10,
        7.582791762421663e-10,
        6.329116056527242e-10,
        3.7479639077868744e-10,
        0.25259697810310033,
        8.450875220544656e-11,
        0.17824450526121516,
    ]
    @test isapprox(w2.weights, w2t, rtol = 8e-6)

    obj = :max_ret
    portfolio.tg_u = r2
    w3 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r3 = calc_risk(portfolio; rm = rm)
    @test isapprox(w3.weights, w2.weights, rtol = 9e-6)

    obj = :sharpe
    portfolio.tg_u = r1
    w4 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r4 = calc_risk(portfolio; rm = rm)
    @test isapprox(w4.weights, w1.weights, rtol = 6e-7)

    portfolio = Portfolio(
        returns = returns[1:100, :],
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ), # "max_step_fraction" => 0.75
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            # :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            # :HiGHS => Dict(
            #     :solver => HiGHS.Optimizer,
            #     :params => Dict("log_to_console" => false),
            # ),
        ),
    )
    asset_statistics!(portfolio)

    rm = :RTG
    portfolio.rtg_u = Inf

    obj = :min_risk
    w1 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r1 = calc_risk(portfolio; rm = rm)
    w1t = [
        0.02315418871225564,
        3.122611999363298e-14,
        0.007325004771381733,
        9.275640969720347e-11,
        0.08929240913364729,
        0.06773822313206025,
        0.004091970458800883,
        0.21769907714189168,
        6.666591902366087e-12,
        1.226463379185659e-11,
        0.37541981893837634,
        4.939341219546928e-12,
        2.8792655667398515e-12,
        1.5495866519268611e-10,
        1.968261186822617e-11,
        5.658963367983038e-12,
        6.3901457915964e-12,
        0.00797806191374815,
        7.16449046986415e-12,
        0.20730124548444573,
    ]
    @test isapprox(w1.weights, w1t, rtol = 1e-8)

    obj = :sharpe
    w2 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r2 = calc_risk(portfolio; rm = rm)
    w2t = [
        8.446019300435742e-12,
        0.24942121610745158,
        4.414767671325562e-12,
        6.514630905923817e-12,
        0.11403343654728389,
        5.2559199788330226e-12,
        9.613683260640885e-12,
        4.512367266433344e-11,
        4.0170137588017504e-12,
        1.1590616822632309e-11,
        2.789163256079482e-13,
        4.955618365720267e-13,
        0.02673618318770693,
        4.437343601822197e-12,
        1.665942288342791e-11,
        8.038261392118368e-12,
        5.1950888122255964e-11,
        0.3207065925852136,
        4.501763836989433e-12,
        0.28910257139100554,
    ]
    @test isapprox(w2.weights, w2t, rtol = 7e-8)

    obj = :max_ret
    portfolio.rtg_u = r2
    w3 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r3 = calc_risk(portfolio; rm = rm)
    @test isapprox(w3.weights, w2.weights, rtol = 7e-8)

    obj = :sharpe
    portfolio.rtg_u = r1
    w4 = opt_port!(portfolio; type = :trad, rm = rm, obj = obj, rf = rf, l = l)
    r4 = calc_risk(portfolio; rm = rm)
    @test isapprox(w4.weights, w1.weights, rtol = 3e-6)
end

@testset "Constraints" begin
    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SD
    kelly = :none
    portfolio.short = true

    portfolio.short_u = 0.3
    portfolio.long_u = 1.3
    portfolio.sum_short_long = portfolio.long_u - portfolio.short_u

    obj = :min_risk
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    w1t = [
        0.003587824703236382,
        0.03755523524543413,
        0.01767899875983944,
        0.03308647263306699,
        0.012486408438682959,
        0.05379626508860939,
        -0.00971112965037303,
        0.14121875540611653,
        -0.010891701181792467,
        0.018593623803989382,
        0.28368308550277926,
        -0.0211415604757037,
        -0.009133130104719252,
        0.14588497161233518,
        0.0007734050697862307,
        0.025431747995892363,
        0.014569199255512567,
        0.20358680770171717,
        -0.06421886110150134,
        0.12316358129715226,
    ]
    @test isapprox(w1t, w1.weights, rtol = 6e-4)
    @test all(w1.weights[w1.weights .< 0] .>= -portfolio.short_u)
    @test all(w1.weights[w1.weights .> 0] .<= portfolio.long_u)
    @test isapprox(sum(w1.weights), portfolio.sum_short_long)

    obj = :sharpe
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    w2t = [
        2.5267713200790443e-8,
        2.3557997670125438e-7,
        5.66536481013349e-7,
        3.683998080561952e-7,
        0.47410790094874317,
        -0.05667373527599564,
        0.06488650583079525,
        0.03063705375737504,
        8.798375683807772e-8,
        9.676849395496075e-8,
        0.03837776267163533,
        -0.07625577190446572,
        -0.046402349834757724,
        1.4595903097759926e-8,
        -0.12066765314099208,
        0.1802766001824138,
        0.24781714990391104,
        4.737709722232439e-7,
        0.26389424388040816,
        4.240771538000945e-7,
    ]
    @test isapprox(w2t, w2.weights, rtol = 6e-5)
    @test all(w2.weights[w2.weights .< 0] .>= -portfolio.short_u)
    @test all(w2.weights[w2.weights .> 0] .<= portfolio.long_u)
    @test isapprox(sum(w2.weights), portfolio.sum_short_long)

    portfolio.short_u = 0.11
    portfolio.long_u = 1.23
    portfolio.sum_short_long = portfolio.long_u - portfolio.short_u

    obj = :min_risk
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    w3t = [
        0.004299730250331464,
        0.04091408957120287,
        0.019260807512482624,
        0.03658001164838576,
        0.014225762488882734,
        0.05746343605850661,
        -0.00996248424417953,
        0.15798886098309115,
        -0.01493472945184608,
        0.017869360599409075,
        0.3178463049624519,
        -0.02187044093991696,
        -0.00946997632593845,
        0.1600574313436442,
        0.0003116825695443617,
        0.027446705294999874,
        0.012821050896696927,
        0.22586523897569308,
        -0.053761904352271,
        0.13704906215838034,
    ]
    @test isapprox(w3t, w3.weights, rtol = 3e-4)
    @test all(w3.weights[w3.weights .< 0] .>= -portfolio.short_u)
    @test all(w3.weights[w3.weights .> 0] .<= portfolio.long_u)
    @test isapprox(sum(w3.weights), portfolio.sum_short_long)

    obj = :sharpe
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    w4t = [
        2.2549529814806565e-8,
        1.1286803738583013e-7,
        1.370876887163495e-7,
        9.945581252127811e-8,
        0.540148036366995,
        -1.6419705392134607e-8,
        0.07154753778886971,
        2.3451969702780645e-7,
        5.56563020211513e-8,
        3.688083646091651e-8,
        1.5644516846888938e-7,
        -1.7332962067334847e-7,
        -0.030700417446881203,
        1.121158663330529e-8,
        -0.07929922509667617,
        0.179809683050173,
        0.25457466107429866,
        1.1853741694129262e-7,
        0.1839188261411397,
        1.0266209809286423e-7,
    ]
    @test isapprox(w4t, w4.weights, rtol = 9e-6)
    @test all(w4.weights[w4.weights .< 0] .>= -portfolio.short_u)
    @test all(w4.weights[w4.weights .> 0] .<= portfolio.long_u)
    @test isapprox(sum(w4.weights), portfolio.sum_short_long)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SD
    kelly = :none

    obj = :sharpe
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    portfolio.turnover = 0.05
    portfolio.turnover_weights = copy(w1.weights)

    obj = :min_risk
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test all(abs.(w2.weights - portfolio.turnover_weights) .<= portfolio.turnover)

    portfolio.turnover = Inf
    obj = :min_risk
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    portfolio.turnover = 0.031
    portfolio.turnover_weights = copy(w3.weights)

    obj = :sharpe
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test all(abs.(w4.weights - portfolio.turnover_weights) .<= portfolio.turnover)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SD
    kelly = :none

    obj = :sharpe
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    portfolio.kind_tracking_err = :weights
    portfolio.tracking_err = 0.0005
    # portfolio.tracking_err_returns = Vector{Float64}(undef, 0)
    portfolio.tracking_err_weights = copy(w1.weights)

    obj = :min_risk
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    w2t = [
        0.00012937963825922666,
        0.0017993432603443194,
        0.0005925682560396107,
        0.0014774704811308095,
        0.4945923286068745,
        0.0017603635964306935,
        0.06023870178607093,
        0.0064070969591296075,
        7.0320195517678365e-9,
        0.0005641226614673089,
        0.013583525225151482,
        8.268177529019035e-10,
        5.371519497662845e-10,
        0.006554406848381595,
        4.8956778188903165e-9,
        0.13751830557487418,
        0.18830514952012686,
        0.009650044091409016,
        0.07135039176230074,
        0.00547678844034211,
    ]
    @test isapprox(w2t, w2.weights, rtol = 3e-4)
    @test norm(
        Matrix(returns[!, 2:end]) * (w2.weights - portfolio.tracking_err_weights),
        2,
    ) / sqrt(nrow(returns) - 1) <= portfolio.tracking_err

    portfolio.tracking_err = Inf
    obj = :min_risk
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    portfolio.kind_tracking_err = :weights
    portfolio.tracking_err = 0.0003
    # portfolio.tracking_err_returns = Vector{Float64}(undef, 0)
    portfolio.tracking_err_weights = copy(w3.weights)

    obj = :sharpe
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    w4t = [
        0.002928171421595965,
        0.030370445108063172,
        0.012247780040264142,
        0.02749763537332101,
        0.021654631904835957,
        0.016980527231975316,
        0.0011918329883360305,
        0.13840549677476377,
        1.6022609102027843e-8,
        7.574054535367988e-8,
        0.2891049073740062,
        2.5460439167122798e-9,
        1.973348841914057e-9,
        0.11538863223449589,
        1.984445672057219e-9,
        0.016999247436935577,
        0.006939722662829445,
        0.19329596637910745,
        0.010767499407052895,
        0.11622740539542435,
    ]
    @test isapprox(w4t, w4.weights, rtol = 4e-4)
    @test norm(
        Matrix(returns[!, 2:end]) * (w4.weights - portfolio.tracking_err_weights),
        2,
    ) / sqrt(nrow(returns) - 1) <= portfolio.tracking_err

    portfolio.tracking_err = Inf
    portfolio.tracking_err = 0.007
    portfolio.kind_tracking_err = :returns
    portfolio.tracking_err_returns = vec(returns[!, argmax(w1.weights) + 1])
    obj = :sharpe
    w5 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test norm(Matrix(returns[!, 2:end]) * w5.weights - portfolio.tracking_err_returns, 2) /
          sqrt(nrow(returns) - 1) <= portfolio.tracking_err

    portfolio.tracking_err = Inf
    portfolio.tracking_err = 0.003
    portfolio.kind_tracking_err = :returns
    portfolio.tracking_err_returns = vec(returns[!, argmax(w1.weights) + 1])
    obj = :min_risk
    w6 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test norm(Matrix(returns[!, 2:end]) * w6.weights - portfolio.tracking_err_returns, 2) /
          sqrt(nrow(returns) - 1) <= portfolio.tracking_err

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
        ),
    )
    asset_statistics!(portfolio)
    rm = :SD
    kelly = :none

    obj = :sharpe
    portfolio.min_number_effective_assets = 5
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    w1t = [
        6.375514180278588e-8,
        0.027882159293922966,
        0.0557336742327969,
        0.013037690322754812,
        0.3584225067124871,
        8.318630557899562e-9,
        0.06661686835404312,
        0.016046763373863127,
        2.0829497312333203e-7,
        5.0307208020413264e-8,
        3.330340050361633e-5,
        6.346688680901949e-9,
        4.289403834579757e-9,
        1.8373274827340437e-8,
        4.353301999181876e-9,
        0.1362408783189624,
        0.1709714790176626,
        0.008864615309020838,
        0.11908068551285773,
        0.027069012112501845,
    ]
    @test isapprox(w1t, w1.weights, rtol = 1e-4)

    obj = :min_risk
    portfolio.min_number_effective_assets = 11
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    w2t = [
        0.04106779420433762,
        0.050740893930896,
        0.03185109806650701,
        0.031446388115378206,
        0.026970772260889782,
        0.07201438611372805,
        7.582110751791772e-9,
        0.12559431248317532,
        7.879051411431053e-8,
        0.03819801450124479,
        0.15650662475502,
        1.32980104284167e-8,
        1.674330566233696e-8,
        0.10145056805736621,
        0.011556811471121392,
        0.03455656383781936,
        0.04719511108868762,
        0.12383171848358535,
        0.021139976473931545,
        0.08587884974237084,
    ]
    @test isapprox(w2t, w2.weights, rtol = 5e-4)

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => (Clarabel.Optimizer),
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :HiGHS => Dict(
                :solver => HiGHS.Optimizer,
                :params => Dict("log_to_console" => false),
            ),
        ),
    )
    asset_statistics!(portfolio)
    rm = :CDaR
    kelly = :none

    obj = :min_risk
    portfolio.max_number_assets = 0
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    portfolio.max_number_assets = 5
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test sum(w2.weights[w2.weights .> length(w2.weights) * eps()]) <=
          portfolio.max_number_assets

    obj = :sharpe
    portfolio.max_number_assets = 0
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    portfolio.max_number_assets = 4
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test sum(w4.weights[w4.weights .> length(w4.weights) * eps()]) <=
          portfolio.max_number_assets

    portfolio = Portfolio(
        returns = returns,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false),
            ),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 0)),
            :HiGHS => Dict(
                :solver => HiGHS.Optimizer,
                :params => Dict("log_to_console" => false),
            ),
        ),
    )
    asset_statistics!(portfolio)
    rm = :CDaR
    kelly = :none

    obj = :min_risk
    portfolio.max_number_assets = 5
    w1 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test sum(w1.weights[w1.weights .> 1e-6]) <= portfolio.max_number_assets

    obj = :sharpe
    portfolio.max_number_assets = 3
    w2 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test sum(w2.weights[w2.weights .> 1e-6]) <= portfolio.max_number_assets

    portfolio.short = true
    portfolio.short_u = 0.4
    portfolio.long_u = 0.6
    portfolio.sum_short_long = portfolio.long_u - portfolio.short_u

    obj = :min_risk
    portfolio.max_number_assets = 10
    w3 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test sum(w3.weights[abs.(w3.weights) .> 1e-6]) <= portfolio.max_number_assets

    obj = :sharpe
    portfolio.max_number_assets = 11
    w4 = opt_port!(portfolio; rm = rm, obj = obj, kelly = kelly, rf = rf, l = l)
    @test sum(w4.weights[abs.(w4.weights) .> 1e-6]) <= portfolio.max_number_assets
end
