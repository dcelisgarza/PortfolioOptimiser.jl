using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

path = joinpath(@__DIR__, "assets/stock_prices.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Variance" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = Variance()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)
    rc3 = risk_contribution(portfolio; type = :RB, rm = SD())
    lrc3, hrc3 = extrema(rc3)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)
    rc4 = risk_contribution(portfolio; type = :RB, rm = SD())
    lrc4, hrc4 = extrema(rc4)

    w1t = [0.050634183840582665, 0.05124435171747801, 0.046907422021349704,
           0.043691027756881995, 0.04571138633336385, 0.05615011293512599,
           0.027633574202234036, 0.07705873910745298, 0.039490700950159684,
           0.04723134145579693, 0.08435961979696782, 0.03385542510757872,
           0.02754539294572688, 0.06206861918855729, 0.0356363624134368,
           0.04413318082759848, 0.05085092588478587, 0.07142214127256237,
           0.045296265218623843, 0.05907922702373602]
    w2t = [0.005640351869226139, 0.011009830028336819, 0.015583508340353323,
           0.019371225492442787, 0.025439557960201274, 0.03265950822082883,
           0.020372156441657057, 0.05956855992983157, 0.03298486504883764,
           0.044832397034598244, 0.08740795990577546, 0.03823116852595384,
           0.031439326532196166, 0.07960566295489545, 0.046145111685504975,
           0.06097590174302487, 0.08329483037608362, 0.1163890903287402,
           0.07928031378039686, 0.1097686738011149]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 5.0e-4)
    @test isapprox(hrc3 / lrc3, hrc1 / lrc1, rtol = 5.0e-10)
    @test isapprox(hrc4 / lrc4, hrc2 / lrc2, rtol = 5.0e-10)

    portfolio.risk_budget = fill(inv(20), 20)
    portfolio.risk_budget[1] = 5
    w3 = optimise!(portfolio, RB())
    rc3 = risk_contribution(portfolio; type = :RB)
    lrc5, hrc5 = extrema(rc3)
    @test isapprox(hrc5 / lrc5, 100, rtol = 0.0005)
end

@testset "MAD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = MAD()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.056165964502088175, 0.051042982720711054, 0.05044269924957646,
           0.04081758598468407, 0.04879062416406081, 0.05670906578708418,
           0.024225097882232604, 0.07090601622033053, 0.04019526097461559,
           0.046491141941875785, 0.08137941000594119, 0.03158826824340507,
           0.023961547974514278, 0.06224837702712137, 0.034594500642199065,
           0.04206573192801628, 0.05465319092541586, 0.07239069554127316,
           0.04770965237555548, 0.06362218590929908]
    w2t = [0.0063961952250897024, 0.01259233570812754, 0.01733507546160919,
           0.019133369264133202, 0.028375436997384727, 0.031889180424327784,
           0.019206369765627646, 0.055400873920331425, 0.03337728881511272,
           0.045145056375019046, 0.08075326190057375, 0.03493485143072039,
           0.02578004094261155, 0.07995311292670913, 0.044767477387302204,
           0.05590147350787757, 0.0922934334228999, 0.12029607659061349,
           0.08192688910102709, 0.11454220083290197]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.01)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end

@testset "SSD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = SSD()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.0505365158950737, 0.052703283951504455, 0.045048046549910084,
           0.04045295744966065, 0.045697002815612627, 0.05415916135002527,
           0.026500694979854727, 0.07776427484079386, 0.038245431266344485,
           0.04454506838012133, 0.08434139677792266, 0.035465952093264266,
           0.029930574993684234, 0.06261437237520955, 0.039240510164013316,
           0.04565079412546282, 0.04984577343193089, 0.07412774594814102,
           0.044280216607099104, 0.05885022600437107]
    w2t = [0.005428687871925218, 0.011165997744344677, 0.014457877462411622,
           0.017041104189619228, 0.024073101414898677, 0.03133995492451869,
           0.019075804783727716, 0.05919921912726392, 0.03194488914093804,
           0.04195795744473274, 0.08814684328464706, 0.038997168516058264,
           0.03418175861909682, 0.07927633692959091, 0.05026058595905142,
           0.06339052851645095, 0.08186275704784614, 0.12155956696235388,
           0.07827595148631002, 0.10836390857421406]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "FLPM" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = FLPM()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.054267058551296765, 0.050962383681933655, 0.05095160961924024,
           0.040031952588503915, 0.055688221897220606, 0.048538866052981175,
           0.024657938379511343, 0.07063025110912476, 0.04179080715877214,
           0.046065793108843366, 0.08265213569184943, 0.02798877922070947,
           0.021904092970555615, 0.05910029975771662, 0.030133599242665763,
           0.04527830359018147, 0.05792703433052645, 0.07483229355619783,
           0.04995841412094677, 0.06664016537122244]
    w2t = [0.006878868232648054, 0.012951149393094247, 0.018381735576801738,
           0.01831469262744488, 0.03608453233241397, 0.02719604369495882,
           0.020207074438070734, 0.052774398319475106, 0.033761212787344494,
           0.04423765658397453, 0.08235670938100217, 0.031749539125101554,
           0.024157585599707543, 0.07391869180768487, 0.03806447452897265,
           0.05744778736381905, 0.0995482799675567, 0.12220053760565629,
           0.08483102562198958, 0.11493800501228298]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-6)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "SLPM" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = SLPM()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05078127940733007, 0.053709289520370206, 0.045765499111031845,
           0.04092124166773404, 0.04906350161785088, 0.05034642655205059,
           0.027363313590193325, 0.07790700226241161, 0.03854385546416795,
           0.04420307984253257, 0.08430309761074495, 0.03395810327214033,
           0.028244948471502513, 0.06056629784190088, 0.036508496019863564,
           0.04757367293185978, 0.051284241663104566, 0.07441627002609454,
           0.04500824504878194, 0.05953213807833387]
    w2t = [0.0054916073854252906, 0.011427939296813336, 0.01482057459914487,
           0.017304371379962005, 0.02632048335008413, 0.029055266030112614,
           0.019768193133877558, 0.05921585232078186, 0.03219069820365385,
           0.04164854622777103, 0.08829470628803716, 0.037308239885460066,
           0.03234413972831064, 0.07667318635649392, 0.047034271680223304,
           0.06579073258270003, 0.08444743344829933, 0.12166293164184272,
           0.07958310033100874, 0.10961772612999747]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "WR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = WR()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04375992127369364, 0.08228122776101159, 0.04366451242725762,
           0.05731640168086471, 0.055185695855985165, 0.048871017501235826,
           0.03646064754099084, 0.04988930973188119, 0.03936615113766571,
           0.0485642283307299, 0.05738085344238412, 0.03891193081729503,
           0.04249668720010548, 0.03781638421341334, 0.04825300489968052,
           0.04988261900333607, 0.06042227215750082, 0.04103921463843018,
           0.0417646563083915, 0.07667326407814674]
    w2t = [0.004526357396910154, 0.01599857045877911, 0.012809686497010154,
           0.0224641869782515, 0.02231992635098638, 0.03129797064790171,
           0.031228767822189207, 0.03944104564090143, 0.03481611261538421,
           0.04185764502618059, 0.06429113998066102, 0.04585693336787605,
           0.060787164422027896, 0.05350837937637914, 0.059728973754469124,
           0.08421019243195493, 0.1021367690898045, 0.07535633202462913, 0.0755885395357398,
           0.12177530658196385]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.5)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)
end

@testset "RG" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = RG()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.03141965254800806, 0.049776721726429926, 0.04187113003186752,
           0.10926863224687966, 0.04030098770429187, 0.05222722528925385,
           0.038366387933800636, 0.059028063952825775, 0.03891563515951284,
           0.06241328458350651, 0.06227667392466888, 0.04249056094979058,
           0.04679952255282753, 0.0365740694818197, 0.06496223086410119,
           0.035639245163702385, 0.04490656092222676, 0.04669630835098361,
           0.04181063398578818, 0.054256472627714575]
    w2t = [0.003211689191745896, 0.00978870997054521, 0.012411317875175435,
           0.0435375180954373, 0.017523030272390296, 0.03363765894379565,
           0.032651841943521866, 0.047222641161857266, 0.03475574810212963,
           0.053549549435304954, 0.07042385745891029, 0.050622193587708024,
           0.06746636445003365, 0.05203641541883015, 0.07903051847142867,
           0.058990450867242764, 0.0761656566896762, 0.08674390154295081,
           0.07669852275934985, 0.09353241376196599]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)
end

@testset "CVaR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = CVaR()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.049387129895525614, 0.04880041958645412, 0.0417360796429531,
           0.03840901174332172, 0.04184139932560737, 0.053780063457719865,
           0.026424802585162756, 0.0773672149813708, 0.03544241306631173,
           0.04094196008260876, 0.09252539697406174, 0.042683400449354016,
           0.03238760400585853, 0.06577416488603098, 0.040197811307927274,
           0.04220372473527037, 0.0487231801118939, 0.08071251851657814,
           0.04276785407281033, 0.05789385057317895]
    w2t = [0.004958218494549572, 0.010722455152671753, 0.013513167408279242,
           0.01594308556557821, 0.022147345900741813, 0.02923512993981011,
           0.019156913359935097, 0.062208957439668394, 0.029581996183043274,
           0.04034973031635409, 0.09432467793690423, 0.04011258944592267,
           0.03579995136507606, 0.07561402622969399, 0.05081433564690067,
           0.06895367453572843, 0.0771448221907743, 0.12898605877835598,
           0.07490792221442769, 0.10552494189558441]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-4)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end

@testset "CVaRRG" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = CVaRRG()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05026686398193886, 0.047782622692513606, 0.04260414924297959,
           0.04204042069840117, 0.043927234385394955, 0.05759584605084417,
           0.029426794666817736, 0.07979509943999454, 0.037595145950307364,
           0.045130733958616796, 0.08755786411737047, 0.03581585908519193,
           0.030136299317118122, 0.06642346796853255, 0.037108167343166486,
           0.04244801008052913, 0.047789552757646195, 0.07556148959170042,
           0.04385391604113499, 0.05714046262980084]
    w2t = [0.005234591248322739, 0.010282579695658864, 0.014110348375517756,
           0.0190038549939815, 0.02405563985464297, 0.03235083075563166,
           0.02331091511917491, 0.060932486850669286, 0.032513361694698366,
           0.044930821162734225, 0.09081652966678729, 0.03890171737036649,
           0.034652969508558985, 0.08231660755870655, 0.04604427381153322,
           0.05818959051939687, 0.08044565782331387, 0.12044858504003186,
           0.07810716559208354, 0.10335147335818895]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end

@testset "EVaR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = EVaR()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.046441175139679075, 0.0636635926954678, 0.0437905327100388,
           0.05092290582010278, 0.0464093742619314, 0.05061665593049196,
           0.03894587929172149, 0.0658674606383409, 0.03842120337585926,
           0.044461528797653645, 0.06810063277922151, 0.04257098857986524,
           0.04132606801031959, 0.049160675556679065, 0.04438749327363125,
           0.053443618688456575, 0.05369210088508193, 0.05226821690404457,
           0.04178234191426879, 0.06372755474714445]
    w2t = [0.004649064633994313, 0.013024128816592357, 0.013065166884888157,
           0.020518113549704037, 0.022282631649315945, 0.030139092451586413,
           0.02817712722231638, 0.04961150619044857, 0.0332088114654529,
           0.04172011248113971, 0.07294398194510515, 0.04732479787078266,
           0.051090244842984446, 0.06410260111407641, 0.05751225215578386,
           0.07959987570465571, 0.08923067053518313, 0.08897533037236685,
           0.0757163816109926, 0.11710810850263045]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-4)
    @test isapprox(hrc1 / lrc1, 1, rtol = 1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.001)
end

@testset "RLVaR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = RLVaR()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))

    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04549547503909051, 0.07569592076334769, 0.04395758113608033,
           0.055867217304234974, 0.04924963189142022, 0.0506624482370855,
           0.04110800894258115, 0.05449490458881057, 0.03937874870789796,
           0.04560096085650905, 0.06132309806744196, 0.04070766992118493,
           0.043671581895838724, 0.04184374433088353, 0.045099565165730104,
           0.05279627171949139, 0.05840798437928582, 0.04473953918265079,
           0.041599376780132175, 0.06830027109030251]
    w2t = [0.004579089349935645, 0.01515744036868148, 0.012920957983932928,
           0.022042709924231012, 0.02225228741879495, 0.031006239647658095,
           0.03102400208745146, 0.041887878568255606, 0.034465254698941274,
           0.041732701803395736, 0.06662190190027566, 0.046533407283221626,
           0.05758114883141428, 0.05644208250978566, 0.0588123028042114,
           0.08358010865542893, 0.09852180375978169, 0.078800315375297, 0.07568030473407306,
           0.12035806229523235]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-7)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.5)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.25)
end

@testset "EVaR < RLVaR < WR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = RLVaR(; kappa = 5e-3)
    w1 = optimise!(portfolio, RB(; rm = rm))
    rm = RLVaR(; kappa = 1 - 5e-3)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rm = EVaR()
    w3 = optimise!(portfolio, RB(; rm = rm))
    rm = WR()
    w4 = optimise!(portfolio, RB(; rm = rm))

    if !Sys.isapple()
        @test isapprox(w1.weights, w3.weights, rtol = 0.01)
    end
    @test isapprox(w2.weights, w4.weights, rtol = 1.0e-4)
end

@testset "MDD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = MDD()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = reverse(1:size(portfolio.returns, 2))
    w2 = optimise!(portfolio, RB(; rm = rm))

    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.08321949009724784, 0.03470001118521989, 0.04179139342635433,
           0.02762932916548522, 0.03998413339708836, 0.056731619169781214,
           0.02234291286299648, 0.0941827990004656, 0.027866356710967193,
           0.03558171362724042, 0.15688809804971873, 0.03203069604160037,
           0.03750492974529328, 0.03997509572794979, 0.02430680394837502,
           0.040543185852144455, 0.044693066504041944, 0.06510424558388941,
           0.035401109346484265, 0.05952301055765607]
    w2t = [0.10304779780409785, 0.06581655353400091, 0.07099464597251968,
           0.048240613246532775, 0.06732650074642522, 0.0693341886181786,
           0.02515019124685212, 0.12492297855493846, 0.032397240862830926,
           0.04506854069981592, 0.13647668024488732, 0.03840453214734835,
           0.035801705831826784, 0.03527625095281626, 0.025258261801191093,
           0.024686395628421485, 0.01652188095116336, 0.021627616169205763,
           0.006761645694441771, 0.006885779292505334]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-4)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-4)
    @test isapprox(hrc1 / lrc1, 1, rtol = 1.0)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.25)
end

@testset "ADD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = ADD()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05760068572489703, 0.04875578167139456, 0.08760931399125951,
           0.028071926271730586, 0.0656282792933219, 0.04203173685954744,
           0.02569142036379359, 0.0852967781988239, 0.03459412389698896, 0.0350939078624161,
           0.10007164884110781, 0.027207224708044018, 0.017629477559201648,
           0.039643268500178935, 0.014602586370220848, 0.04741312005505715,
           0.07495121759286542, 0.04971986510935021, 0.05842386305571274,
           0.05996377407408758]
    w2t = [0.01117366781007166, 0.014033169054553924, 0.045908571107162684,
           0.012856587763064538, 0.050622843054460215, 0.022249776126497384,
           0.01862763478098223, 0.07977869361256462, 0.02874816697896668,
           0.031322680707860255, 0.1105249907447127, 0.02476816915865819,
           0.01688562364522229, 0.05345524384387573, 0.0195961710270235,
           0.061298883594235444, 0.12257905778456897, 0.07478548505959613,
           0.09193002828020275, 0.1088545558657202]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.5)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.1)
end

@testset "UCI" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = UCI()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.054827650621015606, 0.04136677057713142, 0.0616510233668115,
           0.02685181720965714, 0.05953954675813311, 0.0357006890556717,
           0.023357090594641562, 0.09683349095701785, 0.02981058674921562,
           0.032210169214524055, 0.14759402716956618, 0.03543238969939519,
           0.019400560335223485, 0.04196642513390948, 0.017244468289610928,
           0.049642378193525356, 0.06572977221027157, 0.05394757177824973,
           0.0462506904198338, 0.06064288166659467]
    w2t = [0.008154286578153593, 0.008781190376849138, 0.023209195001104903,
           0.011185790713888158, 0.039775349087535154, 0.02070829609129032,
           0.01831077163888364, 0.09098444685646642, 0.02501429263455831,
           0.028901383716111687, 0.17118200009515813, 0.03039066156086129,
           0.018933548234311447, 0.052984908028786755, 0.02121400841973578,
           0.05901915302335803, 0.10465913823759998, 0.08144615071936157,
           0.07867096408121685, 0.1064744649047689]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-4)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.5)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.5)
end

@testset "CDaR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = CDaR()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.046950955979794685, 0.03661874760554526, 0.038223988588657366,
           0.027780516550856046, 0.05445110680774062, 0.034214753641007754,
           0.02235457818855092, 0.0891440899355276, 0.02730742650497644,
           0.028300548271497156, 0.2157867803618873, 0.040306880430063904,
           0.020910697807736522, 0.04004945288518099, 0.026363565986111177,
           0.044699572145195474, 0.058863654906891535, 0.04952002372532478,
           0.03914800563209636, 0.059004654045358026]
    w2t = [0.007051812638889491, 0.005446252661254122, 0.012412530953308716,
           0.009360644286118255, 0.04020995056824593, 0.019328101052668287,
           0.017748063200104077, 0.08718302805443717, 0.022004016111711964,
           0.022400414586845607, 0.2516695149008172, 0.03519171976658421,
           0.01772164131019354, 0.045505473406319996, 0.023422021900167886,
           0.0475329366653845, 0.08639996137790268, 0.07371024833282129,
           0.06435768276056801, 0.111343985465657]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-4)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.1)
end

@testset "EDaR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = EDaR()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05532235699581659, 0.035788140800455205, 0.0407753569587081,
           0.0296057134575646, 0.04696905711931078, 0.040842428659498574,
           0.025301130647021896, 0.08891722046323698, 0.026897229105847707,
           0.03134376238119591, 0.19602182683257213, 0.039031614098305384,
           0.02653355216997208, 0.04068455054688783, 0.025037256367833247,
           0.04705215245370184, 0.053526293400258725, 0.05549120103462624,
           0.038350797642177206, 0.05650835886500902]
    w2t = [0.015858118434283787, 0.005699806949620181, 0.0154094687052691,
           0.009983801600472533, 0.03722669048197105, 0.024786496611501056,
           0.01694263115258482, 0.08449342106400154, 0.022635048127404113,
           0.02448655127646425, 0.21325796783460865, 0.03791611528585942,
           0.02305614958310032, 0.05053900207985802, 0.02457868062265538,
           0.05223407250646271, 0.07671421685304018, 0.0878533864508667,
           0.06349531076756797, 0.11283306361240827]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.5)
end

@testset "RLDaR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = RLDaR()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.0679360045947663, 0.03531369144414196, 0.04202344574685738,
           0.03066508232720293, 0.04581748346998565, 0.046948045136922134,
           0.025815182408722707, 0.09117725057001834, 0.02734160395390057,
           0.03291528526162767, 0.17193313618220957, 0.03568423026702458,
           0.03230471201763436, 0.04018809873037509, 0.02462778965081862,
           0.04706523674092606, 0.04992592620193859, 0.05846705995187727,
           0.03735768546200424, 0.05649304988104601]
    w2t = [0.02829674582948989, 0.0057954601728955795, 0.015516213258281673,
           0.01036741281931746, 0.034589535419083814, 0.028785971922675156,
           0.016410418414304264, 0.08004482782508542, 0.023281484169544498,
           0.025784330230511767, 0.19331589475806557, 0.03679769919331651,
           0.028435096305257097, 0.05165443959250701, 0.02544455456088115,
           0.05270945531830427, 0.07210153554058965, 0.09516078481067097,
           0.06257871546103073, 0.11292942439818753]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-7)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-6)
    @test isapprox(hrc1 / lrc1, 1, rtol = 1.0)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.5)
end

@testset "EDaR < RLDaR < MDD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75,
                                                                           "max_iter" => 300))))
    asset_statistics!(portfolio)

    rm = RLDaR(; kappa = 5e-3)
    w1 = optimise!(portfolio, RB(; rm = rm))
    rm = RLDaR(; kappa = 1 - 5e-3)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rm = EDaR()
    w3 = optimise!(portfolio, RB(; rm = rm))
    rm = MDD()
    w4 = optimise!(portfolio, RB(; rm = rm))

    if !Sys.isapple()
        @test isapprox(w1.weights, w3.weights, rtol = 0.005)
    end
    @test isapprox(w2.weights, w4.weights, rtol = 1.0e-4)
end

@testset "Full Kurt" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = Kurt()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm, str_names = true))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.0462200718999689, 0.05174092671954537, 0.044401282343031694,
           0.05069359658201531, 0.042887514292418626, 0.05510889204986023,
           0.033767406593088044, 0.0787785917536274, 0.0391213651994881,
           0.048201736903453946, 0.07989446732548909, 0.039365382456744646,
           0.034302318515355174, 0.056502995892440684, 0.03863352345516244,
           0.04678779011292719, 0.04923060115270035, 0.06498813842157836,
           0.043444020010514645, 0.055929378320589744]
    w2t = [0.004744836368282065, 0.010553376475749157, 0.014134438703906545,
           0.021524634730045917, 0.02237662977006486, 0.03218224768913213,
           0.02458441565830405, 0.06033530653734529, 0.03328265380416957,
           0.04650306753556985, 0.0845377350345998, 0.044937014879739605,
           0.040213781784835416, 0.07351778359586489, 0.050482836419353125,
           0.065885088003057, 0.07959304260916046, 0.10842590025207983, 0.07730862647432614,
           0.1048765836744142]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "Reduced Kurt" begin
    portfolio = Portfolio(; prices = prices, max_num_assets_kurt = 1,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = Kurt()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.045921530036848776, 0.05179969756961471, 0.04433861222682677,
           0.050735015699262316, 0.042803287659396935, 0.05502423878987333,
           0.033701531390402, 0.07908324070885274, 0.03908642952244211, 0.04807671450296364,
           0.08049505496819197, 0.03924963131871603, 0.034226894953414834,
           0.056406915130355395, 0.03856924971903614, 0.04674664471306371,
           0.04922398837244406, 0.06499634397412841, 0.04336260704722886,
           0.05615237169693716]
    w2t = [0.004657207463258355, 0.01050720618622578, 0.014020923847525419,
           0.021478628294130865, 0.022183892730714548, 0.03188780547430791,
           0.024489222803154475, 0.06026763504443116, 0.03309052187749186,
           0.04623085908386787, 0.08496864880709416, 0.04461491733166125,
           0.04008087770109498, 0.07303871869894975, 0.05021880350149549,
           0.06566297262643818, 0.07989119438573102, 0.10944850522531743,
           0.07704351299115232, 0.10621794592595722]
    @test isapprox(w1.weights, w1t, rtol = 0.0005)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)
end

@testset "Full SKurt" begin
    portfolio = Portfolio(; prices = prices, max_num_assets_kurt = 1,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = SKurt()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.0474930263089476, 0.0545300887771538, 0.04332709582118912, 0.04408977642537243,
           0.04294604641437751, 0.05479952846725636, 0.029936908143268166,
           0.07379426936198005, 0.03854489273152313, 0.04491024182364559,
           0.07821574940941496, 0.041822023191967, 0.03694001773575391, 0.05882544707047249,
           0.041189526693870576, 0.04894370822552273, 0.05062943690177448,
           0.06522327632412397, 0.04376745081445116, 0.06007148935793488]
    w2t = [0.004722755219924028, 0.01099340299327724, 0.01308002868176818,
           0.017759249817031475, 0.021165105296187085, 0.031801586423919975,
           0.020973739885717685, 0.05554909610801033, 0.03254329572377208,
           0.042340761615755194, 0.08231443576994316, 0.04515006940066914,
           0.04355001353914134, 0.07524055481162406, 0.0536534855553139,
           0.07059891123011293, 0.08206096807647194, 0.10766963283148796,
           0.07791292293674282, 0.11091998408312954]
    @test isapprox(w1.weights, w1t, rtol = 0.0005)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.25)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.01)
end

@testset "Reduced SKurt" begin
    portfolio = Portfolio(; prices = prices, max_num_assets_kurt = 1,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = SKurt()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.0474930263089476, 0.0545300887771538, 0.04332709582118912, 0.04408977642537243,
           0.04294604641437751, 0.05479952846725636, 0.029936908143268166,
           0.07379426936198005, 0.03854489273152313, 0.04491024182364559,
           0.07821574940941496, 0.041822023191967, 0.03694001773575391, 0.05882544707047249,
           0.041189526693870576, 0.04894370822552273, 0.05062943690177448,
           0.06522327632412397, 0.04376745081445116, 0.06007148935793488]
    w2t = [0.004722755219924028, 0.01099340299327724, 0.01308002868176818,
           0.017759249817031475, 0.021165105296187085, 0.031801586423919975,
           0.020973739885717685, 0.05554909610801033, 0.03254329572377208,
           0.042340761615755194, 0.08231443576994316, 0.04515006940066914,
           0.04355001353914134, 0.07524055481162406, 0.0536534855553139,
           0.07059891123011293, 0.08206096807647194, 0.10766963283148796,
           0.07791292293674282, 0.11091998408312954]
    @test isapprox(w1.weights, w1t, rtol = 0.0005)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.25)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.01)
end

@testset "Skew" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = Skew()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04591081353431242, 0.07414929020199698, 0.0323997559419677,
           0.029665079749268214, 0.04775001667548281, 0.03526728358046136,
           0.021317283183859607, 0.08111596512642089, 0.028885710868094727,
           0.030723549476101675, 0.06276070821853613, 0.04750490055482056,
           0.043483821283784155, 0.05207326832084091, 0.07523445807351975,
           0.06813713295630598, 0.048718963268605815, 0.07088683910856997,
           0.035340665694213416, 0.06867449418283687]
    w2t = [0.004696966527325601, 0.016545683125543827, 0.00974942505612167,
           0.011301642531742038, 0.024038983655046968, 0.01950942662737644,
           0.014433772744075264, 0.0605373684409659, 0.02319941471330845,
           0.027907785796272337, 0.0652714419701903, 0.04854171728855438,
           0.050392900553137346, 0.06541743599491229, 0.095784206963932,
           0.09373135437492688, 0.07711690092365656, 0.11138763703353748,
           0.059866654324849236, 0.12056928135452505]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.001)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end

@testset "SSkew" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = SSkew()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04762861444408581, 0.05277936940663317, 0.04313986178813018,
           0.04073080697423631, 0.04654851372311383, 0.05132845350860602,
           0.02882166034427583, 0.07957827131848871, 0.037612805257161416,
           0.04380927022767141, 0.08400692097967467, 0.03702450310781179,
           0.03335361510543751, 0.06019867478808539, 0.040427850946278474,
           0.050436605904304314, 0.04930156598640879, 0.07086589045950867,
           0.043803061660139676, 0.058603684069947974]
    w2t = [0.004763915571640253, 0.010492697811449533, 0.012900755339953145,
           0.016255419028500112, 0.02331354126962291, 0.029196294792622965,
           0.02018894810256847, 0.05923863915768778, 0.031326786565196524,
           0.04080723291450329, 0.08737867165100115, 0.040162618006604596,
           0.038616677201321364, 0.07706525576194674, 0.052116693999158616,
           0.07136196407459405, 0.08035630640181515, 0.11755774465762685,
           0.0769670002684789, 0.10993283742370744]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-4)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "DVaR" begin
    portfolio = Portfolio(; prices = prices[(end - 50):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = BDVariance()

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04128284342765488, 0.04983275219701424, 0.043564003316188774,
           0.04375139955674138, 0.05207730254562292, 0.05757218124029658,
           0.04262173039701654, 0.06069203229117991, 0.0438354618111768,
           0.045749552537662524, 0.0748366121861059, 0.03556944049873666,
           0.02745962089165174, 0.06312811050104655, 0.034751555747415355,
           0.05211016208573219, 0.0492019564912932, 0.06008984398353699, 0.0463464000796848,
           0.07552703821424202]
    w2t = [0.00397678215215651, 0.009406035859872739, 0.012693078978626439,
           0.017838584086764402, 0.025445098736939134, 0.03263692535542054,
           0.029645641862972687, 0.04326744515346584, 0.03704865504981173,
           0.042749367951720366, 0.07470548150288066, 0.03832097797627466,
           0.030758417223630156, 0.08150318094822533, 0.048277936906114746,
           0.07414803714623, 0.07857985482796837, 0.09763491080637585, 0.08249193479610582,
           0.138871652678444]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "GMD" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.7))))
    asset_statistics!(portfolio)

    rm = GMD(; owa = OWASettings(; approx = false))

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.048034526698066, 0.05110102905346026, 0.04574327270083649, 0.04016025943159881,
           0.04820091116033776, 0.04947862064636827, 0.029820485230338655,
           0.06382537867100276, 0.047277701928160866, 0.04657284469829766,
           0.06792666888167112, 0.0269397428579643, 0.02314099654779761,
           0.07304770833179834, 0.028267401535028117, 0.0491646207695037,
           0.05682841097538325, 0.07493558079400367, 0.05401810743504692,
           0.07551573165333558]
    w2t = [0.005216843999915085, 0.010874774792569144, 0.01525171027186947,
           0.018480981700494295, 0.028494077115999147, 0.027473269298370104,
           0.02324810771429563, 0.046282794736941614, 0.03778653831314338,
           0.043169591135090925, 0.06425684260974042, 0.028232864847461292,
           0.025156161354234995, 0.09117165252189166, 0.03757692494901132,
           0.06309559240482479, 0.09826122341472683, 0.11627275437789045,
           0.09026494288348534, 0.12943235155804417]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.001)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)

    rm = GMD(; owa = OWASettings(; approx = true))

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.0480435191607846, 0.051047189633827234, 0.045748406597455625,
           0.04013077687117423, 0.04815990471923371, 0.04933536408272662,
           0.029706073516558525, 0.06367306991096036, 0.04748197912129362,
           0.04665134210555755, 0.06756419863994918, 0.02704665821714901,
           0.02321464802190422, 0.07291714191227112, 0.02830150238569247,
           0.049102996055553114, 0.056845099775637925, 0.07510344760986062,
           0.054251021972419204, 0.07567565968999095]
    w2t = [0.005221731385383209, 0.01088151092428963, 0.01524623709025463,
           0.018447190562941047, 0.028324865792466725, 0.0277414764353368,
           0.023135277850088305, 0.04613097567635682, 0.03787935316607699,
           0.043239760952041385, 0.0641834369202214, 0.02851715202074774,
           0.025446968932273897, 0.09092202468791231, 0.03781623716038257,
           0.06292604578427992, 0.09814682089339648, 0.11653701033731113,
           0.09019964639114818, 0.12905627703709094]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end

@testset "TG" begin
    portfolio = Portfolio(; prices = prices[(end - 125):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    rm = TG(; owa = OWASettings(; approx = false))

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    portfolio.risk_budget /= sum(portfolio.risk_budget)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.038763991063993544, 0.07263842075297254, 0.0422816872629888,
           0.04354123420758688, 0.0487727620916042, 0.039675837879028016,
           0.03231527186625781, 0.052976875781158166, 0.03844664825940426,
           0.049786233299200525, 0.05534569043791167, 0.06518919576875985,
           0.034131688235038975, 0.052388079767782675, 0.049270856835427444,
           0.053695701370909135, 0.049277633682941316, 0.04936590629016802,
           0.04217736943703874, 0.08995891570982747]
    w2t = [0.003819881948221029, 0.013369938343993512, 0.011541894562426101,
           0.017571348936903517, 0.01989746041281823, 0.022691138920140796,
           0.021403417604723512, 0.037862601174434095, 0.0314817482343852,
           0.048283288563580114, 0.057253767130212206, 0.07488456565347262,
           0.038556463168947515, 0.07498150469301994, 0.06385960040159129,
           0.08136859199271262, 0.07603774858026555, 0.07779571316749154,
           0.07643067515059271, 0.1509086513600679]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)

    rm = TG(; owa = OWASettings(; approx = true))

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.03874058604050523, 0.07003048272624215, 0.04194226016390561,
           0.044947764369548254, 0.04948042488737771, 0.03992717844412266,
           0.032656838983802304, 0.053336790833033665, 0.03819629391728734,
           0.049170085861050115, 0.05460936027932465, 0.06730549135223622,
           0.03272499867968116, 0.0530741721435316, 0.04968725184755488,
           0.05372872381008022, 0.0492169374226743, 0.04858023663090721,
           0.042413825233982794, 0.09023029637315187]
    w2t = [0.003770059948569851, 0.01320465083937873, 0.011539725614923627,
           0.01732102888549214, 0.020227969943464118, 0.02242605396962551,
           0.021383835037216567, 0.03823173548667696, 0.03146388502298935,
           0.04781944218382115, 0.05671457093776838, 0.07436818666188884,
           0.03858164594693872, 0.07417524349010264, 0.06486311235748, 0.0803521101806884,
           0.07534732363032674, 0.07800659571216939, 0.07565996528431555,
           0.1545428588661634]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.1)
end

@testset "TGRG" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.85))))
    asset_statistics!(portfolio)

    rm = TGRG(;)

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04253488190779744, 0.0499155646820756, 0.0429837499555012, 0.03478162110512582,
           0.04824951793537329, 0.0532050670984131, 0.036593595947319243,
           0.06323743082751611, 0.041120321773829425, 0.04436049526866354,
           0.08117617451193386, 0.043850804186369746, 0.025562873853349606,
           0.06928304176805283, 0.03631809507541593, 0.048489144527540634,
           0.04932834460902806, 0.06279881467188772, 0.04589903156560853,
           0.08031142872919832]
    w2t = [0.0041344453249236015, 0.009450341298025862, 0.01317396662083355,
           0.015385281915140546, 0.0225252397188795, 0.030567262766090824,
           0.026477177029914388, 0.054328575576399904, 0.034569496802012,
           0.040881554572638916, 0.07959063354078794, 0.0472157955395031,
           0.030669932679342874, 0.08497263653160067, 0.0529360020276571,
           0.06602199165673786, 0.07860838479579062, 0.1002302980981733,
           0.07990869136838882, 0.1283522921371587]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.1)
end

@testset "OWA" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.7))))
    asset_statistics!(portfolio)

    rm = OWA(; owa = OWASettings(; approx = false))

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.048034526698066, 0.05110102905346026, 0.04574327270083649, 0.04016025943159881,
           0.04820091116033776, 0.04947862064636827, 0.029820485230338655,
           0.06382537867100276, 0.047277701928160866, 0.04657284469829766,
           0.06792666888167112, 0.0269397428579643, 0.02314099654779761,
           0.07304770833179834, 0.028267401535028117, 0.0491646207695037,
           0.05682841097538325, 0.07493558079400367, 0.05401810743504692,
           0.07551573165333558]
    w2t = [0.005216660374356262, 0.010874374321209126, 0.015251058668187305,
           0.01848157004213322, 0.028494161994685994, 0.02747248255487457,
           0.023249610174422963, 0.046283777791668104, 0.03778685204555321,
           0.04316999575536555, 0.06425760405683642, 0.028232677194594, 0.02515645432118518,
           0.09116841055417137, 0.037576367699632254, 0.06309648657138091,
           0.09826236476510795, 0.11627318427500141, 0.09026525263239465,
           0.12943065420723956]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.001)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)

    rm = OWA(; owa = OWASettings(; approx = true))

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, RB(; rm = rm))
    rc1 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, RB(; rm = rm))
    rc2 = risk_contribution(portfolio; type = :RB, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04804285792411565, 0.05104700133202033, 0.04574834463363973,
           0.040130299540692735, 0.048158978103597476, 0.049336645918596024,
           0.02970624606813772, 0.06367339506214997, 0.047483705540338955,
           0.04665176923616095, 0.06756355191357594, 0.02704670291100915,
           0.023214499054867346, 0.07291727771332068, 0.02830167723618812,
           0.049102757948005366, 0.056845033462047306, 0.07510452446116492,
           0.05425050986780465, 0.07567422207256715]
    w2t = [0.005221885163886571, 0.010882459831658812, 0.015246049090810685,
           0.018447376031650576, 0.02832456112882713, 0.027740837805938705,
           0.023135341062036296, 0.04613125030443857, 0.03788002448590483,
           0.04324047102326525, 0.06418340964785905, 0.028516712391708113,
           0.025447047854659084, 0.09092191088097755, 0.03781591281451579,
           0.06292598340769497, 0.09814694460579099, 0.11653683089302641,
           0.09019918967613041, 0.12905580189922033]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end
