using COSMO, CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries, SCS

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:RP), $(:GMD), blank $(:OWA) and owa_w = owa_gmd(T)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    portfolio.risk_budget = []
    w1 = optimise!(portfolio; class = :Classic, type = :RP, rm = :GMD)
    rc1 = risk_contribution(portfolio; type = :RP, rm = :GMD)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio; class = :Classic, type = :RP, rm = :GMD)
    rc2 = risk_contribution(portfolio; type = :RP, rm = :GMD)
    lrc2, hrc2 = extrema(rc2)

    portfolio.risk_budget = []
    w3 = optimise!(portfolio; class = :Classic, type = :RP, rm = :OWA)
    rc3 = risk_contribution(portfolio; type = :RP, rm = :OWA)
    lrc3, hrc3 = extrema(rc3)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w4 = optimise!(portfolio; class = :Classic, type = :RP, rm = :OWA)
    rc4 = risk_contribution(portfolio; type = :RP, rm = :OWA)
    lrc4, hrc4 = extrema(rc4)

    portfolio.owa_w = owa_gmd(size(portfolio.returns, 1))
    portfolio.risk_budget = []
    w5 = optimise!(portfolio; class = :Classic, type = :RP, rm = :OWA)
    rc5 = risk_contribution(portfolio; type = :RP, rm = :OWA)
    lrc5, hrc5 = extrema(rc5)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w6 = optimise!(portfolio; class = :Classic, type = :RP, rm = :OWA)
    rc6 = risk_contribution(portfolio; type = :RP, rm = :OWA)
    lrc6, hrc6 = extrema(rc6)

    w1t = [0.04802942512713756, 0.051098633965431545, 0.045745604303095315,
           0.04016462272315026, 0.048191113840881636, 0.04947756880152616,
           0.029822819526080995, 0.06381728597897235, 0.04728967147101808,
           0.04658112360117908, 0.06791610810062289, 0.02694231133104885,
           0.02315124666092132, 0.07304653981777988, 0.028275385479207917,
           0.04916987940293215, 0.05681801358545005, 0.07493768749423871,
           0.05401384214831002, 0.07551111664101536]

    w2t = [0.005218035899700653, 0.01087674740630968, 0.015253430555507657,
           0.018486437380714205, 0.028496039178234907, 0.027469099246927333,
           0.02325200625061058, 0.046280910697853825, 0.037790152343153555,
           0.04316409123859577, 0.06424564256322021, 0.028231869870286582,
           0.02515578633314724, 0.09118437505023558, 0.03757948263778634,
           0.06310045219606651, 0.09826555499518812, 0.11626540122133404,
           0.09025737976430415, 0.12942710517082315]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-3)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-4)
    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w2.weights, w4.weights)
    @test isapprox(w1.weights, w5.weights)
    @test isapprox(w2.weights, w6.weights)
    @test isapprox(hrc1 / lrc1, 1, atol = 1e-3)
    @test isapprox(hrc2 / lrc2, 20, atol = 1e-2)
    @test isapprox(rc1, rc3)
    @test isapprox(rc2, rc4)
    @test isapprox(rc1, rc5)
    @test isapprox(rc2, rc6)
end

@testset "$(:Classic), $(:RP), $(:RG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio; class = :Classic, type = :RP, rm = :RG)
    rc1 = risk_contribution(portfolio; type = :RP, rm = :RG)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio; class = :Classic, type = :RP, rm = :RG)
    rc2 = risk_contribution(portfolio; type = :RP, rm = :RG)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04525648415215342, 0.057766352881666934, 0.04597010718397756,
           0.04018043322395846, 0.056959834142059594, 0.06839008647385199,
           0.038684400681021205, 0.06078828361926472, 0.03944682952069196,
           0.05230035622774878, 0.07093362908772874, 0.03857557679026731,
           0.02483608886506268, 0.04201299000564291, 0.041619651661901166,
           0.04179099398612642, 0.052724217150321905, 0.04916876480200959,
           0.04584096241193258, 0.08675395713261208]

    w2t = [0.004113160771993313, 0.00996021962590738, 0.014613077858096003,
           0.014878375261560944, 0.025328251906731773, 0.04415636176137511,
           0.029938902334664085, 0.04258541339860213, 0.03185519932565856,
           0.050853932070479835, 0.07974588464409277, 0.03969037528788452,
           0.031391564578331865, 0.05753733965058486, 0.063382001450656,
           0.060917477338666606, 0.08329777057816251, 0.0830934861741292,
           0.07978920271566133, 0.15287200326676123]

    @test isapprox(w1.weights, w1t, rtol = 0.001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 9e-1)
    @test isapprox(hrc2 / lrc2, 20, atol = 6e-1)
end

@testset "$(:Classic), $(:RP), $(:RCVaR)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio; class = :Classic, type = :RP, rm = :RCVaR)
    rc1 = risk_contribution(portfolio; type = :RP, rm = :RCVaR)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio; class = :Classic, type = :RP, rm = :RCVaR)
    rc2 = risk_contribution(portfolio; type = :RP, rm = :RCVaR)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04093966721708738, 0.04818692689480003, 0.03953773845651885,
           0.034491335246324105, 0.04946849042103668, 0.06049545186182922,
           0.03361149453285966, 0.058942352663501625, 0.044015198222066225,
           0.04435164198392704, 0.08145794183078651, 0.041393851937014566,
           0.025651365210380046, 0.07126392629815041, 0.03650527617306732,
           0.050133827036985705, 0.05096045191695565, 0.0673896804883164,
           0.048839618154183605, 0.07236376345420885]

    w2t = [0.004387553007656474, 0.009477054016177367, 0.013944118056604875,
           0.01569130397497959, 0.024846459246795464, 0.027444369457908634,
           0.02511764730489257, 0.05302484061720872, 0.03670798809714776,
           0.043477185516100456, 0.08277802663000695, 0.03925558908113939,
           0.025754647783602154, 0.08780168059115905, 0.054590987795593945,
           0.06591872513296188, 0.08414084332795661, 0.10979100149151653,
           0.08561118109762815, 0.11023879777296341]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 9e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 4e-1)
end

@testset "$(:Classic), $(:RP), $(:TG)" begin
    portfolio = Portfolio(; prices = prices[(end - 125):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio; class = :Classic, type = :RP, rm = :TG)
    rc1 = risk_contribution(portfolio; type = :RP, rm = :TG)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio; class = :Classic, type = :RP, rm = :TG)
    rc2 = risk_contribution(portfolio; type = :RP, rm = :TG)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.038763856717643874, 0.07263804161391481, 0.04228191397250115,
           0.0435413917826149, 0.048773651417661684, 0.039675758610533296,
           0.03231606313494475, 0.05297622010049861, 0.03844635909937443,
           0.04978675719875652, 0.05534374042603133, 0.06518987496801273,
           0.034131882086129425, 0.052387957021240654, 0.04927168196668566,
           0.05369559817397895, 0.049278651685981445, 0.04936466358405537,
           0.04217762764294142, 0.08995830879649885]

    w2t = [0.0038197667053906006, 0.013370302071012191, 0.01154176451505385,
           0.01757133673602838, 0.019897330983813338, 0.0226914936102597,
           0.021403188430892346, 0.03786231048388992, 0.0314814665445306,
           0.048283495244313315, 0.05725325636356583, 0.07488507977732616,
           0.03855632834997973, 0.07498163188280126, 0.06385935426419453,
           0.08136873456782989, 0.0760382209496565, 0.07779563430429061,
           0.07643092234211506, 0.15090838187305622]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 4e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 6e-1)
end

@testset "$(:Classic), $(:RP), $(:RTG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    w1 = optimise!(portfolio; class = :Classic, type = :RP, rm = :RTG)
    rc1 = risk_contribution(portfolio; type = :RP, rm = :RTG)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio; class = :Classic, type = :RP, rm = :RTG)
    rc2 = risk_contribution(portfolio; type = :RP, rm = :RTG)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.042304471409681986, 0.050291959151292795, 0.04246282744021459,
           0.034154217192553946, 0.04711274092122714, 0.053021558707734416,
           0.03709975559975993, 0.06221043802717917, 0.04085191768667458,
           0.044856782603669765, 0.08316716558090284, 0.04356886923048706,
           0.02551608637477467, 0.07053717563638451, 0.03671465423999034,
           0.04877993854502053, 0.049320416165216964, 0.06298168266461707,
           0.04598643874949792, 0.07906090407311984]

    w2t = [0.004213562156090763, 0.009650456216113297, 0.013281761327902633,
           0.015737255444769692, 0.022026702873995576, 0.031004708791057673,
           0.02760519242868034, 0.054095860792296585, 0.03462758434845054,
           0.04156758184025525, 0.08106190555234895, 0.04651639756905095,
           0.030859657391386564, 0.08623520768951227, 0.0528351286544106,
           0.0671501218631795, 0.07965515136231299, 0.10046755823446475,
           0.08074916456440005, 0.12065904089932113]

    @test isapprox(w1.weights, w1t, rtol = 0.001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 6.2e-4)
    @test isapprox(hrc2 / lrc2, 20, atol = 4e-1)
end
