using COSMO, CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries, SCS

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:RP), $(:RG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = :RG
    opt = OptimiseOpt(; type = :RP, rm = rm, owa_approx = false)

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, opt)
    rc1 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, opt)
    rc2 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    opt.owa_approx = true
    portfolio.risk_budget = []
    w3 = optimise!(portfolio, opt)
    rc3 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc3, hrc3 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w4 = optimise!(portfolio, opt)
    rc4 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc4, hrc4 = extrema(rc2)

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

    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w2.weights, w4.weights)
    @test isapprox(hrc1, hrc3)
    @test isapprox(lrc1, lrc3)
end

@testset "$(:Classic), $(:RP), $(:RCVaR)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = :RCVaR
    opt = OptimiseOpt(; type = :RP, rm = rm, owa_approx = false)

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, opt)
    rc1 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, opt)
    rc2 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    opt.owa_approx = true
    portfolio.risk_budget = []
    w3 = optimise!(portfolio, opt)
    rc3 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc3, hrc3 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w4 = optimise!(portfolio, opt)
    rc4 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc4, hrc4 = extrema(rc2)

    w1t = [0.04093966721708738, 0.04818692689480003, 0.03953773845651885,
           0.034491335246324105, 0.04946849042103668, 0.06049545186182922,
           0.03361149453285966, 0.058942352663501625, 0.044015198222066225,
           0.04435164198392704, 0.08145794183078651, 0.041393851937014566,
           0.025651365210380046, 0.07126392629815041, 0.03650527617306732,
           0.050133827036985705, 0.05096045191695565, 0.0673896804883164,
           0.048839618154183605, 0.07236376345420885]

    w2t = [0.004387744265616181, 0.009477441858249563, 0.013944726856518305,
           0.01569257813807223, 0.024847283284382723, 0.027443397230263705,
           0.025117699869211634, 0.05302598007481011, 0.03671068696345323,
           0.043478309194229, 0.08278289218806542, 0.03925421492629808,
           0.025755980846670346, 0.08779832033252195, 0.0545906993443115,
           0.06591588791922484, 0.08413914831389598, 0.10978713639381915,
           0.08560983493815291, 0.11024003706223308]

    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 9e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 4e-1)

    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w2.weights, w4.weights)
    @test isapprox(hrc1, hrc3)
    @test isapprox(lrc1, lrc3)
end

@testset "$(:Classic), $(:RP), $(:GMD), blank $(:OWA) and owa_w = owa_gmd(T)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = :GMD
    opt = OptimiseOpt(; type = :RP, rm = rm, owa_approx = false)

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, opt)
    rc1 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, opt)
    rc2 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    rm = :OWA
    portfolio.risk_budget = []
    opt.rm = rm
    w3 = optimise!(portfolio, opt)
    rc3 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc3, hrc3 = extrema(rc3)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w4 = optimise!(portfolio, opt)
    rc4 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc4, hrc4 = extrema(rc4)

    portfolio.owa_w = owa_gmd(size(portfolio.returns, 1))
    portfolio.risk_budget = []
    w5 = optimise!(portfolio, opt)
    rc5 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc5, hrc5 = extrema(rc5)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w6 = optimise!(portfolio, opt)
    rc6 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc6, hrc6 = extrema(rc6)

    opt.owa_approx = true
    portfolio.risk_budget = []
    w3 = optimise!(portfolio, opt)
    rc3 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc3, hrc3 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w4 = optimise!(portfolio, opt)
    rc4 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc4, hrc4 = extrema(rc2)

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
    @test isapprox(w2.weights, w2t, rtol = 0.0005)
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

    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w2.weights, w4.weights)
    @test isapprox(hrc1, hrc3)
    @test isapprox(lrc1, lrc3)
end

@testset "$(:Classic), $(:RP), $(:TG)" begin
    portfolio = Portfolio(; prices = prices[(end - 125):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = :TG
    opt = OptimiseOpt(; type = :RP, rm = rm, owa_approx = false)

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, opt)
    rc1 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, opt)
    rc2 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    opt.owa_approx = true
    portfolio.risk_budget = []
    w3 = optimise!(portfolio, opt)
    rc3 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc3, hrc3 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w4 = optimise!(portfolio, opt)
    rc4 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc4, hrc4 = extrema(rc2)

    w1t = [0.03884795714787028, 0.07291399696607788, 0.04219785808808718,
           0.043575947160987435, 0.04840950751874932, 0.03968825269713154,
           0.0324094449511172, 0.052907391784013526, 0.03834129976918106,
           0.049795767780325656, 0.05541676122938239, 0.06546082112864092,
           0.0341451534237343, 0.052690405060622746, 0.04924509147872571,
           0.05382270524704115, 0.049234991484763235, 0.049210766672260015,
           0.04215817781453176, 0.08952770259675667]

    w2t = [0.003805226614697479, 0.013364205009887008, 0.011525394065478603,
           0.01752721720284768, 0.02000506401129493, 0.022673106637119324,
           0.021392712411281652, 0.037872967183538386, 0.031439975771315305,
           0.048025566944921765, 0.05714480019773132, 0.07509769448705383,
           0.03855722461209247, 0.07464724983990394, 0.06404877846320639,
           0.08139496159574147, 0.07619668233424755, 0.07782667521029749,
           0.07630960576210888, 0.1511448916452343]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 4e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 6e-1)

    @test isapprox(w1.weights, w3.weights, rtol = 0.1)
    @test isapprox(w2.weights, w4.weights, rtol = 0.1)
    @test isapprox(hrc1, hrc3)
    @test isapprox(lrc1, lrc3)
end

@testset "$(:Classic), $(:RP), $(:RTG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false)),
                                                :COSMO => Dict(:solver => COSMO.Optimizer,
                                                               :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)

    rm = :RTG
    opt = OptimiseOpt(; type = :RP, rm = rm)

    portfolio.risk_budget = []
    w1 = optimise!(portfolio, opt)
    rc1 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise!(portfolio, opt)
    rc2 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    opt.owa_approx = true
    portfolio.risk_budget = []
    w3 = optimise!(portfolio, opt)
    rc3 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc3, hrc3 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w4 = optimise!(portfolio, opt)
    rc4 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc4, hrc4 = extrema(rc2)

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
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 6.2e-4)
    @test isapprox(hrc2 / lrc2, 20, atol = 4e-1)

    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w2.weights, w4.weights)
    @test isapprox(hrc1, hrc3)
    @test isapprox(lrc1, lrc3)
end
