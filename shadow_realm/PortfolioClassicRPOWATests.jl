using CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:RP), $(:RG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
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
    @test isapprox(hrc2 / lrc2, 20, atol = 6.3e-1)

    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w2.weights, w4.weights)
    @test isapprox(hrc1, hrc3)
    @test isapprox(lrc1, lrc3)
    @test isapprox(hrc2, hrc4)
    @test isapprox(lrc2, lrc4)
end

@testset "$(:Classic), $(:RP), $(:RCVaR)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
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

    w2t = [0.004387994801739041, 0.009476373409804785, 0.013942985177683042,
           0.015691749206978856, 0.024844247016568444, 0.027438678452997547,
           0.025118209296543004, 0.05302595086266504, 0.03670757153719749,
           0.043475336494849956, 0.08277971130556752, 0.039257451588347855,
           0.02575431858451394, 0.08779970749458026, 0.054590515851198625,
           0.06591759725914802, 0.0841450951158135, 0.10979248879490835, 0.085614609766294,
           0.1102394079826009]

    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 9e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 4e-1)

    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w2.weights, w4.weights)
    @test isapprox(hrc1, hrc3)
    @test isapprox(lrc1, lrc3)
    @test isapprox(hrc2, hrc4)
    @test isapprox(lrc2, lrc4)
end

@testset "$(:Classic), $(:RP), $(:GMD), blank $(:OWA) and owa_w = owa_gmd(T)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
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

    opt.owa_approx = true
    portfolio.risk_budget = []
    w9 = optimise!(portfolio, opt)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w10 = optimise!(portfolio, opt)

    opt.owa_approx = false
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
    w7 = optimise!(portfolio, opt)
    rc7 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc7, hrc7 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w8 = optimise!(portfolio, opt)
    rc8 = risk_contribution(portfolio; type = :RP, rm = rm)
    lrc8, hrc8 = extrema(rc2)

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

    @test isapprox(w1.weights, w7.weights, rtol = 0.01)
    @test isapprox(w2.weights, w8.weights, rtol = 0.01)
    @test isapprox(w9.weights, w7.weights)
    @test isapprox(w10.weights, w8.weights)
    @test isapprox(hrc1, hrc7)
    @test isapprox(lrc1, lrc7)
    @test isapprox(hrc2, hrc8)
    @test isapprox(lrc2, lrc8)
end

@testset "$(:Classic), $(:RP), $(:TG)" begin
    portfolio = Portfolio(; prices = prices[(end - 125):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                  :params => Dict("verbose" => false,
                                                                                  "max_step_fraction" => 0.75))))
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

    w1t = [0.03876062398543985, 0.07263607716053425, 0.042283987585277245,
           0.04354582190887695, 0.04877290590437643, 0.03967837050941984,
           0.032317314454973496, 0.05297444577730352, 0.03844930891807633,
           0.04978971862742524, 0.05534539363192573, 0.06518740283130155,
           0.03413268473595869, 0.052389870713266694, 0.049268425614693794,
           0.05369195070486258, 0.049275519328380486, 0.049364433667283644,
           0.04218129180559488, 0.08995445213502887]

    w2t = [0.003820047906094822, 0.013371674705464723, 0.011542106164171402,
           0.017572053568968037, 0.01989698991879131, 0.02269168051164763,
           0.021403015463610165, 0.0378628352742315, 0.03148075801883078,
           0.04828405215754465, 0.05724971378088297, 0.07488964721820487,
           0.03855743115386395, 0.07497925918635083, 0.0638562020027975,
           0.08137054475894451, 0.07603743934843864, 0.07779529288588062,
           0.0764329838153201, 0.150906272159961]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-4)
    @test isapprox(hrc1 / lrc1, 1, atol = 6.4e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 6.4e-1)

    @test isapprox(w1.weights, w3.weights, rtol = 0.1)
    @test isapprox(w2.weights, w4.weights, rtol = 0.1)
    @test isapprox(hrc1, hrc3)
    @test isapprox(lrc1, lrc3)
    @test isapprox(hrc2, hrc4)
    @test isapprox(lrc2, lrc4)
end

@testset "$(:Classic), $(:RP), $(:RTG)" begin
    portfolio = Portfolio(; prices = prices[(end - 200):end],
                          solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
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

    w1t = [0.0425354272767062, 0.04991571895368152, 0.04298329828063256,
           0.03478231208822956, 0.04824894701525979, 0.05320453060693792,
           0.03659423924135794, 0.06323743284904297, 0.04112041989604676,
           0.044360488824348794, 0.08117513451607743, 0.04385016868298808,
           0.025563101387777416, 0.06928268651446567, 0.036317664547475315,
           0.048489360135118655, 0.04932838211979579, 0.06279913028520855,
           0.04589944098267767, 0.08031211579617145]

    w2t = [0.004134397778839289, 0.009449620949961284, 0.013178015687119968,
           0.015383704409587985, 0.02252899120296703, 0.03056873560821377,
           0.026476242607047094, 0.05432975337346941, 0.034568378730063465,
           0.040879422165230495, 0.07959638335531963, 0.047209882758596354,
           0.03067332959942541, 0.08496674384644058, 0.052934757352041416,
           0.06603197134018766, 0.07860555721741971, 0.10022588849748805,
           0.0799099480698288, 0.12834827545075245]

    @test isapprox(w1.weights, w1t, rtol = 0.001)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, atol = 2)

    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w2.weights, w4.weights)
    @test isapprox(hrc1, hrc3)
    @test isapprox(lrc1, lrc3)
end
