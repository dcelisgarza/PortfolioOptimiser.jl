using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "SD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = SD2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)
    rc3 = calc_risk_contribution(portfolio; type = :RP2, rm = Variance2())
    lrc3, hrc3 = extrema(rc3)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)
    rc4 = calc_risk_contribution(portfolio; type = :RP2, rm = Variance2())
    lrc4, hrc4 = extrema(rc4)

    w1t = [0.05063484430993387, 0.051248247145833405, 0.04690544758235205,
           0.04368810360104776, 0.04571303450312897, 0.05614653791155684,
           0.02763256689699135, 0.07706277240162171, 0.039493544309350315,
           0.04723302139657694, 0.08434815328832226, 0.033857024708878705,
           0.027547931971505683, 0.0620621872517023, 0.03563793172255409,
           0.04413334025063814, 0.050849763453794807, 0.07142385153127292,
           0.04529955435624263, 0.05908214140669532]
    w2t = [0.005639940543949097, 0.011009340035755901, 0.015582497207404294,
           0.019370512771987466, 0.025438318732809228, 0.03265843055798135,
           0.02037186992290285, 0.05956750307617451, 0.03298314558343289,
           0.044829563191341355, 0.08741264482620345, 0.03822710208900545,
           0.031439207649716396, 0.0796055307252523, 0.046144090275857204,
           0.06097561787250882, 0.08329103566856179, 0.11639034674107102,
           0.07928340577792381, 0.10977989675016077]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 5.0e-5)
    @test isapprox(hrc3 / lrc3, hrc1 / lrc1, rtol = 5.0e-10)
    @test isapprox(hrc4 / lrc4, hrc2 / lrc2, rtol = 5.0e-10)
end

@testset "MAD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = MAD2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05616718884171783, 0.05104482287669086, 0.050442185055943126,
           0.04081085987206851, 0.048790821392722006, 0.056713242149170624,
           0.024225418284767215, 0.07090469906463681, 0.04019545588230386,
           0.046489535418913604, 0.08138390797380332, 0.03158855211640958,
           0.023961449523877854, 0.062252124307186275, 0.034592269087233674,
           0.04206365189343823, 0.05465542813736888, 0.07239054942796913,
           0.04770615320793009, 0.06362168548584848]
    w2t = [0.006399563121224427, 0.01259953405725646, 0.017344094180032826,
           0.01913715742517366, 0.028376470603859533, 0.031901364268023494,
           0.019207083014157005, 0.05539147091326429, 0.03338308143173257,
           0.045148019521911664, 0.08075958036270268, 0.03494225527115619,
           0.025785184739823736, 0.07994544123466757, 0.04475610582327035,
           0.055914691777513806, 0.09227776882581301, 0.12028022431379208,
           0.08191449169494983, 0.11453641741967499]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-7)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.01)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end

@testset "SSD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = SSD2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05053927750227942, 0.05271325589318637, 0.04504106160525596,
           0.04044991084891684, 0.045697350554167335, 0.05416177290339106,
           0.026499490793228032, 0.07776583172257508, 0.038243904391379806,
           0.044545474025866234, 0.08432742630193225, 0.03546280957236507,
           0.02992913805074365, 0.06261843327177671, 0.03923880111295301,
           0.04565294332231343, 0.04985063996037994, 0.07413165623546289,
           0.044276510952126515, 0.05885431097970036]
    w2t = [0.005428610056694705, 0.011166840163835153, 0.014458218268185663,
           0.017041521745985763, 0.024074418715592797, 0.03134544512417569,
           0.019077413527171835, 0.05919765073190795, 0.03194656075952613,
           0.04195983983599716, 0.08814241216305874, 0.03899786030647434,
           0.03418043720843112, 0.07927476322294401, 0.05025705500730055,
           0.06338780950220609, 0.08186743524309842, 0.12155668843610583,
           0.07827494727076424, 0.1083640727105437]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "FLPM" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = FLPM2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05426765863488127, 0.05096600361558761, 0.05094634029921118,
           0.04002797433009358, 0.05569050595339275, 0.04854248101541967,
           0.024658163117795746, 0.07063013104302747, 0.041789427080531835,
           0.046061116223800595, 0.08265005169742001, 0.027987661093431898,
           0.02190398527822549, 0.059102618815220095, 0.03013261014672915,
           0.04527095950185923, 0.05793299147991312, 0.07483566536141466,
           0.04995915354621185, 0.06664450176583281]
    w2t = [0.0068825675304260625, 0.01296039126512009, 0.018389523192842313,
           0.018318328099045725, 0.036076281241134066, 0.027199453368491867,
           0.020213062610521554, 0.05278002908331459, 0.033763600628745075,
           0.044238787803190134, 0.08233682841314095, 0.031755273488808716,
           0.024158196690959238, 0.07390331556018014, 0.03807087187521186,
           0.05746041768532463, 0.09954082737413884, 0.12219633300138268,
           0.08481965045505808, 0.11493626063296344]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-6)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "SLPM" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = SLPM2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05078363551961249, 0.053717899741000934, 0.04576180467751134,
           0.04092029111604701, 0.04906218280689754, 0.05035005426123335,
           0.027363197627451083, 0.07790260260224377, 0.03854273656605282,
           0.04420143278468031, 0.08429612517822185, 0.03395598885724939,
           0.02824434184473981, 0.06056903214345694, 0.036507052624834876,
           0.04757337554325011, 0.051285915028974366, 0.07441808251273557,
           0.04500593244534493, 0.05953831611846164]
    w2t = [0.005491575041085418, 0.011428841414468762, 0.014820967475246397,
           0.01730505616007213, 0.026322354448980064, 0.02906003755711178,
           0.019769446420805574, 0.0592148116776102, 0.032192641913391676,
           0.04165021324570082, 0.08829065859404443, 0.03730878647615285,
           0.03234182015788028, 0.07667258675867361, 0.04703074043617186,
           0.06578751480948142, 0.08445239530103657, 0.12165870854838648,
           0.07958281325604245, 0.10961803030765727]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "WR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = WR2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.043757321581810414, 0.0822733742327415, 0.0436602082667364,
           0.05731637956632544, 0.05518253405854848, 0.04887765289594206,
           0.0364595791033413, 0.04988408327420239, 0.03936409791792019,
           0.04856653979941317, 0.057383681235270545, 0.03890836447870425,
           0.04249964923744369, 0.037820589663936664, 0.04825224579857682,
           0.0498858692877157, 0.06042687519019553, 0.041036573681297046,
           0.04176201475980447, 0.07668236597007388]
    w2t = [0.004526237015894095, 0.015997474101203387, 0.012811159745908298,
           0.022465579875522047, 0.0223177318630974, 0.031293081065040196,
           0.03122893846568954, 0.03943675894118864, 0.03480951552918271,
           0.04185179466637399, 0.0642910409340559, 0.04586248994132221,
           0.06078494786595186, 0.053510801546698204, 0.059734683261052406,
           0.0842140882259648, 0.10213656258967437, 0.07536518051497272, 0.0755840036469502,
           0.12177793020425705]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.5)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)
end

@testset "RG" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = RG2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.03141997465637757, 0.04977499477148719, 0.04187114359583626,
           0.1092709746839547, 0.040302360122874074, 0.05222655887333725,
           0.03836754735185509, 0.059024793482223635, 0.03891494867941388,
           0.062412354481534905, 0.06227647865499888, 0.04249051606921931,
           0.04679875326560747, 0.036572659740749996, 0.06496218226285785,
           0.035641383527722635, 0.04490696424109597, 0.04669575503705878,
           0.04181298570349804, 0.05425667079829651]
    w2t = [0.0032115713459440046, 0.009789161740529536, 0.012410176155093962,
           0.04353604278840679, 0.017522665838650267, 0.03363737795829702,
           0.03265130856300981, 0.04722460921723296, 0.03475641617373519,
           0.05355268209414306, 0.0704215644082256, 0.05062517817133151,
           0.06746841990954869, 0.05203665550590438, 0.07902655645047767,
           0.058989794563634054, 0.07616307748262995, 0.08674498093786183,
           0.07669598830894978, 0.0935357723863939]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)
end

@testset "CVaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = CVaR2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04938933831427547, 0.04880564679526193, 0.04173143865012399,
           0.038407519877081284, 0.04184150449776414, 0.05378156812124352,
           0.026423698659473648, 0.07736661708123845, 0.03544505151114984,
           0.040941895220711864, 0.09251854782877776, 0.04267847930367274,
           0.03238663840849811, 0.06577589633925136, 0.04019561816260983,
           0.0422064111464128, 0.048725958283461226, 0.08071628760303087,
           0.04277031026985724, 0.05789157392610398]
    w2t = [0.004958200154255492, 0.010723616422779553, 0.013513131194217395,
           0.015940996349843917, 0.022147568548325412, 0.029234237315979867,
           0.019156981154586118, 0.0622031451024865, 0.029582195710747775,
           0.0403536838344944, 0.09432253892060641, 0.04011181192911045,
           0.03580058364514293, 0.0756117191663186, 0.050816304260311834,
           0.06895452443242044, 0.07714685683796904, 0.12898555129346154,
           0.07491029621112891, 0.10552605751581359]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end

@testset "RCVaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = RCVaR2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.050269049067239474, 0.0477841693631293, 0.04260280490906643,
           0.04203989184309871, 0.043925216463068635, 0.05759771772474327,
           0.02942674977284078, 0.07979471443162556, 0.03759416396809604,
           0.04513299513579427, 0.08755599134650839, 0.035814547787941235,
           0.030137053397101272, 0.0664242038465082, 0.03710802437521975,
           0.042448048833865756, 0.04778877261428286, 0.07555923590950067,
           0.04385507971795442, 0.057141569492415]
    w2t = [0.00523446044745023, 0.010280825090916797, 0.014110076080297226,
           0.01900495090158587, 0.024059105238347697, 0.03235151230331019,
           0.0233106211797784, 0.060931395967118704, 0.03251551988754096,
           0.04493317236108299, 0.09081297707962498, 0.038901614588372845,
           0.03465074756319317, 0.08231959509767278, 0.04604470994032463,
           0.05819037345801733, 0.08044369521443982, 0.12044775361725577,
           0.07810426536665324, 0.10335262861701634]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end

@testset "EVaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = EVaR2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04644453616015471, 0.06366537537969691, 0.04378771033344591,
           0.05092181041834212, 0.046407597296895545, 0.05062071789359459,
           0.038944792697317814, 0.06586564571647506, 0.03841903583241513,
           0.0444590054286357, 0.06810122487992437, 0.042570609882092235,
           0.04132505958064079, 0.0491628699322602, 0.044386206615348194,
           0.0534476120977008, 0.05368859367348608, 0.05226903833379673,
           0.041779793367462535, 0.0637327644803145]
    w2t = [0.004647915199384994, 0.013022055859743266, 0.013063284291428971,
           0.02051563692286561, 0.022280099122769625, 0.030140211189309412,
           0.028175264001867355, 0.04961370800009065, 0.03320680258710778,
           0.04171683905151383, 0.07294570884438029, 0.04732109080558564,
           0.05108745694078664, 0.06410624273568749, 0.057508778871881834,
           0.07960000716937425, 0.08923747185076795, 0.08898155729045865,
           0.07571253112228286, 0.11711733814271302]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-6)
    @test isapprox(hrc1 / lrc1, 1, rtol = 1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "RVaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = RVaR2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04549546983882415, 0.07569593834645903, 0.04395756946422178,
           0.05586720104955235, 0.049249589645283996, 0.05066244210696011,
           0.04110802259927475, 0.05449490114189938, 0.03937873998654336,
           0.04560091706356088, 0.06132309022695701, 0.040707768756211224,
           0.04367167227924062, 0.04184372389838059, 0.04509954105105621,
           0.05279627454932386, 0.05840797375507491, 0.044739545949107506,
           0.04159936492787433, 0.06830025336419407]
    w2t = [0.004579088865446104, 0.015157441585595907, 0.012920956868085615,
           0.02204271083803428, 0.02225228792164047, 0.031006238396748376,
           0.031024000145675426, 0.04188787193662531, 0.03446525431216618,
           0.04173270075705968, 0.06662190460862863, 0.04653340500195975,
           0.05758114832301181, 0.05644207642929947, 0.05881230870768455,
           0.08358010623093536, 0.09852182150757373, 0.07880031046381987,
           0.07568030175285387, 0.12035806534715568]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-7)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.25)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.1)
end

@testset "EVaR < RVaR < WR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = RVaR2(; kappa = 5e-4)
    w1 = optimise2!(portfolio; rm = rm, type = RP2())
    rm = RVaR2(; kappa = 1 - 5e-4)
    w2 = optimise2!(portfolio; rm = rm, type = RP2())
    rm = EVaR2()
    w3 = optimise2!(portfolio; rm = rm, type = RP2())
    rm = WR2()
    w4 = optimise2!(portfolio; rm = rm, type = RP2())
    @test isapprox(w1.weights, w3.weights, rtol = 0.01)
    @test isapprox(w2.weights, w4.weights, rtol = 1.0e-4)
end

@testset "MDD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = MDD2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = reverse(1:size(portfolio.returns, 2))
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.0832226755172904, 0.03469976232870371, 0.04178756401104356,
           0.027626625273967375, 0.03998306307780897, 0.05672636461843297,
           0.022342457038002583, 0.09418082088656306, 0.027866153990116427,
           0.0355838714778347, 0.15689409933603563, 0.03203159839417066,
           0.03750630444641577, 0.0399766484092098, 0.024305618157569323,
           0.04053983167664855, 0.04468902801560564, 0.06511478398457014,
           0.03540432378926396, 0.05951840557074665]
    w2t = [0.10305137382242804, 0.06581236519112994, 0.0709894827098822,
           0.048241621907253764, 0.06732850187701794, 0.06933295815596083,
           0.025148756820695124, 0.12492370497852971, 0.03239967250505383,
           0.04506726094560067, 0.1364819458861486, 0.03840359462378298,
           0.035802405548559145, 0.03527540849139431, 0.025258125662223532,
           0.02468672266208908, 0.01652174063302115, 0.021628355205884367,
           0.006760316427956624, 0.006885685945388183]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 1.0)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.25)
end

@testset "ADD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = ADD2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.057606462854449564, 0.048757147583333986, 0.08760313404798067,
           0.028072171525899435, 0.06562456116627945, 0.042031866410813794,
           0.025691993374733636, 0.08529724954646445, 0.034594908564242806,
           0.035096572544019884, 0.10006902786342632, 0.02720565515290973,
           0.017629230206665174, 0.0396434223439632, 0.014602246134429559,
           0.04741538864525304, 0.07494445784801652, 0.04972287802730784,
           0.05842615813453437, 0.05996546802527647]
    w2t = [0.011171906249654688, 0.014031965080399496, 0.04590324999045358,
           0.012856310361700663, 0.05062147229126155, 0.022249533370882774,
           0.018628274730880603, 0.07977835360350231, 0.02874507375363999,
           0.03132516743118289, 0.110532504352518, 0.024766097958708735,
           0.016885285913367288, 0.053455421865007985, 0.019595147396604496,
           0.06130793075234185, 0.12257774709628408, 0.07478174508605419,
           0.09193021772287086, 0.10885659499268396]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.5)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)
end

@testset "UCI" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = UCI2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.054836725033953676, 0.04136772470877948, 0.061654616064266046,
           0.026847127744840178, 0.05953457953147206, 0.03569670239048015,
           0.02335641246953983, 0.09682994754370042, 0.029808144579353775,
           0.032209479526236325, 0.14760109788800663, 0.03543325152670983,
           0.019400742635076487, 0.04196679273855133, 0.017243527333969583,
           0.049643334055546245, 0.06573232587809438, 0.05394654739269968,
           0.04625114194158843, 0.06063977901713551]
    w2t = [0.008155822184804581, 0.00878070253189626, 0.02321245982127177,
           0.011186431374581928, 0.0397794474752201, 0.020705029918419698,
           0.018311318911136066, 0.0909814464844048, 0.02501213733443591,
           0.028899976605608082, 0.1711778541221246, 0.030391010154354883,
           0.018935797209900733, 0.05298402723271922, 0.021213771458155502,
           0.059022411989782635, 0.1046658988397448, 0.08144075179666518,
           0.07867722373925909, 0.10646648081551412]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.5)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.25)
end

@testset "CDaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = CDaR2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04695113393595442, 0.03661746998944636, 0.03822542841829081,
           0.027780427413477166, 0.05444936469032488, 0.034219259816653816,
           0.022355983894753146, 0.0891460376120217, 0.02730933794276768,
           0.02830152377423775, 0.21577208480903373, 0.04030672646888614,
           0.02091262720710264, 0.040046384591098196, 0.02636292031520764,
           0.044700569954147146, 0.058869996420132596, 0.0495191089991847,
           0.03914690160773955, 0.05900671213954006]
    w2t = [0.007053190526800975, 0.005446128277036701, 0.01241372313810428,
           0.009360737063492953, 0.04021058700684525, 0.019329177838430734,
           0.017749230999518865, 0.0871761904766593, 0.02200444636139792,
           0.02240084358896396, 0.25167513373016165, 0.035191371847968535,
           0.017720999126723272, 0.04550428705896632, 0.023422306961792816,
           0.047532660293921586, 0.08639815267463893, 0.07370795016486553,
           0.06435839366448634, 0.11134448919922418]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.1)
end

@testset "EDaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = EDaR2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.05532315448250374, 0.035789843750491365, 0.04077445814199365,
           0.029605826542944313, 0.0469660412474676, 0.04084467161326547,
           0.0253009338967018, 0.08891839246222061, 0.026895879448130804,
           0.03134447114000004, 0.19601820266517858, 0.03903196307834241,
           0.026532875646729508, 0.04068549349390345, 0.02503584422602656,
           0.04705236671296188, 0.05353071766624416, 0.055490746380893664,
           0.03834977161070544, 0.05650834579329473]
    w2t = [0.01585774887523436, 0.005700098123506493, 0.0154093602284636,
           0.009983988990045487, 0.03722637744745142, 0.024786173385869047,
           0.01694269985327135, 0.08449421949038945, 0.022634747150362334,
           0.02448621067406314, 0.21325707882221914, 0.0379160512807431,
           0.02305658860774649, 0.050538403993677206, 0.02457864267628073,
           0.05223382605008592, 0.0767149101174364, 0.08785359264768017,
           0.06349501181374491, 0.11283426977172926]
    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.5)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.5)
end

@testset "RDaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = RDaR2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.06793604284871738, 0.03531368372759915, 0.042023447287659055,
           0.03066508748365364, 0.04581748536071042, 0.046948060881784846,
           0.02581516857259885, 0.09117724220316131, 0.027341611754088595,
           0.032915284784924004, 0.1719331375051792, 0.035684217087994255,
           0.03230472152975499, 0.040188093229804094, 0.024627782457110628,
           0.04706524651000926, 0.04992592584769779, 0.058467051247865144,
           0.03735764770349489, 0.05649306197619242]
    w2t = [0.028296793376123033, 0.0057954596703316125, 0.015516214439508205,
           0.010367412960827622, 0.03458952805098015, 0.028785986871447945,
           0.016410413398176137, 0.08004481670997543, 0.02328148278668714,
           0.025784328928895758, 0.19331586412233384, 0.036797709769958775,
           0.028435126242073838, 0.05165442037924009, 0.025444557800996536,
           0.05270946166132721, 0.07210152030425954, 0.0951607562521504,
           0.06257871598828463, 0.11292943028642197]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-7)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-7)
    @test isapprox(hrc1 / lrc1, 1, rtol = 1.0)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.25)
end

@testset "EDaR < RDaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75,
                                                                            "max_iter" => 300))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    rm = RDaR2(; kappa = 5e-3)
    w1 = optimise2!(portfolio; rm = rm, type = RP2())
    rm = EDaR2()
    w2 = optimise2!(portfolio; rm = rm, type = RP2())
    @test isapprox(w1.weights, w2.weights, rtol = 0.0001)
end

@testset "Full Kurt" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = Kurt2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04622318966891087, 0.051735402412157364, 0.04439927739035058,
           0.05069294216523759, 0.04288956595183918, 0.055104528892227646,
           0.03376709775539053, 0.07877864366411548, 0.03912232452046387,
           0.04820386910393024, 0.07988980034322672, 0.03936329259230218,
           0.03430246703525708, 0.05650251786220336, 0.03863330880463875,
           0.04679014012154553, 0.04922989941972541, 0.0649897843888456,
           0.04344769695204487, 0.0559342509555872]
    w2t = [0.004745115300960629, 0.010553833683548, 0.014134891071172148,
           0.021525684594252546, 0.022377607814903267, 0.03218360364696926,
           0.02458565588014059, 0.0603388597300125, 0.03328288452785557,
           0.046497049074399444, 0.08453681188796948, 0.044939300192438816,
           0.040214301017160343, 0.07351675090554316, 0.05047459462089849,
           0.06588331054429225, 0.07959979719339691, 0.10842127106776223,
           0.07730932876018444, 0.10487934848613988]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "Reduced Kurt" begin
    portfolio = Portfolio2(; prices = prices, max_num_assets_kurt = 1,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = Kurt2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04591842179640149, 0.05179805405164705, 0.04434107746789892,
           0.05073576269255856, 0.04281559309978754, 0.05502400846984758,
           0.033702328915667525, 0.07909071247102473, 0.039084828143279446,
           0.04806917872417939, 0.08049384816589035, 0.03924991745050765,
           0.03422848311786619, 0.056402294323495306, 0.038571074160980434,
           0.04674744638239155, 0.049215020736204144, 0.06499655559593155,
           0.04336580052064711, 0.056149593713793446]
    w2t = [0.004657405843011066, 0.010507218482203191, 0.014020919525534851,
           0.021479156702709834, 0.02218399792303029, 0.031888811326582174,
           0.02449055152562633, 0.060262708467851894, 0.03309109723079836,
           0.04623393516867988, 0.08497008994914156, 0.04461965191283875,
           0.04008280382773206, 0.07302946835465755, 0.05022092884857667,
           0.06566591797487605, 0.07988614830630822, 0.10945172225056968,
           0.07703881166353045, 0.10621865471574114]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-6)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)
end

@testset "Full SKurt" begin
    portfolio = Portfolio2(; prices = prices, max_num_assets_kurt = 1,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = SKurt2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.047491170526917106, 0.054533530611338095, 0.04332743262413755,
           0.04409183402669535, 0.04294281900935894, 0.05479618527188426,
           0.029939286967814335, 0.07378407854092589, 0.038545585072875525,
           0.04491050797340899, 0.07820933163981073, 0.041819191428891594,
           0.03694060939399915, 0.05883380726473094, 0.04119066420625541,
           0.04894563617272387, 0.050636793381874066, 0.06522417318082553,
           0.04376830962159429, 0.0600690530839385]
    w2t = [0.004723138294127937, 0.010993728947886361, 0.01308033043038897,
           0.017759695776427185, 0.021165552963051573, 0.031803263388605195,
           0.020974573806133483, 0.05554783310714788, 0.03254435818391317,
           0.0423428924934359, 0.08231374162568034, 0.04515388494713648,
           0.043553198933333696, 0.0752333223309817, 0.053657628423246104,
           0.0705975363622844, 0.08205662607600783, 0.10767161146945363,
           0.07790694484775723, 0.11092013759300094]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.25)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.01)
end

@testset "Reduced SKurt" begin
    portfolio = Portfolio2(; prices = prices, max_num_assets_kurt = 1,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = SKurt2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.047491170526917106, 0.054533530611338095, 0.04332743262413755,
           0.04409183402669535, 0.04294281900935894, 0.05479618527188426,
           0.029939286967814335, 0.07378407854092589, 0.038545585072875525,
           0.04491050797340899, 0.07820933163981073, 0.041819191428891594,
           0.03694060939399915, 0.05883380726473094, 0.04119066420625541,
           0.04894563617272387, 0.050636793381874066, 0.06522417318082553,
           0.04376830962159429, 0.0600690530839385]
    w2t = [0.004723138294127937, 0.010993728947886361, 0.01308033043038897,
           0.017759695776427185, 0.021165552963051573, 0.031803263388605195,
           0.020974573806133483, 0.05554783310714788, 0.03254435818391317,
           0.0423428924934359, 0.08231374162568034, 0.04515388494713648,
           0.043553198933333696, 0.0752333223309817, 0.053657628423246104,
           0.0705975363622844, 0.08205662607600783, 0.10767161146945363,
           0.07790694484775723, 0.11092013759300094]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.25)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.01)
end

@testset "Skew" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = Skew2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.045906673657994286, 0.07415648620070701, 0.03239691712482279,
           0.029662738346822544, 0.04774712631076141, 0.035264759596118266,
           0.02131567857967313, 0.08112036693805974, 0.028883761129959336,
           0.030721349576933375, 0.06276151903605617, 0.04750351860504847,
           0.04348113069373873, 0.05206788117683908, 0.07524692694700864,
           0.06813705328535936, 0.04871519169184158, 0.07089389122112989,
           0.03533805940098055, 0.06867897048014555]
    w2t = [0.004693934872733258, 0.01653712273816221, 0.009745837375747725,
           0.01129785732641342, 0.024031313716372113, 0.01950335791196889,
           0.01442940919157519, 0.06053833658422826, 0.02319282927646636,
           0.02789984507074273, 0.06528733309435215, 0.04853621916516049,
           0.05038776818805386, 0.06542214777077343, 0.09579505608813803,
           0.09373742473516694, 0.07713111605853692, 0.11139837428417523,
           0.059854520216612875, 0.12058019633461987]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.001)
end

@testset "SSkew" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = SSkew2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04762767201257773, 0.05277763009753849, 0.04313912780375195,
           0.04073166143555511, 0.04654544740226561, 0.05132432402431435,
           0.028822263937020677, 0.07957772403554368, 0.03761397860273206,
           0.04380842958871217, 0.08400524915661846, 0.03702649726168862,
           0.03335525728577381, 0.06020169359842145, 0.040429991668753405,
           0.05043723225818501, 0.04929982465460814, 0.07086786182159446,
           0.043802565657043886, 0.058605567697301054]
    w2t = [0.004763599427643488, 0.010492170146556576, 0.012900259783122791,
           0.016254662775559696, 0.023312371632547447, 0.02919488184598214,
           0.020187912383493548, 0.05923845240591326, 0.03132528772972758,
           0.040805941704882144, 0.08738039411791934, 0.04016178659587393,
           0.03861542497085767, 0.07706578085675182, 0.052116271810969686,
           0.07136137384266647, 0.08035387404428683, 0.1175667208351577,
           0.07696549825130941, 0.10993733483877846]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "DVaR" begin
    portfolio = Portfolio2(; prices = prices[(end - 50):end],
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = DVar2()

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.041282715159969716, 0.04983418210656033, 0.04356426842967805,
           0.04375201152529006, 0.052076995648842765, 0.05757219578037072,
           0.04262153320098835, 0.06069244060278705, 0.04383224021553362,
           0.04575148160420536, 0.07483608960225534, 0.03556830817557716,
           0.02745910033988576, 0.06312322997003714, 0.03475209520623468,
           0.052110338074831834, 0.04920424074835751, 0.060088297745948316,
           0.046346922629732314, 0.07553131323291397]
    w2t = [0.003976406116005965, 0.009405999250255676, 0.012692268233392381,
           0.01783744903447256, 0.02544441595576468, 0.032636904639478555,
           0.02964589511592989, 0.04326901946969708, 0.03704892159886207,
           0.04274803450384758, 0.0747050918423663, 0.038321410427095685,
           0.030758439549623255, 0.08150400702314292, 0.048278203179960955,
           0.07414816531958743, 0.07857960227001794, 0.09763304621423306,
           0.08249398239281651, 0.13887273786344953]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.0005)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)
end

@testset "GMD" begin
    portfolio = Portfolio2(; prices = prices[(end - 200):end],
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = GMD2(; owa = OWASettings(; approx = false))

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04803547750881911, 0.05110378808817606, 0.045744535143979456,
           0.0401591940551555, 0.04820271181945354, 0.049480772022897544,
           0.029817165753273638, 0.06382434140445212, 0.047278147548602704,
           0.04657225285433487, 0.06792646315264042, 0.026940518272695023,
           0.023139927344914143, 0.0730447700398832, 0.028269792959212576,
           0.04916435454256209, 0.056826111828608854, 0.07493459440699926,
           0.05401678163830339, 0.07551829961503662]
    w2t = [0.0052168653727435324, 0.010875041401541045, 0.015252697855293935,
           0.018481723701732653, 0.028493184679311317, 0.02747420746238557,
           0.023248044382348478, 0.04628343130752336, 0.03778803682979785,
           0.04317310948945797, 0.06425862358082006, 0.028232000044163038,
           0.025155426631292603, 0.09117101688071873, 0.03757808278184953,
           0.06309686337156944, 0.09825614373133428, 0.11627448958576403,
           0.09026796539690063, 0.12942304551345193]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.001)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.0005)

    rm = GMD2(; owa = OWASettings(; approx = true))

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04804176713883893, 0.05104724299739746, 0.04574875402236941,
           0.04012962719713228, 0.04815875955840716, 0.04933616200008279,
           0.029706105017130887, 0.06367477406014153, 0.04748519555775158,
           0.04665149434353952, 0.0675631982393003, 0.02704670146538231,
           0.023214330721524467, 0.07291756529139848, 0.028301862554273645,
           0.049102692654923646, 0.05684559150520732, 0.07510436307248146,
           0.05424976913183064, 0.07567404347088627]
    w2t = [0.005221701242699852, 0.01088142933833096, 0.015245989565332926,
           0.01844755738581367, 0.02832444832297849, 0.027741417167144008,
           0.023135277022809514, 0.0461316854171331, 0.03787994983663463,
           0.04323850965363985, 0.06418371868592332, 0.02851724622975968,
           0.025447120396264768, 0.0909237645940213, 0.03781632936438383,
           0.06292575605625167, 0.09814660172282262, 0.11653789569534632,
           0.09019739278381154, 0.12905620951889796]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.005)
end

@testset "TG" begin
    portfolio = Portfolio2(; prices = prices[(end - 125):end],
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = TG2(; owa = OWASettings(; approx = false))

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.038764768981499845, 0.07263706983700537, 0.04228339226242435,
           0.0435411990063813, 0.048774699348804414, 0.039676776202074104,
           0.03231569528296526, 0.05297550687912629, 0.03844632457767705,
           0.04978778365499336, 0.055347007035070256, 0.06518988423572826,
           0.034132382278224085, 0.05239068690649956, 0.04926897149717854,
           0.053690745556068495, 0.049275704404779644, 0.04936498510427762,
           0.04217840509303159, 0.08995801185619062]
    w2t = [0.0038201872636861034, 0.013371366978310375, 0.01154238813703943,
           0.017571807532298662, 0.019897636478913447, 0.022692934297274116,
           0.0214029490852748, 0.03786440897005283, 0.03148120903979511,
           0.048284735121110235, 0.057256192158012524, 0.07488747144376684,
           0.03855613540473131, 0.07497989888742033, 0.06385953604838171,
           0.0813630397309213, 0.07603774634340942, 0.07779457508307593,
           0.07642790345721767, 0.1509078785393079]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)

    rm = TG2(; owa = OWASettings(; approx = true))

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.038740466095828284, 0.07002876653572743, 0.04194119773895824,
           0.04494827890397753, 0.04948102362024745, 0.039928092992390314,
           0.032657023732753664, 0.053336176036027395, 0.03819620335043525,
           0.04917066142522329, 0.05460927920861385, 0.06730801763590079,
           0.03272434270065661, 0.053073199358552196, 0.04968699383594111,
           0.05372969101047005, 0.049217774918763224, 0.04857885542254007,
           0.0424137253877012, 0.0902302300892922]
    w2t = [0.0037698728023252306, 0.013204506974378116, 0.011539552353705813,
           0.01732091437727659, 0.020227283064655915, 0.02242528734969352,
           0.02138248981408943, 0.03823152182275527, 0.031463918665103986,
           0.047817931612775345, 0.05671602710914603, 0.07436882289918073,
           0.0385819640616973, 0.07417489965002216, 0.06486320638120524, 0.0803528521152473,
           0.07534879190537207, 0.07800719011461071, 0.07566097593127415,
           0.15454199099548505]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.1)
end

@testset "RTG" begin
    portfolio = Portfolio2(; prices = prices[(end - 200):end],
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    rm = RTG2(;)

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.04253504052591617, 0.04991570998894274, 0.04298361867322103,
           0.034781792979486484, 0.04824909571411739, 0.053204991729475064,
           0.03659379499268876, 0.06323702251704609, 0.04112095440849473,
           0.04436007858442393, 0.08117585058855487, 0.04385098964509536, 0.025563115008997,
           0.06928261553950593, 0.036317498603680005, 0.048488866652573646,
           0.04932884872222516, 0.0627983762077992, 0.04589951058488181,
           0.08031222833287464]
    w2t = [0.004134421101898551, 0.009449899778879613, 0.013177971399256623,
           0.015383843365554328, 0.022528960922832207, 0.03056818096247944,
           0.026476370967053044, 0.054329900489308594, 0.03456852749670496,
           0.04087959537828222, 0.07959616159750021, 0.0472103000499494,
           0.030673162574130136, 0.08496604265850279, 0.05293487923242808,
           0.06603208610124492, 0.07860589834427198, 0.10022583042887317,
           0.07990952964634396, 0.1283484375045057]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-6)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.1)
end

@testset "OWA" begin
    portfolio = Portfolio2(; prices = prices[(end - 200):end],
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    rm = OWA2(; owa = OWASettings(; approx = false))

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.038764768981499845, 0.07263706983700537, 0.04228339226242435,
           0.0435411990063813, 0.048774699348804414, 0.039676776202074104,
           0.03231569528296526, 0.05297550687912629, 0.03844632457767705,
           0.04978778365499336, 0.055347007035070256, 0.06518988423572826,
           0.034132382278224085, 0.05239068690649956, 0.04926897149717854,
           0.053690745556068495, 0.049275704404779644, 0.04936498510427762,
           0.04217840509303159, 0.08995801185619062]
    w2t = [0.0038201872636861034, 0.013371366978310375, 0.01154238813703943,
           0.017571807532298662, 0.019897636478913447, 0.022692934297274116,
           0.0214029490852748, 0.03786440897005283, 0.03148120903979511,
           0.048284735121110235, 0.057256192158012524, 0.07488747144376684,
           0.03855613540473131, 0.07497989888742033, 0.06385953604838171,
           0.0813630397309213, 0.07603774634340942, 0.07779457508307593,
           0.07642790345721767, 0.1509078785393079]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.05)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.05)

    rm = OWA2(; owa = OWASettings(; approx = true))

    portfolio.risk_budget = []
    w1 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc1 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = optimise2!(portfolio; type = RP2(), rm = rm)
    rc2 = calc_risk_contribution(portfolio; type = :RP2, rm = rm)
    lrc2, hrc2 = extrema(rc2)

    w1t = [0.038740466095828284, 0.07002876653572743, 0.04194119773895824,
           0.04494827890397753, 0.04948102362024745, 0.039928092992390314,
           0.032657023732753664, 0.053336176036027395, 0.03819620335043525,
           0.04917066142522329, 0.05460927920861385, 0.06730801763590079,
           0.03272434270065661, 0.053073199358552196, 0.04968699383594111,
           0.05372969101047005, 0.049217774918763224, 0.04857885542254007,
           0.0424137253877012, 0.0902302300892922]
    w2t = [0.0037698728023252306, 0.013204506974378116, 0.011539552353705813,
           0.01732091437727659, 0.020227283064655915, 0.02242528734969352,
           0.02138248981408943, 0.03823152182275527, 0.031463918665103986,
           0.047817931612775345, 0.05671602710914603, 0.07436882289918073,
           0.0385819640616973, 0.07417489965002216, 0.06486320638120524, 0.0803528521152473,
           0.07534879190537207, 0.07800719011461071, 0.07566097593127415,
           0.15454199099548505]
    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 5.0e-5)
    @test isapprox(hrc1 / lrc1, 1, rtol = 0.1)
    @test isapprox(hrc2 / lrc2, 20, rtol = 0.1)
end
