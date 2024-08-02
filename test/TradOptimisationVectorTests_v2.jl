using CSV, TimeSeries, DataFrames, StatsBase, Statistics, LinearAlgebra, Test, Clarabel,
      PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Add Skew and SSkew to SD" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)

    rm = Skew2(; settings = RiskMeasureSettings(; scale = 1.0))
    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm.settings.scale = 0.99
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 5e-6)

    rm = SSkew2(; settings = RiskMeasureSettings(; scale = 1.0))
    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm.settings.scale = 0.99
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 1e-4)

    rm = SD2()
    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)

    rm = [SD2(), Skew2(; settings = RiskMeasureSettings(; scale = 0.0))]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 3e-4)

    rm = [SD2(), Skew2(; settings = RiskMeasureSettings(; scale = 2))]
    w = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    wt = [0.0026354275749322443, 0.05520947741035115, 2.88292486305246e-7,
          0.011217444462648793, 0.015540235791726633, 0.007294887210979084,
          3.899501931693162e-8, 0.1384846686508059, 5.619219962894404e-8,
          1.4264636900253708e-7, 0.2855982912592649, 8.550398887290524e-8,
          4.0185944557342566e-8, 0.11727545683980922, 0.005180482430773081,
          0.016745180622565338, 0.0077834334790627055, 0.20483183287545345,
          7.577961384734552e-8, 0.1322024537960061]
    @test isapprox(w.weights, wt)

    rm = [SD2(), Skew2(; settings = RiskMeasureSettings(; scale = 8))]
    w = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    wt = [8.413927417714243e-7, 0.09222294098507178, 1.6168463788897735e-7,
          2.0594000236277162e-7, 0.008523442957645658, 3.007500480370547e-7,
          6.833538384822706e-8, 0.13619418248362034, 9.979458409901339e-8,
          1.5596045505028015e-7, 0.26494454649109994, 3.4315324995498946e-6,
          1.2825613036862424e-7, 0.0783181629157472, 0.02532294038010334,
          0.01907855067328539, 0.012932625739071507, 0.21592581988533274,
          1.422385714567375e-7, 0.14653125160396763]
    @test isapprox(w.weights, wt)

    rm = [SD2(), SSkew2(; settings = RiskMeasureSettings(; scale = 0.0))]
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 3e-4)

    rm = [SD2(), SSkew2(; settings = RiskMeasureSettings(; scale = 2))]
    w = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    wt = [1.384008546759806e-6, 0.0316494420628888, 1.4466615477601905e-6,
          0.015775935372681668, 0.010442899482149982, 0.009851951563574745,
          1.6845564712725654e-7, 0.1404230153792723, 2.93065068940981e-7,
          5.00892434748868e-7, 0.32532989744017604, 3.1063572739077716e-7,
          1.7332147477485165e-7, 0.1184225153788876, 1.25268476291211e-6,
          0.014302557449595256, 2.0736860865331673e-6, 0.2083923849842472,
          3.9292677008851197e-7, 0.12540140454845938]
    @test isapprox(w.weights, wt)

    rm = [SD2(), SSkew2(; settings = RiskMeasureSettings(; scale = 8))]
    w = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    wt = [1.5365064547283644e-7, 0.02437633461456116, 1.299823053389551e-7,
          3.5309417854060804e-7, 3.017455267702621e-6, 2.6113474157486046e-7,
          3.341100393369674e-8, 0.13768001144500144, 5.584135855499354e-8,
          1.1036943651763183e-7, 0.38090454974359306, 7.778862184342059e-8,
          4.2133989399698356e-8, 0.09436174496182163, 3.53865987048023e-7,
          0.013926597934786485, 4.4441759524485204e-7, 0.21941318550910147,
          7.686992020172018e-8, 0.12933246577608337]
    @test isapprox(w.weights, wt)
end