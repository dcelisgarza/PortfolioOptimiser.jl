using CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries, Distances

struct POCorDist <: Distances.UnionMetric end
function Distances.pairwise(::POCorDist, mtx, i)
    return sqrt.(clamp!((1 .- mtx) / 2, 0, 1))
end

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:HRP), $(:HERC), $(:NCO), $(:GMD), blank $(:OWA) and owa_w = owa_gmd(T)" begin
    portfolio = HCPortfolio(; prices = prices[(end - 200):end],
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :GMD, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :ward,
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :GMD, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Min_Risk,
                                         rf = rf, l = l), cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Utility,
                                         rf = rf, l = l), cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Sharpe,
                                         rf = rf, l = l), cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Max_Ret,
                                         rf = rf, l = l), cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Equal,
                                         rf = rf, l = l), cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Min_Risk,
                                         l = l, rf = rf),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Sharpe,
                                           rf = rf, l = l), cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Sharpe,
                                         l = l, rf = rf),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Min_Risk,
                                           rf = rf, l = l), cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Equal,
                                          l = l, rf = rf),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Sharpe,
                                            rf = rf, l = l), cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Sharpe,
                                          l = l, rf = rf),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :GMD, obj = :Equal,
                                            rf = rf, l = l), cluster = false)
    w12 = optimise!(portfolio; type = :HRP, rm = :OWA, rf = rf,
                    cluster_opt = ClusterOpt(; linkage = :ward,
                                             max_k = ceil(Int,
                                                          sqrt(size(portfolio.returns, 2)))))
    w13 = optimise!(portfolio; type = :HERC, rm = :OWA, rf = rf, cluster = false)
    w14 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Min_Risk,
                                          rf = rf, l = l), cluster = false)
    w15 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Utility,
                                          rf = rf, l = l), cluster = false)
    w16 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                          rf = rf, l = l), cluster = false)
    w17 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Max_Ret,
                                          rf = rf, l = l), cluster = false)
    w18 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Equal,
                                          rf = rf, l = l), cluster = false)
    w19 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Min_Risk,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                            rf = rf, l = l), cluster = false)
    w20 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :OWA,
                                            obj = :Min_Risk, rf = rf, l = l),
                    cluster = false)
    w21 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Equal,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                            rf = rf, l = l), cluster = false)
    w22 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Equal,
                                            rf = rf, l = l), cluster = false)
    portfolio.owa_w = owa_gmd(200)
    w23 = optimise!(portfolio; type = :HRP, rm = :OWA, rf = rf,
                    cluster_opt = ClusterOpt(; linkage = :ward,
                                             max_k = ceil(Int,
                                                          sqrt(size(portfolio.returns, 2)))))
    w24 = optimise!(portfolio; type = :HERC, rm = :OWA, rf = rf, cluster = false)
    w25 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Min_Risk,
                                          rf = rf, l = l), cluster = false)
    w26 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Utility,
                                          rf = rf, l = l), cluster = false)
    w27 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                          rf = rf, l = l), cluster = false)
    w28 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Max_Ret,
                                          rf = rf, l = l), cluster = false)
    w29 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Equal,
                                          rf = rf, l = l), cluster = false)
    w30 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Min_Risk,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                            rf = rf, l = l), cluster = false)
    w31 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :OWA,
                                            obj = :Min_Risk, rf = rf, l = l),
                    cluster = false)
    w32 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Equal,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                            rf = rf, l = l), cluster = false)
    w33 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Sharpe,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :OWA, obj = :Equal,
                                            rf = rf, l = l), cluster = false)

    w1t = [0.03354709337144272, 0.06342772851033758, 0.028766393880563925,
           0.045655364086333534, 0.05858145457649635, 0.028471101324400793,
           0.031678253063494984, 0.04666120290066344, 0.05852502668524639,
           0.055186672640821416, 0.07991711135169646, 0.03432328155910596,
           0.011681317836655799, 0.06380824647335324, 0.016382117794047368,
           0.050654206083462806, 0.07681116197440706, 0.06184789401548876,
           0.07031525919759005, 0.08375911267439137]
    w2t = [0.10476716806052858, 0.10666764518134146, 0.08983710120012676,
           0.07100744217155575, 0.09074103825321128, 0.02015614093684567,
           0.0492689471985913, 0.04695299820157166, 0.025007783393142056,
           0.022905472495110643, 0.02692824084010223, 0.0201422107106658,
           0.011754366824707138, 0.03667315956832977, 0.011597734535749487,
           0.0297258200838875, 0.11897834625959776, 0.035546466351216284,
           0.029184659238644054, 0.05215725849507475]
    w3t = [0.011848923867459478, 0.027512304541726887, 0.004467805367217499,
           0.0011130712535245661, 0.0068012489288442855, 0.01616434613649352,
           4.803678169296482e-12, 0.11203451597411904, 0.00871900058453107,
           9.727653911104655e-9, 0.058555631749153764, 0.0025218097707053193,
           2.111861556694041e-12, 0.25989975356568945, 2.4143390478138007e-11,
           0.02217645782831401, 0.05925659472288109, 0.22201740753858024,
           0.04188280202702492, 0.14502831638502214]
    w4t = [0.0015584174790040032, 0.03577937128803695, 0.0024784915877609044,
           0.0017813700947509486, 0.01936847193123288, 4.56622597593501e-10,
           8.37192715394709e-12, 0.11408184409088849, 0.002272592900231678,
           0.00326123958176244, 0.05148328598982736, 0.0013071298338763833,
           7.024138691326877e-12, 0.22489613643212875, 2.914539463705695e-11,
           0.02541976751506721, 0.09205025838067595, 0.21246532978762478,
           0.0744318620717884, 0.1373644305341788]
    w5t = [1.1919004092824724e-12, 2.339720336421852e-12, 1.0428923726801206e-12,
           2.0661465205781344e-12, 0.16806533295552967, 4.6054775111852954e-12,
           2.6932333116534955e-12, 0.06273139454492406, 0.048281413824618744,
           1.9633242668226367e-9, 6.735967177298309e-13, 1.1417740375464447e-12,
           1.0544095919192656e-12, 5.554623298492499e-13, 3.876182801683786e-12,
           0.062338209004076724, 0.4220554282143806, 1.690694329647511e-11,
           0.23652821945375865, 1.2395785157624276e-12]
    w6t = [6.73841338839525e-10, 1.1532620090588712e-9, 7.275021485896715e-10,
           1.639344842949527e-9, 0.9999999548054548, 2.536145233985994e-18,
           3.2738173681288396e-10, 1.29989979644734e-16, 1.0862422285522571e-8,
           2.277641034135773e-17, 7.636261501402768e-18, 2.2969579300621857e-17,
           1.2067041288720072e-17, 7.391804256103598e-18, 3.9357442806481615e-18,
           1.300785839282474e-8, 1.6802932009983537e-8, 1.1131110681181213e-17,
           2.554318444434214e-16, 3.7131252628268476e-17]
    w8t = [0.10674711116016065, 0.2478587139253519, 0.040250517389693215,
           0.01002767358101371, 0.06127254116545597, 8.639377268155679e-13,
           4.327639988672433e-11, 1.969672460015776e-10, 4.660054592679874e-13,
           5.199150733499076e-19, 3.1296298011957434e-12, 4.433579430025314e-12,
           3.7128518041236686e-21, 1.3890892981346176e-11, 1.2903946569303e-21,
           3.898830451074731e-11, 0.5338434422072305, 1.1866190736248061e-11,
           2.2385150918169878e-12, 2.549734590472301e-10]
    w9t = [8.032797954495843e-13, 1.5768516048934089e-12, 7.028560148802512e-13,
           1.392476872642417e-12, 0.11326742169671428, 5.797210618350383e-12,
           1.8151012339910275e-12, 0.12227351613130505, 0.06077491947648498,
           2.471362472022988e-9, 8.478994925119785e-13, 2.2255001217653787e-12,
           2.0552128512656386e-12, 6.991961439123301e-13, 4.879200482906079e-12,
           0.12150713465803867, 0.28444372986542604, 2.1281856433807242e-11,
           0.29773327565417584, 2.4161366847114117e-12]
    w16t = [1.1919004092824724e-12, 2.339720336421852e-12, 1.0428923726801206e-12,
            2.0661465205781344e-12, 0.16806533295552967, 4.6054775111852954e-12,
            2.6932333116534955e-12, 0.06273139454492406, 0.048281413824618744,
            1.9633242668226367e-9, 6.735967177298309e-13, 1.1417740375464447e-12,
            1.0544095919192656e-12, 5.554623298492499e-13, 3.876182801683786e-12,
            0.062338209004076724, 0.4220554282143806, 1.690694329647511e-11,
            0.23652821945375865, 1.2395785157624276e-12]
    w20t = [8.032797954495843e-13, 1.5768516048934089e-12, 7.028560148802512e-13,
            1.392476872642417e-12, 0.11326742169671428, 5.797210618350383e-12,
            1.8151012339910275e-12, 0.12227351613130505, 0.06077491947648498,
            2.471362472022988e-9, 8.478994925119785e-13, 2.2255001217653787e-12,
            2.0552128512656386e-12, 6.991961439123301e-13, 4.879200482906079e-12,
            0.12150713465803867, 0.28444372986542604, 2.1281856433807242e-11,
            0.29773327565417584, 2.4161366847114117e-12]
    w27t = [1.1919004092824724e-12, 2.339720336421852e-12, 1.0428923726801206e-12,
            2.0661465205781344e-12, 0.16806533295552967, 4.6054775111852954e-12,
            2.6932333116534955e-12, 0.06273139454492406, 0.048281413824618744,
            1.9633242668226367e-9, 6.735967177298309e-13, 1.1417740375464447e-12,
            1.0544095919192656e-12, 5.554623298492499e-13, 3.876182801683786e-12,
            0.062338209004076724, 0.4220554282143806, 1.690694329647511e-11,
            0.23652821945375865, 1.2395785157624276e-12]
    w31t = [8.032797954495843e-13, 1.5768516048934089e-12, 7.028560148802512e-13,
            1.392476872642417e-12, 0.11326742169671428, 5.797210618350383e-12,
            1.8151012339910275e-12, 0.12227351613130505, 0.06077491947648498,
            2.471362472022988e-9, 8.478994925119785e-13, 2.2255001217653787e-12,
            2.0552128512656386e-12, 6.991961439123301e-13, 4.879200482906079e-12,
            0.12150713465803867, 0.28444372986542604, 2.1281856433807242e-11,
            0.29773327565417584, 2.4161366847114117e-12]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 0.0001)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-5)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isempty(w7)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-6)
    @test isapprox(w9.weights, w9t, rtol = 5.0e-6)
    @test isempty(w10)
    @test isempty(w11)
    @test isapprox(w1.weights, w12.weights)
    @test isapprox(w2.weights, w13.weights)
    @test isapprox(w3.weights, w14.weights)
    @test isapprox(w4.weights, w15.weights)
    @test isapprox(w16.weights, w16t)
    @test isapprox(w6.weights, w17.weights)
    @test isempty(w18)
    @test isapprox(w8.weights, w19.weights)
    @test isapprox(w20.weights, w20t, rtol = 5.0e-6)
    @test isempty(w21)
    @test isempty(w22)
    @test isapprox(w1.weights, w23.weights)
    @test isapprox(w2.weights, w24.weights)
    @test isapprox(w3.weights, w25.weights)
    @test isapprox(w4.weights, w26.weights)
    @test isapprox(w27.weights, w27t)
    @test isapprox(w6.weights, w28.weights)
    @test isempty(w29)
    @test isapprox(w8.weights, w30.weights)
    @test isapprox(w31.weights, w31t, rtol = 5.0e-6)
    @test isempty(w32)
    @test isempty(w33)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:RG)" begin
    portfolio = HCPortfolio(; prices = prices[(end - 200):end],
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :RG, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :ward,
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :RG, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Min_Risk,
                                         rf = rf, l = l), cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Utility,
                                         rf = rf, l = l), cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Sharpe,
                                         rf = rf, l = l), cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Max_Ret,
                                         rf = rf, l = l), cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Equal,
                                         rf = rf, l = l), cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Min_Risk,
                                         rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Sharpe,
                                           rf = rf, l = l), cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Sharpe,
                                         rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Min_Risk,
                                           rf = rf, l = l), cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Equal,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Sharpe,
                                            rf = rf, l = l), cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Sharpe,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RG, obj = :Equal,
                                            rf = rf, l = l), cluster = false)

    w1t = [0.039151540315525625, 0.07627752703443909, 0.03420720478669321,
           0.061642909486682454, 0.043487500726191786, 0.03514534246543564,
           0.03473426200213131, 0.024332660912337017, 0.062392653658646484,
           0.050478792156552536, 0.07427031572948553, 0.025623553443776547,
           0.01982939580430615, 0.05483500235610641, 0.021020503147620707,
           0.0650700102962522, 0.0967257788305686, 0.04693911333638707, 0.05981178374269532,
           0.07402414976816632]
    w2t = [0.11122184940320101, 0.12054938642967195, 0.0971759616257379,
           0.08822499980588872, 0.05945650012179683, 0.020587913173037594,
           0.0497126155127018, 0.029505385383710007, 0.02482281390332758,
           0.024346665863583836, 0.0215685500168411, 0.015148051466011928,
           0.024044800001118423, 0.02774160858061945, 0.012313674111509739,
           0.03846788334898015, 0.13224435032563, 0.0237469946812973, 0.028848105338418063,
           0.05027189090691677]
    w3t = [9.783907111921854e-20, 8.26257806085684e-11, 1.1297889735785012e-18,
           1.6376309712151834e-11, 1.4908869571860509e-19, 0.09101063503985216,
           4.885844076786679e-11, 0.11687711727673525, 3.0291517159871075e-10,
           0.0487740883702542, 0.20910208935945288, 0.034852235943897275,
           1.8772759743994373e-12, 0.005865742203914291, 6.903682877048389e-10,
           0.12833531973049594, 1.24779499459125e-10, 0.12335980790599081,
           0.05046909612089944, 0.19135386678070704]
    w4t = [6.772572377569391e-12, 0.005489128573775947, 7.411588170243905e-11,
           0.001087937300879311, 1.2187165667451235e-11, 0.07643995214422071,
           0.0032458424172611433, 0.1348750585893307, 1.5538160781014355e-10,
           0.04574016405885244, 0.18754877923293478, 3.755821663659992e-9,
           2.6422766797571878e-11, 0.004163587103317427, 6.603489792346652e-10,
           0.15458959581426135, 0.008289552184702589, 0.12885891661174992,
           0.03331344854339574, 0.2163580327342674]
    w5t = [3.259968057806289e-10, 6.467197453623986e-10, 1.037710087476616e-9,
           7.045827239239153e-10, 0.1169005039160808, 2.231986888527516e-17,
           7.612089067274455e-11, 1.0597915762577247e-7, 9.980946033095893e-14,
           5.049276354432049e-14, 1.6875808503416431e-16, 1.4642938519422182e-17,
           3.6543115369408837e-17, 1.7073299177144147e-16, 3.3327624457139784e-17,
           1.1561387802690642e-7, 0.8830990900283096, 3.515771557142767e-16,
           1.816712925576095e-7, 1.3365695977541778e-16]
    w6t = [2.898253958594781e-9, 4.945549821043192e-9, 3.1231712621304313e-9,
           6.944563509697485e-9, 0.9999998082859862, 4.5599059400155283e-17,
           1.384681229591466e-9, 2.2952817707268487e-15, 4.6862438183951184e-8,
           4.0488671410380496e-16, 1.377756097602281e-16, 4.0784507628068783e-16,
           2.0750113284826874e-16, 1.3348645572372e-16, 6.992214575472037e-17,
           5.5463739525073414e-8, 7.009160721324952e-8, 2.0039917512717117e-16,
           4.513305298008089e-15, 6.730870804436965e-16]
    w7t = [0.03667356697916447, 0.05908440289413244, 0.038671098737671356,
           0.04889076250352358, 0.023503491312381584, 0.050135037337424894,
           0.055304174939673184, 0.05152546468534146, 0.04363603725630459,
           0.046721642969149484, 0.05721797425548506, 0.04259085348087018,
           0.030922118304831544, 0.037211819319695874, 0.03399406360487713,
           0.06307906023250544, 0.07051852731674402, 0.04790720822288262,
           0.047695673189754144, 0.11471702245758694]
    w8t = [3.5885805146178224e-10, 0.3030581371073058, 4.1438851062614854e-9,
           0.06006568261749465, 5.468334708059983e-10, 2.7815384260197444e-11,
           0.1792049398144125, 1.2582286990307225e-9, 9.257931111646789e-20,
           1.4906719520915145e-11, 6.390742095796252e-11, 3.751981954360757e-10,
           2.0209623252407234e-20, 1.7927341491649753e-12, 2.1099577203429027e-19,
           1.3815808102270493e-9, 0.457671230174653, 3.770219225112278e-11,
           1.5424761087019648e-11, 2.0600005583976924e-9]
    w9t = [1.5457637418785066e-10, 3.0665206401157264e-10, 4.920461180476515e-10,
           3.3408868077329444e-10, 0.055430162859448934, 1.0820260045247954e-19,
           3.609388518996897e-11, 0.2514859647998535, 4.838578225113628e-16,
           2.447795884290331e-16, 8.181080158640071e-19, 3.474733715156744e-11,
           8.671592444544259e-11, 8.276819988350937e-19, 1.6156616563035538e-19,
           0.2743489221014387, 0.41873494759646424, 1.704381092178718e-18,
           8.807088600446411e-10, 3.171647164815927e-10]
    w10t = [0.11024802407425706, 0.17761944662721826, 0.11625300116105304,
            0.1469753396106673, 0.0706561616302198, 1.008405757585041e-11,
            0.16625533081142696, 1.8087359924263726e-10, 8.77686215955116e-12,
            9.397494502064122e-12, 1.1508704838145715e-11, 1.4950978144390012e-10,
            1.0854816871923674e-10, 7.484708268918382e-12, 6.837495548703011e-12,
            2.2143102892465948e-10, 0.21199269494877682, 9.63595664178708e-12,
            9.593408923331814e-12, 4.026995364916173e-10]
    w11t = [1.208689205340027e-10, 2.3978246450236555e-10, 3.8474885605151247e-10,
            2.612361586294147e-10, 0.04334287167134335, 3.8324992367491514e-11,
            2.8223128946523498e-11, 0.15174655905941592, 1.7138079193247427e-7,
            8.670009610702873e-8, 2.8977107142209354e-10, 2.0966533275223306e-11,
            5.2324363948997076e-11, 2.931621435660069e-10, 5.7226185311023594e-11,
            0.1655420609404017, 0.32742417055493855, 6.036859749755613e-10,
            0.31194407711131444, 1.913770989941342e-10]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.001)
    @test isapprox(w11.weights, w11t, rtol = 5.0e-6)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:RCVaR)" begin
    portfolio = HCPortfolio(; prices = prices[(end - 200):end],
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :RCVaR, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :ward,
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :RCVaR, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Min_Risk,
                                         rf = rf, l = l), cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Utility,
                                         rf = rf, l = l), cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Sharpe,
                                         rf = rf, l = l), cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Max_Ret,
                                         rf = rf, l = l), cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Equal,
                                         rf = rf, l = l), cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Min_Risk,
                                         rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Sharpe,
                                           rf = rf, l = l), cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Sharpe,
                                         rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RCVaR,
                                           obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Equal,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RCVaR,
                                            obj = :Sharpe, rf = rf, l = l), cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Sharpe,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RCVaR, obj = :Equal,
                                            rf = rf, l = l), cluster = false)

    w1t = [0.0330739462814986, 0.06546151173358211, 0.028106124164109084,
           0.049902753874119445, 0.05369883279541233, 0.03061438896454119,
           0.03224227015686936, 0.037953644196072764, 0.06124001970274264,
           0.05512304815654814, 0.08412430173354701, 0.03438416406580439,
           0.012508299633988548, 0.06220968859081304, 0.01951447825505096,
           0.058199439395561846, 0.07082126695922601, 0.06186094004512289,
           0.06952617838102662, 0.07943470291436304]
    w2t = [0.10234965262039762, 0.10970599002859706, 0.08697637772700602,
           0.08022532720069422, 0.08534143017862801, 0.020609217000164205,
           0.05183374608048453, 0.04285710935148786, 0.0254943411163643,
           0.023952403960552822, 0.026749485815295067, 0.01875601434494886,
           0.014124323937001347, 0.034623620684989975, 0.013136898386871272,
           0.031746868066417334, 0.11255343728586818, 0.034429520093362816,
           0.030210940180345928, 0.054323295940522494]
    w3t = [2.664150760557517e-10, 0.030692934306106395, 0.011955189029992448,
           7.0117020129599e-11, 1.4437147999148644e-10, 0.09656433746368384,
           1.8466761981335824e-10, 0.06944343869086692, 1.0384316827963105e-9,
           0.00204981055459017, 0.139465125395689, 0.010312687422429308,
           1.4066975033131039e-11, 0.30331680655348964, 0.003109198569838757,
           0.025430278701225738, 0.0354402935187825, 0.1342376554682851,
           1.7913351495508077e-9, 0.1379822408156151]
    w4t = [4.1882397036196794e-10, 0.03207630903671382, 0.012494026912562552,
           1.5085397882609554e-10, 3.712525164592465e-10, 0.07964082999127103,
           3.885048690264717e-10, 0.06989462203895716, 1.5859184840658755e-9,
           0.007891814612454168, 0.1368684568787778, 0.010379691061455244,
           4.441061174768386e-11, 0.31411339292703533, 1.0160361998532883e-9,
           0.025595505747971096, 0.03703764147533282, 0.13512895939525288,
           6.133105004828555e-9, 0.13887873981331042]
    w5t = [9.101898748943287e-11, 4.3442922199962347e-10, 1.0832444769844242e-10,
           2.596005519121047e-10, 0.16014986748843063, 2.7943188774447678e-12,
           4.746517289878347e-11, 0.10756967880345782, 0.0007130182647158488,
           4.781582511881672e-10, 1.7582486434560834e-11, 7.097066792849838e-12,
           9.15462113409717e-12, 1.8493941335416535e-11, 3.1313265387062985e-12,
           0.16492131232198715, 0.49858112463554966, 5.380421401464259e-11,
           0.06806499687380671, 8.099762673912902e-11]
    w6t = [4.475199930644048e-8, 7.64683932639744e-8, 4.8269213692033513e-8,
           1.0802397270528268e-7, 0.999998206908014, 2.7547207359917206e-15,
           2.15537156236641e-8, 3.486852385218159e-14, 1.8062550275507348e-7,
           2.4598334723466123e-14, 8.309523299193478e-15, 6.1833711173638524e-15,
           3.1904367216872086e-15, 8.046981162305303e-15, 4.2480437211353606e-15,
           2.1495161457268444e-7, 1.0984471844467989e-6, 1.2099213531180133e-14,
           2.7499605593254544e-13, 1.0105537476296299e-14]
    w7t = [0.04834771832465415, 0.054067761388698087, 0.04075649514015394,
           0.032935383643753674, 0.03633777412179309, 0.04415458312649693,
           0.031438759869257996, 0.08288576187670506, 0.03942079220842434,
           0.039926443638758954, 0.062227478636579556, 0.04025323184104151,
           0.029625421447347793, 0.060310731846418796, 0.032942489135279925,
           0.06940251164693978, 0.04635232141187277, 0.05822960840978754, 0.043917274605683,
           0.10646745768035311]
    w8t = [3.41171052089434e-9, 0.3930536080748997, 0.1530981083982288,
           8.97918311574277e-10, 1.848820661714227e-9, 3.152380015400095e-11,
           2.364852885629888e-9, 4.0969240329767493e-10, 3.3900002528744095e-19,
           6.691685561531865e-13, 4.552892772737988e-11, 6.084130875720064e-11,
           8.299031437810702e-20, 9.901894057666953e-11, 1.0150098569418944e-12,
           1.5002989762667217e-10, 0.4538482733473797, 4.38224000212844e-11,
           5.847882639335079e-19, 8.140477620037939e-10]
    w9t = [5.930617064419704e-11, 2.830654820866539e-10, 7.058206597703665e-11,
           1.6915058116653353e-10, 0.10435048369459281, 1.1768044520479227e-11,
           3.0927367148732375e-11, 0.11098042792381643, 0.003002817878381814,
           2.0137242149551567e-9, 7.404719798187162e-11, 7.322095951531874e-12,
           9.44488988201626e-12, 7.788571546040906e-11, 1.3187324615346162e-11,
           0.17015052958086146, 0.3248655920397322, 2.2659202964434094e-10,
           0.28665014575204617, 8.356584658718105e-11]
    w10t = [0.1665805842261949, 0.18628881759119206, 0.1404252570073581,
            0.11347785664360195, 0.12520068893511946, 1.1391157007615839e-10,
            0.10832128521973036, 1.0538904473274267e-9, 1.0169916724710039e-10,
            1.0300366486121344e-10, 1.6053667120049547e-10, 5.11818864311515e-10,
            3.76686513515463e-10, 1.5559177939435756e-10, 8.498620967309399e-11,
            8.824512484308473e-10, 0.1597055052149736, 1.5022282284663522e-10,
            1.132993530811316e-10, 1.353731136204906e-9]
    w11t = [4.7530362760574044e-11, 2.2686012100314036e-10, 5.656732113773022e-11,
            1.3556411409938056e-10, 0.08363069627276075, 1.3395411164711788e-11,
            2.4786442352311022e-11, 0.12881182411716863, 0.0034180683174398706,
            2.2921959366076824e-9, 8.428695700767069e-11, 8.498548379406195e-12,
            1.0962414878445594e-11, 8.865629124994528e-11, 1.5010959134091796e-11,
            0.19748887709153934, 0.2603604189977824, 2.5792674377211685e-10,
            0.32629011184407497, 9.699249979626584e-11]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.01)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.01)
    @test isapprox(w11.weights, w11t, rtol = 5.0e-5)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:TG)" begin
    portfolio = HCPortfolio(; prices = prices[(end - 200):end],
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :TG, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :ward,
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :TG, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Min_Risk,
                                         rf = rf, l = l), cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Utility,
                                         rf = rf, l = l), cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Sharpe,
                                         rf = rf, l = l), cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Max_Ret,
                                         rf = rf, l = l), cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Equal,
                                         rf = rf, l = l), cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Min_Risk,
                                         rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Sharpe,
                                           rf = rf, l = l), cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Sharpe,
                                         rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Min_Risk,
                                           rf = rf, l = l), cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Equal,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Sharpe,
                                            rf = rf, l = l), cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Sharpe,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :TG, obj = :Equal,
                                            rf = rf, l = l), cluster = false)

    w1t = [0.030145659814900954, 0.07840591923890466, 0.026028557387842006,
           0.05855716810600732, 0.06477769251800919, 0.027802001185735552,
           0.03476231810348114, 0.03824054936507268, 0.05886153642717875,
           0.05874967855974159, 0.08231380905109016, 0.034694590939125525,
           0.016958016564560214, 0.053561456905606264, 0.019931906263538635,
           0.05404928225709066, 0.07363612350932364, 0.05533073813005156,
           0.060592164795308885, 0.07260083087743063]
    w2t = [0.09465401610173545, 0.1333280610887615, 0.08172677278325605, 0.0862311331123235,
           0.09580980967417972, 0.016190039007633803, 0.05119089902448161,
           0.04495450832578455, 0.02240269664720525, 0.025437055604183897,
           0.024051120641733644, 0.019599702758806327, 0.01993536467173778,
           0.027559071921068816, 0.011607018420989015, 0.030533574193871215,
           0.10891192174854439, 0.028469423344089584, 0.026234803370180776,
           0.051173007559433006]
    w3t = [1.529520120283453e-13, 0.3116613079543004, 3.84632616801094e-13,
           5.772490351823379e-13, 0.10390363976002721, 4.2909969071001324e-13,
           1.178644258289188e-12, 0.052389487483089456, 1.6380210950769943e-13,
           1.1094581622174947e-10, 0.06622174454468897, 0.023744670180066107,
           6.982091064869122e-12, 0.09824937647393665, 0.007697343961299138,
           0.0577032819777209, 6.322569288900482e-11, 0.10534891854463863,
           2.7735202554320546e-11, 0.1730802289084574]
    w4t = [2.1048602639815493e-13, 0.35409874156556276, 2.7360579421071964e-13,
           8.578028506815234e-13, 0.11805170312109313, 5.086300870253522e-13,
           6.259166098254249e-13, 0.05244929892902187, 1.2060796325379227e-12,
           1.9262816838383824e-11, 0.052635591917857945, 0.023771778286348903,
           1.8079617490318186e-11, 0.07809238166392352, 0.006118136279834698,
           0.05776915886840465, 5.719737679692879e-11, 0.08373538603839674,
           9.525759733857601e-12, 0.1732778232218077]
    w5t = [2.1627382400267953e-11, 2.1515290570132135e-10, 2.1974031498119867e-11,
           1.74378087223813e-11, 0.38240258453692777, 4.6839507089579743e-20,
           3.368858090820401e-11, 0.09614811528651238, 4.2388037126961174e-17,
           2.1539843606557263e-21, 4.109243006477553e-20, 2.393814891986364e-12,
           7.632809105311209e-13, 4.2245085024329126e-20, 4.1016043037731405e-20,
           0.10488907968805851, 0.4165602154209882, 2.3336137425633272e-20,
           4.753590811941956e-9, 8.844873039871597e-13]
    w6t = [6.777583634874746e-10, 1.1595930327449107e-9, 7.316365132060764e-10,
           1.6469289913299042e-9, 0.9999999546644848, 2.552364748847551e-18,
           3.2892874362483696e-10, 1.3036516598005108e-16, 1.0899567652070635e-8,
           2.289958048430665e-17, 7.686888022598514e-18, 2.3065081596688485e-17,
           1.2087994898809687e-17, 7.441217440720305e-18, 3.957832799481596e-18,
           1.3040941650622289e-8, 1.6850159826893964e-8, 1.1202439374005911e-17,
           2.5649550833816185e-16, 3.7333678929830684e-17]
    w8t = [3.6805802035935396e-13, 0.7499701540834167, 9.255655916393775e-13,
           1.389070560929821e-12, 0.25002984564282743, 6.569481731353971e-24,
           2.83624560841634e-12, 1.909130142226733e-11, 2.507797113970194e-24,
           1.6985715175718954e-21, 1.0138495795344833e-12, 8.652817146307999e-12,
           2.5443502404974326e-21, 1.5041900468267656e-12, 1.178457165746973e-13,
           2.1027706171898433e-11, 1.5214395058930657e-10, 1.6128834645667825e-12,
           4.246237190115947e-22, 6.307232574842378e-11]
    w9t = [1.0296803800196524e-11, 1.0243436843384442e-10, 1.0461843548513548e-11,
           8.302146408503475e-12, 0.18206199496505068, 2.0917930609472435e-12,
           1.6039144335586603e-11, 0.194807008347161, 1.8929960505301094e-9,
           9.619421336760611e-14, 1.8351358801143303e-12, 4.850141016858422e-12,
           1.5464938679866482e-12, 1.8866119906858915e-12, 1.8317245322363157e-12,
           0.21251719559378612, 0.19832445414681213, 1.0421623404980396e-12,
           0.21228934488968748, 1.79207179565958e-12]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-6)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isempty(w7)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isempty(w10)
    @test isempty(w11)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:RTG)" begin
    portfolio = HCPortfolio(; prices = prices[(end - 200):end],
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :RTG, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :ward,
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :RTG, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Min_Risk,
                                         rf = rf, l = l), cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Utility,
                                         rf = rf, l = l), cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Sharpe,
                                         rf = rf, l = l), cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Max_Ret,
                                         rf = rf, l = l), cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Equal,
                                         rf = rf, l = l), cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Min_Risk,
                                         rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Sharpe,
                                           rf = rf, l = l), cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Sharpe,
                                         rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Min_Risk,
                                           rf = rf, l = l), cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Equal,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Sharpe,
                                            rf = rf, l = l), cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Sharpe,
                                          rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; owa_approx = false, rm = :RTG, obj = :Equal,
                                            rf = rf, l = l), cluster = false)

    w1t = [0.03421556945613498, 0.06755134870542995, 0.029347182427415453,
           0.05308166337900148, 0.05313714050710668, 0.03148965688772591,
           0.03378837018951474, 0.03435940763610544, 0.060409955275512545,
           0.054204902412732556, 0.08235679662722742, 0.03237339664619154,
           0.013725384532239043, 0.06274773166446919, 0.0200112738782573,
           0.05940396376906876, 0.07564073954231806, 0.06023163289037765,
           0.06424017547355798, 0.0776837080996133]
    w2t = [0.10348056070824181, 0.1112673292168444, 0.08875675433925861,
           0.08280413070085775, 0.08199945610933444, 0.020505334951370077,
           0.05270777973488169, 0.03972896905334218, 0.02494806588285, 0.024822902306374284,
           0.02518596685561075, 0.018064997858342015, 0.015870336971512307,
           0.03280040289010039, 0.013030877889216077, 0.033148590801067475,
           0.11672625668196, 0.0314851513055045, 0.029418512513545252, 0.05324762322978608]
    w3t = [1.4385317590600485e-11, 0.030974118017270874, 0.01119425302508547,
           4.0339412325794937e-13, 3.8901545720276474e-13, 0.11906565355613985,
           0.0034083198514564383, 0.050724577954786236, 6.310637228785733e-11,
           1.179707352570277e-9, 0.15093108954224907, 0.01888712678583965,
           1.5770059414018853e-11, 0.2129948226386454, 0.019891502459598843,
           0.04868252151246453, 0.032236451374686254, 0.14494364545902894,
           6.860622525846417e-10, 0.15606591586292481]
    w4t = [1.0885132755417258e-11, 0.036226120463681184, 0.012476801675232628,
           4.3486091319990824e-13, 3.029076682353541e-13, 0.11197502123052919,
           0.003220556795784826, 0.05079890028661183, 6.103760623776914e-11,
           2.7146915208927678e-9, 0.1482883049850608, 0.01891477228434429,
           4.357058211619562e-11, 0.2133851083512482, 0.015779920824089595,
           0.04875385457857449, 0.03988780645146502, 0.14295739497714574,
           0.0010408855595568688, 0.15629454870575288]
    w5t = [1.2332003082635699e-11, 6.231218991355713e-12, 1.2306818197758671e-11,
           1.3358710371515084e-11, 0.09588041102868011, 4.815315277704397e-13,
           1.696347490104379e-11, 0.11958236042146456, 8.0299742874231e-10,
           4.289839643670496e-13, 4.11129691277955e-13, 7.87658291414414e-12,
           4.201583388090869e-12, 4.039499647978858e-13, 4.646614538032803e-13,
           0.1807672611435324, 0.5888889409158102, 3.0058888121578257e-13,
           0.014881025606576772, 5.1773161359934745e-12]
    w6t = [6.811514944214173e-10, 1.1651950156724173e-9, 7.35203052046739e-10,
           1.6535281968523325e-9, 0.9999999544422623, 2.5760631237238374e-18,
           3.3016467192415186e-10, 1.313989456560998e-16, 1.0970427777944887e-8,
           2.309685955700897e-17, 7.760029950204041e-18, 2.3253660359268447e-17,
           1.2153370536116032e-17, 7.512438643446762e-18, 3.9908115449258506e-18,
           1.3114290523554581e-8, 1.6907776487647915e-8, 1.1307827725842743e-17,
           2.587460832994039e-16, 3.770007980996368e-17]
    w8t = [1.8487002537155558e-10, 0.3980576687063185, 0.14386069878049718,
           5.184138711693228e-12, 4.999354167195719e-12, 6.323445523114105e-12,
           0.04380133934789453, 2.4184856892670488e-11, 3.3515098217259626e-21,
           6.265295620015208e-20, 8.015783678663168e-12, 9.00515049797777e-12,
           7.518970989899964e-21, 1.1311920083030287e-11, 1.0564157539928715e-12,
           2.3211229416295473e-11, 0.41428029280501943, 7.69779712794825e-12,
           3.643600946295079e-20, 7.441031533732544e-11]
    w9t = [7.605890975510201e-12, 3.843169027383243e-12, 7.590357936204107e-12,
           8.23912335881814e-12, 0.05913523927009788, 6.592337018557639e-12,
           1.046239931976785e-11, 0.1488798770564668, 1.0993318962550176e-8,
           5.8729422801422615e-12, 5.628510963319352e-12, 9.806335079436665e-12,
           5.230965637881667e-12, 5.5302179670604594e-12, 6.361379736000372e-12,
           0.2250549957370855, 0.3632033702291312, 4.115168155615156e-12,
           0.2037265066205753, 6.445751589887625e-12]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-6)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-6)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isempty(w7)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isempty(w10)
    @test isempty(w11)
end
