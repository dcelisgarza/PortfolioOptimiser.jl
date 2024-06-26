using CSV, Clarabel, HiGHS, LinearAlgebra, OrderedCollections, PortfolioOptimiser,
      Statistics, Test, TimeSeries, Logging, Distances

struct POCorDist <: Distances.UnionMetric end
function Distances.pairwise(::POCorDist, mtx, i)
    return sqrt.(clamp!((1 .- mtx) / 2, 0, 1))
end

Logging.disable_logging(Logging.Warn)

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

    w2t = [0.125819809633272, 0.1281021818113089, 0.1078895915605406, 0.08527617021593129,
           0.10897517199613894, 0.01544720968623708, 0.05916939125220076,
           0.04888690198154068, 0.0071219088241854635, 0.006523196563266843,
           0.007668831461148527, 0.020971829668395857, 0.012238506609260505,
           0.010444064339267835, 0.008888240949513437, 0.03095016949765901,
           0.14288668056976955, 0.010123195982446138, 0.008311431640900419,
           0.054305515757016115]

    w3t = [0.012012455011706011, 0.027892011483293417, 0.004529467112380566,
           0.0011284331393593362, 0.00689511534507588, 0.013455970662701074,
           4.869975412523311e-12, 0.11189578186437756, 0.009123312795780224,
           0.0018929040295841359, 0.06210358298779455, 0.0025186869738562036,
           2.109246405189902e-12, 0.25476285063386456, 0.0019307745597114349,
           0.022148996370500943, 0.0600744157206004, 0.22008882925890277,
           0.04269768651214994, 0.14484872553138173]

    w4t = [0.0015589945972085593, 0.03579262122061028, 0.0024794094307871834,
           0.0017820297774897417, 0.019375644526443435, 4.747798762019785e-10,
           8.375027473106014e-12, 0.11408827351351102, 0.002281629284344294,
           0.003258337300334062, 0.05145977535445649, 0.0013072035010772553,
           7.02453455761538e-12, 0.2248752932962801, 8.021704227314336e-11,
           0.025421200121891047, 0.092084346730275, 0.21244404271532827,
           0.07441902594315603, 0.1373721721164108]

    w5t = [1.1907785898409796e-12, 2.3409319027431947e-12, 1.0417665009184893e-12,
           2.067352543856232e-12, 0.16806562099391517, 0.0, 2.692147508752728e-12,
           0.06273168367710923, 0.04828158031882784, 1.6004533637958202e-9,
           7.016327896214506e-12, 1.1416122408904733e-12, 1.0542613561219005e-12,
           2.991090845068521e-12, 0.0, 0.062338496336112235, 0.42205615150631676,
           1.0124432939297811e-11, 0.23652646532003407, 1.2395018310408315e-12]

    w6t = [6.738413360065276e-10, 1.1532620042102695e-9, 7.275021455310707e-10,
           1.6393448360573115e-9, 0.999999950601205, 3.1357509152461548e-18,
           3.273817354364893e-10, 1.4781201544507028e-16, 1.2211749626889438e-8,
           2.2377327419571015e-17, 8.059679478220913e-18, 2.6118780998576964e-17,
           1.3721470671965364e-17, 7.821778096813788e-18, 1.071504059509196e-9,
           1.4791276765504043e-8, 1.680293193933981e-8, 1.1387164479269213e-17,
           2.3098462931653324e-16, 4.22220643620723e-17]

    w8t = [0.10674711120788978, 0.2478587140361753, 0.04025051740769016,
           0.010027673585497318, 0.061272541192852345, 1.0634666203827151e-13,
           4.3276399906074224e-11, 3.06285073774096e-11, 5.473553984356847e-14,
           1.1356524351469761e-14, 3.7259197586966925e-13, 6.894238663405698e-13,
           5.773503523165445e-22, 1.5284559976920577e-12, 1.5259503362539836e-14,
           6.062701269278809e-12, 0.5338434424459243, 1.3204283523630718e-12,
           2.561658219583287e-13, 3.964850313949753e-11]

    w9t = [5.355016614543724e-21, 1.0527338448686186e-20, 4.6848985768534726e-21,
           9.297032475153734e-21, 7.558031362288874e-10, 0.0, 1.2106780186649627e-20,
           1.023991625751069e-9, 3.7675250559191143e-10, 1.2488713312847857e-17,
           5.4750053695372e-20, 1.863494339708477e-20, 1.7209083779393615e-20,
           2.3340184038945965e-20, 0.0, 1.0175734886801019e-9, 1.898016745403974e-9,
           7.900332699114801e-20, 1.8456715347689478e-9, 2.0232830057964365e-20]

    w10t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]

    w11t = [6.558740650729216e-15, 1.2893719589937412e-14, 5.737990552092887e-15,
            1.1386860063203745e-14, 0.0009256958680706928, 0.0, 1.4828195143963658e-14,
            0.003016410779882169, 0.0002680005115140795, 5.589053862832027e-13,
            1.0170350659438363e-14, 0.0, 0.0, 5.875165115635068e-15, 3.8872077175972995e-14,
            0.000569512687409757, 0.0023246612438207154, 0.0002724345221003853,
            0.002593735301414956, 4.6222974004725244e-14]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 0.0001)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-5)
    @test isempty(w5)
    @test isapprox(w6.weights, w6t)
    @test isempty(w7)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-6)
    @test isempty(w9)
    @test isempty(w10)
    @test isempty(w11)
    @test isapprox(w1.weights, w12.weights)
    @test isapprox(w2.weights, w13.weights)
    @test isapprox(w3.weights, w14.weights)
    @test isapprox(w4.weights, w15.weights)
    @test isempty(w16)
    @test isapprox(w6.weights, w17.weights)
    @test isempty(w18)
    @test isapprox(w8.weights, w19.weights)
    @test isempty(w20)
    @test isempty(w21)
    @test isempty(w22)
    @test isapprox(w1.weights, w23.weights)
    @test isapprox(w2.weights, w24.weights)
    @test isapprox(w3.weights, w25.weights)
    @test isapprox(w4.weights, w26.weights)
    @test isempty(w27)
    @test isapprox(w6.weights, w28.weights)
    @test isempty(w29)
    @test isapprox(w8.weights, w30.weights)
    @test isempty(w31)
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

    w2t = [0.1292169407470778, 0.14005362261968635, 0.1128985045906986, 0.10249932574849244,
           0.06907623901680716, 0.018565905449536926, 0.057755846783304525,
           0.029971305646125546, 0.0074373142506483905, 0.007294652640437887,
           0.006462284454566657, 0.015387254717294489, 0.0244244920261694,
           0.008311832076575834, 0.011104307045072408, 0.03907533063602432,
           0.15364076817509673, 0.007114981510198903, 0.008643356300103476,
           0.0510657355660822]

    w3t = [1.685416997560173e-19, 1.4233464553712867e-10, 1.9462220132946784e-18,
           2.821052002071675e-11, 2.5682645903498905e-19, 0.03773165252784658,
           8.416560541964733e-11, 0.11735576162995795, 9.255093313315979e-11,
           0.11299994579344845, 0.19435199321918525, 0.034994965558730544,
           1.8849639424595956e-12, 0.021448762872553158, 0.014973155344581206,
           0.1288608886146308, 2.1495041493106466e-10, 0.14514535942195003,
           2.831908545110211e-10, 0.1921375141698281]

    w4t = [2.24522727660641e-19, 1.8197420498758992e-10, 2.4570719853930517e-18,
           3.606702279660163e-11, 4.0402602726681526e-19, 0.006975243463610205,
           1.0760534854620397e-10, 0.12756389913774982, 5.017328436659337e-11,
           0.12203788234734425, 0.2098966197415324, 3.552233624962241e-9,
           2.4990441877601825e-11, 0.023164272753650857, 0.00276800502611664,
           0.1462097723215062, 2.7481314169267e-10, 0.15675436728475792,
           2.2988581816507994e-10, 0.20462993346598893]

    w5t = [2.534393234468539e-12, 4.855005681621374e-13, 1.299278354573542e-12,
           7.780717780037486e-12, 0.11689989340513916, 0.0, 1.022593541446513e-11,
           3.969461115009448e-10, 2.641246129135166e-18, 3.2745478903488603e-21,
           7.875092234589093e-22, 3.517807896506775e-21, 1.9119372109386033e-22,
           7.166900770941184e-22, 0.0, 4.330330562930836e-10, 0.8831001053645018,
           2.1427758915363218e-22, 2.1389975009443748e-11, 8.876093544367921e-21]

    w6t = [2.8979815801944255e-9, 4.945085628817623e-9, 3.122877810772984e-9,
           6.9439120306889625e-9, 0.9999997929279616, 6.580649560499988e-17,
           1.3845506621420994e-9, 2.55737133562738e-15, 5.143447099125221e-8,
           3.931361050019328e-16, 1.4278123478399107e-16, 4.544154331986894e-16,
           2.3119497315191096e-16, 1.3886499030856971e-16, 4.46465888064864e-9,
           6.179345100059182e-8, 7.008504085942441e-8, 2.0177551583840577e-16,
           4.029517077194116e-15, 7.499444163121253e-16]

    w7t = [0.025833509085507574, 0.04162347378527218, 0.02724155432891407,
           0.03444292077450162, 0.016556738340338367, 0.11477028140408123,
           0.038960075319182094, 0.050234055244923845, 0.03879373623687497,
           0.04218170034631946, 0.05106633363062071, 0.0415253505343533, 0.030148301176808,
           0.03345994013637636, 0.10472528790827357, 0.06150767145854596,
           0.04967541308301459, 0.04295911561366888, 0.04243802482313104,
           0.11185651676929217]

    w8t = [3.5885805163348607e-10, 0.30305813725231084, 4.143885108244222e-9,
           0.06006568264623444, 5.468334710676432e-10, 6.750218447954445e-11,
           0.17920493990015718, 9.309010323229759e-10, 1.7751654938169316e-19,
           2.167386084450437e-10, 3.7277522801510084e-10, 2.7759054274173113e-10,
           1.4952098265615022e-20, 4.113962166284878e-11, 2.6787077336855812e-11,
           1.0221631436866233e-9, 0.4576712303936362, 2.7839485233800067e-10,
           5.431718688013756e-19, 1.5240922797863273e-9]

    w9t = [9.511403058993654e-23, 1.8220501563676486e-23, 4.8761020776503483e-23,
           2.9200497337081075e-22, 4.387172395379417e-12, 0.0, 3.83772305179039e-22,
           2.0024252321376067e-11, 1.5189852230492706e-18, 1.8831981626929012e-21,
           4.5289791824163345e-22, 1.7745852874443178e-22, 9.644914517413401e-24,
           4.1216970452067143e-22, 0.0, 2.1844686045472732e-11, 3.3142138044425914e-11,
           1.2323141261415295e-22, 1.2301411671685788e-11, 4.477613751864084e-22]

    w10t = [4.19950323517131e-9, 6.766324785446171e-9, 4.428395505879914e-9,
            5.599051864866957e-9, 2.6914685106848975e-9, 1.5515779370498883e-11,
            6.333361906192175e-9, 4.8872086225577425e-12, 0.15461902131134497,
            0.1681223274028664, 0.20353354159324052, 4.039949595881603e-12,
            2.933090644352322e-12, 0.13336027149932125, 1.4157798010234434e-11,
            5.9840046924348265e-12, 8.075250530622686e-9, 0.17122084792319223,
            0.16914395211827798, 1.088238109742096e-11]

    w11t = [1.0164440246632186e-15, 1.947149103649204e-16, 5.210887173779382e-16,
            3.1205355142010583e-15, 4.688388388960399e-5, 0.0, 4.101214763069201e-15,
            0.00018084128302159894, 4.4529086809731815e-11, 5.520599752652079e-14,
            1.327671290761093e-14, 1.6026480043407179e-15, 8.710431171009255e-17,
            1.2082764384039807e-14, 0.0, 0.0001972818254717155, 0.00035417622374826036,
            3.612531699923921e-15, 0.0003606161665688415, 4.043783521933823e-15]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isempty(w5)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.001)
    @test isapprox(w8.weights, w8t)
    @test isempty(w9)
    @test isapprox(w10.weights, w10t, rtol = 0.001)
    @test isempty(w11)
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

    w2t = [0.12258167369043874, 0.1313921789402529, 0.10416947864836608,
           0.09608391068114182, 0.10221134198893286, 0.016621870431312597,
           0.0620800089254502, 0.04488416639179863, 0.007265132610419133,
           0.006825726160855613, 0.007622811698549928, 0.019643136960109057,
           0.014792376699060406, 0.0098667070696658, 0.010595250797454072,
           0.03324843253018149, 0.1348024968221915, 0.009811394146241121,
           0.008609223736900932, 0.05689268107067708]

    w3t = [1.7818256783215415e-10, 0.02052792221560721, 0.007995820407165864,
           4.689535733316197e-11, 9.65579016576632e-11, 0.04790943920760172,
           1.2350859030041307e-10, 0.06225960370158125, 1.134306861287994e-9,
           0.02640061276164985, 0.15475615311696356, 0.00924585308738149,
           1.2611732378148551e-11, 0.27221362653157954, 0.014669013818270935,
           0.02279954886748706, 0.023703031498917553, 0.21381121400849706,
           4.851684875405448e-9, 0.1237081543335489]

    w4t = [3.120184999612735e-10, 0.023896436103950282, 0.00930788874291692,
           1.1238428329475074e-10, 2.7657837831858305e-10, 0.028175177194387203,
           2.894311616353789e-10, 0.061992163346799606, 1.7610813297061538e-9,
           0.02711854618461621, 0.15896449489924028, 0.009206137539629403,
           3.9389443554156986e-11, 0.2796159806998015, 0.008626735881023603,
           0.022701614616183054, 0.027592564716333937, 0.2196254759409175,
           1.5052116251143666e-8, 0.12317676629120129]

    w5t = [4.61588143601273e-12, 5.419423361307183e-12, 5.084253266633871e-12,
           9.842774479860728e-13, 0.1601501814930634, 0.0, 8.020815728854044e-13,
           0.10756987961365808, 0.0007130068691811849, 4.724333675456978e-12,
           1.935599781796725e-12, 3.4512985417994885e-12, 8.618012146123133e-12,
           3.0012558825122503e-13, 0.0, 0.1649216182679484, 0.4985813769926359,
           9.039562654189643e-13, 0.06806393669426254, 2.3507573335830737e-12]

    w6t = [4.474662368568414e-8, 7.645922049783304e-8, 4.826341698047356e-8,
           1.080110221268728e-7, 0.9999981428125745, 9.328545521248128e-16,
           2.155111739663566e-8, 3.922694402838411e-14, 2.0052306102833784e-7,
           2.4021437840651604e-14, 8.68786387614006e-15, 6.956266629132925e-15,
           3.5892279037593e-15, 8.440657850004864e-15, 1.7498697962026265e-8,
           2.418182099392399e-7, 1.0983156935562923e-6, 1.2276166521840764e-14,
           2.4704686123178475e-13, 1.1368687444697751e-14]

    w7t = [0.03862629006792762, 0.04320027978553068, 0.032868488076019275,
           0.026519416537993043, 0.029338254425792867, 0.12690036687960832,
           0.025240340481867664, 0.06778997177475826, 0.039404989920625216,
           0.041862292331859965, 0.05267933798679749, 0.032922104226344506,
           0.02423030798259358, 0.06290311824333479, 0.07158533902492882,
           0.05676452266165747, 0.037293852173517855, 0.057637315843888065,
           0.0451564624165065, 0.08707694915844807]

    w8t = [3.4117092055407314e-9, 0.39305360813737744, 0.15309810842256522,
           8.97917929106188e-10, 1.8488199264445607e-9, 2.46928250072964e-11,
           2.3648519584037455e-9, 3.7370030430142124e-10, 2.6592088044947717e-19,
           6.189219539774232e-12, 3.628021119885351e-11, 5.549630750690024e-11,
           7.569929693212135e-20, 6.381631788370544e-11, 7.560501588728766e-12,
           1.368495435748918e-10, 0.4538482734195159, 5.012476625112779e-11,
           1.1374032528254833e-18, 7.425324314719382e-10]

    w9t = [2.037198652522541e-21, 2.3918382918086996e-21, 2.243912467738109e-21,
           4.344064548758534e-22, 7.068156720696355e-11, 0.0, 3.539951192738565e-22,
           8.38044443466854e-11, 2.166920626003276e-12, 1.4357864598451946e-20,
           5.882539526834399e-21, 2.688802456680521e-21, 6.714032979052755e-21,
           9.121207041382425e-22, 0.0, 1.2848545177647212e-10, 2.2004666355980244e-10,
           2.7472406805711812e-21, 2.0685515762162978e-10, 1.8314040402608372e-21]

    w10t = [0.06039667745642508, 0.06754864004918927, 0.0513936872869913,
            0.04146617871302297, 0.045873758172815776, 2.2559219942144644e-12,
            0.039466195181905, 5.73620410189285e-13, 0.08357768215808548,
            0.08878960177299135, 0.1117324728524905, 2.785779435542105e-13,
            2.0503031407314075e-13, 0.13341703256069473, 1.2725805664757775e-12,
            4.803260411668026e-13, 0.05831325651183364, 0.12224830595691288,
            0.0957765113208391, 7.368215974519576e-13]

    w11t = [2.9710922901173907e-15, 3.48830601237684e-15, 3.2725679571502745e-15,
            6.335472818326907e-16, 0.0001030834470276089, 0.0, 5.16273741056828e-16,
            0.00015361995472505717, 4.081234366520143e-6, 2.704197363108971e-14,
            1.107932712958236e-14, 4.928780506570128e-15, 1.2307335849610211e-14,
            1.7179117312707317e-15, 0.0, 0.0002355236579467976, 0.00032092056646466417,
            5.17422417051299e-15, 0.0003895963553286423, 3.3571036469652827e-15]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isempty(w5)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.01)
    @test isapprox(w8.weights, w8t)
    @test isempty(w9)
    @test isapprox(w10.weights, w10t, rtol = 0.01)
    @test isempty(w11)
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

    w2t = [0.11177665516399524, 0.15744672357044703, 0.09651091073876485,
           0.10183009688608481, 0.1131415284668777, 0.01475058218968319,
           0.060451185310978796, 0.04404890835934846, 0.006111757773800813,
           0.006939571819398308, 0.006561470070571817, 0.019204870497892646,
           0.01953377056583001, 0.007518486488694224, 0.010575038090719701,
           0.029918481185518066, 0.12861377490259365, 0.007766842634124238,
           0.007157208168590756, 0.05014213711608583]

    w3t = [1.423399913494141e-13, 0.2900378183318111, 3.579462775424237e-13,
           5.371987042516612e-13, 0.09669466251855562, 6.606350311022696e-13,
           1.0968683007440827e-12, 0.05437294388809498, 1.1897658134128507e-11,
           3.075751730971715e-10, 0.07247294612300595, 0.02464363904607377,
           7.2464317543522294e-12, 0.09440521053968351, 3.1071677661334214e-13,
           0.05988791766948051, 5.883900747388565e-11, 0.12785184672586725,
           8.0160486529557e-11, 0.179633014688603]

    w4t = [1.8896856725805472e-13, 0.317900114352252, 2.4563575934347664e-13,
           7.701118143421963e-13, 0.10598357327040948, 1.5439269015864503e-12,
           5.619307229354689e-13, 0.05416661682023877, 2.1147458403644085e-11,
           2.639955659786099e-10, 0.06363484182091061, 0.024550124250752235,
           1.8671588235713624e-11, 0.08289242779254061, 7.261558465755213e-13,
           0.059660661941105725, 5.13502322657248e-11, 0.11226027770333444,
           1.7339598623410748e-10, 0.17895136151585858]

    w5t = [2.162767809819556e-11, 2.1515090568665635e-10, 2.1974324876372395e-11,
           1.7438135373242284e-11, 0.38240257978004244, 0.0, 3.36887932474001e-11,
           0.09614812219364256, 6.928587681914277e-18, 3.1632066830115223e-22,
           1.3879975530126362e-21, 2.3936115491778075e-12, 7.630844937195219e-13,
           6.191928986372573e-22, 0.0, 0.10488908722308733, 0.41656021023917944,
           3.1053739596837623e-21, 1.9501634078652634e-10, 8.844750769566198e-13]

    w6t = [6.777583606412014e-10, 1.1595930278751537e-9, 7.316365101335398e-10,
           1.6469289844135605e-9, 0.9999999504649452, 3.1968578221019446e-18,
           3.289287422434876e-10, 1.4816330123649408e-16, 1.2244712122430014e-8,
           2.250013184854751e-17, 8.109325824395002e-18, 2.621404734127684e-17,
           1.3738311274129839e-17, 7.871038086835928e-18, 1.0739774878794903e-9,
           1.4821359307626416e-8, 1.685015975613105e-8, 1.1457081226302199e-17,
           2.319368510022738e-16, 4.24306683151349e-17]

    w7t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5384571312264972e-5,
           2.8968241081019833e-5, 3.3136657675552645e-5, 0.0, 0.0, 3.5542750929409494e-5,
           0.0, 0.0, 0.0, 3.4506447477018186e-5, 2.8263706479546166e-5, 0.0]

    w8t = [3.6805802039477026e-13, 0.7499701541555825, 9.255655917284398e-13,
           1.389070561063484e-12, 0.2500298456668865, 2.1643177171488077e-12,
           2.836245608689257e-12, 2.1285330397831636e-12, 1.7019229370128389e-22,
           4.39976704699722e-21, 1.036702920264857e-12, 9.64722455308209e-13,
           2.836754515521047e-22, 1.3504371312378655e-12, 1.01794454271911e-12,
           2.344427252378616e-12, 1.5214395060394659e-10, 1.8288808438530737e-12,
           1.1466707912494152e-21, 7.032078446726568e-12]

    w9t = [6.78549644762456e-21, 6.750173086595995e-20, 6.894253868149311e-21,
           5.471063749474055e-21, 1.1997549320268614e-10, 0.0, 1.0569566731440477e-20,
           9.931767328435817e-11, 2.2261295605105806e-18, 1.0163265915559635e-22,
           4.459584730006756e-22, 2.4725176569972243e-21, 7.88239798203844e-22,
           1.9894438500254288e-22, 0.0, 1.0834678679357446e-10, 1.306921535435428e-10,
           9.977451517481344e-22, 6.265802800485691e-11, 9.136320576746894e-22]

    w10t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13662135006690937,
            0.15590888484456766, 0.178343563602023, 0.0, 0.0, 0.19129330794416904, 0.0, 0.0,
            0.0, 0.1857158579646891, 0.15211703556580214, 0.0]

    w11t = [2.4870169549981457e-14, 2.4740702533877843e-13, 2.5268786734316743e-14,
            2.005251702928981e-14, 0.00043973361136133416, 0.0, 3.873952609212424e-14,
            0.000460271026757134, 3.158900918898347e-11, 1.4421779670499545e-15,
            6.3281969528729134e-15, 1.1458466585327136e-14, 3.6529645656455226e-15,
            2.8230414426106456e-15, 0.0, 0.0005021149323598663, 0.00047901226425635867,
            1.4158107113768645e-14, 0.0008891239115278803, 4.2340738698153836e-15]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-6)
    @test isapprox(w4.weights, w4t)
    @test isempty(w5)
    @test isapprox(w6.weights, w6t)
    @test isempty(w7)
    @test isapprox(w8.weights, w8t)
    @test isempty(w9)
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

    w2t = [0.12272565701099011, 0.1319605922941986, 0.10526354820554232,
           0.09820386818485113, 0.09724954191092244, 0.017141155457941353,
           0.06251026137935015, 0.04133365378803519, 0.007261631092835881,
           0.007225199742890979, 0.007330876905674741, 0.018794657524481898,
           0.016511352534700638, 0.009547210056392851, 0.010892984883311026,
           0.0344874888118273, 0.13843475956915705, 0.009164379906494588,
           0.008562843555748122, 0.055398337184653604]

    w3t = [1.1122131485189023e-11, 0.023947904594817403, 0.008654932589380422,
           3.118876209002888e-13, 3.007706321066676e-13, 0.010389329718699282,
           0.002635171680620229, 0.05307058024351695, 2.9552671999819593e-12,
           0.021417541684948155, 0.14074005221548436, 0.019760652884107485,
           1.6499421726538924e-11, 0.22899500156004937, 0.0030216579958167004,
           0.050934079070838405, 0.024923888440213194, 0.20430811856264766,
           0.04391715330559611, 0.16328393542207487]

    w4t = [7.593732814700385e-12, 0.02527222091778704, 0.008704119686236281,
           3.0336952801547483e-13, 2.1131574155202048e-13, 2.2157335406610334e-10,
           0.0022467385902653686, 0.05747591424459326, 3.609806215802843e-11,
           0.043290230492795435, 0.13185162914174783, 0.021400932375252642,
           4.929750500834112e-11, 0.212632122126413, 6.444293489853166e-11,
           0.055162067466843115, 0.027826701939502138, 0.17029092519455724,
           0.06700847324484002, 0.17683792419964636]

    w5t = [1.2331989214426656e-11, 6.231132981694356e-12, 1.2306804018264976e-11,
           1.3358708622483684e-11, 0.09588042510431309, 0.0, 1.696351580189378e-11,
           0.11958236180271992, 5.258941534075055e-9, 1.4425702613451978e-12,
           1.0444838792801463e-12, 7.87656533628801e-12, 4.2015760413110865e-12,
           1.3780269673112635e-12, 0.0, 0.18076726323151485, 0.5888890273669505,
           1.5835940610742327e-12, 0.014880917126768811, 5.177306070082038e-12]

    w6t = [6.811514915839932e-10, 1.1651950108186476e-9, 7.352030489841562e-10,
           1.6535281899643483e-9, 0.9999999502766338, 3.2735065181216375e-18,
           3.3016467054880843e-10, 1.4904946458659182e-16, 1.2296949812172037e-8,
           2.2667015594749685e-17, 8.17348653114203e-18, 2.637727120960746e-17,
           1.3785904919446847e-17, 7.934534285583589e-18, 1.0774940342968887e-9,
           1.4875903084372164e-8, 1.6907776417216398e-8, 1.1548118336675135e-17,
           2.336695627303026e-16, 4.276424504389479e-17]

    w7t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0]

    w8t = [1.8487002538756168e-10, 0.3980576687407824, 0.14386069879295268,
           5.184138712142072e-12, 4.999354167628564e-12, 7.790755991487845e-12,
           0.043801339351686855, 1.0206158655128095e-11, 6.182933620311741e-23,
           4.4809226911564045e-13, 2.9445269807510484e-12, 3.800229007082556e-12,
           3.173052095647566e-21, 4.790974210513392e-12, 2.265882474859349e-12,
           9.795281859827027e-12, 0.4142802928408879, 4.2744816278248645e-12,
           9.188233256309708e-13, 3.1401611648196825e-11]

    w9t = [1.7149792859546157e-21, 8.66548275855857e-22, 1.71147684291976e-21,
           1.857762618528853e-21, 1.3333853940616244e-11, 0.0, 2.359074250825494e-21,
           2.9975251915976844e-11, 1.8532720963928818e-17, 5.083675479399401e-21,
           3.6808030970866095e-21, 1.974388418397284e-21, 1.0531929490588798e-21,
           4.856222321635003e-21, 0.0, 4.53122364522751e-11, 8.189534276261004e-11,
           5.5806490077637644e-21, 5.24409490029177e-11, 1.2977754524772674e-21]

    w10t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0]

    w11t = [8.763691138750252e-15, 4.428135959782229e-15, 8.745793354654594e-15,
            9.493326205884927e-15, 6.813713645510531e-5, 0.0, 1.2055071613360806e-14,
            0.0001949998668016614, 1.6502895120408468e-10, 4.5268778084993766e-14,
            3.2776572629741715e-14, 1.2844111525114656e-14, 6.851401461399121e-15,
            4.324336820875516e-14, 0.0, 0.00029477250424606103, 0.0004184922206066768,
            4.969419517953668e-14, 0.0004669727036274196, 8.442499201704742e-15]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-7)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-6)
    @test isempty(w5)
    @test isapprox(w6.weights, w6t)
    @test isempty(w7)
    @test isapprox(w8.weights, w8t)
    @test isempty(w9)
    @test isempty(w10)
    @test isempty(w11)
end
