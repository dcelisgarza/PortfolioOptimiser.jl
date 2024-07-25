using CSV, Clarabel, DataFrames, HiGHS, LinearAlgebra, OrderedCollections,
      PortfolioOptimiser, Statistics, Test, TimeSeries, Clustering, Distances

struct POCorDist <: Distances.UnionMetric end
function Distances.pairwise(::POCorDist, mtx, i)
    return sqrt.(clamp!((1 .- mtx) / 2, 0, 1))
end
dbht_d(corr, dist) = 2 .- (dist .^ 2) / 2

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "Cluster assets" begin
    hcportfolio = HCPortfolio(; prices = prices)
    portfolio = Portfolio(; prices = prices)
    asset_statistics!(hcportfolio; calc_kurt = false)
    asset_statistics!(portfolio; calc_kurt = false)

    cluster_assets!(hcportfolio)
    clustering_idx, clustering, k = cluster_assets(portfolio)

    @test isapprox(hcportfolio.clusters.heights, clustering.heights)
    @test isapprox(hcportfolio.clusters.merges, clustering.merges)
    @test hcportfolio.clusters.linkage == clustering.linkage
    @test k == hcportfolio.k
    @test clustering_idx == cutree(hcportfolio.clusters; k = k)
end

@testset "Weight bounds" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false,
                      cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    asset_sets = DataFrame("Asset" => portfolio.assets,
                           "PDBHT" => [1, 2, 1, 1, 1, 3, 2, 2, 3, 3, 3, 4, 4, 3, 3, 4, 2, 2,
                                       3, 1],
                           "SPDBHT" => [1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 3, 2, 3, 3, 3, 3, 1,
                                        4, 2, 1],
                           "Pward" => [1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 2, 4, 4, 2, 3, 4, 1, 2,
                                       2, 1],
                           "SPward" => [1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 2, 4, 3, 2, 2, 3, 1,
                                        2, 2, 1],
                           "G2DBHT" => [1, 2, 1, 1, 1, 3, 2, 3, 4, 3, 4, 3, 3, 4, 4, 3, 2,
                                        3, 4, 1],
                           "G2ward" => [1, 1, 1, 1, 1, 2, 3, 4, 2, 2, 4, 2, 3, 3, 3, 2, 1,
                                        4, 2, 2])

    constraints = DataFrame("Enabled" => [true, true, true, true, true, true, false],
                            "Type" => ["Asset", "Asset", "All Assets", "All Assets",
                                       "Each Asset in Subset", "Each Asset in Subset",
                                       "Asset"],
                            "Set" => ["", "", "", "", "PDBHT", "Pward", ""],
                            "Position" => ["WMT", "T", "", "", 3, 2, "AAPL"],
                            "Sign" => [">=", "<=", ">=", "<=", ">=", "<=", ">="],
                            "Weight" => [0.05, 0.04, 0.02, 0.07, 0.04, 0.08, 0.2])

    w_min, w_max = hrp_constraints(constraints, asset_sets)
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    optimise!(portfolio; type = :HRP, rm = :CDaR, cluster_opt = cluster_opt,
              save_opt_params = false)
    optimise!(portfolio; type = :HERC, rm = :CDaR, cluster = false, save_opt_params = false)
    optimise!(portfolio; type = :NCO, nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Min_Risk),
              cluster = false, save_opt_params = false)

    portfolio.w_min = w_min
    portfolio.w_max = w_max
    w1 = optimise!(portfolio; type = :HRP, rm = :CDaR, cluster_opt = cluster_opt)
    w2 = optimise!(portfolio; type = :HERC, rm = :CDaR, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Min_Risk), cluster = false)

    portfolio.w_min = 0.03
    portfolio.w_max = 0.07
    w4 = optimise!(portfolio; type = :HRP, rm = :CDaR, cluster = false)
    w5 = optimise!(portfolio; type = :HERC, rm = :CDaR, cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Min_Risk), cluster = false)

    portfolio.w_min = 0
    portfolio.w_max = 1
    w7 = optimise!(portfolio; type = :HRP, rm = :CDaR, cluster_opt = cluster_opt)
    w8 = optimise!(portfolio; type = :HERC, rm = :CDaR, cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Min_Risk), cluster = false)

    portfolio.w_min = Float64[]
    portfolio.w_max = Float64[]
    w10 = optimise!(portfolio; type = :HRP, rm = :CDaR, cluster_opt = cluster_opt)
    w11 = optimise!(portfolio; type = :HERC, rm = :CDaR, cluster = false)
    w12 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Min_Risk), cluster = false)

    N = length(w_min)

    @test all(abs.(w1.weights .- w_min) .>= -eps() * N)
    @test all(w1.weights .<= w_max)
    @test all(abs.(w2.weights .- w_min) .>= -eps() * N)
    @test all(w2.weights .<= w_max)
    @test all(w3.weights .>= w_min)
    @test !all(w3.weights .<= w_max)

    @test all(w4.weights .>= 0.03)
    @test all(w4.weights .<= 0.07)
    @test all(abs.(w5.weights .- 0.03) .>= -eps() * N)
    @test all(w5.weights .<= 0.07)
    @test all(w6.weights .>= 0.03)
    @test !all(w6.weights .<= 0.07)

    @test isapprox(w7.weights, w10.weights)
    @test isapprox(w8.weights, w11.weights)
    @test isapprox(w9.weights, w12.weights)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:Skew)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    cluster_opt = ClusterOpt(; linkage = :DBHT, genfunc = GenericFunction(; func = dbht_d))
    rm = :Skew
    w1 = optimise!(portfolio; type = :HRP, rm = rm, rf = rf, cluster_opt = cluster_opt)
    w2 = optimise!(portfolio; type = :HERC, rm = rm, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [4.18164303203965e-6, 0.11905148244884775, 0.13754035964680367,
           4.18164303203965e-6, 0.13754035964680367, 1.658199358060021e-6,
           0.028721218707620826, 1.1383557170375464e-10, 8.355416012626526e-16,
           1.8128635217880256e-6, 0.08532736996421339, 0.007291462719384035,
           0.10926054871992093, 0.00021830908182616843, 0.2370505968817962,
           0.10926054871992093, 0.028721218707620826, 3.684313367457248e-6,
           4.824826616435831e-11, 1.0059308456231142e-6]
    w2t = [0.05013622008238557, 0.05843253495835018, 0.0326340699568634,
           0.03810785845352964, 0.04151978371454936, 0.020776349162705425,
           0.014419467673840766, 0.06450029254562262, 0.02337036891691671,
           0.0768704564650501, 0.03525278374131375, 0.007885328106569226,
           0.23227643708915915, 0.029736303978302396, 0.019940256048870408,
           0.05591386120058815, 0.04641584792557699, 0.06294077176658942,
           0.036661874867119304, 0.0522091333460975]
    w3t = [9.512843515038461e-7, 0.02679372963723403, 7.481688741710024e-8,
           8.677981868535554e-8, 0.00024374710251416658, 1.6424163064416956e-10,
           9.925303140364328e-8, 0.018337706834624035, 4.199681081018799e-10,
           0.044847315863413596, 2.880663555289258e-5, 1.6699881321927315e-7,
           0.8850048893895833, 9.222351514908303e-10, 1.29959155424575e-5,
           1.2870890276640751e-6, 1.9217385711076424e-7, 0.02235979951041215,
           2.3395793459459278e-5, 0.002344753415432111]
    w4t = [2.892066058287496e-12, 6.021136122811683e-12, 1.6595810841870378e-11,
           1.2518518518980756e-11, 0.7759709237514477, 3.261400626300735e-22,
           0.22402907609740005, 1.1209584266409124e-11, 5.2093420567391296e-20,
           1.211847041348591e-21, 5.555320744491218e-22, 2.463831250808569e-22,
           1.936357517318602e-22, 5.1125064758334945e-23, 3.530451287716877e-22,
           5.1336508446549383e-11, 1.9893431894515837e-11, 9.98151946790108e-12,
           2.0436673289400196e-11, 2.671258551762943e-13]
    w5t = [1.0310068513022651e-9, 0.04247516664987991, 3.385746665892482e-10,
           4.4084678285666815e-10, 0.3154602214189142, 2.1215416105607568e-19,
           2.7889852555415977e-9, 3.2946681133034345e-9, 2.40803099868887e-18,
           0.14355268678846272, 1.6790419477890336e-18, 2.860421097787122e-19,
           2.7315391405613452e-11, 5.216927699995645e-19, 1.8561334858300276e-19,
           0.49370302263693644, 1.1329559074445164e-9, 0.004808890578177884,
           1.5115378626916289e-9, 1.3617380041860026e-9]
    w6t = [6.1346171755586554e-9, 6.599014059111091e-9, 8.090716526383227e-9,
           7.396914264079482e-9, 4.896724588071184e-7, 7.36517620191656e-18,
           0.9999994421566508, 4.737854348345277e-9, 8.928320134773704e-16,
           6.371336604145683e-17, 1.810543949163722e-17, 7.707398990930562e-18,
           1.3862608503512784e-17, 1.1718109870037684e-17, 5.412442015541036e-18,
           1.015661101908201e-8, 8.50118026514471e-9, 4.925362320042241e-9,
           5.8237163204458354e-9, 5.804903047063175e-9]
    w7t = [0.02587283980010857, 0.03264463657770643, 0.014158174674160097,
           0.015798390683037792, 0.021562504541245746, 0.019702542832291847,
           0.011175680915325443, 0.04053333087337281, 0.02613557862432773,
           0.13685040056428732, 0.05399833667801738, 0.01325291505118253,
           0.3224398305305101, 0.03177231426544304, 0.03083051325239583,
           0.07583058468110429, 0.02019743390256818, 0.039927409011560895,
           0.03821626595613802, 0.029100316585215984]
    w8t = [1.1496042250645375e-5, 0.32379576882040856, 9.041440631805508e-7,
           1.0487132060283437e-6, 0.00294562502215608, 3.8464669151875353e-7,
           1.1994489772875258e-6, 0.22160677006567375, 9.83546879626118e-7,
           6.146120843091661e-12, 0.06746387633725053, 0.00039110389211562547,
           1.212858984355129e-10, 2.1598342541531334e-6, 0.03043586389795064,
           1.7638970242797108e-16, 2.322374773980397e-6, 0.2702127912451197,
           0.05479192159954706, 0.02833578024124957]
    w9t = [4.3494608539055324e-10, 0.017918801836614002, 1.428329265313794e-10,
           1.8597799056168681e-10, 0.133081742598787, 1.6760464463836693e-11,
           1.1765762929485634e-9, 1.3899063781510106e-9, 1.9023769215949662e-10,
           0.1638946377213469, 1.3264657529753092e-10, 2.2597711929108937e-11,
           3.118608420918537e-11, 4.121443147328655e-11, 1.466370453189873e-11,
           0.563662581643284, 4.77954861542065e-10, 0.0020287043964917146,
           0.11941352697150502, 5.744701051211583e-10]
    w10t = [0.10309106955873795, 0.13007348733856158, 0.056413651591213254,
            0.06294913915148705, 0.08591641554217175, 2.5429598049594143e-10,
            0.04452981997759831, 0.16150621519701452, 3.373256258684194e-10,
            2.673662282768832e-10, 6.969435411240823e-10, 1.710521863120272e-10,
            6.299544683808346e-10, 4.1007761675888156e-10, 3.979220176525376e-10,
            1.4815110025705342e-10, 0.08047725257236474, 0.15909190222303052,
            4.93248151009315e-10, 0.11595104304148351]
    w11t = [7.083483903050993e-10, 0.029182362742187502, 2.3261612643373977e-10,
            3.028817011383293e-10, 0.21673545599152041, 3.8084417732649646e-11,
            1.9161591543763263e-9, 2.263586174718662e-9, 4.3227273040850915e-10,
            0.10800148208705941, 3.0140976076984657e-10, 5.1348260829357545e-11,
            2.0550662071143323e-11, 9.365060426750246e-11, 3.331999838703398e-11,
            0.3714361559405905, 7.783920080757015e-10, 0.003303925571302533,
            0.27134060955914413, 9.355756676728832e-10]

    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-6)
    @test isapprox(w8.weights, w8t, rtol = 5.0e-8)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 1.0e-6)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-7)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:SSkew)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    cluster_opt = ClusterOpt(; linkage = :DBHT, genfunc = GenericFunction(; func = dbht_d))
    rm = :SSkew
    w1 = optimise!(portfolio; type = :HRP, rm = rm, rf = rf, cluster_opt = cluster_opt)
    w2 = optimise!(portfolio; type = :HERC, rm = rm, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.03073561997154973, 0.06271788007944147, 0.059839304178611136,
           0.019020703463231065, 0.054114343500047124, 0.07478195087196789,
           0.015025906923294915, 0.037423537539572865, 0.02195663364580543,
           0.07120487867873436, 0.14238956696393482, 0.022992175030255378,
           0.02427916555268004, 0.040541588877064376, 0.010704183865328891,
           0.053000134123666075, 0.07802885174200914, 0.06680453879901624,
           0.030056876315924044, 0.08438215987786517]
    w2t = [0.06664919970362215, 0.065688537645969, 0.05827905623198537, 0.04923706156165528,
           0.055949261956495855, 0.030046317963353407, 0.027319842253653435,
           0.08126769722698507, 0.02377683386943261, 0.08094977263933717,
           0.04476608242725907, 0.012406273321936936, 0.028493003361406897,
           0.03436511159723589, 0.014308519726637069, 0.05423891622864567,
           0.07452547794485952, 0.09337434382384531, 0.028961215530580075,
           0.07539747498510434]
    w3t = [1.0892430594871065e-6, 0.031681259608003894, 6.385841396167424e-7,
           1.0208999440372263e-6, 3.6209703749382284e-6, 9.63334977859699e-6,
           5.195032617345709e-6, 0.16790560775238103, 1.6957722439003348e-6,
           4.9105984264575884e-6, 0.32690571795454637, 0.0064336998061633,
           1.1748062417479142e-8, 0.0780837391349781, 8.910688360428937e-7,
           1.284611349160674e-6, 2.3889090964349337e-6, 0.2998587061455184,
           9.651938161685354e-6, 0.08908923687231873]
    w4t = [8.15581000775344e-12, 2.1122924956231996e-11, 8.468440027459267e-11,
           6.290782166354827e-11, 0.7906420617811222, 3.770466458146047e-21,
           0.20935793754156667, 7.732354627171445e-11, 5.613257710234412e-19,
           2.4647140872454105e-19, 7.937696118180958e-21, 3.0134893967449533e-21,
           3.3302575799562404e-20, 7.080800527624176e-22, 4.270636058014689e-21,
           2.0043380508726013e-10, 9.698265399054417e-11, 5.504855806811231e-11,
           6.228588325811293e-11, 8.36586909877896e-12]
    w5t = [3.7861438714093765e-10, 7.882775220627101e-10, 4.736696170348332e-10,
           3.718007853005574e-10, 0.7671857043598811, 4.915030082168658e-19,
           0.07214396701662078, 8.737251526759338e-10, 6.322008478869382e-18,
           0.02001621273852103, 8.063255787638913e-18, 5.530049009448801e-19,
           3.6820467470084965e-11, 1.1316668921573435e-18, 2.8096469370461567e-19,
           0.14065410623907945, 3.1380239299922003e-9, 9.710437538871152e-10,
           2.04328809940668e-9, 5.706340470933306e-10]
    w6t = [4.808971411822285e-9, 5.0889067778196335e-9, 6.296951167461911e-9,
           5.8264815984649646e-9, 3.6863311384148367e-7, 2.074772259621311e-16,
           0.9999994962043447, 3.680607079860537e-9, 2.4513295260098905e-14,
           2.7849215145189317e-15, 4.759207596372842e-16, 2.1346320729387184e-16,
           4.421019750282232e-16, 3.178085787059933e-16, 1.3941144415954463e-16,
           5.739878651350115e-8, 6.535615364916281e-9, 3.8593171144659175e-9,
           3.7086577022239837e-8, 4.580298321968766e-9]
    w7t = [0.0339264619542522, 0.038638756632197464, 0.03015595919880873,
           0.029485590325139945, 0.03172755919967277, 0.05301160166381721,
           0.02380386625242652, 0.06207621701820662, 0.040253889426890024,
           0.12465166616658238, 0.09072083579161366, 0.031034596824263225,
           0.04653553578551491, 0.05952392664850132, 0.03166034045137151,
           0.08586716691912194, 0.0376304878982509, 0.058963225878939075,
           0.04715290372556418, 0.043179412238865456]
    w8t = [1.2487501420929947e-6, 0.03632061466226695, 7.32097421359956e-7,
           1.1703989656629132e-6, 4.151219721654936e-6, 6.0796903275632165e-15,
           5.95578522404621e-6, 0.1924934473649298, 1.0702165234254124e-15,
           0.25733154982206186, 2.0631302476829858e-10, 4.0603635683280564e-12,
           0.0006156372089006405, 4.927932281807327e-11, 5.623612459000729e-16,
           0.06731787059138405, 2.738737279652763e-6, 0.34376955505537465,
           6.091421616830679e-15, 0.10213532804666114]
    w9t = [1.2679194959464424e-10, 2.639816320734828e-10, 1.5862443754742695e-10,
           1.2451018247103547e-10, 0.25691831705465595, 1.3363625907243063e-10,
           0.024159869619862124, 2.9259668750100066e-10, 1.7189102585665538e-9,
           0.020352032018695936, 2.192343325249284e-9, 1.5035819715346746e-10,
           3.743821784289705e-11, 3.076923792037163e-10, 7.639235156328738e-11,
           0.1430139113294775, 1.050874413312378e-9, 3.2518714258794954e-10,
           0.5555558628268749, 1.9109628633608354e-10]
    w10t = [0.08708302691554774, 0.09917862606251587, 0.0774048354972918,
            0.07568412112549072, 0.08143884544976679, 8.217295414380494e-12,
            0.061100173909951344, 0.159338303083335, 6.2397303725371815e-12,
            2.4264926884526464e-10, 1.4062580351123707e-11, 4.810653558223435e-12,
            9.058694585427884e-11, 9.226766861271822e-12, 4.907649688810532e-12,
            1.6715063594011555e-10, 0.09659058450914755, 0.1513478238068857,
            7.309142289569202e-12, 0.11083365908490689]
    w11t = [1.5279651278133656e-10, 3.1812329527313936e-10, 1.911577271004558e-10,
            1.500467636010365e-10, 0.3096113202857433, 9.031421552219608e-11,
            0.029114970145727778, 3.526071934728881e-10, 1.1616759750162758e-9,
            0.03560694661608765, 1.4816320731329082e-9, 1.016152556013039e-10,
            6.550012416003028e-11, 2.0794502961116357e-10, 5.162756987678539e-11,
            0.25021033287431355, 1.2664048958834443e-9, 3.9188183120156693e-10,
            0.3754564238645099, 2.302894328146739e-10]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 5.0e-8)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-5)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 5.0e-6)
    @test isapprox(w11.weights, w11t, rtol = 5.0e-5)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:DVar)" begin
    portfolio = HCPortfolio(; prices = prices[(end - 25):end],
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    cluster_opt = ClusterOpt(; linkage = :ward, genfunc = GenericFunction(; func = dbht_d))
    rm = :DVar
    w1 = optimise!(portfolio; type = :HRP, rm = rm, rf = rf, cluster_opt = cluster_opt)
    w2 = optimise!(portfolio; type = :HERC, rm = rm, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.03523494732479926, 0.029482678539359645, 0.030682771818436906,
           0.0214446608140303, 0.034133270406532384, 0.034184314896968354,
           0.030296752902823278, 0.05992194286184598, 0.039135371124385865,
           0.05016814136419783, 0.1754879152987106, 0.037699252297632733,
           0.005711291060759251, 0.0706937677899005, 0.01957759763281498,
           0.028586488944922182, 0.026565477167910148, 0.1265533192960246,
           0.04837779487238144, 0.09606224358556376]
    w2t = [0.010552391637986672, 0.016197954498471993, 0.01727913079283032,
           0.006237283272528802, 0.008218082573208793, 0.06024410237795489,
           0.017061742632110556, 0.12850992156474847, 0.011720515975110153,
           0.07300075959954444, 0.23445328475189728, 0.04796185219797715,
           0.010065189383251117, 0.051116602415659206, 0.009609525255688317,
           0.061307215298297926, 0.014595227154193004, 0.18415050243512646,
           0.014590330284512431, 0.023128385898902214]
    w3t = [1.626859598633623e-14, 4.878072199791674e-13, 1.4527911211373106e-8,
           7.815310851572044e-15, 6.5792914715053046e-15, 4.1371829165795685e-7,
           0.011908143119657216, 0.08619438012611065, 1.482991228908767e-14,
           4.635412464266885e-8, 0.5915225354374783, 3.8938657061448187e-7,
           8.404550940317591e-9, 0.12903667758300463, 7.182096717164822e-10,
           1.451552284333764e-7, 6.229477188214056e-8, 0.1813368584615028,
           7.06284550571257e-14, 3.2471198334841704e-7]
    w4t = [8.433592303203916e-11, 1.4810687244348916e-10, 2.5415651255720205e-17,
           1.0575471169642167e-10, 9.826926481083281e-11, 4.680727903817386e-10,
           1.979927381202606e-17, 2.0476454718074495e-9, 8.924219296547511e-11,
           0.38935410621571376, 7.669389718042708e-9, 7.193264475607865e-10,
           0.2556322857685956, 2.0549766418409782e-8, 5.593570762265946e-17,
           5.152412035884537e-10, 1.5881608801684245e-10, 0.022961265815540463,
           9.697611422680339e-11, 0.3320523094492067]
    w5t = [6.99927459172966e-13, 6.351719788969979e-13, 2.1243277127479536e-22,
           6.139347729676585e-13, 6.850737459567674e-13, 1.0400658443295194e-11,
           2.174169572685971e-22, 2.2775316023728395e-11, 7.122348068461382e-13,
           2.0708953804686408e-10, 2.3833834551615515e-11, 1.7237103485625813e-11,
           0.7895758048443616, 1.313760737481902e-10, 1.5378284119183503e-22,
           1.3897709953031573e-11, 6.997871517832699e-13, 3.449570741368316e-11,
           6.80238557191563e-13, 0.21042419468980597]
    w6t = [1.8453590228798314e-14, 3.6180280497773147e-14, 7.8985614511812e-14,
           1.9647369528216504e-14, 1.7956248044052224e-14, 3.7312789562441875e-8,
           3.561384400477929e-14, 4.600416928385553e-8, 1.834673946393253e-14,
           6.0139985334607e-8, 4.555393762100468e-8, 4.373109505228164e-8,
           0.9999990477978071, 2.930842604452552e-7, 6.945292174907096e-13,
           4.136941245290735e-8, 3.309976190391047e-14, 4.9824511263715374e-8,
           2.9235556315226516e-14, 3.3518104994705405e-7]
    w7t = [0.03511491609962473, 0.04430400996713104, 0.06148294701042703,
           0.02965050300792325, 0.0321374957270678, 0.05293260648233608,
           0.06621046805381717, 0.05839707207116222, 0.03536784457798467,
           0.0428379009757812, 0.07938826309297223, 0.04016674098674302,
           0.01906428353023485, 0.11337006961225103, 0.044400169311277646,
           0.04225393795099904, 0.04025217111698174, 0.06571567756307461,
           0.04012684092246267, 0.05682608193974799]
    w8t = [4.203691635544314e-8, 1.260459803726649e-6, 8.576212860322642e-19,
           2.019421767159565e-8, 1.7000429877171876e-8, 4.089511162542749e-12,
           7.029693992445201e-13, 6.723002233827568e-13, 3.8319458112951176e-8,
           2.0927298267345034e-19, 4.612722138123815e-12, 1.807357924424584e-12,
           3.3392805790933e-19, 7.617378697044473e-12, 4.239782948398699e-20,
           4.530214661499659e-12, 0.1609653418887744, 1.4138078575928013e-12,
           1.8249899745769043e-7, 0.8390330975759561]
    w9t = [1.9488236746953094e-12, 1.7685235430542592e-12, 6.696104252694015e-13,
           1.709392315729102e-12, 1.9074661488638952e-12, 2.0340751546127058e-19,
           6.853211034425641e-13, 4.454208809460826e-19, 1.983091298009481e-12,
           4.050086698048424e-18, 4.661225148860031e-19, 3.3710907947565616e-19,
           1.5441873570539128e-8, 0.41411119426086135, 4.847396805665039e-13,
           2.7179996993073934e-19, 1.9484330136929573e-12, 6.746386469040856e-19,
           1.8940034246718974e-12, 0.5858887902822657]
    w10t = [2.1796465196139846e-13, 2.750006813272597e-13, 6.303009282096119e-13,
            1.840482956828167e-13, 1.9948420580386322e-13, 0.7078545168226607,
            6.787850376519974e-13, 6.256977536221414e-13, 2.1953753001098674e-13,
            4.4484162096092496e-13, 8.606893589294194e-13, 1.732558513301338e-12,
            0.29214548316614797, 1.1622407641499858e-12, 4.551872585550205e-13,
            1.958390866411863e-12, 2.4985537270471977e-13, 6.949134769681001e-13,
            2.490738502334285e-13, 3.527326086063564e-13]
    w11t = [1.4146396120098623e-12, 1.28376080979606e-12, 7.128035119111434e-13,
            1.240837800615147e-12, 1.384618427920446e-12, 1.7635745687033476e-12,
            7.295276042396501e-13, 3.861867818524732e-12, 1.4395142776756712e-12,
            3.5114877075827084e-11, 4.041354181475767e-12, 2.9227877745523158e-12,
            0.13388342835012107, 0.4408233540750224, 5.160077167727437e-13,
            2.3565477099367757e-12, 1.4143560334922307e-12, 5.849221244585821e-12,
            1.3748459158277137e-12, 0.4252932175074354]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 5.0e-5)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isempty(w10)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-5)
end

@testset "Shorting with NCO" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; calc_kurt = false,
                      cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    cluster_opt = ClusterOpt(; linkage = :ward)
    portfolio.w_min = -0.2
    portfolio.w_max = 0.8
    w1 = optimise!(portfolio; type = :NCO, cluster_opt = cluster_opt,
                   portfolio_kwargs = (; short = true, solvers = portfolio.solvers))
    w2 = optimise!(portfolio; type = :NCO, cluster_opt = cluster_opt,
                   portfolio_kwargs = (; short = true, short_u = 0.3, long_u = 0.6,
                                       solvers = portfolio.solvers))
    w3 = optimise!(portfolio; type = :NCO, cluster_opt = cluster_opt,
                   portfolio_kwargs_o = (; short = true, short_u = 0.6, long_u = 0.4,
                                         solvers = portfolio.solvers))
    w4 = optimise!(portfolio; type = :NCO, cluster_opt = cluster_opt,
                   portfolio_kwargs = (; short = true, short_u = 0.3, long_u = 0.6,
                                       solvers = portfolio.solvers),
                   portfolio_kwargs_o = (; short = true, solvers = portfolio.solvers))
    w5 = optimise!(portfolio; type = :NCO, cluster_opt = cluster_opt,
                   portfolio_kwargs = (; short = true, solvers = portfolio.solvers),
                   portfolio_kwargs_o = (; short = true, short_u = 0.6, long_u = 0.4,
                                         solvers = portfolio.solvers))

    wt1 = [-0.087301210473091, 0.011322088745794164, 0.013258683926176839,
           0.004456088735487908, 0.22464713851216647, -0.04347232765939294,
           0.04304321233588089, 0.02538694929635001, 3.6391344665779835e-10,
           4.348234024497777e-8, 0.03768812903884124, -1.1037524427811691e-10,
           -0.013820021862556863, -4.136140736693239e-11, -0.01540643428627949,
           0.10308907870563483, 0.18282205944978064, 0.03429683638117709,
           0.11998968304453421, 2.4149790185098565e-9]

    wt2 = [-0.016918145881010444, 0.001712686850666274, 0.003151233891742689,
           0.000877434644273003, 0.02890364374221545, -0.031938949651537776,
           0.007380063055004761, 0.006474271417208838, -1.2243771111394412e-12,
           0.0035435277501784123, 0.0162240125533059, -0.0014765958830886081,
           -0.004205149992108177, -0.005246447380365774, -0.006973410234892313,
           0.01793024969296401, 0.023139966869064287, 0.012461571215558315,
           0.03566741209372675, -0.0007073747516812343]

    wt3 = [1.0339757656912846e-25, 3.308722450212111e-24, 8.271806125530277e-25,
           1.6543612251060553e-24, 5.551115123125783e-17, -7.754818242684634e-26,
           4.163336342344337e-17, -2.7755575615628914e-17, -8.271806125530277e-25,
           -1.2407709188295415e-24, -8.673617379884035e-19, 1.925929944387236e-34,
           3.611118645726067e-35, -2.0679515313825692e-25, 1.2924697071141057e-26,
           1.0339757656912846e-25, 2.7755575615628914e-17, -1.3877787807814457e-17, -0.2,
           4.1359030627651384e-25]

    wt4 = [-0.045115057672989595, 0.004567165136595039, 0.008403290748699927,
           0.0023398258212850554, 0.07707638671287502, -0.08517053748341849,
           0.019680161942876372, 0.017264724808840448, -3.2650058244204215e-12,
           0.009449407897343706, 0.043264036055620944, -0.003937587780318715,
           -0.01121372977736988, -0.013990527181995529, -0.018595754764415592,
           0.0478139841087731, 0.06170658103975119, 0.03323085855661183,
           0.09511310458886758, -0.0018863327543673799]

    wt5 = [-0.004706878592862724, 0.0006104348021668806, 0.0007148470817697388,
           0.00024025175095859297, 0.012111937526189324, 0.028034645035369427,
           0.02059326767499224, -0.016371658716560954, -2.3468226455651654e-10,
           -2.8041102000015e-8, -0.024304503037642658, -5.280709377299777e-11,
           -0.006611946321967141, 2.6673344541851835e-11, -0.007370937436053205,
           0.04932115604168774, 0.009856922180849811, -0.022117509817183564, -0.2,
           1.302045296146306e-10]

    @test isapprox(w1.weights, wt1)
    @test all(w1.weights[w1.weights .>= 0] .<= 0.8)
    @test all(w1.weights[w1.weights .<= 0] .>= -0.2)
    @test sum(w1.weights[w1.weights .>= 0]) <= 0.8
    @test sum(abs.(w1.weights[w1.weights .<= 0])) <= 0.2

    @test isapprox(w2.weights, wt2)
    @test all(w2.weights[w2.weights .>= 0] .<= 0.8)
    @test all(w2.weights[w2.weights .<= 0] .>= -0.2)
    @test sum(w2.weights[w2.weights .>= 0]) <= 0.6
    @test sum(abs.(w2.weights[w2.weights .<= 0])) <= 0.3

    @test isapprox(w3.weights, wt3)
    @test all(w3.weights[w3.weights .>= 0] .<= 0.8)
    @test all(w3.weights[w3.weights .<= 0] .>= -0.2)
    @test sum(w3.weights[w3.weights .>= 0]) <= 0.6
    @test sum(abs.(w3.weights[w3.weights .<= 0])) <= 0.3

    @test isapprox(w4.weights, wt4)
    @test all(w4.weights[w4.weights .>= 0] .<= 0.8)
    @test all(w4.weights[w4.weights .<= 0] .>= -0.2)
    @test sum(w4.weights[w4.weights .>= 0]) <= 0.6
    @test sum(abs.(w4.weights[w4.weights .<= 0])) <= 0.3

    @test isapprox(w5.weights, wt5)
    @test all(w5.weights[w5.weights .>= 0] .<= 0.8)
    @test all(w5.weights[w5.weights .<= 0] .>= -0.2)
    @test sum(w5.weights[w5.weights .>= 0]) <= 0.4
    @test sum(abs.(w5.weights[w5.weights .<= 0])) <= 0.6
end

@testset "$(:HRP), $(:HERC), $(:Variance)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :Variance, rf = rf,
                   cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :Variance)

    w2 = optimise!(portfolio; type = :HERC, rm = :Variance, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :Variance)

    w1t = [0.03360421525593201, 0.053460257098811775, 0.027429766590708997,
           0.04363053745921338, 0.05279180956461212, 0.07434468966041922,
           0.012386194792256387, 0.06206960160806503, 0.025502890164538234,
           0.0542097204834031, 0.12168116639250848, 0.023275086688903004,
           0.009124639465879256, 0.08924750757276853, 0.017850423121104797,
           0.032541204698588386, 0.07175228284814082, 0.08318399209117079,
           0.03809545426566615, 0.07381856017730955]

    w2t = [0.11540105921544155, 0.1145567201312018, 0.09419723372475931,
           0.060483660234210215, 0.07318364748176763, 0.02168713926476103,
           0.005201942574220557, 0.023749598552788367, 0.014061923482203057,
           0.015961970718579934, 0.03686671320264766, 0.012132318435028086,
           0.00441044709602619, 0.02704002898133752, 0.00862810493627861,
           0.016962353908264397, 0.14917275390007956, 0.03182856610967728,
           0.021005280556336624, 0.15346853749439043]

    rc1t = [5.352922093838876e-6, 8.405126658900443e-6, 4.579727314018537e-6,
            7.796610994026335e-6, 9.383639724141158e-6, 1.0935034548478898e-5,
            3.027030700581314e-6, 6.3395584379654415e-6, 5.0005417483053664e-6,
            9.1235145558061e-6, 1.2421076566896583e-5, 5.123109371899597e-6,
            1.9984161851537576e-6, 1.1877508173064802e-5, 3.461507083658984e-6,
            5.354213267505637e-6, 1.1583136431304079e-5, 9.666419624363176e-6,
            6.680309936554301e-6, 1.025204088461852e-5]

    rc2t = [2.5161582685333897e-5, 2.2789434405246687e-5, 2.169855496914287e-5,
            1.3456455013737158e-5, 1.7868635690046716e-5, 2.863484027700964e-6,
            1.2089935206818575e-6, 2.2074171647638546e-6, 2.7341070177970226e-6,
            2.4916478456357633e-6, 3.1641447072315953e-6, 2.6985998436882967e-6,
            7.450030055413553e-7, 3.160990119219948e-6, 1.2202122744455483e-6,
            2.585622933742098e-6, 2.9216053011781054e-5, 3.526778410550578e-6,
            3.613754907458292e-6, 2.669761169053313e-5]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, rc1t)
    @test isapprox(rc2, rc2t)
end

@testset "$(:HRP), $(:HERC), $(:Equal)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :Equal, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :Equal)
    w2 = optimise!(portfolio; type = :HERC, rm = :Equal, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :Equal)

    w1t = [0.05, 0.05000000000000001, 0.05, 0.04999999999999999, 0.04999999999999999,
           0.05000000000000001, 0.05000000000000001, 0.05, 0.05, 0.05000000000000001,
           0.04999999999999999, 0.04999999999999999, 0.05, 0.04999999999999999, 0.05,
           0.04999999999999999, 0.04999999999999999, 0.05, 0.05, 0.04999999999999999]

    w2t = [0.07142857142857142, 0.07142857142857142, 0.07142857142857142,
           0.07142857142857142, 0.07142857142857142, 0.03125, 0.05, 0.03125, 0.03125,
           0.03125, 0.03125, 0.05, 0.05, 0.03125, 0.05, 0.05, 0.07142857142857142, 0.03125,
           0.03125, 0.07142857142857142]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, w1.weights)
    @test isapprox(rc2, w2.weights)
end

@testset "$(:HRP), $(:HERC), $(:VaR)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :VaR, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :VaR)

    w2 = optimise!(portfolio; type = :HERC, rm = :VaR, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :VaR)

    w1t = [0.03403717939776512, 0.05917176810341441, 0.029281627698577888,
           0.04641795773733506, 0.06002080678226616, 0.07336909341225178,
           0.032392046758139795, 0.05383361136188931, 0.02967264551919885,
           0.05284095110693102, 0.09521120378350553, 0.04261320776949539,
           0.01642971834566978, 0.07971670573840162, 0.021147864266696195,
           0.050803448662520866, 0.06563107532995796, 0.05200824347813839,
           0.034112945793535646, 0.07128789895430926]

    w2t = [0.10414300257102177, 0.09871202573006761, 0.08959251861207183,
           0.07555514734087271, 0.09769669156091847, 0.022943300926461043,
           0.016563538745652133, 0.028652238501006684, 0.019401391461582398,
           0.017501979293453335, 0.030610780860929044, 0.02395924092174919,
           0.013446069288029138, 0.025629238086958404, 0.01730739640456328,
           0.028564197108681265, 0.11491557263229325, 0.027680710219061735,
           0.02230467164850204, 0.12482028808612478]

    rc1t = [0.000994634909720906, 0.0001698593509829888, 0.0011572240609587514,
            0.001521302412044034, 0.000757464610978218, 0.0015100223364483602,
            0.002525481611639436, -0.00029729612513407114, 0.00019674334826424382,
            0.0012601389088982937, 0.0011262715165465142, 0.0010755972260840067,
            0.0009847854070050577, 0.00019101400869316903, 6.669414153472188e-6,
            0.0006091200113400707, 0.0012152147454995818, -0.0002511611676685849,
            0.0003233472786960215, 1.9155357514522973e-5]

    rc2t = [0.0020896123610048387, 0.0015508111068072475, 0.0024285188307972497,
            0.002364793270021977, 0.001779735302882169, -1.828512223038174e-5,
            0.0007320901102877016, 5.845255460603851e-5, 0.00016064952284616696,
            0.0003372356972914778, 0.00027264182622373283, 0.0002742980914803267,
            0.00021203416953087038, 0.00022698981669930595, 0.00039521079014403907,
            0.00044869037844667495, 0.0012102901594463994, 0.0001366108591699078,
            0.00019600648867681698, 0.0009111000315422457]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, rc1t)
    @test isapprox(rc2, rc2t)
end

@testset "$(:HRP), $(:HERC), $(:DaR)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :DaR, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :DaR)

    w2 = optimise!(portfolio; type = :HERC, rm = :DaR, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :DaR)

    w1t = [0.057660532702867646, 0.04125282312548106, 0.05922785740065619,
           0.03223509018499126, 0.10384639449297382, 0.02480551517770064,
           0.030513023620218654, 0.037564939241009405, 0.02260728843785307,
           0.05164502511939685, 0.08286147806113833, 0.01696931547695469,
           0.00418787314771699, 0.06308172456210165, 0.007685315296496237,
           0.0694006227527249, 0.09119842783020647, 0.09248808273195344,
           0.044542968010494795, 0.06622570262706394]

    w2t = [0.14733227960307277, 0.06614386458233755, 0.15133705565662284,
           0.035584333675786964, 0.11463609164605132, 0.005518410268992302,
           0.013171538035678308, 0.010620417669043803, 0.014023858241571917,
           0.017001289131758357, 0.026781482271749004, 0.007217550381915265,
           0.0035803904597587304, 0.020388510168558176, 0.0065705021611778035,
           0.029518131826492447, 0.16035186159841677, 0.026148373666210922,
           0.027631100950263884, 0.11644295800454063]

    rc1t = [0.00037063624641478835, 0.009125983148284311, 0.004971401868472875,
            0.004140544853563044, 0.007693244633425402, 0.0005479851716762159,
            -0.0027022094043521083, -0.004242779270002042, 0.003122533803610591,
            0.008238435604051717, -0.002205611884924098, 0.0033524321160804944,
            0.0010212030756203106, 0.005567255853837193, 0.002429940687237058,
            0.009980976169751597, 0.008796451943770132, 0.0068665779885744615,
            0.0051502274017401215, 0.0053992781715604555]

    rc2t = [0.012897562926386988, 0.006688298007058434, -0.0030371877242671617,
            0.0062475823284178205, 0.022331244162834756, 0.00031056017667811136,
            0.0046155645459075405, -0.0007153524251961269, 0.0042670683436316,
            0.002474797532622316, -0.001863230861409188, -0.00016152053742888076,
            0.0006439853929179481, -0.0011570064771656249, -0.0004933606726985137,
            -0.00045035815028852955, 0.01872869301310738, 0.001991160176900652,
            0.003274617796360586, 0.003047190166454803]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, rc1t)
    @test isapprox(rc2, rc2t)
end

@testset "$(:HRP), $(:HERC), $(:DaR_r)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :DaR_r, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :DaR_r)

    w2 = optimise!(portfolio; type = :HERC, rm = :DaR_r, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :DaR_r)

    w1t = [0.05820921825149978, 0.04281381004716568, 0.061012991673039425,
           0.037261284520695284, 0.10225280827507012, 0.03171774621579708,
           0.029035913030571646, 0.03777811595747374, 0.021478656618637317,
           0.05210098246314737, 0.08336111731195082, 0.0224554330786605,
           0.008018398540074134, 0.060923985355596684, 0.009489132849269694,
           0.06345531419794387, 0.0883323204747975, 0.08122137323769983,
           0.039962166984027804, 0.06911923091688174]

    w2t = [0.14774355938966005, 0.0684328493553202, 0.15485994881153173, 0.0411979500128519,
           0.11305584705890495, 0.007216566584313541, 0.011970976561070937,
           0.011775533919110818, 0.013661517786887336, 0.016951592576994145,
           0.02646193157189205, 0.008427695298573155, 0.006529147510212775,
           0.019339548023735872, 0.007726723460706738, 0.02381526338245954,
           0.15152867379279225, 0.025316906660827858, 0.025417970255231653,
           0.11856979798692265]

    rc1t = [0.0027996542438643874, 0.00863503087332812, -0.0002850120378994755,
            0.006461172013997191, 0.0159529662837373, 0.0003986018570652638,
            -0.0011234647419220587, -0.004573534814566288, 0.00700223437754879,
            0.008874480333593358, -0.009222954986530287, 0.0023099671298011737,
            0.002080708519770288, 0.0017671706490277868, 0.0024782243075671976,
            0.004017780500329745, 0.012656330461833405, 0.009367018493098709,
            0.006962552592603614, 0.003747199175510561]

    rc2t = [0.011346536473254676, 0.0015330394857312022, 0.011842189250847353,
            0.01225382568618803, 0.007152494125562007, 0.00043980977341072434,
            0.002717144096906917, 0.0015898757517082715, 0.001827321061913043,
            0.001442775782392507, 0.0020624564567346443, 4.876274896089979e-5,
            -0.0006251187320900545, 0.0009975031068812075, 0.0008789713593400469,
            -0.0024839348401675477, 0.0155998463780046, 0.0036124764392451835,
            0.0032970368949980654, 0.006775161381475716]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, rc1t)
    @test isapprox(rc2, rc2t)
end

@testset "$(:HRP), $(:HERC), $(:MDD_r)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :MDD_r, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :MDD_r)

    w2 = optimise!(portfolio; type = :HERC, rm = :MDD_r, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :MDD_r)

    w1t = [0.054750291482807016, 0.05626561538767235, 0.044927537922610235,
           0.037029800949262545, 0.06286000734501392, 0.038964408950160866,
           0.033959752237812064, 0.04828571977339982, 0.01937785058895953,
           0.05279062837229332, 0.10144994531515807, 0.02678668518638289,
           0.010335110450233378, 0.07434568383344507, 0.01192345099055306,
           0.05711412672965212, 0.06785463405921893, 0.09152771438398988,
           0.032677346602089874, 0.07677368943928506]

    w2t = [0.15079446061615842, 0.08603152475897777, 0.12374041606663375,
           0.050529282163624914, 0.0857760767414279, 0.009923344774006223,
           0.0134887577916876, 0.016005392064562644, 0.0148967949294955,
           0.020046408045779833, 0.029228399800324625, 0.0099793139298191,
           0.008133340149925354, 0.02141948291605225, 0.00938330394572054,
           0.02127772796435491, 0.12849798956868397, 0.030338927541390587,
           0.025120832103469992, 0.14538822412790406]

    rc1t = [0.0034550221680853537, 0.012291022931897917, 0.0016226880627466954,
            0.010304830836857742, 0.014281747474458302, 0.0027506380865027252,
            0.002984547202021771, -0.004875253910488604, 0.007968299975178606,
            0.01269453197008402, -0.00886438306080444, 0.005940337248837948,
            0.003972986385357844, 0.0024925347104599177, 0.001769063618104259,
            0.01272730676962482, 0.012559611640909633, 0.011463366281645982,
            0.007142935943818474, 0.007781726985952924]

    rc2t = [0.010346635225787545, 0.01748241905247152, 0.009205466345910743,
            0.014186559552162302, 0.022392923591497707, 0.00036060233775705406,
            0.0011769124676254982, -0.0017262599696463786, 0.004438397394866984,
            0.003976006338512617, -0.002670785459958172, 0.002247446922654915,
            0.0028025909724020003, 0.0006575612922777944, 0.0012247309732810632,
            0.003764114649294143, 0.02281884397222136, 0.003930952996438839,
            0.003528227645220005, 0.016642396792995746]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, rc1t)
    @test isapprox(rc2, rc2t)
end

@testset "$(:HRP), $(:HERC), $(:ADD_r)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :ADD_r, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :ADD_r)

    w2 = optimise!(portfolio; type = :HERC, rm = :ADD_r, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :ADD_r)

    w1t = [0.05216718533133331, 0.04708278318245931, 0.07158823871210457,
           0.029511984859220904, 0.12867669205129983, 0.03859596234440916,
           0.022402839001206605, 0.026020360199894607, 0.030904469636691148,
           0.054239105004183275, 0.07390114503735276, 0.019880342325388017,
           0.00476824739544596, 0.03991945523395664, 0.0054480039590591904,
           0.059645486519252305, 0.11611402694084015, 0.06374794181935534,
           0.059893035563107475, 0.055492694883439435]

    w2t = [0.12371584689868499, 0.07328365011343618, 0.16977338386193674,
           0.03354181126220188, 0.14624734118080304, 0.009376911249269603,
           0.008150033847862812, 0.008167198562468324, 0.012903517155053858,
           0.011746529657682171, 0.021870250127331976, 0.005917245694323535,
           0.0032768170628257485, 0.011813734015517456, 0.003743956814916424,
           0.01775306443497017, 0.19872759251615896, 0.020009027345803686,
           0.025007088649056427, 0.09497499954969608]

    rc1t = [0.0009687079538756775, 0.001260723964791758, 0.0008797670948275687,
            0.0009757789813260052, 0.00342270121044295, 0.0005219915942389684,
            0.0004334749593047021, 1.7859240000045119e-6, 0.0010172760237372265,
            0.0014089802111120601, 0.00014546336770382634, 0.0006820878343605708,
            0.00015341040785324012, 0.0007523844031235969, 0.0002689991127515771,
            0.0011922843248944387, 0.0023474618914575145, 0.0010873671539977597,
            0.0011893951109258317, 0.0007539191255614216]

    rc2t = [0.003429257847573523, 0.0013995361644898684, 0.0029306085554606253,
            0.0013136264707558488, 0.005363383687100901, 0.00011496812912051169,
            0.00019797307157841686, 3.696220245701304e-6, 0.00027730234244732375,
            0.00021507764559819243, -6.971487016356138e-5, 0.00015612983732422745,
            8.235568082463573e-5, 4.761250063208198e-5, 2.3069718578884633e-5,
            9.24202087881252e-6, 0.004101295526671245, 0.0002473284342232724,
            0.000280947406159675, 0.001089151557439322]
    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, rc1t)
    @test isapprox(rc2, rc2t)
end

@testset "$(:HRP), $(:HERC), $(:CDaR_r)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :CDaR_r, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :CDaR_r)

    w2 = optimise!(portfolio; type = :HERC, rm = :CDaR_r, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :CDaR_r)

    w1t = [0.055226953695215866, 0.047934679854074826, 0.05395157314076186,
           0.036982877818185954, 0.08463665680704126, 0.033820937721103415,
           0.03026563101495272, 0.04118023926554901, 0.02122427296619385,
           0.05288666614109493, 0.09090289747478623, 0.02380032651263032,
           0.008673469487824291, 0.06662055705080387, 0.010138402407857017,
           0.060505269563432176, 0.08319151689594376, 0.08557053458440436,
           0.038056791259321536, 0.07442974633882277]

    w2t = [0.1482304036785224, 0.07559110451590667, 0.14480725317353932,
           0.04480939008007434, 0.10254791389098332, 0.00791620073213515,
           0.011996465335688086, 0.012913340180870978, 0.014391268171680365,
           0.01805225079803855, 0.0270384713414632, 0.008561039338370812,
           0.006787624092846017, 0.01981584825797337, 0.0079340412211216,
           0.021763902803450537, 0.1447239549586514, 0.026833292915610436,
           0.025804676072481453, 0.12948155844059195]

    rc1t = [0.0035078284972479635, 0.007327321104937156, 0.004126129944590046,
            0.0067238287919828025, 0.009407856778007472, 0.002257148119806701,
            0.0025872728909382904, -0.0005681237351201191, 0.004502106214997947,
            0.008405740528208947, -0.0023261277149486067, 0.0031444976568801184,
            0.001436135209826134, 0.005774659096445008, 0.0016894414466280476,
            0.008391486761558408, 0.009289105567599092, 0.008850699400918137,
            0.005076475962265737, 0.004232718475027715]

    rc2t = [0.009044451232082855, 0.012181446118845205, 0.012374311458586114,
            0.008410300213099705, 0.013699300902664057, 0.00038827887913860814,
            0.0009666745614939528, -0.0006216694418470095, 0.002941362902474998,
            0.002441384804663607, -0.0008954941575137244, 0.0013984227575186978,
            0.001106164364662281, 0.0010184575447202322, 0.0014885532721886172,
            0.0024413889937939057, 0.017006785617564434, 0.0026245540197739814,
            0.003029239588452687, 0.009066030325349785]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, rc1t)
    @test isapprox(rc2, rc2t)
end

@testset "$(:HRP), $(:HERC), $(:UCI_r)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :UCI_r, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :UCI_r)

    w2 = optimise!(portfolio; type = :HERC, rm = :UCI_r, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :UCI_r)

    w1t = [0.05276898612268049, 0.04614128625526376, 0.06552577565455027,
           0.0320518739448942, 0.1111735166748811, 0.039154188455666976,
           0.025443440854672504, 0.03239581981059534, 0.027120969094009423,
           0.05445935205465669, 0.08028963951603105, 0.02075298500986429,
           0.0058131044419531056, 0.04948420047533672, 0.006850658262213453,
           0.06288836730781266, 0.1025592250558978, 0.07454814435900761,
           0.050977607512044276, 0.05960085913796832]

    w2t = [0.13177358314927293, 0.07247391549233687, 0.16362956503582105,
           0.03659195084440874, 0.126920998889531, 0.00897565819404876,
           0.010308036952094911, 0.010070348969290802, 0.013930178878301605,
           0.014394572348790095, 0.025070306799710288, 0.006958517882519902,
           0.004449124903473799, 0.01545135954194139, 0.005243228396110122,
           0.021086596858518908, 0.17918434540380382, 0.02317354007700532,
           0.02618369531594223, 0.10413047606707744]

    rc1t = [0.001283438101843825, 0.0021135575930140683, 0.0013701577362353157,
            0.001964759875083139, 0.004228974865110063, 0.0009397927542550052,
            0.0007763919703674528, 3.1663742433858015e-5, 0.001710439572871529,
            0.002639836108019761, -0.0001687502320526006, 0.0008676789176204873,
            0.000274003566895468, 0.0016231943477036503, 0.00046365183669813545,
            0.0023177094086967017, 0.003506698798906536, 0.002535431260631784,
            0.0020515085558793144, 0.001090422920791348]

    rc2t = [0.004938619927487938, 0.0022415895087343645, 0.004027071213871287,
            0.002206508518833801, 0.0069462576421467715, 0.00023201799288260792,
            0.0007754292487379087, 1.2375110105652886e-5, 0.00076276979587084,
            0.0005251951132891504, 5.837067824298913e-6, 0.00018817162773168128,
            0.00011821788119412797, 0.00012553219491850875, 1.6671516192266337e-5,
            0.00017143118108609332, 0.0063472350413808255, 0.0005641484696555832,
            0.0008785186983768231, 0.001971880990312723]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, rc1t)
    @test isapprox(rc2, rc2t)
end

@testset "$(:HRP), $(:HERC), $(:EDaR_r)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :EDaR_r, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :EDaR_r)

    w2 = optimise!(portfolio; type = :HERC, rm = :EDaR_r, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :EDaR_r)

    w1t = [0.055560530905066345, 0.052522781694077786, 0.049959310773007755,
           0.036469678098098916, 0.07360442657586366, 0.03548425014502784,
           0.03187063039148397, 0.0436214861869137, 0.021541877133471526,
           0.054356898936874475, 0.09401419407671989, 0.02488151447327445,
           0.00927044194799896, 0.06971379898104886, 0.010782028044547592,
           0.05939091195793561, 0.07729538157400358, 0.0873665609377898,
           0.03800313067616797, 0.07429016649062732]

    w2t = [0.14882740870423186, 0.07965043278823442, 0.1338236813848157,
           0.04685245027365069, 0.09455930612766184, 0.008714108489009985,
           0.012663733570459775, 0.014222804429578661, 0.015035763684751,
           0.0191035907830612, 0.02818459897621233, 0.009107247269339425,
           0.007272821720022527, 0.020899559288876983, 0.00845868707542321,
           0.021738536865952895, 0.14067163803807126, 0.028485906656925727,
           0.026525363936696727, 0.13520235993702395]

    rc1t = [0.0029771085098667543, 0.00937437695164384, 0.003504867129659025,
            0.007287967333861093, 0.010576971944196838, 0.002375956494596482,
            0.0026442402698086554, -0.00207149642069502, 0.0054261194333417645,
            0.00901422055506963, -0.004085819021196649, 0.004419003319558396,
            0.0020569993192303265, 0.0059688360678046455, 0.001806298969770583,
            0.009677880657258185, 0.01041907884909654, 0.009149883275657777,
            0.005693681507938686, 0.0050346935207989515]

    rc2t = [0.008695351602964986, 0.015208324694851718, 0.009958227528870945,
            0.010543066552653546, 0.015053959694198132, 0.0003489555701996484,
            0.0011743380750797343, -0.0009085227406694986, 0.003573231892030077,
            0.0030069964795315465, -0.001857343118789652, 0.0016898527772629904,
            0.001619401905146628, 0.0008947836837998943, 0.0013149023778996221,
            0.003063584553639108, 0.020186174209623375, 0.0027423620529505887,
            0.0035275112880473066, 0.012119916672564934]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-7)
    @test isapprox(rc1, rc1t, rtol = 1e-1)
    @test isapprox(rc2, rc2t, rtol = 1e-1)
end

@testset "$(:HRP), $(:HERC), $(:RDaR_r)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))
    cluster_opt = ClusterOpt(; linkage = :ward,
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :RDaR_r, rf = rf, cluster_opt = cluster_opt)
    rc1 = risk_contribution(portfolio; type = :HRP, rm = :RDaR_r)

    w2 = optimise!(portfolio; type = :HERC, rm = :RDaR_r, rf = rf, cluster = false)
    rc2 = risk_contribution(portfolio; type = :HERC, rm = :RDaR_r)

    w1t = [0.055263517441901736, 0.054708858603147746, 0.04729258532814955,
           0.03659066794751168, 0.0676324029986855, 0.03707186730732357,
           0.033016951008785926, 0.045900281002362715, 0.020741988813931238,
           0.0538929592253236, 0.09778300255005679, 0.025642106596742546,
           0.009710156615873208, 0.07161140464324384, 0.01126211156930582,
           0.05878170411220516, 0.072570206971881, 0.0896251644842965, 0.03569138424980898,
           0.07521067852946291]

    w2t = [0.1496829925463977, 0.08274800430087025, 0.1280934676404464, 0.04841638920416211,
           0.08949048774983848, 0.009338808954830401, 0.013100309823737586,
           0.015211346672985305, 0.015288347630882252, 0.019793133202881993,
           0.02904499633835502, 0.009477734686286172, 0.007610137222474402,
           0.02127110980613775, 0.00882645026892602, 0.021726662551608086,
           0.13497973958875764, 0.029701766868639134, 0.026307134770233602,
           0.13989098017154974]

    rc1t = [0.003150975185181196, 0.010290205648299808, 0.0029792831227057315,
            0.008262536554226962, 0.012266020634903386, 0.0022783014294430266,
            0.0028679705718045716, -0.002800801206625369, 0.006255518626425792,
            0.010256082901898242, -0.005667652558527162, 0.005219179611186994,
            0.002834883629359977, 0.004119771472473114, 0.0018500045314005823,
            0.010824540073587206, 0.011062290000810555, 0.009825611739104194,
            0.0060877401415700754, 0.006421839644194061]

    rc2t = [0.00872831462077643, 0.015584079367838895, 0.008722059376251022,
            0.012129426553556623, 0.018684122034460345, 0.00039674032941621986,
            0.0011018850278911216, -0.0014176077839749485, 0.004268120679928374,
            0.003558287175207897, -0.0021174843207469106, 0.0018844817233710867,
            0.0021566889363572295, 0.0007496689791092665, 0.001173329861999755,
            0.0035544485998392408, 0.021569812267775067, 0.0034411546443547285,
            0.0036626248116680836, 0.013596075988929432]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-7)
    @test isapprox(w2.weights, w2t)
    @test isapprox(rc1, rc1t, rtol = 1e-1)
    @test isapprox(rc2, rc2t, rtol = 0.1)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:SD)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    cluster_opt = ClusterOpt(; linkage = :DBHT, genfunc = GenericFunction(; func = dbht_d),
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :SD, rf = rf, cluster_opt = cluster_opt)
    w2 = optimise!(portfolio; type = :HERC, rm = :SD, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SD, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SD, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.03692879524929352, 0.06173471900996101, 0.05788485449055379,
           0.02673494374797732, 0.051021461747188426, 0.06928891082994745,
           0.024148698656475776, 0.047867702815293824, 0.03125471206249845,
           0.0647453556976089, 0.10088112288028349, 0.038950479132407664,
           0.025466032983548218, 0.046088041928035575, 0.017522279227008376,
           0.04994166864441962, 0.076922254387969, 0.05541444481863126, 0.03819947378487138,
           0.07900404790602693]

    w2t = [0.04414470568410749, 0.09179295067888649, 0.03988345772432875,
           0.03195896901776689, 0.03515448610066942, 0.04581729858991488,
           0.032884067201388985, 0.050021575151391134, 0.03689357244197575,
           0.08760528183690483, 0.059737295315921264, 0.023064734096888414,
           0.030993923115529922, 0.05116016324833899, 0.01945065635759669,
           0.06078246420345067, 0.10474753188814133, 0.05790789306651554,
           0.045091282572348464, 0.050907691707934126]

    w3t = [0.025110084676904985, 0.0024072043691457027, 0.022131010847542314,
           0.024287124319864878, 0.011436259658313023, 0.06564901809923375,
           2.63164426952892e-5, 0.13509808681926533, 7.488940413328548e-6,
           7.323985175041516e-5, 0.2420077978406655, 0.006175255683734641,
           2.081662938420102e-6, 0.11544046014733363, 5.277576340805123e-7,
           2.542942782872028e-5, 0.004098561315789697, 0.19633523242335496,
           0.041554271004415, 0.10813454871117643]

    w4t = [7.766245844537851e-9, 0.0007499306762073744, 4.521156798112541e-8,
           5.041455975571854e-8, 0.8059997861132897, 2.7911723268365223e-19,
           0.030168550042525746, 1.1371070006277296e-8, 6.886320880414425e-16,
           5.00443088921232e-8, 1.51378253271611e-15, 1.8839581457880405e-19,
           2.1654220382372163e-16, 6.150855869911519e-17, 2.196410145913958e-17,
           7.066817209213057e-8, 0.16308142043655438, 1.3379231760651016e-8,
           4.668474244005647e-8, 1.7191521558665625e-8]

    w5t = [2.8827999879674876e-10, 0.03273675534922566, 0.03166387699337444,
           0.023020837616192273, 0.46917330788347095, 1.7899184506262896e-11,
           0.02979193004065076, 3.955332220592653e-9, 2.70373025377355e-10,
           0.028622875545023355, 5.506596966175203e-10, 1.983395086106402e-11,
           5.3886532079228104e-11, 4.173716737941434e-11, 1.0187054289168763e-11,
           0.11114834385497208, 0.2136479220104585, 0.003039004633947739,
           0.05715512195801641, 1.8906478971120764e-8]

    w6t = [3.5470437040666906e-14, 1.488158342101384e-7, 5.119559450781639e-14,
           4.44103404384708e-14, 1.5080369724560962e-6, 9.215797090311397e-16,
           0.9999980306220989, 2.5081869081132618e-14, 1.0945240779015975e-13,
           1.1011912550096345e-14, 2.104342763173577e-15, 9.605642353231626e-16,
           1.7440131500154973e-15, 1.404582295421992e-15, 6.154293965257424e-16,
           5.700592750757033e-8, 2.1304134377873232e-7, 2.6490033873279397e-14,
           4.2477479461105805e-8, 3.295643720342401e-14]

    w7t = [0.036868517647378266, 0.08857050773217715, 0.03550282387737573,
           0.03365142028202789, 0.03241016006814263, 0.04372860546359345,
           0.035016342750398624, 0.0589104349212394, 0.03280516637754979,
           0.09590014180820772, 0.06533183037607224, 0.02703407226272167,
           0.03467055356530615, 0.04660549160799435, 0.022776594296513932,
           0.06581413753388214, 0.09949899087728453, 0.0602676675157174,
           0.037338533903805926, 0.04729800713261096]

    w8t = [0.0010408874013922965, 0.3566484161420694, 0.0009173959653139009,
           0.001006773256477595, 0.0004740668440810784, 2.2963059738495353e-11,
           0.0038990115363986898, 0.005600215942390388, 2.6195210692923182e-15,
           0.007672333725607141, 8.46507637112972e-11, 2.1600134971057037e-12,
           0.00021806724598806248, 4.037937290388133e-11, 1.8460184827926966e-16,
           0.0026638920217650144, 0.607237765298803, 0.008138677049817413,
           1.4535072040554169e-11, 0.004482497405205028]

    w9t = [9.207376078686123e-11, 0.04506682138768791, 0.010113126987793776,
           0.0073526262822031395, 0.1498492822878779, 9.706876493066787e-11,
           0.04101284857383581, 1.2632937221849326e-9, 1.46625538357711e-9,
           0.028989349803876133, 2.9862732924539796e-9, 1.0756116364437701e-10,
           5.4576470687067085e-11, 2.2634412689621362e-10, 5.524524191508721e-11,
           0.11257143661421957, 0.29411689211049913, 0.0009706278162348274,
           0.30995697574853875, 6.038541103700144e-9]

    w10t = [0.057705337541387534, 0.2075611911365472, 0.055567075230773366,
            0.05267017895610822, 0.05072756298048495, 2.720282854440657e-11,
            0.08205876648985418, 0.0921918677977867, 2.04082874391969e-11,
            3.487417847127784e-10, 4.0638863014412347e-11, 1.6818645833427202e-11,
            1.260803344083395e-10, 2.8992883536950922e-11, 1.416995901864447e-11,
            2.393390477605853e-10, 0.23317197714193869, 0.09431821089387572,
            2.3228632064348586e-11, 0.07402783094562204]

    w11t = [1.2955071122615352e-10, 0.03248044664206342, 0.014229491472841513,
            0.010345379140584112, 0.21084270840269012, 8.420568170415045e-11,
            0.029558677508714423, 1.7774944652846946e-9, 1.2719543121278865e-9,
            0.04511783699736455, 2.5905467997413312e-9, 9.330767848994134e-11,
            8.494058421487245e-11, 1.9635009798094963e-10, 4.792440966662475e-11,
            0.17520157444305318, 0.21197518987514943, 0.0013657042229457167,
            0.26888297652189935, 8.49641948006916e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w11t, rtol = 0.0001)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:MAD)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    cluster_opt = ClusterOpt(; linkage = :DBHT, genfunc = GenericFunction(; func = dbht_d),
                             max_k = ceil(Int, sqrt(size(portfolio.returns, 2))))

    w1 = optimise!(portfolio; type = :HRP, rm = :MAD, rf = rf, cluster_opt = cluster_opt)
    w2 = optimise!(portfolio; type = :HERC, rm = :MAD, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MAD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MAD, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MAD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MAD, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MAD, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MAD, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :MAD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MAD, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :MAD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :MAD, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :MAD, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :MAD, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :MAD, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.03818384697661099, 0.0622147651480733, 0.06006655971419491,
           0.025623934919921716, 0.05562388497425944, 0.0707725175955155,
           0.025813212789699617, 0.0505495225915194, 0.030626076629624386,
           0.06238080360591265, 0.09374966122500297, 0.04170897361395994,
           0.0238168503353984, 0.04472819614772452, 0.016127737407569318,
           0.05166088174144377, 0.07470415476814536, 0.05398093542616027,
           0.03829328495650196, 0.0793741994327616]

    w2t = [0.04613534551516446, 0.08879849775073763, 0.040706520594783996,
           0.03095992637181892, 0.03769576333387878, 0.04693203454569473,
           0.03524876915694045, 0.05391250502938817, 0.03592303405315404,
           0.08262135256691483, 0.057360765752110146, 0.025519651313611693,
           0.029851929945314427, 0.05081167212149685, 0.018321268814608827,
           0.06475150999990342, 0.10201091696487545, 0.05757220451261372,
           0.04491633048971518, 0.04995000116727427]

    w3t = [0.048769433344083575, 0.02544569659248763, 0.025219990928096236,
           0.013126578884206973, 0.00782714947892682, 0.05883083701216104,
           3.728446760634107e-10, 0.15351397397756708, 0.005713867058723632,
           0.0010274658777645247, 0.19717045408027606, 2.0381761138478287e-9,
           9.190595336369797e-6, 0.10390407371192559, 2.2399817305847242e-10,
           0.00037773952649507473, 0.052613073765157614, 0.17923982488273116,
           0.0406425192695901, 0.08656812837945159]

    w4t = [0.024545121562307103, 0.03671551379661241, 0.02610706344511154,
           0.019373559500425738, 0.05227270115699957, 0.023122456748759315,
           0.0005557250043568566, 0.1418175414287909, 0.0048184978354368305,
           0.0156484732735207, 0.17305128527640537, 2.78703928974334e-10,
           9.855054120566305e-12, 0.07061956916846784, 1.8761812530022347e-11,
           0.007128113653725459, 0.08320020524774414, 0.1741902562147324,
           0.06626826933088348, 0.08056564704839982]

    w5t = [1.5141541768464337e-10, 0.02041494145729848, 8.711256350418566e-10,
           7.971722052122192e-10, 0.629985210145492, 4.779327059997394e-12,
           0.015291239641568366, 4.315435479886952e-10, 7.459813055761341e-10,
           4.992584030823763e-9, 1.8138302458210645e-10, 4.678066920424288e-12,
           2.1160570813031135e-12, 1.1097613495239535e-11, 1.736443123719628e-12,
           0.13366593686096678, 0.1394634297151964, 4.735499936913378e-10,
           0.06117923256582268, 9.444926177611966e-10]

    w6t = [3.689723642051115e-12, 1.4935139894877351e-6, 5.666817264569885e-12,
           4.895832761068173e-12, 1.60334545546758e-5, 8.160604552219426e-14,
           0.9999792769483065, 2.4340014867901207e-12, 1.0248348772382915e-11,
           1.4418755263940224e-12, 2.0464538722089622e-13, 8.907750738147742e-14,
           2.359177448026788e-13, 1.3055741215780125e-13, 5.5719386564832484e-14,
           5.712072407706186e-7, 2.2162236060520632e-6, 2.593489715070043e-12,
           4.0861712206000154e-7, 3.4127776858404117e-12]

    w7t = [0.0397269623632107, 0.0853749415795691, 0.03794582324922418,
           0.031835445416380406, 0.03420110618046517, 0.04505298633290963,
           0.03423950450604258, 0.05911662979130967, 0.03283507147517853,
           0.08817478594522044, 0.06547961956448613, 0.027396245853727966,
           0.031758255514355606, 0.04655168865926364, 0.020957432262550254,
           0.06853877551190557, 0.10027761462172244, 0.06319618247932957,
           0.03791663360853293, 0.04942429508461542]

    w8t = [7.417458367797647e-10, 0.3207312747117785, 3.835768019397794e-10,
           1.99645240284566e-10, 1.1904496611403e-10, 2.674224421253834e-11,
           4.699535254956333e-9, 2.3348303081973652e-9, 2.5973050196579657e-12,
           0.011699351445957346, 8.962613320327011e-11, 9.264767620663642e-19,
           0.00010464970872970368, 4.723081048987155e-11, 1.0182098626021327e-19,
           0.0043011720108025, 0.6631635394169465, 2.726100854071717e-9,
           1.8474531910798827e-11, 1.3166351220495404e-9]

    w9t = [3.23499521621336e-11, 0.04846399408243397, 1.861162690810345e-10,
           1.7031609526916309e-10, 0.13459654058151635, 2.713753909854914e-11,
           0.036300596259467105, 9.219941632619864e-11, 4.2357630253641e-9,
           3.816460178543894e-9, 1.0299125503691093e-9, 2.656257300765533e-11,
           1.6175686851657575e-12, 6.301345698819232e-11, 9.859713003690921e-12,
           0.10217769437792956, 0.3310788251130083, 1.0117410681519481e-10,
           0.34738233959137105, 2.0179114828120886e-10]

    w10t = [0.06148357053771031, 0.19870996215540077, 0.058726984435026594,
            0.04927023707380326, 0.05293146012748414, 4.948298010148765e-11,
            0.07969236076459679, 0.0914920563141677, 3.6063695676713464e-11,
            4.2071270893259106e-10, 7.191813408372553e-11, 3.0090078345044824e-11,
            1.5152973228331597e-10, 5.112904761952157e-11, 2.3018145699889617e-11,
            3.2702244301960286e-10, 0.23339589624134885, 0.09780579012454467,
            4.1644920327865464e-11, 0.07649168102330511]

    w11t = [5.548062062971821e-11, 0.033393567882112785, 3.1917641278061624e-10,
            2.920805652606498e-10, 0.23076431955364263, 2.2077626864486315e-11,
            0.025012512839723867, 1.581174296181792e-10, 3.4455863532073866e-9,
            7.479514022548886e-9, 8.377691241827936e-10, 2.1609946284128364e-11,
            3.178125531283781e-12, 5.1260469209582576e-11, 8.023173041056775e-12,
            0.20016806954479488, 0.22812609429875524, 1.7350828207232325e-10,
            0.28253542266753057, 3.460577274365427e-10]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-7)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t, rtol = 1.0e-7)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-5)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:SSD)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :SSD, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :SSD, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SSD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SSD, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SSD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SSD, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SSD, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SSD, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :SSD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SSD, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :SSD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SSD, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SSD, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SSD, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SSD, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.03685118468964678, 0.060679678711961914, 0.05670047906355952,
           0.026116775736648376, 0.05374470957245603, 0.07054365460247647,
           0.025708179197816795, 0.04589886455582043, 0.03028786387008358,
           0.0634613352386214, 0.10260291820653494, 0.038697426540672014,
           0.027823289120792207, 0.04746469792344547, 0.01859405658396672,
           0.052617311519562025, 0.07250127037171229, 0.05655029966386585,
           0.03752315313496762, 0.07563285169538961]

    w2t = [0.04458318301190374, 0.08923261844696723, 0.0386901731987397,
           0.03159651452603008, 0.036673272540490374, 0.046431350386991595,
           0.03571458733541658, 0.04710646865985023, 0.037030915268657386,
           0.08746809871032526, 0.06001952275591192, 0.022636793508911755,
           0.03398552513326154, 0.052297035105064366, 0.020487100360053728,
           0.06427086874342236, 0.10072097804729256, 0.05803814418069801,
           0.0458770123345119, 0.04713983774549962]

    w3t = [0.018187443289737027, 0.00044753319482864043, 6.049805177405795e-9,
           0.019592871512766584, 0.0178723742061117, 0.062187542526660186,
           4.161951729707088e-13, 0.12115687510539876, 1.6611119269554933e-9,
           1.2715247783761365e-9, 0.255724554118471, 0.01005878702480125,
           3.4200535177968427e-11, 0.1280358467816251, 5.293387215900223e-10,
           4.924929542819093e-10, 0.0007744876146257587, 0.22935679935384753,
           0.03671943535904271, 0.09988543987319361]

    w4t = [0.011815073118454212, 0.010780179150189908, 6.30750020542261e-9,
           0.020027927841892904, 0.03729094057380963, 0.03297647882124728,
           1.126343827098314e-11, 0.12201587639766563, 5.970704144934237e-10,
           5.386850811241261e-10, 0.24170004552003954, 0.0074566970240480274,
           2.9765884574429157e-12, 0.10598635914181115, 1.0982477527482656e-10,
           2.2594233563162794e-10, 0.019647153193340476, 0.22911368724636835,
           0.06094459068414003, 0.10024498349372997]

    w5t = [1.3113648513898847e-9, 0.01219354254601431, 5.251429373618801e-9,
           2.530543516012152e-9, 0.6589435966421197, 1.0246367530996652e-11,
           0.018754424451295545, 4.199834668620207e-9, 2.7800232537404553e-10,
           0.019117714777968565, 2.4403236918197824e-10, 1.0930848123322397e-11,
           1.1300536852316676e-11, 2.639376581697133e-11, 4.005323771265213e-12,
           0.1519585412769984, 0.11938083350604307, 1.0819595675333878e-8,
           0.019651317335923686, 4.765957081281191e-9]

    w6t = [8.191863340708682e-12, 9.211591204018666e-7, 1.2626012115872571e-11,
           1.0915767009084885e-11, 4.450729454855096e-5, 1.6702195024225544e-13,
           0.9999505430491591, 5.463121371861769e-12, 2.0302991811258166e-11,
           2.2409245620376714e-12, 4.295640077655596e-13, 1.8430659547832975e-13,
           3.8089532971577737e-13, 2.6765280659944193e-13, 1.1695095842702918e-13,
           1.5578279433378499e-6, 1.3373821711886017e-6, 5.799461049486718e-12,
           1.133212413754835e-6, 7.557214320949884e-12]

    w7t = [0.03754844098300919, 0.08680041787053981, 0.03389183000114803,
           0.03277213059086383, 0.03320752877178498, 0.04331700156529129,
           0.037633843557708824, 0.05764895433246242, 0.03208049588246182,
           0.0939356459430743, 0.06696242782914329, 0.02713647380528876, 0.0380939037248661,
           0.04717402044229326, 0.024240034460649925, 0.0688395222010592,
           0.09548573951989676, 0.06001050286483363, 0.03771243943429284,
           0.04550864621933167]

    w8t = [0.001098555251985528, 0.3506906289283113, 3.654194350053954e-10,
           0.0011834457190567865, 0.0010795244959281231, 1.0338148716144361e-10,
           3.2613390169171995e-10, 0.0073180995998649205, 2.761456947372526e-18,
           0.008377856387202816, 4.251202674093581e-10, 1.6721875787556288e-11,
           0.00022534139873579267, 2.1284867856909692e-10, 8.799804916982116e-19,
           0.0032449507181074636, 0.6068947550902725, 0.013853575375871495,
           6.104293047950748e-11, 0.00603326552399483]

    w9t = [3.0909578666815647e-10, 0.028304414662088006, 1.2377903004267514e-9,
           5.964627906190747e-10, 0.15531656896108528, 1.7300444637227477e-10,
           0.04353394466087989, 9.89923741968133e-10, 4.6939208696199814e-9,
           0.01831898713435886, 4.1203562920737975e-9, 1.8456153580613036e-10,
           1.082840661727534e-11, 4.4564464714261946e-10, 6.762775388382429e-11,
           0.14560979672200497, 0.27711426777817116, 2.550237207558349e-9,
           0.3318020035785962, 1.123361855935011e-9]

    w10t = [0.056331156322712087, 0.2167050029967595, 0.05084541258907015,
            0.04916561015616197, 0.04981880593074, 1.122131530348982e-10,
            0.09395625483181962, 0.08648647382756333, 8.310486561421124e-11,
            8.007799352526373e-10, 1.7346688113336067e-10, 7.029732386602455e-11,
            3.2474183204962084e-10, 1.2220480143768866e-10, 6.279406695323108e-11,
            5.868412100385516e-10, 0.2383886849446839, 0.09002933089587331,
            9.769447525542677e-11, 0.06827326507047765]

    w11t = [4.758967098460148e-10, 0.02161483911959314, 1.90575335174275e-9,
            9.18338883428181e-10, 0.2391318398413062, 1.4017035850748457e-10,
            0.033244962714122916, 1.524127704486835e-9, 3.80307318624984e-9,
            0.02520605599426584, 3.3383640175106555e-9, 1.4953405639627022e-10,
            1.4899373066963704e-11, 3.6106684693239817e-10, 5.4792849003161377e-11,
            0.20035216262610134, 0.21162000300228642, 3.926451115643559e-9,
            0.268830118360281, 1.7295745664108087e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 0.0001)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:FLPM)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :FLPM, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :FLPM, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :FLPM, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :FLPM, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :FLPM, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :FLPM, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :FLPM, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :FLPM, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :FLPM, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :FLPM, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :FLPM, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :FLPM, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :FLPM, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :FLPM, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :FLPM, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.039336538617540336, 0.06395492117890568, 0.06276820986537449,
           0.026249574972701337, 0.06242254734451266, 0.06144221460126541,
           0.026935168312631458, 0.05003386507226852, 0.031591718694359686,
           0.062032838812258684, 0.08971950983943798, 0.03794006123084816,
           0.021868414535268534, 0.04077527283143069, 0.014126622482894119,
           0.053377350594462684, 0.07977242881987609, 0.05414859107042937,
           0.04026174326124142, 0.0812424078622927]

    w2t = [0.047209097585607665, 0.09109520123519059, 0.04231279220538024,
           0.031502994163152144, 0.04207977701425431, 0.041259290117364664,
           0.03636245388481573, 0.053027475329912796, 0.03609227989587427,
           0.08122285780986915, 0.05524398732616691, 0.023361254040976172,
           0.027236696268036157, 0.04712363804034537, 0.01632601815734191,
           0.06648047956970955, 0.107692709790203, 0.057388392301681176,
           0.045997437522764306, 0.05098516774135389]

    w3t = [0.026492938748950267, 0.03023455444435084, 0.030374427574782304,
           0.020581642877391074, 0.052020049877156894, 0.02252542138909628,
           0.00030263189838778103, 0.13226967239072696, 0.0035698997148049874,
           0.014416488080618498, 0.17498668511170243, 1.2353023124069742e-9,
           2.078371058879246e-11, 0.07051698705320793, 1.4431210299193385e-10,
           0.006815051172128916, 0.06808389888188425, 0.18861148724759166,
           0.07248085421301562, 0.08571730792380527]

    w4t = [1.1304636874466858e-8, 0.04105910569753555, 0.036107656290973116,
           0.021499948405940276, 0.08075253280990574, 2.6850430532291318e-9,
           0.0006783875276609762, 0.11011285589969935, 0.004240434884212154,
           0.0175211018028951, 0.1753577430141126, 8.465634405506131e-11,
           3.9784745607253946e-12, 0.047557065796411675, 4.869859912805687e-12,
           0.00978229760983638, 0.1023130059106746, 0.16539700774888363,
           0.09583970043342693, 0.09178114208464726]

    w5t = [1.4643927363432183e-10, 0.016536710102701184, 9.489875252056378e-10,
           3.104686353899987e-10, 0.6486662295311443, 3.703125446798951e-12,
           0.011460020154686461, 3.097072031514349e-10, 6.061483180822724e-10,
           8.570931643984398e-9, 1.3634389556193452e-10, 3.5596075760455198e-12,
           1.1022985423016359e-12, 8.568403313509519e-12, 1.4261626112412809e-12,
           0.14389040710007642, 0.10522349017643833, 4.485586769631565e-10,
           0.07422313080589146, 6.331172358883636e-10]

    w6t = [3.692099183873467e-12, 1.4939646139314987e-6, 5.670082610112128e-12,
           4.898751691669253e-12, 1.6034686690774843e-5, 8.163689752651537e-14,
           0.9999792744266554, 2.435707871829555e-12, 1.0251554687850123e-11,
           1.4425657184130816e-12, 2.047171327871953e-13, 8.911126343306908e-14,
           2.3602983249249915e-13, 1.3060622028590832e-13, 5.5739684918097694e-14,
           5.712972619605853e-7, 2.2169015877985037e-6, 2.595287068381251e-12,
           4.0868799144378423e-7, 3.4150163366229753e-12]

    w7t = [0.0401238956067253, 0.08558543246992718, 0.03799875320276207,
           0.03282162910771337, 0.03806361472814694, 0.03929540405460102,
           0.035694281611878646, 0.05796005155382424, 0.03391254267706462,
           0.08591459255093387, 0.06385378346237242, 0.024862079783365205,
           0.029417260668287995, 0.04405276916050261, 0.018807181412354664,
           0.07078372573476682, 0.10570705434961825, 0.06457179935311778,
           0.039913014977694936, 0.05066113353434207]

    w8t = [0.010426710722913998, 0.22611466014064474, 0.011954331404963752,
           0.00810022770665195, 0.020473304868157155, 7.494378068869082e-11,
           0.002263288151893269, 0.05205679990814005, 1.1877326363202773e-11,
           0.03494632066368692, 5.821939366160095e-10, 4.109944226396341e-18,
           5.0380800841063516e-11, 2.346153495313097e-10, 4.801372818084786e-19,
           0.016520040267007586, 0.5091779237250743, 0.07423100303008641,
           2.4114928411055777e-10, 0.03373538821561933]

    w9t = [3.7942641420226046e-11, 0.04882009126430664, 2.458906128347853e-10,
           8.044417965764992e-11, 0.16809076166570888, 1.6273114959993478e-11,
           0.033832563780413785, 8.024692406173103e-11, 2.663597709241706e-9,
           6.704903932728978e-9, 5.991357648247557e-10, 1.5642454175422475e-11,
           8.703901805967287e-13, 3.7652617108499593e-11, 6.267453869155787e-12,
           0.11246013305501368, 0.31064347603984416, 1.1622462296588133e-10,
           0.32615296342557504, 1.6404548141905757e-10]

    w10t = [0.06274984802805705, 0.18705818360383505, 0.059426333178110115,
            0.05132981749152642, 0.05952777025941103, 6.953482210966254e-11,
            0.07801453227110706, 0.09064385129372175, 6.00096290919783e-11,
            6.91796464283652e-10, 1.1299187731765033e-10, 4.3994465424261686e-11,
            2.368719482335596e-10, 7.795317393243379e-11, 3.3280075503870826e-11,
            5.699605822271128e-10, 0.23103662632889518, 0.1009839781266699,
            7.062770985833279e-11, 0.07922905745164562]

    w11t = [5.442472301307184e-11, 0.03495852581322169, 3.526941460526243e-10,
            1.1538674278102508e-10, 0.2410431386095624, 1.3830577916420247e-11,
            0.024226428924494274, 1.1510374866292715e-10, 2.2628020316251744e-9,
            1.1970954467914404e-8, 5.089867273804682e-10, 1.3294813294830956e-11,
            1.575294972297995e-12, 3.199307373367622e-11, 5.3304797842765034e-12,
            0.20062114759644767, 0.22244196627553878, 1.6670833748569378e-10,
            0.27670877693235, 2.352999662729138e-10]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-5)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-5)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:SLPM)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :SLPM, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :SLPM, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SLPM, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SLPM, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SLPM, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SLPM, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SLPM, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SLPM, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :SLPM, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SLPM, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :SLPM, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SLPM, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SLPM, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SLPM, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SLPM, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.03752644591926642, 0.06162667128035028, 0.058061904976983145,
           0.02658920517037755, 0.056905956197316004, 0.06572780540966205,
           0.026284445157011994, 0.04570757302464124, 0.030832527351205462,
           0.06360209500884424, 0.10038131818271241, 0.03703703080055575,
           0.02666888092950761, 0.04516285946254531, 0.01726932945234202,
           0.053613204742961086, 0.07517335594118413, 0.05681780552493213,
           0.03852165318888371, 0.07648993227871743]

    w2t = [0.04515679764750727, 0.09059570188638273, 0.039470826171407555,
           0.03199565874343434, 0.038685005358890655, 0.04353400590614482,
           0.03634322746049435, 0.046729173297276036, 0.037059295831499514,
           0.08694385655684449, 0.05886348230651655, 0.021718469598558048,
           0.03235988280277077, 0.050292400625876185, 0.019230758319847097,
           0.06505398658267612, 0.10394141316733571, 0.058087728248317386,
           0.04630127544148449, 0.047637054046735884]

    w3t = [0.01020880963628609, 0.010125869992205584, 9.973047214246865e-9,
           0.0203824556126149, 0.036474143292356306, 0.03223328564638525,
           1.5877270708741356e-11, 0.12139620966109949, 5.434398730066544e-10,
           1.8028536786512066e-9, 0.24349582994957766, 0.007496425458235115,
           9.622563025564643e-12, 0.10623328243395391, 1.1369255675001145e-10,
           7.577411857997474e-10, 0.018222544967697693, 0.23103083294264504,
           0.06213411697705032, 0.10056618021361836]

    w4t = [0.000419439291028012, 0.01851092879893747, 0.00088840346312887,
           0.020193037536315517, 0.0595056565454294, 0.006698534750707415,
           1.400685028383808e-11, 0.12206579043150166, 5.110258790626125e-10,
           8.812789656128368e-10, 0.23163082049415953, 0.005421005185472349,
           8.986234477603633e-19, 0.0872187407035392, 6.162473306624368e-11,
           3.970648009688971e-10, 0.03515648404689304, 0.22970975001493654,
           0.08182570556041982, 0.10075570131252981]

    w5t = [1.8823310889998093e-9, 0.011764798189026588, 7.37441465092624e-9,
           3.5138102776051106e-9, 0.6595446478510552, 3.721161506600326e-12,
           0.018947248448056998, 6.193676840939075e-9, 9.863373166039355e-11,
           0.020280462944050892, 8.808223846556176e-11, 3.980627406743843e-12,
           1.2031210097356011e-11, 9.623845868703762e-12, 1.4614894077549446e-12,
           0.1537294789615208, 0.11837110344473635, 1.785387190928891e-8,
           0.017362216488291213, 6.6376228861735174e-9]

    w6t = [9.668867214584299e-12, 9.908537180505388e-7, 1.4881046884436504e-11,
           1.2873353807852623e-11, 4.9526035930584876e-5, 1.8531103534454164e-13,
           0.9999450175294197, 6.428513393464366e-12, 2.243177161035309e-11,
           2.5383173660330557e-12, 4.749674447162343e-13, 2.043562121910656e-13,
           4.336161107966961e-13, 2.961893473592315e-13, 1.3010009399015433e-13,
           1.7521732588631392e-6, 1.4465572918838251e-6, 6.829231784521187e-12,
           1.2667640859428306e-6, 8.91941562821517e-12]

    w7t = [0.03784338270252465, 0.08818470567689848, 0.034419813135385634,
           0.03335457162925605, 0.03528810436731341, 0.040177350663990676,
           0.03855288093494366, 0.05759781328595066, 0.03223197005902945,
           0.09358541826855041, 0.0659416542856297, 0.025849000695795075,
           0.03647205386597262, 0.045269855432872284, 0.02255814790486359,
           0.06999549446693082, 0.09827775216211779, 0.06016586115788434,
           0.0382639600685086, 0.04597020923558208]

    w8t = [0.002632651103007388, 0.29641975750150723, 2.5718526140020564e-9,
           0.00525623419010788, 0.00940596376962165, 1.0303132833834275e-10,
           4.647834444737699e-10, 0.03130569347960495, 1.7370656098217205e-18,
           0.02527253564775949, 7.78316522857037e-10, 2.396177297048058e-11,
           0.00013488979719548474, 3.395668788775821e-10, 3.634099009527976e-19,
           0.010622071750285716, 0.5334378541836996, 0.05957830529166752,
           1.9860713789794444e-10, 0.025934038805423235]

    w9t = [5.581435795916855e-10, 0.02588125121913053, 2.186640923435207e-9,
           1.0419052513182867e-9, 0.19556634473254608, 6.60093638017008e-11,
           0.041681845206051754, 1.8365318317473785e-9, 1.749655279606101e-9,
           0.019636004897306825, 1.5624832496596917e-9, 7.061200708023903e-11,
           1.1648890907665915e-11, 1.7071657384179026e-10, 2.5925234859522595e-11,
           0.1488443735272, 0.26040329941200535, 5.293980445447003e-9, 0.3079868644633374,
           1.9681694784296034e-9]

    w10t = [0.05701688575155151, 0.21202597480198598, 0.05185875080345578,
            0.0502538003757883, 0.053166965303175884, 1.1532776273018472e-10,
            0.09269421606515274, 0.0867799785626763, 9.252080920870121e-11,
            8.739569791141452e-10, 1.8928334829964128e-10, 7.419870573320007e-11,
            3.405979970867478e-10, 1.2994563006045377e-10, 6.475242110814968e-11,
            6.536600682852006e-10, 0.2362930855591633, 0.09064913828525019,
            1.0983543800097608e-10, 0.0692612018477207]

    w11t = [7.016422650624392e-10, 0.020890804009275326, 2.7488261918549485e-9,
            1.3097790375913926e-9, 0.24584644185732304, 5.664549944160638e-11,
            0.033644712598011425, 2.308704070786192e-9, 1.5014551187263128e-9,
            0.026238277441305392, 1.340834677819405e-9, 6.059522736274767e-11,
            1.5565632271804896e-11, 1.4649929996718008e-10, 2.2247569014151942e-11,
            0.1988907615682869, 0.21019209022490218, 6.655062544403099e-9,
            0.26429689295885267, 2.474185749627994e-9]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 0.0001)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:WR)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :WR, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :WR, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :WR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :WR, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :WR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :WR, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :WR, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :WR, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :WR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :WR, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :WR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :WR, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :WR, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :WR, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :WR, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.05025200880802368, 0.055880855583038826, 0.05675550980104009,
           0.03045034754117226, 0.046067750416348405, 0.054698528388037695,
           0.02387947919504387, 0.03257444044988314, 0.02852023603441257,
           0.06948149861799151, 0.10142635777300404, 0.02403801691456373,
           0.0376186609725536, 0.0489137277286356, 0.023542345142850748,
           0.05182849839922806, 0.12037415047550741, 0.06259799738419287,
           0.03040731610608419, 0.050692274268387696]

    w2t = [0.05126516022271462, 0.10579893038895831, 0.04027087582723819,
           0.031064269520033407, 0.032687375431221466, 0.03843044536084045,
           0.028691511315131167, 0.026771517828023953, 0.03722072860450512,
           0.12302196901558414, 0.0451902945840916, 0.010710086504512545,
           0.03862922096045133, 0.04845258952219403, 0.02332039774037855,
           0.05322078098879961, 0.14463114007671204, 0.051446575284934126,
           0.0396834885591531, 0.0294926422645222]

    w3t = [0.06093385974605327, 0.15748529137923045, 3.9033868611415826e-11,
           0.2248210503799943, 1.1007456601591435e-11, 0.011678509116727536,
           0.039029685406592275, 0.12458303296100127, 2.9375952310712597e-12,
           9.444794201870307e-11, 0.039455404565914946, 4.016957421409421e-12,
           1.895010286612365e-11, 3.6104097383671107e-12, 0.014232403196157643,
           3.1424344753727084e-11, 0.2750260937232029, 3.070401320812885e-10,
           1.031909385607574e-11, 0.05275466900233755]

    w4t = [0.06093385987531697, 0.1574852917369013, 3.6112525068374765e-11,
           0.22482105065172406, 7.692849390247625e-12, 0.01167850886654213,
           0.03902968548952245, 0.12458303301820386, 3.531859188222606e-12,
           1.8056875470498115e-11, 0.039455403695254986, 6.701727984921763e-12,
           3.62294445493623e-12, 3.4812756620725853e-12, 0.014232402878869,
           6.007812004315745e-12, 0.2750260944019756, 2.088328772505976e-10,
           2.8312386403846495e-11, 0.0527546690633365]

    w5t = [2.9265977046358034e-9, 0.151705265581498, 0.012854658589842057,
           0.09021572899713891, 0.2603928701676096, 1.290146839107299e-15,
           0.10743003663617062, 0.006034352668086456, 3.407689750709648e-14,
           0.09540890965781988, 5.498278862079726e-7, 1.8775318951805403e-15,
           5.0529241672041286e-11, 3.642708247341742e-15, 5.813365835091464e-16,
           0.04002369930203558, 0.2359329955772524, 1.8272755177923937e-7,
           7.466332748056524e-7, 6.5666524885299e-10]

    w6t = [1.2830407464942537e-15, 8.06090741181965e-9, 1.949752767301775e-15,
           1.6878731230339432e-15, 3.379308881834173e-7, 7.62028131572693e-18,
           0.9999996207892794, 8.5148792511807e-16, 9.389673485389003e-16,
           1.347214982169776e-16, 1.8894805185257976e-17, 8.234049431384787e-18,
           2.1508087527975112e-17, 1.2135571330916248e-17, 5.113883209566018e-18,
           1.243994336023181e-8, 1.1781995430835928e-8, 9.070990979915125e-16,
           8.996977236452035e-9, 1.188714376745067e-15]

    w7t = [0.042391941653638014, 0.09072937228825505, 0.03431374347989877,
           0.045345929018649464, 0.025167250555247336, 0.03593680571351485,
           0.05530371582955835, 0.04066685426478249, 0.028663966083379743,
           0.12309845575070685, 0.04196864748602689, 0.028333590665686977,
           0.055443717439252665, 0.027631889468862886, 0.034339816262967754,
           0.06662335132291418, 0.122636399555027, 0.0376531187240201, 0.030294710346014515,
           0.03345672409159625]

    w8t = [3.947607148375036e-9, 0.3339799994613066, 2.528813690800829e-18,
           1.4565057741032627e-8, 7.131193485356854e-19, 3.9809145527709326e-11,
           0.0827704873065229, 8.07112619375709e-9, 1.0013534680357351e-20,
           5.994480031319286e-10, 1.3449370347876055e-10, 1.36928130953336e-20,
           1.2027367753542976e-10, 1.2306992720795978e-20, 4.8514738001389486e-11,
           1.9944596260891262e-10, 0.583249482088677, 1.989163045461955e-17,
           3.5175235548021734e-20, 3.417717330425231e-9]

    w9t = [1.731619126984906e-9, 0.15727194532057598, 0.0076058874268137115,
           0.05337914454004981, 0.1540701251143918, 4.337394143006759e-19,
           0.11137207916197675, 0.003570425987309125, 1.1456442877571489e-17,
           0.18889793595872603, 1.8484874597300827e-10, 6.31214649264212e-19,
           1.0004169937210818e-10, 1.224656057572098e-18, 1.954417757728137e-19,
           0.07924201434333823, 0.2445903313739081, 1.0811684953721082e-7,
           2.5101350442844104e-10, 3.885378927000986e-10]

    w10t = [0.033040054054251755, 0.2695310254752297, 0.026743949325149372,
            0.03534237610906566, 0.0196152213440997, 1.0186831723921634e-11,
            0.1642915283572523, 0.03169553011048464, 8.12523520758483e-12,
            1.5281334014373858e-10, 1.1896648606694647e-11, 8.03158529299348e-12,
            6.882734312310166e-11, 7.83267746380283e-12, 9.734140883024684e-12,
            8.270564228557204e-11, 0.36431768124263025, 0.029346640644992004,
            8.587494361073358e-12, 0.026075992968103697]

    w11t = [1.456904393125637e-9, 0.11813408494893796, 0.006399242554590085,
            0.04491074796401507, 0.12962749061302703, 2.335974247853824e-10,
            0.08365661551297342, 0.0030039915967536813, 6.170053873744455e-9,
            0.13793914848676125, 0.09955330230645995, 3.3995092835329294e-10,
            7.305356066766326e-11, 6.595584626725848e-10, 1.0525835100695455e-10,
            0.05786498368772074, 0.18372288156871844, 9.096452597860392e-8,
            0.13518741043024146, 3.268978460385678e-10]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 0.0001)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:CVaR)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :CVaR, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :CVaR, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CVaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CVaR, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CVaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CVaR, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CVaR, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CVaR, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :CVaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CVaR, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :CVaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :CVaR, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :CVaR, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :CVaR, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :CVaR, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.03557588495882196, 0.06070721124578364, 0.055279033097091264,
           0.02659326453716366, 0.053554687039556674, 0.06617470037072191,
           0.0262740407048504, 0.04720203604560211, 0.02912774720142737,
           0.06321093829691052, 0.10241165286325946, 0.03903551330484258,
           0.029865777325086773, 0.04683826432227621, 0.01880807994439806,
           0.05468860348477102, 0.07181506647247264, 0.0597967463434774,
           0.035892100831017446, 0.07714865161046885]

    w2t = [0.043702716201920504, 0.09075640411235754, 0.03712132204611378,
           0.03266813725914254, 0.03596337839669679, 0.04343584854550961,
           0.036383811860673396, 0.04605187623507636, 0.03599889412318807,
           0.09033112371096608, 0.06080737064978719, 0.023177508219709795,
           0.0366571877246488, 0.050424482248772794, 0.020248139144617018,
           0.06712466856358004, 0.09944819286261644, 0.05833969448287492,
           0.04435893818837013, 0.04700030542337814]

    w3t = [0.01589098870404606, 1.4573371074753675e-10, 2.5159857637856566e-10,
           0.015961319742193038, 0.03711957968512084, 0.02709174693694417,
           1.0332458354063548e-20, 0.167628017912894, 6.836289085949252e-11,
           8.18675802740808e-11, 0.2515537623316786, 0.024597962426664285,
           1.4822320336606394e-12, 0.1187920848310134, 8.552408573505202e-11,
           3.727650212397274e-11, 2.1472456539781323e-10, 0.24988733831157048,
           2.2605761071911202e-10, 0.09147719800524731]

    w4t = [0.006473684284949211, 6.284881667248203e-10, 2.1142975118946748e-10,
           0.015988782004223403, 0.06098814705076064, 0.024518357378874978,
           5.935562019707601e-20, 0.17428273704596545, 7.626196203543559e-11,
           2.0194137686783987e-10, 0.23011386333700626, 0.022061235750176184,
           1.0323070793548525e-19, 0.10823750709318997, 6.261038402048802e-11,
           9.299420232297465e-11, 9.260166944022488e-10, 0.26763486363461253,
           5.52133413807979e-10, 0.08970081966836528]

    w5t = [6.817876998082789e-10, 0.025987958719689157, 6.279257235145378e-10,
           3.775086203133298e-10, 0.5800469293615287, 2.0205442742765897e-18,
           0.03223815365507193, 1.587982607850203e-9, 3.9759201772656274e-17,
           0.03385838877265157, 1.6852636647880587e-17, 2.000747250916974e-18,
           9.840212869000796e-13, 2.705938021507806e-18, 2.500232217487551e-18,
           0.1691221803107261, 0.15874637011792456, 3.6057735884952043e-9,
           1.056926137719578e-8, 1.6111841502349783e-9]

    w6t = [5.767379032403817e-14, 9.335148963436661e-8, 8.857758852292376e-14,
           7.652638895531039e-14, 4.009755366634321e-6, 1.2758889926635245e-15,
           0.9999955133164664, 3.804569360303192e-14, 1.6023033158076792e-13,
           2.253799229231184e-14, 3.199577709159482e-15, 1.3927033933571582e-15,
           3.687636106018562e-15, 2.0412311180621076e-15, 8.711576873116786e-16,
           1.4285692793934333e-7, 1.3852327695449586e-7, 4.053864186458452e-14,
           1.0219592242762861e-7, 5.3344869267818054e-14]

    w7t = [0.03697381930526495, 0.08435057027197133, 0.03035794987106501,
           0.03130224438246418, 0.032937215456165965, 0.040714836589162993,
           0.0349410177489328, 0.05965947713071079, 0.030709819566359698,
           0.09400317822605378, 0.06843295270512015, 0.02658370738605732,
           0.043454631180914416, 0.05245242724951428, 0.02545531322296032,
           0.07013506828071824, 0.09042868191952276, 0.06543075168097323,
           0.037099985443304534, 0.044576352382763224]

    w8t = [0.0032672308172378406, 0.3562576470635151, 5.172935665783453e-11,
           0.0032816910713807206, 0.007631887287117852, 7.671218293905125e-12,
           2.5258516253506276e-11, 0.034464779766538854, 1.935743236514712e-20,
           8.711888146820745e-10, 7.122921338335441e-11, 6.965085706706446e-12,
           1.577308092123298e-11, 3.363681258531599e-11, 2.4216745143354848e-20,
           3.9667560213892343e-10, 0.5249112785433813, 0.051377521422642805,
           6.400979910454376e-20, 0.018807962548058058]

    w9t = [1.215332431712266e-10, 0.05412551909851164, 1.1193198362308448e-10,
           6.729345068072619e-11, 0.10339726653420324, 5.006776695635197e-11,
           0.06714289568408317, 2.830685805651928e-10, 9.852070425116713e-10,
           0.030494054376838442, 4.1759732515054936e-10, 4.9577209652254495e-11,
           8.862441397369431e-13, 6.705137619797547e-11, 6.195412077607063e-11,
           0.1523173768649735, 0.3306234929920612, 6.427533944572532e-10,
           0.2618993913032025, 2.872044115480477e-10]

    w10t = [0.04371410956734524, 0.25896464109514067, 0.03589189627327679,
            0.03700787814916288, 0.03894116543935061, 4.7092657965854015e-11,
            0.10727182600750135, 0.07053103866709053, 3.5521025142336006e-11,
            9.430357763951511e-11, 7.915474179678662e-11, 3.074823455609446e-11,
            4.359333047510605e-11, 6.066919554333679e-11, 2.944260541078269e-11,
            7.035877241233006e-11, 0.2776211774951054, 0.07735584569477834,
            4.2912289557645716e-11, 0.052700421077451605]

    w11t = [2.775375209258045e-10, 0.03109311399304554, 2.556119237765966e-10,
            1.536747107170206e-10, 0.2361074318993314, 5.10972495334069e-11,
            0.03857111659477083, 6.464220044671037e-10, 1.0055115829823662e-9,
            0.03952665220105085, 4.2620311819380324e-10, 5.05965835573382e-11,
            1.1489320177133444e-12, 6.843090084259769e-11, 6.322856977141958e-11,
            0.19743507719879042, 0.1899309996607892, 1.467803265213944e-9,
            0.2673356033290889, 6.558666385955365e-10]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-5)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:EVaR)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :EVaR, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :EVaR, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EVaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EVaR, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EVaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EVaR, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EVaR, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EVaR, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :EVaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EVaR, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :EVaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :EVaR, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :EVaR, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :EVaR, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :EVaR, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.04298218516072598, 0.05890619211025965, 0.05740162248050834,
           0.030204535049956783, 0.049834984578469854, 0.060682764810121595,
           0.02422026932923119, 0.03449219730798083, 0.028790718728406155,
           0.06651913049525503, 0.11020943123173292, 0.026320956237946017,
           0.03505101302056121, 0.04934069246466456, 0.021946073181284137,
           0.050482012553386094, 0.09622983487219411, 0.06481147607895538,
           0.03384172561560555, 0.05773218469275462]

    w2t = [0.046908260068124694, 0.1002465013186037, 0.03972101611141057,
           0.03296347498532413, 0.03448502219646262, 0.041076975666174534,
           0.03192242102026614, 0.02986356537151306, 0.03748145005038874,
           0.11036260986706951, 0.05255752133505542, 0.012552140080699294,
           0.039169721277886496, 0.05015896331484878, 0.022310028996722173,
           0.05641395756815819, 0.12683134368755505, 0.05611419114375472,
           0.04405714773034778, 0.03480368820963433]

    w3t = [2.2855341178116485e-8, 0.07165106367040126, 8.289105826803737e-9,
           0.08189089596433913, 8.144349496256881e-9, 0.06609004845161012,
           0.015841375805598638, 0.09136511251611182, 1.3448947389500138e-8,
           9.829464392966764e-9, 0.20811399783854218, 0.018316114383509285,
           1.537075625908141e-9, 0.042044829381784464, 0.021986711582760155,
           2.9324550647413686e-9, 0.11534269730094221, 0.14375809847394522,
           0.012367556911982328, 0.11123143068173415]

    w4t = [4.964849718064805e-8, 0.07790393153579288, 1.8554071561495594e-8,
           0.08798525778535686, 2.1990591027929757e-8, 0.052497070196180315,
           0.017523911232090336, 0.09465506351433912, 1.4257989867976142e-8,
           1.4561640630583332e-8, 0.1886597013449402, 0.01586190192306709,
           2.2227140811850335e-9, 0.03648727355398573, 0.016396822749053682,
           4.35659442221111e-9, 0.1258507111236745, 0.1476711543997907,
           0.022468954319380827, 0.11603812073024912]

    w5t = [2.8881703579609315e-8, 0.05683121897007873, 0.10307670471279688,
           0.06288215093518605, 0.46235519200277597, 1.9548955183964208e-16,
           0.05666483073417378, 1.0699821156545776e-7, 3.5077610890655783e-15,
           0.025530468301985427, 9.33355941339848e-15, 2.452111185248173e-16,
           8.492104458784185e-11, 4.391406143559331e-16, 1.2360096158026862e-16,
           0.02032324320769304, 0.20629032469632985, 0.0060455961263924975,
           1.1347449814266133e-7, 2.0873239406307323e-8]

    w6t = [9.446402155866373e-16, 8.622907214779223e-9, 1.3542329098125956e-15,
           1.201237162219277e-15, 2.391479350702934e-7, 1.2578298925328544e-17,
           0.999999725782182, 6.535494692553824e-16, 1.2177974527781532e-15,
           9.926093984448799e-17, 3.1038459423341675e-17, 1.392838196329113e-17,
           1.688042828579514e-17, 2.0131132586026224e-17, 8.573779963075646e-18,
           8.118648640315347e-9, 1.270359107840406e-8, 6.92645246832493e-16,
           5.624728956796762e-9, 8.837228194958539e-16]

    w7t = [0.035355462443285426, 0.10150333878252887, 0.03195963292100241,
           0.03896112396916452, 0.030854396335923243, 0.03937296341435327,
           0.04523458443349084, 0.0487268198304694, 0.029658128727246373,
           0.10954989502529314, 0.05603675121339598, 0.025344951815022564,
           0.05158250282120999, 0.035978272802500835, 0.02868885645546524,
           0.06366656715311136, 0.11102339122919934, 0.04446723582974776,
           0.03187702755056481, 0.0401580972470246]

    w8t = [2.3708662412453847e-15, 0.3532477690612223, 8.598585784270034e-16,
           8.494835373233737e-9, 8.448424867998652e-16, 8.429435366613586e-10,
           0.07809975701030375, 9.477629723569365e-9, 1.7153419527568222e-16,
           5.72367958187117e-9, 2.6543837306338238e-9, 2.3361233041966623e-10,
           8.950363950753286e-10, 5.362595127064301e-10, 2.8042885207891806e-10,
           1.707563353179708e-9, 0.5686524164733802, 1.4912541445841342e-8,
           1.57741632931017e-10, 1.1538433922887621e-8]

    w9t = [5.2408012668434776e-9, 0.10113701502017637, 0.023896961899729303,
           0.01457738945913547, 0.1071930592259488, 1.1220347220945611e-16,
           0.1008432629865652, 1.9005351409798715e-8, 2.0383222033223346e-15,
           0.15802952555394023, 5.227132461350302e-15, 1.406047771117507e-16,
           4.974641316439625e-10, 2.5170508286075082e-16, 7.092643131954439e-17,
           0.12579638252982453, 0.36712450173820727, 0.0014018052502646936,
           6.786538633649107e-8, 3.727197192423296e-9]

    w10t = [0.023970295969538275, 0.321574487102832, 0.02166799151398791,
            0.026414862324169935, 0.020918663228345846, 4.999557183123031e-10,
            0.14330847106098665, 0.033035808677777306, 3.76597283155573e-10,
            2.1825642912702846e-9, 7.11551576901832e-10, 3.2182880056346033e-10,
            1.0276790195547736e-9, 4.568501241146866e-10, 3.642895172965806e-10,
            1.2684300243652437e-9, 0.35173513028412856, 0.030147895972118477,
            4.0477273805846927e-10, 0.027226386251595854]

    w11t = [1.1165472417376397e-8, 0.04948968670520029, 0.03984867332323181,
            0.024309788970545903, 0.1787429987508185, 3.7921048838824565e-10,
            0.04934476220395251, 4.1365488614060555e-8, 6.804350799862205e-9,
            0.14262867361067905, 1.8105235842660336e-8, 4.756603383037981e-10,
            4.74420522909693e-10, 8.518446671715595e-10, 2.39761046717305e-10,
            0.11353796493293959, 0.17964129063607762, 0.0023372667005997013,
            0.22011880623510086, 8.069409441291992e-9]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-6)
    @test isapprox(w3.weights, w3t, rtol = 0.0001)
    @test isapprox(w4.weights, w4t, rtol = 0.0001)
    @test isapprox(w5.weights, w5t, rtol = 5.0e-6)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-5)
    @test isapprox(w9.weights, w9t, rtol = 1.0e-5)
    @test isapprox(w10.weights, w10t, rtol = 5.0e-5)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-5)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:RVaR)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.65))))
    asset_statistics!(portfolio; calc_kurt = false,
                      cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :RVaR, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :RVaR, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RVaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RVaR, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RVaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RVaR, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RVaR, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RVaR, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :RVaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RVaR, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :RVaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :RVaR, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :RVaR, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :RVaR, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :RVaR, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.04746219025128368, 0.057720507695227345, 0.057894030850523526,
           0.030899830154135045, 0.0477807002232772, 0.05649620052589632,
           0.02359615492299298, 0.03264037279266019, 0.028784213764015288,
           0.06836029502022546, 0.10450060319650663, 0.024571310001886896,
           0.036868812002078415, 0.04899693792466318, 0.02279653723295508,
           0.051342217890359566, 0.11116746674144795, 0.06399587204503096,
           0.03166573241138531, 0.052460014353449096]

    w2t = [0.049248271875817956, 0.10352971694202272, 0.040651501329566395,
           0.032062643728263764, 0.033550215968243445, 0.03944969833848993,
           0.0295635855866539, 0.027229931898476747, 0.03757321453987666,
           0.11845903122489482, 0.04742391088929517, 0.011150821593606197,
           0.03875514509177754, 0.049328246840721166, 0.022950680620361073,
           0.05396905875648936, 0.13928153936224671, 0.053387971444982175,
           0.04133457796666694, 0.03110023600154728]

    w3t = [0.011463171964622997, 0.1365816315947654, 2.8511129084145128e-9,
           0.17313840216033252, 4.3542516777215384e-9, 0.04078245658972637,
           0.03735692169672586, 0.09992387594976379, 6.749240140496854e-10,
           4.5560848107050465e-9, 0.11003351446302158, 0.00791074813494162,
           8.08045614278609e-10, 2.95999048236193e-9, 0.030995016529374333,
           1.3978544353146189e-9, 0.20068563074785928, 0.06563745066221992,
           3.1377525472677185e-9, 0.08549115876662991]

    w4t = [0.014254075507599462, 0.13964552388154705, 5.2075174329334345e-9,
           0.1785941865911206, 1.4870340646361e-8, 0.036256886806998664,
           0.03844859655499469, 0.10202225707438087, 1.4553765429694116e-9,
           6.385952224403694e-9, 0.0989865146231353, 0.007464906900184655,
           1.1235783286137348e-9, 5.047281304002713e-9, 0.02717489986234817,
           1.959651451837051e-9, 0.20554547133036372, 0.06415557347784641,
           8.448921883472202e-9, 0.08745106289086073]

    w5t = [1.8409222384647946e-8, 0.08300872761568451, 0.036742854379938655,
           0.11666699599335352, 0.3621864767977365, 2.544328424403338e-16,
           0.0815606837065607, 0.006099559594295306, 3.719308519065499e-15,
           0.049147214837141066, 2.3456123618539025e-8, 3.4372950924603106e-16,
           1.567640004611372e-10, 6.142976882927018e-16, 1.5877586333568593e-16,
           0.024623386412687444, 0.23917366659547037, 0.0007902904154487118,
           9.585639801954156e-8, 5.77317000005512e-9]

    w6t = [4.4719721850382937e-13, 1.8554318200856632e-7, 6.36294092891581e-13,
           5.662772429733862e-13, 4.9294842961383685e-6, 7.031792582667903e-15,
           0.999994309419098, 3.092313371280024e-13, 6.205237967697061e-13,
           5.2565420115495574e-14, 1.7381289232443002e-14, 7.788880844387662e-15,
           9.097886673087828e-15, 1.1275800829257585e-14, 4.775483042725156e-15,
           1.7974732731622715e-7, 2.7127299295497736e-7, 3.2791753051954224e-13,
           1.2452966768647857e-7, 4.1859052423044934e-13]

    w7t = [0.03630230780927064, 0.10648017175438054, 0.03237013061067967,
           0.042615926598813664, 0.028639930564047265, 0.03814861275783586,
           0.050833111117397495, 0.042793280015910554, 0.028912699151375794,
           0.11537856464139956, 0.047348711827175455, 0.0260353783439049,
           0.05308062219710469, 0.03039506214564768, 0.03050029453396626,
           0.06294924319701606, 0.12213472748605388, 0.038517821798921686,
           0.030327995634144653, 0.03623540781495376]

    w8t = [2.7249067637153436e-9, 0.3645830197630867, 6.777370933831525e-16,
           4.1156671518281064e-8, 1.0350477061810805e-15, 6.281216003699347e-10,
           0.09971838205634287, 2.3752871044111503e-8, 1.0395017546338374e-17,
           3.73951585079542e-9, 1.694709759496251e-9, 1.2183944259735302e-10,
           6.632228126352585e-10, 4.558906241389382e-17, 4.773777995224444e-10,
           1.1473225444973184e-9, 0.5356984861492785, 1.5602656386425192e-8,
           4.8326911038780615e-17, 2.0322074682294418e-8]

    w9t = [4.3811387531111694e-9, 0.10908605145235488, 0.011094588682482556,
           0.035225847641813954, 0.10935866170316816, 7.054962474817408e-17,
           0.10718294835846438, 0.0018423308656722395, 1.0312948538757793e-15,
           0.20763231820323327, 6.50226257540096e-9, 9.530995526851443e-17,
           7.693679499736866e-10, 1.703336298459508e-16, 4.40256621750654e-17,
           0.1040298296728622, 0.3143098178199953, 0.00023756600001164434,
           2.6572312405461958e-8, 1.3748584358127235e-9]

    w10t = [0.025502397836221276, 0.3121168533821771, 0.02274003918082762,
            0.029937719314785435, 0.020119574862906196, 5.507075326127284e-10,
            0.14900306116495682, 0.030062312144136332, 4.173793160878481e-10,
            2.5218218007583985e-9, 6.835187944160563e-10, 3.7584276643850806e-10,
            1.16017967733275e-9, 4.387784959205269e-10, 4.402976307095178e-10,
            1.3758776200553119e-9, 0.3580038269875965, 0.02705880215846411,
            4.378103363185387e-10, 0.025455404565714684]

    w11t = [6.505918974948508e-9, 0.06153751547889246, 0.016475281678160692,
            0.05230980425323847, 0.16239581358713545, 4.726276411878093e-10,
            0.06046393884339232, 0.0027358310276292207, 6.908873801898587e-9,
            0.16311937843698385, 0.04356010445647429, 6.385026072224354e-10,
            6.044281683846747e-10, 1.1411029041813966e-9, 2.949377114308613e-10,
            0.08172755234824679, 0.17730814363295289, 0.00035278160185655183,
            0.17801383604700308, 2.0416421596029406e-9]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-7)
    @test isapprox(w3.weights, w3t, rtol = 0.01)
    @test isapprox(w4.weights, w4t, rtol = 0.0001)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-5)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-6)
    @test isapprox(w8.weights, w8t, rtol = 0.01)
    @test isapprox(w9.weights, w9t, rtol = 5.0e-5)
    @test isapprox(w10.weights, w10t, rtol = 1.0e-6)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-5)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:MDD)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :MDD, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :MDD, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MDD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MDD, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MDD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MDD, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MDD, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MDD, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :MDD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :MDD, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :MDD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :MDD, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :MDD, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :MDD, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :MDD, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.06921730086136976, 0.04450738292959614, 0.08140658224086754,
           0.018672719501981218, 0.053316921973365496, 0.02979463558079051,
           0.021114260172381653, 0.029076952717003283, 0.026064071405076886,
           0.07316497416971823, 0.11123520904148268, 0.018576814647316645,
           0.013991436918012332, 0.05469682002741476, 0.012327710930761682,
           0.08330184420498327, 0.07332958900751246, 0.060105980885656114,
           0.049926342635393445, 0.07617245014931591]

    w2t = [0.07530191325935566, 0.0813493515582833, 0.061213037362134175,
           0.020314162597449822, 0.040091238901701284, 0.019672150628411215,
           0.03483719019119278, 0.03195414054455141, 0.035669561097788856,
           0.05794589514287386, 0.07810473051982168, 0.01304386546713144,
           0.007943519082449032, 0.05513728656465975, 0.012426984419478355,
           0.047293912192364396, 0.12098917120658725, 0.06605351597470699,
           0.06832588436952429, 0.07233248891953448]

    w3t = [0.09288698654422871, 6.356361148712279e-10, 3.329053959411766e-10,
           0.01462746465666353, 0.0067642764690766495, 4.426470737527751e-12,
           2.4244819110883616e-10, 0.01980251194994183, 1.3597980185918856e-10,
           9.955474945766373e-11, 0.44521949759838164, 0.05359346334295806,
           4.44567971249619e-21, 3.508212235300306e-11, 2.6011962943743497e-11,
           1.9366245915903663e-11, 9.603511034839303e-10, 0.017785611940728262,
           0.29769706785169064, 0.051623117154568604]

    w4t = [0.09288698709945305, 7.943709984059515e-10, 3.1943399793966327e-10,
           0.014627465356902498, 0.006764276458984734, 1.5766729819361093e-11,
           3.0299381866243656e-10, 0.019802512535639718, 1.1117205460105161e-10,
           8.298042124631033e-11, 0.445219505120722, 0.05359346089998252,
           1.4419900892631536e-20, 3.314229495034821e-11, 4.213228033916901e-11,
           1.6142048926524844e-11, 1.200175739248774e-9, 0.017785611555378274,
           0.2976970604575057, 0.051623117597121106]

    w5t = [6.075357147348992e-10, 2.0421628226787587e-9, 0.5264670554539496,
           2.30450167051231e-10, 0.32265003550898885, 6.708256697384932e-21,
           7.789378073888229e-10, 0.05614503578659163, 1.7583672907404588e-18,
           8.813796040075635e-9, 3.8184665856591184e-10, 3.1294797081461483e-19,
           1.0632091342832445e-10, 6.046055420140702e-19, 3.6160677173813724e-19,
           0.09473785544771883, 3.0854174337530415e-9, 3.3790792639567355e-10,
           7.008607649974044e-10, 7.175148096836271e-10]

    w6t = [1.303643222552151e-13, 4.4436455541826195e-8, 9.884158797244972e-12,
           1.5274111568119e-13, 8.070660504600166e-5, 4.8740254701184495e-17,
           0.9999134216931299, 1.134676029888114e-12, 5.557839575816408e-14,
           2.174910004517825e-15, 4.610439075538003e-15, 5.1153753100600704e-17,
           5.644086385354662e-16, 7.93697374292659e-17, 3.0527548159326436e-17,
           1.5270700152138014e-7, 5.65084397488438e-6, 9.981949051075668e-14,
           2.3702764831928958e-8, 1.6253829716862961e-13]

    w7t = [0.08062857310291681, 0.0792078023198039, 0.04744817612449485,
           0.04061619103951026, 0.04947006766043654, 0.028425939325242434,
           0.03263127775544647, 0.052404318652522125, 0.04912285955085263,
           0.06158168534515238, 0.07886878578979677, 0.022118369972757297,
           0.02100437959787758, 0.040083161546029936, 0.013535829155674497,
           0.05163022803788763, 0.08374695544064996, 0.06281563757705252,
           0.05650909535720665, 0.04815066664868867]

    w8t = [0.25291692851164765, 0.07090395567913133, 9.064500137089194e-10,
           0.0398283287090433, 0.01841808085083987, 1.3385077370027142e-12,
           0.027044617816199875, 0.0539191837901826, 4.111854062911421e-11,
           4.911380010318118e-10, 0.1346286415377346, 0.016205973017995354,
           2.1932075155807414e-20, 1.0608381933232215e-11, 7.865682553743593e-12,
           9.55403871582464e-11, 0.10712527259660852, 0.048427476346293054,
           0.09001975890730979, 0.1405617806829547]

    w9t = [4.894294728515436e-11, 1.94997545168874e-11, 0.042412073426327815,
           1.8565016186357073e-11, 0.025992617876564327, 5.313279820654926e-12,
           7.437749752040159e-12, 0.004523032078904726, 1.3927161503573647e-9,
           6.4670416713044215e-9, 0.3024419364175013, 2.4787067836751064e-10,
           7.801199103606937e-11, 4.788782795225758e-10, 2.864109186575106e-10,
           0.06951302892018757, 2.946135434068335e-11, 2.7221790304189428e-11,
           0.5551173021153398, 5.780283966019497e-11]

    w10t = [0.2113274992541227, 1.2558655928383342e-7, 0.12436167500768941,
            0.10645504132451156, 0.12966105295287814, 1.1790225394636564e-11,
            5.173795735167115e-8, 0.13735172513617322, 2.037468593410109e-11,
            5.801983826866035e-11, 3.2712402233169917e-11, 9.17403517811617e-12,
            1.9789499107924922e-11, 1.6625290856709145e-11, 5.614255164014158e-12,
            4.864396717534949e-11, 1.327835349076225e-7, 0.16463979321905658,
            2.343827458031749e-11, 0.1262029027513344]

    w11t = [2.197525091828318e-10, 0.0711764706485091, 0.19042906224629996,
            8.335642040930649e-11, 0.11670615104820459, 2.0259848486504805e-12,
            0.027148689305733126, 0.02030829166586057, 5.310508601722855e-10,
            1.299691460731517e-8, 0.1153228893378406, 9.451454765192339e-11,
            1.567819162726874e-10, 1.8259910477309654e-10, 1.0921016797053559e-10,
            0.13970141942649306, 0.10753751903281374, 1.222251019935822e-10,
            0.21166949253028083, 2.595331862467074e-10]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-4)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:ADD)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :ADD, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :ADD, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :ADD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :ADD, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :ADD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :ADD, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :ADD, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :ADD, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :ADD, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :ADD, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :ADD, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :ADD, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :ADD, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :ADD, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :ADD, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.05548273429645013, 0.03986073873648723, 0.09599318831154405,
           0.014621119350699974, 0.08212398553813219, 0.024721555415213262,
           0.017757328803291575, 0.024648542319161523, 0.04053046304894661,
           0.06980177121920025, 0.07760681699761883, 0.01027851528431043,
           0.00777588366272761, 0.021591614519396285, 0.0034064806929941845,
           0.06991915064946408, 0.12960468534402678, 0.062161540752826025,
           0.08184195485513887, 0.07027193020237012]

    w2t = [0.06987647219041285, 0.07734286470130464, 0.09294315975375894,
           0.018414237377757756, 0.07951462850378198, 0.018615135796153055,
           0.028723365185807745, 0.025057211716100154, 0.031861833538247786,
           0.03614810576005913, 0.051423236032185425, 0.006810671252523318,
           0.003708260921662954, 0.02889368422079079, 0.004558518648949396,
           0.03334392144677785, 0.20964204403525669, 0.06319217043645177,
           0.06433764990274386, 0.05559282857927396]

    w3t = [2.801843497048092e-9, 0.01823133762490545, 0.126976572838495,
           0.016600200784007566, 0.07054326311130743, 0.0031156156654178445,
           0.009811613736194736, 0.08498448541583929, 8.179724214458373e-10,
           7.555886400615262e-9, 0.11291841839917299, 0.0219129156374651,
           6.3628124011897865e-18, 0.037985918533416804, 7.766238508143918e-11,
           3.445079918796638e-9, 0.11004124140367169, 0.05474954341410615,
           0.19646342841470762, 0.13566543032284778]

    w4t = [1.201264431708132e-9, 0.017693650764110894, 0.13050240482878742,
           0.01650836205303405, 0.07235161625308452, 0.0031131921822646832,
           0.009522240486941303, 0.08465340852262496, 1.037581899542216e-9,
           9.016757277436049e-9, 0.11283114425024635, 0.02189594106371991,
           4.293173690153564e-18, 0.03795643812769797, 1.7856706247898235e-10,
           4.127374260779851e-9, 0.10679570233593967, 0.05447571070411567,
           0.19631149766912737, 0.1353886751967602]

    w5t = [1.2379500077203795e-9, 0.016146438150965198, 0.23426185629539284,
           3.304913016667393e-9, 0.2267610282669818, 1.5907580566772538e-10,
           0.00982658337545211, 0.06079707628169535, 5.005769435982415e-10,
           6.943504632506545e-9, 0.032915088420824565, 2.1050181953817916e-10,
           7.302599386682247e-11, 3.4545967712409064e-10, 1.3242173809826203e-11,
           0.13238294813391252, 0.12249573926535208, 0.00139872734844176,
           0.1111632340167779, 0.05185126765595369]

    w6t = [3.140925812338669e-12, 1.9014616189996336e-7, 1.3229034007720623e-10,
           5.333671168236288e-12, 0.0003252022529196653, 1.8712483536623898e-15,
           0.9996414274026609, 1.9627891734042294e-11, 4.747729019514711e-13,
           1.6138330482272631e-13, 4.863722029546999e-14, 1.8893001368822384e-15,
           3.654983967925628e-14, 2.715429511329365e-15, 1.1577772603803039e-15,
           3.9025606806697874e-7, 3.2693620087778935e-5, 2.3594456864417395e-12,
           9.61538988744323e-8, 4.721496400049383e-12]

    w7t = [0.04489707360283697, 0.07428849410628964, 0.061466374560412644,
           0.029093396292183504, 0.047101115479903415, 0.020649948367959272,
           0.028388091658424618, 0.05525679031619594, 0.04377930143622314,
           0.0741825524956059, 0.05706468520737268, 0.017619608998462304,
           0.014862427309459358, 0.031398210757448336, 0.008229609065734611,
           0.06505318351772757, 0.12743516216930809, 0.04984427029980972,
           0.07602590611192094, 0.07336379824672123]

    w8t = [3.5142773852015812e-9, 0.0509649627239819, 0.1592633203270829,
           0.020821188002294055, 0.08848052879883514, 1.606307472872854e-10,
           0.027427967086966102, 0.10659376781912788, 4.217192857759427e-17,
           1.2348075922242872e-8, 5.821696857954702e-9, 1.1297568095958326e-9,
           1.0398315491149029e-17, 1.9584272053011286e-9, 4.0040134250870274e-18,
           5.630061933174542e-9, 0.30761581413409356, 0.06867088846076362,
           1.0128998795056528e-8, 0.1701615219549293]

    w9t = [1.0701977866298008e-9, 0.007401447368844417, 0.20251748336815384,
           2.85707061947881e-9, 0.19603307809828496, 3.548942312551583e-10,
           0.004504457205295313, 0.052558581577980795, 1.116774916440974e-9,
           5.945963058803583e-9, 0.07343275712338014, 4.69624410256695e-10,
           6.253468310970183e-11, 7.707120892959512e-10, 2.95429658498907e-11,
           0.11336409506154566, 0.056151440869080756, 0.001209188499588471,
           0.24800245590222925, 0.04482500224830162]

    w10t = [0.0765644383415946, 0.11359085744814228, 0.10482060560879627,
            0.049613913954955575, 0.08032306256489549, 7.71384637222446e-11,
            0.04340682512938291, 0.09423119984487083, 1.6353881353346203e-10,
            0.015637530558470936, 2.1316673878566415e-10, 6.581854565975056e-11,
            0.0031329693223819746, 1.172889005967124e-10, 3.07419364471835e-11,
            0.013713078223407579, 0.19485479567190744, 0.08500105360564994,
            2.839969135063144e-10, 0.12510966877385393]

    w11t = [7.213120450006277e-10, 0.02504709201981727, 0.13649654475242767,
            1.925662225239071e-9, 0.13212606325428203, 2.9548314399490514e-10,
            0.015243444761261027, 0.035424421947024394, 9.298211534111268e-10,
            8.758602871392893e-9, 0.06113974259398518, 3.910069114074518e-10,
            9.211568414225699e-11, 6.416909918615075e-10, 2.459732411368391e-11,
            0.16698911155339388, 0.19002098324895322, 0.0008149916214796937,
            0.2064855918600773, 0.030211998607005745]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0005)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 0.0001)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:CDaR)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :CDaR, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :CDaR, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :CDaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :CDaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :CDaR, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :CDaR, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.06282520890507177, 0.03706824571002148, 0.08139181669684024,
           0.015787409071561908, 0.05742532060680956, 0.022473205432788673,
           0.018010510386343107, 0.028690055814249292, 0.031947471924376705,
           0.07945270069786842, 0.10413378630449394, 0.015162575734861551,
           0.011642750066910313, 0.04691738037936903, 0.008558622422573802,
           0.08506702725550051, 0.07867872568683776, 0.06853164241553955,
           0.06271444623431467, 0.08352109825366769]

    w2t = [0.0747775890739847, 0.06939399743726035, 0.07245669105243545,
           0.018790934541577802, 0.05112121688219039, 0.01640191163559003,
           0.031440878203812674, 0.02765262927324115, 0.038622757716471855,
           0.05449451662150961, 0.07421075864027389, 0.010805582781120135,
           0.006125600333633228, 0.05505443171528247, 0.010042975330900426,
           0.04475631681027434, 0.13734914660868333, 0.066053552264685, 0.0758183579584723,
           0.06463015511860094]

    w3t = [1.8244116660407682e-10, 3.5021448909430726e-19, 0.013424622665528558,
           6.483995460755705e-13, 0.08103692763402937, 2.6918169953989202e-11,
           3.859278466294731e-12, 0.11897175620642726, 1.4912744419458638e-10,
           0.0169666215154208, 0.28463551994513464, 0.04146959826942596,
           8.043265317900752e-13, 0.04309141771999534, 2.1399761108135705e-11,
           0.004330236064625966, 9.238315973738486e-11, 0.027869536927597218,
           0.2518844303431581, 0.11631933223107507]

    w4t = [1.8997790704450745e-10, 6.926441204184441e-19, 0.013424623049246836,
           8.318856070041918e-12, 0.08103693856302899, 2.1527796473645622e-11,
           1.5561397972701852e-11, 0.1189717701992731, 1.5832503205707997e-10,
           0.016966572426670986, 0.2846354733594883, 0.04146959768365048,
           1.84563310473958e-12, 0.04309146117360012, 3.50557492120482e-11,
           0.004330225756471268, 3.725077241024316e-10, 0.02786953543219608,
           0.251884460436475, 0.11631934111677891]

    w5t = [1.2318242911290414e-10, 1.0021712244268165e-16, 0.13990025540077664,
           7.482894698992554e-11, 0.3375646803292957, 3.4389601398763396e-20,
           1.4489493723716944e-10, 0.21161159366592713, 5.299670445490732e-19,
           6.387832190124329e-9, 1.49983853164512e-10, 1.0293053431351615e-19,
           1.2329730946776807e-10, 1.6542158258566509e-19, 1.3816885264090829e-19,
           0.2457086313190956, 3.4684740873690043e-9, 3.6469703930570965e-11,
           3.482961441814694e-10, 0.06521482842764527]

    w6t = [6.81843581833036e-14, 3.124173469115393e-8, 3.881371233912884e-12,
           8.723867059823497e-14, 8.604095252688581e-5, 1.911422294729585e-17,
           0.9999078440992752, 8.146100492007672e-13, 5.92518622106533e-15,
           1.5660614920257613e-15, 7.190845803970576e-16, 2.01794943440324e-17,
           3.657325199043091e-16, 2.866306764840243e-17, 1.1881364394548097e-17,
           1.1302670231474434e-7, 5.952036990258397e-6, 5.1027485784203644e-14,
           1.863773469685073e-8, 1.247329045638216e-13]

    w7t = [0.05351961928914926, 0.07089486003848636, 0.051073393011123076,
           0.02618898067675608, 0.050426728384627285, 0.023051735365527026,
           0.026994539271527403, 0.06219336514775715, 0.045211702680940005,
           0.0715611711245754, 0.07048168280367938, 0.016138424448672133,
           0.01867133570753628, 0.03961425273699505, 0.010688467367280088,
           0.055962903217657026, 0.09362014508063544, 0.05338784948755773,
           0.060944545440328554, 0.09937429871918933]

    w8t = [4.202952793522189e-10, 6.409333583198476e-10, 0.030926712640744383,
           1.4937378083154294e-12, 0.18668724154622254, 1.661000714345463e-20,
           0.007062929676298079, 0.2740788630140619, 9.201992251297623e-20,
           1.1153537311952114e-10, 1.7563593764549218e-10, 2.558904727434039e-11,
           5.2874910748494665e-21, 2.6589800026306686e-11, 1.320484213755697e-20,
           2.846615601843261e-11, 0.16907195637681427, 0.06420390214792512,
           1.5542669484872716e-10, 0.26796839301196834]

    w9t = [6.318860720669092e-11, 1.7645441075546338e-18, 0.07176431208815445,
           3.838483274836946e-11, 0.17315977729769882, 3.2485428861552183e-11,
           2.5511958583981114e-12, 0.10854991226291229, 5.00622456336032e-10,
           3.7016136058827227e-9, 0.1416791586462658, 9.723120984599493e-11,
           7.144818221746991e-11, 1.5626209187307524e-10, 1.3051836167865303e-10,
           0.14238295335618134, 6.107015811168749e-11, 1.8707780104749966e-11,
           0.32901078100215786, 0.03345310047254568]

    w10t = [0.10327898804564337, 0.08718180155230068, 0.09855840561473608,
            0.05053794212603172, 0.0973105105602408, 1.313111619019184e-11,
            0.033196095802833836, 0.12001706852417422, 2.575424850432982e-11,
            6.644558223016583e-11, 4.0148958484028675e-11, 9.193040055416301e-12,
            1.7336605209302945e-11, 2.2565735170420845e-11, 6.088544080057372e-12,
            5.196236491315115e-11, 0.11512785137436911, 0.10302470649536136,
            3.471625431429138e-11, 0.1917666296169666]

    w11t = [6.148656367193241e-11, 5.274418729977745e-9, 0.06983127401664745,
            3.73509018025829e-11, 0.16849555865990354, 1.9405786633200198e-11,
            0.007625808367026054, 0.1056260200530183, 2.9905631268866156e-10,
            3.95548810527667e-9, 0.08463473069043505, 5.8082906043837185e-11,
            7.634844286710683e-11, 9.334612224668457e-11, 7.796761708904138e-11,
            0.15214826244951452, 0.18254549966074657, 1.820386875768039e-11,
            0.19654082583798052, 0.03255201029357281]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-4)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 0.0001)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:UCI)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :UCI, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :UCI, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :UCI, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :UCI, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :UCI, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :UCI, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :UCI, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :UCI, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :UCI, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :UCI, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :UCI, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :UCI, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :UCI, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :UCI, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :UCI, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.056222127504695026, 0.03676156065449049, 0.09101083070457834,
           0.014331694009042086, 0.06981598298437515, 0.02449761185808867,
           0.017540488246254482, 0.025783651111609372, 0.03897771757340407,
           0.07792753098849892, 0.09069642716675666, 0.011419250355786286,
           0.009579825814905836, 0.03155851948684042, 0.005020862393829734,
           0.07891456352044066, 0.10481408027151863, 0.06296146942913426,
           0.0767246147903293, 0.0754411911354216]

    w2t = [0.07291516388896417, 0.07339755559165628, 0.08751602925154806,
           0.018586949015554837, 0.0671350603195713, 0.017201289118820004,
           0.03039926629036034, 0.02677983297045136, 0.03471017360643417,
           0.0417825188626436, 0.06116396173886505, 0.0077009272985324164,
           0.004460209471197638, 0.03885125353264208, 0.006181113721020028,
           0.03674132394783595, 0.1816523629456826, 0.06539406027438881,
           0.06832428538802553, 0.059106662765805854]

    w3t = [2.3703125886200163e-9, 0.005413668528957127, 0.11234002020637264,
           5.325868089643136e-10, 0.08167647681635509, 0.0024797209941318714,
           0.002950498790326065, 0.0897671648971775, 1.0247038041716701e-9,
           1.4856579227583637e-9, 0.16362974758546406, 0.030901788450388695,
           9.72514770061732e-19, 0.060771535869062115, 1.5016182058251723e-10,
           6.031391294265412e-10, 0.03283161990128136, 0.06371772032022284,
           0.21177383294752763, 0.14174619852617099]

    w4t = [4.25914824461442e-9, 0.005872404987961267, 0.11257246089846742,
           7.463312499294076e-10, 0.08317300572200355, 0.001369805803798963,
           0.003200513014189802, 0.09100305773356251, 6.721262627730614e-10,
           1.9284208597919347e-9, 0.1619210446056535, 0.030235173222198746,
           1.961376370588077e-18, 0.06040931709105625, 1.6573506178564407e-10,
           7.828895449748772e-10, 0.03561365215328155, 0.06376679474538254,
           0.20786946494570882, 0.14299329652208403]

    w5t = [9.377627082164319e-10, 0.004566461277282273, 0.21458107021948142,
           7.312229582921186e-10, 0.27348254232698305, 3.538922012817086e-11,
           0.002497326685049683, 0.1407656573030659, 1.2972507567856335e-10,
           8.495314219801515e-9, 0.011579421910015182, 6.922994089524877e-11,
           1.5466118772204457e-10, 8.553516188550853e-11, 1.0262638232474694e-11,
           0.18524284235574914, 0.027898538908789215, 2.91899600459833e-9,
           0.03620945531662556, 0.10317667012885948]

    w6t = [2.3925305156408298e-11, 1.4939123402049488e-7, 1.0579346950050932e-9,
           4.5089479240762006e-11, 0.00093832674828407, 9.985462597769679e-15,
           0.9990231151322279, 1.789069180392933e-10, 2.395932350919267e-12,
           2.5014906124740315e-13, 4.4585432301336575e-13, 1.033786806942688e-14,
           7.58520572099388e-14, 1.4177171883670346e-14, 7.137618435256312e-15,
           1.1064306627812483e-6, 3.6977511200118953e-5, 1.8095612352652093e-11,
           3.234267223098279e-7, 3.2507566063008884e-11]

    w7t = [0.048973969314011524, 0.07310557552521323, 0.05845997055102069,
           0.027832882858134678, 0.04886734244161652, 0.02203941659766411,
           0.027888322556591648, 0.05841882873172827, 0.043539671412031355,
           0.07561679787402951, 0.06139231106085604, 0.01582652540855516,
           0.01629627933550747, 0.03263567024354508, 0.008746929502101479,
           0.06491664878126127, 0.11158414579288986, 0.049339707882562096,
           0.0714581499861073, 0.08306085414457287]

    w8t = [3.6370176709848153e-9, 0.03276083087971003, 0.1723750026097795,
           8.172034544457912e-10, 0.12532473181430098, 2.4385810079417996e-11,
           0.017854952028117925, 0.13773911785842563, 1.0077033833774365e-17,
           3.763662029985556e-9, 1.6091503670795757e-9, 3.038911014782725e-10,
           2.4637010025064793e-18, 5.976330141352493e-10, 1.4767055029861839e-18,
           1.5279505500206763e-9, 0.19868064351180353, 0.0977687398160586,
           2.0826038422345858e-9, 0.21749596711830593]

    w9t = [7.066156809243011e-10, 1.9292884682306543e-10, 0.1616894633558081,
           5.50985450854393e-10, 0.20607244367270094, 2.2743338919128936e-10,
           1.0550978717022011e-10, 0.10606864605987947, 8.336949364189651e-10,
           6.4802221866109135e-9, 0.07441664892112852, 4.4491514744585823e-10,
           1.1797549027061238e-10, 5.497027538948563e-10, 6.59541687215039e-11,
           0.14130316382489733, 1.178687962711572e-9, 2.1994992244120266e-9,
           0.2327047364594285, 0.07774488405203203]

    w10t = [0.0915421561413814, 0.10163154185273443, 0.10927339211345093,
            0.05202523185587879, 0.09134284916385084, 1.1383186313720094e-10,
            0.03877041115879424, 0.1091964898141609, 2.248789978287514e-10,
            0.001740699312338166, 3.1708648545154833e-10, 8.174276602380784e-11,
            0.0003751404850578988, 1.685606845395256e-10, 4.51772068258599e-11,
            0.0014943817917427258, 0.15512454011582902, 0.09222579476855723,
            3.690757563028354e-10, 0.15525737010586976]

    w11t = [4.737490419635725e-10, 0.026972185897874996, 0.10840437090246849,
            3.694070716583764e-10, 0.13816084952619803, 1.9317125269699956e-10,
            0.014750669173958762, 0.07111350739840767, 7.081013733640694e-10,
            7.467678977175493e-9, 0.06320601097632407, 3.778900568715544e-10,
            1.3595260519556029e-10, 4.668917345796945e-10, 5.601837723926479e-11,
            0.16283495156752212, 0.16478505609376418, 1.474649768600943e-9,
            0.1976484878066206, 0.052123898933350905]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 0.0001)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:EDaR)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :EDaR, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :EDaR, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EDaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EDaR, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EDaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EDaR, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EDaR, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EDaR, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :EDaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :EDaR, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :EDaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :EDaR, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :EDaR, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :EDaR, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :EDaR, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.06368314296956538, 0.03964178355827289, 0.0825388621979828,
           0.016586300988038757, 0.05587882876522039, 0.025248629100672233,
           0.0190399097684309, 0.02899941110936681, 0.030719783461772187,
           0.07837842004156122, 0.10479226786340595, 0.015985053931477252,
           0.012462053618127768, 0.050366761536315446, 0.00977878156787328,
           0.08402143527479923, 0.07578137420294366, 0.06517737874861304,
           0.05986313124374218, 0.0810566900518186]

    w2t = [0.07520770771397225, 0.07271017597313371, 0.06719834716195543,
           0.019587878653547875, 0.04549329654389771, 0.01759495440823529,
           0.0325513520356299, 0.029652263268033382, 0.038241929416123095,
           0.056434586038089665, 0.07585793159316231, 0.011571398848124902,
           0.006808407073159745, 0.055514112375187226, 0.010778147379197956,
           0.04590352053933134, 0.1295587121696276, 0.06664469104167256,
           0.07452141199172488, 0.06816917577619318]

    w3t = [0.0887784212341269, 3.4891466809257714e-10, 1.0513574803235556e-8,
           2.7727011164722383e-10, 0.013186691640452226, 9.265024794157796e-10,
           5.899144433156251e-10, 0.04140678211830623, 2.734235341397321e-9,
           0.028959097531035326, 0.3439294925362025, 0.04310656989706765,
           2.0037572798132032e-12, 0.0060607979699604, 4.810240678546967e-11,
           0.007870349260050837, 1.168527541783349e-8, 0.05535168725083671,
           0.28817709008698017, 0.08317299334918758]

    w4t = [0.0893877971359108, 5.975399663461963e-10, 4.1631122218719845e-8,
           8.837638508303174e-10, 0.013790913525215696, 1.3927064977882501e-9,
           1.0165863887773937e-9, 0.04207504491830923, 4.088948814631767e-9,
           0.029056112799502702, 0.3432155007817911, 0.04285183836033158,
           7.185782297503246e-12, 0.004867525891677304, 5.826281257326173e-11,
           0.007952956006059251, 2.016307302008043e-8, 0.05509034730151745,
           0.2881977229247516, 0.083514170515744]

    w5t = [1.219454568061737e-8, 7.730879185262923e-15, 0.27138464551898206,
           4.039503580896377e-9, 0.323214034111671, 7.644953371233814e-17,
           8.533476457021957e-10, 0.18025363748588288, 2.1173231269781567e-16,
           8.046975741145786e-9, 6.550457644272192e-9, 9.633235908912312e-17,
           3.251833528823373e-10, 1.1620252642083092e-16, 1.9607658531159103e-17,
           0.22514758283777228, 2.042702603091156e-8, 8.113270327877472e-9,
           1.3424895036980607e-8, 2.6070478424482185e-8]

    w6t = [3.1570881063779455e-13, 4.261295574012588e-8, 8.093971824103465e-12,
           3.8008669233586043e-13, 0.0001556340566844701, 2.6965446955065515e-17,
           0.9998404140550275, 1.7914270959326452e-12, 5.782820369192832e-15,
           1.5021002204024365e-15, 1.0677679567414885e-15, 2.823429413170783e-17,
           2.390846140188947e-16, 4.2495945476640504e-17, 1.6824034825043417e-17,
           1.381756618190805e-7, 3.734304231185893e-6, 2.2111535990390525e-13,
           3.6784284809459755e-8, 3.4347625332286694e-13]

    w7t = [0.060308111118779595, 0.07523098005894394, 0.05376483748449111,
           0.030909814180578095, 0.047433311210150035, 0.02684389268316511,
           0.028588543668545453, 0.06389482826865836, 0.04632654911051497,
           0.06855499020851732, 0.08013916339605456, 0.018093963841450872,
           0.01965684026402137, 0.040696365448656846, 0.012091432008016528,
           0.053188790894120944, 0.0828306280509751, 0.05666325757434575,
           0.06259027496510058, 0.07219342556491347]

    w8t = [0.11881144282680177, 0.011577623088194242, 1.407023209328881e-8,
           3.710683470105646e-10, 0.017647642728208318, 2.7722608355264746e-10,
           0.019574433818810074, 0.05541436148449842, 8.181320309955183e-10,
           5.651732145645521e-9, 0.10290984466030023, 0.012898255334932223,
           3.9105843744802065e-19, 0.0018134989616816145, 1.439309892726978e-11,
           1.5359976554109316e-9, 0.3877386846732448, 0.07407671519441021,
           0.08622773044794005, 0.11130974404219646]

    w9t = [4.819734915293999e-9, 9.034387764386685e-16, 0.10801642903955977,
           1.597410139225079e-9, 0.12864559312540172, 2.1784112519592365e-9,
           1.0224163947412267e-10, 0.07174451329355098, 6.0267780622468395e-9,
           4.424545691067458e-9, 0.18692030127644063, 2.7418074829189674e-9,
           1.7871321361033365e-10, 3.312959588558344e-9, 5.587616728445239e-10,
           0.12158704731351491, 2.4474115012321703e-9, 3.2044740784921893e-9,
           0.38308607402282085, 1.0335461069196405e-8]

    w10t = [0.12209748260667466, 0.08875535493634554, 0.10885022243001703,
            0.06257882114489696, 0.09603165781550807, 7.589045701083328e-10,
            0.03372794476991936, 0.12935901222016002, 1.3096993886943443e-9,
            4.1659071328388785e-9, 2.265616915689193e-9, 5.115350449625224e-10,
            1.1944946796125017e-9, 1.1505282818074127e-9, 3.41837270709725e-10,
            3.2321434617514512e-9, 0.09772120191049807, 0.11471825228442356,
            1.769491715526832e-9, 0.14616003318139834]

    w11t = [5.583668809412522e-9, 6.732304790303556e-8, 0.12513716549399082,
            1.8506016050648948e-9, 0.14903607738328556, 1.1781678482702653e-9,
            0.007618910070624505, 0.08311610662497332, 3.2595113228568036e-9,
            5.2558951944265726e-9, 0.10109362451870711, 1.4828740071998116e-9,
            2.12292512311805e-10, 1.7917748388177926e-9, 3.021996132268625e-10,
            0.14443263157385172, 0.182377827953546, 3.712387149316143e-9,
            0.20718755245495585, 1.1973644322189683e-8]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-6)
    @test isapprox(w3.weights, w3t, rtol = 0.0001)
    @test isapprox(w4.weights, w4t, rtol = 0.0001)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-7)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 1.0e-5)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-4)
    @test isapprox(w9.weights, w9t, rtol = 1.0e-6)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 1.0e-5)
end

@testset "$(:HRP), $(:HERC), $(:NCO), $(:RDaR)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :RDaR, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :RDaR, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :RDaR, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :RDaR, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :RDaR, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :RDaR, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :RDaR, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.06626056846619277, 0.042384457717238096, 0.0822955972759821,
           0.0175713994434378, 0.054533053879138406, 0.027516398529958203,
           0.020210380798873082, 0.028829731825373697, 0.028405213307233126,
           0.07604187167883428, 0.10819510802782517, 0.017147586401631137,
           0.013186029945941988, 0.052505625597297795, 0.010967030873083648,
           0.08422883345382008, 0.07476238309203545, 0.06211533959730764,
           0.05474745892775125, 0.07809593116104428]

    w2t = [0.0751468929534473, 0.07639746285602138, 0.06378327056785309,
           0.019927931552413278, 0.0422658884022649, 0.018661179044012894,
           0.03359489426809132, 0.03080244571310856, 0.03739453637661427,
           0.0578590028072092, 0.077619808112924, 0.012301779538425104,
           0.007386199000120996, 0.05517573403014359, 0.011524745618587183,
           0.04718106420116424, 0.12427446965014705, 0.06636566678755676,
           0.07207324311413196, 0.07026378540576289]

    w3t = [0.0775814021714122, 1.082202635612759e-9, 4.984408724494647e-9,
           2.2371322124893384e-10, 0.00531783754523333, 1.2892994167682342e-10,
           4.278338361471623e-10, 0.022948129537220016, 6.437958766359547e-10,
           0.0408944124346119, 0.3915273105563321, 0.04770173401232437,
           4.591085809125209e-12, 2.982823508853673e-10, 2.4203934241239368e-11,
           0.010921191908881616, 2.685713791634684e-9, 0.04196892809033501,
           0.3026961330431245, 0.058442910196849454]

    w4t = [0.07757592491647358, 1.0567624060296577e-9, 1.2073869686922138e-8,
           3.194044952579666e-10, 0.005608442591732419, 1.6586292089968673e-10,
           4.179279355685602e-10, 0.02312414215920728, 7.867575493253931e-10,
           0.04082878910061243, 0.3913806413096025, 0.04768714763070843,
           8.964341537199215e-12, 4.150256980047205e-10, 3.1912712456061e-11,
           0.010979499109902011, 2.6262066729836297e-9, 0.041654274748570246,
           0.3028178737662861, 0.05834324676421052]

    w5t = [1.5532930061960584e-8, 2.6659293537474916e-9, 0.4042379416231154,
           4.837555938131187e-9, 0.3228437801003447, 7.512844679998556e-17,
           1.2601218271643066e-9, 0.1163011841592106, 2.022638521282633e-16,
           1.1148929234540751e-8, 8.816212610744705e-9, 9.679823890769714e-17,
           3.980764985969551e-10, 1.163723015442524e-16, 2.1057606638090935e-17,
           0.15661699749685937, 1.1551230325124222e-8, 8.677405847728425e-9,
           1.623583552320901e-8, 1.54962422563424e-8]

    w6t = [1.1892904072958567e-13, 2.875413700371397e-8, 4.180692259560764e-12,
           1.5553725016704774e-13, 9.135996935266051e-5, 2.7847543617582435e-17,
           0.9999047524487951, 6.861593098506342e-13, 5.254954787218689e-15,
           1.8909158235087653e-15, 1.0860501394786603e-15, 2.8933200147510487e-17,
           3.1161728002960927e-16, 4.245743959267311e-17, 1.7278744879093537e-17,
           1.570851776487345e-7, 3.676590894169778e-6, 8.82070563462709e-14,
           2.5146247912619646e-8, 1.5757151538167763e-13]

    w7t = [0.07051973885423607, 0.07584689772532657, 0.053095682441402156,
           0.03672128495177433, 0.04714160072070756, 0.027773823262147456,
           0.028817608888299692, 0.061874992398139664, 0.04767372430703749,
           0.06410466549637614, 0.08020872422663865, 0.020687467066990258,
           0.019779415281696616, 0.039720829724268915, 0.012764403877131951,
           0.05171090331715661, 0.08206541972700375, 0.060803959832330584,
           0.05992504535542075, 0.058763812545914826]

    w8t = [0.12129038395978775, 0.1032191871588299, 7.99074600370278e-9,
           3.528051996200124e-10, 0.00831388850569054, 5.542780184291354e-11,
           0.04080626873486547, 0.03587699864067984, 2.490400829562258e-10,
           5.274607820624523e-9, 0.14636256171705903, 0.01783208490426641,
           6.041636242729209e-19, 1.2064851798772764e-10, 6.562015375842511e-12,
           1.408612006539336e-9, 0.2561599523090814, 0.06561400709290342,
           0.11315529590229492, 0.09136935561609187]

    w9t = [3.4469153725460873e-9, 4.950581335348816e-11, 0.0897045161208452,
           1.073502929722926e-9, 0.07164232273754961, 2.1669511355147744e-9,
           2.3400228475882332e-11, 0.025808417209412, 5.833953751364082e-9,
           6.425341114038141e-9, 0.2542885250729898, 2.791979105801307e-9,
           2.2941909838687057e-10, 3.3565593555413946e-9, 6.073705308605338e-10,
           0.09026137057683384, 2.1450420344968988e-10, 1.92560473079096e-9,
           0.4682948166985882, 3.438773974840319e-9]

    w10t = [0.1470443512637328, 0.07678544094103464, 0.11071255093044297,
            0.07656944865416185, 0.09829738748520575, 1.063282841989309e-9,
            0.02917420318715934, 0.12901874375115208, 1.8251233397343315e-9,
            4.908238244330081e-9, 3.0706813189911156e-9, 7.919913858791104e-10,
            1.5144308418799608e-9, 1.5206576464737153e-9, 4.886677467006688e-10,
            3.959297366967328e-9, 0.0830809120574606, 0.1267854784075685,
            2.294148411993727e-9, 0.1225314618855625]

    w11t = [6.3264476759193756e-9, 0.03310925149927765, 0.16464312760687175,
            1.970300799645594e-9, 0.13149188685930713, 9.734096988937452e-10,
            0.015649961030161665, 0.047368613216934745, 2.620653078605901e-9,
            9.943837906057673e-9, 0.11422819488942093, 1.254176661464062e-9,
            3.55048282483979e-10, 1.5077900825947148e-9, 2.7283511652487625e-10,
            0.13968821612189214, 0.14345938665741756, 3.534243303702966e-9,
            0.21036132704846883, 6.311505003695791e-9]

    @test isapprox(w1.weights, w1t, rtol = 5.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-7)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-6)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-5)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-4)
    @test isapprox(w6.weights, w6t, rtol = 1.0e-7)
    @test isapprox(w7.weights, w7t, rtol = 5.0e-6)
    @test isapprox(w8.weights, w8t, rtol = 5.0e-5)
    @test isapprox(w9.weights, w9t, rtol = 1.0e-4)
    @test isapprox(w10.weights, w10t, rtol = 1.0e-5)
    @test isapprox(w11.weights, w11t, rtol = 0.0001)
end

@testset "$(:HRP), $(:HERC), $(:NCO), Full $(:Kurt)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :Kurt, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :Kurt, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :Kurt, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :Kurt, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :Kurt, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :Kurt, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.029481521953684118, 0.05982735944410474, 0.03557911826015237,
           0.028474368121041888, 0.025828635808911045, 0.05929340372085539,
           0.00400811258716055, 0.04856579375773367, 0.028712479333209244,
           0.0657391879397804, 0.16216057544981644, 0.011489226379288793,
           0.009094526296241098, 0.08053893478316498, 0.012724404070883372,
           0.021335119185650623, 0.0855020913192837, 0.11267092582579596,
           0.03851809818436267, 0.08045611757887891]

    w2t = [0.03387678411372887, 0.09937541013097657, 0.03447112089034947,
           0.03271947842878706, 0.025024285899713825, 0.04058093588178494,
           0.005123455337854566, 0.04427043031765973, 0.03694893299425578,
           0.11035502477842635, 0.08911250801239487, 0.006313703407505692,
           0.010111939944752868, 0.0661663716037664, 0.01045367250581684,
           0.02372190006296558, 0.10929487050109292, 0.10270583438786722,
           0.04956738888216057, 0.06980595191813997]

    w3t = [0.0020538089950505375, 8.295179811976934e-8, 0.007795951880502392,
           0.060935174818880104, 2.9266750653280933e-8, 0.07844311049923124,
           8.646889146210331e-9, 0.10886038348364989, 9.30610234600973e-8,
           6.134374631754735e-8, 0.2577211221019369, 0.019735213445093838,
           3.997289090649795e-9, 0.1011328033124083, 0.00020290581517230338,
           1.8185914228557167e-8, 9.575177145090074e-8, 0.20022307113274748,
           0.03137921782586492, 0.13151684348427975]

    w4t = [3.5235313037544633e-9, 0.13968049714572486, 0.06864917032501697,
           0.09164352710177702, 0.1941761526042709, 9.28905966595001e-17,
           0.037602412527191424, 0.08831489917767826, 1.138287388065542e-15,
           2.6193401535198235e-8, 1.5314934663760045e-8, 1.0105363891957816e-16,
           2.442785257037521e-16, 2.2856515462968445e-16, 5.3519197995688067e-17,
           1.07713362810882e-8, 0.2421192174115078, 0.11620888251451723,
           1.910060878501064e-8, 0.021605166288501187]

    w5t = [3.935455957122119e-8, 0.11842055561636398, 0.05075270035956987,
           0.07657636923912364, 0.14720011386390236, 2.734653678774999e-10,
           0.02650291304500535, 0.11095251613583235, 4.6490113814810055e-9,
           0.002069279756004381, 0.030937053998157066, 3.214166222319892e-10,
           7.86080243064354e-12, 8.362735005153349e-10, 1.3817960532717336e-10,
           0.0009946044967541845, 0.18253113088292355, 0.15939603624999776,
           0.039202665829854114, 0.054464014945744564]

    w6t = [7.129505568731354e-16, 3.926336916252277e-8, 9.498402489517851e-16,
           8.275627397169682e-16, 5.512854336701769e-7, 9.896899008241978e-18,
           0.9999993265323346, 5.11355446256894e-16, 1.204583879659665e-15,
           3.4358713294991055e-16, 2.2392366576509616e-17, 1.0561183427310336e-17,
           6.584836543340043e-17, 1.4750445628149622e-17, 6.8480634222257526e-18,
           1.7789775525883463e-8, 5.164735896813484e-8, 5.379122063853694e-16,
           1.3481722318368891e-8, 6.639259744579484e-16]

    w7t = [0.0326177248989493, 0.0987705245071447, 0.03291151547998419, 0.03783556812968726,
           0.028457199397425723, 0.0422369877280818, 0.030089830122294884,
           0.055994149124004064, 0.03174861595333299, 0.11764879900103917,
           0.06442192637197168, 0.025786004293508486, 0.037082311439093345,
           0.04281258898739385, 0.02450454640996756, 0.05980836748959063,
           0.10393663363981313, 0.05549740268794204, 0.034438654988264324,
           0.043400649350510905]

    w8t = [0.001961859579221837, 0.2264772538418214, 0.007446925645361566,
           0.05820709555663211, 2.7956472710074632e-8, 5.2997793493745144e-9,
           0.023607971767903767, 0.1039866835961082, 6.287395887626287e-15,
           1.4982349886792918e-7, 1.741217389418614e-8, 1.333351979621354e-9,
           9.762818111036098e-9, 6.8327420869511385e-9, 1.370873799208554e-11,
           4.441654550620589e-8, 0.26142408893149155, 0.1912590464982992,
           2.1200450820301592e-9, 0.12562880961201764]

    w9t = [3.5039698122031704e-8, 1.07865437883982e-7, 0.04518813877865677,
           0.06818048253250635, 0.13106099037865007, 1.6491657495329543e-9,
           2.4140642694346245e-8, 0.09878760462923351, 2.8036421573359238e-8,
           0.029300677588034445, 0.18656962028210822, 1.9383415488026844e-9,
           1.1130773252634875e-10, 5.043247797687195e-9, 8.333087080029676e-10,
           0.014083444059432395, 1.6626167862215534e-7, 0.14191974330040844,
           0.2364163852950562, 0.04849254223666297]

    w10t = [0.0675996845832932, 0.17216738180204996, 0.068208560606162, 0.07841357665258097,
            0.0589770656705682, 3.0660816347201604e-9, 0.05244972927777891,
            0.1160469294231643, 2.3047062193210685e-9, 3.0983601592893916e-8,
            4.676538170620323e-9, 1.871866306047583e-9, 9.765875840027337e-9,
            3.10786587514612e-9, 1.7788422838875008e-9, 1.57509353767606e-8,
            0.18117245176511212, 0.11501743081467154, 2.499982438704479e-9,
            0.08994711359832316]

    w11t = [1.8467235927225174e-8, 0.08453104860856006, 0.023815845017595733,
            0.03593367306347225, 0.06907406056265367, 1.0324831875294546e-9,
            0.018918328994617623, 0.052064775073687675, 1.7552592224995804e-8,
            0.14871620308071526, 0.11680450937062213, 1.2135257243823717e-9,
            5.649447288414345e-10, 3.1573955274851083e-9, 5.217045231982385e-10,
            0.07148081543594571, 0.13029450686944513, 0.07479703087430342,
            0.14801177088643733, 0.025557389652061918]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-8)
    @test isapprox(w4.weights, w4t, rtol = 1.0e-8)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-6)
    @test isapprox(w6.weights, w6t, rtol = 1.0e-9)
    @test isapprox(w7.weights, w7t, rtol = 0.0005)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-7)
    @test isapprox(w9.weights, w9t, rtol = 1.0e-8)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t, rtol = 0.0001)
end

@testset "$(:NCO), Reduced $(:Kurt)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))),
                            max_num_assets_kurt = 1)
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Min_Risk, rf = rf, l = l),
                   cluster = true)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :Kurt, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :Kurt, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)

    w3t = [1.6445015729827607e-6, 8.313053541228161e-6, 1.3773673270772764e-6,
           0.041510389445998754, 1.5678078293040441e-6, 2.300127689618021e-6,
           1.1750398710528672e-6, 0.0489594416724913, 1.0488930329590219e-6,
           1.8644212441953432e-6, 0.6514623581653005, 1.112823775976261e-6,
           1.0536544854406762e-6, 3.421386733084421e-6, 9.772283319523487e-7,
           3.6357143230324623e-6, 2.575341653170939e-6, 0.12541345407152624,
           1.2336717470066145e-6, 0.1326210556115251]

    w4t = [1.8110308075917118e-7, 1.14853741128717e-6, 0.007974499903959738,
           0.03342410429795055, 0.31598167772742397, 7.140500034239051e-8,
           1.1948567502049891e-5, 5.426079030997925e-7, 5.594829392937265e-7,
           3.8029015475594874e-7, 8.379795164738719e-7, 6.678782431084351e-8,
           1.0373470578595258e-7, 1.3095141830587786e-7, 4.309795036507446e-8,
           4.231678147829965e-6, 0.5345733964005096, 8.152791257967155e-7,
           0.10802472851533884, 5.316521367189196e-7]

    w6t = [4.979922888688266e-12, 5.400638042758381e-12, 7.472107023316153e-12,
           6.486329047389137e-12, 3.298501513943451e-5, 2.290664795215238e-12,
           0.9999652205444823, 4.888239410003251e-7, 5.822186574957655e-12,
           3.841371598346326e-12, 3.3761494223413466e-12, 2.4053232523799823e-12,
           2.5217779908970765e-7, 2.84254687544469e-12, 1.875476136525232e-12,
           1.053369413955893e-6, 7.985539851522891e-12, 3.672425266258579e-12,
           6.126221845713741e-12, 4.6472812006924e-12]

    w7t = [0.02191601579158622, 0.025071711092476606, 0.020836110680854395,
           0.024920982342868964, 0.02005217251791944, 0.02724314975593113,
           0.09234816826271326, 0.26532290817866744, 0.019235648342112034,
           0.02409051080249756, 0.04089789724488654, 0.018779005808836197,
           0.09572092828088434, 0.027577965083938408, 0.019070213145241106,
           0.15236404577538315, 0.024122524955846197, 0.032499642923058666,
           0.021091567048025228, 0.026838831966272996]

    w8t = [1.4128758609929186e-6, 7.1421717513103864e-6, 1.1833670943945169e-6,
           0.035663710021444694, 1.3469843222354956e-6, 1.9761579698390362e-6,
           0.04488051093314426, 0.03556042389974478, 9.01157938294649e-7,
           1.6018201587172325e-6, 0.5597048098457034, 9.560841269151726e-7,
           2.642259013498187e-9, 2.939489268793223e-6, 8.395871086882337e-7,
           0.10247625513808854, 2.212608437324347e-6, 0.10774914710480547,
           1.0599108327839105e-6, 0.11394156819993966]

    w10t = [0.03690898277891147, 0.04222352099715689, 0.035090303712702664,
            0.04196967719286053, 0.033770065562180526, 0.04588046270577139,
            0.06095950467148902, 0.14859141230142467, 0.03239494897940227,
            0.04057107171304008, 0.0688765603867068, 0.031625912693013625,
            2.799219920414802e-9, 0.04644432856957864, 0.03211633789924968,
            0.12649887681276037, 0.040624987070916385, 0.05473297575503918,
            0.03552052035181876, 0.04519954704675691]

    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
end

@testset "$(:HRP), $(:HERC), $(:NCO), Full $(:SKurt)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HRP, rm = :SKurt, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :SKurt, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :SKurt, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Sharpe, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :SKurt, obj = :Min_Risk, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SKurt, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Sharpe, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SKurt, obj = :Equal, rf = rf, l = l),
                    cluster = false)

    w1t = [0.041695764288341326, 0.04968730030116381, 0.04519046227953908,
           0.02228610752801017, 0.036916599549008534, 0.06695518729855642,
           0.008601057788011419, 0.03832875671733724, 0.023603809923922775,
           0.05953659154802947, 0.15949334007294313, 0.010345061127586078,
           0.01551106764201746, 0.07844578132980796, 0.01457102407508962,
           0.034738374409788796, 0.09106192787421133, 0.1097918256490279,
           0.0352173258407637, 0.058022634756843806]

    w2t = [0.05283875180271047, 0.09326996155827437, 0.038193414313502316,
           0.02824195993093501, 0.03120063195855665, 0.046986152168249494,
           0.012083639340153919, 0.030816699662775084, 0.03364535163661155,
           0.10607577588115502, 0.0833002873990155, 0.005403025384597182,
           0.016448287121189113, 0.06537475836063772, 0.012143128181375595,
           0.03683735830458003, 0.1279330427921104, 0.08827371420903507,
           0.050199493871229225, 0.040734566123306254]

    w3t = [4.916851994029914e-8, 0.00963279449728814, 9.53206373056534e-9,
           0.035421206543947203, 9.388265324807899e-9, 0.08718902876968762,
           1.633797603347197e-9, 0.09877145071961631, 2.410751548715072e-8,
           1.3518942541666656e-8, 0.26070221694991313, 0.016438393351695255,
           8.51485170964513e-10, 0.12402775420789341, 4.150875843220202e-7,
           4.354557973085832e-9, 0.01688407789342895, 0.23610676412453788,
           2.7998013093044485e-6, 0.11482298549795064]

    w4t = [6.051573488335999e-9, 0.033989662704861145, 0.12463819316162207,
           3.7848080862585804e-8, 0.5994385074005049, 1.2846546503241694e-17,
           0.034693567827931995, 1.7052306730184856e-8, 9.514216585487543e-17,
           5.50445407733037e-9, 2.8123855973131696e-9, 1.500566470539602e-17,
           1.2992181433275075e-17, 2.9315194996534125e-17, 7.603473547147124e-18,
           3.3146414045203668e-9, 0.1939539514212894, 0.013286028675463624,
           7.155968208314142e-9, 9.06891643807576e-9]

    w5t = [2.42632447371091e-7, 0.08063263505973907, 1.220627679348312e-6,
           0.027594872650097562, 0.290314723471155, 1.185106343417948e-15,
           0.024570085722384328, 0.10342540077876686, 1.0238510395149885e-14,
           3.8348977045567875e-7, 8.329217643449107e-8, 1.519563708984896e-15,
           2.150460821403134e-15, 3.438400792957777e-15, 5.760697058856682e-16,
           2.510726406989156e-7, 0.22004547024650517, 0.20230877382386042,
           1.2935976702208148e-7, 0.05110572777299116]

    w6t = [1.559265699862942e-16, 1.1770656851187672e-8, 2.267590583788695e-16,
           1.9961508329786136e-16, 1.3317659402239818e-7, 4.307908857931094e-18,
           0.9999998313315561, 1.0959511420170268e-16, 5.211656422181268e-16,
           7.556254979832401e-17, 1.0016905333038929e-17, 4.689084408241327e-18,
           1.4837987775382495e-17, 6.583520296388086e-18, 2.9847569825373597e-18,
           4.613088974502009e-9, 1.5646273457261347e-8, 1.162443108442906e-16,
           3.4618289439063874e-9, 1.4580098921215038e-16]

    w7t = [0.03651105842516102, 0.09214153068178792, 0.03206431818112788,
           0.034869492041313004, 0.031170486693440098, 0.04238780833511241,
           0.03963004897710973, 0.05254306718284025, 0.030739527050275137,
           0.10295513290968135, 0.0629780830300844, 0.025256533994634973,
           0.04623721916884042, 0.04366566388556621, 0.027412594303296147,
           0.06864686166032541, 0.1002519644038779, 0.05349142665545054,
           0.034611824035786394, 0.0424353583842889]

    w8t = [3.971014802541533e-8, 0.22094047734221736, 7.698414800203823e-9,
           0.028607371241149544, 7.582278378242912e-9, 3.9822903811350936e-9,
           3.747179496489536e-8, 0.07977118326683406, 1.1010918596555762e-15,
           7.415818115990654e-8, 1.1907368914994026e-8, 7.508107000327215e-10,
           4.670823370833968e-9, 5.6648702198601265e-9, 1.895879981409802e-14,
           2.3886934808259206e-8, 0.38725794815575193, 0.19068785377063519,
           1.2787872857526207e-13, 0.09273494873934807]

    w9t = [1.59608434872206e-7, 0.02358361047960325, 8.02953090459943e-7,
           0.018152454388937117, 0.19097478154980363, 2.2864729704229334e-9,
           0.00718631272186342, 0.06803527938323242, 1.975356676295207e-8,
           0.030656956375471715, 0.16069892050016482, 2.931754915266928e-9,
           1.719122351310185e-10, 6.6338438894054045e-9, 1.1114342766812341e-9,
           0.02007126026812943, 0.06435938319825385, 0.13308272286251838,
           0.24957896175219965, 0.03361836106931188]

    w10t = [0.07278895305406786, 0.173000944964644, 0.06392386995794189,
            0.06951630351710714, 0.06214191509270802, 2.427020740147865e-9,
            0.07440766255243278, 0.10475058832731238, 1.760069053430012e-9,
            1.547355384448209e-8, 3.605968784240134e-9, 1.4461264744950658e-9,
            6.949183398712258e-9, 2.500187578578041e-9, 1.5695771385341319e-9,
            1.0317221493828735e-8, 0.18822874384765095, 0.10664125094044112,
            1.9817871715631175e-9, 0.0845997197149983]

    w11t = [9.696459792465895e-8, 0.060528013305000856, 4.878093882039893e-7,
            0.011027879015659294, 0.11601967776433118, 1.4493738727471603e-9,
            0.018443882026684778, 0.04133233310523048, 1.2521619443347782e-8,
            0.1365969994175233, 0.10190286270489149, 1.8584112103831813e-9,
            7.659391091401999e-10, 4.2051321091802136e-9, 7.0452799784178e-10,
            0.08943072648183441, 0.16518019002523351, 0.08084949663960606,
            0.1582637181195416, 0.020423615115473113]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 1.0e-5)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t, rtol = 1.0e-7)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 0.0001)
    @test isapprox(w8.weights, w8t, rtol = 1.0e-5)
    @test isapprox(w9.weights, w9t, rtol = 1.0e-7)
    @test isapprox(w10.weights, w10t, rtol = 1e-4)
    @test isapprox(w11.weights, w11t, rtol = 0.0005)
end

@testset "$(:NCO), Reduced $(:SKurt)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))),
                            max_num_assets_kurt = 1)
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Min_Risk, rf = rf, l = l),
                   cluster = true)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Equal, rf = rf, l = l),
                   cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Min_Risk, rf = rf, l = l),
                   nco_opt_o = OptimiseOpt(; rm = :SKurt, obj = :Sharpe, rf = rf, l = l),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Equal, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SKurt, obj = :Sharpe, rf = rf, l = l),
                    cluster = false)

    w3t = [2.309605391947925e-6, 0.08123303722880364, 1.7642216615625348e-6,
           3.7965975188801014e-6, 1.8253685785110527e-6, 2.302182220677121e-5,
           2.210121458680488e-6, 0.10719137805742462, 1.4652437013695145e-6,
           2.0710177574500876e-6, 0.39937134055249635, 2.8800328105257137e-6,
           3.1354324492883147e-6, 0.02779824987681546, 2.643445358305059e-6,
           1.0127140895747743e-5, 1.0611528784994818e-5, 0.23165224445120117,
           2.32594813102965e-6, 0.1526835623065537]

    w4t = [2.9928667016944155e-7, 1.2533185281502334e-6, 5.361551938137822e-7,
           4.207434689482753e-7, 0.5838163945053438, 7.260486932651326e-8,
           0.030671151810874462, 4.207946146830618e-7, 5.090067267426231e-7,
           1.9892438860287958e-7, 3.4584146503070664e-7, 7.16815205645096e-8,
           8.945787486580572e-8, 1.3759218865766605e-7, 4.545984569479047e-8,
           0.091668339155504, 0.29383507914740237, 3.700828260980578e-7,
           3.929316269784194e-6, 3.35114424317425e-7]

    w6t = [4.979397704879066e-12, 5.400068545559919e-12, 7.471319373942352e-12,
           6.4856452688010205e-12, 3.2981574337484e-5, 2.2904231510167815e-12,
           0.9999652241935004, 4.887662996057283e-7, 5.821572606187697e-12,
           3.8409664055707176e-12, 3.3757932776038418e-12, 2.4050697104012196e-12,
           2.5214784324409907e-7, 2.8422470157065424e-12, 1.8752784054197944e-12,
           1.0532488024371983e-6, 7.984698167430288e-12, 3.672037880187816e-12,
           6.125575876937054e-12, 4.646791104303914e-12]

    w7t = [0.019882883256399937, 0.023001331642739778, 0.01828339763461694,
           0.01872963697038636, 0.017918935286446544, 0.02349210145617068,
           0.11785479264475786, 0.24821360221974836, 0.016466217447950945,
           0.019212663431670208, 0.03412096524791616, 0.017413780114492503,
           0.11916390584855842, 0.02508118724256985, 0.017412306532890096,
           0.17174294838444773, 0.021189512851042466, 0.027810771886776074,
           0.01844234991862122, 0.024566709981797694]

    w8t = [1.8611172935854082e-6, 0.06545889220906466, 1.4216382831020206e-6,
           3.0593595441911437e-6, 1.470911455471235e-6, 1.8551355823920136e-5,
           0.07734620590403969, 0.031196422878190386, 1.1807170183456965e-6,
           1.6688595277576899e-6, 0.32181987063935524, 2.320776912129606e-6,
           5.552202880385119e-9, 0.02240027831486644, 2.130130925456615e-6,
           0.17203023951991137, 8.550941126993395e-6, 0.18666911661579882,
           1.8742865364507012e-6, 0.12303487827212319]

    w10t = [0.03204378748261461, 0.037069562471018856, 0.02946601359112357,
            0.030185184863085635, 0.028878636303856762, 0.03786050025411932,
            0.10541375171088437, 0.15314887078782796, 0.026537397305032365,
            0.030963643252357575, 0.054990261975911864, 0.028064514691457886,
            6.270713658471298e-9, 0.040421513492208336, 0.028062139827858032,
            0.18860949021095066, 0.03414958675273521, 0.044820585252919834,
            0.02972218535166891, 0.039592368151654456]

    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 5e-5)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
end

@testset "Mixed inner and outer parameters, $(:NCO)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, l = l, obj = :Sharpe, kelly = :Exact,
                                         rf = rf),
                   nco_opt_o = OptimiseOpt(; rm = :CDaR, obj = :Utility, l = 10 * l,
                                           kelly = :None, rf = rf),
                   cluster_opt = ClusterOpt(; dbht_method = :Equal, branchorder = :default,
                                            linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Sharpe, l = l, rf = rf,
                                         kelly = :None),
                   nco_opt_o = OptimiseOpt(; rm = :CDaR, obj = :Utility, l = 10 * l,
                                           kelly = :Exact, rf = rf), cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Utility, l = 10 * l,
                                         kelly = :Exact, rf = rf),
                   nco_opt_o = OptimiseOpt(; rm = :CDaR, obj = :Sharpe, l = l,
                                           kelly = :None, rf = rf), cluster = false)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Utility, l = 10 * l,
                                         kelly = :None, rf = rf),
                   nco_opt_o = OptimiseOpt(; rm = :CDaR, obj = :Sharpe, l = l,
                                           kelly = :Exact, rf = rf), cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Utility, l = 2 * l,
                                           kelly = :None, rf = rf),
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Sharpe, l = l, kelly = :Exact,
                                         rf = rf), cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Utility, l = 2 * l,
                                           kelly = :Exact, rf = rf),
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Sharpe, kelly = :None, l = l,
                                         rf = rf), cluster = false)
    w7 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :CDaR, obj = :Utility, l = 2 * l,
                                         kelly = :Exact, rf = rf),
                   nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Sharpe, l = l, rf = rf,
                                           kelly = :None), cluster = false)
    w8 = optimise!(portfolio; type = :NCO,
                   nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Sharpe, l = l, kelly = :Exact,
                                           rf = rf),
                   nco_opt = OptimiseOpt(; rm = :CDaR, rf = rf, obj = :Utility, l = 2 * l,
                                         kelly = :None), cluster = false)
    w9 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SD, obj = :Utility, l = l, rf = rf),
                   nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Utility, l = l, rf = rf),
                   cluster = false)
    w10 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SD, obj = :Utility, l = l, rf = rf),
                    cluster = false)
    w11 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SD, obj = :Utility, rf = rf, l = l),
                    nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Utility, l = 2 * l, rf = rf),
                    cluster = false)
    w12 = optimise!(portfolio; type = :NCO,
                    nco_opt = OptimiseOpt(; rm = :SD, obj = :Utility, l = 2 * l, rf = rf),
                    nco_opt_o = OptimiseOpt(; rm = :SD, obj = :Utility, l = l, rf = rf),
                    cluster = false)

    w1t = [3.735415577228451e-8, 0.0466001831846816, 0.03972197474935932,
           0.02987869055275237, 0.4774400544550938, 8.8060322718383e-18,
           0.01312757674644295, 0.02529023566474106, 3.833135158114425e-10,
           0.09061070182813595, 1.6733747486061413e-16, 1.0920927060699974e-15,
           1.955001811896496e-9, 1.7194519201439068e-17, 5.5138509906467655e-18,
           0.18919212844229372, 3.254596302056313e-9, 0.05992501952160777,
           1.8930215396337555e-7, 0.02821320260566884]

    w2t = [2.840868390500683e-10, 0.04234055498784264, 0.031203311935144126,
           0.0226859893782639, 0.4623489751616544, 2.9603910070747493e-10,
           0.014040076369707303, 3.897800168693933e-9, 0.017584322217662027,
           0.04451032662538801, 3.1157518567052835e-9, 3.5065068233278475e-12,
           8.379686170186067e-11, 6.547290712588994e-10, 1.74424100925663e-10,
           0.17284249030373613, 0.17058529952807988, 0.002994800971001183,
           0.018863825379910024, 1.8631475894583847e-8]

    w3t = [1.228469386112449e-8, 0.10893990523541322, 0.03423674185505514,
           0.0301577719023965, 0.08870792687404323, 1.8013297902235126e-9,
           0.007841793804218676, 0.13583170571288947, 0.016525127840573795,
           8.142225765479092e-10, 0.13414911640122046, 2.3952050048146996e-10,
           4.179821388359109e-18, 0.027567734739708694, 2.9806544087212303e-10,
           3.370623860081185e-10, 0.1000369397364426, 0.20390686905917438,
           4.6972423825093405e-9, 0.11209834636672687]

    w4t = [7.04627771615497e-8, 0.10988150227300492, 0.03471191262903169,
           0.030544810348287457, 0.09161200182088644, 2.0929124891808844e-8,
           0.007961925924456149, 0.13703046695762694, 0.016276307915609807,
           1.3557718460926932e-7, 0.1296349270458446, 6.814856876321049e-7,
           4.781134484361836e-14, 0.02568147212341953, 2.3050364669916656e-9,
           5.633353588483274e-8, 0.09772886787458748, 0.20579510879552046,
           1.3736745600844884e-5, 0.11312599245272938]

    w5t = [3.275785344242584e-8, 0.07436681280827424, 0.1217821533559664,
           1.1824153426530223e-8, 0.29828910324883257, 1.468770701759108e-14,
           0.04388348162269195, 0.18847691341687808, 2.979814532048435e-14,
           0.0011418622736659148, 4.1277673159546503e-7, 1.6468855376274777e-9,
           6.342239745369124e-10, 3.038720706641237e-14, 1.2426120988201888e-14,
           0.12413260665858533, 7.707891025511953e-7, 2.5820427206646895e-8,
           0.08541011774669942, 0.06251569261894105]

    w6t = [1.0349973688109248e-10, 0.060657837765311556, 0.11754630695183446,
           6.287241110863267e-11, 0.28362694132626287, 1.1447851428446168e-11,
           0.04336726658005853, 0.17779925613691092, 1.5202985337554534e-11,
           3.067163210550636e-9, 0.018802472081689763, 2.0511368577776368e-11,
           5.920208300776516e-11, 3.1381318531651944e-12, 1.0732385631897083e-11,
           0.11797875273896506, 0.035109544302919724, 3.064239590117017e-11,
           0.09031713520330326, 0.0547944835283314]

    w7t = [4.4793017885269395e-11, 0.16279243756592543, 0.02286678804833082,
           4.4757531143129375e-12, 0.1380340028794419, 1.380009734355475e-12,
           0.024869560318423772, 0.2026501786745362, 8.759121682634408e-13,
           8.041581894203742e-10, 0.04697149828513465, 0.006330662330411931,
           4.473434468742288e-22, 0.0003237951532295791, 7.901631980821521e-13,
           2.0523794580039927e-10, 0.04436588652064439, 0.04747149514190659,
           0.10519151253407714, 0.19813218148622666]

    w8t = [5.643977617697122e-10, 0.15253533692692972, 0.022401526828640262,
           2.161447585058125e-11, 0.13522550204689956, 4.867215755904731e-13,
           0.023302599285194264, 0.1985269679553397, 1.9922745799906336e-12,
           6.070497425726566e-8, 0.0626468899375557, 0.005968990494876299,
           2.256352248914527e-17, 0.0004318503319942107, 3.354521672668828e-12,
           1.5493185980108068e-8, 0.05917173072662197, 0.046505609391449744,
           0.09918203188537171, 0.19410088739912074]

    w9t = [8.598863274815516e-9, 1.594733687502297e-6, 5.005868978293449e-8,
           5.5819493108672944e-8, 0.8924108377528648, 3.638226567420588e-11,
           4.451186533713004e-7, 1.2590159805603297e-8, 0.002255892755903678,
           0.01817062655238383, 6.322410963139691e-10, 1.05685248455586e-15,
           7.862447510250934e-11, 1.12926754233254e-10, 9.867406240919562e-11,
           0.02565896088191974, 0.06150130599375543, 1.4813616119662442e-8,
           1.7433653951604228e-7, 1.9034620629860565e-8]

    w10t = [8.598863274815516e-9, 1.594733687502297e-6, 5.005868978293449e-8,
            5.5819493108672944e-8, 0.8924108377528648, 3.638226567420588e-11,
            4.451186533713004e-7, 1.2590159805603297e-8, 0.002255892755903678,
            0.01817062655238383, 6.322410963139691e-10, 1.05685248455586e-15,
            7.862447510250934e-11, 1.12926754233254e-10, 9.867406240919562e-11,
            0.02565896088191974, 0.06150130599375543, 1.4813616119662442e-8,
            1.7433653951604228e-7, 1.9034620629860565e-8]

    w11t = [4.879646527246761e-9, 0.05046935431233224, 2.8407093350726622e-8,
            3.167620963321505e-8, 0.5064215241184382, 1.242553081046635e-10,
            0.014086898147369825, 7.144610585078984e-9, 0.007704485804868947,
            0.051345769365670986, 2.159274876457709e-9, 5.299658893191588e-10,
            2.221736357561518e-10, 3.856755037238032e-10, 3.3699869426455155e-10,
            0.07250598012169168, 0.21004364581151302, 8.406368161008853e-9,
            0.087422247244157, 1.080168593051055e-8]

    w12t = [1.0882961972295098e-8, 1.918083705179376e-8, 0.06012058856566308,
            0.046673447331945055, 0.6105935340349888, 7.77146973370936e-18,
            2.981216163976168e-9, 0.07416512628461075, 3.788497442774938e-9,
            6.982587965979878e-9, 3.923739798834188e-9, 1.3783426364356891e-15,
            1.800056519538815e-17, 1.7845051188805129e-16, 5.072244672310316e-17,
            5.124599215699733e-9, 2.7155942830457237e-8, 0.13808127575944903,
            3.991122343310652e-8, 0.07036590809173547]

    @test isapprox(w1.weights, w1t, rtol = 5e-5)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t, rtol = 5e-8)
    @test isapprox(w4.weights, w4t, rtol = 5.0e-5)
    @test isapprox(w5.weights, w5t, rtol = 5e-5)
    @test isapprox(w6.weights, w6t, rtol = 1e-7)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t, rtol = 5e-5)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t)
    @test !isapprox(w1.weights, w2.weights)
    @test !isapprox(w1.weights, w3.weights)
    @test !isapprox(w1.weights, w4.weights)
    @test !isapprox(w1.weights, w5.weights)
    @test !isapprox(w1.weights, w6.weights)
    @test !isapprox(w1.weights, w7.weights)
    @test !isapprox(w1.weights, w8.weights)
    @test !isapprox(w2.weights, w3.weights)
    @test !isapprox(w2.weights, w4.weights)
    @test !isapprox(w2.weights, w5.weights)
    @test !isapprox(w2.weights, w6.weights)
    @test !isapprox(w2.weights, w7.weights)
    @test !isapprox(w2.weights, w8.weights)
    @test !isapprox(w3.weights, w4.weights)
    @test !isapprox(w3.weights, w5.weights)
    @test !isapprox(w3.weights, w6.weights)
    @test !isapprox(w3.weights, w7.weights)
    @test !isapprox(w3.weights, w8.weights)
    @test !isapprox(w4.weights, w5.weights)
    @test !isapprox(w4.weights, w6.weights)
    @test !isapprox(w4.weights, w7.weights)
    @test !isapprox(w4.weights, w8.weights)
    @test !isapprox(w5.weights, w6.weights)
    @test !isapprox(w5.weights, w7.weights)
    @test !isapprox(w5.weights, w8.weights)
    @test !isapprox(w6.weights, w7.weights)
    @test !isapprox(w6.weights, w8.weights)
    @test !isapprox(w7.weights, w8.weights)
    @test isapprox(w9.weights, w10.weights)
    @test !isapprox(w11.weights, w12.weights)
end

@testset "Mixed inner and outer parameters, $(:HERC)" begin
    portfolio = HCPortfolio(; prices = prices,
                            solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                    :params => Dict("verbose" => false,
                                                                                    "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

    w1 = optimise!(portfolio; type = :HERC, rm_o = :SD, rm = :CDaR, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :SD, rm_o = :CDaR, rf = rf,
                   cluster = false)
    w3 = optimise!(portfolio; type = :HERC, rm = :CDaR, rm_o = :CDaR, rf = rf,
                   cluster = false)
    w4 = optimise!(portfolio; type = :HERC, rm = :CDaR, rf = rf, cluster = false)

    w1t = [0.061732435359141954, 0.06684195886985198, 0.059816424307372225,
           0.0155128049231182, 0.042202981611751615, 0.016416986601765236,
           0.030284606238908746, 0.022828552916191406, 0.03865825582054075,
           0.09276568775491846, 0.07427896560914397, 0.010815514171957245,
           0.010427572590614585, 0.05510503214005871, 0.010052205810662507,
           0.0761884088103524, 0.13229798465965606, 0.05453033048961984,
           0.07588804246885605, 0.05335524884551812]

    w2t = [0.05347326153963788, 0.0952976228086315, 0.0483115366372442, 0.03871246352967465,
           0.04258324980133269, 0.0457752266650704, 0.034139587074985094,
           0.06059201730408066, 0.03685969476575315, 0.051463074362213465,
           0.05968244129623235, 0.02304355480894812, 0.01820715071770149,
           0.051113185215817244, 0.019432795711560417, 0.0357062086855022,
           0.10874681236613978, 0.07014485345792996, 0.0450498773147294,
           0.06166538593681538]

    w3t = [0.0747775890739847, 0.06939399743726035, 0.07245669105243545,
           0.018790934541577802, 0.05112121688219039, 0.01640191163559003,
           0.031440878203812674, 0.02765262927324115, 0.038622757716471855,
           0.05449451662150961, 0.07421075864027389, 0.010805582781120135,
           0.006125600333633228, 0.05505443171528247, 0.010042975330900426,
           0.04475631681027434, 0.13734914660868333, 0.066053552264685, 0.0758183579584723,
           0.06463015511860094]

    w4t = [0.0747775890739847, 0.06939399743726035, 0.07245669105243545,
           0.018790934541577802, 0.05112121688219039, 0.01640191163559003,
           0.031440878203812674, 0.02765262927324115, 0.038622757716471855,
           0.05449451662150961, 0.07421075864027389, 0.010805582781120135,
           0.006125600333633228, 0.05505443171528247, 0.010042975330900426,
           0.04475631681027434, 0.13734914660868333, 0.066053552264685, 0.0758183579584723,
           0.06463015511860094]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test !isapprox(w1.weights, w2.weights)
    @test isapprox(w3.weights, w4.weights)
end
