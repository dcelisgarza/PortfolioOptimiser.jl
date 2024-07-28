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

    w2t = [0.07698450455880468, 0.07670235654605057, 0.06955326092716153,
           0.055733645924582964, 0.061306348146154536, 0.026630781912285295,
           0.0274779863650835, 0.08723325075122691, 0.021443967931440197,
           0.050919570266336714, 0.034721621146346957, 0.013406113465936382,
           0.018014864090601306, 0.029736261019180265, 0.01130547202588631,
           0.03532911363416089, 0.08752722816710831, 0.10098629923304404,
           0.026208793387783046, 0.08877856050082561]
    w3t = [0.013937085585839061, 0.04359528939613315, 0.010055244196525516,
           0.018529417015299345, 0.0060008557954075595, 0.06377945741010448,
           8.830053665226257e-7, 0.13066772968460527, 7.275669461246384e-6,
           1.1522825766672644e-5, 0.2351158704607046, 0.005999396004399579,
           3.2750802699731945e-7, 0.11215293274059893, 5.127280882607535e-7,
           4.000811842370495e-6, 0.037474931372346656, 0.18536090811244715,
           0.04037088344151298, 0.09693547623552375]
    w4t = [3.536321744413884e-9, 8.35986848490829e-9, 1.0627312978295631e-8,
           8.235768003812635e-9, 0.8703125909491561, 1.092557591202154e-18,
           0.12797913827326654, 4.176996018696875e-9, 2.695529928425216e-15,
           0.0007074965433076892, 5.925437102470992e-15, 7.374444588643583e-19,
           3.061344318207294e-12, 2.4076449883278634e-16, 8.597463762416993e-17,
           0.000999064400805917, 1.481241841822526e-6, 4.572656300248877e-9,
           1.827392812927106e-7, 6.3403468197462965e-9]
    w5t = [1.0268536101998741e-9, 4.422568918003183e-9, 4.380602489535938e-9,
           2.956064338817857e-9, 0.4781398601946685, 1.4512115881232926e-11,
           0.06499664007565499, 3.099354900052221e-9, 2.1921024761819662e-10,
           0.028340464844810755, 4.464581793782242e-10, 1.6080765752686552e-11,
           5.332341752653171e-11, 3.3839229356309946e-11, 8.259354624250968e-12,
           0.11005168755732588, 0.27213167660281556, 2.658802782758464e-9,
           0.04633964788062631, 3.5081676586779137e-9]
    w6t = [1.881251684938379e-8, 1.9822795312812863e-8, 2.414660854891215e-8,
           2.2334550044634408e-8, 1.4029799495224622e-6, 3.0636948499419235e-15,
           0.9999980761970937, 1.4630313582860776e-8, 3.638630221309437e-13,
           4.2808125305880613e-14, 6.995666270496466e-15, 3.1932948088578136e-15,
           6.779742676063459e-15, 4.669386166635582e-15, 2.0459303239459237e-15,
           2.2160699849552174e-7, 2.5006723377423327e-8, 1.5281438581326256e-8,
           1.4121190570852067e-7, 1.7968672959672945e-8]
    w7t = [0.03594354904543136, 0.03891177139524374, 0.03393897922702459,
           0.03191180338166861, 0.03154669000464689, 0.05674674429270935,
           0.02049786890241706, 0.05976698398595266, 0.04257331056600222,
           0.11964711125701236, 0.08477538305842097, 0.03508483105116085, 0.043256028579427,
           0.060481128224384596, 0.029559513397421338, 0.08211257818515973,
           0.04003780013424942, 0.05824274268779625, 0.048456009444482875,
           0.04650917317938824]
    w8t = [0.022601387246652074, 0.07069727825830459, 0.016306312144357984,
           0.030048644448520966, 0.009731422312648254, 2.0122167742275325e-11,
           1.4319454456050342e-6, 0.21190025282210542, 2.2954450740334363e-15,
           0.08734103098109872, 7.417813158648544e-11, 1.892785822504333e-12,
           0.0024824543312340127, 3.5383808784753303e-11, 1.617636934698114e-16,
           0.03032546340192833, 0.06077206247830658, 0.3005946715931208,
           1.2736854804055926e-11, 0.15719758789196048]
    w9t = [5.083384080420114e-10, 2.1893691767770292e-9, 2.168593919986438e-9,
           1.463383899260906e-9, 0.23670058997550522, 1.3377400750244865e-10,
           0.03217624032865488, 1.534319128062551e-9, 2.0207000515626317e-9,
           0.03465815910277165, 4.115492208473051e-9, 1.4823396505722006e-10,
           6.521034494875093e-11, 3.119331018877928e-10, 7.613548406809326e-11,
           0.13458455666752087, 0.13471733642262038, 1.3162261499202676e-9,
           0.42716309971451744, 1.7366997058224099e-9]
    w10t = [0.09046786560242673, 0.09793871218692884, 0.0854224775498144,
            0.08032019141501044, 0.0794012218419085, 6.983610695750584e-12,
            0.05159196846855984, 0.15043009436461915, 5.239338938788118e-12,
            2.8069873396589884e-10, 1.0432990989039082e-11, 4.317759625532709e-12,
            1.014810331070678e-10, 7.443187432572064e-12, 3.637779338645444e-12,
            1.926406454539399e-10, 0.10077286238439759, 0.14659366583795952,
            5.963300803379451e-12, 0.11706093972953666]
    w11t = [4.5601306178443335e-10, 1.9640084752282713e-9, 1.945371700377978e-9,
            1.3127518241998705e-9, 0.21233603255882913, 1.1249136381386817e-10,
            0.02886420863907858, 1.3763854004995863e-9, 1.6992187712913765e-9,
            0.05708253091653823, 3.460742037559184e-9, 1.2465082869344956e-10,
            1.0740245956449301e-10, 2.6230641292110363e-10, 6.40227843760461e-11,
            0.2216628729206791, 0.12085033136517515, 1.1807416223726956e-9,
            0.35920400797565893, 1.557934119718829e-9]

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

    w2t = [0.0797145490968226, 0.07455683155558127, 0.070334401927198, 0.05349383521987859,
           0.06513229159697068, 0.026966237124386894, 0.029595506806349627,
           0.09315224544455303, 0.020640678885583085, 0.047472627309343436,
           0.032958383881734096, 0.014663096865841139, 0.017152340172727078,
           0.029195401656427377, 0.010527045845271121, 0.03720496223361964,
           0.08565021870448736, 0.09947562485956811, 0.025808052654604105,
           0.08630566815905283]
    w3t = [0.04059732374917473, 0.045600643372826564, 0.0008508736496987,
           0.006437852670865818, 0.002841800035783749, 0.05930362559033975,
           3.682328495715141e-9, 0.15293340032515007, 0.005759786022651817,
           5.950241253633107e-8, 0.1987549961908217, 2.0545559421162275e-9,
           5.32244045270275e-10, 0.10473908918633106, 2.2579839445845772e-10,
           2.187558110026064e-8, 0.056505427326554836, 0.19033548239854162,
           0.04096913911515197, 0.09437047249318714]
    w4t = [0.019135840114057178, 0.04319664563133952, 6.989818506731821e-8,
           0.010479643822795195, 0.044710601991195954, 0.023524270507582417,
           0.0027081241614039355, 0.15841866168441052, 0.004902231961856615,
           0.010598283527252079, 0.17605850844462068, 2.835425217252423e-10,
           6.674569801134214e-12, 0.07184677071156634, 1.9086830233118526e-11,
           0.00482767668105368, 0.07950214846405654, 0.1805692837235624,
           0.06741985554913642, 0.10210138281662147]
    w5t = [1.7711462588275667e-9, 6.253014446612376e-9, 4.5219875754910405e-9,
           3.3567391619124057e-9, 0.626542835057035, 4.244688293643053e-12,
           0.0410106345340863, 3.5790876174967654e-9, 6.627977950621314e-10,
           4.951758424642855e-9, 1.6115364648536767e-10, 4.154724183802552e-12,
           2.0959897320947726e-12, 9.858343606151467e-12, 1.5411541422962657e-12,
           0.13259044906595502, 0.1453951017104159, 3.983210451398619e-9,
           0.05446094499291796, 5.376799673344027e-9]
    w6t = [1.0697281800757452e-7, 1.1311775297941919e-7, 1.3780259688825507e-7,
           1.268055655932939e-7, 8.50822923994295e-6, 3.204283203701939e-13,
           0.9999862377026062, 8.136173647277906e-8, 4.024041566311586e-11,
           6.94984039611021e-12, 8.035455863957799e-13, 3.4976521519678484e-13,
           1.137123602399115e-12, 5.126371707340376e-13, 2.1878366160025683e-13,
           2.7532186682419475e-6, 1.431604095722679e-7, 8.5069527979556e-8,
           1.6044460686057254e-6, 1.0206247702601848e-7]
    w7t = [0.038482572880612374, 0.04124259860035183, 0.036310167752490984,
           0.03068785441299921, 0.03272027360303007, 0.0569582645840768,
           0.01903357698244691, 0.06024480428552155, 0.041510930079800556,
           0.10991717839342878, 0.08278034364647732, 0.0346366457548056,
           0.039589293259183145, 0.058854479193534, 0.026495842632932323,
           0.08543930360245622, 0.043229183172491234, 0.06156421651978086,
           0.0479372991636419, 0.0523651714799385]
    w8t = [0.05800434349697815, 0.06515294944619106, 0.0012157049502716088,
           0.009198227449941948, 0.004060286004164899, 5.313690390371367e-11,
           5.261210030833962e-9, 0.21850705084472974, 5.1608513534366084e-12,
           0.11357653379214888, 1.780873433595776e-10, 1.8409117583329943e-18,
           0.0010159324843576638, 9.384773463096937e-11, 2.023186182719046e-19,
           0.04175549477997425, 0.08073340588515147, 0.2719461206093044,
           3.6708939571720295e-11, 0.13483394462863404]
    w9t = [8.430155713238247e-10, 2.9762581830461174e-9, 2.1523383066033683e-9,
           1.5977129885579609e-9, 0.2982166850546235, 3.76273164629871e-11,
           0.01951990318687239, 1.7035445704407011e-9, 5.875414320321845e-9,
           4.865726617183371e-9, 1.428557019027885e-9, 3.682982373864523e-11,
           2.059574024864136e-12, 8.738993042788557e-11, 1.3661661497564264e-11,
           0.13028682376625528, 0.06920395993565708, 1.8958956199425605e-9,
           0.48277260198135563, 2.559204710467201e-9]
    w10t = [0.09253278355679952, 0.09916936846831717, 0.08730915430150583,
            0.07379009192109516, 0.07867711976071654, 9.408399233959074e-12,
            0.04576694663042537, 0.14486088164293254, 6.85679954639816e-12,
            2.1564751263015437e-10, 1.367370525484927e-11, 5.72130126797308e-12,
            7.767059474153108e-11, 9.721617064757003e-12, 4.376598678894266e-12,
            1.6762414730820425e-10, 0.10394618525760373, 0.14803345763138517,
            7.918310925072676e-12, 0.12591401031059996]
    w11t = [7.821463045237047e-10, 2.761359835290349e-9, 1.996930436232687e-9,
            1.4823513968167393e-9, 0.2766841872167342, 2.936102474771157e-11,
            0.018110484149536667, 1.5805414937582428e-9, 4.584652892579249e-9,
            9.870065205228937e-9, 1.1147193563605256e-9, 2.873873206737773e-11,
            4.177819988615019e-12, 6.819136072374175e-11, 1.0660350485515304e-11,
            0.2642851822816762, 0.06420714321691791, 1.7590039891815271e-9,
            0.3767129746878156, 2.374419375988825e-9]

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

    w2t = w2t = [0.07837975159477181, 0.07443043390995964, 0.06801950779661087,
                 0.055548455549923444, 0.0644736826243098, 0.02739674778042435,
                 0.029790140405552036, 0.08281583015461562, 0.021850035315318022,
                 0.05161041880598439, 0.035414428250310895, 0.013356805631372573,
                 0.02005310749097011, 0.03085778614009161, 0.012088382453633814,
                 0.03792292849371053, 0.08401306865550462, 0.10203433260227453,
                 0.027069661454433437, 0.08287449489022782]
    w3t = [7.121013120801824e-6, 0.061607987382417705, 1.4912756768376158e-9,
           0.009681807066376894, 0.009020659057053952, 0.059478722069370014,
           1.3179217642229542e-9, 0.11544692256445871, 1.5887557238653072e-9,
           7.210694103149243e-10, 0.24458547584840604, 0.009620637405741693,
           1.9394792891670725e-11, 0.12245874714177621, 5.062813106756657e-10,
           2.792879935708652e-10, 0.029931480605545943, 0.2164485615370284,
           0.03511997743484373, 0.08659189494987324]
    w4t = [2.505966264249908e-9, 0.06462231159494408, 6.312604456450302e-10,
           0.008290344916341286, 0.025721971719375243, 0.03177714442542715,
           3.100206836273914e-9, 0.12015839241220808, 5.753553216235469e-10,
           8.579825113335541e-10, 0.23290956247196232, 0.007185501507127438,
           4.740916222504663e-12, 0.10213169998624982, 1.0583051387480673e-10,
           3.598662360154047e-10, 0.0423881614274072, 0.21855640410841734,
           0.058728073140139496, 0.0875304241491915]
    w5t = [3.376211111543454e-9, 9.027836247432593e-9, 6.624231451997948e-9,
           4.456498283828095e-9, 0.6369289883278642, 1.2292588648044406e-15,
           0.041553493600651026, 8.454859532581521e-9, 3.335199701461747e-14,
           0.018745128572890082, 2.92766141429812e-14, 1.3113761446096982e-15,
           1.1080502561018842e-11, 3.1664656065420014e-15, 4.805195416637262e-16,
           0.14899701281311017, 0.15377296833875437, 1.0423893797290456e-8,
           2.357572635201043e-6, 8.399415392270166e-9]
    w6t = [3.169138468128157e-7, 3.361556961865357e-7, 4.168074813823495e-7,
           3.81870053665129e-7, 2.07656774406466e-5, 1.4691116695512883e-13,
           0.9999738665006, 2.3428043709562976e-7, 1.785834933043541e-11,
           2.4544969294036332e-12, 3.7784102629307113e-13, 1.6211460037242008e-13,
           4.1719665878120677e-13, 2.3542525005240866e-13, 1.0286912965828565e-13,
           1.7063596294143836e-6, 4.329282002176093e-7, 2.458728199437001e-7,
           9.967089115920972e-7, 2.9990312775869386e-7]
    w7t = [0.03596400232271158, 0.03926422387565698, 0.03173743270063896,
           0.030431800778782426, 0.031103389684032656, 0.055774400848792,
           0.02012838723950023, 0.05939418239844191, 0.04130776241819249,
           0.11737585383647636, 0.08621821890143008, 0.034941660524425426,
           0.047599879413746715, 0.060741559517526224, 0.03121118004196171,
           0.08601968722906522, 0.038070323373188976, 0.058782688335425784,
           0.04855866178072313, 0.045374704779281204]
    w8t = [1.13954626102641e-5, 0.0985887127014467, 2.386426752586273e-9,
           0.015493395188726417, 0.014435387389515705, 7.162650790088209e-11,
           2.10901566008699e-9, 0.18474493267121114, 1.9132392299096245e-18,
           0.10881245230852339, 2.945390033408029e-10, 1.1585532391738185e-11,
           0.0029267570449797646, 1.4746941619647781e-10, 6.096829426981602e-19,
           0.04214575052837179, 0.04789810976670943, 0.3463736757955762,
           4.2292794022739487e-11, 0.13856942607937353]
    w9t = [1.46029101751157e-9, 3.9047523226900255e-9, 2.8651364999649027e-9,
           1.9275407249207304e-9, 0.27548682523668083, 2.368777982442693e-10,
           0.017972867053184627, 3.656926365634336e-9, 6.426919378880274e-9,
           0.02075485508451866, 5.641594375924787e-9, 2.527017723436324e-10,
           1.226847946245519e-11, 6.101769306444658e-10, 9.259596518000859e-11,
           0.16497146962409392, 0.06651043936009031, 4.50858017368441e-9,
           0.45430350841212447, 3.6329454659229693e-9]
    w10t = [0.09215604767937866, 0.10061270864988707, 0.0813256637828132,
            0.07798004399993534, 0.0797009586694936, 7.592994736751303e-11,
            0.051578036212642585, 0.15219477120131153, 5.623540869199711e-11,
            7.309568921968852e-10, 1.1737543969417814e-10, 4.7568748461250273e-11,
            2.964277471726999e-10, 8.26921194490307e-11, 4.2490160751147795e-11,
            5.356866952576383e-10, 0.09755339532200438, 0.15062784671050244,
            6.610661122551585e-11, 0.11627052572056149]
    w11t = [1.4297931302039606e-9, 3.823202347463541e-9, 2.8052986943158963e-9,
            1.887284420455039e-9, 0.26973333771258684, 1.9002683353876338e-10,
            0.017597507301320425, 3.5805521862044078e-9, 5.155768704495214e-9,
            0.03163619943991698, 4.52576950357153e-9, 2.0272105695011313e-10,
            1.8700591332399527e-11, 4.894928561823317e-10, 7.428183726816752e-11,
            0.251462623741325, 0.06512138206941398, 4.414419373949669e-9,
            0.36444891758105297, 3.557072122811393e-9]

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

    w2t = [0.08074577526070313, 0.0758057372636474, 0.0723712035349317, 0.05388227729039704,
           0.0719726576355769, 0.023357700765219442, 0.030259361503984363,
           0.09069744656455722, 0.020432554010122735, 0.045981867419973635,
           0.03127471464903142, 0.013225268293206591, 0.01541923284313144,
           0.026677624195275505, 0.00924248201367476, 0.03763591530781149,
           0.08961751171161211, 0.09815629750095696, 0.02604005979180501,
           0.08720431244438119]
    w3t = [0.016087285360179032, 0.03929456125680425, 2.654669650722757e-8,
           0.009017145085378016, 0.04852710668341847, 0.022953259582093585,
           0.00113984978745799, 0.1456189117438582, 0.003637704859085449,
           0.014589771845411147, 0.17831030715846177, 1.2587617733789283e-9,
           2.1033525434368566e-11, 0.07185635646074559, 1.4705271853683893e-10,
           0.006896966942305214, 0.07311971752422704, 0.188190292120055,
           0.07385752447075629, 0.10690321114621834]
    w4t = [8.979792049815534e-9, 0.04319677159738373, 0.010553521731731526,
           0.011677797169390153, 0.07462748970960995, 2.4993569347447565e-9,
           0.0051395703992312735, 0.1393795754745042, 0.003947096902484896,
           0.013483746896893172, 0.16322712734000247, 7.880340298212877e-11,
           3.061723277788993e-12, 0.044267239650700875, 4.5346573333915216e-12,
           0.007528180962873085, 0.08925135943043455, 0.18949901379313963,
           0.08920985591556879, 0.11501164146050306]
    w5t = [4.70569028136048e-10, 1.3574602442249478e-9, 1.2171599066627055e-9,
           7.480173861691719e-10, 0.6552130462915482, 2.118918252960233e-12,
           0.037643941367949886, 8.865144930684747e-10, 3.4692335767765595e-10,
           8.838695321277905e-9, 7.803488985844738e-11, 2.036777057328308e-12,
           1.1596869403076888e-12, 4.9035154832726276e-12, 8.157197134641697e-13,
           0.1480747059921399, 0.11655387517352318, 1.0532930709570267e-9,
           0.042514414741941266, 1.4251951319177548e-9]
    w6t = [1.0701351024251811e-7, 1.1315899899963757e-7, 1.3784659823204616e-7,
           1.2684858651859715e-7, 8.51228975107345e-6, 3.206232650541434e-13,
           0.9999862315870663, 8.139959013478876e-8, 4.026227025375141e-11,
           6.954578268366241e-12, 8.040123708309039e-13, 3.4997831995900936e-13,
           1.1378946431508748e-12, 5.129468913623559e-13, 2.1891375482044552e-13,
           2.7542962972698258e-6, 1.432048377198796e-7, 8.510789499184016e-8,
           1.6050937773801256e-6, 1.0210252984489028e-7]
    w7t = [0.040183831915019924, 0.0434257369874625, 0.037998059938328436,
           0.0317472517897774, 0.03729164492388537, 0.049675475199063394,
           0.02023805337478654, 0.060464070967644855, 0.04287003746789862,
           0.10829254014258542, 0.08072109742895471, 0.031429506535522345,
           0.03707944602714823, 0.05568975882884374, 0.023775343473611613,
           0.08922090474464904, 0.045103020475796764, 0.061381817706047864,
           0.05045637642477227, 0.05295602564820106]
    w8t = [0.023204642673463116, 0.05667931118022407, 3.829151984435438e-8,
           0.013006521918166704, 0.06999653113340545, 5.003155071659266e-11,
           0.0016441435846508278, 0.21004381645886253, 7.929157708446429e-12,
           0.06403526482105995, 3.886655463457313e-10, 2.7437411788801682e-18,
           9.231723330436599e-11, 1.5662633578108312e-10, 3.205329299385556e-19,
           0.030271145381311797, 0.10546943623777214, 0.27144968125386193,
           1.6098831053360938e-10, 0.15419946620914352]
    w9t = [2.4551295254815234e-10, 7.082363109330024e-10, 6.350365292668578e-10,
           3.9026783756505473e-10, 0.3418484428953947, 2.174469971891849e-11,
           0.019640211399809544, 4.6252680830291253e-10, 3.560186537465805e-9,
           8.440920346850948e-9, 8.008073200559956e-10, 2.090175279018548e-11,
           1.1074966083348616e-12, 5.03207103916679e-11, 8.371054522419792e-12,
           0.14141077989803547, 0.06081038979154262, 5.495412496089864e-10,
           0.43629015937615995, 7.435760619018458e-10]
    w10t = [0.09327950330164496, 0.10080500001256203, 0.08820562869586782,
            0.07369550729769486, 0.08656581391136879, 1.5271682041767413e-11,
            0.046978983253377765, 0.14035641298183862, 1.317949307389308e-11,
            4.925807085111835e-10, 2.4816006873765508e-11, 9.662342003105748e-12,
            1.6865999976735617e-10, 1.712064728940623e-11, 7.309230249087832e-12,
            4.058312457650584e-10, 0.10469851049256655, 0.14248679616918566,
            1.5511753730609096e-11, 0.12292784271394983]
    w11t = [2.2043143250733723e-10, 6.358831293923458e-10, 5.701614126176355e-10,
            3.5039820752077555e-10, 0.3069253218037665, 1.7997832815607427e-11,
            0.017633774058245847, 4.152752262928413e-10, 2.9467246235614036e-9,
            1.550352408553005e-8, 6.628188225263423e-10, 1.7300135533424816e-11,
            2.0341502628167806e-12, 4.1649861552417105e-11, 6.92862360632034e-12,
            0.25973061491104893, 0.05459802097590054, 4.934003017597317e-10,
            0.3611122456988977, 6.676125833770062e-10]

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

    w2t = [0.07895203150464367, 0.07513349407323168, 0.06901069326760172,
           0.05594112932547382, 0.06763676613416904, 0.025518328924508955,
           0.030140432803640472, 0.08170116913851065, 0.021723048018546902,
           0.05096388176098718, 0.0345040083464426, 0.012730715665077913,
           0.018968393009821956, 0.029479897263328572, 0.01127249390583484,
           0.03813269633505064, 0.0862014575477909, 0.1015604380650207, 0.02714041935142286,
           0.08328850555889497]
    w3t = [5.889098113924682e-9, 0.06587618283494774, 1.5750531512406229e-9,
           0.008307801198717074, 0.025055535680020887, 0.031007017694062042,
           6.622189397331921e-9, 0.11895069126754958, 5.227655044085646e-10,
           1.0425338132041355e-9, 0.2342323891676781, 0.007211234975414065,
           5.564426798819609e-12, 0.10219179342324933, 1.0936729193190541e-10,
           4.3817799370420595e-10, 0.03983727337898306, 0.22064984507420685,
           0.059770315866896725, 0.08690990323352493]
    w4t = [1.3126689349585126e-9, 0.0683982344676695, 6.641537637834876e-10,
           0.005752802760099117, 0.04380613474339462, 0.006467953935425663,
           0.001194218858958455, 0.12252327476998784, 4.934350529800896e-10,
           5.070902306913777e-10, 0.22365749118830133, 0.005234400227547068,
           5.170702817426386e-19, 0.08421644705458561, 5.950345115515079e-11,
           2.284721290071047e-10, 0.05254721479863481, 0.2198834885600653,
           0.07900905406850742, 0.08730928130149963]
    w5t = [2.0146575390229065e-9, 5.441471606993735e-9, 3.941196024805785e-9,
           2.6032258222664272e-9, 0.6332890387116795, 4.488189568435251e-17,
           0.04170782181673219, 5.14894387005817e-9, 1.1896470812621962e-15,
           0.019835822335139475, 1.0623827899795569e-15, 4.8011381381596766e-17,
           1.1768078009951815e-11, 1.160757066681121e-16, 1.762740290937585e-17,
           0.1503590248352136, 0.15480805233160272, 6.412266352587129e-9,
           2.0941020989105202e-7, 4.9858908525016265e-9]
    w6t = [3.590022276972911e-7, 3.7987197002485197e-7, 4.669822555594107e-7,
           4.2943174301178453e-7, 2.4464659432521346e-5, 1.5740255311730082e-13,
           0.999969595458026, 2.6835123811842975e-7, 1.905346857630097e-11,
           2.6864639955607387e-12, 4.034356965950875e-13, 1.7357945988136158e-13,
           4.589237697745928e-13, 2.5158220448101467e-13, 1.1050657216281399e-13,
           1.8543896347411247e-6, 4.842533754688074e-7, 2.8119376275198016e-7,
           1.0759705564579637e-6, 3.4041248239753164e-7]
    w7t = [0.03651367301522128, 0.04020193126064295, 0.032440500747771495,
           0.03111511622029599, 0.033302134728400365, 0.0520110878935054, 0.020990313244494,
           0.05987395308080474, 0.041723833306903, 0.1172587304056323, 0.08535502205077988,
           0.033461836634349776, 0.045698746344584214, 0.0586004642617431,
           0.029201398506925658, 0.08770180110993163, 0.03939357780871859,
           0.059462235315846254, 0.04953271898182618, 0.04616092508162323]
    w8t = [8.930197630280675e-9, 0.09989429968123478, 2.388402374468406e-9,
           0.012597906358904825, 0.037994083478772395, 6.396479694384992e-11,
           1.0041853424634532e-8, 0.18037620714214933, 1.0784200424778912e-18,
           0.09984751509865623, 4.832011694453475e-10, 1.4876154342475936e-11,
           0.000532926780670009, 2.1081283534397435e-10, 2.256152684449488e-19,
           0.041966009436005564, 0.06040903334326548, 0.3345922729567689,
           1.2330099448512832e-10, 0.131789723466963]
    w9t = [9.58830168313611e-10, 2.5897439320322204e-9, 1.875722089052525e-9,
           1.2389457786123902e-9, 0.30139943083010523, 8.949191237930231e-11,
           0.019849883683917127, 2.4505220475136437e-9, 2.3720876922702357e-9,
           0.021855196116484538, 2.1183300327324584e-9, 9.573192643262366e-11,
           1.2966120006326298e-11, 2.3144826688169208e-10, 3.5148025113174994e-11,
           0.16566623355138332, 0.07367735111219081, 3.05177148325898e-9,
           0.4175518852122588, 2.372920693814788e-9]
    w10t = [0.09140887306415299, 0.10064211370918542, 0.08121203292132384,
            0.07789404554758862, 0.08336906026640174, 7.807552413668657e-11,
            0.052547462922804884, 0.14988934623834815, 6.263299397040623e-11,
            7.92112118012719e-10, 1.281291808000276e-10, 5.023064388024975e-11,
            3.0870648720409374e-10, 8.796704985784526e-11, 4.383516258342232e-11,
            5.924476513638027e-10, 0.09861846963352311, 0.14885864585113545,
            7.435516450530137e-11, 0.1155599476270438]
    w11t = [8.74609690071842e-10, 2.362269370126139e-9, 1.7109648498570807e-9,
            1.13012086942761e-9, 0.2749254993194251, 7.686862854138458e-11,
            0.018106335397526336, 2.2352762765690498e-9, 2.0374928061866106e-9,
            0.03276253042824791, 1.8195289393752012e-9, 8.222845726339001e-11,
            1.9437158055203482e-11, 1.988013260692232e-10, 3.0190219591427273e-11,
            0.24834574756190614, 0.06720577569527518, 2.7837139457529236e-9,
            0.3586540940716257, 2.164491169726737e-9]

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

    w2t = [0.09799469653109426, 0.07931044806875052, 0.07697883394077526,
           0.05938016484003881, 0.062482774302560766, 0.024174945904638732,
           0.021508124985828447, 0.05117445754264079, 0.023413964946189022,
           0.07738784752841718, 0.02842727729869877, 0.006737256345835118,
           0.024299987114092203, 0.030479447210160812, 0.014669862619463056,
           0.033478912079376394, 0.10842038272038522, 0.0983414761742503,
           0.02496318167060388, 0.05637595817620051]
    w3t = [1.9747133396600235e-11, 0.28983188049620723, 2.2178320507471025e-11,
           2.3903571001566017e-9, 8.896231855698167e-11, 1.5285685327330832e-10,
           0.15054354915956833, 0.24258042615731373, 3.844939099024774e-20,
           7.052877593110094e-12, 5.164211395727411e-10, 5.2576871331120963e-20,
           1.4150944216989741e-12, 4.725567845330821e-20, 1.8628408346794537e-10,
           2.3466054659808233e-12, 2.582105425683964e-10, 9.538296837205346e-11,
           1.3506383389403547e-19, 0.3170441404456957]
    w4t = [1.141783373321512e-11, 0.28983188080893435, 8.285628164196383e-12,
           2.486875189454384e-9, 7.827728961310542e-11, 7.65738747390303e-11,
           0.15054354925197597, 0.24258042607837926, 2.3157763218355765e-20,
           1.2910398707352923e-12, 2.587019605725475e-10, 4.3942020776527333e-20,
           2.59035166323434e-13, 2.282609616740485e-20, 9.331929681572565e-11,
           4.295496663376929e-13, 2.666566597352273e-10, 1.0315442168985836e-10,
           1.8563920743873889e-19, 0.3170441404754686]
    w5t = [4.402292614352855e-9, 5.118496511304611e-8, 4.2768740184544925e-9,
           1.1806247921427134e-8, 0.5977202415770091, 5.162354964911019e-11,
           0.1966217669950393, 5.8367058605025425e-8, 1.3635578134809385e-9,
           0.028193636458432692, 0.02200162915850006, 7.512726824369969e-11,
           1.4929495461767638e-11, 1.457593147146519e-10, 2.3261104388259625e-11,
           0.011827130525631805, 0.012053559096343503, 0.10170500941174024,
           0.029876892099256584, 2.966349923966149e-9]
    w6t = [2.469260820591323e-9, 2.6086071830057514e-9, 3.1707427364237383e-9,
           2.9195261819125968e-9, 2.0428633058799499e-7, 7.36816518037098e-18,
           0.9999997520392511, 1.8969261383610923e-9, 9.079017212568097e-16,
           1.5459544528864506e-16, 1.8269672717801768e-17, 7.96162684815466e-18,
           2.4680933724024175e-17, 1.1734067342084149e-17, 4.944690968749843e-18,
           1.4275068260027735e-8, 3.292675858335107e-9, 1.981003055125446e-9,
           8.699313274091695e-9, 2.3612935845290182e-9]
    w7t = [0.03119363952440829, 0.05706544449319162, 0.03036361882099508,
           0.03989035207471949, 0.035078474806200605, 0.04895936407693799,
           0.027872256279024486, 0.03485249409732566, 0.039052219914194025,
           0.1579067332918187, 0.05717436110021421, 0.03859948119378035,
           0.07112283583631317, 0.03764608078209501, 0.04678339289819475,
           0.08546347524280776, 0.042331906477832595, 0.029078372453158717,
           0.04127271389202918, 0.048292782744758324]
    w8t = [1.974713339679921e-11, 0.28983188049912767, 2.21783205076945e-11,
           2.3903571001806876e-9, 8.896231855787809e-11, 1.2274422818471654e-11,
           0.15054354916108525, 0.24258042615975806, 3.0874904985987014e-21,
           5.136434346049085e-10, 4.1468676632895576e-11, 4.2219287874301e-21,
           1.0305778732949605e-10, 3.794636390116836e-21, 1.4958633230194188e-11,
           1.7089740680974092e-10, 2.5821054257099817e-10, 9.538296837301457e-11,
           1.0845641325188149e-20, 0.3170441404488903]
    w9t = [3.639212045004217e-18, 4.231271246150467e-17, 3.5355331429310676e-18,
           9.759787321243212e-18, 4.941131572212958e-10, 9.596938715746581e-10,
           1.6253992304497154e-10, 4.8249882802963156e-17, 2.5348858922527676e-8,
           0.025055444712276446, 0.40901543600915, 1.3966335017969217e-9,
           1.3267715524246204e-11, 2.709699778635513e-9, 4.3242937533753894e-10,
           0.010510670215484443, 9.964230297997063e-12, 8.407575954441219e-11,
           0.5554184174518128, 2.452171474880904e-18]
    w10t = [0.08295753981370832, 0.151761992371126, 0.08075015149354371,
            0.10608590471887734, 0.09328901707870688, 6.717421361347478e-12,
            0.07412452811592576, 0.09268803547029422, 5.358121397316421e-12,
            1.1815549362941528e-10, 7.844551942554426e-12, 5.296003826776969e-12,
            5.321846384494134e-11, 5.1651934616482816e-12, 6.4188693774042995e-12,
            6.394900897568279e-11, 0.1125790664560675, 0.07733211889600791,
            5.662782088088739e-12, 0.12843164530795637]
    w11t = [1.5479035469067402e-9, 1.799730185777955e-8, 1.5038047314835686e-9,
            4.151230877669371e-9, 0.2101662390588305, 3.6930255702128615e-10,
            0.06913477980505216, 2.0522619678363553e-8, 9.754567258308327e-9,
            0.21808674533752423, 0.1573944055009052, 5.374425519555606e-10,
            1.1548439590574342e-10, 1.0427273598904712e-9, 1.6640439078895089e-10,
            0.09148661637954987, 0.004238188714955023, 0.035760808877934744,
            0.21373215757345154, 1.0430073534193542e-9]

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

    w2t = [0.07701588327689891, 0.07520448896018464, 0.06541770521947644,
           0.05757000169964883, 0.06337709857771338, 0.025926148616162506,
           0.030149122854382444, 0.08115573202397372, 0.021487151979485357,
           0.053917172483500996, 0.03629492645395641, 0.013834276128551553,
           0.02188007667915268, 0.030097549937467067, 0.012085783569165435,
           0.04006561840657164, 0.08240686258336018, 0.10281015669469054,
           0.02647712574837549, 0.08282711810728184]
    w3t = [3.664859661161272e-10, 0.0762203721651536, 6.441778816535088e-11,
           0.03138093007735649, 0.027626085711376314, 0.02300652225536527,
           5.844065587830882e-11, 0.13924934619748974, 5.805427791614419e-11,
           1.0937171459308286e-10, 0.21362141189989498, 0.020888781049182764,
           1.980200934283383e-12, 0.10087916256515272, 7.26276933420973e-11,
           4.9799871178339534e-11, 0.014862212321435466, 0.28800164310949844,
           1.9196981699887677e-10, 0.06426353167494636]
    w4t = [1.5789168760766128e-10, 0.07694081488688548, 4.000845881532486e-11,
           0.03378345968044639, 0.027373933478449422, 0.022402731945984695,
           3.630403456173319e-11, 0.14125562537830114, 6.968211398613603e-11,
           2.3479950495644183e-10, 0.21025793521634595, 0.020157628966454504,
           1.2002752293501236e-19, 0.09889797348307146, 5.720839942268613e-11,
           1.0812540256939718e-10, 0.01696672848323463, 0.28844538572619727,
           5.044951919517896e-10, 0.06351778154611436]
    w5t = [7.550077262684875e-10, 1.3306533988959414e-9, 1.1093698572328e-9,
           6.709440849073084e-10, 0.5763326871574024, 2.5772533724661772e-18,
           0.073479023506859, 2.4127955663343554e-9, 5.071595466915273e-17,
           0.029877775100693744, 2.1496819981795957e-17, 2.552000792062951e-18,
           8.626232362752768e-13, 3.4515273144533547e-18, 3.189132459473233e-18,
           0.14923907048182963, 0.17107141857002867, 2.3942026704968923e-9,
           1.3484144929514954e-8, 3.0252058041577887e-9]
    w6t = [2.6743919524671433e-8, 2.8280194662924554e-8, 3.445156994729403e-8,
           3.170223879757576e-8, 2.127112149105521e-6, 1.2520005503949514e-15,
           0.999997376413533, 2.0340977818722356e-8, 1.5723018780652422e-13,
           2.7153417265778693e-14, 3.1396703010908747e-15, 1.3666277289161734e-15,
           4.44280727298183e-15, 2.00301248454596e-15, 8.548473940793335e-16,
           1.7209117914321296e-7, 3.5791059101939696e-8, 2.1267950571210223e-8,
           1.0028872937847331e-7, 2.551630147362986e-8]
    w7t = [0.03442967428619515, 0.03622525747192852, 0.028649166159358926,
           0.028820218103751707, 0.03064487884114398, 0.051338672771389905,
           0.021297430232675357, 0.060896416149046416, 0.038724170445278865,
           0.1180505650585966, 0.08629247235515324, 0.03352163294024383,
           0.05457013256128135, 0.06613966646033598, 0.032097566038840224,
           0.08807616383700578, 0.03507071652584228, 0.06546119522707869,
           0.046780168185706385, 0.04291383634914687]
    w8t = [5.130289583938887e-10, 0.10669783226540461, 9.01758698013453e-11,
           0.04392890140277927, 0.03867264585110126, 3.646089522385327e-12,
           8.180872280278326e-11, 0.1949295570409142, 9.200482023758986e-21,
           0.06912039596643586, 3.3854868764605514e-11, 3.310468434709418e-12,
           0.001251441226646273, 1.5987399293695193e-11, 1.1510086956662074e-20,
           0.03147236767505174, 0.02080501304729041, 0.4031619124356221,
           3.0423509064287326e-20, 0.08995993234694181]
    w9t = [2.9903950310459343e-10, 5.270382240681567e-10, 4.393933986685305e-10,
           2.657440166252003e-10, 0.22827083007783266, 9.096748677772005e-11,
           0.029103186515323173, 9.556474220675886e-10, 1.7900851290266946e-9,
           0.033182494510344746, 7.587580283524162e-10, 9.007616433403822e-11,
           9.58036222768661e-13, 1.218261148456779e-10, 1.1256454950796128e-10,
           0.16574609790396602, 0.06775707085464627, 9.482832453326168e-10,
           0.47594031253929814, 1.1982076593250298e-9]
    w10t = [0.08956526288051826, 0.09423628819191746, 0.07452786445323638,
            0.07497283852526544, 0.0797195061020141, 5.381156481200371e-11,
            0.055403078217149924, 0.15841577458830872, 4.058944447174081e-11,
            2.6520133169447285e-11, 9.044902640169386e-11, 3.5136310040593465e-11,
            1.2259214361923846e-11, 6.932549589318731e-11, 3.364364838370282e-11,
            1.9786365214364175e-11, 0.0912328684533811, 0.170290578709804,
            4.903348521398524e-11, 0.11163593944784997]
    w11t = [3.0787320067555094e-10, 5.42607057721964e-10, 4.523731834736014e-10,
            2.7359415765936813e-10, 0.23501400432820488, 6.715672671430337e-11,
            0.02996290152072703, 9.838774726908922e-10, 1.3215299450795406e-9,
            0.05236069336895692, 5.601529442806046e-10, 6.649870812013323e-11,
            1.5117441180049004e-12, 8.993810196546335e-11, 8.310075343178885e-11,
            0.26154093408333434, 0.06975863073556453, 9.762957564353538e-10,
            0.3513628290030996, 1.2336029966627194e-9]

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

    w2t = [0.08699441730392812, 0.07712958717175648, 0.07366521524310121,
           0.06113285849402033, 0.06395471849180502, 0.025641689880494402,
           0.024561083982710123, 0.05538393000042802, 0.023397235003817983,
           0.06889218377773806, 0.032808220114320844, 0.0078354841769452,
           0.024451109097834427, 0.03131096206947751, 0.01392669730115599,
           0.03521555916294673, 0.09758384164962133, 0.10406746553380812,
           0.027502023704060772, 0.06454571784002953]
    w3t = [1.3785795089567192e-8, 0.1360835835861814, 9.796765065975505e-9,
           0.03979459018377941, 1.6617234434301553e-8, 0.05740929288219373,
           0.04390484034087705, 0.14120918087952655, 7.207773383460931e-9,
           1.0722818337316549e-8, 0.18078058609599582, 0.015910536201136125,
           1.6767734194460038e-9, 0.03652307629311764, 0.019099392747369295,
           3.1989632090928942e-9, 0.04716014099993904, 0.14062385947717723,
           0.010742934281030967, 0.13075792302555284]
    w4t = [1.539901271958622e-8, 0.14215345217490227, 1.1592191304365884e-8,
           0.04131131228118324, 2.2160183751107717e-8, 0.04471285118969558,
           0.04800165632918703, 0.1485469036262668, 1.8677812696498027e-8,
           1.9790701451501078e-8, 0.16069375067272876, 0.013510808701786338,
           3.0208950773188386e-9, 0.031079294902898496, 0.01396492760596718,
           5.921045055010991e-9, 0.059404804181166615, 0.14015472932859627,
           0.019139880880730844, 0.13732553156304858]
    w5t = [1.5891310507284716e-8, 0.0028908394423298495, 1.8841580627264263e-8,
           2.617260936163577e-8, 0.5396359787197313, 4.835893610840537e-16,
           0.14358082220438478, 6.771720274363291e-8, 8.677275256467885e-15,
           0.052940532835842276, 2.308873038393805e-14, 6.065872532873969e-16,
           1.7609411452779525e-10, 1.0863174347982087e-15, 3.057560324418493e-16,
           0.04214272260157403, 0.21880825158555614, 4.3109349482186617e-7,
           2.807036157422449e-7, 1.2014639494904744e-8]
    w6t = [3.616322845273605e-9, 3.83928189118373e-9, 4.759314749439659e-9,
           4.353846320931957e-9, 2.1245024409034897e-7, 2.0442498818823596e-17,
           0.9999997322303765, 2.6610724308919717e-9, 1.979188082489676e-15,
           1.9282537358428684e-16, 5.044432852800647e-17, 2.2636681407881077e-17,
           3.279210237999571e-17, 3.27175189089651e-17, 1.3934272006895045e-17,
           1.5771366511244315e-8, 4.950988636850216e-9, 2.7967080191875943e-9,
           9.144276028456238e-9, 3.4261996722832646e-9]
    w7t = [0.03244158253456704, 0.04235778407533121, 0.029184278892081254,
           0.03478319296608296, 0.02982893493239512, 0.052958781124194984,
           0.028243170894659806, 0.05136667760741622, 0.039891921160963774,
           0.14504526807786997, 0.07537259886425227, 0.03409029432604595,
           0.06829477209996813, 0.04839288789589798, 0.0385877771971674,
           0.08429331145127741, 0.03766316906044964, 0.04276953973125722,
           0.04287636716886095, 0.04155768993926082]
    w8t = [2.0287125722033544e-8, 0.20026010476588305, 1.441688370321708e-8,
           0.05856157362487539, 2.4453861524193335e-8, 6.339497751644587e-10,
           0.06461020274958595, 0.20780291502927517, 7.95927991180445e-17,
           6.530499095588974e-9, 1.9962937384515696e-9, 1.7569421849793738e-10,
           1.0212023513530963e-9, 4.0331094221712257e-10, 2.1090759230946626e-10,
           1.9482589079310657e-9, 0.06940069131439768, 0.20694155819057938,
           1.186302849268938e-10, 0.1924228821287853]
    w9t = [9.07931879678054e-9, 0.0016516481051193474, 1.0764921940941062e-8,
           1.495342149591518e-8, 0.30831485438303746, 2.287646555515597e-10,
           0.08203326322897575, 3.868945053045331e-8, 4.10483365622868e-9,
           0.19498346094337363, 1.0922253214099065e-8, 2.869494972949558e-10,
           6.485661942401577e-10, 5.138885463364689e-10, 1.4463960350073475e-10,
           0.1552144163699879, 0.12501359599013165, 2.463003456456128e-7,
           0.13278841747759348, 6.864427081241984e-9]
    w10t = [0.08763352519519421, 0.11441987868572619, 0.07883466340976727,
            0.09395885092580346, 0.08057605445599274, 4.402975205338157e-10,
            0.07629247511437338, 0.13875534683015436, 3.316600874048118e-10,
            2.157540895132846e-9, 6.26645245446529e-10, 2.834255575260693e-10,
            1.015881218891969e-9, 4.0233683819245326e-10, 3.208174784646232e-10,
            1.2538586680722526e-9, 0.10173844852583798, 0.11553214254059234,
            3.5647267088105485e-10, 0.11225860712762192]
    w11t = [6.241759951291285e-9, 0.0011354586425377689, 7.400561666979751e-9,
            1.0280029759620041e-8, 0.21195723528935181, 5.302830789916956e-10,
            0.05639541341779074, 2.659783936038016e-8, 9.515122975293216e-9,
            0.18750214164634646, 2.5318098418372165e-8, 6.651572227091397e-10,
            6.236813616451639e-10, 1.1912084930814456e-9, 3.3527877854899136e-10,
            0.1492589953165949, 0.08594310589631492, 1.6932411647282603e-7,
            0.3078073870488377, 4.719088183074755e-9]

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

    w2t = [0.09291266842717058, 0.07830058295036317, 0.07669383913367864,
           0.0604899583406831, 0.06329643710739387, 0.024835258181867, 0.022359241907072516,
           0.051372477115718504, 0.02365393139302484, 0.07457498550218726,
           0.029855361973367387, 0.007019914822144945, 0.02439800803722596,
           0.031054222170245907, 0.014448426205745796, 0.03397581214529964,
           0.1053400436785177, 0.10072270065874253, 0.026021869429647074,
           0.05867426081990363]
    w3t = [2.529276503881537e-9, 0.21191459984592514, 2.4711225945029506e-9,
           0.09271239767205439, 7.05231706834582e-9, 0.029526571282439483,
           0.09814897392077784, 0.1951501120999505, 7.101329052225676e-10,
           5.451261076442535e-9, 0.07966445647137076, 0.0057273682001028935,
           9.66807484714734e-10, 2.7329296163754835e-9, 0.022440471292527384,
           1.6725018652286136e-9, 0.0896834273475343, 0.00010556324695181721,
           2.573094273901442e-9, 0.174926032460922]
    w4t = [3.2045890629594967e-9, 0.21575828346428821, 3.048365395458857e-9,
           0.09371783451172656, 1.2254224870649566e-8, 0.0251871911716709,
           0.10039772625098634, 0.19855611913117432, 1.5143763672179137e-9,
           1.0777719587061035e-8, 0.06876464539810181, 0.005185954591406701,
           1.896307599653251e-9, 3.787598431493779e-9, 0.018877883803262885,
           3.30735777657766e-9, 0.09601955141542645, 6.174466512333312e-8,
           1.0543645953386197e-8, 0.17753469818310574]
    w5t = [7.408349638706929e-9, 0.0038186592000146514, 5.835089914154846e-9,
           2.0063286898793054e-8, 0.5334258282960406, 1.8376899941977034e-11,
           0.1792828310954398, 0.0009339923167643227, 2.6910367020951073e-10,
           0.03645665813298778, 0.001470808950642995, 2.4826140427686114e-11,
           1.218766536400032e-10, 4.4366478375481055e-11, 1.1469569402445713e-11,
           0.018265611493780018, 0.13315781240098476, 0.08717716213664908,
           0.006010598751580895, 3.428369265483797e-9]
    w6t = [7.899825126828249e-8, 8.398383302840335e-8, 1.045159719223709e-7,
           9.545583379702279e-8, 3.920187162969964e-6, 1.1063451637644887e-14,
           0.9999948829831524, 5.7757250340023774e-8, 9.7629952535239e-13,
           9.821169011242402e-14, 2.7346804461857104e-14, 1.2254614413840212e-14,
           1.699822475848904e-14, 1.774075055765404e-14, 7.513493183055607e-15,
           3.358367969634551e-7, 1.0880889674288217e-7, 6.077133035237832e-8,
           1.959289193361942e-7, 7.477143351754478e-8]
    w7t = [0.03136070689246619, 0.047254371717960104, 0.029317542180578023,
           0.03807351023889607, 0.03218469553679128, 0.05212310046400624,
           0.030413012215974103, 0.04311656527524499, 0.03950391685866444,
           0.1567530853540612, 0.06469336743911681, 0.035572582903316, 0.0721152246935304,
           0.041529293632031145, 0.04167307947370621, 0.08552272108784345,
           0.039119677573665544, 0.035231385132331074, 0.04143766055839026,
           0.04300450077142654]
    w8t = [2.932014741484665e-9, 0.24565789059857482, 2.8646009497100965e-9,
           0.10747504919912221, 8.175261808774675e-9, 6.493328511430331e-10,
           0.11377729479386053, 0.2262239832621475, 1.561686995851413e-17,
           5.2633657936089765e-9, 1.751938894648434e-9, 1.2595327399665122e-10,
           9.334833486591142e-10, 6.010115305293627e-17, 4.934990610994609e-10,
           1.6148536978412859e-9, 0.10396377408571199, 0.00012237214703363677,
           5.658613813131488e-17, 0.20277961110924478]
    w9t = [4.322586618690752e-9, 0.0022280920804656767, 3.404628940567326e-9,
           1.1706425817506848e-8, 0.31124062171813904, 4.964248482472659e-10,
           0.104607045353951, 0.0005449611434794686, 7.269438755629788e-9,
           0.1670328619106817, 0.039731734537872127, 6.706415681258539e-10,
           5.584002292071377e-10, 1.1984949781711834e-9, 3.0983350119160773e-10,
           0.08368724722999395, 0.07769425123394882, 0.050865692480071216,
           0.16236746037415417, 2.000367670753931e-9]
    w10t = [0.08497087134495143, 0.12803426764266865, 0.07943497936185011,
            0.10315900567079082, 0.08720344324859905, 6.669330535089355e-10,
            0.08240312174968165, 0.11682300827597623, 5.054662455144205e-10,
            3.362441294988746e-9, 8.27773956342839e-10, 4.5516347120009354e-10,
            1.5469118771041557e-9, 5.313816401080388e-10, 5.332214295600482e-10,
            1.8345101686262875e-9, 0.10599356390676425, 0.09545835505713372,
            5.302091633163344e-10, 0.11651937294757167]
    w11t = [2.716437714850606e-9, 0.0014001971258056872, 2.139566716659612e-9,
            7.356654568649945e-9, 0.1955925555251875, 8.008847336155556e-10,
            0.06573807497803892, 0.0003424692513678576, 1.172782253324529e-8,
            0.21990890558861453, 0.06409940949530812, 1.0819494542554886e-9,
            7.351678099788358e-10, 1.933538046536525e-9, 4.998559639855465e-10,
            0.11017934279251476, 0.04882530135229568, 0.031965463652573155,
            0.2619482499893274, 1.2570885591760976e-9]

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

    w2t = [0.11942828911415344, 0.06790696087036098, 0.09708343396881959,
           0.0322181147701282, 0.06358441456867103, 0.010918315222441926,
           0.029080597027890735, 0.05067903550079908, 0.019797098917560686,
           0.032160771893579174, 0.04334920387136981, 0.0072395254377313496,
           0.004408762771082226, 0.030601955352742596, 0.0068971479386845886,
           0.02624877427854606, 0.100996587649217, 0.10476039799498128, 0.03792180923631091,
           0.11471880361492931]
    w3t = [0.09306804224438855, 2.965538175980197e-9, 5.120616924704069e-10,
           0.012498074010119991, 3.7922350595309275e-11, 4.389967746479165e-12,
           1.571554053547651e-10, 0.01890604423249791, 1.3485844134776075e-10,
           8.710330716011039e-11, 0.4415479849421009, 0.05315150363540031,
           3.889652755318321e-21, 3.4792816837581925e-11, 2.5797454703026665e-11,
           1.6944084292719147e-11, 0.02308802678879938, 1.0073993561399218e-10,
           0.29524210224876657, 0.06249821782062269]
    w4t = [0.09810134108090621, 0.011388753754271286, 6.437047367157054e-10,
           0.011075489907070782, 3.148550335615237e-11, 1.4799393254155678e-11,
           1.318931439966537e-10, 0.01841991202305588, 1.0435131278098484e-10,
           8.614741934900651e-11, 0.41790394179337464, 0.05030533996568464,
           1.4970245155556828e-20, 3.110891490720659e-11, 3.954733749975306e-11,
           1.675812001360921e-11, 0.029370582327353862, 2.2607295727840045e-10,
           0.279432445332238, 0.08400219249017586]
    w5t = [8.193729881939037e-10, 2.409987309001213e-9, 0.5264670437187233,
           2.517270977151848e-10, 0.32265001048427105, 3.8303851614533396e-20,
           5.6956445630163935e-9, 0.05614503811357049, 1.0037399347394833e-17,
           8.813801105337669e-9, 2.179845581501531e-9, 1.786411989931377e-18,
           1.0632090753928366e-10, 3.451301121073617e-18, 2.064195656295622e-18,
           0.09473786627587708, 1.5456125575809285e-8, 5.141044640297729e-10,
           4.000999373837102e-9, 1.1596291575584218e-9]
    w6t = [1.8648307834296094e-8, 2.0121129931139206e-8, 1.2732227882968977e-7,
           2.248819943852892e-8, 8.505310254468229e-5, 8.750827164319476e-16,
           0.9999131027072671, 3.013461011003404e-8, 9.978547267173414e-13,
           1.6150100575048547e-14, 8.277584052238747e-14, 9.184146757828695e-16,
           4.191095842512731e-15, 1.4250045646643808e-15, 5.48091714601181e-16,
           1.1339473486081746e-6, 3.3121983434285156e-8, 1.4447566564553116e-8,
           4.2555952904300184e-7, 1.839812964815229e-8]
    w7t = [0.05659875480807075, 0.05765686028748787, 0.04814562158947152,
           0.032543952358064475, 0.031507038395257314, 0.037360792840002946,
           0.02039180126487961, 0.0618246311824005, 0.06456187917628958,
           0.08093640733937797, 0.10365488779001777, 0.02906988764065906,
           0.02760574340527159, 0.052682257268469534, 0.017789884077172368,
           0.06786092277422026, 0.043104486102002494, 0.047256307941542106,
           0.07426879053007172, 0.045179093229270634]
    w8t = [0.4430578096545735, 1.411768010813643e-8, 2.4377103719255615e-9,
           0.05949807433666507, 1.8053236305960425e-10, 1.2847210598042237e-18,
           7.481507936851122e-10, 0.0900037257137861, 3.946623066440409e-17,
           1.975223751571506e-8, 1.2921871592887966e-7, 1.5554751201856835e-8,
           8.820485419167596e-19, 1.0182094061397593e-17, 7.549607482402301e-18,
           3.842375087100669e-9, 0.10991238594479338, 4.79580467596752e-10,
           8.640239938074054e-8, 0.2975277316160487]
    w9t = [6.600850494458528e-11, 1.9414803940906618e-10, 0.04241206747011373,
           2.027907877564386e-11, 0.025992612789648654, 5.314455071369234e-12,
           4.588398540317723e-10, 0.004523031731986403, 1.3926356127821094e-9,
           6.467043114451837e-9, 0.3024419455775335, 2.4785513360341633e-10,
           7.801195929050324e-11, 4.788496188397067e-10, 2.863961354146512e-10,
           0.06951301242845523, 1.2451420246886398e-9, 4.141614081121245e-11,
           0.5551173189269033, 9.341946596177755e-11]
    w10t = [0.12741482518458436, 0.1297968267876931, 0.1083851752397725, 0.0732627778575606,
            0.07092848248744955, 2.2992199022804634e-12, 0.04590591793360314,
            0.1391792911508308, 3.973201483342501e-12, 3.4457091136241505e-9,
            6.379023646422122e-12, 1.7889894496268647e-12, 1.1752604886581708e-9,
            3.242117878157834e-12, 1.0948069465405327e-12, 2.8890459528503267e-9,
            0.09703659700610297, 0.10638315693796643, 4.570574346115142e-12,
            0.10170694188107306]
    w11t = [3.7320482860610613e-10, 1.0976916661375738e-9, 0.23979316580950402,
            1.1465568149297584e-10, 0.14695937454316563, 2.5517155398172393e-12,
            2.594230080280365e-9, 0.025572724056287445, 6.686687321120674e-10,
            1.636654436747313e-8, 0.1452163584146957, 1.1900670671703834e-10,
            1.9742967076689307e-10, 2.299178367715564e-10, 1.3751202323983172e-10,
            0.17592086242391736, 7.03989609072797e-9, 2.341623059930071e-10,
            0.2665374850487746, 5.281833880650424e-10]

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

    w2t = [0.10441410900507704, 0.06337257015393993, 0.1388818998669592,
           0.027515787911662204, 0.1188160882745767, 0.009101280957417223,
           0.02353511836320865, 0.03744216548824482, 0.015577834189637181,
           0.01767347121199722, 0.025141762273107857, 0.0033298619605358548,
           0.0018130367073894856, 0.014126651605875161, 0.002228743288689212,
           0.016302454122944926, 0.17177480035356854, 0.09442597723364114,
           0.03145585583242577, 0.08307053119910199]
    w3t = [4.858382022674378e-10, 0.053481853549220186, 0.12301036094781877,
           1.1572606679018196e-10, 0.04087812887359, 0.002519516122229487,
           0.00042880092524383425, 0.13309997405160942, 6.61472699038919e-10,
           2.3028686401319528e-9, 0.09131414340068031, 0.017720396275592475,
           1.939245823037983e-18, 0.030718209317324146, 6.280352017591961e-11,
           1.049984884248537e-9, 0.18695553460116063, 0.03885091677025856,
           0.1588747870328061, 0.1221473734537721]
    w4t = [4.480528858292739e-10, 0.0564096307290648, 0.12339719843847467,
           7.733392551531556e-11, 0.04562833019774078, 0.002410781036645187,
           0.0008040452742058028, 0.13166519284950062, 8.034784301568693e-10,
           1.2569589667840443e-8, 0.08737372027691452, 0.016955689339267627,
           5.984793645609076e-18, 0.029392551406924643, 1.3827803193332287e-10,
           5.753665011415402e-9, 0.18848310019120762, 0.037265002327532964,
           0.1520188951238704, 0.12819584301825235]
    w5t = [1.02062048455541e-9, 0.0397811102944998, 0.18540374464442913,
           1.07360599890668e-9, 0.16404765384401954, 9.053644415456022e-11,
           0.03456210501430832, 0.030821893166634237, 2.849030118902176e-10,
           6.553423268609104e-9, 0.018734132135076267, 1.1980574003163722e-10,
           6.892277210265349e-11, 1.966174767706324e-10, 7.538937153221054e-12,
           0.12494570939916305, 0.2110201401051579, 2.550949677130291e-9,
           0.06327027556315082, 0.127413223866637]
    w6t = [4.9820377641708154e-8, 5.448876375975373e-8, 1.1275893036261378e-7,
           6.041464983120543e-8, 0.00029562183155109137, 1.335587798324464e-13,
           0.9996693811475731, 3.922730079076675e-8, 3.3886517178782046e-11,
           1.1415914845860577e-11, 3.4714407547192225e-12, 1.348472107004945e-13,
           2.5854586251445246e-12, 1.9381150104653553e-13, 8.26353796979984e-14,
           2.760589173723265e-5, 1.2528267644249627e-7, 3.8541662805900445e-8,
           6.862903785415222e-6, 4.763908748236392e-8]
    w7t = [0.03941964489419902, 0.05878789380633039, 0.06336122678465943,
           0.02644427349083025, 0.04079783708796677, 0.025118942413556285,
           0.02001067299965853, 0.06274910163100314, 0.053259359498068336,
           0.08018126523435438, 0.0694165490338133, 0.021434679840840902,
           0.01606475726803423, 0.0381974113604881, 0.01001038091191075,
           0.07031962100674591, 0.07179495883084511, 0.05835523485418758,
           0.0924829280168134, 0.08179326103569423]
    w8t = [6.756186931515573e-10, 0.07437319633081656, 0.17106126879235903,
           1.6093154808216046e-10, 0.05684614317927392, 0.00023557822297623132,
           0.0005963012364679063, 0.18509213583363687, 6.184860720358851e-11,
           2.967977358332787e-8, 0.008537998008877067, 0.0016568814258448596,
           2.4993339154109853e-17, 0.0028721948234979398, 5.872215521527604e-12,
           1.353238872914074e-8, 0.2599850184180392, 0.05402705158540236,
           0.01485501111689518, 0.16986117690947947]
    w9t = [7.996651587328443e-10, 0.031168851066199533, 0.14526547049987415,
           8.411797769336604e-10, 0.12853278484614483, 3.112698538969441e-10,
           0.027079714360669013, 0.024149225362922088, 9.795140477847068e-10,
           5.072047857655721e-9, 0.0644090965469789, 4.1189949024284296e-10,
           5.3343052059744604e-11, 6.759829573547252e-10, 2.591932881984257e-11,
           0.0967022259521781, 0.16533614246095107, 1.9986915894310024e-9,
           0.2175270921502098, 0.09982938558435937]
    w10t = [0.0752981000097754, 0.11229468756184914, 0.12103051673799935,
            0.050512975328486864, 0.07793067708949149, 4.68877976498372e-10,
            0.03822372476554032, 0.11986125554444421, 9.941557371291465e-10,
            3.3215792015581425e-7, 1.2957508525083775e-9, 4.0010638764950443e-10,
            6.654966526609482e-8, 7.130047376722314e-10, 1.8685687751812173e-10,
            2.9130519444258177e-7, 0.1371403523484868, 0.11146823676210162,
            1.7263150428564876e-9, 0.1562387780539775]
    w11t = [5.994330957552132e-10, 0.023364330284629888, 0.10889174017042325,
            6.305526660346591e-10, 0.09634883336477022, 3.866217884060778e-10,
            0.0202990924815392, 0.018102382930282015, 1.2166339534080966e-9,
            9.65277723955786e-9, 0.08000119441331222, 5.116117592742031e-10,
            1.0151887625304099e-10, 8.396243215735293e-10, 3.2193857314336954e-11,
            0.18403711319025304, 0.12393681859622356, 1.4982294449479619e-9,
            0.2701858607282029, 0.07483261837116671]

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

    w2t = [0.11749818113343935, 0.05870055826126679, 0.11385134924830828,
           0.029526234501200906, 0.08032687433988389, 0.008852903280265263,
           0.026595918536857853, 0.04345063385165379, 0.02084656630749761,
           0.029413320573451408, 0.04005512791398602, 0.005832294513804399,
           0.0033062821268677552, 0.02971553377155059, 0.005420678468821243,
           0.02415714416132541, 0.11618399112873753, 0.10379008396251159,
           0.040922826850160604, 0.10155349706840978]
    w3t = [0.011004045505975613, 0.028251183077981752, 0.014422806875229753,
           4.958526061344831e-12, 0.009984293676628531, 2.3450739422880165e-11,
           6.031901028176002e-12, 0.09832765390526335, 1.2991801782199283e-10,
           2.2291710613941976e-10, 0.24797181174174213, 0.03612792745275501,
           1.056769979910633e-20, 0.03754084144061125, 1.8643272263494993e-11,
           5.689309987552144e-11, 0.09739889660050584, 0.03323021449469062,
           0.2194393677703557, 0.16630095699544786]
    w4t = [0.0071197508044408345, 0.024699895722335642, 0.018315740975714777,
           7.151885014623421e-12, 0.008356484733585173, 1.9089526941426207e-11,
           8.18850804197304e-12, 0.0962626345923765, 1.4039247614012587e-10,
           0.0001446322897575077, 0.25239579994504224, 0.036772480103140416,
           1.5733180507973053e-14, 0.03821064073749522, 3.108510515937393e-11,
           3.6913199117386076e-5, 0.09920775900938193, 0.02887885603246143,
           0.2233543807284419, 0.16624403092078577]
    w5t = [2.86011130169537e-10, 7.279305864441031e-10, 0.13990028972380286,
           5.829103756078371e-11, 0.33756478162542447, 1.4168493517274713e-19,
           3.0069380177645344e-10, 0.2116115968069058, 2.183525693820217e-18,
           6.38781643234779e-9, 6.179520517424919e-10, 4.2408209986768714e-19,
           1.2329702172663889e-10, 6.815529955852682e-19, 5.692773425625955e-19,
           0.24570860427285218, 5.763612679405827e-9, 1.5603214133107417e-10,
           1.4350232532914917e-9, 0.0652147117143545]
    w6t = [1.4702102374546112e-8, 1.6162920494912485e-8, 6.002515523179071e-8,
           1.7472955087571066e-8, 5.886754774299193e-5, 1.2612840120920024e-15,
           0.9999340453627157, 1.4420797125010015e-8, 3.9098351660212424e-13,
           7.836497422018281e-14, 4.745002163035154e-14, 1.3315777617840017e-15,
           1.8301083230580586e-14, 1.891380812749e-15, 7.840115067588502e-16,
           5.655799059022777e-6, 5.258192981293426e-8, 1.1388574409740394e-8,
           1.2300009429028432e-6, 1.4534564600702838e-8]
    w7t = [0.044621929656327136, 0.05974669646641222, 0.04673199751476315,
           0.02202711999783714, 0.034928496248394274, 0.030602363998901337,
           0.014815135143633454, 0.07371498589589642, 0.060020783382639276,
           0.08929648987351317, 0.09357047699806313, 0.021425090153187682,
           0.023300153316325536, 0.052589757331902416, 0.014189312841471253,
           0.0698335955428971, 0.046206226891693514, 0.04462051809754638,
           0.08090941082100037, 0.07684945982759507]
    w8t = [0.023978128534749055, 0.061560132474435376, 0.03142770691908913,
           1.0804769498387138e-11, 0.02175606025773661, 4.717394485274624e-18,
           1.3143684119076717e-11, 0.21425875807042033, 2.613455080283381e-17,
           2.4236765538500844e-8, 4.988247219500176e-8, 7.2675612762873135e-9,
           1.1489780517427515e-18, 7.551785689611238e-9, 3.750315424061929e-18,
           6.185728615995543e-9, 0.21223496945386386, 0.07240958372611976,
           4.414283254378029e-8, 0.36237452127249153]
    w9t = [1.4671443074990454e-10, 3.7340477467532406e-10, 0.07176430985887769,
           2.9901411139078014e-11, 0.1731597813974337, 3.248440825440844e-11,
           1.5424616493598192e-10, 0.10854988387060419, 5.006216079752263e-10,
           3.7016046411499338e-9, 0.14167918915287472, 9.723021045739633e-11,
           7.144801869260131e-11, 1.5626111363648666e-10, 1.3051943442851717e-10,
           0.14238294409042324, 2.9565463162944528e-9, 8.003942636955266e-11,
           0.32901085184289003, 0.033453031355874595]
    w10t = [0.09611356358363932, 0.12869160867682602, 0.10065855172824832,
            0.047445393213208564, 0.07523435832800056, 7.632577112567362e-12,
            0.03191111286748034, 0.15877865521594947, 1.4969864992820254e-11,
            2.2630850203336463e-10, 2.3337539582663204e-11, 5.343660795087057e-12,
            5.905072866396544e-11, 1.3116482706413158e-11, 3.538975761505215e-12,
            1.7698272822708793e-10, 0.09952606623064618, 0.09611052315159904,
            2.0179725894569716e-11, 0.1655301664539414]
    w11t = [1.699037547746042e-10, 4.324242199204653e-10, 0.08310720112199596,
            3.4627555037504955e-11, 0.2005289928536412, 2.53479621278597e-11,
            1.7862593644165625e-10, 0.1257070129754424, 3.9064087176721607e-10,
            4.799958152998788e-9, 0.11055392152527725, 7.586986572314868e-11,
            9.264833311118337e-11, 1.219323670449596e-10, 1.0184583492896904e-10,
            0.18463132603013047, 3.42385079460753e-9, 9.269026230537524e-11,
            0.2567309998955161, 0.038740535657630665]

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

    w2t = [0.10987855273964697, 0.060729473692400754, 0.13188113586803757,
           0.028009359764792043, 0.10116830124988405, 0.00894518902343889,
           0.02515249217180455, 0.040355519105512605, 0.018050336913773852,
           0.021728169704026638, 0.031807104420907735, 0.0040047144063036725,
           0.0023194434166183132, 0.020203823344076888, 0.0032143655180704735,
           0.019106596337899368, 0.15029999715584982, 0.09854472399816692,
           0.03553068862836474, 0.08907001254042422]
    w3t = [6.339221068451245e-10, 0.04651817931744516, 0.11017171572957145,
           2.3217212095726008e-12, 0.033480752472860545, 0.0017843135154844325,
           5.8154031303529526e-11, 0.13022303335424218, 7.373381378336765e-10,
           1.1215711483301387e-9, 0.11774178258081092, 0.02223575914998876,
           7.341828128086774e-19, 0.043728900575733115, 1.0805077202807164e-10,
           4.553292084477986e-10, 0.16441923747282433, 0.039638908228437054,
           0.15238444697953965, 0.13767296750637523]
    w4t = [5.814772490139419e-10, 0.04733086058026881, 0.11032398217999186,
           7.911405751610749e-11, 0.0354118560544708, 0.0010377953218527578,
           7.406664618709402e-11, 0.12696018297146391, 5.092177808027191e-10,
           1.107907204934204e-9, 0.12267498220182824, 0.022906839231032764,
           1.1268406472864974e-18, 0.045767441267552396, 1.255645628894548e-10,
           4.497819877549505e-10, 0.15935092313972446, 0.03748067867036363,
           0.15748652668727153, 0.13326792876704946]
    w5t = [8.828039841744664e-10, 3.9890928005496905e-9, 0.21226108486649664,
           4.1713658186614307e-10, 0.23673047785519769, 5.31587609110726e-17,
           0.0048039761977919015, 0.14638945477130516, 1.9486228895037198e-16,
           7.819440549878561e-9, 1.7393651143857764e-8, 1.0399149605948582e-16,
           1.4235659372375644e-10, 1.2848385093506377e-16, 1.5415688482482198e-17,
           0.17050521930616136, 0.14821155354495932, 1.2548664876774095e-9,
           5.439085290965003e-8, 0.08109814716788621]
    w6t = [5.995207906173523e-8, 6.50199354244668e-8, 1.2333856267120065e-7,
           7.21540711292467e-8, 0.00028068604118484296, 2.976124635366768e-13,
           0.9996800252578011, 4.7195114552088204e-8, 7.140974557928591e-11,
           6.564785028259721e-12, 1.328849866404678e-11, 3.081157588741628e-13,
           1.9906229052224903e-12, 4.2254457795983015e-13, 2.1273367817217905e-13,
           2.903657834747259e-5, 1.4092370999366812e-7, 4.646053553777311e-8,
           9.63956639466522e-6, 5.7417768867966126e-8]
    w7t = [0.042621616770917956, 0.05926643442160448, 0.05891985619040837,
           0.023392613122276185, 0.03818231982232573, 0.028188514974334427,
           0.017007819570827774, 0.07027563215510384, 0.055681856290461615,
           0.08452765001572406, 0.0785253874525583, 0.020242825802686264,
           0.018215706072420134, 0.04174002486233878, 0.011185883589400338,
           0.07256987057081475, 0.054917332512358356, 0.05039480814874462,
           0.09140224204282815, 0.0827416056118659]
    w8t = [9.57405744371646e-10, 0.07025590623094205, 0.16639115810570973,
           3.5064705882190676e-12, 0.05056562059797869, 1.5441767322394795e-10,
           8.782940841968509e-11, 0.19667444759630692, 6.381055718935304e-17,
           7.067883136811594e-9, 1.0189583808892812e-8, 1.924322245228956e-9,
           4.6266510419006584e-18, 3.78437702844659e-9, 9.350906475702045e-18,
           2.869379832815273e-9, 0.24832060712491696, 0.05986621704580805,
           1.3187621756992669e-8, 0.20792600307201034]
    w9t = [6.716658257018889e-10, 3.03503083098134e-9, 0.1614950990106301,
           3.173710038832539e-10, 0.1801122046657988, 1.767241059568185e-10,
           0.003655020477234774, 0.11137783219789629, 6.478116346064185e-10,
           5.973272308287896e-9, 0.05782447512020149, 3.457154352970759e-10,
           1.0874623238941447e-10, 4.271392578990444e-10, 5.1248819913859816e-11,
           0.13024897349157108, 0.11276414388118719, 9.5474301283274e-10,
           0.18082014493841547, 0.06170209350759642]
    w10t = [0.08563371566850694, 0.11907584410098819, 0.11837951242810353,
            0.04699953996637748, 0.07671445071647424, 7.738460052808273e-11,
            0.03417145795049495, 0.14119510141391417, 1.5286077360310543e-10,
            5.521534214436152e-9, 2.1557204219030067e-10, 5.557167483731304e-11,
            1.1898904595155578e-9, 1.1458692140913653e-10, 3.070807858830172e-11,
            4.740425449180453e-9, 0.1103377955583561, 0.10125131328014633,
            2.509222637054648e-10, 0.1662412565671816]
    w11t = [4.989766495382837e-10, 2.2547068160061717e-9, 0.1199737731735411,
            2.357730795857913e-10, 0.1338043130766377, 2.5220189183140663e-10,
            0.0027152935313012124, 0.08274194609330886, 9.244880256350282e-10,
            8.740372949496659e-9, 0.08252095513804042, 4.933683854003776e-10,
            1.5912260129465788e-10, 6.095678251382647e-10, 7.313687777014796e-11,
            0.19058642329529735, 0.08377182900900046, 7.092730513950688e-10,
            0.25804732403535485, 0.04583812769652987]

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

    w2t = [0.11773085192640985, 0.06254095197843257, 0.10519293078600803,
           0.030663051813610273, 0.07121563819691837, 0.009654306716895931,
           0.027998729595426087, 0.04641793515569032, 0.02098324936681848,
           0.030965512189401155, 0.04162305512319709, 0.006349197130850141,
           0.003735755505512061, 0.03046045336577972, 0.005913942062630893,
           0.02518714386655288, 0.11143866905372922, 0.10432623891279905,
           0.04088970700666294, 0.10671268024667503]
    w3t = [0.0726312361265007, 0.033384488741601014, 5.103751149075006e-9,
           3.014518528946845e-10, 4.724506670658462e-9, 1.253054866996639e-9,
           2.205095228542095e-10, 0.08141932914872951, 3.6881820699728994e-9,
           0.014557007444885766, 0.30653075374144134, 0.03842013687740744,
           8.468257870550366e-13, 0.005408493904766368, 6.545441982099049e-11,
           0.00395626991922374, 0.0855973005973881, 0.027809798353575423,
           0.25683830395676144, 0.0734468658299618]
    w4t = [0.07188718659948104, 0.03231379397613412, 7.862356773595368e-9,
           3.304805536050558e-10, 5.767616571417321e-9, 2.062324338537918e-9,
           2.3934067182581034e-10, 0.0806054983964276, 6.055078255784101e-9,
           0.015565533049874125, 0.3095095853423995, 0.03864336921086802,
           3.77609702984125e-12, 0.004388544698817729, 8.319748741533143e-11,
           0.004260470801984409, 0.08586493178147266, 0.02554875960399025,
           0.25989466505205777, 0.07151763908232191]
    w5t = [6.093553106747296e-9, 1.272321786628477e-8, 0.27138040478969777,
           2.0182828498984105e-9, 0.32321508213237843, 9.69583013515116e-17,
           5.175335729487244e-9, 0.18025501290038848, 2.685327727945996e-16,
           8.047029989704588e-9, 8.307726394072881e-9, 1.2217500016022025e-16,
           3.251856167488813e-10, 1.4737564848479143e-16, 2.4867715817468116e-17,
           0.22514939802530334, 2.5334475905700583e-8, 4.060556815673562e-9,
           1.7026345176235333e-8, 1.3040521844213984e-8]
    w6t = [3.2180585982358536e-8, 3.4792072301916835e-8, 1.969392408013349e-7,
           3.867877460090811e-8, 0.0001500291497766489, 6.176351673914708e-16,
           0.9998467424145034, 3.33470605537889e-8, 1.324537001996197e-13,
           2.1097260351535536e-14, 2.445689262829524e-14, 6.466977169358908e-16,
           3.3579852244224604e-15, 9.733564009400666e-16, 3.8534927984276727e-16,
           1.9407013400120445e-6, 5.301555337937153e-8, 2.4899700085263258e-8,
           8.425325908851574e-7, 3.134861737584489e-8]
    w7t = [0.04974172130529782, 0.05733881414227115, 0.050193916021615895,
           0.024534811168682846, 0.03152704024343209, 0.03377228228437697,
           0.017031241107275923, 0.08007434154998308, 0.05828257100946734,
           0.08635967294626462, 0.10082262710759267, 0.022763953645553357,
           0.024761931829072263, 0.05119987144439959, 0.015212168140680805,
           0.06700267065826623, 0.04438899718688257, 0.048204655053479324,
           0.07874539707869038, 0.058041316076715144]
    w8t = [0.19405113284349335, 0.08919437703669776, 1.3635850703467873e-8,
           8.053982924109978e-10, 1.262261143361441e-8, 1.5691263158333906e-16,
           5.891421514306964e-10, 0.21753055433547147, 4.618491732489209e-16,
           2.2178952708606988e-7, 3.8385028858375617e-8, 4.811125946683914e-9,
           1.2902177287902004e-17, 6.772736245248679e-10, 8.196468912402247e-18,
           6.027744629049771e-8, 0.22869297061581562, 0.07430030331937883,
           3.216233800029209e-8, 0.19623027609339982]
    w9t = [2.4253409715576375e-9, 5.0640637802903295e-9, 0.10801415907667186,
           8.033119597547357e-10, 0.12864526944928348, 2.1815317763712712e-9,
           2.0598743567842157e-9, 0.0717446554510012, 6.041904289604935e-9,
           4.345625788287317e-9, 0.18692127301574019, 2.748899696183987e-9,
           1.756095110782521e-10, 3.3159064850704794e-9, 5.595159105036386e-10,
           0.12158709878401262, 1.0083565586553863e-8, 1.6161728042520722e-9,
           0.3830874976116114, 5.190356326630463e-9]
    w10t = [0.10788162341838653, 0.12435846995698004, 0.10886235948496119,
            0.05321197557470049, 0.06837697196208627, 9.597181976091475e-10,
            0.0369379645751663, 0.17366809458703783, 1.656235238419423e-9,
            1.248297512070696e-8, 2.865110185658957e-9, 6.46890855027439e-10,
            3.5792467527671506e-9, 1.4549638051311141e-9, 4.3228925030201903e-10,
            9.684991180646158e-9, 0.09627244399210308, 0.10454797918179005,
            2.237734184442061e-9, 0.12588208126663333]
    w11t = [3.4244722835481194e-9, 7.150230116545913e-9, 0.1525111307344399,
            1.1342403288788212e-9, 0.18164132994286553, 1.466743392546752e-9,
            2.9084498736970356e-9, 0.10130022416059466, 4.062248045691415e-9,
            6.479970680984634e-9, 0.12567570415055218, 1.8482107433971428e-9,
            2.6185975013223644e-10, 2.2294352894414707e-9, 3.761880865751478e-10,
            0.18130434457332809, 1.4237540731568315e-8, 2.2819632532043484e-9,
            0.2575672112481177, 7.328549507358309e-9]

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

    w2t = [0.11833256751510059, 0.06528196928711139, 0.10043845909056107,
           0.031380183529514306, 0.06655539510253948, 0.010311178382485315,
           0.028706985713035555, 0.04850410086672684, 0.02066223864562248,
           0.03196981803885744, 0.04288859214073757, 0.006797311251556611,
           0.004081222059938556, 0.030487185818522944, 0.006367963512320135,
           0.02606975526628787, 0.10619308228152587, 0.10450491803132433,
           0.039823853638821044, 0.11064321982741053]
    w3t = [0.08378986882917888, 0.014222281895084562, 1.4640996523981236e-9,
           7.944954882144362e-11, 9.314212297173224e-10, 9.137860256305013e-11,
           1.8245207791188445e-10, 0.05641447999298819, 4.5648697820494774e-10,
           0.016986220828475156, 0.36206160895224154, 0.04411178251130963,
           1.1583898248023923e-12, 2.1145226385254962e-10, 1.7184793041218614e-11,
           0.004536260375593694, 0.07530115554166356, 0.020468216399785202,
           0.27991576632666104, 0.04219235491193489]
    w4t = [0.08360220460231052, 0.013583941175582762, 3.328795807287142e-9,
           1.87637917696734e-10, 2.2529865423853684e-9, 1.4232546558507974e-10,
           5.772690975589539e-10, 0.05733549479288826, 7.495380224782043e-10,
           0.016330706513896252, 0.36302988958155197, 0.0442327939969194,
           2.342722863753897e-12, 3.371032260258337e-10, 3.742234048075495e-11,
           0.004391597514939037, 0.07796335375207128, 0.017645665083984987,
           0.28088239780352764, 0.04100194756690676]
    w5t = [1.3515804012708479e-8, 2.2763927813695442e-8, 0.40422535276540883,
           4.209877377089907e-9, 0.3228420613543988, 9.382007684504622e-17,
           8.095862116122522e-9, 0.11630847933030657, 2.534082850464609e-16,
           9.151566224458459e-9, 7.013242330109768e-9, 1.212367854592803e-16,
           3.30324485838165e-10, 1.453262655206002e-16, 2.602876462067114e-17,
           0.15662397918383003, 2.8333194936724385e-8, 7.55051684815826e-9,
           1.2915452762775174e-8, 1.3486286320212747e-8]
    w6t = [1.7112615903132398e-8, 1.9024128895262603e-8, 8.647128018693435e-8,
           2.074472300676763e-8, 5.638464427967927e-5, 1.4958618984198002e-15,
           0.9999376754888423, 1.9517941371648137e-8, 2.8227576374538363e-13,
           5.238654504944856e-14, 5.833839585906032e-14, 1.55417915111153e-15,
           8.633146151296927e-15, 2.2806487733165333e-15, 9.281470736582364e-16,
           4.351938694057633e-6, 4.3900989019528236e-8, 1.32625726321481e-8,
           1.3507587848113968e-6, 1.7134740265231196e-8]
    w7t = [0.0560731811241318, 0.056704867975412944, 0.05028698047424797,
           0.028411275849565994, 0.031438244608776644, 0.03500694355093412,
           0.019634352772634988, 0.07479997517345825, 0.060089392631509704,
           0.08300309103129284, 0.10109742147029181, 0.026075081772755374,
           0.025610504206661496, 0.05006530433128356, 0.016088633338122934,
           0.06695558429275707, 0.04410955238183049, 0.05010270397113676,
           0.07553122561040937, 0.048915683432785906]
    w8t = [0.28657029398862566, 0.04864172197443732, 5.007377069317781e-9,
           2.717259363353497e-10, 3.1855600128877307e-9, 4.110340534581181e-18,
           6.240055789159631e-10, 0.19294354249157375, 2.0533438654083565e-17,
           4.511211747348206e-7, 1.628605018625932e-8, 1.984211211631962e-9,
           3.076459348095202e-17, 9.511425945066222e-18, 7.729966254077034e-19,
           1.204743026836388e-7, 0.25753798857533344, 0.07000348458675032,
           1.2591012428833808e-8, 0.14430235683785955]
    w9t = [2.9993937982630666e-9, 5.051714559067786e-9, 0.08970469052712345,
           9.34245575358936e-10, 0.07164431178007301, 3.401743726175629e-9,
           1.7966136975542534e-9, 0.0258108900706693, 9.188119140442163e-9,
           5.274033850640398e-9, 0.2542872900861404, 4.395823241532373e-9,
           1.9036550436031595e-10, 5.269263558584882e-9, 9.437552145124402e-10,
           0.09026216363160257, 6.2876325447071725e-9, 1.6755920244739014e-9,
           0.46829060350325163, 2.992843304949647e-9]
    w10t = [0.1217719879512069, 0.12314379818385374, 0.10920631677480047,
            0.06169968371819093, 0.06827323627728851, 1.2157959044651365e-9,
            0.042639174759252135, 0.1624402520593389, 2.086912767945364e-9,
            2.454154407881538e-8, 3.51112717957597e-9, 9.055911317074574e-10,
            7.572263997150566e-9, 1.7387748197214715e-9, 5.587604211282012e-10,
            1.9796773865012312e-8, 0.09579103188889317, 0.1088061305241422,
            2.623209724743808e-9, 0.10622832331227906]
    w11t = [6.791289140100713e-9, 1.1438196026061368e-8, 0.20311120565286875,
            2.1153380505742483e-9, 0.16221852456445107, 1.889894692203563e-9,
            4.06792969306735e-9, 0.058441548267112786, 5.104610750433436e-9,
            1.0212928880893932e-8, 0.14127348751485952, 2.4421719214506688e-9,
            3.686342204973405e-10, 2.9274260593365717e-9, 5.24318735982997e-10,
            0.17478861226748355, 1.423658695385102e-8, 3.793909931279695e-9,
            0.2601665490435318, 6.7764573783869e-9]

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

    w2t = [0.053466892258046884, 0.10328667521887797, 0.054404919323835864,
           0.05164034525881938, 0.03949521281426285, 0.02166132115612342,
           0.005325106752082301, 0.06987092753743517, 0.019722628041309125,
           0.058905384535264414, 0.04756653865024671, 0.00337013314805977,
           0.005397558580038606, 0.03531833344651656, 0.005579968832386527,
           0.012662292885364586, 0.11359655047116007, 0.16209808354457939,
           0.02645811650511682, 0.11017301104047358]
    w3t = [2.5964437003610723e-8, 0.05105919106221655, 3.467171217276289e-8,
           0.052000269807801806, 2.0285416272445106e-8, 0.07566811426967662,
           0.0033034082607419535, 0.10667421688825035, 8.976013131180923e-8,
           2.4171113199481545e-8, 0.24860400348802075, 0.019037062444765506,
           1.5750411960203906e-9, 0.0975551387747684, 0.00019572743265402633,
           7.1657474133847724e-9, 0.026531138487239897, 0.18667818859089166,
           0.030269149949859745, 0.10242418694951412]
    w4t = [3.953279520738277e-9, 0.00608819450228333, 0.04888864431745161,
           0.03295859019620168, 0.31265900510311656, 2.238222706366946e-12,
           0.08633806381834545, 0.0778256129435594, 2.7427326002014e-11,
           0.036740103394666095, 0.0003690172576887811, 2.434913299244783e-12,
           3.4263659558102143e-10, 5.507335911411046e-12, 1.2895587770429228e-12,
           0.015108385527328301, 0.3825640663960669, 4.1130118915494924e-8,
           0.0004602340413968748, 3.703696274306287e-8]
    w5t = [2.959825072900196e-8, 0.05213258127361044, 0.006609766699105744,
           0.05284860265884723, 0.19124177988566876, 3.480275729654175e-10,
           0.05283328514680718, 0.15481217842854053, 5.916596178137598e-9,
           0.014889219038824623, 0.03937226319131022, 4.090530649072886e-10,
           5.6622891060638855e-11, 1.0642891939159962e-9, 1.7585522121579718e-10,
           0.007156540564504349, 0.23161349618033553, 0.1157559529594864,
           0.04989155324703565, 0.03084274315722841]
    w6t = [8.005212570019472e-9, 8.467651966208832e-9, 1.0409572690089251e-8,
           9.543499190499942e-9, 8.295037015919763e-7, 1.3864613933349198e-17,
           0.9999990580669282, 6.007004624742323e-9, 1.6875079758803041e-15,
           5.123664164923582e-16, 3.1369581021316564e-17, 1.479521351471662e-17,
           9.819487055093616e-17, 2.0663973250659508e-17, 9.593483918575565e-18,
           2.6526189478950564e-8, 1.0708614452849208e-8, 6.2904542992933856e-9,
           1.889382404852362e-8, 7.577344478069048e-9]
    w7t = [0.03168837196142575, 0.037466468353004136, 0.031234374626791957,
           0.034549860557835925, 0.02792805782006853, 0.05605746091442722,
           0.023582207306295348, 0.057710978979585055, 0.04214105060168191,
           0.14962449271688594, 0.08550672279401512, 0.03422542718867126,
           0.04716059334901583, 0.056821918435025244, 0.032524555724384924,
           0.07606724180403476, 0.037144680642580016, 0.05150614556134447,
           0.0457072853415405, 0.041352105321386175]
    w8t = [4.859026903025064e-8, 0.09555299927493346, 6.48852051742806e-8,
           0.09731414932106308, 3.796245741556653e-8, 1.3509880557476973e-9,
           0.0061820518613163915, 0.19963186170647856, 1.6025887053616591e-15,
           0.0078120801189945, 4.438607233244723e-9, 3.3989011392223727e-10,
           0.000509051772356605, 1.7417617517440534e-9, 3.494542268561828e-12,
           0.0023159625476844056, 0.04965080338905666, 0.34935259348969816,
           5.404292208757231e-10, 0.1916782866653138]
    w9t = [1.7377790832481352e-8, 0.030608197127083556, 0.003880740932977206,
           0.031028589196133646, 0.11228229937346212, 1.668733067639664e-9,
           0.031019595944378902, 0.09089367069979797, 2.8369073192110964e-8,
           0.033921015722535905, 0.18878331097542073, 1.961339930666857e-9,
           1.2899977983499341e-10, 5.103085816696665e-9, 8.431959004266849e-10,
           0.016304221489019187, 0.1359854312828611, 0.0679628926912714,
           0.23922151911655093, 0.018108459996288902]
    w10t = [0.08469129373341312, 0.10013400752492095, 0.08347792683439849,
            0.09233899401692375, 0.07464136532865759, 1.7785466364082457e-9,
            0.06302651484560018, 0.15424009407465186, 1.3370178131461105e-9,
            3.1163345946492185e-8, 2.7128894483418966e-9, 1.0858771948120898e-9,
            9.822468627232694e-9, 1.8028007379989707e-9, 1.031913294692049e-9,
            1.5843059705605825e-8, 0.0992739880346011, 0.13765687010117678,
            1.4501644790448746e-9, 0.11051887747757239]
    w11t = [1.2102076686408775e-8, 0.0213158710698896, 0.00270258888622321,
            0.02160863653743642, 0.0781945766664472, 1.3727734534992884e-9,
            0.021602373542131362, 0.06329931023580364, 2.333765137979374e-8,
            0.19217085228363173, 0.1553014815828646, 1.613484770165166e-9,
            7.30815310427805e-10, 4.1980235640674775e-9, 6.93650153312075e-10,
            0.09236740329342016, 0.09470168754382272, 0.04733007475510593,
            0.19679417716188366, 0.012610922392864463]

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

    w1 = optimise!(portfolio; type = :HRP, rm = :Kurt, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :Kurt, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :Kurt, obj = :Min_Risk, rf = rf, l = l),
                   cluster = true)
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

    w2t = [0.053466892258046884, 0.10328667521887797, 0.054404919323835864,
           0.05164034525881938, 0.03949521281426285, 0.02166132115612342,
           0.005325106752082301, 0.06987092753743517, 0.019722628041309125,
           0.058905384535264414, 0.04756653865024671, 0.00337013314805977,
           0.005397558580038606, 0.03531833344651656, 0.005579968832386527,
           0.012662292885364586, 0.11359655047116007, 0.16209808354457939,
           0.02645811650511682, 0.11017301104047358]
    w3t = [6.258921377348572e-7, 3.680102939415165e-6, 7.486004924604795e-7,
           0.03498034415709424, 7.917668013849855e-7, 1.2350801714348813e-6,
           3.0460581400971164e-7, 0.08046935867869996, 5.699668814229727e-7,
           8.85393394989514e-7, 0.6357270365898279, 5.979410095673201e-7,
           2.7962721329567345e-7, 1.8757932499250933e-6, 5.070585649379864e-7,
           0.00127706653757055, 1.2457133780976972e-6, 0.11108722055351541,
           6.616561323348826e-7, 0.13644496428511088]
    w4t = [1.633189843996199e-7, 4.7392695446429325e-7, 8.799602597982136e-7,
           2.682810240405649e-6, 0.33250417011271577, 6.252670516798967e-8,
           0.08180392349672638, 7.710747796092221e-7, 2.9634480103019065e-7,
           2.658102014464364e-7, 4.034001318838129e-7, 5.870525160264156e-8,
           3.865052954354365e-8, 1.0933111816146697e-7, 3.72375730689396e-8,
           0.12820198647078426, 0.4574811481910088, 4.1279374482297674e-7,
           1.7817988595184708e-6, 3.340386297254291e-7]
    w5t = [2.1869958623435718e-8, 1.2027489763838995e-7, 8.307710723713282e-8,
           0.04981239216610764, 0.19930305948126495, 9.890114269259893e-9,
           0.041826524860577356, 0.10466467192338667, 3.332719673064681e-8,
           4.0257776120217934e-8, 0.23960266714976633, 8.639963684421896e-9,
           4.958509460440708e-9, 2.0027087137090508e-8, 5.256105741068125e-9,
           0.08392271758019651, 0.28086715326117795, 2.469501355591167e-7,
           1.0503063749371977e-7, 1.1401803293988891e-7]
    w6t = [4.936466409028226e-8, 5.276389568036465e-8, 6.809010676649967e-8,
           6.105779239028839e-8, 6.716485718335176e-6, 2.444808944581224e-8,
           0.9999924571194886, 3.623677491249064e-8, 5.6060940759989673e-8,
           3.9533919792269964e-8, 3.521613671950229e-8, 2.562615210438123e-8,
           1.9584957360588115e-8, 3.0036845758033746e-8, 2.0122639539235155e-8,
           9.369901721812878e-8, 7.159573942902736e-8, 3.79872737184743e-8,
           5.837684057897428e-8, 4.6593006863234134e-8]
    w7t = [0.045918421796401494, 0.05179805405164706, 0.04434107746789893,
           0.05073576269255857, 0.042815593099787545, 0.055024008469847584,
           0.03370232891566753, 0.07909071247102474, 0.03908482814327945,
           0.048069178724179394, 0.08049384816589036, 0.039249917450507654,
           0.0342284831178662, 0.05640229432349531, 0.03857107416098044,
           0.046747446382391554, 0.04921502073620415, 0.06499655559593157,
           0.04336580052064712, 0.05614959371379345]
    w8t = [6.258921377348572e-7, 3.680102939415165e-6, 7.486004924604795e-7,
           0.03498034415709424, 7.917668013849855e-7, 1.2350801714348813e-6,
           3.0460581400971164e-7, 0.08046935867869996, 5.699668814229727e-7,
           8.85393394989514e-7, 0.6357270365898279, 5.979410095673201e-7,
           2.7962721329567345e-7, 1.8757932499250933e-6, 5.070585649379864e-7,
           0.00127706653757055, 1.2457133780976972e-6, 0.11108722055351541,
           6.616561323348826e-7, 0.13644496428511088]
    w9t = [2.1869958623435718e-8, 1.2027489763838995e-7, 8.307710723713282e-8,
           0.04981239216610764, 0.19930305948126495, 9.890114269259893e-9,
           0.041826524860577356, 0.10466467192338667, 3.332719673064681e-8,
           4.0257776120217934e-8, 0.23960266714976633, 8.639963684421896e-9,
           4.958509460440708e-9, 2.0027087137090508e-8, 5.256105741068125e-9,
           0.08392271758019651, 0.28086715326117795, 2.469501355591167e-7,
           1.0503063749371977e-7, 1.1401803293988891e-7]
    w10t = [0.045918421796401494, 0.05179805405164706, 0.04434107746789893,
            0.05073576269255857, 0.042815593099787545, 0.055024008469847584,
            0.03370232891566753, 0.07909071247102474, 0.03908482814327945,
            0.048069178724179394, 0.08049384816589036, 0.039249917450507654,
            0.0342284831178662, 0.05640229432349531, 0.03857107416098044,
            0.046747446382391554, 0.04921502073620415, 0.06499655559593157,
            0.04336580052064712, 0.05614959371379345]
    w11t = [2.1869958623435718e-8, 1.2027489763838995e-7, 8.307710723713282e-8,
            0.04981239216610764, 0.19930305948126495, 9.890114269259893e-9,
            0.041826524860577356, 0.10466467192338667, 3.332719673064681e-8,
            4.0257776120217934e-8, 0.23960266714976633, 8.639963684421896e-9,
            4.958509460440708e-9, 2.0027087137090508e-8, 5.256105741068125e-9,
            0.08392271758019651, 0.28086715326117795, 2.469501355591167e-7,
            1.0503063749371977e-7, 1.1401803293988891e-7]

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
    @test isapprox(w11.weights, w11t)
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

    w2t = [0.09195751022271098, 0.07943849543837476, 0.06646961117264624,
           0.0491506750340217, 0.05429977685689461, 0.026897931456943396,
           0.010291696410768226, 0.05363160327577018, 0.01926078898577857,
           0.06072467774497688, 0.047686505861964906, 0.0030930433701959084,
           0.009416070036669964, 0.037424766409940795, 0.006951516870884198,
           0.021088101344911066, 0.10896121609215866, 0.1536264710999023,
           0.028737457378644478, 0.07089208493584212]
    w3t = [7.551916314181692e-9, 0.08663346505996936, 4.136135363410196e-9,
           0.005142916560864078, 4.818370239459004e-9, 0.08295324360196754,
           3.871908480663506e-9, 0.10161306822488796, 2.29356993539279e-8,
           1.0037393362885237e-8, 0.24803693892730824, 0.015639781543736138,
           6.322005863046181e-10, 0.11800230131930538, 3.9490169313941265e-7,
           3.2331232148000544e-9, 0.02791297634122693, 0.22534657397026278,
           2.663260304644401e-6, 0.08871561907172698]
    w4t = [1.648507594855836e-9, 7.106093993965583e-9, 2.6821898563545493e-9,
           1.7545831990829108e-9, 0.6557595753775312, 2.1295202363763833e-17,
           0.11251956687680768, 2.1063621510857507e-9, 1.5771333214892832e-16,
           1.0774661320180257e-8, 4.6619729840497305e-9, 2.4874285544821247e-17,
           2.5431768681261486e-17, 4.859461690491088e-17, 1.2603971741332548e-17,
           6.488225359675868e-9, 0.2317208043270109, 2.3331588025652267e-9,
           1.1862146674966285e-8, 2.00074810762526e-9]
    w5t = [5.993030479016171e-8, 0.09420493709134586, 7.120320215549821e-8,
           1.0079243295729554e-7, 0.32925654877467786, 6.9839749367247265e-15,
           0.031402547370096234, 0.14456962329010267, 6.033678074107784e-14,
           1.3583896973113201e-5, 4.908838631043638e-7, 8.954972670064917e-15,
           7.656045679659997e-14, 2.0262911863906142e-14, 3.3948483177339235e-15,
           8.893444358661008e-6, 0.2561544471497953, 0.14438732238772772,
           7.623839948915097e-7, 6.114009483635105e-7]
    w6t = [2.2584896800420005e-9, 2.3868994846955344e-9, 2.935218991892342e-9,
           2.7071881211103927e-9, 1.8108760930357788e-7, 9.121782217691314e-18,
           0.9999997816102653, 1.710765968766494e-9, 1.103542029729593e-15,
           1.8029071739145351e-16, 2.1210296214776463e-17, 9.92890257439193e-18,
           3.540313902777023e-17, 1.3940277230261195e-17, 6.320073872927453e-18,
           1.1006635562617111e-8, 3.022630420456309e-9, 1.7903861779628587e-9,
           7.330238630607758e-9, 2.15367117108765e-9]
    w7t = [0.03371262104831552, 0.0388620407559719, 0.029516120865239113,
           0.03149690911309137, 0.029034868047156617, 0.05575791265829153,
           0.02201380907911114, 0.0549030438553775, 0.04043465605632881, 0.1317092632242419,
           0.08283775502355642, 0.033223382918018206, 0.059150908015103934,
           0.05744126469981375, 0.036058022488107334, 0.08781271342160586,
           0.03715886265808057, 0.05201465381699867, 0.04553081997712453,
           0.04133037227846529]
    w8t = [1.4106110352044103e-8, 0.1618213401042308, 7.725824736390548e-9,
           0.009606376119749783, 9.000160950827322e-9, 1.0621069274077267e-9,
           7.232279335337557e-9, 0.18980151447101565, 2.9366139419013574e-16,
           3.4267090615887674e-7, 3.175786016900885e-9, 2.0024678480868554e-10,
           2.1582968799870234e-8, 1.510863906451459e-9, 5.056195583393224e-15,
           1.1037698949165937e-7, 0.05213822666227577, 0.42092145988299906,
           3.409953723600675e-14, 0.16571056411544613]
    w9t = [3.058055490560294e-8, 0.048069824793777756, 3.633276087273314e-8,
           5.143121732672841e-8, 0.16800929017611627, 2.3683941146465194e-9,
           0.016023735026701735, 0.07376934454421602, 2.0461310027395742e-8,
           0.03912316854880679, 0.16646772975055238, 3.0367956301040948e-9,
           2.2050282487828315e-10, 6.871525404783678e-9, 1.1512553880397138e-9,
           0.025614131446374443, 0.1307075804604388, 0.07367632211134273,
           0.25853840870862826, 3.119787282282833e-7]
    w10t = [0.09110452504787303, 0.10502024628668877, 0.07976396047134356,
            0.08511681548319099, 0.07846342945192147, 2.2249537236765373e-9,
            0.05948981592895711, 0.14836923320076992, 1.6134972467397497e-9,
            1.883953384169017e-8, 3.3055428855487943e-9, 1.3257398997267458e-9,
            8.460874398940114e-9, 2.292125900942151e-9, 1.438852847576898e-9,
            1.256062440665518e-8, 0.10041760113910592, 0.14056368755000662,
            1.8168536557489798e-9, 0.11169063156154387]
    w11t = [2.1529590959660503e-8, 0.03384254041521543, 2.557930954620391e-8,
            3.6209057521025866e-8, 0.118283376677771, 1.942092480765257e-9,
            0.01128117072550784, 0.051935742117885535, 1.6778354626481943e-8,
            0.17662241645098434, 0.13650419254097085, 2.4901843499666803e-9,
            9.954649178193517e-10, 5.634677834018165e-9, 9.440339421237391e-10,
            0.11563556734184247, 0.09202189925352097, 0.051870251647355606,
            0.21200251108417426, 2.196420054378865e-7]

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

    w1 = optimise!(portfolio; type = :HRP, rm = :SKurt, rf = rf,
                   cluster_opt = ClusterOpt(; linkage = :DBHT,
                                            genfunc = GenericFunction(; func = dbht_d),
                                            max_k = ceil(Int,
                                                         sqrt(size(portfolio.returns, 2)))))
    w2 = optimise!(portfolio; type = :HERC, rm = :SKurt, rf = rf, cluster = false)
    w3 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Min_Risk, rf = rf, l = l),
                   cluster = true)
    w4 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Utility, rf = rf, l = l),
                   cluster = false)
    w6 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Max_Ret, rf = rf, l = l),
                   cluster = false)
    w5 = optimise!(portfolio; type = :NCO,
                   nco_opt = OptimiseOpt(; rm = :SKurt, obj = :Sharpe, rf = rf, l = l),
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

    w2t = [0.09195751022271098, 0.07943849543837476, 0.06646961117264624,
           0.0491506750340217, 0.05429977685689461, 0.026897931456943396,
           0.010291696410768226, 0.05363160327577018, 0.01926078898577857,
           0.06072467774497688, 0.047686505861964906, 0.0030930433701959084,
           0.009416070036669964, 0.037424766409940795, 0.006951516870884198,
           0.021088101344911066, 0.10896121609215866, 0.1536264710999023,
           0.028737457378644478, 0.07089208493584212]
    w3t = [9.537055656561948e-7, 0.08065159914970102, 6.97429322507607e-7,
           1.636243884801297e-6, 6.970775807421223e-7, 3.475223837047256e-6,
           2.8901276220144585e-7, 0.1269025346766842, 6.27357228262538e-7,
           8.136543409694392e-7, 0.4622659726775211, 1.0093804242816126e-6,
           4.237381686192652e-7, 2.0106199669190677e-5, 1.10186359354052e-6,
           2.157409439859392e-6, 3.9496933962571675e-6, 0.1701191581088843,
           8.973744700260093e-7, 0.16002190002352545]
    w4t = [2.3518312045771565e-7, 4.548643596588275e-7, 3.289992438366905e-7,
           2.3649353086510158e-7, 0.6546702534218136, 6.07077088037061e-8,
           0.1018376879288071, 2.2938784898777355e-7, 2.504921386432834e-7,
           1.4230718705252731e-7, 2.3118132526823424e-7, 5.8659406953799776e-8,
           3.610622619246454e-8, 1.0854896396294657e-7, 3.8210498866152866e-8,
           0.1938129483117614, 0.04967546347748593, 2.6587343173773683e-7,
           6.969988018821273e-7, 2.72846338644695e-7]
    w5t = [9.329112161227789e-8, 1.8997236041327231e-6, 9.183667269717346e-8,
           1.1278547529605994e-7, 0.323307535710129, 2.5473368939712553e-8,
           6.050657719406614e-6, 0.04416525712754569, 7.422093025293419e-8,
           6.013968073613049e-8, 0.2607281038602079, 2.2840291563131592e-8,
           1.2053370397405268e-8, 6.734594412605242e-8, 1.3612437340851914e-8,
           0.13505441433828766, 0.23673428116977457, 1.0078714608278738e-6,
           2.4778495858369644e-7, 6.28157019375726e-7]
    w6t = [4.936118678171942e-8, 5.2760199097238493e-8, 6.808537976940716e-8,
           6.105353956885902e-8, 6.7161059615987615e-6, 2.444636593372152e-8,
           0.9999924575570034, 3.6234210052048526e-8, 5.6057026646045884e-8,
           3.953112344037875e-8, 3.521364356883661e-8, 2.5624345310355472e-8,
           1.9583592567291033e-8, 3.00347200640912e-8, 2.01212345794344e-8,
           9.36926021227503e-8, 7.159078911490752e-8, 3.7984586231814474e-8,
           5.837277022750332e-8, 4.6589720102054876e-8]
    w7t = [0.047491170526917106, 0.054533530611338095, 0.04332743262413755,
           0.04409183402669535, 0.04294281900935894, 0.05479618527188426,
           0.029939286967814335, 0.07378407854092589, 0.038545585072875525,
           0.04491050797340899, 0.07820933163981073, 0.041819191428891594,
           0.03694060939399915, 0.05883380726473094, 0.04119066420625541,
           0.04894563617272387, 0.050636793381874066, 0.06522417318082553,
           0.04376830962159429, 0.0600690530839385]
    w8t = [9.537055656561948e-7, 0.08065159914970102, 6.97429322507607e-7,
           1.636243884801297e-6, 6.970775807421223e-7, 3.475223837047256e-6,
           2.8901276220144585e-7, 0.1269025346766842, 6.27357228262538e-7,
           8.136543409694392e-7, 0.4622659726775211, 1.0093804242816126e-6,
           4.237381686192652e-7, 2.0106199669190677e-5, 1.10186359354052e-6,
           2.157409439859392e-6, 3.9496933962571675e-6, 0.1701191581088843,
           8.973744700260093e-7, 0.16002190002352545]
    w9t = [9.329112161227789e-8, 1.8997236041327231e-6, 9.183667269717346e-8,
           1.1278547529605994e-7, 0.323307535710129, 2.5473368939712553e-8,
           6.050657719406614e-6, 0.04416525712754569, 7.422093025293419e-8,
           6.013968073613049e-8, 0.2607281038602079, 2.2840291563131592e-8,
           1.2053370397405268e-8, 6.734594412605242e-8, 1.3612437340851914e-8,
           0.13505441433828766, 0.23673428116977457, 1.0078714608278738e-6,
           2.4778495858369644e-7, 6.28157019375726e-7]
    w10t = [0.047491170526917106, 0.054533530611338095, 0.04332743262413755,
            0.04409183402669535, 0.04294281900935894, 0.05479618527188426,
            0.029939286967814335, 0.07378407854092589, 0.038545585072875525,
            0.04491050797340899, 0.07820933163981073, 0.041819191428891594,
            0.03694060939399915, 0.05883380726473094, 0.04119066420625541,
            0.04894563617272387, 0.050636793381874066, 0.06522417318082553,
            0.04376830962159429, 0.0600690530839385]
    w11t = [9.329112161227789e-8, 1.8997236041327231e-6, 9.183667269717346e-8,
            1.1278547529605994e-7, 0.323307535710129, 2.5473368939712553e-8,
            6.050657719406614e-6, 0.04416525712754569, 7.422093025293419e-8,
            6.013968073613049e-8, 0.2607281038602079, 2.2840291563131592e-8,
            1.2053370397405268e-8, 6.734594412605242e-8, 1.3612437340851914e-8,
            0.13505441433828766, 0.23673428116977457, 1.0078714608278738e-6,
            2.4778495858369644e-7, 6.28157019375726e-7]
    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t, rtol = 5e-5)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t, rtol = 0.0001)
    @test isapprox(w11.weights, w11t)
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

    w1t = [8.749535903078065e-9, 0.021318202697098224, 0.010058873436520996,
           0.0006391710067219409, 0.23847337816224912, 5.078733996478347e-9,
           0.03433819538761402, 0.00922817630772192, 3.9969791626882106e-8,
           2.3591257653925676e-8, 5.272610870475486e-8, 4.559863014598392e-9,
           2.7523314119460396e-9, 9.64460078901257e-9, 3.0307822675469274e-9,
           0.11722640236879024, 0.4120447177289777, 0.01690611807209831,
           0.12415863804756225, 0.015607976681640013]

    w2t = [3.1657435876961315e-10, 0.011350048898726263, 0.005901925005901058,
           4.145858254573504e-9, 0.24703174384250007, 1.1554588711505922e-10,
           0.03571166800872377, 2.0093176507817576e-9, 1.1359578457903453e-9,
           5.740514558667733e-10, 9.252049263004662e-10, 1.1349813784657876e-10,
           6.974954102659066e-11, 2.3407404085962843e-10, 6.928260874063481e-11,
           0.1292316915165453, 0.44715804867316167, 2.5109012694019742e-9,
           0.12361485036662549, 1.1467800437648667e-8]
    w3t = [6.4791954821063925e-9, 0.029812930861426164, 0.010894696408080332,
           0.011896393137335998, 0.044384433675000466, 2.746543990901563e-9,
           0.005106858724009885, 0.06946926268324362, 3.8491579305047204e-9,
           0.02819781049675312, 0.28810639904355484, 9.707590311179332e-10,
           4.695579190317573e-10, 0.040824363280960715, 5.178841305932862e-10,
           0.049118688613542884, 0.20174937118162223, 0.10295500372922885,
           0.06140247037868661, 0.05608130275345569]
    w4t = [2.0554538223172535e-8, 0.029818986226667964, 0.010923608025930946,
           0.01186842576568594, 0.04541568191043348, 9.983988311413258e-9,
           0.005261250564448335, 0.06930507652058161, 2.1332962817997016e-8,
           0.027639989232562466, 0.2870831936646783, 2.7962815225575495e-10,
           3.641935755054462e-9, 0.03800725786958468, 3.3607876962123727e-9,
           0.049529929060475084, 0.2034415850017132, 0.10271915204925444,
           0.06299503778536726, 0.05599076716877515]
    w5t = [5.176352072091504e-8, 9.748253834038324e-8, 0.13465562434562653,
           1.9648153063415127e-8, 0.3298109958560023, 3.2882075020315355e-9,
           1.537467302690578e-7, 0.20839279152236462, 7.673588308583527e-9,
           5.981213903737984e-9, 0.10397892851460619, 3.0367305682038695e-9,
           1.571078971760247e-9, 6.130675259765303e-9, 1.6737117754164916e-9,
           0.029512978215898666, 0.12451184847099075, 4.137449755990313e-8,
           2.4255914493335088e-8, 0.06913641544795014]
    w6t = [2.6507703368103426e-10, 6.206863366261981e-10, 0.1274553041157384,
           4.426434822319425e-11, 0.307536338944103, 1.3906592634440003e-12,
           6.214846179668312e-10, 0.19278746427163987, 1.3983773874058484e-10,
           5.832728349828967e-11, 0.1260690819348698, 8.806828202990341e-12,
           4.09459547198998e-11, 6.702988193833297e-11, 4.374194958595253e-11,
           0.035783311293093596, 0.15095500919152735, 1.4991228323663948e-10,
           6.760875344395697e-10, 0.059413487511435624]
    w7t = [3.719680486627233e-11, 0.07905299135793928, 0.0023925141425711507,
           1.9849230810069534e-12, 0.11854062256752976, 1.5944757853507023e-12,
           8.84477407034717e-12, 0.2104859938185179, 2.524805701292716e-12,
           1.6778706984866056e-12, 0.13074334061079762, 0.004587724471183959,
           9.206543630420059e-13, 0.01435429679698965, 1.1060312219295625e-12,
           0.0240868610261247, 0.12128698234619273, 0.07317936750567806,
           6.417141638745798e-12, 0.22128930529420793]
    w8t = [3.4916400295684064e-10, 0.07725941675764415, 0.0023384453052496686,
           5.2970425973337196e-11, 0.11585103475256575, 2.548227295018703e-11,
           5.701909488419777e-11, 0.20571024083695255, 1.4623343192289088e-11,
           2.1557028231014734e-11, 0.1378305483380772, 0.004836407198832101,
           2.950603663314908e-11, 0.015132398067145761, 2.639120634087331e-11,
           0.02539254043238399, 0.1278615845099619, 0.07151903170585491,
           3.5588414749190514e-11, 0.2162683514830301]
    w9t = [4.232017624008816e-9, 1.0625294718226186e-8, 1.3614292906388451e-8,
           1.0520184720959675e-8, 0.8140445091234294, 4.477531126131145e-11,
           0.11970512690059287, 5.128020457259785e-9, 1.578377420905679e-9,
           2.992479895874701e-10, 2.828205379020583e-10, 4.4729923821393855e-11,
           9.515036748696208e-11, 4.864679440371178e-11, 8.204484953291807e-11,
           0.015278913790804101, 0.04711885411046788, 5.6375830018990446e-9,
           0.0038525359151268404, 7.926392399873701e-9]
    w10t = [4.232017624008816e-9, 1.0625294718226186e-8, 1.3614292906388451e-8,
            1.0520184720959675e-8, 0.8140445091234294, 4.477531126131145e-11,
            0.11970512690059287, 5.128020457259785e-9, 1.578377420905679e-9,
            2.992479895874701e-10, 2.828205379020583e-10, 4.4729923821393855e-11,
            9.515036748696208e-11, 4.864679440371178e-11, 8.204484953291807e-11,
            0.015278913790804101, 0.04711885411046788, 5.6375830018990446e-9,
            0.0038525359151268404, 7.926392399873701e-9]
    w11t = [2.5413752274815785e-9, 6.380611609082146e-9, 8.175539377645492e-9,
            6.317491847553192e-9, 0.48884308463583465, 2.968826473976512e-10,
            0.07188430463565379, 3.0794352278131317e-9, 1.0465429588560942e-8,
            1.9841634345916267e-9, 1.8752412359741913e-9, 2.9658170603193523e-10,
            6.308943970378847e-10, 3.225525117111198e-10, 5.43998276025859e-10,
            0.10130681949036092, 0.3124215054367816, 3.3854333929559315e-9,
            0.02554423474584885, 4.759889744801148e-9]

    w12t = [5.800275953464036e-9, 0.06645732886835147, 0.03032885702630591,
            0.008093869418402832, 0.5931082772572596, 4.2703131594422125e-17,
            0.08511943539936005, 0.05763657850972975, 3.2371771823476725e-15,
            2.505194259848522e-15, 2.5594511478480278e-14, 1.209248903401832e-16,
            3.1561307440399247e-16, 5.589139504367345e-16, 2.844558506596257e-16,
            4.800316791252493e-8, 1.732698360619526e-7, 0.09290478266595871,
            5.934650199283608e-8, 0.06635058443481713]

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

    w1t = [0.10871059727246735, 0.05431039601849186, 0.10533650868181384,
           0.027317993835046576, 0.07431929926304212, 0.00954218610227609,
           0.024606833580412473, 0.04020099981391352, 0.022469670005659467,
           0.05391899731269113, 0.04317380104646033, 0.006286394179643389,
           0.006060907562898212, 0.0320291710414021, 0.005842729905950518,
           0.044283643115509565, 0.10749469436263087, 0.09602771642660826,
           0.04410905860746655, 0.09395840186561562]

    w2t = [0.08320752059200986, 0.08290256524137433, 0.07517557492907619,
           0.06023885608558014, 0.06626202578072789, 0.024707098983435642,
           0.029699159972552684, 0.0942847206692912, 0.019894956146041556,
           0.02777710488606625, 0.03221349389416141, 0.01243771863240931,
           0.009827277935290812, 0.027588252827801342, 0.010488817689171098,
           0.0192723640402875, 0.09460246164880029, 0.10914949211014122,
           0.024315592933065365, 0.09595494500271598]
    w3t = [0.11749818113343935, 0.05870055826126679, 0.11385134924830828,
           0.029526234501200906, 0.08032687433988389, 0.008852903280265263,
           0.026595918536857853, 0.04345063385165379, 0.02084656630749761,
           0.029413320573451408, 0.04005512791398602, 0.005832294513804399,
           0.0033062821268677552, 0.02971553377155059, 0.005420678468821243,
           0.02415714416132541, 0.11618399112873753, 0.10379008396251159,
           0.040922826850160604, 0.10155349706840978]
    w4t = [0.11749818113343935, 0.05870055826126679, 0.11385134924830828,
           0.029526234501200906, 0.08032687433988389, 0.008852903280265263,
           0.026595918536857853, 0.04345063385165379, 0.02084656630749761,
           0.029413320573451408, 0.04005512791398602, 0.005832294513804399,
           0.0033062821268677552, 0.02971553377155059, 0.005420678468821243,
           0.02415714416132541, 0.11618399112873753, 0.10379008396251159,
           0.040922826850160604, 0.10155349706840978]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test !isapprox(w1.weights, w2.weights)
    @test isapprox(w3.weights, w4.weights)
end
