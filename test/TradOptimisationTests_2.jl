@testset "WR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = WR()

    obj = MinRisk()
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.8612942682723194e-12, 0.22119703870515675, 2.828265759908966e-12,
          2.1208855227168895e-12, 3.697781891063451e-12, 3.24353226480368e-12,
          0.02918541183751788, 4.420452260557843e-12, 2.3374667530908414e-12,
          2.8919479333342058e-12, 0.5455165570312099, 1.4490684503326206e-12,
          1.9114786537154165e-12, 2.7506310060540026e-12, 3.640894035272413e-11,
          2.7909315847715066e-12, 1.694217734278189e-12, 3.798024068784819e-12,
          2.7258514165515688e-12, 0.20410099234818463]
    riskt = 0.03217605105120276
    rett = 0.0005011526784679896
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [8.31617952752212e-13, 0.22119703875062227, 6.832160545751061e-13,
          6.96193291795788e-13, 4.036019076776619e-12, 4.3471614749828785e-13,
          0.029185411841700936, 9.078156634695366e-13, 6.665895218019583e-13,
          1.8373133578563436e-12, 0.5455165570616287, 2.7833273599028926e-13,
          5.493760560329698e-13, 4.481197759509806e-13, 3.32575216803394e-11,
          1.1556828005777454e-12, 4.0624744116434915e-12, 6.860861190643671e-13,
          8.372817139938184e-13, 0.20410099229467973]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.635110175804277e-10, 0.22119708420320755, 3.820489210458886e-10,
          1.4187173357103828e-9, 2.6363330885995266e-9, 5.189538035605022e-10,
          0.02918540888981707, 7.808080527361188e-10, 3.6109547229216355e-10,
          5.363142216800803e-10, 0.5455165396306245, 1.2773165351656046e-10,
          3.490104383331551e-10, 1.9841993233744004e-10, 1.7436321318876394e-8,
          5.702494806667545e-10, 1.2872101855890893e-9, 3.0511723902483676e-10,
          4.117916217763004e-10, 0.2041009396927171]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [4.619204869507631e-11, 0.2211970401400894, 4.3701774419007517e-11,
          8.353407625940733e-11, 1.358657025936017e-10, 1.0578194758385017e-10,
          0.029185412435982716, 1.8405841561907286e-10, 3.740225068356339e-11,
          6.44640015460276e-11, 0.5455165540499841, 1.992397665103746e-11,
          4.891135996048911e-11, 3.833815555309313e-11, 6.709113288205949e-10,
          1.1391046858565562e-10, 3.582241940912807e-11, 5.843933123280907e-11,
          4.670206096914456e-11, 0.20410099163998444]
    riskt = 0.03217605106792421
    rett = 0.0005011526792393935
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.780104323028494e-11, 0.22119703859779105, 6.276295210763981e-11,
          6.945685168435587e-11, 1.272876224890368e-10, 5.0924754939132304e-11,
          0.029185411803606478, 4.2025073814241164e-11, 5.136255227405268e-11,
          8.72807827590185e-11, 0.5455165573896047, 4.3304404498191726e-11,
          3.360166563092323e-11, 5.6057018320553576e-11, 2.6187078514462165e-10,
          3.8203524552350216e-11, 5.947703697380478e-11, 7.050017028250539e-11,
          6.195177931996556e-11, 0.20410099102512988]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.1929038472765297e-10, 0.22119705255880093, 1.580011094715284e-10,
          4.847576119579443e-10, 1.120981268022772e-9, 1.9690189492108395e-10,
          0.02918541304709744, 1.803814386688887e-10, 1.5236190246315402e-10,
          2.4029948591279455e-10, 0.5455165527590767, 3.7615507614648974e-11,
          1.209905706287236e-10, 6.875025162230649e-11, 1.725560852711778e-9,
          1.970196651568895e-10, 6.270793283913616e-10, 1.034939719979999e-10,
          2.0441141224195639e-10, 0.20410097589712836]
    @test isapprox(w6.weights, wt, rtol = 1.0e-7)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [6.957399908772388e-9, 2.039271731107526e-8, 5.497898695084438e-9,
          1.1584017088731345e-8, 0.3797661371235164, 1.9162230097305403e-9,
          0.17660512608552742, 1.0666210782547244e-8, 1.0225338760635262e-8,
          0.04075088574289245, 0.05638221165264284, 2.089109162284139e-9,
          1.23279550928153e-9, 9.013331222315118e-9, 2.1778889815995123e-9,
          0.15854733523481268, 0.18794817199402036, 1.1268704949879534e-8,
          3.4599644297968083e-8, 4.545308055104026e-9]
    riskt = 0.04173382316607199
    rett = 0.0014131701721435356
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.346041541755758e-8, 2.790563286730082e-7, 2.0230169044344763e-8,
          4.148314099306561e-8, 0.3797663868865331, 6.721680190226824e-9,
          0.17660488556016016, 4.094018803728532e-8, 3.2462164530088474e-8,
          0.040750676423078724, 0.0563819780260819, 8.094382876600906e-9,
          4.282569514075461e-9, 6.146028749441822e-8, 7.254495678879109e-9,
          0.15854739341595234, 0.187947992353654, 3.965732629982512e-8, 9.90370453631504e-8,
          2.319434573454084e-8]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.3449084787703177e-7, 4.694914111656391e-7, 1.0656708224145255e-7,
          2.1792722845632984e-7, 0.37976702290626235, 3.7648369976100124e-8,
          0.1766045318438728, 2.2074894042984511e-7, 1.9142894435540533e-7,
          0.04075031946476531, 0.056380347301646885, 4.0148020588606196e-8,
          2.4311369893956776e-8, 1.85275632936872e-7, 3.9552840747762915e-8,
          0.1585475070017042, 0.18794747366731218, 2.450350381229234e-7,
          7.912653792134936e-7, 9.39233301941589e-8]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.5566238289527526e-9, 1.570956878493887e-9, 1.608762274371583e-9,
          1.6258884417989995e-9, 1.4600423065623333e-7, 1.0857056194353091e-9,
          0.9999998292073036, 1.3820011237558862e-9, 1.5713320679343279e-9,
          1.430630625805485e-9, 1.3707526492120582e-9, 1.0935734943467402e-9,
          8.72584405607747e-10, 1.247652840328386e-9, 9.237093680924222e-10,
          1.3747133091550288e-9, 1.577564048696993e-9, 1.4136760653056833e-9,
          1.5661129788821178e-9, 1.51622582562161e-9]
    riskt = 0.24229070767001235
    rett = 0.001845375606854123
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.5443437027516e-11, 5.2350950229862524e-11, 9.066935701147142e-11,
          7.120923208257929e-11, 0.8503259395366981, 4.355477349277696e-11,
          0.149674059466214, 2.4290625152597366e-12, 6.509078740819352e-11,
          1.0998068341006642e-11, 1.701927079115916e-12, 4.105730646991286e-11,
          5.880134786712474e-11, 2.270001553981933e-11, 5.835110319164303e-11,
          2.0849838100796641e-10, 1.1412283881505805e-10, 8.007551744972036e-12,
          7.751897551597984e-11, 3.458306521316913e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.4427026397905277e-9, 2.953342366847875e-9, 3.488355282693272e-9,
          3.3039868175714513e-9, 0.8533950216137636, 1.3232043581911538e-9,
          0.14660493036607455, 2.166597089120636e-9, 3.3970019446108583e-9,
          2.4108748932365233e-9, 2.131897673067489e-9, 1.3895315918812704e-9,
          1.013781789198535e-9, 1.7170881348384435e-9, 1.04480062674174e-9,
          6.196346478260153e-9, 4.233214436766981e-9, 2.2605081913012243e-9,
          3.736694833735987e-9, 2.810232763884542e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "RG" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = RG()

    obj = MinRisk()
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [3.4835106277243952e-12, 0.12689181906634667, 6.919061275326071e-12,
          0.2504942755637169, 1.6084529746393874e-11, 1.4834264958811167e-11,
          1.7580530855156377e-12, 0.1053793586345684, 3.0155464926527536e-12,
          0.01026273777456012, 0.39071437611839677, 1.3196103594515992e-12,
          4.2161735386690386e-14, 4.265414596070901e-12, 0.013789122525542109,
          4.212243490314627e-11, 1.869748261234954e-11, 1.0033734598934388e-11,
          4.125432454379667e-12, 0.10246831019016761]
    riskt = 0.06215928170399987
    rett = 0.0005126268312684577
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [4.31739476902063e-13, 0.12689181923588902, 1.0001261676136084e-12,
          0.2504942756499968, 2.131936582646773e-12, 2.3266184174579186e-12,
          1.647107055930365e-13, 0.10537935870649043, 4.1805809855091457e-13,
          0.010262737527944765, 0.39071437595899217, 1.2349650797855326e-13,
          1.5343018986211152e-13, 5.450486000680121e-13, 0.013789122514260939,
          3.873306458160529e-12, 2.8657014378612625e-12, 1.5103525345867617e-12,
          6.085879558213952e-13, 0.10246831039027275]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.1239579277328575e-10, 0.12689181129347227, 2.418762432471899e-10,
          0.250494274768601, 4.38531025165919e-10, 4.898101233015144e-10,
          1.6071845088499507e-10, 0.10537935622199111, 1.8804448573894748e-10,
          0.010262743009313536, 0.39071438299790107, 1.4470505605073887e-10,
          1.0335184047755484e-10, 2.0313327148418579e-10, 0.013789122754782501,
          1.2359720171101007e-9, 4.48314862735738e-10, 2.887791589718486e-10,
          1.9495249725613257e-10, 0.10246830460335363]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [2.818778936301644e-13, 0.1268918192291803, 8.441768433371781e-13,
          0.25049427565136034, 2.4865354889173847e-12, 3.866580247555462e-12,
          4.71832585931689e-14, 0.10537935870826316, 2.5951255443382816e-13,
          0.010262737495686404, 0.39071437596255, 4.097093953278833e-14,
          2.8958901395609205e-13, 7.759928478674261e-13, 0.013789122514889859,
          3.4028038727441466e-12, 3.0875826878467018e-12, 1.9463819929445524e-12,
          4.491400980692589e-13, 0.10246831042029175]
    riskt = 0.06215928170296525
    rett = 0.0005126268313959722
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.039072474357371e-13, 0.1268918192260751, 1.1567779084139665e-12,
          0.2504942757008095, 2.864689195079638e-12, 3.814896736433305e-12,
          3.8113550550639973e-13, 0.1053793587367674, 6.167236334759777e-13,
          0.010262737405340411, 0.3907143758931907, 3.674820735298308e-13,
          1.8233065652857641e-13, 8.537775135190291e-13, 0.01378912249381893,
          7.52279880237397e-12, 3.1489482473264856e-12, 2.143105171588191e-12,
          7.948164501810529e-13, 0.10246831051954672]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [5.0436702203930716e-11, 0.12689181408577818, 7.245267592845763e-11,
          0.25049427580583383, 1.828332875862907e-10, 8.19515004442304e-11,
          4.676911589355641e-11, 0.10537935834694909, 5.6959354794897244e-11,
          0.010262739938997605, 0.39071437589107355, 4.403900719334566e-11,
          3.1280083002615885e-11, 5.393777839889425e-11, 0.013789122501750614,
          4.6078935703592717e-10, 1.1657274429329749e-10, 1.0120080504013221e-10,
          5.965245017501347e-11, 0.10246831207074228]
    @test isapprox(w6.weights, wt)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [6.09724566157312e-10, 1.2600001874338616e-9, 1.1376809473114097e-9,
          0.3052627178260572, 0.25494610585828753, 1.8310456179966306e-10,
          0.09576793051248834, 3.7212437966398855e-9, 1.284058570384776e-9,
          2.9231767008643053e-9, 8.664801052496888e-10, 2.2919525949753356e-10,
          9.024587888396662e-11, 4.047257690673743e-10, 8.93417076405552e-11,
          0.12660213909219864, 0.1910488629796444, 1.5541667982439667e-9,
          0.02225468123935576, 0.0041175481388230385]
    riskt = 0.08445623506377935
    rett = 0.0012690611588731012
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.2194625863480789e-8, 3.383078970582277e-8, 2.699313905361916e-8,
          0.3022427423750614, 0.2554082783932047, 4.824015998214284e-9, 0.09551166464482462,
          0.027110130966319045, 2.5938794892273876e-8, 6.239795500214288e-8,
          2.865752282567116e-8, 5.5352340719494474e-9, 2.8034994978963547e-9,
          9.783772028750279e-9, 2.9489655995001644e-9, 0.12507321646088568,
          0.15613331292315097, 4.581868702820075e-8, 0.021599901000487864,
          0.016920491509064172]
    @test isapprox(w8.weights, wt, rtol = 5.0e-8)

    w9 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.00010687613686575052, 0.00018064674908548485, 0.00023319844300520535,
          0.22378979416635794, 0.32378145592084273, 3.688081307945745e-5,
          0.10636835348981784, 0.0002929365679786915, 0.00023413224794467452,
          0.00022854707333556224, 0.00012284815741980983, 3.994849774235545e-5,
          2.3027331773687413e-5, 6.327454574961834e-5, 2.2131018585207834e-5,
          0.1403659005341131, 0.1701490919546012, 0.00018093427234871696,
          0.03297207855612901, 0.000807943523223788]
    @test isapprox(w9.weights, wt, rtol = 5.0e-6)

    obj = MaxRet()
    w10 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [3.1787004844637487e-10, 3.170154234348516e-10, 3.369759520528265e-10,
          3.3317156263838264e-10, 1.159467049603122e-8, 2.0869446614742875e-10,
          0.9999999834622247, 2.7142446908769163e-10, 3.1716632434310177e-10,
          2.811045475363529e-10, 2.690140573125285e-10, 2.0965521139526677e-10,
          1.6191211469487086e-10, 2.4289032625368277e-10, 1.733570647026064e-10,
          2.825117749661399e-10, 3.2313605322415803e-10, 2.7796794747771806e-10,
          3.1506785555133194e-10, 3.041695491311906e-10]
    riskt = 0.7651914993934192
    rett = 0.0018453756417150151
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.2287037120099505e-12, 3.21520946421435e-12, 5.4472659952629796e-12,
          4.329057386853184e-12, 0.8503250088752031, 2.3430474851009177e-12,
          0.14967499106532306, 2.959528005266375e-13, 3.976852765762162e-12,
          8.068081181356933e-13, 5.047120319368428e-14, 2.208096202655667e-12,
          3.2714174793866527e-12, 1.1561895883055539e-12, 3.2286863519798297e-12,
          1.2623311991918676e-11, 6.818098901297006e-12, 6.237598813448599e-13,
          4.683260799268753e-12, 2.167581556647316e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.437805845847735e-9, 2.948680963606501e-9, 3.4821983399797654e-9,
          3.2987045132950436e-9, 0.853395045498853, 1.3212465569277058e-9,
          0.14660490656095476, 2.16332488113349e-9, 3.3911841492352425e-9,
          2.4070897465575284e-9, 2.1284629861243294e-9, 1.3872766233666878e-9,
          1.0120704647945913e-9, 1.7144604541833644e-9, 1.043123164366909e-9,
          6.185764613861165e-9, 4.2256766357794935e-9, 2.256988866515922e-9,
          3.7306944893791315e-9, 2.805438861307354e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-8)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 1e-10

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-10

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 5e-10

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "CVaR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = CVaR()

    obj = MinRisk()
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.965769176302831e-11, 0.04242033378148941, 9.902259604479418e-11,
          2.585550936025974e-10, 0.007574028506215674, 1.1340405766435789e-10,
          1.3814642470526227e-11, 0.09464947974750273, 4.637745432335755e-11,
          6.484701166044592e-11, 0.3040110652312709, 5.940889071027648e-11,
          3.420745138676034e-11, 0.06564166947730173, 9.544192184784114e-11,
          0.029371611149186894, 1.241093002048221e-10, 0.36631011287979914,
          5.953639120278758e-11, 0.09002169815885094]
    riskt = 0.01704950212555889
    rett = 0.0003860990591135937
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [9.204153687833781e-11, 0.04242033344850122, 9.85123787201143e-11,
          2.742462482595728e-10, 0.007574028452637937, 1.0444892006359854e-10,
          1.1253189456551386e-11, 0.09464947950883604, 4.304524873576078e-11,
          5.769528999444338e-11, 0.3040110652564113, 5.2022422074226226e-11,
          2.8850585492519713e-11, 0.06564166930506558, 9.716068090161583e-11,
          0.029371611163095106, 1.2140041190522942e-10, 0.366310112784046,
          5.621940710019265e-11, 0.09002169904451053]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.0779002913123264e-9, 0.0424202478835117, 1.1394636976177199e-9,
          3.2034009048719517e-9, 0.007574041437995398, 1.214157664740724e-9,
          2.0866351379515966e-10, 0.09464947248139145, 5.475138966935703e-10,
          7.320881150782838e-10, 0.304010968264335, 7.160260081039788e-10,
          4.2716139435542236e-10, 0.06564164417407643, 1.1869177983842032e-9,
          0.02937161641119216, 1.343287701885978e-9, 0.3663101613158242,
          6.666278889397393e-10, 0.09002183556846477]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [3.59415027814126e-12, 0.03635916778167782, 3.552811195795529e-12,
          1.2653345986437583e-11, 0.017177347037601237, 3.337778093941137e-12,
          2.5795636217644956e-13, 0.09164255461978174, 1.734150257351883e-12,
          2.0375788095235085e-12, 0.32258428294244085, 1.6071720149229441e-12,
          7.478631306913877e-13, 0.03955071531394016, 3.4490687806596575e-12,
          0.030919378393966135, 4.608903322785033e-12, 0.3733298428934439,
          2.2844085399535255e-12, 0.08843671097728309]
    riskt = 0.017056094628321805
    rett = 0.0004072001780847205
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.2446651084459793e-10, 0.03635920624448488, 1.2099878252057046e-10,
          4.3353705390596115e-10, 0.017177327412953393, 1.1523416518156477e-10,
          1.1529023942129099e-11, 0.091642549012939, 6.068436102923306e-11,
          7.000020425265046e-11, 0.32258415691878095, 5.729097510879122e-11,
          2.6990419802016365e-11, 0.039550772304670946, 1.2030281101532894e-10,
          0.030919377851809874, 1.525456685804843e-10, 0.373329855951351,
          7.83789622418283e-11, 0.08843675293105117]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.8770987535438225e-10, 0.03635922666523639, 1.9543535804493658e-10,
          5.920383859192107e-10, 0.017177311523088035, 1.753401786031763e-10,
          4.144490215167871e-11, 0.0916425570130048, 1.0276360004231398e-10,
          1.1514827254900474e-10, 0.3225840596074403, 9.619163005772905e-11,
          5.6449361466469335e-11, 0.0395507982905123, 1.8272605765002315e-10,
          0.030919377989654204, 2.4645160197705085e-10, 0.3733298632135168,
          1.2924986342087758e-10, 0.08843680357659808]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [2.305962223730381e-9, 3.061529980299523e-9, 3.6226755773356135e-9,
          1.968988878444111e-9, 0.562845489616387, 6.289605285168684e-10,
          0.044341929816432854, 5.465596947274736e-9, 3.128822366888805e-9,
          1.6003971393612084e-9, 4.52394176361636e-9, 5.75356193927518e-10,
          3.1728380155852195e-10, 1.240519587265295e-9, 3.422838872379099e-10,
          0.20959173183485763, 0.18322079783245407, 6.034806498955341e-9,
          1.1803331196573864e-8, 4.279412029260546e-9]
    riskt = 0.03005421217653932
    rett = 0.0015191213711409513
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.304102546507255e-9, 3.4058580184231964e-9, 3.264755599420317e-9,
          2.168529122297692e-9, 0.5593117496370377, 7.139134073206089e-10,
          0.029474976465948034, 4.9115201797004046e-8, 2.799982453416685e-9,
          1.6115667456355964e-9, 8.831047243202884e-9, 6.334407262324075e-10,
          4.1948103829488986e-10, 1.3615475408342457e-9, 4.3425958678632566e-10,
          0.20296977157601848, 0.20824336902427606, 2.80725022409301e-8,
          2.143306519713471e-8, 6.727466637284185e-9]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [6.64088087345985e-8, 9.538860932028128e-8, 1.0453296168596982e-7,
          5.7868546146211784e-8, 0.5622705387604069, 2.1054154746516597e-8,
          0.041923329563299166, 2.0823113529748093e-7, 8.937992205180785e-8,
          4.70377466776035e-8, 1.957973941516057e-7, 1.911064451024525e-8,
          1.2600789308841253e-8, 3.954227708856903e-8, 1.3000474018720965e-8,
          0.20851348525481309, 0.18729086427243372, 2.3196503370395504e-7,
          4.399321628916103e-7, 1.402983868138997e-7]
    @test isapprox(w9.weights, wt)

    obj = MaxRet()
    w10 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.2979096100520238e-8, 1.3939595287583712e-8, 1.6664693909263272e-8,
          1.6061904799427314e-8, 2.5289048640527673e-7, 1.0111632690844616e-8,
          0.9999995079388162, 1.1387303215317185e-8, 1.4887953777768852e-8,
          1.2073758025755345e-8, 1.1359241573256225e-8, 1.0083482858838886e-8,
          9.063418984706977e-9, 1.0812267724771105e-8, 9.4351700847673e-9,
          2.3329395161801666e-8, 1.7731658764712164e-8, 1.1676397354409723e-8,
          1.503052388589023e-8, 1.2543203151639056e-8]
    riskt = 0.08082926752528491
    rett = 0.001845375306063485
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.622468763587675e-11, 2.4305243326642927e-11, 4.2602118130674744e-11,
          3.3294464132139124e-11, 0.8503255339884118, 2.152943423043018e-11,
          0.14967446554001587, 5.218627470261068e-13, 3.039577793482502e-11,
          4.597121407190349e-12, 1.4530843735638327e-12, 2.030952150941086e-11,
          2.8702503009729792e-11, 1.1510274994891938e-11, 2.8604495580622442e-11,
          9.834475691645756e-11, 5.3801390739755065e-11, 3.1622056975748866e-12,
          3.635795488049805e-11, 1.5855285676325412e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.4358139548252643e-9, 2.9443323823815057e-9, 3.4781287468046434e-9,
          3.294150521464759e-9, 0.8533951211371982, 1.3187368397190805e-9,
          0.14660483098984345, 2.1595613343429504e-9, 3.3865724969917085e-9,
          2.4029865798428798e-9, 2.12499035870577e-9, 1.3849425461316505e-9,
          1.0104473225853474e-9, 1.7114267991768829e-9, 1.0412585823551084e-9,
          6.179367364251867e-9, 4.220597690045663e-9, 2.253256560559402e-9,
          3.724982057897676e-9, 2.80140636669256e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 1e-10

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 1e-10

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
end

@testset "RCVaR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = RCVaR()

    obj = MinRisk()
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.022526773522155673, 0.02205967648382663, 2.319996990805667e-11,
          0.029966893048161407, 0.006773557016985066, 0.021160245217902482,
          5.016724411235469e-13, 0.11191320782878157, 2.6370102192581807e-12,
          1.2019874588666696e-11, 0.31895383693063833, 2.262605967359396e-12,
          1.135073650138578e-12, 0.09089853669776103, 7.539013998246915e-13,
          0.045111723951082434, 3.636547744318311e-11, 0.23185661850827086,
          3.242869688736363e-12, 0.09877893071231604]
    riskt = 0.03439845483008025
    rett = 0.00038005938396668074
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.022526773063748538, 0.022059676691070184, 5.723643309528018e-11,
          0.02996689318034236, 0.006773556850550276, 0.021160245111411378,
          5.743328275069653e-13, 0.11191320782571325, 6.241801125982634e-12,
          2.8895570803409416e-11, 0.3189538369788915, 5.107842327278829e-12,
          2.2319962440376507e-12, 0.09089853661111044, 1.3262243693957414e-12,
          0.04511172388317882, 9.937732593535384e-11, 0.2318566188328171,
          7.77470047767543e-12, 0.09877893076239985]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.0225267730544479, 0.022059677015342647, 9.653262727328893e-11,
          0.02996689340532921, 0.006773556341885283, 0.021160244761221407,
          5.4380155368997106e-12, 0.11191320799501756, 1.3674766996945956e-11,
          5.0177662041481294e-11, 0.3189538367970027, 1.1841094059378867e-11,
          7.294442401812083e-12, 0.09089853668947048, 6.293354511679175e-12,
          0.04511172389162617, 1.4998201913543448e-10, 0.23185661881237096,
          1.591846279619127e-11, 0.09877893087913334]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [0.02575961059250994, 0.02352898917138473, 4.482663003519851e-11,
          0.030973036309681387, 0.004266952200873158, 0.016564945623396735,
          4.944858444416546e-14, 0.11212820247498569, 4.9680569601583026e-12,
          2.4961050592557358e-11, 0.31883259040215267, 3.12147570775239e-12,
          9.074138700224762e-13, 0.09274114713379317, 7.00021236550757e-13,
          0.04480467355006564, 2.935393349034395e-10, 0.23106750664936884,
          6.173112445531593e-12, 0.09933234551254147]
    riskt = 0.03439937593563209
    rett = 0.0003821727064259498
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.025759603154404134, 0.023528983143001437, 3.395131727963874e-10,
          0.030973032231232378, 0.004266965420035811, 0.016564959805366864,
          5.690264475199786e-12, 0.11212820314664744, 3.996168497540149e-11,
          1.7897305381947114e-10, 0.31883258681252735, 2.705146728186651e-11,
          1.0620711749158247e-11, 0.09274113900275943, 1.0240175721677321e-11,
          0.0448046740187296, 2.1004785182935535e-9, 0.23106750701746848,
          4.871601460516724e-11, 0.09933234348658192]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.02575960451439615, 0.023528976850317154, 7.983767327331582e-10,
          0.030973027593350605, 0.004266980237179593, 0.016564971038498747,
          4.637836268780547e-11, 0.11212820648963116, 1.1930415539042335e-10,
          4.30109523916112e-10, 0.31883257928869496, 9.140273584033312e-11,
          5.5508973589153297e-11, 0.09274112931817069, 5.494877489631604e-11,
          0.044804674168913386, 4.9315834991818004e-9, 0.23106750191070066,
          1.3836476517736526e-10, 0.09933234192416929]
    @test isapprox(w6.weights, wt, rtol = 1.0e-7)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [4.9839432322477e-10, 8.448693289397689e-10, 1.173858895147354e-9,
          9.375514999679668e-10, 0.573440952697266, 7.451535889801865e-11,
          0.05607344635024722, 1.8370189606252853e-9, 8.256113918546924e-10,
          4.3173297057509325e-10, 1.0216619066483312e-9, 4.6589215379003155e-11,
          6.2787877132137955e-12, 2.0536944124409375e-10, 1.550936856433007e-12,
          0.14189709250803045, 0.2285884636440746, 1.020124064623182e-9,
          3.463719056083844e-8, 1.2380641433681093e-9]
    riskt = 0.0642632537835233
    rett = 0.0015273688513609762
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [4.5479034211371995e-9, 8.671827869678778e-9, 1.0994855997347118e-8,
          1.0595648572365527e-8, 0.4496413256452418, 1.123969446765462e-9,
          0.048473299215131, 0.06272444786721185, 6.224549555694726e-9,
          4.168122980471285e-9, 2.3044528006061617e-8, 8.688279026781881e-10,
          6.459782247496969e-10, 2.2524479146893874e-9, 5.888771273514278e-10,
          0.11095200843066484, 0.27950312813503714, 2.1292024349338408e-8,
          0.04870567343073953, 2.2256412634338147e-8]
    @test isapprox(w8.weights, wt, rtol = 5.0e-6)

    w9 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.9509066172947846e-8, 3.5794865989268365e-8, 4.165842047934432e-8,
          4.21807973700811e-8, 0.542469175785578, 5.406335975721144e-9, 0.05298057718596724,
          1.8018112072748158e-7, 2.7234957043189874e-8, 1.800504632064351e-8,
          4.67699567945515e-8, 4.306522666393123e-9, 3.1169672732397987e-9,
          1.0058751708670444e-8, 2.8508354189681885e-9, 0.13908458466107587,
          0.21472767530635967, 5.431849048951132e-8, 0.05073743421439071,
          6.145449429898624e-8]
    @test isapprox(w9.weights, wt, rtol = 1.0e-5)

    obj = MaxRet()
    w10 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.831302064629739e-9, 1.8517749307082367e-9, 1.912037493612443e-9,
          1.9163167053743124e-9, 1.4740990018868178e-7, 1.2501876054188206e-9,
          0.9999998235090382, 1.6133782010204823e-9, 1.846635679461269e-9,
          1.6667654046232868e-9, 1.5980748195724616e-9, 1.2577511709365595e-9,
          9.960336031414542e-10, 1.445381905668743e-9, 1.0574927845375882e-9,
          1.6733695789610779e-9, 1.882772637074646e-9, 1.6514675241347527e-9,
          1.8483889012850246e-9, 1.7819307614324069e-9]
    riskt = 0.18292982208715033
    rett = 0.0018453756009504528
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.333403861622826e-11, 4.922256307735439e-11, 8.523459626114065e-11,
          6.694101081654318e-11, 0.8503258805420554, 4.091644825463561e-11,
          0.14967411852091908, 2.3117069939643017e-12, 6.118758294159243e-11,
          1.0360634821288679e-11, 1.5673814121497009e-12, 3.856312834148788e-11,
          5.522728487416593e-11, 2.1306894002254063e-11, 5.480721522082859e-11,
          1.9581127116156035e-10, 1.0727320511503262e-10, 7.553739861611402e-12,
          7.287646826587675e-11, 3.2530474858162e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.0070197384895404e-9, 3.367618634441536e-9, 4.184228835659659e-9,
          3.773260903968188e-9, 0.8533949709061769, 1.3228106352178148e-9,
          0.14660497690496824, 2.293045218600837e-9, 3.6474419494561004e-9,
          2.4808328974906687e-9, 2.2035860732098937e-9, 1.3723066061014173e-9,
          9.77344221666783e-10, 1.7596232789413378e-9, 9.932852864994782e-10,
          6.819413329731061e-9, 4.68651011449551e-9, 2.414175137870143e-9,
          3.904568403325466e-9, 2.98178375025297e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 1e-10

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-9

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 5e-10

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10
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

    obj = MinRisk()
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.2500112838486011e-8, 0.15679318740948175, 1.0361131210275206e-8,
          0.01670757435974688, 1.501287503554962e-8, 6.061816596694208e-8,
          0.014452439886462113, 0.15570664400078943, 9.522408711497533e-9,
          1.059085220479153e-8, 0.452447219917494, 8.305434093731495e-9,
          1.2081476879327763e-8, 2.3952270923291378e-8, 0.004794389308245565,
          1.7142647790367886e-7, 0.01841950032750946, 1.9211685081872636e-7,
          1.2233795563491733e-8, 0.18067850606841873]
    riskt = 0.024507972823062964
    rett = 0.00046038550243244597
    @test isapprox(w1.weights, wt, rtol = 1.0e-5)
    @test isapprox(r1, riskt, rtol = 1.0e-5)
    @test isapprox(ret1, rett, rtol = 1.0e-5)

    w2 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1175842274627593e-8, 0.15678962525657866, 9.269168746643073e-9,
          0.016707074814442415, 1.3452573176413182e-8, 5.436355786124658e-8,
          0.014451271346349, 0.15570865514232668, 8.500395881111563e-9,
          9.498747909712649e-9, 0.4524457444044333, 7.375543927729117e-9,
          1.0692599637688104e-8, 2.157786807878457e-8, 0.0047941761453754485,
          1.5347677599844998e-7, 0.018424665778813513, 1.7044737317526053e-7,
          1.0938910978010825e-8, 0.18067830634232349]
    @test isapprox(w2.weights, wt, rtol = 5.0e-5)

    w3 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [7.744149706504591e-10, 0.15679410893441026, 7.274156507917334e-10,
          0.016706213514055972, 1.5810201387153583e-9, 1.6952606171928384e-9,
          0.014453448797547326, 0.15570508057409402, 5.943675126172893e-10,
          6.193547072116074e-10, 0.4524506451372982, 3.8768620409309906e-10,
          4.1492798285227507e-10, 9.744373592147397e-10, 0.004792324631608908,
          4.6408869326508805e-9, 0.01841570061795783, 4.774631989432649e-9,
          7.63777788259464e-10, 0.18068245984484557]
    @test isapprox(w3.weights, wt, rtol = 1.0e-5)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [1.4475965511299271e-8, 0.15139053006670328, 1.2531364593722484e-8,
          0.02042309439335846, 2.1361563540103295e-8, 4.012521850613263e-8,
          0.01722100416172815, 0.1532392038577736, 1.1517423032461546e-8,
          1.2693072417282946e-8, 0.4464468008721702, 9.157260802561859e-9,
          1.1830960942068228e-8, 2.3761186886959645e-8, 1.4669914215262773e-7,
          6.144682912498466e-7, 0.027626609709476953, 1.631595856184254e-7,
          1.4944878185822058e-8, 0.18365166021287574]
    riskt = 0.024511874696597793
    rett = 0.0004794326659607715
    @test isapprox(w4.weights, wt, rtol = 1.0e-5)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett, rtol = 1.0e-6)

    w5 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.4262370368099214e-9, 0.15109830119103693, 4.5753033355549586e-9,
          0.020595549491597333, 7.964018113907466e-9, 1.5349176697692086e-8,
          0.017131627832874405, 0.15324759220272538, 4.0337811229377734e-9,
          4.598713475553217e-9, 0.44642374758452036, 3.14646860021372e-9,
          4.116666022232492e-9, 8.820226856534203e-9, 5.280033483159844e-8,
          3.006208023615729e-7, 0.02786083166919229, 5.994243143472186e-8,
          5.387502906521676e-9, 0.18364187324639056]
    @test isapprox(w5.weights, wt, rtol = 1e-4)

    w6 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.2974521636472227e-10, 0.1511117107080329, 2.3772089567528915e-10,
          0.02058706960215576, 8.800029560475762e-10, 3.33220242180068e-10,
          0.01713205476125333, 0.15324961307673107, 1.971297743563386e-10,
          1.906147975235925e-10, 0.4464267013369022, 1.1241175348079011e-10,
          1.1533719330242375e-10, 2.646022456979656e-10, 1.0697230346457774e-9,
          5.0201710049166106e-9, 0.027855440141728288, 1.2652809114102488e-9,
          2.5056480589466414e-10, 0.18363740020667155]
    @test isapprox(w6.weights, wt, rtol = 1.0e-5)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.0750748140777434e-8, 3.269490337304986e-8, 1.1161941451849754e-8,
          1.3795466025857643e-8, 0.5351874067614019, 2.6718249477546367e-9,
          0.1390764348877217, 1.41282558079161e-8, 1.0656060597300996e-8,
          7.83717309959956e-9, 1.794801260303159e-8, 2.6229370477942236e-9,
          1.8308405319956406e-9, 6.011246604979923e-9, 1.9381716976717685e-9,
          0.18358697053484188, 0.14214899271554252, 1.7344741890623557e-8,
          3.394097823954422e-8, 9.767190110912097e-9]
    riskt = 0.03754976868195822
    rett = 0.0015728602397846448
    @test isapprox(w7.weights, wt, rtol = 1.0e-5)
    @test isapprox(r3, riskt, rtol = 5.0e-7)
    @test isapprox(ret3, rett, rtol = 5.0e-7)

    w8 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.9979357266227307e-8, 1.3433190080573913e-7, 1.9289466264215865e-8,
          2.6567127395872726e-8, 0.4887145767904928, 4.687357399750469e-9,
          0.11042519617829799, 6.329803422124635e-8, 1.9120266630287126e-8,
          1.626296836809088e-8, 1.1949389923023293e-7, 4.433503242518708e-9,
          2.9962250121467016e-9, 1.218049680765375e-8, 3.3390094757338038e-9,
          0.1801155318875559, 0.22074406904792815, 5.5588214838936996e-8,
          1.0546407189021632e-7, 1.9063826338160836e-8]
    @test isapprox(w8.weights, wt, rtol = 5e-7)

    w9 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [8.22397665559172e-8, 3.826190817815927e-7, 9.2494531582376e-8,
          1.116490369754742e-7, 0.5137842676302625, 1.9493350548570787e-8,
          0.11863275644718071, 1.5404503430726993e-7, 8.610208570285454e-8,
          6.013966797550019e-8, 2.0590880890358276e-7, 1.8543652214974115e-8,
          1.2825779526983523e-8, 4.58672619265406e-8, 1.3527189010888518e-8,
          0.18159860062783653, 0.18598228856404617, 1.809789890792263e-7,
          5.372930554387161e-7, 8.300338256936208e-8]
    @test isapprox(w9.weights, wt, rtol = 0.0005)

    obj = MaxRet()
    w10 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [3.4349721061540534e-9, 3.654412215248722e-9, 4.5795242614345734e-9,
          4.166579635738961e-9, 2.721474418332461e-7, 1.6176234392665889e-9,
          0.9999996718661417, 2.521138576067204e-9, 3.865486493372323e-9,
          2.762737441868102e-9, 2.4450889056720585e-9, 1.7125484610603246e-9,
          1.2415710732838844e-9, 2.0536581804908253e-9, 1.281393702188933e-9,
          5.965051328690624e-9, 4.7776301196822434e-9, 2.6493753543706035e-9,
          4.0069166797873965e-9, 3.2507084446056985e-9]
    riskt = 0.14084653100897324
    rett = 0.0018453755649407056
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt, rtol = 1.0e-7)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.714812761322751e-11, 4.773899536284892e-11, 7.269098817719502e-11,
          6.02689343593572e-11, 0.8503244596483243, 1.3212530602986929e-11,
          0.14967553959927737, 1.4434873773786082e-11, 5.6011672428045124e-11,
          2.0563699040590723e-11, 1.1468895310076392e-11, 1.223933138963094e-11,
          2.5822375180978835e-11, 1.5505601538916262e-12, 2.4392582540848822e-11,
          1.499362309475636e-10, 8.733241804718736e-11, 1.8229290852346547e-11,
          6.3532171924317e-11, 3.5824978876083113e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.630332111490297e-9, 2.8208813441505202e-9, 3.4946060056339277e-9,
          3.1476772348502665e-9, 0.8533950846247867, 1.2168423662193806e-9,
          0.1466048715410871, 1.8953132288109335e-9, 3.003500110840219e-9,
          1.9905919789926412e-9, 1.7863560639864318e-9, 1.1568945873970863e-9,
          8.405498765740556e-10, 1.4870828390487235e-9, 8.482710333148072e-10,
          6.0247478526723576e-9, 3.882589512102091e-9, 2.013333761943543e-9,
          3.2065860439340593e-9, 2.3879703550881473e-9]
    @test isapprox(w12.weights, wt)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-6

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 5e-6

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-8

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) <= 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "RVaR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)
    rm = RVaR()

    obj = MinRisk()
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [5.102262457628692e-9, 0.21104494400160803, 5.341766086210288e-9,
          2.5458382238901392e-8, 1.697696472902229e-8, 7.287515616039478e-9,
          0.03667031714382797, 0.061041476346139684, 4.093926758298615e-9,
          4.303160140655642e-9, 0.49353591974074, 2.1672264824822902e-9,
          3.926886474939328e-9, 4.083625597792755e-9, 1.043237724356759e-8,
          1.1621198331723714e-8, 2.5232405645111758e-8, 1.1835541180026409e-8,
          4.9679012600678e-9, 0.19770719993654404]
    riskt = 0.028298069755304314
    rett = 0.000508233446626652
    @test isapprox(w1.weights, wt, rtol = 1.0e-5)
    @test isapprox(r1, riskt, rtol = 5.0e-7)
    @test isapprox(ret1, rett, rtol = 1.0e-7)

    w2 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.6668841800133678e-9, 0.21104236787122874, 1.6385410773041922e-9,
          2.2326054634855798e-8, 3.0730029625651843e-9, 5.094787825563139e-9,
          0.036670498818371776, 0.06104319657703143, 1.3672919980122889e-9,
          1.647423381582632e-9, 0.49353551089576253, 9.658287207358622e-10,
          2.6914612707146334e-9, 1.5502409858734394e-9, 6.688040133571025e-9,
          5.702498460496523e-9, 1.74920185632266e-8, 2.8842309091216022e-9,
          1.6939432548395604e-9, 0.19770834935535725]
    @test isapprox(w2.weights, wt, rtol = 1e-5)

    w3 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.502698952160907e-10, 0.21104666398587518, 1.523573282203632e-10,
          1.1502118020356136e-9, 5.349304212959444e-10, 3.127959428432429e-10,
          0.03667019153803494, 0.06104496281800859, 1.2258127777323074e-10,
          1.3664808057443733e-10, 0.4935347100598141, 7.369173246087541e-11,
          1.7234948097211898e-10, 1.2626925782413405e-10, 5.002396290612599e-10,
          4.3599827752737464e-10, 1.0080007692050082e-9, 4.2016982389760224e-10,
          1.5724233733436136e-10, 0.19770346614451104]
    @test isapprox(w3.weights, wt, rtol = 5.0e-7)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [8.228671862587609e-9, 0.21291612525433196, 9.212632024402435e-9,
          6.77271672761038e-8, 5.647944186841271e-8, 9.525583076397586e-9,
          0.03942301849563698, 0.05652834541257296, 7.053510326818946e-9,
          6.90167215724061e-9, 0.49014702731909277, 3.2234218542855514e-9,
          5.408738468267058e-9, 5.859453487190136e-9, 1.1154624806837896e-8,
          2.753228972694376e-8, 6.081480482292825e-8, 2.3832551410618976e-8,
          8.947215219262381e-9, 0.20098517161658686]
    riskt = 0.028299654429795203
    rett = 0.0005145957821535152
    @test isapprox(w4.weights, wt, rtol = 5.0e-5)
    @test isapprox(r2, riskt, rtol = 1.0e-7)
    @test isapprox(ret2, rett, rtol = 5.0e-7)

    w5 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.409645126108176e-9, 0.2128167193323565, 1.4049598588911328e-9,
          2.6279990910417456e-8, 2.883095904491254e-9, 3.015914001474748e-9,
          0.039303621052848764, 0.05687420123824051, 1.1031129606041379e-9,
          1.3172803496684389e-9, 0.49017424862907016, 7.234519166886258e-10,
          1.6289081525934723e-9, 1.2832030143670635e-9, 3.453683769844097e-9,
          5.307424148337824e-9, 1.9685243970616102e-8, 3.2980253868381807e-9,
          1.40489155857031e-9, 0.20083113554865312]
    @test isapprox(w5.weights, wt, rtol = 1e-5)

    w6 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [8.332438351175533e-11, 0.2128135847060895, 9.145907630512437e-11,
          1.2184106241802656e-9, 4.493973710574867e-10, 1.5387170447945842e-10,
          0.03930172627407741, 0.05687333392336399, 7.36092924910324e-11,
          7.847268644655777e-11, 0.49017736599631556, 4.0892372405286e-11,
          8.781756269721712e-11, 6.49246211802165e-11, 1.9465341973994675e-10,
          2.772073232494246e-10, 8.093463891866714e-10, 1.6289435907455095e-10,
          9.568356005578712e-11, 0.20083398521818877]
    @test isapprox(w6.weights, wt, rtol = 1.0e-5)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [9.496500669050249e-9, 2.64615310020192e-8, 7.273118042494954e-9,
          1.4049587952157727e-8, 0.5059944415194525, 2.377003832919441e-9,
          0.17234053237874894, 1.8314836691951746e-8, 1.2375544635066102e-8,
          4.317304792347554e-8, 1.9197414728022034e-6, 2.401462046149522e-9,
          1.6115997522673463e-9, 9.360121102571334e-9, 2.354326688306667e-9,
          0.1824768715252159, 0.1391859057572847, 2.2814940892439545e-8,
          1.5125718216815985e-7, 5.757021876600399e-9]
    riskt = 0.04189063415633535
    rett = 0.0015775582433052353
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1238164057580054e-8, 7.502987655712217e-8, 8.562395613769455e-9,
          1.7540789268869764e-8, 0.45025091826713454, 2.7589235102909653e-9,
          0.14690695026744388, 3.007054255105741e-8, 1.2072989519488072e-8,
          2.454812073912125e-8, 0.051268956491519475, 2.776064180360977e-9,
          1.7374807154043764e-9, 1.0436142612069122e-8, 2.4318724561358117e-9,
          0.16956901232837346, 0.182003873492148, 2.480966141363538e-8,
          5.707536077061003e-8, 8.0649967819549e-9]
    @test isapprox(w8.weights, wt, rtol = 5e-7)

    w9 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.3388885824612102e-7, 5.044363200075856e-7, 1.0333859856380586e-7,
          2.0019055857457234e-7, 0.4815462218543765, 3.2400656880734604e-8,
          0.15041378144727896, 3.452256577699966e-7, 1.6487301487356635e-7,
          3.1217276266993253e-7, 0.02247517567944038, 3.1543005887384424e-8,
          2.0866510024902028e-8, 1.1730943478258464e-7, 2.855782321665766e-8,
          0.17739887304211172, 0.16816024420968856, 3.5420797416356593e-7,
          3.2692426375081392e-6, 8.551329085377921e-8]
    @test isapprox(w9.weights, wt, rtol = 0.0005)

    obj = MaxRet()
    w10 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [3.831289441927501e-8, 4.0774049973893135e-8, 5.116921734398324e-8,
          4.653543598014566e-8, 3.0359943766315674e-6, 1.8025259997572137e-8,
          0.9999963392770433, 2.8089814289843758e-8, 4.3130715980449956e-8,
          3.078127147470436e-8, 2.7243225373031e-8, 1.906666528886154e-8,
          1.3841167393289765e-8, 2.2873601944820206e-8, 1.4288780218455534e-8,
          6.670750148901588e-8, 5.3397980334051206e-8, 2.9520022848729233e-8,
          4.472247993754918e-8, 3.624849569013439e-8]
    riskt = 0.1896251879021564
    rett = 0.0018453747093886835
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.8450452798034504e-10, 2.3719138287141804e-10, 3.6153220965024256e-10,
          2.9963202552729554e-10, 0.8503242801043888, 6.5628112212073e-11,
          0.14967571615535655, 7.122636503209286e-11, 2.781951307363639e-10,
          1.0170880062937885e-10, 5.654165034078128e-11, 6.098567027533659e-11,
          1.303890693422527e-10, 8.227265244624121e-12, 1.2077210891122954e-10,
          7.456041785896548e-10, 4.3436034951865134e-10, 9.022345600511273e-11,
          3.1568713917821245e-10, 1.778453418556971e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.7740594068572653e-9, 2.9895068386912447e-9, 3.7070853891264206e-9,
          3.337656158169103e-9, 0.8533950905998182, 1.2851548910520028e-9,
          0.14660486291889813, 2.0131312391534185e-9, 3.18366579397459e-9,
          2.1205442212122336e-9, 1.9033528476129074e-9, 1.2334871766943601e-9,
          8.945098419298842e-10, 1.578723339062802e-9, 9.052768476844037e-10,
          6.355287878151272e-9, 4.119720692605665e-9, 2.134978898877822e-9,
          3.400093873856053e-9, 2.5450484421453455e-9]
    @test isapprox(w12.weights, wt)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-5

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 1e-6

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1 * (1.000001)
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-7

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) <= 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "EVaR < RVaR < WR" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false,
                                                                           "max_step_fraction" => 0.75))))
    asset_statistics!(portfolio)

    obj = MinRisk()
    rm = RVaR(; kappa = 5e-4)
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm = RVaR(; kappa = 1 - 5e-4)
    w2 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm = EVaR()
    w3 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm = WR()
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w3.weights, rtol = 5.0e-5)
    @test isapprox(w2.weights, w4.weights, rtol = 5.0e-8)

    obj = Sharpe(; rf = rf)
    rm = RVaR(; kappa = 7e-4)
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm = RVaR(; kappa = 1 - 7e-4)
    w2 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm = EVaR()
    w3 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm = WR()
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w3.weights, rtol = 0.05)
    @test isapprox(w2.weights, w4.weights, rtol = 1.0e-6)
end

@testset "MDD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = MDD()

    obj = MinRisk()
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.03773810166130732, 3.1042180168905816e-13, 1.0258595030881142e-12,
          1.1524205933253063e-13, 0.06767028645381183, 2.3552606686958265e-12,
          2.7134510397943923e-12, 1.3306981320913463e-10, 5.401808545795961e-13,
          5.8034569913613416e-12, 0.4911343243745374, 2.576243810951587e-12,
          0.02869987018637622, 3.4833677676930996e-13, 4.742648529456258e-13,
          0.14543878669281055, 0.09520044600368058, 0.10299334201244474,
          2.646577244814983e-12, 0.031124842463052256]
    riskt = 0.08375062456707072
    rett = 0.0005626385407841893
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.037738100551314574, 2.467272195159355e-12, 2.2786023120985103e-13,
          4.522978123382619e-12, 0.06767028663615178, 1.2727097474407322e-11,
          3.8079504023059085e-12, 1.1076521474478246e-9, 1.018969701428595e-13,
          2.253328836777569e-11, 0.49113432437528093, 1.1223250184241838e-11,
          0.028699869957065707, 5.400643747485063e-12, 5.469963742518727e-12,
          0.14543878619575004, 0.09520044837465244, 0.10299334224439993,
          1.0306609282250803e-11, 0.031124840478943695]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.0377381068327576, 2.8749774898477704e-11, 6.198607129643284e-11,
          2.1186148905531072e-11, 0.06767028481210714, 7.983360633134082e-11,
          6.760391098466892e-11, 1.726246883450978e-9, 3.729951190006856e-11,
          1.1613149763018386e-10, 0.49113431973422234, 7.071577971843237e-11,
          0.028699868727945436, 1.7881244255489216e-11, 1.6754483491379602e-11,
          0.14543878800824347, 0.09520045310963426, 0.10299333590463638,
          7.980405113251942e-11, 0.03112484054626037]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [0.037738101174196806, 9.355145740311047e-13, 4.1393537116826914e-13,
          1.2878100359065746e-12, 0.06767028675828049, 1.827529427308177e-12,
          2.538883782386595e-13, 1.3319672681881275e-10, 4.843334443164101e-13,
          3.44901413847495e-12, 0.4911343242205381, 1.4796211906493498e-12,
          0.028699870219867155, 1.4769994068408537e-12, 1.5002094881337769e-12,
          0.14543878665436505, 0.09520044641837715, 0.10299334274682366,
          1.4220697575424657e-12, 0.031124841659823883]
    riskt = 0.08375062460033456
    rett = 0.0005626385410281567
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.03773810133213691, 1.2816371062509562e-12, 4.571669224729073e-12,
          7.462279682110099e-13, 0.06767028665376185, 1.6635342170474247e-11,
          8.170442011052557e-12, 7.000105510596881e-10, 3.930907631249683e-12,
          2.9000728199320255e-11, 0.4911343235537764, 1.5297426396365432e-11,
          0.02869986990709296, 1.675234796928077e-12, 1.7690865660707167e-12,
          0.14543878689521397, 0.0952004480021012, 0.10299334233085741,
          1.5042175764246885e-11, 0.031124840526927853]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.03773810383869635, 1.2191686693500414e-11, 5.978683409857994e-11,
          9.346412015324103e-12, 0.06767028652043727, 3.3973255778011735e-11,
          6.041730471703603e-11, 9.416456802597202e-10, 1.5733744301731567e-11,
          5.275419118139716e-11, 0.4911343212091556, 3.114377402027074e-11,
          0.028699869573329487, 7.763767234790757e-12, 7.083135870619938e-12,
          0.14543878829128037, 0.09520044740411525, 0.1029933413186109,
          3.28007206036457e-11, 0.031124840579734234]
    @test isapprox(w6.weights, wt, rtol = 1.0e-6)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [5.9506536559648336e-9, 3.055041148244886e-9, 8.889333776832732e-7,
          1.1881054676532963e-9, 0.28539561079025444, 4.893744003117352e-10,
          6.719698385711209e-10, 0.0796418958781452, 7.605101498417779e-10,
          1.0005252999771136e-9, 0.3337062578482136, 3.1450815418195155e-10,
          4.049178064598528e-11, 1.3972691790544688e-9, 2.857481400527557e-10,
          0.30125531607436506, 4.669472443118481e-9, 2.8912262146723897e-9,
          2.0261421488156098e-9, 5.734606087979582e-9]
    riskt = 0.1011843951714193
    rett = 0.0009728069843048797
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.7009588213654687e-9, 7.113672302556916e-10, 0.003258987441540809,
          3.6306552764102385e-10, 0.2865189236346769, 2.011380307125931e-10,
          3.091251734556721e-10, 0.07218283901086149, 2.736657043479687e-10,
          3.225224715732441e-10, 0.3429341815096216, 1.5983473231255888e-10,
          1.0888154936247301e-10, 3.9908709936599927e-10, 1.4762607390713456e-10,
          0.29510505812626575, 1.6121873618192376e-9, 8.613967608375601e-10,
          5.951968473775891e-10, 1.5109798063947234e-9]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.00027239320023962746, 0.00023444170667372637, 0.25128395957475785,
          0.00012796733647458172, 0.3713766908350648, 5.885703792185687e-5,
          0.014884929191308079, 0.15439741495261963, 0.00010135012716798999,
          9.86213985220039e-5, 0.016266649024130896, 5.384251287815895e-5,
          3.073381739446537e-5, 0.0001247855101243992, 5.129650030402543e-5,
          0.18965255146659266, 0.0003311253037677829, 0.00015152523227820314,
          0.0002312378756478585, 0.0002696273961314085]
    @test isapprox(w9.weights, wt, rtol = 1.0e-4)

    obj = MaxRet()
    w10 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.5745374972510783e-8, 1.6795267715474693e-8, 3.9226126821313444e-8,
          1.9658214817464465e-8, 7.384947979194602e-5, 7.74865444801044e-9,
          0.9999258711672797, 1.1978367888341456e-8, 1.8633310016734365e-8,
          1.3278498963909615e-8, 1.1774562333300158e-8, 8.163069275625413e-9,
          6.015311122527167e-9, 9.84067553475408e-9, 6.192326474712178e-9,
          2.7434536807504806e-8, 2.09245908483398e-8, 1.26149029084296e-8,
          1.8431677001104163e-8, 1.4897460492543179e-8]
    riskt = 0.757130363082932
    rett = 0.001845371625961454
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.478981523193168e-11, 5.99217160194925e-11, 5.3505787664539284e-11,
          5.1607944548610926e-11, 0.8503250560578152, 8.512528923095028e-11,
          0.14967494278715762, 7.399007801762696e-11, 4.986305481544514e-11,
          6.85769271157522e-11, 7.424404477319307e-11, 8.247799584139267e-11,
          8.58455454069926e-11, 7.908060870013363e-11, 8.901090836701929e-11,
          1.2282986037910927e-11, 4.0483614118711085e-11, 6.958416327560956e-11,
          5.1460113334693e-11, 6.317652708838199e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.121074390731353e-9, 1.2596371015130967e-9, 1.5268099590099004e-9,
          1.3980608032636215e-9, 0.853396311923405, 5.602282541070632e-10,
          0.14660366839187777, 8.322176259858601e-10, 1.3260868007911922e-9,
          8.974153028402842e-10, 8.392348949474989e-10, 5.808475039100759e-10,
          5.024783660165443e-10, 6.836446068995123e-10, 6.026001720676835e-10,
          2.4262941933969534e-9, 1.7285830901871349e-9, 8.912872944844735e-10,
          1.415330932151256e-9, 1.092885946803343e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 1e-9

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10
end

@testset "ADD" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = ADD()

    obj = MinRisk()
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [7.448857883265971e-10, 0.04441277893847923, 0.1587788958800634,
          1.8957054590408488e-10, 0.06789474999629158, 4.389817844482217e-10,
          2.1768311051594684e-10, 0.06198838865063388, 2.882880177850258e-10,
          4.115990665909508e-10, 0.2102558445383745, 1.30394717412982e-9,
          0.0012129823674060182, 0.01721631518364624, 9.450387009352195e-11,
          0.06906957552145801, 0.10146311184016397, 0.022480775384071998,
          0.13036008594986354, 0.11486649206008832]
    riskt = 0.014157774437043937
    rett = 0.000760933179396814
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [4.972657750580505e-10, 0.04441324137110288, 0.15877992094694032,
          1.0965898910084084e-10, 0.06789510138336641, 2.8327243164871294e-10,
          1.3161176637402985e-10, 0.06198857718964148, 1.7952156924035293e-10,
          2.6726006697309495e-10, 0.21025531290530317, 9.120265809455868e-10,
          0.0012131315389369718, 0.01721616901465811, 4.3713967990876075e-11,
          0.06906942127899034, 0.10146168413625194, 0.022480106182408397,
          0.13036082308663477, 0.11486650854143404]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [7.999732762485758e-10, 0.044413008610723684, 0.15877939178739742,
          2.7363096967652616e-10, 0.06789491611666268, 5.098594379063606e-10,
          3.060178488097915e-10, 0.06198842131679942, 3.6670951519420777e-10,
          4.843892928228615e-10, 0.21025568511029427, 1.3407455931198829e-9,
          0.0012130366014341035, 0.017216206491550923, 1.828652696798942e-10,
          0.06906950014348101, 0.10146224803552259, 0.02248050429281532,
          0.13036056967161253, 0.1148665075575149]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [5.798366327372061e-10, 0.042412234114840804, 0.15457815348085807,
          1.2703441532476687e-10, 0.06853004949414504, 2.734517473068476e-10,
          1.4089105561436528e-10, 0.06477888113569304, 2.0457866358009126e-10,
          3.4447757705514945e-10, 0.20764909776289522, 6.440474691240202e-10,
          0.0007314274843326087, 0.018524387017033964, 2.69090830675864e-11,
          0.07144511559739707, 0.10791260940930285, 0.016305065478820022,
          0.1305913431745452, 0.11654163350890942]
    riskt = 0.014159337775655985
    rett = 0.0007653523686111961
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.0142707062878218e-11, 0.0424121770836295, 0.15457804443757112,
          8.847090098056191e-12, 0.06853006339605403, 5.866935537385589e-13,
          9.750293399119566e-12, 0.06477894353556392, 2.9432366307894726e-12,
          5.368143564920223e-12, 0.2076490424529727, 2.8689007794445397e-11,
          0.0007314290831770678, 0.018524413067004992, 1.3300832976714169e-11,
          0.07144518585673806, 0.10791277202762489, 0.01630490922810095,
          0.13059135481169706, 0.1165416649302377]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.6814487491258234e-10, 0.042412180936881275, 0.15457804711442463,
          6.189037399613695e-11, 0.06853005906181425, 9.815515077593236e-11,
          7.115488721878398e-11, 0.06477893148890453, 7.850778039795153e-11,
          1.0789784605489587e-10, 0.20764906279954667, 1.7223292106660123e-10,
          0.0007314192438345997, 0.018524405084710534, 3.728459379336366e-11,
          0.07144518020236533, 0.10791275425828677, 0.016304930521483248,
          0.13059136057149756, 0.11654166792098217]
    @test isapprox(w6.weights, wt)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.691669135998654e-10, 1.1631413826126582e-9, 0.22302886349067164,
          2.711714483929198e-10, 0.23131321830818175, 6.588409097827075e-11,
          0.0024130708498912596, 0.0693827788407185, 2.059607953371255e-10,
          1.8706873275340954e-10, 0.09196996390552813, 6.137855248572752e-11,
          3.205859049861493e-11, 2.4263452784032555e-10, 4.7562460670403145e-11,
          0.14134965001846883, 0.14559337938522812, 3.8520832394632936e-10,
          0.09494907067142468, 1.6986512754618543e-9]
    riskt = 0.017419673036188653
    rett = 0.0010748787661322586
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.732786951246471e-11, 4.1411290284716676e-10, 0.23659139195646137,
          4.4069736825891224e-11, 0.20617016581466932, 1.5701212099866242e-11,
          0.0015585663302501937, 0.0737221210055692, 4.028191562347591e-11,
          4.273466229817362e-11, 0.1334404823098268, 1.4183237943257261e-11,
          8.73701262926479e-12, 6.052527720960589e-11, 1.1350819173480697e-11,
          0.13268931834298567, 0.14391006397386097, 9.004188734952489e-11,
          0.07191788907020406, 4.171059686466568e-10]
    @test isapprox(w8.weights, wt)

    w9 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [4.167553953047548e-9, 2.5755542087386356e-8, 0.2169553308852588,
          6.3664322411553045e-9, 0.23650465197481196, 1.7649093428417507e-9,
          0.002586135055951778, 0.06889920488574157, 5.01145499843851e-9,
          5.017038145968494e-9, 0.09238650461530447, 1.6720016814640506e-9,
          1.0022011009668063e-9, 5.610891628711742e-9, 1.3382287422343606e-9,
          0.14089647227409152, 0.14604606225385514, 1.02720302420471e-8,
          0.09572550710764308, 6.296905747599441e-8]
    @test isapprox(w9.weights, wt, rtol = 1.0e-5)

    obj = MaxRet()
    w10 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.5476499595124018e-8, 1.6411063538766305e-8, 1.9985512029725137e-8,
          1.8783105810200402e-8, 5.723523093249718e-5, 7.295678963955699e-9,
          0.9999425164590651, 1.1600868184476528e-8, 1.766048206305046e-8,
          1.2667003954387e-8, 1.1218702023911828e-8, 7.72770688897506e-9,
          5.534795162729263e-9, 9.439475106723637e-9, 5.825474042163815e-9,
          2.33247986974336e-8, 2.0744413898136906e-8, 1.2108111903428947e-8,
          1.8035641556123702e-8, 1.4470669126128638e-8]
    riskt = 0.226009263342417
    rett = 0.0018453724844686955
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.493182920360213e-10, 1.3636781645574893e-10, 1.1570567153847813e-10,
          1.171102183969936e-10, 0.8503180435488843, 1.9404909343556765e-10,
          0.14968195383478536, 1.5836219165761555e-10, 1.2996877714955953e-10,
          1.6075200682227455e-10, 1.6134023881961763e-10, 1.8951555485459176e-10,
          1.953087214402432e-10, 1.7614892077835563e-10, 2.0133991994575575e-10,
          2.6598451074410513e-11, 8.403665830424447e-11, 1.583335907300543e-10,
          1.1634384954475185e-10, 1.457302861006041e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.578396576423994e-9, 1.7668913536875456e-9, 2.195938348136312e-9,
          1.979973684702315e-9, 0.8533949984345872, 6.930810292518326e-10,
          0.1466049741867149, 1.2033685035042625e-9, 1.9132151137429597e-9,
          1.3019765686121734e-9, 1.156443581096697e-9, 7.199018490635218e-10,
          5.114053553430424e-10, 9.231823222747696e-10, 5.195704450503038e-10,
          3.576004887560627e-9, 2.4590196894435053e-9, 1.2670150699623182e-9,
          2.048214628941069e-9, 1.5650989103260291e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-9
end

@testset "UCI" begin
    portfolio = Portfolio(; prices = prices,
                          solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                           :check_sol => (allow_local = true,
                                                                          allow_almost = true),
                                                           :params => Dict("verbose" => false))))
    asset_statistics!(portfolio)
    rm = UCI()

    obj = MinRisk()
    w1 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.222285452584206e-10, 0.02501218530242097, 0.09201946209452846,
          1.3682085300046153e-10, 0.01940475597034475, 2.725276389927591e-10,
          5.2162763718278294e-11, 0.06508427427484158, 5.393438835014886e-10,
          3.520844112986984e-10, 0.28536549907916714, 1.2394066227980813e-9,
          9.195480324724614e-10, 0.008371046777200061, 2.10381971522532e-10,
          0.0971653547968288, 0.1751285220443603, 0.02458980120560063, 0.05728457556618117,
          0.1505745182440216]
    riskt = 0.021910190451570975
    rett = 0.0006841173261863873
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [4.925157631881066e-10, 0.025011784565219283, 0.09201903919812783,
          3.759623681562836e-11, 0.019404437138143257, 1.1608305849673475e-10,
          7.189470831446275e-11, 0.06508389984186924, 2.707756606699596e-10,
          1.623209697630627e-10, 0.28536563958992195, 6.755938639252753e-10,
          4.904408473903994e-10, 0.008371445218253549, 8.013472130623573e-11,
          0.09716505026206242, 0.17512903508834968, 0.024590500360709738,
          0.05728431099221822, 0.1505748553477691]
    @test isapprox(w2.weights, wt)

    w3 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [5.194921852041518e-10, 0.025011861706312383, 0.09201887315053964,
          1.802202932981046e-10, 0.019404498174306953, 2.3892072426084615e-10,
          9.85818953549748e-11, 0.06508394126973903, 3.5420283475586296e-10,
          2.731103490061945e-10, 0.2853658078499243, 6.568538290501507e-10,
          5.190279363535191e-10, 0.008371455634173904, 2.1205232048662294e-10,
          0.09716506715750306, 0.17512922169021805, 0.024590543675918626,
          0.05728394902254039, 0.15057477761636134]
    @test isapprox(w3.weights, wt)

    obj = Utility(; l = l)
    w4 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [6.103005918866347e-10, 0.02399478084068091, 0.09614466390677545,
          1.3554508229876816e-11, 0.019578507639319558, 7.842894822380073e-11,
          1.6816731087609036e-10, 0.06625943266894467, 3.008423757485725e-10,
          1.5498777345145633e-10, 0.2838092045866127, 8.096786196491255e-10,
          4.5654529796803774e-10, 0.001542296348720114, 3.170436144910474e-11,
          0.10123984359228022, 0.17626210900329523, 0.020477832035821767,
          0.06087211295464912, 0.1498192137986906]
    riskt = 0.02191286806444714
    rett = 0.0006947880506399113
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.0787572534525563e-10, 0.024112516456871617, 0.09588426169938007,
          5.101205324051336e-10, 0.019666368600616303, 3.8902706464539093e-10,
          7.126792536578957e-10, 0.06619031054408571, 9.744036286872689e-11,
          2.8866867429482205e-10, 0.28389061370830704, 5.660231483311974e-10,
          1.1088893706306199e-10, 0.0019203170265927867, 4.5054094649126116e-10,
          0.10102258859763576, 0.1761368281523827, 0.020639441908370616,
          0.060701128872756985, 0.14983562099973577]
    @test isapprox(w5.weights, wt)

    w6 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.781319025391964e-10, 0.024111133330980725, 0.09587762344981725,
          6.25191922669312e-11, 0.019666353567101703, 7.964277120709312e-11,
          3.39234415421405e-11, 0.06618859529821851, 1.2086406357928116e-10,
          9.381964107077793e-11, 0.2838955384695536, 2.1453501235461293e-10,
          1.5038569080548262e-10, 0.0019253019310517393, 7.094617330925158e-11,
          0.10102214158342449, 0.1761340758261751, 0.020642252907819545,
          0.06070147985648409, 0.14983550277460528]
    @test isapprox(w6.weights, wt, rtol = 1.0e-5)

    obj = Sharpe(; rf = rf)
    w7 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [9.846697842152917e-9, 2.43304437706785e-8, 0.20539739244782326,
          1.1079920282116377e-8, 0.2857551395628918, 2.872382424528247e-9,
          1.221245322708047e-8, 0.1435288044479327, 9.552129263357414e-9,
          8.178392689321128e-9, 0.07582709025042092, 2.9840435618712605e-9,
          1.5141281246528472e-9, 1.2869750440851804e-8, 2.124147094515537e-9,
          0.18476084623054717, 0.1047301703559068, 1.93663720826598e-8,
          1.8714170675082394e-7, 2.5263190989624937e-7]
    riskt = 0.02911587328890136
    rett = 0.0010968342650731456
    @test isapprox(w7.weights, wt, rtol = 1.0e-7)
    @test isapprox(r3, riskt, rtol = 1.0e-7)
    @test isapprox(ret3, rett)

    w8 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.24202248082058e-9, 1.7817399194318174e-8, 0.1929184450504878,
          6.452875009645281e-9, 0.23084520906493095, 2.507336210164012e-9,
          6.732504814516605e-9, 0.1372749711670064, 6.986438943654997e-9,
          6.066029007821234e-9, 0.11254319576114159, 2.5110011915327978e-9,
          1.7128878795388062e-9, 9.672713729659068e-9, 1.8566887674821828e-9,
          0.15375703739424093, 0.12948270229005487, 1.4679552779213837e-8,
          0.013830056152094834, 0.02934829888259256]
    @test isapprox(w8.weights, wt, rtol = 0.0005)

    w9 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.0002472738159425723, 0.00040599319362115286, 0.2059938439469736,
          0.00035535410259227934, 0.4877055130704978, 7.971668223739648e-5,
          0.049893582464332446, 0.008333996152939054, 0.00021117699584221857,
          0.00017986051296359605, 0.0007949025897178295, 7.468177699278661e-5,
          4.722313050472375e-5, 0.00017633046549148763, 5.750514887369428e-5,
          0.23889687043703897, 0.004982509118259884, 0.00026086127256122894,
          0.0007429950398760139, 0.0005598100827413511]
    @test isapprox(w9.weights, wt, rtol = 5.0e-7)

    obj = MaxRet()
    w10 = optimise!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [8.66299431970518e-10, 9.208815443410697e-10, 1.085583564551527e-9,
          1.0469228572790014e-9, 3.030697646530977e-6, 4.1236774249357805e-10,
          0.9999969562915909, 6.499218355634898e-10, 9.87329384691622e-10,
          7.121588886859378e-10, 6.296287328523021e-10, 4.362374055447341e-10,
          3.1335353952467163e-10, 5.32216470301061e-10, 3.29033896532438e-10,
          4.4131943522029303e-10, 1.153440647899276e-9, 6.78694656469276e-10,
          1.0048755804609016e-9, 8.104968341276354e-10]
    riskt = 0.2862458538646685
    rett = 0.0018453754812476895
    @test isapprox(w10.weights, wt, rtol = 5.0e-7)
    @test isapprox(r4, riskt, rtol = 5.0e-7)
    @test isapprox(ret4, rett)

    w11 = optimise!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.944446452914842e-11, 5.7960863478697175e-11, 5.468497251059545e-11,
          5.628215003211347e-11, 0.8503230834197394, 6.607899384201028e-11,
          0.14967691550482193, 6.219269131935343e-11, 5.6720331177869464e-11,
          6.144817451145102e-11, 6.257863597328751e-11, 6.587626965359562e-11,
          6.741125415659148e-11, 6.43570958364287e-11, 6.746197523182665e-11,
          4.3506388341456754e-11, 5.25824585761784e-11, 6.175192399338009e-11,
          5.5632113506790224e-11, 5.946769597040695e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.8525083377008227e-9, 3.1939780222303176e-9, 3.968984428403173e-9,
          3.5787126685669952e-9, 0.8533912913581613, 1.2541645068339838e-9,
          0.1466086591500573, 2.1743933606711545e-9, 3.4588275668077664e-9,
          2.352291982941941e-9, 2.089431604814864e-9, 1.3010557890142628e-9,
          9.265397588811109e-10, 1.6684231493607948e-9, 9.416061925035874e-10,
          6.467073998546227e-9, 4.444561947281658e-9, 2.289311101669237e-9,
          3.7023949823765738e-9, 2.8275221832612232e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-5)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 5e-9

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 1e-9

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    obj = Sharpe(; rf = rf)
    rm.settings.ub = r1 * 1.0000001
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r1) < 1e-8

    rm.settings.ub = r2
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad, rm = rm) - r2) < 5e-9

    rm.settings.ub = r3
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r3

    rm.settings.ub = r4
    optimise!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

    portfolio.mu_l = ret4
    w16 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = Sharpe(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10
end
