@testset "CDaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false))))
    asset_statistics2!(portfolio)
    rm = CDaR2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [3.0804314757988927e-13, 1.657879021331884e-11, 6.262556421442959e-13,
          2.89383317730532e-14, 0.0034099011188261498, 1.5161483827226197e-13,
          5.196600704051868e-13, 0.07904282393551665, 6.175990667294095e-13,
          1.251480348107779e-13, 0.387593170263866, 2.0987865221203183e-12,
          3.138562614833384e-12, 5.06874926896603e-12, 0.0005545151720077135,
          0.09598828927781314, 0.2679088823723124, 1.2303141006668684e-11,
          0.0006560172383583704, 0.16484640057973415]
    riskt = 0.056433122271589295
    rett = 0.0006203230359545646
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1191002119584093e-13, 2.003527800798736e-11, 3.3982948124862123e-13,
          1.3953272189740586e-13, 0.0034099016643585078, 3.243952425105705e-13,
          6.880972687043907e-13, 0.07904282463773535, 4.48866803256831e-13,
          3.0487026196625766e-13, 0.3875931698607013, 2.1136717021819126e-12,
          2.98092441555309e-12, 5.0611076667973155e-12, 0.0005545150891028837,
          0.09598828925066373, 0.26790888138964075, 9.51495342083859e-12,
          0.0006560174191674542, 0.16484640064656664]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.9355981055721525e-11, 5.475696423931913e-10, 3.5866580253301274e-11,
          2.123076239895522e-11, 0.003409901314074457, 1.579965981957622e-11,
          4.54217613467618e-12, 0.07904283113528301, 3.9576286459645786e-11,
          1.6366084837360542e-11, 0.3875931684815596, 8.984745131078415e-11,
          1.0635693908106654e-10, 1.7812934931562238e-10, 0.0005545105505617421,
          0.09598829243516636, 0.2679088658301147, 3.206492164986687e-10,
          0.0006560264361414131, 0.16484640241180865]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [6.947912999651484e-14, 4.277090352016668e-11, 4.528319333601928e-13,
          6.377277979148381e-13, 0.0035594794639014563, 1.0840588017141541e-12,
          1.925986011939725e-12, 0.07997306603154038, 7.196880962156291e-13,
          1.0494792003467066e-12, 0.38715988979257093, 4.26390298658612e-12,
          4.822318130077631e-12, 1.065050908006143e-11, 0.00022182978730093733,
          0.0965257668349744, 0.265939752981685, 1.535632212280552e-11,
          0.0019205820717067992, 0.16469963295251694]
    riskt = 0.056433256072537304
    rett = 0.0006208085760691457
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [8.547206228209242e-13, 5.612511630069217e-11, 1.4707275923352947e-12,
          1.8664758997640343e-13, 0.003559479270748964, 3.536102274931265e-13,
          1.3580723779183722e-12, 0.07997306744622036, 1.7959319382333299e-12,
          3.121823233526299e-13, 0.38715988959811865, 6.237102466648308e-12,
          6.736309588854579e-12, 1.3795415148975926e-11, 0.00022182853299489516,
          0.09652576812266041, 0.26593974806102016, 2.0018419171573694e-11,
          0.0019205854469041023, 0.16469963341208804]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [9.408733235883147e-12, 2.0147441516302233e-10, 1.166880759514716e-11,
          6.910537326560142e-12, 0.0035594785539444875, 4.936341398974611e-12,
          1.0819070878144233e-12, 0.07997306692239549, 1.273789796144024e-11,
          5.124855703752699e-12, 0.38715989060100847, 2.8946181733340445e-11,
          3.038598729409552e-11, 5.571060493427888e-11, 0.00022182746083898803,
          0.09652576801265393, 0.2659397471540211, 9.107594400509428e-11,
          0.001920585919257568, 0.16469963491641776]
    @test isapprox(w6.weights, wt, rtol = 1.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [8.448683544388307e-11, 1.280859126838009e-10, 0.07233499818343535,
          3.1235110568082876e-11, 0.3107255489676791, 6.434229531025652e-12,
          1.7775895803754946e-11, 0.12861324978104444, 2.4818514207226036e-11,
          2.7260857009909587e-11, 0.16438307033445054, 1.6304823775259387e-11,
          3.3660897712740955e-12, 6.255648794327428e-11, 6.889289769976776e-12,
          0.262882733612263, 3.421073479909081e-10, 9.331034331033022e-11,
          1.022392877751668e-10, 0.06106039817425651]
    riskt = 0.082645645820059
    rett = 0.0010548018078272983
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1917953090878462e-9, 2.468191415243125e-9, 0.06188514653019595,
          6.806598635559231e-10, 0.2561783027908721, 3.3840953388531945e-10,
          4.402357109395084e-10, 0.13589729781244972, 6.52268526184591e-10,
          6.172232050103465e-10, 0.18251180950084958, 4.3272403651565954e-10,
          2.3331958795149456e-10, 1.0252502677277383e-9, 3.4626018390081986e-10,
          0.17526252977389256, 2.9545086931034077e-8, 1.7531042789056333e-9,
          2.2441116314699265e-9, 0.18826487162309952]
    @test isapprox(w8.weights, wt, rtol = 5e-5)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.00016240045566614326, 0.00016908214427872127, 0.12664634769824457,
          0.00010910912807997341, 0.4732545303163861, 4.3770595028945213e-5,
          0.0075270442888367185, 0.15692717297206957, 8.178613569555982e-5,
          8.35114431793415e-5, 0.0017318550860133037, 4.525019204311569e-5,
          2.679129520777131e-5, 0.00010299020939058965, 3.78749877643406e-5,
          0.23196483392240672, 0.00034832053680379553, 0.00013211032790113727,
          0.0001872309586056606, 0.00041798730639806884]
    @test isapprox(w9.weights, wt, rtol = 0.0001)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [9.987463130576163e-10, 1.056772644453002e-9, 1.2388504796157802e-9,
          1.2008933981567538e-9, 7.101548970099872e-7, 4.6551092161484706e-10,
          0.9999992748426728, 7.340800975839677e-10, 1.1269034464182206e-9,
          8.091570089471268e-10, 7.151968292259966e-10, 4.91904881032292e-10,
          3.515593821056337e-10, 5.994603655123806e-10, 3.6678454824342996e-10,
          6.356719280937174e-10, 1.3410740971729668e-9, 7.744808193554666e-10,
          1.1566665482207312e-9, 9.387165286514347e-10]
    riskt = 0.636939065675819
    rett = 0.0018453755942528635
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.834537169478786e-12, 5.668884311936744e-12, 4.998965337521077e-12,
          5.333383127537809e-12, 0.8503241084288734, 7.224252515228562e-12,
          0.14967589146682278, 6.3384010795356974e-12, 5.362061372879253e-12,
          6.2630480240251795e-12, 6.353860791883864e-12, 7.065797131767951e-12,
          7.296882305216338e-12, 6.745370726082442e-12, 7.438837439586943e-12,
          1.0411310434870616e-12, 4.296491994171869e-12, 6.21448025575577e-12,
          5.014575868409698e-12, 5.812943509123977e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.0957418977291944e-9, 1.2267710171396468e-9, 1.5232111122482857e-9,
          1.3741851148813044e-9, 0.8533950953646583, 4.843323361944242e-10,
          0.1466048856304184, 8.34629164285368e-10, 1.3279313685992311e-9,
          9.028283191694305e-10, 8.028605805671087e-10, 5.011759106747885e-10,
          3.596194817717271e-10, 6.416776666268118e-10, 3.6693412311538686e-10,
          2.473208420622981e-9, 1.7048588995779374e-9, 8.790581843625214e-10,
          1.4208190333740555e-9, 1.0850806522945768e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 1e-10

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 5e-8

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3 ||
          abs(dot(portfolio.mu, w15.weights) - ret3) < 1e-10

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10
end

@testset "EDaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = EDaR2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.037682575145149035, 9.56278555310282e-9, 2.903645459870996e-9,
          1.1985145087686667e-9, 0.015069295206235518, 1.6728544107938776e-9,
          3.388725311447959e-10, 0.03749724397027287, 2.0700836960731254e-9,
          2.2024819594850123e-9, 0.4198578113688092, 2.938587276054813e-9,
          0.013791892244920912, 3.6397355643491843e-9, 1.9217193756673735e-9,
          0.13381043923267996, 0.20615475605389819, 0.037331096485967066,
          4.8493986688419375e-9, 0.09880485699338833]
    riskt = 0.06867235340781119
    rett = 0.0005981290826955536
    @test isapprox(w1.weights, wt, rtol = 5.0e-5)
    @test isapprox(r1, riskt, rtol = 1.0e-5)
    @test isapprox(ret1, rett, rtol = 1.0e-5)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.03767619706443821, 6.040874388145646e-9, 1.8513086072834443e-9,
          7.472480500864453e-10, 0.015072439269835518, 1.0446644246836178e-9,
          2.0029751881448367e-10, 0.037499075344072465, 1.2971679898632177e-9,
          1.374837670144913e-9, 0.41985700397706, 1.8112006815802752e-9,
          0.013791500822871214, 2.277303301348715e-9, 1.1908575009958289e-9,
          0.13380909413951764, 0.20615223298039678, 0.037335017479152396,
          3.0963938467271572e-9, 0.09880741799050188]
    @test isapprox(w2.weights, wt, rtol = 5.0e-5)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.03767851887227874, 1.4329436113772992e-10, 4.05649576591539e-10,
          2.4972919851808875e-11, 0.01507103782966904, 3.0070283831591244e-11,
          8.063108624511423e-11, 0.03749878898703567, 3.680457025440033e-11,
          3.933577893156735e-11, 0.4198566634663093, 4.7621024416296094e-11,
          0.013791834711444932, 6.105684315724079e-11, 3.206113194015319e-11,
          0.13380928052642502, 0.2061518451407196, 0.037336396558164835,
          8.414716283618674e-11, 0.09880563292230825]
    @test isapprox(w3.weights, wt, rtol = 1.0e-5)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [0.03711752158155375, 9.0393326861564e-9, 2.594461917446991e-9,
          1.0716020851055362e-9, 0.01605305634234447, 1.4991267672133371e-9,
          2.6533968110401274e-10, 0.03761046728458585, 1.892508823748839e-9,
          1.999220600036446e-9, 0.4197926872413624, 2.5684869050894756e-9,
          0.013864230853571753, 3.3691925111834236e-9, 1.703485067349179e-9,
          0.13411510509850116, 0.20508315175097908, 0.03786925224111122,
          4.580320889177859e-9, 0.09849449702291231]
    riskt = 0.06867249456166294
    rett = 0.0005986924375524537
    @test isapprox(w4.weights, wt, rtol = 5.0e-5)
    @test isapprox(r2, riskt, rtol = 5.0e-7)
    @test isapprox(ret2, rett, rtol = 1.0e-6)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.03709773955151592, 2.6802797559890582e-9, 9.678553908839886e-10,
          3.661510676423638e-10, 0.016073974811850546, 5.053614225653502e-10,
          1.2409975221077092e-10, 0.03763771406754832, 6.20378251274211e-10,
          6.611866675389161e-10, 0.41977976347499857, 8.358862174122872e-10,
          0.013859979872712072, 1.0495909493236409e-9, 5.549191627860481e-10,
          0.13412079617477027, 0.2050268226714334, 0.03790770325841322,
          1.4199741910421695e-9, 0.09849549633107485]
    @test isapprox(w5.weights, wt, rtol = 5e-5)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.03710004844591457, 7.668775000534068e-11, 1.5454897939430998e-10,
          1.2929707701830407e-11, 0.01607293826897331, 1.5826864902290516e-11,
          5.224719378370586e-11, 0.0376342410543768, 1.9466611840872193e-11,
          2.0786356940269408e-11, 0.4197807614319933, 2.4613107952468037e-11,
          0.013861415021216943, 3.2702395091715085e-11, 1.666964933633454e-11,
          0.1341203371377317, 0.20503241329417693, 0.03790532548036575,
          4.479632306575932e-11, 0.09849251939397578]
    @test isapprox(w6.weights, wt, rtol = 1.0e-6)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.3046367229418534e-8, 1.427571750756657e-8, 0.10584962927382287,
          4.777864999156633e-9, 0.27127235412812056, 2.636006148622928e-9,
          6.970386746306553e-9, 0.08852131102747707, 4.063322726496132e-9,
          4.421317047959873e-9, 0.2612420770853092, 2.5288753864845775e-9,
          1.497010215619722e-9, 7.163684671131553e-9, 2.295725810569409e-9,
          0.27311446970285214, 2.8481984344497896e-8, 1.0591485932967609e-8,
          1.0029075633034853e-8, 4.600359379735791e-8]
    riskt = 0.0889150147980379
    rett = 0.0010018873392648234
    @test isapprox(w7.weights, wt, rtol = 1.0e-6)
    @test isapprox(r3, riskt, rtol = 5.0e-7)
    @test isapprox(ret3, rett, rtol = 5.0e-7)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.3941605837370376e-8, 1.7379340419048453e-8, 0.09397022998439407,
          4.750316566009424e-9, 0.26905912766854545, 2.542244985356096e-9,
          4.91745934990303e-9, 0.08496764823992713, 4.463542154906836e-9,
          4.642417547118547e-9, 0.3074751732915163, 2.3694756580977916e-9,
          1.4891401987349052e-9, 7.457741148473212e-9, 2.075810974315586e-9,
          0.24452732721152728, 9.005394657577373e-8, 1.4263955038493934e-8,
          1.2495404239355042e-8, 3.107616890388338e-7]
    @test isapprox(w8.weights, wt, rtol = 5e-6)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.430165873074019e-8, 4.4740342832037584e-8, 0.11563921171228772,
          1.3677694099604651e-8, 0.2785792911618146, 7.144205446371835e-9,
          1.7587019090571957e-7, 0.09131072643311786, 1.1575937313951374e-8,
          1.2388850665427953e-8, 0.26202597786762966, 6.90617932726365e-9,
          4.046937461030524e-9, 2.0238788391316287e-8, 6.463579028855831e-9,
          0.25244416958393395, 9.468820075039234e-8, 3.020300472207688e-8,
          2.9735352802248358e-8, 1.312602937255446e-7]
    @test isapprox(w9.weights, wt, rtol = 0.0001)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [3.6548032535862895e-8, 3.861347019610869e-8, 1.0055188636815133e-7,
          4.360588685045703e-8, 0.00013523444753944997, 1.6797112960587062e-8,
          0.9998640251930764, 2.862909912738744e-8, 4.0381167406841796e-8,
          2.915465095874921e-8, 2.669099068489479e-8, 1.7760324653760426e-8,
          1.2662567261861251e-8, 2.1702735893369975e-8, 1.3148979869632596e-8,
          1.581359605252072e-7, 5.07922927421966e-8, 2.808939475892958e-8,
          4.242077057250865e-8, 3.4674060861229493e-8]
    riskt = 0.6612122586242406
    rett = 0.0018453680607163961
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.068095174605424e-11, 5.7846613951361576e-11, 5.0197530596015976e-11,
          5.3741710729620396e-11, 0.850324458152055, 7.296755090668262e-11,
          0.14967554075126555, 6.721896865257403e-11, 5.559668815809807e-11,
          6.580954912658916e-11, 6.808602673456642e-11, 7.36574868892983e-11,
          7.687598262811075e-11, 7.133408967580721e-11, 7.657784586791197e-11,
          2.174156234690718e-11, 4.434054545875643e-11, 6.617876171264754e-11,
          5.2687777046890895e-11, 6.113986905642583e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.092075874030718e-9, 3.3888401472290257e-9, 4.114038166156284e-9,
          3.718861894756244e-9, 0.8533948560167894, 1.4897901950018111e-9,
          0.14660509150098555, 2.3123098668258643e-9, 3.589968894902347e-9,
          2.43011712858619e-9, 2.183443049018459e-9, 1.4358778784726855e-9,
          1.0677252276399053e-9, 1.8145453859060888e-9, 1.0676007002160791e-9,
          6.947245882498193e-9, 4.626650589448291e-9, 2.4467659713923036e-9,
          3.834462937666969e-9, 2.9219050323912107e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1 * 1.0000001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-7

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 5e-7

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1 * 1.000001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-8

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) <= 1e-10

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "RDaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = RDaR2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [0.05620454939911102, 3.025963157432751e-10, 1.55525199050249e-9,
          1.8034522386040858e-10, 0.0299591051407541, 4.1576811387096e-10,
          6.991555930029523e-10, 0.02066796438128093, 2.7039382408143445e-10,
          5.654715437689442e-10, 0.44148521326051954, 4.732606403168718e-10,
          0.021758018837756656, 1.393894367130224e-10, 1.2836530696174218e-10,
          0.14345074805893418, 0.15965735285676014, 0.06193188215818479,
          7.588611460321727e-10, 0.0648851604178394]
    riskt = 0.07576350913162658
    rett = 0.0005794990185578756
    @test isapprox(w1.weights, wt, rtol = 5.0e-7)
    @test isapprox(r1, riskt, rtol = 1.0e-7)
    @test isapprox(ret1, rett, rtol = 1.0e-7)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.056204567156272735, 2.251803354918567e-10, 1.349439978419671e-9,
          1.4142734160611626e-10, 0.029959105214722212, 3.0055053766966263e-10,
          6.785570517824773e-10, 0.020667952362991862, 2.0093566644087126e-10,
          4.049322309533552e-10, 0.4414851698867454, 3.3517191145159927e-10,
          0.021758011042675593, 1.0631991750867355e-10, 9.399616812372313e-11,
          0.14345080105718516, 0.15965734531492787, 0.06193181327504717,
          5.7493911410462e-10, 0.06488523027798172]
    @test isapprox(w2.weights, wt, rtol = 1e-5)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.05620462315809207, 2.6759582643540718e-11, 2.0900233669891437e-10,
          1.7865991735188475e-11, 0.02995917022578265, 3.230037016290626e-11,
          7.788750758486943e-11, 0.020667989723228544, 2.364384785461514e-11,
          4.138502393458586e-11, 0.44148504562796603, 3.5329716380697954e-11,
          0.02175797679639108, 1.5641646506056175e-11, 1.416930296693424e-11,
          0.14345092950544364, 0.1596571436130229, 0.06193197116957979,
          6.508990613526635e-11, 0.06488514962141817]
    @test isapprox(w3.weights, wt, rtol = 1.0e-7)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [0.056636694476851115, 4.49525263570157e-10, 4.045749363336131e-9,
          3.0693192944627693e-10, 0.030740765437852733, 5.423124443348359e-10,
          2.3938922709883823e-9, 0.020936547388877912, 3.938836479209595e-10,
          7.560077918078683e-10, 0.44096731112864757, 5.946878227680669e-10,
          0.02151854391602207, 2.0740284700137118e-10, 1.6147957157877772e-10,
          0.1445623206296534, 0.15845869010706423, 0.06207448298081452,
          1.2756701684425392e-9, 0.06410463280667336]
    riskt = 0.07576390921795187
    rett = 0.0005811018328655167
    @test isapprox(w4.weights, wt, rtol = 1.0e-5)
    @test isapprox(r2, riskt, rtol = 1.0e-7)
    @test isapprox(ret2, rett, rtol = 1.0e-6)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [0.056623385262849484, 1.5419403798990755e-10, 4.479486308560148e-10,
          9.224129291821656e-11, 0.030730621305847673, 2.1383025452685987e-10,
          1.9444808919781441e-10, 0.02094814552253347, 1.3920947418383926e-10,
          2.8368970292298067e-10, 0.4409549799606425, 2.419303961475677e-10,
          0.021521962425106483, 7.438084355416445e-11, 7.235898995328522e-11,
          0.14453046623949664, 0.1584966760550006, 0.06213264412541006,
          3.587506291179255e-10, 0.06406111683013088]
    @test isapprox(w5.weights, wt, rtol = 1e-7)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [0.056623309300318024, 1.3223171280689339e-11, 8.955868396186928e-11,
          8.826463223153956e-12, 0.030730353599472075, 1.6422802037972465e-11,
          3.8184232736014844e-11, 0.020947946880180424, 1.1861017238729038e-11,
          2.069831453013942e-11, 0.44095523371158074, 1.7665679336537252e-11,
          0.021522030578979116, 7.585883445289465e-12, 6.964921324028986e-12,
          0.14453025178982348, 0.15849686136564278, 0.062132221831402344,
          3.135584944354739e-11, 0.06406179068025412]
    @test isapprox(w6.weights, wt, rtol = 1.0e-6)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.620183105236905e-8, 1.1140035827148705e-8, 0.04269821778198461,
          4.539811056982736e-9, 0.27528471583276515, 2.5884954318029336e-9,
          4.359695142864356e-9, 0.09149825388403467, 3.6147826445679577e-9,
          4.148787837320512e-9, 0.3004550564031882, 2.15879252345473e-9,
          1.3866951024663507e-9, 6.227749990512301e-9, 2.022396335125738e-9,
          0.2900636413773165, 1.8039723473841205e-8, 9.883423967231107e-9,
          8.029172726894051e-9, 2.0379317768180928e-8]
    riskt = 0.09342425156101017
    rett = 0.0009783083257672756
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt, rtol = 1.0e-7)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.3990895127156702e-8, 6.304253923649855e-9, 0.047954323907981745,
          2.8276475490355755e-9, 0.27537536599750795, 1.6235093209315257e-9,
          3.188615397144686e-9, 0.08982650391376494, 2.3572615595497384e-9,
          2.6849182857912843e-9, 0.3141349068460301, 1.3353503527862552e-9,
          9.203446261667965e-10, 3.3069743421698417e-9, 1.162462290104251e-9,
          0.2727088136140823, 1.7055521331439893e-8, 8.013519271836428e-9,
          5.639287253039109e-9, 1.531007247532593e-8]
    @test isapprox(w8.weights, wt, rtol = 1e-5)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [7.588979920020583e-8, 7.235173576512984e-8, 0.06299709789211731,
          2.3709342092446354e-8, 0.2783820123620202, 1.2736317346774133e-8,
          1.985609222171503e-8, 0.0813234354283379, 1.8734391596679376e-8,
          2.1498025232682286e-8, 0.3030873017931937, 1.0579582289419378e-8,
          6.789021460898904e-9, 3.452040984496498e-8, 1.0562910024044365e-8,
          0.2742095438849846, 1.0422059021703477e-7, 5.2357203238839924e-8,
          4.3291557031371343e-8, 1.0154236889772837e-7]
    @test isapprox(w9.weights, wt, rtol = 0.0005)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.8157901433862678e-8, 1.9512946463774696e-8, 4.4301242316057525e-8,
          2.2167211764309652e-8, 6.368518874763926e-5, 8.45652568648703e-9,
          0.9999359106455318, 1.4373547170164613e-8, 2.0185920540841476e-8,
          1.4608373243064736e-8, 1.3292999458325192e-8, 8.935605593795477e-9,
          6.428263039630157e-9, 1.0872705786884222e-8, 6.63475540494494e-9,
          1.1646292862752679e-7, 2.682028256528234e-8, 1.4057981627967789e-8,
          2.1459923694047825e-8, 1.743660605718871e-8]
    riskt = 0.6883715306208451
    rett = 0.0018453720271312279
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [4.048605860824538e-11, 3.8731947721508655e-11, 3.2377314098601185e-11,
          3.508937457756166e-11, 0.8503244505829672, 4.8965691320877765e-11,
          0.14967554868689337, 4.5839239874615343e-11, 3.709318311602525e-11,
          4.4977375439214804e-11, 4.651803898201334e-11, 4.992978416397431e-11,
          5.2206716969224834e-11, 4.873447325078158e-11, 5.199127058056857e-11,
          9.428725791518539e-12, 2.7118426619909138e-11, 4.506093061984433e-11,
          3.447179865858097e-11, 4.1119133365515745e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.9694231584880834e-9, 2.186961578189056e-9, 2.653797065880974e-9,
          2.4102059954852963e-9, 0.8533950124938413, 9.542350649976943e-10,
          0.1466049534684535, 1.502079131379396e-9, 2.322678738744387e-9,
          1.5932403325286268e-9, 1.4283982204748175e-9, 9.512058346599107e-10,
          7.146969439357207e-10, 1.183507083890166e-9, 7.187079249837766e-10,
          4.4614325455997415e-9, 3.0107141295967857e-9, 1.5832158786661703e-9,
          2.4873921091996892e-9, 1.9058133726661404e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-7

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 1e-8

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1 * 1.000001
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-7

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) <= 1e-10

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4 ||
          abs(dot(portfolio.mu, w20.weights) - ret4) < 1e-10
end

@testset "EDaR < RDaR" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)

    obj = MinRisk()
    rm = RDaR2(; kappa = 5e-3)
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm = EDaR2()
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 0.0001)

    obj = SR(; rf = rf)
    rm = RDaR2(; kappa = 5e-3)
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    rm = EDaR2()
    w2 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    @test isapprox(w1.weights, w2.weights, rtol = 0.05)
end

@testset "Kurt" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = Kurt2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.2206987375321365e-8, 0.039710885748646646, 1.5461868222416377e-8,
          0.0751622837839456, 1.5932470737314812e-8, 0.011224033304772868,
          8.430423528936607e-9, 0.12881590610168409, 8.394156714410349e-9,
          3.2913090335193894e-8, 0.3694591285102107, 8.438604213700625e-9,
          4.8695034432287695e-9, 0.039009375331486955, 2.101727257447083e-8,
          0.02018194607274747, 4.58542212019818e-8, 0.18478029998119072,
          1.0811006933572572e-8, 0.1316559568357098]
    riskt = 0.00013888508487777416
    rett = 0.0004070691284861725
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.346761464122291e-8, 0.039709686399958165, 4.2439154551774695e-8,
          0.07516162045700765, 4.376521237584917e-8, 0.011226052455372382,
          2.342048287168965e-8, 0.12881618719027788, 2.309188468263431e-8,
          9.036562536424303e-8, 0.3694601232368255, 2.326900392013725e-8,
          1.3534784507986142e-8, 0.03900840597734345, 5.815598324113715e-8,
          0.020181283960573042, 1.2574120903786293e-7, 0.1847801134360715,
          2.9678902026737772e-8, 0.13165601995671322]
    @test isapprox(w2.weights, wt, rtol = 5e-7)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.1194424598122522e-7, 0.03971075349531816, 1.2901126327136693e-7,
          0.07516129185133234, 1.2921254972436994e-7, 0.011225944233697117,
          6.843665706820523e-8, 0.12881622744367655, 7.245942055351997e-8,
          2.602870533812648e-7, 0.36945959008164214, 7.767774331583506e-8,
          4.6098957392625345e-8, 0.03900927746680018, 1.7826227548984752e-7,
          0.020181734532562706, 3.438796553625847e-7, 0.1847790408465989,
          9.217946260056386e-8, 0.13165463059908764]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [6.488257778365516e-9, 4.212509629221129e-8, 0.012824070122197608,
          0.02084004300409522, 0.35357636535400944, 1.9709911723401217e-9,
          0.08635105653507036, 4.974079563787855e-7, 1.1571650668948285e-8,
          9.868829352081284e-9, 2.016748820188375e-8, 1.799041253534452e-9,
          1.1714958351363139e-9, 3.6613705846949442e-9, 1.1209037613139013e-9,
          0.13915737487845023, 0.30507359437131254, 2.243500862811201e-8,
          0.08217685583646381, 2.0110310944520718e-8]
    riskt = 0.0003482031728018851
    rett = 0.0013644412127685368
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.683771781464599e-9, 4.956361469283792e-8, 0.010938887497489176,
          0.02251547139083388, 0.3366005075116465, 1.7156398584585569e-9,
          0.07880524840945159, 0.04437599281224301, 1.0111039348135077e-8,
          9.163869317977593e-9, 3.4033535911268134e-8, 1.4995723607852093e-9,
          9.99975131891246e-10, 3.3104621064556554e-9, 9.63823038421442e-10,
          0.132388560081269, 0.2908997454167932, 2.9820899728381046e-8, 0.08347541834589083,
          2.1668179618436748e-8]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [5.3602657804587796e-9, 3.511084837798627e-8, 0.010740506890032522,
          0.022447066429593616, 0.3365913416668436, 1.7209881919520007e-9,
          0.07876383894774337, 0.04488444185337361, 8.551428803375859e-9,
          7.786222242800678e-9, 2.636763590808593e-8, 1.4087742879225946e-9,
          9.176587512903323e-10, 3.1683149901981274e-9, 8.940291181172879e-10,
          0.13242337878313162, 0.2908217738328046, 2.3766272783511127e-8,
          0.08332751977813833, 1.676589955170418e-8]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [3.6858654756755185e-8, 0.007745692921941757, 7.659402306115789e-7,
          0.06255101954920105, 0.21591225958933066, 1.164996778478312e-8,
          0.04812416132599465, 0.1265864284132457, 7.028702221705754e-8,
          6.689307437956413e-8, 0.19709510156536242, 1.0048124974067658e-8,
          5.6308070308085104e-9, 2.6819277429081264e-8, 6.042626796051015e-9,
          0.09193445125562229, 0.20844322199597035, 0.03380290484590025,
          0.00779744381804698, 6.3145495978703785e-6]
    riskt = 0.00020556674631177048
    rett = 0.0009657710513568699
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1760565974247853e-8, 0.011976065399594853, 1.149574088536465e-7,
          0.060992858119788856, 0.1930027086636508, 4.372814394617514e-9,
          0.04149627583877934, 0.12397070512242318, 2.2187620853197638e-8,
          2.1717199802508656e-8, 0.22978040051543996, 3.660703729436199e-9,
          2.222790420663753e-9, 1.0067084625550425e-8, 2.428676339992691e-9,
          0.08320879001281592, 0.1805759875012898, 0.05910174698075718,
          4.4689262170231523e-7, 0.015893821577973446]
    @test isapprox(w8.weights, wt, rtol = 1e-6)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.0664475786084095e-9, 0.007572090226278144, 1.9754381142249716e-7,
          0.058379179431076934, 0.2178793972737544, 1.0447050126602177e-9,
          0.04711756122071877, 0.12440956472154607, 6.804676118027675e-9,
          6.374737276918507e-9, 0.19573175406539572, 8.775916665728969e-10,
          5.321165284185168e-10, 2.375636780750861e-9, 5.578315708863142e-10,
          0.09176243464543488, 0.20643976967280292, 0.038984207376256505,
          0.011722350109110454, 1.4720800709676136e-6]
    @test isapprox(w9.weights, wt, rtol = 1.0e-5)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [5.934616490250286e-9, 6.321105368581324e-9, 7.96917681590385e-9,
          7.225655982032708e-9, 8.525413620213173e-7, 2.829021876726861e-9,
          0.9999990501666552, 4.36059583429035e-9, 6.709075233436358e-9,
          4.778358200172271e-9, 4.230021296813207e-9, 2.982536918030274e-9,
          2.1818668099291885e-9, 3.5668295461854984e-9, 2.256026580006305e-9,
          1.0491891681968458e-8, 8.312915068047865e-9, 4.580013453306448e-9,
          6.950956345114773e-9, 5.611319150237127e-9]
    riskt = 0.009798228578028432
    rett = 0.0018453754838388065
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [3.928704500020522e-12, 4.9881916832563855e-12, 7.843180381039878e-12,
          6.44134912344952e-12, 0.8503244345698826, 1.9360859919801295e-12,
          0.1496755653518743, 1.1551467682721733e-12, 5.971299860169435e-12,
          2.0062699621274885e-12, 9.524486168090581e-13, 1.4183867779642828e-12,
          2.7164288432625825e-12, 3.9052544214393357e-13, 2.7253698403652935e-12,
          1.4646682952884316e-11, 9.166127868109698e-12, 1.6255496200997712e-12,
          6.644901574411177e-12, 3.686285927356827e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.880823344092973e-9, 2.0626119736752934e-9, 2.5848636966075764e-9,
          2.31687490729002e-9, 0.8533951266408121, 8.626779505686536e-10,
          0.14660484110980923, 1.3830898016174373e-9, 2.2141216509089502e-9,
          1.475058635433737e-9, 1.3169033100692695e-9, 8.467442461877332e-10,
          6.160373769288569e-10, 1.0805017296662783e-9, 6.255772712322293e-10,
          4.467193468114549e-9, 2.899137976718247e-9, 1.4655382099084609e-9,
          2.376423305979761e-9, 1.7751999758969413e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "Kurt Reduced" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))),
                           max_num_assets_kurt = 1)
    asset_statistics2!(portfolio)
    rm = Kurt2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [6.258921377348572e-7, 3.680102939415165e-6, 7.486004924604795e-7,
          0.03498034415709424, 7.917668013849855e-7, 1.2350801714348813e-6,
          3.0460581400971164e-7, 0.08046935867869996, 5.699668814229727e-7,
          8.85393394989514e-7, 0.6357270365898279, 5.979410095673201e-7,
          2.7962721329567345e-7, 1.8757932499250933e-6, 5.070585649379864e-7,
          0.00127706653757055, 1.2457133780976972e-6, 0.11108722055351541,
          6.616561323348826e-7, 0.13644496428511088]
    riskt = 0.0001588490319818568
    rett = 0.0003563680681010386
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [6.651646121453831e-7, 3.9289579493824055e-6, 7.951215187316133e-7,
          0.034981303735917395, 8.407837261774118e-7, 1.315189410869102e-6,
          3.2080440601670075e-7, 0.08046930933469248, 6.034152964641091e-7,
          9.410030616610646e-7, 0.6357234132233586, 6.327666560087137e-7,
          2.939185337655592e-7, 2.001627769134478e-6, 5.362110948191323e-7,
          0.0012778892264629603, 1.3259045130301894e-6, 0.1110872373724336,
          7.019309739133741e-7, 0.13644594430761298]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [7.981264356450749e-8, 4.1055882348367827e-7, 8.863687379738684e-8,
          0.034971480739674564, 9.312166136523345e-8, 1.6943287780592163e-7,
          3.577564832711277e-8, 0.08046249478112905, 6.693601280800449e-8,
          1.0589219505180366e-7, 0.6357235014970133, 7.285250152809561e-8,
          3.425817988268031e-8, 2.38514148267638e-7, 6.334831213823845e-8,
          0.0011915154687944592, 1.442447364394241e-7, 0.11113298741876672,
          7.792046342487458e-8, 0.1365163387895441]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [1.633189843996199e-7, 4.7392695446429325e-7, 8.799602597982136e-7,
          2.682810240405649e-6, 0.33250417011271577, 6.252670516798967e-8,
          0.08180392349672638, 7.710747796092221e-7, 2.9634480103019065e-7,
          2.658102014464364e-7, 4.034001318838129e-7, 5.870525160264156e-8,
          3.865052954354365e-8, 1.0933111816146697e-7, 3.72375730689396e-8,
          0.12820198647078426, 0.4574811481910088, 4.1279374482297674e-7,
          1.7817988595184708e-6, 3.340386297254291e-7]
    riskt = 0.00035186660326760855
    rett = 0.001357118987007538
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [9.358597765840143e-8, 3.2853348804084037e-7, 6.036593066013873e-7,
          1.4009187563345273e-5, 0.3219311277450035, 3.7718006650865916e-8,
          0.07599801296982352, 7.256094514174488e-6, 1.7414044110684151e-7,
          1.7420307762075935e-7, 4.800036051204161e-7, 3.3973421175193506e-8,
          2.253112883489227e-8, 6.784758619739153e-8, 2.2090247195809606e-8,
          0.12771674199358957, 0.47432491366547447, 4.032078181781363e-7,
          5.268419299820633e-6, 2.2843062727063477e-7]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.968353952065472e-9, 1.359488572269217e-8, 2.3112417923669387e-8,
          3.0294830020741113e-7, 0.32186201607634074, 1.5751425390673517e-9,
          0.07594162595742708, 2.238726966389482e-7, 7.091058557142758e-9,
          7.025076335444092e-9, 1.9125798952487517e-8, 1.3432615677339767e-9,
          8.777854917088155e-10, 2.8052805328652988e-9, 8.491480271667892e-10,
          0.12776815597172467, 0.47442742083745953, 1.653563651925195e-8,
          1.4715428093387167e-7, 9.277923860264034e-9]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [2.1869958623435718e-8, 1.2027489763838995e-7, 8.307710723713282e-8,
          0.04981239216610764, 0.19930305948126495, 9.890114269259893e-9,
          0.041826524860577356, 0.10466467192338667, 3.332719673064681e-8,
          4.0257776120217934e-8, 0.23960266714976633, 8.639963684421896e-9,
          4.958509460440708e-9, 2.0027087137090508e-8, 5.256105741068125e-9,
          0.08392271758019651, 0.28086715326117795, 2.469501355591167e-7,
          1.0503063749371977e-7, 1.1401803293988891e-7]
    riskt = 0.00020482110250140048
    rett = 0.0009552956056983061
    @test isapprox(w7.weights, wt, rtol = 1.0e-7)
    @test isapprox(r3, riskt, rtol = 1.0e-7)
    @test isapprox(ret3, rett, rtol = 1.0e-7)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [9.821449555203826e-9, 6.651240517387659e-8, 3.650718258300155e-8,
          0.053308999716557656, 0.181994700448441, 4.514319742356884e-9,
          0.03372737467501634, 0.10801289813580604, 1.511877302486414e-8,
          1.739208041739533e-8, 0.30332966829636443, 3.845883458211269e-9,
          2.305328525313704e-9, 9.226178492543905e-9, 2.490966736743529e-9,
          0.07839193989339778, 0.2412338812171733, 2.3936990707878654e-7,
          4.479630659284112e-8, 8.571646208004972e-8]
    @test isapprox(w8.weights, wt, rtol = 1.0e-5)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.019321105406507e-9, 1.7512676212088132e-8, 1.646738171019031e-8,
          0.04735993141254452, 0.20221078402831122, 1.3026581539183748e-9,
          0.041207302156254164, 0.10602572874014794, 5.168634386630353e-9,
          5.6765942417395675e-9, 0.2391061685008948, 1.1345809773632338e-9,
          6.936147159333732e-10, 2.603160510236683e-9, 7.197443300911477e-10,
          0.08488718199497128, 0.27920270709018613, 7.987377781799362e-8,
          4.670957743132964e-8, 1.519496843349421e-8]
    @test isapprox(w9.weights, wt, rtol = 1.0e-5)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [4.936466409028226e-8, 5.276389568036465e-8, 6.809010676649967e-8,
          6.105779239028839e-8, 6.716485718335176e-6, 2.444808944581224e-8,
          0.9999924571194886, 3.623677491249064e-8, 5.6060940759989673e-8,
          3.9533919792269964e-8, 3.521613671950229e-8, 2.562615210438123e-8,
          1.9584957360588115e-8, 3.0036845758033746e-8, 2.0122639539235155e-8,
          9.369901721812878e-8, 7.159573942902736e-8, 3.79872737184743e-8,
          5.837684057897428e-8, 4.6593006863234134e-8]
    riskt = 0.009798096724216276
    rett = 0.001845374269455763
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.0892479600194666e-10, 2.822452483155382e-10, 4.3716290783456694e-10,
          3.6203185047954e-10, 0.8503240818449798, 1.1465668255679708e-10,
          0.14967591348901493, 6.943764865364773e-11, 3.352853005619898e-10,
          1.0468394630159045e-10, 5.1106014598547553e-11, 1.1040917744355478e-10,
          2.2365100672754073e-10, 4.1667916032801746e-11, 1.5966515635740096e-10,
          9.379284429283585e-10, 5.378011809450025e-10, 9.381817541922242e-11,
          3.8732738194225043e-10, 2.0820251994844338e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.895934412681643e-9, 2.0549992014986265e-9, 2.535516656322881e-9,
          2.2914284217146943e-9, 0.8533951012284242, 8.871289408894401e-10,
          0.14660486676143647, 1.392704877768976e-9, 2.193180923591071e-9,
          1.4666986717592008e-9, 1.3189816250333021e-9, 8.544152685834694e-10,
          6.230780802258776e-10, 1.094522716242892e-9, 6.281374607135794e-10,
          4.3705825482806406e-9, 2.8270575684180838e-9, 1.4782117443345719e-9,
          2.344662860522784e-9, 1.7528972506290458e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-5

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r2) < 5e-5

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r3) < 1e-5

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 5e-5

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4 ||
          abs(dot(portfolio.mu, w16.weights) - ret4) < 1e-10

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "SKurt" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = SKurt2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.84452954864755e-8, 0.09964222365209786, 1.3574917720415848e-8,
          3.86114747805695e-8, 1.3966729610624488e-8, 3.110357119905182e-7,
          6.064715171940905e-9, 0.13536486151149585, 1.1522227247871875e-8,
          1.636443293768575e-8, 0.39008850330648476, 1.8792073784410723e-8,
          8.525841353694015e-9, 0.021330458258521926, 3.206796675755174e-8,
          5.682782161200388e-8, 1.3934759941686095e-7, 0.19923258780375303,
          1.7400011528901028e-8, 0.15434066292082727]
    riskt = 0.0001095671204061385
    rett = 0.0003976490301536407
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.4123555330802712e-8, 0.09964264179900931, 1.0389791635772402e-8,
          2.9665468862085935e-8, 1.069176391564201e-8, 2.3993933009769257e-7,
          4.673843623065419e-9, 0.13536477888417348, 8.830859806797828e-9,
          1.2543332114195561e-8, 0.39008833290541695, 1.4479879451547413e-8,
          6.604538265039614e-9, 0.02133034829887352, 2.474345253016428e-8,
          4.3677213426996515e-8, 1.07055180787398e-7, 0.19923283445217532,
          1.3332188501379585e-8, 0.15434052290995307]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.182876715354597e-7, 0.09964027748413909, 8.301479250366766e-8,
          2.1331482824837186e-7, 8.310916375958217e-8, 1.6554888967503154e-6,
          3.647966541899318e-8, 0.13536463393139103, 7.063079990258532e-8,
          9.996415333091541e-8, 0.3900888723341002, 1.1682097241806513e-7,
          5.6179979676480455e-8, 0.02133194599107856, 1.9185853901637036e-7,
          3.0563695269932246e-7, 6.987045402208702e-7, 0.19923015345472767,
          1.0489268425328608e-7, 0.15434028242092374]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [2.4904484918041937e-9, 5.2070103784409286e-9, 3.632812182685684e-9,
          2.5387115520236563e-9, 0.638391391480839, 5.878002217393133e-10,
          0.10348430082359686, 2.386563225125209e-9, 2.665067360605095e-9,
          1.4354466418425415e-9, 2.389967987814404e-9, 5.620445298774204e-10,
          3.4534457036599057e-10, 1.0740960219367014e-9, 3.6544635783745383e-10,
          0.1920706657954386, 0.06605360256370367, 2.7928083797519015e-9,
          7.900369580148254e-9, 2.9624844928037823e-9]
    riskt = 0.0003259385892067888
    rett = 0.0016255912800443321
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.3609955898035992e-9, 5.988517088367832e-9, 3.4914335221653928e-9,
          2.5948086586120806e-9, 0.5728216410207285, 5.779713009501383e-10,
          0.08065973041121713, 3.057527629258053e-9, 2.7141559634744322e-9,
          1.5053740340839961e-9, 3.082763681025962e-9, 5.28958118015333e-10,
          3.2925176261057116e-10, 1.1042225502596244e-9, 3.4656484003445553e-10,
          0.183720559157383, 0.16279802328843573, 3.4976449308380173e-9,
          1.1683133896657015e-8, 3.2589120945571132e-9]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [4.6584039037999875e-9, 1.0437824029497026e-8, 6.3437394970659326e-9,
          4.841738242179914e-9, 0.5735978132688941, 1.1945265430395104e-9,
          0.08042679742676263, 5.78265473009275e-9, 5.023676582645069e-9,
          2.907591441035866e-9, 5.819176403662165e-9, 1.0408297923295037e-9,
          6.334933753557771e-10, 2.226057030951269e-9, 6.689717713575569e-10,
          0.18385141903001542, 0.1621238880163581, 6.632128953877921e-9,
          1.8074148381643116e-8, 5.973009051050903e-9]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [8.263501742935076e-8, 3.4292121981081834e-5, 8.553642485927413e-8,
          1.0517936993748617e-7, 0.32096393208295665, 1.9848129848992033e-8,
          0.004686373576551095, 0.05273663584046675, 6.751336821499605e-8,
          5.166788731416254e-8, 0.24476417791090582, 1.7957314779078257e-8,
          9.390528572513579e-9, 5.4617904908192196e-8, 1.060832570103225e-8,
          0.135854021084771, 0.24095692751428416, 1.9017748026664851e-6,
          2.5600086153798754e-7, 9.771381477598014e-7]
    riskt = 0.0001695775787535489
    rett = 0.0010654948366056365
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.979181215458543e-9, 0.016330578477593557, 8.411069048603343e-9,
          1.1858991228455916e-8, 0.2826499326544149, 2.2667055298296563e-9,
          0.000680804753932186, 0.0709964873763113, 6.8935420589943575e-9,
          5.529631056001788e-9, 0.2689475964010183, 2.0070917581001112e-9,
          1.1207236579564627e-9, 6.260671141950251e-9, 1.2674092727255503e-9,
          0.12089027511364908, 0.2186100521635675, 0.013238468901051, 2.4337089023812687e-8,
          0.007655726226357345]
    @test isapprox(w8.weights, wt, rtol = 5.0e-5)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.995770968543026e-9, 0.0032260397267616318, 3.170914186391361e-9,
          3.689803553978177e-9, 0.3166856057167533, 7.491713594851984e-10,
          0.008262932499825616, 0.05939179633083242, 2.5569571124615203e-9,
          1.9654623647107404e-9, 0.24031055881428048, 6.558972130243647e-10,
          3.674660509648308e-10, 2.1043975888567013e-9, 4.085505187640167e-10,
          0.1325299031316699, 0.23959030972890885, 2.7596039177896227e-6,
          2.210440904343929e-8, 5.367825006840021e-8]
    @test isapprox(w9.weights, wt, rtol = 1.0e-5)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.3008303869113934e-9, 1.3824739065615774e-9, 1.739049541856439e-9,
          1.5857029726860548e-9, 1.3246917338236657e-7, 6.255681936141374e-10,
          0.9999998462305789, 9.591064844133509e-10, 1.468740987272949e-9,
          1.0517327887109987e-9, 9.304794636340672e-10, 6.590363959442953e-10,
          4.835667782113111e-10, 7.860364519932236e-10, 4.996955739699139e-10,
          2.274893547771803e-9, 1.8028688458061267e-9, 1.0052665761105784e-9,
          1.5147632246682218e-9, 1.2304354253054986e-9]
    riskt = 0.0027556277372339185
    rett = 0.0018453756156264034
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [4.039397479633368e-12, 5.2295216777434724e-12, 8.857647392262786e-12,
          7.614229452258964e-12, 0.8503243913967153, 2.4422914272634276e-12,
          0.14967560851583278, 1.0679826989983144e-12, 6.984952616014042e-12,
          2.4305845082496682e-12, 6.910077937775571e-13, 1.6994904046619193e-12,
          3.113595940215685e-12, 8.238902329531664e-13, 3.187362714522768e-12,
          1.662072504033373e-11, 9.946555734291227e-12, 1.3598132389458005e-12,
          7.453203675411751e-12, 3.889608014795143e-12]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.919021598230146e-9, 2.1069881230082097e-9, 2.6445420486000984e-9,
          2.369369488637192e-9, 0.853395053184621, 8.772300274589101e-10,
          0.1466049138290405, 1.4129460842144394e-9, 2.271697232780608e-9,
          1.5112596835179365e-9, 1.344937975458357e-9, 8.624294781531687e-10,
          6.268800218582154e-10, 1.1019330417508498e-9, 6.366530534406935e-10,
          4.575515169325216e-9, 2.97031119565965e-9, 1.4972209127389936e-9,
          2.440845541320912e-9, 1.8165577763148767e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1 * 1.05
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 * 1.05

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1 * 1.05
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 * 1.05

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "SKurt Reduced" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))),
                           max_num_assets_kurt = 1)
    asset_statistics2!(portfolio)
    rm = SKurt2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [9.537055656561948e-7, 0.08065159914970102, 6.97429322507607e-7,
          1.636243884801297e-6, 6.970775807421223e-7, 3.475223837047256e-6,
          2.8901276220144585e-7, 0.1269025346766842, 6.27357228262538e-7,
          8.136543409694392e-7, 0.4622659726775211, 1.0093804242816126e-6,
          4.237381686192652e-7, 2.0106199669190677e-5, 1.10186359354052e-6,
          2.157409439859392e-6, 3.9496933962571675e-6, 0.1701191581088843,
          8.973744700260093e-7, 0.16002190002352545]
    riskt = 0.00011018128303928912
    rett = 0.00039273793904369474
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.0280901070218869e-6, 0.08065151841675483, 7.508176006083672e-7,
          1.7607419192846997e-6, 7.503921684643635e-7, 3.7461200216006295e-6,
          3.092491161099152e-7, 0.1269020801725786, 6.738163300296364e-7,
          8.759116741354499e-7, 0.46226444562046065, 1.0839909176783647e-6,
          4.532817254443864e-7, 2.1752849487067093e-5, 1.1835593825385768e-6,
          2.3209739554145836e-6, 4.250432993310309e-6, 0.17011825821031992,
          9.657096804042262e-7, 0.16002179164280694]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.2779657967567574e-7, 0.08067090785573523, 8.872973554419637e-8,
          2.02646457064895e-7, 8.770867788746825e-8, 4.782067955434449e-7,
          3.5886085788986314e-8, 0.1269030231796038, 7.927966460562316e-8,
          1.034524947917703e-7, 0.46226041060719314, 1.3078672399148893e-7,
          5.5085725397735354e-8, 2.148976415720433e-6, 1.4202291218909052e-7,
          2.660526149610506e-7, 4.770198981028706e-7, 0.1701351330514935,
          1.1366508082987266e-7, 0.1600259879901123]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [2.3518312045771565e-7, 4.548643596588275e-7, 3.289992438366905e-7,
          2.3649353086510158e-7, 0.6546702534218136, 6.07077088037061e-8,
          0.1018376879288071, 2.2938784898777355e-7, 2.504921386432834e-7,
          1.4230718705252731e-7, 2.3118132526823424e-7, 5.8659406953799776e-8,
          3.610622619246454e-8, 1.0854896396294657e-7, 3.8210498866152866e-8,
          0.1938129483117614, 0.04967546347748593, 2.6587343173773683e-7,
          6.969988018821273e-7, 2.72846338644695e-7]
    riskt = 0.0003318052808415242
    rett = 0.0016372892218660332
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.3097432899096424e-7, 5.148216644696814e-7, 3.2223880964486606e-7,
          2.4930853386638423e-7, 0.5829360395216417, 6.590069056396985e-8,
          0.07877810954999398, 2.9539172416431005e-7, 2.6077691805143425e-7,
          1.5720694494763028e-7, 2.986540829790903e-7, 6.097583569646534e-8,
          3.8615636129108087e-8, 1.193999654951914e-7, 4.090835045979042e-8,
          0.18421305306408883, 0.15406854200678202, 3.3260661739458826e-7,
          9.635346651398179e-7, 3.04542725482918e-7]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [5.592292528115121e-9, 1.202101538600941e-8, 7.411797803589144e-9,
          5.7509092416740405e-9, 0.5838039625958914, 1.5330434045627518e-9,
          0.07852360234608709, 7.000471494174188e-9, 6.03298199957695e-9,
          3.576716251052727e-9, 7.047464245021316e-9, 1.3504294568269924e-9,
          8.300775549641303e-10, 2.785534358896565e-9, 8.758812653158011e-10,
          0.18435005553391626, 0.15332228157359304, 7.986150985540505e-9,
          2.1168960636168925e-8, 6.986785730276258e-9]
    @test isapprox(w6.weights, wt, rtol = 1.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [9.329112161227789e-8, 1.8997236041327231e-6, 9.183667269717346e-8,
          1.1278547529605994e-7, 0.323307535710129, 2.5473368939712553e-8,
          6.050657719406614e-6, 0.04416525712754569, 7.422093025293419e-8,
          6.013968073613049e-8, 0.2607281038602079, 2.2840291563131592e-8,
          1.2053370397405268e-8, 6.734594412605242e-8, 1.3612437340851914e-8,
          0.13505441433828766, 0.23673428116977457, 1.0078714608278738e-6,
          2.4778495858369644e-7, 6.28157019375726e-7]
    riskt = 0.00016836304121264654
    rett = 0.0010573999779863363
    @test isapprox(w7.weights, wt, rtol = 5.0e-7)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.1384620267984809e-8, 0.00662359878216065, 1.1655458374758496e-8,
          1.558786317301833e-8, 0.2859407191685224, 3.282418180451198e-9,
          6.197934976122882e-8, 0.06812668071281264, 9.344910157126645e-9,
          7.755120244286483e-9, 0.29490429019324593, 2.8992877361049625e-9,
          1.589800964346545e-9, 8.974740439225e-9, 1.8174620365057608e-9,
          0.12090221961634619, 0.22236019315291974, 6.143184302567226e-7,
          2.9749578537137244e-8, 0.0011415180349523546]
    @test isapprox(w8.weights, wt, rtol = 5e-6)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.7406841565341014e-9, 1.0355514800819468e-7, 1.8312529676507407e-9,
          2.1482952971960594e-9, 0.3189053797843766, 4.4586576106411933e-10,
          0.0040834499328934555, 0.05250000836272301, 1.4806385027930565e-9,
          1.1384931185319011e-9, 0.2544007225921324, 3.931235158243691e-10,
          2.1965595118456138e-10, 1.225382838435043e-9, 2.4476858055176825e-10,
          0.13215261274875856, 0.23795750631688745, 1.7701947218083515e-7,
          1.0982192168421308e-8, 1.783725546307163e-8]
    @test isapprox(w9.weights, wt, rtol = 1.0e-6)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [4.936118678171942e-8, 5.2760199097238493e-8, 6.808537976940716e-8,
          6.105353956885902e-8, 6.7161059615987615e-6, 2.444636593372152e-8,
          0.9999924575570034, 3.6234210052048526e-8, 5.6057026646045884e-8,
          3.953112344037875e-8, 3.521364356883661e-8, 2.5624345310355472e-8,
          1.9583592567291033e-8, 3.00347200640912e-8, 2.01212345794344e-8,
          9.36926021227503e-8, 7.159078911490752e-8, 3.7984586231814474e-8,
          5.837277022750332e-8, 4.6589720102054876e-8]
    riskt = 0.002755589420705884
    rett = 0.0018453742695479151
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.0892585315372412e-10, 2.822466889970162e-10, 4.371653042827221e-10,
          3.620337871435581e-10, 0.8503240818501003, 1.1465724270578049e-10,
          0.14967591348386972, 6.94378871477989e-11, 3.35287040806797e-10,
          1.0468438517680516e-10, 5.110616760581589e-11, 1.104097550121883e-10,
          2.236522140139172e-10, 4.166819940080418e-11, 1.5966588454708028e-10,
          9.379338672109943e-10, 5.378041567349732e-10, 9.381855296986243e-11,
          3.873294295939177e-10, 2.0820353121112192e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.8791767655278526e-9, 2.037007690615658e-9, 2.5133180718195447e-9,
          2.2713743479155395e-9, 0.8533951165640847, 8.791665528821181e-10,
          0.1466048517068985, 1.3805253841196528e-9, 2.1740229056415066e-9,
          1.4539754234602212e-9, 1.3074949311165239e-9, 8.469108634197527e-10,
          6.175609242439616e-10, 1.084899646412443e-9, 6.226055421512239e-10,
          4.331495744557022e-9, 2.802319919648561e-9, 1.4652556355678416e-9,
          2.3241823968732896e-9, 1.7377241142261507e-9]
    @test isapprox(w12.weights, wt, rtol = 5.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "Skew" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = Skew2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [2.409023468372493e-6, 0.17210378292297784, 5.103256374511177e-7,
          4.2722770517309445e-7, 3.5848058134265105e-6, 6.972041664934061e-7,
          3.2915769657012085e-7, 0.1415418122674741, 4.3289050477603765e-7,
          4.5431144777227054e-7, 0.07897482123611543, 0.023295191901219474,
          2.0444083999934734e-6, 3.3398275530097316e-6, 0.1761574592680367,
          0.042496745295449355, 3.003590887382274e-6, 0.23119283730811144,
          6.400097708092224e-7, 0.1342194770175644]
    riskt = 0.0016553752647584506
    rett = 0.0001952238162305396
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.5158744761559545e-6, 0.1721029776610835, 5.382833339292533e-7,
          4.529918476814139e-7, 3.7504467816068045e-6, 7.337759838178915e-7,
          3.338952681789772e-7, 0.1415418135552326, 4.558168881894815e-7,
          4.811140908680752e-7, 0.07897649570561492, 0.023295942315643442,
          2.148340358809871e-6, 3.469766225848474e-6, 0.17615681981070352,
          0.042498463814182236, 3.130619059216015e-6, 0.23119005044872828,
          6.739369962003686e-7, 0.1342187518275011]
    @test isapprox(w2.weights, wt, rtol = 1e-7)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [4.788427000824502e-6, 0.17210023034570623, 9.601397397225009e-7,
          8.071784548846651e-7, 6.131359251062126e-6, 1.6110036323815667e-6,
          5.6931439969033e-7, 0.1415416746910567, 8.047986677389847e-7,
          8.542515304869642e-7, 0.07897501072680707, 0.023296604182628716,
          4.154988679086088e-6, 6.814423892853844e-6, 0.176153987321236,
          0.04250139969869921, 5.300251165220983e-6, 0.23118179573960823,
          1.1998382537872687e-6, 0.13421530131959014]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [7.333832235462642e-11, 9.456924850163512e-11, 1.8213116779543554e-10,
          1.4644868551189786e-10, 0.7646099139145731, 8.141585690505264e-11,
          0.23539008427414068, 2.8963485990059165e-11, 1.2075936588112584e-10,
          1.2586287937020945e-11, 1.6595054043047734e-11, 1.1914653130961956e-10,
          1.0983720843196118e-10, 5.335725897717632e-11, 6.930324803207184e-11,
          3.047535877238858e-10, 2.0656330346845687e-10, 1.0431583106728689e-11,
          1.3065220329497258e-10, 5.043371915328308e-11]
    riskt = 0.0043590315064662806
    rett = 0.0018073262713875812
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [4.499815592553308e-11, 6.256708180523985e-11, 9.168351622115868e-11,
          7.813851148095504e-11, 0.8609439937379277, 2.618938674017737e-11,
          0.13905600519689662, 1.0764007591635799e-11, 7.324639625912896e-11,
          2.100841086049629e-11, 6.4779402405867144e-12, 3.3986477122662476e-11,
          5.070962707807822e-11, 1.251976295721348e-11, 4.414411320608923e-11,
          2.370137169631524e-10, 1.2301220222279568e-10, 1.760598045824916e-11,
          8.663003235788643e-11, 4.448040269470545e-11]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.2922849413542943e-9, 1.4311920361998716e-9, 1.694421721798563e-9,
          1.54285172701003e-9, 0.8638306021843428, 5.380681284397553e-10,
          0.13616937618340577, 9.288961478143992e-10, 1.4920838282417072e-9,
          9.629619056862926e-10, 8.524203967208526e-10, 4.912755268204297e-10,
          2.9357751693015485e-10, 6.868890552363791e-10, 3.015435213958195e-10,
          3.36439591839203e-9, 1.9643900434173774e-9, 9.91656665863237e-10,
          1.6134657419341152e-9, 1.1898766752904816e-9]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [9.735091341984711e-10, 0.05423761151449597, 2.6660056855991525e-10,
          2.993109735501239e-10, 0.46437248293533817, 1.1905394280455e-10,
          0.005519488091126105, 6.516199104606112e-10, 3.6541663841870745e-10,
          2.6134486312327216e-10, 4.272088655808922e-10, 2.420870547271647e-10,
          7.794962047253004e-11, 2.576929482481253e-10, 1.0413893099420468e-10,
          0.4452656653259548, 0.030604745504608514, 7.911667121694307e-10,
          7.65302298768296e-10, 1.0260740026129097e-9]
    riskt = 0.0026438554509313934
    rett = 0.001430936761062597
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [5.931275589123377e-8, 0.10853460662337316, 1.391899559411138e-8,
          1.4896863515371605e-8, 0.4070400425420224, 6.6277575200097715e-9,
          0.0006329581276294483, 8.256144088137677e-8, 1.9649939896513973e-8,
          1.3826517648949991e-8, 3.2440400187296607e-8, 1.2688002575726309e-8,
          4.527843677701342e-9, 1.5758389458165656e-8, 6.0258358527802494e-9,
          0.383474687777955, 0.10031715780778228, 1.1308265764844824e-7,
          5.3486095574896416e-8, 9.831774174552091e-8]
    @test isapprox(w8.weights, wt, rtol = 5e-8)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.1459769242253624e-8, 0.07940369233211204, 3.836152697130678e-9,
          4.075755403521244e-9, 0.44611179222824576, 1.5649564055064537e-9,
          0.007667989378548945, 1.1957387997008553e-8, 4.986819648281179e-9,
          3.450262919920553e-9, 6.190904157193227e-9, 2.6745863789282485e-9,
          1.0441887652822887e-9, 3.3894565384961806e-9, 1.3213991289732224e-9,
          0.3915895683852205, 0.07522685990821956, 1.3961718881471554e-8,
          1.1609524030505546e-8, 1.624477113954877e-8]
    @test isapprox(w9.weights, wt, rtol = 1.0e-5)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [6.349688730043195e-9, 6.71976111028578e-9, 8.596739423206003e-9,
          7.754992982922198e-9, 7.877744599183851e-7, 3.155665434221977e-9,
          0.9999991072975217, 4.689062890653117e-9, 7.198557674294465e-9,
          5.12679755078563e-9, 4.590755769590718e-9, 3.313424974155189e-9,
          2.513037392410736e-9, 3.917915725822589e-9, 2.580526408074827e-9,
          1.1101066522761444e-8, 8.941249329934282e-9, 4.941832479855464e-9,
          7.450502394466141e-9, 5.986441524548975e-9]
    riskt = 0.009248238571500755
    rett = 0.001845375476375911
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [4.059953701209771e-11, 5.3854869692668857e-11, 8.762056560115626e-11,
          7.379226762543566e-11, 0.8503242042351197, 2.412506182196728e-11,
          0.14967579482348262, 7.1112938835614905e-12, 6.831459725868651e-11,
          1.9081680600373163e-11, 4.3437824460830014e-12, 3.329980889382994e-11,
          4.7734615825477856e-11, 1.2640340301477146e-11, 4.1470563350009406e-11,
          1.874537086069124e-10, 1.0938233352577957e-10, 1.3966473217358029e-11,
          7.808506951083118e-11, 3.852121723218711e-11]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.545071553812398e-9, 2.7546047875565345e-9, 3.4242775386738267e-9,
          3.0760030823433574e-9, 0.8533950804771374, 1.180272815540777e-9,
          0.14660487673003714, 1.8493414965506873e-9, 2.937030243701554e-9,
          1.9460990614339756e-9, 1.7441482041989316e-9, 1.1249336323424014e-9,
          8.210728734888476e-10, 1.448671395348878e-9, 8.282943332945012e-10,
          5.878673483363418e-9, 3.794035048793611e-9, 1.966817335795921e-9,
          3.1328865812486705e-9, 2.340591854075819e-9]
    @test isapprox(w12.weights, wt)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "SSkew" begin
    portfolio = Portfolio2(; prices = prices,
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = SSkew2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [7.610819221245223e-7, 2.0842051720782308e-5, 6.71139244607462e-7,
          8.839948442444747e-7, 2.1807270401133766e-6, 1.0842040956915447e-6,
          2.7830339110192037e-7, 0.1280975813408656, 4.697915971310934e-7,
          7.433008049916792e-7, 0.4996629306075607, 1.043320338349008e-6,
          4.5083144725397534e-7, 0.026586616541717394, 2.6481651179988687e-5,
          0.013510133781810273, 3.2563765357090695e-6, 0.21130849297469684,
          6.622336342182562e-7, 0.1207744357455529]
    riskt = 0.0033523757385970935
    rett = 0.0003452673005217105
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [8.085684277649453e-7, 2.2044416218536275e-5, 7.122197550551237e-7,
          9.365150288739066e-7, 2.3112397057071937e-6, 1.1463876948482304e-6,
          2.897889224018354e-7, 0.12809804062532298, 4.980452459681117e-7,
          7.871575053853804e-7, 0.4996624292271971, 1.1026872071242442e-6,
          4.722205533100804e-7, 0.026584701175178808, 2.783163741902343e-5,
          0.013509903121771059, 3.4423793308795104e-6, 0.21130846849590879,
          7.02783519364156e-7, 0.120773371308087]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [6.848486564371291e-7, 1.3006523492807844e-5, 5.542288571477643e-7,
          7.129829869541138e-7, 1.6479004564845006e-6, 1.0557023195365765e-6,
          2.2246492852752627e-7, 0.12809386822972763, 3.864152131167562e-7,
          6.122990560860172e-7, 0.49966341197142455, 8.502095282205388e-7,
          3.827355480269704e-7, 0.026605678473279253, 1.337984086772962e-5,
          0.013512554143560805, 2.40914743952179e-6, 0.2113068730150808,
          5.44615290757205e-7, 0.1207811642522856]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [3.526947226679415e-11, 4.965403225658518e-11, 9.401426988461822e-11,
          7.144889533157678e-11, 0.8964323505408012, 5.524485694073918e-11,
          0.10356764846871296, 1.261655117472966e-11, 5.570878474427449e-11,
          2.2978432744231737e-12, 9.407994882493524e-12, 4.874361499053944e-11,
          7.080991600332863e-11, 3.428430407494229e-11, 5.4028680780540745e-11,
          1.8873092166991273e-10, 1.1157831669291034e-10, 1.722894627553467e-12,
          6.759785272314914e-11, 2.7326630775060994e-11]
    riskt = 0.006357305722809161
    rett = 0.0018007663766188612
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.7449521008601392e-10, 2.7137878752656457e-10, 4.0711564560695016e-10,
          3.3086060550667526e-10, 0.8832583296126295, 1.309657405691995e-10,
          0.11674166549176974, 4.9133637550014634e-11, 3.1826051767838835e-10,
          8.706532804923367e-11, 3.527633052872797e-11, 1.2600252515254098e-10,
          2.0795615815009207e-10, 5.715515014286958e-11, 2.131871999431445e-10,
          1.2634451642373557e-9, 5.612504144293207e-10, 7.973698061643502e-11,
          3.908903635493841e-10, 1.9142493886032221e-10]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.963067787110398e-9, 3.3288394223204142e-9, 3.942526944023086e-9,
          3.5131273724559455e-9, 0.8863760425157332, 1.2556639631095496e-9,
          0.1136239059445167, 2.230625619729753e-9, 3.4864207152718658e-9,
          2.2553650449903613e-9, 2.0556083005625332e-9, 1.1170405248753545e-9,
          6.817844532528834e-10, 1.6169902374891547e-9, 6.940730235854863e-10,
          8.734864148724232e-9, 4.6836300598640795e-9, 2.362035047463814e-9,
          3.842565346341087e-9, 2.775522155282069e-9]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [3.091279504383203e-10, 5.456174857781897e-10, 3.9798555148028387e-10,
          2.908765475207534e-10, 0.8086224910235427, 1.1178427214665011e-10,
          0.00435469374000496, 5.819507109567311e-10, 3.3752308100189647e-10,
          2.721974270172124e-10, 5.151250328225061e-10, 1.1452836055865946e-10,
          8.167541473284592e-11, 2.0913309558573752e-10, 8.113773847394304e-11,
          0.18702280875990515, 1.092428214335467e-9, 4.864627001832568e-10,
          5.959336285465247e-10, 4.530598274276326e-10]
    riskt = 0.005659517059855914
    rett = 0.001675925645653581
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [2.115637360333395e-8, 4.943128142626069e-8, 2.8824476289201033e-8,
          1.9560049484906765e-8, 0.7615164336573722, 7.869881998305917e-9,
          0.01798316049178013, 9.179605861876573e-8, 2.3905629310735416e-8,
          1.8826515729745752e-8, 6.873632811624254e-8, 6.839870619279722e-9,
          4.837675215317091e-9, 1.501226822940772e-8, 5.096087482118037e-9,
          0.22049960433895407, 2.839568019802063e-7, 5.577507196173848e-8,
          6.046224057226691e-8, 3.942528280601827e-8]
    @test isapprox(w8.weights, wt)

    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [4.569993865266908e-9, 9.332121077658172e-9, 6.2670528516812416e-9,
          4.5065263685530315e-9, 0.7730558190269607, 1.7650304054020667e-9,
          0.020609096522009613, 1.2370503878644851e-8, 5.3658740316800276e-9,
          4.1100044559615e-9, 1.024686041491922e-8, 1.5702284114321352e-9,
          1.1213764350405433e-9, 3.2080872900254548e-9, 1.1637795482051973e-9,
          0.20633496120700334, 2.9352426261083927e-8, 9.329752409789233e-9,
          1.1364488563015936e-8, 7.599919967518768e-9]
    @test isapprox(w9.weights, wt, rtol = 1.0e-5)

    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [4.780168272776335e-9, 5.081624025669024e-9, 6.483732034604582e-9,
          5.902583356891863e-9, 6.020152212638221e-7, 2.367923116315686e-9,
          0.9999993188984393, 3.49745509723335e-9, 5.409340608390766e-9,
          3.860522703075498e-9, 3.4113938301892213e-9, 2.4919961567894646e-9,
          1.872626527112011e-9, 2.9205972291974916e-9, 1.926474965678272e-9,
          8.537390005950894e-9, 6.758217079193345e-9, 3.6864743538101396e-9,
          5.59084080436368e-9, 4.506979158597684e-9]
    riskt = 0.013290076423637484
    rett = 0.0018453755187909182
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.458744096420407e-10, 2.0701558604651032e-10, 3.4745478858082955e-10,
          2.918291043920079e-10, 0.8503238077903578, 1.1459396747732204e-10,
          0.14967618853493264, 8.76210972538846e-12, 2.6054096669079653e-10,
          6.162340135886716e-11, 3.919450488801854e-12, 1.0908947835885885e-10,
          1.834887101082446e-10, 6.066255065369385e-11, 1.8900908914622592e-10,
          7.772370717785006e-10, 4.3260272508600003e-10, 4.126266850166177e-11,
          2.9932376070772586e-10, 1.404199148590055e-10]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.6437612429698908e-9, 2.8421675308077488e-9, 3.5243602538395847e-9,
          3.174885216504512e-9, 0.8533950713694052, 1.2212532037664942e-9,
          0.14660488446854544, 1.906837595143931e-9, 3.029256579335313e-9,
          2.0065742351477204e-9, 1.7974285000088872e-9, 1.1620834091061886e-9,
          8.446377946494509e-10, 1.4970747892711415e-9, 8.531370606590083e-10,
          6.075556080219687e-9, 3.912301966114269e-9, 2.027436910675667e-9,
          3.233683614445602e-9, 2.409613319648957e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-9

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    obj = SR(; rf = rf)
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end

@testset "DVar" begin
    portfolio = Portfolio2(; prices = prices[(end - 50):end],
                           solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                            :params => Dict("verbose" => false,
                                                                            "max_step_fraction" => 0.75))))
    asset_statistics2!(portfolio)
    rm = DVar2()

    obj = MinRisk()
    w1 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r1 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret1 = dot(portfolio.mu, w1.weights)
    wt = [1.6599419520522404e-9, 5.401177113683357e-9, 3.599249264668155e-9,
          1.058226430481034e-8, 3.542026862259862e-8, 5.636730209799928e-8,
          1.07713614651093e-8, 0.03490173549678457, 2.4548865558471906e-9,
          3.133681624832556e-9, 0.31629477278954593, 1.2070573543163117e-9,
          5.256736734453393e-10, 0.07100378778165789, 1.5310130384238334e-9,
          0.0010049805257420585, 6.842541500077776e-9, 4.033774680945785e-8,
          4.013313333846196e-9, 0.5767945395587909]
    riskt = 0.00044380903384026233
    rett = 4.271280183767122e-5
    @test isapprox(w1.weights, wt)
    @test isapprox(r1, riskt)
    @test isapprox(ret1, rett)

    w2 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.6787818359100834e-9, 3.912629594307745e-9, 3.160368058232141e-9,
          8.215692515113953e-9, 2.0298393479301478e-8, 2.8947288835166952e-8,
          8.471542038197599e-9, 0.03490214296601228, 2.2643649100962656e-9,
          2.6720045805244234e-9, 0.31629432718469014, 1.2690337208418477e-9,
          6.291049045691047e-10, 0.07100406325110667, 1.5981756512013748e-9,
          0.0010047516543738734, 5.0514884804146885e-9, 2.0184849034905844e-8,
          3.3106423251525213e-9, 0.5767946032794571]
    @test isapprox(w2.weights, wt)

    w3 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [4.1450486273580134e-9, 9.324978738437985e-9, 7.388430101617496e-9,
          1.5820212421415435e-8, 3.198176908320699e-8, 5.098691423018805e-8,
          1.7028864783234485e-8, 0.03490212037832912, 5.582318578159504e-9,
          6.7569968919626e-9, 0.3162946392019487, 2.8195001214235705e-9,
          1.1064590252280676e-9, 0.07100393015971117, 3.4733358937753346e-9,
          0.0010044525297497812, 1.1243125564724572e-8, 3.524397951087502e-8,
          7.934739930930933e-9, 0.5767946468935877]
    @test isapprox(w3.weights, wt)

    obj = Util(; l = l)
    w4 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r2 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret2 = dot(portfolio.mu, w4.weights)
    wt = [9.223361953845353e-10, 2.4407100887351822e-9, 6.307223673485028e-10,
          6.295156904274784e-10, 1.1277207180021208e-9, 4.977370224886328e-10,
          8.591052170393522e-10, 5.827915013054275e-10, 7.835899932699336e-10,
          6.131084772620629e-10, 2.8641604144729723e-9, 0.08806267952458437,
          0.08378318844152127, 4.2089232860432785e-10, 8.226638552315439e-10,
          6.720359395190258e-10, 1.3420855042508736e-8, 4.944621418604599e-10,
          7.142385199607298e-10, 0.8281541035372488]
    riskt = 0.0007733107354884844
    rett = 0.0016932199573166952
    @test isapprox(w4.weights, wt)
    @test isapprox(r2, riskt)
    @test isapprox(ret2, rett)

    w5 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.347275259850256e-10, 1.0863438649217123e-9, 3.352145840096695e-10,
          3.822248996663145e-10, 4.6310478488656014e-10, 3.190830523950665e-10,
          5.724722014031112e-10, 4.1615484355259076e-10, 4.867220211003578e-10,
          3.489059235106311e-10, 1.5550720467887437e-9, 0.077040781436085,
          0.07386841997050478, 1.825848991116497e-10, 4.284559640982416e-10,
          3.141526236738139e-10, 7.767454487760839e-9, 2.386117334416081e-10,
          3.5675278005991524e-10, 0.8490907826053722]
    @test isapprox(w5.weights, wt)

    w6 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [2.3041181012014065e-10, 1.6909841784268424e-9, 3.443820505122832e-10,
          2.495821441437348e-10, 9.134743851071586e-10, 2.0024170051832926e-10,
          1.4407653932148083e-10, 1.7872296016831914e-10, 3.745481021755366e-10,
          3.4833613162912217e-10, 1.3340036404747291e-9, 0.07736745043455064,
          0.07393385111916854, 3.8938687640205405e-10, 5.069962748293395e-10,
          5.07142640825486e-10, 7.0425476937640065e-9, 4.114487894308365e-10,
          5.11261763780151e-10, 0.8486986830687332]
    @test isapprox(w6.weights, wt, rtol = 5.0e-7)

    obj = SR(; rf = rf)
    w7 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r3 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret3 = dot(portfolio.mu, w7.weights)
    wt = [1.9786762822432265e-12, 9.263422432070166e-12, 2.6636586669065547e-12,
          2.386057574396552e-12, 9.429481906466297e-12, 1.6508810716775093e-12,
          1.2502531952200594e-12, 7.919482593203544e-13, 2.974402756754508e-12,
          2.7649020509809316e-12, 3.744309029297335e-12, 0.48410478175962135,
          0.49717956285476883, 2.5711647004387183e-12, 4.105879754986053e-12,
          2.939952653821243e-12, 1.0489987457836851e-11, 2.8379239254053576e-12,
          3.442694357800877e-12, 0.018715655320324244]
    riskt = 0.004810699356344494
    rett = 0.004941785644668024
    @test isapprox(w7.weights, wt)
    @test isapprox(r3, riskt)
    @test isapprox(ret3, rett)

    w8 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [7.554617701239649e-10, 5.49136147499361e-9, 1.0189476122811958e-9,
          8.418693863163602e-10, 4.61707734965482e-9, 5.531174113787105e-10,
          4.3386775456002837e-10, 4.967828155441755e-10, 1.1641366230404278e-9,
          1.0474934427128516e-9, 1.7641013771274661e-9, 0.2409762038799364,
          0.25881763615937337, 9.537971702020672e-10, 1.8268099244681386e-9,
          1.202220043666255e-9, 8.794395864095084e-9, 1.1017106693079265e-9,
          1.4889193060299604e-9, 0.5002061264086202]
    @test isapprox(w8.weights, wt)

    portfolio.solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                               :params => Dict("verbose" => false,
                                                               "max_step_fraction" => 0.99)))
    w9 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [1.4043980975143159e-9, 1.0689935206604866e-8, 1.9130097739429034e-9,
          1.5805454578154989e-9, 9.052428769807145e-9, 1.022848799461252e-9,
          7.947731812912791e-10, 9.170550288872991e-10, 2.1811772346839784e-9,
          1.9689685903879376e-9, 3.352584496992465e-9, 0.2765068723480663,
          0.29120551080963963, 1.7896176632183133e-9, 3.4535297103975665e-9,
          2.270647987905937e-9, 1.6394700402651242e-8, 2.0797299050677884e-9,
          2.8138854207631618e-9, 0.43228755316245826]
    @test isapprox(w9.weights, wt, rtol = 5.0e-5)

    portfolio.solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                               :params => Dict("verbose" => false,
                                                               "max_step_fraction" => 0.75)))
    obj = MaxRet()
    w10 = optimise2!(portfolio; rm = rm, kelly = NoKelly(), obj = obj)
    r4 = calc_risk(portfolio; type = :Trad2, rm = rm)
    ret4 = dot(portfolio.mu, w10.weights)
    wt = [1.595081606192407e-8, 2.9382334431414644e-8, 1.7658467922714496e-8,
          1.6582896735815844e-8, 2.7054675489076104e-8, 1.4110650817598288e-8,
          1.2211197645107672e-8, 1.3699678443896811e-8, 1.9274099938707518e-8,
          1.861445082317701e-8, 2.1964968347874875e-8, 9.698462231820156e-8,
          0.999999532231261, 1.7731042737188638e-8, 2.6080622753652606e-8,
          2.0173338470245485e-8, 2.7700515356613562e-8, 1.9503708210601273e-8,
          2.1275426991479796e-8, 3.1815225407544205e-8]
    riskt = 0.009987319852162961
    rett = 0.006507502990668492
    @test isapprox(w10.weights, wt)
    @test isapprox(r4, riskt)
    @test isapprox(ret4, rett)

    w11 = optimise2!(portfolio; rm = rm, kelly = AKelly(), obj = obj)
    wt = [1.4505774389724766e-9, 6.001098683290737e-9, 2.0616923853672764e-9,
          1.7050834107556396e-9, 5.355623947260384e-9, 4.3310982362230366e-10,
          4.71749896451958e-10, 5.662910381976156e-10, 2.6020835005189653e-9,
          2.320866683830512e-9, 3.495396369289018e-9, 3.3924485186370256e-8,
          0.9999999122036234, 1.4577471983063948e-9, 4.396567090441705e-9,
          2.808571429124868e-9, 5.907777897474621e-9, 2.4461248981121813e-9,
          3.292930249731605e-9, 7.098599435173618e-9]
    @test isapprox(w11.weights, wt)

    w12 = optimise2!(portfolio; rm = rm, kelly = EKelly(), obj = obj)
    wt = [3.7169130297180648e-9, 7.942145548223315e-9, 4.121538497933915e-9,
          3.825879632370855e-9, 7.411250367960804e-9, 2.9483762146540775e-9,
          2.539299092536898e-9, 2.777208627750765e-9, 4.6923783548998645e-9,
          4.3682174592379056e-9, 5.188547168253687e-9, 4.033774893120325e-8,
          0.9999998686858115, 3.92894752776021e-9, 6.420816665579685e-9,
          4.696699450020883e-9, 7.840288343064783e-9, 4.44303733851266e-9,
          5.211112744769251e-9, 8.903783558490881e-9]
    @test isapprox(w12.weights, wt, rtol = 1.0e-7)

    # Risk upper bound
    obj = MaxRet()
    rm.settings.ub = r1
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
          abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    rm.settings.ub = r2
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    rm.settings.ub = r3
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    rm.settings.ub = r4
    optimise2!(portfolio; rm = rm, obj = obj)
    @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # obj = SR(; rf = rf)
    # rm.settings.ub = r1*1.5
    # optimise2!(portfolio; rm = rm, obj = obj)
    # @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r1 ||
    #       abs(calc_risk(portfolio; type = :Trad2, rm = rm) - r1) < 1e-10

    # rm.settings.ub = r2
    # optimise2!(portfolio; rm = rm, obj = obj)
    # calc_risk(portfolio; type = :Trad2, rm = rm)
    # @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r2

    # rm.settings.ub = r3
    # optimise2!(portfolio; rm = rm, obj = obj)
    # @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r3

    # rm.settings.ub = r4
    # optimise2!(portfolio; rm = rm, obj = obj)
    # @test calc_risk(portfolio; type = :Trad2, rm = rm) <= r4

    # Ret lower bound
    rm.settings.ub = Inf
    obj = MinRisk()
    portfolio.mu_l = ret1
    w13 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w13.weights) >= ret1

    portfolio.mu_l = ret2
    w14 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w14.weights) >= ret2

    portfolio.mu_l = ret3
    w15 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w15.weights) >= ret3

    portfolio.mu_l = ret4
    w16 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w16.weights) >= ret4

    obj = SR(; rf = rf)
    portfolio.mu_l = ret1
    w17 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w17.weights) >= ret1

    portfolio.mu_l = ret2
    w18 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w18.weights) >= ret2

    portfolio.mu_l = ret3
    w19 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w19.weights) >= ret3

    portfolio.mu_l = ret4
    w20 = optimise2!(portfolio; rm = rm, obj = obj)
    @test dot(portfolio.mu, w20.weights) >= ret4
end