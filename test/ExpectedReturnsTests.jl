using Test
using PortfolioOptimiser, CSV, DataFrames, Statistics, CovarianceEstimation

@testset "Expected returns" begin
    df = CSV.read("./assets/stock_prices.csv", DataFrame)
    dropmissing!(df)
    returns = returns_from_prices(df[!, 2:end])

    # Mean returns.
    mu = ret_model(MRet(), Matrix(returns))
    mutest = [
        0.165077005733385
        0.183572017421321
        0.238036763838874
        0.192316447138601
        0.508570682802054
        -0.153295057043960
        0.305495623268028
        0.059318615800612
        0.189931534062078
        0.084295329365638
        0.054465665833134
        -0.182249168372897
        -0.439434885859736
        -0.028695410193010
        -0.353755082717798
        0.256437396469586
        0.262994437027811
        0.084965198887185
        0.214010363955894
        0.154258862256590
    ]
    @test mu ≈ mutest

    mu = ret_model(MRet(), Matrix(returns); compound = false)
    mutest = [
        0.178914374187325
        0.195208843702195
        0.245787816971219
        0.226167230238702
        0.452494357193942
        -0.143808069153082
        0.465034663597819
        0.078266717104239
        0.208793660771767
        0.111571083023889
        0.066312449760358
        -0.109757830255107
        -0.338454468486514
        -0.011028146100837
        -0.311429033491868
        0.290932770652577
        0.253991539266203
        0.096869664526551
        0.217271589766479
        0.163468530861212
    ]
    @test mu ≈ mutest

    # Exponential returns.
    mu = ret_model(EMRet(), Matrix(returns); span = 500)
    mutest = [
        0.174206788637514
        0.261448055846544
        0.182275222944877
        0.388168987273908
        0.602214112048088
        -0.405425982176985
        0.309416246725018
        0.119263386354215
        0.351783576012055
        0.140163506583907
        0.010482845369419
        -0.119858604418692
        -0.396957355595715
        -0.012867616138760
        -0.290725396632509
        0.513672821109176
        0.407679425007418
        0.098478138659613
        0.319774016932130
        0.099739436963900
    ]
    @test mu ≈ mutest

    mu = ret_model(EMRet(), Matrix(returns); compound = false, span = 500)
    mutest = [
        0.160644027761459
        0.232367378080205
        0.167496377111932
        0.328199137640111
        0.471827651094508
        -0.519374112543301
        0.269725670874286
        0.112695969837963
        0.301605231542693
        0.131205823781312
        0.010428497113714
        -0.127640371175951
        -0.505260162800969
        -0.012950788228994
        -0.343278494299677
        0.414880173944252
        0.342174649145516
        0.093943217781961
        0.277613325432957
        0.095091212893796
    ]
    @test mu ≈ mutest

    # CAPM rets.
    mu = ret_model(CAPMRet(), Matrix(returns); fix_method = DFix())
    mutest = [
        0.094883850444685
        0.095679415838370
        0.103105763584424
        0.113528802174820
        0.105209157372728
        0.089111865352351
        0.191417900856481
        0.066415817812618
        0.122436002682103
        0.104694131070045
        0.062870019348941
        0.144651558003671
        0.197305954089759
        0.082330006545561
        0.144756910925415
        0.113935315988995
        0.096108833370246
        0.071786198182351
        0.107079483654873
        0.084533187625037
    ]
    @test mu ≈ mutest

    mu = ret_model(CAPMRet(), Matrix(returns); compound = false)
    mutest = [
        0.100933917383660
        0.101793758637327
        0.109820100917702
        0.121085244725778
        0.112093433526496
        0.094695598148891
        0.205267212404455
        0.070165876111249
        0.130712081830110
        0.111536796868735
        0.066333603088240
        0.154722491395735
        0.211630977236295
        0.087365814796178
        0.154836356062335
        0.121524601880310
        0.102257869962287
        0.075970143901725
        0.114114868480771
        0.089746996778792
    ]
    @test mu ≈ mutest

    mu = ret_model(CAPMRet(), Matrix(returns); compound = false, fix_method = FFix())
    mutest = [
        0.1009339173836596,
        0.10179375863732709,
        0.10982010091770186,
        0.12108524472577818,
        0.11209343352649556,
        0.0946955981488904,
        0.20526721240445442,
        0.07016587611124943,
        0.13071208183010968,
        0.11153679686873515,
        0.06633360308823946,
        0.1547224913957347,
        0.21163097723629473,
        0.08736581479617836,
        0.15483635606233484,
        0.12152460188031008,
        0.10225786996228645,
        0.07597014390172492,
        0.11411486848077082,
        0.08974699677879162,
    ]
    @test mu ≈ mutest

    mu = ret_model(ECAPMRet(), Matrix(returns); rspan = 500, cspan = 500)
    mutest = [
        0.09139391988704085
        0.08583852838696727
        0.09499477329572159
        0.10572143376970136
        0.09067107452064896
        0.08302369806022829
        0.15867778967706206
        0.06801860321851717
        0.10549276014308034
        0.09614499992006319
        0.06136800987467894
        0.14198028714125177
        0.1979620159376844
        0.06945672048624046
        0.1351920895772175
        0.09845786650669847
        0.08355249256001421
        0.06688710785255922
        0.09247933409174176
        0.06700270256064539
    ]
    @test mu ≈ mutest

    mu = ret_model(
        ECAPMRet(),
        Matrix(returns);
        compound = false,
        rspan = 500,
        cspan = 500,
        fix_method = DFix(),
    )
    mutest = [
        0.08723285636839716
        0.08200125066043745
        0.0906238406484665
        0.10072531741428492
        0.08655214071681114
        0.07935047755036757
        0.15059520936231863
        0.06521992710737305
        0.10050997161007391
        0.09170702842619073
        0.05895695138397315
        0.13487089009987602
        0.18758982663369017
        0.06657422635098838
        0.12847833016556906
        0.09388509383098356
        0.07984845223376856
        0.06415437887103413
        0.08825500919934322
        0.0642632363538282
    ]
    @test mu ≈ mutest

    market_returns = vec(mean(Matrix(returns), dims = 2))
    mu = ret_model(CAPMRet(), Matrix(returns), market_returns)
    mu2 = ret_model(CAPMRet(), Matrix(returns))
    @test mu ≈ mu2

    mu = ret_model(ECAPMRet(), Matrix(returns), market_returns)
    mu2 = ret_model(ECAPMRet(), Matrix(returns))
    @test mu ≈ mu2

    returns = returns_from_prices(df[!, 2:end])
    log_returns = returns_from_prices(df[!, 2:end], true)
    @test exp.(log_returns) .- 1 ≈ returns

    rel_prices = prices_from_returns(Matrix(returns[!, :]))
    rel_prices_log = prices_from_returns(Matrix(log_returns[!, :]), true)
    @test rel_prices ≈ rel_prices_log

    reconstructed_prices = (rel_prices' .* Vector(df[1, 2:end]))'

    @test reconstructed_prices ≈ Matrix(df[!, 2:end])

    target = DiagonalUnequalVariance()
    shrinkage = :lw
    method = LinearShrinkage(target, shrinkage)

    capm_ret = ret_model(
        CAPMRet(),
        Matrix(returns),
        cov_type = CustomCov(),
        custom_cov_estimator = method,
    )

    @test capm_ret ≈ [
        0.09278178288462897,
        0.09355501593033733,
        0.10077289837186976,
        0.11090335142923119,
        0.10281724770903872,
        0.08717182341661432,
        0.18660602210741187,
        0.06511287753486789,
        0.11956051755492042,
        0.10231667873029156,
        0.0616666133216439,
        0.14115245913476981,
        0.19232879156294488,
        0.08058033843956112,
        0.1412548546904358,
        0.11129845397787426,
        0.09397237926550572,
        0.07033250574250885,
        0.10463507198733277,
        0.0827216739348617,
    ]

    exp_capm_ret = ret_model(
        ECAPMRet(),
        Matrix(returns),
        cspan = 1503,
        rspan = 1503,
        cov_type = CustomCov(),
        custom_cov_estimator = method,
    )
    @test exp_capm_ret ≈ [
        0.10295512054015153,
        0.10383643503905633,
        0.1120632231754654,
        0.123609697048356,
        0.11439332885534213,
        0.0965610086422304,
        0.20989398306639132,
        0.07141869360015936,
        0.13347694996562873,
        0.11382279103221057,
        0.06749071530819609,
        0.1580869834858518,
        0.2164166733769566,
        0.08904817494761193,
        0.1582036917516621,
        0.12406002648862763,
        0.10431213684802158,
        0.0773679142701998,
        0.11646524611471647,
        0.09148882337760635,
    ]

    target = DiagonalCommonVariance()
    shrinkage = :oas
    method = LinearShrinkage(target, shrinkage)

    mean_ret = ret_model(
        CAPMRet(),
        Matrix(returns),
        cov_type = CustomSCov(),
        custom_cov_estimator = method,
    )
    @test mean_ret ≈ [
        0.10148825983799059,
        0.0995907610147829,
        0.11259682557437534,
        0.1285861753330848,
        0.10936446296833713,
        0.10300894526667283,
        0.20364838848438546,
        0.07624793630742041,
        0.13029045433786315,
        0.11854621479103636,
        0.07102859418763925,
        0.15138698756874633,
        0.20443622850011367,
        0.08963630056470558,
        0.16845735950593438,
        0.11779599802293746,
        0.10057857564887862,
        0.07648833021864435,
        0.11105103412590665,
        0.08906783217752086,
    ]

    mean_ret = ret_model(
        ECAPMRet(),
        Matrix(returns),
        cspan = 447.5,
        rspan = 447.5,
        cov_type = CustomSCov(),
        custom_cov_estimator = method,
    )

    @test mean_ret ≈ [
        0.08777015613036471,
        0.08619209087579147,
        0.09700865546555652,
        0.11030628547661692,
        0.09432044345910705,
        0.08903484246839495,
        0.17273218480086244,
        0.06677890328181402,
        0.1117236584144261,
        0.10195649748470165,
        0.06243820891604167,
        0.12926870543971347,
        0.17338739625068111,
        0.07791340950210028,
        0.14346537344669538,
        0.10133257561415537,
        0.08701361230861653,
        0.06697882819036924,
        0.09572308956911968,
        0.07744063966468716,
    ]
end