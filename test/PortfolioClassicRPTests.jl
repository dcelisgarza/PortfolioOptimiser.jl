using COSMO,
    CSV,
    Clarabel,
    HiGHS,
    LinearAlgebra,
    OrderedCollections,
    PortfolioOptimiser,
    Statistics,
    Test,
    TimeSeries,
    SCS

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "$(:Classic), $(:RP), $(:SD)" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SD)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :SD)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SD)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :SD)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.05063127121116526,
        0.051246073451127395,
        0.04690051417238177,
        0.043692251262490364,
        0.045714486491116986,
        0.05615245606945022,
        0.027633827630176938,
        0.0770545437085184,
        0.03949812820650735,
        0.04723372192739208,
        0.08435518551109286,
        0.033857725965738224,
        0.02754549658244145,
        0.0620682222288943,
        0.03563694530691492,
        0.044132294780985376,
        0.05085457494779759,
        0.0714243017058742,
        0.04529034792706349,
        0.05907763091287083,
    ]

    w2t = [
        0.005639748872225331,
        0.011009072981464817,
        0.015583334801842201,
        0.019370218897423953,
        0.02543776325220892,
        0.03265889172407469,
        0.02036970258263696,
        0.05957032877760121,
        0.03298337846885652,
        0.04482975658758464,
        0.08740735380673535,
        0.038228919621536385,
        0.03143705159669208,
        0.07961157967860712,
        0.04614083596496523,
        0.06097834264239432,
        0.0833005053270252,
        0.11639740018782287,
        0.0792776306508924,
        0.10976818357740997,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 1e-3)
    @test isapprox(hrc2 / lrc2, 20, atol = 1e-2)
end

@testset "$(:Classic), $(:RP), $(:MAD)" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :MAD)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :MAD)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :MAD)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :MAD)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.05616723822258191,
        0.05104350090571541,
        0.05044213212777734,
        0.04081451676582401,
        0.04879048873980569,
        0.056711887975158415,
        0.0242252580652895,
        0.07090406077266315,
        0.040195302909630776,
        0.04648924516109405,
        0.08138369549236708,
        0.031588537618890015,
        0.023961578902842652,
        0.06225067796284232,
        0.03459319058171123,
        0.04206380992172297,
        0.05465474067537367,
        0.07239056092704055,
        0.04770679646373056,
        0.0636227798079387,
    ]

    w2t = [
        0.006401779618828623,
        0.012593951743256573,
        0.017340890783595773,
        0.019134849456476876,
        0.028374307245312213,
        0.03189651794987627,
        0.019206796668855844,
        0.055391316944302715,
        0.03338502811307715,
        0.045151356177118636,
        0.0807580519059978,
        0.03494263054680027,
        0.02578410334386403,
        0.07994586838260086,
        0.044758985906496206,
        0.05591483590172484,
        0.09228051691281969,
        0.12028285344497854,
        0.08191841248798935,
        0.11453694646602786,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-6)
    @test isapprox(hrc1 / lrc1, 1, atol = 1e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 1e-1)
end

@testset "$(:Classic), $(:RP), $(:SSD)" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SSD)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :SSD)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SSD)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :SSD)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.050541924842960946,
        0.052706459151258836,
        0.045042850076469895,
        0.04045020411368036,
        0.045695841041789254,
        0.05416091693414071,
        0.02650035819634097,
        0.0777682856720521,
        0.03824369856805568,
        0.04454331811452701,
        0.08433994049885823,
        0.03546426883470469,
        0.0299297498847143,
        0.06261801736938656,
        0.03923833096664213,
        0.04564914223901187,
        0.049847364656398614,
        0.07412562156298433,
        0.044278287070396695,
        0.05885542020562677,
    ]

    w2t = [
        0.005429547829098365,
        0.011167805664414806,
        0.014459305565898126,
        0.017042626615556288,
        0.024075080878288097,
        0.031344901036748814,
        0.019078016259587233,
        0.059195606446996535,
        0.031947654907047304,
        0.04196068100639579,
        0.08814196888898153,
        0.03899899306400899,
        0.03418229998947243,
        0.07927269058844665,
        0.05025713223285367,
        0.06338856522866418,
        0.08186617870438147,
        0.12155493479073504,
        0.07827458257785366,
        0.10836142772457111,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 1e-3)
    @test isapprox(hrc2 / lrc2, 20, atol = 1e-1)
end

@testset "$(:Classic), $(:RP), $(:FLPM)" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :FLPM)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :FLPM)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :FLPM)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :FLPM)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.05426727063306951,
        0.050966069892303255,
        0.05094730208792428,
        0.040027997647476436,
        0.05569009428444262,
        0.048542563630443555,
        0.02465810993843965,
        0.07063071421200735,
        0.04178928035410628,
        0.04606158063290862,
        0.0826513558447341,
        0.02798793051735644,
        0.021903992548207803,
        0.0591013494881841,
        0.030132543527577153,
        0.045271860898623756,
        0.057932891294237246,
        0.07483274762172051,
        0.04995897456440289,
        0.06664537038183452,
    ]

    w2t = [
        0.006884077432257186,
        0.012954874743778584,
        0.018385799362110936,
        0.01831964358547592,
        0.0360768453411326,
        0.027200592181016465,
        0.02021354469214129,
        0.05276903846646005,
        0.03376615707481203,
        0.0442388899755955,
        0.08234210706887593,
        0.031754949162667905,
        0.024158856974061862,
        0.07390417921315301,
        0.03807087872319826,
        0.05745974916872209,
        0.09954209209763215,
        0.12219715695811513,
        0.0848210415751922,
        0.11493952620360075,
    ]
    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-6)
    @test isapprox(hrc1 / lrc1, 1, atol = 1e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 1e-2)
end

@testset "$(:Classic), $(:RP), $(:SLPM)" begin
    portfolio = Portfolio(
        prices = prices,
        solvers = OrderedDict(
            :Clarabel => Dict(
                :solver => Clarabel.Optimizer,
                :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
            ),
            :COSMO =>
                Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SLPM)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :SLPM)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SLPM)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :SLPM)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.05078187265278322,
        0.05371206006859483,
        0.04576409186702134,
        0.0409205041430174,
        0.04906039669475073,
        0.050350260279864986,
        0.027362427094292905,
        0.07791305167422613,
        0.03854034857251276,
        0.04420141115104938,
        0.08430367098357294,
        0.03395624039861946,
        0.028242664791423695,
        0.06056868051162136,
        0.03650685331711261,
        0.0475715771694708,
        0.05128453593861015,
        0.07441875218075804,
        0.04500417145785451,
        0.05953642905284273,
    ]

    w2t = [
        0.005492527475806188,
        0.011429816988193184,
        0.014822086266724446,
        0.017306287129466233,
        0.026322726877075143,
        0.02905990211343986,
        0.019770052943997973,
        0.0592124634100771,
        0.032193671946941074,
        0.04165106177764175,
        0.08829033644315885,
        0.037309763647838966,
        0.03234339539756486,
        0.07666976289341881,
        0.04703097020934251,
        0.06578728165840875,
        0.08445214229212933,
        0.12165795866775381,
        0.07958227554108589,
        0.10961551631993534,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 1e-3)
    @test isapprox(hrc2 / lrc2, 20, atol = 1e-1)
end
