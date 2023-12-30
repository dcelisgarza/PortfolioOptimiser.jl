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

@testset "$(:Classic), $(:RP), $(:WR)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :WR)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :WR)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :WR)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :WR)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.04375544631866262,
        0.0822719874989624,
        0.04366290510073906,
        0.05731679279782298,
        0.055186348442627035,
        0.048877199550522325,
        0.03646223108130178,
        0.04988484801087561,
        0.03936348977638506,
        0.04856611900494611,
        0.05738895313337345,
        0.03890988533373517,
        0.04249681666902821,
        0.03781821055920519,
        0.04825301800636387,
        0.04988638272418726,
        0.060421157334263315,
        0.04103738762715828,
        0.04176168136676482,
        0.07667913966307555,
    ]

    w2t = [
        0.004525777779486052,
        0.01600249424138224,
        0.012809894506843729,
        0.02246380577589697,
        0.022319750281720237,
        0.03129536460942698,
        0.031230674294772585,
        0.039436834221692155,
        0.03480978663377977,
        0.04185389111188925,
        0.06428733162796366,
        0.045862075754009676,
        0.06078678679058799,
        0.05351363630871811,
        0.05973106973573801,
        0.0842132756214095,
        0.10213623429563772,
        0.07535736865488296,
        0.07558316991135006,
        0.1217807778428124,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 7e-1)
    @test isapprox(hrc2 / lrc2, 20, atol = 5e-1)
end

@testset "$(:Classic), $(:RP), $(:CVaR)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :CVaR)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :CVaR)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :CVaR)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :CVaR)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.04939116495315467,
        0.04880387529683067,
        0.04173386007462713,
        0.038407529094372325,
        0.041841616464888244,
        0.0537805463808844,
        0.02642475939870094,
        0.07736763363606748,
        0.03544320173223963,
        0.04094159908193833,
        0.09252002945495828,
        0.04267908264079304,
        0.03238824553768926,
        0.06577667031467883,
        0.04019656986160158,
        0.04220518109849602,
        0.048723879166031436,
        0.0807120181128111,
        0.042769022520414895,
        0.057893515178821846,
    ]

    w2t = [
        0.004958109406241485,
        0.0107224327723124,
        0.013512957542224755,
        0.01594190083444675,
        0.022145907961592437,
        0.0292363773236835,
        0.019155305967957673,
        0.062200051891773384,
        0.029580061677280296,
        0.040354715673165176,
        0.09432204761488663,
        0.04011163430048787,
        0.03580028151869949,
        0.07561073585707774,
        0.050815793682588054,
        0.06895391725449458,
        0.07714754455472057,
        0.1289861323859113,
        0.07491370270365476,
        0.10553038907680105,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 8e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 3e-2)
end

@testset "$(:Classic), $(:RP), $(:EVaR)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :EVaR)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :EVaR)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :EVaR)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :EVaR)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.04644475994411915,
        0.06366515916971194,
        0.04378790416499597,
        0.05092225480518606,
        0.04640694402649698,
        0.05062044463599438,
        0.03894494558893378,
        0.06586562327514726,
        0.038418534157235326,
        0.04445916135386437,
        0.06810153544170035,
        0.04257037598193672,
        0.04132495187560881,
        0.0491623656772855,
        0.04438603883039922,
        0.05344782724074077,
        0.05368763740936749,
        0.05226851563551302,
        0.041780998049112186,
        0.06373402273665076,
    ]

    w2t = [
        0.004648685027050885,
        0.013021960551074779,
        0.013063415300891827,
        0.020516403357401368,
        0.022279072321956556,
        0.030140059799560755,
        0.028175828803923143,
        0.04961319752572847,
        0.033207697951002486,
        0.04171667117992843,
        0.07294546571134636,
        0.04732334552499035,
        0.051085335258754014,
        0.06410412619041492,
        0.05750784225541883,
        0.07959865715930929,
        0.08923956925095108,
        0.08898375126505782,
        0.07571172084236806,
        0.11711719472287073,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 1e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 3e-2)
end

@testset "$(:Classic), $(:RP), $(:RVaR)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :RVaR)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :RVaR)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :RVaR)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :RVaR)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.0454954710262226,
        0.07569593270295329,
        0.043957575337114425,
        0.0558672104019315,
        0.049249599264164855,
        0.050662448382239635,
        0.0411080172961597,
        0.054494904862553206,
        0.039378742292816286,
        0.04560094224359964,
        0.061323104595475,
        0.04070776601332359,
        0.043671592040505906,
        0.041843737184590446,
        0.045099547484967584,
        0.05279627013440978,
        0.05840797678650866,
        0.04473954015902149,
        0.0415993684979389,
        0.06830025329350349,
    ]

    w2t = [
        0.0045790854398911835,
        0.015157430365661917,
        0.012920946665213115,
        0.022042694386569334,
        0.022252268553027848,
        0.031006204188333178,
        0.03102398500428025,
        0.04188782110186774,
        0.03446522704826401,
        0.04173267131315385,
        0.06662187192278156,
        0.046533375703627526,
        0.05758111935388933,
        0.05644204922297197,
        0.058812278592592686,
        0.08358005446007924,
        0.09852243256021069,
        0.07880025382420974,
        0.07568024623225383,
        0.12035798406112097,
    ]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-6)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 4e-1)
    @test isapprox(hrc2 / lrc2, 20, atol = 2e-0)
end

@testset "$(:Classic), $(:RP), $(:MDD)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :MDD)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :MDD)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = reverse(1:size(portfolio.returns, 2))
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :MDD)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :MDD)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.08321848642371815,
        0.034699747862080424,
        0.04178845732285892,
        0.027628095513838893,
        0.03998679802508808,
        0.05673352734682899,
        0.02234224996736454,
        0.09418046252165299,
        0.02786480561806567,
        0.03558351272814093,
        0.1568919232537885,
        0.032033032913518354,
        0.03750578705533342,
        0.03997579591949542,
        0.024305868557059507,
        0.040541082738431364,
        0.04468652085307903,
        0.06511138995197098,
        0.035403034175987,
        0.05951942125169894,
    ]

    w2t = [
        0.1030562593373048,
        0.06581581599763574,
        0.07099080765747305,
        0.04824197782535705,
        0.06732887343058071,
        0.06933072817173858,
        0.025149256898616135,
        0.12492416770029748,
        0.03239781557296892,
        0.04507100042893705,
        0.13647036788520997,
        0.038402912008787066,
        0.03580166139136979,
        0.035275151440073205,
        0.025261184914901688,
        0.024684971542034622,
        0.01652261459902679,
        0.021626026850536408,
        0.00676102632576781,
        0.00688738002138319,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 9.2e0)
    @test isapprox(hrc2 / lrc2, 20, atol = 3e0)
end

@testset "$(:Classic), $(:RP), $(:ADD)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :ADD)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :ADD)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :ADD)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :ADD)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.057602603655605544,
        0.048754890398656596,
        0.08761279560605131,
        0.028071835205155164,
        0.06562827421390469,
        0.04203173133618097,
        0.025689263771394466,
        0.08530376923054743,
        0.034593393671792774,
        0.0350973523220023,
        0.1000659834124061,
        0.027207701843224387,
        0.017629899203747378,
        0.03964510378994047,
        0.014601469748121416,
        0.047410889655393286,
        0.07494458498999328,
        0.049721437202785275,
        0.058422284556415696,
        0.05996473618668143,
    ]

    w2t = [
        0.011171969543463939,
        0.014032055040105278,
        0.04590343597479641,
        0.012856314838403028,
        0.050621237828349055,
        0.02224976191382803,
        0.018628057674024454,
        0.07977839250021374,
        0.028745234658130975,
        0.03132534235014177,
        0.11053452185772016,
        0.024766797187141528,
        0.016885339255735127,
        0.053454733348957686,
        0.019594922034436565,
        0.06130811987109412,
        0.1225774784670345,
        0.07478040562329874,
        0.09193045782137829,
        0.10885542221174656,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 8e-1)
    @test isapprox(hrc2 / lrc2, 20, atol = 6e-1)
end

@testset "$(:Classic), $(:RP), $(:CDaR)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :CDaR)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :CDaR)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :CDaR)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :CDaR)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.046949957960303226,
        0.03661834983962574,
        0.03822640513283658,
        0.027780437795411443,
        0.05445137396022303,
        0.034218467225507525,
        0.02235614079811853,
        0.0891415257915432,
        0.027308820539438314,
        0.028300876725765575,
        0.215773637466577,
        0.04030484034802265,
        0.020911760438324007,
        0.04004758507289081,
        0.026363442746335643,
        0.04469907928877756,
        0.05886961718462183,
        0.04952351192386208,
        0.03915000538744693,
        0.05900416437436812,
    ]

    w2t = [
        0.007052521038851694,
        0.005446130240302941,
        0.012413366117503868,
        0.009361457504379896,
        0.04021174933870001,
        0.019329482728429888,
        0.01774853717564121,
        0.08718247989210944,
        0.022002271917744995,
        0.022399906586733978,
        0.2516705125798999,
        0.035193041076863406,
        0.017721842358012378,
        0.04550543515629949,
        0.02342205740199043,
        0.0475332023321547,
        0.08640083338898245,
        0.07370827507677156,
        0.06435804060778014,
        0.11133885748084765,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 7.4e-2)
    @test isapprox(hrc2 / lrc2, 20, atol = 2.4e0)
end

@testset "$(:Classic), $(:RP), $(:UCI)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :UCI)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :UCI)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :UCI)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :UCI)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.054827613042077956,
        0.04136400954967725,
        0.061656164141309144,
        0.026847010940154726,
        0.059539021845281426,
        0.03569968719646861,
        0.023356382439714538,
        0.09683090264644328,
        0.02981299842184919,
        0.03221047945907848,
        0.1475969425769354,
        0.0354329253877475,
        0.019401357981471416,
        0.0419699225147063,
        0.017242238058303782,
        0.049644465993875175,
        0.06573071618936731,
        0.05394698803639865,
        0.04624964010900936,
        0.06064053347013054,
    ]

    w2t = [
        0.008156570938968257,
        0.008780638346363395,
        0.023214200309550456,
        0.011186981914492702,
        0.03977999188357741,
        0.02070319042256038,
        0.018310761229369563,
        0.09098077821910543,
        0.025010549901705176,
        0.028897935613459404,
        0.17117718500851792,
        0.03038975118172605,
        0.018936996741878136,
        0.05298425069392209,
        0.021213339647566398,
        0.05902511193454757,
        0.10466281388482923,
        0.08144284262526359,
        0.07867992631467524,
        0.10646618318792157,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 6e-1)
    @test isapprox(hrc2 / lrc2, 20, atol = 3.4e0)
end

@testset "$(:Classic), $(:RP), $(:EDaR)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :EDaR)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :EDaR)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :EDaR)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :EDaR)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.055323158669307405,
        0.03578993243944283,
        0.04077459842398797,
        0.029605864612833273,
        0.04696622602935098,
        0.04084436026235602,
        0.02530074053427011,
        0.0889187240110895,
        0.02689567043466572,
        0.03134448423619236,
        0.19601825114090862,
        0.03903196702101247,
        0.02653347429940722,
        0.04068540522040772,
        0.02503564183517531,
        0.047051944079435214,
        0.053530806932799296,
        0.05549058281069559,
        0.038349807839047685,
        0.05650835916761447,
    ]

    w2t = [
        0.015858049127086833,
        0.005699781835910997,
        0.015409495770182134,
        0.009983767427105456,
        0.03722670417801105,
        0.024786580037670154,
        0.01694267172076037,
        0.08449335249808726,
        0.02263515396947177,
        0.024486630264501656,
        0.2132583216134145,
        0.03791614518718178,
        0.023056163846423814,
        0.05053848499284413,
        0.024578702156905028,
        0.05223422190109789,
        0.07671411552429791,
        0.0878537243106757,
        0.0634950135275836,
        0.11283292011078797,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 5e-1)
    @test isapprox(hrc2 / lrc2, 20, atol = 6e0)
end

@testset "$(:Classic), $(:RP), $(:RDaR)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :RDaR)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :RDaR)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :RDaR)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :RDaR)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.0679360351869923,
        0.035313686084449736,
        0.04202344537462854,
        0.030665075865734883,
        0.04581754161249088,
        0.04694819503920259,
        0.025815132059665905,
        0.09117721014952683,
        0.027341586028619608,
        0.03291529498681437,
        0.17193311027355093,
        0.03568419702139943,
        0.032304737633277854,
        0.04018806182043081,
        0.02462777369887237,
        0.047065236361888634,
        0.04992591548643504,
        0.05846703540464668,
        0.03735766646844597,
        0.056493063442926726,
    ]

    w2t = [
        0.0282967978475588,
        0.0057954592491625415,
        0.015516211638368615,
        0.010367413865032052,
        0.03458953098651808,
        0.02878597498106382,
        0.01641041327188177,
        0.0800448228536231,
        0.023281481505302917,
        0.025784330130272357,
        0.19331586499292608,
        0.03679769961485477,
        0.028435128294964508,
        0.05165443128889416,
        0.025444554657274337,
        0.0527094547050732,
        0.07210154519268194,
        0.09516075182364081,
        0.06257870680405152,
        0.11292942629685457,
    ]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-6)
    @test isapprox(hrc1 / lrc1, 1, atol = 1.1e0)
    @test isapprox(hrc2 / lrc2, 20, atol = 4.3e0)
end

@testset "$(:Classic), $(:RP), Full $(:Kurt)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :Kurt)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :Kurt)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :Kurt)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :Kurt)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.03922265228200221,
        0.04918890700389464,
        0.03789038013384901,
        0.04964943280810259,
        0.03592393225345682,
        0.05445687828419427,
        0.026300844036386613,
        0.10497991002479666,
        0.031077649700123843,
        0.04292805860205612,
        0.12713395600746077,
        0.03206210834279925,
        0.026002501014891655,
        0.056812526930387235,
        0.03142454091482859,
        0.041823272333257604,
        0.04443212280977936,
        0.07509639236063076,
        0.036052465522254135,
        0.057541468634847937,
    ]

    w2t = [
        0.004144915332218853,
        0.010547873695475172,
        0.012586127709339912,
        0.023258316452256424,
        0.01936681212786137,
        0.03210055314795179,
        0.019494819360002687,
        0.08359459443512143,
        0.02612806193145197,
        0.041577133864953375,
        0.13457666907118945,
        0.036402224805583,
        0.03041072158857224,
        0.07162725738900827,
        0.039680319663114305,
        0.05692184096609248,
        0.07140379225584388,
        0.11868158990958652,
        0.06321510265507445,
        0.10428127363930244,
    ]

    @test isapprox(w1.weights, w1t, rtol = 0.0001)
    @test isapprox(w2.weights, w2t, rtol = 1.0e-5)
    @test isapprox(hrc1 / lrc1, 1, atol = 1.4e0)
    @test isapprox(hrc2 / lrc2, 20, atol = 3e0)
end

@testset "$(:Classic), $(:RP), Reduced $(:Kurt)" begin
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
        max_num_assets_kurt = 1,
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :Kurt)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :Kurt)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :Kurt)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :Kurt)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.0500000000000011,
        0.05000000000000058,
        0.0500000000000005,
        0.050000000000001134,
        0.05000000000000087,
        0.0500000000000012,
        0.05000000000000044,
        0.050000000000000794,
        0.05000000000000046,
        0.05000000000000119,
        0.0500000000000009,
        0.050000000000001155,
        0.049999999999997845,
        0.04999999999999776,
        0.050000000000000905,
        0.04999999999999878,
        0.049999999999997484,
        0.049999999999998525,
        0.05000000000000055,
        0.04999999999999802,
    ]

    w2t = [
        0.02610708275409532,
        0.02837943088274139,
        0.030651303698748616,
        0.03292296592900042,
        0.03519875334620729,
        0.03748347710210248,
        0.039781527753448405,
        0.042099870335476915,
        0.044453433199575766,
        0.04686708624986517,
        0.04937400472746767,
        0.05201270392712209,
        0.054818393390558565,
        0.05781799869836054,
        0.061024063467002496,
        0.06442370351110903,
        0.06809573346857908,
        0.0719652815702216,
        0.07608875958076423,
        0.08043442640755293,
    ]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(hrc1 / lrc1, 1, atol = 2.1e0)
    @test isapprox(hrc2 / lrc2, 20, atol = 17e0)
end

@testset "$(:Classic), $(:RP), Full $(:SKurt)" begin
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

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SKurt)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :SKurt)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SKurt)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :SKurt)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.041170947272951605,
        0.055151102657145686,
        0.03582026711008352,
        0.03810281647100041,
        0.03551958231813474,
        0.054217762490842675,
        0.021569898487818123,
        0.09467152302998202,
        0.030283020031477308,
        0.03821071007217146,
        0.1240516019281326,
        0.034933363590817265,
        0.028499637129673648,
        0.06128622298896649,
        0.03461265007418104,
        0.044610124033738624,
        0.04779839493700672,
        0.07608146521350194,
        0.03653809970499657,
        0.06687081045737765,
    ]

    w2t = [
        0.004175205263429701,
        0.011765748187100822,
        0.011015252491488306,
        0.01577359455014908,
        0.017593349506285338,
        0.03187177049780766,
        0.01523068356780147,
        0.0720093728891317,
        0.02536718681887658,
        0.035781514103165026,
        0.13000234588810786,
        0.037090729040813834,
        0.0335443668308728,
        0.0755080628731996,
        0.043604346172737944,
        0.06299741499421682,
        0.07703658809323406,
        0.11862869961574692,
        0.06432382504657572,
        0.11667994356925879,
    ]

    @test isapprox(w1.weights, w1t, rtol = 1.0e-5)
    @test isapprox(w2.weights, w2t, rtol = 0.0001)
    @test isapprox(hrc1 / lrc1, 1, atol = 1.8e0)
    @test isapprox(hrc2 / lrc2, 20, atol = 3.6e0)
end

@testset "$(:Classic), $(:RP), Reduced $(:SKurt)" begin
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
        max_num_assets_kurt = 1,
    )
    asset_statistics!(portfolio)

    w1 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SKurt)
    rc1 = risk_contribution(portfolio, type = :RP, rm = :SKurt)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(portfolio; class = :Classic, type = :RP, rm = :SKurt)
    rc2 = risk_contribution(portfolio, type = :RP, rm = :SKurt)
    lrc2, hrc2 = extrema(rc2)

    w1t = [
        0.0500000000000011,
        0.05000000000000058,
        0.0500000000000005,
        0.050000000000001134,
        0.05000000000000087,
        0.0500000000000012,
        0.05000000000000044,
        0.050000000000000794,
        0.05000000000000046,
        0.05000000000000119,
        0.0500000000000009,
        0.050000000000001155,
        0.049999999999997845,
        0.04999999999999776,
        0.050000000000000905,
        0.04999999999999878,
        0.049999999999997484,
        0.049999999999998525,
        0.05000000000000055,
        0.04999999999999802,
    ]

    w2t = [
        0.02610708275409532,
        0.02837943088274139,
        0.030651303698748616,
        0.03292296592900042,
        0.03519875334620729,
        0.03748347710210248,
        0.039781527753448405,
        0.042099870335476915,
        0.044453433199575766,
        0.04686708624986517,
        0.04937400472746767,
        0.05201270392712209,
        0.054818393390558565,
        0.05781799869836054,
        0.061024063467002496,
        0.06442370351110903,
        0.06809573346857908,
        0.0719652815702216,
        0.07608875958076423,
        0.08043442640755293,
    ]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(hrc1 / lrc1, 1, atol = 2.1e0)
    @test isapprox(hrc2 / lrc2, 20, atol = 17e0)
end

@testset "$(:Classic), $(:RRP)" begin
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

    portfolio.risk_budget = []
    w1 = opt_port!(
        portfolio;
        class = :Classic,
        type = :RRP,
        rrp_penalty = 1,
        rrp_ver = :None,
    )
    rc1 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc1, hrc1 = extrema(rc1)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w2 = opt_port!(
        portfolio;
        class = :Classic,
        type = :RRP,
        rrp_penalty = 1,
        rrp_ver = :None,
    )
    rc2 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc2, hrc2 = extrema(rc2)

    portfolio.risk_budget = []
    w3 = opt_port!(
        portfolio;
        class = :Classic,
        type = :RRP,
        rrp_penalty = 5,
        rrp_ver = :None,
    )
    rc3 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc3, hrc3 = extrema(rc3)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w4 = opt_port!(
        portfolio;
        class = :Classic,
        type = :RRP,
        rrp_penalty = 5,
        rrp_ver = :None,
    )
    rc4 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc4, hrc4 = extrema(rc4)

    portfolio.risk_budget = []
    w5 =
        opt_port!(portfolio; class = :Classic, type = :RRP, rrp_penalty = 1, rrp_ver = :Reg)
    rc5 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc5, hrc5 = extrema(rc5)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w6 =
        opt_port!(portfolio; class = :Classic, type = :RRP, rrp_penalty = 1, rrp_ver = :Reg)
    rc6 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc6, hrc6 = extrema(rc6)

    portfolio.risk_budget = []
    w7 =
        opt_port!(portfolio; class = :Classic, type = :RRP, rrp_penalty = 5, rrp_ver = :Reg)
    rc7 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc7, hrc7 = extrema(rc7)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w8 =
        opt_port!(portfolio; class = :Classic, type = :RRP, rrp_penalty = 5, rrp_ver = :Reg)
    rc8 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc8, hrc8 = extrema(rc8)

    portfolio.risk_budget = []
    w9 = opt_port!(
        portfolio;
        class = :Classic,
        type = :RRP,
        rrp_penalty = 1,
        rrp_ver = :Reg_Pen,
    )
    rc9 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc9, hrc9 = extrema(rc9)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w10 = opt_port!(
        portfolio;
        class = :Classic,
        type = :RRP,
        rrp_penalty = 1,
        rrp_ver = :Reg_Pen,
    )
    rc10 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc10, hrc10 = extrema(rc10)

    portfolio.risk_budget = []
    w11 = opt_port!(
        portfolio;
        class = :Classic,
        type = :RRP,
        rrp_penalty = 5,
        rrp_ver = :Reg_Pen,
    )
    rc11 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc11, hrc11 = extrema(rc11)

    portfolio.risk_budget = 1:size(portfolio.returns, 2)
    w12 = opt_port!(
        portfolio;
        class = :Classic,
        type = :RRP,
        rrp_penalty = 5,
        rrp_ver = :Reg_Pen,
    )
    rc12 = risk_contribution(portfolio, type = :RRP, rm = :SD)
    lrc12, hrc12 = extrema(rc12)

    w1t = [
        0.05063226479993809,
        0.05124795183015634,
        0.04690530569953088,
        0.043690005918066634,
        0.045715234147759326,
        0.05615047368290105,
        0.02763287472025603,
        0.07705758777270617,
        0.03949584162673613,
        0.04723336278374185,
        0.08435232745960597,
        0.033855622876288155,
        0.027546000252391043,
        0.062067256085427125,
        0.03563645240128392,
        0.04413102220290603,
        0.05085269866596615,
        0.07142013082319783,
        0.0452964414517217,
        0.059081144799419644,
    ]

    w2t = [
        0.005639469048865928,
        0.011008565644892861,
        0.015581918684624618,
        0.019369659725540375,
        0.02543728006137944,
        0.03265702133500953,
        0.02037110575860042,
        0.05957155175533342,
        0.03298226975847406,
        0.044829463432875624,
        0.08741168763949635,
        0.038228966150977935,
        0.031437950441703943,
        0.07961004374558595,
        0.0461426883144182,
        0.06097518105577283,
        0.08329762064695105,
        0.11639291280317195,
        0.07928202190618092,
        0.10977262209014467,
    ]

    w3t = [
        0.05063226479993809,
        0.05124795183015634,
        0.04690530569953088,
        0.043690005918066634,
        0.045715234147759326,
        0.05615047368290105,
        0.02763287472025603,
        0.07705758777270617,
        0.03949584162673613,
        0.04723336278374185,
        0.08435232745960597,
        0.033855622876288155,
        0.027546000252391043,
        0.062067256085427125,
        0.03563645240128392,
        0.04413102220290603,
        0.05085269866596615,
        0.07142013082319783,
        0.0452964414517217,
        0.059081144799419644,
    ]

    w4t = [
        0.005639469048865928,
        0.011008565644892861,
        0.015581918684624618,
        0.019369659725540375,
        0.02543728006137944,
        0.03265702133500953,
        0.02037110575860042,
        0.05957155175533342,
        0.03298226975847406,
        0.044829463432875624,
        0.08741168763949635,
        0.038228966150977935,
        0.031437950441703943,
        0.07961004374558595,
        0.0461426883144182,
        0.06097518105577283,
        0.08329762064695105,
        0.11639291280317195,
        0.07928202190618092,
        0.10977262209014467,
    ]

    w5t = [
        0.050632270743810964,
        0.05124795528972891,
        0.046905318486685,
        0.04369002241005243,
        0.04571524917738814,
        0.05615046486057305,
        0.027632904575597795,
        0.07705752368324727,
        0.03949586647658839,
        0.047233374066738165,
        0.08435225287761507,
        0.033855652553976275,
        0.02754602944181694,
        0.06206723048920754,
        0.0356364774817785,
        0.0441310374921869,
        0.05085270331544174,
        0.07142008063335671,
        0.04529645759558092,
        0.059081128348629296,
    ]

    w6t = [
        0.005639461553024654,
        0.011008557376780512,
        0.01558190868333066,
        0.01936965006068913,
        0.02543727063895136,
        0.03265701514849208,
        0.020371093929734484,
        0.059571587418443296,
        0.0329822541618955,
        0.04482945070898587,
        0.08741177238224775,
        0.03822894981166275,
        0.03143793576601198,
        0.07961005120374322,
        0.046142672093931525,
        0.06097516599296141,
        0.08329761439172295,
        0.11639294963195672,
        0.07928200646305023,
        0.10977263258238407,
    ]

    w7t = [
        0.050632270743810964,
        0.05124795528972891,
        0.046905318486685,
        0.04369002241005243,
        0.04571524917738814,
        0.05615046486057305,
        0.027632904575597795,
        0.07705752368324727,
        0.03949586647658839,
        0.047233374066738165,
        0.08435225287761507,
        0.033855652553976275,
        0.02754602944181694,
        0.06206723048920754,
        0.0356364774817785,
        0.0441310374921869,
        0.05085270331544174,
        0.07142008063335671,
        0.04529645759558092,
        0.059081128348629296,
    ]

    w8t = [
        0.005639461553024654,
        0.011008557376780512,
        0.01558190868333066,
        0.01936965006068913,
        0.02543727063895136,
        0.03265701514849208,
        0.020371093929734484,
        0.059571587418443296,
        0.0329822541618955,
        0.04482945070898587,
        0.08741177238224775,
        0.03822894981166275,
        0.03143793576601198,
        0.07961005120374322,
        0.046142672093931525,
        0.06097516599296141,
        0.08329761439172295,
        0.11639294963195672,
        0.07928200646305023,
        0.10977263258238407,
    ]

    w9t = [
        0.0506322668429295,
        0.05124795373281254,
        0.04690530823494499,
        0.043690007639694446,
        0.04571523585825459,
        0.056150474410049,
        0.027632877631675584,
        0.07705757238664261,
        0.03949584655828325,
        0.047233365903979294,
        0.08435231116400815,
        0.03385562638740609,
        0.027546003008159237,
        0.062067255302555704,
        0.03563645411256288,
        0.044131022943107744,
        0.050852702454837334,
        0.07142012425487358,
        0.04529644648715733,
        0.05908114468606597,
    ]

    w10t = [
        0.005639501123313942,
        0.011008597271153931,
        0.015581943905832368,
        0.01936967698794819,
        0.025437294588214254,
        0.03265704415346579,
        0.020371111188622035,
        0.05957154742242699,
        0.032982285361347764,
        0.04482947278788263,
        0.0874116556811304,
        0.0382289721853606,
        0.031437956174044455,
        0.07961003009246612,
        0.04614269021853119,
        0.06097517765079728,
        0.08329760501401974,
        0.11639284823329737,
        0.07928201761012937,
        0.10977257235001557,
    ]

    w11t = [
        0.05063227092483982,
        0.051247957970220706,
        0.046905313223576876,
        0.04369000832670002,
        0.04571523607142845,
        0.056150476586436994,
        0.027632885874903337,
        0.077057507452318,
        0.03949586365421145,
        0.047233374862919744,
        0.08435226765306823,
        0.03385563474314793,
        0.02754601140611397,
        0.062067261204944754,
        0.035636456892157733,
        0.044131020188731974,
        0.05085272421479202,
        0.07142010982372254,
        0.04529647152631459,
        0.059081147399450785,
    ]

    w12t = [
        0.016803340442582207,
        0.019323023095702243,
        0.015068043628123692,
        0.0188459404418481,
        0.024570916279022192,
        0.032040946880747094,
        0.02000931588824339,
        0.0584656081395763,
        0.032347338194528834,
        0.04397847882346323,
        0.08577138764605112,
        0.03753687980177539,
        0.031062491168826374,
        0.07812625162681952,
        0.04562332519102003,
        0.0600409335633433,
        0.08125333541684177,
        0.11408515333305874,
        0.07775046163627136,
        0.10729682880215521,
    ]

    @test isapprox(w1.weights, w1t)
    @test isapprox(w2.weights, w2t)
    @test isapprox(w3.weights, w3t)
    @test isapprox(w4.weights, w4t)
    @test isapprox(w5.weights, w5t)
    @test isapprox(w6.weights, w6t)
    @test isapprox(w7.weights, w7t)
    @test isapprox(w8.weights, w8t)
    @test isapprox(w9.weights, w9t)
    @test isapprox(w10.weights, w10t)
    @test isapprox(w11.weights, w11t)
    @test isapprox(w12.weights, w12t, rtol = 1.0e-6)

    @test isapprox(w1.weights, w3.weights)
    @test isapprox(w2.weights, w4.weights)
    @test isapprox(w5.weights, w7.weights)
    @test isapprox(w6.weights, w8.weights)
    @test !isapprox(w9.weights, w11.weights)
    @test !isapprox(w10.weights, w12.weights)

    @test isapprox(hrc1 / lrc1, 1, atol = 3e-6)
    @test isapprox(hrc2 / lrc2, 20, atol = 3e-4)
    @test isapprox(hrc3 / lrc3, 1, atol = 3e-6)
    @test isapprox(hrc4 / lrc4, 20, atol = 3e-4)
    @test isapprox(hrc5 / lrc5, 1, atol = 5e-6)
    @test isapprox(hrc6 / lrc6, 20, atol = 3e-4)
    @test isapprox(hrc7 / lrc7, 1, atol = 5e-6)
    @test isapprox(hrc8 / lrc8, 20, atol = 3e-4)
    @test isapprox(hrc9 / lrc9, 1, atol = 4e-6)
    @test isapprox(hrc10 / lrc10, 20, atol = 4e-4)
    @test isapprox(hrc11 / lrc11, 1, atol = 4e-6)
    @test isapprox(hrc12 / lrc12, 7, atol = 4e-1)
end
