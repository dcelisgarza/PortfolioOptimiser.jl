using Test, PortfolioOptimiser, CSV, TimeSeries, DataFrames, StatsBase

A = TimeArray(CSV.File("./assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))

@testset "mu" begin
    portfolio = HCPortfolio(returns = returns)

    asset_statistics!(portfolio, mu_type = :JS, mu_target = :GM)
    jsgm = copy(portfolio.mu)
    jsgmt = [
        0.0006341475601844101,
        0.0006789058959280285,
        0.0008178383518190523,
        0.0007639436970922277,
        0.0013856285900496848,
        -0.0002523200512863443,
        0.0014200748323063197,
        0.0003576843329880906,
        0.000716221244111695,
        0.0004491661692147687,
        0.0003248478475254554,
        -0.00015878941972113393,
        -0.0007869829861367987,
        0.00011240543858372297,
        -0.0007127483820581668,
        0.0009418444136399807,
        0.0008403726820401893,
        0.00040878369315988205,
        0.0007395087768546029,
        0.0005917202666292221,
    ]
    @test isapprox(jsgm, jsgmt)

    asset_statistics!(portfolio, mu_type = :BS, mu_target = :GM)
    bsgm = copy(portfolio.mu)
    bsgmt = [
        0.000596098181618497,
        0.0006308701348944901,
        0.0007388042835731622,
        0.0006969344869229568,
        0.0011799103994799723,
        -9.258270551964977e-5,
        0.0012066710711265273,
        0.00038131882803247186,
        0.0006598597637974114,
        0.000452389436043844,
        0.0003558087477472064,
        -1.992042492707195e-5,
        -0.000507952809884533,
        0.00019076602676948228,
        -0.000450281269117916,
        0.0008351423827437062,
        0.0007563108030747699,
        0.000421017010165618,
        0.0006779514329132588,
        0.0005631371734706867,
    ]
    @test isapprox(bsgm, bsgmt)

    asset_statistics!(portfolio, mu_type = :BOP, mu_target = :GM)
    bopgm = copy(portfolio.mu)
    bopgmt = [
        0.00016917378887231472,
        0.00018160113183457628,
        0.00022017632122135324,
        0.00020521224029664158,
        0.0003778257043783738,
        -7.695772718287296e-5,
        0.00038738985056862576,
        9.24125938068311e-5,
        0.00019196189753174021,
        0.00011781291610921848,
        8.32954034613955e-5,
        -5.0988548109821304e-5,
        -0.0002254091731583472,
        2.4309861845846202e-5,
        -0.00020479761847862714,
        0.0002546071335838931,
        0.00022643307419105795,
        0.00010660054926711702,
        0.00019842778039652644,
        0.00015739366972331245,
    ]
    @test isapprox(bsgm, bsgmt)

    asset_statistics!(portfolio, mu_type = :JS, mu_target = :VW)
    jsvw = copy(portfolio.mu)
    jsvwt = [
        0.0005962555831924726,
        0.0006415098938267387,
        0.0007819818843870736,
        0.0007274900136324292,
        0.001356063904844158,
        -0.00030003512981584556,
        0.001390891851892819,
        0.0003167288189678275,
        0.0006792387398634284,
        0.00040922438130752903,
        0.0002835284667077342,
        -0.0002054680690806927,
        -0.000840622757491449,
        6.873194654195269e-5,
        -0.0007655655468742032,
        0.0009073620789126582,
        0.0008047659214344713,
        0.00036839442002307234,
        0.000702784325800534,
        0.0005533581454299117,
    ]
    @test isapprox(jsvw, jsvwt)

    asset_statistics!(portfolio, mu_type = :BS, mu_target = :VW)
    bsvw = copy(portfolio.mu)
    bsvwt = [
        0.0005338285186301338,
        0.0005684298660735894,
        0.0006758344444072082,
        0.0006341700785954429,
        0.0011147763081207544,
        -0.00015147341032380464,
        0.0011414056806530696,
        0.0003201029629179168,
        0.0005972772596601718,
        0.00039082486862451814,
        0.00029471804581950544,
        -7.91676414513528e-5,
        -0.0005648055341960804,
        0.00013048509387008866,
        -0.0005074169542728796,
        0.0007716998683304457,
        0.000693255069558542,
        0.00035960636906711714,
        0.0006152801634355449,
        0.0005010292310561348,
    ]
    @test isapprox(bsvw, bsvwt)

    asset_statistics!(portfolio, mu_type = :BOP, mu_target = :VW)
    bopvw = copy(portfolio.mu)
    bopvwt = [
        0.00015313828431741356,
        0.00016556562727967512,
        0.00020414081666645203,
        0.0001891767357417404,
        0.00036179019982347235,
        -9.299323173777384e-5,
        0.00037135434601372436,
        7.637708925193004e-5,
        0.00017592639297683903,
        0.0001017774115543174,
        6.725989890649445e-5,
        -6.70240526647222e-5,
        -0.00024144467771324788,
        8.27435729094522e-6,
        -0.00022083312303352787,
        0.00023857162902899187,
        0.00021039756963615674,
        9.056504471221594e-5,
        0.00018239227584162526,
        0.00014135816516841132,
    ]
    @test isapprox(bsvw, bsvwt)

    asset_statistics!(portfolio, mu_type = :JS, mu_target = :SE)
    jsse = copy(portfolio.mu)
    jsset = [
        0.0005441589609781389,
        0.0005935020904084804,
        0.0007466660103623951,
        0.0006872506885098965,
        0.0013726175062978338,
        -0.0004331134367079999,
        0.0014105922290353827,
        0.0002393763863887135,
        0.0006346398142441448,
        0.0003402291104782567,
        0.00020317631493160286,
        -0.0003300020501394817,
        -0.0010225442535071647,
        -3.102750669731119e-5,
        -0.0009407054716041901,
        0.000883374559885473,
        0.0007715086313905049,
        0.00029571007933939126,
        0.0006603127916494866,
        0.0004973856526660391,
    ]
    @test isapprox(jsse, jsset)

    asset_statistics!(portfolio, mu_type = :BS, mu_target = :SE)
    bsse = copy(portfolio.mu)
    bsset = [
        0.00039511694346837297,
        0.0004306923523558474,
        0.00054112047359735,
        0.0004982832155335399,
        0.0009924189759160213,
        -0.0003094768985417447,
        0.00101979799107727,
        0.00017537480635680485,
        0.00046035182801232744,
        0.00024808760220746524,
        0.0001492752792552546,
        -0.00023513565224304845,
        -0.000734444722038041,
        -1.958098936102768e-5,
        -0.0006754405980057115,
        0.0006396846020554542,
        0.0005590315062196169,
        0.0002159902716505294,
        0.00047886153096587204,
        0.0003613943241810836,
    ]
    @test isapprox(bsse, bsset)

    asset_statistics!(portfolio, mu_type = :BOP, mu_target = :SE)
    bopse = copy(portfolio.mu)
    bopset = [
        0.00013646827659120816,
        0.00014889561955346972,
        0.00018747080894024668,
        0.00017250672801553501,
        0.00034512019209726716,
        -0.00010966323946397949,
        0.0003546843382875191,
        5.970708152572455e-5,
        0.00015925638525063365,
        8.510740382811192e-5,
        5.058989118028895e-5,
        -8.369406039092783e-5,
        -0.0002581146854394537,
        -8.395650435260338e-6,
        -0.00023750313075973365,
        0.00022190162130278655,
        0.00019372756190995139,
        7.389503698601047e-5,
        0.00016572226811541988,
        0.0001246881574422059,
    ]
    @test isapprox(bsse, bsset)

    asset_statistics!(portfolio, mu_type = :Hist)
    mu1 = copy(portfolio.mu)
    asset_statistics!(portfolio, mu_type = :Exp, mu_alpha = eps())
    mu2 = copy(portfolio.mu)
    @test isapprox(mu1, mu2)
    asset_statistics!(portfolio, mu_type = :Exp, mu_alpha = 1 - eps())
    mu3 = copy(portfolio.mu)
    @test isapprox(mu3, portfolio.returns[end, 1:end])

    asset_statistics!(portfolio, cov_type = :Hist, cov_kwargs = (; corrected = false))
    cov1 = copy(portfolio.cov)
    asset_statistics!(
        portfolio,
        cov_type = :Cov_Est,
        cov_est = StatsBase.SimpleCovariance(; corrected = false),
    )
    cov2 = copy(portfolio.cov)
    @test isapprox(cov1, cov2)
end
