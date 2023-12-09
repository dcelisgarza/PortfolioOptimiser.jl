using Test, PortfolioOptimiser, ECOS

@testset "OWA weights" begin
    w1t = [
        -0.02,
        -0.018029556650246307,
        -0.016175228712174528,
        -0.014433272942339079,
        -0.012799946097294384,
        -0.011271504933594861,
        -0.00984420620779493,
        -0.00851430667644901,
        -0.007278063096111518,
        -0.006131732223336878,
        -0.005071570814679503,
        -0.004093835626693817,
        -0.003194783415934239,
        -0.0023706709389551854,
        -0.0016177549523110784,
        -0.0009322922125563356,
        -0.00031053947624537704,
        0.00025124650006737835,
        0.0007568089598275113,
        0.0012098911464806027,
        0.0016142363034722328,
        0.001973587674247983,
        0.002291688502253433,
        0.0025722820309341636,
        0.0028191115037357567,
        0.003035920164103792,
        0.003226451255483851,
        0.0033944480213215144,
        0.0035436537050623626,
        0.0036778115501519765,
        0.0038006648000359347,
        0.003915956698159822,
        0.004027430487969216,
        0.004138829412909697,
        0.004253896716426849,
        0.004376375641966251,
        0.004510009432973482,
        0.004658541332894126,
        0.004825714585173761,
        0.005015272433257969,
        0.005230958120592331,
        0.0054765148906224265,
        0.005755685986793838,
        0.006072214652552144,
        0.006429844131342926,
        0.006832317666611766,
        0.007283378501804245,
        0.0077867698803659405,
        0.008346235045742436,
        0.00896551724137931,
    ]
    w1 = owa_l_moment_crm(50; k = 4, method = :CRRA, g = 0.5, max_phi = 0.5)
    @test isapprox(w1, w1t)

    w2t = [
        -0.020000000000000004,
        -0.017812025908273858,
        -0.015767183669651033,
        -0.013860587847490365,
        -0.01208735300515068,
        -0.010442593705990821,
        -0.00892142451336962,
        -0.007518959990645907,
        -0.006230314701178517,
        -0.005050603208326285,
        -0.003974940075448045,
        -0.002998439865902631,
        -0.0021162171430488762,
        -0.001323386470245614,
        -0.00061506241085168,
        1.3640471774093475e-5,
        0.0005676076142728719,
        0.0010517244532858214,
        0.0014708764254541078,
        0.001829948967418898,
        0.0021338275158213574,
        0.002387397507302653,
        0.0025955443785039495,
        0.0027631535660664138,
        0.002895110506631212,
        0.002996300636839511,
        0.003071609393332476,
        0.003125922212751273,
        0.003164124531737067,
        0.0031911017869310277,
        0.003211739414974317,
        0.0032309228525081043,
        0.0032535375361735534,
        0.003284468902611832,
        0.0033286023884641055,
        0.00339082343037154,
        0.003476017464975301,
        0.0035890699289165564,
        0.0037348662588364712,
        0.003918291891376211,
        0.004144232263176943,
        0.004417572810879833,
        0.004743198971126047,
        0.00512599618055675,
        0.00557084987581311,
        0.00608264549353629,
        0.006666268470367462,
        0.0073266042429477865,
        0.00806853824791843,
        0.008896955921920562,
    ]
    w2 = owa_l_moment_crm(
        50;
        k = 4,
        method = :E,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict(
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => false)),
        ),
    )
    @test isapprox(w2, w2t, rtol = 2e-2)

    w3t = [
        -0.01,
        -0.009436205569775124,
        -0.00889094183564734,
        -0.008363901334737243,
        -0.007854776604165432,
        -0.007363260181052503,
        -0.006889044602519053,
        -0.00643182240568568,
        -0.005991286127672982,
        -0.005567128305601554,
        -0.005159041476591993,
        -0.004766718177764899,
        -0.004389850946240867,
        -0.004028132319140494,
        -0.003681254833584377,
        -0.0033489110266931147,
        -0.003030793435587304,
        -0.0027265945973875406,
        -0.002436007049214423,
        -0.0021587233281885466,
        -0.0018944359714305108,
        -0.001642837516060911,
        -0.001403620499200346,
        -0.0011764774579694104,
        -0.0009611009294887044,
        -0.0007571834508788228,
        -0.0005644175592603638,
        -0.00038249579175392414,
        -0.00021111068548010131,
        -4.9954777559492104e-5,
        0.00010127939488730606,
        0.00024289929473969622,
        0.000375212384877081,
        0.0004985261281788638,
        0.0006131479875244466,
        0.0007193854257932333,
        0.000817545905864626,
        0.0009079368906180276,
        0.0009908658429328412,
        0.0010666402256884699,
        0.001135567501764316,
        0.0011979551340397829,
        0.0012541105853942729,
        0.0013043413187071894,
        0.0013489547968579348,
        0.0013882584827259118,
        0.0014225598391905245,
        0.0014521663291311743,
        0.0014773854154272648,
        0.0014985245609581985,
        0.0015158912286033785,
        0.0015297928812422078,
        0.0015405369817540892,
        0.001548430993018425,
        0.0015537823779146187,
        0.0015568985993220732,
        0.0015580871201201908,
        0.0015576554031883746,
        0.0015559109114060279,
        0.0015531611076525528,
        0.0015497134548073527,
        0.0015458754157498303,
        0.0015419544533593883,
        0.0015382580305154307,
        0.0015350936100973582,
        0.0015327686549845754,
        0.0015315906280564842,
        0.0015318669921924885,
        0.0015339052102719894,
        0.001538012745174392,
        0.0015444970597790978,
        0.00155366561696551,
        0.0015658258796130307,
        0.0015812853106010642,
        0.0016003513728090118,
        0.001623331529116278,
        0.0016505332424022639,
        0.001682263975546374,
        0.00171883119142801,
        0.0017605423529265753,
        0.0018077049229214724,
        0.001860626364292105,
        0.0019196141399178744,
        0.001984975712678185,
        0.0020570185454524387,
        0.0021360501011200394,
        0.002222377842560389,
        0.0023163092326528908,
        0.002418151734276947,
        0.002528212810311961,
        0.0026467999236373363,
        0.0027742205371324748,
        0.0029107821136767787,
        0.003056792116149653,
        0.0032125580074304997,
        0.0033783872503987196,
        0.003554587307933718,
        0.003741465642914897,
        0.003939329718221661,
        0.004148486996733409,
    ]
    w3 = owa_l_moment_crm(
        100;
        k = 4,
        method = :SS,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict(
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 400),
            ),
        ),
    )
    @test isapprox(w3, w3t, rtol = 3e-2)

    w4t = [
        -0.02,
        -0.018021220988607034,
        -0.016151150043965543,
        -0.014386718673590185,
        -0.012724858384995633,
        -0.011162500685696547,
        -0.0096965770832076,
        -0.008324019085043453,
        -0.007041758198718773,
        -0.005846725931748224,
        -0.004735853791646473,
        -0.0037060732859281885,
        -0.0027543159221080334,
        -0.0018775132077006723,
        -0.0010725966502207745,
        -0.00033649775718300293,
        0.00033385196389797497,
        0.0009415210055074939,
        0.0014895778601308881,
        0.001981091020253492,
        0.00241912897836064,
        0.0028067602269376654,
        0.0031470532584699034,
        0.003443076565442687,
        0.003697898640341351,
        0.00391458797565123,
        0.004096213063857658,
        0.00424584239744597,
        0.0043665444689014975,
        0.0044613877707095784,
        0.004533440795355544,
        0.004585772035324729,
        0.00462144998310247,
        0.004643543131174098,
        0.0046551199720249485,
        0.0046592489981403555,
        0.004658998702005654,
        0.004657437576106178,
        0.00465763411292726,
        0.004662656804954237,
        0.0046755741446724425,
        0.004699454624567208,
        0.004737366737123871,
        0.004792378974827764,
        0.004867559830164222,
        0.004965977795618578,
        0.0050907013636761695,
        0.005244799026822325,
        0.0054313392775423835,
        0.0056533906083216755,
    ]
    w4 = owa_l_moment_crm(
        50;
        k = 4,
        method = :SD,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict(
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 400),
            ),
        ),
    )
    @test isapprox(w4, w4t, rtol = 3e-3)

    w5t = [
        -0.019999999999999997,
        -0.015072010108697215,
        -0.011543217347246466,
        -0.009062700449670655,
        -0.007346942172168242,
        -0.006170585904972388,
        -0.005358003212597124,
        -0.004775637846933882,
        -0.004325091777661593,
        -0.003936918784433643,
        -0.0035650911553048756,
        -0.0031821050358619824,
        -0.0027746899735204417,
        -0.002340088201451322,
        -0.0018828692066011744,
        -0.0014122451262682653,
        -0.0009398525176984103,
        -0.0004779660451636555,
        -3.8109628987057084e-5,
        0.0003699693990231853,
        0.0007389975892669917,
        0.001064578572709812,
        0.001345336619440928,
        0.0015828350129693997,
        0.001781303695794395,
        0.001947210641786668,
        0.0020887114109179143,
        0.0022150113418747653,
        0.0023356748380941694,
        0.002459916202756901,
        0.0025959064782759473,
        0.0027501307458165313,
        0.0029268303403845036,
        0.003127564437019865,
        0.00335092546363216,
        0.0035924427960144904,
        0.003844709190572906,
        0.004097764410307906,
        0.004339770499584811,
        0.00455801316322975,
        0.004740263705488013,
        0.004876535984381512,
        0.0049612728370021005,
        0.004995996431277511,
        0.004995996431277511,
        0.004995996431277511,
        0.004995996431277511,
        0.0051045023497387695,
        0.0054109837723409805,
        0.006040808121419238,
    ]
    w5 = owa_l_moment_crm(
        50;
        k = 8,
        method = :CRRA,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict(
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 400),
            ),
        ),
    )
    @test isapprox(w5, w5t)

    w6t = [
        -0.019999999999999997,
        -0.015711178136335626,
        -0.012494531799108057,
        -0.010109310550524752,
        -0.008356917736020002,
        -0.0070755380656453495,
        -0.006135210346347776,
        -0.005433327256957208,
        -0.004890544057704882,
        -0.004447078126094166,
        -0.004059381210945355,
        -0.0036971662964360346,
        -0.003340770967958539,
        -0.002978839171616101,
        -0.002606303259179227,
        -0.0022226482103238523,
        -0.0018304399239728712,
        -0.0014340994705625539,
        -0.001038905197055451,
        -0.0006502045765213186,
        -0.00027281769410764144,
        8.938573877870413e-5,
        0.000433753950257069,
        0.0007589833619043874,
        0.0010651296390107022,
        0.0013535176453373,
        0.0016265684105559032,
        0.0018875612185473515,
        0.002140348924738217,
        0.0023890446106537935,
        0.002637697683865898,
        0.0028899775315139272,
        0.003148882835577611,
        0.0034164946580798957,
        0.003693791404398406,
        0.003980543772863923,
        0.00427530779882431,
        0.004575534101352356,
        0.004877811440775924,
        0.005178262695208908,
        0.005473111364261387,
        0.005759436708107435,
        0.006036135630089055,
        0.0063051094110346175,
        0.006572693403470305,
        0.006851347793902951,
        0.0071616275413527745,
        0.007534449600314362,
        0.008013675536324451,
        0.008659027642314836,
    ]
    w6 = owa_l_moment_crm(
        50;
        k = 8,
        method = :E,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict(
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 400),
            ),
        ),
    )
    @test isapprox(w6, w6t, rtol = 5e-3)

    w7t = [
        -0.020000000000000004,
        -0.013584639851228918,
        -0.008909762217137511,
        -0.005589147979118064,
        -0.0033029979737387275,
        -0.0017896003303162007,
        -0.0008376802387060904,
        -0.00027940460176353855,
        1.5985973073340157e-5,
        0.0001479453886246249,
        0.00018981802478182228,
        0.00019349000284057082,
        0.00019352329290027426,
        0.00021080020987809312,
        0.00025570584368471883,
        0.00033087596910938584,
        0.0004335379809615361,
        0.0005574724000165859,
        0.0006946224953132242,
        0.0008363795683496787,
        0.0009745714447263803,
        0.0011021817187824752,
        0.0012138272967735964,
        0.0013060217841383592,
        0.0013772522624009842,
        0.0014278970012575132,
        0.0014600116513930274,
        0.0014770114635773233,
        0.0014832770795864666,
        0.0014837114404976704,
        0.0014832753579049243,
        0.0014865292936028145,
        0.0014972088932859709,
        0.001517861819811571,
        0.0015495734315723398,
        0.0015918088515274847,
        0.0016423989724389887,
        0.0016976979438607129,
        0.001752939686427727,
        0.0018028209789933145,
        0.0018423386641610885,
        0.0018679085177596443,
        0.0018787933278071878,
        0.0018788677285135795,
        0.0018787473348672256,
        0.0018983097233542512,
        0.0019696348043573893,
        0.002140392131782019,
        0.0024777026954568094,
        0.0030725027418563554,
    ]
    w7 = owa_l_moment_crm(
        50;
        k = 8,
        method = :SS,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict(
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 400),
            ),
        ),
    )
    @test isapprox(w7, w7t, rtol = 3e-3)

    w8t = [
        -0.02,
        -0.01799382753997565,
        -0.016106415726070942,
        -0.01433328577805163,
        -0.01267003504915822,
        -0.011112337027033443,
        -0.00965594133454403,
        -0.008296673730502196,
        -0.007030436110292338,
        -0.0058532065064084825,
        -0.004761039088907937,
        -0.0037500641657866787,
        -0.0028164881832819615,
        -0.001956593726107653,
        -0.0011667395176277963,
        -0.0004433604199738835,
        0.00021703256588862934,
        0.000817852300139099,
        0.0013624355041202414,
        0.001854042760390961,
        0.0022958585128079537,
        0.002690991066630589,
        0.0030424725886435766,
        0.0033532591072919058,
        0.0036262305128225676,
        0.003864190557427558,
        0.004069866855382658,
        0.004245910883176499,
        0.0043948979796244085,
        0.004519327345961534,
        0.00462162204590975,
        0.00470412900571286,
        0.004769119014134557,
        0.004818786722413688,
        0.004855250644171293,
        0.0048805531552639295,
        0.004896660493577785,
        0.0049054627587580705,
        0.004908773911868193,
        0.00490833177497323,
        0.004905798030642178,
        0.004902758221363486,
        0.0049007217488683825,
        0.004901121873356485,
        0.004905315712618203,
        0.004914584241048418,
        0.004930132288545967,
        0.004953088539293399,
        0.004984505530411521,
        0.005025359650483268,
    ]
    w8 = owa_l_moment_crm(
        50;
        k = 8,
        method = :SD,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict(
            :ECOS => Dict(
                :solver => ECOS.Optimizer,
                :params => Dict("verbose" => false, "maxit" => 400),
            ),
        ),
    )
    @test isapprox(w8, w8t, rtol = 2e-3)
end
