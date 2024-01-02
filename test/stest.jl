using COSMO,
    CovarianceEstimation,
    CSV,
    Clarabel,
    HiGHS,
    LinearAlgebra,
    OrderedCollections,
    PortfolioOptimiser,
    Statistics,
    StatsBase,
    Test,
    TimeSeries,
    SCS

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

cols = [:RRC, :AMZN, :JPM, :MA, :WMT]
PortfolioOptimiser.DenoiseMethods

portfolio = Portfolio(; prices = prices)

asset_statistics!(portfolio, calc_kurt = false, jlogo = false)
cov1 = portfolio.cov

asset_statistics!(portfolio, calc_kurt = false, jlogo = true)
cov2 = portfolio.cov

covt2 = reshape(
    [
        0.00020997582228184732,
        9.80367463317547e-5,
        0.00014077374362954785,
        0.000118227682464823,
        0.000158396388639706,
        4.1342495129394216e-5,
        9.741361595383928e-5,
        3.088472057695862e-5,
        8.723400768938228e-5,
        4.695561415789392e-5,
        2.410793390349615e-5,
        7.697761692810684e-5,
        5.0213913154013656e-5,
        4.6572979116003736e-5,
        6.041300540147272e-5,
        4.007622815237796e-5,
        9.852662188755092e-5,
        4.258121751647648e-5,
        7.89175024133544e-5,
        8.308464134357729e-5,
        9.80367463317547e-5,
        0.00021152344684106057,
        8.070483057492826e-5,
        7.345023656198462e-5,
        8.622173019184438e-5,
        4.428661971580527e-5,
        0.0001302364476786512,
        3.080510219847445e-5,
        9.441947447077733e-5,
        5.0793961901943507e-5,
        2.5625742533150004e-5,
        6.843664330541154e-5,
        5.4343362302752406e-5,
        4.824450286505279e-5,
        6.332376221790689e-5,
        4.3386684391988476e-5,
        9.25843005786566e-5,
        4.325691689179076e-5,
        8.569427917973423e-5,
        5.8006924293976724e-5,
        0.00014077374362954785,
        8.070483057492826e-5,
        0.0002572414426920626,
        9.907520278111455e-5,
        0.00015334295879523456,
        3.975323411216913e-5,
        9.261188549036716e-5,
        3.091630885459727e-5,
        8.337939553449835e-5,
        4.488878095701139e-5,
        2.3287299979458455e-5,
        7.756460767061071e-5,
        4.799509093176812e-5,
        4.566092116229602e-5,
        5.883603777967896e-5,
        3.829743915517546e-5,
        0.00010166996305493103,
        4.2204522581317864e-5,
        7.526496401459766e-5,
        8.802924838121843e-5,
        0.000118227682464823,
        7.345023656198462e-5,
        9.907520278111455e-5,
        0.0004006277432140748,
        0.00013259642444595355,
        3.656822111047864e-5,
        8.81027344671094e-5,
        2.9742503351053513e-5,
        7.622510549864218e-5,
        4.102095002874271e-5,
        2.1533972054045375e-5,
        6.244898023158416e-5,
        4.3865383504766376e-5,
        4.293671627681014e-5,
        5.492306428908421e-5,
        3.4993440090768056e-5,
        0.00010093601577900456,
        4.0169975051465934e-5,
        6.857256380684272e-5,
        6.258847359456461e-5,
        0.000158396388639706,
        8.622173019184438e-5,
        0.00015334295879523456,
        0.00013259642444595355,
        0.0003311044630153933,
        4.038058059540706e-5,
        9.583901646637e-5,
        3.161780823422457e-5,
        8.463726955701539e-5,
        4.555542067291654e-5,
        2.367290551940059e-5,
        7.328255734548563e-5,
        4.871359007313472e-5,
        4.653304028701636e-5,
        5.9897079161003395e-5,
        3.886927014428795e-5,
        0.00010448970386759226,
        4.3091145718550925e-5,
        7.634401145955473e-5,
        7.776687178020369e-5,
        4.1342495129394216e-5,
        4.428661971580527e-5,
        3.975323411216913e-5,
        3.656822111047864e-5,
        4.038058059540706e-5,
        0.00017908772525119732,
        6.489402909991512e-5,
        2.5028507197328294e-5,
        9.962273724791356e-5,
        8.680193208086006e-5,
        5.2773386401160835e-5,
        6.097685395858883e-5,
        7.124243170541177e-5,
        6.946631411775527e-5,
        0.00010068570060338066,
        5.388381392162792e-5,
        5.320627263008016e-5,
        4.007421098942513e-5,
        9.12892869669783e-5,
        3.667147023058035e-5,
        9.741361595383928e-5,
        0.0001302364476786512,
        9.261188549036716e-5,
        8.81027344671094e-5,
        9.583901646637e-5,
        6.489402909991512e-5,
        0.0016481856234771093,
        4.4245960137320026e-5,
        0.00013873440279035015,
        7.462295424808144e-5,
        3.7471916631902205e-5,
        9.514621072757074e-5,
        7.984666848657701e-5,
        7.00488809626995e-5,
        9.224397895599859e-5,
        6.37537383802109e-5,
        0.00013058586088202263,
        6.246142917350008e-5,
        0.00012602293719272406,
        7.362065024292624e-5,
        3.088472057695862e-5,
        3.080510219847445e-5,
        3.091630885459727e-5,
        2.9742503351053513e-5,
        3.161780823422457e-5,
        2.5028507197328294e-5,
        4.4245960137320026e-5,
        0.00016353541427233863,
        5.178413394383988e-5,
        2.8116150979283176e-5,
        1.4697333895237551e-5,
        3.470976783146292e-5,
        2.9917391605612994e-5,
        2.9011081293456074e-5,
        3.7188963010058494e-5,
        2.387232528464302e-5,
        4.601421574505308e-5,
        4.450058196330506e-5,
        4.723912468936478e-5,
        2.563115597593835e-5,
        8.723400768938228e-5,
        9.441947447077733e-5,
        8.337939553449835e-5,
        7.622510549864218e-5,
        8.463726955701539e-5,
        9.962273724791356e-5,
        0.00013873440279035015,
        5.178413394383988e-5,
        0.00027619979891424333,
        0.0001258404123778024,
        5.519747560963825e-5,
        0.00014173738855470427,
        0.00014845072824926932,
        9.14050552075809e-5,
        0.00013536187377374283,
        0.00011844433793628759,
        0.0001098532111513027,
        7.870483515637142e-5,
        0.00020117930541138597,
        7.788398396676921e-5,
        4.695561415789392e-5,
        5.0793961901943507e-5,
        4.488878095701139e-5,
        4.102095002874271e-5,
        4.555542067291654e-5,
        8.680193208086006e-5,
        7.462295424808144e-5,
        2.8116150979283176e-5,
        0.0001258404123778024,
        0.00024332211270197769,
        3.679005664812955e-5,
        7.277466203867355e-5,
        0.00014574432957288077,
        5.726866239843983e-5,
        8.137270380669477e-5,
        0.00010005844523863936,
        5.91066668074322e-5,
        4.341172302514213e-5,
        0.0001081767220673213,
        4.1999000458699555e-5,
        2.410793390349615e-5,
        2.5625742533150004e-5,
        2.3287299979458455e-5,
        2.1533972054045375e-5,
        2.367290551940059e-5,
        5.2773386401160835e-5,
        3.7471916631902205e-5,
        1.4697333895237551e-5,
        5.519747560963825e-5,
        3.679005664812955e-5,
        0.00010534978848760101,
        3.44728314787607e-5,
        3.476298535587925e-5,
        4.85318909193091e-5,
        5.9253564059689944e-5,
        2.7163178806229955e-5,
        3.156891859240837e-5,
        2.412239965940406e-5,
        5.140975472359215e-5,
        2.1234135644375805e-5,
        7.697761692810684e-5,
        6.843664330541154e-5,
        7.756460767061071e-5,
        6.244898023158416e-5,
        7.328255734548563e-5,
        6.097685395858883e-5,
        9.514621072757074e-5,
        3.470976783146292e-5,
        0.00014173738855470427,
        7.277466203867355e-5,
        3.44728314787607e-5,
        0.0007066882567333451,
        7.998599079162287e-5,
        6.019307841605818e-5,
        8.302509502935606e-5,
        6.388032472082356e-5,
        8.383990731316273e-5,
        5.1517912988504366e-5,
        0.00012182889805680229,
        0.00010811590662763469,
        5.0213913154013656e-5,
        5.4343362302752406e-5,
        4.799509093176812e-5,
        4.3865383504766376e-5,
        4.871359007313472e-5,
        7.124243170541177e-5,
        7.984666848657701e-5,
        2.9917391605612994e-5,
        0.00014845072824926932,
        0.00014574432957288077,
        3.476298535587925e-5,
        7.998599079162287e-5,
        0.0019439677607081564,
        5.609119098579761e-5,
        8.143828238380131e-5,
        0.00023386599375945153,
        6.320360210863747e-5,
        4.576565505327077e-5,
        0.00011581541521370356,
        4.4872643276981e-5,
        4.6572979116003736e-5,
        4.824450286505279e-5,
        4.566092116229602e-5,
        4.293671627681014e-5,
        4.653304028701636e-5,
        6.946631411775527e-5,
        7.00488809626995e-5,
        2.9011081293456074e-5,
        9.14050552075809e-5,
        5.726866239843983e-5,
        4.85318909193091e-5,
        6.019307841605818e-5,
        5.609119098579761e-5,
        0.00014363521728517984,
        0.00015453720776819607,
        4.4250794945485826e-5,
        6.444039069141757e-5,
        5.1247884608506786e-5,
        8.77418274795769e-5,
        4.00572789453824e-5,
        6.041300540147272e-5,
        6.332376221790689e-5,
        5.883603777967896e-5,
        5.492306428908421e-5,
        5.9897079161003395e-5,
        0.00010068570060338066,
        9.224397895599859e-5,
        3.7188963010058494e-5,
        0.00013536187377374283,
        8.137270380669477e-5,
        5.9253564059689944e-5,
        8.302509502935606e-5,
        8.143828238380131e-5,
        0.00015453720776819607,
        0.0009937022125140933,
        6.413867647869594e-5,
        8.159223034002055e-5,
        6.341331115400408e-5,
        0.00012057821002300558,
        5.249442561506609e-5,
        4.007622815237796e-5,
        4.3386684391988476e-5,
        3.829743915517546e-5,
        3.4993440090768056e-5,
        3.886927014428795e-5,
        5.388381392162792e-5,
        6.37537383802109e-5,
        2.387232528464302e-5,
        0.00011844433793628759,
        0.00010005844523863936,
        2.7163178806229955e-5,
        6.388032472082356e-5,
        0.00023386599375945153,
        4.4250794945485826e-5,
        6.413867647869594e-5,
        0.0005054585590745463,
        5.040215722614186e-5,
        3.648074552416076e-5,
        9.256991406046915e-5,
        3.582548495164845e-5,
        9.852662188755092e-5,
        9.25843005786566e-5,
        0.00010166996305493103,
        0.00010093601577900456,
        0.00010448970386759226,
        5.320627263008016e-5,
        0.00013058586088202263,
        4.601421574505308e-5,
        0.0001098532111513027,
        5.91066668074322e-5,
        3.156891859240837e-5,
        8.383990731316273e-5,
        6.320360210863747e-5,
        6.444039069141757e-5,
        8.159223034002055e-5,
        5.040215722614186e-5,
        0.0001624387273643112,
        6.127832941435989e-5,
        9.838072451447103e-5,
        7.735879911530243e-5,
        4.258121751647648e-5,
        4.325691689179076e-5,
        4.2204522581317864e-5,
        4.0169975051465934e-5,
        4.3091145718550925e-5,
        4.007421098942513e-5,
        6.246142917350008e-5,
        4.450058196330506e-5,
        7.870483515637142e-5,
        4.341172302514213e-5,
        2.412239965940406e-5,
        5.1517912988504366e-5,
        4.576565505327077e-5,
        5.1247884608506786e-5,
        6.341331115400408e-5,
        3.648074552416076e-5,
        6.127832941435989e-5,
        0.00012202561764009536,
        7.235602099204462e-5,
        3.594821958665302e-5,
        7.89175024133544e-5,
        8.569427917973423e-5,
        7.526496401459766e-5,
        6.857256380684272e-5,
        7.634401145955473e-5,
        9.12892869669783e-5,
        0.00012602293719272406,
        4.723912468936478e-5,
        0.00020117930541138597,
        0.0001081767220673213,
        5.140975472359215e-5,
        0.00012182889805680229,
        0.00011581541521370356,
        8.77418274795769e-5,
        0.00012057821002300558,
        9.256991406046915e-5,
        9.838072451447103e-5,
        7.235602099204462e-5,
        0.00018490114558171504,
        7.089867226693548e-5,
        8.308464134357729e-5,
        5.8006924293976724e-5,
        8.802924838121843e-5,
        6.258847359456461e-5,
        7.776687178020369e-5,
        3.667147023058035e-5,
        7.362065024292624e-5,
        2.563115597593835e-5,
        7.788398396676921e-5,
        4.1999000458699555e-5,
        2.1234135644375805e-5,
        0.00010811590662763469,
        4.4872643276981e-5,
        4.00572789453824e-5,
        5.249442561506609e-5,
        3.582548495164845e-5,
        7.735879911530243e-5,
        3.594821958665302e-5,
        7.089867226693548e-5,
        0.00015789185651061675,
    ],
    20,
    20,
)

@test isapprox(cov1, cov(portfolio.returns))
@test isapprox(cov2, covt2)
@test !isapprox(cov1, cov2)

println("covt = reshape(", vec(portfolio.cov), ", 20, 20)")

# println("kurtt = reshape(", vec(portfolio.kurt), ", 5^2, 5^2)")
# println("skurtt = reshape(", vec(portfolio.skurt), ", 5^2, 5^2)")

asset_statistics!(
    portfolio::AbstractPortfolio;
    # # flags
    # calc_codep::Bool = true,
    # calc_cov::Bool = true,
    # calc_mu::Bool = true,
    # calc_kurt::Bool = true,
    # # cov_mtx
    # alpha::Real = 0.0,
    # cov_args::Tuple = (),
    # cov_est::CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
    # cov_func::Function = cov,
    # cov_kwargs::NamedTuple = (;),
    # cov_type::Symbol = portfolio.cov_type,
    # cov_weights::Union{AbstractWeights, Nothing} = nothing,
    # custom_cov::Union{AbstractMatrix, Nothing} = nothing,
    # denoise::Bool = false,
    # detone::Bool = false,
    # gs_threshold::Real = portfolio.gs_threshold,
    # jlogo::Bool = portfolio.jlogo,
    # kernel = ASH.Kernels.gaussian,
    # m::Integer = 10,
    # method::Symbol = :Fixed,
    # mkt_comp::Integer = 0,
    # n::Integer = 1000,
    # opt_args = (),
    # opt_kwargs = (;),
    # posdef_args::Tuple = (),
    # posdef_fix::Symbol = portfolio.posdef_fix,
    # posdef_func::Function = x -> x,
    # posdef_kwargs::NamedTuple = (;),
    # std_args::Tuple = (),
    # std_func::Function = std,
    # std_kwargs::NamedTuple = (;),
    # target_ret::Union{Real, AbstractVector{<:Real}} = 0.0,
    # # mean_vec
    # custom_mu::Union{AbstractVector, Nothing} = nothing,
    # mean_args::Tuple = (),
    # mean_func::Function = mean,
    # mean_kwargs::NamedTuple = (;),
    # mkt_ret::Union{AbstractVector, Nothing} = nothing,
    # mu_target::Symbol = :GM,
    # mu_type::Symbol = portfolio.mu_type,
    # mu_weights::Union{AbstractWeights, Nothing} = nothing,
    # rf::Real = 0.0,
    # # codep_dist_mtx
    # alpha_tail::Union{Real, Nothing} = isa(portfolio, HCPortfolio) ? portfolio.alpha_tail :
    #                                    nothing,
    # bins_info::Union{Symbol, Integer, Nothing} = isa(portfolio, HCPortfolio) ?
    #                                              portfolio.bins_info : nothing,
    # codep_type::Union{Symbol, Nothing} = isa(portfolio, HCPortfolio) ?
    #                                      portfolio.codep_type : nothing,
    # cor_args::Tuple = (),
    # cor_func::Function = cor,
    # cor_kwargs::NamedTuple = (;),
    # custom_cor::Union{AbstractMatrix, Nothing} = nothing,
    # dist_args::Tuple = (),
    # dist_func::Function = x -> sqrt.(clamp!((1 .- x) / 2, 0, 1)),
    # dist_kwargs::NamedTuple = (;),
    # custom_kurt::Union{AbstractMatrix, Nothing} = nothing,
    # custom_skurt::Union{AbstractMatrix, Nothing} = nothing,
    # uplo::Symbol = :L,
)

########################################
println("w1t = ", w1.weights, "\n")
println("w2t = ", w2.weights, "\n")
println("w3t = ", w3.weights, "\n")
println("w4t = ", w4.weights, "\n")
println("w5t = ", w5.weights, "\n")
println("w6t = ", w6.weights, "\n")
println("w7t = ", w7.weights, "\n")
println("w8t = ", w8.weights, "\n")
println("w9t = ", w9.weights, "\n")
println("w10t = ", w10.weights, "\n")
println("w11t = ", w11.weights, "\n")
println("w12t = ", w12.weights, "\n")
println("w13t = ", w13.weights, "\n")
println("w14t = ", w14.weights, "\n")
println("w15t = ", w15.weights, "\n")
println("w16t = ", w16.weights, "\n")
println("w17t = ", w17.weights, "\n")
println("w18t = ", w18.weights, "\n")
println("w19t = ", w19.weights, "\n")
#######################################

for rtol in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2.5e-1, 5e-1, 1e0]
    a1, a2 = [
        7.142507835689946e-16,
        1.3658938730114826e-16,
        3.660245424824713e-16,
        2.193390803434477e-15,
        3.284406887019229e-5,
        0.0,
        2.8833750726035664e-15,
        0.00013150046307600161,
        3.193448989948215e-11,
        3.958932996584078e-14,
        9.519307751356533e-15,
        1.1653842795246459e-15,
        6.332667663280077e-17,
        8.663055083008026e-15,
        0.0,
        0.0001434553602614673,
        0.000248114859820377,
        2.5929903745042446e-15,
        0.0002587616268781684,
        2.940443665661615e-15,
    ],
    [
        1.0164440246632186e-15,
        1.947149103649204e-16,
        5.210887173779382e-16,
        3.1205355142010583e-15,
        4.688388388960399e-5,
        0.0,
        4.101214763069201e-15,
        0.00018084128302159894,
        4.4529086809731815e-11,
        5.520599752652079e-14,
        1.327671290761093e-14,
        1.6026480043407179e-15,
        8.710431171009255e-17,
        1.2082764384039807e-14,
        0.0,
        0.0001972818254717155,
        0.00035417622374826036,
        3.612531699923921e-15,
        0.0003606161665688415,
        4.043783521933823e-15,
    ]
    if isapprox(a1, a2, rtol = rtol)
        println(", rtol = $(rtol)")
        break
    end
end
portfolio = Portfolio(
    prices = prices[(end - 200):end],
    solvers = OrderedDict(
        :Clarabel => Dict(
            :solver => Clarabel.Optimizer,
            :params => Dict("verbose" => false, "max_step_fraction" => 0.75),
        ),
        :COSMO => Dict(:solver => COSMO.Optimizer, :params => Dict("verbose" => false)),
    ),
)
asset_statistics!(portfolio)
w6 = opt_port!(
    portfolio;
    rf = rf,
    l = l,
    class = :Classic,
    type = :Trad,
    rm = :RTG,
    obj = :Utility,
    kelly = :Exact,
)

isapprox(1.0359036144810312, 1; atol = 0.04)
