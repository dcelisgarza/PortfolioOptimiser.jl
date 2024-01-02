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
