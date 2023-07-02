
using DataFrames

abstract type AbstractPortfolio end
const UnionBoolNothing = Union{Bool, Nothing}
const UnionRealNothing = Union{<:Real, Nothing}
const UnionIntegerNothing = Union{<:Integer, Nothing}
const UnionVecNothing = Union{Vector{<:Real}, Nothing}
const UnionMtxNothing = Union{Matrix{<:Real}, Nothing}
const UnionDataFrameNothing = Union{DataFrame, Nothing}

mutable struct Portfolio{
    # Portfolio characteristics
    r,
    s,
    us,
    ul,
    ssl,
    mnea,
    mna,
    rf,
    l,
    # Risk parameters
    a,
    as,
    b,
    bs,
    k,
    mnak,
    # Benchmark constraints
    kb,
    ato,
    to,
    ate,
    te,
    rbi,
    bw,
    # Risk and return constraints
    ami,
    bvi,
    rbv,
    ler,
    ud,
    uk,
    umad,
    ugmd,
    usd,
    usk,
    uflpm,
    uslpm,
    ucvar,
    utg,
    uevar,
    urvar,
    uwr,
    ur,
    urcvar,
    urtg,
    umd,
    uad,
    ucdar,
    uedar,
    urdar,
    uui,
    # Optimisation model inputs
    tmu,
    tcov,
    tkurt,
    tskurt,
    tl2,
    ts2,
    tmuf,
    tcovf,
    tmufm,
    tcovfm,
    tmubl,
    tcovbl,
    tmublf,
    tcovblf,
    trfm,
    tevar,
    tedar,
    trvar,
    trdar,
    tcovl,
    tcovu,
    tcovmu,
    tcovs,
    tdmu,
    tkmu,
    tks,
    topt,
    trpopt,
    trrpopt,
    twcopt,
    tlim,
    tfront,
    tsolv,
    tsolvp,
} <: AbstractPortfolio
    # Portfolio characteristics.
    returns::r
    short::s
    upper_short::us
    upper_long::ul
    sum_short_long::ssl
    min_number_effective_assets::mnea
    max_number_assets::mna
    returns_factors::rf
    loadings::l
    # Risk parameters
    alpha::a
    a_sim::as
    beta::b
    b_sim::bs
    kappa::k
    max_num_assets_kurt::mnak
    # Benchmark constraints
    kind_benchmark::kb
    allow_turnover::ato
    turnover::to
    allow_tracking_err::ate
    tracking_err::te
    returns_benchmark_index::rbi
    benchmark_weights::bw
    # Risk and return constraints
    a_mtx_inequality::ami
    b_vec_inequality::bvi
    risk_budget_vec::rbv
    lower_expected_return::ler
    upper_deviation::ud
    upper_kurtosis::uk
    upper_mean_absolute_deviation::umad
    upper_gini_mean_difference::ugmd
    upper_semi_deviation::usd
    upper_semi_kurtosis::usk
    upper_first_lower_partial_moment::uflpm
    upper_second_lower_partial_moment::uslpm
    upper_conditional_value_at_risk::ucvar
    upper_tail_gini::utg
    upper_entropic_value_at_risk::uevar
    upper_relativistic_value_at_risk::urvar
    upper_worst_realisation::uwr
    upper_range::ur
    upper_range_conditional_value_at_risk::urcvar
    upper_range_tail_gini::urtg
    upper_max_drawdown::umd
    upper_average_drawdown::uad
    upper_conditional_drawdown_at_risk::ucdar
    upper_entropic_drawdown_at_risk::uedar
    upper_relativistic_drawdown_at_risk::urdar
    upper_ulcer_index::uui
    # Optimisation model inputs
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    mu_f::tmuf
    cov_f::tcovf
    mu_fm::tmufm
    cov_fm::tcovfm
    mu_bl::tmubl
    cov_bl::tcovbl
    mu_bl_fm::tmublf
    cov_bl_fm::tcovblf
    returns_fm::trfm
    z_EVaR::tevar
    z_EDaR::tedar
    z_RVaR::trvar
    z_RDaR::trdar
    # Inputs of Worst Case Optimization Models
    cov_l::tcovl
    cov_u::tcovu
    cov_mu::tcovmu
    cov_sigma::tcovs
    d_mu::tdmu
    k_mu::tkmu
    k_sigma::tks
    # Optimal portfolios
    optimal::topt
    rp_optimal::trpopt
    rrp_optimal::trrpopt
    wc_optimal::twcopt
    limits::tlim
    frontier::tfront
    # Solver params
    solvers::tsolv
    sol_params::tsolvp
end

function Portfolio(;
    # Portfolio characteristics.
    returns::DataFrame,
    short::Bool = false,
    upper_short::Real = 0.2,
    upper_long::Real = 1.0,
    sum_short_long::Real = 0.1,
    min_number_effective_assets::Integer = -1,
    max_number_assets::Integer = -1,
    returns_factors::DataFrame = DataFrame(),
    loadings::DataFrame = DataFrame(),
    # Risk parameters
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta::Real = 0.0,
    b_sim::Integer = 0,
    kappa::Real = 0.3,
    max_num_assets_kurt::Integer = 50,
    # Benchmark constraints
    kind_benchmark::Bool = true,
    allow_turnover::Bool = false,
    turnover::Real = 0.05,
    allow_tracking_err::Bool = false,
    tracking_err::Real = 0.05,
    returns_benchmark_index::DataFrame = DataFrame(),
    benchmark_weights::DataFrame = DataFrame(),
    # Risk and return constraints
    a_mtx_inequality::Matrix{<:Real} = Array{Float64}(undef, 0, 0),
    b_vec_inequality::Vector{<:Real} = Array{Float64}(undef, 0),
    risk_budget_vec::Vector{<:Real} = Array{Float64}(undef, 0),
    lower_expected_return::Real = -Inf,
    upper_deviation::Real = Inf,
    upper_kurtosis::Real = Inf,
    upper_mean_absolute_deviation::Real = Inf,
    upper_gini_mean_difference::Real = Inf,
    upper_semi_deviation::Real = Inf,
    upper_semi_kurtosis::Real = Inf,
    upper_first_lower_partial_moment::Real = Inf,
    upper_second_lower_partial_moment::Real = Inf,
    upper_conditional_value_at_risk::Real = Inf,
    upper_tail_gini::Real = Inf,
    upper_entropic_value_at_risk::Real = Inf,
    upper_relativistic_value_at_risk::Real = Inf,
    upper_worst_realisation::Real = Inf,
    upper_range::Real = Inf,
    upper_range_conditional_value_at_risk::Real = Inf,
    upper_range_tail_gini::Real = Inf,
    upper_max_drawdown::Real = Inf,
    upper_average_drawdown::Real = Inf,
    upper_conditional_drawdown_at_risk::Real = Inf,
    upper_entropic_drawdown_at_risk::Real = Inf,
    upper_relativistic_drawdown_at_risk::Real = Inf,
    upper_ulcer_index::Real = Inf,
    # Optimisation model inputs
    mu::Real = -Inf,
    cov::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    kurt::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    skurt::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    L_2::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    S_2::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_f::Real = -Inf,
    cov_f::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_fm::Real = -Inf,
    cov_fm::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_bl::Real = -Inf,
    cov_bl::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_bl_fm::Real = -Inf,
    cov_bl_fm::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    returns_fm::DataFrame = DataFrame(),
    z_EVaR::Real = -Inf,
    z_EDaR::Real = -Inf,
    z_RVaR::Real = -Inf,
    z_RDaR::Real = -Inf,
    # Inputs of Worst Case Optimization Models
    cov_l::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_u::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_mu::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_sigma::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    d_mu::Real = -Inf,
    k_mu::Real = -Inf,
    k_sigma::Real = -Inf,
    # Optimal portfolios
    optimal::DataFrame = DataFrame(),
    rp_optimal::DataFrame = DataFrame(),
    rrp_optimal::DataFrame = DataFrame(),
    wc_optimal::DataFrame = DataFrame(),
    limits::DataFrame = DataFrame(),
    frontier::DataFrame = DataFrame(),
    # Solver params
    solvers::Vector{AbstractString} = Vector{AbstractString}(undef, 0),
    sol_params::Dict = Dict(),
)
    return Portfolio(
        # Portfolio characteristics.
        returns,
        short,
        upper_short,
        upper_long,
        sum_short_long,
        min_number_effective_assets,
        max_number_assets,
        returns_factors,
        loadings,
        # Risk parameters
        alpha,
        a_sim,
        beta,
        b_sim,
        kappa,
        max_num_assets_kurt,
        # Benchmark constraints
        kind_benchmark,
        allow_turnover,
        turnover,
        allow_tracking_err,
        tracking_err,
        returns_benchmark_index,
        benchmark_weights,
        # Risk and return constraints
        a_mtx_inequality,
        b_vec_inequality,
        risk_budget_vec,
        lower_expected_return,
        upper_deviation,
        upper_kurtosis,
        upper_mean_absolute_deviation,
        upper_gini_mean_difference,
        upper_semi_deviation,
        upper_semi_kurtosis,
        upper_first_lower_partial_moment,
        upper_second_lower_partial_moment,
        upper_conditional_value_at_risk,
        upper_tail_gini,
        upper_entropic_value_at_risk,
        upper_relativistic_value_at_risk,
        upper_worst_realisation,
        upper_range,
        upper_range_conditional_value_at_risk,
        upper_range_tail_gini,
        upper_max_drawdown,
        upper_average_drawdown,
        upper_conditional_drawdown_at_risk,
        upper_entropic_drawdown_at_risk,
        upper_relativistic_drawdown_at_risk,
        upper_ulcer_index,
        # Optimisation model inputs
        mu,
        cov,
        kurt,
        skurt,
        L_2,
        S_2,
        mu_f,
        cov_f,
        mu_fm,
        cov_fm,
        mu_bl,
        cov_bl,
        mu_bl_fm,
        cov_bl_fm,
        returns_fm,
        z_EVaR,
        z_EDaR,
        z_RVaR,
        z_RDaR,
        # Inputs of Worst Case Optimization Models
        cov_l,
        cov_u,
        cov_mu,
        cov_sigma,
        d_mu,
        k_mu,
        k_sigma,
        # Optimal portfolios
        optimal,
        rp_optimal,
        rrp_optimal,
        wc_optimal,
        limits,
        frontier,
        # Solver params
        solvers,
        sol_params,
    )
end

export AbstractPortfolio, Portfolio#, HCPortfolio, OWAPortfolio