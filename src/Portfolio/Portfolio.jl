
using DataFrames

abstract type AbstractPortfolio end
const UnionBoolNothing = Union{Bool, Nothing}
const UnionRealNothing = Union{<:Real, Nothing}
const UnionIntegerNothing = Union{<:Integer, Nothing}
const UnionVecNothing = Union{Vector{<:Real}, Nothing}
const UnionMtxNothing = Union{Matrix{<:Real}, Nothing}
const UnionDataFrameNothing = Union{DataFrame, Nothing}

mutable struct Portfolio{
    # Portfolio characteristics.
    r,
    s,
    us,
    ul,
    ssl,
    mnea,
    mna,
    rf,
    l,
    # Risk parameters.
    a,
    as,
    b,
    bs,
    k,
    mnak,
    # Benchmark constraints.
    kb,
    ato,
    to,
    ate,
    te,
    rbi,
    bw,
    ami,
    bvi,
    rbv,
    # Risk and return constraints.
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
    # Risk parameters.
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
    a_mtx_inequality::ami
    b_vec_inequality::bvi
    risk_budget_vec::rbv
    # Risk and return constraints
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
    # Optimisation model inputs.
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
end

function Portfolio(;
    returns::DataFrame,
    short::Bool = false,
    upper_short::Real = 0.2,
    upper_long::Real = 1.0,
    sum_short_long::Real = 0.1,
    min_number_effective_assets::Integer = -1,
    max_number_assets::Integer = -1,
    returns_factors::DataFrame = DataFrame(),
    loadings::DataFrame = DataFrame(),
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta::Real = 0.0,
    b_sim::Integer = 0,
    kappa::Real = 0.3,
    max_num_assets_kurt::Integer = 50,
    kind_benchmark::Bool = true,
    allow_turnover::Bool = false,
    turnover::Real = 0.05,
    allow_tracking_err::Bool = false,
    tracking_err::Real = 0.05,
    returns_benchmark_index::DataFrame = DataFrame(),
    benchmark_weights::DataFrame = DataFrame(),
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
)
    return Portfolio(
        returns,
        short,
        upper_short,
        upper_long,
        sum_short_long,
        min_number_effective_assets,
        max_number_assets,
        returns_factors,
        loadings,
        alpha,
        a_sim,
        beta,
        b_sim,
        kappa,
        max_num_assets_kurt,
        kind_benchmark,
        allow_turnover,
        turnover,
        allow_tracking_err,
        tracking_err,
        returns_benchmark_index,
        benchmark_weights,
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
        -Inf,
        Matrix{Float64}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0),
        -Inf,
        Matrix{Float64}(undef, 0, 0),
        -Inf,
        Matrix{Float64}(undef, 0, 0),
        -Inf,
        Matrix{Float64}(undef, 0, 0),
        -Inf,
        Matrix{Float64}(undef, 0, 0),
        DataFrame(),
        -Inf,
        -Inf,
        -Inf,
        -Inf,
    )
end

export AbstractPortfolio, Portfolio#, HCPortfolio, OWAPortfolio