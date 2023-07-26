
using DataFrames, JuMP

abstract type AbstractPortfolio end
mutable struct Portfolio{
    # Portfolio characteristics
    r,
    s,
    us,
    ul,
    ssl,
    mnea,
    mna,
    mnaf,
    tretf,
    l,
    # Risk parameters
    ai,
    a,
    as,
    tat,
    bi,
    b,
    bs,
    k,
    tiat,
    lnk,
    topk,
    tomk,
    tk2,
    tkinv,
    tinvopk,
    tinvomk,
    mnak,
    # Benchmark constraints
    to,
    tobw,
    kte,
    te,
    rbi,
    bw,
    # Risk and return constraints
    ami,
    bvi,
    rbv,
    ler,
    ud,
    umad,
    usd,
    ucvar,
    uwr,
    uflpm,
    uslpm,
    umd,
    uad,
    ucdar,
    uuci,
    uevar,
    uedar,
    ugmd,
    utg,
    ur,
    urcvar,
    urtg,
    uk,
    usk,
    urvar,
    urdar,
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
    tf,
    toptpar,
    tmod,
} <: AbstractPortfolio
    # Portfolio characteristics.
    returns::r
    short::s
    short_u::us
    long_u::ul
    sum_short_long::ssl
    min_number_effective_assets::mnea
    max_number_assets::mna
    max_number_assets_factor::mnaf
    returns_factors::tretf
    loadings::l
    # Risk parameters.
    alpha_i::ai
    alpha::a
    a_sim::as
    at::tat
    beta_i::bi
    beta::b
    b_sim::bs
    kappa::k
    invat::tiat
    ln_k::lnk
    opk::topk
    omk::tomk
    invkappa2::tk2
    invk::tkinv
    invopk::tinvopk
    invomk::tinvomk
    max_num_assets_kurt::mnak
    # Benchmark constraints.
    turnover::to
    turnover_weights::tobw
    kind_tracking_err::kte
    tracking_err::te
    tracking_err_returns::rbi
    tracking_err_weights::bw
    # Risk and return constraints.
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget::rbv
    mu_l::ler
    dev_u::ud
    mad_u::umad
    sdev_u::usd
    cvar_u::ucvar
    wr_u::uwr
    flpm_u::uflpm
    slpm_u::uslpm
    mdd_u::umd
    add_u::uad
    cdar_u::ucdar
    uci_u::uuci
    evar_u::uevar
    edar_u::uedar
    gmd_u::ugmd
    tg_u::utg
    rg_u::ur
    rcvar_u::urcvar
    rtg_u::urtg
    krt_u::uk
    skrt_u::usk
    rvar_u::urvar
    rdar_u::urdar
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
    z_evar::tevar
    z_edar::tedar
    z_rvar::trvar
    z_rdar::trdar
    # Inputs of Worst Case Optimization Models.
    cov_l::tcovl
    cov_u::tcovu
    cov_mu::tcovmu
    cov_sigma::tcovs
    d_mu::tdmu
    k_mu::tkmu
    k_sigma::tks
    # Optimal portfolios.
    p_optimal::topt
    rp_optimal::trpopt
    rrp_optimal::trrpopt
    wc_optimal::twcopt
    limits::tlim
    frontier::tfront
    # Solver params.
    solvers::tsolv
    opt_params::toptpar
    fail::tf
    model::tmod
end

function Portfolio(;
    # Portfolio characteristics.
    returns,
    short::Bool = false,
    short_u::Real = 0.2,
    long_u::Real = 1.0,
    sum_short_long::Real = 1,
    min_number_effective_assets::Integer = 0,
    max_number_assets::Integer = 0,
    max_number_assets_factor::Real = 100_000,
    returns_factors = DataFrame(),
    loadings = DataFrame(),
    # Risk parameters.
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Real = Inf,
    beta::Real = Inf,
    b_sim::Integer = -1,
    kappa::Real = 0.3,
    max_num_assets_kurt::Integer = 0,
    # Benchmark constraints.
    turnover::Real = Inf,
    turnover_weights = DataFrame(),
    kind_tracking_err::Symbol = :weights,
    tracking_err::Real = Inf,
    tracking_err_returns = DataFrame(),
    tracking_err_weights = DataFrame(),
    # Risk and return constraints.
    a_mtx_ineq = Matrix{Float64}(undef, 0, 0),
    b_vec_ineq = Vector{Float64}(undef, 0),
    risk_budget = DataFrame(),
    mu_l::Real = -Inf,
    dev_u::Real = Inf,
    mad_u::Real = Inf,
    sdev_u::Real = Inf,
    cvar_u::Real = Inf,
    wr_u::Real = Inf,
    flpm_u::Real = Inf,
    slpm_u::Real = Inf,
    mdd_u::Real = Inf,
    add_u::Real = Inf,
    cdar_u::Real = Inf,
    uci_u::Real = Inf,
    evar_u::Real = Inf,
    edar_u::Real = Inf,
    gmd_u::Real = Inf,
    tg_u::Real = Inf,
    rg_u::Real = Inf,
    rcvar_u::Real = Inf,
    rtg_u::Real = Inf,
    krt_u::Real = Inf,
    skrt_u::Real = Inf,
    rvar_u::Real = Inf,
    rdar_u::Real = Inf,
    # Optimisation model inputs.
    mu = DataFrame(),
    cov = DataFrame(),
    kurt = DataFrame(),
    skurt = DataFrame(),
    L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
    S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
    mu_f = DataFrame(),
    cov_f = DataFrame(),
    mu_fm = DataFrame(),
    cov_fm = DataFrame(),
    mu_bl = DataFrame(),
    cov_bl = DataFrame(),
    mu_bl_fm = DataFrame(),
    cov_bl_fm = DataFrame(),
    returns_fm = DataFrame(),
    z_evar::Real = -Inf,
    z_edar::Real = -Inf,
    z_rvar::Real = -Inf,
    z_rdar::Real = -Inf,
    # Inputs of Worst Case Optimization Models.
    cov_l = DataFrame(),
    cov_u = DataFrame(),
    cov_mu = DataFrame(),
    cov_sigma = DataFrame(),
    d_mu::Real = -Inf,
    k_mu::Real = -Inf,
    k_sigma::Real = -Inf,
    # Optimal portfolios.
    p_optimal = DataFrame(),
    rp_optimal = DataFrame(),
    rrp_optimal = DataFrame(),
    wc_optimal = DataFrame(),
    limits = DataFrame(),
    frontier = DataFrame(),
    # Solutions.
    solvers::AbstractDict = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
    model = JuMP.Model(),
)
    return Portfolio(
        # Portfolio characteristics.
        returns,
        short,
        short_u,
        long_u,
        sum_short_long,
        min_number_effective_assets,
        max_number_assets,
        max_number_assets_factor,
        returns_factors,
        loadings,
        # Risk parameters.
        alpha_i,
        alpha,
        a_sim,
        zero(eltype(alpha)),
        beta_i,
        beta,
        b_sim,
        kappa,
        zero(eltype(kappa)),
        zero(eltype(kappa)),
        zero(eltype(kappa)),
        zero(eltype(kappa)),
        zero(eltype(kappa)),
        zero(eltype(kappa)),
        zero(eltype(kappa)),
        zero(eltype(kappa)),
        max_num_assets_kurt,
        # Benchmark constraints.
        turnover,
        turnover_weights,
        kind_tracking_err,
        tracking_err,
        tracking_err_returns,
        tracking_err_weights,
        # Risk and return constraints.
        a_mtx_ineq,
        b_vec_ineq,
        risk_budget,
        mu_l,
        dev_u,
        mad_u,
        sdev_u,
        cvar_u,
        wr_u,
        flpm_u,
        slpm_u,
        mdd_u,
        add_u,
        cdar_u,
        uci_u,
        evar_u,
        edar_u,
        gmd_u,
        tg_u,
        rg_u,
        rcvar_u,
        rtg_u,
        krt_u,
        skrt_u,
        rvar_u,
        rdar_u,
        # Optimisation model inputs.
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
        z_evar,
        z_edar,
        z_rvar,
        z_rdar,
        # Inputs of Worst Case Optimization Models.
        cov_l,
        cov_u,
        cov_mu,
        cov_sigma,
        d_mu,
        k_mu,
        k_sigma,
        # Optimal portfolios.
        p_optimal,
        rp_optimal,
        rrp_optimal,
        wc_optimal,
        limits,
        frontier,
        # Solutions.
        solvers,
        opt_params,
        fail,
        model,
    )
end

# Asset statistics.
include("_portfolio_asset_statistics.jl")
# Risk constants and functions.
include("_portfolio_risk_setup.jl")
# Optimisation functions.
include("_portfolio_optim_setup.jl")

export AbstractPortfolio, Portfolio#, HCPortfolio, OWAPortfolio