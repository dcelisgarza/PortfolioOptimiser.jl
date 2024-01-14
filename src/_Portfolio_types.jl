"""
```julia
TrackingErrKinds = (:Weights, :Returns)
```

Available kinds of tracking errors for [`Portfolio`](@ref).

  - `:Weights`: provide a vector of asset weights which is used to compute the vector of benchmark returns,    - ``\\bm{b} = \\mathbf{X} \\bm{w}``,where ``\\bm{b}`` is the benchmark returns vector, ``\\mathbf{X}`` the ``T \\times{} N`` asset returns matrix, and ``\\bm{w}`` the asset weights vector.
  - `:Returns`: directly provide the vector of benchmark returns.
    The benchmark is then used as a reference to optimise a portfolio that tracks it up to a given error.
"""
const TrackingErrKinds = (:Weights, :Returns)

const NetworkMethods = (:None, :SDP, :IP)

"""
```julia
AbstractPortfolio
```

Abstract type for portfolios. Concrete portfolios subtype this see [`Portfolio`](@ref) and [`HCPortfolio`](@ref).
"""
abstract type AbstractPortfolio end

"""
```julia
mutable struct Portfolio{
    # Portfolio characteristics
    ast,    dat,    r,    s,    us,    ul,    ssl,    mnea,    mna,    mnaf,    tfa,    tfdat,    tretf,    l,    # Risk parameters
    msvt,    lpmt,    ai,    a,    as,    bi,    b,    bs,    k,    mnak,    # Benchmark constraints
    to,    tobw,    kte,    te,    rbi,    bw,    blbw,    # Risk and return constraints
    ami,    bvi,    rbv,    ler,    ud,    umad,    usd,    ucvar,    uwr,    uflpm,    uslpm,    umd,    uad,    ucdar,    uuci,    uevar,    uedar,    urvar,    urdar,    uk,    usk,    ugmd,    ur,    urcvar,    utg,    urtg,    uowa,    # Cusom OWA weights
    wowa,    # Optimisation model inputs
    tmu,    tcov,    tkurt,    tskurt,    tl2,    ts2,    tmuf,    tcovf,    tmufm,    tcovfm,    tmubl,    tcovbl,    tmublf,    tcovblf,    trfm,    tcovl,    tcovu,    tcovmu,    tcovs,    tdmu,    tkmu,    tks,    topt,    tz,    tlim,    tfront,    tsolv,    tf,    toptpar,    tmod,    # Allocation
    tlp,    taopt,    tasolv,    taoptpar,    taf,    tamod,} <: AbstractPortfolio
    # Portfolio characteristics
    assets::ast
    timestamps::dat
    returns::r
    short::s
    short_u::us
    long_u::ul
    min_number_effective_assets::mnea
    max_number_assets::mna
    max_number_assets_factor::mnaf
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    # Risk parameters
    msv_target::msvt
    lpm_target::lpmt
    alpha_i::ai
    alpha::a
    a_sim::as
    beta_i::bi
    beta::b
    b_sim::bs
    kappa::k
    max_num_assets_kurt::mnak
    # Benchmark constraints
    turnover::to
    turnover_weights::tobw
    kind_tracking_err::kte
    tracking_err::te
    tracking_err_returns::rbi
    tracking_err_weights::bw
    bl_bench_weights::blbw
    # Risk and return constraints
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget::rbv
    # Bounds constraints
    mu_l::ler
    sd_u::ud
    mad_u::umad
    ssd_u::usd
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
    rvar_u::urvar
    rdar_u::urdar
    kurt_u::uk
    skurt_u::usk
    gmd_u::ugmd
    rg_u::ur
    rcvar_u::urcvar
    tg_u::utg
    rtg_u::urtg
    owa_u::uowa
    # Custom OWA weights
    owa_w::wowa
    # Model statistics
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
    z::tz
    limits::tlim
    frontier::tfront
    # Solver params
    solvers::tsolv
    opt_params::toptpar
    fail::tf
    model::tmod
    # Allocation
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_solvers::tasolv
    alloc_params::taoptpar
    alloc_fail::taf
    alloc_model::tamod
end
```
Structure for convex portfolio optimisation.
# Portfolio characteristics
- `assets`: `Na×1` vector of assets, where $(_ndef(:a2)).
- `timestamps`: `T×1` vector of timestamps, where $(_tstr(:t1)).
- `returns`: `T×Na` matrix of asset returns, where $(_tstr(:t1)) and $(_ndef(:a2)).
- `short`: whether or not to allow negative weights, i.e. whether a portfolio accepts shorting.
- `short_u`: absolute value of the sum of all short (negative) weights.
- `long_u`: sum of all long (positive) weights.
- `min_number_effective_assets`: if non-zero, guarantees that at least number of assets make significant contributions to the final portfolio weights.
- `max_number_assets`: if non-zero, guarantees at most this number of assets make non-zero contributions to the final portfolio weights. Requires an optimiser that supports binary variables.
- `max_number_assets_factor`: scaling factor needed to create a decision variable when `max_number_assets` is non-zero.
- `f_assets`: `Nf×1` vector of factors, where $(_ndef(:f2)).
- `f_timestamps`: `T×1` vector of factor timestamps, where $(_tstr(:t1)).
- `f_returns`: `T×Nf` matrix of factor returns, where $(_ndef(:f2)).
- `loadings`: loadings matrix for black litterman models.
# Risk parameters
- `msv_target`: target value for for Absolute Deviation and Semivariance risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the mean returns vector `mu`.
    - Else the target is the value of `msv_target`. If `msv_target` is a vector, its length should be `Na`, where $(_ndef(:a2)).
- `lpm_target`: target value for the First and Second Lower Partial Moment risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the value of the risk free rate `rf`, provided in the [`optimise!`](@ref) function.
    - Else the target is the value of `lpm_target`. If `lpm_target` is a vector, its length should be `Na`, where $(_ndef(:a2)).
$(_isigdef("Tail Gini losses", :a))
$(_sigdef("VaR, CVaR, EVaR, RVaR, DaR, CDaR, EDaR, RDaR, CVaR losses, or Tail Gini losses, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :a))
$(_isigdef("Tail Gini gains", :b))
$(_sigdef("CVaR gains or Tail Gini gains, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :b))
- `kappa`: deformation parameter for relativistic risk measures (RVaR and RDaR).
- `max_num_assets_kurt`: maximum number of assets to use the full kurtosis model, if the number of assets surpases this value use the relaxed kurtosis model.
# Benchmark constraints
- `turnover`: if finite, define the maximum turnover deviations from `turnover_weights` to the optimised portfolio. Else the constraint is disabled.
- `turnover_weights`: target weights for turnover constraint.
    - The turnover constraint is defined as ``\\lvert w_{i} - \\hat{w}_{i}\\rvert \\leq e_{1} \\, \\forall\\, i \\in N``, where ``w_i`` is the optimal weight for the `i'th` asset, ``\\hat{w}_i`` target weight for the `i'th` asset, ``e_{1}`` is the value of the turnover, and $(_ndef(:a3)).
- `kind_tracking_err`: `:Weights` when providing a vector of asset weights for computing the tracking error benchmark from the asset returns, or `:Returns` to directly providing the tracking benchmark. See [`TrackingErrKinds`](@ref) for more information.
- `tracking_err`: if finite, define the maximum tracking error deviation. Else the constraint is disabled.
- `tracking_err_returns`: `T×1` vector of returns to be tracked, where $(_tstr(:t1)). When `kind_tracking_err == :Returns`, this is used directly tracking benchmark.
- `tracking_err_weights`: `Na×1` vector of weights, where $(_ndef(:a2)), when `kind_tracking_err == :Weights`, the returns benchmark is computed from the `returns` field of [`Portfolio`](@ref).
    - The tracking error is defined as ``\\sqrt{\\dfrac{1}{T-1}\\sum\\limits_{i=1}^{T}\\left(\\mathbf{X}_{i} \\bm{w} - b_{i}\\right)^{2}}\\leq e_{2}``, where ``\\mathbf{X}_{i}`` is the `i'th` observation (row) of the returns matrix ``\\mathbf{X}``, ``\\bm{w}`` is the vector of optimal asset weights, ``b_{i}`` is the `i'th` observation of the benchmark returns vector, ``e_{2}`` the tracking error, and $(_tstr(:t2)).
- `bl_bench_weights`: `Na×1` vector of benchmark weights for Black Litterman models, where $(_ndef(:a2)).
# Risk and return constraints
- `a_mtx_ineq`: `C×Na` A matrix of the linear asset constraints ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``, where `C` is the number of constraints, and $(_ndef(:a2)).
- `b_vec_ineq`: `C×1` B vector of the linear asset constraints ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``, where `C` is the number of constraints.
- `risk_budget`: `Na×1` risk budget constraint vector for risk parity optimisations, where $(_ndef(:a2)).
## Bounds constraints
The bounds constraints are only active if they are finite. They define lower bounds denoted by the suffix `_l`, and upper bounds denoted by the suffix `_u`, of various portfolio characteristics. The risk upper bounds are named after their corresponding [`RiskMeasures`](@ref) in lower case, they also bring the same solver requirements as their corresponding risk measure. Multiple bounds constraints can be active at any time but may make finding a solution infeasable.
- `mu_l`: mean expected return.
- `sd_u`: standard deviation.
- `mad_u`: max absolute devia.
- `ssd_u`: semi standard deviation.
- `cvar_u`: critical value at risk.
- `wr_u`: worst realisation.
- `flpm_u`: first lower partial moment.
- `slpm_u`: second lower partial moment.
- `mdd_u`: max drawdown.
- `add_u`: average drawdown.
- `cdar_u`: critical drawdown at risk.
- `uci_u`: ulcer index.
- `evar_u`: entropic value at risk.
- `edar_u`: entropic drawdown at risk.
- `rvar_u`: relativistic value at risk.
- `rdar_u`: relativistic drawdown at risk.
- `kurt_u`: square root kurtosis.
- `skurt_u`: square root semi kurtosis.
- `gmd_u`: gini mean difference.
- `rg_u`: range.
- `rcvar_u`: critical value at risk range.
- `tg_u`: tail gini.
- `rtg_u`: tail gini range.
- `owa_u`: custom ordered weight risk (use with `owa_w`).
# Custom OWA weights
- `owa_w`: `T×1` OWA vector, where $(_tstr(:t1)) containing. Useful for optimising higher OWA L-moments.
# Model statistics
- `mu`: $(_mudef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `cov`: $(_covdef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `kurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the cokurtosis matrix at instance construction. The cokurtosis matrix `kurt` can be computed by calling [`cokurt_mtx`](@ref).
- `skurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the semi cokurtosis matrix at instance construction. The semi cokurtosis matrix `skurt` can be computed by calling [`cokurt_mtx`](@ref).
- `L_2`: `(Na×Na) × ((Na×(Na+1)/2))` elimination matrix, where $(_ndef(:a2)). $(_dircomp("[`cokurt_mtx`](@ref)"))
- `S_2`: `((Na×(Na+1)/2)) × (Na×Na)` summation matrix, where $(_ndef(:a2)). $(_dircomp("[`cokurt_mtx`](@ref)"))
- `mu_f`: $(_mudef("factors", :f2)) $(_dircomp("[`factor_statistics!`](@ref)"))
- `cov_f`: $(_covdef("factors", :f2)) $(_dircomp("[`factor_statistics!`](@ref)"))
- `mu_fm`: $(_mudef("feature selected factors")) $(_dircomp("[`factor_statistics!`](@ref)"))
- `cov_fm`: $(_covdef("feature selected factors")) $(_dircomp("[`factor_statistics!`](@ref)"))
- `mu_bl`: $(_mudef("Black Litterman")) $(_dircomp("[`black_litterman_statistics!`](@ref)"))
- `cov_bl`: $(_covdef("Black Litterman")) $(_dircomp("[`black_litterman_statistics!`](@ref)"))
- `mu_bl_fm`: $(_mudef("Black Litterman feature selected factors")) $(_dircomp("[`black_litterman_factor_satistics!`](@ref)"))
- `cov_bl_fm`: $(_covdef("Black Litterman feature selected factors")) $(_dircomp("[`black_litterman_factor_satistics!`](@ref)"))
- `returns_fm`: `T×Na` matrix of feature selcted adjusted returns, where $(_tstr(:t1)) and $(_ndef(:a2)). $(_dircomp("[`factor_statistics!`](@ref)"))
# Inputs of Worst Case Optimization Models
- `cov_l`: $(_covdef("worst case lower bound asset")) $(_dircomp("[`wc_statistics!`](@ref)"))
- `cov_u`: $(_covdef("worst case upper bound asset")) $(_dircomp("[`wc_statistics!`](@ref)"))
- `cov_mu`: $(_covdef("estimation errors of the mean vector")) $(_dircomp("[`wc_statistics!`](@ref)"))
- `cov_sigma`: $(_covdef("estimation errors of the covariance matrix", :a22)) $(_dircomp("[`wc_statistics!`](@ref)"))
- `d_mu`: $(_mudef("delta", :a2)) $(_dircomp("[`wc_statistics!`](@ref)"))
- `k_mu`: set the percentile of a sample of size `Na`, where `Na` is the number of assets, at instance creation. $(_dircomp("[`wc_statistics!`](@ref)"))
- `k_sigma`: set the percentile of a sample of size `Na×Na`, where `Na` is the number of assets, at instance creation. $(_dircomp("[`wc_statistics!`](@ref)"))
# Optimal portfolios
- `optimal`: $_edst for storing optimal portfolios. $(_filled_by("[`optimise!`](@ref)"))
- `z`: $_edst for storing optimal `z` values of portfolios optimised for entropy and relativistic risk measures. $(_filled_by("[`optimise!`](@ref)"))
- `limits`: $_edst for storing the minimal and maximal risk portfolios for given risk measures. $(_filled_by("[`frontier_limits!`](@ref)"))
- `frontier`: $_edst containing points in the efficient frontier for given risk measures. $(_filled_by("[`efficient_frontier!`](@ref)"))
# Solutions
$(_solver_desc("risk measure `JuMP` model."))
- `opt_params`: $_edst for storing parameters used for optimising. $(_filled_by("[`optimise!`](@ref)"))
- `fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`optimise!`](@ref)"))
- `model`: `JuMP.Model()` for optimising a portfolio. $(_filled_by("[`optimise!`](@ref)"))
# Allocation
- `latest_prices`: `Na×1` vector of asset prices, $(_ndef(:a2)). If `prices` is not empty, this is automatically obtained from the last entry. This is used for discretely allocating stocks according to their prices, weight in the portfolio, and money to be invested.
- `alloc_optimal`: $_edst for storing optimal portfolios after allocating discrete stocks. $(_filled_by("[`allocate!`](@ref)"))
$(_solver_desc("discrete allocation `JuMP` model.", "alloc_", "mixed-integer problems"))
- `alloc_params`: $_edst for storing parameters used for optimising the portfolio allocation. $(_filled_by("[`allocate!`](@ref)"))
- `alloc_fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`allocate!`](@ref)"))
- `alloc_model`: `JuMP.Model()` for optimising a portfolio allocation. $(_filled_by("[`allocate!`](@ref)"))
"""
mutable struct Portfolio{
                         # Portfolio characteristics
                         ast, dat, r, s, us, ul, mnea, mna, mnaf, tfa, tfdat, tretf, l,
                         # Risk parameters
                         msvt, lpmt, ai, a, as, bi, b, bs, k, mnak,
                         # Benchmark constraints
                         to, tobw, kte, te, rbi, bw, blbw,
                         # Risk and return constraints
                         ami, bvi, rbv,
                         # Network constraints
                         nm, nsdp, np, ni, nif, amc, bvc,
                         # Bounds constraints
                         ler, ud, umad, usd, ucvar, uwr, uflpm, uslpm, umd, uad, ucdar,
                         uuci, uevar, uedar, urvar, urdar, uk, usk, ugmd, ur, urcvar, utg,
                         urtg, uowa,
                         # Cusom OWA weights
                         wowa,
                         # Optimisation model inputs
                         tmu, tcov, tkurt, tskurt, tl2, ts2, tmuf, tcovf, tmufm, tcovfm,
                         tmubl, tcovbl, tmublf, tcovblf, trfm, tcovl, tcovu, tcovmu, tcovs,
                         tdmu, tkmu, tks, topt, tz, tlim, tfront, tsolv, tf, toptpar, tmod,
                         # Allocation
                         tlp, taopt, tasolv, taoptpar, taf, tamod} <: AbstractPortfolio
    # Portfolio characteristics
    assets::ast
    timestamps::dat
    returns::r
    short::s
    short_u::us
    long_u::ul
    min_number_effective_assets::mnea
    max_number_assets::mna
    max_number_assets_factor::mnaf
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    # Risk parameters
    msv_target::msvt
    lpm_target::lpmt
    alpha_i::ai
    alpha::a
    a_sim::as
    beta_i::bi
    beta::b
    b_sim::bs
    kappa::k
    max_num_assets_kurt::mnak
    # Benchmark constraints
    turnover::to
    turnover_weights::tobw
    kind_tracking_err::kte
    tracking_err::te
    tracking_err_returns::rbi
    tracking_err_weights::bw
    bl_bench_weights::blbw
    # Risk and return constraints
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget::rbv
    # Network constraints
    network_method::nm
    network_sdp::nsdp
    network_penalty::np
    network_ip::ni
    network_ip_factor::nif
    a_vec_cent::amc
    b_cent::bvc
    # Bounds constraints
    mu_l::ler
    sd_u::ud
    mad_u::umad
    ssd_u::usd
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
    rvar_u::urvar
    rdar_u::urdar
    kurt_u::uk
    skurt_u::usk
    gmd_u::ugmd
    rg_u::ur
    rcvar_u::urcvar
    tg_u::utg
    rtg_u::urtg
    owa_u::uowa
    # Custom OWA weights
    owa_w::wowa
    # Model statistics
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
    z::tz
    limits::tlim
    frontier::tfront
    # Solver params
    solvers::tsolv
    opt_params::toptpar
    fail::tf
    model::tmod
    # Allocation
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_solvers::tasolv
    alloc_params::taoptpar
    alloc_fail::taf
    alloc_model::tamod
end

"""
```julia
Portfolio(;    # Portfolio characteristics
    prices::TimeArray = TimeArray(TimeType[], []),    returns::DataFrame = DataFrame(),    ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    timestamps::AbstractVector = Vector{Date}(undef, 0),    assets::AbstractVector = Vector{String}(undef, 0),    short::Bool = false,    short_u::Real = 0.2,    long_u::Real = 1.0,    min_number_effective_assets::Integer = 0,    max_number_assets::Integer = 0,    max_number_assets_factor::Real = 100_000.0,    f_prices::TimeArray = TimeArray(TimeType[], []),    f_returns::DataFrame = DataFrame(),    f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    f_timestamps::AbstractVector = Vector{Date}(undef, 0),    f_assets::AbstractVector = Vector{String}(undef, 0),    loadings::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    # Risk parameters
    msv_target::Union{<:Real, AbstractVector{<:Real}} = Inf,    lpm_target::Union{<:Real, AbstractVector{<:Real}} = Inf,    alpha_i::Real = 0.0001,    alpha::Real = 0.05,    a_sim::Integer = 100,    beta_i::Real = alpha_i,    beta::Real = alpha,    b_sim::Integer = a_sim,    kappa::Real = 0.3,    max_num_assets_kurt::Integer = 0,    # Benchmark constraints
    turnover::Real = Inf,    turnover_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    kind_tracking_err::Symbol = :Weights,    tracking_err::Real = Inf,    tracking_err_returns::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    tracking_err_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    # Risk and return constraints
    a_mtx_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    b_vec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    # Bounds constraints
    mu_l::Real = Inf,    sd_u::Real = Inf,    mad_u::Real = Inf,    ssd_u::Real = Inf,    cvar_u::Real = Inf,    wr_u::Real = Inf,    flpm_u::Real = Inf,    slpm_u::Real = Inf,    mdd_u::Real = Inf,    add_u::Real = Inf,    cdar_u::Real = Inf,    uci_u::Real = Inf,    evar_u::Real = Inf,    edar_u::Real = Inf,    rvar_u::Real = Inf,    rdar_u::Real = Inf,    kurt_u::Real = Inf,    skurt_u::Real = Inf,    gmd_u::Real = Inf,    rg_u::Real = Inf,    rcvar_u::Real = Inf,    tg_u::Real = Inf,    rtg_u::Real = Inf,    owa_u::Real = Inf,    # Custom OWA weights
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    # Model statistics
    mu::AbstractVector = Vector{Float64}(undef, 0),    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    mu_f::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    cov_f::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    mu_fm::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    cov_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    mu_bl::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    cov_bl::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    mu_bl_fm::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    cov_bl_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    returns_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    # Inputs of Worst Case Optimization Models
    cov_l::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    cov_u::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    cov_mu::AbstractMatrix{<:Real} = Diagonal{Float64}(undef, 0),    cov_sigma::AbstractMatrix{<:Real} = Diagonal{Float64}(undef, 0),    d_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    k_mu::Real = Inf,    k_sigma::Real = Inf,    # Optimal portfolios
    optimal::AbstractDict = Dict(),    z::AbstractDict = Dict(),    limits::AbstractDict = Dict(),    frontier::AbstractDict = Dict(),    # Solutions
    solvers::Union{<:AbstractDict, NamedTuple} = Dict(),    opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),    fail::AbstractDict = Dict(),    model::JuMP.Model = JuMP.Model(),    # Allocation
    latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    alloc_optimal::AbstractDict = Dict(),    alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),    alloc_params::Union{<:AbstractDict, NamedTuple} = Dict(),    alloc_fail::AbstractDict = Dict(),    alloc_model::JuMP.Model = JuMP.Model(),)
```
Creates an instance of [`Portfolio`](@ref) containing all internal data necessary for convex portfolio optimisations as well as failed and successful results.
# Inputs
## Portfolio characteristics
- `prices`: `(T+1)×Na` `TimeArray` of asset prices, where the time stamp field is `timestamp`, where $(_tstr(:t1)) and $(_ndef(:a2)). If `prices` is not empty, then `returns`, `ret`, `timestamps`, `assets`, and `latest_prices` are ignored because their respective fields are obtained from `prices`.
- `returns`: `T×(Na+1)` `DataFrame` of asset returns, where $(_tstr(:t1)) and $(_ndef(:a2)), the extra column is `timestamp`, which contains the timestamps of the returns. If `prices` is empty and `returns` is not empty, `ret`, `timestamps`, and `assets` are ignored because their respective fields are obtained from `returns`.
- `ret`: `T×Na` matrix of asset returns, where $(_tstr(:t1)) and $(_ndef(:a2)). Its value is saved in the `returns` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function, where $(_tstr(:t1)) and $(_ndef(:a2)).
- `timestamps`: `T×1` vector of timestamps, where $(_tstr(:t1)). Its value is saved in the `timestamps` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function.
- `assets`: `Na×1` vector of assets, where $(_ndef(:a2)). Its value is saved in the `assets` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function.
- `short`: whether or not to allow negative weights, i.e. whether a portfolio accepts shorting.
- `short_u`: absolute value of the sum of all short (negative) weights.
- `long_u`: sum of all long (positive) weights.
- `min_number_effective_assets`: if non-zero, guarantees that at least number of assets make significant contributions to the final portfolio weights.
- `max_number_assets`: if non-zero, guarantees at most this number of assets make non-zero contributions to the final portfolio weights. Requires an optimiser that supports binary variables.
- `max_number_assets_factor`: scaling factor needed to create a decision variable when `max_number_assets` is non-zero.
- `f_prices`: `(T+1)×Nf` `TimeArray` of factor prices, where the time stamp field is `f_timestamp`, where $(_tstr(:t1)) and $(_ndef(:f2)). If `f_prices` is not empty, then `f_returns`, `f_ret`, `f_timestamps`, `f_assets`, and `latest_prices` are ignored because their respective fields are obtained from `f_prices`.
- `f_returns`: `T×(Nf+1)` `DataFrame` of factor returns, where $(_tstr(:t1)) and $(_ndef(:f2)), the extra column is `f_timestamp`, which contains the timestamps of the factor returns. If `f_prices` is empty and `f_returns` is not empty, `f_ret`, `f_timestamps`, and `f_assets` are ignored because their respective fields are obtained from `f_returns`.
- `f_ret`: `T×Nf` matrix of factor returns, where $(_ndef(:f2)). Its value is saved in the `f_returns` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function, where $(_tstr(:t1)) and $(_ndef(:f2)).
- `f_timestamps`: `T×1` vector of factor timestamps, where $(_tstr(:t1)). Its value is saved in the `f_timestamps` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function.
- `f_assets`: `Nf×1` vector of factors, where $(_ndef(:f2)). Its value is saved in the `f_assets` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function.
- `loadings`: loadings matrix for black litterman models.
## Risk parameters
- `msv_target`: target value for for Absolute Deviation and Semivariance risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the mean returns vector `mu`.
    - Else the target is the value of `msv_target`. If `msv_target` is a vector, its length should be `Na`, where $(_ndef(:a2)).
- `lpm_target`: target value for the First and Second Lower Partial Moment risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the value of the risk free rate `rf`, provided in the [`optimise!`](@ref) function.
    - Else the target is the value of `lpm_target`. If `lpm_target` is a vector, its length should be `Na`, where $(_ndef(:a2)).
$(_isigdef("Tail Gini losses", :a))
$(_sigdef("VaR, CVaR, EVaR, RVaR, DaR, CDaR, EDaR, RDaR, CVaR losses, or Tail Gini losses, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :a))
$(_isigdef("Tail Gini gains", :b))
$(_sigdef("CVaR gains or Tail Gini gains, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :b))
- `kappa`: deformation parameter for relativistic risk measures (RVaR and RDaR).
- `max_num_assets_kurt`: maximum number of assets to use the full kurtosis model, if the number of assets surpases this value use the relaxed kurtosis model.
## Benchmark constraints
- `turnover`: if finite, define the maximum turnover deviations from `turnover_weights` to the optimised portfolio. Else the constraint is disabled.
- `turnover_weights`: target weights for turnover constraint.
    - The turnover constraint is defined as ``\\lvert w_{i} - \\hat{w}_{i}\\rvert \\leq e_{1} \\, \\forall\\, i \\in N``, where ``w_i`` is the optimal weight for the `i'th` asset, ``\\hat{w}_i`` target weight for the `i'th` asset, ``e_{1}`` is the value of the turnover, and $(_ndef(:a3)).
- `kind_tracking_err`: `:Weights` when providing a vector of asset weights for computing the tracking error benchmark from the asset returns, or `:Returns` to directly providing the tracking benchmark. See [`TrackingErrKinds`](@ref) for more information.
- `tracking_err`: if finite, define the maximum tracking error deviation. Else the constraint is disabled.
- `tracking_err_returns`: `T×1` vector of returns to be tracked, where $(_tstr(:t1)). When `kind_tracking_err == :Returns`, this is used directly tracking benchmark.
- `tracking_err_weights`: `Na×1` vector of weights, where $(_ndef(:a2)), when `kind_tracking_err == :Weights`, the returns benchmark is computed from the `returns` field of [`Portfolio`](@ref).
    - The tracking error is defined as ``\\sqrt{\\dfrac{1}{T-1}\\sum\\limits_{i=1}^{T}\\left(\\mathbf{X}_{i} \\bm{w} - b_{i}\\right)^{2}}\\leq e_{2}``, where ``\\mathbf{X}_{i}`` is the `i'th` observation (row) of the returns matrix ``\\mathbf{X}``, ``\\bm{w}`` is the vector of optimal asset weights, ``b_{i}`` is the `i'th` observation of the benchmark returns vector, ``e_{2}`` the tracking error, and $(_tstr(:t2)).
- `bl_bench_weights`: `Na×1` vector of benchmark weights for Black Litterman models, where $(_ndef(:a2)).
## Risk and return constraints
- `a_mtx_ineq`: `C×Na` A matrix of the linear asset constraints ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``, where `C` is the number of constraints, and $(_ndef(:a2)).
- `b_vec_ineq`: `C×1` B vector of the linear asset constraints ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``, where `C` is the number of constraints.
- `risk_budget`: `Na×1` risk budget constraint vector for risk parity optimisations, where $(_ndef(:a2)).
### Bounds constraints
The bounds constraints are only active if they are finite. They define lower bounds denoted by the suffix `_l`, and upper bounds denoted by the suffix `_u`, of various portfolio characteristics. The risk upper bounds are named after their corresponding [`RiskMeasures`](@ref) in lower case, they also bring the same solver requirements as their corresponding risk measure. Multiple bounds constraints can be active at any time but may make finding a solution infeasable.
- `mu_l`: mean expected return.
- `sd_u`: standard deviation.
- `mad_u`: max absolute devia.
- `ssd_u`: semi standard deviation.
- `cvar_u`: critical value at risk.
- `wr_u`: worst realisation.
- `flpm_u`: first lower partial moment.
- `slpm_u`: second lower partial moment.
- `mdd_u`: max drawdown.
- `add_u`: average drawdown.
- `cdar_u`: critical drawdown at risk.
- `uci_u`: ulcer index.
- `evar_u`: entropic value at risk.
- `edar_u`: entropic drawdown at risk.
- `rvar_u`: relativistic value at risk.
- `rdar_u`: relativistic drawdown at risk.
- `kurt_u`: square root kurtosis.
- `skurt_u`: square root semi kurtosis.
- `gmd_u`: gini mean difference.
- `rg_u`: range.
- `rcvar_u`: critical value at risk range.
- `tg_u`: tail gini.
- `rtg_u`: tail gini range.
- `owa_u`: custom ordered weight risk (use with `owa_w`).
## Custom OWA weights
- `owa_w`: `T×1` OWA vector, where $(_tstr(:t1)) containing. Useful for optimising higher OWA L-moments.
## Model statistics
- `mu`: $(_mudef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `cov`: $(_covdef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `kurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the cokurtosis matrix at instance construction. The cokurtosis matrix `kurt` can be computed by calling [`cokurt_mtx`](@ref).
- `skurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the semi cokurtosis matrix at instance construction. The semi cokurtosis matrix `skurt` can be computed by calling [`cokurt_mtx`](@ref).
- `mu_f`: $(_mudef("factors", :f2)) $(_dircomp("[`factor_statistics!`](@ref)"))
- `cov_f`: $(_covdef("factors", :f2)) $(_dircomp("[`factor_statistics!`](@ref)"))
- `mu_fm`: $(_mudef("feature selected factors")) $(_dircomp("[`factor_statistics!`](@ref)"))
- `cov_fm`: $(_covdef("feature selected factors")) $(_dircomp("[`factor_statistics!`](@ref)"))
- `mu_bl`: $(_mudef("Black Litterman")) $(_dircomp("[`black_litterman_statistics!`](@ref)"))
- `cov_bl`: $(_covdef("Black Litterman")) $(_dircomp("[`black_litterman_statistics!`](@ref)"))
- `mu_bl_fm`: $(_mudef("Black Litterman feature selected factors")) $(_dircomp("[`black_litterman_factor_satistics!`](@ref)"))
- `cov_bl_fm`: $(_covdef("Black Litterman feature selected factors")) $(_dircomp("[`black_litterman_factor_satistics!`](@ref)"))
- `returns_fm`: `T×Na` matrix of feature selcted adjusted returns, where $(_tstr(:t1)) and $(_ndef(:a2)). $(_dircomp("[`factor_statistics!`](@ref)"))
## Inputs of Worst Case Optimization Models
- `cov_l`: $(_covdef("worst case lower bound asset")) $(_dircomp("[`wc_statistics!`](@ref)"))
- `cov_u`: $(_covdef("worst case upper bound asset")) $(_dircomp("[`wc_statistics!`](@ref)"))
- `cov_mu`: $(_covdef("estimation errors of the mean vector")) $(_dircomp("[`wc_statistics!`](@ref)"))
- `cov_sigma`: $(_covdef("estimation errors of the covariance matrix", :a22)) $(_dircomp("[`wc_statistics!`](@ref)"))
- `d_mu`: $(_mudef("delta", :a2)) $(_dircomp("[`wc_statistics!`](@ref)"))
- `k_mu`: set the percentile of a sample of size `Na`, where `Na` is the number of assets, at instance creation. $(_dircomp("[`wc_statistics!`](@ref)"))
- `k_sigma`: set the percentile of a sample of size `Na×Na`, where `Na` is the number of assets, at instance creation. $(_dircomp("[`wc_statistics!`](@ref)"))
## Optimal portfolios
- `optimal`: $_edst for storing optimal portfolios. $(_filled_by("[`optimise!`](@ref)"))
- `z`: $_edst for storing optimal `z` values of portfolios optimised for entropy and relativistic risk measures. $(_filled_by("[`optimise!`](@ref)"))
- `limits`: $_edst for storing the minimal and maximal risk portfolios for given risk measures. $(_filled_by("[`frontier_limits!`](@ref)"))
- `frontier`: $_edst containing points in the efficient frontier for given risk measures. $(_filled_by("[`efficient_frontier!`](@ref)"))
## Solutions
$(_solver_desc("risk measure `JuMP` model."))
- `opt_params`: $_edst for storing parameters used for optimising. $(_filled_by("[`optimise!`](@ref)"))
- `fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`optimise!`](@ref)"))
- `model`: `JuMP.Model()` for optimising a portfolio. $(_filled_by("[`optimise!`](@ref)"))
## Allocation
- `latest_prices`: `Na×1` vector of asset prices, $(_ndef(:a2)). If `prices` is not empty, this is automatically obtained from the last entry. This is used for discretely allocating stocks according to their prices, weight in the portfolio, and money to be invested.
- `alloc_optimal`: $_edst for storing optimal portfolios after allocating discrete stocks. $(_filled_by("[`allocate!`](@ref)"))
$(_solver_desc("discrete allocation `JuMP` model.", "alloc_", "mixed-integer problems"))
- `alloc_params`: $_edst for storing parameters used for optimising the portfolio allocation. $(_filled_by("[`allocate!`](@ref)"))
- `alloc_fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`allocate!`](@ref)"))
- `alloc_model`: `JuMP.Model()` for optimising a portfolio allocation. $(_filled_by("[`allocate!`](@ref)"))

# Outputs
- [`Portfolio`](@ref) instance.

[^jLoGo]:
    [Barfuss, W., Massara, G. P., Di Matteo, T., & Aste, T. (2016). Parsimonious modeling with information filtering networks. Physical Review E, 94(6), 062306.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.062306)
"""
function Portfolio(;
                   # Portfolio characteristics
                   prices::TimeArray                    = TimeArray(TimeType[], []),
                   returns::DataFrame                   = DataFrame(),
                   ret::AbstractMatrix{<:Real}          = Matrix{Float64}(undef, 0, 0),
                   timestamps::AbstractVector           = Vector{Date}(undef, 0),
                   assets::AbstractVector               = Vector{String}(undef, 0),
                   short::Bool                          = false,
                   short_u::Real                        = 0.2,
                   long_u::Real                         = 1.0,
                   min_number_effective_assets::Integer = 0,
                   max_number_assets::Integer           = 0,
                   max_number_assets_factor::Real       = 100_000.0,
                   f_prices::TimeArray                  = TimeArray(TimeType[], []),
                   f_returns::DataFrame                 = DataFrame(),
                   f_ret::AbstractMatrix{<:Real}        = Matrix{Float64}(undef, 0, 0),
                   f_timestamps::AbstractVector         = Vector{Date}(undef, 0),
                   f_assets::AbstractVector             = Vector{String}(undef, 0),
                   loadings::AbstractMatrix{<:Real}     = Matrix{Float64}(undef, 0, 0),
                   # Risk parameters
                   msv_target::Union{<:Real, AbstractVector{<:Real}} = Inf,
                   lpm_target::Union{<:Real, AbstractVector{<:Real}} = Inf,
                   alpha_i::Real                                     = 0.0001,
                   alpha::Real                                       = 0.05,
                   a_sim::Integer                                    = 100,
                   beta_i::Real                                      = alpha_i,
                   beta::Real                                        = alpha,
                   b_sim::Integer                                    = a_sim,
                   kappa::Real                                       = 0.3,
                   max_num_assets_kurt::Integer                      = 0,
                   # Benchmark constraints
                   turnover::Real                               = Inf,
                   turnover_weights::AbstractVector{<:Real}     = Vector{Float64}(undef, 0),
                   kind_tracking_err::Symbol                    = :Weights,
                   tracking_err::Real                           = Inf,
                   tracking_err_returns::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   tracking_err_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   bl_bench_weights::AbstractVector{<:Real}     = Vector{Float64}(undef, 0),
                   # Risk and return constraints
                   a_mtx_ineq::AbstractMatrix{<:Real}  = Matrix{Float64}(undef, 0, 0),
                   b_vec_ineq::AbstractVector{<:Real}  = Vector{Float64}(undef, 0),
                   risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   # Network constraints
                   network_method::Symbol = :None,
                   network_sdp::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   network_penalty::Real = 0.05,
                   network_ip::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   network_ip_factor::Real = 100_000.0,
                   a_vec_cent::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   b_cent::Real = Inf,
                   # Bounds constraints
                   mu_l::Real    = Inf, sd_u::Real    = Inf, mad_u::Real   = Inf,
                   ssd_u::Real   = Inf, cvar_u::Real  = Inf, wr_u::Real    = Inf,
                   flpm_u::Real  = Inf, slpm_u::Real  = Inf, mdd_u::Real   = Inf,
                   add_u::Real   = Inf, cdar_u::Real  = Inf, uci_u::Real   = Inf,
                   evar_u::Real  = Inf, edar_u::Real  = Inf, rvar_u::Real  = Inf,
                   rdar_u::Real  = Inf, kurt_u::Real  = Inf, skurt_u::Real = Inf,
                   gmd_u::Real   = Inf, rg_u::Real    = Inf, rcvar_u::Real = Inf,
                   tg_u::Real    = Inf, rtg_u::Real   = Inf, owa_u::Real   = Inf,
                   # Custom OWA weights
                   owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   # Model statistics
                   mu::AbstractVector                 = Vector{Float64}(undef, 0),
                   cov::AbstractMatrix{<:Real}        = Matrix{Float64}(undef, 0, 0),
                   kurt::AbstractMatrix{<:Real}       = Matrix{Float64}(undef, 0, 0),
                   skurt::AbstractMatrix{<:Real}      = Matrix{Float64}(undef, 0, 0),
                   mu_f::AbstractVector{<:Real}       = Vector{Float64}(undef, 0),
                   cov_f::AbstractMatrix{<:Real}      = Matrix{Float64}(undef, 0, 0),
                   mu_fm::AbstractVector{<:Real}      = Vector{Float64}(undef, 0),
                   cov_fm::AbstractMatrix{<:Real}     = Matrix{Float64}(undef, 0, 0),
                   mu_bl::AbstractVector{<:Real}      = Vector{Float64}(undef, 0),
                   cov_bl::AbstractMatrix{<:Real}     = Matrix{Float64}(undef, 0, 0),
                   mu_bl_fm::AbstractVector{<:Real}   = Vector{Float64}(undef, 0),
                   cov_bl_fm::AbstractMatrix{<:Real}  = Matrix{Float64}(undef, 0, 0),
                   returns_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   # Inputs of Worst Case Optimization Models
                   cov_l::AbstractMatrix{<:Real}     = Matrix{Float64}(undef, 0, 0),
                   cov_u::AbstractMatrix{<:Real}     = Matrix{Float64}(undef, 0, 0),
                   cov_mu::AbstractMatrix{<:Real}    = Diagonal{Float64}(undef, 0),
                   cov_sigma::AbstractMatrix{<:Real} = Diagonal{Float64}(undef, 0),
                   d_mu::AbstractVector{<:Real}      = Vector{Float64}(undef, 0),
                   k_mu::Real                        = Inf,
                   k_sigma::Real                     = Inf,
                   # Optimal portfolios
                   optimal::AbstractDict  = Dict(), z::AbstractDict        = Dict(),
                   limits::AbstractDict   = Dict(), frontier::AbstractDict = Dict(),
                   # Solutions
                   solvers::Union{<:AbstractDict, NamedTuple}    = Dict(),
                   opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),
                   fail::AbstractDict                            = Dict(),
                   model::JuMP.Model                             = JuMP.Model(),
                   # Allocation
                   latest_prices::AbstractVector{<:Real}            = Vector{Float64}(undef, 0),
                   alloc_optimal::AbstractDict                      = Dict(),
                   alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
                   alloc_params::Union{<:AbstractDict, NamedTuple}  = Dict(),
                   alloc_fail::AbstractDict                         = Dict(),
                   alloc_model::JuMP.Model                          = JuMP.Model(),)
    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices)))
        latest_prices = Vector(dropmissing!(DataFrame(prices))[end, colnames(prices)])
    end

    if !isempty(returns)
        assets = setdiff(names(returns), ("timestamp",))
        timestamps = returns[!, "timestamp"]
        returns = Matrix(returns[!, assets])
    else
        @smart_assert(length(assets) == size(ret, 2))
        returns = ret
    end

    if !isempty(f_prices)
        f_returns = dropmissing!(DataFrame(percentchange(f_prices)))
    end

    if !isempty(f_returns)
        f_assets = setdiff(names(f_returns), ("timestamp",))
        f_timestamps = f_returns[!, "timestamp"]
        f_returns = Matrix(f_returns[!, f_assets])
    else
        @smart_assert(length(f_assets) == size(f_ret, 2))
        f_returns = f_ret
    end

    @smart_assert(min_number_effective_assets >= 0)
    @smart_assert(max_number_assets >= 0)
    @smart_assert(max_number_assets_factor >= 0)
    @smart_assert(0 < alpha_i < alpha < 1)
    @smart_assert(a_sim > zero(a_sim))
    @smart_assert(0 < beta_i < beta < 1)
    @smart_assert(b_sim > zero(b_sim))
    @smart_assert(0 < kappa < 1)
    @smart_assert(max_num_assets_kurt >= 0)
    if !isempty(turnover_weights)
        @smart_assert(length(turnover_weights) == size(returns, 2))
    end
    @smart_assert(kind_tracking_err in TrackingErrKinds)
    if !isempty(tracking_err_returns)
        @smart_assert(length(tracking_err_returns) == size(returns, 1))
    end
    if !isempty(tracking_err_weights)
        @smart_assert(length(tracking_err_weights) == size(returns, 2))
    end
    if !isempty(bl_bench_weights)
        @smart_assert(length(bl_bench_weights) == size(returns, 2))
    end
    @smart_assert(network_method in NetworkMethods)
    if !isempty(network_sdp)
        @smart_assert(size(network_sdp) == (size(returns, 2), size(returns, 2)))
    end
    if !isempty(network_ip)
        @smart_assert(size(network_ip) == (size(returns, 2), size(returns, 2)))
    end
    if !isempty(a_vec_cent)
        @smart_assert(size(a_vec_cent, 1) == size(returns, 2))
    end
    if !isempty(a_mtx_ineq)
        @smart_assert(size(a_mtx_ineq, 2) == size(returns, 2))
    end
    if !isempty(risk_budget)
        @smart_assert(length(risk_budget) == size(returns, 2))
    end
    if !isempty(owa_w)
        @smart_assert(length(owa_w) == size(returns, 1))
    end
    if !isempty(mu)
        @smart_assert(length(mu) == size(returns, 2))
    end
    if !isempty(cov)
        @smart_assert(size(cov, 1) == size(cov, 2) == size(returns, 2))
    end
    if !isempty(kurt)
        @smart_assert(size(kurt, 1) == size(kurt, 2) == size(returns, 2)^2)
    end
    if !isempty(skurt)
        @smart_assert(size(skurt, 1) == size(skurt, 2) == size(returns, 2)^2)
    end
    if !isempty(mu_f)
        @smart_assert(length(mu_f) == size(f_returns, 2))
    end
    if !isempty(cov_f)
        @smart_assert(size(cov_f, 1) == size(cov_f, 2) == size(f_returns, 2))
    end
    if !isempty(mu_fm)
        @smart_assert(length(mu_fm) == size(returns, 2))
    end
    if !isempty(cov_fm)
        @smart_assert(size(cov_fm, 1) == size(cov_fm, 2) == size(returns, 2))
    end
    if !isempty(mu_bl)
        @smart_assert(length(mu_bl) == size(returns, 2))
    end
    if !isempty(cov_bl)
        @smart_assert(size(cov_bl, 1) == size(cov_bl, 2) == size(returns, 2))
    end
    if !isempty(mu_bl_fm)
        @smart_assert(length(mu_bl_fm) == size(returns, 2))
    end
    if !isempty(cov_bl_fm)
        @smart_assert(size(cov_bl_fm, 1) == size(cov_bl_fm, 2) == size(returns, 2))
    end
    if !isempty(cov_l)
        @smart_assert(size(cov_l, 1) == size(cov_l, 2) == size(returns, 2))
    end
    if !isempty(cov_u)
        @smart_assert(size(cov_u, 1) == size(cov_u, 2) == size(returns, 2))
    end
    if !isempty(cov_mu)
        @smart_assert(size(cov_mu, 1) == size(cov_mu, 2) == size(returns, 2))
    end
    if !isempty(cov_sigma)
        @smart_assert(size(cov_sigma, 1) == size(cov_sigma, 2) == size(returns, 2)^2)
    end
    if !isempty(d_mu)
        @smart_assert(length(d_mu) == size(returns, 2))
    end
    if !isempty(latest_prices)
        @smart_assert(length(latest_prices) == size(returns, 2))
    end

    L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)
    S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)

    return Portfolio{
                     # Portfolio characteristics    
                     typeof(assets), typeof(timestamps), typeof(returns), typeof(short),
                     typeof(short_u), typeof(long_u), typeof(min_number_effective_assets),
                     typeof(max_number_assets), typeof(max_number_assets_factor),
                     typeof(f_assets), typeof(f_timestamps), typeof(f_returns),
                     typeof(loadings),
                     # Risk parameters
                     Union{<:Real, AbstractVector{<:Real}},
                     Union{<:Real, AbstractVector{<:Real}}, typeof(alpha_i), typeof(alpha),
                     typeof(a_sim), typeof(beta_i), typeof(beta), typeof(b_sim),
                     typeof(kappa), typeof(max_num_assets_kurt),
                     # Benchmark constraints
                     typeof(turnover), typeof(turnover_weights), typeof(kind_tracking_err),
                     typeof(tracking_err), typeof(tracking_err_returns),
                     typeof(tracking_err_weights), typeof(bl_bench_weights),
                     # Risk and return constraints
                     typeof(a_mtx_ineq), typeof(b_vec_ineq), typeof(risk_budget),
                     # Network constraints
                     typeof(network_method), typeof(network_sdp), typeof(network_penalty),
                     typeof(network_ip), typeof(network_ip_factor), typeof(a_vec_cent),
                     typeof(b_cent),
                     # Bounds constraints
                     typeof(mu_l), typeof(sd_u), typeof(mad_u), typeof(ssd_u),
                     typeof(cvar_u), typeof(wr_u), typeof(flpm_u), typeof(slpm_u),
                     typeof(mdd_u), typeof(add_u), typeof(cdar_u), typeof(uci_u),
                     typeof(evar_u), typeof(edar_u), typeof(rvar_u), typeof(rdar_u),
                     typeof(kurt_u), typeof(skurt_u), typeof(gmd_u), typeof(rg_u),
                     typeof(rcvar_u), typeof(tg_u), typeof(rtg_u), typeof(owa_u),
                     # Custom OWA weights
                     typeof(owa_w),
                     # Model statistics
                     typeof(mu), typeof(cov), typeof(kurt), typeof(skurt), typeof(L_2),
                     typeof(S_2), typeof(mu_f), typeof(cov_f), typeof(mu_fm),
                     typeof(cov_fm), typeof(mu_bl), typeof(cov_bl), typeof(mu_bl_fm),
                     typeof(cov_bl_fm), typeof(returns_fm),
                     # Inputs of Worst Case Optimization Models
                     typeof(cov_l), typeof(cov_u), typeof(cov_mu), typeof(cov_sigma),
                     typeof(d_mu), typeof(k_mu), typeof(k_sigma),
                     # Optimal portfolios
                     typeof(optimal), typeof(z), typeof(limits), typeof(frontier),
                     # Solutions
                     Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                     typeof(fail), typeof(model),
                     # Allocation
                     typeof(latest_prices), typeof(alloc_optimal),
                     Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                     typeof(alloc_fail), typeof(alloc_model)
                     #
                     }(
                       # Portfolio characteristics
                       assets, timestamps, returns, short, short_u, long_u,
                       min_number_effective_assets, max_number_assets,
                       max_number_assets_factor, f_assets, f_timestamps, f_returns,
                       loadings,
                       # Risk parameters
                       msv_target, lpm_target, alpha_i, alpha, a_sim, beta_i, beta, b_sim,
                       kappa, max_num_assets_kurt,
                       # Benchmark constraints
                       turnover, turnover_weights, kind_tracking_err, tracking_err,
                       tracking_err_returns, tracking_err_weights, bl_bench_weights,
                       # Risk and return constraints
                       a_mtx_ineq, b_vec_ineq, risk_budget,
                       # Network constraints
                       network_method, network_sdp, network_penalty, network_ip,
                       network_ip_factor, a_vec_cent, b_cent,
                       # Bounds constraints
                       mu_l, sd_u, mad_u, ssd_u, cvar_u, wr_u, flpm_u, slpm_u, mdd_u, add_u,
                       cdar_u, uci_u, evar_u, edar_u, rvar_u, rdar_u, kurt_u, skurt_u,
                       gmd_u, rg_u, rcvar_u, tg_u, rtg_u, owa_u,
                       # Custom OWA weights
                       owa_w,
                       # Model statistics
                       mu, cov, kurt, skurt, L_2, S_2, mu_f, cov_f, mu_fm, cov_fm, mu_bl,
                       cov_bl, mu_bl_fm, cov_bl_fm, returns_fm,
                       # Inputs of Worst Case Optimization Models
                       cov_l, cov_u, cov_mu, cov_sigma, d_mu, k_mu, k_sigma,
                       # Optimal portfolios
                       optimal, z, limits, frontier,
                       # Solutions
                       solvers, opt_params, fail, model,
                       # Allocation
                       latest_prices, alloc_optimal, alloc_solvers, alloc_params,
                       alloc_fail, alloc_model)
end

function Base.getproperty(obj::Portfolio, sym::Symbol)
    if sym == :sum_short_long
        obj.short ? obj.long_u - obj.short_u : one(eltype(obj.returns))
    elseif sym == :at
        obj.alpha * size(obj.returns, 1)
    elseif sym == :invat
        one(typeof(obj.at)) / (obj.at)
    elseif sym == :ln_k
        (obj.invat^obj.kappa - obj.invat^(-obj.kappa)) / (2 * obj.kappa)
    elseif sym == :omk
        one(typeof(obj.kappa)) - obj.kappa
    elseif sym == :opk
        one(typeof(obj.kappa)) + obj.kappa
    elseif sym == :invk2
        one(typeof(obj.kappa)) / (2 * obj.kappa)
    elseif sym == :invk
        one(typeof(obj.kappa)) / obj.kappa
    elseif sym == :invopk
        one(typeof(obj.kappa)) / obj.opk
    elseif sym == :invomk
        one(typeof(obj.kappa)) / obj.omk
    else
        getfield(obj, sym)
    end
end

function Base.setproperty!(obj::Portfolio, sym::Symbol, val)
    if sym == :alpha_i
        @smart_assert(0 < val < obj.alpha < 1)
    elseif sym == :alpha
        @smart_assert(0 < obj.alpha_i < val < 1)
    elseif sym == :a_sim
        @smart_assert(val > zero(val))
    elseif sym == :beta
        @smart_assert(0 < obj.beta_i < val < 1)
    elseif sym == :beta_i
        @smart_assert(0 < val < obj.beta < 1)
    elseif sym == :b_sim
        @smart_assert(val > zero(val))
    elseif sym == :kappa
        @smart_assert(0 < val < 1)
    elseif sym == :max_num_assets_kurt
        @smart_assert(val >= 0)
    elseif sym == :turnover_weights
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :kind_tracking_err
        @smart_assert(val in TrackingErrKinds)
    elseif sym == :tracking_err_returns
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 1))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :tracking_err_weights
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :a_mtx_ineq
        if !isempty(val)
            @smart_assert(size(val, 2) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :network_method
        @smart_assert(val in NetworkMethods)
    elseif sym == :network_sdp
        if !isempty(val)
            @smart_assert(size(val) == (size(obj.returns, 2), size(obj.returns, 2)))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :network_ip
        if !isempty(val)
            @smart_assert(size(val) == (size(obj.returns, 2), size(obj.returns, 2)))
        end
        val = convert(typeof(getfield(obj, sym)), val)

    elseif sym == :a_vec_cent
        if !isempty(val)
            @smart_assert(size(val, 1) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :owa_w
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 1))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :mu_f
        if !isempty(val)
            @smart_assert(length(val) == size(obj.f_returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :cov_f
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.f_returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym in (:L_2, :S_2)
        if !isempty(val)
            N = size(obj.returns, 2)
            @smart_assert(size(val) == (Int(N * (N + 1) / 2), N^2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym in (:risk_budget, :bl_bench_weights)
        if isempty(val)
            N = size(obj.returns, 2)
            val = fill(1 / N, N)
        else
            @smart_assert(length(val) == size(obj.returns, 2))
            isa(val, AbstractRange) ? (val = collect(val / sum(val))) : (val ./= sum(val))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym in
           (:min_number_effective_assets, :max_number_assets, :max_number_assets_factor)
        @smart_assert(val >= 0)
    elseif sym in (:kurt, :skurt, :cov_sigma)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym in (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns)
        throw(ArgumentError("$sym is related to other fields and therefore cannot be manually changed without compromising correctness, please create a new instance of Portfolio instead"))
    elseif sym in (:mu, :mu_fm, :mu_bl, :mu_bl_fm, :d_mu, :latest_prices)
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym in (:cov, :cov_fm, :cov_bl, :cov_bl_fm, :cov_l, :cov_u, :cov_mu)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)

    else
        if (isa(getfield(obj, sym), AbstractArray) && isa(val, AbstractArray)) ||
           (isa(getfield(obj, sym), Real) && isa(val, Real))
            val = convert(typeof(getfield(obj, sym)), val)
        end
    end
    return setfield!(obj, sym, val)
end

"""
```julia
mutable struct HCPortfolio{
    ast,    dat,    r,    # Risk parmeters
    ai,    a,    as,    bi,    b,    bs,    k,    ata,    mnak,    # Custom OWA weights
    wowa,    # Optimisation parameters
    tmu,    tcov,    tkurt,    tskurt,    tl2,    ts2,    tbin,    wmi,    wma,    ttco,    tco,    tdist,    tcl,    tk,    # Optimal portfolios
    topt,    # Solutions
    tsolv,    toptpar,    tf,    # Allocation
    tlp,    taopt,    tasolv,    taoptpar,    taf,    tamod,} <: AbstractPortfolio
    # Portfolio characteristics
    assets::ast
    timestamps::dat
    returns::r
    # Risk parmeters
    alpha_i::ai
    alpha::a
    a_sim::as
    beta_i::bi
    beta::b
    b_sim::bs
    kappa::k
    alpha_tail::ata
    max_num_assets_kurt::mnak
    # Custom OWA weights
    owa_w::wowa
    # Optimisation parameters
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    bins_info::tbin
    w_min::wmi
    w_max::wma
    cor_method::ttco
    cor::tco
    dist::tdist
    clusters::tcl
    k::tk
    # Optimal portfolios
    optimal::topt
    # Solutions
    solvers::tsolv
    opt_params::toptpar
    fail::tf
    # Allocation
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_solvers::tasolv
    alloc_params::taoptpar
    alloc_fail::taf
    alloc_model::tamod
end
```
Structure for hierarchical portfolio optimisation.
# Portfolio characteristics
- `assets`: `Na×1` vector of assets, where $(_ndef(:a2)).
- `timestamps`: `T×1` vector of timestamps, where $(_tstr(:t1)).
- `returns`: `T×Na` matrix of asset returns, where $(_tstr(:t1)) and $(_ndef(:a2)).
# Risk parameters
$(_isigdef("Tail Gini losses", :a))
$(_sigdef("VaR, CVaR, EVaR, RVaR, DaR, CDaR, EDaR, RDaR, CVaR losses, or Tail Gini losses, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :a))
- `at`: protected value of `alpha * T`, where $(_tstr(:t1)). Used when optimising a entropic risk measures (EVaR and EDaR).
$(_isigdef("Tail Gini gains", :b))
$(_sigdef("CVaR gains or Tail Gini gains, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :b))
- `kappa`: deformation parameter for relativistic risk measures (RVaR and RDaR).
- `alpha_tail`: significance level for lower tail dependence index, `0 < alpha_tail < 1`.
- `max_num_assets_kurt`: maximum number of assets to use the full kurtosis model, if the number of assets surpases this value use the relaxed kurtosis model.
# Custom OWA weights
- `owa_w`: `T×1` OWA vector, where $(_tstr(:t1)) containing. Useful for optimising higher OWA L-moments.
# Model statistics
- `mu`: $(_mudef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `cov`: $(_covdef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `kurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the cokurtosis matrix at instance construction. The cokurtosis matrix `kurt` can be computed by calling [`cokurt_mtx`](@ref).
- `skurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the semi cokurtosis matrix at instance construction. The semi cokurtosis matrix `skurt` can be computed by calling [`cokurt_mtx`](@ref).
- `L_2`: `(Na×Na) × ((Na×(Na+1)/2))` elimination matrix, where $(_ndef(:a2)). $(_dircomp("[`cokurt_mtx`](@ref)"))
- `S_2`: `((Na×(Na+1)/2)) × (Na×Na)` summation matrix, where $(_ndef(:a2)). $(_dircomp("[`cokurt_mtx`](@ref)"))
- `bins_info`: selection criterion for computing the number of bins used to calculate the mutual and variation of information statistics, see [`mut_var_info_mtx`](@ref) for available choices.
- `w_min`: `Na×1` vector of the lower bounds for asset weights, where $(_ndef(:a2)).
- `w_max`: `Na×1` vector of the upper bounds for asset weights, where $(_ndef(:a2)).
- `cor_method`: method for estimating the codependence matrix.
- `cor`: `Na×Na` matrix, where where $(_ndef(:a2)). Set the value of the codependence matrix at instance construction. When choosing `:Custom_Val` in `cov_method`, this is the value of `cor` used by [`cor_dist_mtx`](@ref).
- `dist`:  `Na×Na` matrix, where where $(_ndef(:a2)). Set the value of the distance matrix at instance construction. When choosing `:Custom_Val` in `cov_method`, this is the value of `dist` used by [`cor_dist_mtx`](@ref).
- `clusters`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of asset clusters. $(_dircomp("[`asset_statistics!`](@ref) and [`optimise!`](@ref)"))
- `k`: number of clusters to cut the dendrogram into.
    - If `k == 0`, automatically compute `k` using the two difference gap statistic [^TDGS]. $(_dircomp("[`asset_statistics!`](@ref) and [`optimise!`](@ref)"))
    - If `k != 0`, use the value directly.
# Optimal portfolios
- `optimal`: $_edst for storing optimal portfolios. $(_filled_by("[`optimise!`](@ref)"))- `optimal`:
# Solutions
$(_solver_desc("risk measure `JuMP` model for `:NCO` optimisations."))
- `opt_params`: $_edst for storing parameters used for optimising. $(_filled_by("[`optimise!`](@ref)"))
- `fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`optimise!`](@ref)"))
# Allocation
- `latest_prices`: `Na×1` vector of asset prices, $(_ndef(:a2)). If `prices` is not empty, this is automatically obtained from the last entry. This is used for discretely allocating stocks according to their prices, weight in the portfolio, and money to be invested.
- `alloc_optimal`: $_edst for storing optimal portfolios after allocating discrete stocks. $(_filled_by("[`allocate!`](@ref)"))
$(_solver_desc("discrete allocation `JuMP` model.", "alloc_", "mixed-integer problems"))
- `alloc_params`: $_edst for storing parameters used for optimising the portfolio allocation. $(_filled_by("[`allocate!`](@ref)"))
- `alloc_fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`allocate!`](@ref)"))
- `alloc_model`: `JuMP.Model()` for optimising a portfolio allocation. $(_filled_by("[`allocate!`](@ref)"))
"""
mutable struct HCPortfolio{
                           # Portfolio characteristics
                           ast, dat, r,
                           # Risk parmeters
                           ai, a, as, bi, b, bs, k, ata, mnak,
                           # Custom OWA weights
                           wowa,
                           # Optimisation parameters
                           tmu, tcov, tkurt, tskurt, tl2, ts2, tbin, wmi, wma, ttco, tco,
                           tdist, tcl, tk,
                           # Optimal portfolios
                           topt,
                           # Solutions
                           tsolv, toptpar, tf,
                           # Allocation
                           tlp, taopt, tasolv, taoptpar, taf, tamod} <: AbstractPortfolio
    # Portfolio characteristics
    assets::ast
    timestamps::dat
    returns::r
    # Risk parmeters
    alpha_i::ai
    alpha::a
    a_sim::as
    beta_i::bi
    beta::b
    b_sim::bs
    kappa::k
    alpha_tail::ata
    max_num_assets_kurt::mnak
    # Custom OWA weights
    owa_w::wowa
    # Optimisation parameters
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    bins_info::tbin
    w_min::wmi
    w_max::wma
    cor_method::ttco
    cor::tco
    dist::tdist
    clusters::tcl
    k::tk
    # Optimal portfolios
    optimal::topt
    # Solutions
    solvers::tsolv
    opt_params::toptpar
    fail::tf
    # Allocation
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_solvers::tasolv
    alloc_params::taoptpar
    alloc_fail::taf
    alloc_model::tamod
end

"""
```julia
HCPortfolio(;    # Portfolio characteristics
    prices::TimeArray = TimeArray(TimeType[], []),    returns::DataFrame = DataFrame(),    ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),    timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),    assets::AbstractVector = Vector{String}(undef, 0),    # Risk parmeters
    alpha_i::Real = 0.0001,    alpha::Real = 0.05,    a_sim::Integer = 100,    beta_i::Real = alpha_i,    beta::Real = alpha,    b_sim::Integer = a_sim,    kappa::Real = 0.3,    alpha_tail::Real = 0.05,    max_num_assets_kurt::Integer = 0,    # Custom OWA weights
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    # Optimisation parameters
    mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    bins_info::Union{Symbol, <:Integer} = :KN,    w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,    w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,    cor_method::Symbol = :Pearson,    cor::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    clusters::Clustering.Hclust = Hclust{Float64}(
        Matrix{Int64}(undef, 0, 2),        Float64[],        Int64[],        :nothing,    ),    k::Integer = 0,    # Optimal portfolios
    optimal::AbstractDict = Dict(),    # Solutions
    solvers::Union{<:AbstractDict, NamedTuple} = Dict(),    opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),    fail::AbstractDict = Dict(),    # Allocation
    latest_prices::AbstractVector = Vector{Float64}(undef, 0),    alloc_optimal::AbstractDict = Dict(),    alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),    alloc_params::Union{<:AbstractDict, NamedTuple} = Dict(),    alloc_fail::AbstractDict = Dict(),    alloc_model::JuMP.Model = JuMP.Model(),)
```
# Inputs
## Portfolio characteristics
- `prices`: `(T+1)×Na` `TimeArray` of asset prices, where the time stamp field is `timestamp`, where $(_tstr(:t1)) and $(_ndef(:a2)). If `prices` is not empty, then `returns`, `ret`, `timestamps`, `assets`, and `latest_prices` are ignored because their respective fields are obtained from `prices`.
- `returns`: `T×(Na+1)` `DataFrame` of asset returns, where $(_tstr(:t1)) and $(_ndef(:a2)), the extra column is `timestamp`, which contains the timestamps of the returns. If `prices` is empty and `returns` is not empty, `ret`, `timestamps`, and `assets` are ignored because their respective fields are obtained from `returns`.
- `ret`: `T×Na` matrix of asset returns, where $(_tstr(:t1)) and $(_ndef(:a2)). Its value is saved in the `returns` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function, where $(_tstr(:t1)) and $(_ndef(:a2)).
- `timestamps`: `T×1` vector of timestamps, where $(_tstr(:t1)). Its value is saved in the `timestamps` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function.
- `assets`: `Na×1` vector of assets, where $(_ndef(:a2)). Its value is saved in the `assets` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function.
## Risk parmeters
$(_isigdef("Tail Gini losses", :a))
$(_sigdef("VaR, CVaR, EVaR, RVaR, DaR, CDaR, EDaR, RDaR, CVaR losses, or Tail Gini losses, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :a))
- `at`: protected value of `alpha * T`, where $(_tstr(:t1)). Used when optimising a entropic risk measures (EVaR and EDaR).
$(_isigdef("Tail Gini gains", :b))
$(_sigdef("CVaR gains or Tail Gini gains, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :b))
- `kappa`: deformation parameter for relativistic risk measures (RVaR and RDaR).
- `alpha_tail`: significance level for lower tail dependence index, `0 < alpha_tail < 1`.
- `max_num_assets_kurt`: when optimising `:NCO` type of [`HCPortfolio`](@ref), maximum number of assets to use the full kurtosis model, if the number of assets surpases this value use the relaxed kurtosis model.
## Custom OWA weights
- `owa_w`: `T×1` OWA vector, where $(_tstr(:t1)) containing. Useful for optimising higher OWA L-moments.
## Model statistics
- `mu`: $(_mudef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `cov`: $(_covdef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `kurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the cokurtosis matrix at instance construction. The cokurtosis matrix `kurt` can be computed by calling [`cokurt_mtx`](@ref).
- `skurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the semi cokurtosis matrix at instance construction. The semi cokurtosis matrix `skurt` can be computed by calling [`cokurt_mtx`](@ref).
- `bins_info`: selection criterion for computing the number of bins used to calculate the mutual and variation of information statistics, see [`mut_var_info_mtx`](@ref) for available choices.
- `w_min`: `Na×1` vector of the lower bounds for asset weights, where $(_ndef(:a2)).
- `w_max`: `Na×1` vector of the upper bounds for asset weights, where $(_ndef(:a2)).
- `cor_method`: method for estimating the codependence matrix.
- `cor`: `Na×Na` matrix, where where $(_ndef(:a2)). Set the value of the codependence matrix at instance construction. When choosing `:Custom_Val` in `cov_method`, this is the value of `cor` used by [`cor_dist_mtx`](@ref).
- `dist`:  `Na×Na` matrix, where where $(_ndef(:a2)). Set the value of the distance matrix at instance construction. When choosing `:Custom_Val` in `cov_method`, this is the value of `dist` used by [`cor_dist_mtx`](@ref).
- `clusters`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of asset clusters. $(_dircomp("[`asset_statistics!`](@ref) and [`optimise!`](@ref)"))
- `k`: number of clusters to cut the dendrogram into.
    - If `k == 0`, automatically compute `k` using the two difference gap statistic [^TDGS]. $(_dircomp("[`asset_statistics!`](@ref) and [`optimise!`](@ref)"))
    - If `k != 0`, use the value directly.
## Optimal portfolios
- `optimal`: $_edst for storing optimal portfolios. $(_filled_by("[`optimise!`](@ref)"))- `optimal`:
## Solutions
$(_solver_desc("risk measure `JuMP` model for `:NCO` optimisations."))
- `opt_params`: $_edst for storing parameters used for optimising. $(_filled_by("[`optimise!`](@ref)"))
- `fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`optimise!`](@ref)"))
## Allocation
- `latest_prices`: `Na×1` vector of asset prices, $(_ndef(:a2)). If `prices` is not empty, this is automatically obtained from the last entry. This is used for discretely allocating stocks according to their prices, weight in the portfolio, and money to be invested.
- `alloc_optimal`: $_edst for storing optimal portfolios after allocating discrete stocks. $(_filled_by("[`allocate!`](@ref)"))
$(_solver_desc("discrete allocation `JuMP` model.", "alloc_", "mixed-integer problems"))
- `alloc_params`: $_edst for storing parameters used for optimising the portfolio allocation. $(_filled_by("[`allocate!`](@ref)"))
- `alloc_fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`allocate!`](@ref)"))
- `alloc_model`: `JuMP.Model()` for optimising a portfolio allocation. $(_filled_by("[`allocate!`](@ref)"))
# Outputs
- [`HCPortfolio`](@ref) instance.

[^TDGS]: 
    [Yue, S., Wang, X. & Wei, M. Application of two-order difference to gap statistic. Trans. Tianjin Univ. 14, 217–221 (2008). https://doi.org/10.1007/s12209-008-0039-1](https://doi.org/10.1007/s12209-008-0039-1)
"""
function HCPortfolio(;
                     # Portfolio characteristics
                     prices::TimeArray = TimeArray(TimeType[], []),
                     returns::DataFrame = DataFrame(),
                     ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
                     assets::AbstractVector = Vector{String}(undef, 0),
                     # Risk parmeters
                     alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100,
                     beta_i::Real = alpha_i, beta::Real = alpha, b_sim::Integer = a_sim,
                     kappa::Real = 0.3, alpha_tail::Real = 0.05,
                     max_num_assets_kurt::Integer = 0,
                     # Custom OWA weights
                     owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                     # Optimisation parameters
                     mu::AbstractVector{<:Real}                   = Vector{Float64}(undef, 0),
                     cov::AbstractMatrix{<:Real}                  = Matrix{Float64}(undef, 0, 0),
                     kurt::AbstractMatrix{<:Real}                 = Matrix{Float64}(undef, 0, 0),
                     skurt::AbstractMatrix{<:Real}                = Matrix{Float64}(undef, 0, 0),
                     bins_info::Union{Symbol, <:Integer}          = :KN,
                     w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,
                     w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,
                     cor_method::Symbol                           = :Pearson,
                     cor::AbstractMatrix{<:Real}                  = Matrix{Float64}(undef, 0, 0),
                     dist::AbstractMatrix{<:Real}                 = Matrix{Float64}(undef, 0, 0),
                     clusters::Clustering.Hclust                  = Hclust{Float64}(Matrix{Int64}(undef, 0, 2), Float64[], Int64[], :nothing),
                     k::Integer                                   = 0,
                     # Optimal portfolios
                     optimal::AbstractDict = Dict(),
                     # Solutions
                     solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
                     opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),
                     fail::AbstractDict = Dict(),
                     # Allocation
                     latest_prices::AbstractVector = Vector{Float64}(undef, 0),
                     alloc_optimal::AbstractDict = Dict(),
                     alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
                     alloc_params::Union{<:AbstractDict, NamedTuple} = Dict(),
                     alloc_fail::AbstractDict = Dict(),
                     alloc_model::JuMP.Model = JuMP.Model(),)
    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices)))
        latest_prices = Vector(dropmissing!(DataFrame(prices))[end, colnames(prices)])
    end

    if !isempty(returns)
        assets = setdiff(names(returns), ("timestamp",))
        timestamps = returns[!, "timestamp"]
        returns = Matrix(returns[!, assets])
    else
        @smart_assert(length(assets) == size(ret, 2))
        returns = ret
    end

    @smart_assert(0 < alpha_i < alpha < 1)
    @smart_assert(a_sim > zero(a_sim))
    @smart_assert(0 < beta_i < beta < 1)
    @smart_assert(b_sim > zero(b_sim))
    @smart_assert(0 < kappa < 1)
    @smart_assert(0 < alpha_tail < 1)
    @smart_assert(max_num_assets_kurt >= 0)
    if !isempty(owa_w)
        @smart_assert(length(owa_w) == size(returns, 1))
    end
    if !isempty(mu)
        @smart_assert(length(mu) == size(returns, 2))
    end
    if !isempty(cov)
        @smart_assert(size(cov, 1) == size(cov, 2) == size(returns, 2))
    end
    if !isempty(kurt)
        @smart_assert(size(kurt, 1) == size(kurt, 2) == size(returns, 2)^2)
    end
    if !isempty(skurt)
        @smart_assert(size(skurt, 1) == size(skurt, 2) == size(returns, 2)^2)
    end
    @smart_assert(bins_info in BinMethods ||
                  (isa(bins_info, Int) && bins_info > zero(bins_info)))
    if isa(w_min, Real)
        @smart_assert(zero(w_min) <= w_min <= one(w_min) && all(w_min .<= w_max))
    elseif isa(w_min, AbstractVector)
        if !isempty(w_min)
            @smart_assert(length(w_min) == size(returns, 2) &&
                          all(x -> zero(eltype(w_min)) <= x <= one(eltype(w_min)), w_min) &&
                          begin
                              try
                                  all(w_min .<= w_max)
                              catch DimensionMismatch
                                  false
                              end
                          end)
        end
    end
    if isa(w_max, Real)
        @smart_assert(zero(w_max) <= w_max <= one(w_max) && all(w_min .<= w_max))
    elseif isa(w_max, AbstractVector)
        if !isempty(w_max)
            @smart_assert(length(w_max) == size(returns, 2) &&
                          all(x -> zero(eltype(w_max)) <= x <= one(eltype(w_max)), w_max) &&
                          begin
                              try
                                  all(w_min .<= w_max)
                              catch DimensionMismatch
                                  false
                              end
                          end)
        end
    end
    @smart_assert(cor_method in CorMethods)
    if !isempty(cor)
        @smart_assert(size(cor, 1) == size(cor, 2) == size(returns, 2))
    end
    if !isempty(dist)
        @smart_assert(size(dist, 1) == size(dist, 2) == size(returns, 2))
    end
    @smart_assert(k >= zero(k))
    if !isempty(latest_prices)
        @smart_assert(length(latest_prices) == size(returns, 2))
    end

    L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)
    S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)

    return HCPortfolio{
                       # Portfolio characteristics
                       typeof(assets), typeof(timestamps), typeof(returns),
                       # Risk parmeters
                       typeof(alpha_i), typeof(alpha), typeof(a_sim), typeof(beta_i),
                       typeof(beta), typeof(b_sim), typeof(kappa), typeof(alpha_tail),
                       typeof(max_num_assets_kurt),
                       # Custom OWA weights
                       typeof(owa_w),
                       # Optimisation parameters
                       typeof(mu), typeof(cov), typeof(kurt), typeof(skurt), typeof(L_2),
                       typeof(S_2), Union{Symbol, <:Integer},
                       Union{<:Real, AbstractVector{<:Real}},
                       Union{<:Real, AbstractVector{<:Real}}, typeof(cor_method),
                       typeof(cor), typeof(dist), typeof(clusters), typeof(k),
                       # Optimal portfolios
                       typeof(optimal),
                       # Solutions
                       Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                       typeof(fail),
                       # Allocation
                       typeof(latest_prices), typeof(alloc_optimal),
                       Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                       typeof(alloc_fail), typeof(alloc_model)
                       #
                       }(
                         # Portfolio characteristics
                         assets, timestamps, returns,
                         # Risk parmeters
                         alpha_i, alpha, a_sim, beta_i, beta, b_sim, kappa, alpha_tail,
                         max_num_assets_kurt,
                         # Custom OWA weights
                         owa_w,
                         # Optimisation parameters
                         mu, cov, kurt, skurt, L_2, S_2, bins_info, w_min, w_max,
                         cor_method, cor, dist, clusters, k,
                         # Optimal portfolios
                         optimal,
                         # Solutions
                         solvers, opt_params, fail,
                         # Allocation
                         latest_prices, alloc_optimal, alloc_solvers, alloc_params,
                         alloc_fail, alloc_model)
end

function Base.setproperty!(obj::HCPortfolio, sym::Symbol, val)
    if sym == :alpha_i
        @smart_assert(0 < val < obj.alpha < 1)
    elseif sym == :alpha
        @smart_assert(0 < obj.alpha_i < val < 1)
    elseif sym == :a_sim
        @smart_assert(val > zero(val))
    elseif sym == :beta
        @smart_assert(0 < obj.beta_i < val < 1)
    elseif sym == :beta_i
        @smart_assert(0 < val < obj.beta < 1)
    elseif sym == :b_sim
        @smart_assert(val > zero(val))
    elseif sym == :kappa
        @smart_assert(0 < val < 1)
    elseif sym == :alpha_tail
        @smart_assert(0 < val < 1)
    elseif sym == :max_num_assets_kurt
        @smart_assert(val >= 0)
    elseif sym == :owa_w
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 1))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :bins_info
        @smart_assert(val in BinMethods || isa(val, Int) && val > zero(val))
    elseif sym == :cor_method
        @smart_assert(val in CorMethods)
    elseif sym == :k
        @smart_assert(val >= zero(val))
    elseif sym in (:w_min, :w_max)
        if sym == :w_min
            smin = sym
            smax = :w_max
            vmin = val
            vmax = getfield(obj, smax)
        else
            smin = :w_min
            smax = sym
            vmin = getfield(obj, smin)
            vmax = val
        end

        if isa(val, Real)
            @smart_assert(zero(val) <= val <= one(val) && all(vmin .<= vmax))
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(obj.returns, 2) &&
                              all(x -> zero(eltype(val)) <= x <= one(eltype(val)), val) &&
                              begin
                                  try
                                      all(vmin .<= vmax)
                                  catch DimensionMismatch
                                      false
                                  end
                              end)

                if isa(getfield(obj, sym), AbstractVector) &&
                   !isa(getfield(obj, sym), AbstractRange)
                    val = if isa(val, AbstractRange)
                        collect(val)
                    else
                        convert(typeof(getfield(obj, sym)), val)
                    end
                end
            end
        end
    elseif sym in (:mu, :latest_prices)
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym in (:kurt, :skurt)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym in (:L_2, :S_2)
        if !isempty(val)
            N = size(obj.returns, 2)
            @smart_assert(size(val) == (Int(N * (N + 1) / 2), N^2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym in (:cov, :cor, :dist)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    else
        if (isa(getfield(obj, sym), AbstractArray) && isa(val, AbstractArray)) ||
           (isa(getfield(obj, sym), Real) && isa(val, Real))
            val = convert(typeof(getfield(obj, sym)), val)
        end
    end
    return setfield!(obj, sym, val)
end
