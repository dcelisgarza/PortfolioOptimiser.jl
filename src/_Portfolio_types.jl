"""
```julia
TrackingErrKinds = (:Weights, :Returns)
```
Available kinds of tracking errors for [`Portfolio`](@ref).
- `:Weights`: provide a vector of asset weights which is used to compute the vector of benchmark returns,
    - ``\\bm{b} = \\mathbf{X} \\bm{w}``,
where ``\\bm{b}`` is the benchmark returns vector, ``\\mathbf{X}`` the ``T \\times{} N`` asset returns matrix, and ``\\bm{w}`` the asset weights vector.
- `:Returns`: directly provide the vector of benchmark returns.
The benchmark is then used as a reference to optimise a portfolio that tracks it up to a given error.
"""
const TrackingErrKinds = (:Weights, :Returns)

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
    ast,
    dat,
    r,
    s,
    us,
    ul,
    ssl,
    mnea,
    mna,
    mnaf,
    tfa,
    tfdat,
    tretf,
    l,
    # Risk parameters
    msvt,
    lpmt,
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
    gst,
    mnak,
    # Benchmark constraints
    to,
    tobw,
    kte,
    te,
    rbi,
    bw,
    blbw,
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
    ur,
    urcvar,
    utg,
    urtg,
    uowa,
    wowa,
    uk,
    usk,
    urvar,
    urdar,
    # Optimisation model inputs
    ttmu,
    tmu,
    ttcov,
    tjlogo,
    tcov,
    tkurt,
    tskurt,
    tpdf,
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
    tz,
    tcovl,
    tcovu,
    tcovmu,
    tcovs,
    tdmu,
    tkmu,
    tks,
    topt,
    tlim,
    tfront,
    tsolv,
    tf,
    toptpar,
    tmod,
    # Allocation
    tlp,
    taopt,
    tasolv,
    taoptpar,
    taf,
    tamod,
} <: AbstractPortfolio
    # Portfolio characteristics.
    assets::ast
    timestamps::dat
    returns::r
    short::s
    short_u::us
    long_u::ul
    sum_short_long::ssl
    min_number_effective_assets::mnea
    max_number_assets::mna
    max_number_assets_factor::mnaf
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    # Risk parameters.
    msv_target::msvt
    lpm_target::lpmt
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
    invk2::tk2
    invk::tkinv
    invopk::tinvopk
    invomk::tinvomk
    gs_threshold::gst
    max_num_assets_kurt::mnak
    # Benchmark constraints.
    turnover::to
    turnover_weights::tobw
    kind_tracking_err::kte
    tracking_err::te
    tracking_err_returns::rbi
    tracking_err_weights::bw
    bl_bench_weights::blbw
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
    rg_u::ur
    rcvar_u::urcvar
    tg_u::utg
    rtg_u::urtg
    owa_u::uowa
    owa_w::wowa
    kurt_u::uk
    skurt_u::usk
    rvar_u::urvar
    rdar_u::urdar
    # Model statistics.
    mu_type::ttmu
    mu::tmu
    cov_type::ttcov
    jlogo::tjlogo
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    posdef_fix::tpdf
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
    # Inputs of Worst Case Optimization Models.
    cov_l::tcovl
    cov_u::tcovu
    cov_mu::tcovmu
    cov_sigma::tcovs
    d_mu::tdmu
    k_mu::tkmu
    k_sigma::tks
    # Optimal portfolios.
    optimal::topt
    z::tz
    limits::tlim
    frontier::tfront
    # Solver params.
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
Structure for optimising portfolios.
# Fieldnames
## Portfolio characteristics
"""
mutable struct Portfolio{
    # Portfolio characteristics
    ast,
    dat,
    r,
    s,
    us,
    ul,
    ssl,
    mnea,
    mna,
    mnaf,
    tfa,
    tfdat,
    tretf,
    l,
    # Risk parameters
    msvt,
    lpmt,
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
    gst,
    mnak,
    # Benchmark constraints
    to,
    tobw,
    kte,
    te,
    rbi,
    bw,
    blbw,
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
    urvar,
    urdar,
    uk,
    usk,
    ugmd,
    ur,
    urcvar,
    utg,
    urtg,
    uowa,
    # Cusom OWA weights.
    wowa,
    # Optimisation model inputs
    ttmu,
    tmu,
    ttcov,
    tjlogo,
    tcov,
    tkurt,
    tskurt,
    tpdf,
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
    tcovl,
    tcovu,
    tcovmu,
    tcovs,
    tdmu,
    tkmu,
    tks,
    topt,
    tz,
    tlim,
    tfront,
    tsolv,
    tf,
    toptpar,
    tmod,
    # Allocation
    tlp,
    taopt,
    tasolv,
    taoptpar,
    taf,
    tamod,
} <: AbstractPortfolio
    # Portfolio characteristics.
    assets::ast
    timestamps::dat
    returns::r
    short::s
    short_u::us
    long_u::ul
    sum_short_long::ssl
    min_number_effective_assets::mnea
    max_number_assets::mna
    max_number_assets_factor::mnaf
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    # Risk parameters.
    msv_target::msvt
    lpm_target::lpmt
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
    invk2::tk2
    invk::tkinv
    invopk::tinvopk
    invomk::tinvomk
    gs_threshold::gst
    max_num_assets_kurt::mnak
    # Benchmark constraints.
    turnover::to
    turnover_weights::tobw
    kind_tracking_err::kte
    tracking_err::te
    tracking_err_returns::rbi
    tracking_err_weights::bw
    bl_bench_weights::blbw
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
    # Custom OWA weights.
    owa_w::wowa
    # Model statistics.
    mu_type::ttmu
    mu::tmu
    cov_type::ttcov
    jlogo::tjlogo
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    posdef_fix::tpdf
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
    # Inputs of Worst Case Optimization Models.
    cov_l::tcovl
    cov_u::tcovu
    cov_mu::tcovmu
    cov_sigma::tcovs
    d_mu::tdmu
    k_mu::tkmu
    k_sigma::tks
    # Optimal portfolios.
    optimal::topt
    z::tz
    limits::tlim
    frontier::tfront
    # Solver params.
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
Portfolio(;
    # Portfolio characteristics.
    prices::TimeArray = TimeArray(TimeType[], []),
    returns::DataFrame = DataFrame(),
    ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    timestamps::AbstractVector = Vector{Date}(undef, 0),
    assets::AbstractVector = Vector{String}(undef, 0),
    short::Bool = false,
    short_u::Real = 0.2,
    long_u::Real = 1.0,
    min_number_effective_assets::Integer = 0,
    max_number_assets::Integer = 0,
    max_number_assets_factor::Real = 100_000.0,
    f_prices::TimeArray = TimeArray(TimeType[], []),
    f_returns::DataFrame = DataFrame(),
    f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    f_timestamps::AbstractVector = Vector{Date}(undef, 0),
    f_assets::AbstractVector = Vector{String}(undef, 0),
    loadings::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    # Risk parameters.
    msv_target::Union{<:Real, AbstractVector{<:Real}} = Inf,
    lpm_target::Union{<:Real, AbstractVector{<:Real}} = Inf,
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Real = alpha_i,
    beta::Real = alpha,
    b_sim::Integer = a_sim,
    kappa::Real = 0.3,
    gs_threshold::Real = 0.5,
    max_num_assets_kurt::Integer = 0,
    # Benchmark constraints.
    turnover::Real = Inf,
    turnover_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    kind_tracking_err::Symbol = :Weights,
    tracking_err::Real = Inf,
    tracking_err_returns::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    tracking_err_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    # Risk and return constraints.
    a_mtx_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    b_vec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    mu_l::Real = Inf,
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
    rvar_u::Real = Inf,
    rdar_u::Real = Inf,
    kurt_u::Real = Inf,
    skurt_u::Real = Inf,
    gmd_u::Real = Inf,
    rg_u::Real = Inf,
    rcvar_u::Real = Inf,
    tg_u::Real = Inf,
    rtg_u::Real = Inf,
    owa_u::Real = Inf,
    # Custom OWA weights.
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    # Model statistics.
    mu_type::Symbol = :Default,
    mu::AbstractVector = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    mu_f::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_f::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_fm::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_bl::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_bl::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_bl_fm::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_bl_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    returns_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    # Inputs of Worst Case Optimization Models.
    cov_l::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_u::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_mu::AbstractMatrix{<:Real} = Diagonal{Float64}(undef, 0),
    cov_sigma::AbstractMatrix{<:Real} = Diagonal{Float64}(undef, 0),
    d_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    k_mu::Real = Inf,
    k_sigma::Real = Inf,
    # Optimal portfolios.
    optimal::AbstractDict = Dict(),
    z::AbstractDict = Dict(),
    limits::AbstractDict = Dict(),
    frontier::AbstractDict = Dict(),
    # Solutions.
    solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
    model = JuMP.Model(),
    # Allocation.
    latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model = JuMP.Model(),
)
```
Creates an instance of [`Portfolio`](@ref) containing all internal data necessary for convex portfolio optimisations as well as failed and successful results.
# Inputs
## Portfolio characteristics
- `prices`: `(T+1)×Na` `TimeArray` with asset pricing information, where the time stamp field is `timestamp`, where $(_tstr(:t1)) and $(_ndef(:a2)). If `prices` is not empty, then `returns`, `ret`, `timestamps`, `assets`, and `latest_prices` are ignored because their respective fields are obtained from `prices`.
- `returns`: `T×(Na+1)` `DataFrame`, where $(_tstr(:t1)) and $(_ndef(:a2)), the extra column is `timestamp`, which contains the timestamps of the returns. If `prices` is empty and `returns` is not empty, `ret`, `timestamps`, and `assets` are ignored because their respective fields are obtained from `returns`.
- `ret`: `T×Na` matrix of returns, where $(_tstr(:t1)) and $(_ndef(:a2)). Its value is saved in the `returns` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function, where $(_tstr(:t1)) and $(_ndef(:a2)).
- `timestamps`: `T×1` vector of timestamps, where $(_tstr(:t1)). Its value is saved in the `timestamps` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function.
- `assets`: `Na×1` vector of assets, where $(_ndef(:a2)). Its value is saved in the `assets` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function.
- `short`: whether or not to allow negative weights, i.e. whether a portfolio accepts shorting.
- `short_u`: absolute value of the sum of all short (negative) weights.
- `long_u`: sum of all long (positive) weights.
- `min_number_effective_assets`: if non-zero, guarantees that at least number of assets make significant contributions to the final portfolio weights.
- `max_number_assets`: if non-zero, guarantees at most this number of assets make non-zero contributions to the final portfolio weights. Requires an optimiser that supports binary variables.
- `max_number_assets_factor`: scaling factor needed to create a decision variable when `max_number_assets` is non-zero.
- `f_prices`: `(T+1)×Nf` `TimeArray` with factor pricing information, where the time stamp field is `f_timestamp`, where $(_tstr(:t1)) and $(_ndef(:f2)). If `f_prices` is not empty, then `f_returns`, `f_ret`, `f_timestamps`, `f_assets`, and `latest_prices` are ignored because their respective fields are obtained from `f_prices`.
- `f_returns`: `T×(Nf+1)` `DataFrame`, where $(_tstr(:t1)) and $(_ndef(:f2)), the extra column is `f_timestamp`, which contains the timestamps of the factor returns. If `f_prices` is empty and `f_returns` is not empty, `f_ret`, `f_timestamps`, and `f_assets` are ignored because their respective fields are obtained from `f_returns`.
- `f_ret`: `T×Nf` matrix of factor returns, where $(_ndef(:f2)). Its value is saved in the `f_returns` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function, where $(_tstr(:t1)) and $(_ndef(:f2)).
- `f_timestamps`: `T×1` vector of factor timestamps, where $(_tstr(:t1)). Its value is saved in the `f_timestamps` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function.
- `f_assets`: `Nf×1` vector of assets, where $(_ndef(:f2)). Its value is saved in the `f_assets` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function.
- `loadings`: loadings matrix for black litterman models.
## Risk parameters
- `msv_target`: target value for for Absolute Deviation and Semivariance risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the mean returns vector `mu`.
    - Else the target is the value of `msv_target`. If `msv_target` is a vector, its length should be `Na`, where $(_ndef(:a2)).
- `lpm_target`: target value for the First and Second Lower Partial Moment risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the value of the risk free rate `rf`, provided in the [`opt_port!`](@ref) function.
    - Else the target is the value of `lpm_target`. If `lpm_target` is a vector, its length should be `Na`, where $(_ndef(:a2)).
$(_isigdef("Tail Gini losses", :a))
$(_sigdef("VaR, CVaR, EVaR, RVaR, DaR, CDaR, EDaR, RDaR, CVaR losses, or Tail Gini losses, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :a))
$(_isigdef("Tail Gini gains", :b))
$(_sigdef("CVaR gains or Tail Gini gains, depending on the [`RiskMeasures`](@ref) and upper bounds being used", :b))
- `kappa`: deformation parameter for relativistic risk measures (RVaR and RDaR).
- `gs_threshold`: Gerber statistic threshold.
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
- `dev_u`: standard deviation.
- `mad_u`: max absolute devia.
- `sdev_u`: semi standard deviation.
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
- `mu_type`: method for estimating the mean returns vectors `mu`, `mu_fm`, `mu_bl`, `mu_bl_fm` in [`mean_vec`](@ref), see [`MuTypes`](@ref) for available choices.
- `mu`: $(_mudef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `cov_type`: methods for estimating the covariance matrices `cov`, `cov_fm`, `cov_bl`, `cov_bl_fm` in [`covar_mtx`](@ref), see [`CovTypes`](@ref) for available choices.
- `jlogo`: if `true`, apply the j-LoGo transformation to the portfolio covariance matrix in [`covar_mtx`](@ref) [^jLoGo].
- `cov`: $(_covdef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `kurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the cokurtosis matrix at instance construction. The cokurtosis matrix `kurt` can be computed by calling [`cokurt_mtx`](@ref).
- `skurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the semi cokurtosis matrix at instance construction. The semi cokurtosis matrix `skurt` can be computed by calling [`cokurt_mtx`](@ref).
- `posdef_fix`: method for fixing non positive definite matrices when computing portfolio statistics, see [`PosdefFixes`](@ref) for available choices.
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
- `optimal`: $_edst for storing optimal portfolios. $(_filled_by("[`opt_port!`](@ref)"))
- `z`: $_edst for storing optimal `z` values of portfolios optimised for entropy and relativistic risk measures. $(_filled_by("[`opt_port!`](@ref)"))
- `limits`: $_edst for storing the minimal and maximal risk portfolios for given risk measures. $(_filled_by("[`frontier_limits!`](@ref)"))
- `frontier`: $_edst containing points in the efficient frontier for given risk measures. $(_filled_by("[`efficient_frontier!`](@ref)"))
## Solutions
$(_solver_desc("risk measure `JuMP` model."))
- `opt_params`: $_edst for storing parameters used for optimising. $(_filled_by("[`opt_port!`](@ref)"))
- `fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`opt_port!`](@ref)"))
- `model`: `JuMP.Model()` for optimising a portfolio. $(_filled_by("[`opt_port!`](@ref)"))
## Allocation
- `latest_prices`: `Na×1` vector of asset prices, $(_ndef(:a2)). If `prices` is not empty, this is automatically obtained from the last entry. This is used for discretely allocating stocks according to their prices, weight in the portfolio, and money to be invested.
- `alloc_optimal`: $_edst for storing optimal portfolios after allocating discrete stocks. $(_filled_by("[`allocate_port!`](@ref)"))
$(_solver_desc("discrete allocation `JuMP` model.", "alloc_", "mixed-integer problems"))
- `alloc_params`: $_edst for storing parameters used for optimising the portfolio allocation. $(_filled_by("[`allocate_port!`](@ref)"))
- `alloc_fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`allocate_port!`](@ref)"))
- `alloc_model`: `JuMP.Model()` for optimising a portfolio allocation. $(_filled_by("[`allocate_port!`](@ref)"))

[^jLoGo]:
    [Barfuss, W., Massara, G. P., Di Matteo, T., & Aste, T. (2016). Parsimonious modeling with information filtering networks. Physical Review E, 94(6), 062306.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.062306)
"""
function Portfolio(;
    # Portfolio characteristics.
    prices::TimeArray = TimeArray(TimeType[], []),
    returns::DataFrame = DataFrame(),
    ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    timestamps::AbstractVector = Vector{Date}(undef, 0),
    assets::AbstractVector = Vector{String}(undef, 0),
    short::Bool = false,
    short_u::Real = 0.2,
    long_u::Real = 1.0,
    min_number_effective_assets::Integer = 0,
    max_number_assets::Integer = 0,
    max_number_assets_factor::Real = 100_000.0,
    f_prices::TimeArray = TimeArray(TimeType[], []),
    f_returns::DataFrame = DataFrame(),
    f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    f_timestamps::AbstractVector = Vector{Date}(undef, 0),
    f_assets::AbstractVector = Vector{String}(undef, 0),
    loadings::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    # Risk parameters.
    msv_target::Union{<:Real, AbstractVector{<:Real}} = Inf,
    lpm_target::Union{<:Real, AbstractVector{<:Real}} = Inf,
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Real = alpha_i,
    beta::Real = alpha,
    b_sim::Integer = a_sim,
    kappa::Real = 0.3,
    gs_threshold::Real = 0.5,
    max_num_assets_kurt::Integer = 0,
    # Benchmark constraints.
    turnover::Real = Inf,
    turnover_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    kind_tracking_err::Symbol = :Weights,
    tracking_err::Real = Inf,
    tracking_err_returns::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    tracking_err_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    # Risk and return constraints.
    a_mtx_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    b_vec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    mu_l::Real = Inf,
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
    rvar_u::Real = Inf,
    rdar_u::Real = Inf,
    kurt_u::Real = Inf,
    skurt_u::Real = Inf,
    gmd_u::Real = Inf,
    rg_u::Real = Inf,
    rcvar_u::Real = Inf,
    tg_u::Real = Inf,
    rtg_u::Real = Inf,
    owa_u::Real = Inf,
    # Custom OWA weights.
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    # Model statistics.
    mu_type::Symbol = :Default,
    mu::AbstractVector = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    mu_f::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_f::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_fm::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_bl::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_bl::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_bl_fm::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_bl_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    returns_fm::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    # Inputs of Worst Case Optimization Models.
    cov_l::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_u::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_mu::AbstractMatrix{<:Real} = Diagonal{Float64}(undef, 0),
    cov_sigma::AbstractMatrix{<:Real} = Diagonal{Float64}(undef, 0),
    d_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    k_mu::Real = Inf,
    k_sigma::Real = Inf,
    # Optimal portfolios.
    optimal::AbstractDict = Dict(),
    z::AbstractDict = Dict(),
    limits::AbstractDict = Dict(),
    frontier::AbstractDict = Dict(),
    # Solutions.
    solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
    model = JuMP.Model(),
    # Allocation.
    latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model = JuMP.Model(),
)
    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices)))
        latest_prices = Vector(dropmissing!(DataFrame(prices))[end, colnames(prices)])
    end

    if !isempty(returns)
        assets = setdiff(names(returns), ("timestamp",))
        timestamps = returns[!, "timestamp"]
        returns = Matrix(returns[!, assets])
    else
        @assert(
            length(assets) == size(ret, 2),
            "each column of returns must correspond to an asset"
        )
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
        @assert(
            length(f_assets) == size(f_ret, 2),
            "each column of factor returns must correspond to a factor asset"
        )
        f_returns = f_ret
    end

    @assert(
        min_number_effective_assets >= 0,
        "min_number_effective_assets = $min_number_effective_assets, must be greater than or equal to 0"
    )
    @assert(
        max_number_assets >= 0,
        "max_number_assets = $max_number_assets, must be greater than or equal to 0"
    )
    @assert(
        max_number_assets_factor >= 0,
        "max_number_assets_factor = $max_number_assets_factor, must be greater than or equal to 0"
    )
    @assert(
        0 < alpha_i < alpha < 1,
        "0 < alpha_i < alpha < 1: 0 < $alpha_i < $alpha < 1, must hold"
    )
    @assert(a_sim > zero(a_sim), "a_sim = $a_sim, must be greater than zero")
    @assert(
        0 < beta_i < beta < 1,
        "0 < beta_i < beta < 1: 0 < $beta_i < $beta < 1, must hold"
    )
    @assert(b_sim > zero(b_sim), "b_sim = $b_sim, must be greater than or equal to zero")
    @assert(0 < kappa < 1, "kappa = $(kappa), must be greater than 0 and less than 1")
    @assert(
        0 < gs_threshold < 1,
        "gs_threshold = $gs_threshold, must be greater than 0 and less than 1"
    )
    @assert(
        max_num_assets_kurt >= 0,
        "max_num_assets_kurt = $max_num_assets_kurt must be greater than or equal to zero"
    )
    if !isempty(turnover_weights)
        @assert(
            length(turnover_weights) == size(returns, 2),
            "length(turnover_weights) = $(turnover_weights) and size(returns, 2) = $(size(returns, 2)), must be equal"
        )
    end
    @assert(
        kind_tracking_err ∈ TrackingErrKinds,
        "kind_tracking_err = $(kind_tracking_err), must be one of $TrackingErrKinds"
    )
    if !isempty(tracking_err_returns)
        @assert(
            length(tracking_err_returns) == size(returns, 1),
            "length(tracking_err_returns) = $tracking_err_returns and size(returns, 1) = $(size(returns, 1)), must be equal"
        )
    end
    if !isempty(tracking_err_weights)
        @assert(
            length(tracking_err_weights) == size(returns, 2),
            "length(tracking_err_weights) = $tracking_err_weights and size(returns, 2) = $(size(returns, 2)), must be equal"
        )
    end
    if !isempty(bl_bench_weights)
        @assert(
            length(bl_bench_weights) == size(returns, 2),
            "length(bl_bench_weights) = $bl_bench_weights and size(returns, 2) = $(size(returns, 2)), must be equal"
        )
    end
    if !isempty(a_mtx_ineq)
        @assert(
            size(a_mtx_ineq, 2) == size(returns, 2),
            "size(a_mtx_ineq, 2) = a_mtx_ineq must have the same number of columns size(a_mtx_ineq, 2) = $(size(a_mtx_ineq, 2)), as there are assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    if !isempty(risk_budget)
        @assert(
            length(risk_budget) == size(returns, 2),
            "length(risk_budget) = $(length(risk_budget)), and size(returns, 2) = $(size(returns, 2)) must be equal"
        )
    end
    if !isempty(owa_w)
        @assert(
            length(owa_w) == size(returns, 1),
            "length(owa_w) = $(length(owa_w)), and size(returns, 1) = $(size(returns, 1)) must be equal"
        )
    end
    @assert(mu_type ∈ MuTypes, "mu_type = $mu_type, must be one of $MuTypes")
    if !isempty(mu)
        @assert(
            length(mu) == size(returns, 2),
            "length(mu) = $(length(mu)), and size(returns, 2) = $(size(returns, 2)) must be equal"
        )
    end
    @assert(cov_type ∈ CovTypes, "cov_type = $cov_type, must be one of $CovTypes")
    if !isempty(cov)
        @assert(
            size(cov, 1) == size(cov, 2) == size(returns, 2),
            "cov must be a square matrix, size(cov) = $(size(cov)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    if !isempty(kurt)
        @assert(
            size(kurt, 1) == size(kurt, 2) == size(returns, 2)^2,
            "kurt must be a square matrix, size(kurt) = $(size(kurt)), with side length equal to the number of assets squared, size(returns, 2)^2 = $(size(returns, 2))^2"
        )
    end
    if !isempty(skurt)
        @assert(
            size(skurt, 1) == size(skurt, 2) == size(returns, 2)^2,
            "skurt must be a square matrix, size(skurt) = $(size(skurt)), with side length equal to the number of assets squared, size(returns, 2)^2 = $(size(returns, 2))^2"
        )
    end
    @assert(
        posdef_fix ∈ PosdefFixes,
        "posdef_fix = $posdef_fix, must be one of $PosdefFixes"
    )
    if !isempty(mu_f)
        @assert(
            length(mu_f) == size(f_returns, 2),
            "length(mu_f) = $(length(mu_f)), and size(f_returns, 2) = $(size(f_returns, 2)) must be equal"
        )
    end
    if !isempty(cov_f)
        @assert(
            size(cov_f, 1) == size(cov_f, 2) == size(f_returns, 2),
            "cov_f must be a square matrix, size(cov_f) = $(size(cov_f)), with side length equal to the number of assets, size(f_returns, 2) = $(size(f_returns, 2))"
        )
    end
    if !isempty(mu_fm)
        @assert(
            length(mu_fm) == size(returns, 2),
            "length(mu_fm) = $(length(mu_fm)), and size(returns, 2) = $(size(returns, 2)) must be equal"
        )
    end
    if !isempty(cov_fm)
        @assert(
            size(cov_fm, 1) == size(cov_fm, 2) == size(returns, 2),
            "cov_fm must be a square matrix, size(cov_fm) = $(size(cov_fm)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    if !isempty(mu_bl)
        @assert(
            length(mu_bl) == size(returns, 2),
            "length(mu_bl) = $(length(mu_bl)), and size(returns, 2) = $(size(returns, 2)) must be equal"
        )
    end
    if !isempty(cov_bl)
        @assert(
            size(cov_bl, 1) == size(cov_bl, 2) == size(returns, 2),
            "cov_bl must be a square matrix, size(cov_bl) = $(size(cov_bl)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    if !isempty(mu_bl_fm)
        @assert(
            length(mu_bl_fm) == size(returns, 2),
            "length(mu_bl_fm) = $(length(mu_bl_fm)), and size(returns, 2) = $(size(returns, 2)) must be equal"
        )
    end
    if !isempty(cov_bl_fm)
        @assert(
            size(cov_bl_fm, 1) == size(cov_bl_fm, 2) == size(returns, 2),
            "cov_bl_fm must be a square matrix, size(cov_bl_fm) = $(size(cov_bl_fm)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    if !isempty(cov_l)
        @assert(
            size(cov_l, 1) == size(cov_l, 2) == size(returns, 2),
            "cov_l must be a square matrix, size(cov_l) = $(size(cov_l)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    if !isempty(cov_u)
        @assert(
            size(cov_u, 1) == size(cov_u, 2) == size(returns, 2),
            "cov_u must be a square matrix, size(cov_u) = $(size(cov_u)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    if !isempty(cov_mu)
        @assert(
            size(cov_mu, 1) == size(cov_mu, 2) == size(returns, 2),
            "cov_mu must be a square matrix, size(cov_mu) = $(size(cov_mu)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    if !isempty(cov_sigma)
        @assert(
            size(cov_sigma, 1) == size(cov_sigma, 2) == size(returns, 2)^2,
            "cov_sigma must be a square matrix, size(cov_sigma) = $(size(cov_sigma)), with side length equal to the number of assets squared, size(returns, 2)^2 = $(size(returns, 2))^2"
        )
    end
    if !isempty(d_mu)
        @assert(
            length(d_mu) == size(returns, 2),
            "length(d_mu) = $(length(d_mu)), and size(returns, 2) = $(size(returns, 2)) must be equal"
        )
    end
    if !isempty(latest_prices)
        @assert(
            length(latest_prices) == size(returns, 2),
            "length(latest_prices) = $(length(latest_prices)), and size(returns, 2) = $(size(returns, 2)) must be equal"
        )
    end

    at = zero(alpha)
    invat = zero(alpha)
    ln_k = zero(promote_type(typeof(alpha), typeof(kappa)))
    opk = zero(kappa)
    omk = zero(kappa)
    invk2 = zero(kappa)
    invk = zero(kappa)
    invopk = zero(kappa)
    invomk = zero(kappa)

    L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)
    S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)

    return Portfolio{# Portfolio characteristics.
        typeof(assets),
        typeof(timestamps),
        typeof(returns),
        typeof(short),
        typeof(short_u),
        typeof(long_u),
        promote_type(typeof(short_u), typeof(long_u)),
        typeof(min_number_effective_assets),
        typeof(max_number_assets),
        typeof(max_number_assets_factor),
        typeof(f_assets),
        typeof(f_timestamps),
        typeof(f_returns),
        typeof(loadings),
        # Risk parameters.
        Union{<:Real, AbstractVector{<:Real}},
        Union{<:Real, AbstractVector{<:Real}},
        typeof(alpha_i),
        typeof(alpha),
        typeof(a_sim),
        typeof(at),
        typeof(beta_i),
        typeof(beta),
        typeof(b_sim),
        typeof(kappa),
        typeof(invat),
        typeof(ln_k),
        typeof(opk),
        typeof(omk),
        typeof(invk2),
        typeof(invk),
        typeof(invopk),
        typeof(invomk),
        typeof(gs_threshold),
        typeof(max_num_assets_kurt),
        # Benchmark constraints.
        typeof(turnover),
        typeof(turnover_weights),
        typeof(kind_tracking_err),
        typeof(tracking_err),
        typeof(tracking_err_returns),
        typeof(tracking_err_weights),
        typeof(bl_bench_weights),
        # Risk and return constraints.
        typeof(a_mtx_ineq),
        typeof(b_vec_ineq),
        typeof(risk_budget),
        typeof(mu_l),
        typeof(dev_u),
        typeof(mad_u),
        typeof(sdev_u),
        typeof(cvar_u),
        typeof(wr_u),
        typeof(flpm_u),
        typeof(slpm_u),
        typeof(mdd_u),
        typeof(add_u),
        typeof(cdar_u),
        typeof(uci_u),
        typeof(evar_u),
        typeof(edar_u),
        typeof(rvar_u),
        typeof(rdar_u),
        typeof(kurt_u),
        typeof(skurt_u),
        typeof(gmd_u),
        typeof(rg_u),
        typeof(rcvar_u),
        typeof(tg_u),
        typeof(rtg_u),
        typeof(owa_u),
        # Custom OWA weights.
        typeof(owa_w),
        # Model statistics.
        typeof(mu_type),
        typeof(mu),
        typeof(cov_type),
        typeof(jlogo),
        typeof(cov),
        typeof(kurt),
        typeof(skurt),
        typeof(posdef_fix),
        typeof(L_2),
        typeof(S_2),
        typeof(mu_f),
        typeof(cov_f),
        typeof(mu_fm),
        typeof(cov_fm),
        typeof(mu_bl),
        typeof(cov_bl),
        typeof(mu_bl_fm),
        typeof(cov_bl_fm),
        typeof(returns_fm),
        # Inputs of Worst Case Optimization Models.
        typeof(cov_l),
        typeof(cov_u),
        typeof(cov_mu),
        typeof(cov_sigma),
        typeof(d_mu),
        typeof(k_mu),
        typeof(k_sigma),
        # Optimal portfolios.
        typeof(optimal),
        typeof(z),
        typeof(limits),
        typeof(frontier),
        # Solutions.
        Union{<:AbstractDict, NamedTuple},
        typeof(opt_params),
        typeof(fail),
        typeof(model),
        # Allocation.
        typeof(latest_prices),
        typeof(alloc_optimal),
        Union{<:AbstractDict, NamedTuple},
        typeof(alloc_params),
        typeof(alloc_fail),
        typeof(alloc_model),
    }(
        # Portfolio characteristics.
        assets,
        timestamps,
        returns,
        short,
        short_u,
        long_u,
        short ? long_u - short_u : 1.0,
        min_number_effective_assets,
        max_number_assets,
        max_number_assets_factor,
        f_assets,
        f_timestamps,
        f_returns,
        loadings,
        # Risk parameters.
        msv_target,
        lpm_target,
        alpha_i,
        alpha,
        a_sim,
        at,
        beta_i,
        beta,
        b_sim,
        kappa,
        invat,
        ln_k,
        opk,
        omk,
        invk2,
        invk,
        invopk,
        invomk,
        gs_threshold,
        max_num_assets_kurt,
        # Benchmark constraints.
        turnover,
        turnover_weights,
        kind_tracking_err,
        tracking_err,
        tracking_err_returns,
        tracking_err_weights,
        bl_bench_weights,
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
        rvar_u,
        rdar_u,
        kurt_u,
        skurt_u,
        gmd_u,
        rg_u,
        rcvar_u,
        tg_u,
        rtg_u,
        owa_u,
        # Custom OWA weights.
        owa_w,
        # Model statistics.
        mu_type,
        mu,
        cov_type,
        jlogo,
        cov,
        kurt,
        skurt,
        posdef_fix,
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
        # Inputs of Worst Case Optimization Models.
        cov_l,
        cov_u,
        cov_mu,
        cov_sigma,
        d_mu,
        k_mu,
        k_sigma,
        # Optimal portfolios.
        optimal,
        z,
        limits,
        frontier,
        # Solutions.
        solvers,
        opt_params,
        fail,
        model,
        # Allocation.
        latest_prices,
        alloc_optimal,
        alloc_solvers,
        alloc_params,
        alloc_fail,
        alloc_model,
    )
end

function Base.setproperty!(obj::Portfolio, sym::Symbol, val)
    if sym == :short
        setfield!(obj, :sum_short_long, val ? obj.long_u - obj.short_u : 1.0)
    elseif sym == :short_u
        setfield!(obj, :sum_short_long, obj.short ? obj.long_u - val : 1.0)
    elseif sym == :long_u
        setfield!(obj, :sum_short_long, obj.short ? val - obj.short_u : 1.0)
    elseif sym == :alpha_i
        @assert(
            0 < val < obj.alpha < 1,
            "0 < alpha_i < alpha < 1: 0 < $val < $(obj.alpha) < 1 must hold"
        )
    elseif sym == :alpha
        @assert(
            0 < obj.alpha_i < val < 1,
            "0 < alpha_i < alpha < 1: 0 < $(obj.alpha_i) < $val < 1, must hold"
        )
        at = val * size(obj.returns, 1)
        invat = 1 / at
        ln_k = (invat^obj.kappa - invat^(-obj.kappa)) / (2 * obj.kappa)
        setfield!(obj, :at, at)
        setfield!(obj, :invat, invat)
        setfield!(obj, :ln_k, ln_k)
    elseif sym == :a_sim
        @assert(val > zero(val), "a_sim = $val, must be greater than zero")
    elseif sym == :beta
        @assert(
            0 < obj.beta_i < val < 1,
            "0 < beta_i < beta < 1: 0 < $(obj.beta_i) < $val < 1, must hold"
        )
    elseif sym == :beta_i
        @assert(
            0 < val < obj.beta < 1,
            "0 < beta_i < beta < 1: : 0 < $val < $(obj.beta) < 1 must hold"
        )
    elseif sym == :b_sim
        @assert(val > zero(val), "b_sim = $val, must be greater than zero")
    elseif sym == :kappa
        @assert(0 < val < 1, "kappa = $(val), must be greater than 0 and smaller than 1")
        ln_k = (obj.invat^val - obj.invat^(-val)) / (2 * val)
        opk = 1 + val
        omk = 1 - val
        invk2 = 1 / (2 * val)
        invk = 1 / val
        invopk = 1 / opk
        invomk = 1 / omk
        setfield!(obj, :ln_k, ln_k)
        setfield!(obj, :opk, opk)
        setfield!(obj, :omk, omk)
        setfield!(obj, :invk2, invk2)
        setfield!(obj, :invk, invk)
        setfield!(obj, :invopk, invopk)
        setfield!(obj, :invomk, invomk)
    elseif sym == :gs_threshold
        @assert(
            0 < val < 1,
            "gs_threshold = $val, must be greater than zero and smaller than one"
        )
    elseif sym == :max_num_assets_kurt
        @assert(
            val >= 0,
            "max_num_assets_kurt = $val must be greater than or equal to zero"
        )
    elseif sym == :turnover_weights
        if !isempty(val)
            @assert(
                length(val) == size(obj.returns, 2),
                "length(turnover_weights) = $val and size(returns, 2) = $(size(obj.returns, 2)), must be equal"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :kind_tracking_err
        @assert(
            val ∈ TrackingErrKinds,
            "kind_tracking_err = $(val), must be one of $TrackingErrKinds"
        )
    elseif sym == :tracking_err_returns
        if !isempty(val)
            @assert(
                length(val) == size(obj.returns, 1),
                "length(tracking_err_returns) = $val and size(returns, 1) = $(size(obj.returns, 1)), must be equal"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :tracking_err_weights
        if !isempty(val)
            @assert(
                length(val) == size(obj.returns, 2),
                "length(tracking_err_weights) = $val and size(returns, 2) = $(size(obj.returns, 2)), must be equal"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :a_mtx_ineq
        if !isempty(val)
            @assert(
                size(val, 2) == size(obj.returns, 2),
                "size(a_mtx_ineq, 2) = a_mtx_ineq must have the same number of columns size(a_mtx_ineq, 2) = $(size(val, 2)), as there are assets, size(returns, 2) = $(size(obj.returns, 2))"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :owa_w
        if !isempty(val)
            @assert(
                length(val) == size(obj.returns, 1),
                "length(owa_w) = $val and size(returns, 1) = $(size(obj.returns, 1)), must be equal"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :mu_type
        @assert(val ∈ MuTypes, "mu_type = $val, must be one of $MuTypes")
    elseif sym == :cov_type
        @assert(val ∈ CovTypes, "cov_type = $val, must be one of $CovTypes")
    elseif sym == :posdef_fix
        @assert(val ∈ PosdefFixes, "posdef_fix = $val, must be one of $PosdefFixes")
    elseif sym == :mu_f
        if !isempty(val)
            @assert(
                length(val) == size(obj.f_returns, 2),
                "length(mu_f) = $(length(val)), and size(f_returns, 2) = $(size(obj.f_returns, 2)) must be equal"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :cov_f
        if !isempty(val)
            @assert(
                size(val, 1) == size(val, 2) == size(obj.f_returns, 2),
                "cov_f must be a square matrix, size(cov_f) = $(size(val)), with side length equal to the number of assets, size(f_returns, 2) = $(size(obj.f_returns, 2))"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:risk_budget, :bl_bench_weights)
        if isempty(val)
            N = size(obj.returns, 2)
            val = fill(1 / N, N)
        else
            @assert(
                length(val) == size(obj.returns, 2),
                "length($sym) == size(obj.returns, 2) must hold: $(length(val)) == $(size(obj.returns, 2))"
            )
            isa(val, AbstractRange) ? (val = collect(val / sum(val))) : (val ./= sum(val))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈
           (:min_number_effective_assets, :max_number_assets, :max_number_assets_factor)
        @assert(val >= 0, "$sym = $val, must be greater than or equal to 0")
    elseif sym ∈ (:kurt, :skurt, :cov_sigma)
        if !isempty(val)
            @assert(
                size(val, 1) == size(val, 2) == size(obj.returns, 2)^2,
                "$sym must be a square matrix, size($sym) = $(size(val)), with side length equal to the number of assets squared, size(returns, 2)^2 = $(size(obj.returns, 2))^2"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns)
        throw(
            ArgumentError(
                "$sym is related to other fields and therefore cannot be manually changed without compromising correctness, please create a new instance of Portfolio instead",
            ),
        )
    elseif sym ∈ (:mu, :mu_fm, :mu_bl, :mu_bl_fm, :d_mu, :latest_prices)
        if !isempty(val)
            @assert(
                length(val) == size(obj.returns, 2),
                "length($sym) = $(length(val)), and size(returns, 2) = $(size(obj.returns, 2)) must be equal"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:cov, :cov_fm, :cov_bl, :cov_bl_fm, :cov_l, :cov_u, :cov_mu)
        if !isempty(val)
            @assert(
                size(val, 1) == size(val, 2) == size(obj.returns, 2),
                "$sym must be a square matrix, size($sym) = $(size(val)), with side length equal to the number of assets, size(returns, 2) = $(size(obj.returns, 2))"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (
        :sum_short_long,
        :at,
        :invat,
        :ln_k,
        :opk,
        :omk,
        :invk2,
        :invk,
        :invopk,
        :invomk,
    )
        throw(
            ArgumentError(
                "$sym is computed from other fields and therefore cannot be manually changed",
            ),
        )
    else
        if (isa(getfield(obj, sym), AbstractArray) && isa(val, AbstractArray)) ||
           (isa(getfield(obj, sym), Real) && isa(val, Real))
            val = convert(typeof(getfield(obj, sym)), val)
        end
    end

    setfield!(obj, sym, val)
end

"""
```julia
mutable struct HCPortfolio{
    ast,
    dat,
    r,
    # Risk parmeters.
    ai,
    a,
    as,
    bi,
    b,
    bs,
    k,
    ata,
    gst,
    # Optimisation parameters.
    tymu,
    tmu,
    tycov,
    tcov,
    tbin,
    wmi,
    wma,
    ttco,
    tco,
    tdist,
    tcl,
    tk,
    # Optimal portfolios.
    topt,
    # Solutions.
    tsolv,
    toptpar,
    tf,
} <: AbstractPortfolio
    # Portfolio characteristics.
    assets::ast
    timestamps::dat
    returns::r
    # Risk parmeters.
    alpha_i::ai
    alpha::a
    a_sim::as
    beta_i::bi
    beta::b
    b_sim::bs
    kappa::k
    alpha_tail::ata
    gs_threshold::gst
    max_num_assets_kurt::mnak
    # Optimisation parameters.
    mu_type::tymu
    mu::tmu
    cov_type::tycov
    jlogo::tjlogo
    cov::tcov
    bins_info::tbin
    w_min::wmi
    w_max::wma
    codep_type::ttco
    codep::tco
    dist::tdist
    clusters::tcl
    k::tk
    # Optimal portfolios.
    optimal::topt
    # Solutions.
    solvers::tsolv
    opt_params::toptpar
    fail::tf
end
```
"""
mutable struct HCPortfolio{
    ast,
    dat,
    r,
    # Risk parmeters.
    ai,
    a,
    as,
    bi,
    b,
    bs,
    k,
    ata,
    gst,
    mnak,
    # Custom OWA weights.
    wowa,
    # Optimisation parameters.
    ttmu,
    tmu,
    ttcov,
    tjlogo,
    tcov,
    tpdf,
    tbin,
    wmi,
    wma,
    ttco,
    tco,
    tdist,
    tcl,
    tk,
    # Optimal portfolios.
    topt,
    # Solutions.
    tsolv,
    toptpar,
    tf,
    # Allocation
    tlp,
    taopt,
    tasolv,
    taoptpar,
    taf,
    tamod,
} <: AbstractPortfolio
    # Portfolio characteristics.
    assets::ast
    timestamps::dat
    returns::r
    # Risk parmeters.
    alpha_i::ai
    alpha::a
    a_sim::as
    beta_i::bi
    beta::b
    b_sim::bs
    kappa::k
    alpha_tail::ata
    gs_threshold::gst
    max_num_assets_kurt::mnak
    # Custom OWA weights.
    owa_w::wowa
    # Optimisation parameters.
    mu_type::ttmu
    mu::tmu
    cov_type::ttcov
    jlogo::tjlogo
    cov::tcov
    posdef_fix::tpdf
    bins_info::tbin
    w_min::wmi
    w_max::wma
    codep_type::ttco
    codep::tco
    dist::tdist
    clusters::tcl
    k::tk
    # Optimal portfolios.
    optimal::topt
    # Solutions.
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
HCPortfolio(;
    # Portfolio characteristics.
    prices::TimeArray = TimeArray(TimeType[], []),
    returns::DataFrame = DataFrame(),
    ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
    assets::AbstractVector = Vector{String}(undef, 0),
    # Risk parmeters.
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Real = alpha_i,
    beta::Real = alpha,
    b_sim::Integer = a_sim,
    kappa::Real = 0.3,
    alpha_tail::Real = 0.05,
    gs_threshold::Real = 0.5,
    max_num_assets_kurt::Integer = 0,
    # Custom OWA weights.
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    # Optimisation parameters.
    mu_type::Symbol = :Default,
    mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    bins_info::Union{Symbol, <:Integer} = :KN,
    w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,
    w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,
    codep_type::Symbol = :Pearson,
    codep::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    clusters = Hclust{Float64}(Matrix{Int64}(undef, 0, 2), Float64[], Int64[], :nothing),
    k::Integer = 0,
    # Optimal portfolios.
    optimal::AbstractDict = Dict(),
    # Solutions.
    solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
    # Allocation.
    latest_prices::AbstractVector = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model = JuMP.Model(),
)
```
"""
function HCPortfolio(;
    # Portfolio characteristics.
    prices::TimeArray = TimeArray(TimeType[], []),
    returns::DataFrame = DataFrame(),
    ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
    assets::AbstractVector = Vector{String}(undef, 0),
    # Risk parmeters.
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Real = alpha_i,
    beta::Real = alpha,
    b_sim::Integer = a_sim,
    kappa::Real = 0.3,
    alpha_tail::Real = 0.05,
    gs_threshold::Real = 0.5,
    max_num_assets_kurt::Integer = 0,
    # Custom OWA weights.
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    # Optimisation parameters.
    mu_type::Symbol = :Default,
    mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    bins_info::Union{Symbol, <:Integer} = :KN,
    w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,
    w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,
    codep_type::Symbol = :Pearson,
    codep::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    clusters = Hclust{Float64}(Matrix{Int64}(undef, 0, 2), Float64[], Int64[], :nothing),
    k::Integer = 0,
    # Optimal portfolios.
    optimal::AbstractDict = Dict(),
    # Solutions.
    solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
    # Allocation.
    latest_prices::AbstractVector = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model = JuMP.Model(),
)
    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices)))
        latest_prices = Vector(dropmissing!(DataFrame(prices))[end, colnames(prices)])
    end

    if !isempty(returns)
        assets = setdiff(names(returns), ("timestamp",))
        timestamps = returns[!, "timestamp"]
        returns = Matrix(returns[!, assets])
    else
        @assert(
            length(assets) == size(ret, 2),
            "each column of returns must correspond to an asset"
        )
        returns = ret
    end

    @assert(
        0 < alpha_i < alpha < 1,
        "0 < alpha_i < alpha < 1: 0 < $alpha_i < $alpha < 1, must hold"
    )
    @assert(a_sim > zero(a_sim), "a_sim = $a_sim, must be greater than zero")
    @assert(
        0 < beta_i < beta < 1,
        "0 < beta_i < beta < 1: 0 < $beta_i < $beta < 1, must hold"
    )
    @assert(b_sim > zero(b_sim), "b_sim = $b_sim, must be greater than or equal to zero")
    @assert(0 < kappa < 1, "kappa = $(kappa), must be greater than 0 and less than 1")
    @assert(
        0 < gs_threshold < 1,
        "gs_threshold = $gs_threshold, must be greater than 0 and less than 1"
    )
    @assert(
        max_num_assets_kurt >= 0,
        "max_num_assets_kurt = $max_num_assets_kurt must be greater than or equal to zero"
    )
    if !isempty(owa_w)
        @assert(
            length(owa_w) == size(returns, 1),
            "length(owa_w) = $(length(owa_w)), and size(returns, 1) = $(size(returns, 1)) must be equal"
        )
    end
    @assert(mu_type ∈ MuTypes, "mu_type = $mu_type, must be one of $MuTypes")
    if !isempty(mu)
        @assert(
            length(mu) == size(returns, 2),
            "length(mu) = $(length(mu)), and size(returns, 2) = $(size(returns, 2)) must be equal"
        )
    end
    @assert(cov_type ∈ CovTypes, "cov_type = $cov_type, must be one of $CovTypes")
    if !isempty(cov)
        @assert(
            size(cov, 1) == size(cov, 2) == size(returns, 2),
            "cov must be a square matrix, size(cov) = $(size(cov)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    @assert(
        posdef_fix ∈ PosdefFixes,
        "posdef_fix = $posdef_fix, must be one of $PosdefFixes"
    )
    @assert(
        bins_info ∈ BinTypes || isa(bins_info, Int) && bins_info > zero(bins_info),
        "bins_info = $bins_info, has to either be in $BinTypes, or an integer value greater than 0"
    )
    if isa(w_min, Real)
        @assert(
            zero(w_min) <= w_min <= one(w_min) && all(w_min .<= w_max),
            "0 <= w_min .<= w_max <= 1: 0 <= $w_min .<= $w_max <= 1, must be true"
        )
    elseif isa(w_min, AbstractVector)
        if !isempty(w_min)
            @assert(
                length(w_min) == size(returns, 2) &&
                all(x -> zero(eltype(w_min)) <= x <= one(eltype(w_min)), w_min) &&
                begin
                    try
                        all(w_min .<= w_max)
                    catch DimensionMismatch
                        false
                    end
                end,
                "length(w_min) = $(length(w_min)) must be equal to the number of assets size(returns, 2) = $(size(returns, 2)); all entries must be greater than or equal to zero, and less than or equal to one all(x -> 0 <= x <= 1, w_min) = $(all(x -> zero(eltype(w_min)) <= x <= one(eltype(w_min)), w_min)); and all(w_min .<= w_max) = $(begin
                    try
                        all(w_min .<= w_max)
                    catch DimensionMismatch
                        false
                    end
                end)"
            )
        end
    end
    if isa(w_max, Real)
        @assert(
            zero(w_max) <= w_max <= one(w_max) && all(w_min .<= w_max),
            "0 <= w_min .<= w_max <= 1: 0 <= $w_min .<= $w_max <= 1, must be true"
        )
    elseif isa(w_max, AbstractVector)
        if !isempty(w_max)
            @assert(
                length(w_max) == size(returns, 2) &&
                all(x -> zero(eltype(w_max)) <= x <= one(eltype(w_max)), w_max) &&
                begin
                    try
                        all(w_min .<= w_max)
                    catch DimensionMismatch
                        false
                    end
                end,
                "length(w_max) = $(length(w_max)) must be equal to the number of assets size(returns, 2) = $(size(returns, 2)); all entries must be greater than or equal to zero, and less than or equal to one all(x -> 0 <= x <= 1, w_max) = $(all(x -> zero(eltype(w_max)) <= x <= one(eltype(w_max)), w_max)); and all(w_min .<= w_max) = $(begin
                    try
                        all(w_min .<= w_max)
                    catch DimensionMismatch
                        false
                    end
                end)"
            )
        end
    end
    @assert(codep_type ∈ CodepTypes, "codep_type = $codep_type, must be one of $CodepTypes")
    if !isempty(codep)
        @assert(
            size(codep, 1) == size(codep, 2) == size(returns, 2),
            "codep must be a square matrix, size(codep) = $(size(codep)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    if !isempty(dist)
        @assert(
            size(dist, 1) == size(dist, 2) == size(returns, 2),
            "dist must be a square matrix, size(dist) = $(size(dist)), with side length equal to the number of assets, size(returns, 2) = $(size(returns, 2))"
        )
    end
    @assert(k >= zero(k), "a_sim = $k, must be greater than or equal to zero")
    if !isempty(latest_prices)
        @assert(
            length(latest_prices) == size(returns, 2),
            "length(latest_prices) = $(length(latest_prices)), and size(returns, 2) = $(size(returns, 2)) must be equal"
        )
    end

    return HCPortfolio{
        typeof(assets),
        typeof(timestamps),
        typeof(returns),
        # Risk parmeters.
        typeof(alpha_i),
        typeof(alpha),
        typeof(a_sim),
        typeof(beta_i),
        typeof(beta),
        typeof(b_sim),
        typeof(kappa),
        typeof(alpha_tail),
        typeof(gs_threshold),
        typeof(max_num_assets_kurt),
        # Custom OWA weights.
        typeof(owa_w),
        # Optimisation parameters.
        typeof(mu_type),
        typeof(mu),
        typeof(cov_type),
        typeof(jlogo),
        typeof(cov),
        typeof(posdef_fix),
        Union{Symbol, <:Integer},
        Union{<:Real, AbstractVector{<:Real}},
        Union{<:Real, AbstractVector{<:Real}},
        typeof(codep_type),
        typeof(codep),
        typeof(dist),
        typeof(clusters),
        typeof(k),
        # Optimal portfolios.
        typeof(optimal),
        # Solutions.
        Union{<:AbstractDict, NamedTuple},
        typeof(opt_params),
        typeof(fail),
        # Allocation.
        typeof(latest_prices),
        typeof(alloc_optimal),
        Union{<:AbstractDict, NamedTuple},
        typeof(alloc_params),
        typeof(alloc_fail),
        typeof(alloc_model),
    }(
        assets,
        timestamps,
        returns,
        # Risk parmeters.
        alpha_i,
        alpha,
        a_sim,
        beta_i,
        beta,
        b_sim,
        kappa,
        alpha_tail,
        gs_threshold,
        max_num_assets_kurt,
        # Custom OWA weights.
        owa_w,
        # Optimisation parameters.
        mu_type,
        mu,
        cov_type,
        jlogo,
        cov,
        posdef_fix,
        bins_info,
        w_min,
        w_max,
        codep_type,
        codep,
        dist,
        clusters,
        k,
        # Optimal portfolios.
        optimal,
        # Solutions.
        solvers,
        opt_params,
        fail,
        # Allocation.
        latest_prices,
        alloc_optimal,
        alloc_solvers,
        alloc_params,
        alloc_fail,
        alloc_model,
    )
end

function Base.setproperty!(obj::HCPortfolio, sym::Symbol, val)
    if sym == :alpha_i
        @assert(
            0 < val < obj.alpha < 1,
            "0 < alpha_i < alpha < 1: 0 < $val < $(obj.alpha) < 1 must hold"
        )
    elseif sym == :alpha
        @assert(
            0 < obj.alpha_i < val < 1,
            "0 < alpha_i < alpha < 1: 0 < $(obj.alpha_i) < $val < 1, must hold"
        )
        at = val * size(obj.returns, 1)
        invat = 1 / at
        ln_k = (invat^obj.kappa - invat^(-obj.kappa)) / (2 * obj.kappa)
        setfield!(obj, :at, at)
        setfield!(obj, :invat, invat)
        setfield!(obj, :ln_k, ln_k)
    elseif sym == :a_sim
        @assert(val > zero(val), "a_sim = $val, must be greater than zero")
    elseif sym == :beta
        @assert(
            0 < obj.beta_i < val < 1,
            "0 < beta_i < beta < 1: 0 < $(obj.beta_i) < $val < 1, must hold"
        )
    elseif sym == :beta_i
        @assert(
            0 < val < obj.beta < 1,
            "0 < beta_i < beta < 1: : 0 < $val < $(obj.beta) < 1 must hold"
        )
    elseif sym == :b_sim
        @assert(val > zero(val), "b_sim = $val, must be greater than zero")
    elseif sym == :kappa
        @assert(0 < val < 1, "kappa = $(val), must be greater than 0 and smaller than 1")
    elseif sym == :gs_threshold
        @assert(
            0 < val < 1,
            "gs_threshold = $val, must be greater than zero and smaller than one"
        )
    elseif sym == :max_num_assets_kurt
        @assert(
            val >= 0,
            "max_num_assets_kurt = $val must be greater than or equal to zero"
        )
    elseif sym == :owa_w
        if !isempty(val)
            @assert(
                length(val) == size(obj.returns, 1),
                "length(owa_w) = $val and size(returns, 1) = $(size(obj.returns, 1)), must be equal"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :mu_type
        @assert(val ∈ MuTypes, "mu_type = $val, must be one of $MuTypes")
    elseif sym == :cov_type
        @assert(val ∈ CovTypes, "cov_type = $val, must be one of $CovTypes")
    elseif sym == :posdef_fix
        @assert(val ∈ PosdefFixes, "posdef_fix = $val, must be one of $PosdefFixes")
    elseif sym == :bins_info
        @assert(
            val ∈ BinTypes || isa(val, Int) && val > zero(val),
            "bins_info = $val, has to either be in $BinTypes, or an integer value greater than 0"
        )
    elseif sym == :codep_type
        @assert(val ∈ CodepTypes, "codep_type = $val, must be one of $CodepTypes")
    elseif sym == :k
        @assert(val >= zero(val), "k = $val, must be greater than or equal to zero")
    elseif sym ∈ (:w_min, :w_max)
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
            @assert(
                zero(val) <= val <= one(val) && all(vmin .<= vmax),
                "0 <= w_min .<= w_max <= 1: 0 <= $vmin .<= $vmax <= 1, must be true"
            )
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @assert(
                    length(val) == size(obj.returns, 2) &&
                    all(x -> zero(eltype(val)) <= x <= one(eltype(val)), val) &&
                    begin
                        try
                            all(vmin .<= vmax)
                        catch DimensionMismatch
                            false
                        end
                    end,
                    "length(w_min) = $(length(val)) must be equal to the number of assets size(returns, 2) = $(size(obj.returns, 2)); all entries must be greater than or equal to zero all(x -> 0 <= x <= 1, val) = $(all(x -> zero(eltype(val)) <= x <= one(eltype(val)), val)); and all(w_min .<= w_max) = $(begin
                        try
                            all(vmin .<= vmax)
                        catch DimensionMismatch
                            false
                        end
                    end)"
                )

                if isa(getfield(obj, sym), AbstractVector) &&
                   !isa(getfield(obj, sym), AbstractRange)
                    val =
                        isa(val, AbstractRange) ? collect(val) :
                        convert(typeof(getfield(obj, sym)), val)
                end
            end
        end
    elseif sym ∈ (:mu, :latest_prices)
        if !isempty(val)
            @assert(
                length(val) == size(obj.returns, 2),
                "length($sym) = $(length(val)), and size(returns, 2) = $(size(obj.returns, 2)) must be equal"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:cov, :codep, :dist)
        if !isempty(val)
            @assert(
                size(val, 1) == size(val, 2) == size(obj.returns, 2),
                "$sym must be a square matrix, size($sym) = $(size(val)), with side length equal to the number of assets, size(returns, 2) = $(size(obj.returns, 2))"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    else
        if (isa(getfield(obj, sym), AbstractArray) && isa(val, AbstractArray)) ||
           (isa(getfield(obj, sym), Real) && isa(val, Real))
            val = convert(typeof(getfield(obj, sym)), val)
        end
    end
    setfield!(obj, sym, val)
end
