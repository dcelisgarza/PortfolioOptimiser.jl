"""
```julia
AbstractPortfolio
```
Abstract type for portfolios. Concrete portfolios subtype this see [`Portfolio`](@ref) and [`HCPortfolio`](@ref).
"""
abstract type AbstractPortfolio end

"""
```julia
RiskMeasures = (
    :SD,
    :MAD,
    :SSD,
    :FLPM,
    :SLPM,
    :WR,
    :CVaR,
    :EVaR,
    :RVaR,
    :MDD,
    :ADD,
    :CDaR,
    :UCI,
    :EDaR,
    :RDaR,
    :Kurt,
    :SKurt,
    :GMD,
    :RG,
    :RCVaR,
    :TG,
    :RTG,
    :OWA,
)
```
Available risk measures for `:Trad` and `:RP` type (see [`PortTypes`](@ref)) of [`Portfolio`](@ref).
- `:SD` = standard deviation ([`SD`](@ref)).
- `:MAD` = max absolute deviation ([`MAD`](@ref)).
- `:SSD` = semi standard deviation ([`SSD`](@ref)).
- `:FLPM` = first lower partial moment (omega ratio) ([`FLPM`](@ref)).
- `:SLPM` = second lower partial moment (sortino ratio) ([`SLPM`](@ref)).
- `:WR` = worst realisation ([`WR`](@ref)).
- `:CVaR` = conditional value at risk ([`CVaR`](@ref)).
- `:EVaR` = entropic value at risk ([`EVaR`](@ref)).
- `:RVaR` = relativistic value at risk ([`RVaR`](@ref)).
- `:MDD` = maximum drawdown of uncompounded cumulative returns ([`MDD_abs`](@ref)).
- `:ADD` = average drawdown of uncompounded cumulative returns ([`ADD_abs`](@ref)).
- `:CDaR` = conditional drawdown at risk of uncompounded cumulative returns ([`CDaR_abs`](@ref)).
- `:UCI` = ulcer index of uncompounded cumulative returns ([`UCI_abs`](@ref)).
- `:EDaR` = entropic drawdown at risk of uncompounded cumulative returns ([`EDaR_abs`](@ref)).
- `:RDaR` = relativistic drawdown at risk of uncompounded cumulative returns ([`RDaR_abs`](@ref)).
- `:Kurt` = square root kurtosis ([`Kurt`](@ref)).
- `:SKurt` = square root semi-kurtosis ([`SKurt`](@ref)).
- `:GMD` = gini mean difference ([`GMD`](@ref)).
- `:RG` = range of returns ([`RG`](@ref)).
- `:RCVaR` = range of conditional value at risk ([`RCVaR`](@ref)).
- `:TG` = tail gini ([`TG`](@ref)).
- `:RTG` = range of tail gini ([`RTG`](@ref)).
- `:OWA` = ordered weight array (generic OWA weights) ([`OWA`](@ref)).
"""
const RiskMeasures = (
    :SD,    # _mv
    :MAD,   # _mad
    :SSD,   # _mad
    :FLPM,  # _lpm
    :SLPM,  # _lpm
    :WR,    # _wr
    :CVaR,  # _var
    :EVaR,  # _var
    :RVaR,  # _var
    :MDD,   # _dar
    :ADD,   # _dar
    :CDaR,  # _dar
    :UCI,   # _dar
    :EDaR,  # _dar
    :RDaR,  # _dar
    :Kurt,  # _krt
    :SKurt, # _krt
    :GMD,   # _owa
    :RG,    # _owa
    :RCVaR, # _owa
    :TG,    # _owa
    :RTG,   # _owa
    :OWA,   # _owa
)

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
    krt_u::uk
    skrt_u::usk
    rvar_u::urvar
    rdar_u::urdar
    # Optimisation model inputs.
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
    limits::tlim
    frontier::tfront
    # Solver params.
    solvers::tsolv
    opt_params::toptpar
    fail::tf
    model::tmod
    z::tz
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
## Portfolio characteristics.
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
    tz,
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
    krt_u::uk
    skrt_u::usk
    rvar_u::urvar
    rdar_u::urdar
    # Optimisation model inputs.
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
    limits::tlim
    frontier::tfront
    # Solver params.
    solvers::tsolv
    opt_params::toptpar
    fail::tf
    model::tmod
    z::tz
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
    gmd_u::Real = Inf,
    rg_u::Real = Inf,
    rcvar_u::Real = Inf,
    tg_u::Real = Inf,
    rtg_u::Real = Inf,
    owa_u::Real = Inf,
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    krt_u::Real = Inf,
    skrt_u::Real = Inf,
    rvar_u::Real = Inf,
    rdar_u::Real = Inf,
    # Optimisation model inputs.
    mu_type::Symbol = :Default,
    mu::AbstractVector = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    L_2::AbstractMatrix = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
    S_2::AbstractMatrix = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
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
    limits::DataFrame = DataFrame(),
    frontier::AbstractDict = Dict(),
    # Solutions.
    solvers::AbstractDict = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
    model = JuMP.Model(),
    z::AbstractDict = Dict(),
    # Allocation.
    latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::AbstractDict = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model::AbstractDict = Dict(),
)
```
Creates an instance of [`Portfolio`](@ref) containing all internal data necessary for convex portfolio optimisations as well as failed and successful results.
# Inputs
## Portfolio characteristics.
- `prices`: `(T+1)×Na` `TimeArray` with asset pricing information where the time stamp field is `timestamp`, `T` is the number of returns observations and `Na` the number of assets. If `prices` is not empty, then `returns`, `ret`, `timestamps`, `assets` and `latest_prices` are ignored because their respective fields are obtained from `prices`.
- `returns`: `T×(Na+1)` `DataFrame` where `T` is the number of returns observations, `Na` is the number of assets, the extra column is `timestamp`, which contains the timestamps of the returns. If `prices` is empty and `returns` is not empty, `ret`, `timestamps` and `assets` are ignored because their respective fields are obtained from `returns`.
- `ret`: `T×Na` matrix of returns. Its value is saved in the `returns` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function.
- `timestamps`: `T×1` vector of timestamps. Its value is saved in the `timestamps` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function.
- `assets`: `Na×1` vector of assets. Its value is saved in the `assets` field of [`Portfolio`](@ref). If `prices` or `returns` are not empty, this value is obtained from within the function.
- `short`: whether or not to allow negative weights, i.e. whether a portfolio accepts shorting.
- `short_u`: absolute value of the sum of all short (negative) weights.
- `long_u`: sum of all long (positive) weights.
- `min_number_effective_assets`: if non-zero, guarantees that at least number of assets make significant contributions to the final portfolio weights.
- `max_number_assets`: if non-zero, guarantees at most this number of assets make non-zero contributions to the final portfolio weights. Requires an optimiser that supports binary variables.
- `max_number_assets_factor`: scaling factor needed to create a decision variable when `max_number_assets` is non-zero.
- `f_prices`: `(T+1)×Nf` `TimeArray` with factor pricing information where the time stamp field is `f_timestamp`, `T` is the number of factors returns observations and `Nf` the number of factors. If `f_prices` is not empty, then `f_returns`, `f_ret`, `f_timestamps`, `f_assets` and `latest_prices` are ignored because their respective fields are obtained from `f_prices`.
- `f_returns`: `T×(Nf+1)` `DataFrame` where `T` is the number of factor returns observations, `Nf` is the number of factors, the extra column is `f_timestamp`, which contains the timestamps of the factor returns. If `f_prices` is empty and `f_returns` is not empty, `f_ret`, `f_timestamps` and `f_assets` are ignored because their respective fields are obtained from `f_returns`.
- `f_ret`: `T×Nf` matrix of factor returns. Its value is saved in the `f_returns` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function.
- `f_timestamps`: `T×1` vector of factor timestamps. Its value is saved in the `f_timestamps` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function.
- `f_assets`: `Nf×1` vector of assets. Its value is saved in the `f_assets` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function.
- `loadings`: loadings matrix for black litterman models.
## Risk parameters.
- `msv_target`: target value for for Absolute Deviation and Semivariance risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the mean returns vector `mu`.
    - Else the target is the value of `msv_target`. If `msv_target` is a vector, its length should be `Na`, where `Na` is the number of assets.
- `lpm_target`: target value for the First and Second Lower Partial Moment risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the value of the risk free rate `rf`, provided in the [`opt_port!`](@ref) function.
    - Else the target is the value of `lpm_target`. If `lpm_target` is a vector, its length should be `Na`, where `Na` is the number of assets.
- `alpha_i`: initial significance level of Tail Gini losses, `0 < alpha_i < alpha < 1`.
- `alpha`: significance level of VaR, CVaR, EVaR, RVaR, DaR, CDaR, EDaR, RDaR, CVaR losses or Tail Gini losses, `alpha ∈ (0, 1)`.
- `a_sim`: number of CVaRs to approximate the Tail Gini losses, `a_sim`.
- `beta_i`: initial significance level of Tail Gini gains, `0 < beta_i < beta < 1`. 
- `beta`: significance level of CVaR gains, `beta ∈ (0, 1)`.
- `b_sim`: number of CVaRs to approximate the Tail Gini gains, `b_sim > 0`.
- `kappa`: deformation parameter for relativistic risk measures (RVaR and RDaR).
- `gs_threshold`: Gerber statistic threshold.
- `max_num_assets_kurt`: maximum number of assets to use the full kurtosis model, if the number of assets surpases this value use the relaxed kurtosis model.
## Benchmark constraints.
- `turnover`: if finite, define the maximum turnover deviations from `turnover_weights` to the optimised portfolio. Else the constraint is disabled.
- `turnover_weights`: target weights for turnover constraint.
    - The turnover constraint is defined as ``\\lvert w_{i} - \\hat{w}_{i}\\rvert \\leq t \\, \\forall\\, i \\in N``, where ``w_i`` is the optimal weight for the `i'th` asset, ``\\hat{w}_i`` target weight for the `i'th` asset, ``t`` is the value of the turnover, and ``N`` the number of assets.
- `kind_tracking_err`: `:Weights` when providing a vector of asset weights for computing the tracking error benchmark from the asset returns, or `:Returns` to directly providing the tracking benchmark. See [`TrackingErrKinds`](@ref) for more information.
- `tracking_err`: if finite, define the maximum tracking error deviation. Else the constraint is disabled.
- `tracking_err_returns`: `T×1` vector of returns to be tracked, when `kind_tracking_err == :Returns`, this is used directly tracking benchmark.
- `tracking_err_weights`: `N×1` vector of weights, when `kind_tracking_err == :Weights`, the returns benchmark is computed from the `returns` field of [`Portfolio`](@ref).
    - The tracking error is defined as ``\\sqrt{\\dfrac{1}{T-1}\\sum\\limits_{i=1}^{T}\\left(\\mathbf{X}_{i} \\bm{w} - b_{i}\\right)^{2}}\\leq t``, where ``\\mathbf{X}_{i}`` is the `i'th` observation (row) of the returns matrix ``\\mathbf{X}``, ``\\bm{w}`` is the vector of optimal asset weights, ``b_{i}`` is the `i'th` observation of the benchmark returns vector, ``t`` the tracking error, and ``T`` the total number of observations in the returns series.
- `bl_bench_weights`: `N×1` vector of benchmark weights for Black Litterman models.
## Risk and return constraints.
- `a_mtx_ineq`: A matrix of the linear asset constraints ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``.
- `b_vec_ineq`: B vector of the linear asset constraints ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``.
- `risk_budget`: `N×1` risk budget constraint vector for risk parity optimisations.
- `mu_l`:
- `dev_u`:
- `mad_u`:
- `sdev_u`:
- `cvar_u`:
- `wr_u`:
- `flpm_u`:
- `slpm_u`:
- `mdd_u`:
- `add_u`:
- `cdar_u`:
- `uci_u`:
- `evar_u`:
- `edar_u`:
- `gmd_u`:
- `rg_u`:
- `rcvar_u`:
- `tg_u`:
- `rtg_u`:
- `owa_u`:
- `owa_w`:
- `krt_u`:
- `skrt_u`:
- `rvar_u`:
- `rdar_u`:
## Optimisation model inputs.
- `mu_type`:
- `mu`:
- `cov_type`:
- `jlogo`:
- `cov`:
- `kurt`:
- `skurt`:
- `posdef_fix`:
- `L_2`:
- `S_2`:
- `mu_f`:
- `cov_f`:
- `mu_fm`:
- `cov_fm`:
- `mu_bl`:
- `cov_bl`:
- `mu_bl_fm`:
- `cov_bl_fm`:
- `returns_fm`:
## Inputs of Worst Case Optimization Models.
- `cov_l`:
- `cov_u`:
- `cov_mu`:
- `cov_sigma`:
- `d_mu`:
- `k_mu`:
- `k_sigma`:
## Optimal portfolios.
- `optimal`:
- `limits`:
- `frontier`:
## Solutions.
- `solvers`:
- `opt_params`:
- `fail`:
- `model`:
- `z`:
## Allocation.
- `latest_prices`:
- `alloc_optimal`:
- `alloc_solvers`:
- `alloc_params`:
- `alloc_fail`:
- `alloc_model`:
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
    gmd_u::Real = Inf,
    rg_u::Real = Inf,
    rcvar_u::Real = Inf,
    tg_u::Real = Inf,
    rtg_u::Real = Inf,
    owa_u::Real = Inf,
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    krt_u::Real = Inf,
    skrt_u::Real = Inf,
    rvar_u::Real = Inf,
    rdar_u::Real = Inf,
    # Optimisation model inputs.
    mu_type::Symbol = :Default,
    mu::AbstractVector = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    L_2::AbstractMatrix = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
    S_2::AbstractMatrix = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
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
    limits::DataFrame = DataFrame(),
    frontier::AbstractDict = Dict(),
    # Solutions.
    solvers::AbstractDict = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
    model = JuMP.Model(),
    z::AbstractDict = Dict(),
    # Allocation.
    latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::AbstractDict = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model::AbstractDict = Dict(),
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
    @assert(b_sim > zero(b_sim), "a_sim = $b_sim, must be greater than or equal to zero")
    @assert(0 < kappa < 1, "kappa = $(kappa), must be greater than 0 and less than 1")
    @assert(
        0 < gs_threshold < 1,
        "gs_threshold = $gs_threshold, must be greater than 0 and less than 1"
    )
    @assert(
        kind_tracking_err ∈ TrackingErrKinds,
        "kind_tracking_err = $(kind_tracking_err), must be one of $TrackingErrKinds"
    )
    @assert(cov_type ∈ CovTypes, "cov_type = $cov_type, must be one of $CovTypes")
    @assert(mu_type ∈ MuTypes, "mu_type = $mu_type, must be one of $MuTypes")
    @assert(
        posdef_fix ∈ PosdefFixes,
        "posdef_fix = $posdef_fix, must be one of $PosdefFixes"
    )
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

    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices)))
        latest_prices = Vector(returns[end, setdiff(names(returns), ("timestamp",))])
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

    if !isempty(turnover_weights)
        @assert(
            length(turnover_weights) == length(assets),
            "length(turnover_weights) = $(turnover_weights) and length(assets) = $(length(assets)), must be equal"
        )
    end
    if !isempty(tracking_err_returns)
        @assert(
            length(tracking_err_returns) == size(returns, 1),
            "length(tracking_err_returns) = $tracking_err_returns and size(returns, 1) = $(size(returns,1)), must be equal"
        )
    end
    if !isempty(tracking_err_weights)
        @assert(
            length(tracking_err_weights) == length(assets),
            "length(tracking_err_weights) = $tracking_err_weights and length(assets) = $(length(assets)), must be equal"
        )
    end
    if !isempty(bl_bench_weights)
        @assert(
            length(bl_bench_weights) == length(assets),
            "length(bl_bench_weights) = $bl_bench_weights and length(assets) = $(length(assets)), must be equal"
        )
    end
    if !isempty(a_mtx_ineq)
        @assert(
            size(a_mtx_ineq, 2) == length(assets),
            "size(a_mtx_ineq, 2) = a_mtx_ineq must have the same number of columns size(a_mtx_ineq, 2) = $(size(a_mtx_ineq, 2)), as there are assets, length(assets) = $(length(assets))"
        )
    end
    if !isempty(risk_budget)
        @assert(
            length(risk_budget) == length(assets),
            "length(risk_budget) = $(length(risk_budget)), and length(assets) = $(length(assets)) must be equal"
        )
    end

    if !isempty(f_prices)
        f_returns = dropmissing!(DataFrame(percentchange(f_prices)))
    end

    if isa(f_returns, DataFrame) && !isempty(f_returns)
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

    at = zero(alpha)
    invat = zero(alpha)
    ln_k = zero(promote_type(typeof(alpha), typeof(kappa)))
    opk = zero(kappa)
    omk = zero(kappa)
    invk2 = zero(kappa)
    invk = zero(kappa)
    invopk = zero(kappa)
    invomk = zero(kappa)

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
        typeof(gmd_u),
        typeof(rg_u),
        typeof(rcvar_u),
        typeof(tg_u),
        typeof(rtg_u),
        typeof(owa_u),
        typeof(owa_w),
        typeof(krt_u),
        typeof(skrt_u),
        typeof(rvar_u),
        typeof(rdar_u),
        # Optimisation model inputs.
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
        typeof(limits),
        typeof(frontier),
        # Solutions.
        typeof(solvers),
        typeof(opt_params),
        typeof(fail),
        typeof(model),
        typeof(z),
        # Allocation.
        typeof(latest_prices),
        typeof(alloc_optimal),
        typeof(alloc_solvers),
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
        gmd_u,
        rg_u,
        rcvar_u,
        tg_u,
        rtg_u,
        owa_u,
        owa_w,
        krt_u,
        skrt_u,
        rvar_u,
        rdar_u,
        # Optimisation model inputs.
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
        limits,
        frontier,
        # Solutions.
        solvers,
        opt_params,
        fail,
        model,
        z,
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
    elseif sym == :alpha_i
        @assert(
            0 < val < obj.alpha < 1,
            "0 < alpha_i < alpha < 1: 0 < $val < $(obj.alpha) < 1 must hold"
        )
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
    elseif sym == :kind_tracking_err
        @assert(
            val ∈ TrackingErrKinds,
            "kind_tracking_err = $(val), must be one of $TrackingErrKinds"
        )
    elseif sym == :turnover_weights
        if !isempty(val)
            @assert(
                length(val) == length(obj.assets),
                "length(turnover_weights) = $val and length(assets) = $(length(obj.assets)), must be equal"
            )
        end
    elseif sym == :tracking_err_returns
        if !isempty(val)
            @assert(
                length(val) == size(obj.returns, 1),
                "length(tracking_err_returns) = $val and size(returns, 1) = $(size(obj.returns,1)), must be equal"
            )
        end
    elseif sym == :tracking_err_weights
        if !isempty(val)
            @assert(
                length(val) == length(obj.assets),
                "length(tracking_err_weights) = $val and length(assets) = $(length(obj.assets)), must be equal"
            )
        end
    elseif sym == :bl_bench_weights
        if !isempty(val)
            @assert(
                length(val) == length(obj.assets),
                "length(bl_bench_weights) = $val and length(assets) = $(length(obj.assets)), must be equal"
            )
        end
    elseif sym == :a_mtx_ineq
        if !isempty(val)
            @assert(
                size(val, 2) == length(obj.assets),
                "size(a_mtx_ineq, 2) = a_mtx_ineq must have the same number of columns size(a_mtx_ineq, 2) = $(size(val, 2)), as there are assets, length(assets) = $(length(obj.assets))"
            )
        end
    elseif sym == :risk_budget
        if !isempty(val)
            @assert(
                length(val) == length(obj.assets),
                "length(risk_budget) = $(length(val)), and length(assets) = $(length(obj.assets)) must be equal"
            )
        end
    elseif sym == :cov_type
        @assert(val ∈ CovTypes, "cov_type = $val, must be one of $CovTypes")
    elseif sym == :mu_type
        @assert(val ∈ MuTypes, "mu_type = $val, must be one of $MuTypes")
    elseif sym == :posdef_fix
        @assert(val ∈ PosdefFixes, "posdef_fix = $val, must be one of $PosdefFixes")
    elseif sym == :risk_budget
        if isempty(val)
            N = size(obj.returns, 2)
            val = fill(1 / N, N)
        else
            @assert(
                length(val) == size(obj.returns, 2),
                "length(risk_budget) == size(obj.returns, 2) must hold: $(length(val)) == $(size(obj.returns, 2))"
            )
            isa(val, AbstractRange) ? (val = collect(val / sum(val))) : (val ./= sum(val))
        end
    elseif sym ∈
           (:min_number_effective_assets, :max_number_assets, :max_number_assets_factor)
        @assert(val >= 0, "$sym = $val, must be greater than or equal to 0")
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
    elseif sym ∈ (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns)
        throw(
            ArgumentError(
                "$sym is related to other fields and therefore cannot be manually changed without compromising correctness, please create a new instance of Portfolio instead",
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
HRRiskMeasures = (
    :SD,
    :MAD,
    :SSD,
    :FLPM,
    :SLPM,
    :WR,
    :CVaR,
    :EVaR,
    :RVaR,
    :MDD,
    :ADD,
    :CDaR,
    :UCI,
    :EDaR,
    :RDaR,
    :Kurt,
    :SKurt,
    :GMD,
    :RG,
    :RCVaR,
    :TG,
    :RTG,
    :OWA,
    :Variance,
    :Equal,
    :VaR,
    :DaR,
    :DaR_r,
    :MDD_r,
    :ADD_r,
    :CDaR_r,
    :EDaR_r,
    :RDaR_r,
)
```
Available risk measures for optimisations of [`HCPortfolio`](@ref).
- `:SD` = standard deviation ([`SD`](@ref)).
- `:MAD` = max absolute deviation ([`MAD`](@ref)).
- `:SSD` = semi standard deviation ([`SSD`](@ref)).
- `:FLPM` = first lower partial moment (Omega ratio) ([`FLPM`](@ref)).
- `:SLPM` = second lower partial moment (Sortino ratio) ([`SLPM`](@ref)).
- `:WR` = worst realisation ([`WR`](@ref)).
- `:CVaR` = conditional value at risk ([`CVaR`](@ref)).
- `:EVaR` = entropic value at risk ([`EVaR`](@ref)).
- `:RVaR` = relativistic value at risk ([`RVaR`](@ref)).
- `:MDD` = maximum drawdown of uncompounded cumulative returns (Calmar ratio) ([`MDD_abs`](@ref)).
- `:ADD` = average drawdown of uncompounded cumulative returns ([`ADD_abs`](@ref)).
- `:CDaR` = conditional drawdown at risk of uncompounded cumulative returns ([`CDaR_abs`](@ref)).
- `:UCI` = ulcer index of uncompounded cumulative returns ([`UCI_abs`](@ref)).
- `:EDaR` = entropic drawdown at risk of uncompounded cumulative returns ([`EDaR_abs`](@ref)).
- `:RDaR` = relativistic drawdown at risk of uncompounded cumulative returns ([`RDaR_abs`](@ref)).
- `:Kurt` = square root kurtosis ([`Kurt`](@ref)).
- `:SKurt` = square root semi-kurtosis ([`SKurt`](@ref)).
- `:GMD` = gini mean difference ([`GMD`](@ref)).
- `:RG` = range of returns ([`RG`](@ref)).
- `:RCVaR` = range of conditional value at risk ([`RCVaR`](@ref)).
- `:TG` = tail gini ([`TG`](@ref)).
- `:RTG` = range of tail gini ([`RTG`](@ref)).
- `:OWA` = ordered weight array (generic OWA weights) ([`OWA`](@ref)).
- `:Variance` = variance ([`Variance`](@ref)).
- `:Equal` = equal risk contribution, `1/N` where N is the number of assets.
- `:VaR` = value at risk ([`VaR`](@ref)).
- `:DaR` = drawdown at risk of uncompounded cumulative returns ([`DaR_abs`](@ref)).
- `:DaR_r` = drawdown at risk of compounded cumulative returns ([`DaR_rel`](@ref)).
- `:MDD_r` = maximum drawdown of compounded cumulative returns ([`MDD_rel`](@ref)).
- `:ADD_r` = average drawdown of compounded cumulative returns ([`ADD_rel`](@ref)).
- `:CDaR_r` = conditional drawdown at risk of compounded cumulative returns ([`CDaR_rel`](@ref)).
- `:UCI_r` = ulcer index of compounded cumulative returns ([`UCI_rel`](@ref)).
- `:EDaR_r` = entropic drawdown at risk of compounded cumulative returns ([`EDaR_rel`](@ref)).
- `:RDaR_r` = relativistic drawdown at risk of compounded cumulative returns ([`RDaR_rel`](@ref)).
"""
const HRRiskMeasures = (
    RiskMeasures...,
    :Variance,
    :Equal,
    :VaR,
    :DaR,
    :DaR_r,
    :MDD_r,
    :ADD_r,
    :CDaR_r,
    :UCI_r,
    :EDaR_r,
    :RDaR_r,
)

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

```julia
HCPortfolio(;
    # Portfolio characteristics.
    returns = DataFrame(),
    ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
    assets = Vector{String}(undef, 0),
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
    # Optimisation parameters.
    mu_type::Symbol = :Default,
    mu = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    tjlogo::Bool = false,
    cov = Matrix{Float64}(undef, 0, 0),
    bins_info::Union{Symbol, Int} = :KN,
    w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,
    w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,
    codep_type::Symbol = :Pearson,
    codep = Matrix{Float64}(undef, 0, 0),
    dist = Matrix{Float64}(undef, 0, 0),
    clusters = Hclust{Float64}(Matrix{Int64}(undef, 0, 2), Float64[], Int64[], :nothing),
    k::Union{Int, Nothing} = nothing,
    # Solutions.
    solvers::AbstractDict = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
)
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
    solvers::AbstractDict = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
    # Allocation.
    latest_prices::AbstractVector = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::AbstractDict = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model::AbstractDict = Dict(),
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
    solvers::AbstractDict = Dict(),
    opt_params::AbstractDict = Dict(),
    fail::AbstractDict = Dict(),
    # Allocation.
    latest_prices::AbstractVector = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::AbstractDict = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model::AbstractDict = Dict(),
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
    @assert(b_sim > zero(b_sim), "a_sim = $b_sim, must be greater than zero")

    @assert(0 < kappa < 1, "kappa = $(kappa), must be greater than 0 and smaller than 1")
    @assert(
        0 < alpha_tail < 1,
        "alpha_tail = $alpha_tail, must be greater than 0 and smaller than 1"
    )
    @assert(cov_type ∈ CovTypes, "cov_type = $cov_type, must be one of $CovTypes")
    @assert(mu_type ∈ MuTypes, "mu_type = $mu_type, must be one of $MuTypes")
    @assert(
        posdef_fix ∈ PosdefFixes,
        "posdef_fix = $posdef_fix, must be one of $PosdefFixes"
    )
    @assert(
        0 < gs_threshold < 1,
        "gs_threshold = $gs_threshold, must be greater than zero and smaller than one"
    )

    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices)))
        latest_prices = Vector(returns[end, setdiff(names(returns), ("timestamp",))])
    end

    if isa(returns, DataFrame) && !isempty(returns)
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
        typeof(solvers),
        typeof(opt_params),
        typeof(fail),
        # Allocation.
        typeof(latest_prices),
        typeof(alloc_optimal),
        typeof(alloc_solvers),
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
    if sym == :alpha
        @assert(
            0 < obj.alpha_i < val < 1,
            "0 < alpha_i < alpha < 1: 0 < $(obj.alpha_i) < $val < 1, must hold"
        )
    elseif sym == :alpha_i
        @assert(
            0 < val < obj.alpha < 1,
            "0 < alpha_i < alpha < 1: 0 < $val < $(obj.alpha) < 1 must hold"
        )
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
            "0 < beta_i < beta < 1: 0 < $val < $(obj.beta) < 1 must hold"
        )
    elseif sym == :b_sim
        @assert(val >= zero(val), "b_sim = $val, must be greater than zero")
    elseif sym == :kappa
        @assert(0 < val < 1, "kappa = $(val), must be greater than 0 and smaller than 1")
    elseif sym == :alpha_tail
        @assert(0 < val < 1, "alpha_tail = $val, must be greater than 0 and smaller than 1")
    elseif sym == :cov_type
        @assert(val ∈ CovTypes, "cov_type = $val, must be one of $CovTypes")
    elseif sym == :mu_type
        @assert(val ∈ MuTypes, "mu_type = $val, must be one of $MuTypes")
    elseif sym == :codep_type
        @assert(val ∈ CodepTypes, "codep_type = $val, must be one of $CodepTypes")
    elseif sym == :posdef_fix
        @assert(val ∈ PosdefFixes, "posdef_fix = $val, must be one of $PosdefFixes")
    elseif sym == :gs_threshold
        @assert(
            0 < val < 1,
            "gs_threshold = $val, must be greater than zero and smaller than one"
        )
    elseif sym ∈ (:assets, :timestamps, :returns)
        throw(
            ArgumentError(
                "$sym is related to other fields and therefore cannot be manually changed without compromising correctness, please create a new instance of Portfolio instead",
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

export Portfolio, HCPortfolio
