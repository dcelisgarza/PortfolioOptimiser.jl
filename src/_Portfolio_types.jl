"""
```julia
AbstractPortfolio
```

Abstract type for subtyping portfolios.
"""
abstract type AbstractPortfolio end

"""
```julia
mutable struct Portfolio{ast, dat, r, s, us, ul, nal, nau, naus, tfa, tfdat, tretf, l, lo,
                         msvt, lpmt, ai, a, as, bi, b, bs, k, mnak, rb, rbw, to, tobw, kte,
                         te, rbi, bw, blbw, ami, bvi, rbv, frbv, nm, nsdp, np, ni, nis, amc,
                         bvc, ler, ud, umad, usd, ucvar, urcvar, uevar, urvar, uwr, ur,
                         uflpm, uslpm, umd, uad, ucdar, uuci, uedar, urdar, uk, usk, ugmd,
                         utg, urtg, uowa, owap, wowa, tmu, tcov, tkurt, tskurt, tl2, ts2,
                         tmuf, tcovf, trfm, tmufm, tcovfm, tmubl, tcovbl, tmublf, tcovblf,
                         tcovl, tcovu, tcovmu, tcovs, tdmu, tkmu, tks, topt, tz, tlim,
                         tfront, tsolv, tf, toptpar, tmod, tlp, taopt, tasolv, taoptpar,
                         taf, tamod} <: AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    short::s
    short_u::us
    long_u::ul
    num_assets_l::nal
    num_assets_u::nau
    num_assets_u_scale::naus
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    loadings_opt::lo
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
    rebalance::rb
    rebalance_weights::rbw
    turnover::to
    turnover_weights::tobw
    kind_tracking_err::kte
    tracking_err::te
    tracking_err_returns::rbi
    tracking_err_weights::bw
    bl_bench_weights::blbw
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget::rbv
    f_risk_budget::frbv
    network_method::nm
    network_sdp::nsdp
    network_penalty::np
    network_ip::ni
    network_ip_scale::nis
    a_vec_cent::amc
    b_cent::bvc
    mu_l::ler
    sd_u::ud
    mad_u::umad
    ssd_u::usd
    cvar_u::ucvar
    rcvar_u::urcvar
    evar_u::uevar
    rvar_u::urvar
    wr_u::uwr
    rg_u::ur
    flpm_u::uflpm
    slpm_u::uslpm
    mdd_u::umd
    add_u::uad
    cdar_u::ucdar
    uci_u::uuci
    edar_u::uedar
    rdar_u::urdar
    kurt_u::uk
    skurt_u::usk
    gmd_u::ugmd
    tg_u::utg
    rtg_u::urtg
    owa_u::uowa
    owa_p::owap
    owa_w::wowa
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    f_mu::tmuf
    f_cov::tcovf
    fm_returns::trfm
    fm_mu::tmufm
    fm_cov::tcovfm
    bl_mu::tmubl
    bl_cov::tcovbl
    blfm_mu::tmublf
    blfm_cov::tcovblf
    cov_l::tcovl
    cov_u::tcovu
    cov_mu::tcovmu
    cov_sigma::tcovs
    d_mu::tdmu
    k_mu::tkmu
    k_sigma::tks
    optimal::topt
    z::tz
    limits::tlim
    frontier::tfront
    solvers::tsolv
    opt_params::toptpar
    fail::tf
    model::tmod
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_solvers::tasolv
    alloc_params::taoptpar
    alloc_fail::taf
    alloc_model::tamod
end
```
Structure for portfolio optimisation. 
    
Some of these require external data from [`OptimiseOpt`](@ref) given to the [`optimise!`](@ref) function:

  - `type`: one of [`PortTypes`](@ref).
  - `rm`: one of [`RiskMeasures`](@ref).
  - `class`: one of [`PortClasses`](@ref).
  - `hist`: one of [`BLHist`](@ref).
  - `rf`: risk free rate.

In order for a parameter to be considered "appropriately defined", it must meet the following criteria:

  - `:Real`: must be finite.
  - `:Integer`: must be non-zero.
  - `:AbstractArray`: must be non-empty and of the proper shape.

Constraints are set if and only if all their constituents are appropriately defined.

Some constraints define decision variables using scaling factors. The scaling factor should be large enough to be outside of the problem's scale, but not too large as to incur numerical instabilities. Solution quality may be improved by changing the scaling factor.

# Inputs

## Portfolio characteristics

  - `assets`: `Na×1` vector of assets, where $(_ndef(:a2)).
  - `timestamps`: `T×1` vector of timestamps, where $(_tstr(:t1)).
  - `returns`: `T×Na` matrix of asset returns, where $(_tstr(:t1)) and $(_ndef(:a2)).
  - `short`: whether or not to allow negative weights, i.e. whether a portfolio accepts shorting.
  - `short_u`: absolute value of the sum of all short (negative) weights.
  - `long_u`: sum of all long (positive) weights.
  - `num_assets_l`: lower bound for the integer number of assets that make significant contributions to the final portfolio weights. $(_solver_reqs("`MOI.SecondOrderCone`"))
  - `num_assets_u`: upper bound for the integer number of assets that make significant contributions to the final portfolio weights. $(_solver_reqs("MIP constraints"))
  - `num_assets_u_scale`: scaling factor needed to create the decision variable for the `num_assets_u` constraint.
  - `f_assets`: `Nf×1` vector of factors, where $(_ndef(:f2)).
  - `f_timestamps`: `T×1` vector of factor timestamps, where $(_tstr(:t1)).
  - `f_returns`: `T×Nf` matrix of factor returns, where $(_tstr(:t1)) and $(_ndef(:f2)).
  - `loadings`: loadings matrix in dataframe form. Calling [`factor_statistics!`](@ref) will generate and set the dataframe. The number of rows must be equal to the number of asset and factor returns observations, `T`. Must have a few different columns.
    + `tickers`: (optional) contains the list of tickers.
    + `const`: (optional) contains the regression constant.
    + The other columns must be the names of the factors.

## Risk parameters

  - `msv_target`: 
    + `rm ∈ (:MAD, :SSD) || isfinite(mad_u) || isfinite(ssd_u)`: target value for Absolute Deviation and Semivariance risk measures.
      * `msv_target` is appropriately defined: the target for each column of the `returns` matrix is the broadcasted value of `msv_target`.
      * else: the target for each column of the `returns` matrix is the expected returns vector `mu`.
  - `lpm_target`: 
    + `rm ∈ (:FLPM, :SLPM) || isfinite(flpm_u) || isfinite(slpm_u)`: target value for the First and Second Lower Partial Moment risk measures. 
      * `lpm_target` is appropriately defined: the target for each column of the `returns` matrix is the broadcasted value of `lpm_target`.
      * else: the target for each column of the `returns` matrix is `rf`.
  - `alpha_i`:
    + `rm ∈ (:TG, :RTG)`: initial significance level of losses, `0 < alpha_i < alpha < 1`.
  - `a_sim`: 
    + `rm ∈ (:TG, :RTG)`: number of CVaRs to approximate the losses, `a_sim > 0`.
  - `alpha`:
    + `rm ∈ (:VaR, :CVaR, :EVaR, :RVaR, :RCVaR, :TG, :RTG, :DaR, :CDaR, :EDaR, :RDaR, :DaR_r, :CDaR_r, :EDaR_r, :RDaR_r)`: significance level of losses, `alpha ∈ (0, 1)`.
  - `beta_i`:
    + `rm == :RTG`: initial significance level of gains, `0 < beta_i < beta < 1`.
  - `b_sim`: 
    + `rm == :RTG`: number of CVaRs to approximate the gains, `b_sim > 0`.
  - `beta`:
    + `rm ∈ (:RCVaR, :RTG)`: significance level of gains, `beta ∈ (0, 1)`.
  - `kappa`: 
    + `rm ∈ (:RVaR, :RDaR, :RDaR_r)`: relativistic deformation parameter.
  - `max_num_assets_kurt`:
    + `iszero(max_num_assets_kurt)`: use the full kurtosis model.
    + `!iszero(max_num_assets_kurt)`: if the number of assets surpases this value, use the relaxed kurtosis model.

## Benchmark constraints

Only relevant when `type ∈ (:Trad, :WC)`.

  - `rebalance`: the rebalancing penalty is somewhat of an inverse of the `turnover` constraint. It defines a penalty for the objective function that penalises deviations away from a target weights vector.
    + `isa(turnover, Real)`: all assets have the same rebalancing penalty.
    + `isa(turnover, AbstractVector)`: each asset has its own rebalancing penalty.
  - `rebalance_weights`: define the target weights for the rebalancing penalty.

The rebalance penalty is defined as
```math
r_{i} \\rvert w_{i} - \\hat{w}_{i} \\lvert\\, \\forall\\, i \\in N\\,.
```
Where ``r_i`` is the rebalancing coefficient, ``w_i`` is the optimal weight for the `i'th` asset, ``\\hat{w}_i`` target weight for the `i'th` asset, and $(_ndef(:a3)).

  - `turnover`: the turnover constraint is somewhat of an inverse of the `rebalance` penalty.
    + `isa(turnover, Real)`: all assets have the same turnover value.
    + `isa(turnover, AbstractVector)`: each asset has its own turnover value.
  - `turnover_weights`: define the target weights for the turnover constraint.

The turnover constraint is defined as
```math
\\lvert w_{i} - \\hat{w}_{i}\\rvert \\leq t_{i} \\, \\forall\\, i \\in N\\,.
```
Where ``w_i`` is the optimal weight for the `i'th` asset, ``\\hat{w}_i`` target weight for the `i'th` asset, ``t_{i}`` is the value of the turnover for the `i'th` asset, and $(_ndef(:a3)).

  - `kind_tracking_err`: kind of tracking error from [`TrackingErrKinds`](@ref).
  - `tracking_err`: define the value of the tracking error.
  - `tracking_err_returns`: `T×1` vector of benchmark returns for the tracking error constraint as per [`TrackingErrKinds`](@ref), where $(_tstr(:t1)).
  - `tracking_err_weights`: `Na×1` vector of weights used for computing the benchmark vector for the tracking error constraint as per [`TrackingErrKinds`](@ref), where $(_ndef(:a2)).

The tracking error constraint is defined as
```math
\\sqrt{\\dfrac{1}{T-1}\\sum\\limits_{i=1}^{T}\\left(\\mathbf{X}_{i} \\bm{w} - b_{i}\\right)^{2}}\\leq t\\,.
```
Where ``\\mathbf{X}_{i}`` is the `i'th` observation (row) of the returns matrix ``\\mathbf{X}``, ``\\bm{w}`` is the vector of optimal asset weights, ``b_{i}`` is the `i'th` observation of the benchmark returns vector, ``t`` the tracking error, and $(_tstr(:t2)).

  - `bl_bench_weights`: `Na×1` vector of benchmark weights for Black Litterman models, where $(_ndef(:a2)).

## Asset constraints

The constraint is only defined when both `a_mtx_ineq` and `b_vec_ineq` are defined.
- `a_mtx_ineq`: `C×Na` A matrix of the linear asset constraints, where `C` is the number of constraints, and $(_ndef(:a2)).
- `b_vec_ineq`: `C×1` B vector of the linear asset constraints, where `C` is the number of constraints.
The linear asset constraint is defined as
```math
\\mathbf{A} \\bm{w} \\geq \\bm{B}\\,.
```
Where ``\\mathbf{A}`` is the matrix of linear constraints `a_mtx_ineq`, ``\\bm{w}`` the asset weights, and ``\\bm{B}`` is the vector of linear asset constraints `b_vec_ineq`.

## Risk budget constraints

Only relevant when `type ∈ (:RP, :RRP)`.
- `risk_budget`: 
  + `class != :FC || type == :RRP`: `Na×1` asset risk budget constraint vector for risk measure `rm`, where $(_ndef(:a2)). Asset `i` contributes the amount of `rm` risk in position `risk_budget[i]`.
- `f_risk_budget`: 
  + `class == :FC && type == :RP`: `Nf×1` factor risk budget constraint vector for risk measure `rm`, where $(_ndef(:f2)). Factor `i` contributes the amount of `rm` risk in position `f_risk_budget[i]`.

## Network constraints

Only relevant when `type ∈ (:Trad, :WC)`.
- `network_method`: network constraint method from [`NetworkMethods`](@ref).
- `network_sdp`: 
  + `network_method == :SDP`: network matrix.
- `network_penalty`:
  + `network_method == :SDP && rm != :SD`: weight of the SDP network constraint.
- `network_ip`: 
  + `network_method == :IP`: network matrix.
- `network_ip_scale`:
  + `network_method == :IP`: scaling factor needed to create the decision variable for the constraint. Changing the value of the scaling factor can improve the solution.
- `a_vec_cent`: `Na×1` centrality measure vector for the centrality constraint, where $(_ndef(:a2)).
- `b_cent`: average centrality measure for the centraility constraint.
The centrality measure constraint is defined as
```math
\\bm{A} \\cdot \\bm{w} = b\\,.
```
Where ``\\bm{A}`` is the centrality measure vector, ``\\bm{w}`` the portfolio weights, and ``b`` the centrality measure vector. The constraint is only applied when both `a_vec_cent` and `b_cent` are defined.

## Bounds constraints

The bounds constraints are *only active if they are finite*. 
- Lower bounds denoted by the suffix `_l`.
- Upper bounds denoted by the suffix `_u`. 
The risk upper bounds are named after their corresponding [`RiskMeasures`](@ref) in lower case. They have the same solver requirements as their corresponding risk measure. Multiple bounds constraints can be active at any time but may make the problem infeasable.

- `mu_l`: mean expected return.
- `sd_u`: standard deviation.
- `mad_u`: max absolute devia.
- `ssd_u`: semi standard deviation.
- `cvar_u`: critical value at risk.
- `rcvar_u`: critical value at risk range.
- `evar_u`: entropic value at risk.
- `rvar_u`: relativistic value at risk.
- `wr_u`: worst realisation.
- `rg_u`: range.
- `flpm_u`: first lower partial moment.
- `slpm_u`: second lower partial moment.
- `mdd_u`: max drawdown.
- `add_u`: average drawdown.
- `cdar_u`: critical drawdown at risk.
- `uci_u`: ulcer index.
- `edar_u`: entropic drawdown at risk.
- `rdar_u`: relativistic drawdown at risk.
- `kurt_u`: square root kurtosis.
- `skurt_u`: square root semi kurtosis.
- `gmd_u`: gini mean difference.
- `tg_u`: tail gini.
- `rtg_u`: tail gini range.
- `owa_u`: custom ordered weight risk (used with `owa_w`).

## OWA parameters

Only relevant when `rm ∈ (:GMD, :TG, :RTG, :OWA)`.
- `owa_p`: 
  + `owa_approx = true`: C×1` vector containing the order of the p-norms used in the approximate formulation of the risk measures, where `C` is the number of p-norms. The more entries and larger their range, the more accurate the approximation.
- `owa_w`: 
  + `rm == :OWA`: `T×1` OWA vector, where $(_tstr(:t1)). Useful for optimising higher L-moments.

## Model statistics

- `mu`:
  + `:class ∈ (:Classic, :FM)`: `Na×1` asset expected returns vector, where $(_ndef(:a2)).
- `cov`:
  + `:class ∈ (:Classic, :FM)`: `Na×Na` asset covariance matrix, where $(_ndef(:a2)).
- `kurt`: `(Na^2)×(Na^2)` asset cokurtosis matrix, where $(_ndef(:a2)).
- `skurt`: `(Na^2)×(Na^2)` asset semi cokurtosis matrix, where $(_ndef(:a2)).
- `L_2`: `(Na^2)×((Na^2 + Na)/2)` elimination matrix, where $(_ndef(:a2)).
- `S_2`: `((Na^2 + Na)/2)×(Na^2)` summation matrix, where $(_ndef(:a2)).
- `f_mu`: `Nf×1` factor expected returns vector, where $(_ndef(:f2)).
- `f_cov`: `Nf×Nf` factor covariance matrix, where $(_ndef(:f2)).
- `fm_returns`:
  + `:class ∈ (:Classic, :FM) || :hist == 1`: `T×Na` matrix of factor adjusted asset returns, where $(_tstr(:t1)) and $(_ndef(:a2)).
- `fm_mu`: `Na×1` factor adjusted asset expected returns vector, where $(_ndef(:a2)).
- `fm_cov`: `Na×Na` factor adjusted asset covariance matrix, where $(_ndef(:a2)).
- `bl_mu`: `Na×1` Black-Litterman adjusted asset expected returns vector, where $(_ndef(:a2)).
- `bl_cov`: `Na×Na` Black-Litterman adjusted asset covariance matrix, where $(_ndef(:a2)).
- `blfm_mu`: `Na×1` Black-Litterman factor model adjusted asset expected returns vector, where $(_ndef(:a2)).
- `blfm_cov`: `Na×Na` Black-Litterman factor model adjusted asset covariance matrix, where $(_ndef(:a2)).
- `cov_l`: `Na×Na` worst case lower bound for the asset covariance matrix, where $(_ndef(:a2)).
- `cov_u`: `Na×Na` worst case upper bound for the asset covariance matrix, where $(_ndef(:a2)).
- `cov_mu`: `Na×Na` matrix of the estimation errors of the asset expected returns vector for elliptical sets in worst case optimisation, where $(_ndef(:a2)).
- `cov_sigma`: `Na×Na` matrix of the estimation errors of the asset covariance matrix, where $(_ndef(:a2)).
- `d_mu`: `Na×1` absolute deviation of the worst case upper and lower asset expected returns vectors, where $(_ndef(:a2)).
- `k_mu`: distance parameter of the elliptical uncertainty in the asset expected returns vector for the worst case optimisation.
- `k_sigma`: distance parameter of the elliptical uncertainty in the asset covariance matrix for the worst case optimisation.

## Optimal portfolios

- `optimal`: $_edst for storing optimal portfolios. $(_filled_by("[`optimise!`](@ref)"))
- `z`: $_edst for storing optimal `z` values of portfolios optimised for entropy and relativistic risk measures. $(_filled_by("[`optimise!`](@ref)"))
- `limits`: $_edst for storing the minimal and maximal risk portfolios for given risk measures. $(_filled_by("[`frontier_limits!`](@ref)"))
- `frontier`: $_edst containing points in the efficient frontier for given risk measures. $(_filled_by("[`efficient_frontier!`](@ref)"))
# Solutions
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
"""
mutable struct Portfolio{ast, dat, r, s, us, ul, nal, nau, naus, tfa, tfdat, tretf, l, lo,
                         msvt, lpmt, ai, a, as, bi, b, bs, k, mnak, rb, rbw, to, tobw, kte,
                         te, rbi, bw, blbw, ami, bvi, rbv, frbv, nm, nsdp, np, ni, nis, amc,
                         bvc, ler, ud, umad, usd, ucvar, urcvar, uevar, urvar, uwr, ur,
                         uflpm, uslpm, umd, uad, ucdar, uuci, uedar, urdar, uk, usk, ugmd,
                         utg, urtg, uowa, owap, wowa, tmu, tcov, tkurt, tskurt, tl2, ts2,
                         tmuf, tcovf, trfm, tmufm, tcovfm, tmubl, tcovbl, tmublf, tcovblf,
                         tcovl, tcovu, tcovmu, tcovs, tdmu, tkmu, tks, topt, tz, tlim,
                         tfront, tsolv, tf, toptpar, tmod, tlp, taopt, tasolv, taoptpar,
                         taf, tamod} <: AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    short::s
    short_u::us
    long_u::ul
    num_assets_l::nal
    num_assets_u::nau
    num_assets_u_scale::naus
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    loadings_opt::lo
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
    rebalance::rb
    rebalance_weights::rbw
    turnover::to
    turnover_weights::tobw
    kind_tracking_err::kte
    tracking_err::te
    tracking_err_returns::rbi
    tracking_err_weights::bw
    bl_bench_weights::blbw
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget::rbv
    f_risk_budget::frbv
    network_method::nm
    network_sdp::nsdp
    network_penalty::np
    network_ip::ni
    network_ip_scale::nis
    a_vec_cent::amc
    b_cent::bvc
    mu_l::ler
    sd_u::ud
    mad_u::umad
    ssd_u::usd
    cvar_u::ucvar
    rcvar_u::urcvar
    evar_u::uevar
    rvar_u::urvar
    wr_u::uwr
    rg_u::ur
    flpm_u::uflpm
    slpm_u::uslpm
    mdd_u::umd
    add_u::uad
    cdar_u::ucdar
    uci_u::uuci
    edar_u::uedar
    rdar_u::urdar
    kurt_u::uk
    skurt_u::usk
    gmd_u::ugmd
    tg_u::utg
    rtg_u::urtg
    owa_u::uowa
    owa_p::owap
    owa_w::wowa
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    f_mu::tmuf
    f_cov::tcovf
    fm_returns::trfm
    fm_mu::tmufm
    fm_cov::tcovfm
    bl_mu::tmubl
    bl_cov::tcovbl
    blfm_mu::tmublf
    blfm_cov::tcovblf
    cov_l::tcovl
    cov_u::tcovu
    cov_mu::tcovmu
    cov_sigma::tcovs
    d_mu::tdmu
    k_mu::tkmu
    k_sigma::tks
    optimal::topt
    z::tz
    limits::tlim
    frontier::tfront
    solvers::tsolv
    opt_params::toptpar
    fail::tf
    model::tmod
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
    prices::TimeArray = TimeArray(TimeType[], []),    returns::DataFrame = DataFrame(),    ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    timestamps::AbstractVector = Vector{Date}(undef, 0),    assets::AbstractVector = Vector{String}(undef, 0),    short::Bool = false,    short_u::Real = 0.2,    long_u::Real = 1.0,    num_assets_l::Integer = 0,    num_assets_u::Integer = 0,    num_assets_u_scale::Real = 100_000.0,    f_prices::TimeArray = TimeArray(TimeType[], []),    f_returns::DataFrame = DataFrame(),    f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    f_timestamps::AbstractVector = Vector{Date}(undef, 0),    f_assets::AbstractVector = Vector{String}(undef, 0),    loadings::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    # Risk parameters
    msv_target::Union{<:Real, AbstractVector{<:Real}} = Inf,    lpm_target::Union{<:Real, AbstractVector{<:Real}} = Inf,    alpha_i::Real = 0.0001,    alpha::Real = 0.05,    a_sim::Integer = 100,    beta_i::Real = alpha_i,    beta::Real = alpha,    b_sim::Integer = a_sim,    kappa::Real = 0.3,    max_num_assets_kurt::Integer = 0,    # Benchmark constraints
    turnover::Real = Inf,    turnover_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    kind_tracking_err::Symbol = :Weights,    tracking_err::Real = Inf,    tracking_err_returns::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    tracking_err_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    # Asset constraints
    a_mtx_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    b_vec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    # Bounds constraints
    mu_l::Real = Inf,    sd_u::Real = Inf,    mad_u::Real = Inf,    ssd_u::Real = Inf,    cvar_u::Real = Inf,    wr_u::Real = Inf,    flpm_u::Real = Inf,    slpm_u::Real = Inf,    mdd_u::Real = Inf,    add_u::Real = Inf,    cdar_u::Real = Inf,    uci_u::Real = Inf,    evar_u::Real = Inf,    edar_u::Real = Inf,    rvar_u::Real = Inf,    rdar_u::Real = Inf,    kurt_u::Real = Inf,    skurt_u::Real = Inf,    gmd_u::Real = Inf,    rg_u::Real = Inf,    rcvar_u::Real = Inf,    tg_u::Real = Inf,    rtg_u::Real = Inf,    owa_u::Real = Inf,    # OWA parameters
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    # Model statistics
    mu::AbstractVector = Vector{Float64}(undef, 0),    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    f_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    f_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    fm_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    fm_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    bl_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    bl_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    blfm_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    blfm_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    fm_returns::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    
    cov_l::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    cov_u::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    cov_mu::AbstractMatrix{<:Real} = Diagonal{Float64}(undef, 0),    cov_sigma::AbstractMatrix{<:Real} = Diagonal{Float64}(undef, 0),    d_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    k_mu::Real = Inf,    k_sigma::Real = Inf,    # Optimal portfolios
    optimal::AbstractDict = Dict(),    z::AbstractDict = Dict(),    limits::AbstractDict = Dict(),    frontier::AbstractDict = Dict(),    # Solutions
    solvers::Union{<:AbstractDict, NamedTuple} = Dict(),    opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),    fail::AbstractDict = Dict(),    model::JuMP.Model = JuMP.Model(),    # Allocation
    latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    alloc_optimal::AbstractDict = Dict(),    alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),    alloc_params::Union{<:AbstractDict, NamedTuple} = Dict(),    alloc_fail::AbstractDict = Dict(),    alloc_model::JuMP.Model = JuMP.Model())
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
- `num_assets_l`: 
  + `!iszero(num_assets_l)`: guarantees that at least this number of assets make significant contributions to the final portfolio weights.
- `num_assets_u`: if non-zero, guarantees at most this number of assets make non-zero contributions to the final portfolio weights. Requires an optimiser that supports binary variables.
- `num_assets_u_scale`: scaling factor needed to create a decision variable when `num_assets_u` is non-zero.
- `f_prices`: `(T+1)×Nf` `TimeArray` of factor prices, where the time stamp field is `f_timestamp`, where $(_tstr(:t1)) and $(_ndef(:f2)). If `f_prices` is not empty, then `f_returns`, `f_ret`, `f_timestamps`, `f_assets`, and `latest_prices` are ignored because their respective fields are obtained from `f_prices`.
- `f_returns`: `T×(Nf+1)` `DataFrame` of factor returns, where $(_tstr(:t1)) and $(_ndef(:f2)), the extra column is `f_timestamp`, which contains the timestamps of the factor returns. If `f_prices` is empty and `f_returns` is not empty, `f_ret`, `f_timestamps`, and `f_assets` are ignored because their respective fields are obtained from `f_returns`.
- `f_ret`: `T×Nf` matrix of factor returns, where $(_ndef(:f2)). Its value is saved in the `f_returns` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function, where $(_tstr(:t1)) and $(_ndef(:f2)).
- `f_timestamps`: `T×1` vector of factor timestamps, where $(_tstr(:t1)). Its value is saved in the `f_timestamps` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function.
- `f_assets`: `Nf×1` vector of factors, where $(_ndef(:f2)). Its value is saved in the `f_assets` field of [`Portfolio`](@ref). If `f_prices` or `f_returns` are not empty, this value is obtained from within the function.
## Risk parameters
- `msv_target`: target value for for Absolute Deviation and Semivariance risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the mean returns vector `mu`.
    - Else the target is the value of `msv_target`. If `msv_target` is a vector, its length should be `Na`, where $(_ndef(:a2)).
- `lpm_target`: target value for the First and Second Lower Partial Moment risk measures. It can have two meanings depending on its type and value.
    - If it's a `Real` number and infinite, or an empty vector. The target will be the value of the risk-free rate `rf`, provided in the [`optimise!`](@ref) function.
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
## Asset constraints
- `a_mtx_ineq`: `C×Na` A matrix of the linear asset constraints ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``, where `C` is the number of constraints, and $(_ndef(:a2)).
- `b_vec_ineq`: `C×1` B vector of the linear asset constraints ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``, where `C` is the number of constraints.
- `risk_budget`: `Na×1` risk budget constraint vector for risk parity optimisations, where $(_ndef(:a2)).
### Bounds constraints
The bounds constraints are only active if they are finite. They define lower bounds denoted by the suffix `_l`, and upper bounds denoted by the suffix `_u`, of various portfolio characteristics. The risk upper bounds are named after their corresponding [`RiskMeasures`](@ref) in lower case, they also bring the same solver requirements as their corresponding risk measure. Multiple bounds constraints can be active at any time but may make the problem infeasable.
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
## OWA parameters
- `owa_w`: `T×1` OWA vector, where $(_tstr(:t1)) containing. Useful for optimising higher OWA L-moments.
## Model statistics
- `mu`: $(_mudef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `cov`: $(_covdef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `kurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the cokurtosis matrix at instance construction. The cokurtosis matrix `kurt` can be computed by calling [`cokurt_mtx`](@ref).
- `skurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the semi cokurtosis matrix at instance construction. The semi cokurtosis matrix `skurt` can be computed by calling [`cokurt_mtx`](@ref).
- `f_mu`: $(_mudef("factors", :f2)) $(_dircomp("[`factor_statistics!`](@ref)"))
- `f_cov`: $(_covdef("factors", :f2)) $(_dircomp("[`factor_statistics!`](@ref)"))
- `fm_mu`: $(_mudef("feature selected factors")) $(_dircomp("[`factor_statistics!`](@ref)"))
- `fm_cov`: $(_covdef("feature selected factors")) $(_dircomp("[`factor_statistics!`](@ref)"))
- `bl_mu`: $(_mudef("Black Litterman")) $(_dircomp("[`black_litterman_statistics!`](@ref)"))
- `bl_cov`: $(_covdef("Black Litterman")) $(_dircomp("[`black_litterman_statistics!`](@ref)"))
- `blfm_mu`: $(_mudef("Black Litterman feature selected factors")) $(_dircomp("[`black_litterman_factor_satistics!`](@ref)"))
- `blfm_cov`: $(_covdef("Black Litterman feature selected factors")) $(_dircomp("[`black_litterman_factor_satistics!`](@ref)"))
- `fm_returns`: `T×Na` matrix of feature selcted adjusted returns, where $(_tstr(:t1)) and $(_ndef(:a2)). $(_dircomp("[`factor_statistics!`](@ref)"))
#
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
                   prices::TimeArray                                 = TimeArray(TimeType[], []),
                   returns::DataFrame                                = DataFrame(),
                   ret::AbstractMatrix{<:Real}                       = Matrix{Float64}(undef, 0, 0),
                   timestamps::AbstractVector                        = Vector{Date}(undef, 0),
                   assets::AbstractVector                            = Vector{String}(undef, 0),
                   short::Bool                                       = false,
                   short_u::Real                                     = 0.2,
                   long_u::Real                                      = 1.0,
                   num_assets_l::Integer                             = 0,
                   num_assets_u::Integer                             = 0,
                   num_assets_u_scale::Real                          = 100_000.0,
                   f_prices::TimeArray                               = TimeArray(TimeType[], []),
                   f_returns::DataFrame                              = DataFrame(),
                   f_ret::AbstractMatrix{<:Real}                     = Matrix{Float64}(undef, 0, 0),
                   f_timestamps::AbstractVector                      = Vector{Date}(undef, 0),
                   f_assets::AbstractVector                          = Vector{String}(undef, 0),
                   loadings::DataFrame                               = DataFrame(),
                   loadings_opt::Union{LoadingsOpt, Nothing}         = nothing,
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
                   rebalance::Union{Real, AbstractVector{<:Real}}    = Inf,
                   rebalance_weights::AbstractVector{<:Real}         = Vector{Float64}(undef, 0),
                   turnover::Union{Real, AbstractVector{<:Real}}     = Inf,
                   turnover_weights::AbstractVector{<:Real}          = Vector{Float64}(undef, 0),
                   kind_tracking_err::Symbol                         = :None,
                   tracking_err::Real                                = Inf,
                   tracking_err_returns::AbstractVector{<:Real}      = Vector{Float64}(undef, 0),
                   tracking_err_weights::AbstractVector{<:Real}      = Vector{Float64}(undef, 0),
                   bl_bench_weights::AbstractVector{<:Real}          = Vector{Float64}(undef, 0),
                   a_mtx_ineq::AbstractMatrix{<:Real}                = Matrix{Float64}(undef, 0, 0),
                   b_vec_ineq::AbstractVector{<:Real}                = Vector{Float64}(undef, 0),
                   risk_budget::AbstractVector{<:Real}               = Vector{Float64}(undef, 0),
                   f_risk_budget::AbstractVector{<:Real}             = Vector{Float64}(undef, 0),
                   network_method::Symbol                            = :None,
                   network_sdp::AbstractMatrix{<:Real}               = Matrix{Float64}(undef, 0, 0),
                   network_penalty::Real                             = 0.05,
                   network_ip::AbstractMatrix{<:Real}                = Matrix{Float64}(undef, 0, 0),
                   network_ip_scale::Real                            = 100_000.0,
                   a_vec_cent::AbstractVector{<:Real}                = Vector{Float64}(undef, 0),
                   b_cent::Real                                      = Inf,
                   mu_l::Real                                        = Inf,
                   sd_u::Real                                        = Inf,
                   mad_u::Real                                       = Inf,
                   ssd_u::Real                                       = Inf,
                   cvar_u::Real                                      = Inf,
                   rcvar_u::Real                                     = Inf,
                   evar_u::Real                                      = Inf,
                   rvar_u::Real                                      = Inf,
                   wr_u::Real                                        = Inf,
                   rg_u::Real                                        = Inf,
                   flpm_u::Real                                      = Inf,
                   slpm_u::Real                                      = Inf,
                   mdd_u::Real                                       = Inf,
                   add_u::Real                                       = Inf,
                   cdar_u::Real                                      = Inf,
                   uci_u::Real                                       = Inf,
                   edar_u::Real                                      = Inf,
                   rdar_u::Real                                      = Inf,
                   kurt_u::Real                                      = Inf,
                   skurt_u::Real                                     = Inf,
                   gmd_u::Real                                       = Inf,
                   tg_u::Real                                        = Inf,
                   rtg_u::Real                                       = Inf,
                   owa_u::Real                                       = Inf,
                   owa_p::AbstractVector{<:Real}                     = Float64[2, 3, 4, 10, 50],
                   owa_w::AbstractVector{<:Real}                     = Vector{Float64}(undef, 0),
                   mu::AbstractVector                                = Vector{Float64}(undef, 0),
                   cov::AbstractMatrix{<:Real}                       = Matrix{Float64}(undef, 0, 0),
                   kurt::AbstractMatrix{<:Real}                      = Matrix{Float64}(undef, 0, 0),
                   skurt::AbstractMatrix{<:Real}                     = Matrix{Float64}(undef, 0, 0),
                   f_mu::AbstractVector{<:Real}                      = Vector{Float64}(undef, 0),
                   f_cov::AbstractMatrix{<:Real}                     = Matrix{Float64}(undef, 0, 0),
                   fm_returns::AbstractMatrix{<:Real}                = Matrix{Float64}(undef, 0, 0),
                   fm_mu::AbstractVector{<:Real}                     = Vector{Float64}(undef, 0),
                   fm_cov::AbstractMatrix{<:Real}                    = Matrix{Float64}(undef, 0, 0),
                   bl_mu::AbstractVector{<:Real}                     = Vector{Float64}(undef, 0),
                   bl_cov::AbstractMatrix{<:Real}                    = Matrix{Float64}(undef, 0, 0),
                   blfm_mu::AbstractVector{<:Real}                   = Vector{Float64}(undef, 0),
                   blfm_cov::AbstractMatrix{<:Real}                  = Matrix{Float64}(undef, 0, 0),
                   cov_l::AbstractMatrix{<:Real}                     = Matrix{Float64}(undef, 0, 0),
                   cov_u::AbstractMatrix{<:Real}                     = Matrix{Float64}(undef, 0, 0),
                   cov_mu::AbstractMatrix{<:Real}                    = Matrix{Float64}(undef, 0, 0),
                   cov_sigma::AbstractMatrix{<:Real}                 = Matrix{Float64}(undef, 0, 0),
                   d_mu::AbstractVector{<:Real}                      = Vector{Float64}(undef, 0),
                   k_mu::Real                                        = Inf,
                   k_sigma::Real                                     = Inf,
                   optimal::AbstractDict                             = Dict(),
                   z::AbstractDict                                   = Dict(),
                   limits::AbstractDict                              = Dict(),
                   frontier::AbstractDict                            = Dict(),
                   solvers::Union{<:AbstractDict, NamedTuple}        = Dict(),
                   opt_params::Union{<:AbstractDict, NamedTuple}     = Dict(),
                   fail::AbstractDict                                = Dict(),
                   model::JuMP.Model                                 = JuMP.Model(),
                   latest_prices::AbstractVector{<:Real}             = Vector{Float64}(undef, 0),
                   alloc_optimal::AbstractDict                       = Dict(),
                   alloc_solvers::Union{<:AbstractDict, NamedTuple}  = Dict(),
                   alloc_params::Union{<:AbstractDict, NamedTuple}   = Dict(),
                   alloc_fail::AbstractDict                          = Dict(),
                   alloc_model::JuMP.Model                           = JuMP.Model())
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

    @smart_assert(num_assets_l >= zero(num_assets_l))
    @smart_assert(num_assets_u >= zero(num_assets_u))
    @smart_assert(num_assets_u_scale >= zero(num_assets_u_scale))
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    @smart_assert(zero(beta) < beta_i < beta < one(beta))
    @smart_assert(b_sim > zero(b_sim))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    @smart_assert(max_num_assets_kurt >= zero(max_num_assets_kurt))
    if isa(rebalance, AbstractVector) && !isempty(rebalance)
        @smart_assert(length(rebalance) == size(returns, 2) &&
                      all(rebalance .>= zero(rebalance)))
    elseif isa(rebalance, Real)
        @smart_assert(rebalance >= zero(rebalance))
    end
    if !isempty(rebalance_weights)
        @smart_assert(length(rebalance_weights) == size(returns, 2))
    end
    if isa(turnover, AbstractVector) && !isempty(turnover)
        @smart_assert(length(turnover) == size(returns, 2) &&
                      all(turnover .>= zero(turnover)))
    elseif isa(turnover, Real)
        @smart_assert(turnover >= zero(turnover))
    end
    if !isempty(turnover_weights)
        @smart_assert(length(turnover_weights) == size(returns, 2))
    end
    @smart_assert(kind_tracking_err ∈ TrackingErrKinds)
    @smart_assert(tracking_err >= zero(tracking_err))
    if !isempty(tracking_err_returns)
        @smart_assert(length(tracking_err_returns) == size(returns, 1))
    end
    if !isempty(tracking_err_weights)
        @smart_assert(length(tracking_err_weights) == size(returns, 2))
    end
    if !isempty(bl_bench_weights)
        @smart_assert(length(bl_bench_weights) == size(returns, 2))
    end
    @smart_assert(network_method ∈ NetworkMethods)
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
        @smart_assert(all(risk_budget .>= zero(eltype(returns))))

        if isa(risk_budget, AbstractRange)
            risk_budget = collect(risk_budget / sum(risk_budget))
        else
            risk_budget ./= sum(risk_budget)
        end
    end
    if !isempty(f_risk_budget)
        @smart_assert(all(f_risk_budget .>= zero(eltype(returns))))

        if isa(f_risk_budget, AbstractRange)
            f_risk_budget = collect(f_risk_budget / sum(f_risk_budget))
        else
            f_risk_budget ./= sum(f_risk_budget)
        end
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
    if !isempty(f_mu)
        @smart_assert(length(f_mu) == size(f_returns, 2))
    end
    if !isempty(f_cov)
        @smart_assert(size(f_cov, 1) == size(f_cov, 2) == size(f_returns, 2))
    end
    if !isempty(fm_mu)
        @smart_assert(length(fm_mu) == size(returns, 2))
    end
    if !isempty(fm_cov)
        @smart_assert(size(fm_cov, 1) == size(fm_cov, 2) == size(returns, 2))
    end
    if !isempty(bl_mu)
        @smart_assert(length(bl_mu) == size(returns, 2))
    end
    if !isempty(bl_cov)
        @smart_assert(size(bl_cov, 1) == size(bl_cov, 2) == size(returns, 2))
    end
    if !isempty(blfm_mu)
        @smart_assert(length(blfm_mu) == size(returns, 2))
    end
    if !isempty(blfm_cov)
        @smart_assert(size(blfm_cov, 1) == size(blfm_cov, 2) == size(returns, 2))
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

    return Portfolio{typeof(assets), typeof(timestamps), typeof(returns), typeof(short),
                     typeof(short_u), typeof(long_u), typeof(num_assets_l),
                     typeof(num_assets_u), typeof(num_assets_u_scale), typeof(f_assets),
                     typeof(f_timestamps), typeof(f_returns), typeof(loadings),
                     Union{LoadingsOpt, Nothing}, Union{<:Real, AbstractVector{<:Real}},
                     Union{<:Real, AbstractVector{<:Real}}, typeof(alpha_i), typeof(alpha),
                     typeof(a_sim), typeof(beta_i), typeof(beta), typeof(b_sim),
                     typeof(kappa), typeof(max_num_assets_kurt),
                     Union{<:Real, AbstractVector{<:Real}}, typeof(rebalance_weights),
                     Union{<:Real, AbstractVector{<:Real}}, typeof(turnover_weights),
                     typeof(kind_tracking_err), typeof(tracking_err),
                     typeof(tracking_err_returns), typeof(tracking_err_weights),
                     typeof(bl_bench_weights), typeof(a_mtx_ineq), typeof(b_vec_ineq),
                     typeof(risk_budget), typeof(f_risk_budget), typeof(network_method),
                     typeof(network_sdp), typeof(network_penalty), typeof(network_ip),
                     typeof(network_ip_scale), typeof(a_vec_cent), typeof(b_cent),
                     typeof(mu_l), typeof(sd_u), typeof(mad_u), typeof(ssd_u),
                     typeof(cvar_u), typeof(rcvar_u), typeof(evar_u), typeof(rvar_u),
                     typeof(wr_u), typeof(rg_u), typeof(flpm_u), typeof(slpm_u),
                     typeof(mdd_u), typeof(add_u), typeof(cdar_u), typeof(uci_u),
                     typeof(edar_u), typeof(rdar_u), typeof(kurt_u), typeof(skurt_u),
                     typeof(gmd_u), typeof(tg_u), typeof(rtg_u), typeof(owa_u),
                     typeof(owa_p), typeof(owa_w), typeof(mu), typeof(cov), typeof(kurt),
                     typeof(skurt), typeof(L_2), typeof(S_2), typeof(f_mu), typeof(f_cov),
                     typeof(fm_returns), typeof(fm_mu), typeof(fm_cov), typeof(bl_mu),
                     typeof(bl_cov), typeof(blfm_mu), typeof(blfm_cov), typeof(cov_l),
                     typeof(cov_u), typeof(cov_mu), typeof(cov_sigma), typeof(d_mu),
                     typeof(k_mu), typeof(k_sigma), typeof(optimal), typeof(z),
                     typeof(limits), typeof(frontier), Union{<:AbstractDict, NamedTuple},
                     Union{<:AbstractDict, NamedTuple}, typeof(fail), typeof(model),
                     typeof(latest_prices), typeof(alloc_optimal),
                     Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                     typeof(alloc_fail), typeof(alloc_model)}(assets, timestamps, returns,
                                                              short, short_u, long_u,
                                                              num_assets_l, num_assets_u,
                                                              num_assets_u_scale, f_assets,
                                                              f_timestamps, f_returns,
                                                              loadings, loadings_opt,
                                                              msv_target, lpm_target,
                                                              alpha_i, alpha, a_sim, beta_i,
                                                              beta, b_sim, kappa,
                                                              max_num_assets_kurt,
                                                              rebalance, rebalance_weights,
                                                              turnover, turnover_weights,
                                                              kind_tracking_err,
                                                              tracking_err,
                                                              tracking_err_returns,
                                                              tracking_err_weights,
                                                              bl_bench_weights, a_mtx_ineq,
                                                              b_vec_ineq, risk_budget,
                                                              f_risk_budget, network_method,
                                                              network_sdp, network_penalty,
                                                              network_ip, network_ip_scale,
                                                              a_vec_cent, b_cent, mu_l,
                                                              sd_u, mad_u, ssd_u, cvar_u,
                                                              rcvar_u, evar_u, rvar_u, wr_u,
                                                              rg_u, flpm_u, slpm_u, mdd_u,
                                                              add_u, cdar_u, uci_u, edar_u,
                                                              rdar_u, kurt_u, skurt_u,
                                                              gmd_u, tg_u, rtg_u, owa_u,
                                                              owa_p, owa_w, mu, cov, kurt,
                                                              skurt, L_2, S_2, f_mu, f_cov,
                                                              fm_returns, fm_mu, fm_cov,
                                                              bl_mu, bl_cov, blfm_mu,
                                                              blfm_cov, cov_l, cov_u,
                                                              cov_mu, cov_sigma, d_mu, k_mu,
                                                              k_sigma, optimal, z, limits,
                                                              frontier, solvers, opt_params,
                                                              fail, model, latest_prices,
                                                              alloc_optimal, alloc_solvers,
                                                              alloc_params, alloc_fail,
                                                              alloc_model)
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
    elseif sym == :bt
        obj.beta * size(obj.returns, 1)
    elseif sym == :invbt
        one(typeof(obj.bt)) / (obj.bt)
    elseif sym == :ln_kb
        (obj.invbt^obj.kappa - obj.invbt^(-obj.kappa)) / (2 * obj.kappa)
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
        @smart_assert(zero(val) < val < obj.alpha < one(val))
    elseif sym == :alpha
        @smart_assert(zero(val) < obj.alpha_i < val < one(val))
    elseif sym == :a_sim
        @smart_assert(val > zero(val))
    elseif sym == :beta
        @smart_assert(zero(val) < obj.beta_i < val < one(val))
    elseif sym == :beta_i
        @smart_assert(zero(val) < val < obj.beta < one(val))
    elseif sym == :b_sim
        @smart_assert(val > zero(val))
    elseif sym == :kappa
        @smart_assert(zero(val) < val < one(val))
    elseif sym == :max_num_assets_kurt
        @smart_assert(val >= zero(val))
    elseif sym ∈ (:rebalance, :turnover)
        if isa(val, AbstractVector) && !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2) && all(val .>= zero(val)))
        elseif isa(val, Real)
            @smart_assert(val >= zero(val))
        end
    elseif sym ∈ (:rebalance_weights, :turnover_weights)
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :kind_tracking_err
        @smart_assert(val ∈ TrackingErrKinds)
    elseif sym == :tracking_err
        @smart_assert(val >= zero(val))
        val = convert(typeof(getfield(obj, sym)), val)
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
        @smart_assert(val ∈ NetworkMethods)
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
    elseif sym == :f_mu
        if !isempty(val)
            @smart_assert(length(val) == size(obj.f_returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :f_cov
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.f_returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:L_2, :S_2)
        if !isempty(val)
            N = size(obj.returns, 2)
            @smart_assert(size(val) == (Int(N * (N + 1) / 2), N^2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:risk_budget, :bl_bench_weights)
        if isempty(val)
            N = size(obj.returns, 2)
            val = fill(one(eltype(obj.returns)) / N, N)
        else
            @smart_assert(length(val) == size(obj.returns, 2))
            if sym == :risk_budget
                @smart_assert(all(val .>= zero(eltype(obj.returns))))
            end
            if isa(val, AbstractRange)
                val = collect(val / sum(val))
            else
                val ./= sum(val)
            end
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :f_risk_budget
        if !isempty(val)
            if sym == :risk_budget
                @smart_assert(all(val .>= zero(eltype(obj.returns))))
            end
            if isa(val, AbstractRange)
                val = collect(val / sum(val))
            else
                val ./= sum(val)
            end
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:num_assets_l, :num_assets_u, :num_assets_u_scale)
        @smart_assert(val >= zero(val))
    elseif sym ∈ (:kurt, :skurt, :cov_sigma)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns)
        throw(ArgumentError("$sym is related to other fields and therefore cannot be manually changed without compromising correctness, please create a new instance of Portfolio instead"))
    elseif sym ∈ (:mu, :fm_mu, :bl_mu, :blfm_mu, :d_mu, :latest_prices)
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:cov, :fm_cov, :bl_cov, :blfm_cov, :cov_l, :cov_u, :cov_mu)
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

function Base.deepcopy(obj::Portfolio)
    return Portfolio{typeof(obj.assets), typeof(obj.timestamps), typeof(obj.returns),
                     typeof(obj.short), typeof(obj.short_u), typeof(obj.long_u),
                     typeof(obj.num_assets_l), typeof(obj.num_assets_u),
                     typeof(obj.num_assets_u_scale), typeof(obj.f_assets),
                     typeof(obj.f_timestamps), typeof(obj.f_returns), typeof(obj.loadings),
                     Union{LoadingsOpt, Nothing}, Union{<:Real, AbstractVector{<:Real}},
                     Union{<:Real, AbstractVector{<:Real}}, typeof(obj.alpha_i),
                     typeof(obj.alpha), typeof(obj.a_sim), typeof(obj.beta_i),
                     typeof(obj.beta), typeof(obj.b_sim), typeof(obj.kappa),
                     typeof(obj.max_num_assets_kurt), Union{<:Real, AbstractVector{<:Real}},
                     typeof(obj.rebalance_weights), Union{<:Real, AbstractVector{<:Real}},
                     typeof(obj.turnover_weights), typeof(obj.kind_tracking_err),
                     typeof(obj.tracking_err), typeof(obj.tracking_err_returns),
                     typeof(obj.tracking_err_weights), typeof(obj.bl_bench_weights),
                     typeof(obj.a_mtx_ineq), typeof(obj.b_vec_ineq),
                     typeof(obj.risk_budget), typeof(obj.f_risk_budget),
                     typeof(obj.network_method), typeof(obj.network_sdp),
                     typeof(obj.network_penalty), typeof(obj.network_ip),
                     typeof(obj.network_ip_scale), typeof(obj.a_vec_cent),
                     typeof(obj.b_cent), typeof(obj.mu_l), typeof(obj.sd_u),
                     typeof(obj.mad_u), typeof(obj.ssd_u), typeof(obj.cvar_u),
                     typeof(obj.rcvar_u), typeof(obj.evar_u), typeof(obj.rvar_u),
                     typeof(obj.wr_u), typeof(obj.rg_u), typeof(obj.flpm_u),
                     typeof(obj.slpm_u), typeof(obj.mdd_u), typeof(obj.add_u),
                     typeof(obj.cdar_u), typeof(obj.uci_u), typeof(obj.edar_u),
                     typeof(obj.rdar_u), typeof(obj.kurt_u), typeof(obj.skurt_u),
                     typeof(obj.gmd_u), typeof(obj.tg_u), typeof(obj.rtg_u),
                     typeof(obj.owa_u), typeof(obj.owa_p), typeof(obj.owa_w),
                     typeof(obj.mu), typeof(obj.cov), typeof(obj.kurt), typeof(obj.skurt),
                     typeof(obj.L_2), typeof(obj.S_2), typeof(obj.f_mu), typeof(obj.f_cov),
                     typeof(obj.fm_returns), typeof(obj.fm_mu), typeof(obj.fm_cov),
                     typeof(obj.bl_mu), typeof(obj.bl_cov), typeof(obj.blfm_mu),
                     typeof(obj.blfm_cov), typeof(obj.cov_l), typeof(obj.cov_u),
                     typeof(obj.cov_mu), typeof(obj.cov_sigma), typeof(obj.d_mu),
                     typeof(obj.k_mu), typeof(obj.k_sigma), typeof(obj.optimal),
                     typeof(obj.z), typeof(obj.limits), typeof(obj.frontier),
                     Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                     typeof(obj.fail), typeof(obj.model), typeof(obj.latest_prices),
                     typeof(obj.alloc_optimal), Union{<:AbstractDict, NamedTuple},
                     Union{<:AbstractDict, NamedTuple}, typeof(obj.alloc_fail),
                     typeof(obj.alloc_model)}(deepcopy(obj.assets),
                                              deepcopy(obj.timestamps),
                                              deepcopy(obj.returns), deepcopy(obj.short),
                                              deepcopy(obj.short_u), deepcopy(obj.long_u),
                                              deepcopy(obj.num_assets_l),
                                              deepcopy(obj.num_assets_u),
                                              deepcopy(obj.num_assets_u_scale),
                                              deepcopy(obj.f_assets),
                                              deepcopy(obj.f_timestamps),
                                              deepcopy(obj.f_returns),
                                              deepcopy(obj.loadings),
                                              deepcopy(obj.loadings_opt),
                                              deepcopy(obj.msv_target),
                                              deepcopy(obj.lpm_target),
                                              deepcopy(obj.alpha_i), deepcopy(obj.alpha),
                                              deepcopy(obj.a_sim), deepcopy(obj.beta_i),
                                              deepcopy(obj.beta), deepcopy(obj.b_sim),
                                              deepcopy(obj.kappa),
                                              deepcopy(obj.max_num_assets_kurt),
                                              deepcopy(obj.rebalance),
                                              deepcopy(obj.rebalance_weights),
                                              deepcopy(obj.turnover),
                                              deepcopy(obj.turnover_weights),
                                              deepcopy(obj.kind_tracking_err),
                                              deepcopy(obj.tracking_err),
                                              deepcopy(obj.tracking_err_returns),
                                              deepcopy(obj.tracking_err_weights),
                                              deepcopy(obj.bl_bench_weights),
                                              deepcopy(obj.a_mtx_ineq),
                                              deepcopy(obj.b_vec_ineq),
                                              deepcopy(obj.risk_budget),
                                              deepcopy(obj.f_risk_budget),
                                              deepcopy(obj.network_method),
                                              deepcopy(obj.network_sdp),
                                              deepcopy(obj.network_penalty),
                                              deepcopy(obj.network_ip),
                                              deepcopy(obj.network_ip_scale),
                                              deepcopy(obj.a_vec_cent),
                                              deepcopy(obj.b_cent), deepcopy(obj.mu_l),
                                              deepcopy(obj.sd_u), deepcopy(obj.mad_u),
                                              deepcopy(obj.ssd_u), deepcopy(obj.cvar_u),
                                              deepcopy(obj.rcvar_u), deepcopy(obj.evar_u),
                                              deepcopy(obj.rvar_u), deepcopy(obj.wr_u),
                                              deepcopy(obj.rg_u), deepcopy(obj.flpm_u),
                                              deepcopy(obj.slpm_u), deepcopy(obj.mdd_u),
                                              deepcopy(obj.add_u), deepcopy(obj.cdar_u),
                                              deepcopy(obj.uci_u), deepcopy(obj.edar_u),
                                              deepcopy(obj.rdar_u), deepcopy(obj.kurt_u),
                                              deepcopy(obj.skurt_u), deepcopy(obj.gmd_u),
                                              deepcopy(obj.tg_u), deepcopy(obj.rtg_u),
                                              deepcopy(obj.owa_u), deepcopy(obj.owa_p),
                                              deepcopy(obj.owa_w), deepcopy(obj.mu),
                                              deepcopy(obj.cov), deepcopy(obj.kurt),
                                              deepcopy(obj.skurt), deepcopy(obj.L_2),
                                              deepcopy(obj.S_2), deepcopy(obj.f_mu),
                                              deepcopy(obj.f_cov), deepcopy(obj.fm_returns),
                                              deepcopy(obj.fm_mu), deepcopy(obj.fm_cov),
                                              deepcopy(obj.bl_mu), deepcopy(obj.bl_cov),
                                              deepcopy(obj.blfm_mu), deepcopy(obj.blfm_cov),
                                              deepcopy(obj.cov_l), deepcopy(obj.cov_u),
                                              deepcopy(obj.cov_mu), deepcopy(obj.cov_sigma),
                                              deepcopy(obj.d_mu), deepcopy(obj.k_mu),
                                              deepcopy(obj.k_sigma), deepcopy(obj.optimal),
                                              deepcopy(obj.z), deepcopy(obj.limits),
                                              deepcopy(obj.frontier), deepcopy(obj.solvers),
                                              deepcopy(obj.opt_params), deepcopy(obj.fail),
                                              copy(obj.model), deepcopy(obj.latest_prices),
                                              deepcopy(obj.alloc_optimal),
                                              deepcopy(obj.alloc_solvers),
                                              deepcopy(obj.alloc_params),
                                              deepcopy(obj.alloc_fail),
                                              copy(obj.alloc_model))
end

"""
```julia
mutable struct HCPortfolio{
    ast,    dat,    r,    # Risk parmeters
    ai,    a,    as,    bi,    b,    bs,    k,    ata,    mnak,    # OWA parameters
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
    # OWA parameters
    owa_p::owap
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
# OWA parameters
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
mutable struct HCPortfolio{ast, dat, r, ai, a, as, bi, b, bs, k, ata, mnak, owap, wowa, tmu,
                           tcov, tkurt, tskurt, tl2, ts2, tbin, wmi, wma, ttco, tco, tdist,
                           tcl, tk, topt, tsolv, toptpar, tf, tlp, taopt, tasolv, taoptpar,
                           taf, tamod} <: AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    alpha_i::ai
    alpha::a
    a_sim::as
    beta_i::bi
    beta::b
    b_sim::bs
    kappa::k
    alpha_tail::ata
    max_num_assets_kurt::mnak
    owa_p::owap
    owa_w::wowa
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
    optimal::topt
    solvers::tsolv
    opt_params::toptpar
    fail::tf
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
    alpha_i::Real = 0.0001,    alpha::Real = 0.05,    a_sim::Integer = 100,    beta_i::Real = alpha_i,    beta::Real = alpha,    b_sim::Integer = a_sim,    kappa::Real = 0.3,    alpha_tail::Real = 0.05,    max_num_assets_kurt::Integer = 0,    # OWA parameters
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    # Optimisation parameters
    mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    bins_info::Union{Symbol, <:Integer} = :KN,    w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,    w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,    cor_method::Symbol = :Pearson,    cor::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),    clusters::Clustering.Hclust = Hclust{Float64}(
        Matrix{Int64}(undef, 0, 2),        Float64[],        Int64[],        :nothing,    ),    k::Integer = 0,    # Optimal portfolios
    optimal::AbstractDict = Dict(),    # Solutions
    solvers::Union{<:AbstractDict, NamedTuple} = Dict(),    opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),    fail::AbstractDict = Dict(),    # Allocation
    latest_prices::AbstractVector = Vector{Float64}(undef, 0),    alloc_optimal::AbstractDict = Dict(),    alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),    alloc_params::Union{<:AbstractDict, NamedTuple} = Dict(),    alloc_fail::AbstractDict = Dict(),    alloc_model::JuMP.Model = JuMP.Model())
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
## OWA parameters
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
function HCPortfolio(; prices::TimeArray = TimeArray(TimeType[], []),
                     returns::DataFrame = DataFrame(),
                     ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
                     assets::AbstractVector = Vector{String}(undef, 0),
                     alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100,
                     beta_i::Real = alpha_i, beta::Real = alpha, b_sim::Integer = a_sim,
                     kappa::Real = 0.3, alpha_tail::Real = 0.05,
                     max_num_assets_kurt::Integer = 0,
                     owa_p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50],
                     owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                     mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                     cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     bins_info::Union{Symbol, <:Integer} = :KN,
                     w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,
                     w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,
                     cor_method::Symbol = :Pearson,
                     cor::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     clusters::Clustering.Hclust = Hclust{Float64}(Matrix{Int64}(undef, 0,
                                                                                 2),
                                                                   Float64[], Int64[],
                                                                   :nothing),
                     k::Integer = 0, optimal::AbstractDict = Dict(),
                     solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
                     opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),
                     fail::AbstractDict = Dict(),
                     latest_prices::AbstractVector = Vector{Float64}(undef, 0),
                     alloc_optimal::AbstractDict = Dict(),
                     alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
                     alloc_params::Union{<:AbstractDict, NamedTuple} = Dict(),
                     alloc_fail::AbstractDict = Dict(),
                     alloc_model::JuMP.Model = JuMP.Model())
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

    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    @smart_assert(zero(beta) < beta_i < beta < one(beta))
    @smart_assert(b_sim > zero(b_sim))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    @smart_assert(zero(alpha_tail) < alpha_tail < one(alpha_tail))
    @smart_assert(max_num_assets_kurt >= zero(max_num_assets_kurt))
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
    @smart_assert(bins_info ∈ BinMethods ||
                  (isa(bins_info, Int) && bins_info > zero(bins_info)))
    if isa(w_min, Real)
        @smart_assert(all(w_min .<= w_max))
    elseif isa(w_min, AbstractVector)
        if !isempty(w_min)
            @smart_assert(length(w_min) == size(returns, 2) && all(w_min .<= w_max))
        end
    end
    if isa(w_max, Real)
        @smart_assert(all(w_min .<= w_max))
    elseif isa(w_max, AbstractVector)
        if !isempty(w_max)
            @smart_assert(length(w_max) == size(returns, 2) && all(w_min .<= w_max))
        end
    end
    @smart_assert(cor_method ∈ CorMethods)
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

    return HCPortfolio{typeof(assets), typeof(timestamps), typeof(returns), typeof(alpha_i),
                       typeof(alpha), typeof(a_sim), typeof(beta_i), typeof(beta),
                       typeof(b_sim), typeof(kappa), typeof(alpha_tail),
                       typeof(max_num_assets_kurt), typeof(owa_p), typeof(owa_w),
                       typeof(mu), typeof(cov), typeof(kurt), typeof(skurt), typeof(L_2),
                       typeof(S_2), Union{Symbol, <:Integer},
                       Union{<:Real, AbstractVector{<:Real}},
                       Union{<:Real, AbstractVector{<:Real}}, typeof(cor_method),
                       typeof(cor), typeof(dist), typeof(clusters), typeof(k),
                       typeof(optimal), Union{<:AbstractDict, NamedTuple},
                       Union{<:AbstractDict, NamedTuple}, typeof(fail),
                       typeof(latest_prices), typeof(alloc_optimal),
                       Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                       typeof(alloc_fail), typeof(alloc_model)}(assets, timestamps, returns,
                                                                alpha_i, alpha, a_sim,
                                                                beta_i, beta, b_sim, kappa,
                                                                alpha_tail,
                                                                max_num_assets_kurt, owa_p,
                                                                owa_w, mu, cov, kurt, skurt,
                                                                L_2, S_2, bins_info, w_min,
                                                                w_max, cor_method, cor,
                                                                dist, clusters, k, optimal,
                                                                solvers, opt_params, fail,
                                                                latest_prices,
                                                                alloc_optimal,
                                                                alloc_solvers, alloc_params,
                                                                alloc_fail, alloc_model)
end

function Base.setproperty!(obj::HCPortfolio, sym::Symbol, val)
    if sym == :alpha_i
        @smart_assert(zero(val) < val < obj.alpha < one(val))
    elseif sym == :alpha
        @smart_assert(zero(val) < obj.alpha_i < val < one(val))
    elseif sym == :a_sim
        @smart_assert(val > zero(val))
    elseif sym == :beta
        @smart_assert(zero(val) < obj.beta_i < val < one(val))
    elseif sym == :beta_i
        @smart_assert(zero(val) < val < obj.beta < one(val))
    elseif sym == :b_sim
        @smart_assert(val > zero(val))
    elseif sym == :kappa
        @smart_assert(zero(val) < val < one(val))
    elseif sym == :alpha_tail
        @smart_assert(zero(val) < val < one(val))
    elseif sym == :max_num_assets_kurt
        @smart_assert(val >= zero(val))
    elseif sym == :owa_w
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 1))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :bins_info
        @smart_assert(val ∈ BinMethods || isa(val, Int) && val > zero(val))
    elseif sym == :cor_method
        @smart_assert(val ∈ CorMethods)
    elseif sym == :k
        @smart_assert(val >= zero(val))
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

        lmin = length(vmin)
        lmax = length(vmax)

        if isa(val, Real)
            @smart_assert(all(vmin .<= vmax))
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(obj.returns, 2))

                if !isempty(vmin) && !isempty(vmax) && lmin == lmax
                    @smart_assert(all(vmin .<= vmax))
                end

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
    elseif sym ∈ (:mu, :latest_prices)
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:kurt, :skurt)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:L_2, :S_2)
        if !isempty(val)
            N = size(obj.returns, 2)
            @smart_assert(size(val) == (Int(N * (N + 1) / 2), N^2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:cov, :cor, :dist)
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

function Base.deepcopy(obj::HCPortfolio)
    return HCPortfolio{typeof(obj.assets), typeof(obj.timestamps), typeof(obj.returns),
                       typeof(obj.alpha_i), typeof(obj.alpha), typeof(obj.a_sim),
                       typeof(obj.beta_i), typeof(obj.beta), typeof(obj.b_sim),
                       typeof(obj.kappa), typeof(obj.alpha_tail),
                       typeof(obj.max_num_assets_kurt), typeof(obj.owa_p),
                       typeof(obj.owa_w), typeof(obj.mu), typeof(obj.cov), typeof(obj.kurt),
                       typeof(obj.skurt), typeof(obj.L_2), typeof(obj.S_2),
                       Union{Symbol, <:Integer}, Union{<:Real, AbstractVector{<:Real}},
                       Union{<:Real, AbstractVector{<:Real}}, typeof(obj.cor_method),
                       typeof(obj.cor), typeof(obj.dist), typeof(obj.clusters),
                       typeof(obj.k), typeof(obj.optimal),
                       Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                       typeof(obj.fail), typeof(obj.latest_prices),
                       typeof(obj.alloc_optimal), Union{<:AbstractDict, NamedTuple},
                       Union{<:AbstractDict, NamedTuple}, typeof(obj.alloc_fail),
                       typeof(obj.alloc_model)}(deepcopy(obj.assets),
                                                deepcopy(obj.timestamps),
                                                deepcopy(obj.returns),
                                                deepcopy(obj.alpha_i), deepcopy(obj.alpha),
                                                deepcopy(obj.a_sim), deepcopy(obj.beta_i),
                                                deepcopy(obj.beta), deepcopy(obj.b_sim),
                                                deepcopy(obj.kappa),
                                                deepcopy(obj.alpha_tail),
                                                deepcopy(obj.max_num_assets_kurt),
                                                deepcopy(obj.owa_p), deepcopy(obj.owa_w),
                                                deepcopy(obj.mu), deepcopy(obj.cov),
                                                deepcopy(obj.kurt), deepcopy(obj.skurt),
                                                deepcopy(obj.L_2), deepcopy(obj.S_2),
                                                deepcopy(obj.bins_info),
                                                deepcopy(obj.w_min), deepcopy(obj.w_max),
                                                deepcopy(obj.cor_method), deepcopy(obj.cor),
                                                deepcopy(obj.dist), deepcopy(obj.clusters),
                                                deepcopy(obj.k), deepcopy(obj.optimal),
                                                deepcopy(obj.solvers),
                                                deepcopy(obj.opt_params),
                                                deepcopy(obj.fail),
                                                deepcopy(obj.latest_prices),
                                                deepcopy(obj.alloc_optimal),
                                                deepcopy(obj.alloc_solvers),
                                                deepcopy(obj.alloc_params),
                                                deepcopy(obj.alloc_fail),
                                                copy(obj.alloc_model))
end

export Portfolio, HCPortfolio
