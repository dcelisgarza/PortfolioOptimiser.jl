"""
```
AbstractPortfolio
```

Abstract type for subtyping portfolios.
"""
abstract type AbstractPortfolio end

"""
```
mutable struct Portfolio{ast, dat, r, s, us, ul, nal, nau, naus, tfa, tfdat, tretf, l, lo,
                         msvt, lpmt, ai, a, as, bi, b, bs, k, mnak, mnaks, skewf, sskewf,
                         rb, rbw, to, tobw, kte, te, rbi, bw, blbw, ami, bvi, rbv, frbv, nm,
                         nsdp, np, ni, nis, amc, bvc, ler, ud, umad, usd, ucvar, urcvar,
                         uevar, urvar, uwr, ur, uflpm, uslpm, umd, uad, ucdar, uuci, uedar,
                         urdar, uk, usk, ugmd, utg, urtg, uowa, udvar, uskew, usskew, owap,
                         wowa, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew, tv, tsskew, tsv, tmuf,
                         tcovf, trfm, tmufm, tcovfm, tmubl, tcovbl, tmublf, tcovblf, tcovl,
                         tcovu, tcovmu, tcovs, tdmu, tkmu, tks, topt, tz, tlim, tfront,
                         tsolv, tf, toptpar, tmod, tlp, taopt, tasolv, taoptpar, taf,
                         tamod} <: AbstractPortfolio
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
    max_num_assets_kurt_scale::mnaks
    skew_factor::skewf
    sskew_factor::sskewf
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
    dvar_u::udvar
    skew_u::uskew
    sskew_u::usskew
    owa_p::owap
    owa_w::wowa
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    skew::tskew
    V::tv
    sskew::tsskew
    SV::tsv
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

Structure for working with regular portfolios.

Some of these require external information from the arguments of functions that use the an instance of [`Portfolio`](@ref):

  - `type`: one of [`PortTypes`](@ref).
  - `rm`: one of [`RiskMeasures`](@ref).
  - `class`: one of [`PortClasses`](@ref).
  - `hist`: one of [`ClassHist`](@ref).
  - `rf`: risk-free rate.
  - `owa_approx`: flag for using the approximate OWA formulation.

Constraints are set if and only if all their variables are appropriately defined.

In order for a variable to be considered "appropriately defined", it must meet the following criteria:

  - `:Real`: must be finite.
  - `:Integer`: must be non-zero.
  - `:AbstractArray`: must be non-empty and of the proper shape.

Some constraints define decision variables using scaling factors. The scaling factor should be large enough to be outside of the problem's scale, but not too large as to incur numerical instabilities. Solution quality may be improved by changing the scaling factor.

# Inputs

## Portfolio characteristics

  - `assets`: `Na×1` vector of assets, where `Na` is the number of assets.

  - `timestamps`: `T×1` vector of timestamps, where `T` is the number of returns observations.
  - `returns`: `T×Na` matrix of asset returns, where `T` is the number of returns observations and `Na` is the number of assets.
  - `short`: whether or not to allow negative weights, i.e. whether a portfolio accepts shorting.
  - `short_u`: absolute value of the sum of all short (negative) weights.
  - `long_u`: sum of all long (positive) weights.
  - `num_assets_l`: lower bound for the integer number of assets that make significant contributions to the final portfolio weights. Solver must support `MOI.SecondOrderCone`.
  - `num_assets_u`: upper bound for the integer number of assets that make significant contributions to the final portfolio weights [MIP1](@cite). Solver must support MIP constraints.
  - `num_assets_u_scale`: scaling factor needed to create the decision variable for the `num_assets_u` constraint.
  - `f_assets`: `Nf×1` vector of factors, where `Nf` is the number of factors.
  - `f_timestamps`: `T×1` vector of factor timestamps, where `T` is the number of returns observations.
  - `f_returns`: `T×Nf` matrix of factor returns, where `T` is the number of returns observations and `Nf` is the number of factors.
  - `loadings`: loadings matrix in dataframe form. Calling [`factor_statistics!`](@ref) will generate and set the dataframe. The number of rows must be equal to the number of asset and factor returns observations, `T`. Must have a few different columns.

      + `tickers`: (optional) contains the list of tickers.
      + `const`: (optional) contains the regression constant.
      + The other columns must be the names of the factors.
  - `loadings_opt`:

      + `class == :FC`: instance of [`LoadingsOpt`](@ref) used to recover the factor risk budget vector.

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

      + `rm ∈ (:TG, :TGRG)`: initial significance level of losses, `0 < alpha_i < alpha < 1`.
  - `a_sim`:

      + `rm ∈ (:TG, :TGRG)`: number of CVaRs to approximate the losses, `a_sim > 0`.
  - `alpha`:

      + `rm ∈ (:VaR, :CVaR, :EVaR, :RLVaR, :CVaRRG, :TG, :TGRG, :DaR, :CDaR, :EDaR, :RLDaR, :DaR_r, :CDaR_r, :EDaR_r, :RLDaR_r)`: significance level of losses, `alpha ∈ (0, 1)`.
  - `beta_i`:

      + `rm == :TGRG`: initial significance level of gains, `0 < beta_i < beta < 1`.
  - `b_sim`:

      + `rm == :TGRG`: number of CVaRs to approximate the gains, `b_sim > 0`.
  - `beta`:

      + `rm ∈ (:CVaRRG, :TGRG)`: significance level of gains, `beta ∈ (0, 1)`.
  - `kappa`:

      + `rm ∈ (:RLVaR, :RLDaR, :RLDaR_r)`: relativistic deformation parameter.
  - `max_num_assets_kurt`:

      + `iszero(max_num_assets_kurt)`: use the full kurtosis model.
      + `!iszero(max_num_assets_kurt)`: if the number of assets surpases this value, use the relaxed kurtosis model.
  - `max_num_assets_kurt_scale`: the relaxed kurtosis model uses the largest `max_num_assets_kurt_scale * max_num_assets_kurt` eigenvalues to approximate the kurtosis matrix, `max_num_assets_kurt_scale ∈ [1, Na]`, where `Na` is the number of assets.
  - `skew_factor`: factor for adding the multiple of the negative quadratic skewness to the risk function.
  - `sskew_factor`: factor for adding the multiple of the negative quadratic semi skewness to the risk function.

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

Where ``r_i`` is the rebalancing coefficient, ``w_i`` is the optimal weight for the `i'th` asset, ``\\hat{w}_i`` target weight for the `i'th` asset, and ``N`` is the number of assets.

  - `turnover`: the turnover constraint is somewhat of an inverse of the `rebalance` penalty.

      + `isa(turnover, Real)`: all assets have the same turnover value.
      + `isa(turnover, AbstractVector)`: each asset has its own turnover value.

  - `turnover_weights`: define the target weights for the turnover constraint.

The turnover constraint is defined as

```math
\\lvert w_{i} - \\hat{w}_{i}\\rvert \\leq t_{i} \\, \\forall\\, i \\in N\\,.
```

Where ``w_i`` is the optimal weight for the `i'th` asset, ``\\hat{w}_i`` target weight for the `i'th` asset, ``t_{i}`` is the value of the turnover for the `i'th` asset, and ``N`` is the number of assets.

  - `kind_tracking_err`: kind of tracking error from [`TrackingErrKinds`](@ref).
  - `tracking_err`: define the value of the tracking error.
  - `tracking_err_returns`: `T×1` vector of benchmark returns for the tracking error constraint as per [`TrackingErrKinds`](@ref), where `T` is the number of returns observations.
  - `tracking_err_weights`: `Na×1` vector of weights used for computing the benchmark vector for the tracking error constraint as per [`TrackingErrKinds`](@ref), where `Na` is the number of assets.

The tracking error constraint is defined as

```math
\\sqrt{\\dfrac{1}{T-1}\\sum\\limits_{i=1}^{T}\\left(\\mathbf{X}_{i} \\bm{w} - b_{i}\\right)^{2}}\\leq t\\,.
```

Where ``\\mathbf{X}_{i}`` is the `i'th` observation (row) of the returns matrix ``\\mathbf{X}``, ``\\bm{w}`` is the vector of optimal asset weights, ``b_{i}`` is the `i'th` observation of the benchmark returns vector, ``t`` the tracking error, and ``T`` is the number of returns observations.

  - `bl_bench_weights`: `Na×1` vector of benchmark weights for Black-Litterman models, where `Na` is the number of assets.

## Asset constraints

The constraint is only defined when both `a_mtx_ineq` and `b_vec_ineq` are defined.

  - `a_mtx_ineq`: `C×Na` A matrix of the linear asset constraints, where `C` is the number of constraints, and `Na` is the number of assets.
  - `b_vec_ineq`: `C×1` B vector of the linear asset constraints, where `C` is the number of constraints.

The linear asset constraint is defined as

```math
\\mathbf{A} \\bm{w} \\geq \\bm{B}\\,.
```

Where ``\\mathbf{A}`` is the matrix of linear constraints `a_mtx_ineq`, ``\\bm{w}`` the asset weights, and ``\\bm{B}`` is the vector of linear asset constraints `b_vec_ineq`.

## Risk budget constraints

Only relevant when `type ∈ (:RP, :RRP)`. [`PortClasses`](@ref) and [`ClassHist`](@ref) define which one to use when calling [`optimise!`](@ref).

  - `risk_budget`: `Na×1` asset risk budget constraint vector for risk measure `rm`, where `Na` is the number of assets. Asset `i` contributes the amount of `rm` risk in position `risk_budget[i]`.
  - `f_risk_budget`: `Nf×1` factor risk budget constraint vector for risk measure `rm`, where `Nf` is the number of factors. Factor `i` contributes the amount of `rm` risk in position `f_risk_budget[i]`.

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
  - `a_vec_cent`: `Na×1` centrality measure vector for the centrality constraint, where `Na` is the number of assets.
  - `b_cent`: average centrality measure for the centrality constraint.

The centrality measure constraint [NWK1](@cite) is defined as

```math
\\bm{A} \\cdot \\bm{w} = b\\,.
```

Where ``\\bm{A}`` is the centrality measure vector `a_vec_cent`, ``\\bm{w}`` the asset weights, and ``b`` average centrality measure `b_cent`. The constraint is only applied when both `a_vec_cent` and `b_cent` are defined.

## Bounds constraints

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
  - `owa_u`:

      + `owa_w` must be properly defined: ordered weight risk.

## OWA parameters

Only relevant when `rm ∈ (:GMD, :TG, :TGRG, :OWA)`.

  - `owa_p`:

      + `owa_approx = true`: `C×1` vector containing the order of the p-norms used in the approximate formulation of the risk measures, where `C` is the number of p-norms. The more entries and larger their range, the more accurate the approximation.

  - `owa_w`:

      + `rm == :OWA`: `T×1` OWA vector, where `T` is the number of returns observations. Useful for optimising higher L-moments.

## Model statistics

### Asset statistics

These are the default statistics. See [`PortClasses`](@ref) and [`ClassHist`](@ref) for details.

  - `mu`: `Na×1` asset expected returns vector, where `Na` is the number of assets.
  - `cov`: `Na×Na` asset covariance matrix, where `Na` is the number of assets.
  - `kurt`: `(Na^2)×(Na^2)` asset cokurtosis matrix, where `Na` is the number of assets.
  - `skurt`: `(Na^2)×(Na^2)` asset semi cokurtosis matrix, where `Na` is the number of assets.
  - `L_2`: `(Na^2)×((Na^2 + Na)/2)` elimination matrix, where `Na` is the number of assets.
  - `S_2`: `((Na^2 + Na)/2)×(Na^2)` summation matrix, where `Na` is the number of assets.

### Adjusted asset statistics

Only relevant for certain combinations of [`PortClasses`](@ref) and [`ClassHist`](@ref).

  - `f_mu`: `Nf×1` factor expected returns vector, where `Nf` is the number of factors.
  - `f_cov`: `Nf×Nf` factor covariance matrix, where `Nf` is the number of factors.
  - `fm_returns`: `T×Na` matrix of factor adjusted asset returns, where `T` is the number of returns observations and `Na` is the number of assets.
  - `fm_mu`: `Na×1` factor adjusted asset expected returns vector, where `Na` is the number of assets.
  - `fm_cov`: `Na×Na` factor adjusted asset covariance matrix, where `Na` is the number of assets.
  - `bl_mu`: `Na×1` Black-Litterman adjusted asset expected returns vector, where `Na` is the number of assets.
  - `bl_cov`: `Na×Na` Black-Litterman adjusted asset covariance matrix, where `Na` is the number of assets.
  - `blfm_mu`: `Na×1` Black-Litterman factor model adjusted asset expected returns vector, where `Na` is the number of assets.
  - `blfm_cov`: `Na×Na` Black-Litterman factor model adjusted asset covariance matrix, where `Na` is the number of assets.

### Worst case statistics

Only relevant when `type == :WC`.

  - `cov_l`: (box set) `Na×Na` worst case lower bound for the asset covariance matrix, where `Na` is the number of assets.
  - `cov_u`: (box set) `Na×Na` worst case upper bound for the asset covariance matrix, where `Na` is the number of assets.
  - `cov_mu`: (elliptical set) `Na×Na` matrix of the estimation errors of the asset expected returns vector set, where `Na` is the number of assets.
  - `cov_sigma`: (elliptical set) `Na×Na` matrix of the estimation errors of the asset covariance matrix set, where `Na` is the number of assets.
  - `d_mu`: (box set) `Na×1` absolute deviation of the worst case upper and lower asset expected returns vectors, where `Na` is the number of assets.
  - `k_mu`: (elliptical set) distance parameter of the uncertainty in the asset expected returns vector for the worst case optimisation.
  - `k_sigma`: (elliptical set) distance parameter of the uncertainty in the asset covariance matrix for the worst case optimisation.

## Optimal portfolios

  - `optimal`: collection capable of storing key value pairs for storing optimal portfolios.
  - `z`: collection capable of storing key value pairs for storing optimal `z` values of portfolios optimised for entropy and relativistic risk measures.
  - `limits`: collection capable of storing key value pairs for storing the minimal and maximal risk portfolios for a given risk measure.
  - `frontier`: collection capable of storing key value pairs for containing points in the efficient frontier for a given risk measure.

## Solutions

  - `solvers`: provides the solvers and corresponding parameters for solving the portfolio optimisation problem. There can be two `key => value` pairs.

      + `:solver => value`: `value` is a `JuMP` optimiser. The optimiser can be declared alongside its attributes by using `JuMP.solver_with_attributes`.
      + `:params => value`: (optional) `value` must be a `Dict` or `NamedTuple` whose `key => value` pairs are the solver-specific settings.

  - `opt_params`: collection capable of storing key value pairs for storing parameters used for optimising.
  - `fail`: collection capable of storing key value pairs for storing failed optimisation attempts.
  - `model`: `JuMP.Model()` for optimising a portfolio.

## Allocation

  - `latest_prices`: `Na×1` vector of asset prices, `Na` is the number of assets.

  - `alloc_optimal`: collection capable of storing key value pairs for storing optimal portfolios after allocating discrete stocks.
  - `alloc_solvers`: provides the solvers and corresponding parameters for solving the discrete allocation optimisation problem. There can be two `key => value` pairs.

      + `:solver => value`: `value` is a `JuMP` optimiser. The optimiser can be declared alongside its attributes by using `JuMP.solver_with_attributes`. Solver must support MIP constraints.
      + `:params => value`: (optional) `value` must be a `Dict` or `NamedTuple` whose `key => value` pairs are the solver-specific settings.
  - `alloc_params`: collection capable of storing key value pairs for storing parameters used for optimising the portfolio allocation.
  - `alloc_fail`: collection capable of storing key value pairs for storing failed optimisation attempts.
  - `alloc_model`: `JuMP.Model()` for optimising a portfolio allocation.
"""
mutable struct Portfolio{ast, dat, r, s, us, ul, nal, nau, naus, tfa, tfdat, tretf, l, lo,
                         msvt, lpmt, ai, a, as, bi, b, bs, k, mnak, mnaks, skewf, sskewf,
                         rb, rbw, to, tobw, kte, te, rbi, bw, blbw, ami, bvi, rbv, frbv, nm,
                         nsdp, np, ni, nis, amc, bvc, ler, ud, umad, usd, ucvar, urcvar,
                         uevar, urvar, uwr, ur, uflpm, uslpm, umd, uad, ucdar, uuci, uedar,
                         urdar, uk, usk, ugmd, utg, urtg, uowa, udvar, uskew, usskew, owap,
                         wowa, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew, tv, tsskew, tsv,
                         tmuf, tcovf, trfm, tmufm, tcovfm, tmubl, tcovbl, tmublf, tcovblf,
                         tcovl, tcovu, tcovmu, tcovs, tdmu, tkmu, tks, topt, tz, tlim,
                         tfront, tsolv, tf, toptpar, tmod, tlp, taopt, talo, tasolv,
                         taoptpar, taf, tamod} <: AbstractPortfolio
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
    max_num_assets_kurt_scale::mnaks
    skew_factor::skewf
    sskew_factor::sskewf
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
    dvar_u::udvar
    skew_u::uskew
    sskew_u::usskew
    owa_p::owap
    owa_w::wowa
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    skew::tskew
    V::tv
    sskew::tsskew
    SV::tsv
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
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_params::taoptpar
    alloc_fail::taf
    alloc_model::tamod
end

"""
```
Portfolio(; prices::TimeArray                                 = TimeArray(TimeType[], []),
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
          max_num_assets_kurt_scale::Integer                = 2,
          skew_factor::Real                                 = Inf,
          sskew_factor::Real                                = Inf,
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
          dvar_u::Real                                      = Inf,
          skew_u::Real                                      = Inf,
          sskew_u::Real                                     = Inf,
          owa_p::AbstractVector{<:Real}                     = Float64[2, 3, 4, 10, 50],
          owa_w::AbstractVector{<:Real}                     = Vector{Float64}(undef, 0),
          mu::AbstractVector                                = Vector{Float64}(undef, 0),
          cov::AbstractMatrix{<:Real}                       = Matrix{Float64}(undef, 0, 0),
          kurt::AbstractMatrix{<:Real}                      = Matrix{Float64}(undef, 0, 0),
          skurt::AbstractMatrix{<:Real}                     = Matrix{Float64}(undef, 0, 0),
          skew::AbstractMatrix{<:Real}                      = Matrix{Float64}(undef, 0, 0),
          V::AbstractMatrix{<:Real}                         = Matrix{Float64}(undef, 0, 0),
          sskew::AbstractMatrix{<:Real}                     = Matrix{Float64}(undef, 0, 0),
          SV::AbstractMatrix{<:Real}                        = Matrix{Float64}(undef, 0, 0),
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
```

Performs data validation and creates an instance of [`Portfolio`](@ref). Union datatypes remain union datatypes in the instance.

# Inputs

  - `prices`:

      + `!isempty(prices)`: timearray of asset prices, automatically sets `assets`, `timestamps`, computes `returns` and `latest_prices`. Takes priority over the `returns`, `ret`, `timestamps`, `assets` and `latest_prices` arguments.

  - `returns`:

      + `!isempty(returns)`: dataframe of asset returns, automatically sets `assets`, `timestamps`, `returns`. Takes priority over the, `ret`, `timestamps` and `assets` arguments.
  - `ret`: sets `returns`.
  - `timestamps`: sets `timestamps`.
  - `assets`: sets `assets`.
  - `short`: sets `short`.
  - `short_u`: sets `short_u`.
  - `long_u`: sets `long_u`.
  - `num_assets_l`: sets `num_assets_l`.
  - `num_assets_u`: sets `num_assets_u`.
  - `num_assets_u_scale`: sets `num_assets_u_scale`.
  - `f_prices`:

      + `!isempty(f_prices)`: timearray of factor prices automatically sets `f_assets`, `f_timestamps` and `f_returns`. Takes priority over the `f_returns`, `f_ret`, `f_timestamps` and `f_assets` arguments.
  - `f_returns`:

      + `!isempty(f_returns)`: dataframe of factor returns automatically sets `f_assets`, `f_timestamps`, `f_returns`. Takes priority over the, `f_ret`, `f_timestamps` and `f_assets` arguments.
  - `f_ret`: sets `f_returns`.
  - `f_timestamps`: sets `f_timestamps`.
  - `f_assets`: sets `f_assets`.
  - `loadings`: sets `loadings`.
  - `loadings_opt`: sets `loadings_opt`.
  - `msv_target`: sets `msv_target`.
  - `lpm_target`: sets `lpm_target`.
  - `alpha_i`: sets `alpha_i`.
  - `alpha`: sets `alpha`.
  - `a_sim`: sets `a_sim`.
  - `beta_i`: sets `beta_i`.
  - `beta`: sets `beta`.
  - `b_sim`: sets `b_sim`.
  - `kappa`: sets `kappa`.
  - `max_num_assets_kurt`: sets `max_num_assets_kurt`.
  - `max_num_assets_kurt_scale`: sets `max_num_assets_kurt_scale`.
  - `skew_factor`: sets `skew_factor`.
  - `sskew_factor` sets `sskew_factor`.
  - `rebalance`: sets `rebalance`.
  - `rebalance_weights`: sets `rebalance_weights`.
  - `turnover`: sets `turnover`.
  - `turnover_weights`: sets `turnover_weights`.
  - `kind_tracking_err`: sets `kind_tracking_err`.
  - `tracking_err`: sets `tracking_err`.
  - `tracking_err_returns`: sets `tracking_err_returns`.
  - `tracking_err_weights`: sets `tracking_err_weights`.
  - `bl_bench_weights`: sets `bl_bench_weights`.
  - `a_mtx_ineq`: sets `a_mtx_ineq`.
  - `b_vec_ineq`: sets `b_vec_ineq`.
  - `risk_budget`: sets `risk_budget`.
  - `f_risk_budget`: sets `f_risk_budget`.
  - `network_method`: sets `network_method`.
  - `network_sdp`: sets `network_sdp`.
  - `network_penalty`: sets `network_penalty`.
  - `network_ip`: sets `network_ip`.
  - `network_ip_scale`: sets `network_ip_scale`.
  - `a_vec_cent`: sets `a_vec_cent`.
  - `b_cent`: sets `b_cent`.
  - `mu_l`: sets `mu_l`.
  - `sd_u`: sets `sd_u`.
  - `mad_u`: sets `mad_u`.
  - `ssd_u`: sets `ssd_u`.
  - `cvar_u`: sets `cvar_u`.
  - `rcvar_u`: sets `rcvar_u`.
  - `evar_u`: sets `evar_u`.
  - `rvar_u`: sets `rvar_u`.
  - `wr_u`: sets `wr_u`.
  - `rg_u`: sets `rg_u`.
  - `flpm_u`: sets `flpm_u`.
  - `slpm_u`: sets `slpm_u`.
  - `mdd_u`: sets `mdd_u`.
  - `add_u`: sets `add_u`.
  - `cdar_u`: sets `cdar_u`.
  - `uci_u`: sets `uci_u`.
  - `edar_u`: sets `edar_u`.
  - `rdar_u`: sets `rdar_u`.
  - `kurt_u`: sets `kurt_u`.
  - `skurt_u`: sets `skurt_u`.
  - `gmd_u`: sets `gmd_u`.
  - `tg_u`: sets `tg_u`.
  - `rtg_u`: sets `rtg_u`.
  - `owa_u`: sets `owa_u`.
  - `dvar_u`: sets `dvar_u`.
  - `skew_u`: sets `skew_u`.
  - `sskew_u`: sets `sskew_u`.
  - `owa_p`: sets `owa_p`.
  - `owa_w`: sets `owa_w`.
  - `mu`: sets `mu`.
  - `cov`: sets `cov`.
  - `kurt`: sets `kurt`.
  - `skurt`: sets `skurt`.
  - `skew`: sets `skew`.
  - `sskew`: sets `sskew`.
  - `f_mu`: sets `f_mu`.
  - `f_cov`: sets `f_cov`.
  - `fm_returns`: sets `fm_returns`.
  - `fm_mu`: sets `fm_mu`.
  - `fm_cov`: sets `fm_cov`.
  - `bl_mu`: sets `bl_mu`.
  - `bl_cov`: sets `bl_cov`.
  - `blfm_mu`: sets `blfm_mu`.
  - `blfm_cov`: sets `blfm_cov`.
  - `cov_l`: sets `cov_l`.
  - `cov_u`: sets `cov_u`.
  - `cov_mu`: sets `cov_mu`.
  - `cov_sigma`: sets `cov_sigma`.
  - `d_mu`: sets `d_mu`.
  - `k_mu`: sets `k_mu`.
  - `k_sigma`: sets `k_sigma`.
  - `optimal`: sets `optimal`.
  - `z`: sets `z`.
  - `limits`: sets `limits`.
  - `frontier`: sets `frontier`.
  - `solvers`: sets `solvers`.
  - `opt_params`: sets `opt_params`.
  - `fail`: sets `fail`.
  - `model`: sets `model`.
  - `latest_prices`: sets `latest_prices`.
  - `alloc_optimal`: sets `alloc_optimal`.
  - `alloc_solvers`: sets `alloc_solvers`.
  - `alloc_params`: sets `alloc_params`.
  - `alloc_fail`: sets `alloc_fail`.
  - `alloc_model`: sets `alloc_model`.

# Outputs

  - [`Portfolio`](@ref) instance.
"""
function Portfolio(; prices::TimeArray = TimeArray(TimeType[], []),
                   returns::DataFrame = DataFrame(),
                   ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   timestamps::AbstractVector = Vector{Date}(undef, 0),
                   assets::AbstractVector = Vector{String}(undef, 0), short::Bool = false,
                   short_u::Real = 0.2, long_u::Real = 1.0, num_assets_l::Integer = 0,
                   num_assets_u::Integer = 0, num_assets_u_scale::Real = 100_000.0,
                   f_prices::TimeArray = TimeArray(TimeType[], []),
                   f_returns::DataFrame = DataFrame(),
                   f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   f_timestamps::AbstractVector = Vector{Date}(undef, 0),
                   f_assets::AbstractVector = Vector{String}(undef, 0),
                   loadings::DataFrame = DataFrame(),
                   loadings_opt::Union{LoadingsOpt, Nothing} = nothing,
                   msv_target::Union{<:Real, AbstractVector{<:Real}} = Inf,
                   lpm_target::Union{<:Real, AbstractVector{<:Real}} = Inf,
                   alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100,
                   beta_i::Real = alpha_i, beta::Real = alpha, b_sim::Integer = a_sim,
                   kappa::Real = 0.3, max_num_assets_kurt::Integer = 0,
                   max_num_assets_kurt_scale::Integer = 2, skew_factor::Real = Inf,
                   sskew_factor::Real = Inf,
                   rebalance::Union{Real, AbstractVector{<:Real}} = Inf,
                   rebalance_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   turnover::Union{Real, AbstractVector{<:Real}} = Inf,
                   turnover_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   kind_tracking_err::Symbol = :None, tracking_err::Real = Inf,
                   tracking_err_returns::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   tracking_err_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   a_mtx_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   b_vec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   f_risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   network_method::Symbol = :None,
                   network_sdp::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   network_penalty::Real = 0.05,
                   network_ip::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   network_ip_scale::Real = 100_000.0,
                   a_vec_cent::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   b_cent::Real = Inf, mu_l::Real = Inf, sd_u::Real = Inf,
                   mad_u::Real = Inf, ssd_u::Real = Inf, cvar_u::Real = Inf,
                   rcvar_u::Real = Inf, evar_u::Real = Inf, rvar_u::Real = Inf,
                   wr_u::Real = Inf, rg_u::Real = Inf, flpm_u::Real = Inf,
                   slpm_u::Real = Inf, mdd_u::Real = Inf, add_u::Real = Inf,
                   cdar_u::Real = Inf, uci_u::Real = Inf, edar_u::Real = Inf,
                   rdar_u::Real = Inf, kurt_u::Real = Inf, skurt_u::Real = Inf,
                   gmd_u::Real = Inf, tg_u::Real = Inf, rtg_u::Real = Inf,
                   owa_u::Real = Inf, dvar_u::Real = Inf, skew_u::Real = Inf,
                   sskew_u::Real = Inf,
                   owa_p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50],
                   owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   mu::AbstractVector = Vector{Float64}(undef, 0),
                   cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   skew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   V::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   sskew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   SV::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   f_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   f_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   fm_returns::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   fm_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   fm_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   bl_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   bl_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   blfm_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   blfm_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   cov_l::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   cov_u::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   cov_mu::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   cov_sigma::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   d_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   k_mu::Real = Inf, k_sigma::Real = Inf, optimal::AbstractDict = Dict(),
                   z::AbstractDict = Dict(), limits::AbstractDict = Dict(),
                   frontier::AbstractDict = Dict(),
                   solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
                   opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),
                   fail::AbstractDict = Dict(), model::JuMP.Model = JuMP.Model(),
                   latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   alloc_optimal::AbstractDict = Dict(),
                   alloc_leftover::AbstractDict = Dict(),
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
    max_num_assets_kurt_scale = clamp(max_num_assets_kurt_scale, 1, size(returns, 2))
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
    if !isempty(skew)
        @smart_assert(size(skew, 1) == size(returns, 2) &&
                      size(skew, 2) == size(returns, 2)^2)
    end
    if !isempty(V)
        @smart_assert(size(V, 1) == size(V, 2) == size(returns, 2))
    end
    if !isempty(sskew)
        @smart_assert(size(sskew, 1) == size(returns, 2) &&
                      size(sskew, 2) == size(returns, 2)^2)
    end
    if !isempty(SV)
        @smart_assert(size(SV, 1) == size(SV, 2) == size(returns, 2))
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
                     typeof(max_num_assets_kurt_scale), typeof(skew_factor),
                     typeof(sskew_factor), Union{<:Real, AbstractVector{<:Real}},
                     typeof(rebalance_weights), Union{<:Real, AbstractVector{<:Real}},
                     typeof(turnover_weights), typeof(kind_tracking_err),
                     typeof(tracking_err), typeof(tracking_err_returns),
                     typeof(tracking_err_weights), typeof(bl_bench_weights),
                     typeof(a_mtx_ineq), typeof(b_vec_ineq), typeof(risk_budget),
                     typeof(f_risk_budget), typeof(network_method), typeof(network_sdp),
                     typeof(network_penalty), typeof(network_ip), typeof(network_ip_scale),
                     typeof(a_vec_cent), typeof(b_cent), typeof(mu_l), typeof(sd_u),
                     typeof(mad_u), typeof(ssd_u), typeof(cvar_u), typeof(rcvar_u),
                     typeof(evar_u), typeof(rvar_u), typeof(wr_u), typeof(rg_u),
                     typeof(flpm_u), typeof(slpm_u), typeof(mdd_u), typeof(add_u),
                     typeof(cdar_u), typeof(uci_u), typeof(edar_u), typeof(rdar_u),
                     typeof(kurt_u), typeof(skurt_u), typeof(gmd_u), typeof(tg_u),
                     typeof(rtg_u), typeof(owa_u), typeof(dvar_u), typeof(skew_u),
                     typeof(sskew_u), typeof(owa_p), typeof(owa_w), typeof(mu), typeof(cov),
                     typeof(kurt), typeof(skurt), typeof(L_2), typeof(S_2), typeof(skew),
                     typeof(V), typeof(sskew), typeof(SV), typeof(f_mu), typeof(f_cov),
                     typeof(fm_returns), typeof(fm_mu), typeof(fm_cov), typeof(bl_mu),
                     typeof(bl_cov), typeof(blfm_mu), typeof(blfm_cov), typeof(cov_l),
                     typeof(cov_u), typeof(cov_mu), typeof(cov_sigma), typeof(d_mu),
                     typeof(k_mu), typeof(k_sigma), typeof(optimal), typeof(z),
                     typeof(limits), typeof(frontier), Union{<:AbstractDict, NamedTuple},
                     Union{<:AbstractDict, NamedTuple}, typeof(fail), typeof(model),
                     typeof(latest_prices), typeof(alloc_optimal), typeof(alloc_leftover),
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
                                                              max_num_assets_kurt_scale,
                                                              skew_factor, sskew_factor,
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
                                                              dvar_u, skew_u, sskew_u,
                                                              owa_p, owa_w, mu, cov, kurt,
                                                              skurt, L_2, S_2, skew, V,
                                                              sskew, SV, f_mu, f_cov,
                                                              fm_returns, fm_mu, fm_cov,
                                                              bl_mu, bl_cov, blfm_mu,
                                                              blfm_cov, cov_l, cov_u,
                                                              cov_mu, cov_sigma, d_mu, k_mu,
                                                              k_sigma, optimal, z, limits,
                                                              frontier, solvers, opt_params,
                                                              fail, model, latest_prices,
                                                              alloc_optimal, alloc_leftover,
                                                              alloc_solvers, alloc_params,
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
    elseif sym == :max_num_assets_kurt_scale
        val = clamp(val, 1, size(obj.returns, 2))
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
    elseif sym ∈ (:skew, :sskew)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(obj.returns, 2) &&
                          size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns)
        throw(ArgumentError("$sym is related to other fields and therefore cannot be manually changed without compromising correctness, please create a new instance of Portfolio instead"))
    elseif sym ∈ (:mu, :fm_mu, :bl_mu, :blfm_mu, :d_mu, :latest_prices)
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:cov, :fm_cov, :bl_cov, :blfm_cov, :cov_l, :cov_u, :cov_mu, :V, :SV)
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
                     typeof(obj.max_num_assets_kurt), typeof(obj.max_num_assets_kurt_scale),
                     typeof(obj.skew_factor), typeof(obj.sskew_factor),
                     Union{<:Real, AbstractVector{<:Real}}, typeof(obj.rebalance_weights),
                     Union{<:Real, AbstractVector{<:Real}}, typeof(obj.turnover_weights),
                     typeof(obj.kind_tracking_err), typeof(obj.tracking_err),
                     typeof(obj.tracking_err_returns), typeof(obj.tracking_err_weights),
                     typeof(obj.bl_bench_weights), typeof(obj.a_mtx_ineq),
                     typeof(obj.b_vec_ineq), typeof(obj.risk_budget),
                     typeof(obj.f_risk_budget), typeof(obj.network_method),
                     typeof(obj.network_sdp), typeof(obj.network_penalty),
                     typeof(obj.network_ip), typeof(obj.network_ip_scale),
                     typeof(obj.a_vec_cent), typeof(obj.b_cent), typeof(obj.mu_l),
                     typeof(obj.sd_u), typeof(obj.mad_u), typeof(obj.ssd_u),
                     typeof(obj.cvar_u), typeof(obj.rcvar_u), typeof(obj.evar_u),
                     typeof(obj.rvar_u), typeof(obj.wr_u), typeof(obj.rg_u),
                     typeof(obj.flpm_u), typeof(obj.slpm_u), typeof(obj.mdd_u),
                     typeof(obj.add_u), typeof(obj.cdar_u), typeof(obj.uci_u),
                     typeof(obj.edar_u), typeof(obj.rdar_u), typeof(obj.kurt_u),
                     typeof(obj.skurt_u), typeof(obj.gmd_u), typeof(obj.tg_u),
                     typeof(obj.rtg_u), typeof(obj.owa_u), typeof(obj.dvar_u),
                     typeof(obj.skew_u), typeof(obj.sskew_u), typeof(obj.owa_p),
                     typeof(obj.owa_w), typeof(obj.mu), typeof(obj.cov), typeof(obj.kurt),
                     typeof(obj.skurt), typeof(obj.L_2), typeof(obj.S_2), typeof(obj.skew),
                     typeof(obj.V), typeof(obj.sskew), typeof(obj.SV), typeof(obj.f_mu),
                     typeof(obj.f_cov), typeof(obj.fm_returns), typeof(obj.fm_mu),
                     typeof(obj.fm_cov), typeof(obj.bl_mu), typeof(obj.bl_cov),
                     typeof(obj.blfm_mu), typeof(obj.blfm_cov), typeof(obj.cov_l),
                     typeof(obj.cov_u), typeof(obj.cov_mu), typeof(obj.cov_sigma),
                     typeof(obj.d_mu), typeof(obj.k_mu), typeof(obj.k_sigma),
                     typeof(obj.optimal), typeof(obj.z), typeof(obj.limits),
                     typeof(obj.frontier), Union{<:AbstractDict, NamedTuple},
                     Union{<:AbstractDict, NamedTuple}, typeof(obj.fail), typeof(obj.model),
                     typeof(obj.latest_prices), typeof(obj.alloc_optimal),
                     typeof(obj.alloc_leftover), Union{<:AbstractDict, NamedTuple},
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
                                              deepcopy(obj.max_num_assets_kurt_scale),
                                              deepcopy(obj.skew_factor),
                                              deepcopy(obj.sskew_factor),
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
                                              deepcopy(obj.owa_u), deepcopy(obj.dvar_u),
                                              deepcopy(obj.skew_u), deepcopy(obj.sskew_u),
                                              deepcopy(obj.owa_p), deepcopy(obj.owa_w),
                                              deepcopy(obj.mu), deepcopy(obj.cov),
                                              deepcopy(obj.kurt), deepcopy(obj.skurt),
                                              deepcopy(obj.L_2), deepcopy(obj.S_2),
                                              deepcopy(obj.skew), deepcopy(obj.V),
                                              deepcopy(obj.sskew), deepcopy(obj.SV),
                                              deepcopy(obj.f_mu), deepcopy(obj.f_cov),
                                              deepcopy(obj.fm_returns), deepcopy(obj.fm_mu),
                                              deepcopy(obj.fm_cov), deepcopy(obj.bl_mu),
                                              deepcopy(obj.bl_cov), deepcopy(obj.blfm_mu),
                                              deepcopy(obj.blfm_cov), deepcopy(obj.cov_l),
                                              deepcopy(obj.cov_u), deepcopy(obj.cov_mu),
                                              deepcopy(obj.cov_sigma), deepcopy(obj.d_mu),
                                              deepcopy(obj.k_mu), deepcopy(obj.k_sigma),
                                              deepcopy(obj.optimal), deepcopy(obj.z),
                                              deepcopy(obj.limits), deepcopy(obj.frontier),
                                              deepcopy(obj.solvers),
                                              deepcopy(obj.opt_params), deepcopy(obj.fail),
                                              copy(obj.model), deepcopy(obj.latest_prices),
                                              deepcopy(obj.alloc_optimal),
                                              deepcopy(obj.alloc_leftover),
                                              deepcopy(obj.alloc_solvers),
                                              deepcopy(obj.alloc_params),
                                              deepcopy(obj.alloc_fail),
                                              copy(obj.alloc_model))
end

"""
```
mutable struct HCPortfolio{ast, dat, r, ai, a, as, bi, b, bs, k, ata, mnak, mnaks, skewf,
                           sskewf, owap, wowa, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew,
                           tv, tsskew, tsv,tbin, wmi, wma, ttco, tco, tdist, tcl, tk, topt,
                           tsolv, toptpar, tf, tlp, taopt, tasolv, taoptpar, taf, tamod} <:
               AbstractPortfolio
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
    max_num_assets_kurt_scale::mnaks
    skew_factor::skewf
    sskew_factor::sskewf
    owa_p::owap
    owa_w::wowa
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    skew::tskew
    V::tv
    sskew::tsskew
    SV::tsv
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
```

Structure for working with hierarchical portfolios.

Some of these require external information from the arguments of functions that use the an instance of [`HCPortfolio`](@ref):

  - `type`: one of [`PortTypes`](@ref).
  - `rm`: one of [`RiskMeasures`](@ref).
  - `owa_approx`: flag for using the approximate OWA formulation.

# Inputs

## Portfolio characteristics

  - `assets`: `Na×1` vector of assets, where `Na` is the number of assets.
  - `timestamps`: `T×1` vector of timestamps, where `T` is the number of returns observations.
  - `returns`: `T×Na` matrix of asset returns, where `T` is the number of returns observations and `Na` is the number of assets.

## Risk parameters

  - `alpha_i`:

      + `rm ∈ (:TG, :TGRG)`: initial significance level of losses, `0 < alpha_i < alpha < 1`.

  - `a_sim`:

      + `rm ∈ (:TG, :TGRG)`: number of CVaRs to approximate the losses, `a_sim > 0`.
  - `alpha`:

      + `rm ∈ (:VaR, :CVaR, :EVaR, :RLVaR, :CVaRRG, :TG, :TGRG, :DaR, :CDaR, :EDaR, :RLDaR, :DaR_r, :CDaR_r, :EDaR_r, :RLDaR_r)`: significance level of losses, `alpha ∈ (0, 1)`.
  - `beta_i`:

      + `rm == :TGRG`: initial significance level of gains, `0 < beta_i < beta < 1`.
  - `b_sim`:

      + `rm == :TGRG`: number of CVaRs to approximate the gains, `b_sim > 0`.
  - `beta`:

      + `rm ∈ (:CVaRRG, :TGRG)`: significance level of gains, `beta ∈ (0, 1)`.
  - `kappa`:

      + `rm ∈ (:RLVaR, :RLDaR, :RLDaR_r)`: relativistic deformation parameter.
  - `max_num_assets_kurt`:

      + `type == :NCO`:

          * `iszero(max_num_assets_kurt)`: use the full kurtosis model.
          * `!iszero(max_num_assets_kurt)`: if the number of assets surpases this value, use the relaxed kurtosis model.
  - `max_num_assets_kurt_scale`: the relaxed kurtosis model uses the largest `max_num_assets_kurt_scale * max_num_assets_kurt` eigenvalues to approximate the kurtosis matrix, `max_num_assets_kurt_scale ∈ [1, Na]`, where `Na` is the number of assets.
  - `skew_factor`: factor for adding the multiple of the negative quadratic skewness to the risk function.
  - `sskew_factor`: factor for adding the multiple of the negative quadratic semi skewness to the risk function.

##. OWA parameters

Only relevant when `rm ∈ (:GMD, :TG, :TGRG, :OWA)`.

  - `owa_p`:

      + `owa_approx = true`: `C×1` vector containing the order of the p-norms used in the approximate formulation of the risk measures, where `C` is the number of p-norms. The more entries and larger their range, the more accurate the approximation.

  - `owa_w`:

      + `rm == :OWA`: `T×1` OWA vector, where `T` is the number of returns observations. Useful for optimising higher L-moments.

## Asset constraints

  - `w_min`: `Na×1` vector of the lower bounds for asset weights, where `Na` is the number of assets.
  - `w_max`: `Na×1` vector of the upper bounds for asset weights, where `Na` is the number of assets.

## Model statistics

### Asset statistics

  - `mu`: `Na×1` asset expected returns vector, where `Na` is the number of assets.
  - `cov`: `Na×Na` asset covariance matrix, where `Na` is the number of assets.
  - `kurt`: `(Na^2)×(Na^2)` asset cokurtosis matrix, where `Na` is the number of assets.
  - `skurt`: `(Na^2)×(Na^2)` asset semi cokurtosis matrix, where `Na` is the number of assets.
  - `L_2`: `(Na^2)×((Na^2 + Na)/2)` elimination matrix, where `Na` is the number of assets.
  - `S_2`: `((Na^2 + Na)/2)×(Na^2)` summation matrix, where `Na` is the number of assets.

### Clustering statistics

  - `bins_info`: selection criterion for computing the number of bins used to calculate the mutual and variation of information statistics, see [`mut_var_info_mtx`](@ref) for available choices.

  - `cor_method`: method for estimating the codependence matrix.
  - `cor`: `Na×Na` matrix, where where `Na` is the number of assets. Set the value of the codependence matrix at instance construction.
  - `dist`:  `Na×Na` matrix, where where `Na` is the number of assets. Set the value of the distance matrix at instance construction.
  - `clusters`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of asset clusters.
  - `k`: number of clusters to cut the dendrogram into.

      + `iszero(k)`, automatically compute `k` using the two difference gap statistic [^TDGS].
      + `!iszero(k)`, use the value directly.

## Optimal portfolios

  - `optimal`: collection capable of storing key value pairs for storing optimal portfolios.

## Solutions

  - `solvers`: provides the solvers and corresponding parameters for solving the portfolio optimisation problem. There can be two `key => value` pairs.

      + `:solver => value`: `value` is a `JuMP` optimiser. The optimiser can be declared alongside its attributes by using `JuMP.solver_with_attributes`.
      + `:params => value`: (optional) `value` must be a `Dict` or `NamedTuple` whose `key => value` pairs are the solver-specific settings.

  - `opt_params`: collection capable of storing key value pairs for storing parameters used for optimising.
  - `fail`: collection capable of storing key value pairs for storing failed optimisation attempts.
  - `model`: `JuMP.Model()` for optimising a portfolio.

## Allocation

  - `latest_prices`: `Na×1` vector of asset prices, `Na` is the number of assets.

  - `alloc_optimal`: collection capable of storing key value pairs for storing optimal portfolios after allocating discrete stocks.
  - `alloc_solvers`: provides the solvers and corresponding parameters for solving the discrete allocation optimisation problem. There can be two `key => value` pairs.

      + `:solver => value`: `value` is a `JuMP` optimiser. The optimiser can be declared alongside its attributes by using `JuMP.solver_with_attributes`. Solver must support MIP constraints.
      + `:params => value`: (optional) `value` must be a `Dict` or `NamedTuple` whose `key => value` pairs are the solver-specific settings.
  - `alloc_params`: collection capable of storing key value pairs for storing parameters used for optimising the portfolio allocation.
  - `alloc_fail`: collection capable of storing key value pairs for storing failed optimisation attempts.
  - `alloc_model`: `JuMP.Model()` for optimising a portfolio allocation.
"""
mutable struct HCPortfolio{ast, dat, r, ai, a, as, bi, b, bs, k, ata, mnak, mnaks, skewf,
                           sskewf, owap, wowa, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew,
                           tv, tsskew, tsv, tbin, wmi, wma, ttco, tco, tdist, tcl, tk, topt,
                           tsolv, toptpar, tf, tlp, taopt, talo, tasolv, taoptpar, taf,
                           tamod} <: AbstractPortfolio
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
    max_num_assets_kurt_scale::mnaks
    skew_factor::skewf
    sskew_factor::sskewf
    owa_p::owap
    owa_w::wowa
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    skew::tskew
    V::tv
    sskew::tsskew
    SV::tsv
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
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_params::taoptpar
    alloc_fail::taf
    alloc_model::tamod
end

"""
```
HCPortfolio(; prices::TimeArray = TimeArray(TimeType[], []),
            returns::DataFrame = DataFrame(),
            ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
            assets::AbstractVector = Vector{String}(undef, 0), alpha_i::Real = 0.0001,
            alpha::Real = 0.05, a_sim::Integer = 100, beta_i::Real = alpha_i,
            beta::Real = alpha, b_sim::Integer = a_sim, kappa::Real = 0.3,
            alpha_tail::Real = 0.05, max_num_assets_kurt::Integer = 0,
            max_num_assets_kurt_scale::Integer = 2, skew_factor::Real = Inf,
            sskew_factor::Real = Inf,
            owa_p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50],
            owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
            mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
            cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            skew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            V = Matrix{eltype(returns)}(undef, 0, 0),
            sskew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            SV = Matrix{eltype(returns)}(undef, 0, 0),
            bins_info::Union{Symbol, <:Integer} = :KN,
            w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,
            w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,
            cor_method::Symbol = :Pearson,
            cor::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            clusters::Clustering.Hclust = Hclust{Float64}(Matrix{Int64}(undef, 0, 2),
                                                          Float64[], Int64[], :nothing),
            k::Integer = 0, optimal::AbstractDict = Dict(),
            solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
            opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),
            fail::AbstractDict = Dict(),
            latest_prices::AbstractVector = Vector{Float64}(undef, 0),
            alloc_optimal::AbstractDict = Dict(),
            alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
            alloc_params::Union{<:AbstractDict, NamedTuple} = Dict(),
            alloc_fail::AbstractDict = Dict(), alloc_model::JuMP.Model = JuMP.Model())
```

Performs data validation and creates an instance of [`HCPortfolio`](@ref). Union datatypes remain union datatypes in the instance.

# Inputs

  - `prices`:

      + `!isempty(prices)`: timearray of asset prices, automatically sets `assets`, `timestamps`, computes `returns` and `latest_prices`. Takes priority over the `returns`, `ret`, `timestamps`, `assets` and `latest_prices` arguments.

  - `returns`:

      + `!isempty(returns)`: dataframe of asset returns, automatically sets `assets`, `timestamps`, `returns`. Takes priority over the, `ret`, `timestamps` and `assets` arguments.
  - `ret`: sets `returns`.
  - `timestamps`: sets `timestamps`.
  - `assets`: sets `assets`.
  - `alpha_i`: sets `alpha_i`.
  - `alpha`: sets `alpha`.
  - `a_sim`: sets `a_sim`.
  - `beta_i`: sets `beta_i`.
  - `beta`: sets `beta`.
  - `b_sim`: sets `b_sim`.
  - `kappa`: sets `kappa`.
  - `alpha_tail`: sets `alpha_tail`.
  - `max_num_assets_kurt`: sets `max_num_assets_kurt`.
  - `max_num_assets_kurt_scale`: sets `max_num_assets_kurt_scale`.
  - `skew_factor`: sets `skew_factor`.
  - `sskew_factor`: sets `sskew_factor`.
  - `owa_p`: sets `owa_p`.
  - `owa_w`: sets `owa_w`.
  - `mu`: sets `mu`.
  - `cov`: sets `cov`.
  - `kurt`: sets `kurt`.
  - `skurt`: sets `skurt`.
  - `skew`: sets `skew`.
  - `sskew`: sets `sskew`.
  - `bins_info`: sets `bins_info`.
  - `w_min`: sets `w_min`.
  - `w_max`: sets `w_max`.
  - `cor_method`: sets `cor_method`.
  - `cor`: sets `cor`.
  - `dist`: sets `dist`.
  - `clusters`: sets `clusters`.
  - `k`: sets `k`.
  - `optimal`: sets `optimal`.
  - `solvers`: sets `solvers`.
  - `opt_params`: sets `opt_params`.
  - `fail`: sets `fail`.
  - `latest_prices`: sets `latest_prices`.
  - `alloc_optimal`: sets `alloc_optimal`.
  - `alloc_solvers`: sets `alloc_solvers`.
  - `alloc_params`: sets `alloc_params`.
  - `alloc_fail`: sets `alloc_fail`.
  - `alloc_model`: sets `alloc_model`.

# Outputs

  - [`HCPortfolio`](@ref) instance.
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
                     max_num_assets_kurt_scale::Integer = 2, skew_factor::Real = Inf,
                     sskew_factor::Real = Inf,
                     owa_p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50],
                     owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                     mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                     cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     skew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     V = Matrix{eltype(returns)}(undef, 0, 0),
                     sskew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     SV = Matrix{eltype(returns)}(undef, 0, 0),
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
                     alloc_leftover::AbstractDict = Dict(),
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
    max_num_assets_kurt_scale = clamp(max_num_assets_kurt_scale, 1, size(returns, 2))
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
    if !isempty(skew)
        @smart_assert(size(skew, 1) == size(returns, 2) &&
                      size(skew, 2) == size(returns, 2)^2)
    end
    if !isempty(V)
        @smart_assert(size(V, 1) == size(V, 2) == size(returns, 2))
    end
    if !isempty(sskew)
        @smart_assert(size(sskew, 1) == size(returns, 2) &&
                      size(sskew, 2) == size(returns, 2)^2)
    end
    if !isempty(SV)
        @smart_assert(size(SV, 1) == size(SV, 2) == size(returns, 2))
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
                       typeof(max_num_assets_kurt), typeof(max_num_assets_kurt_scale),
                       typeof(skew_factor), typeof(sskew_factor), typeof(owa_p),
                       typeof(owa_w), typeof(mu), typeof(cov), typeof(kurt), typeof(skurt),
                       typeof(L_2), typeof(S_2), typeof(skew), typeof(V), typeof(sskew),
                       typeof(SV), Union{Symbol, <:Integer},
                       Union{<:Real, AbstractVector{<:Real}},
                       Union{<:Real, AbstractVector{<:Real}}, typeof(cor_method),
                       typeof(cor), typeof(dist), typeof(clusters), typeof(k),
                       typeof(optimal), Union{<:AbstractDict, NamedTuple},
                       Union{<:AbstractDict, NamedTuple}, typeof(fail),
                       typeof(latest_prices), typeof(alloc_optimal), typeof(alloc_leftover),
                       Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                       typeof(alloc_fail), typeof(alloc_model)}(assets, timestamps, returns,
                                                                alpha_i, alpha, a_sim,
                                                                beta_i, beta, b_sim, kappa,
                                                                alpha_tail,
                                                                max_num_assets_kurt,
                                                                max_num_assets_kurt_scale,
                                                                skew_factor, sskew_factor,
                                                                owa_p, owa_w, mu, cov, kurt,
                                                                skurt, L_2, S_2, skew, V,
                                                                sskew, SV, bins_info, w_min,
                                                                w_max, cor_method, cor,
                                                                dist, clusters, k, optimal,
                                                                solvers, opt_params, fail,
                                                                latest_prices,
                                                                alloc_optimal,
                                                                alloc_leftover,
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
    elseif sym == :max_num_assets_kurt_scale
        val = clamp(val, 1, size(obj.returns, 2))
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
    elseif sym ∈ (:skew, :sskew)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(obj.returns, 2) &&
                          size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:L_2, :S_2)
        if !isempty(val)
            N = size(obj.returns, 2)
            @smart_assert(size(val) == (Int(N * (N + 1) / 2), N^2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:cov, :cor, :dist, :V, :SV)
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
                       typeof(obj.max_num_assets_kurt),
                       typeof(obj.max_num_assets_kurt_scale), typeof(obj.skew_factor),
                       typeof(obj.sskew_factor), typeof(obj.owa_p), typeof(obj.owa_w),
                       typeof(obj.mu), typeof(obj.cov), typeof(obj.kurt), typeof(obj.skurt),
                       typeof(obj.L_2), typeof(obj.S_2), typeof(obj.skew), typeof(obj.V),
                       typeof(obj.sskew), typeof(obj.SV), Union{Symbol, <:Integer},
                       Union{<:Real, AbstractVector{<:Real}},
                       Union{<:Real, AbstractVector{<:Real}}, typeof(obj.cor_method),
                       typeof(obj.cor), typeof(obj.dist), typeof(obj.clusters),
                       typeof(obj.k), typeof(obj.optimal),
                       Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                       typeof(obj.fail), typeof(obj.latest_prices),
                       typeof(obj.alloc_optimal), typeof(obj.alloc_leftover),
                       Union{<:AbstractDict, NamedTuple}, Union{<:AbstractDict, NamedTuple},
                       typeof(obj.alloc_fail), typeof(obj.alloc_model)}(deepcopy(obj.assets),
                                                                        deepcopy(obj.timestamps),
                                                                        deepcopy(obj.returns),
                                                                        deepcopy(obj.alpha_i),
                                                                        deepcopy(obj.alpha),
                                                                        deepcopy(obj.a_sim),
                                                                        deepcopy(obj.beta_i),
                                                                        deepcopy(obj.beta),
                                                                        deepcopy(obj.b_sim),
                                                                        deepcopy(obj.kappa),
                                                                        deepcopy(obj.alpha_tail),
                                                                        deepcopy(obj.max_num_assets_kurt),
                                                                        deepcopy(obj.max_num_assets_kurt_scale),
                                                                        deepcopy(obj.skew_factor),
                                                                        deepcopy(obj.sskew_factor),
                                                                        deepcopy(obj.owa_p),
                                                                        deepcopy(obj.owa_w),
                                                                        deepcopy(obj.mu),
                                                                        deepcopy(obj.cov),
                                                                        deepcopy(obj.kurt),
                                                                        deepcopy(obj.skurt),
                                                                        deepcopy(obj.L_2),
                                                                        deepcopy(obj.S_2),
                                                                        deepcopy(obj.skew),
                                                                        deepcopy(obj.V),
                                                                        deepcopy(obj.sskew),
                                                                        deepcopy(obj.SV),
                                                                        deepcopy(obj.bins_info),
                                                                        deepcopy(obj.w_min),
                                                                        deepcopy(obj.w_max),
                                                                        deepcopy(obj.cor_method),
                                                                        deepcopy(obj.cor),
                                                                        deepcopy(obj.dist),
                                                                        deepcopy(obj.clusters),
                                                                        deepcopy(obj.k),
                                                                        deepcopy(obj.optimal),
                                                                        deepcopy(obj.solvers),
                                                                        deepcopy(obj.opt_params),
                                                                        deepcopy(obj.fail),
                                                                        deepcopy(obj.latest_prices),
                                                                        deepcopy(obj.alloc_optimal),
                                                                        deepcopy(obj.alloc_leftover),
                                                                        deepcopy(obj.alloc_solvers),
                                                                        deepcopy(obj.alloc_params),
                                                                        deepcopy(obj.alloc_fail),
                                                                        copy(obj.alloc_model))
end

export Portfolio, HCPortfolio
