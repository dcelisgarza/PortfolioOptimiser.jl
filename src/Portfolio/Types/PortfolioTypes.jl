"""
```
abstract type AbstractPortfolio end
```

Abstract type for subtyping portfolios.
"""
abstract type AbstractPortfolio end

"""
```
mutable struct Portfolio{ast, dat, r, tfa, tfdat, tretf, l, lo, s, us, ul, nal, nau, naus,
                         mnak, mnaks, rb, to, kte, blbw, ami, bvi, rbv, frbv, nm, amc, bvc,
                         ler, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew, tv, tsskew, tsv,
                         tmuf, tcovf, trfm, tmufm, tcovfm, tmubl, tcovbl, tmublf, tcovblf,
                         tcovl, tcovu, tcovmu, tcovs, tdmu, tkmu, tks, topt, tlim, tfront,
                         tsolv, tf, tmod, tlp, taopt, talo, tasolv, taf, tamod} <:
               AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    regression_type::lo
    short::s
    short_u::us
    long_u::ul
    num_assets_l::nal
    num_assets_u::nau
    num_assets_u_scale::naus
    max_num_assets_kurt::mnak
    max_num_assets_kurt_scale::mnaks
    rebalance::rb
    turnover::to
    tracking_err::kte
    bl_bench_weights::blbw
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget::rbv
    f_risk_budget::frbv
    network_adj::nm
    a_vec_cent::amc
    b_cent::bvc
    mu_l::ler
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
    limits::tlim
    frontier::tfront
    solvers::tsolv
    fail::tf
    model::tmod
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_fail::taf
    alloc_model::tamod
end
```

Structure for defining a traditional portfolio. `Na` is the number of assets, and `Nf` is the number of factors. For details on how some of these parameters are computed see [`asset_statistics!`](@ref), [`wc_statistics!`](@ref), [`factor_statistics!`](@ref), [`black_litterman_statistics!`](@ref), and [`black_litterman_factor_statistics!`](@ref).

# Parameters

  - `assets`: `Na×1` vector of asset names.

  - `timestamps`: `T×1` vector of asset returns timestamps.
  - `returns`: `T×Na` matrix of asset returns.
  - `f_assets`: `Nf×1` vector of factor names.
  - `f_timestamps`: `T×1` vector of factor returns timestamps.
  - `f_returns`: `T×Nf` matrix of asset returns.
  - `loadings`: loadings matrix for working with factor models.
  - `regression_type`: [`RegressionType`](@ref) used for computing the loadings matrix.
  - `short`:

      + if `true`: shorting is enabled.
      + else: long-only portfolio.
  - `short_u`: upper bound for the absolute value of the sum of the negative weights.
  - `long_u`: upper bound for the sum of the positive weights.
  - `num_assets_l`: lower bound for the minimum number of significant assets.

      + if `> 0`: applies the constraint.
  - `num_assets_u`: upper bound for the maximum number of significant assets.

      + if `> 0`: applies the constraint.
  - `num_assets_u_scale`: scaling factor for the decision variable used for applying the `num_assets_u` constraint when optimising the [`Sharpe`](@ref) objective function.
  - `max_num_assets_kurt`: maximum number of assets to use the complete kurtosis model.

      + if `> 0`: the approximate model will be used if the number of assets in the portfolio exceeds `max_number_assets_kurt`.
  - `max_num_assets_kurt_scale`: multipies `Na` to find the number of eigenvalues when computing the approximate kurtosis model, must be `∈ [1, Na]`.
  - `rebalance`: [`AbstractTR`](@ref) for defining the portfolio rebalancing penalty.

    ```math
    \\begin{align}
    p_{r} &= \\sum\\limits_{i=1}^{N} r_{i} \\lvert w_{i} - b_{i} \\rvert\\,.
    \\end{align}
    ```

    Where:

      + ``p_{r}`` is the portfolio rebalancing penalty.
      + ``N`` is the number of assets.
      + ``r_{i}`` is the rebalancing penalty for the ``i``-th asset.
      + ``w_{i}`` is the weight of the ``i``-th asset.
      + ``b_{i}`` is the benchmark weight of the ``i``-th asset.
  - `turnover`: [`AbstractTR`](@ref) for defining the asset turnover constraint.

    ```math
    \\begin{align}
    \\lvert w_{i} - b_{i} &\\rvert \\leq t_{i}\\quad \\forall i = 1,\\,\\ldots,\\,N\\,.
    \\end{align}
    ```

    Where:

      + ``t_{i}`` is the turnover constraint for the ``i``-th asset.
      + ``w_{i}`` is the weight of the ``i``-th asset.
      + ``b_{i}`` is the benchmark weight of the ``i``-th asset.
      + ``N`` is the number of assets.
  - `tracking_err`: [`TrackingErr`](@ref) for defining the tracking error constraint.

    ```math
    \\begin{align}
    \\left\\lVert \\dfrac{\\mathbf{X} \\bm{w} - \\bm{b}}{T - 1} \\right\\rVert_{2} &\\leq \\epsilon
    \\end{align}
    ```

    Where:

      + ``\\lVert \\cdot \\rVert_{2}`` is the L2 norm.
      + ``\\mathbf{X}`` is the ``T \\times N`` matrix of asset returns.
      + ``T`` is the number of returns observations.
      + ``N`` is the number of assets.
      + ``\\bm{w}`` is the ``N \\times 1`` vector of asset weights.
      + ``\\bm{b}`` is the ``T \\times 1`` vector of benchmark returns.
      + ``\\epsilon`` is the tracking error.
  - `bl_bench_weights`: benchmark weights for Black-Litterman models [`BlackLittermanClass`](@ref).
  - `a_mtx_ineq`: `C×N` matrix of asset weight linear constraints.

      + if `isempty`: the constraint is not set.
  - `b_vec_ineq`: `C×1` vector of asset weight linear constraints.

      + if `isempty`: the constraint is not set.
  - The linear weight constraint is defined as.

    ```math
    \\begin{align}
    \\mathbf{A} \\bm{w} &\\geq \\bm{b}\\,.
    \\end{align}
    ```

    Where:

      + ``\\mathbf{A}`` is the ``C×N`` matrix of asset weight linear constraints.

      + ``\\bm{b}`` is the ``C×1`` vector of asset weight linear constraints.
      + ``C`` is the number of constraints.
      + ``N`` is the number of assets.
  - `risk_budget`: `Na×1` vector of asset risk budgets.
  - `f_risk_budget`: `Nf×1` vector of factor risk budgets.
  - `network_adj`: [`AdjacencyConstraint`](@ref) for defining the asset network constraint. This can be defined in two ways, using an exact mixed-integer approach [`IP`](@ref) or an approximate semi-definite one [`SDP`](@ref). See their docs for the constraint definition for each case.

      + if [`NoAdj`](@ref): the constraint is not set.
  - `a_vec_cent`: centrality vector for defining the centrality constraint.

      + if `isempty`: the constraint is not set.
  - `b_cent`: average centrality of the assets the portfolio.

      + if `isinf`: the constraint is not set.
  - The centrality constraint is defined as.

    ```math
    \\begin{align}
    \\bm{C} \\cdot \\bm{w} &= \\bar{c}
    \\end{align}
    ```

    Where:

      + ``\\bm{w}`` is the ``N\\times 1`` vector of asset weights.
      + ``\\bm{C}`` is the ``N \\times 1`` centrality vector of the asset adjacency matrix.
      + ``\\cdot`` is the dot product.
      + ``\\bar{c}`` is the desired average centrality measure of the portfolio.
  - `mu_l`: lower bound for the expected return of the portfolio.

      + if is `Inf`: the constraint is not applied.
  - `mu`: `Na×1` vector of asset expected returns.
  - `cov`: `Na×Na` asset covariance matrix.
  - `kurt`: `Na^2×Na^2` cokurtosis matrix.
  - `skurt`: `Na^2×Na^2` semi cokurtosis matrix.
  - `L_2`: `(Na^2)×((Na^2 + Na)/2)` elimination matrix.
  - `S_2`: `((Na^2 + Na)/2)×(Na^2)` summation matrix.
  - `skew`: `Na×Na^2` coskew matrix.
  - `V`: `Na×Na` sum of the symmetric negative spectral slices of coskewness.
  - `sskew`: `Na×Na^2` semi coskew matrix.
  - `SV`: `Na×Na` sum of the symmetric negative spectral slices of semi coskewness.
  - `f_mu`: `Nf×1` vector of factor expected returns.
  - `f_cov`: `Nf×Nf` factor covariance matrix.
  - `fm_returns`: `T×Na` factor model adjusted returns matrix.
  - `fm_mu`: `Na×1` factor model adjusted asset expected returns.
  - `fm_cov`: `Na×Na` factor model adjusted asset covariance matrix.
  - `bl_mu`: `Na×1` Black Litterman model adjusted asset expected returns.
  - `bl_cov`: `Na×Na` Black Litterman model adjusted asset covariance matrix.
  - `blfm_mu`: `Na×1` Black Litterman factor model adjusted asset expected returns.
  - `blfm_cov`: `Na×Na` Black Litterman factor model adjusted asset covariance matrix.
  - `cov_l`: `Na×Na` lower bound for the worst case covariance matrix.
  - `cov_u`: `Na×Na` upper bound for the worst case covariance matrix.
  - `cov_mu`: `Na×Na` matrix of the estimation errors of the asset expected returns vector set.
  - `cov_sigma`: `Na×Na` matrix of the estimation errors of the asset covariance matrix set.
  - `d_mu`: absolute deviation of the worst case upper and lower asset expected returns vectors.
  - `k_mu`: distance parameter of the uncertainty in the asset expected returns vector for the worst case optimisation.
  - `k_sigma`: distance parameter of the uncertainty in the asset covariance matrix for the worst case optimisation.
  - `optimal`: collection capable of storing key value pairs for storing optimal portfolios.
  - `limits`: collection capable of storing key value pairs for storing the minimal and maximal risk portfolios.
  - `frontier`: collection capable of storing key value pairs for containing points in the efficient frontier.
  - `solvers`: collection capable of storing key value pairs for storing `JuMP`-supported solvers. They must have the following structure.

    ```
    solvers = Dict(
                   # Key-value pair for the solver, solution acceptance 
                   # criteria, and solver attributes.
                   :Clarabel => Dict(
                                     # Solver we wish to use.
                                     :solver => Clarabel.Optimizer,
                                     # (Optional) Solution acceptance criteria.
                                     :check_sol => (allow_local = true, allow_almost = true),
                                     # (Optional) Solver-specific attributes.
                                     :params => Dict("verbose" => false)))
    ```

    The dictionary contains a key value pair for each solver (plus optional solution acceptance criteria and optional attributes) we want to use.

      + `:solver`: defines the solver to use. One can also use [`JuMP.optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to direcly provide a solver with attributes already attached.
      + `:check_sol`: (optional) defines the keyword arguments passed on to [`JuMP.is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#is_solved_and_feasible) for accepting/rejecting solutions.
      + `:params`: (optional) defines solver-specific parameters.

    Users are also able to provide multiple solvers by adding additional key-value pairs to the top-level dictionary as in the following snippet.

    ```
    using JuMP
    solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                     :check_sol => (allow_local = true, allow_almost = true),
                                     :params => Dict("verbose" => false)),
                   # Provide solver with pre-attached attributes and no arguments 
                   # for the `JuMP.is_solved_and_feasible` function.
                   :COSMO => Dict(:solver => JuMP.optimizer_with_attributes(COSMO.Optimizer,
                                                                            "maxiter" => 5000)))
    ```

    [`optimise!`](@ref) will iterate over the solvers until it finds the first one to successfully solve the problem.
  - `fail`: collection capable of storing key value pairs for storing failed optimisation attempts.
  - `model`: [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#Model) which defines the optimisation model.
  - `latest_prices`: `Na×1` vector of latest asset prices.
  - `alloc_optimal`: collection capable of storing key value pairs for storing optimal discretely allocated portfolios.
  - `alloc_leftover`: collection capable of storing key value pairs for containing points in the leftover investment after allocating.
  - `alloc_solvers`: collection capable of storing key value pairs for storing `JuMP`-supported solvers that support Mixed-Integer Programming, only used in the [`LP`](@ref) allocation.
  - `alloc_fail`: collection capable of storing key value pairs for storing failed discrete asset allocation attempts.
  - `alloc_model`: [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#Model) which defines the discrete asset allocation model.
"""
mutable struct Portfolio{ast, dat, r, tfa, tfdat, tretf, l, lo, s, lb, sb, ul, us, tfee,
                         tsfee, nal, nau, naus, mnak, mnaks, l1t, l2t, rb, to, kte, blbw,
                         ami, bvi, rbv, frbv, nm, cm, amc, bvc, ler, tmu, tcov, tkurt,
                         tskurt, tl2, ts2, tskew, tv, tsskew, tsv, tmuf, tcovf, trfm, tmufm,
                         tcovfm, tmubl, tcovbl, tmublf, tcovblf, tcovl, tcovu, tcovmu,
                         tcovs, tdmu, tkmu, tks, topt, tlim, tfront, tsolv, tf, tmod, tlp,
                         taopt, talo, tasolv, taf, tamod} <: AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    regression_type::lo
    short::s
    budget::lb
    short_budget::sb
    long_u::ul
    short_u::us
    fees::tfee
    short_fees::tsfee
    num_assets_l::nal
    num_assets_u::nau
    num_assets_u_scale::naus
    max_num_assets_kurt::mnak
    max_num_assets_kurt_scale::mnaks
    l1::l1t
    l2::l2t
    rebalance::rb
    turnover::to
    tracking_err::kte
    bl_bench_weights::blbw
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget::rbv
    f_risk_budget::frbv
    network_adj::nm
    cluster_adj::cm
    a_vec_cent::amc
    b_cent::bvc
    mu_l::ler
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
    limits::tlim
    frontier::tfront
    solvers::tsolv
    fail::tf
    model::tmod
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_fail::taf
    alloc_model::tamod
end
"""
```
Portfolio(; prices::TimeArray = TimeArray(TimeType[], []),
            returns::DataFrame = DataFrame(),
            ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            timestamps::AbstractVector = Vector{Date}(undef, 0),
            assets::AbstractVector = Vector{String}(undef, 0),
            f_prices::TimeArray = TimeArray(TimeType[], []),
            f_returns::DataFrame = DataFrame(),
            f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            f_timestamps::AbstractVector = Vector{Date}(undef, 0),
            f_assets::AbstractVector = Vector{String}(undef, 0),
            loadings::DataFrame = DataFrame(),
            regression_type::Union{<:RegressionType, Nothing} = nothing,
            short::Bool = false, short_u::Real = 0.2, long_u::Real = 1.0,
            num_assets_l::Integer = 0, num_assets_u::Integer = 0,
            num_assets_u_scale::Real = 100_000.0, max_num_assets_kurt::Integer = 0,
            max_num_assets_kurt_scale::Integer = 2, rebalance::AbstractTR = NoTR(),
            turnover::AbstractTR = NoTR(), tracking_err::TrackingErr = NoTracking(),
            bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
            a_mtx_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
            b_vec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
            risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
            f_risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
            network_adj::AdjacencyConstraint = NoAdj(),
            a_vec_cent::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
            b_cent::Real = Inf, mu_l::Real = Inf,
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
            limits::AbstractDict = Dict(), frontier::AbstractDict = Dict(),
            solvers::AbstractDict = Dict(), fail::AbstractDict = Dict(),
            model::JuMP.Model = JuMP.Model(),
            latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
            alloc_optimal::AbstractDict = Dict(),
            alloc_leftover::AbstractDict = Dict(),
            alloc_solvers::AbstractDict = Dict(), alloc_fail::AbstractDict = Dict(),
            alloc_model::JuMP.Model = JuMP.Model())
```

Constructor for [`Portfolio`](@ref). Performs data validation checks and automatically extracts the data from `prices`, `returns`, `f_prices`, and `f_returns` if they are provided.

# Inputs

  - `prices`: `(T+1)×Na` [`TimeArray`](https://juliastats.org/TimeSeries.jl/stable/timearray/#The-TimeArray-time-series-type) of asset prices.

      + If provided: will take precedence over `returns`, `ret`, `timestamps`, `assets`, and `latest_prices` because they will be automatically computed from `prices`.

  - `returns`: `T×Na` [`DataFrame`](https://dataframes.juliadata.org/stable/lib/types/#DataFrames.DataFrame) of asset returns.

      + If provided: will take precedence over `ret`, `timestamps`, and `assets` because they will be automatically computed from `returns`.
  - `ret`: set the `returns` matrix directly.
  - `timestamps`: set `timestamps`.
  - `assets`: set `assets`.
  - `f_prices`: `(T+1)×Nf` [`TimeArray`](https://juliastats.org/TimeSeries.jl/stable/timearray/#The-TimeArray-time-series-type) of factor prices.

      + If provided: will take precedence over `f_returns`, `f_ret`, `f_timestamps`, and `f_assets` because they will be automatically computed from `f_prices`.
  - `f_returns`: `T×Nf` [`DataFrame`](https://dataframes.juliadata.org/stable/lib/types/#DataFrames.DataFrame) of factor returns.

      + If provided: will take precedence over `f_ret`, `f_timestamps`, and `f_assets` because they will be automatically computed from `returns`.
  - `f_ret`: set the `f_returns` matrix directly.

The rest of the inputs directly set their corresponding property.

# Outputs

  - `portfolio`: an instance of [`Portfolio`](@ref).
"""
function Portfolio(; prices::TimeArray = TimeArray(TimeType[], []),
                   ret_type::Symbol = :simple, returns::DataFrame = DataFrame(),
                   ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   timestamps::AbstractVector = Vector{Date}(undef, 0),
                   assets::AbstractVector = Vector{String}(undef, 0),
                   f_prices::TimeArray = TimeArray(TimeType[], []),
                   f_returns::DataFrame = DataFrame(),
                   f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   f_timestamps::AbstractVector = Vector{Date}(undef, 0),
                   f_assets::AbstractVector = Vector{String}(undef, 0),
                   loadings::DataFrame = DataFrame(),
                   regression_type::Union{<:RegressionType, Nothing} = nothing,
                   short::Bool = false, budget::Real = 1.0, short_budget::Real = 0.2,
                   long_u::Real = 1.0, short_u::Real = 0.2,
                   fees::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                   short_fees::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                   num_assets_l::Integer = 0, num_assets_u::Integer = 0,
                   num_assets_u_scale::Real = 100_000.0, max_num_assets_kurt::Integer = 0,
                   max_num_assets_kurt_scale::Integer = 2, l1::Real = 0.0, l2::Real = 0.0,
                   rebalance::AbstractTR = NoTR(), turnover::AbstractTR = NoTR(),
                   tracking_err::TrackingErr = NoTracking(),
                   bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   a_mtx_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   b_vec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   f_risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   network_adj::AdjacencyConstraint = NoAdj(),
                   cluster_adj::AdjacencyConstraint = NoAdj(),
                   a_vec_cent::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   b_cent::Real = 0.0, mu_l::Real = Inf,
                   mu::AbstractVector = Vector{Float64}(undef, 0),
                   cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   L_2::AbstractMatrix = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
                   S_2::AbstractMatrix = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
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
                   limits::AbstractDict = Dict(), frontier::AbstractDict = Dict(),
                   solvers::AbstractDict = Dict(), fail::AbstractDict = Dict(),
                   model::JuMP.Model = JuMP.Model(),
                   latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   alloc_optimal::AbstractDict = Dict(),
                   alloc_leftover::AbstractDict = Dict(),
                   alloc_solvers::AbstractDict = Dict(), alloc_fail::AbstractDict = Dict(),
                   alloc_model::JuMP.Model = JuMP.Model())
    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices, ret_type)))
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
    if short
        @smart_assert(short_budget >=
                      short_u >=
                      zero(promote_type(typeof(short_budget), typeof(short_u))))

        @smart_assert(budget + short_budget >=
                      long_u >=
                      zero(promote_type(typeof(budget), typeof(short_budget),
                                        typeof(long_u))))
    else
        @smart_assert(budget >=
                      long_u >=
                      zero(promote_type(typeof(budget), typeof(long_u))))
    end
    if !isempty(f_prices)
        f_returns = dropmissing!(DataFrame(percentchange(f_prices, ret_type)))
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
    @smart_assert(max_num_assets_kurt >= zero(max_num_assets_kurt))
    max_num_assets_kurt_scale = clamp(max_num_assets_kurt_scale, 1, size(returns, 2))
    if isa(fees, AbstractVector) && !isempty(fees)
        @smart_assert(length(fees) == size(returns, 2))
    end
    if isa(short_fees, AbstractVector) && !isempty(short_fees)
        @smart_assert(length(short_fees) == size(returns, 2))
    end
    if isa(rebalance, TR)
        if isa(rebalance.val, Real)
            @smart_assert(rebalance.val >= zero(rebalance.val))
        elseif isa(rebalance.val, AbstractVector) && !isempty(rebalance.val)
            @smart_assert(length(rebalance.val) == size(returns, 2) &&
                          all(rebalance.val .>= zero(rebalance.val)))
        end
        if !isempty(rebalance.w)
            @smart_assert(length(rebalance.w) == size(returns, 2))
        end
    end
    if isa(turnover, TR)
        if isa(turnover.val, Real)
            @smart_assert(turnover.val >= zero(turnover.val))
        elseif isa(turnover.val, AbstractVector) && !isempty(turnover.val)
            @smart_assert(length(turnover.val) == size(returns, 2) &&
                          all(turnover.val .>= zero(turnover.val)))
        end
        if !isempty(turnover.w)
            @smart_assert(length(turnover.w) == size(returns, 2))
        end
    end
    if isa(tracking_err, TrackWeight)
        @smart_assert(length(tracking_err.w) == size(returns, 2))
        @smart_assert(length(tracking_err.err) >= zero(tracking_err.err))
    end
    if isa(tracking_err, TrackRet)
        @smart_assert(length(tracking_err.w) == size(returns, 1))
        @smart_assert(length(tracking_err.err) >= zero(tracking_err.err))
    end
    if !isempty(bl_bench_weights)
        @smart_assert(length(bl_bench_weights) == size(returns, 2))
    end
    if !isa(network_adj, NoAdj) && !isempty(network_adj.A)
        if isa(network_adj, IP)
            @smart_assert(size(network_adj.A, 2) == size(returns, 2))
        else
            @smart_assert(size(network_adj.A) == (size(returns, 2), size(returns, 2)))
        end
    end
    if !isa(cluster_adj, NoAdj) && !isempty(cluster_adj.A)
        if isa(cluster_adj, IP)
            @smart_assert(size(cluster_adj.A, 2) == size(returns, 2))
        else
            @smart_assert(size(cluster_adj.A) == (size(returns, 2), size(returns, 2)))
        end
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
    if !isempty(L_2)
        N = size(returns, 2)
        @smart_assert(size(L_2) == (Int(N * (N + 1) / 2), N^2))
    end
    if !isempty(S_2)
        N = size(returns, 2)
        @smart_assert(size(S_2) == (Int(N * (N + 1) / 2), N^2))
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
    if !isempty(fm_returns)
        @smart_assert(size(fm_returns) == size(returns))
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

    return Portfolio{typeof(assets), typeof(timestamps), typeof(returns), typeof(f_assets),
                     typeof(f_timestamps), typeof(f_returns), typeof(loadings),
                     Union{<:RegressionType, Nothing}, typeof(short), typeof(budget),
                     typeof(short_budget), typeof(long_u), typeof(short_u),
                     Union{<:Real, <:AbstractVector{<:Real}},
                     Union{<:Real, <:AbstractVector{<:Real}}, typeof(num_assets_l),
                     typeof(num_assets_u), typeof(num_assets_u_scale),
                     typeof(max_num_assets_kurt), typeof(max_num_assets_kurt_scale),
                     typeof(l1), typeof(l2), AbstractTR, AbstractTR, TrackingErr,
                     typeof(bl_bench_weights), typeof(a_mtx_ineq), typeof(b_vec_ineq),
                     typeof(risk_budget), typeof(f_risk_budget), AdjacencyConstraint,
                     AdjacencyConstraint, typeof(a_vec_cent), typeof(b_cent), typeof(mu_l),
                     typeof(mu), typeof(cov), typeof(kurt), typeof(skurt), typeof(L_2),
                     typeof(S_2), typeof(skew), typeof(V), typeof(sskew), typeof(SV),
                     typeof(f_mu), typeof(f_cov), typeof(fm_returns), typeof(fm_mu),
                     typeof(fm_cov), typeof(bl_mu), typeof(bl_cov), typeof(blfm_mu),
                     typeof(blfm_cov), typeof(cov_l), typeof(cov_u), typeof(cov_mu),
                     typeof(cov_sigma), typeof(d_mu), typeof(k_mu), typeof(k_sigma),
                     typeof(optimal), typeof(limits), typeof(frontier), typeof(solvers),
                     typeof(fail), typeof(model), typeof(latest_prices),
                     typeof(alloc_optimal), typeof(alloc_leftover), typeof(alloc_solvers),
                     typeof(alloc_fail), typeof(alloc_model)}(assets, timestamps, returns,
                                                              f_assets, f_timestamps,
                                                              f_returns, loadings,
                                                              regression_type, short,
                                                              budget, short_budget, long_u,
                                                              short_u, fees, short_fees,
                                                              num_assets_l, num_assets_u,
                                                              num_assets_u_scale,
                                                              max_num_assets_kurt,
                                                              max_num_assets_kurt_scale, l1,
                                                              l2, rebalance, turnover,
                                                              tracking_err,
                                                              bl_bench_weights, a_mtx_ineq,
                                                              b_vec_ineq, risk_budget,
                                                              f_risk_budget, network_adj,
                                                              cluster_adj, a_vec_cent,
                                                              b_cent, mu_l, mu, cov, kurt,
                                                              skurt, L_2, S_2, skew, V,
                                                              sskew, SV, f_mu, f_cov,
                                                              fm_returns, fm_mu, fm_cov,
                                                              bl_mu, bl_cov, blfm_mu,
                                                              blfm_cov, cov_l, cov_u,
                                                              cov_mu, cov_sigma, d_mu, k_mu,
                                                              k_sigma, optimal, limits,
                                                              frontier, solvers, fail,
                                                              model, latest_prices,
                                                              alloc_optimal, alloc_leftover,
                                                              alloc_solvers, alloc_fail,
                                                              alloc_model)
end
function Base.setproperty!(obj::Portfolio, sym::Symbol, val)
    if sym == :short_u
        if obj.short
            @smart_assert(obj.short_budget >=
                          val >=
                          zero(promote_type(typeof(obj.short_budget), typeof(val))))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :short_budget
        if obj.short
            @smart_assert(val >=
                          obj.short_u >=
                          zero(promote_type(typeof(val), typeof(obj.short_u))))

            @smart_assert(obj.budget + val >=
                          obj.long_u >=
                          zero(promote_type(typeof(obj.budget), typeof(val),
                                            typeof(obj.long_u))))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :long_u
        if obj.short
            @smart_assert(obj.budget + obj.short_budget >=
                          val >=
                          zero(promote_type(typeof(obj.budget), typeof(obj.short_budget),
                                            typeof(val))))
        else
            @smart_assert(obj.budget >=
                          val >=
                          zero(promote_type(typeof(obj.budget), typeof(val))))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :budget
        if obj.short
            @smart_assert(val + obj.short_budget >=
                          obj.long_u >=
                          zero(promote_type(typeof(val), typeof(obj.short_budget),
                                            typeof(obj.long_u))))
        else
            @smart_assert(val >=
                          obj.long_u >=
                          zero(promote_type(typeof(val), typeof(obj.long_u))))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :short
        if val
            @smart_assert(obj.short_budget >=
                          obj.short_u >=
                          zero(promote_type(typeof(obj.short_budget), typeof(obj.short_u))))

            @smart_assert(obj.budget + obj.short_budget >=
                          obj.long_u >=
                          zero(promote_type(typeof(obj.budget), typeof(obj.short_budget),
                                            typeof(obj.long_u))))
        else
            @smart_assert(obj.budget >=
                          obj.long_u >=
                          zero(promote_type(typeof(obj.budget), typeof(obj.long_u))))
        end
    elseif sym == :max_num_assets_kurt
        @smart_assert(val >= zero(val))
    elseif sym == :max_num_assets_kurt_scale
        val = clamp(val, 1, size(obj.returns, 2))
    elseif sym ∈ (:fees, :short_fees)
        if isa(val, AbstractVector) && !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
            val = collect(eltype(obj.returns), val)
        end
    elseif sym ∈ (:rebalance, :turnover)
        if isa(val, TR)
            if isa(val.val, Real)
                @smart_assert(val.val >= zero(val.val))
            elseif isa(val.val, AbstractVector) && !isempty(val.val)
                @smart_assert(length(val.val) == size(obj.returns, 2) &&
                              all(val.val .>= zero(val.val)))
            end
            if !isempty(val.w)
                @smart_assert(length(val.w) == size(obj.returns, 2))
            end
        end
    elseif sym == :tracking_err
        if isa(val, TrackWeight)
            @smart_assert(length(val.w) == size(obj.returns, 2))
            @smart_assert(val.err >= zero(val.err))
        elseif isa(val, TrackRet)
            @smart_assert(length(val.w) == size(obj.returns, 1))
            @smart_assert(val.err >= zero(val.err))
        end
    elseif sym == :a_mtx_ineq
        if !isempty(val)
            @smart_assert(size(val, 2) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:network_adj, :cluster_adj)
        if !isa(val, NoAdj) && !isempty(val.A)
            if isa(val, IP)
                @smart_assert(size(val.A, 2) == size(obj.returns, 2))
            else
                @smart_assert(size(val.A) == (size(obj.returns, 2), size(obj.returns, 2)))
            end
        end
    elseif sym == :a_vec_cent
        if !isempty(val)
            @smart_assert(size(val, 1) == size(obj.returns, 2))
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
            val = fill(inv(N), N)
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
    elseif sym ∈ (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns,
                  :fm_returns)
        if !isempty(val) && !isempty(getfield(obj, sym))
            @smart_assert(size(val) == size(getfield(obj, sym)))
        end
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
                     typeof(obj.f_assets), typeof(obj.f_timestamps), typeof(obj.f_returns),
                     typeof(obj.loadings), Union{<:RegressionType, Nothing},
                     typeof(obj.short), typeof(obj.budget), typeof(obj.short_budget),
                     typeof(obj.long_u), typeof(obj.short_u),
                     Union{<:Real, <:AbstractVector{<:Real}},
                     Union{<:Real, <:AbstractVector{<:Real}}, typeof(obj.num_assets_l),
                     typeof(obj.num_assets_u), typeof(obj.num_assets_u_scale),
                     typeof(obj.max_num_assets_kurt), typeof(obj.max_num_assets_kurt_scale),
                     typeof(obj.l1), typeof(obj.l2), AbstractTR, AbstractTR, TrackingErr,
                     typeof(obj.bl_bench_weights), typeof(obj.a_mtx_ineq),
                     typeof(obj.b_vec_ineq), typeof(obj.risk_budget),
                     typeof(obj.f_risk_budget), AdjacencyConstraint, AdjacencyConstraint,
                     typeof(obj.a_vec_cent), typeof(obj.b_cent), typeof(obj.mu_l),
                     typeof(obj.mu), typeof(obj.cov), typeof(obj.kurt), typeof(obj.skurt),
                     typeof(obj.L_2), typeof(obj.S_2), typeof(obj.skew), typeof(obj.V),
                     typeof(obj.sskew), typeof(obj.SV), typeof(obj.f_mu), typeof(obj.f_cov),
                     typeof(obj.fm_returns), typeof(obj.fm_mu), typeof(obj.fm_cov),
                     typeof(obj.bl_mu), typeof(obj.bl_cov), typeof(obj.blfm_mu),
                     typeof(obj.blfm_cov), typeof(obj.cov_l), typeof(obj.cov_u),
                     typeof(obj.cov_mu), typeof(obj.cov_sigma), typeof(obj.d_mu),
                     typeof(obj.k_mu), typeof(obj.k_sigma), typeof(obj.optimal),
                     typeof(obj.limits), typeof(obj.frontier), typeof(obj.solvers),
                     typeof(obj.fail), typeof(obj.model), typeof(obj.latest_prices),
                     typeof(obj.alloc_optimal), typeof(obj.alloc_leftover),
                     typeof(obj.alloc_solvers), typeof(obj.alloc_fail),
                     typeof(obj.alloc_model)}(deepcopy(obj.assets),
                                              deepcopy(obj.timestamps),
                                              deepcopy(obj.returns), deepcopy(obj.f_assets),
                                              deepcopy(obj.f_timestamps),
                                              deepcopy(obj.f_returns),
                                              deepcopy(obj.loadings),
                                              deepcopy(obj.regression_type),
                                              deepcopy(obj.short), deepcopy(obj.budget),
                                              deepcopy(obj.short_budget),
                                              deepcopy(obj.long_u), deepcopy(obj.short_u),
                                              deepcopy(obj.fees), deepcopy(obj.short_fees),
                                              deepcopy(obj.num_assets_l),
                                              deepcopy(obj.num_assets_u),
                                              deepcopy(obj.num_assets_u_scale),
                                              deepcopy(obj.max_num_assets_kurt),
                                              deepcopy(obj.max_num_assets_kurt_scale),
                                              deepcopy(obj.l1), deepcopy(obj.l2),
                                              deepcopy(obj.rebalance),
                                              deepcopy(obj.turnover),
                                              deepcopy(obj.tracking_err),
                                              deepcopy(obj.bl_bench_weights),
                                              deepcopy(obj.a_mtx_ineq),
                                              deepcopy(obj.b_vec_ineq),
                                              deepcopy(obj.risk_budget),
                                              deepcopy(obj.f_risk_budget),
                                              deepcopy(obj.network_adj),
                                              deepcopy(obj.cluster_adj),
                                              deepcopy(obj.a_vec_cent),
                                              deepcopy(obj.b_cent), deepcopy(obj.mu_l),
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
                                              deepcopy(obj.optimal), deepcopy(obj.limits),
                                              deepcopy(obj.frontier), deepcopy(obj.solvers),
                                              deepcopy(obj.fail), copy(obj.model),
                                              deepcopy(obj.latest_prices),
                                              deepcopy(obj.alloc_optimal),
                                              deepcopy(obj.alloc_leftover),
                                              deepcopy(obj.alloc_solvers),
                                              deepcopy(obj.alloc_fail),
                                              copy(obj.alloc_model))
end

"""
```
mutable struct HCPortfolio{ast, dat, r, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew, tv,
                           tsskew, tsv, wmi, wma, tco, tdist, tcl, tk, topt, tsolv, tf, tlp,
                           taopt, talo, tasolv, taf, tamod} <: AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
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
    w_min::wmi
    w_max::wma
    cor::tco
    dist::tdist
    clusters::tcl
    k::tk
    optimal::topt
    solvers::tsolv
    fail::tf
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_fail::taf
    alloc_model::tamod
end
```

Structure for defining a hierarchical clustering portfolio.

# Parameters

  - `assets`: `N×1` vector of asset names.

  - `timestamps`: `T×1` vector of asset returns timestamps.
  - `returns`: `T×N` matrix of asset returns.
  - `mu`: `N×1` vector of asset expected returns.
  - `cov`: `N×N` asset covariance matrix.
  - `kurt`: `N^2×N^2` cokurtosis matrix.
  - `skurt`: `N^2×N^2` semi cokurtosis matrix.
  - `L_2`: `(N^2)×((N^2 + N)/2)` elimination matrix.
  - `S_2`: `((N^2 + N)/2)×(N^2)` summation matrix.
  - `skew`: `N×N^2` coskew matrix.
  - `V`: `N×N` sum of the symmetric negative spectral slices of coskewness.
  - `sskew`: `N×N^2` semi coskew matrix.
  - `SV`: `N×N` sum of the symmetric negative spectral slices of semi coskewness.
  - `w_min`: minimum allowable asset weights.

      + if isa vector: `N×1` vector of minimum allowable weight per asset.
      + if isa scalar: minimum asset weight for all assets.
  - `w_max`: maximum allowable asset weights.

      + if isa vector: `N×1` vector of maximum allowable weight per asset.
      + if isa scalar: maximum asset weight for all assets.
  - `cor`: `N×N` asset correlation matrix.
  - `dist`: `N×N` asset distance matrix.
  - `clusters`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of asset clusters.
  - `k`: number of asset clusters.

      + if is zero: compute the number of clusters via one of the cluster number methods [`NumClusterMethod`](@ref).
      + if is not zero: use this value directly.
  - `optimal`: collection capable of storing key value pairs for storing optimal portfolios.
  - `solvers`: collection capable of storing key value pairs for storing `JuMP`-supported solvers. They must have the following structure.

    ```
    solvers = Dict(
                   # Key-value pair for the solver, solution acceptance 
                   # criteria, and solver attributes.
                   :Clarabel => Dict(
                                     # Solver we wish to use.
                                     :solver => Clarabel.Optimizer,
                                     # (Optional) Solution acceptance criteria.
                                     :check_sol => (allow_local = true, allow_almost = true),
                                     # (Optional) Solver-specific attributes.
                                     :params => Dict("verbose" => false)))
    ```

    The dictionary contains a key value pair for each solver (plus optional solution acceptance criteria and optional attributes) we want to use.

      + `:solver`: defines the solver to use. One can also use [`JuMP.optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to direcly provide a solver with attributes already attached.
      + `:check_sol`: (optional) defines the keyword arguments passed on to [`JuMP.is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#is_solved_and_feasible) for accepting/rejecting solutions.
      + `:params`: (optional) defines solver-specific parameters.

    Users are also able to provide multiple solvers by adding additional key-value pairs to the top-level dictionary as in the following snippet.

    ```
    using JuMP
    solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                     :check_sol => (allow_local = true, allow_almost = true),
                                     :params => Dict("verbose" => false)),
                   # Provide solver with pre-attached attributes and no arguments 
                   # for the `JuMP.is_solved_and_feasible` function.
                   :COSMO => Dict(:solver => JuMP.optimizer_with_attributes(COSMO.Optimizer,
                                                                            "maxiter" => 5000)))
    ```

    [`optimise!`](@ref) will iterate over the solvers until it finds the first one to successfully solve the problem.
  - `fail`: collection capable of storing key value pairs for storing failed optimisation attempts.
  - `latest_prices`: `Na×1` vector of latest asset prices.
  - `alloc_optimal`: collection capable of storing key value pairs for storing optimal discretely allocated portfolios.
  - `alloc_leftover`: collection capable of storing key value pairs for containing points in the leftover investment after allocating.
  - `alloc_solvers`: collection capable of storing key value pairs for storing `JuMP`-supported solvers that support Mixed-Integer Programming, only used in the [`LP`](@ref) allocation.
  - `alloc_fail`: collection capable of storing key value pairs for storing failed discrete asset allocation attempts.
  - `alloc_model`: [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#Model) which defines the discrete asset allocation model.
"""
mutable struct HCPortfolio{ast, dat, r, tfa, tfdat, tretf, l, lo, blbw, tmu, tcov, tkurt,
                           tskurt, tl2, ts2, tskew, tv, tsskew, tsv, tmuf, tcovf, trfm,
                           tmufm, tcovfm, tmubl, tcovbl, tmublf, tcovblf, wmi, wma, tco,
                           tdist, tcl, tk, topt, tsolv, tf, tlp, taopt, talo, tasolv, taf,
                           tamod} <: AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    regression_type::lo
    bl_bench_weights::blbw
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
    w_min::wmi
    w_max::wma
    cor::tco
    dist::tdist
    clusters::tcl
    k::tk
    optimal::topt
    solvers::tsolv
    fail::tf
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_fail::taf
    alloc_model::tamod
end
"""
```
HCPortfolio(; prices::TimeArray = TimeArray(TimeType[], []),
              returns::DataFrame = DataFrame(),
              ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
              timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
              assets::AbstractVector = Vector{String}(undef, 0),
              mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
              cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
              kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
              skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
              skew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
              V = Matrix{eltype(returns)}(undef, 0, 0),
              sskew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
              SV = Matrix{eltype(returns)}(undef, 0, 0),
              w_min::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              w_max::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
              cor::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
              dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
              clusters::Clustering.Hclust = Hclust{Float64}(Matrix{Int64}(undef, 0, 2),
                                                          Float64[], Int64[],
                                                          :nothing),
              k::Integer = 0, optimal::AbstractDict = Dict(),
              solvers::AbstractDict = Dict(), fail::AbstractDict = Dict(),
              latest_prices::AbstractVector = Vector{Float64}(undef, 0),
              alloc_optimal::AbstractDict = Dict(),
              alloc_leftover::AbstractDict = Dict(),
              alloc_solvers::AbstractDict = Dict(),
              alloc_fail::AbstractDict = Dict(),
              alloc_model::JuMP.Model = JuMP.Model())
```

Constructor for [`HCPortfolio`](@ref). Performs data validation checks and automatically extracts the data from `prices`, `returns`, `f_prices`, and `f_returns` if they are provided.

# Inputs

  - `prices`: `(T+1)×Na` [`TimeArray`](https://juliastats.org/TimeSeries.jl/stable/timearray/#The-TimeArray-time-series-type) of asset prices.

      + If provided: will take precedence over `returns`, `ret`, `timestamps`, `assets`, and `latest_prices` because they will be automatically computed from `prices`.

  - `returns`: `T×Na` [`DataFrame`](https://dataframes.juliadata.org/stable/lib/types/#DataFrames.DataFrame) of asset returns.

      + If provided: will take precedence over `ret`, `timestamps`, and `assets` because they will be automatically computed from `returns`.
  - `ret`: set the `returns` matrix directly.

The rest of the inputs directly set their corresponding property.

# Outputs

  - `portfolio`: an instance of [`HCPortfolio`](@ref).
"""
function HCPortfolio(; prices::TimeArray = TimeArray(TimeType[], []),
                     ret_type::Symbol = :simple, returns::DataFrame = DataFrame(),
                     ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     timestamps::AbstractVector{<:Dates.AbstractTime} = Vector{Date}(undef,
                                                                                     0),
                     assets::AbstractVector = Vector{String}(undef, 0),
                     f_prices::TimeArray = TimeArray(TimeType[], []),
                     f_returns::DataFrame = DataFrame(),
                     f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     f_timestamps::AbstractVector = Vector{Date}(undef, 0),
                     f_assets::AbstractVector = Vector{String}(undef, 0),
                     loadings::DataFrame = DataFrame(),
                     regression_type::Union{<:RegressionType, Nothing} = nothing,
                     bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                     mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                     cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     L_2::AbstractMatrix = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
                     S_2::AbstractMatrix = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
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
                     w_min::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                     w_max::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                     cor::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     clusters::Hclust = Hclust{Float64}(Matrix{Int64}(undef, 0, 2),
                                                        Float64[], Int64[], :nothing),
                     k::Integer = 0, optimal::AbstractDict = Dict(),
                     solvers::AbstractDict = Dict(), fail::AbstractDict = Dict(),
                     latest_prices::AbstractVector = Vector{Float64}(undef, 0),
                     alloc_optimal::AbstractDict = Dict(),
                     alloc_leftover::AbstractDict = Dict(),
                     alloc_solvers::AbstractDict = Dict(),
                     alloc_fail::AbstractDict = Dict(),
                     alloc_model::JuMP.Model = JuMP.Model())
    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices, ret_type)))
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
        f_returns = dropmissing!(DataFrame(percentchange(f_prices, ret_type)))
    end
    if !isempty(f_returns)
        f_assets = setdiff(names(f_returns), ("timestamp",))
        f_timestamps = f_returns[!, "timestamp"]
        f_returns = Matrix(f_returns[!, f_assets])
    else
        @smart_assert(length(f_assets) == size(f_ret, 2))
        f_returns = f_ret
    end
    if !isempty(bl_bench_weights)
        @smart_assert(length(bl_bench_weights) == size(returns, 2))
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
    if !isempty(L_2)
        N = size(returns, 2)
        @smart_assert(size(L_2) == (Int(N * (N + 1) / 2), N^2))
    end
    if !isempty(S_2)
        N = size(returns, 2)
        @smart_assert(size(S_2) == (Int(N * (N + 1) / 2), N^2))
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
    if !isempty(fm_returns)
        @smart_assert(size(fm_returns) == size(returns))
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
    if isa(w_min, Real)
        if isa(w_max, Real)
            @smart_assert(w_min <= w_max)
        elseif !isempty(w_max)
            @smart_assert(all(w_min .<= w_max))
        end
    elseif isa(w_min, AbstractVector)
        if !isempty(w_min)
            @smart_assert(length(w_min) == size(returns, 2))
            if isa(w_max, Real) || !isempty(w_max)
                @smart_assert(all(w_min .<= w_max))
            end
        end
    end
    if isa(w_max, Real)
        if isa(w_min, Real)
            @smart_assert(w_max >= w_min)
        elseif !isempty(w_min)
            @smart_assert(all(w_max .>= w_min))
        end
    elseif isa(w_max, AbstractVector)
        if !isempty(w_max)
            @smart_assert(length(w_max) == size(returns, 2))
            if isa(w_min, Real) || !isempty(w_min)
                @smart_assert(all(w_max .>= w_min))
            end
        end
    end
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

    return HCPortfolio{typeof(assets), typeof(timestamps), typeof(returns),
                       typeof(f_assets), typeof(f_timestamps), typeof(f_returns),
                       typeof(loadings), Union{<:RegressionType, Nothing},
                       typeof(bl_bench_weights), typeof(mu), typeof(cov), typeof(kurt),
                       typeof(skurt), typeof(L_2), typeof(S_2), typeof(skew), typeof(V),
                       typeof(sskew), typeof(SV), typeof(f_mu), typeof(f_cov),
                       typeof(fm_returns), typeof(fm_mu), typeof(fm_cov), typeof(bl_mu),
                       typeof(bl_cov), typeof(blfm_mu), typeof(blfm_cov),
                       Union{<:Real, <:AbstractVector{<:Real}},
                       Union{<:Real, <:AbstractVector{<:Real}}, typeof(cor), typeof(dist),
                       typeof(clusters), typeof(k), typeof(optimal), typeof(solvers),
                       typeof(fail), typeof(latest_prices), typeof(alloc_optimal),
                       typeof(alloc_leftover), typeof(alloc_solvers), typeof(alloc_fail),
                       typeof(alloc_model)}(assets, timestamps, returns, f_assets,
                                            f_timestamps, f_returns, loadings,
                                            regression_type, bl_bench_weights, mu, cov,
                                            kurt, skurt, L_2, S_2, skew, V, sskew, SV, f_mu,
                                            f_cov, fm_returns, fm_mu, fm_cov, bl_mu, bl_cov,
                                            blfm_mu, blfm_cov, w_min, w_max, cor, dist,
                                            clusters, k, optimal, solvers, fail,
                                            latest_prices, alloc_optimal, alloc_leftover,
                                            alloc_solvers, alloc_fail, alloc_model)
end
function Base.setproperty!(obj::HCPortfolio, sym::Symbol, val)
    if sym == :k
        @smart_assert(val >= zero(val))
    elseif sym == :w_min
        if isa(val, Real)
            if isa(obj.w_max, Real)
                @smart_assert(val <= obj.w_max)
            elseif !isempty(obj.w_max)
                @smart_assert(all(val .<= obj.w_max))
            end
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(obj.returns, 2))
                if isa(obj.w_max, Real) || !isempty(obj.w_max)
                    @smart_assert(all(val .<= obj.w_max))
                end
            end
        end
    elseif sym == :w_max
        if isa(val, Real)
            if isa(obj.w_min, Real)
                @smart_assert(val >= obj.w_min)
            elseif !isempty(obj.w_min)
                @smart_assert(all(val .>= obj.w_min))
            end
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(obj.returns, 2))
                if isa(obj.w_min, Real) || !isempty(obj.w_min)
                    @smart_assert(all(val .>= obj.w_min))
                end
            end
        end
    elseif sym ∈ (:mu, :fm_mu, :bl_mu, :blfm_mu, :latest_prices)
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
    elseif sym ∈ (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns,
                  :fm_returns)
        if !isempty(val) && !isempty(getfield(obj, sym))
            @smart_assert(size(val) == size(getfield(obj, sym)))
        end
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
    elseif sym == :bl_bench_weights
        if isempty(val)
            N = size(obj.returns, 2)
            val = fill(inv(N), N)
        else
            @smart_assert(length(val) == size(obj.returns, 2))
            if isa(val, AbstractRange)
                val = collect(val / sum(val))
            else
                val ./= sum(val)
            end
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:cov, :cor, :dist, :V, :SV)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    end
    return setfield!(obj, sym, val)
end
function Base.deepcopy(obj::HCPortfolio)
    return HCPortfolio{typeof(obj.assets), typeof(obj.timestamps), typeof(obj.returns),
                       typeof(obj.f_assets), typeof(obj.f_timestamps),
                       typeof(obj.f_returns), typeof(obj.loadings),
                       Union{<:RegressionType, Nothing}, typeof(obj.bl_bench_weights),
                       typeof(obj.mu), typeof(obj.cov), typeof(obj.kurt), typeof(obj.skurt),
                       typeof(obj.L_2), typeof(obj.S_2), typeof(obj.skew), typeof(obj.V),
                       typeof(obj.sskew), typeof(obj.SV), typeof(obj.f_mu),
                       typeof(obj.f_cov), typeof(obj.fm_returns), typeof(obj.fm_mu),
                       typeof(obj.fm_cov), typeof(obj.bl_mu), typeof(obj.bl_cov),
                       typeof(obj.blfm_mu), typeof(obj.blfm_cov),
                       Union{<:Real, <:AbstractVector{<:Real}},
                       Union{<:Real, <:AbstractVector{<:Real}}, typeof(obj.cor),
                       typeof(obj.dist), typeof(obj.clusters), typeof(obj.k),
                       typeof(obj.optimal), typeof(obj.solvers), typeof(obj.fail),
                       typeof(obj.latest_prices), typeof(obj.alloc_optimal),
                       typeof(obj.alloc_leftover), typeof(obj.alloc_solvers),
                       typeof(obj.alloc_fail), typeof(obj.alloc_model)}(deepcopy(obj.assets),
                                                                        deepcopy(obj.timestamps),
                                                                        deepcopy(obj.returns),
                                                                        deepcopy(obj.f_assets),
                                                                        deepcopy(obj.f_timestamps),
                                                                        deepcopy(obj.f_returns),
                                                                        deepcopy(obj.loadings),
                                                                        deepcopy(obj.regression_type),
                                                                        deepcopy(obj.bl_bench_weights),
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
                                                                        deepcopy(obj.f_mu),
                                                                        deepcopy(obj.f_cov),
                                                                        deepcopy(obj.fm_returns),
                                                                        deepcopy(obj.fm_mu),
                                                                        deepcopy(obj.fm_cov),
                                                                        deepcopy(obj.bl_mu),
                                                                        deepcopy(obj.bl_cov),
                                                                        deepcopy(obj.blfm_mu),
                                                                        deepcopy(obj.blfm_cov),
                                                                        deepcopy(obj.w_min),
                                                                        deepcopy(obj.w_max),
                                                                        deepcopy(obj.cor),
                                                                        deepcopy(obj.dist),
                                                                        deepcopy(obj.clusters),
                                                                        deepcopy(obj.k),
                                                                        deepcopy(obj.optimal),
                                                                        deepcopy(obj.solvers),
                                                                        deepcopy(obj.fail),
                                                                        deepcopy(obj.latest_prices),
                                                                        deepcopy(obj.alloc_optimal),
                                                                        deepcopy(obj.alloc_leftover),
                                                                        deepcopy(obj.alloc_solvers),
                                                                        deepcopy(obj.alloc_fail),
                                                                        copy(obj.alloc_model))
end

export Portfolio, HCPortfolio
