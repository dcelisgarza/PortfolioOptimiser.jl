"""
```julia
mutable struct HCPortfolio{
    ast,
    dat,
    r,
    # Risk parmeters
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
    # Custom OWA weights
    wowa,
    # Optimisation parameters
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
    tbin,
    wmi,
    wma,
    ttco,
    tco,
    tdist,
    tcl,
    tk,
    # Optimal portfolios
    topt,
    # Solutions
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
    gs_threshold::gst
    max_num_assets_kurt::mnak
    # Custom OWA weights
    owa_w::wowa
    # Optimisation parameters
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
    bins_info::tbin
    w_min::wmi
    w_max::wma
    codep_type::ttco
    codep::tco
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
- `gs_threshold`: Gerber statistic threshold.
- `max_num_assets_kurt`: maximum number of assets to use the full kurtosis model, if the number of assets surpases this value use the relaxed kurtosis model.
# Custom OWA weights
- `owa_w`: `T×1` OWA vector, where $(_tstr(:t1)) containing. Useful for optimising higher OWA L-moments.
# Model statistics
- `mu_type`: method for estimating the mean returns vectors `mu`, `mu_fm`, `mu_bl`, `mu_bl_fm` in [`mean_vec`](@ref), see [`MuTypes`](@ref) for available choices.
- `mu`: $(_mudef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `cov_type`: method for estimating the covariance matrices `cov`, `cov_fm`, `cov_bl`, `cov_bl_fm` in [`covar_mtx`](@ref), see [`CovTypes`](@ref) for available choices.
- `jlogo`: if `true`, apply the j-LoGo transformation to the portfolio covariance matrix in [`covar_mtx`](@ref) [^jLoGo].
- `cov`: $(_covdef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `kurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the cokurtosis matrix at instance construction. The cokurtosis matrix `kurt` can be computed by calling [`cokurt_mtx`](@ref).
- `skurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the semi cokurtosis matrix at instance construction. The semi cokurtosis matrix `skurt` can be computed by calling [`cokurt_mtx`](@ref).
- `posdef_fix`: method for fixing non positive definite matrices when computing portfolio statistics, see [`PosdefFixes`](@ref) for available choices.
- `L_2`: `(Na×Na) × ((Na×(Na+1)/2))` elimination matrix, where $(_ndef(:a2)). $(_dircomp("[`cokurt_mtx`](@ref)"))
- `S_2`: `((Na×(Na+1)/2)) × (Na×Na)` summation matrix, where $(_ndef(:a2)). $(_dircomp("[`cokurt_mtx`](@ref)"))
- `bins_info`: selection criterion for computing the number of bins used to calculate the mutual and variation of information statistics, see [`mut_var_info_mtx`](@ref) for available choices.
- `w_min`: `Na×1` vector of the lower bounds for asset weights, where $(_ndef(:a2)).
- `w_max`: `Na×1` vector of the upper bounds for asset weights, where $(_ndef(:a2)).
- `codep_type`: method for estimating the codependence matrix.
- `codep`: `Na×Na` matrix, where where $(_ndef(:a2)). Set the value of the codependence matrix at instance construction. When choosing `:Custom_Val` in `cov_type`, this is the value of `codep` used by [`codep_dist_mtx`](@ref).
- `dist`:  `Na×Na` matrix, where where $(_ndef(:a2)). Set the value of the distance matrix at instance construction. When choosing `:Custom_Val` in `cov_type`, this is the value of `dist` used by [`codep_dist_mtx`](@ref).
- `clusters`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of asset clusters. $(_dircomp("[`asset_statistics!`](@ref) and [`opt_port!`](@ref)"))
- `k`: number of clusters to cut the dendrogram into.
    - If `k == 0`, automatically compute `k` using the two difference gap statistic [^TDGS]. $(_dircomp("[`asset_statistics!`](@ref) and [`opt_port!`](@ref)"))
    - If `k != 0`, use the value directly.
# Optimal portfolios
- `optimal`: $_edst for storing optimal portfolios. $(_filled_by("[`opt_port!`](@ref)"))- `optimal`:
# Solutions
$(_solver_desc("risk measure `JuMP` model for `:NCO` optimisations."))
- `opt_params`: $_edst for storing parameters used for optimising. $(_filled_by("[`opt_port!`](@ref)"))
- `fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`opt_port!`](@ref)"))
# Allocation
- `latest_prices`: `Na×1` vector of asset prices, $(_ndef(:a2)). If `prices` is not empty, this is automatically obtained from the last entry. This is used for discretely allocating stocks according to their prices, weight in the portfolio, and money to be invested.
- `alloc_optimal`: $_edst for storing optimal portfolios after allocating discrete stocks. $(_filled_by("[`allocate_port!`](@ref)"))
$(_solver_desc("discrete allocation `JuMP` model.", "alloc_", "mixed-integer problems"))
- `alloc_params`: $_edst for storing parameters used for optimising the portfolio allocation. $(_filled_by("[`allocate_port!`](@ref)"))
- `alloc_fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`allocate_port!`](@ref)"))
- `alloc_model`: `JuMP.Model()` for optimising a portfolio allocation. $(_filled_by("[`allocate_port!`](@ref)"))
"""
mutable struct HCPortfolio{
    ast,
    dat,
    r,
    # Risk parmeters
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
    # Custom OWA weights
    wowa,
    # Optimisation parameters
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
    tbin,
    wmi,
    wma,
    ttco,
    tco,
    tdist,
    tcl,
    tk,
    # Optimal portfolios
    topt,
    # Solutions
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
    gs_threshold::gst
    max_num_assets_kurt::mnak
    # Custom OWA weights
    owa_w::wowa
    # Optimisation parameters
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
    bins_info::tbin
    w_min::wmi
    w_max::wma
    codep_type::ttco
    codep::tco
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
HCPortfolio(;
    # Portfolio characteristics
    prices::TimeArray = TimeArray(TimeType[], []),
    returns::DataFrame = DataFrame(),
    ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
    assets::AbstractVector = Vector{String}(undef, 0),
    # Risk parmeters
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
    # Custom OWA weights
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    # Optimisation parameters
    mu_type::Symbol = :Default,
    mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    bins_info::Union{Symbol, <:Integer} = :KN,
    w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,
    w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,
    codep_type::Symbol = :Pearson,
    codep::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    clusters::Clustering.Hclust = Hclust{Float64}(
        Matrix{Int64}(undef, 0, 2),
        Float64[],
        Int64[],
        :nothing,
    ),
    k::Integer = 0,
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
    alloc_model::JuMP.Model = JuMP.Model(),
)
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
- `gs_threshold`: Gerber statistic threshold.
- `max_num_assets_kurt`: when optimising `:NCO` type of [`HCPortfolio`](@ref), maximum number of assets to use the full kurtosis model, if the number of assets surpases this value use the relaxed kurtosis model.
## Custom OWA weights
- `owa_w`: `T×1` OWA vector, where $(_tstr(:t1)) containing. Useful for optimising higher OWA L-moments.
## Model statistics
- `mu_type`: method for estimating the mean returns vectors `mu`, `mu_fm`, `mu_bl`, `mu_bl_fm` in [`mean_vec`](@ref), see [`MuTypes`](@ref) for available choices.
- `mu`: $(_mudef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `cov_type`: method for estimating the covariance matrices `cov`, `cov_fm`, `cov_bl`, `cov_bl_fm` in [`covar_mtx`](@ref), see [`CovTypes`](@ref) for available choices.
- `jlogo`: if `true`, apply the j-LoGo transformation to the portfolio covariance matrix in [`covar_mtx`](@ref) [^jLoGo].
- `cov`: $(_covdef("asset")) $(_dircomp("[`asset_statistics!`](@ref)"))
- `kurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the cokurtosis matrix at instance construction. The cokurtosis matrix `kurt` can be computed by calling [`cokurt_mtx`](@ref).
- `skurt`: `(Na×Na)×(Na×Na)` matrix, where $(_ndef(:a2)). Set the semi cokurtosis matrix at instance construction. The semi cokurtosis matrix `skurt` can be computed by calling [`cokurt_mtx`](@ref).
- `posdef_fix`: method for fixing non positive definite matrices when computing portfolio statistics, see [`PosdefFixes`](@ref) for available choices.
- `bins_info`: selection criterion for computing the number of bins used to calculate the mutual and variation of information statistics, see [`mut_var_info_mtx`](@ref) for available choices.
- `w_min`: `Na×1` vector of the lower bounds for asset weights, where $(_ndef(:a2)).
- `w_max`: `Na×1` vector of the upper bounds for asset weights, where $(_ndef(:a2)).
- `codep_type`: method for estimating the codependence matrix.
- `codep`: `Na×Na` matrix, where where $(_ndef(:a2)). Set the value of the codependence matrix at instance construction. When choosing `:Custom_Val` in `cov_type`, this is the value of `codep` used by [`codep_dist_mtx`](@ref).
- `dist`:  `Na×Na` matrix, where where $(_ndef(:a2)). Set the value of the distance matrix at instance construction. When choosing `:Custom_Val` in `cov_type`, this is the value of `dist` used by [`codep_dist_mtx`](@ref).
- `clusters`: [`Clustering.Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) of asset clusters. $(_dircomp("[`asset_statistics!`](@ref) and [`opt_port!`](@ref)"))
- `k`: number of clusters to cut the dendrogram into.
    - If `k == 0`, automatically compute `k` using the two difference gap statistic [^TDGS]. $(_dircomp("[`asset_statistics!`](@ref) and [`opt_port!`](@ref)"))
    - If `k != 0`, use the value directly.
## Optimal portfolios
- `optimal`: $_edst for storing optimal portfolios. $(_filled_by("[`opt_port!`](@ref)"))- `optimal`:
## Solutions
$(_solver_desc("risk measure `JuMP` model for `:NCO` optimisations."))
- `opt_params`: $_edst for storing parameters used for optimising. $(_filled_by("[`opt_port!`](@ref)"))
- `fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`opt_port!`](@ref)"))
## Allocation
- `latest_prices`: `Na×1` vector of asset prices, $(_ndef(:a2)). If `prices` is not empty, this is automatically obtained from the last entry. This is used for discretely allocating stocks according to their prices, weight in the portfolio, and money to be invested.
- `alloc_optimal`: $_edst for storing optimal portfolios after allocating discrete stocks. $(_filled_by("[`allocate_port!`](@ref)"))
$(_solver_desc("discrete allocation `JuMP` model.", "alloc_", "mixed-integer problems"))
- `alloc_params`: $_edst for storing parameters used for optimising the portfolio allocation. $(_filled_by("[`allocate_port!`](@ref)"))
- `alloc_fail`: $_edst for storing failed optimisation attempts. $(_filled_by("[`allocate_port!`](@ref)"))
- `alloc_model`: `JuMP.Model()` for optimising a portfolio allocation. $(_filled_by("[`allocate_port!`](@ref)"))
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
    # Custom OWA weights
    owa_w::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    # Optimisation parameters
    mu_type::Symbol = :Default,
    mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    bins_info::Union{Symbol, <:Integer} = :KN,
    w_min::Union{<:Real, AbstractVector{<:Real}} = 0.0,
    w_max::Union{<:Real, AbstractVector{<:Real}} = 1.0,
    codep_type::Symbol = :Pearson,
    codep::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    clusters::Clustering.Hclust = Hclust{Float64}(
        Matrix{Int64}(undef, 0, 2),
        Float64[],
        Int64[],
        :nothing,
    ),
    k::Integer = 0,
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
    alloc_model::JuMP.Model = JuMP.Model(),
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
        0 < alpha_tail < 1,
        "alpha_tail = $alpha_tail, must be greater than 0 and less than 1"
    )
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
    @assert(
        bins_info ∈ BinTypes || isa(bins_info, Int) && bins_info > zero(bins_info),
        "bins_info = $bins_info, has to either be in $BinTypes, or an integer value greater than 0"
    )
    if isa(w_min, Real)
        @assert(
            zero(w_min) <= w_min <= one(w_min) && all(w_min .<= w_max),
            "0 .<= w_min .<= w_max .<= 1: 0 .<= $w_min .<= $w_max .<= 1, must be true"
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
            "0 .<= w_min .<= w_max .<= 1: 0 .<= $w_min .<= $w_max .<= 1, must be true"
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

    L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)
    S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)

    return HCPortfolio{
        typeof(assets),
        typeof(timestamps),
        typeof(returns),
        # Risk parmeters
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
        # Custom OWA weights
        typeof(owa_w),
        # Optimisation parameters
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
        Union{Symbol, <:Integer},
        Union{<:Real, AbstractVector{<:Real}},
        Union{<:Real, AbstractVector{<:Real}},
        typeof(codep_type),
        typeof(codep),
        typeof(dist),
        typeof(clusters),
        typeof(k),
        # Optimal portfolios
        typeof(optimal),
        # Solutions
        Union{<:AbstractDict, NamedTuple},
        Union{<:AbstractDict, NamedTuple},
        typeof(fail),
        # Allocation
        typeof(latest_prices),
        typeof(alloc_optimal),
        Union{<:AbstractDict, NamedTuple},
        Union{<:AbstractDict, NamedTuple},
        typeof(alloc_fail),
        typeof(alloc_model),
    }(
        assets,
        timestamps,
        returns,
        # Risk parmeters
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
        # Custom OWA weights
        owa_w,
        # Optimisation parameters
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
        bins_info,
        w_min,
        w_max,
        codep_type,
        codep,
        dist,
        clusters,
        k,
        # Optimal portfolios
        optimal,
        # Solutions
        solvers,
        opt_params,
        fail,
        # Allocation
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
    elseif sym == :alpha_tail
        @assert(0 < val < 1, "alpha_tail = $val, must be greater than 0 and less than 1")
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
                "0 .<= w_min .<= w_max .<= 1: 0 .<= $vmin .<= $vmax .<= 1, must be true"
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
    elseif sym ∈ (:kurt, :skurt)
        if !isempty(val)
            @assert(
                size(val, 1) == size(val, 2) == size(obj.returns, 2)^2,
                "$sym must be a square matrix, size($sym) = $(size(val)), with side length equal to the number of assets squared, size(returns, 2)^2 = $(size(obj.returns, 2))^2"
            )
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:L_2, :S_2)
        if !isempty(val)
            N = size(obj.returns, 2)
            @assert(
                size(val) == (Int(N * (N + 1) / 2), N^2),
                "size($sym) == $(size(val)), must be equal to (N * (N + 1) / 2, N^2) = $((Int(N * (N + 1) / 2), N^2)), where N = size(obj.returns, 2) = $N"
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
