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
    invkappa2::tk2
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
    z::tz
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
- `assets`: `N×1` vector of assets in the portfolio.
- `timestamps`: `T×1` vector of timestamps of the returns series.
- `returns`: `T×N` matrix of the returns series.
- `short`: whether or not to enable short investments (produce negative weights).
- `short_u`: sum of the absolute values of the short (negative) weights.
- `long_u`: sum of the absolute values of the long (positive) weights.
- `sum_short_long`: if shorting is enabled, the maximum value of the sum of the long (positive) weights and short (negative) weights, `sum_short_long = long_u - short_u`.
- `min_number_effective_assets`: if finite, constraints are added to the optimisations such that at least this amount of assets significantly contribute to the final weights vector.
- `max_number_assets`: if finite, maximum number of assets with non-zero weights in the final weights vector.
- `max_number_assets_factor`: factor to use in the binary decision variable used when `max_number_assets` is in finite.
- `f_assets`:
- `f_timestamps`:
- `f_returns`:
- `loadings`:
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
    invkappa2::tk2
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
    z::tz
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
    returns = DataFrame(),
    ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
    assets = Vector{String}(undef, 0),
    short::Bool = false,
    short_u::Real = 0.2,
    long_u::Real = 1.0,
    sum_short_long::Real = short ? long_u - short_u : 1.0,
    min_number_effective_assets::Integer = 0,
    max_number_assets::Integer = 0,
    max_number_assets_factor::Real = 100_000.0,
    f_returns = DataFrame(),
    f_ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    f_timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
    f_assets = Vector{String}(undef, 0),
    loadings = Matrix{Float64}(undef, 0, 0),
    # Risk parameters.
    msv_target::Union{<:Real, AbstractVector{<:Real}, Nothing} = Inf,
    lpm_target::Union{<:Real, AbstractVector{<:Real}, Nothing} = Inf,
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{<:Integer, Nothing} = nothing,
    kappa::Real = 0.3,
    gs_threshold::Real = 0.5,
    max_num_assets_kurt::Integer = 0,
    # Benchmark constraints.
    turnover::Real = Inf,
    turnover_weights = Vector{Float64}(undef, 0),
    kind_tracking_err::Symbol = :Weights,
    tracking_err::Real = Inf,
    tracking_err_returns = Vector{Float64}(undef, 0),
    tracking_err_weights = Vector{Float64}(undef, 0),
    bl_bench_weights = Vector{Float64}(undef, 0),
    # Risk and return constraints.
    a_mtx_ineq = Matrix{Float64}(undef, 0, 0),
    b_vec_ineq = Vector{Float64}(undef, 0),
    risk_budget::Union{<:Real, AbstractVector{<:Real}} = Vector{Float64}(undef, 0),
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
    owa_w::Union{<:Real, AbstractVector{<:Real}, Nothing} = Vector{Float64}(undef, 0),
    krt_u::Real = Inf,
    skrt_u::Real = Inf,
    rvar_u::Real = Inf,
    rdar_u::Real = Inf,
    # Optimisation model inputs.
    mu_type::Symbol = :Default,
    mu = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov = Matrix{Float64}(undef, 0, 0),
    kurt = Matrix{Float64}(undef, 0, 0),
    skurt = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
    S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
    mu_f = Vector{Float64}(undef, 0),
    cov_f = Matrix{Float64}(undef, 0, 0),
    mu_fm = Vector{Float64}(undef, 0),
    cov_fm = Matrix{Float64}(undef, 0, 0),
    mu_bl = Vector{Float64}(undef, 0),
    cov_bl = Matrix{Float64}(undef, 0, 0),
    mu_bl_fm = Vector{Float64}(undef, 0),
    cov_bl_fm = Matrix{Float64}(undef, 0, 0),
    returns_fm = Matrix{Float64}(undef, 0, 0),
    z::AbstractDict = Dict(),
    # Inputs of Worst Case Optimization Models.
    cov_l = Matrix{Float64}(undef, 0, 0),
    cov_u = Matrix{Float64}(undef, 0, 0),
    cov_mu = Diagonal{Float64}(undef, 0),
    cov_sigma = Diagonal{Float64}(undef, 0),
    d_mu = Vector{Float64}(undef, 0),
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
    # Allocation.
    latest_prices = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::AbstractDict = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model::AbstractDict = Dict(),
)
```
"""
function Portfolio(;
    # Portfolio characteristics.
    returns = DataFrame(),
    ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
    assets = Vector{String}(undef, 0),
    short::Bool = false,
    short_u::Real = 0.2,
    long_u::Real = 1.0,
    sum_short_long::Real = short ? long_u - short_u : 1.0,
    min_number_effective_assets::Integer = 0,
    max_number_assets::Integer = 0,
    max_number_assets_factor::Real = 100_000.0,
    f_returns = DataFrame(),
    f_ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    f_timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
    f_assets = Vector{String}(undef, 0),
    loadings = Matrix{Float64}(undef, 0, 0),
    # Risk parameters.
    msv_target::Union{<:Real, AbstractVector{<:Real}, Nothing} = Inf,
    lpm_target::Union{<:Real, AbstractVector{<:Real}, Nothing} = Inf,
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{<:Integer, Nothing} = nothing,
    kappa::Real = 0.3,
    gs_threshold::Real = 0.5,
    max_num_assets_kurt::Integer = 0,
    # Benchmark constraints.
    turnover::Real = Inf,
    turnover_weights = Vector{Float64}(undef, 0),
    kind_tracking_err::Symbol = :Weights,
    tracking_err::Real = Inf,
    tracking_err_returns = Vector{Float64}(undef, 0),
    tracking_err_weights = Vector{Float64}(undef, 0),
    bl_bench_weights = Vector{Float64}(undef, 0),
    # Risk and return constraints.
    a_mtx_ineq = Matrix{Float64}(undef, 0, 0),
    b_vec_ineq = Vector{Float64}(undef, 0),
    risk_budget::Union{<:Real, AbstractVector{<:Real}} = Vector{Float64}(undef, 0),
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
    owa_w::Union{<:Real, AbstractVector{<:Real}, Nothing} = Vector{Float64}(undef, 0),
    krt_u::Real = Inf,
    skrt_u::Real = Inf,
    rvar_u::Real = Inf,
    rdar_u::Real = Inf,
    # Optimisation model inputs.
    mu_type::Symbol = :Default,
    mu = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov = Matrix{Float64}(undef, 0, 0),
    kurt = Matrix{Float64}(undef, 0, 0),
    skurt = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
    S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0),
    mu_f = Vector{Float64}(undef, 0),
    cov_f = Matrix{Float64}(undef, 0, 0),
    mu_fm = Vector{Float64}(undef, 0),
    cov_fm = Matrix{Float64}(undef, 0, 0),
    mu_bl = Vector{Float64}(undef, 0),
    cov_bl = Matrix{Float64}(undef, 0, 0),
    mu_bl_fm = Vector{Float64}(undef, 0),
    cov_bl_fm = Matrix{Float64}(undef, 0, 0),
    returns_fm = Matrix{Float64}(undef, 0, 0),
    z::AbstractDict = Dict(),
    # Inputs of Worst Case Optimization Models.
    cov_l = Matrix{Float64}(undef, 0, 0),
    cov_u = Matrix{Float64}(undef, 0, 0),
    cov_mu = Diagonal{Float64}(undef, 0),
    cov_sigma = Diagonal{Float64}(undef, 0),
    d_mu = Vector{Float64}(undef, 0),
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
    # Allocation.
    latest_prices = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::AbstractDict = Dict(),
    alloc_params::AbstractDict = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model::AbstractDict = Dict(),
)
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

    return Portfolio{# Portfolio characteristics.
        typeof(assets),
        typeof(timestamps),
        typeof(returns),
        typeof(short),
        typeof(short_u),
        typeof(long_u),
        typeof(sum_short_long),
        typeof(min_number_effective_assets),
        typeof(max_number_assets),
        typeof(max_number_assets_factor),
        typeof(f_assets),
        typeof(f_timestamps),
        typeof(f_returns),
        typeof(loadings),
        # Risk parameters.
        Union{<:Real, AbstractVector{<:Real}, Nothing},
        Union{<:Real, AbstractVector{<:Real}, Nothing},
        typeof(alpha_i),
        typeof(alpha),
        typeof(a_sim),
        typeof(alpha),
        Union{<:Real, Nothing},
        Union{<:Real, Nothing},
        Union{<:Integer, Nothing},
        typeof(kappa),
        typeof(kappa),
        typeof(kappa),
        typeof(kappa),
        typeof(kappa),
        typeof(kappa),
        typeof(kappa),
        typeof(kappa),
        typeof(kappa),
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
        Union{<:Real, Vector{<:Real}},
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
        Union{<:Real, AbstractVector{<:Real}, Nothing},
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
        typeof(z),
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
        sum_short_long,
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
        zero(typeof(alpha)),
        beta_i,
        beta,
        b_sim,
        kappa,
        zero(typeof(kappa)),
        zero(typeof(kappa)),
        zero(typeof(kappa)),
        zero(typeof(kappa)),
        zero(typeof(kappa)),
        zero(typeof(kappa)),
        zero(typeof(kappa)),
        zero(typeof(kappa)),
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
        z,
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
        # Allocation.
        latest_prices,
        alloc_optimal,
        alloc_solvers,
        alloc_params,
        alloc_fail,
        alloc_model,
    )
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
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{<:Integer, Nothing} = nothing,
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
    w_min::Union{<:Real, AbstractVector{<:Real}, Nothing} = 0.0,
    w_max::Union{<:Real, AbstractVector{<:Real}, Nothing} = 1.0,
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

function HCPortfolio(;
    # Portfolio characteristics.
    returns = DataFrame(),
    ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
    assets = Vector{String}(undef, 0),
    # Risk parmeters.
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{<:Integer, Nothing} = nothing,
    kappa::Real = 0.3,
    alpha_tail::Real = 0.05,
    gs_threshold::Real = 0.5,
    owa_w::Union{<:Real, AbstractVector{<:Real}, Nothing} = Vector{Float64}(undef, 0),
    # Optimisation parameters.
    mu_type::Symbol = :Default,
    mu = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Full,
    jlogo::Bool = false,
    cov = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    bins_info::Union{Symbol, <:Integer} = :KN,
    w_min::Union{<:Real, AbstractVector{<:Real}, Nothing} = 0.0,
    w_max::Union{<:Real, AbstractVector{<:Real}, Nothing} = 1.0,
    codep_type::Symbol = :Pearson,
    codep = Matrix{Float64}(undef, 0, 0),
    dist = Matrix{Float64}(undef, 0, 0),
    clusters = Hclust{Float64}(Matrix{Int64}(undef, 0, 2), Float64[], Int64[], :nothing),
    k::Union{<:Integer, Nothing} = nothing,
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
        Union{<:Real, Nothing},
        Union{<:Real, Nothing},
        Union{<:Integer, Nothing},
        typeof(kappa),
        typeof(alpha_tail),
        typeof(gs_threshold),
        Union{<:Real, AbstractVector{<:Real}, Nothing},
        # Optimisation parameters.
        typeof(mu_type),
        typeof(mu),
        typeof(cov_type),
        typeof(jlogo),
        typeof(cov),
        typeof(posdef_fix),
        Union{Symbol, <:Integer},
        Union{<:Real, AbstractVector{<:Real}, Nothing},
        Union{<:Real, AbstractVector{<:Real}, Nothing},
        typeof(codep_type),
        typeof(codep),
        typeof(dist),
        typeof(clusters),
        Union{<:Integer, Nothing},
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

export Portfolio, HCPortfolio