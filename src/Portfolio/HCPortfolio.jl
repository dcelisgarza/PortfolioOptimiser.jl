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
    hrpopt,
    hercopt,
    herc2opt,
    ncoopt,
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
    hrp_optimal::hrpopt
    herc_optimal::hercopt
    herc2_optimal::herc2opt
    nco_optimal::ncoopt
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
    beta_i::Union{Real, Nothing} = nothing,
    beta::Union{Real, Nothing} = nothing,
    b_sim::Union{Integer, Nothing} = nothing,
    kappa::Real = 0.3,
    alpha_tail::Real = 0.05,
    gs_threshold::Real = 0.5,
    # Optimisation parameters.
    mu_type::Symbol = :Historical,
    mu = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Historical,
    cov = Matrix{Float64}(undef, 0, 0),
    bins_info::Union{Symbol, Int} = :KN,
    w_min::Union{AbstractFloat, AbstractVector, Nothing} = 0.0,
    w_max::Union{AbstractFloat, AbstractVector, Nothing} = 1.0,
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
    # Optimisation parameters.
    ttmu,
    tmu,
    ttcov,
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
    hrpopt,
    hercopt,
    herc2opt,
    ncoopt,
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
    mu_type::ttmu
    mu::tmu
    cov_type::ttcov
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
    hrp_optimal::hrpopt
    herc_optimal::hercopt
    herc2_optimal::herc2opt
    nco_optimal::ncoopt
    # Solutions.
    solvers::tsolv
    opt_params::toptpar
    fail::tf
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
    beta_i::Union{Real, Nothing} = nothing,
    beta::Union{Real, Nothing} = nothing,
    b_sim::Union{Integer, Nothing} = nothing,
    kappa::Real = 0.3,
    alpha_tail::Real = 0.05,
    gs_threshold::Real = 0.5,
    # Optimisation parameters.
    mu_type::Symbol = :Historical,
    mu = Vector{Float64}(undef, 0),
    cov_type::Symbol = :Historical,
    cov = Matrix{Float64}(undef, 0, 0),
    posdef_fix::Symbol = :None,
    bins_info::Union{Symbol, Int} = :KN,
    w_min::Union{AbstractFloat, AbstractVector, Nothing} = 0.0,
    w_max::Union{AbstractFloat, AbstractVector, Nothing} = 1.0,
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
    if isa(returns, DataFrame) && !isempty(returns)
        assets = names(returns)[2:end]
        timestamps = returns[!, 1]
        returns = Matrix(returns[!, 2:end])
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
        Union{Real, Nothing},
        Union{Real, Nothing},
        Union{Int, Nothing},
        typeof(kappa),
        typeof(alpha_tail),
        typeof(gs_threshold),
        # Optimisation parameters.
        typeof(mu_type),
        typeof(mu),
        typeof(cov_type),
        typeof(cov),
        typeof(posdef_fix),
        Union{Symbol, Int},
        Union{AbstractFloat, AbstractVector, Nothing},
        Union{AbstractFloat, AbstractVector, Nothing},
        typeof(codep_type),
        typeof(codep),
        typeof(dist),
        typeof(clusters),
        Union{Int, Nothing},
        # Optimal portfolios.
        DataFrame,
        DataFrame,
        DataFrame,
        DataFrame,
        # Solutions.
        typeof(solvers),
        typeof(opt_params),
        typeof(fail),
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
        # Optimisation parameters.
        mu_type,
        mu,
        cov_type,
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
        DataFrame(),
        DataFrame(),
        DataFrame(),
        DataFrame(),
        # Solutions.
        solvers,
        opt_params,
        fail,
    )
end

export HCPortfolio
