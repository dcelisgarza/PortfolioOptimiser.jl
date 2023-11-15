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

export HCPortfolio
