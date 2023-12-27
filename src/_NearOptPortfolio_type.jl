mutable struct NearOptPortfolio{
    tp,
    tc,
    te,
    topt,
    tsolv,
    toptpar,
    tf,
    tmod,
    tlp,
    taopt,
    tasolv,
    taoptpar,
    taf,
    tamod,
} <: AbstractPortfolio
    optimum_portfolio::tp
    c1::tc
    c2::tc
    e1::te
    e2::te
    # Optimal portfolios
    optimal::topt
    # Solutions
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

function NearOptPortfolio(;
    portfolio::Union{Portfolio, HCPortfolio} = portfolio,
    c1::Real = zero(eltype(portfolio.returns)),
    c2::Real = zero(eltype(portfolio.returns)),
    e1::Real = zero(eltype(portfolio.returns)),
    e2::Real = zero(eltype(portfolio.returns)),
    # Optimal portfolios
    optimal::AbstractDict = Dict(),
    # Solutions
    solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    opt_params::Union{<:AbstractDict, NamedTuple} = Dict(),
    fail::AbstractDict = Dict(),
    model::JuMP.Model = JuMP.Model(),
    # Allocation
    latest_prices::AbstractVector = Vector{Float64}(undef, 0),
    alloc_optimal::AbstractDict = Dict(),
    alloc_solvers::Union{<:AbstractDict, NamedTuple} = Dict(),
    alloc_params::Union{<:AbstractDict, NamedTuple} = Dict(),
    alloc_fail::AbstractDict = Dict(),
    alloc_model::JuMP.Model = JuMP.Model(),
)
    return NearOptPortfolio{
        typeof(portfolio),
        typeof(c1),
        typeof(c2),
        typeof(e1),
        typeof(e2),
        typeof(optimal),
        typeof(solvers),
        typeof(opt_params),
        typeof(fail),
        typeof(model),
        typeof(latest_prices),
        typeof(alloc_optimal),
        typeof(alloc_solvers),
        typeof(alloc_params),
        typeof(alloc_fail),
        typeof(alloc_model),
    }
    (
        portfolio,
        c1,
        c2,
        e1,
        e2,
        optimal,
        solvers,
        opt_params,
        fail,
        model,
        latest_prices,
        alloc_optimal,
        alloc_solvers,
        alloc_params,
        alloc_fail,
        alloc_model,
    )
end
