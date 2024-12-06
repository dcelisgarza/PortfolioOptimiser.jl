"""
```
abstract type AbstractPortfolio end
```

Abstract type for subtyping portfolios.
"""
abstract type AbstractPortfolio end

mutable struct OmniPortfolio{
                             # Assets and factors
                             T_assets, T_timestamps, T_returns, T_latest_prices, T_f_assets,
                             T_f_timestamps, T_f_returns, T_loadings, T_regression_type,
                             # Statistics
                             T_mu_l, T_mu, T_cov, T_cor, T_dist, T_clusters, T_k,
                             T_min_cluster_size, T_max_num_assets_kurt,
                             T_max_num_assets_kurt_scale, T_kurt, T_skurt, T_L_2, T_S_2,
                             T_skew, T_V, T_sskew, T_SV, T_f_mu, T_f_cov, T_fm_returns,
                             T_fm_mu, T_fm_cov, T_bl_bench_weights, T_bl_mu, T_bl_cov,
                             T_blfm_mu, T_blfm_cov, T_cov_l, T_cov_u, T_cov_mu, T_cov_sigma,
                             T_d_mu, T_k_mu, T_k_sigma,
                             # Min and max weights
                             T_w_min, T_w_max,
                             # Risk budgetting
                             T_risk_budget, T_f_risk_budget,
                             # Budget and shorting
                             T_short, T_long_l, T_long_u, T_short_l, T_short_u,
                             T_min_budget, T_budget, T_max_budget, T_min_short_budget,
                             T_short_budget, T_max_short_budget,
                             # Cardinality
                             T_card_scale, T_card, T_a_card_ineq, T_b_card_ineq,
                             T_a_card_eq, T_b_card_eq, T_nea, T_a_ineq, T_b_ineq, T_a_eq,
                             T_b_eq, T_tracking, T_turnover, T_network_adj, T_cluster_adj,
                             T_l1, T_l2, T_long_fees, T_short_fees, T_rebalance,
                             T_constraint_scale, T_obj_scale, T_model, T_solvers, T_optimal,
                             T_fail, T_limits, T_frontier, T_walking, T_alloc_model,
                             T_alloc_solvers, T_alloc_optimal, T_alloc_leftover,
                             T_alloc_fail, T_alloc_walking} <: AbstractPortfolio
    # Assets and factors
    assets::T_assets
    timestamps::T_timestamps
    returns::T_returns
    latest_prices::T_latest_prices
    f_assets::T_f_assets
    f_timestamps::T_f_timestamps
    f_returns::T_f_returns
    loadings::T_loadings
    regression_type::T_regression_type
    # Statistics
    mu_l::T_mu_l
    mu::T_mu
    cov::T_cov
    cor::T_cor
    dist::T_dist
    clusters::T_clusters
    k::T_k
    min_cluster_size::T_min_cluster_size
    max_num_assets_kurt::T_max_num_assets_kurt
    max_num_assets_kurt_scale::T_max_num_assets_kurt_scale
    kurt::T_kurt
    skurt::T_skurt
    L_2::T_L_2
    S_2::T_S_2
    skew::T_skew
    V::T_V
    sskew::T_sskew
    SV::T_SV
    f_mu::T_f_mu
    f_cov::T_f_cov
    fm_returns::T_fm_returns
    fm_mu::T_fm_mu
    fm_cov::T_fm_cov
    bl_bench_weights::T_bl_bench_weights
    bl_mu::T_bl_mu
    bl_cov::T_bl_cov
    blfm_mu::T_blfm_mu
    blfm_cov::T_blfm_cov
    cov_l::T_cov_l
    cov_u::T_cov_u
    cov_mu::T_cov_mu
    cov_sigma::T_cov_sigma
    d_mu::T_d_mu
    k_mu::T_k_mu
    k_sigma::T_k_sigma
    # Min and max weights
    w_min::T_w_min
    w_max::T_w_max
    # Risk budgetting
    risk_budget::T_risk_budget
    f_risk_budget::T_f_risk_budget
    # Budget and shorting
    short::T_short
    long_l::T_long_l
    long_u::T_long_u
    short_l::T_short_l
    short_u::T_short_u
    min_budget::T_min_budget
    budget::T_budget
    max_budget::T_max_budget
    min_short_budget::T_min_short_budget
    short_budget::T_short_budget
    max_short_budget::T_max_short_budget
    # Cardinality
    card_scale::T_card_scale
    card::T_card
    a_card_ineq::T_a_card_ineq
    b_card_ineq::T_b_card_ineq
    a_card_eq::T_a_card_eq
    b_card_eq::T_b_card_eq
    # Effective assets
    nea::T_nea
    # Linear constraints
    a_ineq::T_a_ineq
    b_ineq::T_b_ineq
    a_eq::T_a_eq
    b_eq::T_b_eq
    # Tracking
    tracking::T_tracking
    # Turnover
    turnover::T_turnover
    # Adjacency
    network_adj::T_network_adj
    cluster_adj::T_cluster_adj
    # Regularisation
    l1::T_l1
    l2::T_l2
    # Fees
    long_fees::T_long_fees
    short_fees::T_short_fees
    # Rebalance cost
    rebalance::T_rebalance
    # Solution
    constr_scale::T_constraint_scale
    obj_scale::T_obj_scale
    model::T_model
    solvers::T_solvers
    optimal::T_optimal
    fail::T_fail
    limits::T_limits
    frontier::T_frontier
    walking::T_walking
    alloc_model::T_alloc_model
    alloc_solvers::T_alloc_solvers
    alloc_optimal::T_alloc_optimal
    alloc_leftover::T_alloc_leftover
    alloc_fail::T_alloc_fail
    alloc_walking::T_alloc_walking
end
function setup_returns_assets_timestamps(returns, assets, timestamps, ret)
    if !isempty(returns)
        assets = setdiff(names(returns), ("timestamp",))
        timestamps = returns[!, "timestamp"]
        returns = Matrix(returns[!, assets])
    else
        @smart_assert(length(assets) == size(ret, 2))
        returns = ret
    end
    return assets, timestamps, returns
end
function vector_assert(x::AbstractVector, n::Integer, name = "")
    if !isempty(x)
        @smart_assert(length(x) == n, "Length of $name must be equal to $n")
    end
    return nothing
end
function matrix_assert(x::AbstractMatrix, n1::Integer, n2::Integer, name = "")
    if !isempty(x)
        @smart_assert(size(x) == (n1, n2), "Size of $name must be equal to ($n1, $n2)")
    end
    return nothing
end
function risk_budget_assert(risk_budget, n::Integer, name = "")
    if !isempty(risk_budget)
        @smart_assert(length(risk_budget) == n, "Length of $name must be equal to $n")
        # @smart_assert(all(risk_budget .>= zero(eltype(returns))))
        # if isa(risk_budget, AbstractRange)
        #     risk_budget = collect(risk_budget / sum(risk_budget))
        # else
        #     risk_budget ./= sum(risk_budget)
        # end
        if isa(risk_budget, AbstractRange)
            risk_budget = collect(risk_budget)
        end
    end
    return risk_budget
end
function factor_risk_budget_assert(risk_budget, n::Integer, name = "")
    if !isempty(risk_budget)
        @smart_assert(length(risk_budget) <= n,
                      "Length of $name must be less than or equal to $n")
        # @smart_assert(all(risk_budget .>= zero(eltype(returns))))
        # if isa(risk_budget, AbstractRange)
        #     risk_budget = collect(risk_budget / sum(risk_budget))
        # else
        #     risk_budget ./= sum(risk_budget)
        # end
        if isa(risk_budget, AbstractRange)
            risk_budget = collect(risk_budget)
        end
    end
    return risk_budget
end
function real_or_vector_assert(x::Union{<:Real, AbstractVector{<:Real}}, n::Integer,
                               name = "", f = >=, val = 0.0)
    if isa(x, AbstractVector) && !isempty(x)
        @smart_assert(length(x) == n, "Length of $name must be equal to $n")
        @smart_assert(all(f.(x, val)))
    elseif isa(x, Real)
        @smart_assert(f(x, val))
    end
    return nothing
end
function set_default_budget(budget, val, min_budget_flag, budget_flag, max_budget_flag)
    if !min_budget_flag && !budget_flag && !max_budget_flag
        budget = val
        budget_flag = true
    end
    return budget, budget_flag
end
function short_budget_assert(budget_flag, min_budget_flag, max_budget_flag, min_budget,
                             budget, max_budget, short_budget, long_u, short_u, name = "")
    if budget_flag
        @smart_assert(all(short_budget .<= short_u .<= 0), "all($name .<= short_u .<= 0)")
        @smart_assert(all(budget .- short_budget .>= long_u .>= 0),
                      "all(budget .- $name .>= long_u .>= 0")
    else
        if min_budget_flag
            @smart_assert(all(short_budget .<= short_u .<= 0),
                          "all($name .<= short_u .<= 0)")
            @smart_assert(all(min_budget .- short_budget .>= long_u .>= 0),
                          "all(min_budget .- $name .>= long_u .>= 0)")
        end
        if max_budget_flag
            @smart_assert(all(short_budget .<= short_u .<= 0),
                          "all($name .<= short_u .<= 0)")
            @smart_assert(all(max_budget .- short_budget .>= long_u .>= 0),
                          "all(max_budget .- $name .>= long_u .>= 0)")
        end
        if min_budget_flag && max_budget_flag
            @smart_assert(max_budget >= min_budget)
        end
    end
    return nothing
end
function linear_constraint_assert(A::AbstractMatrix, B::AbstractVector, n::Integer,
                                  name = "")
    if !(isempty(A) || isempty(B))
        C, N = size(A)
        @smart_assert(C == length(B),
                      "length(B) = $(length(B)), must be equal to the number of constraints C = $(size(A,1))")
        @smart_assert(N == n, "size(A, 2)  = $N, must be equal to the number of assets $n")
    end
    return nothing
end
function tracking_assert(tracking, t, n)
    if !isa(tracking, NoTracking)
        @smart_assert(tracking.err >= zero(tracking.err))
        if isa(tracking, TrackRet)
            @smart_assert(length(tracking.w) == t)
        else
            @smart_assert(length(tracking.w) == n)
        end
    end
    return nothing
end
function tr_assert(tr, n::Integer)
    if isa(tr, TR)
        if isa(tr.val, Real)
            @smart_assert(tr.val >= zero(tr.val))
        elseif isa(tr.val, AbstractVector) && !isempty(tr.val)
            @smart_assert(length(tr.val) == n && all(tr.val .>= zero(tr.val)))
        end
        if !isempty(tr.w)
            @smart_assert(length(tr.w) == n)
        end
    end
    return nothing
end
function adj_assert(adj, n::Integer)
    if !isa(adj, NoAdj) && !isempty(adj.A)
        if isa(adj, IP)
            @smart_assert(size(adj.A, 2) == n)
        else
            @smart_assert(size(adj.A) == (n, n))
        end
    end
end
function long_short_budget_assert(N, long_l, long_u, min_budget, budget, max_budget, short,
                                  short_l, short_u, min_short_budget, short_budget,
                                  max_short_budget)
    real_or_vector_assert(long_l, N, :long_l, >=, 0)
    real_or_vector_assert(long_u, N, :long_u, >=, 0)
    @smart_assert(all(long_l .<= long_u))
    min_budget_flag = isfinite(min_budget)
    budget_flag = isfinite(budget)
    max_budget_flag = isfinite(max_budget)
    budget, budget_flag = set_default_budget(budget, 1.0, min_budget_flag, budget_flag,
                                             max_budget_flag)

    if short
        real_or_vector_assert(short_l, N, :short_l, <=, 0)
        real_or_vector_assert(short_u, N, :short_u, <=, 0)
        @smart_assert(short_u <= short_l)
        min_short_budget_flag = isfinite(min_short_budget)
        short_budget_flag = isfinite(short_budget)
        max_short_budget_flag = isfinite(max_short_budget)

        short_budget, short_budget_flag = set_default_budget(short_budget, -0.2,
                                                             min_budget_flag, budget_flag,
                                                             max_budget_flag)
        if short_budget_flag
            short_budget_assert(budget_flag, min_budget_flag, max_budget_flag, min_budget,
                                budget, max_budget, short_budget, long_u, short_u,
                                :short_budget)
        else
            if min_short_budget_flag
                short_budget_assert(budget_flag, min_budget_flag, max_budget_flag,
                                    min_budget, budget, max_budget, min_short_budget,
                                    long_u, short_u, :min_short_budget)
            end
            if max_short_budget_flag
                short_budget_assert(budget_flag, min_budget_flag, max_budget_flag,
                                    min_budget, budget, max_budget, max_short_budget,
                                    long_u, short_u, :max_short_budget)
            end
            if min_short_budget_flag && max_short_budget_flag
                @smart_assert(max_short_budget <= min_short_budget)
            end
        end
    else
        if budget_flag
            @smart_assert(all(budget .>= long_u .>= 0))
        else
            if min_budget_flag
                @smart_assert(all(min_budget .>= long_u .>= 0))
            end
            if max_budget_flag
                @smart_assert(all(max_budget .>= long_u .>= 0))
            end
            if min_budget_flag && max_budget_flag
                @smart_assert(max_budget >= min_budget)
            end
        end
    end
    return budget, short_budget
end
function OmniPortfolio(;
                       # Assets and factors
                       prices::TimeArray = TimeArray(TimeType[], []),
                       ret_type::Symbol = :simple, returns::DataFrame = DataFrame(),
                       ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       timestamps::AbstractVector = Vector{Date}(undef, 0),
                       assets::AbstractVector = Vector{String}(undef, 0),
                       latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       f_prices::TimeArray = TimeArray(TimeType[], []),
                       f_returns::DataFrame = DataFrame(),
                       f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       f_timestamps::AbstractVector = Vector{Date}(undef, 0),
                       f_assets::AbstractVector = Vector{String}(undef, 0),
                       loadings::DataFrame = DataFrame(),
                       regression_type::Union{<:RegressionType, Nothing} = nothing,
                       # Statistics
                       mu_l::Real = Inf, mu::AbstractVector = Vector{Float64}(undef, 0),
                       cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       cor::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       clusters::Hclust = Hclust{Float64}(Matrix{Int64}(undef, 0, 2),
                                                          Float64[], Int64[], :nothing),
                       k::Integer = 0, min_cluster_size::Integer = 2,
                       max_num_assets_kurt::Integer = 0,
                       max_num_assets_kurt_scale::Integer = 2,
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
                       bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       bl_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       bl_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       blfm_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       blfm_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       cov_l::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       cov_u::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       cov_mu::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       cov_sigma::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       d_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       k_mu::Real = Inf, k_sigma::Real = Inf,
                       # Min and max weights
                       w_min::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                       w_max::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                       # Risk budgetting
                       risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       f_risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       # Budget and shorting
                       short::Bool = false,
                       long_l::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                       long_u::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                       short_l::Union{<:Real, <:AbstractVector{<:Real}} = -0.0,
                       short_u::Union{<:Real, <:AbstractVector{<:Real}} = -0.2,
                       min_budget::Real = Inf, budget::Real = 1.0, max_budget::Real = Inf,
                       min_short_budget::Real = -Inf, short_budget::Real = -0.2,
                       max_short_budget::Real = -Inf,
                       # Cardinality
                       card_scale::Real = 1e6, card::Integer = 0,
                       a_card_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       b_card_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       a_card_eq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       b_card_eq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       # Effective assets
                       nea::Real = 0.0,
                       # Linear constraints
                       a_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       b_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       a_eq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                       b_eq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                       # Tracking
                       tracking::TrackingErr = NoTracking(),
                       # Turnover
                       turnover::AbstractTR = NoTR(),
                       # Adjacency
                       network_adj::AdjacencyConstraint = NoAdj(),
                       cluster_adj::AdjacencyConstraint = NoAdj(),
                       # Regularisation
                       l1::Real = 0.0, l2::Real = 0.0,
                       # Fees
                       long_fees::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                       short_fees::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                       rebalance::AbstractTR = NoTR(),
                       # Solution
                       constr_scale::Real = 1.0, obj_scale::Real = 1.0,
                       model::JuMP.Model = JuMP.Model(), solvers::AbstractDict = Dict(),
                       optimal::AbstractDict = Dict(), fail::AbstractDict = Dict(),
                       limits::AbstractDict = Dict(), frontier::AbstractDict = Dict(),
                       walking::AbstractDict = Dict(),
                       alloc_model::JuMP.Model = JuMP.Model(),
                       alloc_solvers::AbstractDict = Dict(),
                       alloc_optimal::AbstractDict = Dict(),
                       alloc_leftover::AbstractDict = Dict(),
                       alloc_fail::AbstractDict = Dict(),
                       alloc_walking::AbstractDict = Dict())
    # Assets and factors
    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices, ret_type)))
        latest_prices = Vector(dropmissing!(DataFrame(prices))[end, colnames(prices)])
    end
    assets, timestamps, returns = setup_returns_assets_timestamps(returns, assets,
                                                                  timestamps, ret)
    if !isempty(f_prices)
        f_returns = dropmissing!(DataFrame(percentchange(f_prices, ret_type)))
    end
    f_assets, f_timestamps, f_returns = setup_returns_assets_timestamps(f_returns, f_assets,
                                                                        f_timestamps, f_ret)
    T, N = size(returns)
    Tf, Nf = size(f_returns)
    vector_assert(latest_prices, N, :latest_prices)
    if !isempty(f_ret)
        @smart_assert(T == Tf)
    end
    # Statistics
    vector_assert(mu, N, :mu)
    matrix_assert(cov, N, N, :cov)
    matrix_assert(cor, N, N, :cor)
    matrix_assert(dist, N, N, :dist)
    @smart_assert(max_num_assets_kurt >= zero(max_num_assets_kurt))
    max_num_assets_kurt_scale = clamp(max_num_assets_kurt_scale, 1, N)
    matrix_assert(kurt, N^2, N^2, :kurt)
    matrix_assert(skurt, N^2, N^2, :skurt)
    matrix_assert(L_2, Int(N * (N + 1) / 2), N^2, :L_2)
    matrix_assert(S_2, Int(N * (N + 1) / 2), N^2, :S_2)
    matrix_assert(skew, N, N^2, :skew)
    matrix_assert(V, N, N, :V)
    matrix_assert(sskew, N, N^2, :sskew)
    matrix_assert(SV, N, N, :SV)
    vector_assert(f_mu, Nf, :f_mu)
    matrix_assert(f_cov, Nf, Nf, :f_cov)
    matrix_assert(fm_returns, T, N, :fm_returns)
    vector_assert(fm_mu, N, :fm_mu)
    matrix_assert(fm_cov, N, N, :fm_cov)
    vector_assert(bl_bench_weights, N, :bl_bench_weights)
    vector_assert(bl_mu, N, :bl_mu)
    matrix_assert(bl_cov, N, N, :bl_cov)
    vector_assert(blfm_mu, N, :blfm_mu)
    matrix_assert(blfm_cov, N, N, :blfm_cov)
    matrix_assert(cov_l, N, N, :cov_l)
    matrix_assert(cov_u, N, N, :cov_u)
    matrix_assert(cov_mu, N, N, :cov_mu)
    matrix_assert(cov_sigma, N^2, N^2, :cov_sigma)
    vector_assert(d_mu, N, :d_mu)
    # Min and max weights
    if isa(w_min, Real)
        if isa(w_max, Real)
            @smart_assert(w_min <= w_max)
        elseif !isempty(w_max)
            @smart_assert(all(w_min .<= w_max))
        end
    elseif isa(w_min, AbstractVector)
        if !isempty(w_min)
            @smart_assert(length(w_min) == N)
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
            @smart_assert(length(w_max) == N)
            if isa(w_min, Real) || !isempty(w_min)
                @smart_assert(all(w_max .>= w_min))
            end
        end
    end
    # Risk budgetting
    risk_budget = risk_budget_assert(risk_budget, N, :risk_budget)
    f_risk_budget = factor_risk_budget_assert(f_risk_budget, Nf, :f_risk_budget)
    # Budget and shorting
    budget, short_budget = long_short_budget_assert(N, long_l, long_u, min_budget, budget,
                                                    max_budget, short, short_l, short_u,
                                                    min_short_budget, short_budget,
                                                    max_short_budget)
    # Cardinality
    @smart_assert(card >= zero(card))
    linear_constraint_assert(a_card_ineq, b_card_ineq, N, "card_ineq")
    linear_constraint_assert(a_card_eq, b_card_eq, N, "card_eq")
    # Effective assets
    @smart_assert(nea >= zero(nea))
    # Linear constraints
    linear_constraint_assert(a_ineq, b_ineq, N, "ineq")
    linear_constraint_assert(a_eq, b_eq, N, "eq")
    # Tracking
    tracking_assert(tracking, T, N)
    # Turnover
    tr_assert(turnover, N)
    # Adjacency
    adj_assert(network_adj, N)
    adj_assert(cluster_adj, N)
    # Regularisation
    @smart_assert(l1 >= zero(l1))
    @smart_assert(l2 >= zero(l2))
    # Fees
    real_or_vector_assert(long_fees, N, :long_fees, >=, 0)
    real_or_vector_assert(short_fees, N, :short_fees, >=, 0)
    tr_assert(rebalance, N)
    # Constraint and objective scales
    @smart_assert(constr_scale > zero(constr_scale))
    @smart_assert(obj_scale > zero(obj_scale))

    return OmniPortfolio{
                         # Assets and factors
                         typeof(assets), typeof(timestamps), typeof(returns),
                         typeof(latest_prices), typeof(f_assets), typeof(f_timestamps),
                         typeof(f_returns), typeof(loadings),
                         Union{<:RegressionType, Nothing},
                         # Statistics
                         typeof(mu_l), typeof(mu), typeof(cov), typeof(cor), typeof(dist),
                         typeof(clusters), typeof(k), typeof(min_cluster_size),
                         typeof(max_num_assets_kurt), typeof(max_num_assets_kurt_scale),
                         typeof(kurt), typeof(skurt), typeof(L_2), typeof(S_2),
                         typeof(skew), typeof(V), typeof(sskew), typeof(SV), typeof(f_mu),
                         typeof(f_cov), typeof(fm_returns), typeof(fm_mu), typeof(fm_cov),
                         typeof(bl_bench_weights), typeof(bl_mu), typeof(bl_cov),
                         typeof(blfm_mu), typeof(blfm_cov), typeof(cov_l), typeof(cov_u),
                         typeof(cov_mu), typeof(cov_sigma), typeof(d_mu), typeof(k_mu),
                         typeof(k_sigma),
                         # Min and max weights
                         Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}},
                         # Risk budgetting
                         typeof(risk_budget), typeof(f_risk_budget),
                         # Budget and shorting
                         typeof(short), Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}}, typeof(min_budget),
                         typeof(budget), typeof(max_budget), typeof(min_short_budget),
                         typeof(short_budget), typeof(max_short_budget),
                         # Cardinality
                         typeof(card_scale), typeof(card), typeof(a_card_ineq),
                         typeof(b_card_ineq), typeof(a_card_eq), typeof(b_card_eq),
                         # Effective assets
                         typeof(nea),
                         # Linear constraints
                         typeof(a_ineq), typeof(b_ineq), typeof(a_eq), typeof(b_eq),
                         # Tracking
                         TrackingErr,
                         # Turnover
                         AbstractTR,
                         # Adjacency
                         AdjacencyConstraint, AdjacencyConstraint,
                         # Regularisation
                         typeof(l1), typeof(l2),
                         # Fees
                         Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}},
                         # Rebalance cost
                         AbstractTR,
                         # Solution
                         typeof(constr_scale), typeof(obj_scale), typeof(model),
                         typeof(solvers), typeof(optimal), typeof(fail), typeof(limits),
                         typeof(frontier), typeof(walking), typeof(alloc_model),
                         typeof(alloc_solvers), typeof(alloc_optimal),
                         typeof(alloc_leftover), typeof(alloc_fail), typeof(alloc_walking)}(
                                                                                            # Assets and factors
                                                                                            assets,
                                                                                            timestamps,
                                                                                            returns,
                                                                                            latest_prices,
                                                                                            f_assets,
                                                                                            f_timestamps,
                                                                                            f_returns,
                                                                                            loadings,
                                                                                            regression_type,
                                                                                            # Statistics
                                                                                            mu_l,
                                                                                            mu,
                                                                                            cov,
                                                                                            cor,
                                                                                            dist,
                                                                                            clusters,
                                                                                            k,
                                                                                            min_cluster_size,
                                                                                            max_num_assets_kurt,
                                                                                            max_num_assets_kurt_scale,
                                                                                            kurt,
                                                                                            skurt,
                                                                                            L_2,
                                                                                            S_2,
                                                                                            skew,
                                                                                            V,
                                                                                            sskew,
                                                                                            SV,
                                                                                            f_mu,
                                                                                            f_cov,
                                                                                            fm_returns,
                                                                                            fm_mu,
                                                                                            fm_cov,
                                                                                            bl_bench_weights,
                                                                                            bl_mu,
                                                                                            bl_cov,
                                                                                            blfm_mu,
                                                                                            blfm_cov,
                                                                                            cov_l,
                                                                                            cov_u,
                                                                                            cov_mu,
                                                                                            cov_sigma,
                                                                                            d_mu,
                                                                                            k_mu,
                                                                                            k_sigma,
                                                                                            # Min and max weights
                                                                                            w_min,
                                                                                            w_max,
                                                                                            # Risk budgetting
                                                                                            risk_budget,
                                                                                            f_risk_budget,
                                                                                            # Budget and shorting
                                                                                            short,
                                                                                            long_l,
                                                                                            long_u,
                                                                                            short_l,
                                                                                            short_u,
                                                                                            min_budget,
                                                                                            budget,
                                                                                            max_budget,
                                                                                            min_short_budget,
                                                                                            short_budget,
                                                                                            max_short_budget,
                                                                                            # Cardinality
                                                                                            card_scale,
                                                                                            card,
                                                                                            a_card_ineq,
                                                                                            b_card_ineq,
                                                                                            a_card_eq,
                                                                                            b_card_eq,
                                                                                            # Effective assets
                                                                                            nea,
                                                                                            # Linear constraints
                                                                                            a_ineq,
                                                                                            b_ineq,
                                                                                            a_eq,
                                                                                            b_eq,
                                                                                            # Tracking
                                                                                            tracking,
                                                                                            # Turnover
                                                                                            turnover,
                                                                                            # Adjacency
                                                                                            network_adj,
                                                                                            cluster_adj,
                                                                                            # Regularisation
                                                                                            l1,
                                                                                            l2,
                                                                                            # Fees
                                                                                            long_fees,
                                                                                            short_fees,
                                                                                            # Rebalance cost
                                                                                            rebalance,
                                                                                            # Solution
                                                                                            constr_scale,
                                                                                            obj_scale,
                                                                                            model,
                                                                                            solvers,
                                                                                            optimal,
                                                                                            fail,
                                                                                            limits,
                                                                                            frontier,
                                                                                            walking,
                                                                                            alloc_model,
                                                                                            alloc_solvers,
                                                                                            alloc_optimal,
                                                                                            alloc_leftover,
                                                                                            alloc_fail,
                                                                                            alloc_walking)
end
function Base.setproperty!(port::OmniPortfolio, sym::Symbol, val)
    if sym ∈ (:latest_prices, :mu, :fm_mu, :bl_bench_weights, :bl_mu, :blfm_mu, :d_mu)
        vector_assert(val, size(port.returns, 2), sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :f_mu
        vector_assert(val, size(port.f_returns, 2), sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:cov, :cor, :dist, :V, :SV, :fm_cov, :bl_cov, :blfm_cov, :cov_l, :cov_u,
                  :cov_mu)
        N = size(port.returns, 2)
        matrix_assert(val, N, N, sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:max_num_assets_kurt, :card, :nea, :l1, :l2)
        @smart_assert(val >= zero(val))
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:kurt, :skurt, :cov_sigma)
        N = size(port.returns, 2)
        matrix_assert(val, N^2, N^2, sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:L_2, :S_2)
        N = size(port.returns, 2)
        matrix_assert(val, Int(N * (N + 1) / 2), N^2, sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:skew, :sskew)
        N = size(port.returns, 2)
        matrix_assert(val, N, N^2, sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :f_cov
        Nf = size(port.f_returns, 2)
        matrix_assert(val, Nf, Nf, sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :fm_returns
        T, N = size(port.returns)
        matrix_assert(val, T, N, sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :f_ret
        if !isempty(val)
            T = size(port.returns, 1)
            @smart_assert(size(port.returns, 1) == size(val, 1))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :max_num_assets_kurt_scale
        N = size(port.returns, 2)
        val = clamp(val, 1, N)
    elseif sym == :w_min
        if isa(val, Real)
            if isa(port.w_max, Real)
                @smart_assert(val <= port.w_max)
            elseif !isempty(port.w_max)
                @smart_assert(all(val .<= port.w_max))
            end
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(port.returns, 2))
                if isa(port.w_max, Real) || !isempty(port.w_max)
                    @smart_assert(all(val .<= port.w_max))
                end
            end
        end
    elseif sym == :w_max
        if isa(val, Real)
            if isa(port.w_min, Real)
                @smart_assert(val >= port.w_min)
            elseif !isempty(port.w_min)
                @smart_assert(all(val .>= port.w_min))
            end
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(port.returns, 2))
                if isa(port.w_min, Real) || !isempty(port.w_min)
                    @smart_assert(all(val .>= port.w_min))
                end
            end
        end
    elseif sym == :risk_budget
        N = size(port.returns, 2)
        val = risk_budget_assert(val, N, sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :f_risk_budget
        Nf = size(port.f_returns, 2)
        val = factor_risk_budget_assert(val, Nf, sym)
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :long_l
        N = size(port.returns, 2)
        long_short_budget_assert(N, val, port.long_u, port.min_budget, port.budget,
                                 port.max_budget, port.short, port.short_l, port.short_u,
                                 port.min_short_budget, port.short_budget,
                                 port.max_short_budget)
    elseif sym == :long_u
        N = size(port.returns, 2)
        long_short_budget_assert(N, port.long_l, val, port.min_budget, port.budget,
                                 port.max_budget, port.short, port.short_l, port.short_u,
                                 port.min_short_budget, port.short_budget,
                                 port.max_short_budget)
    elseif sym == :min_budget
        N = size(port.returns, 2)
        long_short_budget_assert(N, port.long_l, port.long_u, val, port.budget,
                                 port.max_budget, port.short, port.short_l, port.short_u,
                                 port.min_short_budget, port.short_budget,
                                 port.max_short_budget)
    elseif sym == :budget
        N = size(port.returns, 2)
        val = long_short_budget_assert(N, port.long_l, port.long_u, port.min_budget, val,
                                       port.max_budget, port.short, port.short_l,
                                       port.short_u, port.min_short_budget,
                                       port.short_budget, port.max_short_budget)[1]
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :max_budget
        N = size(port.returns, 2)
        long_short_budget_assert(N, port.long_l, port.long_u, port.min_budget, port.budget,
                                 val, port.short, port.short_l, port.short_u,
                                 port.min_short_budget, port.short_budget,
                                 port.max_short_budget)
    elseif sym == :short_l
        N = size(port.returns, 2)
        long_short_budget_assert(N, port.long_l, port.long_u, port.min_budget, port.budget,
                                 port.max_budget, port.short, val, port.short_u,
                                 port.min_short_budget, port.short_budget,
                                 port.max_short_budget)
    elseif sym == :short_u
        N = size(port.returns, 2)
        long_short_budget_assert(N, port.long_l, port.long_u, port.min_budget, port.budget,
                                 port.max_budget, port.short, port.short_l, val,
                                 port.min_short_budget, port.short_budget,
                                 port.max_short_budget)
    elseif sym == :min_short_budget
        N = size(port.returns, 2)
        long_short_budget_assert(N, port.long_l, port.long_u, port.min_budget, port.budget,
                                 port.max_budget, port.short, port.short_l, port.short_u,
                                 val, port.short_budget, port.max_short_budget)
    elseif sym == :short_budget
        N = size(port.returns, 2)
        val = long_short_budget_assert(N, port.long_l, port.long_u, port.min_budget,
                                       port.budget, port.max_budget, port.short,
                                       port.short_l, port.short_u, port.min_short_budget,
                                       val, port.max_short_budget)[2]
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :max_short_budget
        N = size(port.returns, 2)
        long_short_budget_assert(N, port.long_l, port.long_u, port.min_budget, port.budget,
                                 port.max_budget, port.short, port.short_l, port.short_u,
                                 port.min_short_budget, port.short_budget, val)
    elseif sym == :a_card_ineq
        N = size(port.returns, 2)
        linear_constraint_assert(val, port.b_card_ineq, N, "card_ineq")
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :b_card_ineq
        N = size(port.returns, 2)
        linear_constraint_assert(port.a_card_ineq, val, N, "card_ineq")
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :a_card_eq
        N = size(port.returns, 2)
        linear_constraint_assert(val, port.b_card_eq, N, "card_eq")
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :b_card_eq
        N = size(port.returns, 2)
        linear_constraint_assert(port.a_card_eq, :b_card_eq, N, "card_eq")
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :a_ineq
        N = size(port.returns, 2)
        linear_constraint_assert(val, port.b_ineq, N, "ineq")
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :b_ineq
        N = size(port.returns, 2)
        linear_constraint_assert(port.a_ineq, val, N, "ineq")
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :a_eq
        N = size(port.returns, 2)
        linear_constraint_assert(val, port.b_eq, N, "eq")
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :b_eq
        N = size(port.returns, 2)
        linear_constraint_assert(port.a_eq, val, N, "eq")
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :tracking
        T, N = size(port.returns)
        tracking_assert(val, T, N)
    elseif sym == :turnover
        N = size(port.returns, 2)
        tr_assert(val, N)
    elseif sym ∈ (:network_adj, :cluster_adj)
        N = size(port.returns, 2)
        adj_assert(val, N)
    elseif sym ∈ (:rebalance, :turnover)
        if isa(val, TR)
            if isa(val.val, Real)
                @smart_assert(val.val >= zero(val.val))
            elseif isa(val.val, AbstractVector) && !isempty(val.val)
                @smart_assert(length(val.val) == size(port.returns, 2) &&
                              all(val.val .>= zero(val.val)))
            end
            if !isempty(val.w)
                @smart_assert(length(val.w) == size(port.returns, 2))
            end
        end
    elseif sym ∈ (:constr_scale, :obj_scale)
        @smart_assert(val > zero(val))
    else
        if (isa(getfield(port, sym), AbstractArray) && isa(val, AbstractArray)) ||
           (isa(getfield(port, sym), Real) && isa(val, Real))
            val = convert(typeof(getfield(port, sym)), val)
        end
    end
    return setfield!(port, sym, val)
end
function Base.deepcopy(port::OmniPortfolio)
    return OmniPortfolio{
                         # Assets and factors
                         typeof(port.assets), typeof(port.timestamps), typeof(port.returns),
                         typeof(port.latest_prices), typeof(port.f_assets),
                         typeof(port.f_timestamps), typeof(port.f_returns),
                         typeof(port.loadings), Union{<:RegressionType, Nothing},
                         # Statistics
                         typeof(port.mu_l), typeof(port.mu), typeof(port.cov),
                         typeof(port.cor), typeof(port.dist), typeof(port.clusters),
                         typeof(port.k), typeof(port.min_cluster_size),
                         typeof(port.max_num_assets_kurt),
                         typeof(port.max_num_assets_kurt_scale), typeof(port.kurt),
                         typeof(port.skurt), typeof(port.L_2), typeof(port.S_2),
                         typeof(port.skew), typeof(port.V), typeof(port.sskew),
                         typeof(port.SV), typeof(port.f_mu), typeof(port.f_cov),
                         typeof(port.fm_returns), typeof(port.fm_mu), typeof(port.fm_cov),
                         typeof(port.bl_bench_weights), typeof(port.bl_mu),
                         typeof(port.bl_cov), typeof(port.blfm_mu), typeof(port.blfm_cov),
                         typeof(port.cov_l), typeof(port.cov_u), typeof(port.cov_mu),
                         typeof(port.cov_sigma), typeof(port.d_mu), typeof(port.k_mu),
                         typeof(port.k_sigma),
                         # Min and max weights
                         Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}},
                         # Risk budgetting
                         typeof(port.risk_budget), typeof(port.f_risk_budget),
                         # Budget and shorting
                         typeof(port.short), Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}}, typeof(port.min_budget),
                         typeof(port.budget), typeof(port.max_budget),
                         typeof(port.min_short_budget), typeof(port.short_budget),
                         typeof(port.max_short_budget),
                         # Cardinality
                         typeof(port.card_scale), typeof(port.card),
                         typeof(port.a_card_ineq), typeof(port.b_card_ineq),
                         typeof(port.a_card_eq), typeof(port.b_card_eq),
                         # Effective assets
                         typeof(port.nea),
                         # Linear constraints
                         typeof(port.a_ineq), typeof(port.b_ineq), typeof(port.a_eq),
                         typeof(port.b_eq),
                         # Tracking
                         TrackingErr,
                         # Turnover
                         AbstractTR,
                         # Adjacency
                         AdjacencyConstraint, AdjacencyConstraint,
                         # Regularisation
                         typeof(port.l1), typeof(port.l2),
                         # Fees
                         Union{<:Real, <:AbstractVector{<:Real}},
                         Union{<:Real, <:AbstractVector{<:Real}},
                         # Rebalance cost
                         AbstractTR,
                         # Solution
                         typeof(port.constr_scale), typeof(port.obj_scale),
                         typeof(port.model), typeof(port.solvers), typeof(port.optimal),
                         typeof(port.fail), typeof(port.limits), typeof(port.frontier),
                         typeof(port.walking), typeof(port.alloc_model),
                         typeof(port.alloc_solvers), typeof(port.alloc_optimal),
                         typeof(port.alloc_leftover), typeof(port.alloc_fail),
                         typeof(port.alloc_walking)}(
                                                     # Assets and factors
                                                     deepcopy(port.assets),
                                                     deepcopy(port.timestamps),
                                                     deepcopy(port.returns),
                                                     deepcopy(port.latest_prices),
                                                     deepcopy(port.f_assets),
                                                     deepcopy(port.f_timestamps),
                                                     deepcopy(port.f_returns),
                                                     deepcopy(port.loadings),
                                                     deepcopy(port.regression_type),
                                                     # Statistics
                                                     deepcopy(port.mu_l), deepcopy(port.mu),
                                                     deepcopy(port.cov), deepcopy(port.cor),
                                                     deepcopy(port.dist),
                                                     deepcopy(port.clusters),
                                                     deepcopy(port.k),
                                                     deepcopy(port.min_cluster_size),
                                                     deepcopy(port.max_num_assets_kurt),
                                                     deepcopy(port.max_num_assets_kurt_scale),
                                                     deepcopy(port.kurt),
                                                     deepcopy(port.skurt),
                                                     deepcopy(port.L_2), deepcopy(port.S_2),
                                                     deepcopy(port.skew), deepcopy(port.V),
                                                     deepcopy(port.sskew),
                                                     deepcopy(port.SV), deepcopy(port.f_mu),
                                                     deepcopy(port.f_cov),
                                                     deepcopy(port.fm_returns),
                                                     deepcopy(port.fm_mu),
                                                     deepcopy(port.fm_cov),
                                                     deepcopy(port.bl_bench_weights),
                                                     deepcopy(port.bl_mu),
                                                     deepcopy(port.bl_cov),
                                                     deepcopy(port.blfm_mu),
                                                     deepcopy(port.blfm_cov),
                                                     deepcopy(port.cov_l),
                                                     deepcopy(port.cov_u),
                                                     deepcopy(port.cov_mu),
                                                     deepcopy(port.cov_sigma),
                                                     deepcopy(port.d_mu),
                                                     deepcopy(port.k_mu),
                                                     deepcopy(port.k_sigma),
                                                     # Min and max weights
                                                     deepcopy(port.w_min),
                                                     deepcopy(port.w_max),
                                                     # Risk budgetting
                                                     deepcopy(port.risk_budget),
                                                     deepcopy(port.f_risk_budget),
                                                     # Budget and shorting
                                                     deepcopy(port.short),
                                                     deepcopy(port.long_l),
                                                     deepcopy(port.long_u),
                                                     deepcopy(port.short_l),
                                                     deepcopy(port.short_u),
                                                     deepcopy(port.min_budget),
                                                     deepcopy(port.budget),
                                                     deepcopy(port.max_budget),
                                                     deepcopy(port.min_short_budget),
                                                     deepcopy(port.short_budget),
                                                     deepcopy(port.max_short_budget),
                                                     # Cardinality
                                                     deepcopy(port.card_scale),
                                                     deepcopy(port.card),
                                                     deepcopy(port.a_card_ineq),
                                                     deepcopy(port.b_card_ineq),
                                                     deepcopy(port.a_card_eq),
                                                     deepcopy(port.b_card_eq),
                                                     # Effective assets
                                                     deepcopy(port.nea),
                                                     # Linear constraints
                                                     deepcopy(port.a_ineq),
                                                     deepcopy(port.b_ineq),
                                                     deepcopy(port.a_eq),
                                                     deepcopy(port.b_eq),
                                                     # Tracking
                                                     deepcopy(port.tracking),
                                                     # Turnover
                                                     deepcopy(port.turnover),
                                                     # Adjacency
                                                     deepcopy(port.network_adj),
                                                     deepcopy(port.cluster_adj),
                                                     # Regularisation
                                                     deepcopy(port.l1), deepcopy(port.l2),
                                                     # Fees
                                                     deepcopy(port.long_fees),
                                                     deepcopy(port.short_fees),
                                                     # Rebalance cost
                                                     deepcopy(port.rebalance),
                                                     # Solution
                                                     deepcopy(port.constr_scale),
                                                     deepcopy(port.obj_scale),
                                                     deepcopy(port.model),
                                                     deepcopy(port.solvers),
                                                     deepcopy(port.optimal),
                                                     deepcopy(port.fail),
                                                     deepcopy(port.limits),
                                                     deepcopy(port.frontier),
                                                     deepcopy(port.walking),
                                                     deepcopy(port.alloc_model),
                                                     deepcopy(port.alloc_solvers),
                                                     deepcopy(port.alloc_optimal),
                                                     deepcopy(port.alloc_leftover),
                                                     deepcopy(port.alloc_fail),
                                                     deepcopy(port.alloc_walking))
end

export OmniPortfolio
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
                         tsfee, nal, nau, naus, mnak, mnaks, l1t, l2t, rb, to, kte, asmi,
                         bsvi, acmi, bcvi, ami, bvi, rbv, frbv, nm, cm, amc, bvc, ler, tmu,
                         tcov, tkurt, tskurt, tl2, ts2, tskew, tv, tsskew, tsv, tmuf, tcovf,
                         trfm, tmufm, tcovfm, blbw, tmubl, tcovbl, tmublf, tcovblf, tcovl,
                         tcovu, tcovmu, tcovs, tdmu, tkmu, tks, topt, tlim, tfront, tsolv,
                         tf, tos, tmod, tlp, taopt, talo, tasolv, taf, tamod} <:
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
    a_smtx_ineq::asmi
    b_svec_ineq::bsvi
    a_cmtx_ineq::acmi
    b_cvec_ineq::bcvi
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
    bl_bench_weights::blbw
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
    obj_scale::tos
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
                   a_smtx_ineq::Union{<:BitMatrix, SparseMatrixCSC{Bool, <:Integer}} = BitMatrix(undef,
                                                                                                 0,
                                                                                                 0),
                   b_svec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   a_cmtx_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   b_cvec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
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
                   bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
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
                   obj_scale::Real = 1.0, model::JuMP.Model = JuMP.Model(),
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
        @smart_assert(tracking_err.err >= zero(tracking_err.err))
    end
    if isa(tracking_err, TrackRet)
        @smart_assert(tracking_err.w == size(returns, 1))
        @smart_assert(tracking_err.err >= zero(tracking_err.err))
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
    if !isempty(a_smtx_ineq)
        if !(isa(a_smtx_ineq, BitMatrix) ||
             isa(a_smtx_ineq, SparseMatrixCSC{Bool, <:Integer}))
            a_smtx_ineq = BitMatrix(a_smtx_ineq)
        end
        @smart_assert(size(a_smtx_ineq, 2) == size(returns, 2))
    end
    if !isempty(a_cmtx_ineq)
        @smart_assert(size(a_cmtx_ineq, 2) == size(returns, 2))
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
    if !isempty(bl_bench_weights)
        @smart_assert(length(bl_bench_weights) == size(returns, 2))
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
                     typeof(a_smtx_ineq), typeof(b_svec_ineq), typeof(a_cmtx_ineq),
                     typeof(b_cvec_ineq), typeof(a_mtx_ineq), typeof(b_vec_ineq),
                     typeof(risk_budget), typeof(f_risk_budget), AdjacencyConstraint,
                     AdjacencyConstraint, typeof(a_vec_cent), typeof(b_cent), typeof(mu_l),
                     typeof(mu), typeof(cov), typeof(kurt), typeof(skurt), typeof(L_2),
                     typeof(S_2), typeof(skew), typeof(V), typeof(sskew), typeof(SV),
                     typeof(f_mu), typeof(f_cov), typeof(fm_returns), typeof(fm_mu),
                     typeof(fm_cov), typeof(bl_bench_weights), typeof(bl_mu),
                     typeof(bl_cov), typeof(blfm_mu), typeof(blfm_cov), typeof(cov_l),
                     typeof(cov_u), typeof(cov_mu), typeof(cov_sigma), typeof(d_mu),
                     typeof(k_mu), typeof(k_sigma), typeof(optimal), typeof(limits),
                     typeof(frontier), typeof(solvers), typeof(fail), typeof(obj_scale),
                     typeof(model), typeof(latest_prices), typeof(alloc_optimal),
                     typeof(alloc_leftover), typeof(alloc_solvers), typeof(alloc_fail),
                     typeof(alloc_model)}(assets, timestamps, returns, f_assets,
                                          f_timestamps, f_returns, loadings,
                                          regression_type, short, budget, short_budget,
                                          long_u, short_u, fees, short_fees, num_assets_l,
                                          num_assets_u, num_assets_u_scale,
                                          max_num_assets_kurt, max_num_assets_kurt_scale,
                                          l1, l2, rebalance, turnover, tracking_err,
                                          a_smtx_ineq, b_svec_ineq, a_cmtx_ineq,
                                          b_cvec_ineq, a_mtx_ineq, b_vec_ineq, risk_budget,
                                          f_risk_budget, network_adj, cluster_adj,
                                          a_vec_cent, b_cent, mu_l, mu, cov, kurt, skurt,
                                          L_2, S_2, skew, V, sskew, SV, f_mu, f_cov,
                                          fm_returns, fm_mu, fm_cov, bl_bench_weights,
                                          bl_mu, bl_cov, blfm_mu, blfm_cov, cov_l, cov_u,
                                          cov_mu, cov_sigma, d_mu, k_mu, k_sigma, optimal,
                                          limits, frontier, solvers, fail, obj_scale, model,
                                          latest_prices, alloc_optimal, alloc_leftover,
                                          alloc_solvers, alloc_fail, alloc_model)
end
function Base.setproperty!(port::Portfolio, sym::Symbol, val)
    if sym == :short_u
        if port.short
            @smart_assert(port.short_budget >=
                          val >=
                          zero(promote_type(typeof(port.short_budget), typeof(val))))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :short_budget
        if port.short
            @smart_assert(val >=
                          port.short_u >=
                          zero(promote_type(typeof(val), typeof(port.short_u))))

            @smart_assert(port.budget + val >=
                          port.long_u >=
                          zero(promote_type(typeof(port.budget), typeof(val),
                                            typeof(port.long_u))))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :long_u
        if port.short
            @smart_assert(port.budget + port.short_budget >=
                          val >=
                          zero(promote_type(typeof(port.budget), typeof(port.short_budget),
                                            typeof(val))))
        else
            @smart_assert(port.budget >=
                          val >=
                          zero(promote_type(typeof(port.budget), typeof(val))))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :budget
        if port.short
            @smart_assert(val + port.short_budget >=
                          port.long_u >=
                          zero(promote_type(typeof(val), typeof(port.short_budget),
                                            typeof(port.long_u))))
        else
            @smart_assert(val >=
                          port.long_u >=
                          zero(promote_type(typeof(val), typeof(port.long_u))))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :short
        if val
            @smart_assert(port.short_budget >=
                          port.short_u >=
                          zero(promote_type(typeof(port.short_budget),
                                            typeof(port.short_u))))

            @smart_assert(port.budget + port.short_budget >=
                          port.long_u >=
                          zero(promote_type(typeof(port.budget), typeof(port.short_budget),
                                            typeof(port.long_u))))
        else
            @smart_assert(port.budget >=
                          port.long_u >=
                          zero(promote_type(typeof(port.budget), typeof(port.long_u))))
        end
    elseif sym == :max_num_assets_kurt
        @smart_assert(val >= zero(val))
    elseif sym == :max_num_assets_kurt_scale
        val = clamp(val, 1, size(port.returns, 2))
    elseif sym ∈ (:fees, :short_fees)
        if isa(val, AbstractVector) && !isempty(val)
            @smart_assert(length(val) == size(port.returns, 2))
            val = collect(eltype(port.returns), val)
        end
    elseif sym ∈ (:rebalance, :turnover)
        if isa(val, TR)
            if isa(val.val, Real)
                @smart_assert(val.val >= zero(val.val))
            elseif isa(val.val, AbstractVector) && !isempty(val.val)
                @smart_assert(length(val.val) == size(port.returns, 2) &&
                              all(val.val .>= zero(val.val)))
            end
            if !isempty(val.w)
                @smart_assert(length(val.w) == size(port.returns, 2))
            end
        end
    elseif sym == :tracking_err
        if isa(val, TrackWeight)
            @smart_assert(length(val.w) == size(port.returns, 2))
            @smart_assert(val.err >= zero(val.err))
        elseif isa(val, TrackRet)
            @smart_assert(length(val.w) == size(port.returns, 1))
            @smart_assert(val.err >= zero(val.err))
        end
    elseif sym ∈ (:a_mtx_ineq, :a_cmtx_ineq)
        if !isempty(val)
            @smart_assert(size(val, 2) == size(port.returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :a_smtx_ineq
        if !(isa(val, BitMatrix) || isa(val, SparseMatrixCSC{Bool, <:Integer}))
            val = if isa(getfield(port, sym), BitMatrix)
                BitMatrix(val)
            else
                sparse(BitMatrix(val))
            end
        end
        if !isempty(val)
            @smart_assert(size(val, 2) == size(port.returns, 2))
        end
    elseif sym ∈ (:network_adj, :cluster_adj)
        if !isa(val, NoAdj) && !isempty(val.A)
            if isa(val, IP)
                @smart_assert(size(val.A, 2) == size(port.returns, 2))
            else
                @smart_assert(size(val.A) == (size(port.returns, 2), size(port.returns, 2)))
            end
        end
    elseif sym == :a_vec_cent
        if !isempty(val)
            @smart_assert(size(val, 1) == size(port.returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :f_mu
        if !isempty(val)
            @smart_assert(length(val) == size(port.f_returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :f_cov
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(port.f_returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:L_2, :S_2)
        if !isempty(val)
            N = size(port.returns, 2)
            @smart_assert(size(val) == (Int(N * (N + 1) / 2), N^2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :bl_bench_weights
        if !isempty(val)
            @smart_assert(length(val) == size(port.returns, 2))
            if isa(val, AbstractRange)
                val = collect(val)
            end
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :risk_budget
        if isempty(val)
            N = size(port.returns, 2)
            val = fill(inv(N), N)
        else
            @smart_assert(length(val) == size(port.returns, 2) &&
                          all(val .>= zero(eltype(port.returns))))

            if isa(val, AbstractRange)
                val = collect(val / sum(val))
            else
                val ./= sum(val)
            end
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :f_risk_budget
        if !isempty(val)
            if isa(val, AbstractRange)
                val = collect(val / sum(val))
            else
                val ./= sum(val)
            end
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:num_assets_l, :num_assets_u, :num_assets_u_scale)
        @smart_assert(val >= zero(val))
    elseif sym ∈ (:kurt, :skurt, :cov_sigma)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(port.returns, 2)^2)
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:skew, :sskew)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(port.returns, 2) &&
                          size(val, 2) == size(port.returns, 2)^2)
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns,
                  :fm_returns)
        if !isempty(val) && !isempty(getfield(port, sym))
            @smart_assert(size(val) == size(getfield(port, sym)))
        end
    elseif sym ∈ (:mu, :fm_mu, :bl_mu, :blfm_mu, :d_mu, :latest_prices)
        if !isempty(val)
            @smart_assert(length(val) == size(port.returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:cov, :fm_cov, :bl_cov, :blfm_cov, :cov_l, :cov_u, :cov_mu, :V, :SV)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(port.returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    else
        if (isa(getfield(port, sym), AbstractArray) && isa(val, AbstractArray)) ||
           (isa(getfield(port, sym), Real) && isa(val, Real))
            val = convert(typeof(getfield(port, sym)), val)
        end
    end
    return setfield!(port, sym, val)
end
function Base.deepcopy(port::Portfolio)
    return Portfolio{typeof(port.assets), typeof(port.timestamps), typeof(port.returns),
                     typeof(port.f_assets), typeof(port.f_timestamps),
                     typeof(port.f_returns), typeof(port.loadings),
                     Union{<:RegressionType, Nothing}, typeof(port.short),
                     typeof(port.budget), typeof(port.short_budget), typeof(port.long_u),
                     typeof(port.short_u), Union{<:Real, <:AbstractVector{<:Real}},
                     Union{<:Real, <:AbstractVector{<:Real}}, typeof(port.num_assets_l),
                     typeof(port.num_assets_u), typeof(port.num_assets_u_scale),
                     typeof(port.max_num_assets_kurt),
                     typeof(port.max_num_assets_kurt_scale), typeof(port.l1),
                     typeof(port.l2), AbstractTR, AbstractTR, TrackingErr,
                     typeof(port.a_smtx_ineq), typeof(port.b_svec_ineq),
                     typeof(port.a_cmtx_ineq), typeof(port.b_cvec_ineq),
                     typeof(port.a_mtx_ineq), typeof(port.b_vec_ineq),
                     typeof(port.risk_budget), typeof(port.f_risk_budget),
                     AdjacencyConstraint, AdjacencyConstraint, typeof(port.a_vec_cent),
                     typeof(port.b_cent), typeof(port.mu_l), typeof(port.mu),
                     typeof(port.cov), typeof(port.kurt), typeof(port.skurt),
                     typeof(port.L_2), typeof(port.S_2), typeof(port.skew), typeof(port.V),
                     typeof(port.sskew), typeof(port.SV), typeof(port.f_mu),
                     typeof(port.f_cov), typeof(port.fm_returns), typeof(port.fm_mu),
                     typeof(port.fm_cov), typeof(port.bl_bench_weights), typeof(port.bl_mu),
                     typeof(port.bl_cov), typeof(port.blfm_mu), typeof(port.blfm_cov),
                     typeof(port.cov_l), typeof(port.cov_u), typeof(port.cov_mu),
                     typeof(port.cov_sigma), typeof(port.d_mu), typeof(port.k_mu),
                     typeof(port.k_sigma), typeof(port.optimal), typeof(port.limits),
                     typeof(port.frontier), typeof(port.solvers), typeof(port.fail),
                     typeof(port.obj_scale), typeof(port.model), typeof(port.latest_prices),
                     typeof(port.alloc_optimal), typeof(port.alloc_leftover),
                     typeof(port.alloc_solvers), typeof(port.alloc_fail),
                     typeof(port.alloc_model)}(deepcopy(port.assets),
                                               deepcopy(port.timestamps),
                                               deepcopy(port.returns),
                                               deepcopy(port.f_assets),
                                               deepcopy(port.f_timestamps),
                                               deepcopy(port.f_returns),
                                               deepcopy(port.loadings),
                                               deepcopy(port.regression_type),
                                               deepcopy(port.short), deepcopy(port.budget),
                                               deepcopy(port.short_budget),
                                               deepcopy(port.long_u),
                                               deepcopy(port.short_u), deepcopy(port.fees),
                                               deepcopy(port.short_fees),
                                               deepcopy(port.num_assets_l),
                                               deepcopy(port.num_assets_u),
                                               deepcopy(port.num_assets_u_scale),
                                               deepcopy(port.max_num_assets_kurt),
                                               deepcopy(port.max_num_assets_kurt_scale),
                                               deepcopy(port.l1), deepcopy(port.l2),
                                               deepcopy(port.rebalance),
                                               deepcopy(port.turnover),
                                               deepcopy(port.tracking_err),
                                               deepcopy(port.a_smtx_ineq),
                                               deepcopy(port.b_svec_ineq),
                                               deepcopy(port.a_cmtx_ineq),
                                               deepcopy(port.b_cvec_ineq),
                                               deepcopy(port.a_mtx_ineq),
                                               deepcopy(port.b_vec_ineq),
                                               deepcopy(port.risk_budget),
                                               deepcopy(port.f_risk_budget),
                                               deepcopy(port.network_adj),
                                               deepcopy(port.cluster_adj),
                                               deepcopy(port.a_vec_cent),
                                               deepcopy(port.b_cent), deepcopy(port.mu_l),
                                               deepcopy(port.mu), deepcopy(port.cov),
                                               deepcopy(port.kurt), deepcopy(port.skurt),
                                               deepcopy(port.L_2), deepcopy(port.S_2),
                                               deepcopy(port.skew), deepcopy(port.V),
                                               deepcopy(port.sskew), deepcopy(port.SV),
                                               deepcopy(port.f_mu), deepcopy(port.f_cov),
                                               deepcopy(port.fm_returns),
                                               deepcopy(port.fm_mu), deepcopy(port.fm_cov),
                                               deepcopy(port.bl_bench_weights),
                                               deepcopy(port.bl_mu), deepcopy(port.bl_cov),
                                               deepcopy(port.blfm_mu),
                                               deepcopy(port.blfm_cov),
                                               deepcopy(port.cov_l), deepcopy(port.cov_u),
                                               deepcopy(port.cov_mu),
                                               deepcopy(port.cov_sigma),
                                               deepcopy(port.d_mu), deepcopy(port.k_mu),
                                               deepcopy(port.k_sigma),
                                               deepcopy(port.optimal),
                                               deepcopy(port.limits),
                                               deepcopy(port.frontier),
                                               deepcopy(port.solvers), deepcopy(port.fail),
                                               deepcopy(port.obj_scale), copy(port.model),
                                               deepcopy(port.latest_prices),
                                               deepcopy(port.alloc_optimal),
                                               deepcopy(port.alloc_leftover),
                                               deepcopy(port.alloc_solvers),
                                               deepcopy(port.alloc_fail),
                                               copy(port.alloc_model))
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
mutable struct HCPortfolio{ast, dat, r, tfa, tfdat, tretf, l, lo, tmu, tcov, tkurt, tskurt,
                           tl2, ts2, tskew, tv, tsskew, tsv, tmuf, tcovf, trfm, tmufm,
                           tcovfm, blbw, tmubl, tcovbl, tmublf, tcovblf, wmi, wma, tco,
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
    bl_bench_weights::blbw
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
                     bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
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
    if !isempty(bl_bench_weights)
        @smart_assert(length(bl_bench_weights) == size(returns, 2))
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
                       typeof(loadings), Union{<:RegressionType, Nothing}, typeof(mu),
                       typeof(cov), typeof(kurt), typeof(skurt), typeof(L_2), typeof(S_2),
                       typeof(skew), typeof(V), typeof(sskew), typeof(SV), typeof(f_mu),
                       typeof(f_cov), typeof(fm_returns), typeof(fm_mu), typeof(fm_cov),
                       typeof(bl_bench_weights), typeof(bl_mu), typeof(bl_cov),
                       typeof(blfm_mu), typeof(blfm_cov),
                       Union{<:Real, <:AbstractVector{<:Real}},
                       Union{<:Real, <:AbstractVector{<:Real}}, typeof(cor), typeof(dist),
                       typeof(clusters), typeof(k), typeof(optimal), typeof(solvers),
                       typeof(fail), typeof(latest_prices), typeof(alloc_optimal),
                       typeof(alloc_leftover), typeof(alloc_solvers), typeof(alloc_fail),
                       typeof(alloc_model)}(assets, timestamps, returns, f_assets,
                                            f_timestamps, f_returns, loadings,
                                            regression_type, mu, cov, kurt, skurt, L_2, S_2,
                                            skew, V, sskew, SV, f_mu, f_cov, fm_returns,
                                            fm_mu, fm_cov, bl_bench_weights, bl_mu, bl_cov,
                                            blfm_mu, blfm_cov, w_min, w_max, cor, dist,
                                            clusters, k, optimal, solvers, fail,
                                            latest_prices, alloc_optimal, alloc_leftover,
                                            alloc_solvers, alloc_fail, alloc_model)
end
function Base.setproperty!(port::HCPortfolio, sym::Symbol, val)
    if sym == :k
        @smart_assert(val >= zero(val))
    elseif sym == :w_min
        if isa(val, Real)
            if isa(port.w_max, Real)
                @smart_assert(val <= port.w_max)
            elseif !isempty(port.w_max)
                @smart_assert(all(val .<= port.w_max))
            end
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(port.returns, 2))
                if isa(port.w_max, Real) || !isempty(port.w_max)
                    @smart_assert(all(val .<= port.w_max))
                end
            end
        end
    elseif sym == :w_max
        if isa(val, Real)
            if isa(port.w_min, Real)
                @smart_assert(val >= port.w_min)
            elseif !isempty(port.w_min)
                @smart_assert(all(val .>= port.w_min))
            end
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(port.returns, 2))
                if isa(port.w_min, Real) || !isempty(port.w_min)
                    @smart_assert(all(val .>= port.w_min))
                end
            end
        end
    elseif sym ∈ (:mu, :fm_mu, :bl_mu, :blfm_mu, :latest_prices)
        if !isempty(val)
            @smart_assert(length(val) == size(port.returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:kurt, :skurt)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(port.returns, 2)^2)
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:skew, :sskew)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(port.returns, 2) &&
                          size(val, 2) == size(port.returns, 2)^2)
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns,
                  :fm_returns)
        if !isempty(val) && !isempty(getfield(port, sym))
            @smart_assert(size(val) == size(getfield(port, sym)))
        end
    elseif sym == :f_mu
        if !isempty(val)
            @smart_assert(length(val) == size(port.f_returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :f_cov
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(port.f_returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:L_2, :S_2)
        if !isempty(val)
            N = size(port.returns, 2)
            @smart_assert(size(val) == (Int(N * (N + 1) / 2), N^2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym == :bl_bench_weights
        if !isempty(val)
            @smart_assert(length(val) == size(port.returns, 2))
            if isa(val, AbstractRange)
                val = collect(val)
            end
        end
        val = convert(typeof(getfield(port, sym)), val)
    elseif sym ∈ (:cov, :cor, :dist, :V, :SV)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(port.returns, 2))
        end
        val = convert(typeof(getfield(port, sym)), val)
    end
    return setfield!(port, sym, val)
end
function Base.deepcopy(port::HCPortfolio)
    return HCPortfolio{typeof(port.assets), typeof(port.timestamps), typeof(port.returns),
                       typeof(port.f_assets), typeof(port.f_timestamps),
                       typeof(port.f_returns), typeof(port.loadings),
                       Union{<:RegressionType, Nothing}, typeof(port.mu), typeof(port.cov),
                       typeof(port.kurt), typeof(port.skurt), typeof(port.L_2),
                       typeof(port.S_2), typeof(port.skew), typeof(port.V),
                       typeof(port.sskew), typeof(port.SV), typeof(port.f_mu),
                       typeof(port.f_cov), typeof(port.fm_returns), typeof(port.fm_mu),
                       typeof(port.fm_cov), typeof(port.bl_bench_weights),
                       typeof(port.bl_mu), typeof(port.bl_cov), typeof(port.blfm_mu),
                       typeof(port.blfm_cov), Union{<:Real, <:AbstractVector{<:Real}},
                       Union{<:Real, <:AbstractVector{<:Real}}, typeof(port.cor),
                       typeof(port.dist), typeof(port.clusters), typeof(port.k),
                       typeof(port.optimal), typeof(port.solvers), typeof(port.fail),
                       typeof(port.latest_prices), typeof(port.alloc_optimal),
                       typeof(port.alloc_leftover), typeof(port.alloc_solvers),
                       typeof(port.alloc_fail), typeof(port.alloc_model)}(deepcopy(port.assets),
                                                                          deepcopy(port.timestamps),
                                                                          deepcopy(port.returns),
                                                                          deepcopy(port.f_assets),
                                                                          deepcopy(port.f_timestamps),
                                                                          deepcopy(port.f_returns),
                                                                          deepcopy(port.loadings),
                                                                          deepcopy(port.regression_type),
                                                                          deepcopy(port.mu),
                                                                          deepcopy(port.cov),
                                                                          deepcopy(port.kurt),
                                                                          deepcopy(port.skurt),
                                                                          deepcopy(port.L_2),
                                                                          deepcopy(port.S_2),
                                                                          deepcopy(port.skew),
                                                                          deepcopy(port.V),
                                                                          deepcopy(port.sskew),
                                                                          deepcopy(port.SV),
                                                                          deepcopy(port.f_mu),
                                                                          deepcopy(port.f_cov),
                                                                          deepcopy(port.fm_returns),
                                                                          deepcopy(port.fm_mu),
                                                                          deepcopy(port.fm_cov),
                                                                          deepcopy(port.bl_bench_weights),
                                                                          deepcopy(port.bl_mu),
                                                                          deepcopy(port.bl_cov),
                                                                          deepcopy(port.blfm_mu),
                                                                          deepcopy(port.blfm_cov),
                                                                          deepcopy(port.w_min),
                                                                          deepcopy(port.w_max),
                                                                          deepcopy(port.cor),
                                                                          deepcopy(port.dist),
                                                                          deepcopy(port.clusters),
                                                                          deepcopy(port.k),
                                                                          deepcopy(port.optimal),
                                                                          deepcopy(port.solvers),
                                                                          deepcopy(port.fail),
                                                                          deepcopy(port.latest_prices),
                                                                          deepcopy(port.alloc_optimal),
                                                                          deepcopy(port.alloc_leftover),
                                                                          deepcopy(port.alloc_solvers),
                                                                          deepcopy(port.alloc_fail),
                                                                          copy(port.alloc_model))
end

export Portfolio, HCPortfolio
