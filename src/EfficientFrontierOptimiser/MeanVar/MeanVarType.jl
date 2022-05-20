abstract type AbstractEfficientMeanVar <: AbstractEfficient end

struct EfficientMeanVar{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14} <:
       AbstractEfficientMeanVar
    tickers::T1
    mean_ret::T2
    weights::T3
    cov_mtx::T4
    rf::T5
    market_neutral::T6
    risk_aversion::T7
    target_volatility::T8
    target_ret::T9
    extra_vars::T10
    extra_constraints::T11
    extra_obj_terms::T12
    extra_obj_func::T13
    model::T14
end
function EfficientMeanVar(
    tickers,
    mean_ret,
    cov_mtx;
    weight_bounds = (0, 1),
    rf = 0.02,
    market_neutral = false,
    risk_aversion = 1.0,
    target_volatility = rank(cov_mtx) < size(cov_mtx, 1) ? 1 / sum(diag(cov_mtx)) :
                        sqrt(1 / sum(inv(cov_mtx))),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
    extra_obj_func = [],
)
    return EfficientMeanVar(
        tickers,
        mean_ret,
        cov_mtx,
        weight_bounds,
        rf,
        market_neutral,
        risk_aversion,
        target_volatility,
        target_ret,
        extra_vars,
        extra_constraints,
        extra_obj_terms,
        extra_obj_func,
    )
end
function EfficientMeanVar(
    tickers::AbstractVector{<:AbstractString},
    mean_ret::AbstractVector{<:Real},
    cov_mtx::AbstractArray{<:Real, 2},
    weight_bounds::Union{Tuple{<:Real, <:Real}, AbstractVector} = (0.0, 1.0),
    rf::Real = 0.02,
    market_neutral::Bool = false,
    risk_aversion::Real = 1.0,
    target_volatility::Real = rank(cov_mtx) < size(cov_mtx, 1) ? 1 / sum(diag(cov_mtx)) :
                              sqrt(1 / sum(inv(cov_mtx))),
    target_ret::Real = minimum(mean_ret),
    extra_vars::AbstractVector{<:Any} = [],
    extra_constraints::AbstractVector{<:Any} = [],
    extra_obj_terms::AbstractVector{<:Any} = [],
    extra_obj_func::AbstractArray{<:Any} = [],
)
    weights, model = _EfficientFrontier(
        tickers,
        mean_ret,
        cov_mtx,
        weight_bounds,
        extra_vars,
        extra_constraints,
        extra_obj_func,
    )

    return EfficientMeanVar(
        tickers,
        mean_ret,
        weights,
        cov_mtx,
        rf,
        market_neutral,
        risk_aversion,
        target_volatility,
        target_ret,
        extra_vars,
        extra_constraints,
        extra_obj_terms,
        extra_obj_func,
        model,
    )
end

@inline function _EfficientFrontier(
    tickers,
    mean_ret,
    cov_mtx,
    weight_bounds,
    extra_vars,
    extra_constraints,
    extra_obj_func,
)
    num_tickers = length(tickers)
    @assert num_tickers == length(mean_ret) == size(cov_mtx, 1) == size(cov_mtx, 2)
    weights = zeros(num_tickers)

    model = Model()
    @variable(model, w[1:num_tickers])

    lower_bounds, upper_bounds = _create_weight_bounds(num_tickers, weight_bounds)

    @constraint(model, lower_bounds, w .>= lower_bounds)
    @constraint(model, upper_bounds, w .<= upper_bounds)

    # Add extra variables for extra variables and objective functions.
    if !isempty(extra_vars)
        _add_var_to_model!.(model, extra_vars)
    end

    # We need to add the extra constraints.
    if !isempty(extra_constraints)
        constraint_keys =
            [Symbol("extra_constraint$(i)") for i in 1:length(extra_constraints)]
        _add_constraint_to_model!.(model, constraint_keys, extra_constraints)
    end

    if !isempty(extra_obj_func)
        for func in extra_obj_func
            !isdefined(PortfolioOptimiser, func.args[2].args[1].args[1]) && eval(func)
        end
    end

    return weights, model
end