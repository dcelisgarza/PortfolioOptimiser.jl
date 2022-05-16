abstract type AbstractEfficientCDaR <: AbstractEfficient end

struct EfficientCDaR{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13} <:
       AbstractEfficientCDaR
    tickers::T1
    mean_ret::T2
    weights::T3
    returns::T4
    beta::T5
    market_neutral::T6
    target_cdar::T7
    target_ret::T8
    extra_vars::T9
    extra_constraints::T10
    extra_obj_terms::T11
    extra_obj_func::T12
    model::T13
end
function EfficientCDaR(
    tickers,
    mean_ret,
    returns;
    weight_bounds = (0.0, 1.0),
    beta = 0.95,
    market_neutral = false,
    target_cdar = mean(maximum(returns, dims = 2)),
    target_ret = mean(mean_ret),
    extra_vars = [],
    extra_constraints = [],
    extra_obj_terms = [],
    extra_obj_func = [],
)
    return EfficientCDaR(
        tickers,
        mean_ret,
        returns,
        weight_bounds,
        beta,
        market_neutral,
        target_cdar,
        target_ret,
        extra_vars,
        extra_constraints,
        extra_obj_terms,
        extra_obj_func,
    )
end
function EfficientCDaR(
    tickers::AbstractVector{<:AbstractString},
    mean_ret::AbstractVector{<:Real},
    returns::AbstractArray{<:Real, 2},
    weight_bounds::Union{Tuple{<:Real, <:Real}, AbstractVector} = (0.0, 1.0),
    beta::Real = 0.95,
    market_neutral::Bool = false,
    target_cdar::Real = mean(maximum(returns, dims = 2)),
    target_ret::Real = mean(mean_ret),
    extra_vars::AbstractVector{<:Any} = [],
    extra_constraints::AbstractVector{<:Any} = [],
    extra_obj_terms::AbstractVector{<:Any} = [],
    extra_obj_func::AbstractArray{<:Any} = [],
)
    weights, model = _EfficientCDaR(
        tickers,
        mean_ret,
        returns,
        weight_bounds,
        beta,
        market_neutral,
        extra_vars,
        extra_constraints,
        extra_obj_func,
    )

    return EfficientCDaR(
        tickers,
        mean_ret,
        weights,
        returns,
        beta,
        market_neutral,
        target_cdar,
        target_ret,
        extra_vars,
        extra_constraints,
        extra_obj_terms,
        extra_obj_func,
        model,
    )
end

@inline function _EfficientCDaR(
    tickers,
    mean_ret,
    returns,
    weight_bounds,
    beta,
    market_neutral,
    extra_vars,
    extra_constraints,
    extra_obj_func,
)
    num_tickers = length(tickers)
    @assert num_tickers == length(mean_ret) == size(returns, 2)
    weights = zeros(num_tickers)

    if !(0 <= beta <= 1)
        @warn("beta: $beta must be between 0 and 1. Defaulting to 0.95")
        beta = 0.95
    end
    if beta <= 0.2
        @warn(
            "beta: $beta is the confidence level, not percentile. It is typically 0.8, 0.9, 0.95"
        )
    end

    model = Model()
    samples = size(returns, 1)
    @variable(model, w[1:num_tickers])

    # CDaR variables
    @variable(model, alpha)
    @variable(model, u[1:(samples + 1)])
    @variable(model, z[1:samples])
    # CDaR constraints.
    @constraint(model, z_geq_uma, z .>= u[2:end] .- alpha)
    @constraint(model, uf_geq_uimvw, u[2:end] .>= u[1:(end - 1)] .- returns * w)
    @constraint(model, u1_eq_0, u[1] == 0)
    @constraint(model, z_geq_0, z .>= 0)
    @constraint(model, u2e_geq_0, u[2:end] .>= 0)

    lower_bounds, upper_bounds = _create_weight_bounds(num_tickers, weight_bounds)

    @constraint(model, lower_bounds, w .>= lower_bounds)
    @constraint(model, upper_bounds, w .<= upper_bounds)

    _make_weight_sum_constraint(model, market_neutral)

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