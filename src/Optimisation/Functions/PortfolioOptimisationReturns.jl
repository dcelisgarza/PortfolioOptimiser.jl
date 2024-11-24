function _return_bounds(::Any, model, mu_l::Real)
    if isfinite(mu_l)
        ret = model[:ret]
        @constraint(model, ret >= mu_l)
    end
    return nothing
end
function _return_bounds(::Sharpe, model, mu_l::Real)
    if isfinite(mu_l)
        ret = model[:ret]
        k = model[:k]
        @constraint(model, ret >= mu_l * k)
    end
    return nothing
end
function set_returns(obj::Any, kelly::NoKelly, port; mu::AbstractVector, kwargs...)
    if !isempty(mu)
        _wc_return_constraints(kelly.mu, port)
        _return_bounds(obj, port.model, port.mu_l)
    end
    return nothing
end
function set_returns(obj::Sharpe, kelly::NoKelly, port; mu::AbstractVector, kwargs...)
    if !isempty(mu)
        model = port.model
        _wc_return_constraints(kelly.mu, port)
        if !all(mu .< zero(eltype(mu)))
            ret = model[:ret]
            k = model[:k]
            @constraint(model, ret - obj.rf * k == 1)
        else
            risk = model[:risk]
            @constraint(model, alt_sr, risk <= 1)
        end
        _return_bounds(obj, model, port.mu_l)
    end
    return nothing
end
function set_returns(obj::Any, kelly::AKelly, port; mu::AbstractVector,
                     kelly_approx_idx::Union{AbstractVector{<:Integer}, Nothing} = nothing,
                     adjacency_constraint::AdjacencyConstraint, sigma::AbstractMatrix,
                     kwargs...)
    if !isempty(mu)
        model = port.model
        w = model[:w]
        if isnothing(kelly_approx_idx) ||
           isempty(kelly_approx_idx) ||
           iszero(kelly_approx_idx[1])
            if !haskey(model, :sd_risk)
                _sd_risk(adjacency_constraint, kelly.formulation, model, sigma)
            end
            sd_risk = model[:sd_risk]
            @expression(model, ret, dot(mu, w) - 0.5 * sd_risk)
        else
            sd_risk = model[:sd_risk]
            @expression(model, ret, dot(mu, w) - 0.5 * sd_risk[kelly_approx_idx[1]])
        end
        _return_bounds(obj, model, port.mu_l)
    end
    return nothing
end
function _set_returns(adjacency_constraint::SDP, obj::Sharpe, kelly::AKelly, port;
                      kwargs...)
    return set_returns(obj, EKelly(), port; kwargs...)
end
function _set_returns(adjacency_constraint::Union{NoAdj, IP}, obj::Sharpe, kelly::AKelly,
                      port; mu::AbstractVector, kelly_approx_idx::AbstractVector{<:Integer},
                      sigma::AbstractMatrix, kwargs...)
    if !isempty(mu)
        model = port.model
        @variable(model, tapprox_kelly)
        risk = model[:risk]
        @constraint(model, risk <= 1)
        w = model[:w]
        @expression(model, ret, dot(mu, w) - 0.5 * tapprox_kelly)
        k = model[:k]
        if isnothing(kelly_approx_idx) ||
           isempty(kelly_approx_idx) ||
           iszero(kelly_approx_idx[1])
            if !haskey(model, :sd_risk)
                _sd_risk(adjacency_constraint, kelly.formulation, model, sigma)
            end
            dev = model[:dev]
            @constraint(model,
                        [k + tapprox_kelly
                         2 * dev
                         k - tapprox_kelly] ∈ SecondOrderCone())
        else
            dev = model[:dev]
            @constraint(model,
                        [k + tapprox_kelly
                         2 * dev[kelly_approx_idx[1]]
                         k - tapprox_kelly] ∈ SecondOrderCone())
        end
        _return_bounds(obj, model, port.mu_l)
    end
    return nothing
end
function set_returns(obj::Sharpe, kelly::AKelly, port; mu::AbstractVector,
                     kelly_approx_idx::AbstractVector{<:Integer},
                     adjacency_constraint::AdjacencyConstraint, sigma::AbstractMatrix,
                     kwargs...)
    _set_returns(adjacency_constraint, obj, kelly, port; mu = mu,
                 kelly_approx_idx = kelly_approx_idx, sigma = sigma, kwargs...)
    return nothing
end
function set_returns(obj::Any, ::EKelly, port; mu::AbstractVector, returns::AbstractMatrix,
                     kwargs...)
    if !isempty(mu)
        model = port.model
        T = size(returns, 1)
        @variable(model, texact_kelly[1:T])
        @expression(model, ret, sum(texact_kelly) / T)
        w = model[:w]
        @expression(model, kret, 1 .+ returns * w)
        @constraint(model, [i = 1:T], [texact_kelly[i], 1, kret[i]] ∈ MOI.ExponentialCone())
        _return_bounds(obj, model, port.mu_l)
    end
    return nothing
end
function set_returns(obj::Sharpe, ::EKelly, port; mu::AbstractVector,
                     returns::AbstractMatrix, kwargs...)
    if !isempty(mu)
        model = port.model
        T = size(returns, 1)
        @variable(model, texact_kelly[1:T])
        k = model[:k]
        @expression(model, ret, sum(texact_kelly) / T - obj.rf * k)
        w = model[:w]
        @expression(model, kret, k .+ returns * w)
        risk = model[:risk]
        @constraint(model, [i = 1:T], [texact_kelly[i], k, kret[i]] ∈ MOI.ExponentialCone())
        @constraint(model, risk <= 1)
        _return_bounds(obj, model, port.mu_l)
    end
    return nothing
end
function return_constraints(port, obj, kelly, mu, sigma, returns, kelly_approx_idx)
    set_returns(obj, kelly, port; mu = mu, sigma = sigma, returns = returns,
                kelly_approx_idx = kelly_approx_idx,
                adjacency_constraint = _get_ntwk_clust_method(port))
    return nothing
end
