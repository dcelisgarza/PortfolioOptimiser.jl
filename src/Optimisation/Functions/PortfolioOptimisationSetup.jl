function set_w_ini(w, w_ini)
    if !isempty(w_ini)
        @smart_assert(length(w_ini) == length(w))
        set_start_value.(w, w_ini)
    end
    return nothing
end
function initial_w(port, w_ini)
    model = port.model
    N = size(port.returns, 2)
    @variable(model, w[1:N])
    set_w_ini(w, w_ini)
    return nothing
end
function mu_sigma_returns_class(port, ::Union{Classic, FC})
    return port.mu, port.cov, port.returns
end
function mu_sigma_returns_class(port, class::FM)
    mu = port.fm_mu
    if class.type == 1
        sigma = port.fm_cov
        returns = port.fm_returns
    else
        sigma = port.cov
        returns = port.returns
    end
    return mu, sigma, returns
end
function mu_sigma_returns_class(port, class::BL)
    mu = port.bl_mu
    returns = port.returns
    if class.type == 1
        sigma = port.bl_cov
    else
        sigma = port.cov
    end
    return mu, sigma, returns
end
function mu_sigma_returns_class(port, class::BLFM)
    mu = port.blfm_mu
    if class.type == 1
        sigma = port.blfm_cov
        returns = port.fm_returns
    elseif class.type == 2
        sigma = port.cov
        returns = port.returns
    else
        sigma = port.fm_cov
        returns = port.fm_returns
    end
    return mu, sigma, returns
end
##########
##########
function optimal_homogenisation_factor(port, mu, ::Sharpe, ohf)
    if iszero(ohf)
        if !isempty(mu)
            ohf = min(1e3, max(1e-3, mean(abs.(mu))))
        else
            ohf = one(eltype(port.returns))
        end
    end
    model = port.model
    @expression(model, ohf, ohf)
    return nothing
end
function optimal_homogenisation_factor(args...)
    return nothing
end
function set_k(port, ::Sharpe)
    model = port.model
    @variable(model, k >= 0)
    return nothing
end
function set_k(port, ::Any)
    model = port.model
    @expression(model, k, 1)
    return nothing
end
##########
##########
# sharpe ratio/RP k
function set_sr_k(::Sharpe, model)
    @variable(model, k >= 0)
    return nothing
end
function set_sr_k(::Any, ::Any)
    return nothing
end
# SDP setup
function _sdp_m2(::Sharpe, model)
    w = model[:w]
    k = model[:k]
    @expression(model, M2, vcat(w, k))
    return nothing
end
function _sdp_m2(::Any, model)
    w = model[:w]
    @expression(model, M2, vcat(w, 1))
    return nothing
end
function _sdp(port, obj)
    model = port.model
    if !haskey(model, :W)
        N = size(port.returns, 2)
        @variable(model, W[1:N, 1:N], Symmetric)
        w = model[:w]
        @expression(model, M1, vcat(W, transpose(w)))
        _sdp_m2(obj, model)
        M2 = model[:M2]
        @expression(model, M3, hcat(M1, M2))
        @constraint(model, M3 ∈ PSDCone())
    end
    return nothing
end
function _sdp(::SDP, port, obj)
    model = port.model
    if !haskey(model, :W)
        N = size(port.returns, 2)
        @variable(model, W[1:N, 1:N], Symmetric)
        w = model[:w]
        @expression(model, M1, vcat(W, transpose(w)))
        _sdp_m2(obj, model)
        M2 = model[:M2]
        @expression(model, M3, hcat(M1, M2))
        @constraint(model, M3 ∈ PSDCone())
    end
    return nothing
end
function _sdp(::Any, ::Any, ::Any)
    return nothing
end
