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
