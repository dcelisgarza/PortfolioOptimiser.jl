function mv(w, cov)
    return dot(w, cov, w)
end

function msd(w, cov)
    return sqrt(mv(w, cov))
end

function mad(x)
    mu = mean(x)
    return mean(abs.(x .- mu))
end

function gmd(x)
    T = length(x)
    w = owa_gmd(T)
    return dot(w, sort!(x))
end

function msv(x)
    T = length(x)
    mu = mean(x)
    val = mu .- x
    val = sum(val[val .>= 0] .^ 2) / (T - 1)
    return sqrt(val)
end

function flpm(x, min_ret = 0.0)
    T = length(x)
    val = min_ret .- x
    return sum(val[val .>= 0]) / T
end

function slpm(x, min_ret = 0.0)
    T = length(x)
    val = min_ret .- x
    val = sum(val[val .>= 0] .^ 2) / (T - 1)
    return sqrt(val)
end

function var(x, alpha = 0.05)
    sort!(x)
    idx = Int(ceil(alpha * length(x)))
    return -x[idx]
end

function cvar(x, alpha = 0.05)
    sort!(x)
    idx = Int(ceil(alpha * length(x)))
    sum_var = 0.0
    var = -x[idx]
    for i in 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / (alpha * length(x))
end

function tg(x; alpha_i = 0.0001, alpha = 0.05, a_sim = 100)
    T = length(x)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    return dot(w, sort!(x))
end

function _optimize_rm(model, solvers)
    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) in solvers
        if haskey(val, :solver)
            set_optimizer(model, val[:solver])
        end

        if haskey(val, :params)
            for (attribute, value) in val[:params]
                set_attribute(model, attribute, value)
            end
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, key => Dict(:jump_error => jump_error))
            continue
        end

        term_status = termination_status(model)

        term_status in ValidTermination && break

        push!(
            solvers_tried,
            key => Dict(
                :objective_val => objective_value(model),
                :term_status => term_status,
                :params => haskey(val, :params) ? val[:params] : missing,
            ),
        )
    end

    return solvers_tried
end

function _optimize_entropic_rm(x, alpha, solvers)
    model = JuMP.Model()
    T = length(x)
    at = alpha * T
    @variable(model, t)
    @variable(model, s >= 0)
    @variable(model, u[1:T])
    @constraint(model, sum(u) <= s)
    @constraint(model, [i = 1:T], [x[i] - t, s, u[i]] in MOI.ExponentialCone())
    @expression(model, risk, t - s * log(at))
    @objective(model, Min, risk)

    solvers_tried = _optimize_rm(model, solvers)
    return model, solvers_tried
end

function entropic_rm(x, alpha, solvers)
    model, solvers_tried = _optimize_entropic_rm(x, alpha, solvers)
    term_status = termination_status(model)

    if term_status ∉ ValidTermination
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.entropic_rm))"
        @warn(
            "$funcname: model could not be optimised satisfactorily. Solvers: $solvers_tried"
        )
    end

    z = value(model[:s])
    arg = -x / z
    val = mean(exp.(arg))
    return z * (log(val) - log(alpha))
end

function evar(x, alpha, solvers)
    return entropic_rm(x, alpha, solvers)
end

function sharpe_risk(
    w,
    cov,
    returns,
    rm = :mv,
    rf = 0.0,
    alpha_i = 0.0001,
    alpha = 0.05,
    a_sim = 100,
    beta_i = 0.0001,
    beta = Inf,
    b_sim = -1,
    kappa = 0.3,
    solvers = nothing,
)
    if rm != :mv || rm != :msd
        x = returns * w
    end

    risk = if rm == :mv
        mv(w, cov)
    elseif rm == :msd
        msd(w, cov)
    elseif rm == :mad
        mad(x)
    elseif rm == :msv
        msv(x)
    elseif rm == :flpm
        flpm(x, rf)
    elseif rm == :slpm
        slpm(x, rf)
    elseif rm == :var
        var(x, alpha)
    elseif rm == :cvar
        cvar(x, alpha)
    elseif rm == :tg
        tg(x; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim)
    elseif rm == :evar
        evar(x, alpha, solvers)
    end
end