function owa_l_moment(T, k = 2)
    w = Vector(undef, T)
    for i in 1:T
        a = 0.0
        for j in 0:(k - 1)
            a +=
                (-1)^j *
                binomial(k - 1, j) *
                binomial(i - 1, k - 1 - j) *
                binomial(T - i, j)
        end
        a *= 1 / (k * binomial(T, k))
        w[i] = a
    end
    w = 1 / convert(typeof(T), 1) * w

    return w
end

function owa_gmd(T)
    w = Vector(undef, T)
    for i in 1:T
        w[i] = 2 * i - 1 - T
    end
    w = 2 / (T * (T - 1)) * w

    return w
end

function owa_cvar(T, alpha = 0.05)
    k = ceil(Int, T * alpha)
    w = zeros(T)
    w[1:(k - 1)] .= -1 / (T * alpha)
    w[k] = -1 - sum(w[1:(k - 1)])

    return w
end

function owa_wcvar(T, alphas, weights)
    w = zeros(T)
    for (i, j) in zip(alphas, weights)
        w .+= owa_cvar(T, i) * j
    end

    return w
end

function owa_tg(T; alpha_i = 0.0001, alpha = 0.05, a_sim = 100)
    alphas = range(start = alpha_i, stop = alpha, length = a_sim)
    n = length(alphas)
    w = Vector(undef, n)

    w[1] = alphas[2] * alphas[1] / alphas[n]^2
    for i in 2:(n - 1)
        w[i] = (alphas[i + 1] - alphas[i - 1]) * alphas[i] / alphas[n]^2
    end
    w[n] = (alphas[n] - alphas[n - 1]) / alphas[n]

    w = owa_wcvar(T, alphas, w)

    return w
end

function owa_wr(T)
    w = zeros(T)
    w[1] = -1

    return w
end

function owa_rg(T)
    w = zeros(T)
    w[1] = -1
    w[T] = 1
    return w
end

function owa_rcvar(T; alpha = 0.05, beta = nothing)
    isnothing(beta) && (beta = alpha)
    w = owa_cvar(T, alpha) .- reverse(owa_cvar(T, beta))

    return w
end

function owa_wcvrg(T, alphas, weights_a, betas = nothing, weights_b = nothing)
    if isnothing(betas) || isnothing(weights_b)
        betas = alphas
        weights_b = weights_a
    end

    w = owa_wcvar(T, alphas, weights_a) .- reverse(owa_wcvar(T, betas, weights_b))

    return w
end

function owa_rtg(
    T;
    alpha_i = 0.0001,
    alpha = 0.05,
    a_sim = 100,
    beta_i = nothing,
    beta = nothing,
    b_sim = nothing,
)
    isnothing(beta) && (beta = alpha)
    isnothing(b_sim) && (b_sim = a_sim)
    isnothing(beta_i) && (beta_i = alpha_i)

    w =
        owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) .-
        reverse(owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim))

    return w
end

function _optimize_owa(model, solvers)
    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) in solvers
        haskey(val, :solver) && set_optimizer(model, val[:solver])

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

    return term_status, solvers_tried
end

function _crra_method(ws, k, g)
    phis = Vector{eltype(ws)}(undef, k - 1)
    e = 1
    for i in 1:(k - 1)
        e *= g + i - 1
        phis[i] = e / factorial(i + 1)
    end

    phis ./= sum(phis)
    a = ws * phis

    w = similar(a)
    w[1] = a[1]
    for i in 2:length(a)
        w[i] = maximum(a[1:i])
    end

    return w
end

const OWAMethods = (:crra, :me, :mss, :msd)
function owa_l_moment_crm(
    T;
    k = 4,
    method = :msd,
    g = 0.5,
    max_phi = 0.5,
    solvers = Dict(),
    sol_params = Dict(),
)
    @assert(k >= 2, "k = $k, must be an integer bigger than or equal to 2")
    @assert(method ∈ OWAMethods, "method = $method, must be one of $OWAMethods")
    @assert(0 < g < 1, "risk aversion, g = $g, must be in the interval (0, 1)")
    @assert(
        0 < max_phi < 1,
        "the constraint on the maximum weight of the L-moments, max_phi = $max_phi, must be in the interval (0, 1)"
    )

    ws = Matrix{typeof(max_phi)}(undef, T, 0)
    for i in 2:k
        wi = (-1)^i * owa_l_moment(T, i)
        ws = hcat(ws, wi)
    end

    if method == :crra || isempty(solvers)
        w = _crra_method(ws, k, g)
    else
        n = size(ws, 2)
        model = JuMP.Model()
        @variable(model, theta[1:T])
        @variable(model, phi[1:n] >= 0)

        @constraint(model, sum(phi) == 1)
        @constraint(model, theta .== ws * phi)
        @constraint(model, phi .<= max_phi)
        @constraint(model, phi[2:end] .<= phi[1:(end - 1)])
        @constraint(model, theta[2:end] .>= theta[1:(end - 1)])

        if method == :me
            # Maximise entropy.
            @variable(model, t[1:T])
            @variable(model, x[1:T] >= 0)
            @constraint(model, sum(x) == 1)
            @constraint(model, [i = 1:T], [t[i], x[i], 1] in MOI.ExponentialCone())
            @constraint(model, x .- theta .>= 0)
            @constraint(model, x .+ theta .>= 0)
            @objective(model, Max, sum(t))
        elseif method == :mss
            @variable(model, t)
            @constraint(model, [t; theta] in SecondOrderCone())
            @objective(model, Min, t)
        elseif method == :msd
            @variable(model, t)
            @expression(model, theta_diff, theta[2:end] .- theta[1:(end - 1)])
            @constraint(model, [t; theta_diff] in SecondOrderCone())
            @objective(model, Min, t)
        end

        term_status, solvers_tried = _optimize_owa(model, solvers, sol_params)
        # Error handling.
        if term_status ∉ ValidTermination
            funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.owa_l_moment_crm))"

            @warn(
                "$funcname: model could not be optimised satisfactorily.\nMethod: $method\nSolvers: $solvers_tried. Reverting to crra method."
            )

            w = _crra_method(ws, k, g)
        else
            phis = value.(phi)
            phis ./= sum(phis)
            w = ws * phis
        end
    end

    return w
end

export owa_l_moment_crm