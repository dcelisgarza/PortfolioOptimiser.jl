"""
```julia
owa_gmd(T::Integer)
```
Computes the Gini Mean Difference (GMD) of a returns series [^OWA].
# Inputs
- `T`: number of observations of the returns series.
# Outputs
- `w`: `T×1` ordered weight vector.

[^OWA]:
    [Cajas, Dany, OWA Portfolio Optimization: A Disciplined Convex Programming Framework (December 18, 2021). Available at SSRN: https://ssrn.com/abstract=3988927 or http://dx.doi.org/10.2139/ssrn.3988927](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3988927)
"""
function owa_gmd(T::Integer)
    w = Vector(undef, T)
    for i in 1:T
        w[i] = 2 * i - 1 - T
    end
    w = 2 / (T * (T - 1)) * w

    return w
end

"""
```julia
owa_cvar(T::Integer, α::Real = 0.05)
```
Calculate the OWA weights corresponding to the Critical Value at Risk (CVaR) of a returns series [^OWA].
# Inputs
- `T`: number of observations of the returns series.
- `α`: significance level of CVaR, α ∈ (0, 1).
# Outputs
- `w`: `T×1` ordered weight vector.
"""
function owa_cvar(T::Integer, alpha::Real = 0.05)
    @assert(0 < alpha < 1, "alpha = $alpha, must be between 0 and 1")

    k = floor(Int, T * alpha)
    w = zeros(T)
    w[1:k] .= -1 / (T * alpha)
    w[k + 1] = -1 - sum(w[1:k])

    return w
end

"""
```julia
owa_wcvar(
    T::Integer,
    αs::AbstractVector{<:Real},
    ws::AbstractVector{<:Real},
)
```
Compute the OWA weights for the Weighted Conditional Value at Risk (WCVaR) of a returns series [^OWA].
# Inputs
- `T`: number of observations of the returns series.
- `αs`: `N×1` vector of significance levels of each CVaR model, where `N` is the number of models, each α_n ∈ (0, 1).
- `ws`: `N×1` vector of weights of each CVaR model, where `N` is the number of models.
# Outputs
- `w`: `T×1` ordered weight vector.
"""
function owa_wcvar(
    T::Integer,
    alphas::AbstractVector{<:Real},
    weights::AbstractVector{<:Real},
)
    w = zeros(T)
    for (i, j) in zip(alphas, weights)
        w .+= owa_cvar(T, i) * j
    end

    return w
end

"""
```julia
owa_tg(
    T::Integer;
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
)
```
Compute the OWA weights for the Tail Gini of a returns series [^OWA].
# Inputs
- `T`: number of observations of the returns series.
- `alpha_i`: initial significance level of Tail Gini.
- `alpha`: significance level of Tail Gini.
- `a_sim`: number of CVaRs to approximate the Tail Gini.
# Outputs
- `w`: `T×1` ordered weight vector.
"""
function owa_tg(T::Integer; alpha_i::Real = 1e-4, alpha::Real = 0.05, a_sim::Integer = 100)
    @assert(
        0 < alpha_i < alpha < 1,
        "alpha_i = $alpha_i, alpha = $alpha, please ensure 0 < alpha_i < alpha < 1, holds"
    )

    alphas = range(start = alpha_i, stop = alpha, length = a_sim)
    n = length(alphas)
    w = Vector{typeof(alpha)}(undef, n)

    w[1] = alphas[2] * alphas[1] / alphas[n]^2
    for i in 2:(n - 1)
        w[i] = (alphas[i + 1] - alphas[i - 1]) * alphas[i] / alphas[n]^2
    end
    w[n] = (alphas[n] - alphas[n - 1]) / alphas[n]

    w = owa_wcvar(T, alphas, w)

    return w
end

"""
```julia
owa_wr(T::Integer)
```
Compute the OWA weights for the Worst Realisation (WR) of a returns series [^OWA].
# Inputs
- `T`: number of observations of the returns series.
# Outputs
- `w`: `T×1` ordered weight vector.
"""
function owa_wr(T::Integer)
    w = zeros(T)
    w[1] = -1

    return w
end

"""
```julia
owa_rg(T::Integer)
```
Compute the OWA weights for the Range of a returns series [^OWA].
# Inputs
- `T`: number of observations of the returns series.
# Outputs
- `w`: `T×1` ordered weight vector.
"""
function owa_rg(T::Integer)
    w = zeros(T)
    w[1] = -1
    w[T] = 1
    return w
end

"""
```julia
owa_rcvar(T::Integer; alpha::Real = 0.05, beta::Union{<:Real, Nothing} = nothing)
```
Compute the OWA weights for the CVaR Range of a returns series [^OWA].
# Inputs
- `T`: number of observations of the returns series.
- `alpha`: significance level of CVaR losses, alpha ∈ (0, 1).
- `beta`: significance level of CVaR gains, beta ∈ (0, 1).
# Outputs
- `w`: `T×1` ordered weight vector.
"""
function owa_rcvar(T::Integer; alpha::Real = 0.05, beta::Union{<:Real, Nothing} = nothing)
    isnothing(beta) && (beta = alpha)
    w = owa_cvar(T, alpha) .- reverse(owa_cvar(T, beta))

    return w
end

"""
```julia
owa_rwcvar(
    T::Integer,
    αs::AbstractVector{<:Real},
    wα::AbstractVector{<:Real},
    βs::Union{AbstractVector{<:Real}, Nothing} = nothing,
    wβ::Union{AbstractVector{<:Real}, Nothing} = nothing,
)
```
Compute the OWA weights for the Weighted Conditional Value at Risk (WCVaR) of a returns series [^OWA].
# Inputs
- `T`: number of observations of the returns series.
- `αs`: `N×1` vector of significance levels of the losses for each CVaR model, where `N` is the number of losses models, each α_n ∈ (0, 1).
- `wα`: `N×1` vector of weights of the losses for each CVaR model, where `N` is the number of losses models.
- `βs`: `M×1` vector of significance levels of the gains for each CVaR model, where `M` is the number of gains models, each β_m ∈ (0, 1).
- `wβ`: `M×1` vector of weights of the gains for each CVaR model, where `M` is the number of gains models.
# Outputs
- `w`: `T×1` ordered weight vector.
"""
function owa_rwcvar(
    T::Integer,
    alphas::AbstractVector{<:Real},
    weights_a::AbstractVector{<:Real},
    betas::Union{AbstractVector{<:Real}, Nothing} = nothing,
    weights_b::Union{AbstractVector{<:Real}, Nothing} = nothing,
)
    isnothing(betas) && (betas = alphas)
    isnothing(weights_b) && (weights_b = weights_a)

    w = owa_wcvar(T, alphas, weights_a) .- reverse(owa_wcvar(T, betas, weights_b))

    return w
end

"""
```julia
owa_rtg(
    T::Integer;
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{<:Integer, Nothing} = nothing,
)
```
Compute the OWA weights for the Tail Gini Range of a returns series [^OWA].
# Inputs
- `T`: number of observations of the returns series.
- `alpha_i`: initial significance level of Tail Gini losses.
- `alpha`: significance level of Tail Gini losses.
- `a_sim`: number of CVaRs to approximate the Tail Gini losses.
- `beta_i`: initial significance level of Tail Gini gains.
- `beta`: significance level of Tail Gini gains.
- `b_sim`: number of CVaRs to approximate the Tail Gini gains.
# Outputs
- `w`: `T×1` ordered weight vector.
"""
function owa_rtg(
    T::Integer;
    alpha_i::Real = 0.0001,
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta_i::Union{<:Real, Nothing} = nothing,
    beta::Union{<:Real, Nothing} = nothing,
    b_sim::Union{<:Integer, Nothing} = nothing,
)
    isnothing(beta) && (beta = alpha)
    isnothing(b_sim) && (b_sim = a_sim)
    isnothing(beta_i) && (beta_i = alpha_i)

    w =
        owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) .-
        reverse(owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim))

    return w
end

"""
```julia
_optimize_owa(model, solvers)
```
Internal function to optimise an OWA JuMP model.
# Inputs
- `model`: JuMP model.
- `solver`: `Dict` or `NamedTuple` with the keys:
    - `:solver`: which contains the JuMP optimiser.
    - `:params`: for the solver-specific parameters.
# Outputs
- `term_status`: JuMP termination status.
- `solvers_tried`: Dictionary that contains a dictionary of failed optimisations. `Dict(key => Dict(...))`, where `key` is the solver key used for the iteration of `solver` that failed.
    - If an MOI call fails on a model:
        - `Dict(:jump_error => jump_error)`: [`JuMP`](https://jump.dev/JuMP.jl/stable/moi/reference/errors/) error code.
    - If the optimiser fails to optimise the model satisfactorily:
        - `Dict(:objective_val => JuMP.objective_value(model), :term_status => term_status, :params => haskey(val, :params) ? val[:params] : missing)`, where `val` is the value of the dictionary corresponding to `key`.
"""
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

"""
```julia
_crra_method(ws::AbstractMatrix, k::Integer, g::Real)
```
Internal function for computing the Normalized Constant Relative Risk Aversion coefficients.
# Inputs
- `ws`: `T×(k-1)` matrix where T is the number of observations and `k` the order of the l-moments to combine, the `i`'th column contains the weights for the `(i+1)`'th l-moment.
- `k`: the maximum order of the l-moments.
- `g`: the risk aversion coefficient.
# Outputs
- `w`: `T×1` ordered weight vector of the combined l-moments.
"""
function _crra_method(ws::AbstractMatrix, k::Integer, g::Real)
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

"""
```julia
owa_l_moment(T::Integer, k::Integer = 2)
```
Calculates the OWA weights of the k'th linear moment (l-moment) of a returns series [^OWAL].
# Inputs
- `T`: number of observations of the returns series.
- `k`: order of the l-moment.
# Outputs
- `w`: `T×1` ordered weight vector.

[^OWAL]:
    [Cajas, Dany, Higher Order Moment Portfolio Optimization with L-Moments (March 19, 2023). Available at SSRN: https://ssrn.com/abstract=4393155 or http://dx.doi.org/10.2139/ssrn.4393155](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4393155)
"""
function owa_l_moment(T::Integer, k::Integer = 2)
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

"""
```julia
owa_l_moment_crm(
    T::Integer;
    k::Integer = 2,
    method::Symbol = :SD,
    g::Real = 0.5,
    max_phi::Real = 0.5,
    solvers = Dict(),
)
```
"""
function owa_l_moment_crm(
    T::Integer;
    k::Integer = 2,
    method::Symbol = :SD,
    g::Real = 0.5,
    max_phi::Real = 0.5,
    solvers = Dict(),
)
    @assert(k >= 2, "k = $k, must be an integer bigger than or equal to 2")
    @assert(method ∈ OWAMethods, "method = $method, must be one of $OWAMethods")
    @assert(0 < g < 1, "risk aversion, g = $g, must be in the interval (0, 1)")
    @assert(
        0 < max_phi < 1,
        "the constraint on the maximum weight of the L-moments, max_phi = $max_phi, must be in the interval (0, 1)"
    )

    rg = 2:k
    ws = Matrix{typeof(max_phi)}(undef, T, length(rg))
    for i in rg
        wi = (-1)^i * owa_l_moment(T, i)
        ws[:, i - 1] .= wi
    end

    if method == :CRRA || isempty(solvers)
        w = _crra_method(ws, k, g)
    else
        n = size(ws, 2)
        model = JuMP.Model()
        @variable(model, theta[1:T])
        @variable(model, 0 .<= phi[1:n] .<= max_phi)

        @constraint(model, sum(phi) == 1)
        @constraint(model, theta .== ws * phi)
        @constraint(model, phi[2:end] .<= phi[1:(end - 1)])
        @constraint(model, theta[2:end] .>= theta[1:(end - 1)])

        if method == :E
            # Maximise entropy.
            @variable(model, t)
            @variable(model, x[1:T])
            @constraint(model, sum(x) == 1)
            @constraint(model, [t; ones(T); x] in MOI.RelativeEntropyCone(2 * T + 1))
            @constraint(model, x .>= theta)
            @constraint(model, x .>= -theta)
            @objective(model, Max, -t)
        elseif method == :SS
            @variable(model, t)
            @constraint(model, [t; theta] in SecondOrderCone())
            @objective(model, Min, t)
        elseif method == :SD
            @variable(model, t)
            @expression(model, theta_diff, theta[2:end] .- theta[1:(end - 1)])
            @constraint(model, [t; theta_diff] in SecondOrderCone())
            @objective(model, Min, t)
        end

        term_status, solvers_tried = _optimize_owa(model, solvers)
        # Error handling.
        if term_status ∉ ValidTermination
            funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.owa_l_moment_crm))"
            @warn(
                "$funcname: model could not be optimised satisfactorily.\nMethod: $method\nSolvers: $solvers_tried.\nReverting to crra method."
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

export owa_l_moment,
    owa_l_moment_crm,
    owa_gmd,
    owa_cvar,
    owa_wcvar,
    owa_tg,
    owa_wr,
    owa_rg,
    owa_rcvar,
    owa_rwcvar,
    owa_rtg
