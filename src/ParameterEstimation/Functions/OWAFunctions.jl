# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```julia
owa_gmd(T::Integer)
```

Computes the Gini Mean Difference (GMD) of a returns series [^OWA].

# Inputs

# Outputs

[^OWA]: [Cajas, Dany, OWA Portfolio Optimization: A Disciplined Convex Programming Framework (December 18, 2021). Available at SSRN: https://ssrn.com/abstract=3988927 or http://dx.doi.org/10.2139/ssrn.3988927](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3988927)
"""
function owa_gmd(T::Integer)
    w = Vector{typeof(inv(T))}(undef, T)
    for i ∈ eachindex(w)
        w[i] = 2 * i - 1 - T
    end
    w = 2 / (T * (T - 1)) * w

    # w = (4 * range(1; stop = T) .- 2 * (T + 1)) / (T * (T - 1))

    return w
end

"""
```julia
owa_cvar(T::Integer; alpha::Real = 0.05)
```

Calculate the OWA weights corresponding to the Critical Value at Risk (CVaR) of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_cvar(T::Integer, alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))

    k = floor(Int, T * alpha)
    w = zeros(typeof(alpha), T)
    w[1:k] .= -1 / (T * alpha)
    w[k + 1] = -1 - sum(w[1:k])

    return w
end

"""
```julia
owa_wcvar(T::Integer, alphas::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
```

Compute the OWA weights for the Weighted Conditional Value at Risk (WCVaR) of a returns series [^OWA].

# Inputs

  - `alphas`: `N×1` vector of significance levels of each CVaR model, where `N` is the number of models, each .
  - `weights`: `N×1` vector of weights of each CVaR model, where `N` is the number of models.

# Outputs
"""
function owa_wcvar(T::Integer, alphas::AbstractVector{<:Real},
                   weights::AbstractVector{<:Real})
    w = zeros(promote_type(eltype(alphas), eltype(weights)), T)
    for (i, j) ∈ zip(alphas, weights)
        w .+= owa_cvar(T, i) * j
    end

    return w
end

"""
```julia
owa_tg(T::Integer; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100)
```

Compute the OWA weights for the Tail Gini of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_tg(T::Integer; alpha_i::Real = 1e-4, alpha::Real = 0.05, a_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))

    alphas = range(; start = alpha_i, stop = alpha, length = a_sim)
    n = length(alphas)
    w = Vector{typeof(alpha)}(undef, n)

    w[1] = alphas[2] * alphas[1] / alphas[n]^2
    for i ∈ 2:(n - 1)
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

# Outputs
"""
function owa_wr(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1

    return w
end

"""
```julia
owa_rg(T::Integer)
```

Compute the OWA weights for the Range of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_rg(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1
    w[T] = 1
    return w
end

"""
```julia
owa_cvarrg(T::Integer; alpha::Real = 0.05, beta::Real = alpha)
```

Compute the OWA weights for the CVaR Range of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_cvarrg(T::Integer; alpha::Real = 0.05, beta::Real = alpha)
    w = owa_cvar(T, alpha) .- reverse(owa_cvar(T, beta))
    return w
end

"""
```julia
owa_wcvarrg(T::Integer, alphas::AbstractVector{<:Real}, weights_a::AbstractVector{<:Real};
            betas::AbstractVector{<:Real} = alphas,
            weights_b::AbstractVector{<:Real} = weights_b)
```

Compute the OWA weights for the Weighted Conditional Value at Risk (WCVaR) of a returns series [^OWA].

# Inputs

  - `alphas`: `N×1` vector of significance levels of the losses for each CVaR model, where `N` is the number of losses models, each .
  - `weights_a`: `N×1` vector of weights of the losses for each CVaR model, where `N` is the number of losses models.
  - `betas`: `M×1` vector of significance levels of the gains for each CVaR model, where `M` is the number of gains models, each .
  - `weights_b`: `M×1` vector of weights of the gains for each CVaR model, where `M` is the number of gains models.

# Outputs
"""
function owa_wcvarrg(T::Integer, alphas::AbstractVector{<:Real},
                     weights_a::AbstractVector{<:Real},
                     betas::AbstractVector{<:Real} = alphas,
                     weights_b::AbstractVector{<:Real} = weights_a)
    w = owa_wcvar(T, alphas, weights_a) .- reverse(owa_wcvar(T, betas, weights_b))

    return w
end

"""
```julia
owa_tgrg(T::Integer; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100,
         beta_i::Real = alpha_i, beta::Real = alpha, b_sim::Integer = a_sim)
```

Compute the OWA weights for the Tail Gini Range of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_tgrg(T::Integer; alpha_i::Real = 0.0001, alpha::Real = 0.05,
                  a_sim::Integer = 100, beta_i::Real = alpha_i, beta::Real = alpha,
                  b_sim::Integer = a_sim)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) .-
        reverse(owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim))

    return w
end

"""
```julia
optimise_JuMP_model(model, solvers)
```

Internal function to optimise an OWA JuMP model.

# Inputs

  - `model`: JuMP model.

# Outputs

  - `term_status`: JuMP termination status.

  - `solvers_tried`: Dictionary that contains a dictionary of failed optimisations. `Dict(key => Dict(...))`, where `key` is the solver key used for the iteration of `solver` that failed.

      + If an MOI call fails on a model:

          * `Dict(:jump_error => jump_error)`: [`JuMP`](https://jump.dev/JuMP.jl/stable/moi/reference/errors/) error code.

      + If the optimiser fails to optimise the model satisfactorily:

          * `Dict(:objective_val => JuMP.objective_value(model), :term_status => term_status, :params => haskey(val, :params) ? val[:params] : missing)`, where `val` is the value of the dictionary corresponding to `key`.
"""
function optimise_JuMP_model(model, solvers)
    solvers_tried = Dict()

    sucess = false
    for solver ∈ solvers
        name = solver.name
        solver_i = solver.solver
        params = solver.params
        add_bridges = solver.add_bridges
        check_sol = solver.check_sol

        set_optimizer(model, solver_i; add_bridges = add_bridges)
        if !isnothing(params)
            set_attribute.(model, getindex.(params, 1), getindex.(params, 2))
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, name => Dict(:jump_error => jump_error))
            continue
        end

        if is_solved_and_feasible(model; check_sol...)
            sucess = true
            break
        else
            term_status = termination_status(model)
        end

        push!(solvers_tried,
              name => Dict(:objective_val => objective_value(model),
                           :term_status => term_status, :params => params))
    end

    return sucess, solvers_tried
end

"""
```julia
crra_weights(weights::AbstractMatrix{<:Real}, k::Integer, g::Real)
```

Internal function for computing the Normalized Constant Relative Risk Aversion coefficients.

# Inputs

  - `weights`: `T×(k-1)` matrix where T is the number of observations and `k` the order of the L-moments to combine, the `i`'th column contains the weights for the `(i+1)`'th L-moment.
  - `k`: the maximum order of the L-moments.
  - `g`: the risk aversion coefficient.

# Outputs

  - `w`: `T×1` ordered weight vector of the combined L-moments.
"""
function crra_weights(weights::AbstractMatrix{<:Real}, k::Integer, g::Real)
    phis = Vector{eltype(weights)}(undef, k - 1)
    e = 1
    for i ∈ eachindex(phis)
        e *= g + i - 1
        phis[i] = e / factorial(i + 1)
    end

    phis ./= sum(phis)
    a = weights * phis

    w = similar(a)
    w[1] = a[1]
    for i ∈ 2:length(a)
        w[i] = maximum(a[1:i])
    end

    return w
end

"""
```julia
owa_l_moment(T::Integer; k::Integer = 2)
```

Calculates the OWA weights of the k'th linear moment (L-moment) of a returns series [OWAL](@cite).

# Inputs

  - `k`: order of the L-moment.

# Outputs
"""
function owa_l_moment(T::Integer, k::Integer = 2)
    T, k = promote(T, k)
    w = Vector{typeof(inv(T * k))}(undef, T)
    for i ∈ eachindex(w)
        a = zero(k)
        for j ∈ 0:(k - 1)
            a += (-1)^j *
                 binomial(k - 1, j) *
                 binomial(i - 1, k - 1 - j) *
                 binomial(T - i, j)
        end
        a *= 1 / (k * binomial(T, k))
        w[i] = a
    end

    return w
end

function owa_l_moment_crm(type::CRRA, ::Any, k, weights)
    return crra_weights(weights, k, type.g)
end
function owa_model_setup(type, T, weights)
    N = size(weights, 2)
    model = JuMP.Model()
    max_phi = type.max_phi
    scale_constr = type.scale_constr
    @variables(model, begin
                   theta[1:T]
                   phi[1:N]
               end)
    @constraints(model,
                 begin
                     scale_constr * 0 .<= scale_constr * phi .<= scale_constr * max_phi
                     scale_constr * sum(phi) == scale_constr * 1
                     scale_constr * theta .== scale_constr * weights * phi
                     scale_constr * phi[2:end] .<= scale_constr * phi[1:(end - 1)]
                     scale_constr * theta[2:end] .>= scale_constr * theta[1:(end - 1)]
                 end)
    return model
end
function owa_model_solve(model, weights, type, k)
    solvers = type.solvers
    success = optimise_JuMP_model(model, solvers)[1]
    return if success
        phi = model[:phi]
        phis = value.(phi)
        phis ./= sum(phis)
        w = weights * phis
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.owa_l_moment_crm))"
        @warn("$funcname: model could not be optimised satisfactorily.\nType: $type\nReverting to crra type.")
        w = crra_weights(weights, k, 0.5)
    end
end
function owa_l_moment_crm(type::MaxEntropy, T, k, weights)
    scale_constr = type.scale_constr
    scale_obj = type.scale_obj
    ovec = range(; start = 1, stop = 1, length = T)
    model = owa_model_setup(type, T, weights)
    theta = model[:theta]
    @variables(model, begin
                   t
                   x[1:T]
               end)
    @constraints(model,
                 begin
                     scale_constr * sum(x) == scale_constr * 1
                     [scale_constr * t; scale_constr * ovec; scale_constr * x] ∈
                     MOI.RelativeEntropyCone(2 * T + 1)
                     [i = 1:T],
                     [scale_constr * x[i]; scale_constr * theta[i]] ∈ MOI.NormOneCone(2)
                 end)
    @objective(model, Max, -scale_obj * t)
    return owa_model_solve(model, weights, type, k)
end
function owa_l_moment_crm(type::MinSumSq, T, k, weights)
    scale_constr = type.scale_constr
    scale_obj = type.scale_obj
    model = owa_model_setup(type, T, weights)
    theta = model[:theta]
    @variable(model, t)
    @constraint(model, [scale_constr * t; scale_constr * theta] ∈ SecondOrderCone())
    @objective(model, Min, scale_obj * t)
    return owa_model_solve(model, weights, type, k)
end
function owa_l_moment_crm(type::MinSqDist, T, k, weights)
    scale_constr = type.scale_constr
    scale_obj = type.scale_obj
    model = owa_model_setup(type, T, weights)
    theta = model[:theta]
    @variable(model, t)
    @constraint(model,
                [scale_constr * t; scale_constr * (theta[2:end] .- theta[1:(end - 1)])] ∈
                SecondOrderCone())
    @objective(model, Min, scale_obj * t)
    return owa_model_solve(model, weights, type, k)
end
"""
```julia
owa_l_moment_crm(T::Integer; k::Integer = 2, type::Symbol = :SD, g::Real = 0.5,
                 max_phi::Real = 0.5, solvers = Dict())
```

Compute the OWA weights for the convex risk measure considering higher order L-moments [OWAL](@cite).

# Inputs

  - `k`: order of the L-moment, `k ≥ 2`.

  - `type`: type for computing the weights used to combine L-moments higher than 2, used in [`OWATypes`](@ref).

      + `:CRRA:` Normalised Constant Relative Risk Aversion Coefficients.
      + `:E`: Maximum Entropy. Solver must support `MOI.RelativeEntropyCone` and `MOI.NormOneCone`.
      + `:SS`: Minimum Sum of Squares. Solver must support `MOI.SecondOrderCone`.
      + `:SD`: Minimum Square Distance. Solver must support `MOI.SecondOrderCone`.
  - `g`: the risk aversion coefficient.
  - `max_phi`: maximum weight constraint of the L-moments.

# Outputs
"""
function owa_l_moment_crm(T::Integer; k::Integer = 2, type::OWATypes = CRRA())
    @smart_assert(k >= 2)
    rg = 2:k
    weights = Matrix{typeof(inv(T * k))}(undef, T, length(rg))
    for i ∈ rg
        wi = (-1)^i * owa_l_moment(T, i)
        weights[:, i - 1] .= wi
    end
    return owa_l_moment_crm(type, T, k, weights)
end

export owa_gmd, owa_cvar, owa_wcvar, owa_tg, owa_wr, owa_rg, owa_cvarrg, owa_wcvarrg,
       owa_tgrg, owa_l_moment, owa_l_moment_crm
