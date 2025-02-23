# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function calc_linear_constraints!(port::AbstractPortfolio;
                                  asset_sets::DataFrame = port.asset_sets,
                                  a_ineq_constraints::DataFrame = port.a_ineq_constraints,
                                  a_eq_constraints::DataFrame = port.a_eq_constraints,
                                  a_card_ineq_constraints::DataFrame = port.a_card_ineq_constraints,
                                  a_card_eq_constraints::DataFrame = port.a_card_eq_constraints,
                                  f_ineq_constraints::DataFrame = port.f_ineq_constraints,
                                  f_eq_constraints::DataFrame = port.f_eq_constraints,
                                  f_card_ineq_constraints::DataFrame = port.f_card_ineq_constraints,
                                  f_card_eq_constraints::DataFrame = port.f_card_eq_constraints,
                                  loadings::DataFrame = port.loadings)
    port.asset_sets = asset_sets
    port.loadings = loadings

    asset_set_flag = isempty(asset_sets)
    loadings_flag = isempty(loadings)
    if asset_set_flag && loadings_flag
        return nothing
    end

    returns = port.returns

    N = size(returns, 2)

    A_ineq = Matrix{eltype(returns)}(undef, 0, N)
    B_ineq = Vector{eltype(returns)}(undef, 0)
    if !asset_set_flag && !isempty(a_ineq_constraints)
        port.a_ineq_constraints = a_ineq_constraints
        A, B = asset_constraints(a_ineq_constraints, asset_sets)
        A_ineq = vcat(A_ineq, A)
        push!(B_ineq, B)
    end
    if !loadings_flag && !isempty(f_ineq_constraints)
        port.f_ineq_constraints = f_ineq_constraints
        A, B = factor_constraints(f_ineq_constraints, loadings)
        A_ineq = vcat(A_ineq, A)
        push!(B_ineq, B)
    end
    if !isempty(A_ineq)
        port.a_ineq, port.b_ineq = A_ineq, B_ineq
    end

    A_eq = Matrix{eltype(returns)}(undef, 0, N)
    B_eq = Vector{eltype(returns)}(undef, 0)
    if !asset_set_flag && !isempty(a_eq_constraints)
        port.a_eq_constraints = a_eq_constraints
        A, B = asset_constraints(a_eq_constraints, asset_sets)
        A_eq = vcat(A_eq, A)
        push!(B_eq, B)
    end
    if !loadings_flag && !isempty(f_eq_constraints)
        port.f_eq_constraints = f_eq_constraints
        A, B = factor_constraints(f_eq_constraints, loadings)
        A_eq = vcat(A_eq, A)
        push!(B_eq, B)
    end
    if !isempty(A_eq)
        port.a_eq, port.b_eq = A_eq, B_eq
    end

    A_card_ineq = Matrix{eltype(returns)}(undef, 0, N)
    B_card_ineq = Vector{eltype(returns)}(undef, 0)
    if !asset_set_flag && !isempty(a_card_ineq_constraints)
        port.a_card_ineq_constraints = a_card_ineq_constraints
        A, B = asset_constraints(a_card_ineq_constraints, asset_sets)
        A_card_ineq = vcat(A_card_ineq, A)
        push!(B_card_ineq, B)
    end
    if !loadings_flag && !isempty(f_card_ineq_constraints)
        port.f_card_ineq_constraints = f_card_ineq_constraints
        A, B = factor_constraints(f_card_ineq_constraints, loadings)
        A_card_ineq = vcat(A_card_ineq, A)
        push!(B_card_ineq, B)
    end
    if !isempty(A_card_ineq)
        port.a_card_ineq, port.b_card_ineq = A_card_ineq, B_card_ineq
    end

    A_card_eq = Matrix{eltype(returns)}(undef, 0, N)
    B_card_eq = Vector{eltype(returns)}(undef, 0)
    if !asset_set_flag && !isempty(a_card_eq_constraints)
        port.a_card_eq_constraints = a_card_eq_constraints
        A, B = asset_constraints(a_card_eq_constraints, asset_sets)
        A_card_eq = vcat(A_card_eq, A)
        push!(B_card_eq, B)
    end
    if !loadings_flag && !isempty(f_card_eq_constraints)
        port.f_card_eq_constraints = f_card_eq_constraints
        A, B = factor_constraints(f_card_eq_constraints, loadings)
        A_card_eq = vcat(A_card_eq, A)
        push!(B_card_eq, B)
    end
    if !isempty(A_card_eq)
        port.a_card_eq, port.b_card_eq = A_card_eq, B_card_eq
    end

    return nothing
end

function calc_hc_constraints!(port::AbstractPortfolio;
                              hc_constraints::DataFrame = port.hc_constraints,
                              asset_sets::DataFrame = port.asset_sets)
    port.hc_constraints = hc_constraints
    port.asset_sets = asset_sets
    if isempty(hc_constraints) || isempty(asset_sets)
        return nothing
    end
    port.w_min, port.w_max = calc_hc_constraints(hc_constraints, asset_sets)

    return nothing
end

function calc_rb_constraints!(port::AbstractPortfolio;
                              rb_constraints::DataFrame = port.rb_constraints,
                              asset_sets::DataFrame = port.asset_sets)
    port.rb_constraints = rb_constraints
    port.asset_sets = asset_sets
    if isempty(rb_constraints) || isempty(asset_sets)
        return nothing
    end
    port.rb_constraints = calc_rb_constraints(rb_constraints, asset_sets)
    return nothing
end

function calc_frb_constraints!(port::AbstractPortfolio;
                               frb_constraints::DataFrame = port.frb_constraints,
                               asset_sets::DataFrame = port.asset_sets)
    port.frb_constraints = frb_constraints
    port.asset_sets = asset_sets
    if isempty(frb_constraints) || isempty(asset_sets)
        return nothing
    end
    port.frb_constraints = calc_rb_constraints(frb_constraints, asset_sets)
    return nothing
end

export calc_linear_constraints!, calc_hc_constraints!, calc_rb_constraints!,
       calc_frb_constraints!