# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function symmetric_step_up_matrix(n1::Integer, n2::Integer)
    @smart_assert(abs(n1 - n2) <= 1)
    if n1 == n2
        return I(n1)
    elseif n1 < n2
        return transpose(symmetric_step_up_matrix(n2, n1)) * n1 / n2
    end

    m = zeros(n1, n2)
    e = vcat(ones(1, n2) / n2, I(n2))
    m .+= e
    for i ∈ 1:(n1 - 1)
        s1 = e[i, :]
        s2 = e[i + 1, :]
        e[i, :] .= s2
        e[i + 1, :] .= s1
        m .+= e
    end
    return m / n1
end
function single_schur_augmentation(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix,
                                   gamma::Real = 0.5, delta::Real = gamma)
    Na = size(A, 1)
    Nd = size(D, 1)

    A_aug = A - gamma * B * (D \ transpose(B))
    m = symmetric_step_up_matrix(Na, Nd)
    r = I - delta * (transpose(B) \ transpose(D)) * transpose(m)
    A_aug = r \ A_aug
    A_aug = (A_aug + transpose(A_aug)) / 2

    return A_aug
end
function schur_augmentation(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix,
                            gamma::Real = 0.5, tol::Real = 1e-2, max_iter::Integer = 10)
    Na = size(A, 1)
    Nd = size(D, 1)
    if iszero(gamma) || isone(Na) || isone(Nd)
        return A, D
    end

    counter = 0
    valid_A_aug = nothing
    valid_D_aug = nothing
    low = 0
    high = gamma
    old_gamma = gamma
    while counter <= max_iter
        A_aug = single_schur_augmentation(A, B, D, gamma, gamma)
        D_aug = single_schur_augmentation(D, transpose(B), A, gamma, gamma)

        if isposdef(A_aug) && isposdef(D_aug)
            valid_A_aug = A_aug
            valid_D_aug = D_aug
            if abs(gamma - old_gamma) <= tol
                break
            else
                low = gamma
            end
        else
            high = gamma
        end

        old_gamma = gamma
        gamma = (low + high) / 2
        counter += 1
    end

    return if isnothing(valid_A_aug)
        A, D
    else
        valid_A_aug, valid_D_aug
    end
end
function naive_portfolio_variance(sigma::AbstractMatrix)
    w = inv.(diag(sigma))
    w ./= sum(w)
    return dot(w, sigma, w)
end
function schur_optimise(port::Portfolio, params, sigma::AbstractMatrix)
    N = size(port.returns, 2)
    items = [port.clusters.order]

    min_cluster_size = port.min_cluster_size

    weights = zeros(eltype(sigma), N)

    for param ∈ params
        (; rm, gamma, prop_coef, tol, max_iter) = param
        scale = rm.settings.scale
        sigma_i = if isnothing(rm.sigma) || isempty(rm.sigma)
            copy(sigma)
        else
            copy(rm.sigma)
        end
        weights_i = ones(eltype(sigma), N)
        while length(items) > 0
            items = [i[j:k] for i ∈ items
                     for (j, k) ∈
                         ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                     if length(i) > 1]

            for i ∈ 1:2:length(items)
                lc = items[i]
                rc = items[i + 1]

                A = sigma_i[lc, lc]
                D = sigma_i[rc, rc]

                A_aug, D_aug = if length(lc) < min_cluster_size
                    A, D
                else
                    B = sigma_i[lc, rc]
                    schur_augmentation(A, B, D, gamma, tol, max_iter)
                end

                sigma_i[lc, lc] .= A * (1 - prop_coef) + A_aug * prop_coef
                sigma_i[rc, rc] .= D * (1 - prop_coef) + D_aug * prop_coef

                lrisk = naive_portfolio_variance(A_aug)
                rrisk = naive_portfolio_variance(D_aug)

                alpha_1 = one(lrisk) - lrisk / (lrisk + rrisk)
                # Weight constraints.
                weights_i[lc] *= alpha_1
                weights_i[rc] *= one(alpha_1) - alpha_1
            end
        end
        weights .+= weights_i * inv(scale)
    end
    weights ./= sum(weights)

    return weights
end
