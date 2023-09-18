function asset_constraints(constraints, asset_classes)
    m = nrow(asset_classes)
    asset_list = asset_classes[!, "Assets"]

    A = Matrix{Float64}(undef, 0, m)
    B = Float64[]

    for row in eachrow(constraints)
        !row["Enabled"] && continue

        if row["Sign"] == ">="
            d = 1
        elseif row["Sign"] == "<="
            d = -1
        end

        if row["Type"] == "Assets"
            idx = findfirst(x -> x == row["Position"], asset_list)
            A1 = zeros(m)
            if row["Weight"] != ""
                A1[idx] = d
                push!(B, row["Weight"] * d)
            else
                A1[idx] = 1
                if row["Type Relative"] == "Assets" && row["Relative"] != ""
                    idx2 = findfirst(x -> x == row["Relative"], asset_list)
                    A2 = zeros(m)
                    A2[idx2] = 1
                elseif row["Type Relative"] == "Classes" &&
                       row["Relative Set"] != "" &&
                       row["Relative"] != ""
                    A2 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
                end
                A1 = (A1 - A2 * row["Factor"]) * d
                push!(B, 0)
            end
            A = vcat(A, transpose(A1))
        elseif row["Type"] == "All Assets"
            A1 = I(m)
            if row["Weight"] != ""
                A1 *= d
                B1 = d * row["Weight"]
                B = vcat(B, fill(B1, m))
            else
                if row["Type Relative"] == "Assets" && row["Relative"] != ""
                    idx = findfirst(x -> x == row["Relative"], asset_list)
                    A2 = zeros(m, m)
                    A2[:, idx] .= 1
                elseif row["Type Relative"] == "Classes" &&
                       row["Relative Set"] != "" &&
                       row["Relative"] != ""
                    A2 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
                    A2 = ones(m, m) .* transpose(A2)
                end
                A1 = (A1 - A2 * row["Factor"]) * d
                B = vcat(B, zeros(m))
            end
            A = vcat(A, A1)
        elseif row["Type"] == "Classes"
            A1 = asset_classes[!, row["Set"]] .== row["Position"]
            if row["Weight"] != ""
                A1 = A1 * d
                push!(B, row["Weight"] * d)
            else
                if row["Type Relative"] == "Assets" && row["Relative"] != ""
                    idx = findfirst(x -> x == row["Relative"], asset_list)
                    A2 = zeros(m)
                    A2[idx] = 1
                elseif row["Type Relative"] == "Classes" &&
                       row["Relative Set"] != "" &&
                       row["Relative"] != ""
                    A2 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
                end
                A1 = (A1 - A2 * row["Factor"]) * d
                push!(B, 0)
            end
            A = vcat(A, transpose(A1))
        elseif row["Type"] == "All Classes"
            if row["Weight"] != ""
                for val in sort!(unique(asset_classes[!, row["Set"]]))
                    A1 = (asset_classes[!, row["Set"]] .== val) * d
                    A = vcat(A, transpose(A1))
                    push!(B, row["Weight"] * d)
                end
            else
                for val in sort!(unique(asset_classes[!, row["Set"]]))
                    A1 = asset_classes[!, row["Set"]] .== val
                    if row["Type Relative"] == "Assets" && row["Relative"] != ""
                        idx = findfirst(x -> x == row["Relative"], asset_list)
                        A2 = zeros(m)
                        A2[idx] = 1
                    elseif row["Type Relative"] == "Classes" &&
                           row["Relative Set"] != "" &&
                           row["Relative"] != ""
                        A2 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
                    end
                    A1 = (A1 - A2 * row["Factor"]) * d
                    A = vcat(A, transpose(A1))
                    push!(B, 0)
                end
            end
        elseif row["Type"] == "Each asset in a class"
            A1 = asset_classes[!, row["Set"]] .== row["Position"]
            if row["Weight"] != ""
                for (i, j) in pairs(A1)
                    !j && continue
                    A2 = zeros(m)
                    A2[i] = d
                    A = vcat(A, transpose(A2))
                    push!(B, row["Weight"] * d)
                end
            else
                for (i, j) in pairs(A1)
                    !j && continue
                    A2 = zeros(m)
                    A2[i] = 1
                    if row["Type Relative"] == "Assets" && row["Relative"] != ""
                        idx = findfirst(x -> x == row["Relative"], asset_list)
                        A3 = zeros(m)
                        A3[idx] = 1
                    elseif row["Type Relative"] == "Classes" &&
                           row["Relative Set"] != "" &&
                           row["Relative"] != ""
                        A3 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
                    end
                    A2 = (A2 - A3 * row["Factor"]) * d
                    A = vcat(A, transpose(A2))
                    push!(B, 0)
                end
            end
        end
    end

    return A, B
end

function factor_constraints(constraints, loadings)
    m = nrow(loadings)

    C = Matrix{Float64}(undef, 0, m)
    D = Float64[]
    for row in eachrow(constraints)
        !row["Enabled"] && continue

        if row["Sign"] == ">="
            d = 1
        elseif row["Sign"] == "<="
            d = -1
        end

        C1 = loadings[!, row["Factor"]]
        if row["Relative Factor"] != ""
            C2 = loadings[!, row["Relative Factor"]]
            C1 = C1 - C2
        end

        C = vcat(C, transpose(C1) * d)
        push!(D, row["Value"] * d)
    end

    return C, D
end

function asset_views(views, asset_classes)
    m = nrow(asset_classes)
    asset_list = asset_classes[!, "Assets"]

    P = Matrix{Float64}(undef, 0, m)
    Q = Float64[]

    for row in eachrow(views)
        valid = false

        !row["Enabled"] || row["Return"] == "" && continue

        if row["Sign"] == ">="
            d = 1
        elseif row["Sign"] == "<="
            d = -1
        end

        if row["Type"] == "Assets"
            idx = findfirst(x -> x == row["Position"], asset_list)
            P1 = zeros(m)
            P1[idx] = 1
        elseif row["Type"] == "Classes"
            P1 = asset_classes[!, row["Set"]] .== row["Position"]
            P1 = P1 / sum(P1)
        end

        if row["Type Relative"] == "Assets" && row["Relative"] != ""
            idx2 = findfirst(x -> x == row["Relative"], asset_list)
            P2 = zeros(m)
            P2[idx2] = 1
            valid = true
        elseif row["Type Relative"] == "Classes" &&
               row["Relative Set"] != "" &&
               row["Relative"] != ""
            P2 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
            P2 = P2 / sum(P2)
            valid = true
        elseif row["Type Relative"] == "" &&
               row["Relative Set"] == "" &&
               row["Relative"] == ""
            P2 = zeros(m)
            valid = true
        end

        if valid
            P1 = (P1 - P2) * d
            P = vcat(P, transpose(P1))
            push!(Q, row["Return"] * d)
        end
    end

    for i in eachindex(view(Q, :, 1))
        if Q[i, 1] < 0
            P[i, :] .= -P[i, :]
            Q[i] .= -Q[i]
        end
    end

    return P, Q
end

function factor_views(views, loadings)
    m = ncol(loadings)
    factor_list = names(loadings)
    P = Matrix{Float64}(undef, 0, m)
    Q = Float64[]

    for row in eachrow(views)
        !row["Enabled"] && continue

        if row["Sign"] == ">="
            d = 1
        elseif row["Sign"] == "<="
            d = -1
        end

        idx = findfirst(x -> x == row["Factor"], factor_list)
        P1 = zeros(m)
        P1[idx] = d

        if row["Relative Factor"] != ""
            idx = findfirst(x -> x == row["Relative Factor"], factor_list)
            P1[idx] = -d
        end

        P = vcat(P, transpose(P1))
        push!(Q, row["Value"] * d)
    end

    return P, Q
end

function hrp_constraints(constraints, asset_classes)
    n = nrow(asset_classes)
    w = zeros(n, 2)
    w[:, 2] .= 1
    for row in eachrow(constraints)
        !row["Enabled"] && continue

        if row["Sign"] == ">="
            i = 1
            op = <=
        elseif row["Sign"] == "<="
            i = 2
            op = >=
        end

        if row["Type"] == "Assets"
            idx = findfirst(x -> x == row["Position"], asset_classes[!, "Assets"])
            op(w[idx, i], row["Weight"]) && (w[idx, i] = row["Weight"])
        elseif row["Type"] == "All Assets"
            !isempty(w[op.(w[:, i], row["Weight"]), i]) &&
                (w[op.(w[:, i], row["Weight"]), i] .= row["Weight"])
        elseif row["Type"] == "Each asset in a class"
            assets =
                asset_classes[asset_classes[!, row["Set"]] .== row["Position"], "Assets"]
            idx =
                [findfirst(x -> x == asset, asset_classes[!, "Assets"]) for asset in assets]

            for ind in idx
                if !isnothing(ind) && op(w[ind, i], row["Weight"])
                    w[ind, i] = row["Weight"]
                end
            end
        end
    end
    return w[:, 1], w[:, 2]
end

function risk_parity_constraints(asset_classes, type = :assets, class_col = nothing)
    @assert(type ∈ RPConstraintTypes, "type must be one of $RPConstraintTypes")
end

export asset_constraints, factor_constraints, asset_views, factor_views, hrp_constraints
