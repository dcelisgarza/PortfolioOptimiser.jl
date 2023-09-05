function asset_constraints(constraints, asset_classes)
    n = nrow(constraints)
    m = nrow(asset_classes)

    asset_list = asset_classes[!, "Assets"]
    num_assets = length(asset_list)

    A = Matrix{Float64}(undef, 0, num_assets)
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
                if row["Type Relative"] == "Assets"
                    idx2 = findfirst(x -> x == row["Relative"], asset_list)
                    A2 = zeros(m)
                    A2[idx2] = 1
                elseif row["Type Relative"] == "Classes"
                    A2 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
                end
                A1 = (A1 - A2 * row["Factor"]) * d
                push!(B, 0)
            end
            A = vcat(A, transpose(A1))
        elseif row["Type"] == "All Assets"
            A1 = I(num_assets)
            if row["Weight"] != ""
                A1 *= d
                B1 = d * row["Weight"]
                B = vcat(B, fill(B1, num_assets))
            else
                if row["Type Relative"] == "Assets"
                    idx = findfirst(x -> x == row["Relative"], asset_list)
                    A2 = zeros(num_assets, num_assets)
                    A2[:, idx] .= 1
                elseif row["Type Relative"] == "Classes"
                    A2 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
                    A2 = ones(num_assets, num_assets) .* transpose(A2)
                end
                A1 = (A1 - A2 * row["Factor"]) * d
                B = vcat(B, zeros(num_assets))
            end
            A = vcat(A, transpose(A1))
        elseif row["Type"] == "Classes"
            A1 = asset_classes[!, row["Set"]] .== row["Position"]
            if row["Weight"] != ""
                A1 = A1 * d
                push!(B, row["Weight"] * d)
            else
                if row["Type Relative"] == "Assets"
                    idx = findfirst(x -> x == row["Relative"], asset_list)
                    A2 = zeros(m)
                    A2[idx] = 1
                elseif row["Type Relative"] == "Classes"
                    A2 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
                end
                A1 = (A1 - A2 * row["Factor"]) * d
                push!(B, 0)
            end
            A = vcat(A, transpose(A1))
        elseif row["Type"] == "All Classes"
            if row["Weight"] != ""
                for val in unique(asset_classes[!, row["Set"]])
                    A1 = (asset_classes[!, row["Set"]] .== val) * d
                    A = vcat(A, transpose(A1))
                    push!(B, row["Weight"] * d)
                end
            else
                for val in unique(asset_classes[!, row["Set"]])
                    A1 = asset_classes[!, row["Set"]] .== val
                    if row["Type Relative"] == "Assets"
                        idx = findfirst(x -> x == row["Relative"], asset_list)
                        A2 = zeros(m)
                        A2[idx] = 1
                    elseif row["Type Relative"] == "Classes"
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
                    j == 0 && continue
                    A2 = zeros(m)
                    A2[i] = d
                    A = vcat(A, transpose(A2))
                    push!(B, row["Weight"] * d)
                end
            else
                for (i, j) in pairs(A1)
                    j == 0 && continue
                    A2 = zeros(m)
                    A2[i] = 1
                    if row["Type Relative"] == "Assets"
                        idx = findfirst(x -> x == row["Relative"], asset_list)
                        A3 = zeros(m)
                        A3[idx] = 1
                    elseif row["Type Relative"] == "Classes"
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

export asset_constraints
