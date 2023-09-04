function asset_constraints(constraints, asset_classes)
    n = nrow(constraints)
    m = nrow(constraints)

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
        elseif row["Type"] == "All Assets"
            A1 = I(num_assets)
            if row["Weight"] != ""
                A1 *= d
                B1 = d * row["Weight"]

                A = vcat(A, A1)
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
                A1 = (A1 + A2 * row["Factor"] * -1) * d

                A = vcat(A, A1)
                B = vcat(B, fill(0, num_assets))
            end
        elseif row["Type"] == "Classes"
        elseif row["Type"] == "All Classes"
        elseif row["Type"] == "Each asset in a class"
        end
    end

    return A, B
end

export asset_constraints
