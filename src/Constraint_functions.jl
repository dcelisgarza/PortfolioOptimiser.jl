"""
```julia
asset_constraints(constraints::DataFrame, asset_classes::DataFrame)
```
Create the linear constraint matrix `A` and vector `B` of the constraints:
- ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``.
# Inputs
- `constraints`: `Nc×10` Dataframe, where `Nc` is the number of constraints. The required columns are:
    - `Enabled`: (Bool) indicates if the constraint is enabled.
    - `Type`: (String) specifies the object(s) to which a constraint applies:
        - `Assets`: specific asset.
        - `Classes`: whole class.
        - `All Assets`: all assets.
        - `All Classes`: all asset classes.
        - `Each Asset in Class`: specific assets in a class.
    - `Set`: (String) if `Type` is `Classes`, `All Classes` or `Each Asset in Class`, specifies the asset class set.
    - `Position`: (String) name of the asset or asset class to which the constraint applies.
    - `Sign`: (String) specifies whether the constraint is a lower or upper bound:
        - `>=`: lower bound.
        - `<=`: upper bound.
    - `Weight`: (<:Real) value of the constraint.
    - `Type Relative`: (String) specifies to what the constraint is relative to:
        - Empty string: nothing.
        - `Assets`: other asset.
        - `Classes`: other class.
    - `Relative Set`: (String) if `Type Relative` is `Classes`, specifies the name of the set of asset classes.
    - `Relative`: (String) name of the asset or asset class of the relative constraint.
    - `Factor`: (<:Real) the factor of the relative constraint.
- `asset_classes`: `Na×C` DataFrame where `Na` is the number of assets and `C` the number of columns.
    - `Assets`: list of assets, this is the only mandatory column.
    - Subsequent columns specify the asset class sets.
# Outputs
- `A`: `C×N` matrix of constraints where `C` is the number of constraints and `N` the number of assets.
- `A`: `C×1` vector of constraints where `C` is the number of constraints.
# Examples
```@repl constraint_examples
asset_classes = DataFrame(
    "Assets" => ["FB", "GOOGL", "NTFX", "BAC", "WFC", "TLT", "SHV", "FCN", "TKO", "ZOO", "ZVO", "ZX", "ZZA", "ZZB", "ZZC"],
    "Class 1" => ["Equity", "Equity", "Equity", "Equity", "Equity", "Fixed Income", "Fixed Income", "Equity", "Equity", "Equity", "Fixed Income", "Fixed Income", "Equity", "Fixed Income", "Equity"],
    "Class 2" => ["Technology", "Technology", "Technology", "Financial", "Financial", "Treasury", "Treasury", "Financial", "Entertainment", "Treasury", "Financial", "Financial", "Entertainment", "Technology", "Treasury"],
)
constraints = DataFrame(
    "Enabled" => [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true],
    "Type" => ["Classes", "All Classes", "Assets", "Assets", "Classes", "All Assets", "Each Asset in Class", "Assets", "All Assets", "All Assets", "Classes", "All Classes", "All Classes", "Each Asset in Class", "Each Asset in Class"],
    "Set" => ["Class 1", "Class 1", "", "", "Class 2", "", "Class 1", "Class 1", "Class 2", "", "Class 1", "Class 2", "Class 2", "Class 2", "Class 1"],
    "Position" => ["Equity", "Fixed Income", "BAC", "WFC", "Financial", "", "Equity", "FCN", "TKO", "ZOO", "Fixed Income", "Treasury", "Entertainment", "Treasury", "Equity"],
    "Sign" => ["<=", "<=", "<=", "<=", ">=", ">=", ">=", "<=", ">=", "<=", ">=", "<=", ">=", "<=", ">="],
    "Weight" => [0.6, 0.5, 0.1, "", "", 0.02, "", "", "", "", "", "", "", 0.27, ""],
    "Type Relative" => ["", "", "", "Assets", "Classes", "", "Assets", "Classes", "Assets", "Classes", "Assets", "Assets", "Classes", "", "Classes"],
    "Relative Set" => ["", "", "", "", "Class 1", "", "", "Class 1", "", "Class 2", "", "Class 2", "Class 2", "", "Class 2"],
    "Relative" => ["", "", "", "FB", "Fixed Income", "", "TLT", "Equity", "NTFX", "Financial", "WFC", "ZOO", "Entertainment", "", "Entertainment"],
    "Factor" => ["", "", "", 1.2, 0.5, "", 0.4, 0.7, 0.21, 0.11, 0.13, -0.17, 0.23, "", -0.31],
)
A, B = asset_constraints(constraints, asset_classes)
A
B
```
"""
function asset_constraints(constraints::DataFrame, asset_classes::DataFrame)
    N = nrow(asset_classes)
    asset_list = asset_classes[!, "Assets"]

    A = Matrix{Float64}(undef, 0, N)
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
            A1 = zeros(N)
            if row["Weight"] != ""
                A1[idx] = d
                push!(B, row["Weight"] * d)
            else
                A1[idx] = 1
                if row["Type Relative"] == "Assets" && row["Relative"] != ""
                    idx2 = findfirst(x -> x == row["Relative"], asset_list)
                    A2 = zeros(N)
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
            A1 = I(N)
            if row["Weight"] != ""
                A1 *= d
                B1 = d * row["Weight"]
                B = vcat(B, fill(B1, N))
            else
                if row["Type Relative"] == "Assets" && row["Relative"] != ""
                    idx = findfirst(x -> x == row["Relative"], asset_list)
                    A2 = zeros(N, N)
                    A2[:, idx] .= 1
                elseif row["Type Relative"] == "Classes" &&
                       row["Relative Set"] != "" &&
                       row["Relative"] != ""
                    A2 = asset_classes[!, row["Relative Set"]] .== row["Relative"]
                    A2 = ones(N, N) .* transpose(A2)
                end
                A1 = (A1 - A2 * row["Factor"]) * d
                B = vcat(B, zeros(N))
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
                    A2 = zeros(N)
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
                        A2 = zeros(N)
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
        elseif row["Type"] == "Each Asset in Class"
            A1 = asset_classes[!, row["Set"]] .== row["Position"]
            if row["Weight"] != ""
                for (i, j) in pairs(A1)
                    !j && continue
                    A2 = zeros(N)
                    A2[i] = d
                    A = vcat(A, transpose(A2))
                    push!(B, row["Weight"] * d)
                end
            else
                for (i, j) in pairs(A1)
                    !j && continue
                    A2 = zeros(N)
                    A2[i] = 1
                    if row["Type Relative"] == "Assets" && row["Relative"] != ""
                        idx = findfirst(x -> x == row["Relative"], asset_list)
                        A3 = zeros(N)
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

"""
```julia
factor_constraints(constraints::DataFrame, loadings::DataFrame)
```
Create the factor constraints matrix `C` and vector `D` of the constraints:
- ``\\mathbf{C} \\bm{w} \\geq \\bm{D}``.
# Inputs
- `constraints`: `Nc×4` Dataframe, where `Nc` is the number of constraints. The required columns are:
    - `Enabled`: (Bool) indicates if the constraint is enabled.
    - `Factor`: (String) name of the constraint's factor.
    - `Sign`: (String) specifies whether the constraint is a lower or upper bound:
        - `>=`: lower bound.
        - `<=`: upper bound.
    - `Value`: (<:Real) the upper or lower bound of the factor's value.
- `loadings`: `Na×Nf` where `Na` is the number of assets and `Nf` is the number of factors.
# Examples
```@repl constraint_examples
loadings = DataFrame(
        "const" => [0.0004, 0.0002, 0.0000, 0.0006, 0.0001, 0.0003, -0.0003],
        "MTUM" => [0.1916, 1.0061, 0.8695, 1.9996, 0.0000, 0.0000, 0.0000],
        "QUAL" => [0.0000, 2.0129, 1.4301, 0.0000, 0.0000, 0.0000, 0.0000],
        "SIZE" => [0.0000, 0.0000, 0.0000, 0.4717, 0.0000, -0.1857, 0.0000],
        "USMV" => [-0.7838, -1.6439, -1.0176, -1.4407, 0.0055, 0.5781, 0.0000],
        "VLUE" => [1.4772, -0.7590, -0.4090, 0.0000, -0.0054, -0.4844, 0.9435],
    )
constraints = DataFrame(
        "Enabled" => [true, true, true, true],
        "Factor" => ["MTUM", "USMV", "VLUE", "const"],
        "Sign" => ["<=", "<=", ">=", ">="],
        "Value" => [0.9, -1.2, 0.3, -0.1],
        "Relative Factor" => ["USMV", "", "", "SIZE"],
    )
C, D = factor_constraints(constraints, loadings)
C
D
```
"""
function factor_constraints(constraints::DataFrame, loadings::DataFrame)
    N = nrow(loadings)

    C = Matrix{Float64}(undef, 0, N)
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
    N = nrow(asset_classes)
    asset_list = asset_classes[!, "Assets"]

    P = Matrix{Float64}(undef, 0, N)
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
            P1 = zeros(N)
            P1[idx] = 1
        elseif row["Type"] == "Classes"
            P1 = asset_classes[!, row["Set"]] .== row["Position"]
            P1 = P1 / sum(P1)
        end

        if row["Type Relative"] == "Assets" && row["Relative"] != ""
            idx2 = findfirst(x -> x == row["Relative"], asset_list)
            P2 = zeros(N)
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
            P2 = zeros(N)
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
            Q[i] = -Q[i]
        end
    end

    return P, Q
end

function factor_views(views, loadings)
    factor_list = names(loadings)
    "const" ∈ factor_list && (factor_list = setdiff(factor_list, ("const",)))
    "ticker" ∈ factor_list && (factor_list = setdiff(factor_list, ("ticker",)))

    N = length(factor_list)

    P = Matrix{Float64}(undef, 0, N)
    Q = Float64[]

    for row in eachrow(views)
        !row["Enabled"] && continue

        if row["Sign"] == ">="
            d = 1
        elseif row["Sign"] == "<="
            d = -1
        end

        idx = findfirst(x -> x == row["Factor"], factor_list)
        P1 = zeros(N)
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
    N = nrow(asset_classes)
    w = zeros(N, 2)
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
        elseif row["Type"] == "Each Asset in Class"
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

"""
"""
function rp_constraints(asset_classes, type = :Assets, class_col = nothing)
    @assert(type ∈ RPConstraintTypes, "type = $type, must be one of $RPConstraintTypes")
    N = nrow(asset_classes)

    w = if type == :Assets
        fill(1 / N, N)
    else
        classes = names(asset_classes)
        isa(class_col, Symbol) && (class_col = String(class_col))
        if class_col ∈ classes
            A = DataFrame(a = asset_classes[!, class_col])
            rename!(A, [class_col])
        elseif isa(class_col, Int) && class_col < N
            A = DataFrame(a = asset_classes[!, class_col])
            rename!(A, [classes[class_col]])
        else
            throw(ArgumentError("class_col must be a valid index of asset_classes"))
        end

        col = names(A)[1]
        A[!, :weight] .= 1
        B = combine(groupby(A, col), nrow => :count)
        A = leftjoin(A, B, on = col)
        A[!, :weight] ./= A[!, :count]
        A[!, :weight] ./= sum(A[!, :weight])
        A[!, :weight]
    end

    return w
end

export asset_constraints,
    factor_constraints, asset_views, factor_views, hrp_constraints, rp_constraints
