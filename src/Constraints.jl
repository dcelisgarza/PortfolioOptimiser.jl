"""
```julia
asset_constraints(constraints::DataFrame, asset_sets::DataFrame)
```
Create the linear constraint matrix `A` and vector `B`:
- ``\\mathbf{A} \\bm{w} \\geq \\bm{B}``.
# Inputs
- `constraints`: `Nc×10` Dataframe, where $(_ndef(:c1)). The required columns are:
    - `Enabled`: (Bool) indicates if the constraint is enabled.
    - `Type`: (String) specifies the object(s) to which a constraint applies:
        - `Asset`: specific asset.
        - `Subset`: whole class.
        - `All Assets`: all assets.
        - `All Subsets`: all asset classes.
        - `Each Asset in Subset`: specific assets in a class.
    - `Set`: (String) if `Type` is `Subset`, `All Subsets` or `Each Asset in Subset`, specifies the asset class set.
    - `Position`: (String) name of the asset or asset class to which the constraint applies.
    - `Sign`: (String) specifies whether the constraint is a lower or upper bound:
        - `>=`: lower bound.
        - `<=`: upper bound.
    - `Weight`: (<:Real) value of the constraint.
    - `Relative_Type`: (String) specifies to what the constraint is relative:
        - Empty string: nothing.
        - `Asset`: other asset.
        - `Subset`: other class.
    - `Relative_Set`: (String) if `Relative_Type` is `Subset`, specifies the name of the set of asset classes.
    - `Relative_Position`: (String) name of the asset or asset class of the relative constraint.
    - `Factor`: (<:Real) the factor of the relative constraint.
- `asset_sets`: `Na×D` DataFrame where $(_ndef(:a2)) and `D` the number of columns.
    - `Asset`: list of assets, this is the only mandatory column.
    - Subsequent columns specify the asset class sets.
# Outputs
- `A`: `Nc×Na` matrix of constraints where $(_ndef(:c1)) and $(_ndef(:a2)).
- `B`: `Nc×1` vector of constraints where $(_ndef(:c1)).
# Examples
```julia
asset_sets = DataFrame(
    "Asset" => ["FB", "GOOGL", "NTFX", "BAC", "WFC", "TLT", "SHV", "FCN", "TKO", "ZOO", "ZVO", "ZX", "ZZA", "ZZB", "ZZC"],    "Class 1" => ["Equity", "Equity", "Equity", "Equity", "Equity", "Fixed Income", "Fixed Income", "Equity", "Equity", "Equity", "Fixed Income", "Fixed Income", "Equity", "Fixed Income", "Equity"],    "Class 2" => ["Technology", "Technology", "Technology", "Financial", "Financial", "Treasury", "Treasury", "Financial", "Entertainment", "Treasury", "Financial", "Financial", "Entertainment", "Technology", "Treasury"],)
constraints = DataFrame(
    "Enabled" => [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true],    "Type" => ["Subset", "All Subsets", "Asset", "Asset", "Subset", "All Assets", "Each Asset in Subset", "Asset", "All Assets", "All Assets", "Subset", "All Subsets", "All Subsets", "Each Asset in Subset", "Each Asset in Subset"],    "Set" => ["Class 1", "Class 1", "", "", "Class 2", "", "Class 1", "Class 1", "Class 2", "", "Class 1", "Class 2", "Class 2", "Class 2", "Class 1"],    "Position" => ["Equity", "Fixed Income", "BAC", "WFC", "Financial", "", "Equity", "FCN", "TKO", "ZOO", "Fixed Income", "Treasury", "Entertainment", "Treasury", "Equity"],    "Sign" => ["<=", "<=", "<=", "<=", ">=", ">=", ">=", "<=", ">=", "<=", ">=", "<=", ">=", "<=", ">="],    "Weight" => [0.6, 0.5, 0.1, "", "", 0.02, "", "", "", "", "", "", "", 0.27, ""],    "Relative_Type" => ["", "", "", "Asset", "Subset", "", "Asset", "Subset", "Asset", "Subset", "Asset", "Asset", "Subset", "", "Subset"],    "Relative_Set" => ["", "", "", "", "Class 1", "", "", "Class 1", "", "Class 2", "", "Class 2", "Class 2", "", "Class 2"],    "Relative_Position" => ["", "", "", "FB", "Fixed Income", "", "TLT", "Equity", "NTFX", "Financial", "WFC", "ZOO", "Entertainment", "", "Entertainment"],    "Factor" => ["", "", "", 1.2, 0.5, "", 0.4, 0.7, 0.21, 0.11, 0.13, -0.17, 0.23, "", -0.31],)
A, B = asset_constraints(constraints, asset_sets)
```
"""
function asset_constraints(constraints::DataFrame, asset_sets::DataFrame)
    N = nrow(asset_sets)
    asset_list = asset_sets[!, "Asset"]

    A = Matrix(undef, 0, N)
    B = Vector(undef, 0)

    for row in eachrow(constraints)
        if !row["Enabled"]
            continue
        end

        if row["Sign"] == ">="
            d = 1
        elseif row["Sign"] == "<="
            d = -1
        end

        if row["Type"] == "Asset"
            idx = findfirst(x -> x == row["Position"], asset_list)
            A1 = zeros(N)
            if row["Weight"] != ""
                A1[idx] = d
                push!(B, row["Weight"] * d)
            else
                A1[idx] = 1
                if row["Relative_Type"] == "Asset" && row["Relative_Position"] != ""
                    idx2 = findfirst(x -> x == row["Relative_Position"], asset_list)
                    A2 = zeros(N)
                    A2[idx2] = 1
                elseif row["Relative_Type"] == "Subset" &&
                       row["Relative_Set"] != "" &&
                       row["Relative_Position"] != ""
                    A2 = asset_sets[!, row["Relative_Set"]] .== row["Relative_Position"]
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
                if row["Relative_Type"] == "Asset" && row["Relative_Position"] != ""
                    idx = findfirst(x -> x == row["Relative_Position"], asset_list)
                    A2 = zeros(N, N)
                    A2[:, idx] .= 1
                elseif row["Relative_Type"] == "Subset" &&
                       row["Relative_Set"] != "" &&
                       row["Relative_Position"] != ""
                    A2 = asset_sets[!, row["Relative_Set"]] .== row["Relative_Position"]
                    A2 = ones(N, N) .* transpose(A2)
                    # else
                    #     @warn(
                    #         """
                    #         Constraints DataFrame not created correctly.
                    #         - row["Type"] = $(row["Type"])
                    #         First check to see if this holds.
                    #         - row["Relative_Type"] == "Asset" && row["Relative_Position"] != ""
                    #         The items evaluate to:
                    #         - row["Relative_Type"] = $(row["Relative_Type"])
                    #         - row["Relative_Position"] != "" = $(row["Relative_Position"] != "")
                    #         Otherwise check this holds.
                    #         - row["Relative_Type"] == "Subset" && row["Relative_Set"] != "" && row["Relative_Position"] != ""
                    #         The items evaluate to:
                    #         - row["Relative_Type"] = $(row["Relative_Type"])
                    #         - row["Relative_Set"] != "" = $(row["Relative_Set"] != "")
                    #         - row["Relative_Position"] != "" = $(row["Relative_Position"] != "")
                    #         """
                    #     )
                end
                A1 = (A1 - A2 * row["Factor"]) * d
                B = vcat(B, zeros(N))
            end
            A = vcat(A, A1)
        elseif row["Type"] == "Subset"
            A1 = asset_sets[!, row["Set"]] .== row["Position"]
            if row["Weight"] != ""
                A1 = A1 * d
                push!(B, row["Weight"] * d)
            else
                if row["Relative_Type"] == "Asset" && row["Relative_Position"] != ""
                    idx = findfirst(x -> x == row["Relative_Position"], asset_list)
                    A2 = zeros(N)
                    A2[idx] = 1
                elseif row["Relative_Type"] == "Subset" &&
                       row["Relative_Set"] != "" &&
                       row["Relative_Position"] != ""
                    A2 = asset_sets[!, row["Relative_Set"]] .== row["Relative_Position"]
                end
                A1 = (A1 - A2 * row["Factor"]) * d
                push!(B, 0)
            end
            A = vcat(A, transpose(A1))
        elseif row["Type"] == "All Subsets"
            if row["Weight"] != ""
                for val in sort!(unique(asset_sets[!, row["Set"]]))
                    A1 = (asset_sets[!, row["Set"]] .== val) * d
                    A = vcat(A, transpose(A1))
                    push!(B, row["Weight"] * d)
                end
            else
                for val in sort!(unique(asset_sets[!, row["Set"]]))
                    A1 = asset_sets[!, row["Set"]] .== val
                    if row["Relative_Type"] == "Asset" && row["Relative_Position"] != ""
                        idx = findfirst(x -> x == row["Relative_Position"], asset_list)
                        A2 = zeros(N)
                        A2[idx] = 1
                    elseif row["Relative_Type"] == "Subset" &&
                           row["Relative_Set"] != "" &&
                           row["Relative_Position"] != ""
                        A2 = asset_sets[!, row["Relative_Set"]] .== row["Relative_Position"]
                    end
                    A1 = (A1 - A2 * row["Factor"]) * d
                    A = vcat(A, transpose(A1))
                    push!(B, 0)
                end
            end
        elseif row["Type"] == "Each Asset in Subset"
            A1 = asset_sets[!, row["Set"]] .== row["Position"]
            if row["Weight"] != ""
                for (i, j) in pairs(A1)
                    if !j
                        continue
                    end
                    A2 = zeros(N)
                    A2[i] = d
                    A = vcat(A, transpose(A2))
                    push!(B, row["Weight"] * d)
                end
            else
                for (i, j) in pairs(A1)
                    if !j
                        continue
                    end
                    A2 = zeros(N)
                    A2[i] = 1
                    if row["Relative_Type"] == "Asset" && row["Relative_Position"] != ""
                        idx = findfirst(x -> x == row["Relative_Position"], asset_list)
                        A3 = zeros(N)
                        A3[idx] = 1
                    elseif row["Relative_Type"] == "Subset" &&
                           row["Relative_Set"] != "" &&
                           row["Relative_Position"] != ""
                        A3 = asset_sets[!, row["Relative_Set"]] .== row["Relative_Position"]
                    end
                    A2 = (A2 - A3 * row["Factor"]) * d
                    A = vcat(A, transpose(A2))
                    push!(B, 0)
                end
            end
        end
    end

    A = convert.(typeof(promote(A...)[1]), A)
    B = convert.(typeof(promote(B...)[1]), B)

    return A, B
end

"""
```julia
factor_constraints(constraints::DataFrame, loadings::DataFrame)
```
Create the factor constraints matrix `C` and vector `D`:
- ``\\mathbf{C} \\bm{w} \\geq \\bm{D}``.
# Inputs
- `constraints`: `Nc×4` Dataframe, where $(_ndef(:c1)). The required columns are:
    - `Enabled`: (Bool) indicates if the constraint is enabled.
    - `Factor`: (String) name of the constraint's factor.
    - `Sign`: (String) specifies whether the constraint is a lower or upper bound:
        - `>=`: lower bound.
        - `<=`: upper bound.
    - `Value`: (<:Real) the upper or lower bound of the factor's value.
    - `Relative_Factor`: (String) factor to which the constraint is relative.
- `loadings`: `Nl×Nf` loadings DataFrame, where `Nl` is the number of data points, and $(_ndef(:f2)).
# Outputs
- `C`: `Nc×Nf` matrix of constraints where $(_ndef(:c1)) and $(_ndef(:f2)).
- `D`: `Nc×1` vector of constraints where $(_ndef(:c1)).
# Examples
```julia
loadings = DataFrame(
        "const" => [0.0004, 0.0002, 0.0000, 0.0006, 0.0001, 0.0003, -0.0003],        "MTUM" => [0.1916, 1.0061, 0.8695, 1.9996, 0.0000, 0.0000, 0.0000],        "QUAL" => [0.0000, 2.0129, 1.4301, 0.0000, 0.0000, 0.0000, 0.0000],        "SIZE" => [0.0000, 0.0000, 0.0000, 0.4717, 0.0000, -0.1857, 0.0000],        "USMV" => [-0.7838, -1.6439, -1.0176, -1.4407, 0.0055, 0.5781, 0.0000],        "VLUE" => [1.4772, -0.7590, -0.4090, 0.0000, -0.0054, -0.4844, 0.9435],    )
constraints = DataFrame(
        "Enabled" => [true, true, true, true],        "Factor" => ["MTUM", "USMV", "VLUE", "const"],        "Sign" => ["<=", "<=", ">=", ">="],        "Value" => [0.9, -1.2, 0.3, -0.1],        "Relative_Factor" => ["USMV", "", "", "SIZE"],    )
C, D = factor_constraints(constraints, loadings)
```
"""
function factor_constraints(constraints::DataFrame, loadings::DataFrame)
    N = nrow(loadings)

    C = Matrix(undef, 0, N)
    D = Vector(undef, 0)

    for row in eachrow(constraints)
        if !row["Enabled"]
            continue
        end

        if row["Sign"] == ">="
            d = 1
        elseif row["Sign"] == "<="
            d = -1
        end

        C1 = loadings[!, row["Factor"]]
        if row["Relative_Factor"] != ""
            C2 = loadings[!, row["Relative_Factor"]]
            C1 = C1 - C2
        end

        C = vcat(C, transpose(C1) * d)
        push!(D, row["Value"] * d)
    end

    C = convert.(typeof(promote(C...)[1]), C)
    D = convert.(typeof(promote(D...)[1]), D)

    return C, D
end

"""
```julia
asset_views(views::DataFrame, asset_sets::DataFrame)
```
Create the asset views matrix `P` and vector `Q`:
- ``\\mathbf{P} \\bm{w} \\geq \\bm{Q}``.
# Inputs
- `views`: `Nv×9` DataFrame, where `Nv` is the number of views. The required columns are:
    - `Enabled`: (Bool) indicates if the view is enabled.
    - `Type`: (String) specifies the object(s) to which a view applies:
        - `Asset`: specific asset.
        - `Subset`: whole class.
    - `Set`: (String) if `Type` is `Subset`, specifies the asset class set.
    - `Position`: (String) name of the asset or asset class to which the view applies.
    - `Sign`: (String) specifies whether the view is a lower or upper bound:
        - `>=`: lower bound.
        - `<=`: upper bound.
    - `Return`: (<:Real) the view's return.
    - `Relative_Type`: (String) specifies to what the view is relative:
        - Empty string: nothing.
        - `Asset`: other asset.
        - `Subset`: other class.
    - `Relative_Set`: (String) if `Relative_Type` is `Subset`, specifies the name of the set of asset classes.
    - `Relative_Position`: (String) name of the asset or asset class of the relative view.
- `asset_sets`: `Na×D` DataFrame where $(_ndef(:a2)) and `D` the number of columns.
    - `Asset`: list of assets, this is the only mandatory column.
    - Subsequent columns specify the asset class sets.
# Outputs
- `P`: `Nv×Na` matrix of views where `Nv` is the number of views and $(_ndef(:a2)).
- `Q`: `Nv×1` vector of views where `Nv` is the number of views.
# Examples
```julia
asset_sets = DataFrame(
        "Asset" => ["FB", "GOOGL", "NTFX", "BAC", "WFC", "TLT", "SHV"],        "Class 1" => ["Equity", "Equity", "Equity", "Equity", "Equity", "Fixed Income", "Fixed Income"],        "Class 2" => ["Technology", "Technology", "Technology", "Financial", "Financial", "Treasury", "Treasury"],    )
views = DataFrame(
        "Enabled" => [true, true, true, true, true],        "Type" => ["Asset", "Subset", "Subset", "Asset", "Subset"],        "Set" => ["", "Class 2", "Class 1", "", "Class 1"],        "Position" => ["WFC", "Financial", "Equity", "FB", "Fixed Income"],        "Sign" => ["<=", ">=", ">=", ">=", "<="],        "Return" => [0.3, 0.1, 0.05, 0.03, 0.017],        "Relative_Type" => ["Asset", "Subset", "Asset", "", ""],        "Relative_Set" => ["", "Class 1", "", "", ""],        "Relative_Position" => ["FB", "Fixed Income", "TLT", "", ""],    )
P, Q = asset_views(views, asset_sets)
```
"""
function asset_views(views::DataFrame, asset_sets::DataFrame)
    N = nrow(asset_sets)
    asset_list = asset_sets[!, "Asset"]

    P = Matrix(undef, 0, N)
    Q = Vector(undef, 0)

    for row in eachrow(views)
        valid = false

        if !row["Enabled"] || row["Return"] == ""
            continue
        end

        if row["Sign"] == ">="
            d = 1
        elseif row["Sign"] == "<="
            d = -1
        end

        if row["Type"] == "Asset"
            idx = findfirst(x -> x == row["Position"], asset_list)
            P1 = zeros(N)
            P1[idx] = 1
        elseif row["Type"] == "Subset"
            P1 = asset_sets[!, row["Set"]] .== row["Position"]
            P1 = P1 / sum(P1)
        end

        if row["Relative_Type"] == "Asset" && row["Relative_Position"] != ""
            idx2 = findfirst(x -> x == row["Relative_Position"], asset_list)
            P2 = zeros(N)
            P2[idx2] = 1
            valid = true
        elseif row["Relative_Type"] == "Subset" &&
               row["Relative_Set"] != "" &&
               row["Relative_Position"] != ""
            P2 = asset_sets[!, row["Relative_Set"]] .== row["Relative_Position"]
            P2 = P2 / sum(P2)
            valid = true
        elseif row["Relative_Type"] == "" &&
               row["Relative_Set"] == "" &&
               row["Relative_Position"] == ""
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

    P = convert.(typeof(promote(P...)[1]), P)
    Q = convert.(typeof(promote(Q...)[1]), Q)

    return P, Q
end

"""
```julia
factor_views(views::DataFrame, loadings::DataFrame)
```
Create the factor views matrix `P` and vector `Q`:
- ``\\mathbf{P} \\bm{w} \\geq \\bm{Q}``.
# Inputs
- `views`: `Nv×4` DataFrame, where `Nv` is the number of views. The required columns are:
    - `Enabled`: (Bool) indicates if the view is enabled.
    - `Factor`: (String) name of the view's factor.
    - `Sign`: (String) specifies whether the view is a lower or upper bound:
        - `>=`: lower bound.
        - `<=`: upper bound.
    - `Value`: (<:Real) the upper or lower bound of the factor's value.
    - `Relative_Factor`: (String) factor to which the view is relative.
- `loadings`: `Nl×Nf` loadings DataFrame, where `Nl` is the number of data points, and $(_ndef(:f2)).
# Outputs
- `P`: `Nv×Nf` matrix of views where `Nv` is the number of views and $(_ndef(:f2)).
- `Q`: `Nv×1` vector of views where `Nv` is the number of views.
# Examples
```julia
loadings = DataFrame(
        "const" => [0.0004, 0.0002, 0.0000, 0.0006, 0.0001, 0.0003, -0.0003],        "MTUM" => [0.1916, 1.0061, 0.8695, 1.9996, 0.0000, 0.0000, 0.0000],        "QUAL" => [0.0000, 2.0129, 1.4301, 0.0000, 0.0000, 0.0000, 0.0000],        "SIZE" => [0.0000, 0.0000, 0.0000, 0.4717, 0.0000, -0.1857, 0.0000],        "USMV" => [-0.7838, -1.6439, -1.0176, -1.4407, 0.0055, 0.5781, 0.0000],        "VLUE" => [1.4772, -0.7590, -0.4090, 0.0000, -0.0054, -0.4844, 0.9435],    )
views = DataFrame(
    "Enabled" => [true, true, true],    "Factor" => ["MTUM", "USMV", "VLUE"],    "Sign" => ["<=", "<=", ">="],    "Value" => [0.9, -1.2, 0.3],    "Relative_Factor" => ["USMV", "", ""],)
P, Q = factor_views(views, loadings)
```
"""
function factor_views(views::DataFrame, loadings::DataFrame)
    factor_list = names(loadings)
    if "const" in factor_list
        (factor_list = setdiff(factor_list, ("const",)))
    end
    if "ticker" in factor_list
        (factor_list = setdiff(factor_list, ("ticker",)))
    end

    N = length(factor_list)

    P = Matrix(undef, 0, N)
    Q = Vector(undef, 0)

    for row in eachrow(views)
        if !row["Enabled"]
            continue
        end

        if row["Sign"] == ">="
            d = 1
        elseif row["Sign"] == "<="
            d = -1
        end

        idx = findfirst(x -> x == row["Factor"], factor_list)
        P1 = zeros(N)
        P1[idx] = d

        if row["Relative_Factor"] != ""
            idx = findfirst(x -> x == row["Relative_Factor"], factor_list)
            P1[idx] = -d
        end

        P = vcat(P, transpose(P1))
        push!(Q, row["Value"] * d)
    end

    P = convert.(typeof(promote(P...)[1]), P)
    Q = convert.(typeof(promote(Q...)[1]), Q)

    return P, Q
end

"""
```julia
hrp_constraints(constraints::DataFrame, asset_sets::DataFrame)
```
Create the upper and lower bounds constraints for hierarchical risk parity portfolios.
# Inputs
- `constraints`: `Nc×4` Dataframe, where $(_ndef(:c1)). The required columns are:
    - `Enabled`: (Bool) indicates if the constraint is enabled.
    - `Type`: (String) specifies the object(s) to which a constraint applies:
        - `Asset`: specific asset.
        - `All Assets`: all assets.
        - `Each Asset in Subset`: specific assets in a class.
    - `Position`: (String) name of the asset or asset class to which the constraint applies.
    - `Sign`: (String) specifies whether the constraint is a lower or upper bound:
        - `>=`: lower bound.
        - `<=`: upper bound.
    - `Weight`: (<:Real) value of the constraint.
- `asset_sets`: `Na×D` DataFrame where $(_ndef(:a2)) and `D` the number of columns.
    - `Asset`: list of assets, this is the only mandatory column.
    - Subsequent columns specify the asset class sets.
# Outputs
- `w_min`: `Na×1` vector of the lower bounds for asset weights.
- `w_max`: `Na×1` vector of the upper bounds for asset weights.
# Examples
```julia
asset_sets = DataFrame(
        "Asset" => ["FB", "GOOGL", "NTFX", "BAC", "WFC", "TLT", "SHV"],        "Class 1" => ["Equity", "Equity", "Equity", "Equity", "Equity", "Fixed Income", "Fixed Income"],        "Class 2" => ["Technology", "Technology", "Technology", "Financial", "Financial", "Treasury", "Treasury"],    )
constraints = DataFrame(
    "Enabled" => [true, true, true, true, true, true],    "Type" => ["Asset", "Asset", "All Assets", "All Assets", "Each Asset in Subset", "Each Asset in Subset"],    "Set" => ["", "", "", "", "Class 1", "Class 2"],    "Position" => ["BAC", "FB", "", "", "Fixed Income", "Financial"],    "Sign" => [">=", "<=", "<=", ">=", "<=", "<="],    "Weight" => [0.02, 0.085, 0.09, 0.01, 0.07, 0.06],)
w_min, w_max = hrp_constraints(constraints, asset_sets)
```
"""
function hrp_constraints(constraints::DataFrame, asset_sets::DataFrame)
    N = nrow(asset_sets)
    w = Matrix(undef, N, 2)
    w .= 0
    w[:, 2] .= 1
    for row in eachrow(constraints)
        if !row["Enabled"]
            continue
        end

        if row["Sign"] == ">="
            i = 1
            op = <=
        elseif row["Sign"] == "<="
            i = 2
            op = >=
        end

        if row["Type"] == "Asset"
            idx = findfirst(x -> x == row["Position"], asset_sets[!, "Asset"])
            if op(w[idx, i], row["Weight"])
                (w[idx, i] = row["Weight"])
            end
        elseif row["Type"] == "All Assets"
            if !isempty(w[op.(w[:, i], row["Weight"]), i])
                (w[op.(w[:, i], row["Weight"]), i] .= row["Weight"])
            end
        elseif row["Type"] == "Each Asset in Subset"
            assets = asset_sets[asset_sets[!, row["Set"]] .== row["Position"], "Asset"]
            idx = [findfirst(x -> x == asset, asset_sets[!, "Asset"]) for asset in assets]

            for ind in idx
                if !isnothing(ind) && op(w[ind, i], row["Weight"])
                    w[ind, i] = row["Weight"]
                end
            end
        end
    end

    w = convert.(typeof(promote(w...)[1]), w)

    return w[:, 1], w[:, 2]
end

"""
```julia
RPConstraintTypes = (:Asset, :Subset)
```

Types of risk parity constraints for building the set of linear constraints via [`rp_constraints`](@ref).

  - `:Asset`: equal risk contribution per asset.
  - `:Subset`: equal risk contribution per class.
"""
const RPConstraintTypes = (:Asset, :Subset)

"""
```julia
rp_constraints(
    asset_sets::DataFrame,    type::Symbol = :Asset,    class_col::Union{String, Symbol, Nothing} = nothing,)
```
Constructs risk contribution constraint vector for the risk parity optimisation (`:RP` and `:RRP` types of [`PortTypes`](@ref)).
# Inputs
- `asset_sets`: `Na×D` DataFrame where $(_ndef(:a2)) and `D` the number of columns.
    - `Asset`: list of assets, this is the only mandatory column.
    - Subsequent columns specify the asset class sets. They are only used if `type == :Subset`.
- `type`: what the risk parity is applied relative to, must be one of [`RPConstraintTypes`](@ref).
- `class_col`: index of set of classes from `asset_sets` to use in when `type == :Subset`.
# Outputs
- `rw`: risk contribution constraint vector.
# Examples
```julia
asset_sets = DataFrame(
        "Asset" => ["FB", "GOOGL", "NTFX", "BAC", "WFC", "TLT", "SHV"],        "Class 1" => ["Equity", "Equity", "Equity", "Equity", "Equity", "Fixed Income", "Fixed Income"],        "Class 2" => ["Technology", "Technology", "Technology", "Financial", "Financial", "Treasury", "Treasury"],    )

rw_a = rp_constraints(asset_sets, :Asset)
rw_c = rp_constraints(asset_sets, :Subset, "Class 2")
```
"""
function rp_constraints(asset_sets::DataFrame, type::Symbol = :Asset,
                        class_col::Union{String,Symbol,Int,Nothing} = nothing)
    @smart_assert(type in RPConstraintTypes)
    N = nrow(asset_sets)

    rw = if type == :Asset
        fill(1 / N, N)
    else
        classes = names(asset_sets)
        A = DataFrame(; a = asset_sets[!, class_col])
        if isa(class_col, String) || isa(class_col, Symbol)
            DataFrames.rename!(A, [class_col])
        elseif isa(class_col, Int)
            DataFrames.rename!(A, [classes[class_col]])
        end

        col = names(A)[1]
        A[!, :weight] .= 1
        B = combine(groupby(A, col), nrow => :count)
        A = leftjoin(A, B; on = col)
        A[!, :weight] ./= A[!, :count]
        A[!, :weight] ./= sum(A[!, :weight])
        A[!, :weight]
    end

    return rw
end

export asset_constraints, factor_constraints, asset_views, factor_views, hrp_constraints,
       rp_constraints
