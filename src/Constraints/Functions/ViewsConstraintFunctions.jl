"""
```julia
asset_views(views::DataFrame, asset_sets::DataFrame)
```

Create the asset views matrix `P` and vector `Q`:

  - ``\\mathbf{P} \\bm{w} \\geq \\bm{Q}``.

# Inputs

  - `views`: `Nv×9` DataFrame, where `Nv` is the number of views. The required columns are:

      + `Enabled`: (Bool) indicates if the view is enabled.

      + `Type`: (String) specifies the object(s) to which a view applies:

          * `Asset`: specific asset.
          * `Subset`: whole class.
      + `Set`: (String) if `Type` is `Subset`, specifies the asset class set.
      + `Position`: (String) name of the asset or asset class to which the view applies.
      + `Sign`: (String) specifies whether the view is a lower or upper bound:

          * `>=`: lower bound.
          * `<=`: upper bound.
      + `Return`: (<:Real) the view's return.
      + `Relative_Type`: (String) specifies to what the view is relative:

          * Empty string: nothing.
          * `Asset`: other asset.
          * `Subset`: other class.
      + `Relative_Set`: (String) if `Relative_Type` is `Subset`, specifies the name of the set of asset classes.
      + `Relative_Position`: (String) name of the asset or asset class of the relative view.

  - `asset_sets`: `Na×D` DataFrame where  and `D` the number of columns.

      + `Asset`: list of assets, this is the only mandatory column.
      + Subsequent columns specify the asset class sets.

# Outputs

  - `P`: `Nv×Na` matrix of views where `Nv` is the number of views and .
  - `Q`: `Nv×1` vector of views where `Nv` is the number of views.

# Examples

```julia
asset_sets = DataFrame("Asset" => ["FB", "GOOGL", "NTFX", "BAC", "WFC", "TLT", "SHV"],
                       "Class 1" => ["Equity", "Equity", "Equity", "Equity", "Equity",
                                     "Fixed Income", "Fixed Income"],
                       "Class 2" => ["Technology", "Technology", "Technology", "Financial",
                                     "Financial", "Treasury", "Treasury"])
views = DataFrame("Enabled" => [true, true, true, true, true],
                  "Type" => ["Asset", "Subset", "Subset", "Asset", "Subset"],
                  "Set" => ["", "Class 2", "Class 1", "", "Class 1"],
                  "Position" => ["WFC", "Financial", "Equity", "FB", "Fixed Income"],
                  "Sign" => ["<=", ">=", ">=", ">=", "<="],
                  "Return" => [0.3, 0.1, 0.05, 0.03, 0.017],
                  "Relative_Type" => ["Asset", "Subset", "Asset", "", ""],
                  "Relative_Set" => ["", "Class 1", "", "", ""],
                  "Relative_Position" => ["FB", "Fixed Income", "TLT", "", ""])
P, Q = asset_views(views, asset_sets)
```
"""
function asset_views(views::DataFrame, asset_sets::DataFrame)
    N = nrow(asset_sets)
    asset_list = asset_sets[!, "Asset"]

    P = Matrix(undef, 0, N)
    Q = Vector(undef, 0)

    for row ∈ eachrow(views)
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
            P1 .= (P1 .- P2) * d
        elseif row["Relative_Type"] == "Subset" &&
               row["Relative_Set"] != "" &&
               row["Relative_Position"] != ""
            P2 = asset_sets[!, row["Relative_Set"]] .== row["Relative_Position"]
            P2 = P2 / sum(P2)
            P1 .= (P1 .- P2) * d
        elseif row["Relative_Type"] == "" &&
               row["Relative_Set"] == "" &&
               row["Relative_Position"] == ""
            P1 .*= d
        end

        P = vcat(P, transpose(P1))
        push!(Q, row["Return"] * d)
    end

    for i ∈ eachindex(Q)
        if Q[i] < 0
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

      + `Enabled`: (Bool) indicates if the view is enabled.

      + `Factor`: (String) name of the view's factor.
      + `Sign`: (String) specifies whether the view is a lower or upper bound:

          * `>=`: lower bound.
          * `<=`: upper bound.
      + `Value`: (<:Real) the upper or lower bound of the factor's value.
      + `Relative_Factor`: (String) factor to which the view is relative.

  - `loadings`: `Nl×Nf` loadings DataFrame, where `Nl` is the number of data points, and .

# Outputs

  - `P`: `Nv×Nf` matrix of views where `Nv` is the number of views and .
  - `Q`: `Nv×1` vector of views where `Nv` is the number of views.

# Examples

```julia
loadings = DataFrame("const" => [0.0004, 0.0002, 0.0000, 0.0006, 0.0001, 0.0003, -0.0003],
                     "MTUM" => [0.1916, 1.0061, 0.8695, 1.9996, 0.0000, 0.0000, 0.0000],
                     "QUAL" => [0.0000, 2.0129, 1.4301, 0.0000, 0.0000, 0.0000, 0.0000],
                     "SIZE" => [0.0000, 0.0000, 0.0000, 0.4717, 0.0000, -0.1857, 0.0000],
                     "USMV" => [-0.7838, -1.6439, -1.0176, -1.4407, 0.0055, 0.5781, 0.0000],
                     "VLUE" => [1.4772, -0.7590, -0.4090, 0.0000, -0.0054, -0.4844, 0.9435])
views = DataFrame("Enabled" => [true, true, true], "Factor" => ["MTUM", "USMV", "VLUE"],
                  "Sign" => ["<=", "<=", ">="], "Value" => [0.9, -1.2, 0.3],
                  "Relative_Factor" => ["USMV", "", ""])
P, Q = factor_views(views, loadings)
```
"""
function factor_views(views::DataFrame, loadings::DataFrame)
    factor_list = names(loadings)
    factor_list = setdiff(factor_list, ("const", "tickers"))

    N = length(factor_list)

    P = Matrix(undef, 0, N)
    Q = Vector(undef, 0)

    for row ∈ eachrow(views)
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

export asset_views, factor_views
