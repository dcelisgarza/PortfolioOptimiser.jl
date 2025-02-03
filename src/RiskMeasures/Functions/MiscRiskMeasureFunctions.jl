# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

"""
```
get_rm_symbol(rm::Union{AbstractVector, <:AbstractRiskMeasure})
```

Get a symbol for the risk measure(s). If multiple measures are given, they are concatenated by underscores.

# Inputs

  - `rm`: risk measure or vector of risk measures [`AbstractRiskMeasure`](@ref).
"""
function get_rm_symbol(rm::Union{AbstractVector, <:AbstractRiskMeasure})
    rmsym = ""
    if !isa(rm, AbstractVector)
        rmsym *= String(rm)
    else
        rm = reduce(vcat, rm)
        for (i, r) ∈ pairs(rm)
            rmsym *= String(r)
            if i != length(rm)
                rmsym *= '_'
            end
        end
    end
    return Symbol(rmsym)
end

"""
```
get_first_rm(rm::Union{AbstractVector, <:AbstractRiskMeasure})
```

Get the first risk measure, used in [`efficient_frontier!`](@ref).

# Inputs

  - `rm`: risk measure or vector of risk measures [`AbstractRiskMeasure`](@ref).
"""
function get_first_rm(rm::Union{AbstractVector, <:AbstractRiskMeasure})
    return if !isa(rm, AbstractVector)
        rm
    else
        reduce(vcat, rm)[1]
    end
end

"""
```
set_rm_solvers!(rm::RMSolvers, solvers)
```
"""
function set_rm_solvers!(rm::RMSolvers, solvers)
    flag = false
    if isnothing(rm.solvers) || isempty(rm.solvers)
        rm.solvers = solvers
        flag = true
    end
    return flag
end
function set_rm_solvers!(args...)
    return false
end
function set_rm_sigma!(rm::RMSigma, sigma)
    flag = false
    if isnothing(rm.sigma) || isempty(rm.sigma)
        rm.sigma = sigma
        flag = true
    end
    return flag
end
function set_rm_sigma!(args...)
    return false
end
function set_rm_skew!(rm::Skew, V)
    flag = false
    if isnothing(rm.V) || isempty(rm.V)
        rm.V = V
        flag = true
    end
    return flag
end
function set_rm_sskew!(rm::SSkew, V)
    flag = false
    if isnothing(rm.V) || isempty(rm.V)
        rm.V = V
        flag = true
    end
    return flag
end
function set_rm_skew!(args...)
    return false
end
function set_rm_sskew!(args...)
    return false
end
"""
```
set_rm_properties!(rm::AbstractRiskMeasure, solvers::AbstractDict,
                   sigma::Union{Nothing, <:AbstractMatrix{<:Real}})
```

Set properties for risk measures that use solvers or covariance matrices.

# Inputs

  - `rm`: risk measure [`AbstractRiskMeasure`](@ref).
  - `solvers`: solvers.
  - `sigma`: covariance matrix.
"""
function set_rm_properties!(rm::AbstractRiskMeasure,
                            solvers::Union{PortOptSolver, <:AbstractVector{PortOptSolver}},
                            sigma::Union{Nothing, <:AbstractMatrix{<:Real}},
                            V::Union{Nothing, <:AbstractMatrix{<:Real}},
                            SV::Union{Nothing, <:AbstractMatrix{<:Real}})
    solver_flag = set_rm_solvers!(rm, solvers)
    sigma_flag = set_rm_sigma!(rm, sigma)
    skew_flag = set_rm_skew!(rm, V)
    sskew_flag = set_rm_sskew!(rm, SV)
    return solver_flag, sigma_flag, skew_flag, sskew_flag
end
"""
```
unset_rm_solvers!(rm::RMSolvers, flag)
```
"""
function unset_rm_solvers!(rm::RMSolvers, flag)
    if flag
        rm.solvers = nothing
    end
end
function unset_rm_solvers!(args...)
    return nothing
end
function unset_rm_sigma!(rm::RMSigma, flag)
    if flag
        rm.sigma = nothing
    end
end
function unset_rm_sigma!(args...)
    return nothing
end
function unset_rm_skew!(rm::Skew, flag)
    if flag
        rm.V = nothing
    end
end
function unset_rm_sskew!(rm::SSkew, flag)
    if flag
        rm.V = nothing
    end
end
function unset_rm_skew!(args...)
    return nothing
end
function unset_rm_sskew!(args...)
    return nothing
end
"""
```
unset_set_rm_properties!(rm::AbstractRiskMeasure, solver_flag::Bool, sigma_flag::Bool)
```

Unset properties for risk measures that use solvers or covariance matrices.

# Inputs

  - `rm`: risk measure [`AbstractRiskMeasure`](@ref).
  - `solvers`: solvers.
  - `sigma`: covariance matrix.
"""
function unset_set_rm_properties!(rm::AbstractRiskMeasure, solver_flag::Bool,
                                  sigma_flag::Bool, skew_flag::Bool, sskew_flag::Bool)
    unset_rm_solvers!(rm, solver_flag)
    unset_rm_sigma!(rm, sigma_flag)
    unset_rm_skew!(rm, skew_flag)
    unset_rm_sskew!(rm, sskew_flag)
    return nothing
end
function number_effective_assets(w::AbstractVector)
    return inv(dot(w, w))
end
export number_effective_assets
