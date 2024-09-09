function get_rm_string(rm::Union{AbstractVector, <:RiskMeasure})
    rmstr = ""
    if !isa(rm, AbstractVector)
        rmstr *= String(rm)
    else
        rm = reduce(vcat, rm)
        for (i, r) ∈ enumerate(rm)
            rmstr *= String(r)
            if i != length(rm)
                rmstr *= '_'
            end
        end
    end
    return Symbol(rmstr)
end

function get_first_rm(rm::Union{AbstractVector, <:RiskMeasure})
    return if !isa(rm, AbstractVector)
        rm
    else
        reduce(vcat, rm)[1]
    end
end

function _set_rm_solvers(rm::RMSolvers, solvers)
    flag = false
    if isnothing(rm.solvers) || isempty(rm.solvers)
        rm.solvers = solvers
        flag = true
    end
    return flag
end
function _set_rm_solvers(args...)
    return false
end
function _set_rm_sigma(rm::RMSigma, sigma)
    flag = false
    if isnothing(rm.sigma) || isempty(rm.sigma)
        rm.sigma = sigma
        flag = true
    end
    return flag
end
function _set_rm_sigma(args...)
    return false
end
function set_rm_properties(rm, solvers, sigma)
    solver_flag = _set_rm_solvers(rm, solvers)
    sigma_flag = _set_rm_sigma(rm, sigma)
    return solver_flag, sigma_flag
end
function _unset_rm_solvers(rm::RMSolvers, flag)
    if flag
        rm.solvers = nothing
    end
end
function _unset_rm_solvers(args...)
    return nothing
end
function _unset_rm_sigma(rm::RMSigma, flag)
    if flag
        rm.sigma = nothing
    end
end
function _unset_rm_sigma(args...)
    return nothing
end
function unset_set_rm_properties(rm, solver_flag, sigma_flag)
    _unset_rm_solvers(rm, solver_flag)
    _unset_rm_sigma(rm, sigma_flag)
    return nothing
end

export get_rm_string, get_first_rm, set_rm_properties, unset_set_rm_properties
