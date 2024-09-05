function get_rm_string(rm::Union{AbstractVector, <:TradRiskMeasure})
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

function get_first_rm(rm::Union{AbstractVector, <:TradRiskMeasure})
    return if !isa(rm, AbstractVector)
        rm
    else
        reduce(vcat, rm)[1]
    end
end

function set_rm_properties(rm, solvers, sigma)
    solver_flag = false
    sigma_flag = false
    if hasproperty(rm, :solvers) && (isnothing(rm.solvers) || isempty(rm.solvers))
        rm.solvers = solvers
        solver_flag = true
    end
    if hasproperty(rm, :sigma) && (isnothing(rm.sigma) || isempty(rm.sigma))
        rm.sigma = sigma
        sigma_flag = true
    end
    return solver_flag, sigma_flag
end

function unset_set_rm_properties(rm, solver_flag, sigma_flag)
    if solver_flag
        rm.solvers = nothing
    end
    if sigma_flag
        rm.sigma = nothing
    end
    return nothing
end

export get_rm_string, get_first_rm, set_rm_properties, unset_set_rm_properties
