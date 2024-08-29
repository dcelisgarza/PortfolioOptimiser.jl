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
    return rmi = if !isa(rm, AbstractVector)
        rm
    else
        reduce(vcat, rm)[1]
    end
end

export get_rm_string, get_first_rm
