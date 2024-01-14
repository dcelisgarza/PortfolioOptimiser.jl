function optimise!(portfolio::NearOptCentering, optimisation::Function = max_sharpe!;
                   target = nothing, initial_guess = nothing,
                   n = Int(ceil(length(portfolio.opt_port.tickers) /
                                log(length(portfolio.opt_port.tickers)))),
                   optimiser = Ipopt.Optimizer, silent = true, optimiser_attributes = (),
                   nloptimiser = Ipopt.Optimizer, nlsilent = true,
                   nloptimiser_attributes = (),)
    model = portfolio.model

    if termination_status(model) != OPTIMIZE_NOT_CALLED
        throw("Nonlinear models must be reconstructed before being used again.")
    end

    opt_port = portfolio.opt_port

    c1, c2, w_min, w_max = calc_c1_c2(opt_port, n, optimiser, silent, optimiser_attributes)

    if isnothing(target)
        optimisation(opt_port; optimiser, silent, optimiser_attributes)
    else
        optimisation(opt_port, target; optimiser, silent, optimiser_attributes)
    end

    e1, e2 = calc_e1_e2(opt_port, c1, c2)

    portfolio.c12 .= (c1, c2)
    portfolio.e12 .= (e1, e2)

    w = model[:w]
    ret = model[:ret]
    risk = model[:risk]
    num_tickers = length(w)

    if !isnothing(initial_guess)
        @assert num_tickers == length(initial_guess)
        set_start_value.(w, initial_guess)
    elseif optimisation == max_sharpe!
        set_start_value.(w, w_max)
    else
        set_start_value.(w, opt_port.weights)
    end

    # Add stuff to refresh the model so we can continually call it.

    @constraint(model, ret_geq_e1, ret >= e1)
    @constraint(model, risk_get_e2, risk <= e2)

    @NLexpression(model, lret, -log(ret - e1))
    @NLexpression(model, lrisk, -log(e2 - risk))
    @NLexpression(model, slw, -sum(log(1 - w[i]) + log(w[i]) for i ∈ 1:num_tickers))

    @NLobjective(model, Min, lret + lrisk + slw)

    _setup_and_optimise(model, nloptimiser, nlsilent, nloptimiser_attributes)

    portfolio.weights .= value.(w)
    return nothing
end

function calc_c1_c2(portfolio::AbstractEfficient,
                    n = length(portfolio.tickers) / log(length(portfolio.tickers)),
                    optimiser = Ipopt.Optimizer, silent = true, optimiser_attributes = ())
    max_port = max_return(portfolio; optimiser, silent, optimiser_attributes)
    w_max = value.(max_port[:w])

    min_port = min_risk(portfolio; optimiser, silent, optimiser_attributes)
    w_min = value.(min_port[:w])

    max_ret = value(max_port[:ret])
    min_ret = value(min_port[:ret])

    c1 = (max_ret - min_ret) / n

    max_risk_val = value(max_port[:risk])
    min_risk_val = value(min_port[:risk])

    c2 = (max_risk_val - min_risk_val) / n

    return c1, c2, w_min, w_max
end

function calc_e1_e2(portfolio::AbstractEfficient, c1, c2)
    model = portfolio.model

    ret = value(model[:ret])
    risk = value(model[:risk])

    if haskey(model, :k)
        ret /= value(model[:k])
        risk /= value(model[:k])
    end

    e1 = ret - c1
    e2 = risk + c2

    return e1, e2
end
