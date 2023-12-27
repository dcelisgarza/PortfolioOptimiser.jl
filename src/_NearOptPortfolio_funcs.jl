function _optimize_near_opt_portfolio(portfolio, type, obj)
    solvers = portfolio.solvers
    model = portfolio.model

    optimum_portfolio = portfolio.optimum_portfolio

    N = size(optimum_portfolio.returns, 2)
    rtype = eltype(optimum_portfolio.returns)
    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) in solvers
        key = Symbol(String(key) * "_" * String(type))

        haskey(val, :solver) && set_optimizer(model, val[:solver])

        if haskey(val, :params)
            for (attribute, value) in val[:params]
                set_attribute(model, attribute, value)
            end
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, key => Dict(:jump_error => jump_error))
            continue
        end

        term_status = termination_status(model)

        all_finite_weights = all(isfinite.(value.(model[:w])))
        all_non_zero_weights = all(abs.(0.0 .- value.(model[:w])) .> eps())

        term_status ∈ ValidTermination &&
            all_finite_weights &&
            all_non_zero_weights &&
            break

        weights = Vector{rtype}(undef, N)
        if type == :Trad || type == :WC
            if obj == :Sharpe
                val_k = value(model[:k])
                val_k = val_k > 0 ? val_k : 1
                weights .= value.(model[:w]) / val_k
            else
                weights .= value.(model[:w])
            end

            short = optimum_portfolio.short
            sum_short_long = optimum_portfolio.sum_short_long
            if short == false
                sum_w = sum(abs.(weights))
                sum_w = sum_w > eps() ? sum_w : 1
                weights .= abs.(weights) / sum_w * sum_short_long
            end
        elseif type == :RP || type == :RRP
            weights .= value.(model[:w])
            sum_w = sum(abs.(weights))
            sum_w = sum_w > eps() ? sum_w : 1
            weights .= abs.(weights) / sum_w
        end

        push!(
            solvers_tried,
            key => Dict(
                :objective_val => objective_value(model),
                :term_status => term_status,
                :params => haskey(val, :params) ? val[:params] : missing,
                :finite_weights => all_finite_weights,
                :nonzero_weights => all_non_zero_weights,
                :portfolio =>
                    DataFrame(tickers = optimum_portfolio.assets, weights = weights),
            ),
        )
    end

    isempty(solvers_tried) &&
        term_status ∉ ValidTermination &&
        push!(solvers_tried, :error => term_status)

    return term_status, solvers_tried
end

function opt_port!(
    portfolio::NearOptPortfolio;
    M = sqrt(size(portfolio.optimum_portfolio.returns, 2)),
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    l::Real = 2.0,
    obj::Symbol = :Sharpe,
    rf::Real = 0.0,
    rm::Symbol = :SD,
    rrp_penalty::Real = 1.0,
    rrp_ver::Symbol = :None,
    save_opt_params::Bool = true,
    string_names::Bool = false,
    type::Symbol = :Trad,
    u_cov::Symbol = :Box,
    u_mu::Symbol = :Box,
)
    optimum_portfolio = portfolio.optimum_portfolio

    mu, sigma, returns = _setup_model_class(optimum_portfolio, class, hist)

    fl = frontier_limits!(
        optimum_portfolio;
        class = class,
        hist = hist,
        kelly = kelly,
        rf = rf,
        rm = rm,
    )

    w1 = fl.w_min
    w2 = fl.w_max

    ret1 = dot(mu, w1)
    ret2 = dot(mu, w2)

    alpha_i = optimum_portfolio.alpha_i
    alpha = optimum_portfolio.alpha
    a_sim = optimum_portfolio.a_sim
    beta_i = optimum_portfolio.beta_i
    beta = optimum_portfolio.beta
    b_sim = optimum_portfolio.b_sim
    kappa = optimum_portfolio.kappa
    owa_w = optimum_portfolio.owa_w
    solvers = optimum_portfolio.solvers

    risk1, risk2 = _ul_risk(
        rm,
        returns,
        w1,
        w2,
        sigma,
        rf,
        solvers,
        alpha,
        kappa,
        alpha_i,
        beta,
        a_sim,
        beta_i,
        b_sim,
        owa_w,
        0,
    )

    c1 = (ret2 - ret1) / M
    c2 = (risk2 - risk1) / M

    w_opt = opt_port!(
        optimum_portfolio;
        class = class,
        hist = hist,
        kelly = kelly,
        l = l,
        obj = obj,
        rf = rf,
        rm = rm,
        rrp_penalty = rrp_penalty,
        rrp_ver = rrp_ver,
        save_opt_params = save_opt_params,
        string_names = string_names,
        type = type,
        u_cov = u_cov,
        u_mu = u_mu,
    )

    w3 = w_opt.weights

    ret3 = dot(mu, w3)
    risk3 = calc_risk(
        w3,
        returns;
        rm = rm,
        rf = rf,
        sigma = sigma,
        alpha_i = alpha_i,
        alpha = alpha,
        a_sim = a_sim,
        beta_i = beta_i,
        beta = beta,
        b_sim = b_sim,
        kappa = kappa,
        owa_w = owa_w,
        solvers = solvers,
    )

    e1 = ret3 - c1
    e2 = risk3 + c2

    portfolio.c1 = c1
    portfolio.c2 = c2
    portfolio.e1 = e1
    portfolio.e2 = e2

    model = portfolio.model = copy(optimum_portfolio.model)
    N = length(w3)

    @constraint(model, model[:ret] >= e1)
    @constraint(model, model[:risk] <= e2)

    # @NLexpression(model, lret, -log(ret - e1))
    # @NLexpression(model, lrisk, -log(e2 - risk))
    # @NLexpression(model, slw, -sum(log(1 - w[i]) + log(w[i]) for i in 1:num_tickers))

    @variable(model, log_ret)
    @constraint(model, [-log_ret, 1, model[:ret] - e1] in MOI.ExponentialCone())

    @variable(model, log_risk)
    @constraint(model, [-log_risk, 1, e2 - model[:risk]] in MOI.ExponentialCone())

    @variable(model, log_w[1:N])
    @constraint(model, [i = 1:N], [log_w[i], 1, model[:w][i]] in MOI.ExponentialCone())
    @variable(model, log_omw[1:N])
    @constraint(
        model,
        [i = 1:N],
        [log_omw[i], 1, 1 - model[:w][i]] in MOI.ExponentialCone()
    )

    @expression(model, sum_log_w, -sum(log_w + log_omw))

    @objective(model, Min, log_ret + log_risk + sum_log_w)

    term_status, solvers_tried = _optimize_near_opt_portfolio(portfolio, type, obj)
    retval = _handle_errors_and_finalise(
        portfolio,
        term_status,
        optimum_portfolio.returns,
        N,
        solvers_tried,
        type,
        rm,
        obj,
    )
end
