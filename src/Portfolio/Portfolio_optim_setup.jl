
function _setup_sharpe_k(model, obj)
    obj == :Sharpe && @variable(model, k >= 0)
    return nothing
end

function _setup_risk_budget(portfolio, N)
    model = portfolio.model
    if isempty(portfolio.risk_budget) || isa(portfolio.risk_budget, Real)
        portfolio.risk_budget = fill(1 / N, N)
    else
        portfolio.risk_budget ./= sum(portfolio.risk_budget)
    end
    @variable(model, k >= 0)
    return nothing
end

function _setup_ret(kelly, model, T, returns, mu, mu_l)
    if kelly == :Exact
        @variable(model, texact_kelly[1:T])
        @expression(model, _ret, sum(texact_kelly) / T)
        @expression(model, kret, 1 .+ returns * model[:w])
        @constraint(
            model,
            [i = 1:T],
            [texact_kelly[i], 1, kret[i]] in MOI.ExponentialCone()
        )
    elseif kelly == :Approx && (!isempty(mu) || !isnothing(mu))
        @expression(model, _ret, dot(mu, model[:w]) - 0.5 * model[:dev_risk])
    elseif !isempty(mu) || !isnothing(mu)
        @expression(model, _ret, dot(mu, model[:w]))
    end

    !isinf(mu_l) && @constraint(model, _ret >= mu_l)

    return nothing
end

function _setup_sharpe_ret(kelly, model, T, rf, returns, mu, mu_l)
    if kelly == :Exact
        @variable(model, texact_kelly[1:T])
        @expression(model, _ret, sum(texact_kelly) / T - rf * model[:k])
        @expression(model, kret, model[:k] .+ returns * model[:w])
        @constraint(
            model,
            [i = 1:T],
            [texact_kelly[i], model[:k], kret[i]] in MOI.ExponentialCone()
        )
        @constraint(model, model[:risk] <= 1)
    elseif kelly == :Approx && (!isempty(mu) || !isnothing(mu))
        @variable(model, tapprox_kelly)
        @constraint(
            model,
            [
                model[:k] + tapprox_kelly
                2 * model[:dev]
                model[:k] - tapprox_kelly
            ] in SecondOrderCone()
        )
        @expression(model, _ret, dot(mu, model[:w]) - 0.5 * tapprox_kelly)
        @constraint(model, model[:risk] <= 1)
    elseif !isempty(mu) || !isnothing(mu)
        @expression(model, _ret, dot(mu, model[:w]))
        @constraint(model, _ret - rf * model[:k] == 1)
    end

    !isinf(mu_l) && @constraint(model, _ret >= mu_l * model[:k])

    return nothing
end

function _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
    model = portfolio.model
    mu_l = portfolio.mu_l

    if class == :Classic && (kelly == :Exact || kelly == :Approx)
        obj == :Sharpe ? _setup_sharpe_ret(kelly, model, T, rf, returns, mu, mu_l) :
        _setup_ret(kelly, model, T, returns, mu, mu_l)
    else
        obj == :Sharpe ? _setup_sharpe_ret(kelly, model, T, rf, returns, mu, mu_l) :
        _setup_ret(kelly, model, T, returns, mu, mu_l)
    end

    haskey(model, :_ret) && @expression(model, ret, model[:_ret])

    return nothing
end

function _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    model = portfolio.model
    mu_l = portfolio.mu_l
    _setup_ret(kelly, model, T, returns, mu, mu_l)
    haskey(model, :_ret) && @expression(model, ret, model[:_ret])

    @objective(model, Min, model[:risk])
    return nothing
end

function _setup_weights(portfolio, obj, N)
    max_number_assets = portfolio.max_number_assets
    factor = portfolio.max_number_assets_factor
    short = portfolio.short
    short_u = portfolio.short_u
    long_u = portfolio.long_u
    sum_short_long = portfolio.sum_short_long

    model = portfolio.model

    # Boolean variables max number of assets.
    if max_number_assets > 0
        if obj == :Sharpe
            @variable(model, tass_bin[1:N], binary = true)
            @variable(model, tass_bin_sharpe[1:N] >= 0)
        else
            @variable(model, tass_bin[1:N], binary = true)
        end
    end

    # Weight constraints.
    if obj == :Sharpe
        @constraint(model, sum(model[:w]) == sum_short_long * model[:k])

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
            @constraint(model, tass_bin_sharpe .<= model[:k])
            @constraint(model, tass_bin_sharpe .<= factor * tass_bin)
            @constraint(model, tass_bin_sharpe .>= model[:k] .- factor * (1 .- tass_bin))
            @constraint(model, model[:w] .<= long_u * tass_bin_sharpe)
        end

        if short == false
            @constraint(model, model[:w] .<= long_u * model[:k])
            @constraint(model, model[:w] .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum(tw_ulong) <= long_u * model[:k])
            @constraint(model, sum(tw_ushort) <= short_u * model[:k])

            @constraint(model, model[:w] .<= tw_ulong)
            @constraint(model, model[:w] .>= -tw_ushort)

            # Maximum number of assets constraints.
            max_number_assets > 0 &&
                @constraint(model, model[:w] .>= -short_u * tass_bin_sharpe)
        end
    else
        @constraint(model, sum(model[:w]) == sum_short_long)

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
            @constraint(model, model[:w] .<= long_u * tass_bin)
        end

        if short == false
            @constraint(model, model[:w] .<= long_u)
            @constraint(model, model[:w] .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum(tw_ulong) <= long_u)
            @constraint(model, sum(tw_ushort) <= short_u)

            @constraint(model, model[:w] .<= tw_ulong)
            @constraint(model, model[:w] .>= -tw_ushort)

            # Maximum number of assets constraints.
            max_number_assets > 0 && @constraint(model, model[:w] .>= -short_u * tass_bin)
        end
    end

    return nothing
end

function _setup_linear_constraints(portfolio, obj, type)
    A = portfolio.a_mtx_ineq
    B = portfolio.b_vec_ineq

    (isempty(A) || isempty(B)) && (return nothing)

    model = portfolio.model

    # Linear weight constraints.
    obj == :Sharpe || type == :RP ?
    @constraint(model, A * model[:w] .- B * model[:k] .>= 0) :
    @constraint(model, A * model[:w] .- B .>= 0)

    return nothing
end

function _setup_min_number_effective_assets(portfolio, obj)
    mnea = portfolio.min_number_effective_assets

    (mnea < 1) && (return nothing)

    model = portfolio.model

    @variable(model, tmnea >= 0)
    @constraint(model, [tmnea; model[:w]] in SecondOrderCone())

    obj == :Sharpe ? @constraint(model, tmnea * sqrt(mnea) <= model[:k]) :
    @constraint(model, tmnea * sqrt(mnea) <= 1)

    return nothing
end

function _setup_tracking_err(portfolio, returns, obj, T)
    tracking_err = portfolio.tracking_err

    isinf(tracking_err) && (return nothing)

    kind_tracking_err = portfolio.kind_tracking_err
    tracking_err_weights = portfolio.tracking_err_weights
    tracking_err_returns = portfolio.tracking_err_returns

    tracking_err_flag = false

    if kind_tracking_err == :Weights && !isempty(tracking_err_weights)
        benchmark = returns * tracking_err_weights
        tracking_err_flag = true
    elseif kind_tracking_err == :Returns && !isempty(tracking_err_returns)
        benchmark = tracking_err_returns
        tracking_err_flag = true
    end

    !(tracking_err_flag) && (return nothing)

    model = portfolio.model

    @variable(model, t_track_err >= 0)
    if obj == :Sharpe
        @expression(model, track_err, returns * model[:w] .- benchmark * model[:k])
        @constraint(model, [t_track_err; track_err] in SecondOrderCone())
        @constraint(model, t_track_err <= tracking_err * model[:k] * sqrt(T - 1))
    else
        @expression(model, track_err, returns * model[:w] .- benchmark)
        @constraint(model, [t_track_err; track_err] in SecondOrderCone())
        @constraint(model, t_track_err <= tracking_err * sqrt(T - 1))
    end

    return nothing
end

function _setup_turnover(portfolio, N, obj)
    turnover = portfolio.turnover
    turnover_weights = portfolio.turnover_weights

    (isinf(turnover) || isempty(turnover_weights)) && (return nothing)

    model = portfolio.model

    @variable(model, t_turnov[1:N] >= 0)
    if obj == :Sharpe
        @expression(model, turnov, model[:w] .- turnover_weights * model[:k])
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] in MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover * model[:k])
    else
        @expression(model, turnov, model[:w] .- turnover_weights)
        @constraint(model, [i = 1:N], [t_turnov[i]; turnov[i]] in MOI.NormOneCone(2))
        @constraint(model, t_turnov .<= turnover)
    end

    return nothing
end

function _setup_trad_wc_constraints(portfolio, obj, T, N, type, class, kelly, l, returns)
    _setup_weights(portfolio, obj, N)
    _setup_min_number_effective_assets(portfolio, obj)
    _setup_tracking_err(portfolio, returns, obj, T)
    _setup_turnover(portfolio, N, obj)
    _setup_trad_wc_objective_function(portfolio, type, obj, class, kelly, l)

    return nothing
end

function _setup_trad_wc_objective_function(portfolio, type, obj, class, kelly, l)
    model = portfolio.model
    if obj == :Sharpe
        type == :Trad && class == :Classic && (kelly == :Exact || kelly == :Approx) ?
        @objective(model, Max, model[:ret]) : @objective(model, Min, model[:risk])
    elseif obj == :Min_Risk
        @objective(model, Min, model[:risk])
    elseif obj == :Utility
        @objective(model, Max, model[:ret] - l * model[:risk])
    elseif obj == :Max_Ret
        @objective(model, Max, model[:ret])
    end
    return nothing
end

function _optimize_portfolio(portfolio, type, obj)
    solvers = portfolio.solvers
    model = portfolio.model

    N = length(portfolio.assets)
    rtype = eltype(portfolio.returns)
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

        term_status in ValidTermination &&
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

            short = portfolio.short
            sum_short_long = portfolio.sum_short_long
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
                :portfolio => DataFrame(tickers = portfolio.assets, weights = weights),
            ),
        )
    end

    return term_status, solvers_tried
end

function _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj)
    model = portfolio.model

    if type == :Trad || type == :RP
        if rm == :EVaR
            portfolio.z_evar = value(portfolio.model[:z_evar])
        elseif rm == :EDaR
            portfolio.z_edar = value(portfolio.model[:z_edar])
        elseif rm == :RVaR
            portfolio.z_rvar = value(portfolio.model[:z_rvar])
        elseif rm == :RDaR
            portfolio.z_rdar = value(portfolio.model[:z_rdar])
        end
    end

    weights = Vector{eltype(returns)}(undef, N)
    if type == :Trad || type == :WC
        if obj == :Sharpe
            val_k = value(model[:k])
            val_k = val_k > 0 ? val_k : 1
            weights .= value.(model[:w]) / val_k
        else
            weights .= value.(model[:w])
        end

        short = portfolio.short
        sum_short_long = portfolio.sum_short_long
        if short == false
            sum_w = sum(abs.(weights))
            sum_w = sum_w > eps() ? sum_w : 1
            weights .= abs.(weights) / sum_w * sum_short_long
        end

        type == :Trad ?
        (portfolio.p_optimal = DataFrame(tickers = portfolio.assets, weights = weights)) :
        (portfolio.wc_optimal = DataFrame(tickers = portfolio.assets, weights = weights))

    elseif type == :RP || type == :RRP
        weights .= value.(model[:w])
        sum_w = sum(abs.(weights))
        sum_w = sum_w > eps() ? sum_w : 1
        weights .= abs.(weights) / sum_w

        type == :RP ?
        (portfolio.rp_optimal = DataFrame(tickers = portfolio.assets, weights = weights)) :
        (portfolio.rrp_optimal = DataFrame(tickers = portfolio.assets, weights = weights))
    end

    isempty(solvers_tried) ? portfolio.fail = Dict() : portfolio.fail = solvers_tried

    retval = if type == :Trad
        portfolio.p_optimal
    elseif type == :RP
        portfolio.rp_optimal
    elseif type == :RRP
        portfolio.rrp_optimal
    elseif type == :WC
        portfolio.wc_optimal
    end

    return retval
end

function _save_opt_params(
    portfolio,
    type,
    class,
    rm,
    obj,
    kelly,
    rrp_ver,
    rf,
    l,
    rrp_penalty,
    u_mu,
    u_cov,
    string_names,
    save_opt_params,
)
    !save_opt_params && return nothing

    opt_params_dict = if type == :Trad
        Dict(
            :class => class,
            :rm => rm,
            :obj => obj,
            :kelly => kelly,
            :rf => rf,
            :l => l,
            :string_names => string_names,
        )
    elseif type == :RP
        Dict(
            :class => class,
            :rm => rm,
            :obj => :Min_Risk,
            :kelly => (kelly == :Exact) ? :None : kelly,
            :rf => rf,
            :string_names => string_names,
        )
    elseif type == :RRP
        Dict(
            :class => class,
            :rm => :SD,
            :obj => :Min_Risk,
            :kelly => kelly,
            :rrp_penalty => rrp_penalty,
            :rrp_ver => rrp_ver,
            :string_names => string_names,
        )
    elseif type == :WC
        Dict(
            :rm => :SD,
            :obj => obj,
            :kelly => kelly,
            :rf => rf,
            :l => l,
            :u_mu => u_mu,
            :u_cov => u_cov,
            :string_names => string_names,
        )
    end

    portfolio.opt_params[type] = opt_params_dict

    return nothing
end

function opt_port!(
    portfolio::Portfolio;
    type::Symbol = :Trad,
    class::Symbol = :Classic,
    rm::Symbol = :SD,
    obj::Symbol = :Sharpe,
    kelly::Symbol = :None,
    rrp_ver::Symbol = :None,
    rf::Real = 0.0,#1.0329^(1 / 252) - 1
    l::Real = 2.0,
    rrp_penalty::Real = 1.0,
    u_mu = :Box,
    u_cov = :Box,
    string_names = false,
    save_opt_params = true,
)
    @assert(type ∈ PortTypes, "type must be one of $PortTypes")
    @assert(class ∈ PortClasses, "class must be one of $PortClasses")
    @assert(rm ∈ RiskMeasures, "rm = $rm, must be one of $RiskMeasures")
    @assert(obj ∈ ObjFuncs, "obj must be one of $ObjFuncs")
    @assert(kelly ∈ KellyRet, "kelly must be one of $KellyRet")
    @assert(rrp_ver ∈ RRPVersions)
    @assert(u_mu ∈ UncertaintyTypes, "u_mu must be one of $UncertaintyTypes")
    @assert(u_cov ∈ UncertaintyTypes, "u_cov must be one of $UncertaintyTypes")
    @assert(
        0 < portfolio.alpha < 1,
        "portfolio.alpha must be greater than 0 and smaller than 1"
    )
    @assert(
        0 < portfolio.kappa < 1,
        "portfolio.kappa must be greater than 0 and smaller than 1"
    )
    @assert(
        portfolio.kind_tracking_err ∈ TrackingErrKinds,
        "portfolio.kind_tracking_err must be one of $TrackingErrKinds"
    )

    portfolio.model = JuMP.Model()

    # Returns, mu, sigma.
    returns = portfolio.returns
    T, N = size(returns)
    mu = !isempty(portfolio.mu) ? portfolio.mu : nothing
    sigma = portfolio.cov
    kurtosis = portfolio.kurt
    skurtosis = portfolio.skurt

    # Model variables.
    model = portfolio.model
    set_string_names_on_creation(model, string_names)
    @variable(model, w[1:N])

    if type == :Trad
        _setup_sharpe_k(model, obj)
        _risk_setup(
            portfolio,
            :Trad,
            rm,
            kelly,
            obj,
            rf,
            T,
            N,
            mu,
            returns,
            sigma,
            kurtosis,
            skurtosis,
        )
        _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :Trad, class, kelly, l, returns)
    elseif type == :RP
        _setup_risk_budget(portfolio, N)
        _rp_setup(portfolio, N)
        _risk_setup(
            portfolio,
            :RP,
            rm,
            kelly,
            obj,
            rf,
            T,
            N,
            mu,
            returns,
            sigma,
            kurtosis,
            skurtosis,
        )
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    elseif type == :RRP
        _setup_risk_budget(portfolio, N)
        _mv_setup(portfolio, sigma, rm, kelly, obj, :RRP)
        _rrp_setup(portfolio, sigma, N, rrp_ver, rrp_penalty)
        _setup_rp_rrp_return_and_obj(portfolio, kelly, T, returns, mu)
    else
        _setup_sharpe_k(model, obj)
        _wc_setup(portfolio, kelly, obj, T, N, rf, mu, sigma, u_mu, u_cov)
        _setup_trad_wc_constraints(portfolio, obj, T, N, :WC, class, kelly, l, returns)
    end

    _setup_linear_constraints(portfolio, obj, type)

    term_status, solvers_tried = _optimize_portfolio(portfolio, type, obj)

    # Error handling.
    if term_status ∉ ValidTermination || any(.!isfinite.(value.(w)))
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.opt_port!))"

        @warn(
            "$funcname: model could not be optimised satisfactorily.\nPortfolio type: $type\nClass: $class\nRisk measure: $rm\nKelly return: $kelly\nObjective: $obj\nSolvers: $solvers_tried"
        )

        retval = if type == :Trad
            portfolio.p_optimal = DataFrame()
        elseif type == :RP
            portfolio.rp_optimal = DataFrame()
        elseif type == :RRP
            portfolio.rp_optimal = DataFrame()
        elseif type == :WC
            portfolio.wc_optimal = DataFrame()
        end

        portfolio.fail = solvers_tried
    else
        retval = _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj)
    end

    _save_opt_params(
        portfolio,
        type,
        class,
        rm,
        obj,
        kelly,
        rrp_ver,
        rf,
        l,
        rrp_penalty,
        u_mu,
        u_cov,
        string_names,
        save_opt_params,
    )

    return retval
end

export opt_port!
