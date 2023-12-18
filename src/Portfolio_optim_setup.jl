
function _setup_sharpe_k(model, obj)
    obj == :Sharpe && @variable(model, k >= 0)
    return nothing
end

function _setup_risk_budget(portfolio)
    model = portfolio.model
    if isempty(portfolio.risk_budget)
        portfolio.risk_budget = ()
    else
        !isapprox(sum(portfolio.risk_budget), one(eltype(portfolio.risk_budget))) &&
            (portfolio.risk_budget .= portfolio.risk_budget)
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
    elseif kelly == :Approx && !isempty(mu)
        @expression(model, _ret, dot(mu, model[:w]) - 0.5 * model[:dev_risk])
    elseif !isempty(mu)
        @expression(model, _ret, dot(mu, model[:w]))
    end

    !isinf(mu_l) && @constraint(model, _ret >= mu_l)

    return nothing
end

function _setup_sharpe_ret(kelly, model, T, rf, returns, mu, mu_l, trad = true)
    if kelly == :Exact
        @variable(model, texact_kelly[1:T])
        @expression(model, _ret, sum(texact_kelly) / T - rf * model[:k])
        @expression(model, kret, model[:k] .+ returns * model[:w])
        @constraint(
            model,
            [i = 1:T],
            [texact_kelly[i], model[:k], kret[i]] in MOI.ExponentialCone()
        )
        trad && @constraint(model, model[:risk] <= 1)
    elseif kelly == :Approx && !isempty(mu)
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
        trad && @constraint(model, model[:risk] <= 1)
    elseif !isempty(mu)
        @expression(model, _ret, dot(mu, model[:w]))
        trad && @constraint(model, _ret - rf * model[:k] == 1)
    end

    !isinf(mu_l) && @constraint(model, _ret >= mu_l * model[:k])

    return nothing
end

function _setup_trad_return(portfolio, class, kelly, obj, T, rf, returns, mu)
    model = portfolio.model
    mu_l = portfolio.mu_l

    kelly = class == :Classic ? kelly : :None

    obj == :Sharpe ? _setup_sharpe_ret(kelly, model, T, rf, returns, mu, mu_l) :
    _setup_ret(kelly, model, T, returns, mu, mu_l)

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
        (type == :Trad && class == :Classic || type == :WC) && kelly != :None ?
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

    isempty(solvers_tried) &&
        term_status ∉ ValidTermination &&
        push!(solvers_tried, :error => term_status)

    return term_status, solvers_tried
end

function _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj)
    model = portfolio.model

    if (type == :Trad || type == :RP) && rm ∈ (:EVaR, :EDaR, :RVaR, :RDaR)
        z_key = "z_" * lowercase(string(rm))
        z_key2 = Symbol(string(type) * "_" * z_key)
        portfolio.z[z_key2] = value(portfolio.model[Symbol(z_key)])
        type == :Trad &&
            obj == :Sharpe &&
            (portfolio.z[z_key2] /= value(portfolio.model[:k]))
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
    elseif type == :RP || type == :RRP
        weights .= value.(model[:w])
        sum_w = sum(abs.(weights))
        sum_w = sum_w > eps() ? sum_w : 1
        weights .= abs.(weights) / sum_w
    end

    portfolio.optimal[type] = DataFrame(tickers = portfolio.assets, weights = weights)

    isempty(solvers_tried) ? portfolio.fail = Dict() : portfolio.fail = solvers_tried

    return portfolio.optimal[type]
end

function _handle_errors_and_finalise(
    portfolio,
    term_status,
    returns,
    N,
    solvers_tried,
    type,
    rm,
    obj,
)
    retval =
        if term_status ∉ ValidTermination || any(.!isfinite.(value.(portfolio.model[:w])))
            funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.opt_port!))"
            @warn(
                "$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried."
            )
            portfolio.fail = solvers_tried
            DataFrame()
        else
            _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj)
        end

    return retval
end

function _p_save_opt_params(
    portfolio,
    type,
    class,
    hist,
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
            :hist => hist,
        )
    elseif type == :RP
        Dict(
            :class => class,
            :rm => rm,
            :obj => :Min_Risk,
            :kelly => (kelly == :Exact) ? :None : kelly,
            :rf => rf,
            :string_names => string_names,
            :hist => hist,
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
            :hist => hist,
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
            :hist => hist,
        )
    end

    portfolio.opt_params[type] = opt_params_dict

    return nothing
end

function _setup_model_class(portfolio, class, hist)
    mu, sigma, returns = (
        Vector{eltype(portfolio.mu)}(undef, 0),
        Matrix{eltype(portfolio.cov)}(undef, 0, 0),
        Matrix{eltype(portfolio.returns)}(undef, 0, 0),
    )

    class != :Classic && @assert(hist ∈ BLHist, "hist = $hist, must be one of $BLHist")

    if class == :Classic
        mu = portfolio.mu
        sigma = portfolio.cov
        returns = portfolio.returns
    elseif class == :FM
        mu = portfolio.mu_fm
        if hist == 1
            sigma = portfolio.cov_fm
            returns = portfolio.returns_fm
        elseif hist == 2
            sigma = portfolio.cov
            returns = portfolio.returns
        else
            throw(AssertionError("for class = $class, hist = $hist can only be 1 or 2"))
        end
    elseif class == :BL
        mu = portfolio.mu_bl
        if hist == 1
            sigma = portfolio.cov_bl
        elseif hist == 2
            sigma = portfolio.cov
        else
            throw(AssertionError("for class = $class, hist = $hist can only be 1 or 2"))
        end
        returns = portfolio.returns
    elseif class == :BLFM
        mu = portfolio.mu_bl_fm
        if hist == 1
            sigma = portfolio.cov_bl_fm
            returns = portfolio.returns_fm
        elseif hist == 2
            sigma = portfolio.cov
            returns = portfolio.returns
        else
            sigma = portfolio.cov_fm
            returns = portfolio.returns_fm
        end
    end

    return mu, sigma, returns
end

"""
```julia
opt_port!(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    l::Real = 2.0,
    obj::Symbol = :Sharpe,
    rf::Real = 0.0,#1.0329^(1 / 252) - 1
    rm::Symbol = :SD,
    rrp_penalty::Real = 1.0,
    rrp_ver::Symbol = :None,
    save_opt_params::Bool = true,
    string_names::Bool = false,
    type::Symbol = :Trad,
    u_cov::Symbol = :Box,
    u_mu::Symbol = :Box,
)
```
"""
function opt_port!(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    l::Real = 2.0,
    obj::Symbol = :Sharpe,
    rf::Real = 0.0,#1.0329^(1 / 252) - 1
    rm::Symbol = :SD,
    rrp_penalty::Real = 1.0,
    rrp_ver::Symbol = :None,
    save_opt_params::Bool = true,
    string_names::Bool = false,
    type::Symbol = :Trad,
    u_cov::Symbol = :Box,
    u_mu::Symbol = :Box,
)
    @assert(type ∈ PortTypes, "type = $type, must be one of $PortTypes")
    @assert(class ∈ PortClasses, "class = $class, must be one of $PortClasses")
    @assert(rm ∈ RiskMeasures, "rm = $rm, must be one of $RiskMeasures")
    @assert(obj ∈ ObjFuncs, "obj = $obj, must be one of $ObjFuncs")
    @assert(kelly ∈ KellyRet, "kelly = $kelly, must be one of $KellyRet")
    @assert(rrp_ver ∈ RRPVersions)
    @assert(u_mu ∈ UncertaintyTypes, "u_mu = $u_mu, must be one of $UncertaintyTypes")
    @assert(u_cov ∈ UncertaintyTypes, "u_cov = $u_cov, must be one of $UncertaintyTypes")
    @assert(
        0 < portfolio.alpha < 1,
        "portfolio.alpha = $(portfolio.alpha), must be greater than 0 and less than 1"
    )
    @assert(
        0 < portfolio.kappa < 1,
        "portfolio.kappa = $(portfolio.kappa), must be greater than 0 and less than 1"
    )
    @assert(
        portfolio.kind_tracking_err ∈ TrackingErrKinds,
        "portfolio.kind_tracking_err = $(portfolio.kind_tracking_err), must be one of $TrackingErrKinds"
    )

    _p_save_opt_params(
        portfolio,
        type,
        class,
        hist,
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

    portfolio.model = JuMP.Model()

    # mu, sigma, returns
    mu, sigma, returns = _setup_model_class(portfolio, class, hist)
    T, N = size(returns)
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
        _setup_risk_budget(portfolio)
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
        _setup_risk_budget(portfolio)
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

    retval = _handle_errors_and_finalise(
        portfolio,
        term_status,
        returns,
        N,
        solvers_tried,
        type,
        rm,
        obj,
    )

    return retval
end

function frontier_limits!(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    rf::Real = 0.0,
    rm::Symbol = :SD,
)
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)

    w_min = opt_port!(
        portfolio;
        class = class,
        hist = hist,
        kelly = kelly,
        obj = :Min_Risk,
        rf = rf,
        rm = rm,
        save_opt_params = false,
    )

    w_max = opt_port!(
        portfolio;
        class = class,
        hist = hist,
        kelly = kelly,
        obj = :Max_Ret,
        rf = rf,
        rm = rm,
        save_opt_params = false,
    )

    limits = hcat(w_min, DataFrame(x1 = w_max[!, 2]))
    DataFrames.rename!(limits, :weights => :w_min, :x1 => :w_max)
    portfolio.limits[rm] = limits

    portfolio.optimal = optimal1
    portfolio.fail = fail1

    return portfolio.limits[rm]
end

function efficient_frontier(
    portfolio::Portfolio;
    class::Symbol = :Classic,
    hist::Integer = 1,
    kelly::Symbol = :None,
    rf::Real = 0.0,#1.0329^(1 / 252) - 1
    rm::Symbol = :SD,
    points::Integer = 20,
)
    optimal1 = deepcopy(portfolio.optimal)
    fail1 = deepcopy(portfolio.fail)

    mu, sigma, returns = _setup_model_class(portfolio, class, hist)

    fl = frontier_limits!(
        portfolio;
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

    alpha_i = portfolio.alpha_i
    alpha = portfolio.alpha
    a_sim = portfolio.a_sim
    beta_i = portfolio.beta_i
    beta = portfolio.beta
    b_sim = portfolio.b_sim
    kappa = portfolio.kappa
    owa_w = portfolio.owa_w
    solvers = portfolio.solvers

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

    mus = range(ret1, stop = ret2, length = points)
    risks = range(risk1, stop = risk2, length = points)

    ur = (
        SD = :dev_u,
        MAD = :mad_u,
        SSD = :sdev_u,
        FLPM = :flpm_u,
        SLPM = :slpm_u,
        WR = :wr_u,
        CVaR = :cvar_u,
        EVaR = :evar_u,
        RVaR = :rvar_u,
        MDD = :mdd_u,
        ADD = :add_u,
        CDaR = :cdar_u,
        UCI = :uci_u,
        EDaR = :edar_u,
        RDaR = :rdar_u,
        Kurt = :kurt_u,
        SKurt = :skurt_u,
        GMD = :gmd_u,
        RG = :rg_u,
        RCVaR = :rcvar_u,
        TG = :tg_u,
        RTG = :rtg_u,
        OWA = :owa_u,
    )

    rmf = ur[rm]

    frontier = Vector{typeof(risk1)}(undef, 0)
    srisk = Vector{typeof(risk1)}(undef, 0)

    i = 0
    for (j, r) in enumerate(risks)
        if i == 0
            w = opt_port!(
                portfolio;
                class = class,
                hist = hist,
                kelly = kelly,
                obj = :Min_Risk,
                rf = rf,
                rm = rm,
                save_opt_params = false,
            )
        else
            j != length(risks) ? setfield!(portfolio, rmf, r) :
            setfield!(portfolio, rmf, Inf)
            w = opt_port!(
                portfolio;
                class = class,
                hist = hist,
                kelly = kelly,
                obj = :Max_Ret,
                rf = rf,
                rm = rm,
                save_opt_params = false,
            )
        end
        isempty(w) && continue
        rk = calc_risk(
            w.weights,
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
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
    end
    setfield!(portfolio, rmf, Inf)

    w = opt_port!(
        portfolio;
        class = class,
        hist = hist,
        kelly = kelly,
        obj = :Sharpe,
        rf = rf,
        rm = rm,
        save_opt_params = false,
    )
    sharpe = false
    if !isempty(w)
        rk = calc_risk(
            w.weights,
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
        append!(frontier, w.weights)
        push!(srisk, rk)
        i += 1
        sharpe = true
    end

    portfolio.frontier[rm] = Dict(
        :weights => hcat(
            DataFrame(tickers = portfolio.assets),
            DataFrame(reshape(frontier, length(w1), :), string.(range(1, i))),
        ),
        :class => class,
        :hist => hist,
        :kelly => kelly,
        :rf => rf,
        :points => points,
        :risk => srisk,
        :sharpe => sharpe,
    )

    portfolio.optimal = optimal1
    portfolio.fail = fail1

    return portfolio.frontier
end

export opt_port!, frontier_limits!, efficient_frontier
