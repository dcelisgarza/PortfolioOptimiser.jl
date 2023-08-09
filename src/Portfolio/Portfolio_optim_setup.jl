function _setup_k_and_risk_budged(portfolio, obj, N, type)
    model = portfolio.model
    if obj == :sharpe && (type == :trad || type == :wc)
        @variable(model, k >= 0)
    elseif type == :rp || type == :rrp
        if isempty(portfolio.risk_budget) || isa(portfolio.risk_budget, Real)
            portfolio.risk_budget = fill(1 / N, N)
        else
            portfolio.risk_budget ./= sum(portfolio.risk_budget)
        end
        @variable(model, k >= 0)
    end
    return nothing
end

const KellyRet = (:none, :approx, :exact)
function _setup_return(portfolio, type, class, kelly, obj, T, rf, returns, mu)
    model = portfolio.model

    if type == :trad
        if class == :classic && (kelly == :exact || kelly == :approx)
            if kelly == :exact
                @variable(model, texact_kelly[1:T])
                if obj == :sharpe
                    @expression(model, ret, sum(texact_kelly) / T - rf * model[:k])
                    @expression(model, kret, model[:k] .+ returns * model[:w])
                    @constraint(
                        model,
                        [i = 1:T],
                        [texact_kelly[i], model[:k], kret[i]] in MOI.ExponentialCone()
                    )
                    @constraint(model, model[:risk] <= 1)
                else
                    @expression(model, ret, sum(texact_kelly) / T)
                    @expression(model, kret, 1 .+ returns * model[:w])
                    @constraint(
                        model,
                        [i = 1:T],
                        [texact_kelly[i], 1, kret[i]] in MOI.ExponentialCone()
                    )
                end
            elseif kelly == :approx
                if obj == :sharpe
                    @variable(model, tapprox_kelly)
                    @constraint(
                        model,
                        [
                            model[:k] + tapprox_kelly
                            2 * model[:dev]
                            model[:k] - tapprox_kelly
                        ] in SecondOrderCone()
                    )
                    @expression(model, ret, dot(mu, model[:w]) - 0.5 * tapprox_kelly)
                    @constraint(model, model[:risk] <= 1)
                else
                    @expression(model, ret, dot(mu, model[:w]) - 0.5 * model[:dev_risk])
                end
            end
        else
            obj == :min_risk && isnothing(mu) && return nothing
            @expression(model, ret, dot(mu, model[:w]))
            if obj == :sharpe
                @constraint(model, ret - rf * model[:k] == 1)
            end
        end
    elseif type == :rp || type == :rrp
        # Exact is infeasable with exponential cone rp weight constraints.
        if kelly == :exact && type == :rrp
            @variable(model, texact_kelly[1:T])
            @expression(model, ret, sum(texact_kelly) / T)
            @expression(model, kret, 1 .+ returns * model[:w])
            @constraint(
                model,
                [i = 1:T],
                [texact_kelly[i], 1, kret[i]] in MOI.ExponentialCone()
            )
        elseif kelly == :approx
            @expression(model, ret, dot(mu, model[:w]) - 0.5 * model[:dev_risk])
        else
            isempty(mu) && return nothing
            @expression(model, ret, dot(mu, model[:w]))
        end
    end

    # Return constraints.
    mu_l = portfolio.mu_l
    !isfinite(mu_l) && (return nothing)

    if obj == :sharpe || type == :rp
        @constraint(model, ret >= mu_l * model[:k])
    else
        @constraint(model, ret >= mu_l)
    end

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
        if obj == :sharpe
            @variable(model, tass_bin[1:N], binary = true)
            @variable(model, tass_bin_sharpe[1:N] >= 0)
        else
            @variable(model, tass_bin[1:N], binary = true)
        end
    end

    # Weight constraints.
    if obj == :sharpe
        @constraint(model, sum(model[:w]) == sum_short_long * model[:k])

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum(tass_bin) <= max_number_assets)
            @constraint(model, tass_bin_sharpe .<= model[:k])
            @constraint(model, tass_bin_sharpe .<= factor * tass_bin)
            @constraint(model, tass_bin_sharpe .>= model[:k] - factor * (1 .- tass_bin))
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
            if max_number_assets > 0
                @constraint(model, model[:w] .>= -short_u * tass_bin_sharpe)
            end
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
            if max_number_assets > 0
                @constraint(model, model[:w] .>= -short_u * tass_bin)
            end
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
    if obj == :sharpe || type == :rp
        @constraint(model, A * model[:w] .- B * model[:k] .>= 0)
    else
        @constraint(model, A * model[:w] .- B .>= 0)
    end

    return nothing
end

function _setup_min_number_effective_assets(portfolio, obj)
    mnea = portfolio.min_number_effective_assets

    (mnea < 1) && (return nothing)

    model = portfolio.model

    @variable(model, tmnea >= 0)
    @constraint(model, [tmnea; model[:w]] in SecondOrderCone())
    if obj == :sharpe
        @constraint(model, tmnea * sqrt(mnea) <= model[:k])
    else
        @constraint(model, tmnea * sqrt(mnea) <= 1)
    end

    return nothing
end

const TrackingErrKinds = (:weights, :returns)
function _setup_tracking_err(portfolio, returns, obj, T)
    tracking_err = portfolio.tracking_err

    !isfinite(tracking_err) && (return nothing)

    kind_tracking_err = portfolio.kind_tracking_err
    tracking_err_weights = portfolio.tracking_err_weights
    tracking_err_returns = portfolio.tracking_err_returns

    tracking_err_flag = false

    if kind_tracking_err == :weights && !isempty(tracking_err_weights)
        benchmark = returns * tracking_err_weights
        tracking_err_flag = true
    elseif kind_tracking_err == :returns && !isempty(tracking_err_returns)
        benchmark = tracking_err_returns
        tracking_err_flag = true
    end

    !(tracking_err_flag) && (return nothing)

    model = portfolio.model

    @variable(model, t_track_err >= 0)
    if obj == :sharpe
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
    if obj == :sharpe
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

const ObjFuncs = (:min_risk, :utility, :sharpe, :max_ret)
function _setup_objective_function(portfolio, type, obj, class, kelly, l)
    model = portfolio.model

    if type == :trad || type == :wc
        if obj == :sharpe
            if type == :trad && class == :classic && (kelly == :exact || kelly == :approx)
                @objective(model, Max, model[:ret])
            else
                @objective(model, Min, model[:risk])
            end
        elseif obj == :min_risk
            @objective(model, Min, model[:risk])
        elseif obj == :utility
            @objective(model, Max, model[:ret] - l * model[:risk])
        elseif obj == :max_ret
            @objective(model, Max, model[:ret])
        end
    elseif type == :rp || type == :rrp
        @objective(model, Min, model[:risk])
    end

    return nothing
end

const ValidTermination =
    (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
function _optimize_portfolio(portfolio)
    solvers = portfolio.solvers
    model = portfolio.model

    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) in solvers
        if haskey(val, :solver)
            set_optimizer(model, val[:solver])
        end

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

        push!(
            solvers_tried,
            key => Dict(
                :objective_val => objective_value(model),
                :term_status => term_status,
                :params => haskey(val, :params) ? val[:params] : missing,
                :finite_weights => all_finite_weights,
                :nonzero_weights => all_non_zero_weights,
            ),
        )
    end

    return term_status, solvers_tried
end

function _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj)
    model = portfolio.model

    if type == :trad || type == :rp
        if rm == :evar
            portfolio.z_evar = value(portfolio.model[:z_evar])
        elseif rm == :edar
            portfolio.z_edar = value(portfolio.model[:z_edar])
        elseif rm == :rvar
            portfolio.z_rvar = value(portfolio.model[:z_rvar])
        elseif rm == :rdar
            portfolio.z_rdar = value(portfolio.model[:z_rdar])
        end
    end

    weights = Vector{eltype(returns)}(undef, N)
    if type == :trad || type == :wc
        if obj == :sharpe
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

        if type == :trad
            portfolio.p_optimal = DataFrame(tickers = portfolio.assets, weights = weights)
        else
            portfolio.wc_optimal = DataFrame(tickers = portfolio.assets, weights = weights)
        end
    elseif type == :rp || type == :rrp
        weights .= value.(model[:w])
        sum_w = sum(abs.(weights))
        sum_w = sum_w > eps() ? sum_w : 1
        weights .= abs.(weights) / sum_w
        if type == :rp
            portfolio.rp_optimal = DataFrame(tickers = portfolio.assets, weights = weights)
        else
            portfolio.rrp_optimal = DataFrame(tickers = portfolio.assets, weights = weights)
        end
    end

    if isempty(solvers_tried)
        portfolio.fail = Dict()
    else
        portfolio.fail = solvers_tried
    end

    retval = if type == :trad
        portfolio.p_optimal
    elseif type == :rp
        portfolio.rp_optimal
    elseif type == :rrp
        portfolio.rrp_optimal
    elseif type == :wc
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

    opt_params_dict = if type == :trad
        Dict(
            :class => class,
            :rm => rm,
            :obj => obj,
            :kelly => kelly,
            :rf => rf,
            :l => l,
            :string_names => string_names,
        )
    elseif type == :rp
        Dict(
            :class => class,
            :rm => rm,
            :obj => :min_risk,
            :kelly => (kelly == :exact) ? :none : kelly,
            :rf => rf,
            :string_names => string_names,
        )
    elseif type == :rrp
        Dict(
            :class => class,
            :rm => :mv,
            :obj => :min_risk,
            :kelly => kelly,
            :rrp_penalty => rrp_penalty,
            :rrp_ver => rrp_ver,
            :string_names => string_names,
        )
    elseif type == :wc
        Dict(
            :rm => :mv,
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

const PortClasses = (:classic,)
const PortTypes = (:trad, :rp, :rrp, :owa, :wc)
const UncertaintyTypes = (:none, :box, :ellipse)
const RRPVersions = (:none, :reg, :reg_pen)
function opt_port!(
    portfolio::Portfolio;
    type::Symbol = :trad,
    class::Symbol = :classic,
    rm::Symbol = :mv,
    obj::Symbol = :sharpe,
    kelly::Symbol = :none,
    rrp_ver::Symbol = :none,
    rf::Real = 0.0,#1.0329^(1 / 252) - 1
    l::Real = 2.0,
    rrp_penalty::Real = 1.0,
    u_mu = :box,
    u_cov = :box,
    string_names = false,
    save_opt_params = true,
)
    @assert(type ∈ PortTypes, "type must be one of $PortTypes")
    @assert(class ∈ PortClasses, "class must be one of $PortClasses")
    @assert(rm ∈ RiskMeasures, "rm must be one of $RiskMeasures")
    @assert(obj ∈ ObjFuncs, "obj must be one of $ObjFuncs")
    @assert(kelly ∈ KellyRet, "kelly must be one of $KellyRet")
    @assert(rrp_ver ∈ RRPVersions)
    @assert(u_mu ∈ UncertaintyTypes, "u_mu must be one of $UncertaintyTypes")
    @assert(u_cov ∈ UncertaintyTypes, "u_cov must be one of $UncertaintyTypes")
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

    _setup_k_and_risk_budged(portfolio, obj, N, type)

    type != :wc && _mv_setup(portfolio, sigma, rm, kelly, obj, type)

    # Risk variables, functions and constraints.
    if type == :trad || type == :rp
        _calc_var_dar_constants(portfolio, rm, T)
        ## Mean variance.
        ## Mean Absolute Deviation and Mean Semi Deviation.
        _mad_setup(portfolio, rm, T, returns, mu, obj, type)
        ## Lower partial moments, Omega and Sortino ratios.
        _lpm_setup(portfolio, rm, T, returns, obj, rf, type)
        ## Worst realisation.
        _wr_setup(portfolio, rm, returns, obj, type)
        ## Conditional and Entropic Value at Risk
        _var_setup(portfolio, rm, T, returns, obj, type)
        ## Drawdown, Max Drawdown, Average Drawdown, Conditional Drawdown, Ulcer Index, Entropic Drawdown at Risk
        _drawdown_setup(portfolio, rm, T, returns, obj, type)
        ## Kurtosis setup
        _kurtosis_setup(portfolio, kurtosis, skurtosis, rm, N, obj, type)
        ## OWA methods
        _owa_setup(portfolio, rm, T, returns, obj, type)
        ## RP setupt
        if type == :rp
            _rp_setup(portfolio, N)
        end
    elseif type == :rrp
        _rrp_setup(portfolio, sigma, N, rrp_ver, rrp_penalty)
    elseif type == :wc
        _wc_setup(portfolio, obj, N, rf, mu, sigma, u_mu, u_cov)
    end

    # Constraints.
    ## Return constraints.
    (type == :trad || type == :rp || type == :rrp) &&
        _setup_return(portfolio, type, class, kelly, obj, T, rf, returns, mu)

    ## Linear weight constraints.
    _setup_linear_constraints(portfolio, obj, type)

    if type == :trad || type == :wc
        ## Weight constraints.
        _setup_weights(portfolio, obj, N)
        ## Minimum number of effective assets.
        _setup_min_number_effective_assets(portfolio, obj)
        ## Tracking error variables and constraints.
        _setup_tracking_err(portfolio, returns, obj, T)
        ## Turnover variables and constraints
        _setup_turnover(portfolio, N, obj)
    end

    # Objective functions.
    _setup_objective_function(portfolio, type, obj, class, kelly, l)

    # Optimize.
    term_status, solvers_tried = _optimize_portfolio(portfolio)

    # Error handling.
    if term_status ∉ ValidTermination || any(.!isfinite.(value.(w)))
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.opt_port!))"

        @warn(
            "$funcname: model could not be optimised satisfactorily.\nPortfolio type: $type\nClass: $class\nRisk measure: $rm\nKelly return: $kelly\nObjective: $obj\nSolvers: $solvers_tried"
        )

        retval = if type == :trad
            portfolio.p_optimal = DataFrame()
        elseif type == :rp
            portfolio.rp_optimal = DataFrame()
        elseif type == :rrp
            portfolio.rp_optimal = DataFrame()
        elseif type == :wc
            portfolio.wc_optimal = DataFrame()
        end

        portfolio.fail = solvers_tried
    else
        retval = _finalise_portfolio(portfolio, returns, N, solvers_tried, type, rm, obj)
    end

    # Save optimisation parameters.
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

const HRTypes = (:hrp, :herc, :herc2, :nco)
const CodepTypes = (
    :pearson,
    :spearman,
    :kendall,
    :gerber1,
    :gerber2,
    :abs_pearson,
    :abs_spearman,
    :abs_kendall,
    :distance,
    :mutual_info,
    :tail,
    :custom_cov,
    :custom_cor,
)
const LinkageTypes = (:single, :complete, :average, :ward_presquared, :ward, :dbht)
const BranchOrderTypes = (:optimal, :barjoseph, :r, :default)
const HRObjFuncs = (:min_risk, :utility, :sharpe, :erc)

function _setup_hr_weights(w_max, w_min, N)
    @assert(
        all(w_max .>= w_min),
        "all upper bounds must be bigger than their corresponding lower bounds"
    )

    upper_bound = if isnothing(w_max)
        ones(N)
    elseif isa(w_max, Real)
        fill(min(1.0, w_max), N)
    else
        min.(1.0, w_max)
    end

    lower_bound = if isnothing(w_min)
        zeros(N)
    elseif isa(w_min, Real)
        fill(max(0.0, w_min), N)
    else
        max.(0.0, w_min)
    end

    return upper_bound, lower_bound
end

function opt_port!(
    portfolio::HCPortfolio;
    type::Symbol = :hrp,
    rm::Symbol = :mv,
    obj::Symbol = :min_risk,
    kelly::Symbol = :none,
    rf::Real = 0.0,#1.0329^(1 / 252) - 1
    l::Real = 2.0,
    linkage::Symbol = :single,
    k = nothing,
    max_k = 10,
    branchorder = :optimal,
    max_iter = 100,
    save_opt_params = true,
)
    @assert(type ∈ HRTypes, "type must be one of $HRTypes")
    @assert(rm ∈ HRRiskMeasures, "rm must be one of $HRRiskMeasures")
    @assert(obj ∈ HRObjFuncs, "obj must be one of $HRObjFuncs")
    @assert(kelly ∈ KellyRet, "kelly must be one of $KellyRet")
    @assert(linkage ∈ LinkageTypes, "linkage must be one of $LinkageTypes")
    @assert(portfolio.codep_type ∈ CodepTypes, "codep_type must be one of $CodepTypes")
    @assert(
        0 < portfolio.kappa < 1,
        "portfolio.kappa must be greater than 0 and smaller than 1"
    )

    N = length(portfolio.assets)

    portfolio.clusters, tk =
        _hierarchical_clustering(portfolio, type, linkage, max_k, branchorder)

    portfolio.k = isnothing(k) ? tk : k

    portfolio.sort_order = leaves_list(portfolio.clusters)

    upper_bound, lower_bound = _setup_hr_weights(portfolio.w_max, portfolio.w_min, N)

    if type == :hrp
        weights = _recursive_bisection(
            portfolio;
            rm = rm,
            rf = rf,
            upper_bound = upper_bound,
            lower_bound = lower_bound,
        )
    elseif type == :herc || type == :herc2
        weights = _hierarchical_recursive_bisection(
            portfolio;
            rm = rm,
            rf = rf,
            type = type,
            upper_bound = upper_bound,
            lower_bound = lower_bound,
        )
    else
        intra_weights = _intra_weights(portfolio; obj = obj, rm = rm, rf = rf, l = l)
        weights =
            _inter_weights(portfolio, intra_weights, obj = obj, rm = rm, rf = rf, l = l)
    end

    weights = _opt_weight_bounds(upper_bound, lower_bound, weights, max_iter)

    portfolio.p_optimal = DataFrame(tickers = portfolio.assets, weights = weights)

    return portfolio.p_optimal
end

export opt_port!