
using DataFrames, JuMP

abstract type AbstractPortfolio end
const UnionBoolNothing = Union{Bool, Nothing}
const UnionRealNothing = Union{<:Real, Nothing}
const UnionIntegerNothing = Union{<:Integer, Nothing}
const UnionVecNothing = Union{Vector{<:Real}, Nothing}
const UnionMtxNothing = Union{Matrix{<:Real}, Nothing}
const UnionDataFrameNothing = Union{DataFrame, Nothing}
const PortClasses = (:classic,)
const KellyRet = (:exact, :approx, :none)
const ObjFuncs = (:min_risk, :utility, :sharpe, :max_ret)
const RiskMeasures = (:mean_var, :mean_abs_dev, :mean_semi_dev)
const TrackingErrKinds = (:weights, :returns)
const ValidTermination =
    (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)

mutable struct Portfolio{
    # Portfolio characteristics
    r,
    s,
    us,
    ul,
    ssl,
    mnea,
    mna,
    rf,
    l,
    # Risk parameters
    a,
    as,
    b,
    bs,
    k,
    mnak,
    # Benchmark constraints
    kb,
    ato,
    turnover,
    tobw,
    ate,
    te,
    rbi,
    bw,
    # Risk and return constraints
    ami,
    bvi,
    rbv,
    ler,
    ud,
    uk,
    umad,
    ugmd,
    usd,
    usk,
    uflpm,
    uslpm,
    ucvar,
    utg,
    uevar,
    urvar,
    uwr,
    ur,
    urcvar,
    urtg,
    umd,
    uad,
    ucdar,
    uedar,
    urdar,
    uui,
    # Optimisation model inputs
    tmu,
    tcov,
    tkurt,
    tskurt,
    tl2,
    ts2,
    tmuf,
    tcovf,
    tmufm,
    tcovfm,
    tmubl,
    tcovbl,
    tmublf,
    tcovblf,
    trfm,
    tevar,
    tedar,
    trvar,
    trdar,
    tcovl,
    tcovu,
    tcovmu,
    tcovs,
    tdmu,
    tkmu,
    tks,
    topt,
    trpopt,
    trrpopt,
    twcopt,
    tlim,
    tfront,
    tsolv,
    tsolvp,
    tmod,
    tf,
} <: AbstractPortfolio
    # Portfolio characteristics.
    returns::r
    short::s
    upper_short::us
    upper_long::ul
    sum_short_long::ssl
    min_number_effective_assets::mnea
    max_number_assets::mna
    returns_factors::rf
    loadings::l
    # Risk parameters
    alpha::a
    a_sim::as
    beta::b
    b_sim::bs
    kappa::k
    max_num_assets_kurt::mnak
    # Benchmark constraints
    kind_tracking_err::kb
    allow_turnover::ato
    turnover::turnover
    turnover_weights::tobw
    allow_tracking_err::ate
    tracking_err::te
    tracking_err_returns::rbi
    tracking_err_weights::bw
    # Risk and return constraints
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget_vec::rbv
    lower_mu::ler
    upper_dev::ud
    upper_krt::uk
    upper_mean_abs_dev::umad
    upper_gini_mean_diff::ugmd
    upper_mean_semi_dev::usd
    upper_semi_krt::usk
    upper_first_lower_partial_moment::uflpm
    upper_second_lower_partial_moment::uslpm
    upper_conditional_value_at_risk::ucvar
    upper_tail_gini::utg
    upper_entropic_value_at_risk::uevar
    upper_relativistic_value_at_risk::urvar
    upper_worst_realisation::uwr
    upper_range::ur
    upper_range_conditional_value_at_risk::urcvar
    upper_range_tail_gini::urtg
    upper_max_drawdown::umd
    upper_average_drawdown::uad
    upper_conditional_drawdown_at_risk::ucdar
    upper_entropic_drawdown_at_risk::uedar
    upper_relativistic_drawdown_at_risk::urdar
    upper_ulcer_index::uui
    # Optimisation model inputs
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    mu_f::tmuf
    cov_f::tcovf
    mu_fm::tmufm
    cov_fm::tcovfm
    mu_bl::tmubl
    cov_bl::tcovbl
    mu_bl_fm::tmublf
    cov_bl_fm::tcovblf
    returns_fm::trfm
    z_evar::tevar
    z_edar::tedar
    z_rvar::trvar
    z_rdar::trdar
    # Inputs of Worst Case Optimization Models
    cov_l::tcovl
    cov_u::tcovu
    cov_mu::tcovmu
    cov_sigma::tcovs
    d_mu::tdmu
    k_mu::tkmu
    k_sigma::tks
    # Optimal portfolios
    p_optimal::topt
    rp_optimal::trpopt
    rrp_optimal::trrpopt
    wc_optimal::twcopt
    limits::tlim
    frontier::tfront
    # Solver params
    solvers::tsolv
    sol_params::tsolvp
    model::tmod
    fail::tf
end

function Portfolio(;
    # Portfolio characteristics.
    returns::DataFrame,
    short::Bool = false,
    upper_short::Real = 0.2,
    upper_long::Real = 1.0,
    sum_short_long::Real = 1,
    min_number_effective_assets::Integer = 0,
    max_number_assets::Integer = 0,
    returns_factors::DataFrame = DataFrame(),
    loadings::DataFrame = DataFrame(),
    # Risk parameters
    alpha::Real = 0.05,
    a_sim::Integer = 100,
    beta::Real = 0.0,
    b_sim::Integer = 0,
    kappa::Real = 0.3,
    max_num_assets_kurt::Integer = 50,
    # Benchmark constraints
    kind_tracking_err::Symbol = :weights,
    allow_turnover::Bool = false,
    turnover::Real = 0.05,
    turnover_weights::DataFrame = DataFrame(),
    allow_tracking_err::Bool = false,
    tracking_err::Real = 0.05,
    tracking_err_returns::DataFrame = DataFrame(),
    tracking_err_weights::DataFrame = DataFrame(),
    # Risk and return constraints
    a_mtx_ineq::Matrix{<:Real} = Array{Float64}(undef, 0, 0),
    b_vec_ineq::Vector{<:Real} = Array{Float64}(undef, 0),
    risk_budget_vec::Vector{<:Real} = Array{Float64}(undef, 0),
    lower_mu::Real = -Inf,
    upper_dev::Real = Inf,
    upper_krt::Real = Inf,
    upper_mean_abs_dev::Real = Inf,
    upper_gini_mean_diff::Real = Inf,
    upper_mean_semi_dev::Real = Inf,
    upper_semi_krt::Real = Inf,
    upper_first_lower_partial_moment::Real = Inf,
    upper_second_lower_partial_moment::Real = Inf,
    upper_conditional_value_at_risk::Real = Inf,
    upper_tail_gini::Real = Inf,
    upper_entropic_value_at_risk::Real = Inf,
    upper_relativistic_value_at_risk::Real = Inf,
    upper_worst_realisation::Real = Inf,
    upper_range::Real = Inf,
    upper_range_conditional_value_at_risk::Real = Inf,
    upper_range_tail_gini::Real = Inf,
    upper_max_drawdown::Real = Inf,
    upper_average_drawdown::Real = Inf,
    upper_conditional_drawdown_at_risk::Real = Inf,
    upper_entropic_drawdown_at_risk::Real = Inf,
    upper_relativistic_drawdown_at_risk::Real = Inf,
    upper_ulcer_index::Real = Inf,
    # Optimisation model inputs
    mu::Vector{<:Real} = Array{Float64}(undef, 0),
    cov::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    kurt::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    skurt::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    L_2::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    S_2::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_f::Vector{<:Real} = Array{Float64}(undef, 0),
    cov_f::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_fm::Vector{<:Real} = Array{Float64}(undef, 0),
    cov_fm::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_bl::Vector{<:Real} = Array{Float64}(undef, 0),
    cov_bl::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    mu_bl_fm::Vector{<:Real} = Array{Float64}(undef, 0),
    cov_bl_fm::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    returns_fm::DataFrame = DataFrame(),
    z_evar::Real = -Inf,
    z_edar::Real = -Inf,
    z_rvar::Real = -Inf,
    z_rdar::Real = -Inf,
    # Inputs of Worst Case Optimization Models
    cov_l::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_u::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_mu::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    cov_sigma::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
    d_mu::Real = -Inf,
    k_mu::Real = -Inf,
    k_sigma::Real = -Inf,
    # Optimal portfolios
    p_optimal::DataFrame = DataFrame(),
    rp_optimal::DataFrame = DataFrame(),
    rrp_optimal::DataFrame = DataFrame(),
    wc_optimal::DataFrame = DataFrame(),
    limits::DataFrame = DataFrame(),
    frontier::DataFrame = DataFrame(),
    # Solver params
    solvers::Dict = Dict(),
    sol_params::Dict = Dict(),
    model = JuMP.Model(),
)
    return Portfolio(
        # Portfolio characteristics.
        returns,
        short,
        upper_short,
        upper_long,
        sum_short_long,
        min_number_effective_assets,
        max_number_assets,
        returns_factors,
        loadings,
        # Risk parameters
        alpha,
        a_sim,
        beta,
        b_sim,
        kappa,
        max_num_assets_kurt,
        # Benchmark constraints
        kind_tracking_err,
        allow_turnover,
        turnover,
        turnover_weights,
        allow_tracking_err,
        tracking_err,
        tracking_err_returns,
        tracking_err_weights,
        # Risk and return constraints
        a_mtx_ineq,
        b_vec_ineq,
        risk_budget_vec,
        lower_mu,
        upper_dev,
        upper_krt,
        upper_mean_abs_dev,
        upper_gini_mean_diff,
        upper_mean_semi_dev,
        upper_semi_krt,
        upper_first_lower_partial_moment,
        upper_second_lower_partial_moment,
        upper_conditional_value_at_risk,
        upper_tail_gini,
        upper_entropic_value_at_risk,
        upper_relativistic_value_at_risk,
        upper_worst_realisation,
        upper_range,
        upper_range_conditional_value_at_risk,
        upper_range_tail_gini,
        upper_max_drawdown,
        upper_average_drawdown,
        upper_conditional_drawdown_at_risk,
        upper_entropic_drawdown_at_risk,
        upper_relativistic_drawdown_at_risk,
        upper_ulcer_index,
        # Optimisation model inputs
        mu,
        cov,
        kurt,
        skurt,
        L_2,
        S_2,
        mu_f,
        cov_f,
        mu_fm,
        cov_fm,
        mu_bl,
        cov_bl,
        mu_bl_fm,
        cov_bl_fm,
        returns_fm,
        z_evar,
        z_edar,
        z_rvar,
        z_rdar,
        # Inputs of Worst Case Optimization Models
        cov_l,
        cov_u,
        cov_mu,
        cov_sigma,
        d_mu,
        k_mu,
        k_sigma,
        # Optimal portfolios
        p_optimal,
        rp_optimal,
        rrp_optimal,
        wc_optimal,
        limits,
        frontier,
        # Solver params
        solvers,
        sol_params,
        model,
        Dict(),
    )
end

function _mv_setup(model, sigma, rm, kelly, upper_dev, obj)
    w = model[:w]
    k = model[:k]
    if rm == :mean_var || kelly == :approx || isfinite(upper_dev)
        @variable(model, tdev >= 0)
        @expression(model, dev_risk, tdev * tdev)
        G = sqrt(sigma)
        @constraint(
            model,
            tdev_sqrt_sigma_soc,
            [tdev; transpose(G) * w] in SecondOrderCone()
        )

        if isfinite(upper_dev)
            if obj == :sharpe
                @constraint(model, tdev_leq_udev, tdev <= upper_dev * k)
            else
                @constraint(model, tdev_leq_udev, tdev <= upper_dev)
            end
        end

        if rm == :mean_var
            @expression(model, risk, dev_risk)
        end
    end

    return nothing
end

function _mad_setup(model, T, rm, upper_mean_abs_dev, upper_mean_semi_dev, returns, mu, obj)
    w = model[:w]
    k = model[:k]
    if rm == :mean_abs_dev ||
       rm == :mean_semi_dev ||
       isfinite(upper_mean_abs_dev) ||
       isfinite(upper_mean_semi_dev)
        @variable(model, tmad[1:T] >= 0)
        abs_dev = returns .- transpose(mu)
        @constraint(model, abs_dev_w_geq_neg_tmad, abs_dev * w >= -tmad)

        if rm == :mean_abs_dev || isfinite(upper_mean_abs_dev)
            @expression(model, mad, sum(tmad) / T)

            if isfinite(upper_mean_abs_dev)
                if obj == :sharpe
                    @constraint(
                        model,
                        mad_leq_umad_div_2,
                        mad * 2 <= upper_mean_abs_dev * k
                    )
                else
                    @constraint(model, mad_leq_umad_div_2, mad * 2 <= upper_mean_abs_dev)
                end
            end

            if rm == :mean_abs_dev
                @expression(model, risk, mad)
            end
        end

        if rm == :mean_semi_dev || isfinite(upper_mean_semi_dev)
            @variable(model, tmsd >= 0)
            @constraint(model, tmsd_tmad_soc, [tmsd; tmad] in SecondOrderCone())
            @expression(model, msd, tmsd / sqrt(T - 1))

            if isfinite(upper_mean_semi_dev)
                if obj == :sharpe
                    @constraint(model, msd_leq_umsd, msd <= upper_mean_semi_dev * k)
                else
                    @constraint(model, msd_leq_umsd, msd <= upper_mean_semi_dev)
                end
            end

            if rm == :mean_semi_dev
                @expression(model, risk, msd)
            end
        end
    end

    return nothing
end

function _return_setup(model, class, kelly, obj, T, rf, returns, mu, lower_mu)
    w = model[:w]
    k = model[:k]
    risk = model[:risk]

    if class == :classic && (kelly == :exact || kelly == :approx)
        if kelly == :exact
            @variable(model, texact_kelly[1:T])
            if obj == :sharpe
                @expression(model, ret, sum(texact_kelly) / T - rf * k)
                @expression(model, kret, k .+ returns * w)
                @constraint(
                    model,
                    texact_kelly_k_ec[i = 1:T],
                    [texact_kelly[i], k, kret[i]] in MOI.ExponentialCone()
                )
                @constraint(model, sharpe, risk <= 1)
            else
                @expression(model, ret, sum(texact_kelly) / T)
                @expression(model, kret, 1 .+ returns * w)
                @constraint(
                    model,
                    texact_kelly_k_ec[i = 1:T],
                    [texact_kelly[i], 1, kret[i]] in MOI.ExponentialCone()
                )
            end
        elseif kelly == :approx
            tdev = model[:tdev]
            dev_risk = model[:dev_risk]
            if obj == :sharpe
                @variable(model, tapprox_kelly >= 0)
                @constraint(
                    model,
                    quad_over_lin_tdev_k,
                    [k + tapprox_kelly; 2 * tdev + k - tapprox_kelly] in SecondOrderCone()
                )
                @expression(model, ret, dot(mu, w) - 0.5 * tapprox_kelly)
                @constraint(model, sharpe, risk <= 1)
            else
                @expression(model, ret, dot(mu, w) - 0.5 * dev_risk)
            end
        end
    else
        @expression(model, ret, dot(mu, w))
        if obj == :sharpe
            @constraint(model, sharpe, ret - rf * k == 1)
        end
    end

    # Return constraints.
    if isfinite(lower_mu)
        if obj == :sharpe
            @constraint(model, ret_geq_lmu, ret >= lower_mu * k)
        else
            @constraint(model, ret_geq_lmu, ret >= lower_mu)
        end
    end

    return nothing
end

function _setup_weights(
    model,
    max_number_assets,
    obj,
    N,
    short,
    upper_short,
    upper_long,
    sum_short_long,
)
    w = model[:w]
    k = model[:k]

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
        @constraint(model, sum_w_eq_sum_ls, sum(w) == sum_short_long * k)

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum_tass_bin_leq_mna, sum(tass_bin) <= max_number_assets)
            @constraint(model, tass_bin_sharpe_leq_k, tass_bin_sharpe .<= k)
            @constraint(
                model,
                tass_bin_sharpe_leq_tass_bin,
                tass_bin_sharpe .<= 100000 * tass_bin
            )
            @constraint(
                model,
                tass_bin_sharpe_geq_o_m_tass_bin,
                tass_bin_sharpe .>= k - 100000 * (1 .- tass_bin)
            )
            @constraint(model, w_leq_ul, w .<= upper_long * tass_bin_sharpe)
        end

        if short == false
            @constraint(model, w_leq_ul, w .<= upper_long * k)
            @constraint(model, w_geq_0, w .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum_ulong, sum(tw_ulong) <= upper_long * k)
            @constraint(model, sum_ushort, sum(tw_ushort) <= upper_short * k)

            @constraint(model, w_leq_tw_ulong, w .- tw_ulong .<= 0)
            @constraint(model, w_geq_neg_tw_ushort, w .+ tw_ushort .>= 0)

            # Maximum number of assets constraints.
            if max_number_assets > 0
                @constraint(
                    model,
                    w_geq_neg_us_tass_bin,
                    w .>= -upper_short * tass_bin_sharpe
                )
            end
        end
    else
        @constraint(model, sum_w_eq_sum_ls, sum(w) == sum_short_long)

        # Maximum number of assets constraints.
        if max_number_assets > 0
            @constraint(model, sum_tass_bin_leq_mna, sum(tass_bin) <= max_number_assets)
            @constraint(model, wgule, w .<= upper_long * tass_bin)
        end

        if short == false
            @constraint(model, w_leq_ul, w .<= upper_long)
            @constraint(model, w_geq_0, w .>= 0)
        else
            @variable(model, tw_ulong[1:N] .>= 0)
            @variable(model, tw_ushort[1:N] .>= 0)

            @constraint(model, sum_ulong, sum(tw_ulong) <= upper_long)
            @constraint(model, sum_ushort, sum(tw_ushort) <= upper_short)

            @constraint(model, w_leq_tw_ulong, w .- tw_ulong .<= 0)
            @constraint(model, w_geq_neg_tw_ushort, w .+ tw_ushort .>= 0)

            # Maximum number of assets constraints.
            if max_number_assets > 0
                @constraint(model, w_geq_neg_us_tass_bin, w .>= -upper_short * tass_bin)
            end
        end
    end

    return nothing
end

function _setup_linear_constraints(model, A, B, obj)
    w = model[:w]
    k = model[:k]
    # Linear weight constraints.
    if !isempty(A) && !isempty(B)
        if obj == :sharpe
            @constraint(model, aw_geq_b, A * w .- B * k .>= 0)
        else
            @constraint(model, aw_geq_b, A * w .- B .>= 0)
        end
    end

    return nothing
end

function _setup_min_number_effective_assets(model, mnea)
    w = model[:w]
    k = model[:k]

    if mnea > 0
        @variable(model, tmnea >= 0)
        @constraint(model, tmnea_w_soc, [tmnea; w] in SecondOrderCone())
        if obj == :sharpe
            @constraint(model, tmnea_sqrt_mnea_leq_1, tmnea * sqrt(mnea) <= k)
        else
            @constraint(model, tmnea_sqrt_mnea_leq_1, tmnea * sqrt(mnea) <= 1)
        end
    end

    return nothing
end

function _setup_tracking_err(
    model,
    allow_tracking_err,
    kind_tracking_err,
    tracking_err_weights,
    returns,
    tracking_err_returns,
    obj,
    upper_tracking_err,
    T,
)
    w = model[:w]
    k = model[:k]
    if allow_tracking_err == true
        tracking_err_flag = false

        if kind_tracking_err == :weights && !isempty(tracking_err_weights)
            benchmark = returns * tracking_err_weights[!, :weights]
            tracking_err_flag = true
        elseif kind_tracking_err == :returns && !isempty(tracking_err_returns)
            benchmark = tracking_err_returns[!, :returns]
            tracking_err_flag = true
        end

        if tracking_err_flag == true
            @variable(model, t_tracking_err >= 0)
            if obj == :sharpe
                @expression(model, tracking_err, returns * w .- benchmark * k)
                @constraint(
                    model,
                    t_track_err_track_err_soc,
                    [t_tracking_err; tracking_err] in SecondOrderCone()
                )
                @constraint(
                    model,
                    t_track_err_leq_track_err_sqrt_tm1,
                    t_tracking_err <= upper_tracking_err * k * sqrt(T - 1)
                )
            else
                @expression(model, tracking_err, returns * w .- benchmark)
                @constraint(
                    model,
                    t_track_err_track_err_soc,
                    [t_tracking_err; tracking_err] in SecondOrderCone()
                )
                @constraint(
                    model,
                    t_track_err_leq_track_err_sqrt_tm1,
                    t_tracking_err <= upper_tracking_err * sqrt(T - 1)
                )
            end
        end
    end
end

function _setup_turnover(model, allow_turnover, turnover_weights, N, upper_turnover, obj)
    w = model[:w]
    k = model[:k]
    if allow_turnover == true && !isempty(turnover_weights)
        @variable(model, t_turnover[1:N] >= 0)
        if obj == :sharpe
            @expression(model, turnover, w .- turnover_weights[!, :weights] * k)
            @constraint(
                model,
                t_turnover_turnover_soc[i = 1:N],
                [t_turnover[i]; turnover[i]] in MOI.NormOneCone(2)
            )
            @constraint(model, t_turnover_leq_turnover, t_turnover .<= upper_turnover * k)
        else
            @expression(model, turnover, w .- turnover_weights[!, :weights])
            @constraint(
                model,
                t_turnover_turnover_soc[i = 1:N],
                [t_turnover[i]; turnover[i]] in MOI.NormOneCone(2)
            )
            @constraint(model, t_turnover_leq_turnover, t_turnover .<= upper_turnover)
        end
    end
end

function _setup_objective_function(model, obj, kelly)
    ret = model[:ret]
    risk = model[:risk]
    if obj == :sharpe
        if model == :classic && (kelly == :exact || kelly == :approx)
            @objective(model, Max, ret)
        else
            @objective(model, Min, risk)
        end
    elseif obj == :min_risk
        @objective(model, Min, risk)
    elseif obj == :utility
        @objective(model, Max, ret - l * risk)
    elseif obj == :max_ret
        @objective(model, Max, ret)
    end
end

function optimize(
    portfolio::Portfolio;
    class::Symbol = :classic,
    rm::Symbol = :mean_var,
    obj::Symbol = :sharpe,
    kelly::Symbol = :none,
    rf::Real = 1.0329^(1 / 252) - 1,
    l::Real = 2.0,
)
    @assert(class ∈ PortClasses)
    @assert(rm ∈ RiskMeasures)
    @assert(obj ∈ ObjFuncs)
    @assert(kelly ∈ KellyRet)
    @assert(portfolio.kind_tracking_err ∈ TrackingErrKinds)

    portfolio.model = JuMP.Model()
    term_status = termination_status(portfolio.model)

    mu = portfolio.mu
    sigma = portfolio.cov
    returns = Matrix(portfolio.returns[!, 2:end])
    T, N = size(returns)

    model = portfolio.model

    # Model variables.
    @variable(model, w[1:N])
    @variable(model, k >= 0)

    # Risk Variables.

    # Mean variance.
    upper_dev = portfolio.upper_dev
    _mv_setup(model, sigma, rm, kelly, upper_dev, obj)

    # Mean absolute deviation and mean semi deviation.
    upper_mean_abs_dev = portfolio.upper_mean_abs_dev
    upper_mean_semi_dev = portfolio.upper_mean_semi_dev
    _mad_setup(model, T, rm, upper_mean_abs_dev, upper_mean_semi_dev, returns, mu, obj)

    ###############################
    # Return variables.
    lower_mu = portfolio.lower_mu
    _return_setup(model, class, kelly, obj, T, rf, returns, mu, lower_mu)

    # Weight constraints.
    max_number_assets = portfolio.max_number_assets
    short = portfolio.short
    upper_short = portfolio.upper_short
    upper_long = portfolio.upper_long
    sum_short_long = portfolio.sum_short_long
    _setup_weights(
        model,
        max_number_assets,
        obj,
        N,
        short,
        upper_short,
        upper_long,
        sum_short_long,
    )

    # Linear weight constraints.
    A = portfolio.a_mtx_ineq
    B = portfolio.b_vec_ineq
    _setup_linear_constraints(model, A, B, obj)

    # Minimum number of effective assets.
    mnea = portfolio.min_number_effective_assets
    _setup_min_number_effective_assets(model, mnea)

    # Tracking error variables and constraints.
    allow_tracking_err = portfolio.allow_tracking_err
    kind_tracking_err = portfolio.kind_tracking_err
    tracking_err_weights = portfolio.tracking_err_weights
    tracking_err_returns = portfolio.tracking_err_returns
    upper_tracking_err = portfolio.tracking_err
    _setup_tracking_err(
        model,
        allow_tracking_err,
        kind_tracking_err,
        tracking_err_weights,
        returns,
        tracking_err_returns,
        obj,
        upper_tracking_err,
        T,
    )

    # Turnover variables and constraints
    allow_turnover = portfolio.allow_turnover
    turnover_weights = portfolio.turnover_weights
    upper_turnover = portfolio.turnover
    _setup_turnover(model, allow_turnover, turnover_weights, N, upper_turnover, obj)

    # Objective functions.
    _setup_objective_function(model, obj, kelly)

    solvers = portfolio.solvers
    sol_params = portfolio.sol_params
    solvers_tried = Dict()
    for (solver_name, solver) in solvers
        set_optimizer(model, solver)
        if haskey(sol_params, solver_name)
            for (attribute, value) in sol_params[solver_name]
                set_attribute(model, attribute, value)
            end
        end
        try
            optimize!(model)
        catch jump_error
            push!(solvers_tried, solver_name => Dict("error" => jump_error))
            continue
        end
        term_status = termination_status(model)

        if term_status in ValidTermination &&
           all(isfinite.(value.(w))) &&
           all(abs.(0.0 .- value.(w)) .> N^2 * eps())
            break
        end
        push!(
            solvers_tried,
            solver_name => Dict(
                "objective_val" => objective_value(model),
                "term_status" => term_status,
                "sol_params" =>
                    haskey(sol_params, solver_name) ? sol_params[solver_name] : missing,
            ),
        )
    end

    if term_status ∉ ValidTermination || any(.!isfinite.(value.(w)))
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.optimize))"

        @warn(
            "$funcname: model for could not be optimised satisfactorily.\nPortfolio: $class\nRisk measure: $rm\nKelly return: $kelly\nObjective: $obj\nSolvers: $solvers_tried"
        )

        portfolio.p_optimal = DataFrame()
        portfolio.fail = solvers_tried

        return portfolio.p_optimal
    end

    weights = Vector{eltype(returns)}(undef, N)
    if obj == :sharpe
        weights .= value.(w) / value(k)
    else
        weights .= value.(w)
    end

    if short == false
        weights .= abs.(weights) / sum(abs.(weights)) * sum_short_long
    end

    portfolio.p_optimal =
        DataFrame(tickers = names(portfolio.returns)[2:end], weights = weights)

    if isempty(solvers_tried)
        portfolio.fail = Dict()
    else
        portfolio.fail = solvers_tried
    end

    return portfolio.p_optimal
end

export AbstractPortfolio, Portfolio, optimize#, HCPortfolio, OWAPortfolio