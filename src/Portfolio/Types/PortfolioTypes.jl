abstract type AbstractPortfolio end
"""
```
mutable struct Portfolio{ast, dat, r, s, us, ul, nal, nau, naus, tfa, tfdat, tretf, l, lo,
                         mnak, mnaks, rb, to, kte, blbw, ami, bvi, rbv, frbv, nm, amc, bvc,
                         ler, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew, tv, tsskew, tsv,
                         tmuf, tcovf, trfm, tmufm, tcovfm, tmubl, tcovbl, tmublf, tcovblf,
                         tcovl, tcovu, tcovmu, tcovs, tdmu, tkmu, tks, topt, tz, tlim,
                         tfront, tsolv, tf, tmod, tlp, taopt, talo, tasolv, taf, tamod} <:
               AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    short::s
    short_u::us
    long_u::ul
    num_assets_l::nal
    num_assets_u::nau
    num_assets_u_scale::naus
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    loadings_opt::lo
    max_num_assets_kurt::mnak
    max_num_assets_kurt_scale::mnaks
    rebalance::rb
    turnover::to
    tracking_err::kte
    bl_bench_weights::blbw
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget::rbv
    f_risk_budget::frbv
    network_method::nm
    a_vec_cent::amc
    b_cent::bvc
    mu_l::ler
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    skew::tskew
    V::tv
    sskew::tsskew
    SV::tsv
    f_mu::tmuf
    f_cov::tcovf
    fm_returns::trfm
    fm_mu::tmufm
    fm_cov::tcovfm
    bl_mu::tmubl
    bl_cov::tcovbl
    blfm_mu::tmublf
    blfm_cov::tcovblf
    cov_l::tcovl
    cov_u::tcovu
    cov_mu::tcovmu
    cov_sigma::tcovs
    d_mu::tdmu
    k_mu::tkmu
    k_sigma::tks
    optimal::topt
    z::tz
    limits::tlim
    frontier::tfront
    solvers::tsolv
    fail::tf
    model::tmod
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_fail::taf
    alloc_model::tamod
end
```
"""
mutable struct Portfolio{ast, dat, r, s, us, ul, nal, nau, naus, tfa, tfdat, tretf, l, lo,
                         mnak, mnaks, rb, to, kte, blbw, ami, bvi, rbv, frbv, nm, amc, bvc,
                         ler, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew, tv, tsskew, tsv,
                         tmuf, tcovf, trfm, tmufm, tcovfm, tmubl, tcovbl, tmublf, tcovblf,
                         tcovl, tcovu, tcovmu, tcovs, tdmu, tkmu, tks, topt, tz, tlim,
                         tfront, tsolv, tf, tmod, tlp, taopt, talo, tasolv, taf, tamod} <:
               AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    short::s
    short_u::us
    long_u::ul
    num_assets_l::nal
    num_assets_u::nau
    num_assets_u_scale::naus
    f_assets::tfa
    f_timestamps::tfdat
    f_returns::tretf
    loadings::l
    loadings_opt::lo
    max_num_assets_kurt::mnak
    max_num_assets_kurt_scale::mnaks
    rebalance::rb
    turnover::to
    tracking_err::kte
    bl_bench_weights::blbw
    a_mtx_ineq::ami
    b_vec_ineq::bvi
    risk_budget::rbv
    f_risk_budget::frbv
    network_method::nm
    a_vec_cent::amc
    b_cent::bvc
    mu_l::ler
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    skew::tskew
    V::tv
    sskew::tsskew
    SV::tsv
    f_mu::tmuf
    f_cov::tcovf
    fm_returns::trfm
    fm_mu::tmufm
    fm_cov::tcovfm
    bl_mu::tmubl
    bl_cov::tcovbl
    blfm_mu::tmublf
    blfm_cov::tcovblf
    cov_l::tcovl
    cov_u::tcovu
    cov_mu::tcovmu
    cov_sigma::tcovs
    d_mu::tdmu
    k_mu::tkmu
    k_sigma::tks
    optimal::topt
    z::tz
    limits::tlim
    frontier::tfront
    solvers::tsolv
    fail::tf
    model::tmod
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_fail::taf
    alloc_model::tamod
end
function Portfolio(; prices::TimeArray = TimeArray(TimeType[], []),
                   returns::DataFrame = DataFrame(),
                   ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   timestamps::AbstractVector = Vector{Date}(undef, 0),
                   assets::AbstractVector = Vector{String}(undef, 0), short::Bool = false,
                   short_u::Real = 0.2, long_u::Real = 1.0, num_assets_l::Integer = 0,
                   num_assets_u::Integer = 0, num_assets_u_scale::Real = 100_000.0,
                   f_prices::TimeArray = TimeArray(TimeType[], []),
                   f_returns::DataFrame = DataFrame(),
                   f_ret::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   f_timestamps::AbstractVector = Vector{Date}(undef, 0),
                   f_assets::AbstractVector = Vector{String}(undef, 0),
                   loadings::DataFrame = DataFrame(),
                   loadings_opt::Union{<:RegressionType, Nothing} = nothing,
                   max_num_assets_kurt::Integer = 0, max_num_assets_kurt_scale::Integer = 2,
                   rebalance::AbstractTR = NoTR(), turnover::AbstractTR = NoTR(),
                   tracking_err::TrackingErr = NoTracking(),
                   bl_bench_weights::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   a_mtx_ineq::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   b_vec_ineq::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   f_risk_budget::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   network_method::NetworkMethods = NoNtwk(),
                   a_vec_cent::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   b_cent::Real = Inf, mu_l::Real = Inf,
                   mu::AbstractVector = Vector{Float64}(undef, 0),
                   cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   skew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   V::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   sskew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   SV::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   f_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   f_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   fm_returns::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   fm_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   fm_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   bl_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   bl_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   blfm_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   blfm_cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   cov_l::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   cov_u::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   cov_mu::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   cov_sigma::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                   d_mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   k_mu::Real = Inf, k_sigma::Real = Inf, optimal::AbstractDict = Dict(),
                   z::AbstractDict = Dict(), limits::AbstractDict = Dict(),
                   frontier::AbstractDict = Dict(), solvers::AbstractDict = Dict(),
                   fail::AbstractDict = Dict(), model::JuMP.Model = JuMP.Model(),
                   latest_prices::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                   alloc_optimal::AbstractDict = Dict(),
                   alloc_leftover::AbstractDict = Dict(),
                   alloc_solvers::AbstractDict = Dict(), alloc_fail::AbstractDict = Dict(),
                   alloc_model::JuMP.Model = JuMP.Model())
    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices)))
        latest_prices = Vector(dropmissing!(DataFrame(prices))[end, colnames(prices)])
    end

    if !isempty(returns)
        assets = setdiff(names(returns), ("timestamp",))
        timestamps = returns[!, "timestamp"]
        returns = Matrix(returns[!, assets])
    else
        @smart_assert(length(assets) == size(ret, 2))
        returns = ret
    end

    @smart_assert(short_u >= zero(short_u))
    @smart_assert(long_u >= zero(long_u))

    if !isempty(f_prices)
        f_returns = dropmissing!(DataFrame(percentchange(f_prices)))
    end

    if !isempty(f_returns)
        f_assets = setdiff(names(f_returns), ("timestamp",))
        f_timestamps = f_returns[!, "timestamp"]
        f_returns = Matrix(f_returns[!, f_assets])
    else
        @smart_assert(length(f_assets) == size(f_ret, 2))
        f_returns = f_ret
    end

    @smart_assert(num_assets_l >= zero(num_assets_l))
    @smart_assert(num_assets_u >= zero(num_assets_u))
    @smart_assert(num_assets_u_scale >= zero(num_assets_u_scale))
    @smart_assert(max_num_assets_kurt >= zero(max_num_assets_kurt))
    max_num_assets_kurt_scale = clamp(max_num_assets_kurt_scale, 1, size(returns, 2))
    if isa(rebalance, TR)
        if isa(rebalance.val, Real)
            @smart_assert(rebalance.val >= zero(rebalance.val))
        elseif isa(rebalance.val, AbstractVector) && !isempty(rebalance.val)
            @smart_assert(length(rebalance.val) == size(returns, 2) &&
                          all(rebalance.val .>= zero(rebalance.val)))
        end
        if !isempty(rebalance.w)
            @smart_assert(length(rebalance.w) == size(returns, 2))
        end
    end
    if isa(turnover, TR)
        if isa(turnover.val, Real)
            @smart_assert(turnover.val >= zero(turnover.val))
        elseif isa(turnover.val, AbstractVector) && !isempty(turnover.val)
            @smart_assert(length(turnover.val) == size(returns, 2) &&
                          all(turnover.val .>= zero(turnover.val)))
        end
        if !isempty(turnover.w)
            @smart_assert(length(turnover.w) == size(returns, 2))
        end
    end
    if isa(tracking_err, TrackWeight)
        @smart_assert(length(tracking_err.w) == size(returns, 2))
        @smart_assert(length(tracking_err.err) >= zero(tracking_err.err))
    end
    if isa(tracking_err, TrackRet)
        @smart_assert(length(tracking_err.w) == size(returns, 1))
        @smart_assert(length(tracking_err.err) >= zero(tracking_err.err))
    end
    if !isempty(bl_bench_weights)
        @smart_assert(length(bl_bench_weights) == size(returns, 2))
    end
    if !isa(network_method, NoNtwk) && !isempty(network_method.A)
        @smart_assert(size(network_method.A) == (size(returns, 2), size(returns, 2)))
    end
    if !isempty(a_vec_cent)
        @smart_assert(size(a_vec_cent, 1) == size(returns, 2))
    end
    if !isempty(a_mtx_ineq)
        @smart_assert(size(a_mtx_ineq, 2) == size(returns, 2))
    end
    if !isempty(risk_budget)
        @smart_assert(length(risk_budget) == size(returns, 2))
        @smart_assert(all(risk_budget .>= zero(eltype(returns))))

        if isa(risk_budget, AbstractRange)
            risk_budget = collect(risk_budget / sum(risk_budget))
        else
            risk_budget ./= sum(risk_budget)
        end
    end
    if !isempty(f_risk_budget)
        @smart_assert(all(f_risk_budget .>= zero(eltype(returns))))
        if isa(f_risk_budget, AbstractRange)
            f_risk_budget = collect(f_risk_budget / sum(f_risk_budget))
        else
            f_risk_budget ./= sum(f_risk_budget)
        end
    end
    if !isempty(mu)
        @smart_assert(length(mu) == size(returns, 2))
    end
    if !isempty(cov)
        @smart_assert(size(cov, 1) == size(cov, 2) == size(returns, 2))
    end
    if !isempty(kurt)
        @smart_assert(size(kurt, 1) == size(kurt, 2) == size(returns, 2)^2)
    end
    if !isempty(skurt)
        @smart_assert(size(skurt, 1) == size(skurt, 2) == size(returns, 2)^2)
    end
    if !isempty(skew)
        @smart_assert(size(skew, 1) == size(returns, 2) &&
                      size(skew, 2) == size(returns, 2)^2)
    end
    if !isempty(V)
        @smart_assert(size(V, 1) == size(V, 2) == size(returns, 2))
    end
    if !isempty(sskew)
        @smart_assert(size(sskew, 1) == size(returns, 2) &&
                      size(sskew, 2) == size(returns, 2)^2)
    end
    if !isempty(SV)
        @smart_assert(size(SV, 1) == size(SV, 2) == size(returns, 2))
    end
    if !isempty(f_mu)
        @smart_assert(length(f_mu) == size(f_returns, 2))
    end
    if !isempty(f_cov)
        @smart_assert(size(f_cov, 1) == size(f_cov, 2) == size(f_returns, 2))
    end
    if !isempty(fm_mu)
        @smart_assert(length(fm_mu) == size(returns, 2))
    end
    if !isempty(fm_cov)
        @smart_assert(size(fm_cov, 1) == size(fm_cov, 2) == size(returns, 2))
    end
    if !isempty(bl_mu)
        @smart_assert(length(bl_mu) == size(returns, 2))
    end
    if !isempty(bl_cov)
        @smart_assert(size(bl_cov, 1) == size(bl_cov, 2) == size(returns, 2))
    end
    if !isempty(blfm_mu)
        @smart_assert(length(blfm_mu) == size(returns, 2))
    end
    if !isempty(blfm_cov)
        @smart_assert(size(blfm_cov, 1) == size(blfm_cov, 2) == size(returns, 2))
    end
    if !isempty(cov_l)
        @smart_assert(size(cov_l, 1) == size(cov_l, 2) == size(returns, 2))
    end
    if !isempty(cov_u)
        @smart_assert(size(cov_u, 1) == size(cov_u, 2) == size(returns, 2))
    end
    if !isempty(cov_mu)
        @smart_assert(size(cov_mu, 1) == size(cov_mu, 2) == size(returns, 2))
    end
    if !isempty(cov_sigma)
        @smart_assert(size(cov_sigma, 1) == size(cov_sigma, 2) == size(returns, 2)^2)
    end
    if !isempty(d_mu)
        @smart_assert(length(d_mu) == size(returns, 2))
    end
    if !isempty(latest_prices)
        @smart_assert(length(latest_prices) == size(returns, 2))
    end

    L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)
    S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)

    return Portfolio{typeof(assets), typeof(timestamps), typeof(returns), typeof(short),
                     typeof(short_u), typeof(long_u), typeof(num_assets_l),
                     typeof(num_assets_u), typeof(num_assets_u_scale), typeof(f_assets),
                     typeof(f_timestamps), typeof(f_returns), typeof(loadings),
                     Union{<:RegressionType, Nothing}, typeof(max_num_assets_kurt),
                     typeof(max_num_assets_kurt_scale), AbstractTR, AbstractTR, TrackingErr,
                     typeof(bl_bench_weights), typeof(a_mtx_ineq), typeof(b_vec_ineq),
                     typeof(risk_budget), typeof(f_risk_budget), NetworkMethods,
                     typeof(a_vec_cent), typeof(b_cent), typeof(mu_l), typeof(mu),
                     typeof(cov), typeof(kurt), typeof(skurt), typeof(L_2), typeof(S_2),
                     typeof(skew), typeof(V), typeof(sskew), typeof(SV), typeof(f_mu),
                     typeof(f_cov), typeof(fm_returns), typeof(fm_mu), typeof(fm_cov),
                     typeof(bl_mu), typeof(bl_cov), typeof(blfm_mu), typeof(blfm_cov),
                     typeof(cov_l), typeof(cov_u), typeof(cov_mu), typeof(cov_sigma),
                     typeof(d_mu), typeof(k_mu), typeof(k_sigma), typeof(optimal),
                     typeof(z), typeof(limits), typeof(frontier), typeof(solvers),
                     typeof(fail), typeof(model), typeof(latest_prices),
                     typeof(alloc_optimal), typeof(alloc_leftover), typeof(alloc_solvers),
                     typeof(alloc_fail), typeof(alloc_model)}(assets, timestamps, returns,
                                                              short, short_u, long_u,
                                                              num_assets_l, num_assets_u,
                                                              num_assets_u_scale, f_assets,
                                                              f_timestamps, f_returns,
                                                              loadings, loadings_opt,
                                                              max_num_assets_kurt,
                                                              max_num_assets_kurt_scale,
                                                              rebalance, turnover,
                                                              tracking_err,
                                                              bl_bench_weights, a_mtx_ineq,
                                                              b_vec_ineq, risk_budget,
                                                              f_risk_budget, network_method,
                                                              a_vec_cent, b_cent, mu_l, mu,
                                                              cov, kurt, skurt, L_2, S_2,
                                                              skew, V, sskew, SV, f_mu,
                                                              f_cov, fm_returns, fm_mu,
                                                              fm_cov, bl_mu, bl_cov,
                                                              blfm_mu, blfm_cov, cov_l,
                                                              cov_u, cov_mu, cov_sigma,
                                                              d_mu, k_mu, k_sigma, optimal,
                                                              z, limits, frontier, solvers,
                                                              fail, model, latest_prices,
                                                              alloc_optimal, alloc_leftover,
                                                              alloc_solvers, alloc_fail,
                                                              alloc_model)
end
function Base.getproperty(obj::Portfolio, sym::Symbol)
    if sym == :budget
        obj.short ? obj.long_u - obj.short_u : obj.long_u
    else
        getfield(obj, sym)
    end
end
function Base.setproperty!(obj::Portfolio, sym::Symbol, val)
    if sym == :short_u
        @smart_assert(val >= zero(val))
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :long_u
        @smart_assert(val >= zero(val))
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :max_num_assets_kurt
        @smart_assert(val >= zero(val))
    elseif sym == :max_num_assets_kurt_scale
        val = clamp(val, 1, size(obj.returns, 2))
    elseif sym ∈ (:rebalance, :turnover)
        if isa(val, TR)
            if isa(val.val, Real)
                @smart_assert(val.val >= zero(val.val))
            elseif isa(val.val, AbstractVector) && !isempty(val.val)
                @smart_assert(length(val.val) == size(obj.returns, 2) &&
                              all(val.val .>= zero(val.val)))
            end
            if !isempty(val.w)
                @smart_assert(length(val.w) == size(obj.returns, 2))
            end
        end
    elseif sym == :tracking_err
        if isa(val, TrackWeight)
            @smart_assert(length(val.w) == size(obj.returns, 2))
            @smart_assert(val.err >= zero(val.err))
        elseif isa(val, TrackRet)
            @smart_assert(length(val.w) == size(obj.returns, 1))
            @smart_assert(val.err >= zero(val.err))
        end
    elseif sym == :a_mtx_ineq
        if !isempty(val)
            @smart_assert(size(val, 2) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :network_method
        if !isa(val, NoNtwk) && !isempty(val.A)
            @smart_assert(size(val.A) == (size(obj.returns, 2), size(obj.returns, 2)))
        end
    elseif sym == :a_vec_cent
        if !isempty(val)
            @smart_assert(size(val, 1) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :f_mu
        if !isempty(val)
            @smart_assert(length(val) == size(obj.f_returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :f_cov
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.f_returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:L_2, :S_2)
        if !isempty(val)
            N = size(obj.returns, 2)
            @smart_assert(size(val) == (Int(N * (N + 1) / 2), N^2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:risk_budget, :bl_bench_weights)
        if isempty(val)
            N = size(obj.returns, 2)
            val = fill(inv(N), N)
        else
            @smart_assert(length(val) == size(obj.returns, 2))
            if sym == :risk_budget
                @smart_assert(all(val .>= zero(eltype(obj.returns))))
            end
            if isa(val, AbstractRange)
                val = collect(val / sum(val))
            else
                val ./= sum(val)
            end
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym == :f_risk_budget
        if !isempty(val)
            if isa(val, AbstractRange)
                val = collect(val / sum(val))
            else
                val ./= sum(val)
            end
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:num_assets_l, :num_assets_u, :num_assets_u_scale)
        @smart_assert(val >= zero(val))
    elseif sym ∈ (:kurt, :skurt, :cov_sigma)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:skew, :sskew)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(obj.returns, 2) &&
                          size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:assets, :timestamps, :returns, :f_assets, :f_timestamps, :f_returns)
        if !isempty(val) && !isempty(getfield(obj, sym))
            @smart_assert(size(val) == size(getfield(obj, sym)))
        end
    elseif sym ∈ (:mu, :fm_mu, :bl_mu, :blfm_mu, :d_mu, :latest_prices)
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:cov, :fm_cov, :bl_cov, :blfm_cov, :cov_l, :cov_u, :cov_mu, :V, :SV)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    else
        if (isa(getfield(obj, sym), AbstractArray) && isa(val, AbstractArray)) ||
           (isa(getfield(obj, sym), Real) && isa(val, Real))
            val = convert(typeof(getfield(obj, sym)), val)
        end
    end
    return setfield!(obj, sym, val)
end
function Base.deepcopy(obj::Portfolio)
    return Portfolio{typeof(obj.assets), typeof(obj.timestamps), typeof(obj.returns),
                     typeof(obj.short), typeof(obj.short_u), typeof(obj.long_u),
                     typeof(obj.num_assets_l), typeof(obj.num_assets_u),
                     typeof(obj.num_assets_u_scale), typeof(obj.f_assets),
                     typeof(obj.f_timestamps), typeof(obj.f_returns), typeof(obj.loadings),
                     Union{<:RegressionType, Nothing}, typeof(obj.max_num_assets_kurt),
                     typeof(obj.max_num_assets_kurt_scale), AbstractTR, AbstractTR,
                     TrackingErr, typeof(obj.bl_bench_weights), typeof(obj.a_mtx_ineq),
                     typeof(obj.b_vec_ineq), typeof(obj.risk_budget),
                     typeof(obj.f_risk_budget), NetworkMethods, typeof(obj.a_vec_cent),
                     typeof(obj.b_cent), typeof(obj.mu_l), typeof(obj.mu), typeof(obj.cov),
                     typeof(obj.kurt), typeof(obj.skurt), typeof(obj.L_2), typeof(obj.S_2),
                     typeof(obj.skew), typeof(obj.V), typeof(obj.sskew), typeof(obj.SV),
                     typeof(obj.f_mu), typeof(obj.f_cov), typeof(obj.fm_returns),
                     typeof(obj.fm_mu), typeof(obj.fm_cov), typeof(obj.bl_mu),
                     typeof(obj.bl_cov), typeof(obj.blfm_mu), typeof(obj.blfm_cov),
                     typeof(obj.cov_l), typeof(obj.cov_u), typeof(obj.cov_mu),
                     typeof(obj.cov_sigma), typeof(obj.d_mu), typeof(obj.k_mu),
                     typeof(obj.k_sigma), typeof(obj.optimal), typeof(obj.z),
                     typeof(obj.limits), typeof(obj.frontier), typeof(obj.solvers),
                     typeof(obj.fail), typeof(obj.model), typeof(obj.latest_prices),
                     typeof(obj.alloc_optimal), typeof(obj.alloc_leftover),
                     typeof(obj.alloc_solvers), typeof(obj.alloc_fail),
                     typeof(obj.alloc_model)}(deepcopy(obj.assets),
                                              deepcopy(obj.timestamps),
                                              deepcopy(obj.returns), deepcopy(obj.short),
                                              deepcopy(obj.short_u), deepcopy(obj.long_u),
                                              deepcopy(obj.num_assets_l),
                                              deepcopy(obj.num_assets_u),
                                              deepcopy(obj.num_assets_u_scale),
                                              deepcopy(obj.f_assets),
                                              deepcopy(obj.f_timestamps),
                                              deepcopy(obj.f_returns),
                                              deepcopy(obj.loadings),
                                              deepcopy(obj.loadings_opt),
                                              deepcopy(obj.max_num_assets_kurt),
                                              deepcopy(obj.max_num_assets_kurt_scale),
                                              deepcopy(obj.rebalance),
                                              deepcopy(obj.turnover),
                                              deepcopy(obj.tracking_err),
                                              deepcopy(obj.bl_bench_weights),
                                              deepcopy(obj.a_mtx_ineq),
                                              deepcopy(obj.b_vec_ineq),
                                              deepcopy(obj.risk_budget),
                                              deepcopy(obj.f_risk_budget),
                                              deepcopy(obj.network_method),
                                              deepcopy(obj.a_vec_cent),
                                              deepcopy(obj.b_cent), deepcopy(obj.mu_l),
                                              deepcopy(obj.mu), deepcopy(obj.cov),
                                              deepcopy(obj.kurt), deepcopy(obj.skurt),
                                              deepcopy(obj.L_2), deepcopy(obj.S_2),
                                              deepcopy(obj.skew), deepcopy(obj.V),
                                              deepcopy(obj.sskew), deepcopy(obj.SV),
                                              deepcopy(obj.f_mu), deepcopy(obj.f_cov),
                                              deepcopy(obj.fm_returns), deepcopy(obj.fm_mu),
                                              deepcopy(obj.fm_cov), deepcopy(obj.bl_mu),
                                              deepcopy(obj.bl_cov), deepcopy(obj.blfm_mu),
                                              deepcopy(obj.blfm_cov), deepcopy(obj.cov_l),
                                              deepcopy(obj.cov_u), deepcopy(obj.cov_mu),
                                              deepcopy(obj.cov_sigma), deepcopy(obj.d_mu),
                                              deepcopy(obj.k_mu), deepcopy(obj.k_sigma),
                                              deepcopy(obj.optimal), deepcopy(obj.z),
                                              deepcopy(obj.limits), deepcopy(obj.frontier),
                                              deepcopy(obj.solvers), deepcopy(obj.fail),
                                              copy(obj.model), deepcopy(obj.latest_prices),
                                              deepcopy(obj.alloc_optimal),
                                              deepcopy(obj.alloc_leftover),
                                              deepcopy(obj.alloc_solvers),
                                              deepcopy(obj.alloc_fail),
                                              copy(obj.alloc_model))
end

"""
```
mutable struct HCPortfolio{ast, dat, r, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew, tv,
                           tsskew, tsv, wmi, wma, ttco, tco, tdist, tcl, tk, topt, tsolv,
                           tf, tlp, taopt, talo, tasolv, taf, tamod} <: AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    skew::tskew
    V::tv
    sskew::tsskew
    SV::tsv
    w_min::wmi
    w_max::wma
    cor_type::ttco
    cor::tco
    dist::tdist
    clusters::tcl
    k::tk
    optimal::topt
    solvers::tsolv
    fail::tf
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_fail::taf
    alloc_model::tamod
end
```
"""
mutable struct HCPortfolio{ast, dat, r, tmu, tcov, tkurt, tskurt, tl2, ts2, tskew, tv,
                           tsskew, tsv, wmi, wma, ttco, tco, tdist, tcl, tk, topt, tsolv,
                           tf, tlp, taopt, talo, tasolv, taf, tamod} <: AbstractPortfolio
    assets::ast
    timestamps::dat
    returns::r
    mu::tmu
    cov::tcov
    kurt::tkurt
    skurt::tskurt
    L_2::tl2
    S_2::ts2
    skew::tskew
    V::tv
    sskew::tsskew
    SV::tsv
    w_min::wmi
    w_max::wma
    cor_type::ttco
    cor::tco
    dist::tdist
    clusters::tcl
    k::tk
    optimal::topt
    solvers::tsolv
    fail::tf
    latest_prices::tlp
    alloc_optimal::taopt
    alloc_leftover::talo
    alloc_solvers::tasolv
    alloc_fail::taf
    alloc_model::tamod
end
function HCPortfolio(; prices::TimeArray = TimeArray(TimeType[], []),
                     returns::DataFrame = DataFrame(),
                     ret::Matrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     timestamps::Vector{<:Dates.AbstractTime} = Vector{Date}(undef, 0),
                     assets::AbstractVector = Vector{String}(undef, 0),
                     mu::AbstractVector{<:Real} = Vector{Float64}(undef, 0),
                     cov::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     kurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     skurt::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     skew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     V = Matrix{eltype(returns)}(undef, 0, 0),
                     sskew::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     SV = Matrix{eltype(returns)}(undef, 0, 0),
                     w_min::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                     w_max::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                     cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                     cor::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     dist::AbstractMatrix{<:Real} = Matrix{Float64}(undef, 0, 0),
                     clusters::Clustering.Hclust = Hclust{Float64}(Matrix{Int64}(undef, 0,
                                                                                 2),
                                                                   Float64[], Int64[],
                                                                   :nothing),
                     k::Integer = 0, optimal::AbstractDict = Dict(),
                     solvers::AbstractDict = Dict(), fail::AbstractDict = Dict(),
                     latest_prices::AbstractVector = Vector{Float64}(undef, 0),
                     alloc_optimal::AbstractDict = Dict(),
                     alloc_leftover::AbstractDict = Dict(),
                     alloc_solvers::AbstractDict = Dict(),
                     alloc_fail::AbstractDict = Dict(),
                     alloc_model::JuMP.Model = JuMP.Model())
    if !isempty(prices)
        returns = dropmissing!(DataFrame(percentchange(prices)))
        latest_prices = Vector(dropmissing!(DataFrame(prices))[end, colnames(prices)])
    end
    if !isempty(returns)
        assets = setdiff(names(returns), ("timestamp",))
        timestamps = returns[!, "timestamp"]
        returns = Matrix(returns[!, assets])
    else
        @smart_assert(length(assets) == size(ret, 2))
        returns = ret
    end
    if !isempty(mu)
        @smart_assert(length(mu) == size(returns, 2))
    end
    if !isempty(cov)
        @smart_assert(size(cov, 1) == size(cov, 2) == size(returns, 2))
    end
    if !isempty(kurt)
        @smart_assert(size(kurt, 1) == size(kurt, 2) == size(returns, 2)^2)
    end
    if !isempty(skurt)
        @smart_assert(size(skurt, 1) == size(skurt, 2) == size(returns, 2)^2)
    end
    if !isempty(skew)
        @smart_assert(size(skew, 1) == size(returns, 2) &&
                      size(skew, 2) == size(returns, 2)^2)
    end
    if !isempty(V)
        @smart_assert(size(V, 1) == size(V, 2) == size(returns, 2))
    end
    if !isempty(sskew)
        @smart_assert(size(sskew, 1) == size(returns, 2) &&
                      size(sskew, 2) == size(returns, 2)^2)
    end
    if !isempty(SV)
        @smart_assert(size(SV, 1) == size(SV, 2) == size(returns, 2))
    end
    if isa(w_min, Real)
        if isa(w_max, Real)
            @smart_assert(w_min <= w_max)
        elseif !isempty(w_max)
            @smart_assert(all(w_min .<= w_max))
        end
    elseif isa(w_min, AbstractVector)
        if !isempty(w_min)
            @smart_assert(length(w_min) == size(returns, 2))
            if isa(w_max, Real) || !isempty(w_max)
                @smart_assert(all(w_min .<= w_max))
            end
        end
    end
    if isa(w_max, Real)
        if isa(w_min, Real)
            @smart_assert(w_max >= w_min)
        elseif !isempty(w_min)
            @smart_assert(all(w_max .>= w_min))
        end
    elseif isa(w_max, AbstractVector)
        if !isempty(w_max)
            @smart_assert(length(w_max) == size(returns, 2))
            if isa(w_min, Real) || !isempty(w_min)
                @smart_assert(all(w_max .>= w_min))
            end
        end
    end
    if !isempty(cor)
        @smart_assert(size(cor, 1) == size(cor, 2) == size(returns, 2))
    end
    if !isempty(dist)
        @smart_assert(size(dist, 1) == size(dist, 2) == size(returns, 2))
    end
    @smart_assert(k >= zero(k))
    if !isempty(latest_prices)
        @smart_assert(length(latest_prices) == size(returns, 2))
    end

    L_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)
    S_2 = SparseMatrixCSC{Float64, Int}(undef, 0, 0)

    return HCPortfolio{typeof(assets), typeof(timestamps), typeof(returns), typeof(mu),
                       typeof(cov), typeof(kurt), typeof(skurt), typeof(L_2), typeof(S_2),
                       typeof(skew), typeof(V), typeof(sskew), typeof(SV),
                       Union{<:Real, <:AbstractVector{<:Real}},
                       Union{<:Real, <:AbstractVector{<:Real}}, typeof(cor_type),
                       typeof(cor), typeof(dist), typeof(clusters), typeof(k),
                       typeof(optimal), typeof(solvers), typeof(fail),
                       typeof(latest_prices), typeof(alloc_optimal), typeof(alloc_leftover),
                       typeof(alloc_solvers), typeof(alloc_fail), typeof(alloc_model)}(assets,
                                                                                       timestamps,
                                                                                       returns,
                                                                                       mu,
                                                                                       cov,
                                                                                       kurt,
                                                                                       skurt,
                                                                                       L_2,
                                                                                       S_2,
                                                                                       skew,
                                                                                       V,
                                                                                       sskew,
                                                                                       SV,
                                                                                       w_min,
                                                                                       w_max,
                                                                                       cor_type,
                                                                                       cor,
                                                                                       dist,
                                                                                       clusters,
                                                                                       k,
                                                                                       optimal,
                                                                                       solvers,
                                                                                       fail,
                                                                                       latest_prices,
                                                                                       alloc_optimal,
                                                                                       alloc_leftover,
                                                                                       alloc_solvers,
                                                                                       alloc_fail,
                                                                                       alloc_model)
end
function Base.setproperty!(obj::HCPortfolio, sym::Symbol, val)
    if sym == :k
        @smart_assert(val >= zero(val))
    elseif sym == :w_min
        if isa(val, Real)
            if isa(obj.w_max, Real)
                @smart_assert(val <= obj.w_max)
            elseif !isempty(obj.w_max)
                @smart_assert(all(val .<= obj.w_max))
            end
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(obj.returns, 2))
                if isa(obj.w_max, Real) || !isempty(obj.w_max)
                    @smart_assert(all(val .<= obj.w_max))
                end
            end
        end
    elseif sym == :w_max
        if isa(val, Real)
            if isa(obj.w_min, Real)
                @smart_assert(val >= obj.w_min)
            elseif !isempty(obj.w_min)
                @smart_assert(all(val .>= obj.w_min))
            end
        elseif isa(val, AbstractVector)
            if !isempty(val)
                @smart_assert(length(val) == size(obj.returns, 2))
                if isa(obj.w_min, Real) || !isempty(obj.w_min)
                    @smart_assert(all(val .>= obj.w_min))
                end
            end
        end
    elseif sym ∈ (:mu, :latest_prices)
        if !isempty(val)
            @smart_assert(length(val) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:kurt, :skurt)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:skew, :sskew)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(obj.returns, 2) &&
                          size(val, 2) == size(obj.returns, 2)^2)
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:L_2, :S_2)
        if !isempty(val)
            N = size(obj.returns, 2)
            @smart_assert(size(val) == (Int(N * (N + 1) / 2), N^2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    elseif sym ∈ (:cov, :cor, :dist, :V, :SV)
        if !isempty(val)
            @smart_assert(size(val, 1) == size(val, 2) == size(obj.returns, 2))
        end
        val = convert(typeof(getfield(obj, sym)), val)
    end
    return setfield!(obj, sym, val)
end
function Base.deepcopy(obj::HCPortfolio)
    return HCPortfolio{typeof(obj.assets), typeof(obj.timestamps), typeof(obj.returns),
                       typeof(obj.mu), typeof(obj.cov), typeof(obj.kurt), typeof(obj.skurt),
                       typeof(obj.L_2), typeof(obj.S_2), typeof(obj.skew), typeof(obj.V),
                       typeof(obj.sskew), typeof(obj.SV),
                       Union{<:Real, <:AbstractVector{<:Real}},
                       Union{<:Real, <:AbstractVector{<:Real}}, typeof(obj.cor_type),
                       typeof(obj.cor), typeof(obj.dist), typeof(obj.clusters),
                       typeof(obj.k), typeof(obj.optimal), typeof(obj.solvers),
                       typeof(obj.fail), typeof(obj.latest_prices),
                       typeof(obj.alloc_optimal), typeof(obj.alloc_leftover),
                       typeof(obj.alloc_solvers), typeof(obj.alloc_fail),
                       typeof(obj.alloc_model)}(deepcopy(obj.assets),
                                                deepcopy(obj.timestamps),
                                                deepcopy(obj.returns), deepcopy(obj.mu),
                                                deepcopy(obj.cov), deepcopy(obj.kurt),
                                                deepcopy(obj.skurt), deepcopy(obj.L_2),
                                                deepcopy(obj.S_2), deepcopy(obj.skew),
                                                deepcopy(obj.V), deepcopy(obj.sskew),
                                                deepcopy(obj.SV), deepcopy(obj.w_min),
                                                deepcopy(obj.w_max), deepcopy(obj.cor_type),
                                                deepcopy(obj.cor), deepcopy(obj.dist),
                                                deepcopy(obj.clusters), deepcopy(obj.k),
                                                deepcopy(obj.optimal),
                                                deepcopy(obj.solvers), deepcopy(obj.fail),
                                                deepcopy(obj.latest_prices),
                                                deepcopy(obj.alloc_optimal),
                                                deepcopy(obj.alloc_leftover),
                                                deepcopy(obj.alloc_solvers),
                                                deepcopy(obj.alloc_fail),
                                                copy(obj.alloc_model))
end

export Portfolio, HCPortfolio
