using Test,
    PortfolioOptimiser,
    DataFrames,
    TimeSeries,
    CSV,
    Dates,
    ECOS,
    SCS,
    Clarabel,
    COSMO,
    OrderedCollections,
    LinearAlgebra,
    StatsBase

A = TimeArray(CSV.File("./test/assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
RET = dropmissing!(DataFrame(Y))

PTypes = PortfolioOptimiser.PortTypes
RMs = PortfolioOptimiser.RiskMeasures
Kret = PortfolioOptimiser.KellyRet
ObjF = PortfolioOptimiser.ObjFuncs

rf = 1.0329^(1 / 252) - 1
l = 2.0

tickers = names(RET)[2:end]
mu = ret_model(MRet(), Matrix(RET[!, 2:end]), compound = false)
sigma = cov(Cov(), Matrix(RET[!, 2:end]))

portfolio1 = Portfolio(
    returns = RET,
    solvers = OrderedDict(
        :COSMO => Dict(:solver => COSMO.Optimizer),
        :Clarabel => Dict(:solver => Clarabel.Optimizer),
        :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
        :ECOS => Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => true)),
    ),
)
asset_statistics!(portfolio1)
wc_statistics!(portfolio1, box = :normal, ellipse = :normal)
dfs = gen_dataframes(portfolio1)

@testset "mv" begin
    portfolio1 = Portfolio(
        returns = RET,
        solvers = OrderedDict(
            :COSMO => Dict(:solver => COSMO.Optimizer),
            :Clarabel => Dict(:solver => Clarabel.Optimizer),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => true)),
        ),
    )
    asset_statistics!(portfolio1)

    # Mean variance
    ## Min Risk
    mv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[1],
        obj = ObjF[1],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    mv2 = EffMeanVar(tickers, mu, sigma; rf = rf, risk_aversion = 2 * l)
    min_risk!(mv2, optimiser = COSMO.Optimizer, silent = false)
    mv1_2 = hcat(
        mv1,
        DataFrame(weights2 = mv2.weights),
        DataFrame(abs_diff = abs.(mv1.weights - mv2.weights)),
    )
    display(mv1_2)
    @test rmsd(mv1.weights, mv2.weights) < 1e-3

    ## Max Util
    mv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        obj = ObjF[2],
        rm = RMs[1],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    mv2 = EffMeanVar(tickers, mu, sigma; rf = rf, risk_aversion = 2 * l)
    max_utility!(mv2, optimiser = COSMO.Optimizer, silent = false)
    mv1_2 = hcat(
        mv1,
        DataFrame(weights2 = mv2.weights),
        DataFrame(abs_diff = abs.(mv1.weights - mv2.weights)),
    )
    display(mv1_2)
    @test rmsd(mv1.weights, mv2.weights) < 1e-3

    ## Sharpe
    mv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        obj = ObjF[3],
        rm = RMs[1],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    mv2 = EffMeanVar(tickers, mu, sigma; rf = rf, risk_aversion = 2 * l)
    max_sharpe!(mv2, optimiser = COSMO.Optimizer, silent = false)
    mv1_2 = hcat(
        mv1,
        DataFrame(weights2 = mv2.weights),
        DataFrame(abs_diff = abs.(mv1.weights - mv2.weights)),
    )
    display(mv1_2)
    @test rmsd(mv1.weights, mv2.weights) < 1e-3

    ## Max Ret
    mv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        obj = ObjF[4],
        rm = RMs[1],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    mv2 = EffMeanVar(tickers, mu, sigma; rf = rf, risk_aversion = 2 * l)
    efficient_risk!(mv2, 10, optimiser = COSMO.Optimizer, silent = false)
    mv1_2 = hcat(
        mv1,
        DataFrame(weights2 = mv2.weights),
        DataFrame(abs_diff = abs.(mv1.weights - mv2.weights)),
    )
    display(mv1_2)
    @test rmsd(mv1.weights, mv2.weights) < 1e-3
end

@testset "msv target mu" begin
    portfolio1 = Portfolio(
        returns = RET,
        solvers = OrderedDict(
            :COSMO => Dict(:solver => COSMO.Optimizer),
            :Clarabel => Dict(:solver => Clarabel.Optimizer),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => true)),
        ),
    )
    asset_statistics!(portfolio1)

    # Mean Semivar
    ## Min Risk
    returns = Matrix(RET[!, 2:end])
    msv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[3],
        obj = ObjF[1],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    msv2 = EffMeanSemivar(
        tickers,
        mu,
        returns;
        target = transpose(mu),
        rf = rf,
        risk_aversion = l,
    )
    min_risk!(msv2, optimiser = COSMO.Optimizer, silent = false)
    msv1_2 = hcat(
        msv1,
        DataFrame(weights2 = msv2.weights),
        DataFrame(abs_diff = abs.(msv1.weights - msv2.weights)),
    )
    display(msv1_2)
    @test rmsd(msv1.weights, msv2.weights) < 1e-3

    ## Max util
    msv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[3],
        obj = ObjF[2],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    msv2 = EffMeanSemivar(
        tickers,
        mu,
        returns;
        target = transpose(mu),
        rf = rf,
        risk_aversion = 2 * l,
    )
    max_utility!(msv2, optimiser = COSMO.Optimizer, silent = false)
    msv1_2 = hcat(
        msv1,
        DataFrame(weights2 = msv2.weights),
        DataFrame(abs_diff = abs.(msv1.weights - msv2.weights)),
    )
    display(msv1_2)
    @test rmsd(msv1.weights, msv2.weights) < 1e-3

    ## Max Sharpe
    msv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[3],
        obj = ObjF[3],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    msv2 = EffMeanSemivar(
        tickers,
        mu,
        returns;
        target = transpose(mu),
        rf = rf,
        risk_aversion = 2 * l,
    )
    max_sharpe!(msv2, optimiser = COSMO.Optimizer, silent = false)
    msv1_2 = hcat(
        msv1,
        DataFrame(weights2 = msv2.weights),
        DataFrame(abs_diff = abs.(msv1.weights - msv2.weights)),
    )
    display(msv1_2)
    @test rmsd(msv1.weights, msv2.weights) < 1e-3

    ## Max Ret
    msv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[3],
        obj = ObjF[4],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    msv2 = EffMeanSemivar(
        tickers,
        mu,
        returns;
        target = transpose(mu),
        rf = rf,
        risk_aversion = 2 * l,
    )
    efficient_risk!(msv2, 10, optimiser = COSMO.Optimizer, silent = false)
    mv1_2 = hcat(
        msv1,
        DataFrame(weights2 = msv2.weights),
        DataFrame(abs_diff = abs.(msv1.weights - msv2.weights)),
    )
    display(mv1_2)
    @test rmsd(msv1.weights, msv2.weights) < 1e-3

    portfolio1 = Portfolio(
        returns = RET,
        msv_target = [],
        solvers = OrderedDict(
            :COSMO => Dict(:solver => COSMO.Optimizer),
            :Clarabel => Dict(:solver => Clarabel.Optimizer),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => true)),
        ),
    )
    asset_statistics!(portfolio1)

    msv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[3],
        obj = ObjF[3],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    msv2 = EffMeanSemivar(
        tickers,
        mu,
        returns;
        target = transpose(mu),
        rf = rf,
        risk_aversion = 2 * l,
    )
    max_sharpe!(msv2, optimiser = COSMO.Optimizer, silent = false)
    msv1_2 = hcat(
        msv1,
        DataFrame(weights2 = msv2.weights),
        DataFrame(abs_diff = abs.(msv1.weights - msv2.weights)),
    )
    display(msv1_2)
    @test rmsd(msv1.weights, msv2.weights) < 1e-3
end

@testset "msv target rf" begin
    portfolio1 = Portfolio(
        returns = RET,
        msv_target = fill(rf, length(tickers)),
        solvers = OrderedDict(
            :COSMO => Dict(:solver => COSMO.Optimizer),
            :Clarabel => Dict(:solver => Clarabel.Optimizer),
            :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
            :ECOS =>
                Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => true)),
        ),
    )
    asset_statistics!(portfolio1)

    # Mean variance
    # MSV target = rf
    ## Min Risk
    returns = Matrix(RET[!, 2:end])
    msv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[3],
        obj = ObjF[1],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    msv2 = EffMeanSemivar(tickers, mu, returns; target = rf, rf = rf, risk_aversion = l)
    min_risk!(msv2, optimiser = COSMO.Optimizer, silent = false)
    msv1_2 = hcat(
        msv1,
        DataFrame(weights2 = msv2.weights),
        DataFrame(abs_diff = abs.(msv1.weights - msv2.weights)),
    )
    display(msv1_2)
    @test rmsd(msv1.weights, msv2.weights) < 1e-3

    ## Max util
    msv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[3],
        obj = ObjF[2],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    msv2 = EffMeanSemivar(tickers, mu, returns; target = rf, rf = rf, risk_aversion = 2 * l)
    max_utility!(msv2, optimiser = COSMO.Optimizer, silent = false)
    msv1_2 = hcat(
        msv1,
        DataFrame(weights2 = msv2.weights),
        DataFrame(abs_diff = abs.(msv1.weights - msv2.weights)),
    )
    display(msv1_2)
    @test rmsd(msv1.weights, msv2.weights) < 1e-3

    ## Max Sharpe
    msv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[3],
        obj = ObjF[3],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    msv2 = EffMeanSemivar(tickers, mu, returns; target = rf, rf = rf, risk_aversion = 2 * l)
    max_sharpe!(msv2, optimiser = COSMO.Optimizer, silent = false)
    msv1_2 = hcat(
        msv1,
        DataFrame(weights2 = msv2.weights),
        DataFrame(abs_diff = abs.(msv1.weights - msv2.weights)),
    )
    display(msv1_2)
    @test rmsd(msv1.weights, msv2.weights) < 1.2e-3

    ## Max Ret
    msv1 = opt_port!(
        portfolio1,
        type = PTypes[1],
        rm = RMs[3],
        obj = ObjF[4],
        kelly = Kret[1],
        rf = rf,
        l = l,
    )
    msv2 = EffMeanSemivar(
        tickers,
        mu,
        returns;
        target = transpose(mu),
        rf = rf,
        risk_aversion = 2 * l,
    )
    efficient_risk!(msv2, 10, optimiser = COSMO.Optimizer, silent = false)
    mv1_2 = hcat(
        msv1,
        DataFrame(weights2 = msv2.weights),
        DataFrame(abs_diff = abs.(msv1.weights - msv2.weights)),
    )
    display(mv1_2)
    @test rmsd(msv1.weights, msv2.weights) < 1e-3
end

using PortfolioOptimiser,
    DataFrames,
    TimeSeries,
    Dates,
    Statistics,
    ECOS,
    MarketData,
    CSV,
    StatsBase,
    SCS,
    JuMP,
    LinearAlgebra,
    Clarabel,
    COSMO,
    # SparseArrays,
    OrderedCollections

A = TimeArray(CSV.File("./test/assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
RET = dropmissing!(DataFrame(Y))
N = ncol(RET[!, 2:end])
a_mtx_ineq = rand(3, N)
b_vec_ineq = ones(3)
wghts1 = rand(N)
wghts1 ./= sum(wghts1)
wghts2 = rand(N)
wghts2 ./= sum(wghts2)
tracking_err_weights = wghts1
turnover_weights = wghts2

test1 = Portfolio(
    returns = RET,
    # short_u = 0.2,
    # long_u = 1,
    # short = true,
    # sum_short_long = 0.8,
    # a_mtx_ineq = a_mtx_ineq,
    # b_vec_ineq = b_vec_ineq,
    # kind_tracking_err = :weights,
    # tracking_err = 0.05,
    # tracking_err_weights = tracking_err_weights,
    # turnover = 0.05,
    # turnover_weights = turnover_weights,
    # ## max_number_assets = 10,
    # min_number_effective_assets = 10,
    # mu_l = -1000000.0,
    # dev_u = 1000000.0,
    # mad_u = 1000000.0,
    # sdev_u = 1000000.0,
    # cvar_u = 1000000.0,
    # wr_u = 1000000.0,
    # flpm_u = 1000000.0,
    # slpm_u = 1000000.0,
    # mdd_u = 1000000.0,
    # add_u = 1000000.0,
    # cdar_u = 1000000.0,
    # uci_u = 1000000.0,
    # evar_u = 1000000.0,
    # edar_u = 1000000.0,
    # gmd_u = 1000000.0,
    # tg_u = 1000000.0,
    # rg_u = 1000000.0,
    # rcvar_u = 1000000.0,
    # rtg_u = 1000000.0,
    # krt_u = 1000000.0,
    # skrt_u = 1000000.0,
    # # rvar_u = 1000000.0,
    # # rdar_u = 1000000.0,
    solvers = OrderedDict(
        :Clarabel => Dict(:solver => Clarabel.Optimizer),
        :COSMO => Dict(:solver => COSMO.Optimizer),
        :SCS => Dict(:solver => SCS.Optimizer, :params => Dict("verbose" => 1)),
        :ECOS => Dict(:solver => ECOS.Optimizer, :params => Dict("verbose" => true)),
        # :GLPK => Dict(:solver => GLPK.Optmizer, :params => Dict("it_lim" => 2)),
    ),
)
asset_statistics!(test1)
wc_statistics!(test1, box = :delta, ellipse = :normal)

test2 = deepcopy(test1)
asset_statistics!(test2)

wc1 = opt_port!(test1, type = :wc, u_mu = :none, u_cov = :none, obj = :sharpe)
mv = opt_port!(test2, type = :trad, kelly = :none, rm = :mv, obj = :sharpe)
display(hcat(wc1, mv[!, 2], wc1[!, 2] - mv[!, 2], makeunique = true))

test.krt_u = Inf
test.max_num_assets_kurt = 20
krt1 = opt_port!(test, type = :trad, rm = :mv, kelly = :exact, obj = :sharpe)
krt2 = opt_port!(test, type = :trad, rm = :krt, kelly = :approx, obj = :sharpe)
krt3 = opt_port!(test, type = :trad, rm = :krt, kelly = :exact, obj = :sharpe)
display(hcat(krt1, krt2, krt3, makeunique = true))

(:none, :reg, :reg_pen)
rrp1 = opt_port!(test, type = :wc, rrp_ver = :none)
rrp2 = opt_port!(test, type = :rrp, rrp_ver = :reg)
rrp3 = opt_port!(test, type = :rrp, rrp_ver = :reg_pen, rrp_penalty = 1000)
rp1 = opt_port!(test, type = :rp, rm = :mv, kelly = :none)
rp2 = opt_port!(test, type = :rp, rm = :mv, kelly = :approx)
rp3 = opt_port!(test, type = :rp, rm = :msv, kelly = :exact)

display(hcat(rrp1, rrp2, rrp3, rp1, rp2, makeunique = true))

rrp1 = opt_port!(test, type = :rrp, rrp_ver = :none, kelly = :none)
rrp2 = opt_port!(test, type = :rrp, rrp_ver = :none, kelly = :approx)
rrp3 = opt_port!(test, type = :rrp, rrp_ver = :none, kelly = :exact)
display(hcat(rrp1, rrp2, rrp3, makeunique = true))

rrp11 = opt_port!(test, type = :rrp, rrp_ver = :reg, kelly = :none)
rrp21 = opt_port!(test, type = :rrp, rrp_ver = :reg, kelly = :approx)
rrp31 = opt_port!(test, type = :rrp, rrp_ver = :reg, kelly = :exact)
display(hcat(rrp11, rrp21, rrp31, makeunique = true))

rrp12 = opt_port!(test, type = :rrp, rrp_ver = :reg_pen, rrp_penalty = 1000, kelly = :none)
rrp22 =
    opt_port!(test, type = :rrp, rrp_ver = :reg_pen, rrp_penalty = 1000, kelly = :approx)
rrp32 = opt_port!(test, type = :rrp, rrp_ver = :reg_pen, rrp_penalty = 1000, kelly = :exact)
display(hcat(rrp12, rrp22, rrp32, makeunique = true))

rp1 = opt_port!(test, type = :rp, rm = :mv, kelly = :none)
rp2 = opt_port!(test, type = :rp, rm = :mv, kelly = :approx)

risk_budget = 100:-5:1
tr1 = opt_port!(
    test,
    type = :trad,
    rm = :rvar,
    kelly = :none,
    obj = :min_risk,
    risk_budget = risk_budget,
)
rp1 = opt_port!(
    test,
    type = :rp,
    rm = :rvar,
    kelly = :none,
    obj = :min_risk,
    risk_budget = risk_budget,
)

tr2 = opt_port!(
    test,
    type = :trad,
    rm = :rvar,
    kelly = :approx,
    obj = :min_risk,
    risk_budget = risk_budget,
)
rp2 = opt_port!(
    test,
    type = :rp,
    rm = :rvar,
    kelly = :approx,
    obj = :min_risk,
    risk_budget = risk_budget,
)

tr3 = opt_port!(
    test,
    type = :trad,
    rm = :rvar,
    kelly = :exact,
    obj = :min_risk,
    risk_budget = risk_budget,
)

wakao = hcat(tr1, rp1, tr2, rp2, tr3, makeunique = true)

risk_budget = nothing
tr1 = opt_port!(
    test,
    type = :trad,
    rm = :rvar,
    kelly = :none,
    obj = :min_risk,
    risk_budget = risk_budget,
)
rp1 = opt_port!(
    test,
    type = :rp,
    rm = :rvar,
    kelly = :none,
    obj = :min_risk,
    risk_budget = risk_budget,
)

tr2 = opt_port!(
    test,
    type = :trad,
    rm = :rvar,
    kelly = :approx,
    obj = :min_risk,
    risk_budget = risk_budget,
)
rp2 = opt_port!(
    test,
    type = :rp,
    rm = :rvar,
    kelly = :approx,
    obj = :min_risk,
    risk_budget = risk_budget,
)

tr3 = opt_port!(
    test,
    type = :trad,
    rm = :rvar,
    kelly = :exact,
    obj = :min_risk,
    risk_budget = risk_budget,
)

wakao2 = hcat(tr1, rp1, tr2, rp2, tr3, makeunique = true)
(
    :mv,
    :mad,
    :msv,
    :cvar,
    :wr,
    :flpm,
    :slpm,
    :mdd,
    :add,
    :cdar,
    :uci,
    :evar,
    :edar,
    :rdar,
    :rvar,
    :gmd,
    :tg,
    :rg,
    :rcvar,
    :rtg,
    :krt,
    :skrt,
)

mtx = duplication_matrix(4)

rms = PortfolioOptimiser.RiskMeasures
kellies = PortfolioOptimiser.KellyRet
objs = PortfolioOptimiser.ObjFuncs

weights = DataFrame[]
for rm in rms[19:end]
    for kelly in kellies[3:3]
        for obj in objs[1:1]
            @time push!(weights, optimize(test, rm = rm, kelly = kelly, obj = obj))
        end
    end
end

# test.wr_u = 0.035#0.04429675707220074
###########################
@time display(
    owa_l_moment_crm(
        100;
        k = 4,
        method = :msd,
        g = 0.5,
        max_phi = 0.5,
        solvers = Dict("SCS" => SCS.Optimizer),
        sol_params = Dict("SCS" => Dict("verbose" => 0)),
    ),
)

println(fieldnames(Portfolio))

A = TimeArray(CSV.File("./test/assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
RET = dropmissing!(DataFrame(Y))

test = Portfolio(
    returns = RET,
    # short_u = 0.2,
    # long_u = 1,
    # short = true,
    # sum_short_long = 0.8,
    solvers = Dict("ECOS" => ECOS.Optimizer),#, "SCS" => SCS.Optimizer),
    sol_params = Dict(
        "ECOS" => Dict("maxit" => 100, "feastol" => 1e-12, "verbose" => false),
    ),
)
test.mu = vec(mean(Matrix(RET[!, 2:end]), dims = 1))
test.cov = cov(Matrix(RET[!, 2:end]))

rm = :wr
obj = :max_ret
kelly = :exact
test.wr_u = 0.035#0.04429675707220074
w1 = optimize(test, rm = rm, kelly = kelly, obj = obj)
maximum(-value.(test.model[:hist_ret])) - test.wr_u

test = Portfolio(
    returns = RET,
    # short_u = 0.2,
    # long_u = 1,
    # short = true,
    # sum_short_long = 0.8,
    solvers = Dict("ECOS" => ECOS.Optimizer),#, "SCS" => SCS.Optimizer),
    sol_params = Dict(
        "ECOS" => Dict("maxit" => 100, "feastol" => 1e-12, "verbose" => false),
    ),
)
test.mu = vec(mean(Matrix(RET[!, 2:end]), dims = 1))
test.cov = cov(Matrix(RET[!, 2:end]))

test.tracking_err_weights =
    DataFrame(tickers = names(RET)[2:end], weights = collect(1:2:40) / sum(1:2:40))

obj = :max_ret
test.kind_tracking_err = :weights
test.tracking_err = Inf
w1 = optimize(test, kelly = :exact, obj = obj)

test.tracking_err = 0.1
w2 = optimize(test, kelly = :exact, obj = obj)

test.tracking_err = 0.002
w3 = optimize(test, kelly = :exact, obj = obj)

test.tracking_err = 0.00001
w4 = optimize(test, kelly = :exact, obj = obj)
sh2 = hcat(
    w1,
    w2[!, :weights],
    w3[!, :weights],
    w4[!, :weights],
    test.tracking_err_weights[!, :weights],
    makeunique = true,
)
display(sh2)

test = Portfolio(
    returns = RET,
    # short_u = 0.4,
    # long_u = 1,
    # short = true,
    # sum_short_long = 0.6,
    solvers = Dict("ECOS" => ECOS.Optimizer, "SCS" => SCS.Optimizer),
    sol_params = Dict(
        "ECOS" => Dict("maxit" => 1000, "feastol" => 1e-12, "verbose" => false),
    ),
)
test.mu = vec(mean(Matrix(RET[!, 2:end]), dims = 1))
test.cov = cov(Matrix(RET[!, 2:end]))

test.turnover_weights =
    DataFrame(tickers = names(RET)[2:end], weights = collect(1:2:40) / sum(1:2:40))

test.turnover = Inf
obj = :max_ret
w1 = optimize(test, kelly = :exact, obj = obj)

test.turnover = 0.4
w2 = optimize(test, kelly = :exact, obj = obj)

test.turnover = 0.05
w3 = optimize(test, kelly = :exact, obj = obj)

test.turnover = 0.005
w4 = optimize(test, kelly = :exact, obj = obj)
sh2 = hcat(
    w1,
    w2[!, :weights],
    w3[!, :weights],
    w4[!, :weights],
    test.turnover_weights[!, :weights],
    makeunique = true,
)
display(sh2)

test = Portfolio(
    returns = RET,
    # short_u = 0.2,
    # long_u = 1,
    # short = true,
    # sum_short_long = 0.8,
    solvers = Dict("ECOS" => ECOS.Optimizer),#, "SCS" => SCS.Optimizer),
    sol_params = Dict(
        "ECOS" => Dict("maxit" => 1000, "feastol" => 1e-12, "verbose" => false),
    ),
)
test.mu = vec(mean(Matrix(RET[!, 2:end]), dims = 1))
test.cov = cov(Matrix(RET[!, 2:end]))
test.dev_u = Inf#0.007720653477634564
test.mad_u = Inf#9.883349909235248
test.sdev_u = Inf#0.0010573893959405641
test.min_number_effective_assets = 0
obj = :sharpe
rm = :msd
@time w1 = optimize(test, rm = rm, kelly = :exact, obj = obj)
r1 = sqrt(dot(w1[!, :weights], test.cov, w1[!, :weights]))
mu1 = dot(w1[!, :weights], test.mu)
@time w2 = optimize(test, rm = rm, kelly = :approx, obj = obj)
r2 = sqrt(dot(w2[!, :weights], test.cov, w2[!, :weights]))
mu2 = dot(w2[!, :weights], test.mu)
@time w3 = optimize(test, rm = rm, kelly = :none, obj = obj)
r3 = sqrt(dot(w3[!, :weights], test.cov, w3[!, :weights]))
mu3 = dot(w3[!, :weights], test.mu)
sh3 = hcat(w1, w2[!, :weights], w3[!, :weights], makeunique = true)
display(sh3)

test.min_number_effective_assets = 20
@time w4 = optimize(test, rm = rm, kelly = :none, obj = obj)
r4 = sqrt(dot(w4[!, :weights], test.cov, w4[!, :weights]))
mu4 = dot(w4[!, :weights], test.mu)
sh4 = hcat(w1, w2[!, :weights], w3[!, :weights], w4[!, :weights], makeunique = true)
display(sh4)

w12 = @test rmsd(w1[!, :weights], w2[!, :weights])
w13 = @test rmsd(w1[!, :weights], w3[!, :weights])
w23 = @test rmsd(w2[!, :weights], w3[!, :weights])

println((w12, w13, w23))
display(sh3)

value.(test.model[:w])

using JuMP, LinearAlgebra

boo = rand(10)
wak = JuMP.Model()
@variable(wak, a[1:10, 1:10] >= 0)
@expression(wak, tra, sum(diag(a)))

@variable(wak, b >= 2)
@variable(wak, t >= 0)
@expression(wak, booa, 2 * dot(boo, a))
@expression(wak, b2, -2 * b)
@variable(wak, ab[1:2] >= 0)
@constraint(wak, cab1, ab[1] == booa)
@constraint(wak, cab2, ab[2] == b2)
@constraint(wak, cnst, [t; a] in NormOneCone())

value.(test.model[:w])

isinf(test.add_u)

push!(test.sol_params, "ECOS" => Dict("max_iters" => 500, "abstol" => 1e-8))

Dict("A" => 5, "B" => 10)

using JuMP, Convex, Statistics, ECOS, LinearAlgebra, TimeSeries, Dates

A = TimeArray(collect(Date(2023, 03, 01):Day(1):(Date(2023, 03, 20))), rand(20, 10))
Y = percentchange(A)

A = values(Y)
cv = cov(A)
cr = cor(A)
G = sqrt(cv)
mu = vec(mean(A, dims = 1))
T, n = size(A)

model1 = JuMP.Model(ECOS.Optimizer)
@variable(model1, w[1:n] .>= 0)
@variable(model1, g >= 0)
@variable(model1, t >= 0)
@variable(model1, k >= 0)
@constraint(model1, sum_w, sum(w) == k)
# @constraint(model1, sqrt_g, [g; transpose(G) * w] in SecondOrderCone())
# @expression(model1, risk, g * g)
@expression(model1, risk, dot(w, cv, w))
#! This is equivalent to quadoverlin(g, k)
@constraint(model1, qol, [k + t, 2 * g + k - t] in SecondOrderCone())
@expression(model1, ret, transpose(mu) * w - 0.5 * t)
@constraint(model1, risk_leq_1, risk <= 1)
@objective(model1, Max, ret)
optimize!(model1)
# println((value(t), value(k), value(risk), value(ret)))
println(solution_summary(model1))
println(value.(w) / value(k))

rf = 0.00012846213956385633
model2 = JuMP.Model(ECOS.Optimizer)
@variable(model2, w[1:n] .>= 0)
@variable(model2, g >= 0)
@variable(model2, gr[1:n] .>= 0)
# @variable(model2, t >= 0)
@variable(model2, k >= 0)
@constraint(model2, sum_w, sum(w) == k)
@constraint(model2, sqrt_g, [g; transpose(G) * w] in SecondOrderCone())
@expression(model2, risk, g * g)
# @expression(model2, risk, dot(w, cv, w))
@expression(model2, kret, k .+ A * w)
@constraint(model2, exp_gr[i = 1:n], [gr[i], k, kret[i]] in MOI.ExponentialCone())
@expression(model2, ret, 1 / T * sum(gr) - rf * k)
@constraint(model2, risk_leq_1, risk <= 1)
@objective(model2, Max, ret)
optimize!(model2)
# println((value(t), value(k), value(risk), value(ret)))
println(solution_summary(model2))
println(value.(w) / value(k))

rf = 0.00012846213956385633
model3 = JuMP.Model(ECOS.Optimizer)
@variable(model3, w[1:n] .>= 0)
@variable(model3, g >= 0)
@variable(model3, gr[1:n] .>= 0)
@variable(model3, k >= 0)
@constraint(model3, sum_w, sum(w) == k)
@constraint(model3, sqrt_g, [g; transpose(G) * w] in SecondOrderCone())
@expression(model3, risk, g * g)
# @expression(model3, risk, dot(w, cv, w))
@expression(model3, ret, transpose(mu) * w)
@constraint(model3, sret, ret - rf * k == 1)
@objective(model3, Min, risk)
optimize!(model3)
# println((value(t), value(k), value(risk), value(ret)))
println(solution_summary(model3))
println(value.(w) / value(k))

w2 = Variable(10, Positive())
g2 = Variable(Positive())
sqrt_g2 = g2 <= quadform(w2, cv)