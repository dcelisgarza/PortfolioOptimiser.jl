using CSV, TimeSeries, JuMP, Test, Clarabel, StatsBase, PortfolioOptimiser
prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
portfolio = Portfolio2(; prices = prices,
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => false))))
asset_statistics2!(portfolio)
@time w1 = optimise2!(portfolio; rm = Skew2(; settings = RiskMeasureSettings(; ub = r1)),
                      obj = MaxRet())

r1 = calc_risk(portfolio2; rm = :Skew)

deepcopy(portfolio)
String(SR())
portfolio2 = Portfolio(; prices = prices,
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => false))))
asset_statistics!(portfolio2)
portfolio2.skew_u = r1
@time w2 = optimise!(portfolio2, OptimiseOpt(; obj = :Max_Ret, rm = :Skew))

a = CVaR2()
size(a)
b = CVaR2(; alpha = BigFloat(0.5))

a isa CVaR2
b isa CVaR2
typeof(a) == typeof(b)

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
portfolio = Portfolio2(; prices = prices,
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => false))))
asset_statistics2!(portfolio)

portfolio.model = JuMP.Model()
@variable(portfolio.model, w[1:length(portfolio.assets)])
@constraint(portfolio.model, sum(portfolio.model[:w]) == 1)
@constraint(portfolio.model, portfolio.model[:w] .>= 0)
set_rm(portfolio, SSD2(), Trad2(), MinRisk(), 1, 1; mu = portfolio.mu,
       returns = portfolio.returns)
set_rm(portfolio, MAD2(; settings = RiskMeasureSettings(; scale = 2)), Trad2(), MinRisk(),
       1, 1; mu = portfolio.mu, returns = portfolio.returns)
@objective(portfolio.model, Min, portfolio.model[:risk])
set_optimizer(portfolio.model, portfolio.solvers[:Clarabel][:solver])
JuMP.optimize!(portfolio.model)
w1 = value.(portfolio.model[:w])

portfolio.model = JuMP.Model()
@variable(portfolio.model, w[1:length(portfolio.assets)])
@constraint(portfolio.model, sum(portfolio.model[:w]) == 1)
@constraint(portfolio.model, portfolio.model[:w] .>= 0)
set_rm(portfolio, SSD2(), Trad2(), MinRisk(), 1, 1; mu = portfolio.mu,
       returns = portfolio.returns)
set_rm(portfolio, MAD2(; settings = RiskMeasureSettings(; scale = 2)), Trad2(), MinRisk(),
       1, 1; mu = portfolio.mu, returns = portfolio.returns)
@objective(portfolio.model, Min, portfolio.model[:risk])
set_optimizer(portfolio.model, portfolio.solvers[:Clarabel][:solver])
JuMP.optimize!(portfolio.model)
w2 = value.(portfolio.model[:w])

using Random
mu2 = shuffle(portfolio.mu)

portfolio.model = JuMP.Model()
@variable(portfolio.model, w[1:length(portfolio.assets)])
@constraint(portfolio.model, sum(portfolio.model[:w]) == 1)
@constraint(portfolio.model, portfolio.model[:w] .>= 0)
set_rm(portfolio, SSD2(), Trad2(), MinRisk(), 1, 1; mu = portfolio.mu,
       returns = portfolio.returns)
set_rm(portfolio, MAD2(), Trad2(), MinRisk(), 1, 1; mu = mu2, returns = portfolio.returns)
@objective(portfolio.model, Min, portfolio.model[:risk])
set_optimizer(portfolio.model, portfolio.solvers[:Clarabel][:solver])
JuMP.optimize!(portfolio.model)
w3 = value.(portfolio.model[:w])

portfolio.model = JuMP.Model()
@variable(portfolio.model, w[1:length(portfolio.assets)])
@constraint(portfolio.model, sum(portfolio.model[:w]) == 1)
@constraint(portfolio.model, portfolio.model[:w] .>= 0)
set_rm(portfolio, SSD2(), Trad2(), MinRisk(), 1, 1; mu = portfolio.mu,
       returns = portfolio.returns)
set_rm(portfolio, MAD2(), Trad2(), MinRisk(), 1, 1; mu = mu2, returns = portfolio.returns)
@objective(portfolio.model, Min, portfolio.model[:risk])
set_optimizer(portfolio.model, portfolio.solvers[:Clarabel][:solver])
JuMP.optimize!(portfolio.model)
w4 = value.(portfolio.model[:w])

portfolio.model = JuMP.Model()
@variable(portfolio.model, w[1:length(portfolio.assets)])
@constraint(portfolio.model, sum(portfolio.model[:w]) == 1)
@constraint(portfolio.model, portfolio.model[:w] .>= 0)
set_rm(portfolio, SSD2(), Trad2(), MinRisk(), 2, 1; mu = portfolio.mu,
       returns = portfolio.returns)
set_rm(portfolio, SSD2(), Trad2(), MinRisk(), 2, 2; mu = mu2, returns = portfolio.returns)
@objective(portfolio.model, Min, portfolio.model[:risk])
set_optimizer(portfolio.model, portfolio.solvers[:Clarabel][:solver])
JuMP.optimize!(portfolio.model)
w5 = value.(portfolio.model[:w])

SD2[SD2()]
portfolio = Portfolio2(; prices = prices,
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => false))))
@variable(portfolio.model, w[1:length(portfolio.assets)])
@constraint(portfolio.model, sum(portfolio.model[:w]) == 1)
@constraint(portfolio.model, portfolio.model[:w] .>= 0)

setup_rm(portfolio, CVaR2(), Trad2(), MinRisk(), 1, 1)
# setup_rm(portfolio, CVaR2(; alpha = 0.3), true, Inf, 1, Trad2(), MinRisk(), 3, 2)
# setup_rm(portfolio, CVaR2(; alpha = 0.1), true, 0.3, 8, Trad2(), MinRisk(), 3, 3)

@objective(portfolio.model, Min, portfolio.model[:risk])
set_optimizer(portfolio.model, portfolio.solvers[:Clarabel][:solver])
JuMP.optimize!(portfolio.model)

ws2 = [portfolio.assets value.(portfolio.model[:w])]

portfolio = Portfolio2(; prices = prices,
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => false))))
@variable(portfolio.model, w[1:length(portfolio.assets)])
@constraint(portfolio.model, sum(portfolio.model[:w]) == 1)
@constraint(portfolio.model, portfolio.model[:w] .>= 0)

# setup_rm(portfolio, CVaR2(),  Trad2(), MinRisk(), 1, 1)
setup_rm(portfolio, CVaR2(; alpha = 0.3), Trad2(), MinRisk(), 1, 1)
# setup_rm(portfolio, CVaR2(; alpha = 0.1),  Trad2(), MinRisk(), 3, 3)

@objective(portfolio.model, Min, portfolio.model[:risk])
set_optimizer(portfolio.model, portfolio.solvers[:Clarabel][:solver])
JuMP.optimize!(portfolio.model)

ws3 = [portfolio.assets value.(portfolio.model[:w])]

portfolio = Portfolio2(; prices = prices,
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => false))))
@variable(portfolio.model, w[1:length(portfolio.assets)])
@constraint(portfolio.model, sum(portfolio.model[:w]) == 1)
@constraint(portfolio.model, portfolio.model[:w] .>= 0)

# setup_rm(portfolio, CVaR2(),  Trad2(), MinRisk(), 1, 1)
# setup_rm(portfolio, CVaR2(; alpha = 0.3),  Trad2(), MinRisk(), 1, 1)
setup_rm(portfolio, CVaR2(; alpha = 0.1), Trad2(), MinRisk(), 1, 1)

@objective(portfolio.model, Min, portfolio.model[:risk])
set_optimizer(portfolio.model, portfolio.solvers[:Clarabel][:solver])
JuMP.optimize!(portfolio.model)

ws4 = [portfolio.assets value.(portfolio.model[:w])]

################################################################

#######################################

for rtol ∈
    [1e-10, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 2.5e-1,
     5e-1, 1e0]
    a1, a2 = [4.738149345821625e-10, 3.663234596561653e-9, 7.073221682639396e-10,
              5.144311130281193e-10, 1.9363370662174556e-9, 4.015636234393278e-10,
              3.069936409025867e-10, 3.622860672267401e-10, 7.657470758194004e-10,
              7.055829167252015e-10, 2.3414048284473663e-9, 0.07736690505803431,
              0.07393383942482808, 7.686604081466158e-10, 1.0567885892794253e-9,
              1.0211218845834348e-9, 1.492719283255675e-8, 8.097139283006201e-10,
              1.0442229399923509e-9, 0.8486992237107189],
             [4.738499709522624e-10, 3.6633514859378074e-9, 7.073819084571228e-10,
              5.144786593921031e-10, 1.9364112760613397e-9, 4.015980331044767e-10,
              3.070219555232118e-10, 3.6232263189906873e-10, 7.657950831485577e-10,
              7.05607330254185e-10, 2.341495160152209e-9, 0.07736689078887454,
              0.07393381793501849, 7.687155728007624e-10, 1.0569020022084006e-9,
              1.02114427823091e-9, 1.4933139682479794e-8, 8.097577466777486e-10,
              1.044281070245392e-9, 0.8486992594628531]
    if isapprox(a1, a2; rtol = rtol)
        println(", rtol = $(rtol)")
        break
    end
end

cov_optn = CovOpt(; method = :SB1, gerber = GerberOpt(; normalise = true))
asset_statistics!(portfolio; calc_kurt = false, cov_opt = cov_optn)
mu15n = portfolio.mu
cov15n = portfolio.cov

println("covt15n = reshape($(vec(cov15n)), $(size(cov15n)))")

function f(rpe, warm)
    rpe = rpe * 10

    if warm == 1
        rpew = 0.6 * rpe[1]
        repsw = range(; start = 6, stop = 10)
    elseif warm == 2
        rpew = [0.5; 0.7] * rpe[1]
        repsw = [range(; start = 6, stop = 10), range(; start = 4, stop = 6)]
    elseif warm == 3
        rpew = [0.45; 0.65; 0.85] * rpe[1]
        repsw = [range(; start = 6, stop = 10), range(; start = 4, stop = 6),
                 range(; start = 3, stop = 4)]
    elseif warm == 4
        rpew = [0.3; 0.5; 0.7; 0.9] * rpe[1]
        repsw = [range(; start = 6, stop = 10), range(; start = 4, stop = 6),
                 range(; start = 3, stop = 4), range(; start = 2, stop = 3)]
    end

    return rpew, repsw
end

r = collect(range(; start = 9.5, stop = 10, length = 2))
f(r, 1)
display(r * 10)

# %%

f(w, h, l) = w * h * l

alex = 2 * f(32, 56, 7) + 3 * f(56, 32, 13)
trotten = 3 * f(15.6, 47, 40)
