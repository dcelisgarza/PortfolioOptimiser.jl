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
    a1, a2 = [2.456179251844397e-9, 2.750239127257306e-9, 3.417527096163251e-9,
              3.081490792154701e-9, 0.8533950554131274, 1.0799615276036851e-9,
              0.14660490197119067, 1.8722597362203355e-9, 2.978315168038638e-9,
              2.0254145313597473e-9, 1.7991048312512529e-9, 1.1202766836170355e-9,
              7.978750454249214e-10, 1.4366086070472023e-9, 8.108644462561822e-10,
              5.568684463807914e-9, 3.827014432168025e-9, 1.9712211044381082e-9,
              3.188016792573601e-9, 2.4346281865154164e-9],
             [1.578396576423994e-9, 1.7668913536875456e-9, 2.195938348136312e-9,
              1.979973684702315e-9, 0.8533949984345872, 6.930810292518326e-10,
              0.1466049741867149, 1.2033685035042625e-9, 1.9132151137429597e-9,
              1.3019765686121734e-9, 1.156443581096697e-9, 7.199018490635218e-10,
              5.114053553430424e-10, 9.231823222747696e-10, 5.195704450503038e-10,
              3.576004887560627e-9, 2.4590196894435053e-9, 1.2670150699623182e-9,
              2.048214628941069e-9, 1.5650989103260291e-9]
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
