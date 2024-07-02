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
    a1, a2 = [9.329022920510884e-8, 1.899704089201269e-6, 9.183579394059247e-8,
              1.1278439761516814e-7, 0.3233075357692419, 2.5473125618494352e-8,
              6.050608019992438e-6, 0.04416525698551672, 7.422022106006065e-8,
              6.013910567558717e-8, 0.2607281040507928, 2.284007335953662e-8,
              1.2053255537762558e-8, 6.734529963676938e-8, 1.361230758830987e-8,
              0.1350544143519127, 0.23673428114195488, 1.0078612571953623e-6,
              2.4778257564912495e-7, 6.281508294539916e-7],
             [9.314737762842888e-8, 1.8967448813980331e-6, 9.169510138324964e-8,
              1.1261141448788833e-7, 0.32330753683109287, 2.5434124390553046e-8,
              6.037612527333865e-6, 0.04416526579069986, 7.410651227385625e-8,
              6.004705250723869e-8, 0.2607280742124454, 2.2805101952467936e-8,
              1.2034803177431616e-8, 6.724220742605958e-8, 1.359147113792245e-8,
              0.13505441457541317, 0.23673432060735197, 1.006319666654265e-6,
              2.4740263628665907e-7, 6.271881186590633e-7]
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
