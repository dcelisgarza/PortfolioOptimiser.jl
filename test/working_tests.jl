using PortfolioOptimiser, DataFrames, TimeSeries, Dates, Statistics, ECOS, MarketData, CSV

println(fieldnames(Portfolio))

A = TimeArray(CSV.File("./test/assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
RET = dropmissing!(DataFrame(Y))

test = Portfolio(
    returns = RET,
    # upper_short = 0.2,
    # upper_long = 1,
    # short = true,
    # sum_short_long = 0.8,
)
test.mu = vec(mean(Matrix(RET[!, 2:end]), dims = 1))
test.cov = cov(Matrix(RET[!, 2:end]))
w1 = optimize(test, ECOS.Optimizer, kelly = :exact, obj = :utility)
w2 = optimize(test, ECOS.Optimizer, kelly = :approx, obj = :utility)
w3 = optimize(test, ECOS.Optimizer, kelly = :none, obj = :utility)
sh3 = hcat(w1, w2, w3, makeunique = true)
display(sh3)

value.(test.model[:w])

using JuMP
value.(test.model[:w])

isinf(test.upper_average_drawdown)

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