using COSMO, CSV, Clarabel, DataFrames, Graphs, HiGHS, JuMP, LinearAlgebra,
      OrderedCollections, Pajarito, PortfolioOptimiser, Statistics, Test, TimeSeries,
      Logging, GLPK, Ipopt, SCS, NLopt, ECOS, PyCall

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0
returns = dropmissing!(DataFrame(percentchange(prices)))

portfolio = Portfolio(; prices = prices,
                      solvers = OrderedDict(#:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                            #                 :params => Dict("verbose" => false,
                                            #                                "max_step_fraction" => 0.75))
                                            :COSMO => Dict(:solver => COSMO.Optimizer,
                                                           :params => Dict("verbose" => false))))
asset_statistics!(portfolio)

A = block_vec_pq(portfolio.kurt, 20, 20)
vals_A, vecs_A = eigen(A)

portfolio.solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                           :params => Dict("verbose" => false)))
portfolio.max_num_assets_kurt = 1
portfolio.risk_budget = rand(1.0:20, 20)
rm = :SKurt
obj = :Min_Risk
opt = OptimiseOpt(; type = :RP, rm = rm, obj = obj, rf = rf, l = l)
w1 = optimise!(portfolio, opt)
fig1 = plot_risk_contribution(portfolio; type = :RP, rm = rm)
fig2 = plot_risk_contribution(portfolio.assets, vec(py"np.array(w)"), portfolio.returns;
                              rm = rm)

norm(vec(py"np.array(w)") - w1.weights)

portfolio.solvers = Dict(:COSMO => Dict(:solver => COSMO.Optimizer,
                                        :params => Dict("verbose" => false)))
portfolio.skurt = portfolio.kurt
portfolio.max_num_assets_kurt = 0

rm = :Kurt
obj = :Min_Risk
opt = OptimiseOpt(; type = :Trad, rm = rm, obj = obj, rf = rf, l = l)
w1 = optimise!(portfolio, opt)
opt.obj = :Utility
w2 = optimise!(portfolio, opt)
opt.obj = :Sharpe
w3 = optimise!(portfolio, opt)
opt.obj = :Max_Ret
w4 = optimise!(portfolio, opt)

rm = :SKurt
obj = :Min_Risk
opt = OptimiseOpt(; type = :Trad, rm = rm, obj = obj, rf = rf, l = l)
w5 = optimise!(portfolio, opt)
opt.obj = :Utility
w6 = optimise!(portfolio, opt)
opt.obj = :Sharpe
w7 = optimise!(portfolio, opt)
opt.obj = :Max_Ret
w8 = optimise!(portfolio, opt)

portfolio.max_num_assets_kurt = 1
rm = :Kurt
obj = :Min_Risk
opt = OptimiseOpt(; type = :Trad, rm = rm, obj = obj, rf = rf, l = l)
w9 = optimise!(portfolio, opt)
opt.obj = :Utility
w10 = optimise!(portfolio, opt)
opt.obj = :Sharpe
w11 = optimise!(portfolio, opt)
opt.obj = :Max_Ret
w12 = optimise!(portfolio, opt)

rm = :SKurt
obj = :Min_Risk
opt = OptimiseOpt(; type = :Trad, rm = rm, obj = obj, rf = rf, l = l)
w13 = optimise!(portfolio, opt)
opt.obj = :Utility
w14 = optimise!(portfolio, opt)
opt.obj = :Sharpe
w15 = optimise!(portfolio, opt)
opt.obj = :Max_Ret
w16 = optimise!(portfolio, opt)

@test isapprox(w1.weights, w5.weights)
@test isapprox(w2.weights, w6.weights)
@test isapprox(w3.weights, w7.weights)
@test isapprox(w4.weights, w8.weights)

@test isapprox(w9.weights, w13.weights)
@test isapprox(w10.weights, w14.weights)
@test isapprox(w11.weights, w15.weights)
@test isapprox(w12.weights, w16.weights)

# For relaxed utility and sharpe use cosmo.

opt = OptimiseOpt(; type = :RP, rm = rm)

portfolio.risk_budget = []
w1 = optimise!(portfolio, opt)
rc1 = risk_contribution(portfolio; type = :RP, rm = rm)
lrc1, hrc1 = extrema(rc1)

portfolio.risk_budget = 1:size(portfolio.returns, 2)
w2 = optimise!(portfolio, opt)
rc2 = risk_contribution(portfolio; type = :RP, rm = rm)
lrc2, hrc2 = extrema(rc2)

################################################################

#######################################

for rtol ∈ [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2.5e-1, 5e-1, 1e0]
    a1, a2 = [0.03269374269936843, 0.09876949827014882, 0.03290849405260685,
              0.03787936798623205, 0.028265730462756423, 0.04226582299997201,
              0.030089696780537235, 0.055992584587106854, 0.03176986081290085,
              0.11784259915845924, 0.06434041938139494, 0.025699904281409433,
              0.037243031562668956, 0.04284496917584549, 0.024516462564458785,
              0.05943055504242573, 0.10394176249369234, 0.05553854076852158,
              0.03445765298857188, 0.04350930393092219],
             [0.030795768440128966, 0.08980112455006087, 0.031361118018054905,
              0.03855780121358287, 0.024995134758674688, 0.043678051636418476,
              0.02230983327992818, 0.0665890737582307, 0.028172231626379267,
              0.11584085840938199, 0.09578101395782831, 0.023019314875309647,
              0.02574967780514567, 0.04520934606750117, 0.02031296669900166,
              0.046895996926926115, 0.09725285208056758, 0.07302873790866801,
              0.0319043655040979, 0.048744732484112936]
    if isapprox(a1, a2; rtol = rtol)
        println(", rtol = $(rtol)")
        break
    end
end
