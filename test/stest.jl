using COSMO, CSV, Clarabel, DataFrames, Graphs, HiGHS, JuMP, LinearAlgebra,
      OrderedCollections, Pajarito, PortfolioOptimiser, Statistics, Test, TimeSeries,
      Logging, GLPK, Ipopt, SCS, NLopt, ECOS

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

solvers = Dict(:PClGL => Dict(:solver => optimizer_with_attributes(Pajarito.Optimizer,
                                                                   "verbose" => false,
                                                                   "oa_solver" => optimizer_with_attributes(GLPK.Optimizer,
                                                                                                            MOI.Silent() => true),
                                                                   "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                               "verbose" => false,
                                                                                                               "max_step_fraction" => 0.75))))
portfolio = Portfolio(; prices = prices, solvers = solvers)
asset_statistics!(portfolio; calc_kurt = false)

rm = :SD
L_A = cluster_matrix(portfolio; cor_opt = CorOpt(;),
                     cluster_opt = ClusterOpt(; linkage = :ward))

portfolio.network_method = :None
portfolio.network_ip = L_A
portfolio.network_sdp = L_A
w1 = optimise!(portfolio; obj = :Sharpe, rm = rm, l = l, rf = rf)
r1 = calc_risk(portfolio; rm = rm)

portfolio.sd_u = r1
portfolio.network_method = :IP
w2 = optimise!(portfolio; obj = :Min_Risk, rm = rm, l = l, rf = rf)
r2 = calc_risk(portfolio; rm = rm)
w3 = optimise!(portfolio; obj = :Utility, rm = rm, l = l, rf = rf)
r3 = calc_risk(portfolio; rm = rm)
w4 = optimise!(portfolio; obj = :Sharpe, rm = rm, l = l, rf = rf)
r4 = calc_risk(portfolio; rm = rm)
w5 = optimise!(portfolio; obj = :Max_Ret, rm = rm, l = l, rf = rf)
r5 = calc_risk(portfolio; rm = rm)

portfolio.network_method = :SDP
w6 = optimise!(portfolio; obj = :Min_Risk, rm = rm, l = l, rf = rf)
r6 = calc_risk(portfolio; rm = rm)
w7 = optimise!(portfolio; obj = :Utility, rm = rm, l = l, rf = rf)
r7 = calc_risk(portfolio; rm = rm)
w8 = optimise!(portfolio; obj = :Sharpe, rm = rm, l = l, rf = rf)
r8 = calc_risk(portfolio; rm = rm)
w9 = optimise!(portfolio; obj = :Max_Ret, rm = rm, l = l, rf = rf)
r9 = calc_risk(portfolio; rm = rm)

@test r2 <= r1 + sqrt(eps())
@test r3 <= r1 + length(w5.weights) * sqrt(eps())
@test r4 <= r1 + sqrt(eps())
@test r5 <= r1 + length(w5.weights) * sqrt(eps())

@test r6 <= r1 + sqrt(eps())
@test r7 <= r1 + sqrt(eps())
@test r8 <= r1 + sqrt(eps())
@test r9 <= r1 + length(w5.weights) * sqrt(eps())

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
