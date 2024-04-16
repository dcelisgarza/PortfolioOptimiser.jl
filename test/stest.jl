using CSV, Clarabel, DataFrames, OrderedCollections, Test, TimeSeries, PortfolioOptimiser,
      LinearAlgebra, PyCall, MultivariateStats, JuMP, NearestCorrelationMatrix, StatsBase,
      AverageShiftedHistograms, Distances, Aqua, StatsPlots, GraphRecipes, BenchmarkTools,
      COSMO

ret = randn(100, 200)
T, N = size(ret)
q = T / N
X = cov(ret)

X1 = copy(X)
X2 = copy(X)
je = PortfolioOptimiser.JLoGo(; flag = true)
posdef = PortfolioOptimiser.PosdefNearest()
display(@allocated PortfolioOptimiser.jlogo!(je, posdef, X1))

opt = CovOpt(; jlogo = JlogoOpt(; flag = true))
display(@allocated c = PortfolioOptimiser._denoise_logo_mtx(0, 0, X2, opt, :cov, true))

@test isapprox(X1, c)
@test isapprox(X1, X2)

isapprox(X, X1)

me = PortfolioOptimiser.MeanBOP(; target = PortfolioOptimiser.TargetSE(), sigma = X)
mu1 = mean(me, ret)
display(@benchmark mean($me, $ret) setup = ())

opt = MuOpt(; method = :BOP, target = :SE, sigma = X)
mu2 = PortfolioOptimiser.mu_estimator(ret, opt)
display(@benchmark PortfolioOptimiser.mu_estimator($ret, $opt) setup = ())
@test isapprox(mu1, mu2)

normalise = true
a, b, c, d = 1, 2, 3, 4
ce = PortfolioOptimiser.CorGerberSB1(; normalise = normalise)
a = cor(ce, X)
b = cov(ce, X)
display(@benchmark PortfolioOptimiser.cov($ce, $X) setup = ())

opt = GerberOpt(; normalise = normalise)
c, d = PortfolioOptimiser.gerbersb1(X, opt)
display(@benchmark PortfolioOptimiser.gerbersb1($X, $opt) setup = ())
@test isapprox(a, c)
@test isapprox(b, d)

bins = :HGR
c, d = PortfolioOptimiser.mut_var_info_mtx(X, bins, normalise)
display(@benchmark PortfolioOptimiser.mut_var_info_mtx($X, $bins, $normalise) setup = ())

bins = PortfolioOptimiser.FD()
bins = PortfolioOptimiser.SC()
bins = PortfolioOptimiser.HGR()

prices_assets = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
prices_factors = TimeArray(CSV.File("./test/assets/factor_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0
portfolio = Portfolio(; prices = prices_assets,
                      solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                              :params => Dict("verbose" => false
                                                                              #   "max_step_fraction" => 0.65
                                                                              #   #   "max_iter" => 400,
                                                                              #   #   "max_iter"=>150,
                                                                              #   "tol_gap_abs" => 1e-8,
                                                                              #   "tol_gap_rel" => 1e-8,
                                                                              #   "tol_feas" => 1e-8,
                                                                              #   "tol_ktratio" => 1e-8,
                                                                              #   "equilibrate_max_iter" => 30,
                                                                              #   "reduced_tol_gap_abs" => 1e-8,
                                                                              #   "reduced_tol_gap_rel" => 1e-8,
                                                                              #   "reduced_tol_feas" => 1e-8,
                                                                              #   "reduced_tol_ktratio" => 1e-8

                                                                              ))))
asset_statistics!(portfolio)
# :SKew
# :SSKew
# :DVar

rm = :SSkew
opt = OptimiseOpt(; rf = rf, l = l, class = :Classic, type = :Trad, rm = rm,
                  obj = :Min_Risk, kelly = :None)

w1 = optimise!(portfolio, opt)
risk1 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
opt.kelly = :Approx
w2 = optimise!(portfolio, opt)
opt.kelly = :Exact
w3 = optimise!(portfolio, opt)
opt.obj = :Utility
opt.kelly = :None
w4 = optimise!(portfolio, opt)
opt.kelly = :Approx
w5 = optimise!(portfolio, opt)
opt.kelly = :Exact
w6 = optimise!(portfolio, opt)
opt.obj = :Sharpe
opt.kelly = :None
w7 = optimise!(portfolio, opt)
risk7 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
ret7 = dot(portfolio.mu, w7.weights)
opt.kelly = :Approx
w8 = optimise!(portfolio, opt)
risk8 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
ret8 = dot(portfolio.mu, w8.weights)
opt.kelly = :Exact
w9 = optimise!(portfolio, opt)
risk9 = calc_risk(portfolio; type = :Trad, rm = rm, rf = rf)
ret9 = dot(portfolio.mu, w9.weights)
opt.obj = :Max_Ret
opt.kelly = :None
w10 = optimise!(portfolio, opt)
opt.kelly = :Approx
w11 = optimise!(portfolio, opt)
opt.kelly = :Exact
w12 = optimise!(portfolio, opt)
setproperty!(portfolio, :mu_l, ret7)
opt.obj = :Min_Risk
opt.kelly = :None
w13 = optimise!(portfolio, opt)
setproperty!(portfolio, :mu_l, ret8)
w14 = optimise!(portfolio, opt)
setproperty!(portfolio, :mu_l, ret9)
w15 = optimise!(portfolio, opt)
setproperty!(portfolio, :mu_l, Inf)
rmf = Symbol(lowercase(string(rm)) * "_u")
setproperty!(portfolio, rmf, risk7)
opt.obj = :Max_Ret
opt.kelly = :None
w16 = optimise!(portfolio, opt)
setproperty!(portfolio, rmf, risk8)
w17 = optimise!(portfolio, opt)
setproperty!(portfolio, rmf, risk9)
w18 = optimise!(portfolio, opt)
setproperty!(portfolio, rmf, risk1)
opt.obj = :Sharpe
w19 = optimise!(portfolio, opt)
setproperty!(portfolio, rmf, Inf)

println("w1t = $(w1.weights)")
println("w2t = $(w2.weights)")
println("w3t = $(w3.weights)")
println("w4t = $(w4.weights)")
println("w5t = $(w5.weights)")
println("w6t = $(w6.weights)")
println("w7t = $(w7.weights)")
println("w8t = $(w8.weights)")
println("w9t = $(w9.weights)")
println("w10t = $(w10.weights)")
println("w11t = $(w11.weights)")
println("w12t = $(w12.weights)")

w1t = [1.6044126474808694e-6, 6.387919740147824e-5, 1.414858054840168e-6,
       1.6813775168781085e-6, 2.0469710540271765e-6, 2.116339953996922e-6,
       5.874226424834042e-7, 0.12812799249943374, 1.0974501778662702e-6,
       1.7086142924218442e-6, 0.4997282356314232, 1.3273919118913505e-6,
       9.728050097433342e-7, 0.026495764704323903, 0.00015023141200918842,
       0.013458959792367066, 2.946452848774722e-6, 0.21129032992645078,
       1.5467266618233742e-6, 0.12066555601381827]
w2t = [1.6741408439788703e-6, 5.4181619830474946e-5, 1.4772206315003974e-6,
       1.7971431052953999e-6, 2.6482468158924754e-6, 2.246087598334583e-6,
       5.917923438339016e-7, 0.12812195127012185, 1.1231990478506012e-6,
       1.7619259095515968e-6, 0.4996871899259452, 1.5623277400739484e-6,
       9.939378327706881e-7, 0.026506706873806737, 0.00016017104781535386,
       0.01342123105488912, 3.830684541612858e-6, 0.21131359319071602, 1.588305210829121e-6,
       0.12071368000525379]
w3t = [9.112967407024484e-8, 2.4284652557869215e-6, 8.159064709668178e-8,
       1.1181858109975545e-7, 2.6663481590850287e-7, 1.7003046143834034e-7,
       3.159882605610237e-8, 0.12809183981977293, 5.851550821538698e-8, 9.74327693873567e-8,
       0.49967137044020066, 1.9733654127513186e-7, 8.948653620120993e-8,
       0.026607801299592827, 5.951230209656851e-6, 0.013510867564644547,
       3.9524058040306547e-7, 0.21131359797294272, 8.142442397451755e-8,
       0.12079447096801577]
w4t = [7.052084116651649e-11, 1.1011456306472704e-10, 2.1349200076014008e-10,
       1.5432621008187172e-10, 0.8964274785082526, 1.739778466305075e-10,
       0.10357251888949207, 3.345459809184482e-11, 1.2749228017926342e-10,
       1.3909874692879892e-11, 4.59595184139058e-11, 1.589986592408071e-10,
       2.1698240512924053e-10, 1.0784688621030566e-10, 2.1379411467057645e-10,
       4.6422703030242035e-10, 2.6128939335401695e-10, 1.9650064748911383e-11,
       1.5795421790524158e-10, 5.826476193012757e-11]
w5t = [4.03951510116872e-11, 6.56162536985027e-11, 1.0181240275254102e-10,
       7.787365194556045e-11, 0.8832591266867846, 4.5587652947206215e-11,
       0.11674087205513205, 1.0713501752351081e-11, 7.651053985807375e-11,
       1.5882559822214765e-11, 4.861705899662827e-12, 4.362777191010949e-11,
       6.282059377511313e-11, 2.1620509050839107e-11, 6.211225286087416e-11,
       3.305784345455929e-10, 1.414471943424064e-10, 1.5975960718954705e-11,
       9.638055375788389e-11, 4.4266689812649654e-11]
w6t = [1.5811311060775533e-9, 1.9187489530488656e-9, 2.3737110193614567e-9,
       2.072788553699979e-9, 0.886376590169756, 5.364744856000569e-10, 0.11362337985191061,
       1.2353120458872547e-9, 2.079805228313373e-9, 1.3122687984673233e-9,
       1.1673486922757801e-9, 5.645804291400642e-10, 3.228190976064559e-10,
       8.348336683389837e-10, 3.319787180535584e-10, 5.435061631443339e-9,
       2.9131005505429675e-9, 1.2988360236514801e-9, 2.3385785703096787e-9,
       1.6609558545100205e-9]
w7t = [1.9208950784660518e-10, 3.2176497330463307e-10, 2.4046864614545796e-10,
       1.8337883326718426e-10, 0.8086229923872494, 7.988007958328952e-11,
       0.0043547547791044595, 3.386058392712632e-10, 1.9622464070952902e-10,
       1.6355298824641334e-10, 3.1940809699051557e-10, 7.012901224488476e-11,
       4.874296829843509e-11, 1.3641899280890118e-10, 5.2647081948336176e-11,
       0.18702224906335355, 5.376052134706978e-10, 2.9754670175260314e-10,
       3.153043649354607e-10, 2.765246466590213e-10]
w8t = [1.5640157344076053e-8, 3.582061272235252e-8, 2.0967647646182747e-8,
       1.434265151901145e-8, 0.7615402768341621, 5.8799811509887846e-9, 0.01795493375117831,
       6.82072789192539e-8, 1.7161833781871267e-8, 1.3937346954368635e-8,
       5.331381082832199e-8, 4.989596015087121e-9, 3.52895190702238e-9,
       1.1379989369077216e-8, 3.725672571225845e-9, 0.22050422976420622,
       1.7751905872404807e-7, 4.198766806450605e-8, 4.217111232106812e-8,
       2.907708355516215e-8]
w9t = [5.553592439719559e-8, 1.1663023650724744e-7, 7.633484799626114e-8,
       5.326938535959939e-8, 0.7729653491506971, 2.083948647969848e-8, 0.020610259502484927,
       1.520542775416144e-7, 6.471991580131279e-8, 4.987187222564725e-8,
       1.2708076393696777e-7, 1.8038849954898925e-8, 1.2669331427630937e-8,
       3.910989923638194e-8, 1.3355344444324209e-8, 0.2064228749643929,
       3.6443089845109477e-7, 1.1604485798825767e-7, 1.4158630547059004e-7,
       9.481022777802042e-8]
w10t = [4.3422955344620045e-9, 4.482608372372868e-9, 5.2151136335546805e-9,
        4.997973799097022e-9, 1.479369787463968e-7, 2.2326679314415314e-9,
        0.999999785026607, 3.327627412942966e-9, 4.59970993047624e-9, 3.582706666614823e-9,
        3.263665683434885e-9, 2.3305636210161792e-9, 1.7153315191007083e-9,
        2.77851132735056e-9, 1.7865323395456848e-9, 4.945270403094947e-9,
        5.191706124856787e-9, 3.4811228279287047e-9, 4.650003219981197e-9,
        4.113003750877124e-9]
w11t = [5.619586642787703e-12, 8.276729930516474e-12, 1.4303142954340837e-11,
        1.1318044807100396e-11, 0.8503236954639315, 7.1381171385102224e-12,
        0.14967630438138319, 4.2876945553380653e-13, 1.0193743602633687e-11,
        1.732551494667518e-12, 4.5941464265574075e-14, 6.44661322933117e-12,
        9.08817518586835e-12, 3.5643431740902176e-12, 8.77780354909866e-12,
        3.082287714886444e-11, 1.7759280864727746e-11, 1.3797700206848313e-12,
        1.2160188596177151e-11, 5.629568200254058e-12]
w12t = [1.4913543300318003e-10, 1.6699242865726496e-10, 2.07522285731361e-10,
        1.871090607872307e-10, 0.8533951274589937, 6.55647304423056e-11,
        0.14660486995341881, 1.1367885040236333e-10, 1.808379346386064e-10,
        1.2297986365358102e-10, 1.0923688096630237e-10, 6.801869384941162e-11,
        4.844089767435111e-11, 8.72246410303023e-11, 4.9229146078698996e-11,
        3.381458846828161e-10, 2.3238337104439516e-10, 1.196872652847151e-10,
        1.9357080965121847e-10, 1.4782934867899478e-10]

@test isapprox(w1.weights, w1t)
@test isapprox(w2.weights, w2t)
@test isapprox(w3.weights, w3t)
@test isapprox(w2.weights, w3.weights, rtol = 5e-4)
@test isapprox(w4.weights, w4t)
@test isapprox(w5.weights, w5t)
@test isapprox(w6.weights, w6t)
@test isapprox(w5.weights, w6.weights, rtol = 5e-3)
@test isapprox(w7.weights, w7t)
@test isapprox(w8.weights, w8t)
@test isapprox(w9.weights, w9t)
@test isapprox(w8.weights, w9.weights, rtol = 3e-2)
@test isapprox(w10.weights, w10t)
@test isapprox(w11.weights, w11t)
@test isapprox(w12.weights, w12t)
@test isapprox(w11.weights, w12.weights, rtol = 6e-3)
@test isapprox(w13.weights, w7.weights, rtol = 7e-4)
@test isapprox(w14.weights, w8.weights, rtol = 4e-2)
@test isapprox(w15.weights, w9.weights, rtol = 4e-2)
@test isapprox(w16.weights, w7.weights, rtol = 4e-5)
@test isapprox(w17.weights, w8.weights, rtol = 4e-2)
@test isapprox(w18.weights, w9.weights, rtol = 4e-2)
@test isapprox(w13.weights, w16.weights, rtol = 8e-4)
@test isapprox(w14.weights, w17.weights, rtol = 4e-3)
@test isapprox(w15.weights, w18.weights, rtol = 4e-3)
@test isapprox(w19.weights, w1.weights, rtol = 6e-3)

##################
portfolio.skew_factor = Inf
portfolio.skew_u = Inf

portfolio.risk_budget = []
w1 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :DVar, obj = :Min_Risk))
plot_risk_contribution(portfolio; type = :Trad, rm = :DVar)
dvar1 = calc_risk(portfolio; type = :Trad, rm = :DVar)
value(portfolio.model[:risk])

portfolio.dvar_u = dvar1
w2 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :DVar, obj = :Max_Ret))
plot_risk_contribution(portfolio; type = :Trad, rm = :DVar)
dvar2 = calc_risk(portfolio; type = :Trad, rm = :DVar)
value(portfolio.model[:risk])

portfolio.dvar_u = Inf
w3 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :DVar, obj = :Sharpe))
plot_risk_contribution(portfolio; type = :Trad, rm = :DVar)
dvar3 = calc_risk(portfolio; type = :Trad, rm = :DVar)

portfolio.dvar_u = dvar3
w4 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :DVar, obj = :Max_Ret))
plot_risk_contribution(portfolio; type = :Trad, rm = :DVar)
dvar4 = calc_risk(portfolio; type = :Trad, rm = :DVar)

portfolio.dvar_u = Inf
w5 = optimise!(portfolio, OptimiseOpt(; type = :RP, rm = :SSkew, obj = :Max_Ret))
plot_risk_contribution(portfolio; type = :RP, rm = :SSkew, percentage = false)
dvar5 = calc_risk(portfolio; type = :RP, rm = :SSkew)

portfolio.dvar_u = dvar5
w6 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :DVar, obj = :Max_Ret))
plot_risk_contribution(portfolio; type = :Trad, rm = :DVar)
dvar6 = calc_risk(portfolio; type = :Trad, rm = :DVar)

DVar(portfolio.returns * w5.weights) / 20

##################
##################
w3 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :DVar, obj = :Sharpe))
plot_risk_contribution(portfolio; type = :Trad, rm = :DVar)
dvar3 = calc_risk(portfolio; type = :Trad, rm = :DVar)
value(portfolio.model[:risk] / portfolio.model[:k]^2)

w4 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :DVar, obj = :Max_Ret))
plot_risk_contribution(portfolio; type = :Trad, rm = :DVar)
dvar4 = calc_risk(portfolio; type = :Trad, rm = :DVar)
value(portfolio.model[:risk])

########################################################################################
########################################################################################

w2 = optimise!(portfolio, OptimiseOpt(; type = :RP, rm = :DVar))
plot_risk_contribution(portfolio; type = :RP, rm = :DVar)

portfolio.risk_budget = 1:2:40
w3 = optimise!(portfolio, OptimiseOpt(; type = :RP, rm = :DVar))
plot_risk_contribution(portfolio; type = :RP, rm = :DVar)

efficient_frontier!(portfolio, OptimiseOpt(; rm = :DVar); points = 7)
plot_frontier(portfolio; rm = :DVar)

efficient_frontier!(portfolio, OptimiseOpt(; rm = :Skew))
plot_frontier(portfolio; rm = :Skew)

efficient_frontier!(portfolio, OptimiseOpt(; rm = :SSkew))
plot_frontier(portfolio; rm = :SSkew)

portfolio.skew_factor = Inf
efficient_frontier!(portfolio, OptimiseOpt(; rm = :SD))
plot_frontier(portfolio; rm = :SD)
portfolio.skew_factor = 3
efficient_frontier!(portfolio, OptimiseOpt(; rm = :SD))
plot_frontier(portfolio; rm = :SD)

portfolio.skew_factor = Inf
w1 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :Skew))
portfolio.skew_factor = 64
portfolio.sskew_factor = 2
w2 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :CDaR))
portfolio.skew_factor = Inf
w3 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :CDaR))
portfolio.skew_factor = 0
w4 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :CDaR))

portfolio.dvar_u = Inf
w1 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :DVar, obj = :Min_Risk))
dvar1 = calc_risk(portfolio; rm = :DVar)

portfolio.dvar_u = dvar1
w2 = optimise!(portfolio, OptimiseOpt(; type = :Trad, rm = :DVar, obj = :Max_Ret))
display(hcat(w1, w2.weights; makeunique = true))

skew1 = sqrt(transpose(w1.weights) * portfolio.V * w1.weights)

portfolio.skew_u = skew1
w2 = optimise!(portfolio, OptimiseOpt(; rm = :Skew, obj = :Max_Ret))
skew2 = calc_risk(portfolio; rm = :Skew)

portfolio.skew_factor = 8

portfolio.skew_factor = Inf

w3 = optimise!(portfolio, OptimiseOpt(; rm = :SSkew, obj = :Min_Risk))
w4 = optimise!(portfolio, OptimiseOpt(; rm = :SSkew, obj = :Min_Risk, sd_cone = false))

w3 = optimise!(portfolio, OptimiseOpt(; rm = :Skew, obj = :Sharpe))
w4 = optimise!(portfolio, OptimiseOpt(; rm = :Skew, obj = :Sharpe))

r1 = dot(portfolio.mu, w1.weights) / dvar1
dot(portfolio.mu, w2.weights) / dvar1

w2 = optimise!(portfolio, OptimiseOpt(; rm = :DVar, obj = :Sharpe))
dvar2 = DVar(portfolio.returns * w2.weights)
r2 = dot(portfolio.mu, w2.weights) / dvar2

w3 = optimise!(portfolio, OptimiseOpt(; rm = :DVar, obj = :Sharpe))
dvar3 = DVar(portfolio.returns * w3.weights)
r3 = dot(portfolio.mu, w3.weights) / dvar3

w4 = optimise!(portfolio, OptimiseOpt(; rm = :DVar, obj = :Sharpe))
dvar4 = DVar(portfolio.returns * w4.weights)
r4 = dot(portfolio.mu, w4.weights) / dvar4

asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = AugClampDist())),
                  calc_kurt = false)
w1 = optimise!(portfolio; type = :NCO, cluster_opt = ClusterOpt(; linkage = :single))
display(plot_clusters(portfolio; cluster = false))

asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = AugClampDist())),
                  calc_kurt = false)
w1 = optimise!(portfolio; type = :NCO, cluster_opt = ClusterOpt(; linkage = :average))
display(plot_clusters(portfolio; cluster = false))

asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = AugClampDist())),
                  calc_kurt = false)
w1 = optimise!(portfolio; type = :NCO, cluster_opt = ClusterOpt(; linkage = :complete))
display(plot_clusters(portfolio; cluster = false))

asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = AugClampDist())),
                  calc_kurt = false)
w1 = optimise!(portfolio; type = :NCO, cluster_opt = ClusterOpt(; linkage = :ward))
display(plot_clusters(portfolio; cluster = false))

asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = AugClampDist())),
                  calc_kurt = false)
w1 = optimise!(portfolio; type = :NCO, cluster_opt = ClusterOpt(; linkage = :DBHT))
display(plot_clusters(portfolio; cluster = false))

asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = AugClampDist())),
                  calc_kurt = false)
w1 = optimise!(portfolio; type = :NCO,
               cluster_opt = ClusterOpt(; linkage = :DBHT,
                                        genfunc = GenericFunction(;
                                                                  func = (corr, dist) -> ceil(maximum(dist)^2) .-
                                                                                         dist .^
                                                                                         2)))
display(plot_clusters(portfolio; cluster = false))

(corr, dist, args...; kwargs...) -> ceil(maximum(dist)^2) .- dist .^ 2

####################################

Aqua.test_all(PortfolioOptimiser; ambiguities = false, deps_compat = false)

test = rand(1000, 300)
T, N = size(test)

dims = 1
corrected = true
mu = nothing
w = pweights(rand(T))
# args = (1)
std(test; corrected = corrected, mean = mu, dims = 1)

mean(test, w; dims = dims)

X = cov(test)

ce = CovType(; ce = SemiCov())
de = DistType()
@time begin
    c1 = cor(ce, X)
    d1 = dist(de, c1)
end

@time c2, d2 = PortfolioOptimiser.cor_dist_mtx(X, CorOpt(; method = :Semi_Pearson))

@test isapprox(c1, c2)
@test isapprox(d1, d2)

function jlogo2(X)
    corr = cov2cor(X)
    dist = sqrt.(clamp!((1 .- corr) / 2, 0, 1))
    separators, cliques = PMFG_T2s(1 .- dist .^ 2, 4)[3:4]
    return J_LoGo(X, separators, cliques) \ I
end

struct ClampDist <: Distances.UnionMetric end
function Distances.pairwise(::ClampDist, i, X)
    return sqrt.(clamp!((1 .- X) / 2, zero(eltype(X)), one(eltype(X))))
end

ce = CovType(; ce = Gerber_SB1())
@time c1 = StatsBase.cov(ce, X)
@time c2 = PortfolioOptimiser.covar_mtx(X, CovOpt(; method = :Gerber_SB1))
@test isapprox(c1, c2)

ce = JLoGoCov(; metric = ClampDist(),
              func = (corr, dist, args...; kwargs...) -> 1 .- dist .^ 2)
@allocations c1 = jlogo2(X)
@allocations c2 = PortfolioOptimiser.jlogo(ce, X, true)

@test isapprox(c1, c2)

@time dc1 = denoise_cov(cov2cor(X), T / N, DenoiseOpt(; method = :None))
@time dc2 = PortfolioOptimiser.denoise(NoDenoiser(), cov2cor(X), T / N, false)

isapprox(dc1, dc2)

ce = MutualInfoCov()
@time c1, d1 = PortfolioOptimiser.mut_var_info_mtx(test)
@time c2 = PortfolioOptimiser.mutual_info_mtx(test)
@time d2 = PortfolioOptimiser.variation_info_mtx(test)

@time c3 = cor(MutualInfoCov(), test)
@time d3 = pairwise(VariationInfo(), 1, test)

@test isapprox(c1, c2)
@test isapprox(c1, c3)
@test isapprox(d1, d2)
@test isapprox(d1, d3)

test = DenoiseOpt()

test.kernel
typeof(AverageShiftedHistograms.Kernels)

cv = cov(test)
cr = similar(cv)
StatsBase.cov2cor!(cv, sqrt.(diag(cv)))

@time cor1, cov1 = PortfolioOptimiser.gerbersb0(test, GerberOpt(; normalise = false))
@time cov2 = cov(Gerber_SB0(; normalise = false), test)
@time cor2 = cor(Gerber_SB0(; normalise = false), test)
@test isapprox(cov1, cov2)
@test isapprox(cor1, cor2)

@test isapprox(cor(Cov2Cor(), cov2), cor2)

@time c1 = PortfolioOptimiser.ltdi_mtx(test, 0.05)
ltdi = LTDI()
@time c2 = cov(ltdi, test)
isapprox(c1, c2)

@time cc1, cv1 = PortfolioOptimiser.gerber0(test, GerberOpt(; normalise = true))
@time cc2, cv2 = cov(Gerber0(; normalise = true), test)

@time cc3, cv3 = PortfolioOptimiser.gerber1(test, GerberOpt(; normalise = true))
@time cc4, cv4 = cov(Gerber1(; normalise = true), test)

@time cc5, cv5 = PortfolioOptimiser.gerber2(test, GerberOpt(; normalise = true))
@time cc6, cv6 = cov(Gerber2(; normalise = true), test)

isapprox(cc1, cc2)
isapprox(cv1, cv2)
isapprox(cc3, cc3)
isapprox(cv4, cv4)
isapprox(cc5, cc5)
isapprox(cv6, cv6)

c1 = cov(FullCov(;), test)
c2 = cov(SemiCov(;), test)

prices_assets = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
prices_factors = TimeArray(CSV.File("./test/assets/factor_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

portfolio = Portfolio(; prices = prices_assets, f_prices = prices_factors,
                      solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                              :params => Dict("verbose" => false
                                                                              #   "max_step_fraction" => 0.75
                                                                              #   "max_iter" => 400,
                                                                              #   "max_iter"=>150,
                                                                              #   "tol_gap_abs" => 1e-8,
                                                                              #   "tol_gap_rel" => 1e-8,
                                                                              #   "tol_feas" => 1e-8,
                                                                              #   "tol_ktratio" => 1e-8,
                                                                              #   "equilibrate_max_iter" => 30,
                                                                              #   "reduced_tol_gap_abs" => 1e-6,
                                                                              #   "reduced_tol_gap_rel" => 1e-5,
                                                                              #   "reduced_tol_feas" => 1e-6,
                                                                              #   "reduced_tol_ktratio" => 1e-6
                                                                              ))))
asset_statistics!(portfolio; calc_kurt = false)
w1 = optimise!(portfolio)

using GraphRecipes, StatsPlots
plot_drawdown(portfolio)

mvr_opt = MVROpt(;)
loadings_opt = LoadingsOpt(; mvr_opt = mvr_opt)
factor_opt = FactorOpt(; loadings_opt = loadings_opt)
posdef = PosdefFixOpt(; method = :Nearest)
cov_opt = CovOpt(; posdef = posdef)
mu_opt = MuOpt(;)
loadings_opt.method = :MVR
mvr_opt.pca_genfunc.kwargs = (; pratio = 0.9)
factor_statistics!(portfolio; cov_opt = cov_opt, mu_opt = mu_opt, factor_opt = factor_opt)
portfolio.loadings

mvr_opt = MVROpt(;)
loadings_opt = LoadingsOpt(; mvr_opt = mvr_opt)
factor_opt = FactorOpt(; loadings_opt = loadings_opt)
posdef = PosdefFixOpt(; method = :Nearest)
cov_opt = CovOpt(; posdef = posdef)
mu_opt = MuOpt(;)
loadings_opt.method = :BReg
# mvr_opt.pca_genfunc.kwargs = (; pratio = 0.9)
# mvr_opt.pca_genfunc.args = (MultivariateStats.PCA,)
factor_statistics!(portfolio; cov_opt = cov_opt, mu_opt = mu_opt, factor_opt = factor_opt)
breg2 = copy(portfolio.loadings)

mvr_opt.pca_genfunc.args = (MultivariateStats.FactorAnalysis,)
factor_statistics!(portfolio; cov_opt = cov_opt, mu_opt = mu_opt, factor_opt = factor_opt)
fca = copy(portfolio.loadings)

mvr_opt.pca_genfunc.args = (MultivariateStats.PPCA,)
factor_statistics!(portfolio; cov_opt = cov_opt, mu_opt = mu_opt, factor_opt = factor_opt)
ppca = copy(portfolio.loadings)

test = rand(10, 10)
using NearestCorrelationMatrix
Newton
nearest_cor(test, Newton)
wak = PortfolioOptimiser.posdef_nearest(test)
wak2 = PortfolioOptimiser.posdef_nearest(test)
wak3 = PortfolioOptimiser.posdef_nearest(test,
                                         JuMPAlgorithm(optimizer_with_attributes(Clarabel.Optimizer)))
wak4 = PortfolioOptimiser.posdef_psd(test,
                                     Dict(:clarabel => Dict(:solver => Clarabel.Optimizer)))

w1 = optimise!(portfolio, OptimiseOpt(; type = :RP, class = :FC))
portfolio.f_risk_budget = 1:3
w2 = optimise!(portfolio, OptimiseOpt(; type = :RP, class = :FC))

loadings_opt.method = :BReg
factor_statistics!(portfolio; cov_f_opt = cov_f_opt, mu_f_opt = mu_f_opt,
                   cov_fm_opt = cov_fm_opt, mu_fm_opt = mu_fm_opt, factor_opt = factor_opt)
w3 = optimise!(portfolio, OptimiseOpt(; type = :RP, class = :FC))
portfolio.f_risk_budget = 1:5
w4 = optimise!(portfolio, OptimiseOpt(; type = :RP, class = :FC))

w1t = [-0.21773678293532656, 0.2678286433406289, 0.22580201434765002, 0.13009711209340935,
       -0.278769279772971, -0.331380980832027, -0.051650987176009834, 0.13460603215061595,
       -0.37358551690369, -0.9529827367064128, 0.30660697444264795, -0.09510590387915907,
       -0.15495259672856798, 0.23632593974465374, 0.29817635929005415, 0.12937409438378805,
       0.340897478693138, 0.3861788696698697, 0.30571854158323103, 0.6945527251944786]
w2t = [-0.060392894539612385, 0.20233103102025976, 0.27317239156127254, 0.14623348260674732,
       -0.3038293219871962, -0.3213998509220644, -0.030081946832234717,
       0.046976552558119423, -0.4473942798228809, -0.9617196645566132, 0.4467481399314188,
       -0.07727045650413009, -0.14070387560969616, 0.1455351423668605, 0.2583135617369249,
       0.1530902041977934, 0.07799306488349253, 0.5004440951067239, 0.371710254291287,
       0.7202443705135283]
w3t = [-2.30147257861377, 1.015594350375363, 1.4946718630838827, -0.5828485475120959,
       -0.4635674617121809, -0.07533572762551083, -0.5191177391071375, 1.1904188964246432,
       -0.4104970685871811, -0.8832844775538449, -0.9184915810813326, -0.8998509840079932,
       -0.6699607121244088, 1.0150741280736368, 0.8241685984816146, 0.495292306033691,
       2.8349366425935556, -0.8499048852808291, 0.9994635077334864, -0.29528852959358637]
w4t = [-2.2577765287237592, 0.9178982669292839, 1.2033733533210704, -0.3633175736239475,
       -0.43199497771239365, -0.5355766908797392, -0.46103809474646834, 1.0896247234056198,
       -0.1390372227481579, -0.6979034498174019, -0.7912768221485822, -0.76610343722116,
       -0.5799298212148584, 1.0421468464220744, 0.6556120082685148, 0.3203723397086974,
       2.6063724055147026, -0.8934040981887952, 1.1777198625438983, -0.0957610890885971]

@test isapprox(w1.weights, w1t; rtol = 5e-5)
@test isapprox(w2.weights, w2t; rtol = 5e-5)
@test isapprox(w3.weights, w3t; rtol = 5e-5)
@test isapprox(w4.weights, w4t; rtol = 1e-4)

asset_statistics!(portfolio; calc_kurt = false)
loadings_opt.method = :MVR
loadings_opt.mvr_opt.pca_genfunc.kwargs = (; pratio = 0.95)
factor_statistics!(portfolio; cov_f_opt = cov_f_opt, cov_fm_opt = cov_fm_opt,
                   factor_opt = factor_opt)

w1 = optimise!(portfolio, OptimiseOpt(; type = :RP, class = :FC))

loadings_opt.method = :FReg
factor_statistics!(portfolio; cov_f_opt = cov_f_opt, cov_fm_opt = cov_fm_opt,
                   factor_opt = factor_opt)

w2 = optimise!(portfolio, OptimiseOpt(; type = :RP, class = :FC))

loadings_opt.method = :BReg
factor_statistics!(portfolio; cov_f_opt = cov_f_opt, cov_fm_opt = cov_fm_opt,
                   factor_opt = factor_opt)

w3 = optimise!(portfolio, OptimiseOpt(; type = :RP, class = :FC))

w4 = optimise!(portfolio, OptimiseOpt(; type = :RP))

#################

portfolio = Portfolio(; prices = prices)

###################

cor_opt = CorOpt(; method = :Gerber1, gerber = GerberOpt(; normalise = true))
cor_opt = CorOpt(; method = :Gerber2, gerber = GerberOpt(; normalise = true))

cov_opt = CovOpt(; method = :Gerber1)

cluster_opt = ClusterOpt(; linkage = :DBHT)
port_opt = OptimiseOpt(; obj = :Sharpe, rf = rf, type = :Trad, kelly = :Exact)

using StructTypes
function StructTypes.StructType(::Type{typeof(cor_opt.estimation.cor_genfunc.func)})
    return StructTypes.StringType()
end
function StructTypes.StructType(::Type{typeof(cor_opt.estimation.cor_genfunc.func)})
    return StructTypes.StringType()
end
function StructTypes.StructType(::Type{typeof(cor_opt.estimation.dist_genfunc.func)})
    return StructTypes.StringType()
end
function StructTypes.StructType(::Type{typeof(cor_opt.gerber.mean_func.func)})
    return StructTypes.StringType()
end
function StructTypes.StructType(::Type{typeof(cor_opt.gerber.std_func.func)})
    return StructTypes.StringType()
end
function StructTypes.StructType(::Type{typeof(cor_opt.gerber.posdef.genfunc.func)})
    return StructTypes.StringType()
end
StructTypes.StructType(::Type{typeof(cor_opt.denoise.kernel)}) = StructTypes.StringType()
function StructTypes.StructType(::Type{typeof(cov_opt.estimation.genfunc.func)})
    return StructTypes.StringType()
end

println(JSON3.write(AllocOpt(;); allow_inf = true))

cor_opt.denoise.kernel

propertynames(cov_opt.estimation)

asset_statistics!(portfolio; calc_kurt = false,
                  cov_opt = CovOpt(; method = :SB1, gerber = GerberOpt(; normalise = false),
                                   sb = SBOpt(;)))
cov1 = copy(portfolio.cov)

asset_statistics!(portfolio; calc_kurt = false,
                  cov_opt = CovOpt(; method = :SB1, gerber = GerberOpt(; normalise = false),
                                   sb = SBOpt(;)))
cov2 = copy(portfolio.cov)

plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber0, gerber = GerberOpt(; normalise = false),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber0, gerber = GerberOpt(; normalise = true),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :SB0, gerber = GerberOpt(; normalise = false),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :SB0, gerber = GerberOpt(; normalise = true),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber_SB0,
                               gerber = GerberOpt(; normalise = false), sb = SBOpt(;)),
              cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber_SB0,
                               gerber = GerberOpt(; normalise = true), sb = SBOpt(;)),
              cluster_opt = ClusterOpt(; linkage = :DBHT))

plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber1, gerber = GerberOpt(; normalise = false),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber1, gerber = GerberOpt(; normalise = true),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :SB1, gerber = GerberOpt(; normalise = false),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :SB1, gerber = GerberOpt(; normalise = true),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber_SB1,
                               gerber = GerberOpt(; normalise = false), sb = SBOpt(;)),
              cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber_SB1,
                               gerber = GerberOpt(; normalise = true), sb = SBOpt(;)),
              cluster_opt = ClusterOpt(; linkage = :DBHT))

plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber2, gerber = GerberOpt(; normalise = false),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))
plot_clusters(portfolio;
              cor_opt = CorOpt(; method = :Gerber2, gerber = GerberOpt(; normalise = true),
                               sb = SBOpt(;)), cluster_opt = ClusterOpt(; linkage = :DBHT))

asset_statistics!(portfolio; calc_kurt = false,
                  cov_opt = CovOpt(; method = :SB0, sb = SBOpt(; c1 = 0.8, c2 = 0.9)))
sb0 = copy(portfolio.cov)

asset_statistics!(portfolio; calc_kurt = false, cov_opt = CovOpt(; method = :SB1))
sb1 = copy(portfolio.cov)

asset_statistics!(portfolio; calc_kurt = false, cov_opt = CovOpt(; method = :Gerber0))
g0 = copy(portfolio.cov)

asset_statistics!(portfolio; calc_kurt = false, cov_opt = CovOpt(; method = :Gerber1))
g1 = copy(portfolio.cov)

asset_statistics!(portfolio; calc_kurt = false, cov_opt = CovOpt(; method = :Gerber2))
g2 = copy(portfolio.cov)

portfolio.cov

frontier = efficient_frontier!(portfolio,
                               OptimiseOpt(; type = :Trad, rm = :CVaR, near_opt = true,
                                           n = 5); points = 10)
frontiert = reshape([0.051486360012397016, 0.05236983288917021, 0.05132527587399171,
                     0.04986422716287365, 0.0570083550281748, 0.045796008131193965,
                     0.048068240664882085, 0.054942698822322146, 0.048930962313161616,
                     0.0480730247923202, 0.05347248935088241, 0.04327427528324469,
                     0.03961302483795674, 0.049721873111004, 0.04044184014873579,
                     0.05357451247151647, 0.053409588164720044, 0.05403200381208819,
                     0.0513814825682843, 0.05321392456108, 0.026821268798996076,
                     0.030277546224179892, 0.024110298236767504, 0.025200174485100337,
                     0.31781909387081353, 0.009521499550705926, 0.03860941857585781,
                     0.06023468191033268, 0.01779430032575157, 0.01575469608851992,
                     0.0722861132897799, 0.00819289606352098, 0.0056255371943901775,
                     0.01952983067904024, 0.006286692485911797, 0.08181427464482686,
                     0.0708798405394817, 0.09271216923235816, 0.030463759170697696,
                     0.046065908632967174, 0.02257872852038554, 0.024536111583012415,
                     0.029400229801466445, 0.025726947684143738, 0.48305625882020437,
                     0.010183813115227603, 0.14591193416608383, 0.01743842685190719,
                     0.02485440605927779, 0.01807893337221225, 0.016667535309586895,
                     0.010600203390568098, 0.007523273071820424, 0.01348604256468001,
                     0.0078053379956584895, 0.04209240187898889, 0.03264573581864106,
                     0.018289629452301608, 0.02697553593745076, 0.022148514606382497,
                     0.014013249672769559, 0.014958382016165537, 0.01841752710637425,
                     0.016524037217702566, 0.5149873461783283, 0.006405976988549117,
                     0.2581687568487945, 0.010418560222877753, 0.015630531492763264,
                     0.011218062480867912, 0.010053571291418718, 0.006786733521275428,
                     0.004806947357755105, 0.008309759095310669, 0.004963807223459894,
                     0.024223287105782892, 0.01938348921393103, 0.010940957426087802,
                     0.016376042899818352, 0.01341297463996747, 0.013705614739666137,
                     0.01457432851387161, 0.017960327728509074, 0.016294074841513725,
                     0.4275436268346442, 0.006325912155529782, 0.35118351295240374,
                     0.010148253965269931, 0.015266596621300794, 0.011008571260714623,
                     0.009815309133938126, 0.006702068862146002, 0.004764770631800752,
                     0.008160951964809232, 0.004929037960149233, 0.02320956714853085,
                     0.018791844825991565, 0.010656903238207656, 0.01592250360425309,
                     0.013036223016749877, 0.013554075423113217, 0.014396084352292829,
                     0.017711980783940522, 0.0161759895553917, 0.3654973533468737,
                     0.006300460121646455, 0.41588289621708935, 0.010035193992124537,
                     0.01511337361258717, 0.010916281514823229, 0.009709630367625439,
                     0.006669503537459898, 0.004752988819649873, 0.008098702507701094,
                     0.00492184924431572, 0.02265453206113468, 0.018483477580504094,
                     0.010537420565674754, 0.01570804931151351, 0.012880157084538213,
                     0.013393451640655354, 0.014212370934066495, 0.017453791369429785,
                     0.01599064164771719, 0.3342206137150228, 0.006244873165037023,
                     0.44992011891561634, 0.009906436778978085, 0.014923301817736568,
                     0.010800667517320332, 0.009593622103020144, 0.006615331583175004,
                     0.004714295554363717, 0.008014788859958061, 0.004886151718028586,
                     0.02227727835404245, 0.018217692262057873, 0.010404188109198109,
                     0.015485194434977053, 0.0127251895195991, 0.013219786577852288,
                     0.014040310612926713, 0.017268474735408938, 0.015815957803047603,
                     0.3112376210069283, 0.006183192397466366, 0.47551503526062255,
                     0.009785091923076392, 0.014745627777312004, 0.010682448633460056,
                     0.009481947125956531, 0.006549133069360527, 0.00467367551739762,
                     0.00792412116318791, 0.004842492994898218, 0.02194439632282262,
                     0.017974475149029796, 0.010273919741916538, 0.015283057325334635,
                     0.012559234861994378, 0.013048478140206, 0.013854832126576695,
                     0.01703853587914295, 0.015617020614165346, 0.29558456027489594,
                     0.006111767158909742, 0.49393025152843606, 0.009659092048917668,
                     0.014557783735968648, 0.01054868135128449, 0.009359445370684135,
                     0.006473959448772166, 0.004622300082268987, 0.007830217556712617,
                     0.004788840759744425, 0.02163031654357611, 0.01772571058792744,
                     0.010142630347936782, 0.015085137290175736, 0.012390439153697941,
                     0.012879927764277755, 0.013673903235410852, 0.01681093862961253,
                     0.015413410656460427, 0.2843568453030244, 0.006036800615322954,
                     0.5079302504339147, 0.009533387434620046, 0.014368969457392269,
                     0.010416381319155859, 0.009242115746001067, 0.0063964700498197365,
                     0.00456858581674156, 0.0077323479994042595, 0.004733448544173203,
                     0.02129758789417074, 0.017485726294248365, 0.010012155696351537,
                     0.014883553766076174, 0.012227193343821576], 20, :)
@test isapprox(Matrix(frontier[:weights][!, 2:end]), frontiert)

frontier2 = efficient_frontier!(portfolio,
                                OptimiseOpt(; type = :Trad, rm = :CVaR, near_opt = false);
                                points = 50)

f1 = plot_frontier_area(portfolio; rm = :CVaR)
f2 = plot_frontier_area(portfolio; rm = :CVaR, near_opt = true)

f3 = plot_frontier(portfolio; rm = :CVaR)
f4 = plot_frontier(portfolio; rm = :CVaR, near_opt = true)

A = PortfolioOptimiser.block_vec_pq(portfolio.kurt, 20, 20)
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

for rtol ∈
    [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 2.5e-1, 5e-1,
     1e0]
    a1, a2 = w13.weights, w7.weights
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

r = collect(range(; start = 7.5, stop = 8, length = 2))
f(r, 1)
display(r * 10)
