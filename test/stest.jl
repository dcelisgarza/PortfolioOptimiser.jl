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
                                                              :params => Dict("verbose" => true
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
@time asset_statistics!(portfolio)

skew = portfolio.skew
V = portfolio.V
sskew = portfolio.sskew
SV = portfolio.SV
kurt = portfolio.kurt
skurt = portfolio.skurt
clusters = rand(1:20, 10)
N = size(portfolio.returns, 2)
Nc = length(clusters)

idx = Int[]
sizehint!(idx, Nc^2)
for c ∈ clusters
    append!(idx, (((c - 1) * N + 1):(c * N))[clusters])
end

skew[clusters, idx]
@time skew2, V2, sskew2, SV2 = PortfolioOptimiser.coskew_mtx(portfolio.returns[:, clusters],
                                                             portfolio.mu[clusters])
@test isapprox(skew[clusters, idx], skew2)
@test isapprox(sskew[clusters, idx], sskew2)

V = zeros(eltype(skew), Nc, Nc)
for i ∈ 1:Nc
    j = (i - 1) * Nc + 1
    k = i * Nc
    vals, vecs = eigen(skew[clusters, idx][:, j:k])
    vals = clamp.(real.(vals), -Inf, 0) .+ clamp.(imag.(vals), -Inf, 0)im
    V .-= real(vecs * Diagonal(vals) * transpose(vecs))
end
isapprox(V, V2)

@time kurt2, skurt2, L_2, S_2 = PortfolioOptimiser.cokurt_mtx(portfolio.returns[:,
                                                                                clusters],
                                                              portfolio.mu[clusters])
@test isapprox(kurt[idx, idx], kurt2)
@test isapprox(skurt[idx, idx], skurt2)

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
    a1, a2 = [3.5231149124516024e-9, 0.13968049714554842, 0.06864917026613944,
              0.0916435270956175, 0.1941761525025078, 9.289060359835379e-17,
              0.037602412536222, 0.08831489903626148, 1.1382874727175016e-15,
              2.6193402629947552e-8, 1.5314935309760717e-8, 1.0105364646520988e-16,
              2.4427988768871966e-16, 2.2856517165425115e-16, 5.351920200768704e-17,
              1.0771336730926371e-8, 0.24211921739567008, 0.11620888241657644,
              1.9100609590696612e-8, 0.021605166702055836],
             [3.5235313037544633e-9, 0.13968049714572486, 0.06864917032501697,
              0.09164352710177702, 0.1941761526042709, 9.28905966595001e-17,
              0.037602412527191424, 0.08831489917767826, 1.138287388065542e-15,
              2.6193401535198235e-8, 1.5314934663760045e-8, 1.0105363891957816e-16,
              2.442785257037521e-16, 2.2856515462968445e-16, 5.3519197995688067e-17,
              1.07713362810882e-8, 0.2421192174115078, 0.11620888251451723,
              1.910060878501064e-8, 0.021605166288501187]
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
