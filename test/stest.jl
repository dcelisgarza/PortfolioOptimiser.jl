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

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

struct POCorDist <: Distances.UnionMetric end
function Distances.pairwise(::POCorDist, mtx, i)
    return sqrt.(clamp!((1 .- mtx) / 2, 0, 1))
end
dbht_d(corr, dist) = 2 .- (dist .^ 2) / 2

portfolio = HCPortfolio(; prices = prices[(end - 50):end],
                        solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                                :params => Dict("verbose" => false,
                                                                                "max_step_fraction" => 0.75))))
asset_statistics!(portfolio; cor_opt = CorOpt(; dist = DistOpt(; method = POCorDist())))

cluster_opt = ClusterOpt(; linkage = :DBHT, genfunc = GenericFunction(; func = dbht_d))
rm = :SSkew
w1 = optimise!(portfolio; type = :HRP, rm = rm, rf = rf, cluster_opt = cluster_opt)
w2 = optimise!(portfolio; type = :HERC, rm = rm, rf = rf, cluster = false)
w3 = optimise!(portfolio; type = :NCO,
               nco_opt = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
               cluster = false)
w4 = optimise!(portfolio; type = :NCO,
               nco_opt = OptimiseOpt(; rm = rm, obj = :Utility, rf = rf, l = l),
               cluster = false)
w5 = optimise!(portfolio; type = :NCO,
               nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
               cluster = false)
w6 = optimise!(portfolio; type = :NCO,
               nco_opt = OptimiseOpt(; rm = rm, obj = :Max_Ret, rf = rf, l = l),
               cluster = false)
w7 = optimise!(portfolio; type = :NCO,
               nco_opt = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
               cluster = false)
w8 = optimise!(portfolio; type = :NCO,
               nco_opt = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
               nco_opt_o = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
               cluster = false)
w9 = optimise!(portfolio; type = :NCO,
               nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
               nco_opt_o = OptimiseOpt(; rm = rm, obj = :Min_Risk, rf = rf, l = l),
               cluster = false)
w10 = optimise!(portfolio; type = :NCO,
                nco_opt = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                nco_opt_o = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                cluster = false)
w11 = optimise!(portfolio; type = :NCO,
                nco_opt = OptimiseOpt(; rm = rm, obj = :Sharpe, rf = rf, l = l),
                nco_opt_o = OptimiseOpt(; rm = rm, obj = :Equal, rf = rf, l = l),
                cluster = false)

w1t = [0.03073561997154973, 0.06271788007944147, 0.059839304178611136, 0.019020703463231065,
       0.054114343500047124, 0.07478195087196789, 0.015025906923294915,
       0.037423537539572865, 0.02195663364580543, 0.07120487867873436, 0.14238956696393482,
       0.022992175030255378, 0.02427916555268004, 0.040541588877064376,
       0.010704183865328891, 0.053000134123666075, 0.07802885174200914, 0.06680453879901624,
       0.030056876315924044, 0.08438215987786517]
w2t = [0.03919512340681375, 0.10686864893708017, 0.03383085583616111, 0.028829387374514404,
       0.0324748288822625, 0.039829364218452665, 0.0367071612631816, 0.04152962896457131,
       0.03151854338033432, 0.10730692465497858, 0.0593418669070617, 0.016445741516008028,
       0.0377702921849933, 0.04555435204684619, 0.018967357142203968, 0.07189904439937453,
       0.12128239171624568, 0.051485815846593244, 0.03839095369301088, 0.04077171762931205]
w3t = [9.639142250374431e-7, 3.821423686703969e-6, 5.747448406456808e-7,
       0.002352011118061013, 0.026164457969268935, 1.031686400035221e-5,
       2.2517736637352832e-11, 0.15312219714162972, 1.8160922231624558e-6,
       1.8417694901450321e-6, 0.350100630690602, 0.0068901895442682005,
       4.406229353280931e-9, 0.08362400782979643, 9.542927649989849e-7,
       4.818064488496433e-7, 8.585537441700159e-6, 0.2995924956347618,
       1.0336771283354825e-5, 0.07811431242646]
w4t = [3.930740279153447e-11, 4.6437148972040797e-11, 1.1200848920020919e-10,
       8.438496711900486e-11, 0.7769738594452223, 1.1992697975099713e-21,
       0.22302614002619753, 8.25942589163891e-12, 1.785404804253808e-19,
       1.0725226098366154e-19, 2.5247366894795724e-21, 9.585011117191017e-22,
       1.4491646595304253e-20, 2.25217746957559e-22, 1.358360655354307e-21,
       8.721895527116661e-11, 1.0120357182367101e-10, 3.797391966007319e-14,
       1.981122476678983e-11, 2.99109628188828e-11]
w5t = [5.078506378670576e-10, 7.129857385069814e-10, 6.761039899688519e-10,
       5.13410014938181e-10, 0.7305480078641171, 4.2591678548089477e-19,
       0.0062235536602822895, 2.428853706836181e-9, 5.4783989078124194e-18,
       0.023338990470018972, 6.987294100997418e-18, 4.792118583212526e-19,
       4.2889387228036694e-11, 9.806571219930943e-19, 2.43472729870113e-19,
       0.1640032932998666, 0.07588614463961668, 2.6595179086652414e-9,
       1.7707097444429373e-9, 7.537773623308645e-10]
w6t = [2.3060896376689725e-15, 1.0202741343537544e-8, 3.311158818575788e-15,
       2.9681833532660824e-15, 3.933662877761243e-7, 6.022146260993428e-17,
       0.999999556914032, 1.6453580554840533e-15, 7.115125356552783e-15,
       6.931495294112464e-16, 1.3813874799307706e-16, 6.19589282485867e-17,
       1.100364137175803e-16, 9.224577469128846e-17, 4.0464976495275276e-17,
       1.428619860674334e-8, 1.4466444340415644e-8, 1.7460856825889383e-15,
       1.076427343104852e-8, 2.17351197577772e-15]
w7t = [0.03581114005948653, 0.09165618425796519, 0.031837470062627035, 0.03284253312462643,
       0.034009977404041346, 0.04053550659280367, 0.03517279051716404, 0.05778092197461075,
       0.030779690659077617, 0.09970888642242545, 0.0693703667745115, 0.02373010226414456,
       0.03722266543223331, 0.04551526159136513, 0.02420856429532943, 0.06868450200987816,
       0.10203132992528399, 0.05982317887939177, 0.03605499049226807, 0.043223937260766045]
w8t = [6.5559159346787484e-15, 0.30800585983405954, 3.909039581833177e-15,
       1.599684574303155e-11, 1.7795357975581224e-10, 1.1010312831529227e-14,
       1.8149243326868086e-6, 1.0414373251466815e-9, 1.9381610058292314e-15,
       9.300403859635975e-10, 3.736326723205436e-10, 7.35331418038803e-12,
       2.225018532610631e-12, 8.924480214151074e-11, 1.0184356288059165e-15,
       2.432983378460604e-10, 0.6919923197914748, 2.0376327737728045e-9,
       1.1031558184136092e-14, 5.312826102791567e-10]
w9t = [7.842532904347133e-11, 3.940507192242069e-9, 1.0440801670272057e-10,
       7.928384125861935e-11, 0.11281558715654076, 6.626462445547879e-11,
       0.034396140953675866, 3.7507809768065373e-10, 8.523356172347408e-10,
       0.01967034572985398, 1.0870912707509277e-9, 7.455633332322455e-11,
       3.614762497978687e-11, 1.5257176548017964e-10, 3.78797679734014e-11,
       0.13822369413051988, 0.419405161413342, 4.106986415534946e-10, 0.27548906320441635,
       1.1640280282919093e-10]
w10t = [0.01712792814440894, 0.34391989640653353, 0.015227381720509945,
        0.015708088223512353, 0.016266459213578697, 3.192428814167094e-11,
        0.13197824640993092, 0.02763574346013218, 2.4240962950894444e-11,
        4.2965143396829856e-10, 5.463357346559132e-11, 1.8688963972300786e-11,
        1.603946463841395e-10, 3.584616174857993e-11, 1.9065783236010362e-11,
        2.959655436820271e-10, 0.38285037395144494, 0.028612524133950636,
        2.839559690182128e-11, 0.02067335723719085]
w11t = [1.6325276024268907e-10, 2.3622806445933734e-9, 2.1733918271144124e-10,
        1.650398676802274e-10, 0.23484065959731076, 6.335681049978989e-11,
        0.020620020230782132, 7.807749804785943e-10, 8.149335580954624e-10,
        0.02861732343823022, 1.0393877004949617e-9, 7.128466388717538e-11,
        5.258922693973869e-11, 1.458766349704544e-10, 3.62175338800758e-11,
        0.20109418594288503, 0.2514277088492203, 8.549238833842325e-10, 0.2634000949320061,
        2.4230792645214744e-10]

@test isapprox(w1.weights, w1t)
@test isapprox(w2.weights, w2t)
@test isapprox(w3.weights, w3t)
@test isapprox(w4.weights, w4t)
@test isapprox(w5.weights, w5t)
@test isapprox(w6.weights, w6t)
@test isapprox(w7.weights, w7t)
@test isapprox(w8.weights, w8t)
@test isapprox(w9.weights, w9t)
@test isapprox(w10.weights, w10t)
@test isapprox(w11.weights, w11t)

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
    a1, a2 = 19.999587506769174, 20
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
