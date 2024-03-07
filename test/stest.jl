using COSMO, CSV, Clarabel, DataFrames, OrderedCollections, Test, TimeSeries,
      PortfolioOptimiser, LinearAlgebra, PyCall

prices_assets = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
prices_factors = TimeArray(CSV.File("./test/assets/factor_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

portfolio = Portfolio(; prices = prices_assets[(end - 400):end],
                      f_prices = prices_factors[(end - 400):end],
                      solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                              :params => Dict("verbose" => true
                                                                              #   "max_step_fraction" => 0.75,
                                                                              #   "max_iter"=>150,
                                                                              #   "tol_gap_abs" => 1e-8,
                                                                              #   "tol_gap_rel" => 1e-8,
                                                                              #   "tol_feas" => 1e-8,
                                                                              #   "tol_ktratio" => 1e-8,
                                                                              #   "equilibrate_max_iter" => 30,
                                                                              #   "reduced_tol_gap_abs" => 1e-6,
                                                                              #   "reduced_tol_gap_rel" => 1e-5,
                                                                              #   "reduced_tol_feas" => 1e-6,
                                                                              #   "reduced_tol_ktratio" => 1e-6,
                                                                              ))))
asset_statistics!(portfolio; calc_kurt = false)

w1 = optimise!(portfolio,
               OptimiseOpt(; obj = :Min_Risk, type = :Trad, rm = :RTG, owa_approx = false);
               string_names = true)

portfolio.owa_w = owa_rtg(400)
w2 = optimise!(portfolio,
               OptimiseOpt(; obj = :Min_Risk, type = :Trad, rm = :OWA, owa_approx = false);
               string_names = true)
w3 = optimise!(portfolio,
               OptimiseOpt(; obj = :Min_Risk, type = :Trad, rm = :RTG, owa_approx = true);
               string_names = true)

# portfolio.solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
#                                                   :params => Dict("verbose" => true,
#                                                                   "max_step_fraction" => 0.75
#                                                                   #   "max_iter"=>250,
#                                                                   #   "tol_gap_abs" => 1e-10,
#                                                                   #   "tol_gap_rel" => 1e-10,
#                                                                   #   "tol_feas" => 1e-10,
#                                                                   #   "tol_ktratio" => 1e-10,
#                                                                   #   "equilibrate_max_iter" => 30,
#                                                                   #   "reduced_tol_gap_abs" => 1e-10,
#                                                                   #   "reduced_tol_gap_rel" => 1e-10,
#                                                                   #   "reduced_tol_feas" => 1e-10,
#                                                                   #   "reduced_tol_ktratio" => 1e-10,
#                                                                   )))
# portfolio.owa_p = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]

w4 = optimise!(portfolio,
               OptimiseOpt(; obj = :Min_Risk, type = :Trad, rm = :OWA, owa_approx = true);
               string_names = true)

display(DataFrame(; tickers = w1.tickers, w1 = w1.weights, w2 = w2.weights, w3 = w3.weights,
                  w4 = w4.weights, d12 = w1.weights - w2.weights,
                  d13 = w1.weights - w3.weights, d34 = w3.weights - w4.weights))

loadings_opt = LoadingsOpt(;)
factor_opt = FactorOpt(; loadings_opt = loadings_opt)
posdef = PosdefFixOpt(; method = :Nearest)
cov_f_opt = CovOpt(; posdef = posdef)
cov_fm_opt = CovOpt(; posdef = posdef)

test = factor_risk_contribution(portfolio.optimal[:RP].weights, portfolio.assets,
                                portfolio.returns, portfolio.f_assets, portfolio.f_returns,
                                DataFrame(); loadings_opt = loadings_opt, rm = :SD,
                                rf = 0.0, sigma = portfolio.cov,
                                solvers = portfolio.solvers)

py"""
import riskfolio as rk
import numpy as np
import pandas as pd
import riskfolio.src.ParamsEstimation as pe
from scipy.linalg import null_space
from numpy.linalg import pinv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
from scipy.linalg import sqrtm, norm, null_space
import riskfolio.src.OwaWeights as owa

"""

py"""
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.
model = 'FC' # Factor Contribution Model
rm = 'MV' # Risk measure used, this time will be variance
rf = $rf # Risk free rate
b_f = None # Risk factor contribution vector
port = rk.Portfolio(returns = pd.DataFrame($(portfolio.returns), columns = $(portfolio.assets)), factors = pd.DataFrame($(portfolio.f_returns), columns = $(portfolio.f_assets)))

port.assets_stats(method_mu=method_mu,
                  method_cov=method_cov)

feature_selection = 'PCR' # Method to select best model, could be PCR or Stepwise
n_components = 0.95 # 95% of explained variance. See PCA in scikit learn for more information
port.factors_stats(method_mu=method_mu,
                   method_cov=method_cov,
                   feature_selection=feature_selection,
                   dict_risk=dict(n_components=n_components)
                  )

w1 = port.rp_optimization(model=model,
                         rm=rm,
                         rf=rf,
                         b_f=b_f,
                         )

feature_selection = 'stepwise' # Method to select best model, could be PCR or Stepwise
stepwise = 'Forward' # Forward or Backward regression

port.factors_stats(method_mu=method_mu,
                    method_cov=method_cov,
                    feature_selection=feature_selection,
                    dict_risk=dict(stepwise=stepwise)
                    )

w2 = port.rp_optimization(model=model,
                         rm=rm,
                         rf=rf,
                         b_f=b_f,
                         )

feature_selection = 'stepwise' # Method to select best model, could be PCR or Stepwise
stepwise = 'Backward' # Forward or Backward regression

port.factors_stats(method_mu=method_mu,
                    method_cov=method_cov,
                    feature_selection=feature_selection,
                    dict_risk=dict(stepwise=stepwise)
                    )

w3 = port.rp_optimization(model=model,
                         rm=rm,
                         rf=rf,
                         b_f=b_f,
                         )

w4 = port.rp_optimization(model="Classic",
                         rm=rm,
                         rf=rf,
                         b_f=b_f,
                         )

"""

asset_statistics!(portfolio; calc_kurt = false)
loadings_opt.method = :PCR
loadings_opt.pcr_opt.pca_genfunc.kwargs = (; pratio = 0.95)
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

py"""


def Factors_Risk_Contribution(
    w,
    cov=None,
    returns=None,
    factors=None,
    B=None,
    const=False,
    rm="MV",
    rf=0,
    alpha=0.05,
    a_sim=100,
    beta=None,
    b_sim=None,
    kappa=0.3,
    solver="CLARABEL",
    feature_selection="stepwise",
    stepwise="Forward",
    criterion="pvalue",
    threshold=0.05,
    n_components=0.95,
):
    w_ = np.array(w, ndmin=2)
    if w_.shape[0] == 1 and w_.shape[1] > 1:
        w_ = w_.T
    if w_.shape[0] > 1 and w_.shape[1] > 1:
        raise ValueError("weights must have n_assets x 1 size")

    RM = rk.Risk_Margin(
        w=w,
        cov=cov,
        returns=returns,
        rm=rm,
        rf=rf,
        alpha=alpha,
        a_sim=a_sim,
        beta=beta,
        b_sim=b_sim,
        kappa=kappa,
        solver=solver,
    ).reshape(-1, 1)

    if B is None:
        B = pe.loadings_matrix(
            X=factors,
            Y=returns,
            feature_selection=feature_selection,
            stepwise=stepwise,
            criterion=criterion,
            threshold=threshold,
            n_components=n_components,
        )
        const = True
    elif not isinstance(B, pd.DataFrame):
        raise ValueError("B must be a DataFrame")

    if const == True or factors.shape[1] + 1 == B.shape[1]:
        B = B.iloc[:, 1:].to_numpy()

    if feature_selection == "PCR":
        scaler = StandardScaler()
        scaler.fit(factors)
        factors_std = scaler.transform(factors)
        pca = PCA(n_components=n_components)
        pca.fit(factors_std)
        V_p = pca.components_.T
        std = np.array(np.std(factors, axis=0, ddof=1), ndmin=2)
        B = (pinv(V_p) @ (B.T * std.T)).T

    B1 = pinv(B.T)
    B2 = pinv(null_space(B.T).T)
    print(B2)
    B3 = pinv(B2.T)

    RC_F = (B.T @ w_) * (B1.T @ RM)
    RC_OF = np.array(((B2.T @ w.to_numpy()) * (B3.T @ RM)).sum(), ndmin=2)
    RC_F = np.vstack([RC_F, RC_OF]).ravel()

    return RC_F

test = Factors_Risk_Contribution(
    w=pd.DataFrame($(portfolio.optimal[:Trad].weights)),
    cov=pd.DataFrame($(portfolio.cov), columns = $(portfolio.assets)),
    returns=pd.DataFrame($(portfolio.returns), columns = $(portfolio.assets)),
    factors=pd.DataFrame($(portfolio.f_returns), columns = $(portfolio.f_assets)),
    B=None,
    const=False,
    rm="MV",
    rf=0,
    alpha=0.05,
    a_sim=100,
    beta=None,
    b_sim=None,
    kappa=0.3,
    solver="CLARABEL",
    feature_selection="PCR",
    stepwise="Forward",
    criterion="pvalue",
    threshold=0.05,
    n_components=0.99,
)
"""
loadings_opt.method = :PCR
loadings_opt.pcr_opt.pca_genfunc.kwargs = (; pratio = 0.99)
test = factor_risk_contribution(portfolio.optimal[:Trad].weights, portfolio.assets,
                                portfolio.returns, portfolio.f_assets, portfolio.f_returns,
                                DataFrame(); loadings_opt = loadings_opt, rm = :SD,
                                rf = 0.0, sigma = portfolio.cov,
                                solvers = portfolio.solvers)

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

for rtol ∈
    [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2.5e-1, 5e-1, 1e0]
    a1, a2 = [0.0425354272767062, 0.04991571895368152, 0.04298329828063256,
              0.03478231208822956, 0.04824894701525979, 0.05320453060693792,
              0.03659423924135794, 0.06323743284904297, 0.04112041989604676,
              0.044360488824348794, 0.08117513451607743, 0.04385016868298808,
              0.025563101387777416, 0.06928268651446567, 0.036317664547475315,
              0.048489360135118655, 0.04932838211979579, 0.06279913028520855,
              0.04589944098267767, 0.08031211579617145],
             [0.042304471409681986, 0.050291959151292795, 0.04246282744021459,
              0.034154217192553946, 0.04711274092122714, 0.053021558707734416,
              0.03709975559975993, 0.06221043802717917, 0.04085191768667458,
              0.044856782603669765, 0.08316716558090284, 0.04356886923048706,
              0.02551608637477467, 0.07053717563638451, 0.03671465423999034,
              0.04877993854502053, 0.049320416165216964, 0.06298168266461707,
              0.04598643874949792, 0.07906090407311984]
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

r = range(; start = 7, stop = 8, length = 3)
f(r, 1)
