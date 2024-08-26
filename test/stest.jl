using CSV, TimeSeries, JuMP, Test, Clarabel, StatsBase, PyCall, DataFrames,
      PortfolioOptimiser, Clustering

rms = EVaR()
for (i, r) ∈ enumerate(rms)
    println(i)
    println(r)
end
test = Trad()
Symbol(test)

String(test)
prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

portfolio = Portfolio(; prices = prices,
                      solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                       :params => Dict("verbose" => true,
                                                                       "max_step_fraction" => 0.75))))

using StatsBase, Statistics

ve = SimpleVariance(; corrected = false)
w = eweights(1:size(portfolio.returns, 1), 0.03)
@test isapprox(var(ve, portfolio.returns, w), std(ve, portfolio.returns, w) .^ 2)
@test !isapprox(var(ve, portfolio.returns, w), var(ve, portfolio.returns))

asset_statistics!(portfolio)
rm = TG(; owa = OWASettings(; approx = true))
portfolio.risk_budget = []
@time w1 = optimise!(portfolio; type = RP(), rm = rm)

###
portfolio2 = Portfolio(; prices = prices[(end - 200):end],
                       solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                        :params => Dict("verbose" => true,
                                                                        "max_step_fraction" => 0.75))))
asset_statistics!(portfolio2)

rm = :TG
opt = OptimiseOpt(; type = :RP, rm = rm, owa_approx = true)

portfolio2.risk_budget = []
@time w2 = optimise!(portfolio2, opt)

#############
portfolio = HCPortfolio(; prices = prices,
                        solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                         :params => Dict("verbose" => true))))
asset_statistics!(portfolio; calc_kurt = false)

clustering_idx, clustering, k = cluster_assets(portfolio,
                                               ClusterOpt(; k_method = :Std_Sil,
                                                          linkage = :single))
# idx = sortperm(portfolio.assets[clustering.order])
# df = DataFrame([portfolio.assets[clustering.order][idx] clustering_idx[idx]],
#                [:Assets, :Cluster])
ret = randn(30000, 2000)
corr = cor(ret)
dist = ((1 .- corr) ./ 2) .^ 2

hclust_opt = HCType()
hclust_opt.k_method
clusteringt = hclust(dist; linkage = :ward, branchorder = hclust_opt.branchorder)
kt = PortfolioOptimiser.calc_k(hclust_opt, dist, clusteringt)
idxt = cutree(clusteringt; k = kt)

hclust_opt.k_method = StdSilhouette()
clusterings = hclust(dist; linkage = :ward, branchorder = hclust_opt.branchorder)
ks = PortfolioOptimiser.calc_k(hclust_opt, dist, clusterings)
idxs = cutree(clusterings; k = ks)

py"""
import numpy as np
import pandas as pd
import riskfolio as rp
import scipy.cluster.hierarchy as hr
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples
"""

py"""
def two_diff_gap_stat(dist, clustering, max_k=10):
    flag = False
    # Check if linkage matrix is monotonic
    if hr.is_monotonic(clustering):
        flag = True
    # cluster levels over from 1 to N-1 clusters
    cluster_lvls = pd.DataFrame(hr.cut_tree(clustering), index=dist.columns)
    level_k = cluster_lvls.columns.tolist()
    cluster_lvls = cluster_lvls.iloc[:, ::-1]  # reverse order to start with 1 cluster
    cluster_lvls.columns = level_k
    # Fix for nonmonotonic linkage matrices
    if flag is False:
        for i in cluster_lvls.columns:
            unique_vals, indices = np.unique(cluster_lvls[i], return_inverse=True)
            cluster_lvls[i] = indices
    cluster_lvls = cluster_lvls.T.drop_duplicates().T
    level_k = cluster_lvls.columns.tolist()
    cluster_k = cluster_lvls.nunique(axis=0).tolist()
    W_list = []
    n = dist.shape[0]


    # get within-cluster dissimilarity for each k
    for k in cluster_k:
        if k == 1:
            W_list.append(-np.inf)
        elif k > min(max_k, np.sqrt(n)) + 2:
            break
        else:
            level = cluster_lvls[level_k[cluster_k.index(k)]]  # get k clusters
            D_list = []  # within-cluster distance list

            for i in range(np.max(level.unique()) + 1):
                cluster = level.loc[level == i]
                # Based on correlation distance
                cluster_dist = dist.loc[cluster.index, cluster.index]  # get distance
                cluster_pdist = squareform(cluster_dist, checks=False)
                if cluster_pdist.shape[0] != 0:
                    D = np.nan_to_num(cluster_pdist.std())
                    D_list.append(D)  # append to list

            W_k = np.sum(D_list)
            W_list.append(W_k)
    W_list = pd.Series(W_list)
    gaps = W_list.shift(-2) + W_list - 2 * W_list.shift(-1)
    k_index = int(gaps.idxmax())
    k = cluster_k[k_index]
    node_k = level_k[k_index]


    if flag:
        clustering_inds = cluster_lvls[node_k].tolist()
    else:
        clustering_inds = hr.fcluster(clustering, k, criterion="maxclust")
        j = len(np.unique(clustering_inds))
        while k != j:
            j += 1
            clustering_inds = hr.fcluster(clustering, j, criterion="maxclust")
            k = len(np.unique(clustering_inds))
        unique_vals, indices = np.unique(clustering_inds, return_inverse=True)
        clustering_inds = indices

    return k, clustering_inds

def std_silhouette_score(dist, clustering, max_k=10):
    flag = False
    # Check if linkage matrix is monotonic
    if hr.is_monotonic(clustering):
        flag = True
    # cluster levels over from 1 to N-1 clusters
    cluster_lvls = pd.DataFrame(hr.cut_tree(clustering), index=dist.columns)
    level_k = cluster_lvls.columns.tolist()
    cluster_lvls = cluster_lvls.iloc[:, ::-1]  # reverse order to start with 1 cluster
    cluster_lvls.columns = level_k
    # Fix for nonmonotonic linkage matrices
    if flag is False:
        for i in cluster_lvls.columns:
            unique_vals, indices = np.unique(cluster_lvls[i], return_inverse=True)
            cluster_lvls[i] = indices
    cluster_lvls = cluster_lvls.T.drop_duplicates().T
    level_k = cluster_lvls.columns.tolist()
    cluster_k = cluster_lvls.nunique(axis=0).tolist()
    scores_list = []
    n = dist.shape[0]
    # print(f"level_k = {level_k}, cluster_k = {cluster_k}")

    # get within-cluster dissimilarity for each k
    for k in cluster_k:
        if k == 1:
            scores_list.append(-np.inf)
        elif k > min(max_k, np.sqrt(n)):
            break
        else:
            level = cluster_lvls[level_k[cluster_k.index(k)]]  # get k clusters
            b = silhouette_samples(dist, level, metric="precomputed")
            scores_list.append(b.mean() / b.std())

    scores_list = pd.Series(scores_list)
    k_index = int(scores_list.idxmax())
    k = cluster_k[k_index]
    node_k = level_k[k_index]
    if flag:
        clustering_inds = cluster_lvls[node_k].tolist()
    else:
        clustering_inds = hr.fcluster(clustering, k, criterion="maxclust")
        j = len(np.unique(clustering_inds))
        while k != j:
            j += 1
            clustering_inds = hr.fcluster(clustering, j, criterion="maxclust")
            k = len(np.unique(clustering_inds))
        unique_vals, indices = np.unique(clustering_inds, return_inverse=True)
        clustering_inds = indices

    return k, clustering_inds
"""
py"""
dist = pd.DataFrame($(dist))
leaf_order=True
linkage="ward"
max_k=1000
"""

py"""
p_dist = squareform(dist, checks=False)
clustering = hr.linkage(p_dist, method=linkage, optimal_ordering=leaf_order)
kt, clusterst = two_diff_gap_stat(dist, clustering, max_k)
ks, clusterss = std_silhouette_score(dist, clustering, max_k)
"""

isequal(py"clusterst", idxt .- 1)
isequal(py"clusterss", idxs .- 1)
################################################################

#######################################

function find_rtol(a1, a2)
    for rtol ∈
        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
         5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0]
        if isapprox(a1, a2; rtol = rtol)
            println(", rtol = $(rtol)")
            break
        end
    end
end

function get_rtol(a1, a2)
    for rtol ∈
        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
         5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0]
        if isapprox(a1, a2; rtol = rtol)
            return rtol
            break
        end
    end
end

function find_atol(a1, a2)
    for atol ∈
        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
         5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 2e0]
        if isapprox(a1, a2; atol = atol)
            println(", atol = $(atol)")
            break
        end
    end
end

find_rtol(
          #    
          w10, w10t
          #
          )

str = "println(\""
for i ∈ 1:11
    if i ∈ (7, 10, 11)
        continue
    end
    str *= "w$(i)t = \$(w$(i).weights)\\n"
end
str *= "\")"
println(str)

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

r = collect(range(; start = 9, stop = 9.5, length = 3))
p, reps = f(r, 1)
display([p reps])
display(r * 10)

# %%

f(w, h, l) = w * h * l

alex = 2 * f(32, 56, 7) + 3 * f(56, 32, 13)
trotten = 3 * f(15.6, 47, 40)

using Optimization, OptimizationOptimJL, AverageShiftedHistograms
const ASH = AverageShiftedHistograms

function errPDF(x, vals; kernel = ASH.Kernels.gaussian, m = 10, n = 1000, q = 1000)
    e_min, e_max = x[1] * (1 - sqrt(1.0 / q))^2, x[1] * (1 + sqrt(1.0 / q))^2
    rg = range(e_min, e_max; length = n)
    pdf1 = q ./ (2 * pi * x[1] * rg) .*
           sqrt.(clamp.((e_max .- rg) .* (rg .- e_min), 0, Inf))

    e_min, e_max = x[1] * (1 - sqrt(1.0 / q))^2, x[1] * (1 + sqrt(1.0 / q))^2
    res = ash(vals; rng = range(e_min, e_max; length = n), kernel = kernel, m = m)
    pdf2 = [ASH.pdf(res, i) for i ∈ pdf1]
    pdf2[.!isfinite.(pdf2)] .= 0.0
    sse = sum((pdf2 - pdf1) .^ 2)

    return sse
end
function find_max_eval(vals, q; kernel = ASH.Kernels.gaussian, m::Integer = 10,
                       n::Integer = 1000, args = (), kwargs = ())
    res = Optim.optimize(x -> errPDF(x, vals; kernel = kernel, m = m, n = n, q = q), 0.0,
                         1.0, args...; kwargs...)

    x = Optim.converged(res) ? Optim.minimizer(res) : 1.0

    e_max = x * (1.0 + sqrt(1.0 / q))^2

    return e_max, x
end

function find_max_eval2(vals, q; kernel = ASH.Kernels.gaussian, m::Integer = 10,
                        n::Integer = 1000, args = (), kwargs = ())
    u0 = [0.0]
    p = [vals, kernel, m, n, q]
    erpdf(u, p) = errPDF(u, p[1]; kernel = p[2], m = p[3], n = p[4], q = p[5])
    prob = OptimizationProblem(erpdf, u0, p; lb = [0.0], ub = [1.0])
    sol = solve(prob, SAMIN())

    # res = Optim.optimize(x -> errPDF(x, vals; kernel = kernel, m = m, n = n, q = q), 0.0,
    #                      1.0, args...; kwargs...)
    x = sol.u[1]

    # x = Optim.converged(res) ? Optim.minimizer(res) : 1.0

    e_max = x * (1.0 + sqrt(1.0 / q))^2

    return e_max, x
end

X = randn(100, 20)
X = cov(X)
vals, vecs = eigen(X)

max_val = find_max_eval2(vals, 100 / 20)[1]

using PortfolioOptimiser
