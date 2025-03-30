# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

function calc_centrality(type::BetweennessCentrality, G)
    return Graphs.betweenness_centrality(G, type.args...; type.kwargs...)
end
function calc_centrality(type::ClosenessCentrality, G)
    return Graphs.closeness_centrality(G, type.args...; type.kwargs...)
end
function calc_centrality(type::DegreeCentrality, G)
    return Graphs._degree_centrality(G, type.type; type.kwargs...)
end
function calc_centrality(::EigenvectorCentrality, G)
    return Graphs.eigenvector_centrality(G)
end
function calc_centrality(type::KatzCentrality, G)
    return Graphs.katz_centrality(G, type.alpha)
end
function calc_centrality(type::Pagerank, G)
    return Graphs.pagerank(G, type.alpha, type.n, type.epsilon)
end
function calc_centrality(::RadialityCentrality, G)
    return Graphs.radiality_centrality(G)
end
function calc_centrality(type::StressCentrality, G)
    return Graphs.stress_centrality(G, type.args...; type.kwargs...)
end
function calc_mst(type::KruskalTree, G)
    return Graphs.kruskal_mst(G, type.args...; type.kwargs...)
end
function calc_mst(type::BoruvkaTree, G)
    return Graphs.boruvka_mst(G, type.args...; type.kwargs...)[1]
end
function calc_mst(type::PrimTree, G)
    return Graphs.prim_mst(G, type.args...; type.kwargs...)
end
function calc_adjacency(nt::TMFG, rho::AbstractMatrix, delta::AbstractMatrix)
    S = dbht_similarity(nt.similarity, rho, delta)
    Rpm = PMFG_T2s(S)[1]
    return adjacency_matrix(SimpleGraph(Rpm))
end
function calc_adjacency(nt::MST, ::Any, delta::AbstractMatrix)
    G = SimpleWeightedGraph(delta)
    tree = calc_mst(nt.tree, G)
    return adjacency_matrix(SimpleGraph(G[tree]))
end
function calc_adjacency(nt::NetworkType, X::AbstractMatrix,
                        cor_type::PortfolioOptimiserCovCor, dist_type::DistType)
    S = cor(cor_type, X)
    dist_type = default_dist(dist_type, cor_type)
    D = dist(dist_type, S, X)
    return calc_adjacency(nt, S, D)
end
function connection_matrix(X::AbstractMatrix;
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           dist_type::DistType = DistCanonical(),
                           network_type::NetworkType = MST(), kwargs...)
    A = calc_adjacency(network_type, X, cor_type, dist_type)

    A_p = similar(Matrix(A))
    fill!(A_p, zero(eltype(A_p)))
    for i ∈ 0:(network_type.steps)
        A_p .+= A^i
    end

    A_p .= clamp!(A_p, 0, 1) - I

    return A_p
end
function centrality_vector(X::AbstractMatrix;
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           dist_type::DistType = DistCanonical(),
                           network_type::NetworkType = MST())
    Adj = connection_matrix(X; cor_type = cor_type, dist_type = dist_type,
                            network_type = network_type)
    G = SimpleGraph(Adj)

    return calc_centrality(network_type.centrality, G)
end
function centrality_constraint(X::AbstractMatrix; val = nothing, sign = ">=",
                               cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                               dist_type::DistType = DistCanonical(),
                               network_type::NetworkType = MST())
    if sign == ">="
        d = 1
    elseif sign == "<="
        d = -1
    end
    A = centrality_vector(X; cor_type = cor_type, dist_type = dist_type,
                          network_type = network_type)
    B = isnothing(val) ? minimum(A) : val

    return A * d, B * d
end
function average_centrality(X::AbstractMatrix, w::AbstractVector;
                            cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                            dist_type::DistType = DistCanonical(),
                            network_type::NetworkType = MST())
    return dot(centrality_vector(X; cor_type = cor_type, dist_type = dist_type,
                                 network_type = network_type), w)
end
function cluster_matrix(X::AbstractMatrix;
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistType = DistCanonical(), clust_alg::ClustAlg = HAC(),
                        clust_opt::ClustOpt = ClustOpt(), kwargs...)
    clusters = cluster_assets(X; cor_type = cor_type, dist_type = dist_type,
                              clust_alg = clust_alg, clust_opt = clust_opt)[1]

    N = size(X, 2)
    A_c = Vector{Int}(undef, 0)
    for i ∈ unique(clusters)
        idx = clusters .== i
        tmp = zeros(Int, N)
        tmp[idx] .= 1
        append!(A_c, tmp)
    end

    A_c = reshape(A_c, N, :)
    A_c = A_c * transpose(A_c) - I

    return A_c
end
function con_rel_assets(A::AbstractMatrix, w::AbstractVector)
    ovec = range(; start = 1, stop = 1, length = size(A, 1))
    aw = abs.(w * transpose(w))
    C_a = sum(A .* aw)
    C_a /= sum(aw)
    return C_a
end
function connected_assets(returns::AbstractMatrix, w::AbstractVector;
                          cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                          dist_type::DistType = DistCanonical(),
                          network_type::NetworkType = MST())
    A_c = connection_matrix(returns; cor_type = cor_type, dist_type = dist_type,
                            network_type = network_type)
    C_a = con_rel_assets(A_c, w)
    return C_a
end
function related_assets(returns::AbstractMatrix, w::AbstractVector;
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistType = DistCanonical(), clust_alg::ClustAlg = HAC(),
                        clust_opt::ClustOpt = ClustOpt())
    A_c = cluster_matrix(returns; cor_type = cor_type, dist_type = dist_type,
                         clust_alg = clust_alg, clust_opt = clust_opt)
    R_a = con_rel_assets(A_c, w)
    return R_a
end

export calc_centrality, connection_matrix, centrality_vector, cluster_matrix,
       con_rel_assets, connected_assets, related_assets, average_centrality,
       centrality_constraint
