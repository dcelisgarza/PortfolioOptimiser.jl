function calc_centrality(method::BetweennessCentrality, G)
    return Graphs.betweenness_centrality(G, method.args...; method.kwargs...)
end
function calc_centrality(method::ClosenessCentrality, G)
    return Graphs.closeness_centrality(G, method.args...; method.kwargs...)
end
function calc_centrality(method::DegreeCentrality, G)
    return Graphs._degree_centrality(G, method.type; method.kwargs...)
end
function calc_centrality(::EigenvectorCentrality, G)
    return Graphs.eigenvector_centrality(G)
end
function calc_centrality(method::KatzCentrality, G)
    return Graphs.katz_centrality(G, method.alpha)
end
function calc_centrality(method::Pagerank, G)
    return Graphs.pagerank(G, method.alpha, method.n, method.epsilon)
end
function calc_centrality(::RadialityCentrality, G)
    return Graphs.radiality_centrality(G)
end
function calc_centrality(method::StressCentrality, G)
    return Graphs.stress_centrality(G, method.args...; method.kwargs...)
end
function clac_mst(method::KruskalTree, G)
    return Graphs.kruskal_mst(G, method.args...; method.kwargs...)
end
function clac_mst(method::BoruvkaTree, G)
    return Graphs.boruvka_mst(G, method.args...; method.kwargs...)[1]
end
function clac_mst(method::PrimTree, G)
    return Graphs.prim_mst(G, method.args...; method.kwargs...)
end
function _calc_adjacency(nt::TMFG, rho::AbstractMatrix, delta::AbstractMatrix)
    S = dbht_similarity(nt.similarity, rho, delta)
    Rpm = PMFG_T2s(S)[1]
    return adjacency_matrix(SimpleGraph(Rpm))
end
function _calc_adjacency(nt::MST, ::Any, delta::AbstractMatrix)
    G = SimpleWeightedGraph(delta)
    tree = clac_mst(nt.tree, G)
    return adjacency_matrix(SimpleGraph(G[tree]))
end
function _calc_adjacency(nt::NetworkType, X::AbstractMatrix,
                         cor_type::PortfolioOptimiserCovCor, dist_type::DistanceMethod)
    S = cor(cor_type, X)
    dist_type = _get_default_dist(dist_type, cor_type)
    D = dist(dist_type, S, X)
    return _calc_adjacency(nt, S, D)
end
function connection_matrix(X::AbstractMatrix;
                           cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                           dist_type::DistanceMethod = DistanceCanonical(),
                           network_type::NetworkType = MST())
    A = _calc_adjacency(network_type, X, cor_type, dist_type)

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
                           dist_type::DistanceMethod = DistanceCanonical(),
                           network_type::NetworkType = MST())
    Adj = connection_matrix(X; cor_type = cor_type, dist_type = dist_type,
                            network_type = network_type)
    G = SimpleGraph(Adj)

    return calc_centrality(network_type.centrality, G)
end
function cluster_matrix(X::AbstractMatrix;
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceCanonical(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
    clusters = cluster_assets(X; cor_type = cor_type, dist_type = dist_type,
                              hclust_alg = hclust_alg, hclust_opt = hclust_opt)[1]

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
function _con_rel(A::AbstractMatrix, w::AbstractVector)
    ovec = range(; start = 1, stop = 1, length = size(A, 1))
    aw = abs.(w * transpose(w))
    C_a = transpose(ovec) * (A .* aw) * ovec
    C_a /= transpose(ovec) * aw * ovec
    return C_a
end
function connected_assets(returns::AbstractMatrix, w::AbstractVector;
                          cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                          dist_type::DistanceMethod = DistanceCanonical(),
                          network_type::NetworkType = MST())
    A_c = connection_matrix(returns; cor_type = cor_type, dist_type = dist_type,
                            network_type = network_type)
    C_a = _con_rel(A_c, w)
    return C_a
end
function related_assets(returns::AbstractMatrix, w::AbstractVector;
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceCanonical(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
    A_c = cluster_matrix(returns; cor_type = cor_type, dist_type = dist_type,
                         hclust_alg = hclust_alg, hclust_opt = hclust_opt)
    R_a = _con_rel(A_c, w)
    return R_a
end

export calc_centrality, connection_matrix, centrality_vector, cluster_matrix,
       connected_assets, related_assets
