abstract type CentralityType end
@kwdef mutable struct DegreeCentrality <: CentralityType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
function _cent_func(::DegreeCentrality)
    return Graphs.degree_centrality
end
abstract type NetworkType end
@kwdef mutable struct TMFG{T1 <: Integer} <: NetworkType
    similarity::DBHTSimilarity = DBHTMaxDist()
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end
abstract type TreeType end
@kwdef mutable struct KruskalTree <: TreeType
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
@kwdef mutable struct MST{T1 <: Integer} <: NetworkType
    tree::TreeType
    steps::T1 = 1
    centrality::CentralityType = DegreeCentrality()
end
function _tree_func(::KruskalTree)
    return Graphs.kruskal_mst
end
function _calc_adjacency(nt::TMFG, X::AbstractMatrix, cor_type::PortfolioOptimiserCovCor,
                         dist_type::DistanceMethod)
    S = cor(cor_type, X)
    dist_type = _get_default_dist(dist_type, cor_type)
    D = dist(dist_type, S, X)
    S = dbht_similarity(nt.similarity, S, D)

    Rpm = PMFG_T2s(S)[1]
    return adjacency_matrix(SimpleGraph(Rpm))
end
function _calc_adjacency(nt::MST, X::AbstractMatrix, cor_type::PortfolioOptimiserCovCor,
                         dist_type::DistanceMethod)
    S = cor(cor_type, X)
    dist_type = _get_default_dist(dist_type, cor_type)
    D = dist(dist_type, S, X)

    G = SimpleWeightedGraph(D)
    tree_func = _tree_func(nt.tree)

    return adjacency_matrix(SimpleGraph(G[tree_func(G, nt.tree.args...; nt.tree.kwargs...)]))
end
function connection_matrix2(X::AbstractMatrix;
                            cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                            dist_type::DistanceMethod = DistanceDefault(),
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
function connection_matrix2(portfolio::AbstractPortfolio;
                            cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                            dist_type::DistanceMethod = DistanceDefault(),
                            network_type::NetworkType = MST())
    return connection_matrix2(portfolio.returns; cor_type = cor_type, dist_type = dist_type,
                              network_type = network_type)
end
function centrality_vector2(X::AbstractMatrix;
                            cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                            dist_type::DistanceMethod = DistanceDefault(),
                            network_type::NetworkType = MST())
    Adj = connection_matrix2(X; cor_type = cor_type, dist_type = dist_type,
                             network_type = network_type)
    G = SimpleGraph(Adj)
    cent_func = _cent_func(network_type.centrality)

    return cent_func(G, network_type.centrality.args...; network_type.centrality.kwargs...)
end
function centrality_vector2(portfolio::AbstractPortfolio;
                            cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                            dist_type::DistanceMethod = DistanceDefault(),
                            network_type::NetworkType = MST())
    return centrality_vector2(portfolio.returns; cor_type = cor_type, dist_type = dist_type,
                              network_type = network_type)
end
