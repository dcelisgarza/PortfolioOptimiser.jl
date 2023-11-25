"""
```julia
PMFG_T2s(W::AbstractMatrix{<:Number}, nargout::Integer = 3)
```
Constructs a Triangulated Maximally Filtered Graph (TMFG) starting from a tetrahedron and recursively inserting vertices inside existing triangles (T2 move) in order to approximate a Maximal Planar Graph with the largest total weight, aka Planar Maximally Filtered Graph (PMFG). All weights are non-negative [^PMFG].
# Arguments
- `W`: `N×N` matrix of non-negative weights.
- `nargout`: number of output arguments, the same arguments are always returne, this only controls whether some arguments are empty or not.
# Outputs
- `A`: adjacency matrix of the PMFG with weights.
- `tri`: list of triangles (triangular faces).
- `clique3`: list of 3-cliques taht are not triangular faces, all 3-cliques are given by `[tri; clique3]`.
- `cliques`: list of all 4-cliques, if `nargout <= 3`, this will be returned as an empty array.
- `cliqueTree`: 4-cliques tree structure (adjacency matrix), if `nargout <= 4`, it is returned as an empty array.

[^PMFG]:
    [Guido Previde Massara, T. Di Matteo, Tomaso Aste, Network Filtering for Big Data: Triangulated Maximally Filtered Graph, Journal of Complex Networks, Volume 5, Issue 2, June 2017, Pages 161–178, https://doi.org/10.1093/comnet/cnw015](https://academic.oup.com/comnet/article-abstract/5/2/161/2555365).
"""
function PMFG_T2s(W::AbstractMatrix{<:Number}, nargout::Integer = 3)
    N = size(W, 1)

    @assert(N >= 9, "W Matrix too small")
    @assert(all(W .>= 0), "W Matrix has negative elements")

    A = spzeros(N, N)  # Initialize adjacency matrix
    in_v = zeros(Int, N)    # Initialize list of inserted vertices
    tri = zeros(Int, 2 * N - 4, 3)  # Initialize list of triangles
    clique3 = zeros(Int, N - 4, 3)   # Initialize list of 3-cliques (non-face triangles)

    # Find 3 vertices with largest strength
    s = sum(W .* (W .> mean(W)), dims = 2)
    j = sortperm(vec(s), rev = true)

    in_v[1:4] .= j[1:4]
    ou_v = sort!(setdiff(1:N, in_v))  # List of vertices not inserted yet

    # Build the tetrahedron with largest strength
    tri[1, :] = in_v[[1, 2, 3]]
    tri[2, :] = in_v[[2, 3, 4]]
    tri[3, :] = in_v[[1, 2, 4]]
    tri[4, :] = in_v[[1, 3, 4]]
    A[in_v[1], in_v[2]] = 1
    A[in_v[1], in_v[3]] = 1
    A[in_v[1], in_v[4]] = 1
    A[in_v[2], in_v[3]] = 1
    A[in_v[2], in_v[4]] = 1
    A[in_v[3], in_v[4]] = 1

    # Build initial gain table
    gain = zeros(N, 2 * N - 4)
    gain[ou_v, 1] .= sum(W[ou_v, tri[1, :]], dims = 2)
    gain[ou_v, 2] .= sum(W[ou_v, tri[2, :]], dims = 2)
    gain[ou_v, 3] .= sum(W[ou_v, tri[3, :]], dims = 2)
    gain[ou_v, 4] .= sum(W[ou_v, tri[4, :]], dims = 2)

    kk = 4  # Number of triangles
    for k in 5:N
        # Find best vertex to add in a triangle
        if length(ou_v) == 1  # Special case for the last vertex
            ve = ou_v[1]
            v = 1
            tr = argmax(vec(gain[ou_v, :]))
        else
            gij, v = findmax(gain[ou_v, :], dims = 1)
            v = vec(getindex.(v, 1))
            tr = argmax(vec(gij))
            ve = ou_v[v[tr]]
            v = v[tr]
        end

        # Update vertex lists
        ou_v = ou_v[deleteat!(collect(1:length(ou_v)), v)]
        # vcat(ou_v[1:(v - 1)], ou_v[(v + 1):end])
        in_v[k] = ve

        # Update adjacency matrix
        A[ve, tri[tr, :]] .= 1

        # Update 3-clique list
        clique3[k - 4, :] .= tri[tr, :]

        # Update triangle list replacing 1 and adding 2 triangles
        tri[kk + 1, :] .= vcat(tri[tr, [1, 3]], ve) # add
        tri[kk + 2, :] .= vcat(tri[tr, [2, 3]], ve) # add
        tri[tr, :] .= vcat(tri[tr, [1, 2]], ve)     # replace

        # # Update gain table
        gain[ve, :] .= 0
        gain[ou_v, tr] .= sum(W[ou_v, tri[tr, :]], dims = 2)
        gain[ou_v, kk + 1] .= sum(W[ou_v, tri[kk + 1, :]], dims = 2)
        gain[ou_v, kk + 2] .= sum(W[ou_v, tri[kk + 2, :]], dims = 2)

        # # Update number of triangles
        kk += 2
    end

    A = sparse(W .* ((A + A') .== 1))

    cliques =
        nargout > 3 ? vcat(transpose(in_v[1:4]), hcat(clique3, in_v[5:end])) :
        Matrix{Int}(undef, 0, 0)

    cliqueTree = if nargout > 4
        M = size(cliques, 1)
        cliqueTree = spzeros(Int, M, M)
        ss = zeros(Int, M)
        for i in 1:M
            ss .= 0
            for j in 1:3
                ss .+= vec(sum(cliques .== cliques[i, j], dims = 2))
            end
            cliqueTree[i, ss .== 2] .= 1
        end
        cliqueTree
    else
        Matrix{Int}(undef, 0, 0)
    end

    return A, tri, clique3, cliques, cliqueTree
end

"""
```julia
distance_wei(L::AbstractMatrix{<:Number})
```
The distance matrix contains lengths of shortest paths between all node pairs. An entry `[u, v]` represents the length of the shortest path from node `u` to node `v`. The average shortest path length is the characteristic path length of the network. The function uses Dijkstra's algorithm.
# Inputs
- `L`: Directed/undirected connection-length matrix. 
    - Lengths between disconnected nodes are set to `Inf`.
    - Lengths on the main diagonal are set to 0.
!!! note
    The input matrix must be a connection-length matrix typically obtained via a mapping from weight to length. For instance, in a weighted correlation network, higher correlations are more naturally interpreted as shorter distances, and the input matrix should therefore be some inverse of the connectivity matrix, i.e. a distance matrix.

    The number of edges in the shortest weighted path may in general exceed the number of edges in the shortest binary paths (i.e. the shortest weighted paths computed on the binarised connectivity matrix), because the shortest weighted paths have the minimal weighted distance, not necessarily the minimal number of edges.
# Outputs
- `D`: distance (shortest weighted path) matrix.
- `B`: number of edged in the shortest weigthed path matrix.

!!! note
    Based on a Matlab implementation by:
    - Mika Rubinov, UNSW/U Cambridge, 2007-2012.
    - Rick Betzel and Andrea Avena, IU, 2012

    Modification history:
    - 2007: original (MR)
    - 2009-08-04: min() function vectorized (MR)
    - 2012: added number of edges in shortest path as additional output (RB/AA)
    - 2013: variable names changed for consistency with other functions (MR)
"""
function distance_wei(L::AbstractMatrix{<:Number})
    N = size(L, 1)
    D = fill(Inf, N, N)
    D[diagind(D)] .= 0  # Distance matrix
    B = zeros(N, N)     # Number of edges matrix

    for u in 1:N
        S = fill(true, N)   # Distance permanence (true is temporary)
        L1 = copy(L)
        V = [u]
        while true
            S[V] .= false   # Distance u -> V is now permanent
            L1[:, V] .= 0   # No inside edges as already shortest
            dropzeros!(L1)
            for v in V
                T = findnz(L1[v, :])[1] # neighbours of shortest nodes
                d, wi = findmin(
                    vcat(vcat(transpose(D[u, T]), transpose(D[u, v] .+ L1[v, T]))),
                    dims = 1,
                )
                wi = vec(getindex.(wi, 2))
                D[u, T] .= vec(d)   # Smallest of old/new path lengths
                ind = T[wi .== 3]   # Indices of lengthened paths
                B[u, ind] .= B[u, v] + 1    # Increment number of edges in lengthened paths
            end

            dus = D[u, S]
            minD = !isempty(dus) ? minimum(dus) : Float64[]

            # isempty: all nodes reached
            # isinf: some nodes cannot be reached
            (isempty(minD) || isinf(minD)) && break

            V = findall(D[u, :] .== minD)
        end
    end

    return D, B
end

"""
```julia
clique3(A::AbstractMatrix{<:Number})
```
Computes the list of 3-cliques.
# Inputs
- `A`: `N×N` adjacency matrix of a Maximal Planar Graph (MPG).
# Outputs
- `K3`: vector of vectors with the corresponding indices of candidate cliques.
- `E`: matrix with non-zero indices and entries of candidate cliques.
- `CliqList`: `Nc×3` matrix. Each row vector lists the three vertices consisting of a 3-clique in the MPG.
"""
function clique3(A::AbstractMatrix{<:Number})
    A = A - Diagonal(A)
    A = A .!= 0
    A2 = A * A
    P = (A2 .!= 0) .* (A .!= 0)
    P = sparse(UpperTriangular(P))
    r, c = findnz(P .!= 0)[1:2]
    E = hcat(r, c)

    lr = length(r)
    N3 = Vector{Int}(undef, lr)
    K3 = Vector{Vector{Int}}(undef, lr)
    for n in 1:lr
        i = r[n]
        j = c[n]
        a = A[i, :] .* A[j, :]
        idx = findnz(a .!= 0)[1]
        K3[n] = idx
        N3[n] = length(idx)
    end

    clique = zeros(Int, 1, 3)
    for n in 1:lr
        temp = K3[n]
        for m in eachindex(temp)
            candidate = transpose(E[n, :])
            candidate = hcat(candidate, temp[m])
            sort!(candidate, dims = 2)
            a = clique[:, 1] .== candidate[1]
            b = clique[:, 2] .== candidate[2]
            c = clique[:, 3] .== candidate[3]
            check = a .* b .* c
            check = sum(check)

            check == 0 && (clique = vcat(clique, candidate))

            candidate, check, a, b, c = nothing, nothing, nothing, nothing, nothing
        end
    end

    isort = sortperm(collect(zip(clique[:, 1], clique[:, 2], clique[:, 3])))
    clique = clique[isort, :]
    clique = clique[2:size(clique, 1), :]

    return K3, E, clique
end

"""
```julia
breadth(CIJ::AbstractMatrix{<:Number}, source::Integer)
```
Breadth-first search.
# Inputs
- `CIJ`: binary (directed/undirected) connection matrix.
- `source`: source vertex.
# Outputs
- `distance`: distance between `source` and i'th vertex (0 for source vertex).
- `branch`: vertex that precedes i in the breadth-first search tree (-1 for source vertex).
!!! note
    Breadth-first search tree does not contain all paths (or all shortest paths), but allows the determination of at least one path with minimum distace. The entire graph is explored, starting from source vertex `source`.

    Original written by:
    Olaf Sporns, Indiana University, 2002/2007/2008
"""
function breadth(CIJ::AbstractMatrix{<:Number}, source::Integer)
    N = size(CIJ, 1)
    # Colours
    white = 0
    gray = 1
    black = 2
    # Initialise colours
    color = zeros(Int, N)
    # Initialise distances
    distance = fill(Inf, N)
    # Initialise branches
    branch = zeros(Int, N)
    # Start on vertex `source`
    color[source] = gray
    distance[source] = 0
    branch[source] = -1
    Q = [source]
    # Keep going until the entire graph is explored
    while !isempty(Q)
        u = Q[1]
        ns = findnz(CIJ[u, :])[1]
        for v in ns
            # This allows the `source` distance to itself to be recorded
            all(distance[v] .== 0) && (distance[v] = distance[u] + 1)
            if all(color[v] .== white)
                color[v] = gray
                distance[v] = distance[u] + 1
                branch[v] = u
                Q = vcat(Q, v)
            end
        end
        Q = Q[2:length(Q)]
        color[u] = black
    end

    return distance, branch
end

"""
```julia
FindDisjoint(Adj::AbstractMatrix{<:Number}, Cliq::AbstractVector{<:Number})
```
Finds disjointed cliques in adjacency matrix.
# Inputs
- `Adj`: `N×N` adjacency matrix.
- `Cliq`: `3×1` vector of 3-cliques.
# Outputs
- `T`: `N×1` vector containing the adjacency number of each node.
- `IndxNot`: `N×1` vector of nodes with no adjacencies.
"""
function FindDisjoint(Adj::AbstractMatrix{<:Number}, Cliq::AbstractVector{<:Number})
    N = size(Adj, 1)
    Temp = copy(Adj)
    T = zeros(Int, N)
    IndxTotal = 1:N
    IndxNot =
        findall(IndxTotal .!= Cliq[1] .&& IndxTotal .!= Cliq[2] .&& IndxTotal .!= Cliq[3])
    Temp[Cliq, :] .= 0
    Temp[:, Cliq] .= 0
    dropzeros!(Temp)
    d = breadth(Temp, IndxNot[1])[1]
    d[isinf.(d)] .= -1
    d[IndxNot[1]] = 0
    Indx1 = d .== -1
    Indx2 = d .!= -1
    T[Indx1] .= 1
    T[Indx2] .= 2
    T[Cliq] .= 0
    return T, IndxNot
end

"""
```julia
BuildHierarchy(M::AbstractMatrix{<:Number})
```
Builds the predicted hierarchy.
# Inputs
- `M`: `N×Nc` matrix of nodes and 3-cliques.
# Outputs
- `Pred`: `Nc×1` vector of predicted hierarchies.
"""
function BuildHierarchy(M::AbstractMatrix{<:Number})
    N = size(M, 2)
    Pred = zeros(Int, N)
    dropzeros!(M)
    for n in 1:N
        Children = findnz(M[:, n] .== 1)[1]
        ChildrenSum = vec(sum(M[Children, :], dims = 1))
        Parents = findall(ChildrenSum .== length(Children))
        Parents = Parents[Parents .!= n]
        if !isempty(Parents)
            ParentSum = vec(sum(M[:, Parents], dims = 1))
            a = findall(ParentSum .== minimum(ParentSum))
            length(a) == 1 ? Pred[n] = Parents[a[1]] : Pred = Int[]
        else
            Pred[n] = 0
        end
    end
    return Pred
end

"""
```julia
AdjCliq(
    A::AbstractMatrix{<:Number},
    CliqList::AbstractMatrix{<:Number},
    CliqRoot::AbstractVector{<:Number},
)
```
Find adjacent clique to the root candidates.
# Inputs
- `A`: `N×N` adjacency matrix.
- `CliqList`: `Nc×3` matrix. Each row vector lists the three vertices consisting of a 3-clique in the MPG.
- `CliqRoot`: `Nc×1` vector of root cliques.
# Outputs
- `Adj`: `Nc×Nc` adjacency matrix of the cliques with the root cliques.
"""
function AdjCliq(
    A::AbstractMatrix{<:Number},
    CliqList::AbstractMatrix{<:Number},
    CliqRoot::AbstractVector{<:Number},
)
    Nc = size(CliqList, 1)
    N = size(A, 1)
    Adj = spzeros(Nc, Nc)
    Indicator = zeros(Int, N)
    for n in eachindex(CliqRoot)
        Indicator[CliqList[CliqRoot[n], :]] .= 1
        Indi = hcat(
            Indicator[CliqList[CliqRoot, 1]],
            Indicator[CliqList[CliqRoot, 2]],
            Indicator[CliqList[CliqRoot, 3]],
        )

        adjacent = CliqRoot[vec(sum(Indi, dims = 2)) .== 2]
        Adj[adjacent, n] .= 1
    end
    Adj = Adj + transpose(Adj)
end

"""
```julia
BubbleHierarchy(Pred::AbstractVector{<:Number}, Sb::AbstractVector{<:Number})
```
Build the bubble hierarchy.
# Inputs
- `Pred`: `Nc×1` vector of predicted hierarchies.
- `Sb`: `Nc×1` vector. `Sb[n] = 1` indicates 3-clique `n` is separating.
# Outputs
- `Mb`: `Nc×Nb` bubble membership matrix for 3-cliques. `Mb[n, bi] = 1` indicated that 3-clique `n` belongs to bubble `bi`.
- `H2`: `Nb×Nb` adjacency matrix for the bubble hierarchical tree where `Nb` is the number of bubbles.
"""
function BubbleHierarchy(Pred::AbstractVector{<:Number}, Sb::AbstractVector{<:Number})
    Nc = size(Pred, 1)
    Root = findall(Pred .== 0)
    CliqCount = zeros(Nc)
    CliqCount[Root] .= 1
    Mb = Matrix{Int}(undef, Nc, 0)

    if length(Root) > 1
        TempVec = zeros(Nc)
        TempVec[Root] .= 1
        Mb = hcat(Mb, TempVec)
    end

    while sum(CliqCount) < Nc
        NxtRoot = Int[]
        for n in eachindex(Root)
            DirectChild = findall(Pred .== Root[n])
            TempVec = zeros(Int, Nc)
            TempVec[[Root[n]; DirectChild]] .= 1
            Mb = hcat(Mb, TempVec)
            CliqCount[DirectChild] .= 1

            for m in eachindex(DirectChild)
                Sb[DirectChild[m]] != 0 && (NxtRoot = [NxtRoot; DirectChild[m]])
            end

            DirectChild, TempVec = nothing, nothing
        end
        Root = sort!(unique(NxtRoot))
    end
    Nb = size(Mb, 2)
    H = spzeros(Int, Nb, Nb)

    for n in 1:Nb
        Indx = Mb[:, n] .== 1
        JointSum = vec(sum(Mb[Indx, :], dims = 1))
        Neigh = JointSum .>= 1
        H[n, Neigh] .= 1
    end

    H = H + transpose(H)
    H = H - Diagonal(H)
    return H, Mb
end

"""
```julia
DBHTRootMethods = (:Unique, :Equal)
```
Methods for finding the root of a Direct Bubble Hierarchical Clustering Tree in [`DBHTs`](@ref), in case there is more than one candidate.
- `:Unique`: create a unique root.
- `:Equal`: the root is created from the candidate's adjacency tree. 
"""
const DBHTRootMethods = (:Unique, :Equal)

"""
```julia
CliqHierarchyTree2s(Apm::AbstractMatrix{<:Number}, method::Symbol = :Unique)
```
Looks for 3-cliques of a Maximal Planar Graph (MPG), then construct a hierarchy of the cliques with the definition of "inside" a clique being a subgraph of smaller size when the entire graph is made disjoint by removing the clique [^NHPG].
# Inputs
- `Apm`: `N×N` adjacency matrix of an MPG.
- `method`: method for finding the root of the graph [`DBHTRootMethods`](@ref). Uses Voronoi tesselation between tiling triangles.
# Outputs
- `H1`: `Nc×Nc` adjacency matrix for 3-clique hierarchical tree where `Nc` is the number of 3-cliques.
- `H2`: `Nb×Nb` adjacency matrix for the bubble hierarchical tree where `Nb` is the number of bubbles.
- `Mb`: `Nc×Nb` bubble membership matrix for 3-cliques. `Mb[n, bi] = 1` indicated that 3-clique `n` belongs to bubble `bi`.
- `CliqList`: `Nc×3` matrix. Each row vector lists the three vertices consisting of a 3-clique in the MPG.
- `Sb`: `Nc×1` vector. `Sb[n] = 1` indicates 3-clique `n` is separating.
[^NHPG]:
    [Song, W. M., Di Matteo, T., & Aste, T. (2011). Nested hierarchies in planar graphs. Discrete Applied Mathematics, 159(17), 2135-2146.](https://www.sciencedirect.com/science/article/pii/S0166218X11002794)
"""
function CliqHierarchyTree2s(Apm::AbstractMatrix{<:Number}, method::Symbol = :Unique)
    @assert(method ∈ DBHTRootMethods, "method = $method, must be one of $DBHTRootMethods")
    N = size(Apm, 1)
    A = Apm .!= 0
    K3, E, clique = clique3(A)

    Nc = size(clique, 1)
    M = spzeros(Int, N, Nc)
    CliqList = copy(clique)
    Sb = zeros(Int, Nc)

    for n in 1:Nc
        cliq_vec = CliqList[n, :]
        T = FindDisjoint(A, cliq_vec)[1]
        indx0 = findall(T .== 0)
        indx1 = findall(T .== 1)
        indx2 = findall(T .== 2)

        indx_s = length(indx1) > length(indx2) ? vcat(indx2, indx0) : vcat(indx1, indx0)

        Sb[n] = isempty(indx_s) ? 0 : length(indx_s) - 3

        M[indx_s, n] .= 1
    end

    Pred = BuildHierarchy(M)
    Root = findall(Pred .== 0)

    if method == :Unique
        if length(Root) > 1
            push!(Pred, 0)
            Pred[Root] .= length(Pred)
        end

        H = spzeros(Int, Nc + 1, Nc + 1)
        for n in eachindex(Pred)
            Pred[n] != 0 && (H[n, Pred[n]] = 1)
        end
        H = H + transpose(H)
    else
        length(Root) > 1 && (Adj = AdjCliq(A, CliqList, Root))

        H = spzeros(Int, Nc, Nc)
        for n in eachindex(Pred)
            Pred[n] != 0 && (H[n, Pred[n]] = 1)
        end

        if !isempty(Pred)
            H .+= transpose(H)
            H .+= Adj
        else
            H = spzeros(Int, 0, 0)
        end
    end

    if !isempty(H)
        H2, Mb = BubbleHierarchy(Pred, Sb)
        H2 = H2 .!= 0
        Mb = Mb[1:size(CliqList, 1), :]
    else
        H2 = spzeros(Int, 0, 0)
        Mb = spzeros(Int, 0, 0)
    end

    return H, H2, Mb, CliqList, Sb
end

"""
```julia
DirectHb(
    Rpm::AbstractMatrix{<:Number},
    Hb::AbstractMatrix{<:Number},
    Mb::AbstractMatrix{<:Number},
    Mv::AbstractMatrix{<:Number},
    CliqList::AbstractMatrix{<:Number},
)
```
Computes the directions on each separating 3-clique of a Maximal Planar Graph (MPH), hence computes the Directed Bubble Hierarchy Tree (DBHT).
# Inputs
- `Rpm`: `N×N` sparse weighted adjacency matrix of the Planar Maximally Filtered Graph (MPFG).
- `Hb`: Undirected bubble tree of the PMFG.
- `Mb`: `Nc×Nb` bubble membership matrix for 3-cliques. `Mb[n, bi] = 1` indicated that 3-clique `n` belongs to bubble `bi`.
- `Mv`: `N×Nb` bubble membership matrix for vertices.
- `CliqList`: `Nc×3` matrix. Each row vector lists the three vertices consisting of a 3-clique in the MPG.
# Outputs
- `Hc`: `Nb×Nb` unweighted directed adjacency matrix of the DBHT. `Hc[i, j]=1` indicates a directed edge from bubble `i` to bubble `j`.
"""
function DirectHb(
    Rpm::AbstractMatrix{<:Number},
    Hb::AbstractMatrix{<:Number},
    Mb::AbstractMatrix{<:Number},
    Mv::AbstractMatrix{<:Number},
    CliqList::AbstractMatrix{<:Number},
)
    Hb = Hb .!= 0
    r, c, _ = findnz(sparse(UpperTriangular(Hb) .!= 0))
    CliqEdge = Matrix{Int}(undef, 0, 3)
    for n in eachindex(r)
        data = findall(Mb[:, r[n]] .!= 0 .&& Mb[:, c[n]] .!= 0)
        data = hcat(r[n], c[n], data)
        CliqEdge = vcat(CliqEdge, data)
    end

    kb = vec(sum(Hb, dims = 1))
    sMv = size(Mv, 2)
    Hc = spzeros(sMv, sMv)

    sCE = size(CliqEdge, 1)
    for n in 1:sCE
        Temp = copy(Hb)
        Temp[CliqEdge[n, 1], CliqEdge[n, 2]] = 0
        Temp[CliqEdge[n, 2], CliqEdge[n, 1]] = 0
        dropzeros!(Temp)
        d, _ = breadth(Temp, 1)
        d[isinf.(d)] .= -1
        d[1] = 0

        vo = CliqList[CliqEdge[n, 3], :]
        b = CliqEdge[n, 1:2]
        bleft = b[d[b] .!= -1]
        bright = b[d[b] .== -1]

        vleft = getindex.(findall(Mv[:, d .!= -1] .!= 0), 1)
        vleft = setdiff(vleft, vo)

        vright = getindex.(findall(Mv[:, d .== -1] .!= 0), 1)
        vright = setdiff(vright, vo)

        left = sum(Rpm[vo, vleft])
        right = sum(Rpm[vo, vright])

        left > right ? Hc[bright, bleft] .= left : Hc[bleft, bright] .= right
    end

    Sep = vec(Int.(sum(Hc, dims = 2) .== 0))
    Sep[vec(sum(Hc, dims = 1) .== 0) .&& kb .> 1] .= 2

    return Hc, Sep
end

"""
```julia
BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)
```
Obtains non-discrete and discrete clusterings from the bubble topology of the Planar Maximally Filtered Graph (PMFG).
# Inputs
- `Rpm`: `N×N` sparse weighted adjacency matrix of the PMFG.
- `Dpm`: `N×N` shortest path lengths matrix of the PMFG.
- `Hb`: undirected bubble tree of the PMFG.
- `Mb`: `Nc×Nb` bubble membership matrix for 3-cliques. `Mb[n, bi] = 1` indicated that 3-clique `n` belongs to bubble `bi`.
- `Mv`: `N×Nb` bubble membership matrix for vertices.
- `CliqList`: `Nc×3` matrix. Each row vector lists the three vertices consisting of a 3-clique in the MPG.
# Outputs
- `Adjv`: `N×Nk` cluster membership matrix for vertices for non-discrete clustering via the bubble topology. `Adjv[n, k] = 1` indicates cluster membership of vertex `n` to the `k`'th non-discrete cluster.
- `Tc`: `N×1` cluster membership vector. `Tc[n] = k` indicates cluster membership of vertex `n` to the `k`'th discrete cluster.
"""
function BubbleCluster8s(
    Rpm::AbstractMatrix{<:Number},
    Dpm::AbstractMatrix{<:Number},
    Hb::AbstractMatrix{<:Number},
    Mb::AbstractMatrix{<:Number},
    Mv::AbstractMatrix{<:Number},
    CliqList::AbstractMatrix{<:Number},
)
    Hc, Sep = DirectHb(Rpm, Hb, Mb, Mv, CliqList)   # Assign directions on the bubble tree

    N = size(Rpm, 1)    # Number of vertices in the PMFG
    indx = findall(Sep .== 1)   # Look for the converging bubbles
    Adjv = spzeros(0, 0)

    dropzeros!(Hc)
    lidx = length(indx)
    if lidx > 1
        Adjv = spzeros(size(Mv, 1), lidx)   # Set the non-discrete cluster membership matrix 'Adjv' at default

        # Identify the non-discrete cluster membership of vertices by each converging bubble
        for n in eachindex(indx)
            d, _ = breadth(transpose(Hc), indx[n])
            d[isinf.(d)] .= -1
            d[indx[n]] = 0
            r = getindex.(findall(Mv[:, d .!= -1] .!= 0), 1)
            Adjv[unique(r), n] .= 1
        end

        Tc = zeros(Int, N)  # Set the discrete cluster membership vector at default
        Bubv = Mv[:, indx]  # Gather the list of vertices in the converging bubbles
        cv = findall(vec(sum(Bubv, dims = 2) .== 1))    # Identify vertices which belong to single converging bubbles
        uv = findall(vec(sum(Bubv, dims = 2) .> 1)) # Identify vertices which belong to more than one converging bubbles
        Mdjv = spzeros(N, lidx) # Set the cluster membership matrix for vertices in the converging bubbles at default
        Mdjv[cv, :] = Bubv[cv, :]   # Assign vertices which belong to single converging bubbles to the rightful clusters

        # Assign converging bubble membership of vertices in `uv'
        for v in eachindex(uv)
            v_cont = vec(sum(Rpm[:, uv[v]] .* Bubv, dims = 1))  # sum of edge weights linked to uv(v) in each converging bubble
            all_cont = vec(3 * (sum(Bubv, dims = 1) .- 2))  # number of edges in converging bubble
            imx = argmax(v_cont ./ all_cont)    # computing chi(v,b_{alpha})
            Mdjv[uv[v], imx] = 1    # Pick the most strongly associated converging bubble
        end

        # Assign discrete cluster memebership of vertices in the converging bubbles
        v, ci, _ = findnz(Mdjv)
        Tc[v] .= ci

        # Compute the distance between a vertex and the converging bubbles
        Udjv = Dpm * Mdjv * diagm(1 ./ vec(sum(Mdjv .!= 0, dims = 1)))
        Udjv[Adjv .== 0] .= Inf

        imn = vec(getindex.(argmin(Udjv[vec(sum(Mdjv, dims = 2)) .== 0, :], dims = 2), 2))  # Look for the closest converging bubble
        Tc[Tc .== 0] .= imn # Assign discrete cluster membership according to the distances to the converging bubbles
    else
        Tc = ones(Int, N)   # If there is one converging bubble, all vertices belong to a single cluster
    end

    return Adjv, Tc
end

"""
```julia
BubbleMember(
    Rpm::AbstractMatrix{<:Number},
    Mv::AbstractMatrix{<:Number},
    Mc::AbstractMatrix{<:Number},
)
```
Assigns each vertex in the to a specific bubble.
# Inputs
- `Rpm`: `N×N` sparse weighted adjacency matrix of the PMFG.
- `Mv`: `N×Nb` bubble membership matrix. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
- `Mc`: Matrix of the bubbles which coincide with the cluster.
# Outputs
- `Mvv`: Matrix of the vertices belonging to the bubble.
"""
function BubbleMember(
    Rpm::AbstractMatrix{<:Number},
    Mv::AbstractMatrix{<:Number},
    Mc::AbstractMatrix{<:Number},
)
    Mvv = zeros(size(Mv, 1), size(Mv, 2))

    vu = findall(vec(sum(Mc, dims = 2) .> 1))
    v = findall(vec(sum(Mc, dims = 2) .== 1))

    Mvv[v, :] = Mc[v, :]

    for n in eachindex(vu)
        bub = findall(Mc[vu[n], :] .!= 0)
        vu_bub = vec(sum(Rpm[:, vu[n]] .* Mv[:, bub], dims = 1))
        all_bub = diag(transpose(Mv[:, bub]) * Rpm * Mv[:, bub]) / 2
        frac = vu_bub ./ all_bub
        imx = vec(argmax(frac, dims = 1))
        Mvv[vu[n], bub[imx]] .= 1
    end

    return Mvv
end

"""
```julia
DendroConstruct(
    Zi::AbstractMatrix{<:Number},
    LabelVec1::AbstractVector{<:Number},
    LabelVec2::AbstractVector{<:Number},
    LinkageDist::Union{<:Number, AbstractVector{<:Number}},
)
```
Construct the linkage matrix by continuially adding rows to the matrix.
# Inputs
- `Zi`: Linkage matrix at iteration `i` in the same format as the output from Matlab.
- `LabelVec1`: label vector for the vertices in the bubble for the previous valid iteration.
- `LabelVec2`: label vector for the vertices in the bubble for the trial iteration.
# Outputs
- `Z`: Linkage matrix at iteration `i + 1` in the same format as the output from Matlab.
"""
function DendroConstruct(
    Zi::AbstractMatrix{<:Number},
    LabelVec1::AbstractVector{<:Number},
    LabelVec2::AbstractVector{<:Number},
    LinkageDist::Union{<:Number, AbstractVector{<:Number}},
)
    indx = LabelVec1 .!= LabelVec2
    Z = vcat(Zi, hcat(transpose(sort!(unique(LabelVec1[indx]))), LinkageDist))
    return Z
end

"""
```julia
LinkageFunction(d::AbstractMatrix{<:Number}, labelvec::AbstractVector{<:Number})
```
Looks for the pair of clusters with the best linkage.
# Inputs
- `d`: `Nv×Nv` distance matrix for a list of vertices assigned to a bubble.
- `labelvec`: label vector for the vertices in the bubble.
# Outputs
- `PairLink`: pair of links with the best linkage.
- `dvu`: value of the best linkage.
"""
function LinkageFunction(d::AbstractMatrix{<:Number}, labelvec::AbstractVector{<:Number})
    lvec = sort!(unique(labelvec))
    Links = Matrix{Int}(undef, 0, 3)
    for r in 1:(length(lvec) - 1)
        vecr = labelvec .== lvec[r]
        for c in (r + 1):length(lvec)
            vecc = labelvec .== lvec[c]
            x1 = vecr .|| vecc
            dd = d[x1, x1]
            de = dd[dd .!= 0]

            Link1 =
                isempty(de) ? hcat(lvec[r], lvec[c], 0) :
                hcat(lvec[r], lvec[c], vec(maximum(de, dims = 1)))

            Links = vcat(Links, Link1)
        end
    end
    dvu, imn = findmin(Links[:, 3])
    PairLink = Links[imn, 1:2]
    return PairLink, dvu
end

"""
```julia
_build_link_and_dendro(
    rg::AbstractRange,
    dpm::AbstractMatrix{<:Number},
    LabelVec::AbstractVector{<:Number},
    LabelVec1::AbstractVector{<:Number},
    LabelVec2::AbstractVector{<:Number},
    V::AbstractVector{<:Number},
    nc::Number,
    Z::AbstractMatrix{<:Number},
)
```
Computes iterates over the vertices to construct the linkage matrix iteration by iteration.
# Inputs
- `rg`: range of indices of the vertices in a bubble.
- `dpm`: `Nv×Nv` distance matrix for a list of vertices assigned to a bubble.
- `LabelVec`: vector labels of all vertices.
- `LabelVec1`: label vector for the vertices in the bubble for the previous valid iteration.
- `LabelVec2`: label vector for the vertices in the bubble for the trial iteration.
# Outputs
- `Z`: updated linkage matrix in the same format as the output from Matlab.
- `nc`: updated inverse of the linkage distance.
- `LabelVec1`: updated `LabelVec1` for the next iteration.
"""
function _build_link_and_dendro(
    rg::AbstractRange,
    dpm::AbstractMatrix{<:Number},
    LabelVec::AbstractVector{<:Number},
    LabelVec1::AbstractVector{<:Number},
    LabelVec2::AbstractVector{<:Number},
    V::AbstractVector{<:Number},
    nc::Number,
    Z::AbstractMatrix{<:Number},
)
    for _ in rg
        PairLink, dvu = LinkageFunction(dpm, LabelVec)  # Look for the pair of clusters which produces the best linkage
        LabelVec[LabelVec .== PairLink[1] .|| LabelVec .== PairLink[2]] .=
            maximum(LabelVec1) + 1  # Merge the cluster pair by updating the label vector with a same label.
        LabelVec2[V] = LabelVec
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, 1 / nc)
        nc -= 1
        LabelVec1 = copy(LabelVec2)
    end
    return Z, nc, LabelVec1
end

"""
```julia
HierarchyConstruct4s(
    Rpm::AbstractMatrix{<:Number},
    Dpm::AbstractMatrix{<:Number},
    Tc::AbstractVector{<:Number},
    Mv::AbstractMatrix{<:Number},
)
```
Constructs the intra- and inter-cluster hierarchy by utilizing Bubble Hierarchy structure of a Maximal Planar graph, in this a Planar Maximally Filtered Graph (PMFG). 
# Inputs
- `Rpm`: `N×N` sparse weighted adjacency matrix of the PMFG.
- `Dpm`: `N×N` shortest path lengths matrix of the PMFG.
- `Tc`: `N×1` cluster membership vector. `Tc[n] = k` indicates cluster membership of vertex `n` to the `k`'th discrete cluster.
- `Mv`: `N×Nb` bubble membership matrix. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
# Outputs
- `Z`: `(N-1)×3` linkage matrix in the same format as the output from Matlab.
"""
function HierarchyConstruct4s(
    Rpm::AbstractMatrix{<:Number},
    Dpm::AbstractMatrix{<:Number},
    Tc::AbstractVector{<:Number},
    Mv::AbstractMatrix{<:Number},
)
    N = size(Dpm, 1)
    kvec = sort!(unique(Tc))
    LabelVec1 = collect(1:N)
    E = sparse(LabelVec1, Tc, ones(Int, N), N, maximum(Tc))
    Z = Matrix{Float64}(undef, 0, 3)

    # Intra-cluster hierarchy construction
    for n in eachindex(kvec)
        Mc = vec(E[:, kvec[n]]) .* Mv   # Get the list of bubbles which coincide with nth cluster
        Mvv = BubbleMember(Rpm, Mv, Mc) # Assign each vertex in the nth cluster to a specific bubble
        Bub = findall(vec(sum(Mvv, dims = 1) .> 0)) # Get the list of bubbles which contain the vertices of nth cluster 
        nc = sum(Tc .== kvec[n]) - 1

        # Apply the linkage within the bubbles.
        for m in eachindex(Bub)
            V = vec(findall(Mvv[:, Bub[m]] .!= 0)) # Retrieve the list of vertices assigned to mth bubble
            if length(V) > 1
                dpm = Dpm[V, V] # Retrieve the distance matrix for the vertices in V
                LabelVec = LabelVec1[V] # Initiate the label vector which labels for the clusters
                LabelVec2 = copy(LabelVec1)
                Z, nc, LabelVec1 = _build_link_and_dendro(
                    1:(length(V) - 1),
                    dpm,
                    LabelVec,
                    LabelVec1,
                    LabelVec2,
                    V,
                    nc,
                    Z,
                )
            end
        end

        V = findall(E[:, kvec[n]] .!= 0)
        dpm = Dpm[V, V]

        # Perform linkage merging between the bubbles
        LabelVec = LabelVec1[V] # Initiate the label vector which labels for the clusters.
        LabelVec2 = copy(LabelVec1)
        Z, nc, LabelVec1 = _build_link_and_dendro(
            1:(length(Bub) - 1),
            dpm,
            LabelVec,
            LabelVec1,
            LabelVec2,
            V,
            nc,
            Z,
        )
    end

    # Inter-cluster hierarchy construction
    LabelVec2 = copy(LabelVec1)
    dcl = ones(Int, length(LabelVec1))
    for _ in 1:(length(kvec) - 1)
        PairLink, dvu = LinkageFunction(Dpm, LabelVec1)
        LabelVec2[LabelVec1 .== PairLink[1] .|| LabelVec1 .== PairLink[2]] .=
            maximum(LabelVec1) + 1

        dvu =
            unique(dcl[LabelVec1 .== PairLink[1]]) + unique(dcl[LabelVec1 .== PairLink[2]])

        dcl[LabelVec1 .== PairLink[1] .|| LabelVec1 .== PairLink[2]] .= dvu

        Z = DendroConstruct(Z, LabelVec1, LabelVec2, dvu)
        LabelVec1 = copy(LabelVec2)
    end

    return Z
end

"""
```julia
turn_into_Hclust_merges(Z::AbstractMatrix{<:Number})
```
Turns a Matlab-style linkage matrix to a useable format for [`Hclust`](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust).
# Inputs
- `Z`: Matlab-style linkage matrix.
# Outputs
- `Z`: Hclust-style linkage matrix.
"""
function turn_into_Hclust_merges(Z::AbstractMatrix{<:Number})
    N = size(Z, 1) + 1
    Z = hcat(Z, zeros(N - 1))

    for i in eachindex(view(Z, :, 1))

        # Cluster indices.
        a = Int(Z[i, 1])
        b = Int(Z[i, 2])

        # If the cluster index is less than N, it represents a leaf, 
        # so only one add one to the count.
        if a <= N
            Z[i, 1] = -a
            Z[i, 4] += 1
        else
            # Clusters in index Z[i, 1:2] are combined to form cluster i + N.
            # If a cluster has index a > N, it's a combined cluster.
            # The index of the child is j = a - N, so we need to go to index j
            # which is being combined by cluster a, get the count at index j
            # and add it to the count at index i, which contains cluster a.
            j = a - N
            Z[i, 1] = j
            Z[i, 4] += Z[j, 4]
        end

        if b <= N
            Z[i, 2] = -b
            Z[i, 4] += 1
        else
            # Do the same with the other side of the cluster, to wherever that side leads.
            j = b - N
            Z[i, 2] = j
            Z[i, 4] += Z[j, 4]
        end
    end
    return Z
end

"""
```julia
DBHTs(
    D::AbstractMatrix{<:Number},
    S::AbstractMatrix{<:Number};
    branchorder::Symbol = :optimal,
    method::Symbol = :Unique,
)
```
Perform Direct Bubble Hierarchical Tree clustering, a deterministic clustering algorithm [^DBHTs]. This version uses a graph-theoretic filtering technique called Triangulated Maximally Filtered Graph (TMFG).
# Arguments
- `D`: `N×N` dissimilarity matrix, e.g. a distance matrix.
- `S`: `N×N` non-negative similarity matrix, e.g. a correlation matrix + 1, `S = 2 .- D .^2 / 2`; or another choice could be `S = exp.(-D).`
- `branchorder`: is a parameter for ordering the final dendrogram's branches. The choices are defined by [`BranchOrderTypes`](@ref).
- `method`: method for finding the root of a Direct Bubble Hierarchical Clustering Tree in case there is more than one candidate [`DBHTRootMethods`](@ref).
# Outputs
- `T8`: `N×1` cluster membership vector.
- `Rpm`: `N×N` adjacency matrix of the Planar Maximally Filtered Graph (PMFG).
- `Adjv`: Bubble cluster membership matrix from [`BubbleCluster8s`](@ref).
- `Dpm`: `N×N` shortest path length matrix of the PMFG.
- `Mv`: `N×Nb` bubble membership matrix. `Mv[n, bi] = 1` means vertex `n` is a vertex of bubble `bi`.
- `Z`: `(N-1)×3` linkage matrix in the same format as the output from Matlab.
- `Z_hclust`: Z matrix in [Clustering.Hclust](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust) format.

[^DBHTs]:
    [Song, Won-Min, T. Di Matteo, and Tomaso Aste. "Hierarchical information clustering by means of topologically embedded graphs." PloS one 7.3 (2012): e31929](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0031929).
"""
function DBHTs(
    D::AbstractMatrix{<:Number},
    S::AbstractMatrix{<:Number};
    branchorder::Symbol = :optimal,
    method::Symbol = :Unique,
)
    @assert(
        branchorder ∈ BranchOrderTypes,
        "branchorder = $branchorder, must be one of $BranchOrderTypes"
    )
    @assert(issymmetric(D), "D must be symmetric")
    @assert(issymmetric(S), "S must be symmetric")

    Rpm = PMFG_T2s(S)[1]
    Apm = copy(Rpm)
    Apm[Apm .!= 0] .= D[Apm .!= 0]
    Dpm = distance_wei(Apm)[1]

    H1, Hb, Mb, CliqList, Sb = CliqHierarchyTree2s(Rpm, method)

    Mb = Mb[1:size(CliqList, 1), :]

    sRpm = size(Rpm, 1)
    Mv = spzeros(Int, sRpm, 0)

    nMb = size(Mb, 2)
    for n in 1:nMb
        vc = spzeros(Int, sRpm)
        vc[sort!(unique(CliqList[Mb[:, n] .!= 0, :]))] .= 1
        Mv = hcat(Mv, vc)
    end

    Adjv, T8 = BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)

    Z = HierarchyConstruct4s(Rpm, Dpm, T8, Mv)
    Z = turn_into_Hclust_merges(Z)

    n = size(Z, 1)
    hmer = Clustering.HclustMerges{eltype(D)}(n + 1)
    resize!(hmer.mleft, n) .= Int.(Z[:, 1])
    resize!(hmer.mright, n) .= Int.(Z[:, 2])
    resize!(hmer.heights, n) .= Z[:, 3]

    if branchorder == :barjoseph || branchorder == :optimal
        Clustering.orderbranches_barjoseph!(hmer, D)
    elseif branchorder == :r
        Clustering.orderbranches_r!(hmer)
    end

    Z_hclust = Hclust(hmer, :DBHT)

    return T8, Rpm, Adjv, Dpm, Mv, Z, Z_hclust
end

function _jlogo!(jlogo, sigma, source, sign)
    tmp = Matrix{eltype(sigma)}(undef, size(source, 2), size(source, 2))
    for i in axes(source, 1)
        v = source[i, :]
        idx = Iterators.product(v, v)
        for (j, k) in enumerate(idx)
            tmp[j] = sigma[k[1], k[2]]
        end

        tmp = inv(tmp)

        for (j, k) in enumerate(idx)
            jlogo[k[1], k[2]] += sign * tmp[j]
        end
    end
    return nothing
end

"""
```julia
JLogo(sigma, separators, cliques)
```
Compute the sparse inverse covariance from a clique tree and separators [^JLogo].
# Inputs
- `sigma`: `N×N` covariance matrix.
- `separators`: list of 3-cliques taht are not triangular faces.
- `cliques`: list of all 4-cliques.
# Outputs
- `jlogo`: JLogo covariance matrix.

[^JLogo]:
    [Barfuss, W., Massara, G. P., Di Matteo, T., & Aste, T. (2016). Parsimonious modeling with information filtering networks. Physical Review E, 94(6), 062306.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.062306)
"""
function JLogo(sigma, separators, cliques)
    jlogo = zeros(eltype(sigma), size(sigma))

    _jlogo!(jlogo, sigma, cliques, 1)
    _jlogo!(jlogo, sigma, separators, -1)

    return jlogo
end

export DBHTs, PMFG_T2s, JLogo
