function PMFG_T2s(W, nargout = 3)
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
    ou_v = setdiff(1:N, in_v)  # List of vertices not inserted yet

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
        tri[kk + 1, :] .= vcat(tri[tr, [1, 3]], ve)
        tri[kk + 2, :] .= vcat(tri[tr, [2, 3]], ve)
        tri[tr, :] .= vcat(tri[tr, [1, 2]], ve)

        # # Update gain table
        gain[ve, :] .= 0
        gain[ou_v, tr] .= sum(W[ou_v, tri[tr, :]], dims = 2)
        gain[ou_v, kk + 1] .= sum(W[ou_v, tri[kk + 1, :]], dims = 2)
        gain[ou_v, kk + 2] .= sum(W[ou_v, tri[kk + 2, :]], dims = 2)

        # # Update number of triangles
        kk += 2
    end

    A = W .* ((A + A') .== 1)

    cliques = if nargout > 3
        vcat(transpose(in_v[1:4]), hcat(clique3, in_v[5:end]))
    else
        Matrix{Int}(undef, 0, 0)
    end

    cliqueTree = if nargout > 4
        M = size(cliques, 1)
        cliqueTree = zeros(Int, M, M)
        ss = zeros(Int, M)
        for i in 1:M
            ss .= 0
            for j in 1:3
                ss .+= sum(cliques .== cliques[i, j], dims = 2)
            end
            cliqueTree[i, ss .== 2] .= 1
        end
        cliqueTree
    else
        Matrix{Int}(undef, 0, 0)
    end

    return sparse(A), tri, clique3, cliques, cliqueTree
end

function distance_wei(L)
    N = size(L, 1)
    D = fill(Inf, N, N)
    D[diagind(D)] .= 0
    B = zeros(N, N)

    for u in 1:N
        S = fill(true, N)
        L1 = copy(L)
        V = [u]
        while true
            S[V] .= false
            L1[:, V] .= 0
            for v in vec(V)
                T = if issparse(L1)
                    findnz(L1[v, :])[1]
                else
                    getindex.(findall(L1[v, :] .!= 0), 1)
                end
                d, wi = findmin([D[u, T] D[u, v] .+ L1[v, T]], dims = 2)
                wi = vec(getindex.(wi, 2))
                D[u, T] .= vec(d)
                ind = T[wi .== 3]
                B[u, ind] .= B[u, v] + 1
            end

            dus = D[u, S]
            minD = !isempty(dus) ? minimum(dus) : Float64[]
            if isempty(minD) || isinf(minD)
                break
            end

            V = findall(D[u, :] .== minD)
        end
    end

    return D, B
end

function DBHTs(D, S)
    Rpm = PMFG_T2s(S)[1]
    Apm = copy(Rpm)
    Apm[Apm != 0] = D[Apm != 0]
    Dpm = distance_wei(Apm)[1]
    (H1, Hb, Mb, CliqList, Sb) = CliqHierarchyTree2s(Rpm, method1 = "uniqueroot")

    Clustering.orderbranches_barjoseph!(Z, D)
    return T8, Rpm, Adjv, Dpm, Mv, Z
end
