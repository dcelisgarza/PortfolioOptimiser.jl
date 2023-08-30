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

    A = sparse(W .* ((A + A') .== 1))

    cliques =
        nargout > 3 ? vcat(transpose(in_v[1:4]), hcat(clique3, in_v[5:end])) :
        Matrix{Int}(undef, 0, 0)

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

    return A, tri, clique3, cliques, cliqueTree
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
            dropzeros!(L1)
            for v in V
                T = findnz(L1[v, :])[1]
                d, wi = findmin(
                    vcat(vcat(transpose(D[u, T]), transpose(D[u, v] .+ L1[v, T]))),
                    dims = 1,
                )
                wi = vec(getindex.(wi, 2))
                D[u, T] .= vec(d)
                ind = T[wi .== 3]
                B[u, ind] .= B[u, v] + 1
            end

            dus = D[u, S]
            minD = !isempty(dus) ? minimum(dus) : Float64[]

            (isempty(minD) || isinf(minD)) && break

            V = findall(D[u, :] .== minD)
        end
    end

    return D, B
end

function clique3(A)
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

function breadth(CIJ, source)
    N = size(CIJ, 1)
    white = 0
    gray = 1
    black = 2
    color = zeros(Int, N)
    distance = fill(Inf, N)
    branch = zeros(Int, N)
    color[source] = gray
    distance[source] = 0
    branch[source] = -1
    Q = [source]
    while !isempty(Q)
        u = Q[1]
        ns = findnz(CIJ[u, :])[1]
        for v in ns
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

function FindDisjoint(Adj, Cliq)
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

function BuildHierarchy(M)
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

function AdjCliq(A, CliqList, CliqRoot)
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

function BubbleHierarchy(Pred, Sb)
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

function CliqHierarchyTree2s(Apm, method = :unique)
    @assert(method ∈ DBHTRootMethods, "method must be one of $DBHTRootMethods")
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

    if method == :unique
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
            H = H + transpose(H)
            H = H + Adj
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

function DirectHb(Rpm, Hb, Mb, Mv, CliqList)
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

function BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)
    Hc, Sep = DirectHb(Rpm, Hb, Mb, Mv, CliqList)

    N = size(Rpm, 1)
    indx = findall(Sep .== 1)
    Adjv = spzeros(0, 0)

    dropzeros!(Hc)
    lidx = length(indx)
    if lidx > 1
        Adjv = spzeros(size(Mv, 1), lidx)

        for n in eachindex(indx)
            d, _ = breadth(transpose(Hc), indx[n])
            d[isinf.(d)] .= -1
            d[indx[n]] = 0
            r = getindex.(findall(Mv[:, d .!= -1] .!= 0), 1)
            Adjv[unique(r), n] .= 1
        end
        Tc = zeros(Int, N)
        Bubv = Mv[:, indx]
        cv = findall(vec(sum(Bubv, dims = 2) .== 1))
        uv = findall(vec(sum(Bubv, dims = 2) .> 1))
        Mdjv = spzeros(N, lidx)
        Mdjv[cv, :] = Bubv[cv, :]
        for v in eachindex(uv)
            v_cont = vec(sum(Rpm[:, uv[v]] .* Bubv, dims = 1))
            all_cont = vec(3 * (sum(Bubv, dims = 1) .- 2))
            imx = argmax(v_cont ./ all_cont)
            Mdjv[uv[v], imx] = 1
        end
        v, ci, _ = findnz(Mdjv)
        Tc[v] .= ci

        Udjv = Dpm * Mdjv * diagm(1 ./ vec(sum(Mdjv .!= 0, dims = 1)))
        Udjv[Adjv .== 0] .= Inf
        imn = vec(getindex.(argmin(Udjv[vec(sum(Mdjv, dims = 2)) .== 0, :], dims = 2), 2))
        Tc[Tc .== 0] .= imn
    else
        Tc = ones(Int, N)
    end

    return Adjv, Tc
end

function DendroConstruct(Zi, LabelVec1, LabelVec2, LinkageDist)
    indx = LabelVec1 .!= LabelVec2
    Z = vcat(Zi, hcat(transpose(sort!(unique(LabelVec1[indx]))), LinkageDist))
    return Z
end

function BubbleMember(Dpm, Rpm, Mv, Mc)
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

function LinkageFunction(d, labelvec)
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

function _build_link_and_dendro(rg, dpm, LabelVec, LabelVec1, LabelVec2, V, nc, Z)
    for _ in rg
        PairLink, dvu = LinkageFunction(dpm, LabelVec)
        LabelVec[LabelVec .== PairLink[1] .|| LabelVec .== PairLink[2]] .=
            maximum(LabelVec1) + 1
        LabelVec2[V] = LabelVec
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, 1 / nc)
        nc -= 1
        LabelVec1 = copy(LabelVec2)
    end
    return Z, nc, LabelVec1
end

function HierarchyConstruct4s(Rpm, Dpm, Tc, Adjv, Mv)
    N = size(Dpm, 1)
    kvec = sort!(unique(Tc))
    LabelVec1 = collect(1:N)
    E = sparse(LabelVec1, Tc, ones(Int, N), N, maximum(Tc))
    Z = Matrix{Float64}(undef, 0, 3)

    for n in eachindex(kvec)
        Mc = vec(E[:, kvec[n]]) .* Mv
        Mvv = BubbleMember(Dpm, Rpm, Mv, Mc)
        Bub = findall(vec(sum(Mvv, dims = 1) .> 0))
        nc = sum(Tc .== kvec[n]) - 1
        for m in eachindex(Bub)
            V = vec(findall(Mvv[:, Bub[m]] .!= 0))
            if length(V) > 1
                dpm = Dpm[V, V]
                LabelVec = LabelVec1[V]
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

        LabelVec = LabelVec1[V]
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

    LabelVec2 = copy(LabelVec1)
    dcl = ones(Int, length(LabelVec1))
    for n in 1:(length(kvec) - 1)
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

function turn_into_Hclust_merges(Z)
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

function DBHTs(D, S; branchorder = :optimal)
    @assert(branchorder ∈ BranchOrderTypes, "branchorder must be one of $BranchOrderTypes")
    @assert(issymmetric(D), "D must be symmetric")
    @assert(issymmetric(S), "S must be symmetric")

    Rpm = PMFG_T2s(S)[1]
    Apm = copy(Rpm)
    Apm[Apm .!= 0] .= D[Apm .!= 0]
    Dpm = distance_wei(Apm)[1]

    H1, Hb, Mb, CliqList, Sb = CliqHierarchyTree2s(Rpm, :unique)

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

    Z = HierarchyConstruct4s(Rpm, Dpm, T8, Adjv, Mv)
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

    hclust = Hclust(hmer, :dbht)

    return T8, Rpm, Adjv, Dpm, Mv, Z, hclust
end

export DBHTs, PMFG_T2s, distance_wei, clique3, breadth, FindDisjoint, CliqHierarchyTree2s
