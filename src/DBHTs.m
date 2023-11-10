function [T8,Rpm,Adjv,Dpm,Mv,Z]=DBHTs(D,S)
% Perform DBHT clustering, a deterministic technique which only requires a
% similarity matrix S, and related dissimilarity matrix D. 
% see: Song, Won-Min, T. Di Matteo, and Tomaso Aste. "Hierarchical 
% information clustering by means of topologically embedded graphs." 
% PloS one 7.3 (2012): e31929.
% This version makes extensive use of graph-theoretic filtering technique 
% called Triangulated Maximally Filtered Graph (TMFG).
%
% Function call: [T8,Rpm,Adjv,Dpm,Mv,Z]=DBHTs(D,S);
% 
% Input
% D = NxN dissimilarity matrix - e.g. a distance: 
%       D=pdist(data,'euclidean'); D=squareform(D);
% S = NxN similarity matrix (non-negative)- e.g. correlation coefficiant+1:
%       S = 2-D.^2/2; or another possible choice can be S = exp(-D);
%
% Output
%
% T8 = Nx1 cluster membership vector. 
% Rpm = NxN adjacency matrix of PMFG. 
% Adjv = Bubble cluster membership matrix from BubbleCluster8. 
% Dpm = NxN shortest path length matrix of PMFG
% Mv = NxNb bubble membership matrix. Nv(n,bi)=1 indicates vertex n is a
% vertex of bubble bi. 
% Z DBHT hierarchy
%
% This version of the code makes use of funtions from 
% Brain Connectivity Toolbox
% https://sites.google.com/site/bctnet/measures/list
%
% TA 14/10/2014
% 
Rpm =PMFG_T2s(S);
Apm=Rpm;Apm(Apm~=0)=D(Apm~=0);
Dpm=distance_wei(Apm);
[H1,Hb,Mb,CliqList,Sb]=CliqHierarchyTree2s(Rpm,'uniqueroot');
clear H1 Sb
Mb=Mb(1:size(CliqList,1),:);
Mv=[];
for n=1:size(Mb,2);
    vec=sparse(size(Rpm,1),1);
    vec(unique(CliqList((Mb(:,n)~=0),:)))=1;
    Mv=[Mv vec];
end
[Adjv,T8]=BubbleCluster8s(Rpm,Dpm,Hb,Mb,Mv,CliqList);
Z = HierarchyConstruct4s(Rpm,Dpm,T8,Adjv,Mv);
end

function [A,tri,clique3,clique4,cliqueTree]=PMFG_T2s(W)
% PMFG_T2  - this is the TMFG construction
%           Computes a Triangulated Maximally Filtered Graph (TMFG) starting from 
%           a tetrahedron and inserting recursively vertices inside 
%           existing triangles (T2 move) in order to approxiamte a
%           maxiaml planar graph with the largest total weight - non
%           negative weights
% Function call 
%           [A,tri,clique3]=PMFG_T2(W)
%           [A,tri,clique3,clique4]=PMFG_T2(W)
%           [A,tri,clique3,clique4,cliqueTree]=PMFG_T2(W)
% Input     
%           W:  a NxN matrix of -non-negative- weights 
% Output
%           A: adjacency matrix of the PMFG (with weights)
%           tri: list of triangles (triangular faces)
%           clique3: list of 3-cliques that are not triangular faces (all
%           3-cliques are given by: [tri;clique3])
%           clique4: (optional) list of all 4-cliques
%           cliqueTree: (optional) 4-cliques tree structure (adjacency matrix)
% Example:
%    [A,tri,clique3]=PMFG_T2(corr(randn(80,100)));
%    [A,tri,clique3,clique4]=PMFG_T2(corr(randn(80,100)));
%    [A,tri,clique3,clique4,cliqueTree]=PMFG_T2(corr(randn(80,100)));
%
% Reference 
%           Guido Previde Massara, Tomaso Aste & Tiziana Di Matteo, 
%           Planar Random Markov Fields and Dependence Modelling in 
%           Financial Networks, to be submitted 2014.
%
% TA 28/12/2013
%
N    = size(W,1);
if N< 9, fprintf('W Matrix too small \n'), end
if any(W<0), fprintf('W Matrix has negative elements! \n'), end
A    = sparse(N,N);     % ininzialize adjacency matrix
in_v = zeros(N,1);      % ininzialize list of inserted vertices
tri  = zeros(2*N-4,3);  % ininzialize list of triangles
clique3=zeros(N-4,3);   % ininzialize list of 3-cliques (non face-triangles)
%% find 3 vertices with largest strength
s    = sum(W.*(W>mean(W(:))),2);
[~,j]=sort(s,'descend');
in_v(1:4)  = j(1:4);
ou_v = setdiff([1:N],in_v); % list of vertices not inserted yet
%% build the tetrahedron with largest strength
tri(1,:)=in_v([1 2 3]);
tri(2,:)=in_v([2 3 4]);
tri(3,:)=in_v([1 2 4]);
tri(4,:)=in_v([1 3 4]);
A(in_v(1),in_v(2)) = 1; 
A(in_v(1),in_v(3)) = 1;
A(in_v(1),in_v(4)) = 1;
A(in_v(2),in_v(3)) = 1;
A(in_v(2),in_v(4)) = 1;
A(in_v(3),in_v(4)) = 1;
%% build initial gain table
gain = zeros(N,2*N-4);
gain(ou_v,1) = sum(W(ou_v,tri(1,:)),2);
gain(ou_v,2) = sum(W(ou_v,tri(2,:)),2);
gain(ou_v,3) = sum(W(ou_v,tri(3,:)),2);
gain(ou_v,4) = sum(W(ou_v,tri(4,:)),2);
kk = 4;  % number of triangles
for k=5:N
    %% find best vertex to add in a triangle
    if length(ou_v)==1 %special case for the last vertex
        ve = ou_v;
        v  = 1;
        [~,tr] = max(gain(ou_v,:));
    else
        [gij,v]= max(gain(ou_v,:));
        [~,tr] = max( gij );
        ve = ou_v(v(tr));
        v  = v(tr);
    end
    %% update vertex lists
    ou_v = ou_v([1:(v-1),(v+1):end]);
    in_v(k) = ve;
    %% update adjacency matrix
    A(ve,tri(tr,:))=1;
    %% update 3-clique list
    clique3(k-4,:) = tri(tr,:); 
    %% update triangle list replacing 1 and adding 2 triangles 
    tri(kk+1,:) = [tri(tr,[1,3]),ve]; % add
    tri(kk+2,:) = [tri(tr,[2,3]),ve]; % add
    tri(tr,:)   = [tri(tr,[1,2]),ve]; % replace
    %% update gain table
    gain(ve,:)=0;
    gain(ou_v,tr)  = sum(W(ou_v,tri(tr,:)),2);
    gain(ou_v,kk+1)= sum(W(ou_v,tri(kk+1,:)),2);
    gain(ou_v,kk+2)= sum(W(ou_v,tri(kk+2,:)),2);
    %% update number of triangles
    kk = kk+2; 
    if mod(k,1000)==0,fprintf('PMFG T2: %0.2f per-cent done\n',k/N*100);end
end
A = W.*((A+A')==1);
end

function [D B]=distance_wei(L)
%DISTANCE_WEI       Distance matrix
%
%   D = distance_wei(L);
%   [D B] = distance_wei(L);
%
%   The distance matrix contains lengths of shortest paths between all
%   pairs of nodes. An entry (u,v) represents the length of shortest path 
%   from node u to node v. The average shortest path length is the 
%   characteristic path length of the network.
%
%   Input:      L,      Directed/undirected connection-length matrix.
%
%   Output:     D,      distance (shortest weighted path) matrix
%               B,      number of edges in shortest weighted path matrix
%
%   Notes:
%       The input matrix must be a connection-length matrix, typically
%   obtained via a mapping from weight to length. For instance, in a
%   weighted correlation network higher correlations are more naturally
%   interpreted as shorter distances and the input matrix should
%   consequently be some inverse of the connectivity matrix. 
%       The number of edges in shortest weighted paths may in general 
%   exceed the number of edges in shortest binary paths (i.e. shortest
%   paths computed on the binarized connectivity matrix), because shortest 
%   weighted paths have the minimal weighted distance, but not necessarily 
%   the minimal number of edges.
%       Lengths between disconnected nodes are set to Inf.
%       Lengths on the main diagonal are set to 0.
%
%   Algorithm: Dijkstra's algorithm.
%
%
%   Mika Rubinov, UNSW/U Cambridge, 2007-2012.
%   Rick Betzel and Andrea Avena, IU, 2012

%Modification history
%2007: original (MR)
%2009-08-04: min() function vectorized (MR)
%2012: added number of edges in shortest path as additional output (RB/AA)
%2013: variable names changed for consistency with other functions (MR)
n=length(L);
D=inf(n);
D(1:n+1:end)=0;                             %distance matrix
B=zeros(n);                                 %number of edges matrix
for u=1:n
    S=true(1,n);                            %distance permanence (true is temporary)
    L1=L;
    V=u;
    while 1
        S(V)=0;                             %distance u->V is now permanent
        L1(:,V)=0;                          %no in-edges as already shortest
        for v=V
            T=find(L1(v,:));                %neighbours of shortest nodes
            [d wi]=min([D(u,T);D(u,v)+L1(v,T)]);
            D(u,T)=d;                       %smallest of old/new path lengths
            ind=T(wi==2);                   %indices of lengthened paths
            B(u,ind)=B(u,v)+1;              %increment no. of edges in lengthened paths
        end
        minD=min(D(u,S));
        if isempty(minD)||isinf(minD),      %isempty: all nodes reached;
            break,                          %isinf: some nodes cannot be reached
        end;

        V=find(D(u,:)==minD);
    end
end
end

%%
function [H1,H2,Mb,CliqList,Sb]=CliqHierarchyTree2s(Apm,method1);
%% ClqHierarchyTree2 looks for 3-cliques of a maximal planar graph, then 
% construct hierarchy of the cliques with the definition of 'inside' a
% clique to be a subgraph with smaller size, when the entire graph is
% made disjoint by removing the clique. Refer and cite to:
%
% Won-Min Song, T. Di Matteo, and Tomaso Aste, Nested hierarchies in planar
% graphs, Discrete Applied Mathematics, Volume 159, Issue 17, 28 October 2011, Pages 2135-2146.
%
% Function call: [H1,Hb,Mb,CliqList,Sb]=CliqHierarchyTree2s(Apm,method1);
%
% Input
% 
% Apm = N x N Adjacency matrix of a maximal planar graph
%
% method = Choose between 'uniqueroot' and 'equalroot'. Assigns
%          connections between final root cliques. Uses Voronoi
%          tesselation between tiling triangles. 
%
% Output   

% H1 = Nc x Nc adjacency matrix for 3-clique hierarchical tree where Nc is the number of 3-cliques
% H2 = Nb x Nb adjacency matrix for bubble hierarchical tree where Nb is the number of bubbles
% Mb = Nc x Nb matrix bubble membership matrix. Mb(n,bi)=1 indicates that 3-clique n belongs to bi bubble.
% CliqList = Nc x 3 matrix. Each row vector lists three vertices consisting a 3-clique in the maximal planar graph.
% Sb = Nc x 1 vector. Sb(n)=1 indicates nth 3-clique is separating. 
N=size(Apm,1);
%IndxTotal=1:N;
if issparse(Apm)~=1;
    A=sparse(Apm~=0);
else
    A=(Apm~=0);
end
[K3,E,clique]=clique3(A);
clear K3 E N3
Nc=size(clique,1);
M=sparse(N,Nc);
CliqList=clique;
clear clique
for n=1:Nc;
    cliq_vec=CliqList(n,:);
    [T,IndxNot]=FindDisjoint(A,cliq_vec);
    indx1=find(T==1);indx2=find(T==2);indx0=find(T==0);
    if length(indx1)>length(indx2);
        indx_s=[indx2(:);indx0];
        clear indx1 indx2;
    else
        indx_s=[indx1(:);indx0];
        clear indx1 indx2
    end
    if isempty(indx_s)==1;
        Sb(n)=0;
    else
        Sb(n)=length(indx_s)-3;
    end
    M(indx_s,n)=sparse(1);
    clear Indicator InsideCliq count T Temp cliq_vec IndxNot InsideCliq
end
Pred=BuildHierarchy(M);
Root=find(Pred==0);
% for n=1:length(Root);
%     Components{n}=find(M(:,Root(n))==1);
% end
clear n
switch lower(method1)
    case 'uniqueroot'
        
        if length(Root)>1;
            Pred=[Pred(:);0];
            Pred(Root)=length(Pred);
        end
        H=sparse(Nc+1,Nc+1);
        for n=1:length(Pred);
            if Pred(n)~=0;
                H(n,Pred(n))=sparse(1);
            end
        end
        H=H+H'; 
    case 'equalroot'
        if length(Root)>1;
            %RootCliq=CliqList(Root,:);
            Adj=AdjCliq(A,CliqList,Root);
        end
        H=sparse(Nc,Nc);
        for n=1:length(Pred);
            if Pred(n)~=0;
                H(n,Pred(n))=sparse(1);
            end
        end
        if isempty(Pred)~=1;
            H=H+H';H=H+Adj;
        else
            H=[];
        end   
end
H1=H;
if isempty(H1)~=1;
    [H2,Mb]=BubbleHierarchy(Pred,Sb,A,CliqList);
else
    H2=[];Mb=[];
end
H2=double(H2~=0);
Mb=Mb(1:size(CliqList,1),:);
end

%%
function Pred=BuildHierarchy(M)
Pred=zeros(size(M,2),1);
for n=1:size(M,2);
    Children=find(M(:,n)==1);
    ChildrenSum=sum(M(Children,:));
    Parents=find((ChildrenSum==length(Children)));
    Parents=Parents(Parents~=n);
    if isempty(Parents)~=1;
        ParentSum=sum(M(:,Parents));
        a=find(ParentSum==min(ParentSum));
        if length(a)==1;
            Pred(n)=Parents(a);
        else
            Pred=[];
            break
        end
    else
        Pred(n)=0;
    end
end
end

%%
function [T,IndxNot]=FindDisjoint(Adj,Cliq)
N=size(Adj,1);
Temp=Adj;
T=zeros(N,1);
IndxTotal=1:N;
IndxNot=find((IndxTotal~=Cliq(1))&(IndxTotal~=Cliq(2))&(IndxTotal~=Cliq(3)));
Temp(Cliq,:)=0;Temp(:,Cliq)=0;
%d = bfs(Temp,IndxNot(1));
d = breadth(Temp,IndxNot(1))';
d(isinf(d))=-1;
d(IndxNot(1))=0;
Indx1= d==-1;Indx2= d~=-1;
T(Indx1)=1;T(Indx2)=2;T(Cliq)=0;
clear Temp
end

%%
function Adj=AdjCliq(A,CliqList,CliqRoot)
Nc=size(CliqList,1);
N=size(A,1);
Adj=sparse(Nc,Nc);
Indicator=zeros(N,1);
for n=1:length(CliqRoot);
    Indicator(CliqList(CliqRoot(n),:))=1;
    Indi=[Indicator(CliqList(CliqRoot,1)) Indicator(CliqList(CliqRoot,2)) Indicator(CliqList(CliqRoot,3))];
    adjacent=CliqRoot(sum(Indi')==2);
    Adj(adjacent,n)=1;
end
Adj=Adj+Adj';
end

%%
function [H,Mb]=BubbleHierarchy(Pred,Sb,A,CliqList);
Nc=size(Pred,1);
Root=find(Pred==0);
CliqCount=zeros(Nc,1);
CliqCount(Root)=1;
Mb=[];
k=1;
if length(Root)>1;
    TempVec=sparse(Nc,1);TempVec(Root)=1;
    Mb=[Mb TempVec];
    clear TempVec
end
while sum(CliqCount)<Nc;
    NxtRoot=[];
    for n=1:length(Root);
        DirectChild=find(Pred==Root(n));
        TempVec=sparse(Nc,1);TempVec([Root(n);DirectChild(:)])=1;
        Mb=[Mb TempVec];
        CliqCount(DirectChild)=1;
        for m=1:length(DirectChild);
            if Sb(DirectChild(m))~=0;
                NxtRoot=[NxtRoot;DirectChild(m)];
            end
        end
        clear DirectChild TempVec
    end
    Root=unique(NxtRoot);
    k=k+1;
end
Nb=size(Mb,2);
H=sparse(Nb,Nb);
% if sum(IdentifyJoint==0)==0;
    for n=1:Nb;
        Indx= Mb(:,n)==1;
        JointSum=sum(Mb(Indx,:));
        Neigh= JointSum>=1;
        H(n,Neigh)=sparse(1);
    end
% else
%     H=[];
% end
H=H+H';H=H-diag(diag(H));
end

%%
function [K3,E,clique]=clique3(A)
% Computes the list of 3-cliques. 
% 
% Input
%
% A = NxN sparse adjacency matrix
%
% Output
%
% clique = Nc x 3 matrix. Each row vector contains the list of vertices for
% a 3-clique. 
A=A-diag(diag(A));
A=(A~=0);
A2=A^2;
P=(A2~=0).*(A~=0);
P=sparse(triu(P));
[r,c]=find(P~=0);
K3=cell(length(r),1);
for n=1:length(r);
    i=r(n);j=c(n);
    a=A(i,:).*A(j,:);
    indx=find(a~=0);
    K3{n}=indx;
    N3(n)=length(indx);
end 
E=[r c];
clique=[0 0 0];
for n=1:length(r);
    temp=K3{n};
    for m=1:length(temp);
        candidate=sort([E(n,:) temp(m)],'ascend');
        a=(clique(:,1)==candidate(1));b=(clique(:,2)==candidate(2));c=(clique(:,3)==candidate(3));
        check=(a.*b).*c;
        check=sum(check);
        if check==0;
            clique=[clique;candidate];
        end
        clear candidate check a b c
    end
end
[~,isort]=sort(clique(:,1),'ascend');
clique=clique(isort,:);
clique=clique(2:size(clique,1),:);
end

%%
function [distance,branch] = breadth(CIJ,source)
%BREADTH        Auxiliary function for breadthdist.m
%
%   [distance,branch] = breadth(CIJ,source);
%
%   Implementation of breadth-first search.
%
%   Input:      CIJ,        binary (directed/undirected) connection matrix
%               source,     source vertex
%
%   Outputs:    distance,   distance between 'source' and i'th vertex
%                           (0 for source vertex)
%               branch,     vertex that precedes i in the breadth-first search tree
%                           (-1 for source vertex)
%        
%   Notes: Breadth-first search tree does not contain all paths (or all 
%   shortest paths), but allows the determination of at least one path with
%   minimum distance. The entire graph is explored, starting from source 
%   vertex 'source'.
%
%
%   Olaf Sporns, Indiana University, 2002/2007/2008
N = size(CIJ,1);
% colors: white, gray, black
white = 0; 
gray = 1; 
black = 2;
% initialize colors
color = zeros(1,N);
% initialize distances
distance = inf*ones(1,N);
% initialize branches
branch = zeros(1,N);
% start on vertex 'source'
color(source) = gray;
distance(source) = 0;
branch(source) = -1;
Q = source;
% keep going until the entire graph is explored
while ~isempty(Q)
   u = Q(1);
   ns = find(CIJ(u,:));
   for v=ns
% this allows the 'source' distance to itself to be recorded
      if (distance(v)==0)
         distance(v) = distance(u)+1;
      end;
      if (color(v)==white)
         color(v) = gray;
         distance(v) = distance(u)+1;
         branch(v) = u;
         Q = [Q v];
      end;
   end;
   Q = Q(2:length(Q));
   color(u) = black;
end
end

%%
function [Adjv,Tc]=BubbleCluster8s(Rpm,Dpm,Hb,Mb,Mv,CliqList)
% Obtains non-discrete and discrete clusterings from the bubble topology of
% PMFG. 
% 
% Function call: [Adjv,T8]=BubbleCluster8s(Rpm,Dpm,Hb,Mb,Mv,CliqList);
%
% Input
% 
% Rpm = N x N sparse weighted adjacency matrix of PMFG
% Dpm = N x N shortest path lengths matrix of PMFG
% Hb = Undirected bubble tree of PMFG
% Mb = Nc x Nb bubble membership matrix for 3-cliques. Mb(n,bi)=1 indicates that
% 3-clique n belongs to bi bubble. 
% Mv = N x Nb bubble membership matrix for vertices. 
% CliqList = Nc x 3 matrix of list of 3-cliques. Each row vector contains
% the list of vertices for a particular 3-clique. 
%
% Output
% 
% Adjv = N x Nk cluster membership matrix for vertices for non-discrete
% clustering via the bubble topology. Adjv(n,k)=1 indicates cluster
% membership of vertex n to kth non-discrete cluster.
% Tc = N x 1 cluster membership vector. Tc(n)=k indicates cluster
% membership of vertex n to kth discrete cluster.
[Hc,Sep]=DirectHb(Rpm,Hb,Mb,Mv,CliqList);%Assign directions on the bubble tree
N=size(Rpm,1);% Number of vertices in the PMFG
indx=find(Sep==1);% Look for the converging bubbles
Adjv=[];
if length(indx)>1;
    Adjv=sparse(size(Mv,1),length(indx));%Set the non-discrete cluster membership matrix 'Adjv' at default
    % Identify the non-discrete cluster membership of vertices by each
    % converging bubble
    for n=1:length(indx);
        %[d dt p]=bfs(Hc',indx(n));
        d = breadth(Hc',indx(n))';d(isinf(d))=-1;d(indx(n))=0;
        [r c]=find(Mv(:,d~=-1)~=0);
        Adjv(unique(r),n)=1;
        clear d dt p r c
    end
    Tc=zeros(N,1);% Set the discrete cluster membership vector at default
    Bubv=Mv(:,indx);% Gather the list of vertices in the converging bubbles
    cv=find(sum(Bubv')'==1);% Identify vertices which belong to single converging bubbles
    uv=find(sum(Bubv')'>1);% Identify vertices which belong to more than one converging bubbles.
    Mdjv=sparse(N,length(indx));% Set the cluster membership matrix for vertices in the converging bubbles at default
    Mdjv(cv,:)=Bubv(cv,:);% Assign vertices which belong to single converging bubbles to the rightful clusters.
    % Assign converging bubble membership of vertices in `uv'
    for v=1:length(uv);
        v_cont=sum(bsxfun(@times,Rpm(:,uv(v)),Bubv))';% sum of edge weights linked to uv(v) in each converging bubble
        all_cont=3*(full(sum(Bubv))-2);% number of edges in converging bubble
        [mx, imx]=max(v_cont(:)./all_cont(:));% computing chi(v,b_{alpha})
        Mdjv(uv(v),imx(1))=1;% Pick the most strongly associated converging bubble
    end
    [v, ci]=find(Mdjv~=0);Tc(v)=ci;clear v ci% Assign discrete cluster memebership of vertices in the converging bubbles.
    
    Udjv=Dpm*(Mdjv*diag(1./sum(Mdjv~=0)));Udjv(Adjv==0)=inf;% Compute the distance between a vertex and the converging bubbles.
    [mn, imn]=min(Udjv(sum(Mdjv')==0,:)');% Look for the closest converging bubble
    Tc(Tc==0)=imn;% Assign discrete cluster membership according to the distances to the converging bubbles
else
    Tc=ones(N,1); % if there is one converging bubble, all vertices belong to a single cluster
end
end

%%
function [Hc,Sep]=DirectHb(Rpm,Hb,Mb,Mv,CliqList);
% Computes directions on each separating 3-clique of a maximal planar
% graph, hence computes Directed Bubble Hierarchical Tree (DBHT). 
% 
% Function call: Hc=DirectHb(Rpm,Hb,Mb,Mv,CliqList);
%
% Input
% Rpm = N x N sparse weighted adjacency matrix of PMFG
% Hb = Undirected bubble tree of PMFG
% Mb = Nc x Nb bubble membership matrix for 3-cliques. Mb(n,bi)=1 indicates that
% 3-clique n belongs to bi bubble. 
% Mv = N x Nb bubble membership matrix for vertices. 
% CliqList = Nc x 3 matrix of list of 3-cliques. Each row vector contains
% the list of vertices for a particular 3-clique. 
%
% Output
% Hc = Nb x Nb unweighted directed adjacency matrix of DBHT. Hc(i,j)=1
% indicates a directed edge from bubble i to bubble j. 
Hb=(Hb~=0);
[r,c]=find(triu(Hb)~=0);
CliqEdge=[];
for n=1:length(r);
    CliqEdge=[CliqEdge;r(n) c(n) find((Mb(:,r(n))~=0)&(Mb(:,c(n))~=0))];
end
clear r c
kb=sum(Hb~=0);
Hc=sparse(size(Mv,2),size(Mv,2));
for n=1:size(CliqEdge,1);
    Temp=Hb;
    Temp(CliqEdge(n,1),CliqEdge(n,2))=0;
    Temp(CliqEdge(n,2),CliqEdge(n,1))=0;
    d = breadth(Temp,1)';d(isinf(d))=-1;d(1)=0;
    vo=CliqList(CliqEdge(n,3),:);
    bleft=CliqEdge(n,1:2);bleft=bleft(d(bleft)~=-1);
    bright=CliqEdge(n,1:2);bright=bright(d(bright)==-1);
    [vleft c]=find(Mv(:,(d~=-1))~=0);vleft=setdiff(vleft,vo);
    [vright c]=find(Mv(:,(d==-1))~=0);vright=setdiff(vright,vo);clear c
    left=sum(sum(Rpm(vo,vleft)));
    right=sum(sum(Rpm(vo,vright)));
    if left>right;
        Hc(bright,bleft)=left;
    else
        Hc(bleft,bright)=right;
    end
    clear vleft vright vo Temp bleft bright right left
end
Sep=double((sum(Hc')==0));
Sep((sum(Hc)==0)&(kb>1))=2;
end

function Z=HierarchyConstruct4s(Rpm,Dpm,Tc,Adjv,Mv);
% Constructs intra- and inter-cluster hierarchy by utilizing Bubble
% hierarchy structure of a maximal planar graph, namely Planar Maximally Filtered Graph (PMFG). 
%
% Input
% Rpm = NxN Weighted adjacency matrix of PMFG.
% Dpm = NxN shortest path length matrix of PMFG. 
% Tc = Nx1 cluster membership vector from DBHT clustering. Tc(n)=z_i
% indicate cluster of nth vertex. 
% Adjv = Bubble cluster membership matrix from BubbleCluster8s. 
% Mv = Bubble membership of vertices from BubbleCluster8s. 
%
% Output
%
% Z = (N-1)x3 linkage matrix, in the same format as the output from matlab
% function 'linkage'. To plot the respective dendrogram, use dendrogram(Z).
% Use 'help linkage' for the details 
N=size(Dpm,1);
kvec=unique(Tc);
LabelVec1=[1:N];LinkageDist=[0];
E=sparse(1:N,Tc,ones(N,1),N,max(Tc));
Z=[];
% Intra-cluster hierarchy construction
for n=1:length(kvec);
    Mc=bsxfun(@times,E(:,kvec(n)),Mv);%Get the list of bubbles which coincide with nth cluster
    Mvv=BubbleMember(Dpm,Rpm,Mv,Mc);%Assign each vertex in the nth cluster to a specific bubble.
    Bub=find(sum(Mvv)>0);%Get the list of bubbles which contain the vertices of nth cluster 
    nc=sum(Tc==kvec(n))-1;
    %Apply the linkage within the bubbles.
    for m=1:length(Bub);
        V=find(Mvv(:,Bub(m))~=0);%Retrieve the list of vertices assigned to mth bubble.
        if length(V)>1;
            dpm=Dpm(V,V);%Retrieve the distance matrix for the vertices in V
            LabelVec=LabelVec1(V);%Initiate the label vector which labels for the clusters.
            LabelVec2=LabelVec1;
            for v=1:(length(V)-1);
                [PairLink,dvu]=LinkageFunction(dpm,LabelVec);%Look for the pair of clusters which produces the best linkage
                LabelVec((LabelVec==PairLink(1))|(LabelVec==PairLink(2)))=max(LabelVec1)+1;%Merge the cluster pair by updating the label vector with a same label.
                LabelVec2(V)=LabelVec;
                Z=DendroConstruct(Z,LabelVec1,LabelVec2,1/nc);
                nc=nc-1;
                LabelVec1=LabelVec2;
                clear PairLink dvu Vect
            end
            clear LabelVec dpm rpm LabelVec2
        end
        clear V 
    end
    V=find(E(:,kvec(n))~=0);
    dpm=Dpm(V,V);
    %Perform linkage merging between the bubbles
    LabelVec=LabelVec1(V);%Initiate the label vector which labels for the clusters.
    LabelVec2=LabelVec1;
    for b=1:(length(Bub)-1);
        [PairLink,dvu]=LinkageFunction(dpm,LabelVec);
        %[PairLink,dvu]=LinkageFunction(rpm,LabelVec);
        LabelVec((LabelVec==PairLink(1))|(LabelVec==PairLink(2)))=max(max(LabelVec1))+1;%Merge the cluster pair by updating the label vector with a same label.
        LabelVec2(V)=LabelVec;
        Z=DendroConstruct(Z,LabelVec1,LabelVec2,1/nc);
        nc=nc-1;
        LabelVec1=LabelVec2;
        clear PairLink dvu Vect
    end
    clear LabelVec V dpm rpm LabelVec2
end
%Inter-cluster hierarchy construction
LabelVec2=LabelVec1;
dcl=ones(1,length(LabelVec1));
for n=1:(length(kvec)-1);
    [PairLink,dvu]=LinkageFunction(Dpm,LabelVec1);
    %[PairLink,dvu]=LinkageFunction(Rpm,LabelVec);
    LabelVec2((LabelVec1==PairLink(1))|(LabelVec1==PairLink(2)))=max(LabelVec1)+1;%Merge the cluster pair by updating the label vector with a same label.
    dvu=unique(dcl(LabelVec1==PairLink(1)))+unique(dcl(LabelVec1==PairLink(2)));
    dcl((LabelVec1==PairLink(1))|(LabelVec1==PairLink(2)))=dvu;
    Z=DendroConstruct(Z,LabelVec1,LabelVec2,dvu);
    LabelVec1=LabelVec2;
    clear PairLink dvu
end
clear LabelVec1 
if length(unique(LabelVec2))>1;
    disp('Something Wrong in Merging. Check the codes.');
    return
end
end

%%
function [PairLink,dvu]=LinkageFunction(d,labelvec);
lvec=unique(labelvec);
Links=[];
for r=1:(length(lvec)-1);
    vecr=(labelvec==lvec(r));
    for c=(r+1):length(lvec);
        vecc=(labelvec==lvec(c));
        dd=d((vecr|vecc),(vecr|vecc));
        Links=[Links;[lvec(r) lvec(c) max(dd(dd~=0))]];
        clear vecc
    end
end
[dvu imn]=min(Links(:,3));PairLink=Links(imn,1:2);
end

%%
function Mvv=BubbleMember(Dpm,Rpm,Mv,Mc);
Mvv=sparse(size(Mv,1),size(Mv,2));
vu=find(sum(Mc')>1);
v=find(sum(Mc')==1);
Mvv(v,:)=Mc(v,:);
for n=1:length(vu);
    bub=find(Mc(vu(n),:)~=0);
    vu_bub=sum(bsxfun(@times,Rpm(:,vu(n)),Mv(:,bub)))';
    all_bub=diag(Mv(:,bub)'*Rpm*Mv(:,bub))/2;
    frac=vu_bub./all_bub;
    [mx, imx]=max(frac);
    Mvv(vu(n),bub(imx(1)))=1;
    clear v_bub all_bub frac bub vec mx imx
end
end
    
%%
function Z=DendroConstruct(Zi,LabelVec1,LabelVec2,LinkageDist);
indx=(bsxfun(@eq,LabelVec1',LabelVec2')~=1);
if length(unique(LabelVec1(indx)))~=2;
    disp('Check the codes');
    return
end
Z=[Zi;[sort(unique(LabelVec1(indx))) LinkageDist]];
end