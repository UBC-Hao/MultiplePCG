-module(main).
-export([mpcg/4,mergeMat/2,sumMat/2,nodes_helper/2]).

mergeMat(0,X)->X;
mergeMat(X,0)->X;
mergeMat(X,Sum)->mat:merge(X,Sum).


mergeMat_tuple({X,Y},{Z,W}) -> {mergeMat(X,Z),mergeMat(Y,W)}.


sumMat(0,X) -> X;
sumMat(X,0) -> X;
sumMat(X,Sum) -> mat:add(Sum,X).

trimloc({X,Y},Size) when X<0->
    trimloc({0,Y},Size);
trimloc({X,Y},Size) when Y<0->
    trimloc({X,0},Size);
trimloc({X,Y},{M,N}) when X>M-2->
    trimloc({M-2,Y},{M,N});
trimloc({X,Y},{M,N}) when Y>N-2->
    trimloc({X,N-2},{M,N});
trimloc({X,Y},_Size) ->
    {X,Y}.

% broadcast the residual
broadcast_r(PidMap_Infos,R)->
    [Pid ! {ri,mat:subrestrict(SubInfo,R)}||{Pid,SubInfo}<-PidMap_Infos].

% using Matrix A to multiply B parallelly
% B, a matrix
% K, the number of columns of B
% Ww, the wtree(local)
amul(B,K,W2)->
    Leaf = fun(ProcState)->
            Index = wtree:get(ProcState, index),
            if Index=<K ->
               mat:amul(B,Index-1);
            true-> 0
            end
           end,
    wtree:reduce(W2,Leaf, fun mergeMat/2).
    


% subodmianinfo(K, {M,N}, L, W) divide the whole domain of size {M,N} into L*W subdomains and return the k-th subdomain
subdomaininfo(K, {M,N}, L, W)->
    Delta = 2,% Delta is overlapping lines
    SubWidth = (M-1) div L + 1,
    SubHeight = (N-1) div W + 1,
    I = K rem L,
    J = K div L,
    {X_start,Y_start} = trimloc({SubWidth * I - Delta, SubHeight * J - Delta},{M,N}),
    {X_end,Y_end} = trimloc({SubWidth * I + SubWidth + Delta, SubHeight * J + SubHeight + Delta},{M,N}),
    mat:subcreate(X_start,Y_start,X_end,Y_end).



%apply the precondioners, Ps is a list of r(residual), returns for Mr for each r
apply_precondioners(W2,W,_R,_K,SubDomainsInfos)->
    T0 = erlang:monotonic_time(),
    P_s_before_prolong = workers:retrieve(W, fun(ProcState)-> 
        Info = workers:get(ProcState,subdomain_info),
        receive {ri,Ri}->
              T1 = erlang:monotonic_time(),
              Z = mat:subsolve(Info, Ri),
              T2 = erlang:monotonic_time(),
              T_elapsed2 = time_it:elapsed(T1, T2),
              io:format("Time elapsed for solving Ri: ~p~n",[T_elapsed2]),
              Z
            end
        end),
    T1 = erlang:monotonic_time(),
    T_elapsed = time_it:elapsed(T0, T1),
    io:format("Total time elapsed for Ri: ~p~n",[T_elapsed]),

    Leaf = fun(ProcState)->
            Index = wtree:get(ProcState, index),
            if Index=<length(SubDomainsInfos) ->
               {SubInfo,RR} = {lists:nth(Index,SubDomainsInfos),lists:nth(Index,P_s_before_prolong)},
               Z = mat:subprolong(SubInfo,RR),
               {Z,mat:amul(Z)};
            true-> {0,0}
            end
           end,
    wtree:reduce(W2,Leaf,fun mergeMat_tuple/2).




mpcg_itrs(W2,W,PidMap_Infos,History,X_i,R_i,K,B,TOL,MAXITRS,SubDomainsInfos,Truncate,Itrs)->
    broadcast_r(PidMap_Infos, R_i),
    {Z_new,AZ_new} = apply_precondioners(W2,W,R_i,K,SubDomainsInfos),%most of the time is spent on this
    LHistory = length(History),
    Leaf = fun(ProcState)->
            Index = wtree:get(ProcState, index),
            if Index=<LHistory ->
                {P,Inv} = lists:nth(Index,History),
                mat:multiply(P,mat:multiply(Inv,mat:tmultiply(P,AZ_new)));
            true-> 0
            end
           end,
    

    Tmp = wtree:reduce(W2,Leaf, fun sumMat/2),

    P_new = mat:sub(Z_new, Tmp),
    NewInv = mat:inverse(mat:tmultiply(P_new,amul(P_new,K,W2))),
    Alpha_new = mat:multiply(NewInv,mat:tmultiply(P_new,R_i)),
    Tmp2 = mat:multiply(P_new,Alpha_new),
    X_new = mat:add(X_i, Tmp2),
    R_new = mat:sub(R_i, mat:amul(Tmp2)),
    Error = mat:norm(R_new)/mat:norm(B),
    io:format("~p,~n", [Error]),
    if Error<TOL ;  MAXITRS =:= 0 -> X_new;
       true -> 
            History_new = [{P_new,NewInv} | History],
            if 
                Itrs > Truncate ->
                    mpcg_itrs(W2,W,PidMap_Infos,lists:droplast(History_new),X_new, R_new, K, B, TOL,MAXITRS-1, SubDomainsInfos,Truncate,Itrs+1);
                true ->
                    mpcg_itrs(W2,W,PidMap_Infos,History_new,X_new, R_new, K, B, TOL,MAXITRS-1, SubDomainsInfos,Truncate,Itrs+1)
            end
    end.

%K is the number of process we need, and Allnodes is a list of available remote vm, we'll ping each of them and get a new list.
ping_and_gennodes(K,Allnodes) -> 
    AliveNodes = lists:filter(fun (Node)->net_adm:ping(Node)==pong end,Allnodes),
    nodes_helper(K, AliveNodes).

%No available nodes
nodes_helper(_K,[])->
    [];

nodes_helper(K, [Node])->
    [Node || _<-lists:seq(1,K)];

nodes_helper(K, [Node | AliveNodes])->
    L = length(AliveNodes)+1,
    W = K div L,
    [Node || _<-lists:seq(1,W)]++nodes_helper(K-W, AliveNodes).

retrieve_helper(_, Index) ->  {Index, self()}.


%%%%%%%%%%%%% The main function %%%%%%%%%%%%%%%%%%%%
%%% MPCG(RemoteNodes, Size, L, W)  runs mpcg algortihm with a domain of size Size by Size. And the domain is 
%%%  divided into L * W subdomains. RemoteNodes is a list of remote nodes, we'll ping each of them and get a new list.
%%%  
mpcg(RemoteNodes,Size,L_Subdomains,W_Subdomains)->
    {{M,N},{MAXITRS,TOL},B} = mat:initpde(Size),
    Truncate = 40, %We must have truncate >= L*W
    K = L_Subdomains * W_Subdomains,%K the number of subdomains i.e. the preconditioners
    Nodes = ping_and_gennodes(K,RemoteNodes),% Ping and generate K workers remotely
    %W = workers:create(Nodes,remote),%W is the worker pool for solving subproblems on GPU
    W = workers:create(Nodes,remote),
    W2 = wtree:create(Truncate),%W2 is the worker pool for scan on this machine only, the communication cost is big and also we don't have that many computers.
    Indices = lists:seq(1,Truncate),
    wtree:update(W2,index,Indices),
    {_,PidMap} = lists:unzip(lists:sort(workers:retrieve(W,fun retrieve_helper/2))),
    SubDomainsInfos = [subdomaininfo(KK,{M,N},L_Subdomains,W_Subdomains)||KK<-lists:seq(0,K - 1)],
    workers:update(W,subdomain_info,SubDomainsInfos),
    R_0 = B,
    X_0 = mat:zeros((M-1)*(N-1),1),
    PidMap_Infos = lists:zip(PidMap,SubDomainsInfos),
    broadcast_r(PidMap_Infos, R_0),
    {P_1,AP_1} = apply_precondioners(W2,W,R_0,K,SubDomainsInfos),
    Inv0 = mat:inverse(mat:tmultiply(P_1,AP_1)),
    Alpha_1 = mat:multiply(Inv0,mat:tmultiply(P_1,R_0)),%(P_1^T (A P_1))\(P'*r)
    Tmp = mat:multiply(P_1,Alpha_1),%P_1 Alpha_1
    X_1 = mat:add(X_0,Tmp),%X_1 = X_0 + P_1 alpha_1
    R_1 = mat:sub(R_0,mat:amul(Tmp)),%R_1 = R_0 - A P_1 alpha_1
    mpcg_itrs(W2,W,PidMap_Infos,[{P_1,Inv0}],X_1,R_1,K,B, TOL, MAXITRS,SubDomainsInfos,Truncate,1).

