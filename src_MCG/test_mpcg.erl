-module(test_mpcg).
-export([test_mpcg/3]).

test_mpcg(M,N,Z)->
    RemoteNodes = [t1@lin06,hao9@lin02,hao9@lin03,hao9@lin20],%[hao3@lin01]++nodes(),
    {mean, T1M} = lists:keyfind(mean, 1, time_it:t(fun()->main:mpcg(RemoteNodes,M,N,Z) end)),
    T1M.