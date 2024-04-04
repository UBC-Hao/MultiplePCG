-module(mat).
-export([foo/1, zeros/2,rndmat/2,add/2,sub/2,multiply/2,tmultiply/2,backslash/2,merge/2,amul/1,amul/2,initpde/1,vece/2,norm/1,subcreate/4,subsolve/2,subrestrict/2,subprolong/2,inverse/1]).
-nifs([foo/1, zeros/2,rndmat/2,add/2,sub/2,multiply/2,tmultiply/2,backslash/2,merge/2,amul/1,amul/2,initpde/1,vece/2,norm/1,subcreate/4,subsolve/2,subrestrict/2,subprolong/2,inverse/1]).
-on_load(init/0).

init() ->
    ok = erlang:load_nif("./mat_nif", 0).

foo(_X) ->
    exit(nif_library_not_loaded).
zeros(_M,_N) ->
    exit(nif_library_not_loaded).
rndmat(_M,_N) ->
    exit(nif_library_not_loaded).
add(_M,_N) ->
    exit(nif_library_not_loaded).
sub(_M,_N) ->
    exit(nif_library_not_loaded).
multiply(_M,_N) ->
    exit(nif_library_not_loaded).
tmultiply(_M,_N) ->
    exit(nif_library_not_loaded).
backslash(_M,_N) ->
    exit(nif_library_not_loaded).
merge(_M,_N) ->
    exit(nif_library_not_loaded).
amul(_M) ->
    exit(nif_library_not_loaded).
amul(_M,_I) ->
    exit(nif_library_not_loaded).
inverse(_M) ->
    exit(nif_library_not_loaded).
initpde(_N) ->
    exit(nif_library_not_loaded).
vece(_M,_N) ->
    exit(nif_library_not_loaded).
norm(_M) ->
    exit(nif_library_not_loaded).
subcreate(_M,_N,_X,_Y) ->
    exit(nif_library_not_loaded).
subsolve(_M,_N) ->
    exit(nif_library_not_loaded).
subrestrict(_M,_N) ->
    exit(nif_library_not_loaded).
subprolong(_M,_N) ->
    exit(nif_library_not_loaded).