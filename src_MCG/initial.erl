-module(initial).
-export([connect/0]).

connect()->
    mat:vece(3,3),%just loading the mat module
    net_adm:ping(t1@lin06).
