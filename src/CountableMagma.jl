module CountableMagma

#   This file is part of CountableMagma.jl
#   It is licensed under the AGPL license
#   Magma Copyright (C) 2026 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com
#
# .|'''',                            ||            '||     '||`        
# ||                                 ||             ||      ||         
# ||      .|''|, '||  ||` `||''|,  ''||''   '''|.   ||''|,  ||  .|''|, 
# ||      ||  ||  ||  ||   ||  ||    ||    .|''||   ||  ||  ||  ||..|| 
# `|....' `|..|'  `|..'|. .||  ||.   `|..' `|..||. .||..|' .||. `|...  
#
#       .        :    :::.      .,-:::::/  .        :    :::.
#       ;;,.    ;;;   ;;`;;   ,;;-'````'   ;;,.    ;;;   ;;`;;
#       [[[[, ,[[[[, ,[[ '[[, [[[   [[[[[[/[[[[, ,[[[[, ,[[ '[[,
#       $$$$$$$$"$$$c$$$cc$$$c"$$c.    "$$ $$$$$$$$"$$$c$$$cc$$$c
#       888 Y88" 888o888   888,`Y8bo,,,o88o888 Y88" 888o888   888,
#       MMM  M'  "MMMYMM   ""`   `'YMUP"YMMMMM  M'  "MMMYMM   ""`

using LinearAlgebra, StaticVectors, Combinatorics

export Semimagma, grouplaw, groupinverse, compose, order, orders

struct Semimagma{T,F,G} <: DenseVector{T}
    v::Vector{T}
    Semimagma(v::Vector{T},f=*,g=groupinverse(f)) where T = new{T,f,g}(v)
end

Base.@pure grouplaw(::Semimagma{T,F} where T) where F = F
Base.@pure groupinverse(::Semimagma{T,F,G} where {T,F}) where G = G
Base.@pure groupinverse(::typeof(*)) = inv
Base.@pure groupinverse(::typeof(+)) = -
(::Semimagma{T,F} where T)(a,b) where F = F(a,b)

Base.size(G::Semimagma) = size(G.v)
Base.getindex(G::Semimagma,i::Int) = G.v[i]

order(G::Semimagma) = length(G)
order(n,f=*,g=groupinverse(f)) = order(group(n,f,g))
orders(G::Semimagma{T,X,Y} where T) where {X,Y} = order.(G,X,Y)
Base.length(G::Semimagma) = length(G.v)
Base.abs(G::Semimagma) = length(G)

Base.iseven(G::Semimagma) = prod(iseven.(G.v))
Base.isodd(G::Semimagma) = prod(isodd.(G.v))

Base.:(==)(G::Semimagma,H::Semimagma) = G ⊆ H && H ⊆ G

function gexp(a,n,op)
    if isone(n)
        return a
    else
        op(a,gexp(a,n-1,op))
    end
end
gequal(a,b) = a ≈ b
function Base.in(g,G::Semimagma)
    for h ∈ G.v
        gequal(g,h) && (return true)
    end
    return false
end

compose(g::T,H::Semimagma{T,X,Y},F=X) where {T,X,Y} = Semimagma(F.(Ref(g),H.v),X,Y)
compose(H::Semimagma{T,X,Y},g::T,F=X) where {T,X,Y} = Semimagma(F.(H.v,Ref(g)),X,Y)
function compose(G::Semimagma{T,X,Y} where T,H::Semimagma{S,X,Y} where S,F=X) where {X,Y}
    out = Semimagma(eltype(G.v)[],X,Y)
    for g ∈ G.v
        for h ∈ H.v
            gh = F(g,h)
            gh ∉ out && push!(out.v,gh)
        end
    end
    out
end

Base.:∘(g::T,H::Semimagma{T}) where T = compose(g,H)
Base.:∘(H::Semimagma{T},g::T) where T = compose(H,g)
Base.:∘(G::Semimagma,H::Semimagma) = compose(G,H)

for fun ∈ (:*,:+)
    @eval begin
        Base.$fun(g::Number,H::Semimagma) = compose(g,H,$fun)
        Base.$fun(H::Semimagma,g::Number) = compose(H,g,$fun)
        Base.$fun(g::T,H::Semimagma{T}) where T = compose(g,H,$fun)
        Base.$fun(H::Semimagma{T},g::T) where T = compose(H,g,$fun)
        Base.$fun(G::Semimagma,H::Semimagma) = compose(G,H,$fun)
    end
end

export cayley, group, magma, group, issubgroup, isnormal, subgroup
export center, centralizer, normalizer, commutator, leftcosets, rightcosets, subsemigroup

cayley(G::Semimagma) = [G(g,h) for g ∈ G.v, h ∈ G.v]

for fun ∈ (:iscategory,:ismonoid,:isgroupoid,:issemicategory,:issemigroup,:ismagma,:iscyclic,:isabelian,:isassociative,:isinvertible,:isgroup)
    @eval begin
        $fun(G) = false
        export $fun
    end
end

iscategory(G::Semimagma) = isone(G) ∈ G && issemicategory(G)
ismonoid(G::Semimagma) = iscategory(G)
isgroupoid(G::Semimagma) = isgroup(G)
issemicategory(G::Semimagma) = issemigroup(G)
issemigroup(G::Semimagma) = ismagma(G) && isassociative(G)
isgroup(G::Semimagma) = isinvertible(G) && issemigroup(G)

function isassociative(G::Semimagma)
    for f ∈ G.v
        for g ∈ G.v
            for h ∈ G.v
                !gequal(G(G(f,g),h), G(f,G(g,h))) && (return false)
            end
        end
    end
    return true
end

function ismagma(G::Semimagma)
    for g ∈ G.v
        for h ∈ G.v
            G(g,h) ∉ G && (return false)
        end
    end
    return true
end

function isinvertible(G::Semimagma)
    F = groupinverse(G)
    for g ∈ G.v
        try 
            F(g) ∉ G && (return false)
        catch
            return false
        end
    end
    return true
end

iscyclic(G::Semimagma{T,X,Y} where T) where {X,Y} = G == group(G[1],X,Y) || G == group(G[2],X,Y)

function magma(p,F::Function=*,G::Function=groupinverse(F))
    out = Semimagma([p],F,G)
    p ∉ out && push!(out.v,p)
    pn = F(p,p)
    while pn ∉ out
        push!(out.v,pn)
        pn = F(pn,p)
    end
    out
end
function magma(p::AbstractVector,F::Function=*,G::Function=groupinverse(F))
    magma(Semimagma(p,F,G))
end
function magma(G::Semimagma{T,X,Y} where T,out=Semimagma(copy(G.v),X,Y)) where {X,Y}
    i = 1
    while i <= length(out)
        g = out[i]
        j = 1
        while j <= length(out)
            gh = X(g,out[j])
            gh ∉ out && push!(out.v,gh)
            j += 1
        end
        i += 1
    end
    return out
end
function group(G::Semimagma{T,X,Y} where T,out::Semimagma=Semimagma(copy(G.v),X,Y)) where {X,Y}
    F = groupinverse(G)
    for i ∈ 1:length(out)
        ig = F(out[i])
        ig ∉ out && push!(out.v,ig)
    end
    return magma(G,out)
end
function group(p::AbstractVector,F::Function=*,G::Function=groupinverse(F))
    group(Semimagma(p,F,G))
end
function group(p,F::Function=*,G::Function=groupinverse(F))
    magma(p,F,G)
end

function subsemigroup(G::Semimagma{T,X,Y} where T,out=Semigroup(copy(G.v),X,Y)) where {X,Y}
    i = 1
    while i ≤ length(out)
        g = out[i]
        bool = false
        for j ∈ 1:length(out)
            bool = bool || (X(g,out[j]) ∈ out)
        end
        if bool
            i += 1
        else
            deleteat!(out.v,i)
        end
    end
    return out
end

function subgroup(G::Semimagma{T,X,Y} where T,out=Semimagma(copy(G.v),X,Y)) where {X,Y}
    i = 1
    while i ≤ length(out)
        try
            if Y(out[i]) ∉ out
                deleteat!(out.v,i)
            else
                i += 1
            end
        catch
            deleteat!(out.v,i)
        end
    end
    return subsemigroup(G,out)
end

issubgroup(H::Semimagma,G::Semimagma) = H ⊆ G && isgroup(H)
Base.issubset(H::Semimagma,G::Semimagma) = prod(H.v .∈ Ref(G))

function isabelian(G::Semimagma)
    for g ∈ G.v
        for h ∈ G.v
            !gequal(G(g,h), G(h,g)) && (return false)
        end
    end
    return true
end

#center(G::Semimagma) = centralizer(G,G)
function center(G::Semimagma{T,X,Y} where T) where {X,Y}
    out = copy(G.v)
    i = 1
    while i ≤ length(out)
        g = out[i]
        j = 1
        while j ≤ length(out)
            h = out[j]
            if gequal(X(g,h), X(h,g))
                j += 1
            else
                deleteat!(out,j)
                j < i && (i -= 1)
            end
        end
        i += 1
    end
    return Semimagma(out,X,Y)
end

function centralizer(H::Semimagma{T,X,Y} where T,G::(Semimagma{S,X,Y} where S)=defaultgroup(H)) where {X,Y}
    out = copy(G.v)
    i = 1
    while i ≤ length(out)
        g = out[i]
        commute = true
        for h ∈ H.v
            if !gequal(X(g,h), X(h,g))
                commute = false
                break
            end
        end
        if commute
            i += 1
        else
            deleteat!(out,i)
        end
    end
    return Semimagma(out,X,Y)
end

function isnormal(H::Semimagma,G=defaultgroup(H))
    for g ∈ G.v
        g∘H ≠ H∘g && (return false)
    end
    return true
end

function normalizer(H::Semimagma{T,X,Y} where T,G::Semimagma{S,X,Y} where S=defaultgroup(H)) where {X,Y}
    out = copy(G.v)
    i = 1
    while i ≤ length(out)
        g = out[i]
        if g∘H ≠ H∘g
            deleteat!(out,i)
        else
            i += 1
        end
    end
    return Semimagma(out,X,Y)
end

function commutator(G::Semimagma{T,F,Q} where T,H::(Semimagma{S,F,Q} where S)=G) where {F,Q}
    out = Semimagma(eltype(G.v)[],F,Q)
    for g ∈ G.v
        for h ∈ H.v
            gh = F(F(Q(g),Q(h)),F(g,h))
            gh ∉ out && push!(out.v,gh)
        end
    end
    return group(G,out)
end

Base.:/(G::Semimagma,N::Semimagma) = leftcosets(N,G)

function leftcosets(H::Semimagma{T,X,Y},G=defaultgroup(H)) where {T,X,Y}
    out = Semimagma(typeof(H)[],X,Y)
    for g ∈ G.v
        gH = g∘H
        gH ∉ out && push!(out.v,gH)
    end
    return out
end
function rightcosets(H::Semimagma{T,X,Y},G=defaultgroup(H)) where {T,X,Y}
    out = Semimagma(typeof(H)[],X,Y)
    for g ∈ G.v
        Hg = H∘g
        Hg ∉ out && push!(out.v,Hg)
    end
    return out
end

include("perm.jl")
include("count.jl")

end # module Magma
