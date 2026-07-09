
#   This file is part of Magma.jl
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

export CountableArray, CountableVector, CountableMatrix, CountableCache

mutable struct CountableArray{T,N,F} <: AbstractArray{T,N}
    n::NTuple{N,Int}
end

const CountableVector{T,F} = CountableArray{T,1,F}
const CountableMatrix{T,F} = CountableArray{T,2,F}

CountableArray{T,N}(f,n::NTuple{N}) where {T,N} = CountableArray{T,N,f}(n)
CountableArray{T,N}(f,n::Vararg{Int,N}) where {T,N} = CountableArray{T,N,f}(n)
CountableVector{T}(f,n::NTuple) where T = CountableVector{T,f}(n)
CountableVector{T}(f,n::Int=100) where T = CountableVector{T,f}((n,))
CountableMatrix{T}(f,n::NTuple) where T = CountableMatrix{T,f}(n)
CountableMatrix{T}(f,n::Int=100,m::Int=100) where T = CountableMatrix{T,f}((n,))
CountableVector(f,n=100) = CountableVector{typeof(f(1))}(f,n)
CountableVector(n::Int=100) = Naturals(n)
CountableVector(n::Tuple{Int}) = Naturals(n[1])
CountableMatrix(f,n::Int=100,m::Int=100) = CountableMatrix{typeof(f(1,1))}(f,(n,m))
CountableMatrix(f,n::Tuple{Int,Int}) = CountableMatrix{typeof(f(1,1))}(f,n)
CountableMatrix(n::Int=100,m::Int=100) = CountableArray(n,m)
CountableMatrix(n::Tuple{Int,Int}) = CountableArray(n)
CountableArray(f,n::Vararg{Int,N}) where N = CountableArray{typeof(f(one.(n)...)),N}(f,n)
CountableArray(f,n::NTuple{N,Int}) where N = CountableArray{typeof(f(one.(n)...)),N}(f,n)
CountableArray(n::Vararg{Int,N}) where N = countabletuple(Naturals.(n)...)
CountableArray(n::NTuple{N,Int}) where N = countabletuple(Naturals.(n)...)

(::CountableArray{T,N,F})(n::Vararg{Int,N}) where {T,N,F} = CountableArray{T,N,F}(n)
(::CountableArray{T,N,F})(n::NTuple{N,Int}) where {T,N,F} = CountableArray{T,N,F}(n)

Base.@pure counter(::CountableArray{T,N,F} where {T,N}) where F = F
Base.getindex(::CountableVector{T,F} where T,i::Int) where F = F(i)
Base.getindex(::CountableVector{T,F} where T,i::Int,j::Int) where F = isone(j) ? F(i) : F(i,j)
Base.getindex(::CountableArray{T,N,F} where T,i::Vararg{Int}) where {N,F} = F(i...)
Base.size(x::CountableArray) = x.n
Base.resize!(x::CountableVector,i::Int) = (x.n = i; return x)

Semimagma(v::CountableVector,f=*,g=groupinverse(f)) = Semimagma(collect(v),f,g)

Base.map(f,x::CountableArray{T,N,F} where {T,N}) where F = CountableArray(f∘F,x.n)
mapmap(f,x::CountableArray{<:CountableArray,N,F} where N) where F = CountableArray((u...)->map(f,F(u...)),x.n)
Base.broadcast(f,x::CountableArray) = map(f,x)

for fun ∈ (:*,:+,:/,:-,:^)
    @eval begin
        Base.$fun(a::Number,x::CountableVector{T,F} where T) where F = CountableVector(n->$fun(a,F(n)),x.n)
        Base.$fun(x::CountableVector{T,F} where T,b::Number) where F = CountableVector(n->$fun(F(n),b),x.n)
        Base.$fun(a::Number,x::CountableArray{T,N,F} where T) where {N,F} = CountableVector((n::Vararg{Int,N})->$fun(a,F(n...)),x.n)
        Base.$fun(x::CountableArray{T,N,F} where T,b::Number) where {N,F} = CountableVector((n::Vararg{Int,N})->$fun(F(n...),b),x.n)
    end
end

export elegantpair, elegantproduct, countabletuple, countableproduct, mapmap

function cantorinversion(n::Int)
    w = Int(floor((sqrt(8n+1)-1)/2))
    t = (w^2+2)÷2
    return (n-t,w-n+t)
end

function elegantinversion(n::Int)
    fsn = Int(floor(sqrt(n)))
    nfsn2 = n - fsn^2
    if nfsn2 < fsn
        (nfsn2,fsn)
    else
        (fsn,nfsn2-fsn)
    end
end

elegantinversion1(n::Int) = elegantinversion(n,1)
function elegantinversion(n::Int,k::Int)
    fsn = Int(floor(sqrt(n-k)))
    nfsn2 = n-fsn^2-k
    if nfsn2 < fsn
        (nfsn2+k,fsn+k)
    else
        (fsn+k,nfsn2-fsn+k)
    end
end

elegantpair(a::CountableVector,b::CountableVector) = elegantproduct(a,b,tuple)
function elegantproduct(a::CountableVector{T,F} where T,b::CountableVector{S,G} where S,op=*) where {F,G}
    function myprod(n)
        i,j = elegantinversion1(n)
        op(F(i),G(j))
    end
    CountableVector(myprod)
end

countabletuple(x::CountableVector) = x
countabletuple(x::CountableVector,y::CountableVector) = countableproduct(x,y,tuple)
countabletuple(x::CountableVector,y::CountableVector,z::CountableVector) = countableproduct(x,y,z,tuple)
countableproduct(x::CountableVector) = x
function countableproduct(x::CountableVector{T,F} where T,y::CountableVector{S,G} where S,op::Function=*) where {F,G}
    CountableArray((i,j) -> op(F(i),G(j)),(length(x),length(y)))
end
function countableproduct(x::CountableVector{T,F} where T,y::CountableVector{S,G} where S,z::CountableVector{R,H} where R,op::Function=*) where {F,G,H}
    CountableArray((i,j,k) -> op(F(i),G(j),H(k)),(length(x),length(y),length(z)))
end

using ElasticArrays

struct CountableCache{T,N,V<:AbstractArray{T,N},F} <: DenseArray{T,N}
    v::V
end

CountableCache{T}(v::V,f) where {T,N,V<:AbstractArray{T,N}} = CountableCache{T,N,V,f}(v)
CountableCache(v::AbstractArray{T},f) where T = CountableCache{T}(v,f)
(c::CountableCache{T,N} where T)(n::Vararg{Int,N}) where N = c[n...]

Base.size(c::CountableCache) = size(c.v)
function Base.getindex(c::CountableCache{T,1,V,F} where {T,V},n::Int) where F
    N = length(c)
    n ≤ N && (return extract(c.v,n))
    resize!(c.v,n)
    for k ∈ N+1:n
        assign!(c.v,k,F(c.v,k))
    end
    return extract(c.v,n)
end

function Base.cumsum(x::CountableVector{T,F} where T) where F
    CountableCache(cumsum(view(x,:)),(x,k) -> x[k-1] + F(k))
end

#ElasticArrays.resize_lastdim!(c::CountableCache,i) = ElasticArrays.resize_lastdim!(c.v,i)

extract(x::AbstractVector,i) = (@inbounds x[i])
extract(x::AbstractMatrix,i) = (@inbounds x[:,i])
extract(x::AbstractArray{T,3} where T,i) = (@inbounds x[:,:,i])
extract(x::AbstractArray{T,4} where T,i) = (@inbounds x[:,:,:,i])
extract(x::AbstractArray{T,5} where T,i) = (@inbounds x[:,:,:,:,i])

assign!(x::AbstractVector,i,s) = (@inbounds x[i] = s)
assign!(x::AbstractMatrix,i,s) = (@inbounds x[:,i] = s)
assign!(x::AbstractArray{T,3} where T,i,s) = (@inbounds x[:,:,i] = s) # .= s
assign!(x::AbstractArray{T,4} where T,i,s) = (@inbounds x[:,:,:,i] = s)
assign!(x::AbstractArray{T,5} where T,i,s) = (@inbounds x[:,:,:,:,i] = s)

export FixedPoint, fixedpoint, fixedpointhold, fixedpointerror, errornorm

struct FixedPoint{T,F}
    v0::T
    v::T
    n::Int
    FixedPoint(v0::T,v::T,n::Int,F) where T = new{T,F}(v0,v,n)
end

function Base.show(io::IO,x::FixedPoint)
    println(io,"(f^$(x.n))($(x.v0)) = $(x.v)")
    #show(io,x.v)
end

function Base.sum(x::CountableVector{T,F} where T) where F
    countsum(u) = (first(u)+1) => (last(u)+F(first(u)+1))
    FixedPoint(1=>x[1],length(x)=>sum(view(x,:)),length(x),countsum)
end

function Base.getindex(x::FixedPoint{T,F} where T,i::Int) where F
    i == x.n && (return x)
    xn = i<x.n ? x.v0 : x.v
    for k ∈ (i<x.n ? (2:i) : (x.n+1:i))
        xn = F(xn)
    end
    return FixedPoint(x.v0,xn,i,F)
end

errornorm(a::Number,b::Number,ϵ=5eps()) = norm(a-b)
fixedpointerror(f,x,ϵ=5eps()) = fixedpoint(f,x,ϵ,Val(true))
fixedpoint(f,x,n::Int,v::Val=Val(false)) = fixedpoint(f,x,1:n,v)
function fixedpoint(f,x,n::AbstractVector{Int},::Val{print}=Val(false)) where print
    out = print ? zeros(length(n)) : nothing
    xn = x
    for i ∈ n
        if print
            xi = f(xn)
            out[i] = errornorm(xi,xn)
            xn = xi
        else
            xn = f(xn)
        end
    end
    return print ? (xn,out) : xn
end
function fixedpoint(f,x,ϵ::AbstractFloat=5eps(),::Val{print}=Val(false)) where print
    change = 5ϵ
    print && (out = Float64[])
    while change > ϵ
        xi = f(x)
        change = errornorm(xi,x)
        print && push!(out,change)
        x = xi
    end
    return print ? (x,out) : x
end

function fixedpointhold(f,x,n::AbstractVector{Int},::Val{print}=Val(false)) where print
    print && (out = zeros(length(n)))
    xi = x
    for i ∈ n
        if print
            y = f(x,xi) #  hold x constant, iterate xi
            out[i] = errornorm(y,xi)
            xi = y
        else
            xi = f(x,xi) # hold x constant, iterate xi
        end
    end
    return xi
end

function sternbrocot(n::Int)
    if isone(n)
        1
    elseif iseven(n)
        sternbrocot(n÷2)
    else
        kk = (n-1)÷2
        sternbrocot(kk) + sternbrocot(kk+1)
    end
end
function sternbrocot(a,n)
    if iseven(n)
        @inbounds a[n÷2]
    else
        k = (n-1)÷2
        @inbounds a[k] + a[k+1]
    end
end
const SternBrocot = CountableCache([1],sternbrocot)
export SternBrocot, sternbrocot

integer(n) = iseven(n) ? n÷2 : -(n÷2)
positiverational(n) = Rational(sternbrocot(n),sternbrocot(n+1))
nonzerorational(n) = rational(n+1)
function rational(z::Int)
    n = integer(z)
    iszero(n) ? Rational(0,1) : (n>0 ? (+) : (-))(positiverational(abs(n)))
end

complextuple(n) = Complex(n...)

export Naturals, Integers, CantorPairs, ElegantPairs0, ElegantPairs1, ElegantPairs
export PositiveRationals, Rationals, NonzeroRationals
export GaussianNaturals, GaussianIntegers, GaussianRationals

const Naturals = CountableVector(identity)
const Integers = CountableVector(integer)
const CantorPairs = CountableVector(cantorinversion)
const ElegantPairs0 = CountableVector(elegantinversion)
const ElegantPairs1 = CountableVector(elegantinversion1)
const ElegantPairs = ElegantPairs1
const PositiveRationals = CountableVector(positiverational)
const Rationals = CountableVector(rational)
const NonzeroRationals = CountableVector(nonzerorational)
const GaussianNaturals = map(complextuple,ElegantPairs1)
const GaussianIntegers = map(complextuple,elegantpair(Integers,Integers))
const GaussianRationals = map(complextuple,elegantpair(Rationals,Rationals))

export SequenceArray

sequence(i::NTuple{N,Int},j) where N = j ∈ Base.OneTo(N) ? i[j] : 1
sequence(i::Vararg{Int}) = CountableVector(Base.Fix{1}(sequence,i))
SequenceArray(n::Vararg{Int}) = CountableArray(sequence,n)
SequenceArray(fun::Function,n::Vararg{Int}) = mapmap(fun,CountableArray(sequence,n))


