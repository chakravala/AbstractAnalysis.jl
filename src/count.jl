
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
CountableMatrix(f,n=100,m=100) = CountableVector{typeof(f(1))}(f,(n,m))
CountableArray(f,n::Vararg{Int,N}) where N = CountableArray{typeof(f(one.(n)...)),N}(f,n)
CountableArray(f,n::NTuple{N,Int}) where N = CountableArray{typeof(f(one.(n)...)),N}(f,n)

(::CountableArray{T,N,F})(n::Vararg{Int,N}) where {T,N,F} = CountableArray{T,N,F}(n)
(::CountableArray{T,N,F})(n::NTuple{N,Int}) where {T,N,F} = CountableArray{T,N,F}(n)

Base.@pure counter(::CountableArray{T,N,F} where {T,N}) where F = F
Base.getindex(::CountableVector{T,F} where T,i::Int) where F = F(i)
Base.getindex(::CountableArray{T,N,F} where T,i::Vararg{Int,N}) where {N,F} = F(i...)
Base.size(x::CountableArray) = x.n
Base.resize!(x::CountableVector,i::Int) = (x.n = i; return x)

Base.map(f,x::CountableArray{T,N,F} where {T,N}) where F = CountableArray(f∘F,x.n)
Base.broadcast(f,x::CountableArray) = map(f,x)

for fun ∈ (:*,:+,:/,:-,:^)
    @eval begin
        Base.$fun(a::Number,x::CountableVector{T,F} where T) where F = CountableVector(n->$fun(a,F(n)),x.n)
        Base.$fun(x::CountableVector{T,F} where T,b::Number) where F = CountableVector(n->$fun(F(n),b),x.n)
        Base.$fun(a::Number,x::CountableArray{T,N,F} where T) where {N,F} = CountableVector((n::Vararg{Int,N})->$fun(a,F(n...)),x.n)
        Base.$fun(x::CountableArray{T,N,F} where T,b::Number) where {N,F} = CountableVector((n::Vararg{Int,N})->$fun(F(n...),b),x.n)
    end
end

export elegantpair, elegantproduct, countablepair, countableproduct

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

countablepair(x::CountableVector,y::CountableVector) = countableproduct(x,y,tuple)
function countableproduct(x::CountableVector{T,F} where T,y::CountableVector{S,G} where S,op=*) where {F,G}
    CountableArray((i,j) -> op(F(i),G(j)),(length(x),length(y)))
end
function countableproduct(x::CountableVector{T,F} where T,y::CountableVector{S,G} where S,z::CountableVector{R,H} where R,op=*) where {F,G,H}
    CountableArray((i,j,k) -> op(F(i),G(j),H(k)),(length(x),length(y),length(z)))
end


struct CountableCache{T,F} <: DenseVector{T}
    v::Vector{T}
end

CountableCache{T}(v::Vector{T},f) where T = CountableCache{T,f}(v)
CountableCache(v::Vector{T},f) where T = CountableCache{T}(v,f)
(c::CountableCache)(n::Int) = c[n]

Base.size(c::CountableCache) = size(c.v)
function Base.getindex(c::CountableCache{T,F} where T,n::Int) where F
    N = length(c)
    n ≤ N && (return c.v[n])
    resize!(c.v,n)
    for k ∈ N+1:n
        @inbounds  c.v[k] = F(c.v,k)
    end
    return c.v[end]
end

function Base.cumsum(x::CountableVector{T,F} where T) where F
    out = cumsum(view(x,:))
    CountableCache(out,(x,k) -> x[k-1] + F(k))
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

function rational(z::Int)
    n = Integers[z]
    iszero(n) ? Rational(0,1) : (n>0 ? (+) : (-))(PositiveRationals[abs(n)])
end

complextuple(n) = Complex(n...)

export Naturals, Integers, CantorPairs, ElegantPairs0, ElegantPairs1, ElegantPairs
export PositiveRationals, Rationals, NonzeroRationals
export GaussianNaturals, GaussianIntegers, GaussianRationals

const Naturals = CountableVector(identity)
const Integers = CountableVector(n -> iseven(n) ? n÷2 : -(n÷2))
const CantorPairs = CountableVector(cantorinversion)
const ElegantPairs0 = CountableVector(elegantinversion)
const ElegantPairs1 = CountableVector(elegantinversion1)
const ElegantPairs = ElegantPairs1
const PositiveRationals = CountableVector(n -> Rational(sternbrocot(n),sternbrocot(n+1)))
const Rationals = CountableVector(rational)
const NonzeroRationals = CountableVector(n -> rational(n+1))
const GaussianNaturals = map(complextuple,ElegantPairs1)
const GaussianIntegers = map(complextuple,elegantpair(Integers,Integers))
const GaussianRationals = map(complextuple,elegantpair(Rationals,Rationals))

Semimagma(v::CountableVector,f=*,g=groupinverse(f)) = Semimagma(collect(v),f,g)


