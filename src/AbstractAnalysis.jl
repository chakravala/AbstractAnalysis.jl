module AbstractAnalysis

#   This file is part of AbstractAnalysis.jl
#   It is licensed under the AGPL license
#   AbstractAnalysis Copyright (C) 2026 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com
#       _       __               _                         _
#      / \     [  |             / |_                      / |_
#     / _ \     | |.--.   .--. `| |-'_ .--.  ,--.   .---.`| |-'
#    / ___ \    | '/'`\ \( (`\] | | [ `/'`\]`'_\ : / /'`\]| |
#  _/ /   \ \_  |  \__/ | `'.'. | |, | |    // | |,| \__. | |,
# |____|_|____|[__;.__.' [\__) )\__/[___]   \'-;__/'.___.'\__/
#      / \                     [  |                  (_)
#     / _ \     _ .--.   ,--.   | |   _   __  .--.   __   .--.
#    / ___ \   [ `.-. | `'_\ :  | |  [ \ [  ]( (`\] [  | ( (`\]
#  _/ /   \ \_  | | | | // | |, | |   \ '/ /  `'.'.  | |  `'.'.
# |____| |____|[___||__]\'-;__/[___][\_:  /  [\__) )[___][\__) )
#                                    \__.'

import Base: Fix1, Fix2
if VERSION >= v"1.12"
    import Base: Fix
end

include("magma.jl")
include("perm.jl")

export CountableArray, CountableVector, CountableMatrix
export SequenceArray, SequenceVector, SequenceMatrix
export FunctionArray, FunctionVector, FunctionMatrix, CountableFunction
export Ones, Zeros, Series, Product, AbstractCountable

abstract type AbstractCountable{T,N,F} <: AbstractArray{T,N} end
abstract type CountableFunction{T,N,F} <: AbstractCountable{T,N,F} end

counter(x::CountableFunction) = x.f
Base.broadcast(f,x::CountableFunction) = map(f,x)
Base.getindex(x::CountableFunction{T,1} where T,ϵ::AbstractFloat) = limit(x,ϵ)

struct CountableArray{T,N,F} <: CountableFunction{T,N,F}
    f::F
    n::Variables{N,Int}
end

const CountableVector{T,F} = CountableArray{T,1,F}
const CountableMatrix{T,F} = CountableArray{T,2,F}
const Ones = CountableVector{Int,typeof(one)}
const Zeros = CountableVector{Int,typeof(zero)}

CountableArray{T,N}(f::F,n::NTuple{N}) where {T,N,F} = CountableArray{T,N,F}(f,n)
CountableArray{T,N}(f::F,n::Vararg{Int,N}) where {T,N,F} = CountableArray{T,N,F}(f,n)
CountableVector{T}(f::F,n::NTuple) where {T,F} = CountableArray{T,1,F}(f,n)
CountableVector{T}(f::F,n::Int=100) where {T,F} = CountableArray{T,1,F}(f,(n,))
CountableMatrix{T}(f::F,n::NTuple) where {T,F} = CountableArray{T,2,F}(f,n)
CountableMatrix{T}(f::F,n::Int=100,m::Int=100) where {T,F} = CountableArray{T,2,F}(f,(n,))
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
Ones(n::Int) = CountableVector(one,n)
Zeros(n::Int) = CountableVector(zero,n)

(x::CountableArray{T,N,F})(n::Vararg{Int,N}) where {T,N,F} = CountableArray{T,N,F}(counter(x),n)
(x::CountableArray{T,N,F})(n::NTuple{N,Int}) where {T,N,F} = CountableArray{T,N,F}(counter(x),n)

Base.size(x::CountableArray) = Tuple(x.n)
Base.resize!(x::CountableVector,i::Int) = (x.n[1] = i; return x)
Base.getindex(x::CountableVector,i::Int) = counter(x)(i)
Base.getindex(x::CountableVector,i::Int,j::Int) = isone(j) ? counter(x)(i) : counter(x)(i,j)
Base.getindex(x::CountableArray,i::Vararg{Int}) = counter(x)(i...)

Semimagma(v::CountableVector,f=*,g=groupinverse(f)) = Semimagma(collect(v),f,g)

countmapmap(x) = Fix1(countmapmap,x)
countmapmap(x,u...) = map(f,counter(x)(u...))

Base.map(f,x::CountableArray{T,N,identity} where {T,N}) = CountableArray(f,size(x))
Base.map(f,x::CountableArray) = CountableArray(f∘counter(x),size(x))
mapmap(f,x::CountableArray{<:CountableArray}) = CountableArray(countmapmap(x),size(x))

binop(a,b,f) = Fix1(Fix1(Fix1(binop,a),b),f)
binop(a::CountableFunction,b::CountableFunction,f,x,n...) = f(counter(a)(x,n...),counter(b)(x,n...))

for fun ∈ (:*,:+,:/,:-,:^)
    @eval begin
        Base.$fun(a::Number,x::CountableArray{T,N} where T) where N = map(Fix1($fun,a),x)
        Base.$fun(x::CountableArray{T,N} where T,b::Number) where N = map(Fix2($fun,b),x)
        Base.$fun(a::CountableArray{T,N} where T,b::CountableArray{S,N} where S) where N = CountableArray(binop(a,b,$fun),min.(size(a),size(b)))
    end
end
const unaryops = [:inv,:-,:abs,:!,:~,:real,:imag,:conj,:floor,:ceil,:round,:exp,:exp2,:exp10,:log,:log2,:log10,:sinh,:cosh,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:acsch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:abs2,:angle]
for fun ∈ unaryops
    @eval Base.$fun(x::AbstractCountable) = map($fun,x)
end

LinearAlgebra.dot(a::CountableVector,b::CountableVector,Σ::Function=sum) = Σ(a*b)
LinearAlgebra.dot(a::AbstractVector,b::CountableVector,Σ::Function=sum) = Σ(a.*b)
LinearAlgebra.dot(a::CountableVector,b::AbstractVector,Σ::Function=sum) = Σ(a.*b)
LinearAlgebra.dot(a::Ones,b::CountableVector,Σ::Function=sum) = Σ(b)
LinearAlgebra.dot(a::CountableVector,b::Ones,Σ::Function=sum) = Σ(a)
LinearAlgebra.dot(a::Ones,b::Ones,Σ::Function=sum) = Σ(a)

countabletuple(x::CountableVector) = x
countabletuple(x::CountableVector,y::CountableVector) = countableproduct(x,y,tuple)
countabletuple(x::CountableVector,y::CountableVector,z::CountableVector) = countableproduct(x,y,z,tuple)
countableproduct(x::CountableVector) = x
function countableproduct(x::CountableVector,y::CountableVector,op::Function=*)
    CountableArray((i,j) -> op(counter(x)(i),counter(y)(j)),(length(x),length(y)))
end
function countableproduct(x::CountableVector,y::CountableVector,z::CountableVector,op::Function=*)
    CountableArray((i,j,k) -> op(counter(x)(i),counter(y)(j),counter(z)(k)),(length(x),length(y),length(z)))
end

struct FunctionArray{T,N,F} <: CountableFunction{T,N,F}
    f::F
    n::Variables{N,Int}
end

const FunctionVector{T,F} = FunctionArray{T,1,F}
const FunctionMatrix{T,F} = FunctionArray{T,2,F}

FunctionArray{T,N}(f::F,n::NTuple{N}) where {T,N,F} = FunctionArray{T,N,F}(f,n)
FunctionArray{T,N}(f::F,n::Vararg{Int,N}) where {T,N,F} = FunctionArray{T,N,F}(f,n)
FunctionVector{T}(f::F,n::NTuple) where {T,F} = FunctionVector{T,F}(f,n)
FunctionVector{T}(f::F,n::Int=100) where {T,F} = FunctionVector{T,F}(f,(n,))
FunctionMatrix{T}(f::F,n::NTuple) where {T,F} = FunctionMatrix{T,F}(f,n)
FunctionMatrix{T}(f::F,n::Int=100,m::Int=100) where {T,F} = FunctionMatrix{T,F}(f,(n,))
FunctionVector(f,n=100) = FunctionVector{typeof(Fix2(f,1))}(f,n)
FunctionMatrix(f,n::Int=100,m::Int=100) = FunctionMatrix{typeof(Fix2(Fix{3}(f,1),1))}(f,(n,m))
FunctionMatrix(f,n::Tuple{Int,Int}) = FunctionMatrix{typeof(Fix2(Fix{3}(f,1),1))}(f,n)
FunctionMatrix(n::Int=100,m::Int=100) = FunctionArray(^,n,m)
FunctionMatrix(n::Tuple{Int,Int}) = FunctionArray(^,n)
#FunctionArray(f,n::Vararg{Int,N}) where N = FunctionArray{typeof(f(one.(n)...)),N}(f,n)
#FunctionArray(f,n::NTuple{N,Int}) where N = FunctionArray{typeof(f(one.(n)...)),N}(f,n)
FunctionArray(n::Vararg{Int,N}) where N = FunctionArray(^,n)
FunctionArray(n::NTuple{N,Int}) where N = FunctionArray(^,n)

(x::FunctionArray{T,N})(u,n::Vararg{Int,N}) where {T,N} = CountableArray(Fix1(counter(x),u),n)
(x::FunctionArray{T,N})(u,n::NTuple{N,Int}=size(x)) where {T,N} = CountableArray(Fix1(counter(x),u),n)

Base.size(x::FunctionArray) = Tuple(x.n)
Base.resize!(x::FunctionVector,i::Int) = (x.n[1] = i; return x)
Base.getindex(x::FunctionVector,i::Int) = Fix2(counter(x),i)
Base.getindex(x::FunctionVector,i::Int,j::Int) = isone(j) ? Fix2(counter(x),i) : Fix2(Fix{3}(counter(x),j),i)
#Base.getindex(x::CountableArray,i::Vararg{Int}) = counter(x)(i...)
Base.map(f,x::FunctionArray) = FunctionArray(f∘counter(x),size(x))
Base.map(f,x::FunctionVector) = FunctionVector(f∘counter(x),size(x))
Base.map(f,x::FunctionMatrix) = FunctionMatrix(f∘counter(x),size(x))

for fun ∈ (:*,:+,:/,:-,:^)
    @eval begin
        Base.$fun(a::Number,b::FunctionArray{T,N} where T) where N = map(Fix1($fun,a),b)
        Base.$fun(a::FunctionArray{T,N} where T,b::Number) where N = map(Fix2($fun,b),a)
        Base.$fun(a::FunctionArray{T,N} where T,b::FunctionArray{S,N} where S) where N = FunctionArray(binop(a,b,$fun),min.(size(a),size(b)))
    end
end

LinearAlgebra.dot(c::AbstractArray,f::FunctionArray) = Series(c,f)
LinearAlgebra.dot(f::FunctionArray,c::AbstractArray) = Series(c,f)

functiontuple(x::CountableVector) = x
functiontuple(x::CountableVector,y::CountableVector) = functionproduct(x,y,tuple)
functiontuple(x::CountableVector,y::CountableVector,z::CountableVector) = functionproduct(x,y,z,tuple)
functionproduct(x::CountableVector) = x
function functionproduct(a::FunctionVector,b::FunctionVector,op::Function=*)
    FunctionArray((x,i,j) -> op(counter(a)(x,i),counter(b)(x,j)),(length(x),length(y)))
end
function functionproduct(a::FunctionVector,b::FunctionVector,c::FunctionVector,op::Function=*)
    FunctionArray((x,i,j,k) -> op(counter(a)(x,i),counter(b)(x,j),counter(c)(x,k)),(length(x),length(y),length(z)))
end

struct Series{N,C<:AbstractArray{S,N} where S,F<:FunctionArray{R,N} where R}
    v::C
    f::F
end

Base.length(x::Series) = length(x.f)
Series(f::FunctionArray) = Series(CountableArray(one,size(f)),f)
Base.sum(x::FunctionArray) = Series(x)
function Base.resize!(f::Series,i::Int)
    if typeof(f.v) <: AbstractCountable
        resize!(f.v,i)
        resize!(f.f,i)
    else
        resize!(f.f,length(f.v) < i ? length(f.v) : i)
    end
    return f
end

function (f::Series{1})(x,Σ=sum)
    fv = length(f) < length(f.v) ? view(f.v,Base.OneTo(length(f))) : f.v
    LinearAlgebra.dot(fv,f.f(x),Σ)
end

struct Product{N,F<:FunctionArray{S,N} where S}
    v::F
end

Base.length(x::Product) = length(x.v)
Base.prod(x::FunctionArray) = Product(x)
Base.log(x::Product) = sum(log(x.v))
function Base.resize!(f::Product,i::Int)
    resize!(f.v,i)
    return f
end

(f::Product{1})(x,Π=prod) = Π(f.v(x))

using ElasticArrays
import ElasticArrays: resize_lastdim!
export resize_lastdim!, extract, assign!

struct SequenceArray{T,N,V<:AbstractArray{T,N},F} <: AbstractCountable{T,N,F}
    v::V
    f::F
end

const SequenceVector{T,F} = SequenceArray{T,1,F}
const SequenceMatrix{T,F} = SequenceArray{T,2,F}

SequenceVector(v,f) = SequenceArray(v,f)
SequenceMatrix(v,f) = SequenceArray(v,f)

#SequenceArray{T}(v::V,f::F) where {T,N,V<:AbstractArray{T,N}} = SequenceArray{T,N,V,F}(v,f)
#SequenceArray(v::AbstractArray{T},f) where T = SequenceArray{T}(v,f)
(c::SequenceArray{T,N} where T)(n::Vararg{Int,N}) where N = c[n...]

counter(x::SequenceArray) = x.f
Base.size(c::SequenceArray) = size(c.v)
function Base.resize!(c::SequenceArray{T,1} where T,n::Int)
    m = length(c)
    resize!(c.v,n)
    F = counter(c)
    n > m && for k ∈ m+1:n
        assign!(c.v,k,F(c.v,k))
    end
    return c
end
function ElasticArrays.resize_lastdim!(c::SequenceArray{T,N} where T,n::Int) where N
    m = size(c)[end]
    resize_lastdim!(c.v,n)
    F = counter(c)
    n > m && for k ∈ m+1:n
        assign!(c.v,k,F(c.v,k))
    end
    return c
end

Base.getindex(x::SequenceArray,ϵ::AbstractFloat) = limit(x,ϵ)
Base.getindex(c::SequenceArray{T,1} where T,n::Int) = extract(c,n)
function Base.getindex(c::SequenceArray{T,N} where T,n::Vararg{Int,N}) where N
    n[end] > size(c)[end] && resize_lastdim!(c,n[end])
    c.v[n...]
end
function extract(c::SequenceArray{T,1} where T,n::Int)
    n > length(c) && resize!(c,n)
    return extract(c.v,n)
end
function extract(c::SequenceArray,n::Int)
    n > size(c)[end] && resize_lastdim!(c,n)
    return extract(c.v,n)
end

countcumsum(x) = Fix1(countcumsum,x)
countcumsum(x::CountableVector,u,k) = u[k-1] + counter(x)(k)
countcumsum(x::SequenceArray{T,1} where T,u,k) = u[k-1] + counter(x)(x,k)

countcumprod(x) = Fix1(countcumprod,x)
countcumprod(x::CountableVector,u,k) = u[k-1] * counter(x)(k)
countcumprod(x::SequenceArray{T,1} where T,u,k) = u[k-1] * counter(x)(x,k)

Base.cumsum(x::Zeros) = x
Base.cumprod(x::Zeros) = x
Base.cumsum(x::Ones) = Naturals(length(x))
Base.cumprod(x::Ones) = x
Base.cumsum(x::CountableVector) = SequenceArray(cumsum(view(x,:)),countcumsum(x))
Base.cumprod(x::CountableVector) = SequenceArray(cumprod(view(x,:)),countcumprod(x))
Base.cumsum(x::SequenceArray{T,1} where T) = SequenceArray(cumsum(x.v),countcumsum(x))
Base.cumprod(x::SequenceArray{T,1} where T) = SequenceArray(cumprod(x.v),countcumprod(x))

extract(x::AbstractVector,i) = (@inbounds x[i])
extract(x::AbstractMatrix,i) = view(x,:,i)
extract(x::AbstractArray{T,3} where T,i) = view(x,:,:,i)
extract(x::AbstractArray{T,4} where T,i) = view(x,:,:,:,i)
extract(x::AbstractArray{T,5} where T,i) = view(x,:,:,:,:,i)

assign!(x::AbstractVector,i,s) = (@inbounds x[i] = s)
assign!(x::AbstractMatrix,i,s) = (@inbounds x[:,i] = s)
assign!(x::AbstractArray{T,3} where T,i,s) = (@inbounds x[:,:,i] = s) # .= s
assign!(x::AbstractArray{T,4} where T,i,s) = (@inbounds x[:,:,:,i] = s)
assign!(x::AbstractArray{T,5} where T,i,s) = (@inbounds x[:,:,:,:,i] = s)

Base.map(f,x::SequenceArray{T,N} where T) where N = CountableArray(f∘Fix1(getindex,x),size(x))

binop(a::SequenceArray,b::SequenceArray,f,n...) = f(a[n...],b[n...])

for fun ∈ (:*,:+,:/,:-,:^)
    @eval begin
        Base.$fun(a::Number,x::SequenceArray{T,N} where T) where N = map(Fix1($fun,a),x)
        Base.$fun(x::SequenceArray{T,N} where T,b::Number) where N = map(Fix2($fun,b),x)
        Base.$fun(a::SequenceArray{T,N} where T,b::SequenceArray{S,N} where S) where N = CountableArray(binop(a,b,$fun),min.(size(a),size(b)))
    end
end

export elegantpair, elegantproduct, countabletuple, countableproduct, mapmap
export functiontuple, functionproduct

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
function elegantproduct(a::CountableVector,b::CountableVector,op=*)
    function myprod(n)
        i,j = elegantinversion1(n)
        op(counter(a)(i),counter(b)(j))
    end
    CountableVector(myprod)
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
const SternBrocot = SequenceArray([1],sternbrocot)
export SternBrocot, sternbrocot

integer(n) = iseven(n) ? n÷2 : -(n÷2)
positiverational(n) = Rational(sternbrocot(n),sternbrocot(n+1))
nonzerorational(n) = rational(n+1)
function rational(z::Int)
    n = integer(z)
    iszero(n) ? Rational(0,1) : (n>0 ? (+) : (-))(positiverational(abs(n)))
end

complextuple(n) = Complex(n...)

prime(u,i) = prime(i)

export Naturals, Integers, CantorPairs, ElegantPairs0, ElegantPairs1, ElegantPairs
export PositiveRationals, Rationals, NonzeroRationals, PrimeIntegers, PrimeCache
export GaussianNaturals, GaussianIntegers, GaussianRationals, SequenceArray

const Naturals = CountableVector(identity)
const Integers = CountableVector(integer)
const PrimeIntegers = CountableVector{Int}(prime)
const PrimeCache = SequenceArray([2],prime)
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

function Base.prod(x::typeof(Naturals))
    countprod = length(x)>20 ? factorial∘big : factorial
    val = prod(length(x)>20 ? big.(x) : view(x,:))
    Limit(1=>countprod(x[1]),length(x)=>val,length(x),supnorm(val/x[end],val),countprod)
end
Base.cumprod(x::typeof(Naturals)) = CountableVector(length(x)>20 ? factorial∘big : factorial,length(x))

sequence(i::NTuple{N,Int},j) where N = j ∈ Base.OneTo(N) ? i[j] : 1
sequence(i::Vararg{Int}) = CountableVector(Fix1(sequence,i))
SequenceArray(n::Vararg{Int}) = CountableArray(sequence,n)
SequenceArray(fun::Function,n::Vararg{Int}) = mapmap(fun,CountableArray(sequence,n))

include("metric.jl")

end # module
