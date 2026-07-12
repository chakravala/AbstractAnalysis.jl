
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
export FunctionArray, FunctionVector, FunctionMatrix, CountableFunction
export OnesArray, OnesVector, OnesMatrix, Series, Product, AbstractCountable

abstract type AbstractCountable{T,N,F} <: AbstractArray{T,N} end
abstract type CountableFunction{T,N,F} <: AbstractCountable{T,N,F} end

counter(x::CountableFunction) = x.f
Base.size(x::CountableFunction) = x.n
Base.resize!(x::CountableFunction,i::Int) = (x.n = (i,); return x)
Base.broadcast(f,x::CountableFunction) = map(f,x)

mutable struct CountableArray{T,N,F} <: CountableFunction{T,N,F}
    f::F
    n::NTuple{N,Int}
end

const CountableVector{T,F} = CountableArray{T,1,F}
const CountableMatrix{T,F} = CountableArray{T,2,F}
const OnesArray{N} = CountableArray{Int,N,typeof(one)}
const OnesVector = OnesArray{1}
const OnesMatrix = OnesArray{2}

CountableArray{T,N}(f::F,n::NTuple{N}) where {T,N,F} = CountableArray{T,N,F}(f,n)
CountableArray{T,N}(f::F,n::Vararg{Int,N}) where {T,N,F} = CountableArray{T,N,F}(f,n)
CountableVector{T}(f::F,n::NTuple) where {T,F} = CountableVector{T,F}(f,n)
CountableVector{T}(f::F,n::Int=100) where {T,F} = CountableVector{T,F}(f,(n,))
CountableMatrix{T}(f::F,n::NTuple) where {T,F} = CountableMatrix{T,F}(f,n)
CountableMatrix{T}(f::F,n::Int=100,m::Int=100) where {T,F} = CountableMatrix{T,F}(f,(n,))
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

(x::CountableArray{T,N,F})(n::Vararg{Int,N}) where {T,N,F} = CountableArray{T,N,F}(counter(x),n)
(x::CountableArray{T,N,F})(n::NTuple{N,Int}) where {T,N,F} = CountableArray{T,N,F}(counter(x),n)

Base.getindex(x::CountableVector,i::Int) = counter(x)(i)
Base.getindex(x::CountableVector,i::Int,j::Int) = isone(j) ? counter(x)(i) : counter(x)(i,j)
Base.getindex(x::CountableArray,i::Vararg{Int}) = counter(x)(i...)

Semimagma(v::CountableVector,f=*,g=groupinverse(f)) = Semimagma(collect(v),f,g)

Base.map(f,x::CountableArray{T,N,identity} where {T,N}) = CountableArray(f,size(x))
Base.map(f,x::CountableArray) = CountableArray(f∘counter(x),size(x))
mapmap(f,x::CountableArray{<:CountableArray}) = CountableArray((u...)->map(f,counter(x)(u...)),size(x))

for fun ∈ (:*,:+,:/,:-,:^)
    @eval begin
        Base.$fun(a::Number,x::CountableArray{T,N} where T) where N = CountableArray((n::Vararg{Int,N})->$fun(a,counter(x)(n...)),size(x))
        Base.$fun(x::CountableArray{T,N} where T,b::Number) where N = CountableArray((n::Vararg{Int,N})->$fun(counter(x)(n...),b),size(x))
        Base.$fun(a::CountableArray{T,N} where T,b::CountableArray{S,N} where S) where N = CountableArray((n::Vararg{Int,N})->$fun(counter(a)(n...),counter(b)(n...)),min.(size(a),size(b)))
    end
end
for fun ∈ (:inv,:-,:abs,:!,:~,:real,:imag,:conj,:floor,:ceil,:round,:exp,:exp2,:exp10,:log,:log2,:log10,:sinh,:cosh,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:acsch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:abs2,:angle)
    @eval Base.$fun(x::CountableArray) = map($fun,x)
end

LinearAlgebra.dot(a::CountableVector,b::CountableVector,Σ=sum) = Σ(a*b)
LinearAlgebra.dot(a::OnesVector,b::CountableVector,Σ=sum) = Σ(b)
LinearAlgebra.dot(a::CountableVector,b::OnesVector,Σ=sum) = Σ(a)
LinearAlgebra.dot(a::OnesVector,b::OnesVector,Σ=sum) = Σ(a)

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
    n::NTuple{N,Int}
end

const FunctionVector{T,F} = FunctionArray{T,1,F}
const FunctionMatrix{T,F} = FunctionArray{T,2,F}

FunctionArray{T,N}(f::F,n::NTuple{N}) where {T,N,F} = FunctionArray{T,N,F}(f,n)
FunctionArray{T,N}(f::F,n::Vararg{Int,N}) where {T,N,F} = FunctionArray{T,N,F}(f,n)
FunctionVector{T}(f::F,n::NTuple) where {T,F} = FunctionVector{T,F}(f,n)
FunctionVector{T}(f::F,n::Int=100) where {T,F} = FunctionVector{T,F}(f,(n,))
FunctionMatrix{T}(f::F,n::NTuple) where {T,F} = FunctionMatrix{T,F}(f,n)
FunctionMatrix{T}(f::F,n::Int=100,m::Int=100) where {T,F} = FunctionMatrix{T,F}(f,(n,))
FunctionVector(f,n=100) = FunctionVector{typeof(Base.Fix2(f,1))}(f,n)
FunctionMatrix(f,n::Int=100,m::Int=100) = FunctionMatrix{typeof(Base.Fix2(Base.Fix{3}(f,1),1))}(f,(n,m))
FunctionMatrix(f,n::Tuple{Int,Int}) = FunctionMatrix{typeof(Base.Fix2(Base.Fix{3}(f,1),1))}(f,n)
FunctionMatrix(n::Int=100,m::Int=100) = FunctionArray(^,n,m)
FunctionMatrix(n::Tuple{Int,Int}) = FunctionArray(^,n)
#FunctionArray(f,n::Vararg{Int,N}) where N = FunctionArray{typeof(f(one.(n)...)),N}(f,n)
#FunctionArray(f,n::NTuple{N,Int}) where N = FunctionArray{typeof(f(one.(n)...)),N}(f,n)
FunctionArray(n::Vararg{Int,N}) where N = FunctionArray(^,n)
FunctionArray(n::NTuple{N,Int}) where N = FunctionArray(^,n)

(x::FunctionArray{T,N})(u,n::Vararg{Int,N}) where {T,N} = CountableArray(k->counter(x)(u,k),n)
(x::FunctionArray{T,N})(u,n::NTuple{N,Int}=size(x)) where {T,N} = CountableArray(k->counter(x)(u,k),n)

Base.getindex(x::FunctionVector,i::Int) = Base.Fix2(counter(x),i)
Base.getindex(x::FunctionVector,i::Int,j::Int) = isone(j) ? Base.Fix2(counter(x),i) : Base.Fix2(Base.Fix{3}(counter(x),j),i)
#Base.getindex(x::CountableArray,i::Vararg{Int}) = counter(x)(i...)
Base.map(f,x::FunctionArray) = FunctionArray(f∘counter(x),size(x))

for fun ∈ (:*,:+,:/,:-,:^)
    @eval begin
        Base.$fun(a::Number,x::FunctionArray{T,N} where T) where N = FunctionArray((x,n::Vararg{Int,N})->$fun(a,counter(x)(x,n...)),size(x))
        Base.$fun(x::FunctionArray{T,N} where T,b::Number) where N = FunctionArray((x,n::Vararg{Int,N})->$fun(counter(x)(x,n...),b),size(x))
        Base.$fun(a::FunctionArray{T,N} where T,b::FunctionArray{S,N} where S) where N = FunctionArray((x,n::Vararg{Int,N})->$fun(counter(a)(x,n...),counter(b)(x,n...)),min.(size(a),size(b)))
    end
end
for fun ∈ (:inv,:-,:abs,:!,:~,:real,:imag,:conj,:floor,:ceil,:round,:exp,:exp2,:exp10,:log,:log2,:log10,:sinh,:cosh,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:acsch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:abs2,:angle)
    @eval Base.$fun(x::FunctionArray) = map($fun,x)
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

Series(f::FunctionArray) = Series(CountableArray(one,size(f)),f)
Base.sum(x::FunctionArray) = Series(x)

(f::Series{1})(x,Σ=sum) = LinearAlgebra.dot(f.v,f.f(x),Σ)

struct Product{N,F<:FunctionArray{S,N} where S}
    v::F
end

Base.prod(x::FunctionArray) = Product(x)

(f::Product{1})(x,Π=prod) = Π(f.v(x))

using ElasticArrays
import ElasticArrays: resize_lastdim!
export resize_lastdim!, extract, assign!

struct CountableCache{T,N,V<:AbstractArray{T,N},F} <: AbstractCountable{T,N,F}
    v::V
    f::F
end

#CountableCache{T}(v::V,f::F) where {T,N,V<:AbstractArray{T,N}} = CountableCache{T,N,V,F}(v,f)
#CountableCache(v::AbstractArray{T},f) where T = CountableCache{T}(v,f)
(c::CountableCache{T,N} where T)(n::Vararg{Int,N}) where N = c[n...]

counter(x::CountableCache) = x.f
Base.size(c::CountableCache) = size(c.v)
function Base.resize!(c::CountableCache{T,1} where T,n::Int)
    m = length(c)
    resize!(c.v,n)
    F = counter(c)
    n > m && for k ∈ m+1:n
        assign!(c.v,k,F(c.v,k))
    end
    return c
end
function ElasticArrays.resize_lastdim!(c::CountableCache{T,N} where T,n::Int) where N
    m = size(c)[end]
    resize_lastdim!(c.v,n)
    F = counter(c)
    n > m && for k ∈ m+1:n
        assign!(c.v,k,F(c.v,k))
    end
    return c
end

Base.getindex(c::CountableCache{T,1} where T,n::Int) = extract(c,n)
function Base.getindex(c::CountableCache{T,N} where T,n::Vararg{Int,N}) where N
    n[end] > size(c)[end] && resize_lastdim!(c,n[end])
    c.v[n...]
end
function extract(c::CountableCache{T,1} where T,n::Int)
    n > length(c) && resize!(c,n)
    return extract(c.v,n)
end
function extract(c::CountableCache,n::Int)
    n > size(c)[end] && resize_lastdim!(c,n)
    return extract(c.v,n)
end

Base.cumsum(x::OnesArray) = Naturals(length(x))
function Base.cumsum(x::CountableVector)
    CountableCache(cumsum(view(x,:)),(u,k) -> u[k-1] + counter(x)(k))
end
function Base.cumprod(x::CountableVector)
    CountableCache(cumprod(view(x,:)),(u,k) -> u[k-1] * counter(x)(k))
end
function Base.cumsum(x::CountableCache{T,1} where T)
    CountableCache(cumsum(x.v),(u,k) -> u[k-1] + counter(x)(x,k))
end
function Base.cumprod(x::CountableCache{T,1} where T)
    CountableCache(cumprod(x.v),(u,k) -> u[k-1] * counter(x)(x,k))
end

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

export distance, distance2, distancemax

distance2(a,b) = Inf
distance2(a::Pair{Int},b::Pair{Int}) = distance2(last(a),last(b))
distance2(a::Pair{<:Pair},b::Pair{<:Pair}) = distance2(last(a),last(b))
distance2(a::AbstractArray,b::AbstractArray) = norm(a-b)
distance2(a::Number,b::Number,ϵ=5eps()) = norm(a-b)

distance(a,b) = Inf
distance(a::Pair{Int},b::Pair{Int}) = distance(last(a),last(b))
distance(a::Pair{<:Pair},b::Pair{<:Pair}) = distance(last(a),last(b))
distance(a::AbstractArray,b::AbstractArray) = norm(a-b)
distance(a::Number,b::Number,ϵ=5eps()) = abs(a-b)

distancemax(a,b) = Inf

export FixedPoint, fixedpoint, fixedpointhold, fixedpointerror, residual, residuals

struct FixedPoint{T,F,D}
    v0::T
    v::T
    n::Int
    r::Float64
    f::F
    FixedPoint(v0::T,v::T,n::Int,f::F,D=distance2) where {T,F} = new{T,F,D}(v0,v,n,Inf,f)
    FixedPoint(v0::T,v::T,n::Int,r::Real,f::F,D=distance2) where {T,F} = new{T,F,D}(v0,v,n,r,f)
end

counter(x::FixedPoint) = x.f
initial(x::FixedPoint) = x.v0
final(x::FixedPoint) = x.v
Base.first(x::FixedPoint) = initial(x)
Base.first(x::FixedPoint{<:Pair{<:Union{Int,Pair}}}) = last(initial(x))
Base.last(x::FixedPoint) = final(x)
Base.last(x::FixedPoint{<:Pair{<:Union{Int,Pair}}}) = last(final(x))
Base.lastindex(x::FixedPoint) = length(x)
Base.length(x::FixedPoint) = x.n
residual(x::FixedPoint) = x.r
(x::FixedPoint{T,F,D} where {T,F})(u) where D = FixedPoint(u,length(x)-1,counter(x),D)
(::FixedPoint{T,F,D} where {T,F})(x,y) where D = D(x,y)

function Base.show(io::IO,x::FixedPoint)
    if !get(io,:compact,false)
        println(io,"(::FixedPoint)[$(length(x))] with residual: $(residual(x))")
    end
    show(io,last(x))
end

function Base.sum(x::CountableVector)
    countsum(u) = (first(u)+1) => (last(u)+counter(x)(first(u)+1))
    FixedPoint(1=>x[1],length(x)=>sum(view(x,:)),length(x),x[end],countsum)
end
function Base.prod(x::CountableVector)
    countprod(u) = (first(u)+1) => (last(u)*counter(x)(first(u)+1))
    val = prod(view(x,:))
    FixedPoint(1=>x[1],length(x)=>val,length(x),distance2(val/x[end],val),countprod)
end

function Base.sum(x::FixedPoint{<:Pair{<:Union{Int,Pair}}})
    function countsum(u)
        fp1 = counter(x)(first(u))
        fp1 => (last(u) + last(fp1))
    end
    FixedPoint(x.v0=>last(x.v0),length(x)-1,countsum)
end
function Base.sum(x::FixedPoint)
    function countsum(u)
        fp1 = counter(x)(last(first(u)))
        ((first(first(u))+1)=>fp1) => (last(u) + fp1)
    end
    FixedPoint((1=>first(x))=>first(x),length(x)-1,countsum)
end
function Base.prod(x::FixedPoint{<:Pair{<:Union{Int,Pair}}})
    function countprod(u)
        fp1 = counter(x)(first(u))
        fp1 => (last(u) + last(fp1))
    end
    FixedPoint(x.v0=>last(x.v0),length(x)-1,countprod)
end
function Base.prod(x::FixedPoint)
    function countprod(u)
        fp1 = counter(x)(last(first(u)))
        ((first(first(u))+1)=>fp1) => (last(u) * fp1)
    end
    FixedPoint((1=>first(x))=>first(x),length(x)-1,countprod)
end

function Base.cumsum(x::FixedPoint{<:Pair{Int},F,D} where F) where D
    CountableCache([first(x)],(u,k) -> u[k-1] + last(FixedPoint(x.v0,k-1,counter(x),D)))
end
function Base.cumsum(x::FixedPoint{T,F,D} where {T,F}) where D
    CountableCache([x],(u,k) -> u[k-1] + last(FixedPoint(x.v0,k-1,counter(x),D)))
end
function Base.cumprod(x::FixedPoint{<:Pair{Int},F,D} where F) where D
    CountableCache([first(x)],(u,k) -> u[k-1] * last(FixedPoint(x.v0,k-1,counter(x),D)))
end
function Base.cumprod(x::FixedPoint{T,F,D} where {T,F}) where D
    CountableCache([x],(u,k) -> u[k-1] * last(FixedPoint(x.v0,k-1,counter(x),D)))
end

#=function Base.sum(x::CountableCache{T,1} where T)
    countsum(u) = (first(u)+1) => (last(u)+counter(x)(Values((last(u),)),2))
    FixedPoint(1=>x[1],length(x)=>sum(view(x,:)),length(x),x[end],countsum)
end
function Base.prod(x::CountableCache{T,1} where T)
    countprod(u) = (first(u)+1) => (last(u)*counter(x)(first(u)+1))
    val = prod(view(x,:))
    FixedPoint(1=>x[1],length(x)=>val,length(x),distance2(val/x[end],val),countprod)
end=#

function FixedPoint(v0,n::Int,F,D=distance2)
    x0 = v0
    xn = x0
    for k ∈ 2:n+1
        x0 = xn
        xn = F(xn)
    end
    return FixedPoint(v0,xn,n+1,D(x0,xn),F,D)
end

function Base.getindex(x::FixedPoint{T,F,D} where {T,F},i::Int) where D
    n = length(x)
    i == n && (return x)
    xn = i < n ? initial(x) : final(x)
    x0 = xn
    F = counter(x)
    for k ∈ (i<n ? (2:i) : (n+1:i))
        x0 = xn
        xn = F(xn)
    end
    return FixedPoint(initial(x),xn,i,D(x0,xn),F,D)
end

function Base.collect(x::FixedPoint)
    resize!(CountableCache([first(x)],(u,k) -> counter(x)(extract(u,k-1))),length(x))
end
function Base.collect(x::FixedPoint{<:Pair{Int}})
    resize!(CountableCache([first(x)],(u,k) -> last(counter(x)(k-1=>extract(u,k-1)))),length(x))
end
function Base.collect(x::FixedPoint{<:AbstractArray})
    val = ElasticArray(reshape(first(x),size(first(x))...,1))
    out = CountableCache(val,(u,k) -> counter(x)(extract(u,k-1)))
    return resize_lastdim!(out,length(x))
end

fixedpointerror(f,x,ϵ=5eps()) = fixedpoint(f,x,ϵ,Val(true))
fixedpoint(f,x,n::Int,v::Val=Val(false)) = fixedpoint(f,x,1:n,v)
function fixedpoint(f,x,n::AbstractVector{Int},::Val{print}=Val(false),d=distance2) where print
    out = print ? zeros(length(n)) : nothing
    x0 = x
    xn = x
    for i ∈ n
        x0 = xn
        xn = f(xn)
        print && (out[i] = d(x0,xn))
    end
    fp = FixedPoint(x,xn,length(n)+1,d(x0,xn),f,d)
    return print ? (fp,out) : fp
end
function fixedpoint(f,x,ϵ::AbstractFloat=5eps(),::Val{print}=Val(false),d=distance2) where print
    change = 5ϵ
    print && (out = Float64[])
    x0 = x
    xn = x
    n = 1
    while change > ϵ
        n += 1
        x0 = xn
        xn = f(xn)
        print && push!(out,d(x0,xn))
    end
    fp = FixedPoint(x,xn,n,d(x0,xn),f,d)
    return print ? (fp,out) : fp
end

function fixedpointhold(f,x,n::AbstractVector{Int},::Val{print}=Val(false),d=distance2) where print
    print && (out = zeros(length(n)))
    x0 = x
    xn = x
    for i ∈ n
        x0 = xn
        xn = f(x,xn) # hold x constant, iterate xn
        print && (out[i] = d(x0,xn))
    end
    return FixedPoint(x,xn,length(n)+1,d(x0,xn),f,d)
end

residuals(x::FixedPoint,d=distance2) = residuals(collect(x),d)
function residuals(x::AbstractArray,d=distance2)
    x0 = extract(x,1)
    xi = x0
    out = Vector{Float64}(undef,size(x)[end])
    for i ∈ 1:size(x)[end]
        x0 = xi
        xi = extract(x,i)
        out[i] = d(x0,xi)
    end
    return out
end
function residuals(x::CountableVector,d=distance2)
    CountableVector(k -> d(counter(x)(isone(k) ? 1 : k-1),counter(x)(k)),length(x))
end
function residuals(x::CountableCache,d=distance2)
    CountableCache(residuals(x.v),(u,k) -> d(extract(u,k-1),extract(u,k)))
end

export FixedCycle

struct FixedCycle{F,D}
    n::Int
    f::F
    FixedCycle(n::Int,f::F,d=distance2) where F = new{F,d}(n,f)
    FixedCycle(f::F,d=distance2) where F = new{F,d}(100,f)
end

counter(x::FixedCycle) = x.f
Base.length(x::FixedCycle) = x.n

(x::FixedCycle{F,D} where F)(u,n=length(x)) where D = FixedPoint(u,n,counter(x),D)

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

prime(u,i) = prime(i)

export Naturals, Integers, CantorPairs, ElegantPairs0, ElegantPairs1, ElegantPairs
export PositiveRationals, Rationals, NonzeroRationals, PrimeIntegers, PrimeCache
export GaussianNaturals, GaussianIntegers, GaussianRationals

const Naturals = CountableVector(identity)
const Integers = CountableVector(integer)
const PrimeIntegers = CountableVector{Int}(prime)
const PrimeCache = CountableCache([2],prime)
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


