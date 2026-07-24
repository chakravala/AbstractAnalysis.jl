
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

export @metric, @norm, supnorm, infnorm, maxabs, minabs, residual, residuals, lipschitz

#=abstract type AbstractMetric{D} end
abstract type NormedSpace{D,N} <: AbstractMetric{D} end

struct MetricSpace{D} <: AbstractMetric{D} end
struct BanachSpace{D,N} <: NormedSpace{D,N} end
struct HilbertSpace{D,N,P} <: NormedSpace{D,N} end=#

macro metric(f)
    d = esc(f)
    quote
        $d(a::Number,b::Number) = $d(a-b)
        $d(a::AbstractArray,b::AbstractArray) = $d(a-b)
        $d(a::Pair{Int},b::Pair{Int}) = $d(last(a),last(b))
        $d(a::Pair{<:Pair},b::Pair{<:Pair}) = $d(last(a),last(b))
        $d(a,b) = Inf
        $d
    end
end

if  VERSION < v"1.12"
    macro norm(f)
        n = esc(f)
        quote
            $n(x::Pair{Int}) = $n(last(x))
            $n(x::Pair{<:Pair}) = $n(last(x))
            $n(::Fix1) = Inf
            $n(::Fix2) = Inf
            @metric $f
        end
    end
else
    macro norm(f)
        n = esc(f)
        quote
            $n(x::Pair{Int}) = $n(last(x))
            $n(x::Pair{<:Pair}) = $n(last(x))
            $n(::Fix) = Inf
            @metric $f
        end
    end
end

supnorm(x) = Inf
supnorm(x::AbstractArray) = norm(x)
supnorm(x::Number) = norm(x)
@norm supnorm

infnorm(x) = 0.0
infnorm(x::AbstractArray) = norm(x)
infnorm(x::Number) = norm(x)
@norm infnorm

maxabs(x) = maximum(norm,x)
@norm maxabs

minabs(x) = minimum(norm,x)
@norm minabs

export Limit, orbit, orbithold, orbiterror, limit

struct Limit{T,F,D}
    v0::T
    v::T
    n::Int
    r::Float64
    f::F
    Limit(v0::T,v::T,n::Int,f::F,D=supnorm) where {T,F} = new{T,F,D}(v0,v,n,Inf,f)
    Limit(v0::T,v::T,n::Int,r::Real,f::F,D=supnorm) where {T,F} = new{T,F,D}(v0,v,n,r,f)
end

counter(x::Limit) = x.f
initial(x::Limit) = x.v0
final(x::Limit) = x.v
Base.first(x::Limit) = initial(x)
Base.first(x::Limit{<:Pair{<:Union{Int,Pair}}}) = last(initial(x))
Base.last(x::Limit) = final(x)
Base.last(x::Limit{<:Pair{<:Union{Int,Pair}}}) = last(final(x))
Base.lastindex(x::Limit) = length(x)
Base.length(x::Limit) = x.n
residual(x::Limit) = x.r
(x::Limit{T,F,D} where {T,F})(u) where D = Limit(u,length(x)-1,counter(x),D)
(::Limit{T,F,D} where {T,F})(x,y) where D = D(x,y)
(x::Limit{<:Pair{Int,<:Union{Series,Product}}})(u) = last(x)(u)

countfix(x,u) = Fix1(Fix1(countfix,x),u)
countfix(x,u,z) = (first(z)+1) => last(x).f(u,first(z)+1)

if VERSION < v"1.12"
    function (x::Limit{<:Pair{Int,<:Union{Fix1,Fix2}},F,D} where F)(u) where D
        Limit(1=>first(x)(u),length(x)=>last(x)(u),length(x),residual(x),countfix(x,u),D)
    end
else
    function (x::Limit{<:Pair{Int,<:Fix},F,D} where F)(u) where D
        Limit(1=>first(x)(u),length(x)=>last(x)(u),length(x),residual(x),countfix(x,u),D)
    end
end

Base.show(io::IO, ::MIME"text/plain", x::Limit) = show(io, x)
function Base.show(io::IO, x::Limit)
    if typeof(last(x)) <: Number || get(io,:compact,false)
        println(io,"$(last(x)) (n → $(length(x)), Δ → $(residual(x)))")
    else
        println(io,"Limit as n → $(length(x)), Δ → $(residual(x))")
        show(io,last(x))
    end
end

fixmap(x,f) = Fix1(Fix1(fixmap,x),f)
function fixmap(x,f,u)
    xn = counter(x)(first(u))
    xn => f(last(xn))
end

function Base.map(f,x::Limit{T,F,D} where {T,F}) where D
    v0,vn = (1=>first(x))=>f(first(x)),(length(x)=>last(x))=>f(last(x))
    Limit(v0,vn,length(x),Inf,fixmap(x,f),D)
end
function Base.map(f,x::Limit{<:Pair{Int},F,D} where F) where D
    v0,vn = initial(x)=>f(first(x)),final(x)=>f(last(x))
    Limit(v0,vn,length(x),Inf,fixmap(x,f),D)
end

binopleft(x,f,a) = Fix1(Fix1(Fix1(binopleft,x),f),a)
function binopleft(x::Limit,f,a,u)
    y = last(first(u))
    p = counter(x)(y)
    ((first(first(u))+1)=>p) => f(a,last(p))
end
binopright(x,f,b) = Fix1(Fix1(Fix1(binopright,x),f),b)
function binopright(x::Limit,f,b,u)
    y = last(first(u))
    p = counter(x)(y)
    ((first(first(u))+1)=>p) => f(last(p),b)
end
function binop(a::Limit,b::Limit,f,u)
    x,y = last(first(u))
    p,q = counter(a)(x),counter(b)(y)
    ((first(first(u))+1)=>(p,q)) => f(last(p),last(q))
end

for fun ∈ (:*,:+,:/,:-,:^)
    @eval begin
        function Base.$fun(a::Number,x::Limit{T,F,D} where {T,F}) where D
            v0 = (1 => initial(x)) => $fun(a,first(x))
            vn = (length(x) => final(x)) => $fun(a,last(x))
            vn1 = binopleft(x,$fun,a,vn)
            Limit(v0,vn1,length(x)+1,D(last(vn1),last(vn)),binopleft(x,$fun,a),D)
        end
        function Base.$fun(x::Limit{T,F,D} where {T,F},b::Number) where D
            v0 = (1 => initial(x)) => $fun(first(x),b)
            vn = (length(x) => final(x)) => $fun(last(x),b)
            vn1 = binopright(x,$fun,b,vn)
            Limit(v0,vn1,length(x)+1,D(last(vn1),last(vn)),binopright(x,$fun,b),D)
        end
        function Base.$fun(a::Limit{T,F,D} where {T,F},b::Limit{S,G,D} where {S,G}) where D
            v0 = (1 => (final(a),final(b))) => $fun(last(a),last(b))
            vn = binop(a,b,$fun,v0)
            Limit(v0,vn,2,D(last(vn),last(v0)),binop(a,b,$fun),D)
        end
    end
end
for fun ∈ unaryops
    @eval Base.$fun(x::Limit) = map($fun,x)
end

countsum(x) = Fix1(countsum,x)
countsum(x::CountableVector,u) = (first(u)+1) => (last(u)+counter(x)(first(u)+1))
function countsum(x::Limit{<:Pair{<:Union{Int,Pair}}},u)
    fp1 = counter(x)(first(u))
    fp1 => (last(u) + last(fp1))
end
function countsum(x::Limit,u)
    fp1 = counter(x)(last(first(u)))
    ((first(first(u))+1)=>fp1) => (last(u) + fp1)
end

countprod(x) = Fix1(countprod,x)
countprod(x::CountableVector,u) = (first(u)+1) => (last(u)*counter(x)(first(u)+1))
function countprod(x::Limit{<:Pair{<:Union{Int,Pair}}},u)
    fp1 = counter(x)(first(u))
    fp1 => (last(u) * last(fp1))
end
function countprod(x::Limit,u)
    fp1 = counter(x)(last(first(u)))
    ((first(first(u))+1)=>fp1) => (last(u) * fp1)
end

function Base.sum(x::CountableVector)
    Limit(1=>x[1],length(x)=>sum(view(x,:)),length(x),x[end],countsum(x))
end
function Base.prod(x::CountableVector)
    val = prod(view(x,:))
    Limit(1=>x[1],length(x)=>val,length(x),supnorm(val,val/x[end]),countprod(x))
end
function Base.sum(x::Limit{<:Pair{<:Union{Int,Pair}}})
    Limit(x.v0=>last(x.v0),length(x)-1,countsum(x))
end
function Base.sum(x::Limit)
    Limit((1=>first(x))=>first(x),length(x)-1,countsum(x))
end
function Base.prod(x::Limit{<:Pair{<:Union{Int,Pair}}})
    Limit(x.v0=>last(x.v0),length(x)-1,countprod(x))
end
function Base.prod(x::Limit)
    Limit((1=>first(x))=>first(x),length(x)-1,countprod(x))
end

countcumsum(x,u,k) = u[k-1] + last(Limit(x.v0,k-1,counter(x),D))
countcumprod(x,u,k) = u[k-1] * last(Limit(x.v0,k-1,counter(x),D))

function Base.cumsum(x::Limit{<:Pair{Int},F,D} where F) where D
    SequenceArray([first(x)],countcumsum(x))
end
function Base.cumsum(x::Limit{T,F,D} where {T,F}) where D
    SequenceArray([x],countcumsum(x))
end
function Base.cumprod(x::Limit{<:Pair{Int},F,D} where F) where D
    SequenceArray([first(x)],countcumprod(x))
end
function Base.cumprod(x::Limit{T,F,D} where {T,F}) where D
    SequenceArray([x],countcumprod(x))
end

#=function Base.sum(x::SequenceArray{T,1} where T)
    countsum(u) = (first(u)+1) => (last(u)+counter(x)(Values((last(u),)),2))
    Limit(1=>x[1],length(x)=>sum(view(x,:)),length(x),x[end],countsum)
end
function Base.prod(x::SequenceArray{T,1} where T)
    countprod(u) = (first(u)+1) => (last(u)*counter(x)(first(u)+1))
    val = prod(view(x,:))
    Limit(1=>x[1],length(x)=>val,length(x),supnorm(val/x[end],val),countprod)
end=#

function Limit(v0,n::Int,F,D=supnorm)
    x0 = v0
    xn = x0
    for k ∈ 2:n+1
        x0 = xn
        xn = F(xn)
    end
    return Limit(v0,xn,n+1,D(xn,x0),F,D)
end

Base.getindex(x::Limit,ϵ::AbstractFloat) = limit(x,ϵ)
function Base.getindex(x::Limit{T,F,D} where {T,F},i::Int) where D
    n = length(x)
    i == n && (return x)
    xn = i < n ? initial(x) : final(x)
    x0 = xn
    F = counter(x)
    for k ∈ (i<n ? (2:i) : (n+1:i))
        x0 = xn
        xn = F(xn)
    end
    return Limit(initial(x),xn,i,D(xn,x0),F,D)
end

extract1(u,k) = extract(u,k-1)
counterextract(x) = counter(x)∘extract1
counterextract(x::Limit{<:Pair{Int}}) = Fix1(counterextract,x)
counterextract(x::Limit{<:Pair{Int}},u,k) = last(counter(x)(k-1=>extract(u,k-1)))

function Base.collect(x::Limit)
    resize!(SequenceArray([first(x)],counterextract(x)),length(x))
end
function Base.collect(x::Limit{<:Pair{<:Pair,T}}) where T
    xn = x[1]
    out = Vector{T}(undef,length(x))
    out[1] = last(xn)
    for k ∈ 2:length(x)
        xn = xn[k]
        out[k] = last(xn)
    end
    return out
end
function Base.collect(x::Limit{<:AbstractArray})
    val = ElasticArray(reshape(first(x),size(first(x))...,1))
    out = SequenceArray(val,counterextract(x))
    return resize_lastdim!(out,length(x))
end

orbiterror(f,x,ϵ=5eps()) = orbit(f,x,ϵ,Val(true))
orbit(f,x,n,d) = orbit(f,x,n,Val(false),d)
orbit(f,x,n::Int,v::Val=Val(false),d=supnorm) = orbit(f,x,1:n,v,d)
function orbit(f,x,n::AbstractVector{Int},::Val{print}=Val(false),d=supnorm) where print
    out = print ? zeros(length(n)) : nothing
    x0 = x
    xn = x
    for i ∈ n
        x0 = xn
        xn = f(xn)
        print && (out[i] = d(xn,x0))
    end
    fp = Limit(x,xn,length(n)+1,d(xn,x0),f,d)
    return print ? (fp,out) : fp
end
function orbit(f,x,ϵ::AbstractFloat=5eps(),::Val{print}=Val(false),d=supnorm) where print
    change = 5ϵ
    print && (out = Float64[])
    x0 = x
    xn = x
    n = 1
    while change > ϵ
        n += 1
        x0 = xn
        xn = f(xn)
        change = d(xn,x0)
        print && push!(out,change)
    end
    fp = Limit(x,xn,n,change,f,d)
    return print ? (fp,out) : fp
end

function orbithold(f,x,n::AbstractVector{Int},::Val{print}=Val(false),d=supnorm) where print
    print && (out = zeros(length(n)))
    x0 = x
    xn = x
    for i ∈ n
        x0 = xn
        xn = f(x,xn) # hold x constant, iterate xn
        print && (out[i] = d(xn,x0))
    end
    return Limit(x,xn,length(n)+1,d(xn,x0),f,d)
end

countresiduals(x) = Fix1(countresiduals,x)
countresiduals(x::CountableVector,k) = d(counter(x)(k+1),counter(x)(k))
countresiduals(x::AbstractArray,k) = d(extract(x,k+1),extract(x,k))
countresiduals(x::AbstractArray,u,k) = d(extract(x,k+1),extract(x,k))

lipschitz(x,d=supnorm,r=/) = residuals(residuals(x,d),r)
residuals(x::Limit,d=supnorm) = residuals(collect(x),d)
residuals(x::Ones) = Zeros(length(x)-1)
residuals(x::Zeros) = Zeros(length(x)-1)
residuals(x::typeof(Naturals)) = Ones(length(x)-1)
residuals(x::typeof(Integers)) = Naturals(length(x)-1)
function residuals(x::AbstractArray,d=supnorm)
    x0 = extract(x,1)
    xi = x0
    out = Vector{Float64}(undef,size(x)[end]-1)
    for i ∈ 2:size(x)[end]
        x0 = xi
        xi = extract(x,i)
        out[i-1] = d(xi,x0)
    end
    return out
end
function residuals(x::CountableVector,d=supnorm)
    CountableVector(countresiduals(x),length(x)-1)
end
function residuals(x::SequenceArray,d=supnorm)
    SequenceArray(residuals(x.v,d),countresiduals(x))
end

residualproduct(x::CountableVector,d::Function=supnorm) = countableproduct(x,x,d)
residualproduct(x::AbstractVector,d::Function=supnorm) = [d(a,b) for a ∈ x, b ∈ x]

export FixedCycle

struct FixedCycle{F,D}
    n::Int
    f::F
    FixedCycle(n::Int,f::F,d=supnorm) where F = new{F,d}(n,f)
    FixedCycle(f::F,d=supnorm) where F = new{F,d}(100,f)
end

counter(x::FixedCycle) = x.f
Base.length(x::FixedCycle) = x.n
Base.getindex(x::FixedCycle{F,D} where F,i::Int) where D = FixedCycle(i,x.f,D)

(x::FixedCycle{F,D} where F)(u,n=length(x)) where D = Limit(u,n,counter(x),D)

export isconverging, isdiverging, ismonotonic, isdecreasing, isincreasing, iscauchy

function isbounded(x::AbstractArray)
    for i ∈ x
        !isinf(i) && (return false)
    end
    return true
end

isconverging(x::AbstractVector) = !isdiverging(x)
function isdiverging(x::AbstractVector,d=supnorm)
    r0 = d(x[end],x[end-1])
    ri = r0
    for i ∈ length(x)-1:-1:2
        r0 = ri
        ri = d(x[i],x[i-1])
        r0 < ri && (return false)
    end
    return true
end

function iscauchy(x::AbstractVector,d=supnorm)
    ϵ0 = d(x[end-1],x[end])
    N = length(x)
    for n ∈ N-2:-1:1
        ϵmax = 0.0
        for i ∈ n+1:N
            rij = d(x[n],x[i])
            ϵmax = max(ϵmax,rij)
        end
        ϵ0 > ϵmax && (return false)
        ϵ0 = ϵmax
    end
    return true
end

ismonotonic(x) = isincreasing(x) || isdecreasing(x)
function isincreasing(x::AbstractVector)
    for i ∈ 2:length(x)
        x[i-1] > x[i] && (return false)
    end
    return true
end
function isdecreasing(x::AbstractVector)
    for i ∈ 2:length(x)
        x[i-1] < x[i] && (return false)
    end
    return true
end

limit(x::Limit) = x
limit(x::FixedCycle) = x
limit(x::AbstractVector) = last(x)

limit(x::Limit,n::Int) = x[n]
limit(x::FixedCycle,n::Int) = x[n]
limit(x::AbstractVector,n::Int) = x[n]
limit(x::AbstractArray,n::Int=size(x)[end]) = extract(x,n)

countlimit(x) = Fix1(countlimit,x)
countlimit(x::CountableVector,u) = (first(u)+1) => counter(x)(first(u)+1)
countlimit(x::AbstractCountable,u) = (first(u)+1) => extract(x,first(u)+1)
countlimit(x::Union{Series,Product},u) = (first(u)+1) => resize!(last(u),first(u)+1)

function limit(x::CountableFunction,n::Int=length(x),d=supnorm)
    Limit(1=>x[1],n=>x[n],n,d(x[n],x[n-1]),countlimit(x),d)
end
function limit(x::SequenceArray,n::Int=length(x),d=supnorm)
    n > length(x) && (return limit(x,length(x),d)[n])
    v0,vn = extract(x,1),extract(x,n)
    Limit(1=>v0,n=>vn,n,d(vn,extract(x,n-1)),countlimit(x),d)
end
function limit(x::Union{Series,Product},n::Int=length(x),d=supnorm)
    n == length(x) ? x : resize!(x,n)
end

limit(x::CountableVector,ϵ::AbstractFloat,p::Val=Val(false)) = limit(limit(x,2),ϵ,p)
limit(x::SequenceArray,ϵ::AbstractFloat,p::Val=Val(false)) = limit(limit(x,2),ϵ,p)
function limit(L::Limit{T,F,D} where F,ϵ::AbstractFloat,::Val{print}=Val(false)) where {T,D,print}
    x = final(L)
    change = 5ϵ
    print && (out = Float64[])
    x0 = x
    xn = x
    n = 1
    f = counter(L)
    while change > ϵ
        n += 1
        x0 = xn
        xn = f(xn)
        change = T<:Pair{<:Union{Int,Pair}} ? D(last(xn),last(x0)) : D(xn,x0)
        print && push!(out,change)
    end
    fp = Limit(x,xn,n+length(L),change,f,D)
    return print ? (fp,out) : fp
end

export supseq, infseq, limsup, liminf, residualproduct, iscauchy

function supremum(x,u,k)
    xk = extract(x,k)
    for i ∈ 1:k-1
        assign!(u,i,max(extract(u,i),xk))
    end
    return xk
end
function infimum(x,u,k)
    xk = extract(x,k)
    for i ∈ 1:k-1
        assign!(u,i,min(extract(u,i),xk))
    end
    return xk
end

function supseq(x::T) where T<:AbstractVector
    out = resize!(SequenceArray([extract(x,1)],Fix1(supremum,x)),length(x))
    T <: AbstractCountable ? out : out.v
end
function infseq(x::T) where T<:AbstractVector
    out = resize!(SequenceArray([extract(x,1)],Fix1(infimum,x)),length(x))
    T <: AbstractCountable ? out : out.v
end

function countsup(x::T,n,k) where T
    xk = T<:CountableVector ? counter(x)(k+n) : extract(x,k+n)
    iszero(n) && (return xk)
    if isone(n)
        xk1 = T<:CountableVector ? counter(x)(k) : extract(x,k)
        max(xk1,xk)
    else
        max(countsup(x,n-1,k),xk)
    end
end
function countinf(x::T,n,k) where T
    xk = T<:CountableVector ? counter(x)(k+n) : extract(x,k+n)
    iszero(n) && (return xk)
    if isone(n)
        xk1 = T<:CountableVector ? counter(x)(k) : extract(x,k)
        min(xk1,xk)
    else
        min(countinf(x,n-1,k),xk)
    end
end


supseq(x::T,n) where T<:AbstractVector = CountableVector(Fix1(Fix1(countsup,x),n),length(x))
infseq(x::T,n) where T<:AbstractVector = CountableVector(Fix1(Fix1(countinf,x),n),length(x))

limsup(x::AbstractCountable,args...) = limit(supseq(x,5),args...)
liminf(x::AbstractCountable,args...) = limit(infseq(x,5),args...)
limsup(x::AbstractCountable,m::Int,args...) = limit(supseq(x,m),args...)
liminf(x::AbstractCountable,m::Int,args...) = limit(infseq(x,m),args...)
limsup(x::AbstractVector,m::Int=5) = supseq(x,m)[end-m]
liminf(x::AbstractVector,m::Int=5) = infseq(x,m)[end-m]
limsup(x::AbstractVector,m::Int,n::Int) = supseq(x,m)[n]
liminf(x::AbstractVector,m::Int,n::Int) = infseq(x,m)[n]

export derivative, derivative2

#derivative(f) = Fix1(derivative,f)
#derivative(f,x,h=cbrt(eps(typeof(x)))) = (f(x+h)-f(x-h))/2h

derivative(f) = Fix1(derivative,f)
derivative(f,x,h=eps()^(1/5)) = (-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h))/12h

derivative2(f) = Fix1(derivative2,f)
derivative2(f,x,h=sqrt(sqrt(eps(typeof(x))))) = (f(x+h)-2f(x)+f(x-h))/h^2

