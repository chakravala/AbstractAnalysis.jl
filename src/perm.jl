
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

export AbstractPermutation, Permutation, Cycle, CycleProduct

abstract type AbstractPermutation{N} <: AbstractVector{Int} end

Base.one(G::Semimagma{<:AbstractPermutation{N}}) where N = Permutation{N}(I)
Base.isone(a::AbstractPermutation) = a==one(a)
Base.isodd(c::AbstractPermutation) = isodd(order(c))
Base.iseven(c::AbstractPermutation) = iseven(order(c))
Base.:*(a::AbstractPermutation,::LinearAlgebra.UniformScaling{Bool}) = a
Base.:*(::LinearAlgebra.UniformScaling{Bool},b::AbstractPermutation) = b
Base.:∘(a::AbstractPermutation{N},b::AbstractPermutation{N}) where N = a(b)
Base.:*(a::AbstractPermutation{N},b::AbstractPermutation{N}) where N = a(b)
Base.:/(a::AbstractPermutation{N},b::AbstractPermutation{N}) where N = a(inv(b))
Base.:\(a::AbstractPermutation{N},b::AbstractPermutation{N}) where N = inv(a)(b)
Base.:^(a::AbstractPermutation{N},n::Integer) where N = isone(n) ? a : n>0 ? a*a^(n-1) : n<0 ? inv(a)^-n : one(a)

defaultgroup(G::Semimagma{<:AbstractPermutation}) = SymmetricGroup(G)
SymmetricGroup(G::Semimagma{<:AbstractPermutation{N}}) where N = SymmetricGroup(N)

struct Permutation{N,T<:AbstractVector{Int}} <: AbstractPermutation{N}
    v::T
    Permutation(v::T) where T<:AbstractVector = new{length(v),T}(v)
end

Permutation(n::Int...) = Permutation(Values(n))

Base.length(p::Permutation{N}) where N = N
Base.size(p::Permutation{N}) where N = (N,)
Base.getindex(σ::Permutation,i::Int) = σ.v[i]
(σ::Permutation)(i::Int) = σ[i]
(σ::Permutation)(ρ::Permutation) = Permutation(σ.v[ρ.v])

Permutation{N}(I::LinearAlgebra.UniformScaling{Bool}) where N = Permutation(count(1,N))
Base.one(a::Permutation{N}) where N = Permutation{N}(I)
Base.inv(a::Permutation) = Permutation(sortperm(a.v))
Base.inv(a::Permutation{N,<:Values} where N) = Permutation(Values(sortperm(a.v)))
commutator(g::Permutation,h::Permutation) = commutator(group(g),group(h))

struct Cycle{N,T<:AbstractVector{Int}} <: AbstractPermutation{N}
    v::T
    Cycle{N}(v::T) where {N,T<:AbstractVector} = new{N,T}(v)
end

const Transposition{N} = Cycle{N,Values{2,Int}}

Cycle{N}(n::Int...) where N = Cycle{N}(Values(n))
Permutation(c::Cycle{N}) where N = Permutation(evalperm.(Ref(c),count(1,N)))

function evalperm(c::Cycle{N},i) where N
    j = findfirst(isequal(i),c.v)
    isnothing(j) ? i : j==length(c.v) ? c.v[1] : c.v[j+1]
end

Base.size(c::Cycle) = size(c.v)
Base.getindex(c::Cycle,i::Int) = c.v[i]

struct CycleProduct{N,T<:Tuple} <: AbstractPermutation{N}
    v::T
    CycleProduct{N}(v::T) where {N,T<:Tuple} = new{N,T}(v)
    CycleProduct(v::T) where {N,T<:NTuple{K,<:Cycle{N}} where K} = new{N,T}(v)
end

CycleProduct(v...) = CycleProduct(v)
CycleProduct{N}(v...) where N = CycleProduct{N}(v)

Permutation(c::CycleProduct{N}) where N = isempty(c.v) ? Permutation{N}(I) : *(Permutation.(c.v)...)

export decompose, order, levicivita, isdisjoint

function CycleProduct(p::Permutation{N,T}) where {N,T}
    out = T<:Values ? (Values{N,Int} where N)[] : Vector{Int}[]
    for i ∈ 1:N
        if length(out)>0 ? prod(i .∉ out) : true
            c = getcycle(p,i)
            length(c)≠1 && push!(out,c)
        end
    end
    CycleProduct{N}(Cycle{N}.(out)...)
end
function decompose(p::Permutation)
    out = CycleProduct(p)
    length(out.v) ≠ 1 ? out : out.v[1]
end
function getcycle(p::Permutation{N,T},i,out=Int[i]) where {N,T}
    pi = p[i]
    pi ∉ out ? getcycle(p,pi,push!(out,pi)) : T<:Values ? Values(out...) : out
end

decompose(G::Semimagma) = Semimagma(decompose.(G.v))

Base.size(c::CycleProduct) = (length(c.v),)
Base.getindex(c::CycleProduct,i::Int) = c.v[i]

order(c::Cycle) = length(c.v)-1
order(c::CycleProduct) = isempty(c.v) ? 0 : sum(order.(c.v))
order(p::Permutation) = order(decompose(p))

levicivita(c::Cycle) = isodd(c) ? -1 : 1
levicivita(c::CycleProduct) = prod(levicivita.(c.v))
levicivita(p::Permutation) = levicivita(decompose(p))
const ε = levicivita

isdisjoint(a::Cycle,b::Cycle) = prod(a.v .∉ Ref(b.v))
isabelian(a::Cycle,b::Cycle) = isdisjoint(a,b) || a==b
Base.:(==)(a::Cycle,b::Cycle) = length(a.v)==length(b.v) && prod(a.v .∈ Ref(b.v))

export SymmetricGroup, AlternatingGroup, DihedralGroup, unityroots

unityroots(n) = Semimagma(cis.((2π/n).*(0:n-1)))

SymmetricGroup(N::Int) = Semimagma(Permutation.(Values{N}.(collect(permutations(count(1,N))))))

AlternatingGroup(n::Int) = AlternatingGroup(SymmetricGroup(n))
AlternatingGroup(Sn::Semimagma) = Semimagma(Sn.v[findall(iseven.(Sn.v))])

DihedralGroup(r::Permutation,s::Permutation) = DihedralGroup(decompose(r),decompose(s))
function DihedralGroup(r::Cycle,s::Cycle)
    order(s)≠1 && throw(error("order($s)≠1"))
    group(Permutation(s))*group(Permutation(r))
end

#=
struct PermutationProduct{T<:NTuple{<:Permutation}} <: Number
    v::T
end

struct SemimagmaProduct{T<:NTuple{<:Permutation}}
    v::T
end
=#

