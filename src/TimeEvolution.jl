module TimeEvolution


using LinearAlgebra
import DiffEqBase: solve


export ExactDiagonalizationExpectationValue
export EDEV
export ExactDiagonalizationOperatorAutoCorrelation
export EDOAC

export solve
export oacmean


#mutable struct ExactDiagonalization
#    H::AbstractArray
#    A::AbstractArray
#end

#const ED = ExactDiagonalization

#######################################################################################
#
# Exact Diagonlization Expectation Value
#
#######################################################################################

struct ExactDiagonalizationExpectationValue{T}
    eigenvalues::Vector{T}
    iscoef::Vector{T}
    Acoef::Matrix{T}
end

const EDEV = ExactDiagonalizationExpectationValue

function ExactDiagonalizationExpectationValue(
        A::Matrix{T},
        initialstate::Vector{T},
        eigenvalues::Vector{T},
        eigenstates::Matrix{T},
    ) where T<:Real

    @assert length(initialstate) == size(A)[1] == size(A)[2]
    @assert length(initialstate) == length(eigenvalues)
    @assert length(initialstate) == size(eigenstates)[1] == size(eigenstates)[2]

    #@assert ishermitian(A) "Only hermitian A implemented"

    t = 0.0

    eigenvalues = deepcopy(eigenvalues)
    iscoef = (initialstate' * eigenstates)'
    Acoef = eigenstates' * A * eigenstates

    ExactDiagonalizationExpectationValue(
        eigenvalues,
        iscoef,
        Acoef,
    )
end

function solve(edev::ExactDiagonalizationExpectationValue, t)
    eigenvalues = edev.eigenvalues
    iscoef = edev.iscoef
    Acoef = edev.Acoef
    ψ = iscoef .* exp.(-1im*t .* eigenvalues)
    value = ψ' * Acoef * ψ
    return value
end

#######################################################################################
#
# Exact Diagonlization Operator Auto Correlation
#
#######################################################################################

struct ExactDiagonalizationOperatorAutoCorrelation{T}
    eigenvalues::Vector{T}
    iscoef::Vector{T}
    Acoef::Matrix{T}
    isAcoef::Vector{T}

    # cache
    c_1::Vector{Complex{T}}
    c_2::Vector{Complex{T}}
end

EDOAC = ExactDiagonalizationOperatorAutoCorrelation

"""
Creates an ExactDiagonalizationOperatorAutoCorrelation (EDOAC) object to evolve
<ψ|A(0)A(t)|ψ>.
"""
function ExactDiagonalizationOperatorAutoCorrelation(
        A::Matrix{T},
        initialstate::Vector{T},
        eigenvalues::Vector{T},
        eigenstates::Matrix{T},
    ) where T<:Real

    D = length(initialstate)
    @assert D == size(A)[1] == size(A)[2]
    @assert D == length(eigenvalues)
    @assert D == size(eigenstates)[1] == size(eigenstates)[2]

    t = 0.0

    eigenvalues = deepcopy(eigenvalues)
    iscoef = (initialstate' * eigenstates)'
    Acoef = eigenstates' * A * eigenstates
    isAcoef = (initialstate' * A * eigenstates)'

    ExactDiagonalizationOperatorAutoCorrelation(
        eigenvalues,
        iscoef,
        Acoef,
        isAcoef,
        zeros(complex(T), D),
        zeros(complex(T), D),
    )
end

"""
Solves <ψ|A(0)A(t)|ψ> for a specific t
"""
function solve end

function solve(edoac::ExactDiagonalizationOperatorAutoCorrelation, t::Real)
    eigenvalues = edoac.eigenvalues
    iscoef = edoac.iscoef
    Acoef = edoac.Acoef
    isAcoef = edoac.isAcoef
    c_1 = edoac.c_1
    c_2 = edoac.c_2

    c_1 .= iscoef .* exp.(-1im*t .* eigenvalues)
    mul!(c_2, Acoef, c_1)
    #c_2 .= isAcoef .* exp.(-1im*t .* eigenvalues)
    c_1 .= isAcoef .* exp.(1im*t .* eigenvalues)

    #value = c_2' * Acoef * c_1
    value = sum(c_1 .* c_2)
    return value
end

solve(edoac::ExactDiagonalizationOperatorAutoCorrelation, t::AbstractArray{<:Real}) = [solve(edoac,s) for s in t]

@doc raw"""
Calculates
    ``\sum_{i} <ψ|A|E_i> <E_i|A|E_i> <E_i|ψ>``
where
    ψ is the initial value
    ``E_i`` are the eigenstates of the Hamiltonian
    A is the operator
"""
function oacmean(A::AbstractMatrix, initialstate::Vector, eigenstates::Matrix;
    cache::Vector=zeros(length(initialstate)))

    D = length(initialstate)

    @assert D == size(eigenstates)[1] == size(eigenstates)[2]
    @assert D == size(A)[1] == size(A)[2]

    ψ = initialstate
    #cache = zeros(D)
    value = 0
    for i in 1:D
        E = view(eigenstates, :, i)
        mul!(cache, A, E)
        value += dot(ψ, cache)*dot(E, cache)*dot(E,ψ)
    end

    #mul!(cache, A, ψ)
    #value -= dot(ψ, cache)^2

    return value
end

end
