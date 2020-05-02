
using StaticArrays
using LinearAlgebra


struct Point{d, RT}  
    position::SVector{d, RT}
end

function Point(v::AbstractVecOrMat)
    @assert size(v,1) == 1 || size(v, 2) == 1
    return Point{length(v), eltype(v)}(SVector{length(v), eltype(v)}(vec(v)))
end

function matern12(l)
    function f(u, v)
        r = norm(u.position - v.position)
        return exp(-r/l)
    end
end

function matern32(l)
    function f(u, v)
        r = norm(u.position - v.position)
        return ( 1 + sqrt(3) * r / l ) * exp( - sqrt(3) * r / l )
    end
end

function matern52(l)
    function f(u, v)
        r = norm(u.position - v.position)
        return ( 1 + sqrt(5) * r / l + 5 * r^2 / 3 / l^2 ) * exp( - sqrt(5) * r / l )
    end
end

function cknn_selection(xTrain, xTest, covariance_function, k_max::Int)
    N = length(xTrain)
    indices = Vector{Int}(undef, k_max)
    factors = zeros(N+1, k_max)
    conditional_covariance = zeros(N)
    conditional_variance = zeros(N)

    # Selecting the first pivot
    reduction = 0.0
    for i in 1 : N
        conditional_variance[i] = covariance_function(xTrain[i], xTrain[i])
        conditional_covariance[i] = covariance_function(xTrain[i], xTest)

        new_reduction = conditional_covariance[i]^2 / conditional_variance[i]
        if new_reduction > reduction
           indices[1] = i 
           reduction = new_reduction
        end
    end

    # Adding the first pivot to factors
    for i in 1 : N
        factors[i, 1] = covariance_function(xTrain[i], xTrain[indices[1]])
    end
    factors[N + 1, 1] = covariance_function(xTest, xTrain[indices[1]])
    factors[:, 1] ./= sqrt(factors[indices[1], 1])

    for k = 2 : k_max
        reduction = 0.0
        for i in 1 : N
            # updating the conditional variance and the covariance with the prediction point
            conditional_variance[i] -= factors[i, k-1]^2 
            conditional_covariance[i] -= factors[i, k-1] * factors[N + 1, k-1]

            new_reduction = conditional_covariance[i]^2 / conditional_variance[i] 
            if conditional_variance[i] < 1e-16
                new_reduction = 0.0
            end
            if new_reduction > reduction
               indices[k] = i 
               reduction = new_reduction
            end
        end

        if reduction ≤ 1e-16
            return indices[1 : (k - 2)]
        end

        # Adding the first pivot to factors
        for i in 1 : N
            factors[i, k] = covariance_function(xTrain[i], xTrain[indices[k]])
        end
        factors[N + 1, k] = covariance_function(xTest, xTrain[indices[k]])



        # Substracting the remaining parts of the factor
        factors[ :, k] .-= factors[ :, 1 : (k - 1)] * factors[indices[k], 1 : (k - 1)]
        # Dividing factor by square root of diagonal entry.


        factors[:, k] ./= sqrt(factors[indices[k],k])
    end
    return indices
end


function cknn_selection(M, v, k_max::Int)
    @assert issymmetric(M)
    N = size(M, 1)
    @assert length(v) == N + 1
    Mv = vcat(hcat(M, v[1 : N]), v')

    xTrain = 1 : N
    xTest = N + 1

    covariance_function(i, j) = Mv[i, j]

    return cknn(xTrain, xTest, covariance_function, k_max)
end

# Perform estimation on the set of Test points
function cknn_estimation(xTrain, yTrain, xTest, covariance_function, k_max::Int)
    
    yPred = zeros(length(xTest))
    σ2Pred = zeros(length(xTest))
    K = zeros(k_max, k_max)
    v = zeros(k_max + 1)

    for ind_test = 1 : length(xTest)
        indices = cknn_selection(xTrain, xTest[ind_test], covariance_function, k_max)
        @show length(indices)

        if mod(ind_test, 10) == 0
            println("Treating test point number $ind_test")
        end

        for (ind_i, i) in enumerate(indices)
            # @show typeof(xTest)
            # @show typeof(xTrain)
            # @show indices
            # @show i
            # @show ind_test
            v[ind_i] = covariance_function(xTrain[i], xTest[ind_test])
            for (ind_j, j) in enumerate(indices)
                K[ind_i, ind_j] = covariance_function(xTrain[i], xTrain[j])
            end
        end
        v[end] = covariance_function(xTest[ind_test], xTest[ind_test]) 

        yPred[ind_test] = yTrain[indices]' * (K[1 : length(indices), 1 : length(indices)] \ v[1 : length(indices)])
        σ2Pred[ind_test] = v[end] - v[1 : length(indices)]' * (K[1 : length(indices), 1 : length(indices)] \ v[1 : length(indices)])
    end
    return yPred, σ2Pred
end

function estimate(xTrain, yTrain, xTest, covariance_function)
    N = length(xTrain)
    M = length(xTest)

    Θ = zeros(N, N)
    V = zeros(N, M)
    for i in 1 : N
        for j = 1 : N
            Θ[i, j] = covariance_function(xTrain[i], xTrain[j])
        end
        for j in 1 : M
            V[i, j] = covariance_function(xTrain[i], xTest[j])
        end
    end
    return (V' * (Θ \ yTrain))[:]
end