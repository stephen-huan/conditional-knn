include("./CKNN.jl")

d = 3 
N = 1000

xTrain = mapslices(Point, rand(d, N), dims=1)[:]
xTest = Point(rand(3))

cov_func = matern52(0.1)

M = zeros(N, N)
for i in 1 : N
    for j in 1 : N
        M[i, j] = cov_func(xTrain[i], xTrain[j])
    end
end

v = zeros(N + 1)
x = vcat(xTrain, [xTest])

for i in 1 : (N + 1)
    v[i] = cov_func(x[i], x[end])
end


indices_old = cknn(M, v, 20)
indices_new = cknn(xTrain, xTest, cov_func, 20)

@show indices_new 
@show indices_old 