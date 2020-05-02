using Random

include("./CKNN.jl")

Random.seed!(123)

using MAT
# vars = matread("./data/water.mat") 
vars = matread("./data/qm7b.mat") 
varsTest = matread("./data/gdb13.mat")

x = Matrix(vars["offdiag_features"]')
y = vars["offdiag_mp2"]


#Randomly shuffling dataset
# tempPerm = randperm( size(x,2) )
# x = x[ :, tempPerm ] 
#remove duplicates in a hacky way for now:
# x += eps(Float32) * randn( size(x) )
# y = y[ tempPerm ]


#optionally shortening datavector
# x = x[ :, 1 : 40000]
# y = y[ 1 : size(x,2) ]
inds = randperm(size(x, 2))[1:20000]
x = x[:, inds]
y = y[inds]

N = size(x,2)
d = size(x,1)

NTrain = floor( Int64, 0.99 * N)
NTest = N - NTrain
setSplit = randperm(N)
trainInds = setSplit[1:NTrain]
testInds = setSplit[(NTrain + 1) : end]
xTrain = x[:, trainInds]
yTrain = y[trainInds]
xTest = x[:, testInds]
yTest = y[testInds]

xTest = xTest[:, 1 : 10]
yTest = yTest[1 : 10]


xTrain = x[:, :]
yTrain = y[:]
xTest = Matrix(varsTest["offdiag_features"]')[1:d,:]
yTest = varsTest["offdiag_mp2"][:]
test_indices = randperm(length(yTest))[1:10]
xTest = xTest[:, test_indices]
yTest = yTest[test_indices]

cov_func = matern32(10.1)

xTrainPoints = mapslices(Point, xTrain, dims=1)[:]
xTestPoints = mapslices(Point, xTest, dims=1)[:]

@time yPred, σ2Pred = cknn_estimation(xTrainPoints, yTrain, xTestPoints, cov_func, 100)

# training_indices = randperm(length(xTrainPoints))#[1 : 10000]
# yPred_exact = estimate(xTrainPoints[training_indices], yTrain[training_indices], xTestPoints, cov_func)
 
@show norm(yTest - yPred)
# @show norm(yTest - yPred_exact)
