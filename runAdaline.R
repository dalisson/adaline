rm(list = ls())

source("trainAdaline.R")

baseDirectory <- 'dados/'

filex <- 'x'
filey <- 'y'
file_t <- 't'

#opens the files and loads them as a matrix
t <- read.table(paste(baseDirectory, filex, sep = ''))
x <- as.matrix(t, ncol = dim(t)[2])
y <- as.matrix(read.table(paste(baseDirectory, filey, sep = '')))
t <- as.matrix(read.table(paste(baseDirectory, file_t, sep = '')))

#Adalaine result

result <- runAdaline(x, y, 0.01, 0.01, 2000, 0.7) 

param <- matrix(unlist(result[1]), ncol=1)

#these are the result that will be obtained applying the parameters from the adaline to the x dataset

obtainedResult <- (cbind(x, 1)) %*% param 

#Plotting the graphics

plot(t, y, type = 'l', col = 'green')
par(new=T)
plot(t,obtainedResult, col = 'red')