if (!require("caTools")) 
install.packages("caTools")
#install.packages("caTools")
library(caTools)

trainAdaline <- function(x, y, tol, learnRate, maxEpoch){
  
  dimofx <- dim(x)
  X <- matrix(cbind(x, 1), ncol=dimofx[2] + 1)
  
  dimensionOfX <- dim(X)
  numberOfLines <- dimensionOfX[1]
  numberOfCol <- dimensionOfX[2]
  
  w <- as.matrix(runif(numberOfCol) - 0.5)
  
  currentEpoch <- 0
  
  
  epochError <- tol + 1
  
  errorVector <- matrix(nrow = 1, ncol = maxEpoch)
  while(currentEpoch < maxEpoch && epochError > tol){
    
    xSequence <- sample(numberOfLines)
    error2 <- 0
    for(i in 1:numberOfLines){
      currentSeq <- xSequence[i]
      yPred <- (X[currentSeq,] %*% w)
      error <- (y[currentSeq] - yPred)
      newW <- w + as.vector(learnRate*error)*X[currentSeq,]
      w <-newW
      error2 <- error2 +(error*error)
    }
    currentEpoch <- currentEpoch + 1
    errorVector[currentEpoch] <- error2/numberOfLines
    epochError <- errorVector[currentEpoch]
    
    
  }
  return(list(w, errorVector[1:currentEpoch]))
  
}

#This function runs the training and returns the results

runAdaline <- function(x, y, tol, learnRate, maxEpoch, splitRatio = 0.7){
  #gets the dimensions of x, whatever they are dynamicaly
  dimofx <- dim(x)
  X <- matrix(cbind(x, 1), ncol=dimofx[2] + 1) #adds the column of 1 to the dimension of x
  Y <- matrix(y, ncol = 1)
  
  #set up dataset
  #join data
  data <- cbind(X, Y)
  #setting up collumn names
  columnNames <- vector()
  for(i in (dimofx[2]):0){
    columnNames <- cbind(columnNames, paste("x", i, sep = ""))
  }
  
  columnNames <- cbind(columnNames, 'y')
  
  colnames(data, do.NULL = FALSE)
  colnames(data) <- columnNames
  dataSet <- data.frame(data)
  
  
  #starting seed
  set.seed(42)
  #inicia o metodo de split de dados
  split <- sample.split(dataSet$y, SplitRatio = splitRatio)
 
  trainingSet <- subset(dataSet, split == TRUE)
  testingSet <- subset(dataSet, split == FALSE)
 
  trainingX <- data.matrix(subset(trainingSet, select = -c(x0,y)))
  trainingY <- data.matrix(subset(trainingSet, select = y))

  #trains my adaline
  trainedAdaline <- trainAdaline(trainingX,trainingY,tol,learnRate, maxEpoch)
  adjustedValues <- as.matrix(unlist(trainedAdaline[1]))
  
  #validating model
  validationX <- data.matrix(subset(testingSet, select = -y))
  validationY <- data.matrix(subset(testingSet, select = y))
  
  
  
  #training set error
  finalTraingResult <- (cbind(trainingX,1) %*% adjustedValues)
  trainingError <- as.vector(trainingY - finalTraingResult)
  mediumTrainingError <- sum(trainingError * trainingError)/length(trainingError)
  
  
  #validation set error
  finalResult <- (validationX %*% adjustedValues)
  errorVector <- as.vector(finalResult - validationY)
  errorVector <- (errorVector * errorVector)
  mediumValidationError <- sum(errorVector)/length(errorVector)
  
  return(list(adjustedValues, validationY, mediumTrainingError,mediumValidationError))
  
  

}

