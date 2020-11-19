safelog <- function(x) {
  return (log(x + 1e-100))
}

# read data into memory
data_set1 <- read.csv("hw03_data_set_images.csv",header = FALSE)
data_set2 <- read.csv("hw03_data_set_labels.csv",header = FALSE)

# get X and y values
X <- as.matrix(data_set1)
y_label <-as.matrix(data_set2)



train_X=rbind(X[1:25,1:320],X[40:64,1:320],X[79:103,1:320],X[118:142,1:320],X[157:181,1:320])
test_X=rbind(X[26:39,1:320],X[65:78,1:320],X[104:117,1:320],X[143:156,1:320],X[182:195,1:320])
train_y<-c(1,125)
test_y<-c(1,70)

for (i in 1:125){
  if (i<=25){
    train_y[i]<-1
  }
  else if(i<=50){
    train_y[i]<-2
  }
  else if(i<=75){
    train_y[i]<-3
  }
  else if(i<=100){
    train_y[i]<-4
  }
  else if(i<=125){
    train_y[i]<-5
  }
}
for (i in 1:70){
  if (i<=14){
    test_y[i]<-1
  }
  else if(i<=28){
    test_y[i]<-2
  }
  else if(i<=42){
    test_y[i]<-3
  }
  else if(i<=56){
    test_y[i]<-4
  }
  else if(i<=70){
    test_y[i]<-5
  }
}



# get number of classes and number of samples
K <- max(train_y)
N <- length(train_y)
D <- ncol(X)

# one-of-K-encoding
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, train_y)] <- 1


# define the sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

############### define the softmax function
softmax <- function(X,V) {
  scores <- cbind(1,X) %*% V
  scores <- exp(scores - matrix(apply(scores, MARGIN = 2, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

# set learning parameters
eta <- 0.005
epsilon <- 1e-3
H <- 20
max_iteration <- 200

# randomly initalize W and v
set.seed(521)
W <- matrix(runif((D + 1) * H, min = -0.01, max = 0.01), D + 1, H)
v <- matrix(runif((H + 1) * K, min = -0.01, max = 0.01), H + 1, K)
Z <- sigmoid(cbind(1, train_X) %*% W)
y_predicted<- softmax(Z,v)
objective_values <- c()

objective_values <- c(objective_values, -sum(Y_truth * safelog(y_predicted + 1e-100)))

# learn W and v using gradient descent and online learning
iteration <- 1
while (1) {
  
  
  
  # calculate hidden nodes
  Z <- sigmoid(cbind(1, train_X) %*% W)
  # calculate output node
  y_predicted <- softmax(Z, v)
  Z_bind<-cbind(1,Z)
  v <-v+t(Z_bind)%*%(Y_truth-y_predicted)*eta

  train_X_bind<-cbind(1,train_X)
  a<-t(train_X_bind)%*%(((Y_truth-y_predicted)%*%t(v[-1,]))*(Z)*(1-Z))*eta
  W<-W+a
  
  Z <- sigmoid(cbind(1, train_X) %*% W)
  y_predicted<- softmax(Z,v)
  objective_values <- c(objective_values, -sum(Y_truth * safelog(y_predicted + 1e-100)))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}
print(W)
print(v)

# plot objective function during iterations
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix
Y_predicted <- apply(y_predicted, 1, which.max)
confusion_matrix <- table(Y_predicted, train_y)
print(confusion_matrix)




Z_test <- sigmoid(cbind(1, test_X) %*% W)
y_predicted_test<- softmax(Z_test,v)

# calculate confusion matrix test
Y_predicted_test <- apply(y_predicted_test, 1, which.max)
confusion_matrix_test <- table(Y_predicted_test, test_y)
print(confusion_matrix_test)

