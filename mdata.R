### Analyzing the business problem ###

# Output Variable (y) = Type of program (prog) which the student take the most
# Input Variables = x,id,female,ses,schtyp,read,write,math,science,honors

# Importing required packages
require('mlogit')
require('nnet')

### Importing Dataset ###
program <- read.csv(file.choose())
View(program)
attach(program)

head(program) # Shows first 6 rows of the dataset
tail(program ) # Showa last 6 rows of the dataset

# Tabular represntation of output variable
table(program$prog)
# Student who opt academic = 105 , general = 45 , vocation = 50

### Data Preprocessing ##

# Checking of NA values
sum(is.na(program)) # No NA Values found

# Removing Uncessary columns
program  <- program[ , 3:11]
View(program)

# Renaming the column names
colnames(program) <- c("gender","ses","schtyp","prog","read","write","math","science","honors")
View(program)

# Reorder the variables
program  <- program[,c(4,1,2,3,5,6,7,8,9)]
View(program)

# Creating dummy variables

install.packages("dummies")
library(dummies)

str(program)


program$gender <- as.factor(program$gender)
program$gender <- as.numeric(program$gender)

program$ses <- as.factor(program$ses)
program$ses <- as.numeric(program$ses)

program$schtyp <- as.factor(program$schtyp)
program$schtyp <- as.numeric(program$schtyp)

program$honors <- as.factor(program$honors)
program$honors = as.numeric(program$honors)

str(program)


#Exploratory data analysis
summary(program)

install.packages("Hmisc")
library(Hmisc)
describe(program)

install.packages("lattice") # Highly used for data visualization
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(program$science, main = "Dot Plot of Science")
dotplot(program$read, main = "Dot Plot of Read")
dotplot(program$write, main = "Dot Plot of Write")
dotplot(program$math, main = "Dot Plot of Math")


#Boxplot Representation

boxplot(program$read, col = "dodgerblue4")
boxplot(program$math, col = "dodgerblue4")
boxplot(program$science, col = "dodgerblue4")
boxplot(program$write, col = "red", horizontal = T)

#Histogram Representation

hist(program$read,col="blue",main="Read")
hist(program$write, col="blue",main="Write")
hist(program$math ,col="blue",main="Math")
hist(program$science , col="blue",main="Science")


# Normal QQ plot
qqnorm(program$read ,main = "Read")
qqline(program$read , main = "Read")

qqnorm(program$write , main = "Write")
qqline(program$write , main ="Write")

qqnorm(program$math, main = "Math")
qqline(program$math, main ="Math")

qqnorm(program$science, main = "Science")
qqline(program$science, main ="Science")

#Scatter plot for all pairs of variables
plot(program)

# correlation matrix
cor(program)

# Data Partitioning
n <- nrow(program)
n1 <- n * 0.7
n2 <- n - n1
train_index <- sample(1:n,n1)
train <- program[train_index,]
test <- program[-train_index,]

## Model Building ##
program.prog <- multinom(prog ~ ., data=train)
summary(program.prog)

#Residual Deviance: 256.2956 
#AIC: 292.2956


##### Significance of Regression Coefficients###
z <- summary(program.prog)$coefficients / summary(program.prog)$standard.errors
z

p_value <- (1-pnorm(abs(z),0,1))*2

summary(program.prog)$coefficients
p_value

# odds ratio 
exp(coef(program.prog))

# predict probabilities
prob <- fitted(program.prog)
prob

# Prediction on test data
pred_test <- predict(program.prog, newdata = test , type = "probs")
pred_test

# Find the accuracy of the model
class(pred_test)
pred_test <- data.frame(pred_test)
View(pred_test)
pred_test["prediction"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

predtest_name <- apply(pred_test, 1, get_names)
pred_test$prediction <- predtest_name
View(pred_test)

# Confusion matrix
table(predtest_name, test$prog)

# confusion matrix visualization
barplot(table(predtest_name, test$prog),beside = T,col=c("red","lightgreen","blue","orange"), legend=c("academic","general","Vocational"), main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")
barplot(table(predtest_name, test$prog),beside = T,col=c("red","lightgreen","blue","orange"), main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")

# Accuracy 
mean(predtest_name == test$prog)


# Training Data
pred_train <- predict(program.prog , newdata = train , type = "probs")
pred_train

# # Find the accuracy of the model
class(pred_train)
pred_train <- data.frame(pred_train)
View(pred_train)
pred_train["prediction"] <- NULL

predtrain_name <- apply(pred_train, 1, get_names)
pred_train$prediction <- predtrain_name
View(pred_train)

# Confusion Matrix
table(predtrain_name , train$prog)

# confusion matrix visualization
barplot(table(predtrain_name, train$prog),beside = T,col=c("red","lightgreen","blue","orange"), legend=c("academic","general","Vocational"), main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")
barplot(table(predtrain_name, train$prog),beside = T,col=c("red","lightgreen","blue","orange"), main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")

# Accuracy 
mean(predtrain_name == train$prog)


