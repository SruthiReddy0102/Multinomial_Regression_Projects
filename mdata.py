### Multinomial Regression ####
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

program = pd.read_csv("C:/Users/personal/Desktop/mdata.csv")

#Removing of unnecessary columns
program1 = program.drop(["index","id"], axis = 1)
program1

program1.columns = "gender","ses","schtyp","prog","read","write","math","science","honors"
program1.head() # Shows first 5 columns of the dataset

program1.describe()
program1.prog.value_counts()
# academic=105 , vocation = 50 and general = 45
program1.gender.value_counts()
program1.ses.value_counts()
program1.schtyp.value_counts()
program1.honors.value_counts()


# Rearrange the order of the variables
program = program1.iloc[:, [3, 0,1,2,4,5,6,7,8]]
program.columns

# Creation of Dummy variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
program['gender'] = le.fit_transform(program['gender'])
program['ses'] = le.fit_transform(program['ses'])
program['schtyp'] = le.fit_transform(program['schtyp'])
program['honors'] = le.fit_transform(program['honors'])

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "prog", y = "read", data = program)
sns.boxplot(x = "prog", y = "write", data = program)
sns.boxplot(x = "prog", y = "math", data = program)
sns.boxplot(x = "prog", y = "science", data = program)

# Scatter plot for each categorical choice of car
sns.stripplot(x = "prog", y = "read", jitter = True, data = program)
sns.stripplot(x = "prog", y = "write", jitter = True, data = program)
sns.stripplot(x = "prog", y = "math", jitter = True, data = program)
sns.stripplot(x = "prog", y = "science", jitter = True, data = program)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(program) # Normal
sns.pairplot(program, hue = "prog") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
program.corr()

train, test = train_test_split(program, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:],train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions
# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 

# Conclusion
# as the test accuracy = 0.55 and train accuracy = 0.65 , which indiactes that there is no major variance 
#which indicate the model is Over fit but at split of 80% and 20% the model is right fit