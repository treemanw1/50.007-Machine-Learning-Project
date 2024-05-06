# 50.007 Machine Learning Project
## Task 1: Implement Logistics Regression (10 marks)
Recalled that you have learned about Logistic Regression in your earlier class. Your task is to implement a Logistic Regression model from scratch. Note that you are NOT TO USE the sklearn logistic regression package or any other pre-defined logistic regression package for this task! Usage of any logistic regression packages will result in 0 marks for this task.

## Key Task Deliverables
1a. Code implementation of the Logistic Regression model.

1b. Prediction made by your Logistic Regression on the Test set. Note that you are welcome to submit your predicted labels to Kaggle but you will need to submit the final prediction output in the final project submission. Please label the file as "LogRed_Prediction.csv".

## Tips
Check out the Logistic Regression implementation in this awesome blog.

Your implementation should have the following functions:

-- sigmoid(z): A function that takes in a Real Number input and returns an output value between 0 and 1.

-- loss(y, y_hat): A loss function that allows us to minimize and determine the optimal parameters. The function takes in the actual labels y and the predicted labels yhat, and returns the overall training loss. Note that you should be using the Log Loss function taught in class.

-- gradients(X, y, y_hat): The Gradient Descent Algorithm to find the optimal values of our parameters. The function takes in the training feature X, actual labels y and the predicted labels yhat, and returns the partial derivative of the Loss function with respect to weights (w) and bias (db).

-- train(X, y, bs, epochs, lr): The training function for your model.

-- predict(X): The prediction function where you can apply your validation and test sets.
