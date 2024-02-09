import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] 
Y = iris.target

# create an instance of Logistic Regression Classifier and fit the data
log_reg = LogisticRegression(C=1e5)
log_reg.fit(X, Y)

_, ax = plt.subplots(figsize=(4,3))
DecisionBoundaryDisplay.from_estimator(
  log_reg,
  X,
  cmap=plt.cm.Paired,
  ax=ax,
  response_method="predict",
  plot_method="pcolormesh",
  shading="auto",
  xlabel="Sepal length", 
  ylabel="Sepal width",
  eps=0.5,
)

plt.scatter(X[:,0], X[:,1], c=Y, edgecolors="k", cmap=plt.cm.Paired)

plt.xticks(())
plt.yticks(())
plt.show()