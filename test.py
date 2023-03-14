from skmini.linear_model import LogisticRegression

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=1000)
model.fit(X[20:80], y[20:80])
print(model.score(X[:100], y[:100]))
