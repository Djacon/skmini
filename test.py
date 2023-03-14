from skmini.neighbors import KNeighborsClassifier, KNeighborsRegressor
from skmini.linear_model import Linear

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X[30:120], y[30:120])
print(model.score(X, y))

model = KNeighborsRegressor(n_neighbors=1)
model.fit(X[30:120], y[30:120])
print(model.score(X, y))
