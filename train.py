from sklearn.inspection import permutation_importance
from sklearn.svm import NuSVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = load_iris()
X = df.data
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = NuSVC()
model.fit(X_train, y_train)

importance_model = permutation_importance(model, X_test, y_test, n_repeats=40, random_state=42)

importance_features = importance_model.importances_mean
importance_names = df.feature_names

plt.barh(importance_names, importance_features)
plt.show()