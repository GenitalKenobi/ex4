import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy before permutation: {accuracy * 100:.2f}%")
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30,
random_state=42)
feature_importances = perm_importance.importances_mean
feature_names = X.columns
for feature_name, importance in zip(feature_names, feature_importances):
 print(f"{feature_name}: {importance:.4f}")
import matplotlib.pyplot as plt
plt.barh(feature_names, feature_importances)
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importance')
plt.show()