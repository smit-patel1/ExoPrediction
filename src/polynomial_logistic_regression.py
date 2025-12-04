from preprocessing import X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

poly = PolynomialFeatures(2)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

lg = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1200)

lg.fit(X_train_poly, y_train)

y_pred = lg.predict(X_test_poly)

class_report = classification_report(y_test, y_pred, target_names=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])

conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')

test_accuracy = accuracy_score(y_test, y_pred)

print(class_report)

plt.title("Polynomial Logistic Regression Confusion Matrix")
disp_conf_matrix = sns.heatmap(conf_matrix, annot=True, xticklabels=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"], yticklabels=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])
plt.show()

print(f"Test accuracy: {test_accuracy:.4f}")
