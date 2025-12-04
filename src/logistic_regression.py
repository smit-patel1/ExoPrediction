from preprocessing import y_train, y_test, X_train_scaled, X_test_scaled
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

lg = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)

lg.fit(X_train_scaled, y_train)

y_pred = lg.predict(X_test_scaled)

class_report = classification_report(y_test, y_pred, target_names=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])

conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')

test_accuracy = accuracy_score(y_test, y_pred)

print(class_report)

plt.title("Logistic Regression Confusion Matrix")
dis_conf_matrix = sns.heatmap(conf_matrix, annot=True, xticklabels=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"], yticklabels=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])
plt.show()

print(f"Test accuracy: {test_accuracy:.4f}")