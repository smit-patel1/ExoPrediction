from preprocessing import y_train, y_test, X_train_scaled, X_test_scaled
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

gnb = GaussianNB()

gnb.fit(X_train_scaled, y_train)

y_pred = gnb.predict(X_test_scaled)

class_report = classification_report(y_test, y_pred, target_names=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])

conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')

print(class_report)

plt.title("Guassian Naive Bayes Confusion Matrix")
disp_conf_matrix = sns.heatmap(conf_matrix, annot=True, xticklabels=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"], yticklabels=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])
plt.show()

