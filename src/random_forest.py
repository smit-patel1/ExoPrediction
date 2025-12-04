from sklearn.ensemble import RandomForestClassifier
from preprocessing import y_train, y_test, X_train_scaled, X_test_scaled
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


rf = RandomForestClassifier(n_estimators=200, max_depth=15, criterion='gini', max_features='sqrt', random_state=42)

rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
 
class_report = classification_report(y_test, y_pred, target_names=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])

conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')

test_accuracy = accuracy_score(y_test, y_pred)


print(class_report)

plt.title("Random Forest Confusion Matrix")
disp_conf_matrix = sns.heatmap(conf_matrix, annot=True, xticklabels=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"], yticklabels=["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])
plt.show()

print(f"Test accuracy: {test_accuracy:.4f}")
