# AI-Ethics-
# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load Dataset
dataset = CompasDataset()

# Initial Metrics
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{'race': 1}], unprivileged_groups=[{'race': 0}])
print("Disparate Impact (DI):", metric.disparate_impact())

# Train/Test Split
from sklearn.model_selection import train_test_split
X = dataset.features
y = dataset.labels.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Post-prediction Fairness Evaluation
classified_dataset = dataset.copy()
classified_dataset.labels = y_pred

classified_metrics = ClassificationMetric(dataset, classified_dataset,
                                          unprivileged_groups=[{'race': 0}],
                                          privileged_groups=[{'race': 1}])

print("False Positive Rate Difference:", classified_metrics.false_positive_rate_difference())
print("Equal Opportunity Difference:", classified_metrics.equal_opportunity_difference())

# Visualize False Positive Rate
fpr_unpriv = classified_metrics.false_positive_rate(privileged=False)
fpr_priv = classified_metrics.false_positive_rate(privileged=True)

plt.bar(['Unprivileged (Black)', 'Privileged (White)'], [fpr_unpriv, fpr_priv], color=['red', 'blue'])
plt.title('False Positive Rate by Race')
plt.ylabel('Rate')
plt.show()
300-Word Report Summary

The COMPAS dataset, widely used for recidivism prediction, has shown evidence of racial bias, particularly disadvantaging Black defendants. Using IBM’s AI Fairness 360, we conducted a fairness audit comparing privileged (White) and unprivileged (Black) groups.

Initial metrics indicated a disparate impact ratio < 1, showing potential systemic disadvantage against the unprivileged group. After training a logistic regression classifier, we observed a False Positive Rate (FPR) Difference > 0, meaning Black defendants were more likely to be incorrectly classified as high-risk compared to White defendants.

We also found a significant Equal Opportunity Difference, indicating the model fails to predict positive outcomes (non-recidivism) equally across groups. These disparities reflect real-world harm, such as unjust pretrial detainment or longer sentences.

Remediation strategies include preprocessing techniques like Reweighing, which adjusts instance weights to reduce bias during training. We recommend implementing this or adopting post-processing techniques like threshold optimization to balance error rates.

In conclusion, while COMPAS attempts to aid judicial decisions, our audit highlights that unmitigated use of such tools can reinforce societal inequities. AI systems in criminal justice must undergo continuous bias monitoring, transparent documentation, and public accountability to ensure fairness.
