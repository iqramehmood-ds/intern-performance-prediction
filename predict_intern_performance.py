# Task 3 - Predict Intern Performance Based on Engagement and Task Completion
# Data Analytics Internship

# internee.pk didn't give us a dataset for this one, same as task 1, so I made
# my own synthetic dataset for 300 interns instead of leaving it blank.
# Not real internee.pk data - just numbers built to behave like real intern
# data would (better attendance + more finished tasks = more likely to succeed).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, recall_score

np.random.seed(42)  # so I get the same numbers every time I rerun this

# ------------------ building the dataset ------------------

n = 300

attendance_rate = np.clip(np.random.normal(0.80, 0.15, n), 0.2, 1.0)
tasks_assigned = np.random.randint(4, 11, n)
task_completion_rate = np.clip(np.random.normal(0.75, 0.18, n), 0.1, 1.0)
avg_submission_delay = np.clip(np.random.exponential(1.5, n), 0, 10)
mentor_feedback_score = np.clip(np.random.normal(7.0, 1.8, n), 1, 10)
# keeping engagement separate from attendance on purpose, otherwise they'd be
# correlated just because of how I built them, not because of anything real
engagement_score = np.clip(np.random.normal(6, 2, n), 0, 10)

tasks_completed = (task_completion_rate * tasks_assigned).round().astype(int)

# this is basically me deciding what "success" means, since there's no real
# outcome column to use. attendance + task completion matter most, feedback
# and engagement a bit less, and being late pulls the score down. added some
# random noise too so it's not a perfectly clean formula
success_score = (
    0.30 * attendance_rate
    + 0.30 * task_completion_rate
    + 0.15 * (mentor_feedback_score / 10)
    + 0.15 * (engagement_score / 10)
    - 0.10 * (avg_submission_delay / 10)
)
noise = np.random.normal(0, 0.05, n)
success = (success_score + noise > 0.55).astype(int)

df = pd.DataFrame({
    "intern_id": [f"INT{1000+i}" for i in range(n)],
    "attendance_rate": attendance_rate.round(2),
    "tasks_assigned": tasks_assigned,
    "tasks_completed": tasks_completed,
    "task_completion_rate": task_completion_rate.round(2),
    "avg_submission_delay_days": avg_submission_delay.round(2),
    "mentor_feedback_score": mentor_feedback_score.round(2),
    "engagement_score": engagement_score.round(2),
    "success": success
})

df.to_csv("intern_dataset.csv", index=False)
print(f"Dataset built: {n} interns, {success.sum()} labelled successful ({success.mean():.0%})")

# ------------------ train/test split ------------------

features = [
    "attendance_rate", "task_completion_rate", "avg_submission_delay_days",
    "mentor_feedback_score", "engagement_score"
]
X = df[features]
y = df["success"]

# holding back 25% so I can actually check if the model learned something
# real instead of just memorizing the training interns
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ------------------ training the models ------------------

# using Random Forest as the main model - handles the mixed feature scales
# fine and gives feature importance for free, which I need for the mentor
# insights later. class_weight='balanced' matters here because only ~15-20%
# of interns end up "at risk" - without it the model can just predict
# success for basically everyone and still get decent accuracy, which
# defeats the whole point of this task
rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

# logistic regression as a simpler baseline to compare against
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

for name, model in [("Random Forest", rf), ("Logistic Regression", log_reg)]:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, probs):.2f}")

print("\nClassification report (Random Forest):")
print(classification_report(y_test, rf.predict(X_test)))

# accuracy alone is misleading with imbalanced classes like this, so I'm
# also checking recall specifically for the at-risk class - basically,
# of the interns who were actually at risk, how many did the model catch
at_risk_recall = recall_score(y_test, rf.predict(X_test), pos_label=0)
print(f"At-risk recall (Random Forest): {at_risk_recall:.2f}")

# ------------------ predicting for every intern ------------------

df["success_probability"] = rf.predict_proba(X)[:, 1].round(3)
df["risk_flag"] = np.where(df["success_probability"] < 0.5, "At Risk", "On Track")

df.to_csv("intern_predictions.csv", index=False)
print("\nSaved predictions to intern_predictions.csv")
print(df["risk_flag"].value_counts())

# ------------------ charts ------------------

plt.style.use("seaborn-v0_8-whitegrid")

# which features the model actually relied on
importances = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="#4C72B0")
plt.title("What drives predicted intern success")
plt.xlabel("Relative importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# spread of predicted probabilities across all interns
plt.figure(figsize=(7, 4))
plt.hist(df["success_probability"], bins=20, color="#55A868", edgecolor="white")
plt.axvline(0.5, color="red", linestyle="--", label="Risk threshold (0.5)")
plt.title("Distribution of predicted success probability")
plt.xlabel("Predicted probability of success")
plt.ylabel("Number of interns")
plt.legend()
plt.tight_layout()
plt.savefig("probability_distribution.png", dpi=150)
plt.close()

# how many the model got right vs wrong on the test set
cm = confusion_matrix(y_test, rf.predict(X_test))
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion matrix (test set)")
plt.xticks([0, 1], ["Predicted: At Risk", "Predicted: Success"])
plt.yticks([0, 1], ["Actual: At Risk", "Actual: Success"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# attendance vs completion, colored by whether the model flagged them at risk
plt.figure(figsize=(6.5, 5))
colors = df["risk_flag"].map({"At Risk": "#C44E52", "On Track": "#55A868"})
plt.scatter(df["attendance_rate"], df["task_completion_rate"], c=colors, alpha=0.6)
plt.xlabel("Attendance rate")
plt.ylabel("Task completion rate")
plt.title("Attendance vs task completion (colour = risk flag)")
plt.tight_layout()
plt.savefig("attendance_vs_completion.png", dpi=150)
plt.close()

print("\nSaved charts: feature_importance.png, probability_distribution.png, "
      "confusion_matrix.png, attendance_vs_completion.png")

# ------------------ insights for mentors ------------------

top_feature = importances.idxmax()
at_risk_count = (df["risk_flag"] == "At Risk").sum()
at_risk_pct = at_risk_count / n

print("\nInsights for mentors:")
print(f"1. {top_feature.replace('_', ' ').title()} is the strongest predictor of intern success.")
print(f"2. {at_risk_count} interns ({at_risk_pct:.0%}) are flagged 'At Risk' (success probability below 0.5).")
print("3. At-risk interns should be prioritised for one-on-one check-ins, especially where "
      "attendance and task completion rate are both trending down together.")
print("4. Submission delay has a smaller but real negative effect - mentors can catch this "
      "early by watching for tasks turned in more than 2-3 days late.")
