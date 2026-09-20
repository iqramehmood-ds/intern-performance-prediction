# Intern Performance Prediction

Task 3 of my Data Analytics Internship at Internee.pk: predict intern performance from engagement and task completion data using machine learning, and turn that into insights mentors can use.

## About the Data

Internee.pk didn't provide a dataset for this task, so I generated a synthetic dataset of 300 interns instead of leaving the project without data.

The dataset is designed to simulate plausible relationships between attendance, task completion, mentor feedback, engagement, and a "success" outcome, with randomness added so the relationships are not perfectly deterministic.

**This is simulated data, not real Internee.pk records.** The model is learning a success label generated from the same underlying variables, so the results demonstrate the machine-learning workflow rather than proving predictive performance on a real intern cohort.

### Columns

- `attendance_rate` – percentage of sessions attended
- `tasks_assigned` – number of assigned tasks
- `tasks_completed` – number of completed tasks
- `task_completion_rate` – percentage of assigned tasks completed
- `avg_submission_delay_days` – average days late per task
- `mentor_feedback_score` – mentor rating out of 10
- `engagement_score` – general engagement measure
- `success` – outcome label used to train the model

## What the Script Does

1. Generates the synthetic dataset (`intern_dataset.csv`)
2. Splits the data into training (75%) and test (25%) sets
3. Trains a Random Forest classifier and a Logistic Regression baseline
4. Evaluates the models using accuracy, ROC-AUC, and classification metrics
5. Generates success probabilities and risk flags
6. Saves predictions to `intern_predictions.csv`
7. Generates visualizations to explore the model and the data
8. Produces insights that could help mentors identify interns who may need additional support

## Results

- Random Forest accuracy: **0.83**
- Random Forest ROC-AUC: **0.85**
- Logistic Regression accuracy: **0.88**
- Logistic Regression ROC-AUC: **0.84**
- Random Forest at-risk recall: **0.43**

The at-risk recall is particularly important for this use case. A model can achieve high overall accuracy simply by predicting "success" for most interns when the classes are imbalanced. That would be less useful for identifying interns who may need support.

The Random Forest used `class_weight='balanced'` to give additional weight to the minority class during training.

## Key Insights

1. Task completion rate showed the highest feature importance, followed by mentor feedback score and attendance.
2. **56 of the 300 synthetic interns (19%)** were flagged as at risk by the model in this run.
3. Lower attendance combined with lower task completion appeared to be a stronger warning pattern than either measure alone.
4. Submission delay showed a weaker individual relationship with the model's predictions, although consistently delayed submissions may still warrant mentor attention.

## Limitations

- The dataset is synthetic and does not contain real Internee.pk intern records.
- The success label was generated using a formula based partly on the same features used by the model. Therefore, the model is partly learning the structure used to create the dataset.
- The results cannot be treated as evidence of how accurately the model would predict real intern performance.
- At-risk recall (0.43) is still moderate, so the model should be viewed as a starting point for identifying interns who may benefit from a check-in, not as a final decision-making tool.

## Visualizations

### Attendance vs Task Completion

<img width="975" height="750" alt="attendance_vs_completion" src="https://github.com/user-attachments/assets/8d34843b-0b8d-43b0-a497-2b1559095a91" />


### Confusion Matrix

<img width="675" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/f848b1ff-862e-4ed7-8afa-9ab9d9a9dfa4" />


### Feature Importance

<img width="1050" height="600" alt="feature_importance" src="https://github.com/user-attachments/assets/c44a03f5-a15a-4639-a820-63934aad5ee2" />


### Success Probability Distribution

<img width="1050" height="600" alt="probability_distribution" src="https://github.com/user-attachments/assets/e25fff8c-b863-4bbc-965d-dad46c821d8b" />

## Tools

Python, pandas, numpy, scikit-learn, matplotlib

## How to Run

```bash
pip install -r requirements.txt
python predict_intern_performance.py
```
