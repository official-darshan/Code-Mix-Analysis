# Code-Mixed Support Analytics — Version 3

This version improves the original project after reviewing the testing mistakes.

## Main improvements

1. 50,000 scenario-driven training conversations.
2. Balanced intent classes, including a new **Product Feedback** intent to avoid classifying positive product messages as Product Complaint.
3. Better English/Hindi/Hinglish detection using strong Hindi markers and script detection.
4. Word + character TF-IDF features.
5. `SGDClassifier(loss="log_loss")` for fast scalable text classification on 50,000 rows. It supports probability/confidence output and is much faster than full Logistic Regression on this sparse text dataset.
6. Separate models for intent, sentiment, resolution and escalation.
7. Resolution is trained using **customer message + agent response**, because resolution is a conversation-level problem.
8. High-precision business rules correct obvious cases such as refund, payment failure, delivery delay, product feedback and escalation signals.
9. Confidence threshold and manual-review state.
10. FastAPI exception handling and validation.
11. Streamlit now calls FastAPI instead of directly importing the prediction function.
12. Curated unseen test set and evaluation script.

## New intents

- Payment Issue
- Refund Request
- Order Cancellation
- Order Tracking
- Delivery Delay
- Login Problem
- Technical Problem
- Product Complaint
- Product Feedback
- General Query

## Run order

### 1. Install

```powershell
pip install -r requirements.txt
```

### 2. Generate 50,000 rows

```powershell
python data/generate_dataset.py
```

### 3. Train models

```powershell
python src/train_models.py
```

### 4. Create unseen test set

```powershell
python data/create_test_set.py
```

### 5. Evaluate

```powershell
python src/evaluate.py
```

### 6. Start API

```powershell
uvicorn api.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### 7. Start dashboard in a second terminal

```powershell
streamlit run dashboard/app.py
```

## Important evaluation note

The 50,000-row training dataset is synthetic educational data. A 100% score on the generated train/test split does not mean 100% real-world accuracy.

The project therefore also includes `data/unseen_test_set.csv` and `src/evaluate.py`. The unseen set is manually curated and should be used to discuss generalization.
