# Project Report Guide

## 1. Abstract
This project develops an NLP and machine learning based support conversation analytics system for code-mixed customer messages. The system predicts customer intent, sentiment, resolution status and escalation requirement. Predictions are exposed through a FastAPI service and visualized through a Streamlit dashboard.

## 2. Objectives
1. Process English, Hindi and Hinglish support messages.
2. Predict customer intent.
3. Predict resolution status.
4. Predict escalation requirement.
5. Analyze sentiment.
6. Calculate escalation probability and priority.
7. Store predictions in SQLite.
8. Provide API and dashboard access.

## 3. NLP Method
Text is normalized using lowercase conversion, URL/symbol removal and whitespace normalization. TF-IDF with unigrams and bigrams converts text into numerical features.

## 4. Machine Learning Method
Logistic Regression is used because it is simple, fast and effective for sparse TF-IDF text classification. Separate models are trained for intent, sentiment, resolution and escalation.

## 5. Analytics
The dashboard displays total conversations, resolution count, escalation count, high-priority count, intent distribution, sentiment distribution, resolution distribution and escalation distribution.

## 6. API
FastAPI exposes:
- GET /
- POST /analyze
- GET /analytics

The POST /analyze endpoint accepts a customer message and returns all predictions.

## 7. Database
SQLite stores each analyzed conversation and its predictions.

## 8. Limitations
The included dataset is intentionally small for a self-contained educational project. A production system needs a much larger, diverse and validated dataset, privacy controls, human review and monitoring.

## 9. Future Scope
- Larger real-world support dataset
- Multilingual transformer model
- Better Hindi/Hinglish language identification
- Agent performance analytics
- Conversation-level features
- Human feedback loop
- Authentication and deployment
