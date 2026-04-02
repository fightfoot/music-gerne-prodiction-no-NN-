#  Music Genre Classification using Logistic Regression (OvR)

## Overview

This project explores how machine learning can be used to classify music genres using metadata.
A logistic regression model is implemented from scratch and extended to multiclass classification using a One-vs-Rest (OvR) approach.

The focus of this project is on understanding the **mathematics behind machine learning models** and applying it to a real dataset.

---

##  What This Project Demonstrates

* Implementation of **logistic regression from first principles**
* Use of **gradient descent optimisation**
* Extension to **multiclass classification (OvR)**
* Working with real-world datasets and preprocessing techniques
* Interpreting model behaviour beyond just accuracy

---

##  Features Used

The model uses a set of metadata features, including:

* Duration
* Tempo
* Key and mode
* Loudness
* Time signature
* Year of release
* Word counts from artist names and song titles

These features allow the model to learn patterns without directly analysing audio.

---

##  Model

* One-vs-Rest logistic regression
* Binary classifier trained per genre
* Gradient descent optimisation
* Learning rate: `a = 0.01`

Predictions are made by selecting the class with the highest probability across all classifiers.

---

##  Results

* Achieved **~60% accuracy** on the test set
* Model converges efficiently within relatively few training epochs
* Demonstrates strong capability in learning patterns from structured metadata

---

##  Data Distribution

The dataset contains an uneven distribution of genres, which is common in real-world data.
This provides a useful opportunity to observe how models behave under imbalance and how dominant classes can influence predictions.

With more time, this could be improved by:

* Applying **class weighting** during training
* Using **resampling techniques** (oversampling minority classes or undersampling majority classes)
* Exploring more advanced models that handle imbalance more effectively

---

##  Key Strengths

* Built entirely from scratch without relying on high-level ML libraries
* Clear mapping between mathematical formulas and code implementation
* Efficient vectorised operations using NumPy
* Reproducible results using a defined train/test split 80/20

---

##  Future Scope

* Extend to more advanced models
* Incorporate additional feature engineering
* Explore alternative classification techniques

---

##  Final Thoughts

This project demonstrates how core machine learning concepts can be implemented and applied to real-world data. It highlights the importance of understanding both the theory and practical behaviour of models when solving classification problems.

---
