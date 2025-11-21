# Exoplanet Multiclass Classification – Final Project

ITCS 3156: Introduction to Machine Learning  
Smit Patel

## Overview

This project applies machine learning algorithms to classify Kepler Objects of Interest (KOIs) collected by NASA.  
Each KOI is labeled as one of the following:

- **CONFIRMED** – a validated exoplanet
- **CANDIDATE** – a promising transit signal awaiting confirmation
- **FALSE POSITIVE** – a signal caused by noise, instrument artifacts, or non-planet astrophysical sources

The goal is to build a complete ML pipeline that explores the dataset, preprocesses it, trains multiple models, and evaluates performance.

---

## Dataset

The dataset used is the **NASA Kepler KOI cumulative table**, sourced from Kaggle.

It contains ~49 columns and thousands of KOIs.

### **Selected Features**

From the full dataset, 14 scientifically meaningful features were selected:

#### Transit Features

- `koi_period`
- `koi_duration`
- `koi_depth`
- `koi_prad`
- `koi_model_snr`
- `koi_impact`

#### Stellar Features

- `koi_steff`
- `koi_slogg`
- `koi_srad`
- `koi_kepmag`

#### Diagnostic (False Positive) Flags

- `koi_fpflag_nt`
- `koi_fpflag_ss`
- `koi_fpflag_co`
- `koi_fpflag_ec`

### **Target Variable**

- `koi_disposition`  
  Encoded as:
  - **0** → CONFIRMED
  - **1** → CANDIDATE
  - **2** → FALSE POSITIVE

---

## Machine Learning Models Used:

### **1. Logistic Regression**

Provides a linear baseline for multiclass classification.

### **2. Polynomial Logistic Regression (Degree = 2)**

Logistic Regression with nonlinear feature interactions.

### **3. Gaussian Naive Bayes**

A generative model with strong independence assumptions.  
Useful as a baseline comparison.

### **4. Random Forest Classifier**

A nonlinear ensemble model.

All models use **scikit-learn** implementations.

---
