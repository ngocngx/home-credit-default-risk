# Home Credit Fault Risk Prediction Project

## Overview

This repository contains the code and resources for a machine learning project focused on predicting credit default risk for Home Credit customers. The goal is to build a robust model that can assess the likelihood of a customer defaulting on a loan based on various features.

## Table of Contents

- [Project Description](#project-description)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)


## Repo Structure
    ├── README.md             <- Overview and guidance for developers.
    ├── requirements.txt      <- List of Python packages needed for the project.
    ├── eda                   <- Notebooks for initial data exploration.
    ├── processed-data        <- Final data sets for model training.
    ├── raw-data              <- Original, unmodified data.
    ├── feature-processing
    │   ├── app_train.py      <- Processes application training data.
    │   ├── bureau.py         <- Transforms bureau data.
    │   ├── credit_card_balance.py <- Handles credit card balance data.
    │   ├── installment_payment.py <- Manages installment payment data.
    │   ├── posc_ash.py       <- Processes POS cash data.
    │   └── previous.py       <- Handles previous application data.
    └── pipeline              <- Scripts for data and model pipelines.

## Project Description

Home Credit, as a financial institution, faces challenges in predicting credit default risk accurately. This project aims to develop a machine learning model that can assist in identifying customers who are more likely to default on loans. The model will analyze various features such as income, employment history, and previous credit behavior to provide risk assessments.

## Data

The dataset used for this project is sourced from Home Credit and includes information on customer demographics, financial history, and loan performance. The data preprocessing steps involve handling missing values, encoding categorical variables, and scaling numerical features.

[Link to Kaggle Dataset](https://www.kaggle.com/competitions/home-credit-default-risk)

## Installation

To set up the project locally, follow these steps:

```bash
git clone https://github.com/ngocngx/home-credit-default-risk.git
pip install -r requirements.txt
