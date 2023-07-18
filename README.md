
# email-discriminator

This is a project for the MLOps Zoomcamp Course where I'll be creating an email discriminator

## Problem Description

### Introduction

Emails have become a primary source of information in our digital age. However, with the sheer volume of emails received daily, it can be challenging to manage and filter out relevant information. An email discriminator, a tool that classifies or filters emails based on their relevance, can assist with this information overload.

In this project, we focus on a specific type of emails: the "TLDR" (Too Long, Didn't Read) newsletters from [tldr.tech](https://tldr.tech/). These newsletters provide summarized versions of the latest news in tech and AI. But, as with any newsletter, not all articles are of interest to every reader.

### Problem Statement

The goal of this project is to develop an email discriminator to filter the TLDR newsletters and identify the articles that are most relevant to the user.

The problem is a binary classification task with two classes:

Class 1: Relevant TLDR articles
Class 0: Irrelevant TLDR articles
The model will take as input the text of a TLDR article and output a prediction of whether it is relevant (Class 1) or not (Class 0).

### Data

The data for this project comes from the user's Gmail account. The user has categorized the TLDRs into different labels:

TLDRs that the user cares about are in a Gmail label called "TLDRs". These TLDRs will form the positive class (Class 1).
"Archived" TLDRs that the user doesn't care about will form the negative class (Class 0).
TLDRs in the "inbox" are the ones on which the model will make predictions.
Each TLDR contains a title and text, which are the features that the model will use to make predictions. The label (relevant or not) is the target variable.

Where and how these are stored in Gmail depends on the user, but in this case this is how I'm storing it.

### Approach

The approach to solving this problem involves several steps:

1. **Data Collection**: Using the Gmail API to fetch the TLDRs from the different Gmail labels.

2. **Data Preprocessing**: Processing and cleaning the text data.

3. **Feature Engineering**: Converting the text data into a form that can be used to train a machine learning model.

4. **Model Training**: Training a binary classification model on the processed data.

5. **Model Evaluation**: Evaluating the model's performance using suitable metrics.

6. **Model Deployment**: Deploying the model to a production environment where it can make predictions on new, unseen TLDRs.
