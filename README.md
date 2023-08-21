
<img width="580" height="200" src="./images/email-discriminator-title.PNG" alt="TLDR email-discriminator">

[![codecov](https://codecov.io/gh/ugm2/email-discriminator/graph/badge.svg?token=6FCPGWHIZ4)](https://codecov.io/gh/ugm2/email-discriminator)

This is a project for the MLOps Zoomcamp Course where I'll be creating an email discriminator

## Problem Description

### Introduction

Emails have become a primary source of information in our digital age. However, with the sheer volume of emails received daily, it can be challenging to manage and filter out relevant information. An email discriminator, a tool that classifies or filters emails based on their relevance, can assist with this information overload.

In this project, we focus on a specific type of emails: the "TLDR" (Too Long, Didn't Read) newsletters from [tldr.tech](https://tldr.tech/). These newsletters provide summarized versions of the latest news in tech and AI. But, as with any newsletter, not all articles are of interest to every reader.

### Problem Statement

The goal of this project is to develop an email discriminator to filter the TLDR newsletters and identify the articles that are most relevant to the user.

The problem is a binary classification task with two classes:

* Class 1: Relevant TLDR articles

* Class 0: Irrelevant TLDR articles

The model will take as input the text of a TLDR article and output a prediction of whether it is relevant (Class 1) or not (Class 0).

### Data

The data for this project comes from the user's Gmail account. The user has MANUALLY categorised the TLDRs into different labels:

* TLDRs that the user cares about are in a Gmail label called "TLDRs". These TLDRs will form the positive class (Class 1).

* "Archived" TLDRs that the user doesn't care about will form the negative class (Class 0).

* TLDRs in the "inbox" are the ones on which the model will make predictions.
Each TLDR contains a title and text, which are the features that the model will use to make predictions. The label (relevant or not) is the target variable.

Where and how these are stored in Gmail depends on the user, but in this case this is how I'm storing it.

### Approach

```mermaid
graph TD

    style A fill:#f9d5e5,stroke:#333,stroke-width:4px;
    style B fill:#eeac99,stroke:#333,stroke-width:4px;
    style C fill:#e06377,stroke:#333,stroke-width:4px;
    style D fill:#c83349,stroke:#333,stroke-width:4px;
    style E fill:#5b9aa0,stroke:#333,stroke-width:4px;
    style F fill:#d6eafd,stroke:#333,stroke-width:4px;
    style H fill:#f3c623,stroke:#333,stroke-width:4px;
    style I fill:#93c6ed,stroke:#333,stroke-width:4px;
    style J fill:#c6c3ed,stroke:#333,stroke-width:4px;
    style K fill:#b2c9ab,stroke:#333,stroke-width:4px;
    style L fill:#f6e3b4,stroke:#333,stroke-width:4px;
    style M fill:#a9f1e1,stroke:#333,stroke-width:4px;
    style N fill:#f6d8e0,stroke:#333,stroke-width:4px;
    style O fill:#e9b0df,stroke:#333,stroke-width:4px;

    A[Start]
    B[Get and store initial labeled data]
    C[Train & deploy initial ML model]
    D{Weekly Batch Process}
    H[Fetch emails, delete from source & predict]
    E[Review & Label via UI]
    F[Re-train & deploy new model]
    G[Loop to next batch]
    I[Prefect Cloud]
    J[MLFlow GCP Server]
    K[Main Prefect Flow Queue GCP Server]
    L[Streamlit Interface]
    M[GCP Cloud Run]
    N[Fetch & Predict Flow]
    O[Train & Deploy Flow]

    A --> B
    B --> C
    C --> D
    D --> H
    H --> E
    E --> F
    F --> G
    G --> D

    N -.- H
    O -.- F
    N -.-> I
    O -.-> I
    I -.-> K
    O -.-> J
    E -.-> L

    classDef GCP style fill:#e1eef6,stroke:#333,stroke-width:3px;
    classDef streamlit style fill:#d9f2d9,stroke:#333,stroke-width:3px;
    classDef prefect style fill:#f5d5e5,stroke:#333,stroke-width:3px;
    classDef flows style fill:#ffe6e6,stroke:#333,stroke-width:3px;

    class I prefect;
    class J,K,M GCP;
    class L streamlit;
    class N,O flows;

```

The approach to solving this problem involves several steps:

1. **Data Collection**: Using the Gmail API to fetch the TLDRs from the different Gmail labels. I'm dumping the data into a CSV file in the `data` folder.

2. **Data Exploration**: Loading, analysing and visualising the data. This will be done in the notebook [notebooks/tld_articles_exploration.ipynb](notebooks/tld_articles_exploration.ipynb).

3. **Model Exploration**: Experimenting with different machine learning models to solve the task at hand. This will be done in the notebook [notebooks/tld_articles_model_exploration.ipynb](notebooks/tld_articles_model_exploration.ipynb) where we will use MLFlow as model experimentation tool.

4. **Model as Service**: Code refactoring and creation of a batch model service.

5. **Model Deployment**: Deploying the model to a production environment where it can make predictions on new, unseen TLDRs.

### Deployment procedure

MLFLOW:

https://www.youtube.com/watch?v=MWfKAgEHsHo

PREFECT:

https://medium.com/@danilo.drobac/7-a-complete-google-cloud-deployment-of-prefect-2-0-32b8e3c2febe

1. `prefect init` to create a new prefect project.

2. `prefect deploy` to deploy flows.

3. `docker build --platform linux/amd64 -t europe-west1-docker.pkg.dev/mlops-389311/email-discriminator/email-discriminator:1 .`

4. `docker push europe-west1-docker.pkg.dev/mlops-389311/email-discriminator/email-discriminator:1`

5. `python flow-deployment.py`

TODO: Automatically refresh GMAIL API token

INTERFACE:

1. Create VM instance in GCP.

2. Execute this in the machine:

```shell
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    python3-dateutil \
    python3-distutils \
    git-all
sudo ln -s /usr/bin/python3 /usr/bin/python
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
PATH="$HOME/.local/bin:$PATH"
export PATH
rm get-pip.py
```
