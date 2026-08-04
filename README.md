#  Fantasy Premier League Points Predictor

A Machine Learning project that predicts the Fantasy Premier League (FPL) points of every player for the next Gameweek.

The project collects data directly from the official Fantasy Premier League API, builds training datasets from historical Gameweeks, trains several regression models and predicts the expected score of each player.

---

# Project Overview

The prediction is based on the player's performances during the previous **5 Gameweeks** together with season statistics.

Different feature sets are created depending on the player's position:

- Goalkeepers
- Defenders
- Midfielders & Forwards

The final objective of the project is to build an intelligent Fantasy Premier League assistant capable of recommending the optimal team every Gameweek.

---

# Implemented Models

Currently implemented regression models:

- Linear Regression
- Random Forest Regressor
- Neural Network (MLP)

Planned models:

- XGBoost
- LightGBM
- CatBoost

---

# Project Structure

```
Fantasy-Premier-League/
│
├── src/
│   │
│   ├── donne_entr.py
│   ├── prediction.py
|
├── data/
│   │
│   ├── gk_training.csv
│   ├── def_training.csv
│   ├── att_training.csv
│   │
│   ├── gk_training_year.csv
│   ├── def_training_year.csv
│   ├── att_training_year.csv
│   │
│   ├── X1_GK.csv
│   ├── X1_def.csv
│   └── X1_att.csv
│
├── prediction/
│   │
│   ├── prediction_GK.csv
│   ├── prediction_att.csv
│   ├── prediction_def.csv
│   └── ...
│
└── README.md
```

---

# Data Pipeline

The project follows the pipeline below.

```
Fantasy Premier League API
            │
            ▼
Download current player statistics
            │
            ▼
Download every player's Gameweek history
            │
            ▼
Feature Engineering
(last 5 Gameweeks)
            │
            ▼
Training datasets
            │
            ▼
Train ML models
            │
            ▼
Predict next Gameweek points
            │
            ▼
Prediction CSV files
```

---

# Dataset Generation

The training datasets are automatically generated from the FPL API.

For every player, the project computes rolling statistics over the previous five Gameweeks.

Examples of generated features:

## Common features

- Average points
- Average minutes played
- Average influence
- Average creativity
- Average BPS

## Goalkeepers

- Saves
- Clean sheets
- Goals conceded
- Penalties saved

## Defenders

- Tackles
- Defensive contributions
- Goals conceded
- Clean sheets

## Midfielders & Forwards

- Goals scored
- Assists

Season statistics such as ICT Index, Expected Goals (xG), Expected Assists (xA), Expected Goal Involvement (xGI), player cost and availability are also collected.

---

# Data Folder

The **data/** directory contains the datasets used by the machine learning models.

### Training datasets

These files are used to train the models.

- gk_training.csv
- def_training.csv
- att_training.csv

### Season statistics

These files contain season-wide player statistics.

- gk_training_year.csv
- def_training_year.csv
- att_training_year.csv

### Prediction datasets

The X1 files contain the features of the latest five Gameweeks and are used to predict the next Gameweek.

- X1_GK.csv
- X1_def.csv
- X1_att.csv

---

# Prediction Folder

The **prediction/** directory stores the final prediction results.

Each CSV contains

- player id
- player name
- predicted points
- player cost

These files can later be used for team optimization.

---

# Current Status

 Automatic data collection from the FPL API

 Automatic dataset generation

 Feature engineering

 Linear Regression

 Random Forest

 Neural Network

 Next Gameweek prediction

---

# Future Work

The project is still under development.

Planned improvements include:

- XGBoost implementation
- Hyperparameter optimization
- Cross-validation
- Feature selection
- Better neural network architecture
- Automatic squad optimization under the £100M budget
- Captain recommendation
- Transfer recommendation
- Fixture Difficulty Rating (FDR)
- Injury prediction

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Requests
- Fantasy Premier League API

---

# Author

**Ilyass Bouharaoui**

Engineering Student at ENSIMAG
Interested in Machine Learning, Artificial Intelligence and Data Science.
