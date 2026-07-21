import pandas as pd
import statsmodels.api as sm
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

with open("dico.pkl", "rb") as f:
    dico = pickle.load(f)
#========================================================donnee d'entrainement======================================================
df_GK_week = pd.read_csv("gk_training.csv")
df_GK = pd.read_csv("gk_training_year.csv")


X1 = df_GK_week[["minutes","avg_points_5","avg_influence_5","avg_creativity_5","bps","clean_sheets_5","saves_5","goals_conceded_5","penalties_saved_5"]]
X1 = X1.rename(columns={
    "minutes" : "minutes_avg" ,
    "bps" : "bps_5"
})

rows = []

for _, row in df_GK.iterrows():
    n = dico[row["id"]]-5
    
    for _ in range(n):
        rows.append(row.copy())

X2 = pd.DataFrame(rows).reset_index(drop=True)

X2 = X2[[
    "minutes",
    "starts",
    "chance_next",
    "influence",
    "creativity",
    "threat",
    "ict",
    "bps",
    "cost",
    "clean_sheets",
    "goals_conceded",
    "xGC",
    "saves"
]]


X = pd.concat([X1,X2],axis=1)
Y = df_GK_week["points"]
#========================================================linear regression=========================================================

mod1 = linear_model.LinearRegression()
mod1.fit(X,Y)
#============================================================decision tree===========================================================

mod2 = DecisionTreeRegressor(random_state=0)
mod2.fit(X,Y)
#==============================================================random forest=============================================================

mod3 = RandomForestRegressor(n_estimators=10)
mod3.fit(X,Y)
#==========================================================prediction================================================================
df_pred_GK = pd.read_csv("X1_GK.csv")
L =[]

for _,row in df_pred_GK.iterrows():
    d = {}
    l= []
    id = row["id"]
    x1 = pd.DataFrame([row])[["minutes_5","avg_points_5","avg_influence_5","avg_creativity_5","bps_5","clean_sheets_5","saves_5","goals_conceded_5","penalties_saved_5"]]
    x1 = x1.rename(columns={
        "minutes_5" : "minutes_avg"
    })
    x2 = df_GK[df_GK["id"] == id][[
    "minutes",
    "starts",
    "chance_next",
    "influence",
    "creativity",
    "threat",
    "ict",
    "bps",
    "cost",
    "clean_sheets",
    "goals_conceded",
    "xGC",
    "saves"
    ]]
    x = pd.concat(
    [
     x1.reset_index(drop=True),
     x2.reset_index(drop=True)],
    axis=1
    )
    d['id'] = id
    d['pre_linear'] = mod1.predict(x)[0]
    d['pre_tree'] = mod2.predict(x)[0]
    d['pre_random_forest'] = mod3.predict(x)[0]
    L.append(d)    
    
pd.DataFrame(L).to_csv("prediction_GK.csv")