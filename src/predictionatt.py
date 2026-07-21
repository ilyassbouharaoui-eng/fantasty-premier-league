import pandas as pd
import pickle
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

with open("../data/dico.pkl", "rb") as f:
    dico = pickle.load(f)
#========================================================donnee d'entrainement======================================================
df_att_week = pd.read_csv("../data/att_training.csv")
df_att = pd.read_csv("../data/att_training_year.csv")


X1 = df_att_week[["minutes","avg_points_5","avg_influence_5","avg_creativity_5","bps","goals_scored_5","assists_5"]]
X1 = X1.rename(columns={
    "minutes" : "minutes_avg" ,
    "bps" : "bps_5"
})

rows = []

for _, row in df_att.iterrows():
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
    "goals",
    "assists",
    "xG",
    "xA",
    "xGI"
]]


X = pd.concat([X1,X2],axis=1)
Y = df_att_week["points"]
#========================================================linear regression=========================================================

mod1 = linear_model.LinearRegression()
mod1.fit(X,Y)

#==============================================================random forest=============================================================

mod2 = RandomForestRegressor(n_estimators=10)
mod2.fit(X,Y)
#==========================================================prediction================================================================
df_pred_att = pd.read_csv("../data/X1_att.csv")
L =[]

for _,row in df_pred_att.iterrows():
    d = {}
    l= []
    id = row["id"]
    x1 = pd.DataFrame([row])[["minutes_5","avg_points_5","avg_influence_5","avg_creativity_5","bps_5","goals_scored_5","assists_5"]]
    x1 = x1.rename(columns={
        "minutes_5" : "minutes_avg"
    })
    x2 = df_att[df_att["id"] == id][[
    "minutes",
    "starts",
    "chance_next",
    "influence",
    "creativity",
    "threat",
    "ict",
    "bps",
    "cost",
    "goals",
    "assists",
    "xG",
    "xA",
    "xGI"
    ]]
    x = pd.concat(
    [
     x1.reset_index(drop=True),
     x2.reset_index(drop=True)],
    axis=1
    )
    d['id'] = id
    d['cost'] = df_att[df_att["id"] == id]['cost'].iloc[0]
    d['pre_linear'] = mod1.predict(x)[0]
    d['pre_random_forest'] = mod2.predict(x)[0]
    L.append(d)    
    
pd.DataFrame(L).to_csv("../prediction/prediction_att.csv")