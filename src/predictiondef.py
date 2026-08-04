import pandas as pd
import pickle
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

with open("../data/dico.pkl", "rb") as f:
    dico = pickle.load(f)
#========================================================donnee d'entrainement======================================================
df_def_week = pd.read_csv("../data/def_training.csv")
df_def = pd.read_csv("../data/def_training_year.csv")


X1 = df_def_week[["minutes","avg_points_5","avg_influence_5","avg_creativity_5","bps","clean_sheets_5","goals_conceded_5","tackles_5","defensive_contribution_5"]]
X1 = X1.rename(columns={
    "minutes" : "minutes_avg" ,
    "bps" : "bps_5"
})

rows = []

for _, row in df_def.iterrows():
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
    "xGC"
]]


X = pd.concat([X1,X2],axis=1)
Y = df_def_week["points"]

X,X_test,Y,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
#========================================================linear regression=========================================================

mod1 = linear_model.LinearRegression()
mod1.fit(X,Y)
print(mod1.score(X_test,Y_test))
#==============================================================random forest=============================================================

mod2 = RandomForestRegressor(n_estimators=10)
mod2.fit(X,Y)
print(mod2.score(X_test,Y_test))
#=============================================================neural network ========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
X_test_scaled = scaler.fit_transform(X_test) 
mod3 = MLPRegressor(hidden_layer_sizes=(16,),max_iter = 1000)
mod3.fit(X_scaled,Y) 
print(mod3.score(X_test_scaled,Y_test))
#==========================================================prediction================================================================
df_pred_GK = pd.read_csv("../data/X1_def.csv")
L =[]

for _,row in df_pred_GK.iterrows():
    d = {}
    l= []
    id = row["id"]
    x1 = pd.DataFrame([row])[["minutes_5","avg_points_5","avg_influence_5","avg_creativity_5","bps_5","clean_sheets_5","goals_conceded_5","tackles_5","defensive_contribution_5"]]
    x1 = x1.rename(columns={
        "minutes_5" : "minutes_avg"
    })
    x2 = df_def[df_def["id"] == id][[
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
    "xGC"
    ]]
    x = pd.concat(
    [
     x1.reset_index(drop=True),
     x2.reset_index(drop=True)],
    axis=1
    )
    d['id'] = id
    d['cost'] = df_def[df_def["id"] == id]['cost'].iloc[0]
    d['pre_linear'] = mod1.predict(x)[0]
    d['pre_random_forest'] = mod2.predict(x)[0]
    x = scaler.transform(x)
    d['neural_network'] = mod3.predict(x)[0]
    L.append(d)    
    
pd.DataFrame(L).to_csv("../prediction/prediction_def.csv")