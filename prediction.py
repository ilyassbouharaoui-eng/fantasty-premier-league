import pandas as pd
import statsmodels.api as sm
import numpy as np

import pickle

with open("dico.pkl", "rb") as f:
    dico = pickle.load(f)

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
X = sm.add_constant(X)
Y = df_GK_week["points"]
#==============================================lineaire regression=========================================================

model = sm.OLS(Y,X).fit()
print(model.summary())

#==============================================decision tree===============================================================

X = pd.concat([X1,X2],axis=1)
X = X.to_numpy()
Y = Y.to_numpy()

def moyenne_s(L):
    l=[]
    for i in range(len(L)-1):
        s = (L[i]+L[i+1])/2
        l.append(s)
    return l    

def split(L,s):
    l1=[]
    l2=[]
    for i in range(len(L)) :
        if L[i] < s:
            l1.append(i)
        else:
            l2.append(i)    
    return l1,l2        


def erreur(R , Y):

    if len(R) == 0:
        return np.inf

    y_1 = sum(Y[i] for i in R) / len(R)

    s1 = sum((Y[i] - y_1) ** 2 for i in R)

    return s1 

def divide(x,y):
    d  = {}
    for j in range(22):
        S = moyenne_s(sorted(x[:, j]))
        for s in S :
            R1 = split(x[:, j],s)[0]
            R2 = split(x[:, j],s)[1]
            err = erreur(R1,y) + erreur(R2,y)
            d[(j,s)] = err
    return min(d,key = d.get)        



def build_tree(X, Y):

    if len(Y) < 5:
        return {"value": np.mean(Y)}

    j, s = divide(X, Y)

    R1, R2 = split(X[:, j], s)

    left = build_tree(X[R1,:], Y[R1])
    right = build_tree(X[R2,:], Y[R2])

    return {
        "feature": j,
        "threshold": s,
        "left": left,
        "right": right
    }

def predict(tree, x):

    if "value" in tree:
        return tree["value"]

    if x[tree["feature"]] < tree["threshold"]:
        return predict(tree["left"], x)
    else:
        return predict(tree["right"], x)