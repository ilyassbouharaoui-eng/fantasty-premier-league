import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def split_by_gw(df, test_gw_start):
    """
    train = toutes les lignes avec GW < test_gw_start
    test = toutes les lignes avec GW >= test_gw_start
    """
    train_df = df[df["GW"] < test_gw_start].reset_index(drop=True)
    test_df = df[df["GW"] >= test_gw_start].reset_index(drop=True)
    return train_df, test_df


def train_and_eval(df_full,feature_cols):
    train_df, test_df = split_by_gw(df_full, test_gw_start=30)
    X = train_df[feature_cols].rename(columns={"minutes":"minutes_avg","bps":"bps_5"})
    Y = train_df["points"]

    X_test = test_df[feature_cols].rename(columns={"minutes":"minutes_avg","bps":"bps_5"})
    Y_test = test_df["points"]
    mod1 = linear_model.LinearRegression().fit(X, Y)
    mod2 = RandomForestRegressor(n_estimators=10).fit(X, Y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    mod3 = MLPRegressor(hidden_layer_sizes=(16,), max_iter=1000).fit(X_train_scaled, Y)
    
    return {
        "linear": mod1.score(X_test, Y_test),
        "rf": mod2.score(X_test, Y_test),
        "nn": mod3.score(X_test_scaled, Y_test),
    }, (mod1, mod2, mod3, scaler)

def predict(df_pred,df,feature_cols,mod1,mod2,mod3,scaler):
    L =[]
    for _,row in df_pred.iterrows():
        d = {}
        l= []
        id = row["id"]
        x = pd.DataFrame([row])[feature_cols].rename(columns={
                "minutes_5" : "minutes_avg"
            })
        d['id'] = id
        d['cost'] = df[df["id"] == id]['cost'].iloc[0]
        d['pre_linear'] = mod1.predict(x)[0]
        d['pre_random_forest'] = mod2.predict(x)[0]
        x = scaler.transform(x)
        d['neural_network'] = mod3.predict(x)[0]
        L.append(d)    
        


##GK
feature_cols_Gk = ["minutes","avg_points_5","avg_influence_5","avg_creativity_5",
                 "bps","clean_sheets_5","saves_5","goals_conceded_5","penalties_saved_5"]
feature_pred_Gk = ["minutes_5","avg_points_5","avg_influence_5","avg_creativity_5",
                 "bps_5","clean_sheets_5","saves_5","goals_conceded_5","penalties_saved_5"]
df_GK_week = pd.read_csv("../data/gk_training.csv")
df_pred_GK = pd.read_csv("../data/X1_GK.csv")
df_GK = pd.read_csv("../data/gk_training_year.csv")
R2,(mod1,mod2,mod3,scaler) = train_and_eval(df_GK_week,feature_cols_Gk)
L = predict(df_pred_GK,df_GK,feature_pred_Gk,mod1,mod2,mod3,scaler)
print("GK : ",R2)
pd.DataFrame(L).to_csv("../prediction/prediction_GK.csv")

##def
feature_cols_def = ["minutes","avg_points_5","avg_influence_5","avg_creativity_5",
                "bps","clean_sheets_5","goals_conceded_5","tackles_5","defensive_contribution_5"]
feature_pred_def = ["minutes_5","avg_points_5","avg_influence_5","avg_creativity_5",
                "bps_5","clean_sheets_5","goals_conceded_5","tackles_5","defensive_contribution_5"]
df_def_week = pd.read_csv("../data/def_training.csv")
df_pred_def = pd.read_csv("../data/X1_def.csv")
df_def = pd.read_csv("../data/def_training_year.csv")
R2,(mod1,mod2,mod3,scaler) = train_and_eval(df_def_week,feature_cols_def)
L = predict(df_pred_def,df_def,feature_pred_def,mod1,mod2,mod3,scaler)
print("DEF : ",R2)
pd.DataFrame(L).to_csv("../prediction/prediction_def.csv")

##att
feature_cols_att = ["minutes","avg_points_5","avg_influence_5","avg_creativity_5",
                    "bps","goals_scored_5","assists_5"]
feature_pred_att = ["minutes_5","avg_points_5","avg_influence_5","avg_creativity_5",
                    "bps_5","goals_scored_5","assists_5"]
df_att_week = pd.read_csv("../data/att_training.csv")
df_att = pd.read_csv("../data/att_training_year.csv")
df_pred_att = pd.read_csv("../data/X1_att.csv")
R2,(mod1,mod2,mod3,scaler) = train_and_eval(df_att_week,feature_cols_att)
L = predict(df_pred_att,df_att,feature_pred_att,mod1,mod2,mod3,scaler)
print("ATT : ",R2)
pd.DataFrame(L).to_csv("../prediction/prediction_att.csv")