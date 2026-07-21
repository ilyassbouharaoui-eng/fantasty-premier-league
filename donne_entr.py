import requests
import pandas as pd
import pickle
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
data = requests.get(url).json()



rows_GK = []
rows_def = []
rows_att_mid = []

for d in data['elements']:
    row = {
         "id": d["id"],
        "name": d["web_name"],
        "position": d["element_type"],

        "minutes": d["minutes"],
        "starts": d["starts"],
        "chance_next": d["chance_of_playing_next_round"] or 0,

        "influence": float(d["influence"]),
        "creativity": float(d["creativity"]),
        "threat": float(d["threat"]),
        "ict": float(d["ict_index"]),
        "bps": d["bps"],
        "cost":d["now_cost"]
    }

    # attackers + mids
    if d["element_type"] in [3, 4]:
        row.update({
            "goals": d["goals_scored"],
            "assists": d["assists"],
            "xG": float(d["expected_goals"]),
            "xA": float(d["expected_assists"]),
            "xGI": float(d["expected_goal_involvements"]),
        })
        rows_att_mid.append(row)

    # defenders 
    if d["element_type"] == 2:
        row.update({
            "clean_sheets": d["clean_sheets"],
            "goals_conceded": d["goals_conceded"],
            "xGC": float(d["expected_goals_conceded"]),
        })
        rows_def.append(row)

    # GK 
    if d["element_type"] == 1:
        row.update({
            "clean_sheets": d["clean_sheets"],
            "goals_conceded": d["goals_conceded"],
            "xGC": float(d["expected_goals_conceded"]),
            "saves": d["saves"],
        })
        rows_GK.append(row)

    


df_GK = pd.DataFrame(rows_GK)
df_def = pd.DataFrame(rows_def)
df_att = pd.DataFrame(rows_att_mid)

df_GK.to_csv("gk_training_year.csv", index=False)
df_def.to_csv("def_training_year.csv", index=False)
df_att.to_csv("att_training_year.csv", index=False)


rows_GK_week = []
rows_def_week = []
rows_att_mid_week = []

all_histories = {}

for d in data["elements"]:
    id = d["id"]
    url_week = f"https://fantasy.premierleague.com/api/element-summary/{id}/" 
    all_histories[id] = requests.get(url_week).json()["history"]

#le dict qu on va definir ici va nous servir dans les donnes de l entrainement voir fichier prediction.py( len(X1) doit etre = len(X2))
dico = {}
for d in data['elements']:
    id = d['id'] 
    
    weeks = all_histories[id]
    dico[id] = len(weeks)
    for i in range(5,len(weeks)):
        point = 0 
        influence = 0
        creativity = 0
        bps = 0 
        saves = 0
        goals_conceded = 0
        penalties_saved = 0
        clean_sheets = 0
        tackles = 0
        defensive_contribution = 0 
        minutes = 0
        goals_scored = 0
        assists = 0
        for j in range(i-5,i):
            minutes += weeks[j]['minutes']
            point += weeks[j]['total_points']
            influence += float(weeks[j]['influence'])
            creativity += float(weeks[j]['creativity'])
            bps += weeks[j]['bps']
            saves += weeks[j]['saves']
            goals_conceded += weeks[j]['goals_conceded']
            penalties_saved += weeks[j]['penalties_saved']
            clean_sheets += weeks[j]['clean_sheets']
            tackles += weeks[j]['tackles']
            defensive_contribution += weeks[j]['defensive_contribution']
            goals_scored += weeks[j]['goals_scored']
            assists += weeks[j]['assists']

        row = {
        "id" : id ,
        "minutes" : minutes/5,
        "position": d["element_type"],
        "GW" : i + 1,
        "avg_points_5" : point/5,
        "points" : weeks[i]['total_points'],
        "avg_influence_5" : influence/5,
        "avg_creativity_5" : creativity/5,
        "bps" : bps/5,
        
        }
        # GK 
        if d["element_type"] == 1:
            row.update({
                "clean_sheets_5" : clean_sheets,
                "saves_5" : saves,
                "goals_conceded_5" : goals_conceded,
                "penalties_saved_5" : penalties_saved,

            })
            rows_GK_week.append(row)
        # defenders 
        if d["element_type"] == 2:
            row.update({  
                "clean_sheets_5" : clean_sheets,
                "goals_conceded_5" : goals_conceded,
                "tackles_5" : tackles,
                "defensive_contribution_5" : defensive_contribution,
            })
            rows_def_week.append(row)
        # attackers + mids
        if d["element_type"] in [3, 4]:
            row.update({  
                "goals_scored_5" : goals_scored,
                "assists_5" : assists,
            })  
            rows_att_mid_week.append(row)   

df_GK_week = pd.DataFrame(rows_GK_week)
df_def_week = pd.DataFrame(rows_def_week)
df_att_week = pd.DataFrame(rows_att_mid_week)    

df_GK_week.to_csv("gk_training.csv", index=False)
df_def_week.to_csv("def_training.csv", index=False)
df_att_week.to_csv("att_training.csv", index=False)

with open("dico.pkl", "wb") as f:
    pickle.dump(dico, f)

#===============================================donne pour la prediction====================================================
X1_GK = []
X1_def = []
X1_att = []

for d in data['elements']:
    id = d['id'] 
    weeks = all_histories[id]
    point = 0 
    influence = 0
    creativity = 0
    bps = 0 
    saves = 0
    goals_conceded = 0
    penalties_saved = 0
    clean_sheets = 0
    tackles = 0
    defensive_contribution = 0 
    minutes = 0
    goals_scored = 0
    assists = 0
    if len(weeks) < 5 :
        continue
    for j in range(len(weeks)-5,len(weeks)):
        minutes += weeks[j]['minutes']
        point += weeks[j]['total_points']
        influence += float(weeks[j]['influence'])
        creativity += float(weeks[j]['creativity'])
        bps += weeks[j]['bps']
        saves += weeks[j]['saves']
        goals_conceded += weeks[j]['goals_conceded']
        penalties_saved += weeks[j]['penalties_saved']
        clean_sheets += weeks[j]['clean_sheets']
        tackles += weeks[j]['tackles']
        defensive_contribution += weeks[j]['defensive_contribution']
        goals_scored += weeks[j]['goals_scored']
        assists += weeks[j]['assists']
    row = {
    "id" : id ,
    "minutes_5" : minutes/5,
    "position": d["element_type"],
    "avg_points_5" : point/5,
    "avg_influence_5" : influence/5,
    "avg_creativity_5" : creativity/5,
    "bps_5" : bps/5,
        
    }
    # GK 
    if d["element_type"] == 1:
        row.update({
            "clean_sheets_5" : clean_sheets,
            "saves_5" : saves,
            "goals_conceded_5" : goals_conceded,
            "penalties_saved_5" : penalties_saved,

        })
        X1_GK.append(row)
    # defenders 
    if d["element_type"] == 2:
        row.update({  
            "clean_sheets_5" : clean_sheets,
            "goals_conceded_5" : goals_conceded,
            "tackles_5" : tackles,
            "defensive_contribution_5" : defensive_contribution,
        })
        X1_def.append(row)
    # attackers + mids
    if d["element_type"] in [3, 4]:
        row.update({  
            "goals_scored_5" : goals_scored,
            "assists_5" : assists,
        })  
        X1_att.append(row)

pd.DataFrame(X1_GK).to_csv("X1_GK.csv",index=False)  
pd.DataFrame(X1_def).to_csv("X1_def.csv",index=False)  
pd.DataFrame(X1_att).to_csv("X1_att.csv",index=False)  
     

