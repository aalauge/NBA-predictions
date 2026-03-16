#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 🐍 OPERATION KOBRA – Automatisches NBA Vorhersagesystem
# Läuft täglich automatisch und generiert Vorhersagen für heutige Spiele

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, date
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Konfiguration
API_KEY = "dein_balldontlie_key_hier"
HEADERS = {"Authorization": API_KEY}
HEUTE = date.today().isoformat()

print(f"🐍 Operation Kobra gestartet – {HEUTE}")


# In[3]:


# 🐍 OPERATION KOBRA – Automatisches NBA Vorhersagesystem

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, date
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ⬇️ HIER deinen API Key eintragen
API_KEY = "98ee804c-c05f-4eb3-88cd-3b7d42ae474e"

# Ab hier nichts ändern
HEADERS = {"Authorization": API_KEY}
HEUTE = date.today().isoformat()

# Verbindung testen
test = requests.get("https://api.balldontlie.io/v1/teams", headers=HEADERS)
if test.status_code == 200:
    print(f"✅ API Verbindung erfolgreich!")
else:
    print(f"❌ Fehler: {test.status_code} – API Key prüfen!")

print(f"🐍 Operation Kobra gestartet – {HEUTE}")


# In[4]:


# 📊 Schritt 1: Historische Spieldaten laden
def lade_saison(saison, max_versuche=3):
    """Lädt alle Spiele einer NBA Saison von balldontlie.io"""
    alle_spiele = []
    cursor = None

    while True:
        params = {
            "seasons[]": saison,
            "per_page": 100,
            "postseason": "false"
        }
        if cursor:
            params["cursor"] = cursor

        for versuch in range(max_versuche):
            response = requests.get(
                "https://api.balldontlie.io/v1/games",
                headers=HEADERS,
                params=params
            )
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                print(f"Rate limit – warte 30 Sekunden...")
                time.sleep(30)
            else:
                print(f"Fehler: {response.status_code}")
                return pd.DataFrame()

        daten = response.json()
        alle_spiele.extend(daten["data"])
        cursor = daten["meta"].get("next_cursor")
        if not cursor:
            break
        time.sleep(15)

    # In DataFrame umwandeln
    rows = []
    for spiel in alle_spiele:
        if spiel["status"] != "Final":
            continue
        rows.append({
            "GAME_DATE": spiel["date"],
            "HOME_TEAM_ID": spiel["home_team"]["id"],
            "AWAY_TEAM_ID": spiel["visitor_team"]["id"],
            "HOME_TEAM": spiel["home_team"]["full_name"],
            "AWAY_TEAM": spiel["visitor_team"]["full_name"],
            "PTS_HOME": spiel["home_team_score"],
            "PTS_AWAY": spiel["visitor_team_score"],
            "SEASON": saison,
            "HOME_WIN": 1 if spiel["home_team_score"] > spiel["visitor_team_score"] else 0
        })

    df = pd.DataFrame(rows)
    print(f"✅ Saison {saison}: {len(df)} Spiele geladen")
    return df

# Saisons laden – 2022 bis 2025
print("Lade historische Daten...")
saisons_dfs = []
for saison in [2022, 2023, 2024, 2025]:
    df = lade_saison(saison)
    if len(df) > 0:
        saisons_dfs.append(df)

daten = pd.concat(saisons_dfs, ignore_index=True)
daten["GAME_DATE"] = pd.to_datetime(daten["GAME_DATE"])
daten = daten.sort_values("GAME_DATE").reset_index(drop=True)

print(f"\n📊 Gesamt: {len(daten)} Spiele")
print(f"Zeitraum: {daten['GAME_DATE'].min().date()} bis {daten['GAME_DATE'].max().date()}")


# In[5]:


# 📊 Schritt 2: Features berechnen
print("Berechne Features...")

daten["HomeWinRate"] = 0.0
daten["AwayWinRate"] = 0.0
daten["HomePtsScored"] = 0.0
daten["HomePtsConceded"] = 0.0
daten["AwayPtsScored"] = 0.0
daten["AwayPtsConceded"] = 0.0
daten["HomeElo"] = 0.0
daten["AwayElo"] = 0.0
daten["EloDiff"] = 0.0

elo = {}

for i, row in daten.iterrows():
    home = row["HOME_TEAM_ID"]
    away = row["AWAY_TEAM_ID"]

    if home not in elo: elo[home] = 1500
    if away not in elo: elo[away] = 1500

    # Elo vor dem Spiel speichern
    daten.at[i, "HomeElo"] = elo[home]
    daten.at[i, "AwayElo"] = elo[away]
    daten.at[i, "EloDiff"] = elo[home] - elo[away]

    # Form letzte 5 Spiele
    for team_id, prefix in [(home, "Home"), (away, "Away")]:
        vergangene = daten[
            ((daten["HOME_TEAM_ID"] == team_id) |
             (daten["AWAY_TEAM_ID"] == team_id)) &
            (daten["GAME_DATE"] < row["GAME_DATE"])
        ].tail(5)

        if len(vergangene) < 3:
            continue

        siege = 0
        pts_scored = []
        pts_conceded = []

        for _, spiel in vergangene.iterrows():
            if spiel["HOME_TEAM_ID"] == team_id:
                pts_scored.append(spiel["PTS_HOME"])
                pts_conceded.append(spiel["PTS_AWAY"])
                if spiel["HOME_WIN"] == 1: siege += 1
            else:
                pts_scored.append(spiel["PTS_AWAY"])
                pts_conceded.append(spiel["PTS_HOME"])
                if spiel["HOME_WIN"] == 0: siege += 1

        daten.at[i, f"{prefix}WinRate"] = siege / len(vergangene)
        daten.at[i, f"{prefix}PtsScored"] = sum(pts_scored) / len(pts_scored)
        daten.at[i, f"{prefix}PtsConceded"] = sum(pts_conceded) / len(pts_conceded)

    # Elo aktualisieren
    erwartung_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
    if row["HOME_WIN"] == 1:
        ergebnis_home, ergebnis_away = 1, 0
    else:
        ergebnis_home, ergebnis_away = 0, 1

    elo[home] += 32 * (ergebnis_home - erwartung_home)
    elo[away] += 32 * (ergebnis_away - (1 - erwartung_home))

# Spiele ohne Features entfernen
daten = daten[daten["HomeWinRate"] > 0].reset_index(drop=True)
print(f"✅ Features berechnet: {len(daten)} Spiele")


# In[6]:


# 🤖 Schritt 3: Modell trainieren
print("Trainiere Modell...")

FEATURES = [
    "HomeWinRate", "AwayWinRate",
    "HomePtsScored", "HomePtsConceded",
    "AwayPtsScored", "AwayPtsConceded",
    "HomeElo", "AwayElo", "EloDiff"
]

# Zeitbasierter Split – trainiere auf allem außer aktueller Saison
train = daten[daten["SEASON"] != 2025]
test = daten[daten["SEASON"] == 2025]

modell = LogisticRegression(max_iter=2000)
modell.fit(train[FEATURES], train["HOME_WIN"])

vorhersagen = modell.predict(test[FEATURES])
genauigkeit = accuracy_score(test["HOME_WIN"], vorhersagen)

print(f"✅ Modell trainiert!")
print(f"📊 Trainingsspiele: {len(train)}")
print(f"📊 Testspiele: {len(test)}")
print(f"🎯 Genauigkeit: {genauigkeit:.1%}")


# In[7]:


# 🏀 Schritt 4: Heutige Spiele vorhersagen
print(f"Lade Spiele für heute ({HEUTE})...")

response = requests.get(
    "https://api.balldontlie.io/v1/games",
    headers=HEADERS,
    params={"dates[]": HEUTE, "per_page": 100}
)

heute_spiele = response.json()["data"]
print(f"Spiele heute: {len(heute_spiele)}")

# Features für heutige Spiele berechnen
def get_team_stats(team_id):
    vergangene = daten[
        ((daten["HOME_TEAM_ID"] == team_id) |
         (daten["AWAY_TEAM_ID"] == team_id))
    ].tail(5)

    if len(vergangene) < 3:
        return None

    siege = 0
    pts_scored = []
    pts_conceded = []

    for _, spiel in vergangene.iterrows():
        if spiel["HOME_TEAM_ID"] == team_id:
            pts_scored.append(spiel["PTS_HOME"])
            pts_conceded.append(spiel["PTS_AWAY"])
            if spiel["HOME_WIN"] == 1: siege += 1
        else:
            pts_scored.append(spiel["PTS_AWAY"])
            pts_conceded.append(spiel["PTS_HOME"])
            if spiel["HOME_WIN"] == 0: siege += 1

    return {
        "WinRate": siege / len(vergangene),
        "PtsScored": sum(pts_scored) / len(pts_scored),
        "PtsConceded": sum(pts_conceded) / len(pts_conceded),
        "Elo": elo.get(team_id, 1500)
    }

# Vorhersagen generieren
vorhersage_rows = []
for spiel in heute_spiele:
    heim_id = spiel["home_team"]["id"]
    auswaerts_id = spiel["visitor_team"]["id"]

    heim_stats = get_team_stats(heim_id)
    auswaerts_stats = get_team_stats(auswaerts_id)

    if not heim_stats or not auswaerts_stats:
        continue

    vorhersage_rows.append({
        "Datum": HEUTE,
        "Heimteam": spiel["home_team"]["full_name"],
        "Auswärtsteam": spiel["visitor_team"]["full_name"],
        "HomeWinRate": heim_stats["WinRate"],
        "AwayWinRate": auswaerts_stats["WinRate"],
        "HomePtsScored": heim_stats["PtsScored"],
        "HomePtsConceded": heim_stats["PtsConceded"],
        "AwayPtsScored": auswaerts_stats["PtsScored"],
        "AwayPtsConceded": auswaerts_stats["PtsConceded"],
        "HomeElo": heim_stats["Elo"],
        "AwayElo": auswaerts_stats["Elo"],
        "EloDiff": heim_stats["Elo"] - auswaerts_stats["Elo"]
    })

heute_df = pd.DataFrame(vorhersage_rows)

# Modell anwenden
vorhersagen_heute = modell.predict(heute_df[FEATURES])
wahrscheinlichkeiten = modell.predict_proba(heute_df[FEATURES])

# Ausgabe
print("\n🏀 NBA Vorhersagen – Heute")
print("=" * 60)

for idx, row in heute_df.iterrows():
    prob_home = wahrscheinlichkeiten[idx][1]
    prob_away = wahrscheinlichkeiten[idx][0]
    tipp = row["Heimteam"] if vorhersagen_heute[idx] == 1 else row["Auswärtsteam"]
    konfidenz = max(prob_home, prob_away)

    print(f"\n{row['Heimteam']} vs {row['Auswärtsteam']}")
    print(f"  🏆 Tipp: {tipp} ({konfidenz:.0%})")
    print(f"  Heim: {prob_home:.0%} | Auswärts: {prob_away:.0%}")


# In[8]:


# 💾 Schritt 5: Ergebnisse speichern
heute_df["Tipp"] = [
    row["Heimteam"] if vorhersagen_heute[idx] == 1 else row["Auswärtsteam"]
    for idx, row in heute_df.iterrows()
]
heute_df["Tipp_Prob"] = [max(w[0], w[1]) for w in wahrscheinlichkeiten]
heute_df["Echtes_Ergebnis"] = ""  # Später ausfüllen
heute_df["Richtig"] = ""          # Später ausfüllen

# Nur relevante Spalten speichern
ergebnis = heute_df[[
    "Datum", "Heimteam", "Auswärtsteam",
    "Tipp", "Tipp_Prob",
    "HomeElo", "AwayElo",
    "Echtes_Ergebnis", "Richtig"
]]

# CSV speichern
dateiname = f"predictions_{HEUTE}.csv"
ergebnis.to_csv(os.path.expanduser(f"~/{dateiname}"), index=False)
print(f"✅ Gespeichert: ~/{dateiname}")
print(ergebnis[["Heimteam", "Auswärtsteam", "Tipp", "Tipp_Prob"]].to_string(index=False))


# In[ ]:




