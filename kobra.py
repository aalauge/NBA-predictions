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


# In[11]:


# 🔄 Schritt 5: Back-to-Back Feature berechnen
print("Berechne Back-to-Back Spiele...")

# Für jedes Spiel prüfen ob das Team gestern auch gespielt hat
daten["Home_B2B"] = 0
daten["Away_B2B"] = 0

for i, row in daten.iterrows():
    spieldatum = row["GAME_DATE"]
    gestern = spieldatum - pd.Timedelta(days=1)

    # Hat Heimteam gestern gespielt?
    heim_gestern = daten[
        ((daten["HOME_TEAM_ID"] == row["HOME_TEAM_ID"]) |
         (daten["AWAY_TEAM_ID"] == row["HOME_TEAM_ID"])) &
        (daten["GAME_DATE"] == gestern)
    ]
    if len(heim_gestern) > 0:
        daten.at[i, "Home_B2B"] = 1

    # Hat Auswärtsteam gestern gespielt?
    auswaerts_gestern = daten[
        ((daten["HOME_TEAM_ID"] == row["AWAY_TEAM_ID"]) |
         (daten["AWAY_TEAM_ID"] == row["AWAY_TEAM_ID"])) &
        (daten["GAME_DATE"] == gestern)
    ]
    if len(auswaerts_gestern) > 0:
        daten.at[i, "Away_B2B"] = 1

# Statistik anzeigen
b2b_spiele = daten[daten["Home_B2B"] == 1]
b2b_heimsiege = b2b_spiele["HOME_WIN"].mean()

normal_spiele = daten[daten["Home_B2B"] == 0]
normal_heimsiege = normal_spiele["HOME_WIN"].mean()

print(f"Heimsiege normal:      {normal_heimsiege:.1%}")
print(f"Heimsiege bei B2B:     {b2b_heimsiege:.1%}")
print(f"\nAuswärtsteam B2B Spiele: {daten['Away_B2B'].sum()}")
print(f"Heimteam B2B Spiele:     {daten['Home_B2B'].sum()}")


# In[12]:


# 🤖 Modell mit Back-to-Back Feature testen
FEATURES_B2B = [
    "HomeWinRate", "AwayWinRate",
    "HomePtsScored", "HomePtsConceded",
    "AwayPtsScored", "AwayPtsConceded",
    "HomeElo", "AwayElo", "EloDiff",
    "Home_B2B", "Away_B2B"  # NEU!
]

train_b2b = daten[daten["SEASON"] != 2025]
test_b2b = daten[daten["SEASON"] == 2025]

modell_b2b = LogisticRegression(max_iter=2000)
modell_b2b.fit(train_b2b[FEATURES_B2B], train_b2b["HOME_WIN"])

vorhersagen_b2b = modell_b2b.predict(test_b2b[FEATURES_B2B])
genauigkeit_b2b = accuracy_score(test_b2b["HOME_WIN"], vorhersagen_b2b)

print(f"Modell ohne B2B: 65.2%")
print(f"Modell mit B2B:  {genauigkeit_b2b:.1%}")


# In[13]:


# B2B Vorteil/Nachteil als kombiniertes Feature
daten["B2B_Advantage"] = daten["Away_B2B"] - daten["Home_B2B"]
# +1 = Auswärtsteam müde, Heimteam frisch → Heimvorteil
# -1 = Heimteam müde, Auswärtsteam frisch → Auswärtsvorteil
#  0 = beide gleich

FEATURES_B2B_V2 = [
    "HomeWinRate", "AwayWinRate",
    "HomePtsScored", "HomePtsConceded",
    "AwayPtsScored", "AwayPtsConceded",
    "HomeElo", "AwayElo", "EloDiff",
    "B2B_Advantage"  # Kombiniert!
]

train_b2b = daten[daten["SEASON"] != 2025]
test_b2b = daten[daten["SEASON"] == 2025]

modell_b2b_v2 = LogisticRegression(max_iter=2000)
modell_b2b_v2.fit(train_b2b[FEATURES_B2B_V2], train_b2b["HOME_WIN"])

vorhersagen_b2b_v2 = modell_b2b_v2.predict(test_b2b[FEATURES_B2B_V2])
genauigkeit_b2b_v2 = accuracy_score(test_b2b["HOME_WIN"], vorhersagen_b2b_v2)

print(f"Modell ohne B2B:     65.2%")
print(f"Modell B2B einzeln:  64.5%")
print(f"Modell B2B kombiniert: {genauigkeit_b2b_v2:.1%}")


# In[14]:


# 😴 Rest Days Feature
daten["Home_RestDays"] = 0
daten["Away_RestDays"] = 0

for i, row in daten.iterrows():
    for team_id, prefix in [
        (row["HOME_TEAM_ID"], "Home"),
        (row["AWAY_TEAM_ID"], "Away")
    ]:
        # Letztes Spiel vor diesem
        letztes_spiel = daten[
            ((daten["HOME_TEAM_ID"] == team_id) |
             (daten["AWAY_TEAM_ID"] == team_id)) &
            (daten["GAME_DATE"] < row["GAME_DATE"])
        ]["GAME_DATE"].max()

        if pd.notna(letztes_spiel):
            rest = (row["GAME_DATE"] - letztes_spiel).days
            daten.at[i, f"{prefix}_RestDays"] = min(rest, 7)  # Max 7 Tage

# Statistik
print("Heimsiege nach Rest Days:")
for tage in range(1, 6):
    spiele = daten[daten["Home_RestDays"] == tage]
    if len(spiele) > 50:
        print(f"  {tage} Tag(e) Pause: {spiele['HOME_WIN'].mean():.1%} ({len(spiele)} Spiele)")


# In[15]:


# Rest Days Differenz als Feature
daten["RestDays_Advantage"] = daten["Home_RestDays"] - daten["Away_RestDays"]

FEATURES_REST = [
    "HomeWinRate", "AwayWinRate",
    "HomePtsScored", "HomePtsConceded",
    "AwayPtsScored", "AwayPtsConceded",
    "HomeElo", "AwayElo", "EloDiff",
    "Home_RestDays", "Away_RestDays",
    "RestDays_Advantage"
]

train_rest = daten[daten["SEASON"] != 2025]
test_rest = daten[daten["SEASON"] == 2025]

modell_rest = LogisticRegression(max_iter=2000)
modell_rest.fit(train_rest[FEATURES_REST], train_rest["HOME_WIN"])

vorhersagen_rest = modell_rest.predict(test_rest[FEATURES_REST])
genauigkeit_rest = accuracy_score(test_rest["HOME_WIN"], vorhersagen_rest)

print(f"Modell ohne Rest Days: 65.2%")
print(f"Modell mit Rest Days:  {genauigkeit_rest:.1%}")


# In[16]:


# 🏠 Team-spezifischer Heimvorteil
heim_vorteil = {}

for team_id in daten["HOME_TEAM_ID"].unique():
    heim_spiele = daten[daten["HOME_TEAM_ID"] == team_id]
    auswaerts_spiele = daten[daten["AWAY_TEAM_ID"] == team_id]

    if len(heim_spiele) > 20 and len(auswaerts_spiele) > 20:
        heim_win_rate = heim_spiele["HOME_WIN"].mean()
        auswaerts_win_rate = 1 - auswaerts_spiele["HOME_WIN"].mean()
        vorteil = heim_win_rate - auswaerts_win_rate
        heim_vorteil[team_id] = vorteil

# Teams mit stärkstem Heimvorteil
# Teams direkt von balldontlie API laden
response_teams = requests.get(
    "https://api.balldontlie.io/v1/teams",
    headers=HEADERS,
    params={"per_page": 100}
)
teams_data = response_teams.json()["data"]
teams = pd.DataFrame([{
    "TEAM_ID": t["id"],
    "NICKNAME": t["name"],
    "CITY": t["city"],
    "FULL_NAME": t["full_name"]
} for t in teams_data])
vorteil_df = pd.DataFrame(
    list(heim_vorteil.items()),
    columns=["TEAM_ID", "Heimvorteil"]
).merge(teams[["TEAM_ID", "NICKNAME"]], on="TEAM_ID")
vorteil_df = vorteil_df.sort_values("Heimvorteil", ascending=False)

print("🏠 Top 10 Heimteams:")
print(vorteil_df.head(10)[["NICKNAME", "Heimvorteil"]].to_string(index=False))
print("\n✈️  Schlechteste Heimteams:")
print(vorteil_df.tail(5)[["NICKNAME", "Heimvorteil"]].to_string(index=False))


# In[17]:


# Team Namen direkt aus unseren Daten holen
team_namen = {}
for _, row in daten.iterrows():
    team_namen[row["HOME_TEAM_ID"]] = row["HOME_TEAM"]
    team_namen[row["AWAY_TEAM_ID"]] = row["AWAY_TEAM"]

# Heimvorteil berechnen
vorteil_liste = []
for team_id, vorteil in heim_vorteil.items():
    vorteil_liste.append({
        "Team": team_namen.get(team_id, str(team_id)),
        "Heimvorteil": vorteil
    })

vorteil_df = pd.DataFrame(vorteil_liste).sort_values("Heimvorteil", ascending=False)

print("🏠 Top 10 Heimteams:")
print(vorteil_df.head(10).to_string(index=False))
print("\n✈️  Schlechteste Heimteams:")
print(vorteil_df.tail(5).to_string(index=False))


# In[18]:


# Heimvorteil zum Datensatz hinzufügen
daten["Home_Advantage"] = daten["HOME_TEAM_ID"].map(heim_vorteil).fillna(0)
daten["Away_Advantage"] = daten["AWAY_TEAM_ID"].map(heim_vorteil).fillna(0)

FEATURES_ADV = [
    "HomeWinRate", "AwayWinRate",
    "HomePtsScored", "HomePtsConceded",
    "AwayPtsScored", "AwayPtsConceded",
    "HomeElo", "AwayElo", "EloDiff",
    "Home_Advantage", "Away_Advantage"
]

train_adv = daten[daten["SEASON"] != 2025]
test_adv = daten[daten["SEASON"] == 2025]

modell_adv = LogisticRegression(max_iter=2000)
modell_adv.fit(train_adv[FEATURES_ADV], train_adv["HOME_WIN"])

vorhersagen_adv = modell_adv.predict(test_adv[FEATURES_ADV])
genauigkeit_adv = accuracy_score(test_adv["HOME_WIN"], vorhersagen_adv)

print(f"Modell ohne Heimvorteil: 65.2%")
print(f"Modell mit Heimvorteil:  {genauigkeit_adv:.1%}")


# In[19]:


from xgboost import XGBClassifier

# Alle Features zusammen
FEATURES_ALLE = [
    "HomeWinRate", "AwayWinRate",
    "HomePtsScored", "HomePtsConceded",
    "AwayPtsScored", "AwayPtsConceded",
    "HomeElo", "AwayElo", "EloDiff",
    "Home_B2B", "Away_B2B",
    "Home_RestDays", "Away_RestDays",
    "RestDays_Advantage",
    "Home_Advantage", "Away_Advantage"
]

train_xgb = daten[daten["SEASON"] != 2025]
test_xgb = daten[daten["SEASON"] == 2025]

modell_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
modell_xgb.fit(train_xgb[FEATURES_ALLE], train_xgb["HOME_WIN"])

vorhersagen_xgb = modell_xgb.predict(test_xgb[FEATURES_ALLE])
genauigkeit_xgb = accuracy_score(test_xgb["HOME_WIN"], vorhersagen_xgb)

print(f"Logistische Regression: 65.2%")
print(f"XGBoost (alle Features): {genauigkeit_xgb:.1%}")


# In[20]:


import requests
from bs4 import BeautifulSoup

# ESPN NBA Injury Report scrapen
url = "https://www.espn.com/nba/injuries"
headers_espn = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

response = requests.get(url, headers=headers_espn)
print(f"Status: {response.status_code}")

soup = BeautifulSoup(response.text, "html.parser")

# Verletzungen suchen
tabellen = soup.find_all("table")
print(f"Tabellen gefunden: {len(tabellen)}")

if len(tabellen) > 0:
    print("\nErste Tabelle:")
    print(tabellen[0].get_text()[:500])


# In[24]:


# Korrekte Extraktion mit span für Teamnamen
alle_verletzungen = []

for tabelle in tabellen:
    # Team aus span Element holen
    team_span = tabelle.find_previous("span")
    team_name = team_span.get_text().strip() if team_span else "Unbekannt"

    zeilen = tabelle.find_all("tr")
    for zeile in zeilen[1:]:
        spalten = zeile.find_all("td")
        if len(spalten) >= 4:
            alle_verletzungen.append({
                "Team": team_name,
                "Spieler": spalten[0].get_text().strip(),
                "Status": spalten[3].get_text().strip(),
            })

verletzungen_df = pd.DataFrame(alle_verletzungen)
print(f"Verletzte Spieler: {len(verletzungen_df)}")
print(verletzungen_df[["Team", "Spieler", "Status"]].head(20).to_string(index=False))


# In[25]:


# Korrekte Extraktion mit span für Teamnamen
alle_verletzungen = []

for tabelle in tabellen:
    # Team aus span Element holen
    team_span = tabelle.find_previous("span")
    team_name = team_span.get_text().strip() if team_span else "Unbekannt"

    zeilen = tabelle.find_all("tr")
    for zeile in zeilen[1:]:
        spalten = zeile.find_all("td")
        if len(spalten) >= 4:
            alle_verletzungen.append({
                "Team": team_name,
                "Spieler": spalten[0].get_text().strip(),
                "Status": spalten[3].get_text().strip(),
            })

verletzungen_df = pd.DataFrame(alle_verletzungen)
print(f"Verletzte Spieler: {len(verletzungen_df)}")
print(verletzungen_df[["Team", "Spieler", "Status"]].head(20).to_string(index=False))


# In[26]:


# Heutige Spiele mit Verletzungen abgleichen
print("🏥 Verletzungscheck für heutige Spiele:")
print("=" * 60)

for _, spiel in heute_df.iterrows():
    heim = spiel["Heimteam"]
    auswaerts = spiel["Auswärtsteam"]

    heim_kurz = heim.split()[-1]
    auswaerts_kurz = auswaerts.split()[-1]

    heim_verletzt = verletzungen_df[
        verletzungen_df["Team"].str.contains(heim_kurz, case=False)
    ]
    auswaerts_verletzt = verletzungen_df[
        verletzungen_df["Team"].str.contains(auswaerts_kurz, case=False)
    ]

    print(f"\n{heim} vs {auswaerts}")

    if len(heim_verletzt) > 0:
        out_heim = heim_verletzt[heim_verletzt["Status"] == "Out"]["Spieler"].tolist()
        dtd_heim = heim_verletzt[heim_verletzt["Status"] == "Day-To-Day"]["Spieler"].tolist()
        if out_heim: print(f"  ❌ {heim_kurz} Out: {', '.join(out_heim)}")
        if dtd_heim: print(f"  ⚠️  {heim_kurz} Day-To-Day: {', '.join(dtd_heim)}")

    if len(auswaerts_verletzt) > 0:
        out_aus = auswaerts_verletzt[auswaerts_verletzt["Status"] == "Out"]["Spieler"].tolist()
        dtd_aus = auswaerts_verletzt[auswaerts_verletzt["Status"] == "Day-To-Day"]["Spieler"].tolist()
        if out_aus: print(f"  ❌ {auswaerts_kurz} Out: {', '.join(out_aus)}")
        if dtd_aus: print(f"  ⚠️  {auswaerts_kurz} Day-To-Day: {', '.join(dtd_aus)}")


# In[27]:


# Verletzungs-Score berechnen
# Wichtige Positionen bekommen mehr Gewicht
def berechne_verletzungs_score(team_kurz, verletzungen_df):
    verletzt = verletzungen_df[
        verletzungen_df["Team"].str.contains(team_kurz, case=False)
    ]
    score = 0
    for _, spieler in verletzt.iterrows():
        if spieler["Status"] == "Out":
            score += 1.0
        elif spieler["Status"] == "Day-To-Day":
            score += 0.5
    return score

# Score für heutige Spiele
print("📊 Verletzungs-Score heute:")
print("=" * 60)

for _, spiel in heute_df.iterrows():
    heim_kurz = spiel["Heimteam"].split()[-1]
    auswaerts_kurz = spiel["Auswärtsteam"].split()[-1]

    heim_score = berechne_verletzungs_score(heim_kurz, verletzungen_df)
    auswaerts_score = berechne_verletzungs_score(auswaerts_kurz, verletzungen_df)

    vorteil = auswaerts_score - heim_score

    print(f"\n{spiel['Heimteam']} vs {spiel['Auswärtsteam']}")
    print(f"  Verletzungen Heim: {heim_score:.1f} | Auswärts: {auswaerts_score:.1f}")
    if vorteil > 1:
        print(f"  → Heimvorteil durch Verletzungen! (+{vorteil:.1f})")
    elif vorteil < -1:
        print(f"  → Auswärtsvorteil durch Verletzungen! ({vorteil:.1f})")
    else:
        print(f"  → Ausgeglichen")


# In[31]:


# Spieler Stats aus unserem eigenen Dataset
spieler_stats = pd.read_csv(
    os.path.expanduser("~/Downloads/nba_daten/games_details.csv")
)

print(f"Einträge: {len(spieler_stats)}")
print(f"Spalten: {spieler_stats.columns.tolist()}")
spieler_stats.head(3)


# In[32]:


# Durchschnittliche Stats pro Spieler berechnen
# Nur Spieler mit mindestens 20 Spielen für Zuverlässigkeit

# Minuten als Zahl konvertieren
def min_zu_zahl(min_str):
    try:
        if pd.isna(min_str):
            return 0
        teile = str(min_str).split(":")
        return int(teile[0]) + int(teile[1])/60 if len(teile) == 2 else float(teile[0])
    except:
        return 0

spieler_stats["MIN_NUM"] = spieler_stats["MIN"].apply(min_zu_zahl)

# Durchschnitt pro Spieler
spieler_avg = spieler_stats.groupby("PLAYER_NAME").agg(
    Spiele=("GAME_ID", "count"),
    MIN=("MIN_NUM", "mean"),
    PTS=("PTS", "mean"),
    REB=("REB", "mean"),
    AST=("AST", "mean")
).reset_index()

# Nur Spieler mit mindestens 20 Spielen
spieler_avg = spieler_avg[spieler_avg["Spiele"] >= 20]

# Impact Score = gewichtete Kombination
spieler_avg["Impact"] = (
    spieler_avg["PTS"] * 1.0 +
    spieler_avg["AST"] * 1.5 +  # Assists wertvoller
    spieler_avg["REB"] * 0.8 +
    spieler_avg["MIN"] * 0.3
)

# Normalisieren auf 0-10 Skala
max_impact = spieler_avg["Impact"].max()
spieler_avg["Impact_Score"] = (spieler_avg["Impact"] / max_impact * 10).round(2)

print(f"Spieler mit Impact Score: {len(spieler_avg)}")
print("\nTop 10 Spieler nach Impact:")
print(spieler_avg.nlargest(10, "Impact_Score")[
    ["PLAYER_NAME", "MIN", "PTS", "REB", "AST", "Impact_Score"]
].to_string(index=False))


# In[33]:


# Verletzungs-Impact für heutige Spiele
def berechne_impact_verlust(team_kurz, verletzungen_df, spieler_avg):
    verletzt = verletzungen_df[
        verletzungen_df["Team"].str.contains(team_kurz, case=False)
    ]

    total_impact = 0
    details = []

    for _, spieler in verletzt.iterrows():
        name = spieler["Spieler"]
        status = spieler["Status"]

        # Impact Score suchen
        match = spieler_avg[
            spieler_avg["PLAYER_NAME"].str.contains(
                name.split()[0] + " " + name.split()[-1] 
                if len(name.split()) > 1 else name, 
                case=False, na=False
            )
        ]

        if len(match) > 0:
            impact = match.iloc[0]["Impact_Score"]
            gewicht = 1.0 if status == "Out" else 0.5
            total_impact += impact * gewicht
            details.append(f"{name} ({impact:.1f})")

    return total_impact, details

# Ausgabe
print("💥 Impact-Verlust durch Verletzungen heute:")
print("=" * 65)

for _, spiel in heute_df.iterrows():
    heim_kurz = spiel["Heimteam"].split()[-1]
    auswaerts_kurz = spiel["Auswärtsteam"].split()[-1]

    heim_impact, heim_details = berechne_impact_verlust(
        heim_kurz, verletzungen_df, spieler_avg)
    auswaerts_impact, auswaerts_details = berechne_impact_verlust(
        auswaerts_kurz, verletzungen_df, spieler_avg)

    print(f"\n{spiel['Heimteam']} vs {spiel['Auswärtsteam']}")
    print(f"  Heim Impact-Verlust:     {heim_impact:.1f}")
    if heim_details:
        print(f"  Fehlende Spieler: {', '.join(heim_details[:3])}")
    print(f"  Auswärts Impact-Verlust: {auswaerts_impact:.1f}")
    if auswaerts_details:
        print(f"  Fehlende Spieler: {', '.join(auswaerts_details[:3])}")


# In[37]:


get_ipython().system('pip install nba_api')


# In[39]:


from nba_api.stats.endpoints import leaguedashplayerstats

# Aktuelle Saison Stats laden
stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2025-26",
    season_type_all_star="Regular Season",
    per_mode_detailed="PerGame"  # Changed from per_mode_simple to per_mode_detailed
    # Alternatively, you might try just "per_mode" instead of "per_mode_detailed"
)

stats_df = stats.get_data_frames()[0]
print(f"Spieler: {len(stats_df)}")
print(stats_df[["PLAYER_NAME", "MIN", "PTS", "REB", "AST"]].head(10))


# In[41]:


from nba_api.stats.endpoints import leaguedashplayerstats

# Letzte Saison laden
print("Lade Saison 2024/25...")
stats_2425 = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2024-25",
    season_type_all_star="Regular Season",
    per_mode_detailed="PerGame",  # Changed from per_mode_simple to per_mode_detailed
    timeout=30
).get_data_frames()[0]

print(f"✅ Saison 2024/25: {len(stats_2425)} Spieler")

# Impact Score berechnen
def berechne_impact(df):
    df["Impact"] = (
        df["PTS"] * 1.0 +
        df["AST"] * 1.5 +
        df["REB"] * 0.8 +
        df["MIN"] * 0.3
    )
    max_val = df["Impact"].max()
    df["Impact_Score"] = (df["Impact"] / max_val * 10).round(2)
    return df[["PLAYER_NAME", "Impact_Score", "MIN", "PTS", "REB", "AST"]]

stats_2526_impact = berechne_impact(stats_df.copy())  # Note: stats_df is not defined in the snippet
stats_2425_impact = berechne_impact(stats_2425.copy())

# Zusammenführen mit Gewichtung
kombiniert = stats_2526_impact.merge(
    stats_2425_impact[["PLAYER_NAME", "Impact_Score"]],
    on="PLAYER_NAME",
    how="outer",
    suffixes=("_2526", "_2425")
).fillna(0)

# 60% aktuelle Saison, 40% letzte Saison
kombiniert["Impact_Final"] = (
    kombiniert["Impact_Score_2526"] * 0.6 +
    kombiniert["Impact_Score_2425"] * 0.4
).round(2)

print(f"\nSpieler mit kombiniertem Impact Score: {len(kombiniert)}")
print("\nTop 10:")
print(kombiniert.nlargest(10, "Impact_Final")[
    ["PLAYER_NAME", "Impact_Score_2526", "Impact_Score_2425", "Impact_Final"]
].to_string(index=False))


# In[42]:


# Verletzungs-Impact mit kombinierten aktuellen Scores
def berechne_impact_verlust_neu(team_kurz, verletzungen_df, kombiniert):
    verletzt = verletzungen_df[
        verletzungen_df["Team"].str.contains(team_kurz, case=False)
    ]

    total_impact = 0
    details = []

    for _, spieler in verletzt.iterrows():
        name = spieler["Spieler"]
        status = spieler["Status"]

        match = kombiniert[
            kombiniert["PLAYER_NAME"].str.contains(
                name.split()[0], case=False, na=False
            ) &
            kombiniert["PLAYER_NAME"].str.contains(
                name.split()[-1], case=False, na=False
            )
        ]

        if len(match) > 0:
            impact = match.iloc[0]["Impact_Final"]
            gewicht = 1.0 if status == "Out" else 0.5
            total_impact += impact * gewicht
            details.append(f"{name} ({impact:.1f})")

    return total_impact, details

# Finale Ausgabe
print("💥 AKTUALISIERTER Impact-Verlust:")
print("=" * 65)

for _, spiel in heute_df.iterrows():
    heim_kurz = spiel["Heimteam"].split()[-1]
    auswaerts_kurz = spiel["Auswärtsteam"].split()[-1]

    heim_impact, heim_details = berechne_impact_verlust_neu(
        heim_kurz, verletzungen_df, kombiniert)
    auswaerts_impact, auswaerts_details = berechne_impact_verlust_neu(
        auswaerts_kurz, verletzungen_df, kombiniert)

    print(f"\n{spiel['Heimteam']} vs {spiel['Auswärtsteam']}")
    print(f"  Heim Impact-Verlust:     {heim_impact:.1f}")
    if heim_details:
        print(f"  Fehlende Spieler: {', '.join(heim_details[:3])}")
    print(f"  Auswärts Impact-Verlust: {auswaerts_impact:.1f}")
    if auswaerts_details:
        print(f"  Fehlende Spieler: {', '.join(auswaerts_details[:3])}")


# In[43]:


# 🎯 ULTIMATIVE VORHERSAGE mit Impact-Korrektur
print("🏀 ULTIMATIVE NBA VORHERSAGEN – 16. März 2026")
print("(Modell + Verletzungs-Impact)")
print("=" * 65)

for idx, spiel in heute_df.iterrows():
    heim_kurz = spiel["Heimteam"].split()[-1]
    auswaerts_kurz = spiel["Auswärtsteam"].split()[-1]

    prob_home = wahrscheinlichkeiten[idx][1]
    prob_away = wahrscheinlichkeiten[idx][0]

    heim_impact, heim_details = berechne_impact_verlust_neu(
        heim_kurz, verletzungen_df, kombiniert)
    auswaerts_impact, auswaerts_details = berechne_impact_verlust_neu(
        auswaerts_kurz, verletzungen_df, kombiniert)

    # Impact Korrektur
    impact_diff = (auswaerts_impact - heim_impact) / 30
    impact_diff = max(-0.15, min(0.15, impact_diff))

    prob_home_adj = min(0.95, max(0.05, prob_home + impact_diff))
    prob_away_adj = 1 - prob_home_adj

    tipp = spiel["Heimteam"] if prob_home_adj > 0.5 else spiel["Auswärtsteam"]
    konfidenz = max(prob_home_adj, prob_away_adj)

    original_tipp = spiel["Heimteam"] if vorhersagen_heute[idx] == 1 else spiel["Auswärtsteam"]
    geaendert = " ⚡ GEÄNDERT!" if tipp != original_tipp else ""

    if konfidenz >= 0.70:
        label = "🔒 Sicher"
    elif konfidenz >= 0.60:
        label = "✅ Gut"
    else:
        label = "❓ Knapp"

    print(f"\n{spiel['Heimteam']} vs {spiel['Auswärtsteam']}")
    print(f"  🏆 Tipp: {tipp} ({konfidenz:.0%}) {label}{geaendert}")
    print(f"  Original: {prob_home:.0%}/{prob_away:.0%} → Angepasst: {prob_home_adj:.0%}/{prob_away_adj:.0%}")
    if heim_details:
        print(f"  ❌ Heim fehlt: {', '.join(heim_details[:2])}")
    if auswaerts_details:
        print(f"  ❌ Auswärts fehlt: {', '.join(auswaerts_details[:2])}")


# In[44]:


# 💾 Komplette Vorhersage mit Verletzungen speichern
ergebnisse = []

for idx, spiel in heute_df.iterrows():
    heim_kurz = spiel["Heimteam"].split()[-1]
    auswaerts_kurz = spiel["Auswärtsteam"].split()[-1]

    prob_home = wahrscheinlichkeiten[idx][1]
    prob_away = wahrscheinlichkeiten[idx][0]

    heim_impact, heim_details = berechne_impact_verlust_neu(
        heim_kurz, verletzungen_df, kombiniert)
    auswaerts_impact, auswaerts_details = berechne_impact_verlust_neu(
        auswaerts_kurz, verletzungen_df, kombiniert)

    impact_diff = (auswaerts_impact - heim_impact) / 30
    impact_diff = max(-0.15, min(0.15, impact_diff))

    prob_home_adj = min(0.95, max(0.05, prob_home + impact_diff))
    prob_away_adj = 1 - prob_home_adj

    tipp = spiel["Heimteam"] if prob_home_adj > 0.5 else spiel["Auswärtsteam"]
    konfidenz = max(prob_home_adj, prob_away_adj)

    ergebnisse.append({
        "Datum": HEUTE,
        "Heimteam": spiel["Heimteam"],
        "Auswärtsteam": spiel["Auswärtsteam"],
        "Tipp": tipp,
        "Konfidenz": f"{konfidenz:.0%}",
        "Heim_Prob": f"{prob_home_adj:.0%}",
        "Auswaerts_Prob": f"{prob_away_adj:.0%}",
        "Heim_Impact_Verlust": round(heim_impact, 1),
        "Auswaerts_Impact_Verlust": round(auswaerts_impact, 1),
        "Fehlende_Heim": ", ".join(heim_details[:3]),
        "Fehlende_Auswaerts": ", ".join(auswaerts_details[:3]),
        "Echtes_Ergebnis": "",
        "Richtig": ""
    })

finale_df = pd.DataFrame(ergebnisse)
dateiname = f"predictions_{HEUTE}.csv"
finale_df.to_csv(os.path.expanduser(f"~/{dateiname}"), index=False)
print(f"✅ Gespeichert: {dateiname}")
print(finale_df[["Heimteam", "Auswärtsteam", "Tipp", "Konfidenz"]].to_string(index=False))


# In[ ]:




