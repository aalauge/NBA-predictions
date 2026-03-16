import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nba_api.stats.endpoints import leaguedashplayerstats

# API Key
API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "98ee804c-c05f-4eb3-88cd-3b7d42ae474e")
HEADERS = {"Authorization": API_KEY}

# ─── SCHRITT 1: Spieldaten laden ───────────────────────────────────────────────

def lade_spiele(saison):
    spiele = []
    cursor = None
    while True:
        params = {"seasons[]": saison, "per_page": 100}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get("https://api.balldontlie.io/v1/games",
                             headers=HEADERS, params=params)
            if r.status_code == 429:
                print("Rate limit, warte 60 Sekunden...")
                time.sleep(60)
                continue
            if r.status_code != 200 or not r.text.strip():
                print(f"Fehler: {r.status_code} - {r.text}")
                time.sleep(30)
                continue
            data = r.json()
            spiele.extend(data["data"])
            cursor = data["meta"].get("next_cursor")
            if not cursor:
                break
            time.sleep(1.5)
        except Exception as e:
            print(f"Fehler: {e}, warte 30 Sekunden...")
            time.sleep(30)
            continue
    return spiele

alle_spiele = []
for saison in [2022, 2023, 2024, 2025]:
    print(f"Lade Saison {saison}...")
    spiele = lade_spiele(saison)
    alle_spiele.extend(spiele)
    print(f"  → {len(spiele)} Spiele")

print(f"\n✅ Gesamt: {len(alle_spiele)} Spiele")

# ─── SCHRITT 2: DataFrame erstellen ────────────────────────────────────────────

df_raw = pd.DataFrame([{
    "game_id": g["id"],
    "date": g["date"][:10],
    "home_team": g["home_team"]["full_name"],
    "away_team": g["visitor_team"]["full_name"],
    "home_score": g["home_team_score"],
    "away_score": g["visitor_team_score"],
    "season": g["season"],
    "status": g["status"]
} for g in alle_spiele])

df = df_raw[df_raw["status"] == "Final"].copy()
df["date"] = pd.to_datetime(df["date"])
df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

print(f"✅ Gespielte Spiele: {len(df)}")

# ─── SCHRITT 3: Features berechnen ─────────────────────────────────────────────

df = df.sort_values("date").reset_index(drop=True)

def berechne_form(df, window=10):
    home_stats = []
    away_stats = []
    for idx, row in df.iterrows():
        datum = row["date"]
        heim = row["home_team"]
        auswaerts = row["away_team"]
        heim_spiele = df[(df["date"] < datum) &
                         ((df["home_team"] == heim) | (df["away_team"] == heim))].tail(window)
        auswaerts_spiele = df[(df["date"] < datum) &
                              ((df["home_team"] == auswaerts) | (df["away_team"] == auswaerts))].tail(window)
        if len(heim_spiele) > 0:
            h_wins = sum((r["home_team"] == heim and r["home_win"] == 1) or
                         (r["away_team"] == heim and r["home_win"] == 0)
                         for _, r in heim_spiele.iterrows())
            h_winrate = h_wins / len(heim_spiele)
        else:
            h_winrate = 0.5
        if len(auswaerts_spiele) > 0:
            a_wins = sum((r["home_team"] == auswaerts and r["home_win"] == 1) or
                         (r["away_team"] == auswaerts and r["home_win"] == 0)
                         for _, r in auswaerts_spiele.iterrows())
            a_winrate = a_wins / len(auswaerts_spiele)
        else:
            a_winrate = 0.5
        home_stats.append(h_winrate)
        away_stats.append(a_winrate)
    df["home_winrate"] = home_stats
    df["away_winrate"] = away_stats
    return df

print("Berechne Form-Features...")
df = berechne_form(df)
print("✅ Form-Features fertig!")

def berechne_punkte(df, window=10):
    home_pts_scored, home_pts_conceded = [], []
    away_pts_scored, away_pts_conceded = [], []
    for idx, row in df.iterrows():
        datum = row["date"]
        heim = row["home_team"]
        auswaerts = row["away_team"]
        heim_spiele = df[(df["date"] < datum) &
                         ((df["home_team"] == heim) | (df["away_team"] == heim))].tail(window)
        auswaerts_spiele = df[(df["date"] < datum) &
                              ((df["home_team"] == auswaerts) | (df["away_team"] == auswaerts))].tail(window)
        if len(heim_spiele) > 0:
            h_scored = [r["home_score"] if r["home_team"] == heim else r["away_score"] for _, r in heim_spiele.iterrows()]
            h_conceded = [r["away_score"] if r["home_team"] == heim else r["home_score"] for _, r in heim_spiele.iterrows()]
        else:
            h_scored, h_conceded = [110], [110]
        if len(auswaerts_spiele) > 0:
            a_scored = [r["home_score"] if r["home_team"] == auswaerts else r["away_score"] for _, r in auswaerts_spiele.iterrows()]
            a_conceded = [r["away_score"] if r["home_team"] == auswaerts else r["home_score"] for _, r in auswaerts_spiele.iterrows()]
        else:
            a_scored, a_conceded = [110], [110]
        home_pts_scored.append(np.mean(h_scored))
        home_pts_conceded.append(np.mean(h_conceded))
        away_pts_scored.append(np.mean(a_scored))
        away_pts_conceded.append(np.mean(a_conceded))
    df["home_pts_scored"] = home_pts_scored
    df["home_pts_conceded"] = home_pts_conceded
    df["away_pts_scored"] = away_pts_scored
    df["away_pts_conceded"] = away_pts_conceded
    return df

print("Berechne Punkte-Features...")
df = berechne_punkte(df)
print("✅ Punkte-Features fertig!")

# ─── SCHRITT 4: Elo Rating ─────────────────────────────────────────────────────

def berechne_elo(df, start_elo=1500, k=20):
    elo = {team: start_elo for team in pd.concat([df["home_team"], df["away_team"]]).unique()}
    home_elo_list, away_elo_list = [], []
    for _, row in df.iterrows():
        heim = row["home_team"]
        auswaerts = row["away_team"]
        h_elo = elo[heim]
        a_elo = elo[auswaerts]
        home_elo_list.append(h_elo)
        away_elo_list.append(a_elo)
        expected_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        expected_a = 1 - expected_h
        actual_h = row["home_win"]
        actual_a = 1 - actual_h
        elo[heim] = h_elo + k * (actual_h - expected_h)
        elo[auswaerts] = a_elo + k * (actual_a - expected_a)
    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    return df

df = berechne_elo(df)
print("✅ Elo fertig!")

# ─── SCHRITT 5: Modell trainieren ──────────────────────────────────────────────

features = ["home_winrate", "away_winrate",
            "home_pts_scored", "home_pts_conceded",
            "away_pts_scored", "away_pts_conceded",
            "home_elo", "away_elo"]

train = df[df["season"] < 2025]
test = df[df["season"] == 2025]

model = LogisticRegression(max_iter=1000)
model.fit(train[features], train["home_win"])

acc = accuracy_score(test["home_win"], model.predict(test[features]))
print(f"✅ Modell trainiert! Genauigkeit: {acc:.1%}")

# ─── SCHRITT 6: Heutige Spiele laden ───────────────────────────────────────────

heute = datetime.now().strftime("%Y-%m-%d")
print(f"Lade Spiele für: {heute}")

try:
    r = requests.get("https://api.balldontlie.io/v1/games",
                     headers=HEADERS,
                     params={"dates[]": heute, "per_page": 100})
    if r.status_code == 429:
        print("Rate limit, warte 60 Sekunden...")
        time.sleep(60)
        r = requests.get("https://api.balldontlie.io/v1/games",
                         headers=HEADERS,
                         params={"dates[]": heute, "per_page": 100})
    print(f"Status: {r.status_code}")
    data = r.json()
    heute_spiele = data["data"]
except Exception as e:
    print(f"Fehler beim Laden der heutigen Spiele: {e}")
    heute_spiele = []

print(f"✅ {len(heute_spiele)} Spiele heute")
for s in heute_spiele:
    print(f"  {s['home_team']['full_name']} vs {s['visitor_team']['full_name']}")

# ─── SCHRITT 7: Basis-Vorhersagen ──────────────────────────────────────────────

vorhersagen = []
for spiel in heute_spiele:
    heim = spiel["home_team"]["full_name"]
    auswaerts = spiel["visitor_team"]["full_name"]
    heim_stats = df[df["home_team"] == heim].tail(1)
    if len(heim_stats) == 0:
        heim_stats = df[df["away_team"] == heim].tail(1)
    auswaerts_stats = df[df["home_team"] == auswaerts].tail(1)
    if len(auswaerts_stats) == 0:
        auswaerts_stats = df[df["away_team"] == auswaerts].tail(1)
    if len(heim_stats) == 0 or len(auswaerts_stats) == 0:
        print(f"⚠️ Keine Stats für {heim} vs {auswaerts}")
        continue
    x = pd.DataFrame([{
        "home_winrate": heim_stats.iloc[0]["home_winrate"],
        "away_winrate": auswaerts_stats.iloc[0]["away_winrate"],
        "home_pts_scored": heim_stats.iloc[0]["home_pts_scored"],
        "home_pts_conceded": heim_stats.iloc[0]["home_pts_conceded"],
        "away_pts_scored": auswaerts_stats.iloc[0]["away_pts_scored"],
        "away_pts_conceded": auswaerts_stats.iloc[0]["away_pts_conceded"],
        "home_elo": heim_stats.iloc[0]["home_elo"],
        "away_elo": auswaerts_stats.iloc[0]["away_elo"]
    }])
    prob = model.predict_proba(x)[0][1]
    gewinner = heim if prob > 0.5 else auswaerts
    konfidenz = prob if prob > 0.5 else 1 - prob
    vorhersagen.append({
        "Heimteam": heim,
        "Auswärtsteam": auswaerts,
        "Heimsieg %": round(prob * 100, 1),
        "Tipp": gewinner,
        "Konfidenz": round(konfidenz * 100, 1)
    })

vorhersagen_df = pd.DataFrame(vorhersagen)
print(vorhersagen_df.to_string(index=False))

# ─── SCHRITT 8: Verletzungen laden ─────────────────────────────────────────────

def lade_verletzungen():
    url = "https://www.espn.com/nba/injuries"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    verletzungen = []
    tabellen = soup.find_all("div", class_="ResponsiveTable")
    for tabelle in tabellen:
        team_span = tabelle.find("span", class_="injuries__teamName")
        team_name = team_span.text.strip() if team_span else "Unbekannt"
        rows = tabelle.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 4:
                verletzungen.append({
                    "Team": team_name,
                    "Spieler": cols[0].text.strip(),
                    "Status": cols[3].text.strip()
                })
    return pd.DataFrame(verletzungen)

verletzungen_df = lade_verletzungen()
verletzungen_df = verletzungen_df[
    verletzungen_df["Status"].str.contains("Out|Doubtful", case=False, na=False)
]
print(f"✅ {len(verletzungen_df)} verletzte Spieler")

# ─── SCHRITT 9: Spieler Impact Scores ──────────────────────────────────────────

def lade_spieler_stats(saison):
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=saison,
        per_mode_detailed="PerGame"
    )
    df = stats.get_data_frames()[0]
    return df[["PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PTS", "REB", "AST", "GP"]]

print("Lade Spieler Stats...")
stats_aktuell = lade_spieler_stats("2025-26")
stats_vorjahr = lade_spieler_stats("2024-25")

stats_aktuell = stats_aktuell[stats_aktuell["GP"] >= 20]
stats_vorjahr = stats_vorjahr[stats_vorjahr["GP"] >= 20]

def berechne_impact(df):
    df = df.copy()
    df["Impact"] = (df["PTS"] * 1.0 + df["AST"] * 1.5 + df["REB"] * 0.8 + df["MIN"] * 0.3)
    max_impact = df["Impact"].max()
    df["Impact_Score"] = (df["Impact"] / max_impact * 10).round(2)
    return df

stats_aktuell = berechne_impact(stats_aktuell)
stats_vorjahr = berechne_impact(stats_vorjahr)

kombiniert = stats_aktuell.merge(
    stats_vorjahr[["PLAYER_NAME", "Impact_Score"]],
    on="PLAYER_NAME",
    suffixes=("_Aktuell", "_Vorjahr"),
    how="left"
)
kombiniert["Impact_Final"] = (
    kombiniert["Impact_Score_Aktuell"] * 0.6 +
    kombiniert["Impact_Score_Vorjahr"].fillna(kombiniert["Impact_Score_Aktuell"]) * 0.4
).round(2)

print(f"✅ {len(kombiniert)} Spieler mit Impact Score")

# ─── SCHRITT 10: Finale Vorhersagen mit Verletzungskorrektur ───────────────────

def berechne_impact_verlust(team_name, verletzungen_df, kombiniert):
    verletzt = verletzungen_df[verletzungen_df["Team"] == team_name]
    total_impact = 0
    details = []
    for _, spieler in verletzt.iterrows():
        name = spieler["Spieler"]
        status = spieler["Status"]
        teile = name.split()
        if len(teile) >= 2:
            match = kombiniert[
                kombiniert["PLAYER_NAME"].str.contains(teile[0], case=False, na=False) &
                kombiniert["PLAYER_NAME"].str.contains(teile[-1], case=False, na=False)
            ]
        else:
            match = kombiniert[kombiniert["PLAYER_NAME"].str.contains(name, case=False, na=False)]
        if len(match) > 0:
            impact = match.iloc[0]["Impact_Final"]
            gewicht = 1.0 if status == "Out" else 0.5
            total_impact += impact * gewicht
            details.append(f"{name} ({impact:.1f})")
    return total_impact, details

ergebnisse = []
for _, spiel in vorhersagen_df.iterrows():
    heim = spiel["Heimteam"]
    auswaerts = spiel["Auswärtsteam"]
    heim_impact, heim_details = berechne_impact_verlust(heim, verletzungen_df, kombiniert)
    auswaerts_impact, auswaerts_details = berechne_impact_verlust(auswaerts, verletzungen_df, kombiniert)
    diff = (auswaerts_impact - heim_impact) * 0.01
    neue_prob = min(0.95, max(0.05, spiel["Heimsieg %"] / 100 + diff))
    gewinner = heim if neue_prob > 0.5 else auswaerts
    konfidenz = neue_prob if neue_prob > 0.5 else 1 - neue_prob
    ergebnisse.append({
        "Heimteam": heim,
        "Auswärtsteam": auswaerts,
        "Heimsieg %": round(neue_prob * 100, 1),
        "Tipp": gewinner,
        "Konfidenz": round(konfidenz * 100, 1)
    })

finale_df = pd.DataFrame(ergebnisse)
print("\n🏀 FINALE VORHERSAGEN MIT VERLETZUNGSKORREKTUR")
print("=" * 65)
print(finale_df.to_string(index=False))

# ─── SCHRITT 11: CSV speichern ─────────────────────────────────────────────────

dateiname = f"predictions_{heute}.csv"
finale_df.to_csv(dateiname, index=False)
print(f"\n✅ Gespeichert als {dateiname}")
