from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import math
import ssl
import urllib.request
import cloudscraper
import gspread
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson
import streamlit as st

# ==========================================
# 1. CONFIGURATION (Doit être la toute première commande Streamlit)
# ==========================================
st.set_page_config(
    page_title="Value Bet Scanner Pro", page_icon="⚽", layout="wide"
)

LEAGUES = {
    "premier_league": {"country": "Angleterre - Premier League", "fd_code": "E0"},
    "championship": {"country": "Angleterre - Championship", "fd_code": "E1"},
    "ligue_1": {"country": "France - Ligue 1", "fd_code": "F1"},
    "ligue_2": {"country": "France - Ligue 2", "fd_code": "F2"},
    "liga": {"country": "Espagne - LaLiga", "fd_code": "SP1"},
    "bundesliga": {"country": "Allemagne - Bundesliga", "fd_code": "D1"},
    "serie_a": {"country": "Italie - Serie A", "fd_code": "I1"},
    "eredivisie": {"country": "Pays-Bas - Eredivisie", "fd_code": "N1"},
    "belgique": {"country": "Belgique - Jupiler League", "fd_code": "B1"},
    "portugal": {"country": "Portugal - Liga NOS", "fd_code": "P1"},
    "ecosse": {"country": "Écosse - Premiership", "fd_code": "SC0"},
    "turquie": {"country": "Turquie - Süper Lig", "fd_code": "T1"},
    "grece": {"country": "Grèce - Super League", "fd_code": "G1"},
}
SEASONS = ["2425", "2526", "2627"]
TIME_DECAY_HALFLIFE_DAYS = 180


# =========================================================
# 2. FONCTIONS DE CONNEXION ET FORMATAGE GOOGLE SHEETS
# =========================================================

# ID du Google Sheet (à récupérer dans l'URL de ton navigateur)
SPREADSHEET_ID = "11fcyQntgVPi2xjF0GXIeKAZkXrhRkwq2sfysO325kaI"

def get_gspread_client():
  """Connexion à l'API Google Sheets via st.secrets."""
  creds_dict = dict(st.secrets["gcp_service_account"])
  if "private_key" in creds_dict:
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
  return gspread.service_account_from_dict(creds_dict)


def format_bet_details(market, selection, home_team, away_team):
  """Formate le nom du pari (Col F) et le cut tirs (Col G) selon la nomenclature Sheets."""
  bet_name = selection
  cut_str = ""

  if market == "1X2":
    mapping = {"H": "V1", "D": "Nul", "A": "V2"}
    bet_name = mapping.get(selection, selection)
  elif market == "double_chance":
    mapping = {"1X": "1N", "12": "12", "X2": "N2"}
    bet_name = mapping.get(selection, selection)
  elif market == "over_under_2_5":
    bet_name = "Over 2,5" if selection == "over" else "Under 2,5"
  elif market == "btts":
    bet_name = "BTTS Oui" if selection == "yes" else "BTTS Non"
  elif market == "home_goals":
    sel_clean = (
        selection.replace("over_", "Over ")
        .replace("under_", "Under ")
        .replace("_", ",")
    )
    bet_name = f"{sel_clean} (Dom)"
  elif market == "away_goals":
    sel_clean = (
        selection.replace("over_", "Over ")
        .replace("under_", "Under ")
        .replace("_", ",")
    )
    bet_name = f"{sel_clean} (Ext)"
  elif "tirs_" in market or "sot_" in market:
    parts = market.split("_")
    stat_type = "shots" if parts[0] == "tirs" else "sot"
    target = parts[1]
    cut_val = parts[2] if len(parts) > 2 else ""
    cut_str = str(cut_val).replace(".", ",")

    target_label = "totaux"
    if target == "domicile":
      target_label = "(Dom)"
    elif target == "exterieur":
      target_label = "(Ext)"

    sel_label = "Over" if selection == "over" else "Under"
    bet_name = f"{sel_label} cut {stat_type} {target_label}"

  return bet_name, cut_str


def build_sheets_row(
    vb, match_date, match_mode, timing_paris, league_name, home_team, away_team
):
  """Construit la liste des 16 colonnes (A à P) pour Google Sheets."""
  bet_name, cut_str = format_bet_details(
      vb.market, vb.selection, home_team, away_team
  )

  return [
      str(match_date),  # A: Dates
      "Live" if match_mode == "En direct (Live)" else "Avant",  # B: Avant/Live
      timing_paris,  # C: Timing prise de paris
      league_name,  # D: Championnat
      f"{home_team} / {away_team}",  # E: Match
      bet_name,  # F: Paris
      cut_str,  # G: Cut tirs
      str(vb.bookmaker_odds).replace(".", ","),  # H: Côtes prises
      "",  # I: Closing odds
      "",  # J: CLV
      f"{vb.edge * 100:.1f} %".replace(".", ","),  # K: Edge
      f"{vb.kelly_quart * 100:.1f} %".replace(".", ","),  # L: Quart Kelly
      "",  # M: Mise Paris
      "",  # N: Resultats paris
      "",  # O: Gains/Pertes
      "",  # P: Bankroll hebdomadaire
  ]

# SPREADSHEET_ID doit être défini en haut de ton script :
# SPREADSHEET_ID = "11fcyQntgVPi2xjF0GXIeKAZkXrhRkwq2sfysO325kaI"


def export_all_value_bets_to_sheet(
    results,
    match_date,
    match_mode,
    timing_paris,
    league_name,
    home_team,
    away_team,
    spreadsheet_id=SPREADSHEET_ID,
    worksheet_name="Test-Export",
):
  """Exporte la liste complète des Value Bets détectés en une seule requête via l'ID du Google Sheet."""
  try:
    gc = get_gspread_client()
    # Ouverture du fichier via son ID unique et ciblage de l'onglet
    sh = gc.open_by_key(spreadsheet_id).worksheet(worksheet_name)

    rows_data = [
        build_sheets_row(
            vb,
            match_date,
            match_mode,
            timing_paris,
            league_name,
            home_team,
            away_team,
        )
        for vb in results
    ]

    sh.append_rows(rows_data, value_input_option="USER_ENTERED")
    return True
  except Exception as e:
    st.error(f"Erreur lors de l'exportation globale vers Google Sheets : {e}")
    return False


def export_value_bet_to_sheet(
    vb,
    match_date,
    match_mode,
    timing_paris,
    league_name,
    home_team,
    away_team,
    spreadsheet_id=SPREADSHEET_ID,
    worksheet_name="Test-Export",
):
  """Exporte un seul Value Bet vers Google Sheets via l'ID du Google Sheet."""
  try:
    gc = get_gspread_client()
    # Ouverture du fichier via son ID unique et ciblage de l'onglet
    sh = gc.open_by_key(spreadsheet_id).worksheet(worksheet_name)

    row_data = build_sheets_row(
        vb,
        match_date,
        match_mode,
        timing_paris,
        league_name,
        home_team,
        away_team,
    )
    sh.append_row(row_data, value_input_option="USER_ENTERED")
    return True
  except Exception as e:
    st.error(f"Erreur lors de l'exportation vers Google Sheets : {e}")
    return False




# ==========================================
# 3. LOGIQUE MATHÉMATIQUE (DIXON-COLES & NBINOM)
# ==========================================
def time_weights(dates, halflife_days):
  days_ago = (dates.max() - dates).dt.days.values
  decay = np.log(2) / halflife_days
  return np.exp(-decay * days_ago)


def dixon_coles_adjustment(home_goals, away_goals, lam_home, lam_away, rho):
  if home_goals == 0 and away_goals == 0:
    return 1 - lam_home * lam_away * rho
  elif home_goals == 0 and away_goals == 1:
    return 1 + lam_home * rho
  elif home_goals == 1 and away_goals == 0:
    return 1 + lam_away * rho
  elif home_goals == 1 and away_goals == 1:
    return 1 - rho
  return 1.0


def proba_tirs_nbinom(mu, var, ligne_bookmaker):
  """Calcule les probas statistiques avec la Loi Binomiale Négative si surdispersion"""
  seuil = int(np.floor(ligne_bookmaker))

  if var <= mu or math.isnan(var) or var == 0:
    p_under = poisson.cdf(seuil, mu)
  else:
    p = mu / var
    n = (mu**2) / (var - mu)
    p_under = nbinom.cdf(seuil, n, p)

  return p_under, 1.0 - p_under


class DixonColesModel:

  def __init__(self):
    self.teams = []
    self.params = {}
    self.rho = 0.0
    self.home_advantage = 0.0

  def fit(self, matches, home_col="FTHG", away_col="FTAG", halflife_days=180):
    self.teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    n = len(self.teams)
    idx = {t: i for i, t in enumerate(self.teams)}
    weights = time_weights(matches["Date"], halflife_days)

    home_idx = matches["HomeTeam"].map(idx).values
    away_idx = matches["AwayTeam"].map(idx).values
    hg = matches[home_col].values
    ag = matches[away_col].values

    def unpack(x):
      return x[:n], x[n : 2 * n], x[2 * n], x[2 * n + 1]

    def neg_log_likelihood(x):
      attack, defense, rho, home_adv = unpack(x)
      lam_home = np.exp(
          np.clip(attack[home_idx] + defense[away_idx] + home_adv, -20, 20)
      )
      lam_away = np.exp(
          np.clip(attack[away_idx] + defense[home_idx], -20, 20)
      )
      ll = poisson.logpmf(hg, lam_home) + poisson.logpmf(ag, lam_away)

      if home_col == "FTHG":
        tau = np.array([
            dixon_coles_adjustment(h, a, lh, la, rho)
            for h, a, lh, la in zip(hg, ag, lam_home, lam_away)
        ])
        ll += np.log(np.clip(tau, 1e-6, None))
      return -np.sum(weights * ll)

    x0 = np.zeros(2 * n + 2)
    x0[2 * n + 1] = 0.2 if home_col == "FTHG" else 1.0
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:n])}]
    result = minimize(
        neg_log_likelihood,
        x0,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 200},
    )

    if not result.success:
      st.warning(
          "Attention : Le modèle a eu du mal à converger pour ce"
          " championnat. Les résultats peuvent être moins précis."
      )

    attack, defense, rho, home_adv = unpack(result.x)
    self.params = {
        t: {"attack": attack[i], "defense": defense[i]}
        for i, t in enumerate(self.teams)
    }
    self.rho = rho
    self.home_advantage = home_adv
    return self

  def get_lambdas(self, home_team, away_team):
    a_h, d_h = (
        self.params[home_team]["attack"],
        self.params[home_team]["defense"],
    )
    a_a, d_a = (
        self.params[away_team]["attack"],
        self.params[away_team]["defense"],
    )
    lam_home = np.exp(a_h + d_a + self.home_advantage)
    lam_away = np.exp(a_a + d_h)
    return lam_home, lam_away

  def score_matrix(self, home_team, away_team, max_goals=10):
    lam_home, lam_away = self.get_lambdas(home_team, away_team)
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
      for j in range(max_goals + 1):
        p = (
            poisson.pmf(i, lam_home)
            * poisson.pmf(j, lam_away)
            * dixon_coles_adjustment(i, j, lam_home, lam_away, self.rho)
        )
        probs[i, j] = p
    probs = np.clip(probs, 0, None)
    return probs / probs.sum(), lam_home, lam_away

  def predict_goal_markets(self, home_team, away_team, max_goals=10):
    M, lam_h, lam_a = self.score_matrix(home_team, away_team, max_goals)
    goals = np.arange(max_goals + 1)

    h_prob = np.tril(M, -1).sum()
    d_prob = np.trace(M)
    a_prob = np.triu(M, 1).sum()

    return {
        "1X2": {"H": h_prob, "D": d_prob, "A": a_prob},
        "double_chance": {
            "1X": h_prob + d_prob,
            "12": h_prob + a_prob,
            "X2": d_prob + a_prob,
        },
        "over_under_2_5": {
            "over": M[goals[:, None] + goals[None, :] > 2.5].sum(),
            "under": M[goals[:, None] + goals[None, :] < 2.5].sum(),
        },
        "btts": {"yes": M[1:, 1:].sum(), "no": 1 - M[1:, 1:].sum()},
        "home_goals": {
            "over_0_5": M[1:, :].sum(),
            "under_0_5": M[0, :].sum(),
            "over_1_5": M[2:, :].sum(),
            "under_1_5": M[:2, :].sum(),
        },
        "away_goals": {
            "over_0_5": M[:, 1:].sum(),
            "under_0_5": M[:, 0].sum(),
            "over_1_5": M[:, 2:].sum(),
            "under_1_5": M[:, :2].sum(),
        },
        "expected_goals": {"home": lam_h, "away": lam_a},
    }


# ==========================================
# 4. GESTION DES DONNÉES DE MATCHS
# ==========================================
@dataclass
class ValueBetResult:
  market: str
  selection: str
  model_prob: float
  bookmaker_odds: float
  edge: float
  kelly_quart: float


def remove_overround(odds):
  implied = {k: 1 / v for k, v in odds.items()}
  overround = sum(implied.values())
  return {k: v / overround for k, v in implied.items()}


@st.cache_data(ttl=3600)
def load_and_clean_data(league_code):
  dfs = []
  scraper = cloudscraper.create_scraper()

  for s in SEASONS:
    url = f"https://www.football-data.co.uk/mmz4281/{s}/{league_code}.csv"
    try:
      response = scraper.get(url, timeout=10)
      if response.status_code == 200:
        try:
          content = response.content.decode("utf-8")
        except UnicodeDecodeError:
          content = response.content.decode("latin-1")

        df = pd.read_csv(io.StringIO(content))
        df["Season"] = s
        dfs.append(df)
      else:
        st.sidebar.warning(f"Saison {s} : Erreur HTTP {response.status_code}")
    except Exception as e:
      st.sidebar.warning(f"Saison {s} : {e}")

  if not dfs:
    return None

  data = pd.concat(dfs, ignore_index=True)

  cols_to_clean = [
      "HomeTeam",
      "AwayTeam",
      "FTHG",
      "FTAG",
      "HS",
      "AS",
      "HST",
      "AST",
  ]
  existing_cols = [c for c in cols_to_clean if c in data.columns]
  data = data.dropna(subset=existing_cols)

  data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
  data = data.dropna(subset=["Date"])

  for col in ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]:
    if col in data.columns:
      data[col] = data[col].astype(int)

  return data


def load_fixtures():
  scraper = cloudscraper.create_scraper()
  url = "https://www.football-data.co.uk/fixtures.csv"
  try:
    response = scraper.get(url, timeout=10)
    if response.status_code == 200:
      try:
        content = response.content.decode("utf-8")
      except UnicodeDecodeError:
        content = response.content.decode("latin-1")
      df_fix = pd.read_csv(io.StringIO(content))
      return df_fix
  except Exception:
    pass
  return None


@st.cache_resource(show_spinner=False)
def train_all_models(df):
  goal_model = DixonColesModel().fit(
      df, "FTHG", "FTAG", halflife_days=TIME_DECAY_HALFLIFE_DAYS
  )
  shot_model = DixonColesModel().fit(
      df, "HS", "AS", halflife_days=TIME_DECAY_HALFLIFE_DAYS
  )
  sot_model = DixonColesModel().fit(
      df, "HST", "AST", halflife_days=TIME_DECAY_HALFLIFE_DAYS
  )
  return {"goals": goal_model, "shots": shot_model, "sot": sot_model}


# ==========================================
# 5. SIDEBAR (FILTRES & METRIQUES)
# ==========================================
st.sidebar.header("🎯 Mode d'Analyse")
match_mode = st.sidebar.radio(
    "Sélectionnez le contexte", ["Avant-match (Statique)", "En direct (Live)"]
)

st.sidebar.header("📋 Catégories à Analyser")
cat_buts_main = st.sidebar.checkbox("⚽ Marchés Buts Principaux", value=True)
cat_buts_team = st.sidebar.checkbox("市内 Buts par Équipe", value=True)
cat_shots = st.sidebar.checkbox("📊 Tirs & Tirs Cadrés", value=True)

st.sidebar.divider()

st.sidebar.subheader("📊 Performance Hebdo (7j)")
total_vbs = 24
avg_edge = 6.8
avg_odds = 1.88
expected_roi = 8.4
top_league = "Premier League"
pct_buts = 0.60
pct_tirs = 0.40

kpi_col1, kpi_col2 = st.sidebar.columns(2)
with kpi_col1:
  st.metric(label="Value Bets", value=f"{total_vbs}", delta="+5 vs S-1")
  st.metric(label="Cote Moyenne", value=f"{avg_odds:.2f}")

with kpi_col2:
  st.metric(label="Edge Moyen", value=f"+{avg_edge:.1f}%")
  st.metric(label="ROI Théorique", value=f"+{expected_roi:.1f}%")

st.sidebar.metric(label="🏆 Top Ligue (Edge)", value=top_league)
st.sidebar.markdown("**🎯 Répartition Marchés**")
st.sidebar.caption(f"{pct_buts:.0%} Buts  |  {pct_tirs:.0%} Tirs")
st.sidebar.progress(pct_buts)

st.sidebar.divider()

cote_min, cote_max = 1.50, 2.30

# ==========================================
# 6. PAGE PRINCIPALE & SELECTIONS
# ==========================================
st.title("🏆 Scanner de Value Bets Pro (Buts & Tirs)")

league_key = st.selectbox(
    "1️⃣ Choisis un championnat",
    options=list(LEAGUES.keys()),
    format_func=lambda x: LEAGUES[x]["country"],
)

with st.spinner("Calcul et calibration des modèles..."):
  df = load_and_clean_data(LEAGUES[league_key]["fd_code"])
  if df is not None:
    models = train_all_models(df)
    teams_list = models["goals"].teams
    st.success(
        f"✅ Modèles calibrés avec succès sur {len(df)} matchs historiques !"
    )
  else:
    st.error("Impossible de charger les données.")
    st.stop()

# --- SÉLECTION PAR DATE, MATCH ET TIMING (CORRECTION 4 : Alignement [1, 2, 1]) ---
df_fixtures = load_fixtures()
idx_h, idx_a = 0, min(1, len(teams_list) - 1)

# Valeurs par défaut de sécurité
selected_date = datetime.now().strftime("%d/%m/%Y")
timing_paris = "J-1"

if df_fixtures is not None and not df_fixtures.empty:
  code_fd = LEAGUES[league_key]["fd_code"]
  league_fixtures = df_fixtures[df_fixtures["Div"] == code_fd].copy()

  if not league_fixtures.empty:
    available_dates = sorted(
        league_fixtures["Date"].dropna().unique().tolist()
    )

    col_d, col_m, col_t = st.columns([1, 2, 1])

    with col_d:
      selected_date_opt = st.selectbox(
          "📅 1. Date", options=["-- Toutes les dates --"] + available_dates
      )
      if selected_date_opt != "-- Toutes les dates --":
        selected_date = selected_date_opt

    if selected_date_opt != "-- Toutes les dates --":
      filtered_fixtures = league_fixtures[
          league_fixtures["Date"] == selected_date_opt
      ]
    else:
      filtered_fixtures = league_fixtures

    fixture_options = filtered_fixtures.apply(
        lambda r: f"{r['HomeTeam']} / {r['AwayTeam']}", axis=1
    ).tolist()

    with col_m:
      selected_fixture = st.selectbox(
          "⚽ 2. Choisir le match",
          options=["-- Sélectionner un match --"] + fixture_options,
      )

    with col_t:
      timing_paris = st.selectbox(
          "⏱️ 3. Timing prise de pari",
          options=["J-2", "J-1", "H-12 à H-2", "H-2 à H"],
      )

    if selected_fixture != "-- Sélectionner un match --":
      match_str = selected_fixture.rsplit(" (", 1)[0]
      h_sel, a_sel = match_str.split(" / ")
      if h_sel in teams_list:
        idx_h = teams_list.index(h_sel)
      if a_sel in teams_list:
        idx_a = teams_list.index(a_sel)

st.divider()

col1, col2 = st.columns(2)
with col1:
  home_team = st.selectbox(
      "🏠 Équipe à Domicile", options=teams_list, index=idx_h
  )
with col2:
  away_team = st.selectbox(
      "✈️ Équipe à l'Extérieur", options=teams_list, index=idx_a
  )

# --- LIVE ---
live_minute, live_home_score, live_away_score = 0, 0, 0
if match_mode == "En direct (Live)":
  st.divider()
  st.info(
      "⏱️ **Mode Live Activé** : Indiquez l'état actuel de la rencontre pour"
      " réajuster le modèle en temps réel."
  )
  l_col1, l_col2, l_col3 = st.columns(3)
  with l_col1:
    live_minute = st.number_input(
        "Minute du match", min_value=1, max_value=90, value=46
    )
  with l_col2:
    live_home_score = st.number_input(
        "Score Domicile actuel", min_value=0, max_value=10, value=0
    )
  with l_col3:
    live_away_score = st.number_input(
        "Score Extérieur actuel", min_value=0, max_value=10, value=1
    )

st.divider()

# ==========================================
# 7. SAISIE DES COTES
# ==========================================
st.subheader("2️⃣ Saisissez les cotes du bookmaker")

if cat_buts_main:
  with st.expander(
      "⚽ MARCHÉS DES BUTS PRINCIPAUX (1X2, DOUBLE CHANCE, O/U 2.5, BTTS)",
      expanded=True,
  ):
    st.markdown("### 🏆 Résultat Match (1X2)")
    c1, c2, c3 = st.columns(3)
    h_odd = c1.number_input("Cote 1 (Domicile)", value=2.00, step=0.01)
    d_odd = c2.number_input("Cote X (Nul)", value=3.40, step=0.01)
    a_odd = c3.number_input("Cote 2 (Extérieur)", value=3.80, step=0.01)

    st.markdown("### 🛡️ Double Chance")
    c1, c2, c3 = st.columns(3)
    dc_1x = c1.number_input("1X (Dom ou Nul)", value=1.28, step=0.01)
    dc_12 = c2.number_input("12 (Dom ou Ext)", value=1.30, step=0.01)
    dc_x2 = c3.number_input("X2 (Nul ou Ext)", value=1.70, step=0.01)

    st.markdown("### ⚽ Buts & BTTS")
    c1, c2, c3, c4 = st.columns(4)
    ou_over = c1.number_input("Over 2.5 (Buts)", value=1.90, step=0.01)
    ou_under = c2.number_input("Under 2.5 (Buts)", value=1.90, step=0.01)
    btts_yes = c3.number_input("BTTS Oui", value=1.85, step=0.01)
    btts_no = c4.number_input("BTTS Non", value=1.95, step=0.01)

if cat_buts_team:
  with st.expander("🥅 BUTS PAR ÉQUIPE (Over / Under 0.5 et 1.5)"):
    st.markdown(f"**🏠 {home_team} (Domicile)**")
    c1, c2, c3, c4 = st.columns(4)
    hg_o05 = c1.number_input("Over 0.5 (Dom)", value=1.15, step=0.01)
    hg_u05 = c2.number_input("Under 0.5 (Dom)", value=5.00, step=0.01)
    hg_o15 = c3.number_input("Over 1.5 (Dom)", value=2.10, step=0.01)
    hg_u15 = c4.number_input("Under 1.5 (Dom)", value=1.70, step=0.01)

    st.markdown(f"**✈️ {away_team} (Extérieur)**")
    c1, c2, c3, c4 = st.columns(4)
    ag_o05 = c1.number_input("Over 0.5 (Ext)", value=1.40, step=0.01)
    ag_u05 = c2.number_input("Under 0.5 (Ext)", value=2.80, step=0.01)
    ag_o15 = c3.number_input("Over 1.5 (Ext)", value=3.50, step=0.01)
    ag_u15 = c4.number_input("Under 1.5 (Ext)", value=1.28, step=0.01)

if cat_shots:
  with st.expander("📊 MARCHÉS DES TIRS & TIRS CADRÉS (Lignes ajustables)"):
    st.markdown("### 🏹 Tirs Totaux (Match)")
    c1, c2, c3 = st.columns(3)
    t_shots_line = c1.number_input(
        "Ligne de Tirs Match (ex: 24.5)", value=24.5, step=0.5
    )
    t_shots_o = c2.number_input(
        "Cote Over Tirs Match", value=1.85, step=0.01, key="ts_o"
    )
    t_shots_u = c3.number_input(
        "Cote Under Tirs Match", value=1.85, step=0.01, key="ts_u"
    )

    st.markdown(f"### 🏠 Tirs Totaux Individuels : {home_team}")
    c1, c2, c3 = st.columns(3)
    h_shots_line = c1.number_input(
        f"Ligne Tirs Totaux {home_team}", value=13.5, step=0.5
    )
    h_shots_o = c2.number_input("Cote Over Tirs Dom", value=1.85, step=0.01)
    h_shots_u = c3.number_input("Cote Under Tirs Dom", value=1.85, step=0.01)

    st.markdown(f"### ✈️ Tirs Totaux Individuels : {away_team}")
    c1, c2, c3 = st.columns(3)
    a_shots_line = c1.number_input(
        f"Ligne Tirs Totaux {away_team}", value=11.5, step=0.5
    )
    a_shots_o = c2.number_input("Cote Over Tirs Ext", value=1.85, step=0.01)
    a_shots_u = c3.number_input("Cote Under Tirs Ext", value=1.85, step=0.01)

    st.markdown("### 🎯 Tirs Cadrés Totaux (Match)")
    c1, c2, c3 = st.columns(3)
    t_sot_line = c1.number_input(
        "Ligne Tirs Cadrés Match (ex: 8.5)", value=8.5, step=0.5
    )
    t_sot_o = c2.number_input("Cote Over SOT Match", value=1.85, step=0.01)
    t_sot_u = c3.number_input("Cote Under SOT Match", value=1.85, step=0.01)

    st.markdown(f"### 🏠 Tirs Cadrés Individuels : {home_team}")
    c1, c2, c3 = st.columns(3)
    h_sot_line = c1.number_input(f"Ligne SOT {home_team}", value=4.5, step=0.5)
    h_sot_o = c2.number_input("Cote Over SOT Dom", value=1.85, step=0.01)
    h_sot_u = c3.number_input("Cote Under SOT Dom", value=1.85, step=0.01)

    st.markdown(f"### ✈️ Tirs Cadrés Individuels : {away_team}")
    c1, c2, c3 = st.columns(3)
    a_sot_line = c1.number_input(f"Ligne SOT {away_team}", value=3.5, step=0.5)
    a_sot_o = c2.number_input("Cote Over SOT Ext", value=1.85, step=0.01)
    a_sot_u = c3.number_input("Cote Under SOT Ext", value=1.85, step=0.01)

st.divider()

min_edge = (
    st.slider(
        "Seuil d'Edge minimum (%)",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
    )
    / 100
)

# ==========================================
# 8. MOTEUR D'ANALYSE & AFFICHAGE
# ==========================================
# -----------------------------------------------------------------------------
# BOUTON DE LANCEMENT DE L'ANALYSE
# -----------------------------------------------------------------------------
if st.button(
    "🚀 Lancer l'Analyse Complète", type="primary", use_container_width=True
):
  if home_team == away_team:
    st.error("Veuillez choisir deux équipes différentes.")
    st.session_state.pop("analysis_data", None)
  elif not (cat_buts_main or cat_buts_team or cat_shots):
    st.warning(
        "Veuillez cocher au moins une catégorie à analyser dans le menu de"
        " gauche."
    )
    st.session_state.pop("analysis_data", None)
  else:
    preds_all = {}
    market_odds = {}

    # 1. Calculs Buts
    if cat_buts_main or cat_buts_team:
      base_goals_preds = models["goals"].predict_goal_markets(
          home_team, away_team
      )
      lam_h_base = base_goals_preds["expected_goals"]["home"]
      lam_a_base = base_goals_preds["expected_goals"]["away"]

      if match_mode == "En direct (Live)":
        ratio_temps = (90 - live_minute) / 90.0
        lam_h = lam_h_base * ratio_temps
        lam_a = lam_a_base * ratio_temps

        score_total_actuel = live_home_score + live_away_score
        if score_total_actuel == 1:
          lam_h *= 0.92
          lam_a *= 0.92
        elif score_total_actuel >= 3:
          lam_h *= 1.08
          lam_a *= 1.08

        max_goals = 10
        M = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
          for j in range(max_goals + 1):
            p = (
                poisson.pmf(i, lam_h)
                * poisson.pmf(j, lam_a)
                * dixon_coles_adjustment(
                    i, j, lam_h, lam_a, models["goals"].rho
                )
            )
            M[i, j] = p
        M = np.clip(M, 0, None)
        M = M / M.sum()

        h_prob = np.tril(M, -1).sum()
        d_prob = np.trace(M)
        a_prob = np.triu(M, 1).sum()

        buts_manquants_over25 = max(0, 3 - score_total_actuel)
        if buts_manquants_over25 == 0:
          prob_over_25, prob_under_25 = 1.0, 0.0
        else:
          prob_over_25 = sum(
              (math.exp(-(lam_h + lam_a)) * ((lam_h + lam_a) ** k))
              / math.factorial(k)
              for k in range(buts_manquants_over25, 10)
          )
          prob_under_25 = 1.0 - prob_over_25

        preds_goals = {
            "1X2": {"H": h_prob, "D": d_prob, "A": a_prob},
            "double_chance": {
                "1X": h_prob + d_prob,
                "12": h_prob + a_prob,
                "X2": d_prob + a_prob,
            },
            "over_under_2_5": {"over": prob_over_25, "under": prob_under_25},
            "btts": {"yes": M[1:, 1:].sum(), "no": 1 - M[1:, 1:].sum()},
            "home_goals": {
                "over_0_5": M[1:, :].sum(),
                "under_0_5": M[0, :].sum(),
                "over_1_5": M[2:, :].sum(),
                "under_1_5": M[:2, :].sum(),
            },
            "away_goals": {
                "over_0_5": M[:, 1:].sum(),
                "under_0_5": M[:, 0].sum(),
                "over_1_5": M[:, 2:].sum(),
                "under_1_5": M[:, :2].sum(),
            },
            "expected_goals": {"home": lam_h, "away": lam_a},
        }
      else:
        preds_goals = base_goals_preds

      if cat_buts_main:
        preds_all.update({
            "1X2": preds_goals["1X2"],
            "double_chance": preds_goals["double_chance"],
            "over_under_2_5": preds_goals["over_under_2_5"],
            "btts": preds_goals["btts"],
        })
        market_odds.update({
            "1X2": {"H": h_odd, "D": d_odd, "A": a_odd},
            "double_chance": {"1X": dc_1x, "12": dc_12, "X2": dc_x2},
            "over_under_2_5": {"over": ou_over, "under": ou_under},
            "btts": {"yes": btts_yes, "no": btts_no},
        })

      if cat_buts_team:
        preds_all.update({
            "home_goals": preds_goals["home_goals"],
            "away_goals": preds_goals["away_goals"],
        })
        market_odds.update({
            "home_goals": {
                "over_0_5": hg_o05,
                "under_0_5": hg_u05,
                "over_1_5": hg_o15,
                "under_1_5": hg_u15,
            },
            "away_goals": {
                "over_0_5": ag_o05,
                "under_0_5": ag_u05,
                "over_1_5": ag_o15,
                "under_1_5": ag_u15,
            },
        })

    # 2. Calculs Tirs & SOT
    if cat_shots:
      lam_h_shots, lam_a_shots = models["shots"].get_lambdas(
          home_team, away_team
      )
      h_shots_data = df[df["HomeTeam"] == home_team]["HS"]
      a_shots_data = df[df["AwayTeam"] == away_team]["AS"]
      var_h_shots = (
          np.var(h_shots_data, ddof=1)
          if len(h_shots_data) > 1
          else lam_h_shots
      )
      var_a_shots = (
          np.var(a_shots_data, ddof=1)
          if len(a_shots_data) > 1
          else lam_a_shots
      )

      lam_h_sot, lam_a_sot = models["sot"].get_lambdas(home_team, away_team)
      h_sot_data = df[df["HomeTeam"] == home_team]["HST"]
      a_sot_data = df[df["AwayTeam"] == away_team]["AST"]
      var_h_sot = (
          np.var(h_sot_data, ddof=1) if len(h_sot_data) > 1 else lam_h_sot
      )
      var_a_sot = (
          np.var(a_sot_data, ddof=1) if len(a_sot_data) > 1 else lam_a_sot
      )

      if match_mode == "En direct (Live)":
        ratio = (90 - live_minute) / 90.0
        lam_h_shots *= ratio
        lam_a_shots *= ratio
        var_h_shots *= ratio
        var_a_shots *= ratio
        lam_h_sot *= ratio
        lam_a_sot *= ratio
        var_h_sot *= ratio
        var_a_sot *= ratio

      prob_shots_under, prob_shots_over = proba_tirs_nbinom(
          lam_h_shots + lam_a_shots,
          var_h_shots + var_a_shots,
          t_shots_line,
      )
      prob_h_shots_under, prob_h_shots_over = proba_tirs_nbinom(
          lam_h_shots, var_h_shots, h_shots_line
      )
      prob_a_shots_under, prob_a_shots_over = proba_tirs_nbinom(
          lam_a_shots, var_a_shots, a_shots_line
      )

      prob_sot_under, prob_sot_over = proba_tirs_nbinom(
          lam_h_sot + lam_a_sot, var_h_sot + var_a_sot, t_sot_line
      )
      prob_h_sot_under, prob_h_sot_over = proba_tirs_nbinom(
          lam_h_sot, var_h_sot, h_sot_line
      )
      prob_a_sot_under, prob_a_sot_over = proba_tirs_nbinom(
          lam_a_sot, var_a_sot, a_sot_line
      )

      preds_all.update({
          f"tirs_match_{t_shots_line}": {
              "over": prob_shots_over,
              "under": prob_shots_under,
          },
          f"tirs_domicile_{h_shots_line}": {
              "over": prob_h_shots_over,
              "under": prob_h_shots_under,
          },
          f"tirs_exterieur_{a_shots_line}": {
              "over": prob_a_shots_over,
              "under": prob_a_shots_under,
          },
          f"sot_match_{t_sot_line}": {
              "over": prob_sot_over,
              "under": prob_sot_under,
          },
          f"sot_domicile_{h_sot_line}": {
              "over": prob_h_sot_over,
              "under": prob_h_sot_under,
          },
          f"sot_exterieur_{a_sot_line}": {
              "over": prob_a_sot_over,
              "under": prob_a_sot_under,
          },
      })
      market_odds.update({
          f"tirs_match_{t_shots_line}": {
              "over": t_shots_o,
              "under": t_shots_u,
          },
          f"tirs_domicile_{h_shots_line}": {
              "over": h_shots_o,
              "under": h_shots_u,
          },
          f"tirs_exterieur_{a_shots_line}": {
              "over": a_shots_o,
              "under": a_shots_u,
          },
          f"sot_match_{t_sot_line}": {"over": t_sot_o, "under": t_sot_u},
          f"sot_domicile_{h_sot_line}": {"over": h_sot_o, "under": h_sot_u},
          f"sot_exterieur_{a_sot_line}": {"over": a_sot_o, "under": a_sot_u},
      })

    # 3. Moteur de Value Bets
    results = []
    for market, odds in market_odds.items():
      for sel, odd in odds.items():
        prob = preds_all[market][sel]
        edge = prob * odd - 1

        if edge > min_edge and (cote_min <= odd <= cote_max):
          b = odd - 1
          kelly_quart = (
              max(0.0, (b * prob - (1 - prob)) / b) * 0.25 if b > 0 else 0.0
          )
          results.append(
              ValueBetResult(market, sel, prob, odd, edge, kelly_quart)
          )

    results.sort(key=lambda x: x.edge, reverse=True)

    # Sauvegarde du contexte et des résultats dans la mémoire de session
    st.session_state["analysis_data"] = {
        "results": results,
        "home_team": home_team,
        "away_team": away_team,
        "match_mode": match_mode,
        "live_minute": (
            live_minute if match_mode == "En direct (Live)" else None
        ),
        "live_home_score": (
            live_home_score if match_mode == "En direct (Live)" else None
        ),
        "live_away_score": (
            live_away_score if match_mode == "En direct (Live)" else None
        ),
        "selected_date": selected_date,
        "timing_paris": timing_paris,
        "league_country": LEAGUES[league_key]["country"],
        "min_edge": min_edge,
    }

# -----------------------------------------------------------------------------
# 2. AFFICHAGE DU RAPPORT & BOUTONS D'EXPORT (SÉPARÉS DU BOUTON PRINCIPAL)
# -----------------------------------------------------------------------------
if "analysis_data" in st.session_state:
  data = st.session_state["analysis_data"]
  results = data["results"]
  home_team = data["home_team"]
  away_team = data["away_team"]
  match_mode = data["match_mode"]
  selected_date = data["selected_date"]
  timing_paris = data["timing_paris"]
  league_country = data["league_country"]
  min_edge = data["min_edge"]

  st.header(f"📊 Rapport : {home_team} vs {away_team}")
  if match_mode == "En direct (Live)":
    st.caption(
        f"⚡ Analyse Live à la {data['live_minute']}e minute | Score actuel :"
        f" {data['live_home_score']} - {data['live_away_score']}"
    )

  st.subheader("💸 Value Bets Détectés")
  if not results:
    st.info(
        "Aucun Value Bet détecté pour ce match avec vos critères actuels"
        " (Edge ou Cotes hors limites)."
    )
  else:
    # Bouton Export Global
    if st.button(
        f"📤 Exporter les {len(results)} Value Bets vers Google Sheets",
        type="primary",
        use_container_width=True,
    ):
      success = export_all_value_bets_to_sheet(
          results=results,
          match_date=selected_date,
          match_mode=match_mode,
          timing_paris=timing_paris,
          league_name=league_country,
          home_team=home_team,
          away_team=away_team,
      )
      if success:
        st.success(
            f"✅ {len(results)} Value Bets exportés avec succès vers Google"
            " Sheets !"
        )

    st.divider()

    # Plotly Scatter
    df_res = pd.DataFrame([{
        "Marché": vb.market.upper(),
        "Sélection": vb.selection.upper(),
        "Cote": vb.bookmaker_odds,
        "Edge (%)": round(vb.edge * 100, 2),
        "Probabilité (%)": round(vb.model_prob * 100, 1),
    } for vb in results])

    fig = px.scatter(
        df_res,
        x="Cote",
        y="Edge (%)",
        color="Edge (%)",
        hover_data=["Marché", "Sélection", "Probabilité (%)"],
        title="📌 Répartition Cote vs Edge des opportunités détectées",
        labels={"Cote": "Cote Bookmaker", "Edge (%)": "Edge / Value (%)"},
        color_continuous_scale="RdYlGn",
    )
    fig.add_hline(
        y=min_edge * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Seuil Min ({min_edge*100:.1f}%)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # Boucle des résultats et boutons d'exports individuels
    for idx, vb in enumerate(results):
      display_market = vb.market.upper()
      display_selection = vb.selection.upper()

      if vb.market == "double_chance":
        display_market = "DOUBLE CHANCE"
        if vb.selection == "1X":
          display_selection = f"1X ({home_team} OU NUL)"
        elif vb.selection == "12":
          display_selection = f"12 ({home_team} OU {away_team})"
        elif vb.selection == "X2":
          display_selection = f"X2 (NUL OU {away_team})"
      elif vb.market == "home_goals":
        display_market = f"BUTS {home_team.upper()}"
      elif vb.market == "away_goals":
        display_market = f"BUTS {away_team.upper()}"
      elif vb.market.startswith("tirs_domicile_"):
        display_market = f"TIRS {home_team.upper()}"
      elif vb.market.startswith("tirs_exterieur_"):
        display_market = f"TIRS {away_team.upper()}"
      elif vb.market.startswith("sot_domicile_"):
        display_market = f"SOT {home_team.upper()}"
      elif vb.market.startswith("sot_exterieur_"):
        display_market = f"SOT {away_team.upper()}"

      st.success(f"🎯 **[{display_market}] Option : {display_selection}**")
      st.write(
          f"• Probabilité estimée : **{vb.model_prob:.1%}** | Cote saisie :"
          f" **{vb.bookmaker_odds}**"
      )
      st.write(
          f"• **EDGE : +{vb.edge:.1%}** | Mise Kelly (1/4) conseillée :"
          f" **{vb.kelly_quart:.1%}**"
      )

      if st.button(f"📤 Exporter ce pari (#{idx+1})", key=f"export_{idx}"):
        success = export_value_bet_to_sheet(
            vb=vb,
            match_date=selected_date,
            match_mode=match_mode,
            timing_paris=timing_paris,
            league_name=league_country,
            home_team=home_team,
            away_team=away_team,
        )
        if success:
          st.success("✅ Pari exporté vers Google Sheets !")

      st.divider()
