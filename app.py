from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import math
import ssl
import urllib.request
import cloudscraper
from fpdf import FPDF
import gspread
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson
import streamlit as st

# ==========================================
# 1. CONFIGURATION
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

# ID du Google Sheet
SPREADSHEET_ID = "11fcyQntgVPi2xjF0GXIeKAZkXrhRkwq2sfysO325kaI"


# =========================================================
# 2. FONCTIONS GOOGLE SHEETS
# =========================================================


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
    bet_name = "Over 2.5" if selection == "over" else "Under 2.5"
  elif market == "btts":
    bet_name = "BTTS Oui" if selection == "yes" else "BTTS Non"
  elif market == "home_goals":
    sel_clean = (
        selection.replace("over_", "Over ")
        .replace("under_", "Under ")
        .replace("_", ".")
    )
    bet_name = f"{sel_clean} (Dom)"
  elif market == "away_goals":
    sel_clean = (
        selection.replace("over_", "Over ")
        .replace("under_", "Under ")
        .replace("_", ".")
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
    bet_name = f"{sel_label} Cut {stat_type} {target_label}"

  return bet_name, cut_str


def build_sheets_row(
    vb, match_date, match_mode, timing_paris, league_name, home_team, away_team
):
  """Construit la liste des 16 colonnes (A à P) pour Google Sheets."""
  bet_name, cut_str = format_bet_details(
      vb.market, vb.selection, home_team, away_team
  )
  clean_league = (
      league_name.split(" - ")[-1] if " - " in str(league_name) else league_name
  )
  edge_val = round(float(vb.edge), 4)
  kelly_val = round(float(vb.kelly_quart), 4)

  return [
      str(match_date),
      "Live" if match_mode == "En direct (Live)" else "Avant",
      timing_paris,
      clean_league,
      f"{home_team} / {away_team}",
      bet_name,
      cut_str,
      float(vb.bookmaker_odds),
      "",
      "",
      edge_val,
      kelly_val,
      "",
      "",
      "",
      "",
  ]


def export_all_value_bets_to_sheet(
    results,
    match_date,
    match_mode,
    timing_paris,
    league_name,
    home_team,
    away_team,
    spreadsheet_id=SPREADSHEET_ID,
    worksheet_name="Suivi Value Bets Global",
):
  try:
    gc = get_gspread_client()
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
    sh.append_rows(rows_data, value_input_option="USER_ENTERED", table_range="A1")
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
    worksheet_name="Suivi Value Bets Global",
):
  try:
    gc = get_gspread_client()
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
    sh.append_row(row_data, value_input_option="USER_ENTERED", table_range="A1")
    return True
  except Exception as e:
    st.error(f"Erreur lors de l'exportation vers Google Sheets : {e}")
    return False


@st.cache_data(ttl=60)
def fetch_raw_data_from_sheets(
    spreadsheet_id=SPREADSHEET_ID, worksheet_name="Suivi Value Bets Global"
):
  """Récupère l'ensemble des lignes du Google Sheet sous forme de DataFrame brut."""
  try:
    gc = get_gspread_client()
    sh = gc.open_by_key(spreadsheet_id).worksheet(worksheet_name)
    data = sh.get_all_values()

    if not data or len(data) < 2:
      return pd.DataFrame()

    header_idx = -1
    for i, row in enumerate(data):
      if row and str(row[0]).strip().lower() == "date":
        header_idx = i
        break

    if header_idx != -1:
      df = pd.DataFrame(data[header_idx + 1 :], columns=data[header_idx])
    else:
      df = pd.DataFrame(data[1:])

    return df
  except Exception as e:
    st.error(f"Erreur lors de la récupération du Google Sheet : {e}")
    return pd.DataFrame()


# ==========================================
# 3. HELPER FUNCTIONS & DATA PREPARATION
# ==========================================


def compute_top_league(df_subset):
  """Calcule dynamiquement la meilleure ligue sur le DataFrame filtré.

  Index 4-facteurs : PnL (40%), ROI (25%), Volume Log (20%), Edge (15%).
  Format de retour : Ligue (PnL | ROI | Volume | Edge Moyen ± Écart-type)
  """
  if df_subset is None or df_subset.empty:
    return "Aucun pari"

  col_league = (
      "league" if "league" in df_subset.columns else df_subset.columns[3]
  )
  col_gains = (
      "gains_num" if "gains_num" in df_subset.columns else df_subset.columns[14]
  )
  col_mises = (
      "mises_num" if "mises_num" in df_subset.columns else df_subset.columns[12]
  )
  col_edge = (
      "edge_pct" if "edge_pct" in df_subset.columns else df_subset.columns[10]
  )

  df_valid = df_subset[
      ~df_subset[col_league]
      .astype(str)
      .str.contains("€|Total|Bankroll|Live|Avant", case=False, na=False)
      & (df_subset[col_league].astype(str).str.strip() != "")
  ].copy()

  if df_valid.empty:
    return "Aucun pari"

  df_valid["league_clean"] = df_valid[col_league].astype(str).str.strip()

  stats = (
      df_valid.groupby("league_clean")
      .agg(
          pnl=(col_gains, "sum"),
          mises=(col_mises, "sum"),
          nb_bets=("league_clean", "count"),
          avg_edge=(col_edge, "mean"),
          std_edge=(col_edge, "std"),
      )
      .reset_index()
  )

  stats["std_edge"] = stats["std_edge"].fillna(0.0)
  stats["roi"] = np.where(
      stats["mises"] > 0, (stats["pnl"] / stats["mises"]) * 100.0, 0.0
  )

  pos_stats = stats[stats["pnl"] > 0].copy()
  if pos_stats.empty:
    return "Aucun P&L positif"

  def normalize(series):
    s_min, s_max = series.min(), series.max()
    return (series - s_min) / (s_max - s_min) if s_max > s_min else 1.0

  norm_pnl = normalize(pos_stats["pnl"])
  norm_roi = normalize(pos_stats["roi"])
  norm_vol = normalize(np.log1p(pos_stats["nb_bets"]))
  norm_edge = normalize(pos_stats["avg_edge"])

  w_pnl, w_roi, w_vol, w_edge = 0.40, 0.25, 0.20, 0.15
  pos_stats["score"] = (
      (w_pnl * norm_pnl)
      + (w_roi * norm_roi)
      + (w_vol * norm_vol)
      + (w_edge * norm_edge)
  )

  best = pos_stats.sort_values(by="score", ascending=False).iloc[0]
  pnl_formatted = f"{best['pnl']:+,.2f} €".replace(",", " ")
  nb_bets_val = int(best["nb_bets"])
  bets_label = "pari" if nb_bets_val == 1 else "paris"
  std_val = best["std_edge"]

  # FORMAT AVEC ÉCART-TYPE : (PnL | ROI | Volume | Edge ± σ)
  return (
      f"{best['league_clean']} ({pnl_formatted} | {best['roi']:+.1f}% |"
      f" {nb_bets_val} {bets_label} | {best['avg_edge']:+.1f}% ± {std_val:.1f}% edge)"
  )


def prepare_dataframe(df_raw):
  """Prépare et nettoie le DataFrame brut de Google Sheets pour tous les modules."""
  if df_raw is None or df_raw.empty:
    return pd.DataFrame()

  df = df_raw.copy()

  col_date, col_league, col_paris = 0, 3, 5
  col_odds, col_edge, col_mises, col_gains_pertes = 7, 10, 12, 14

  raw_dates = (
      df.iloc[:, col_date]
      .astype(str)
      .str.strip()
      .str.extract(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")[0]
  )
  df["parsed_date"] = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")
  df = df.dropna(subset=["parsed_date"]).sort_values("parsed_date").copy()

  def clean_num(series):
    s = (
        series.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    extracted = s.str.extract(r"(-?\d+\.?\d*)")[0]
    return pd.to_numeric(extracted, errors="coerce").fillna(0.0)

  df["league"] = df.iloc[:, col_league].astype(str).str.strip()
  df["pari"] = df.iloc[:, col_paris].astype(str).str.strip()
  df["odds_num"] = clean_num(df.iloc[:, col_odds])
  df["edge_num"] = clean_num(df.iloc[:, col_edge])
  df["mises_num"] = clean_num(df.iloc[:, col_mises])
  df["gains_num"] = clean_num(df.iloc[:, col_gains_pertes])

  df["edge_pct"] = df["edge_num"].apply(
      lambda x: x if abs(x) > 1.0 else x * 100.0
  )

  df["type_marche"] = df["pari"].apply(
      lambda x: (
          "Tirs & Cadrés"
          if any(k in str(x).lower() for k in ["shot", "sot", "tir"])
          else "Buts & Match"
      )
  )

  return df


def filter_by_period(df, period_choice):
  """Filtre le DataFrame selon la période choisie dans le sélecteur (sans MTD)."""
  if df is None or df.empty:
    return pd.DataFrame()

  now = pd.Timestamp.now()

  if period_choice == "7 Derniers Jours":
    cutoff = now - pd.Timedelta(days=7)
    return df[df["parsed_date"] >= cutoff].copy()
  elif period_choice == "30 Derniers Jours":
    cutoff = now - pd.Timedelta(days=30)
    return df[df["parsed_date"] >= cutoff].copy()
  elif period_choice == "Année en Cours (YTD)":
    start_of_year = pd.Timestamp(now.year, 1, 1)
    return df[df["parsed_date"] >= start_of_year].copy()
  else:  # "Tout l'Historique"
    return df.copy()


def get_financial_summary(df_sub):
  """Helper pour calculer les métriques financières d'un sous-ensemble."""
  if df_sub is None or df_sub.empty:
    return {"mises": 0.0, "pnl": 0.0, "roi": 0.0, "vbs": 0}
  mises = df_sub["mises_num"].sum()
  pnl = df_sub["gains_num"].sum()
  roi = (pnl / mises * 100.0) if mises > 0 else 0.0
  return {"mises": mises, "pnl": pnl, "roi": roi, "vbs": len(df_sub)}


def generate_pdf_report(df_filtered, period_label):
  pdf = FPDF()
  pdf.add_page()
  pdf.set_font("Helvetica", "B", 16)

  pdf.cell(
      0,
      10,
      f"Bilan de Performance Value Bets - {period_label}",
      ln=True,
      align="C",
  )
  pdf.set_font("Helvetica", "", 10)
  pdf.cell(
      0,
      10,
      f'Généré le {pd.Timestamp.now().strftime("%d/%m/%Y à %H:%M")}',
      ln=True,
      align="C",
  )
  pdf.ln(5)

  total_vbs = len(df_filtered)
  total_mises = df_filtered["mises_num"].sum() if total_vbs > 0 else 0.0
  total_pnl = df_filtered["gains_num"].sum() if total_vbs > 0 else 0.0
  roi_reel = (total_pnl / total_mises * 100) if total_mises > 0 else 0.0
  avg_edge = df_filtered["edge_pct"].mean() if total_vbs > 0 else 0.0

  pdf.set_font("Helvetica", "B", 12)
  pdf.cell(0, 8, "Résumé Global :", ln=True)
  pdf.set_font("Helvetica", "", 11)

  pdf.cell(100, 7, f"Nombre total de Value Bets : {total_vbs}", ln=True)
  pdf.cell(100, 7, f"Total Mises : {total_mises:,.2f} EUR", ln=True)
  pdf.cell(100, 7, f"P&L Net Réel : {total_pnl:+,.2f} EUR", ln=True)
  pdf.cell(100, 7, f"ROI Réel : {roi_reel:+.2f} %", ln=True)
  pdf.cell(100, 7, f"Edge Moyen Théorique : +{avg_edge:.2f} %", ln=True)
  pdf.ln(8)

  if total_vbs > 0:
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top Ligues par Volume :", ln=True)
    pdf.set_font("Helvetica", "B", 10)

    pdf.cell(70, 7, "Ligue", border=1)
    pdf.cell(30, 7, "Bets", border=1, align="C")
    pdf.cell(45, 7, "Mises (EUR)", border=1, align="R")
    pdf.cell(45, 7, "P&L (EUR)", border=1, align="R")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    league_summary = (
        df_filtered.groupby("league")
        .agg({"mises_num": "sum", "gains_num": "sum", "pari": "count"})
        .reset_index()
        .sort_values("pari", ascending=False)
        .head(8)
    )

    for _, row in league_summary.iterrows():
      pdf.cell(70, 7, str(row["league"])[:30], border=1)
      pdf.cell(30, 7, str(row["pari"]), border=1, align="C")
      pdf.cell(45, 7, f"{row['mises_num']:.2f}", border=1, align="R")
      pdf.cell(45, 7, f"{row['gains_num']:+.2f}", border=1, align="R")
      pdf.ln()

  pdf_buffer = io.BytesIO()
  pdf.output(pdf_buffer)
  pdf_buffer.seek(0)
  return pdf_buffer.getvalue()


# ==========================================
# 4. MODÈLES MATHÉMATIQUES (DIXON-COLES & NBINOM)
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
  if ligne_bookmaker is None:
    return 0.0, 0.0

  seuil = int(np.floor(ligne_bookmaker))

  if var <= mu or math.isnan(var) or var == 0:
    p_under = poisson.cdf(seuil, mu)
  else:
    r = (mu**2) / (var - mu)
    p = mu / var
    p_under = nbinom.cdf(seuil, r, p)

  p_over = 1.0 - p_under
  return p_under, p_over


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
      st.warning("Attention : Le modèle a eu du mal à converger.")

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


@dataclass
class ValueBetResult:
  market: str
  selection: str
  model_prob: float
  bookmaker_odds: float
  edge: float
  kelly_quart: float


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
    except Exception:
      pass

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
      return pd.read_csv(io.StringIO(content))
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
# 5. CHARGEMENT INITIAL DE LA DONNÉE
# ==========================================
df_raw = fetch_raw_data_from_sheets()
df_all = prepare_dataframe(df_raw)


# ==========================================
# 6. EN-TÊTE PRINCIPAL & FILTRE TEMPOREL DYNAMIQUE
# ==========================================
st.title("🏆 Scanner de Value Bets Pro (Buts & Tirs)")

st.subheader("🗓️ Filtre Temporel & Période Glissante")
period_choice = st.selectbox(
    "Choisir l'horizon d'analyse :",
    [
        "7 Derniers Jours",
        "30 Derniers Jours",
        "Année en Cours (YTD)",
        "Tout l'Historique",
    ],
    index=1,  # Par défaut sur 30 jours
)

# Application immédiate du filtre temporel
df_filtered = filter_by_period(df_all, period_choice)

# CALCUL DES MÉTRIQUES DYNAMIQUES & FINANCIAL BREAKDOWN
if not df_filtered.empty:
  total_vbs = len(df_filtered)
  avg_odds = (
      df_filtered["odds_num"][df_filtered["odds_num"] > 1.0].mean()
      if not df_filtered[df_filtered["odds_num"] > 1.0].empty
      else 0.0
  )

  # Edge Moyen + Écart-type Global
  valid_edges = df_filtered["edge_pct"][df_filtered["edge_pct"] != 0]
  avg_edge = valid_edges.mean() if not valid_edges.empty else 0.0
  std_edge = valid_edges.std() if len(valid_edges) > 1 else 0.0
  expected_roi = avg_edge

  # Masques de marchés
  tirs_mask = df_filtered["type_marche"] == "Tirs & Cadrés"
  df_buts = df_filtered[~tirs_mask]
  df_tirs = df_filtered[tirs_mask]

  tirs_count = tirs_mask.sum()
  pct_tirs = tirs_count / total_vbs if total_vbs > 0 else 0.5
  pct_buts = 1.0 - pct_tirs

  # Bilans Financiers Découpés (Global, Buts, Tirs)
  fin_global = get_financial_summary(df_filtered)
  fin_buts = get_financial_summary(df_buts)
  fin_tirs = get_financial_summary(df_tirs)

  # Top Ligues raccordées au filtre temporel actif
  top_league_global = compute_top_league(df_filtered)
  top_league_buts = compute_top_league(df_buts)
  top_league_tirs = compute_top_league(df_tirs)
else:
  total_vbs = 0
  avg_odds = 0.0
  avg_edge = 0.0
  std_edge = 0.0
  expected_roi = 0.0
  pct_buts, pct_tirs = 0.5, 0.5
  fin_global = {"mises": 0.0, "pnl": 0.0, "roi": 0.0, "vbs": 0}
  fin_buts = {"mises": 0.0, "pnl": 0.0, "roi": 0.0, "vbs": 0}
  fin_tirs = {"mises": 0.0, "pnl": 0.0, "roi": 0.0, "vbs": 0}
  top_league_global = "Aucun pari"
  top_league_buts = "Aucun pari"
  top_league_tirs = "Aucun pari"


# ==========================================
# 7. BARRE LATÉRALE (SIDEBAR DYNAMIQUE)
# ==========================================
st.sidebar.header("🎯 Mode d'Analyse")
match_mode = st.sidebar.radio(
    "Sélectionnez le contexte", ["Avant-match (Statique)", "En direct (Live)"]
)

st.sidebar.header("📋 Catégories à Analyser")
cat_buts_main = st.sidebar.checkbox("⚽ Marchés Buts Principaux", value=True)
cat_buts_team = st.sidebar.checkbox("🥅 Buts par Équipe", value=True)
cat_shots = st.sidebar.checkbox("📊 Tirs & Tirs Cadrés", value=True)

st.sidebar.header("⚙️ Filtre de Cotes")
cote_min, cote_max = st.sidebar.slider(
    "Plage de cotes autorisées",
    min_value=1.10,
    max_value=5.00,
    value=(1.50, 2.30),
    step=0.05,
)

st.sidebar.divider()

if st.sidebar.button("🔄 Purger le cache & Actualiser"):
  st.cache_data.clear()
  st.rerun()

st.sidebar.subheader(f"📊 Performance ({period_choice})")

# 1. KPIs de Volume & Cotes (+ Écart-type d'Edge)
kpi_col1, kpi_col2 = st.sidebar.columns(2)
with kpi_col1:
  st.sidebar.metric(label="Value Bets", value=f"{total_vbs}")
  st.sidebar.metric(label="Cote Moyenne", value=f"{avg_odds:.2f}")

with kpi_col2:
  st.sidebar.metric(
      label="Edge Moyen", value=f"+{avg_edge:.1f}% ± {std_edge:.1f}%"
  )
  st.sidebar.metric(label="ROI Théorique", value=f"+{expected_roi:.1f}%")

# 2. Répartition des Marchés (Buts vs Tirs)
st.sidebar.markdown("**🎯 Répartition Marchés**")
st.sidebar.caption(f"{pct_buts:.0%} Buts  |  {pct_tirs:.0%} Tirs")
st.sidebar.progress(pct_buts)

# 3. Bilan Financier Tripartite (Global, Buts, Tirs)
st.sidebar.markdown("---")
st.sidebar.markdown("**💰 Bilan Financier**")

# A. Global
st.sidebar.markdown("🌐 **Global**")
fg1, fg2 = st.sidebar.columns(2)
with fg1:
  st.sidebar.metric(
      label="Mises Totales",
      value=f"{fin_global['mises']:,.2f} €".replace(",", " "),
  )
  st.sidebar.metric(label="ROI Réel", value=f"{fin_global['roi']:+.1f}%")
with fg2:
  st.sidebar.metric(
      label="P&L Net", value=f"{fin_global['pnl']:+,.2f} €".replace(",", " ")
  )

# B. Marchés Buts
st.sidebar.markdown("⚽ **Marchés Buts**")
fb1, fb2 = st.sidebar.columns(2)
with fb1:
  st.sidebar.metric(
      label="Mises (Buts)",
      value=f"{fin_buts['mises']:,.2f} €".replace(",", " "),
  )
  st.sidebar.metric(label="ROI (Buts)", value=f"{fin_buts['roi']:+.1f}%")
with fb2:
  st.sidebar.metric(
      label="P&L (Buts)", value=f"{fin_buts['pnl']:+,.2f} €".replace(",", " ")
  )

# C. Marchés Tirs
st.sidebar.markdown("📊 **Marchés Tirs**")
ft1, ft2 = st.sidebar.columns(2)
with ft1:
  st.sidebar.metric(
      label="Mises (Tirs)",
      value=f"{fin_tirs['mises']:,.2f} €".replace(",", " "),
  )
  st.sidebar.metric(label="ROI (Tirs)", value=f"{fin_tirs['roi']:+.1f}%")
with ft2:
  st.sidebar.metric(
      label="P&L (Tirs)", value=f"{fin_tirs['pnl']:+,.2f} €".replace(",", " ")
  )

# --- TOP LIGUES : AFFICHAGE COMPACT (POLICE RÉDUITE) ---
st.sidebar.markdown("---")
st.sidebar.markdown("**🏆 Top Ligues (Période Active)**")

st.sidebar.markdown(
    f"""
    <div style="font-size: 0.72rem; line-height: 1.4; background-color: #1e2129; padding: 8px 10px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #00CC96;">
        <span style="font-weight: bold; color: #a0aab8; font-size: 0.68rem; text-transform: uppercase;">🌐 Global</span><br>
        <span style="color: #ffffff; font-weight: 500;">{top_league_global}</span>
    </div>
    <div style="font-size: 0.72rem; line-height: 1.4; background-color: #1e2129; padding: 8px 10px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #00CC96;">
        <span style="font-weight: bold; color: #a0aab8; font-size: 0.68rem; text-transform: uppercase;">⚽ Marchés Buts</span><br>
        <span style="color: #ffffff; font-weight: 500;">{top_league_buts}</span>
    </div>
    <div style="font-size: 0.72rem; line-height: 1.4; background-color: #1e2129; padding: 8px 10px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #00CC96;">
        <span style="font-weight: bold; color: #a0aab8; font-size: 0.68rem; text-transform: uppercase;">📊 Marchés Tirs</span><br>
        <span style="color: #ffffff; font-weight: 500;">{top_league_tirs}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()


# ==========================================
# 8. GRAPHIQUES & TABLEAU DE BORD PRINCIPAL
# ==========================================

# A. COURBE DE VARIANCE (P&L RÉEL VS P&L THÉORIQUE)
st.markdown("---")
st.subheader("📈 P&L Réel vs P&L Théorique (Courbe de Variance)")

if not df_filtered.empty:
  df_chart = df_filtered.copy()
  df_chart["date_jour"] = df_chart["parsed_date"].dt.date
  df_chart["pnl_theo_step"] = df_chart["mises_num"] * (
      df_chart["edge_pct"] / 100.0
  )

  df_pnl = (
      df_chart.groupby("date_jour")
      .agg({"gains_num": "sum", "pnl_theo_step": "sum", "mises_num": "sum"})
      .reset_index()
      .sort_values("date_jour")
  )

  df_pnl["pnl_real_cum"] = df_pnl["gains_num"].cumsum()
  df_pnl["pnl_theo_cum"] = df_pnl["pnl_theo_step"].cumsum()

  fig_pnl = go.Figure()

  fig_pnl.add_trace(
      go.Scatter(
          x=df_pnl["date_jour"],
          y=df_pnl["pnl_real_cum"],
          mode="lines+markers",
          name="P&L Réel (€)",
          line=dict(color="#00CC96", width=2.5),
          marker=dict(size=6),
      )
  )

  fig_pnl.add_trace(
      go.Scatter(
          x=df_pnl["date_jour"],
          y=df_pnl["pnl_theo_cum"],
          mode="lines+markers",
          name="P&L Théorique Attendu (€)",
          line=dict(color="#AB63FA", width=2, dash="dash"),
          marker=dict(size=6),
      )
  )

  fig_pnl.update_layout(
      template="plotly_dark",
      xaxis_title="Date",
      yaxis_title="Euros (€)",
      hovermode="x unified",
      legend=dict(
          orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
      ),
      margin=dict(l=20, r=20, t=30, b=20),
  )

  st.plotly_chart(fig_pnl, use_container_width=True)
else:
  st.info("Aucun pari trouvé sur la période sélectionnée.")

# B. HEATMAP DE RENTABILITÉ
st.markdown("---")
st.subheader("🔥 Matrice de Rentabilité (Ligue × Type de Marché)")

if not df_filtered.empty:
  pivot_pnl = df_filtered.pivot_table(
      index="league",
      columns="type_marche",
      values="gains_num",
      aggfunc="sum",
      fill_value=0.0,
  )

  if not pivot_pnl.empty:
    fig_heatmap = px.imshow(
        pivot_pnl,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Gain Net (€) par Ligue et Marché",
    )
    fig_heatmap.update_layout(
        template="plotly_dark", xaxis_title="Marché", yaxis_title="Ligue"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
  st.info("Pas assez de données pour afficher la heatmap.")

# C. EXPORT PDF DU BILAN
st.markdown("---")
st.subheader("📄 Exportation du Bilan")

if not df_filtered.empty:
  pdf_bytes = generate_pdf_report(df_filtered, period_choice)
  st.download_button(
      label=f"📥 Télécharger le Bilan PDF ({period_choice})",
      data=pdf_bytes,
      file_name=f"bilan_value_bets_{period_choice.lower().replace(' ', '_')}.pdf",
      mime="application/pdf",
  )
else:
  st.warning("Aucune donnée à exporter pour cette période.")


# ==========================================
# 9. MODULE VALUE BET SCANNER (SELECTION & MATCH)
# ==========================================
st.markdown("---")
st.header("⚽ Analyse de Match & Scanner")

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

df_fixtures = load_fixtures()
idx_h, idx_a = 0, min(1, len(teams_list) - 1)
selected_date = datetime.now().strftime("%d/%m/%Y")
timing_paris = "1. J-2+"
match_date = datetime.today().date()

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

    filtered_fixtures = (
        league_fixtures[league_fixtures["Date"] == selected_date_opt]
        if selected_date_opt != "-- Toutes les dates --"
        else league_fixtures
    )

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
          options=["1. J-2+", "2. J-1", "3. H-12 à H-2", "4. H-2 à H"],
      )

    if selected_fixture != "-- Sélectionner un match --":
      match_str = selected_fixture.rsplit(" (", 1)[0]
      h_sel, a_sel = match_str.split(" / ")
      if h_sel in teams_list:
        idx_h = teams_list.index(h_sel)
      if a_sel in teams_list:
        idx_a = teams_list.index(a_sel)

      match_row = filtered_fixtures[
          (filtered_fixtures["HomeTeam"] == h_sel)
          & (filtered_fixtures["AwayTeam"] == a_sel)
      ]
      if not match_row.empty:
        match_date = match_row["Date"].values[0]

st.divider()

col1, col2, col3, col4 = st.columns([2, 2, 1, 1.2])

with col1:
  home_team = st.selectbox(
      "🏠 Équipe à Domicile", options=teams_list, index=idx_h
  )

with col2:
  away_team = st.selectbox(
      "✈️ Équipe à l'Extérieur", options=teams_list, index=idx_a
  )

with col3:
  if isinstance(match_date, str):
    final_match_date = match_date
    st.text_input("📅 Date match", value=final_match_date, disabled=True)
  else:
    user_date = st.date_input("📅 Date match", value=match_date)
    final_match_date = user_date.strftime("%d/%m/%Y")

  selected_date = final_match_date

with col4:
  timing_paris = st.selectbox(
      "⏱️ Timing pari",
      options=["1. J-2+", "2. J-1", "3. H-12 à H-2", "4. H-2 à H"],
      key="timing_manual",
  )

live_minute, live_home_score, live_away_score = 0, 0, 0
if match_mode == "En direct (Live)":
  st.divider()
  st.info("⏱️ **Mode Live Activé** : Indiquez l'état actuel de la rencontre.")
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
# 10. SAISIE DES COTES BOOKMAKER
# ==========================================
st.subheader("2️⃣ Saisissez les cotes du bookmaker")

if cat_buts_main:
  with st.expander(
      "⚽ MARCHÉS DES BUTS PRINCIPAUX (1X2, DOUBLE CHANCE, O/U 2.5, BTTS)",
      expanded=True,
  ):
    st.markdown("### 🏆 Résultat Match (1X2)")
    c1, c2, c3 = st.columns(3)
    h_odd = c1.number_input("Cote 1 (Domicile)", value=None, step=0.01)
    d_odd = c2.number_input("Cote X (Nul)", value=None, step=0.01)
    a_odd = c3.number_input("Cote 2 (Extérieur)", value=None, step=0.01)

    st.markdown("### 🛡️ Double Chance")
    c1, c2, c3 = st.columns(3)
    dc_1x = c1.number_input("1X (Dom ou Nul)", value=None, step=0.01)
    dc_12 = c2.number_input("12 (Dom ou Ext)", value=None, step=0.01)
    dc_x2 = c3.number_input("X2 (Nul ou Ext)", value=None, step=0.01)

    st.markdown("### ⚽ Buts & BTTS")
    c1, c2, c3, c4 = st.columns(4)
    ou_over = c1.number_input("Over 2.5 (Buts)", value=None, step=0.01)
    ou_under = c2.number_input("Under 2.5 (Buts)", value=None, step=0.01)
    btts_yes = c3.number_input("BTTS Oui", value=None, step=0.01)
    btts_no = c4.number_input("BTTS Non", value=None, step=0.01)

if cat_buts_team:
  with st.expander("角 BUTS PAR ÉQUIPE (Over / Under 0.5 et 1.5)"):
    st.markdown(f"**🏠 {home_team} (Domicile)**")
    c1, c2, c3, c4 = st.columns(4)
    hg_o05 = c1.number_input("Over 0.5 (Dom)", value=None, step=0.01)
    hg_u05 = c2.number_input("Under 0.5 (Dom)", value=None, step=0.01)
    hg_o15 = c3.number_input("Over 1.5 (Dom)", value=None, step=0.01)
    hg_u15 = c4.number_input("Under 1.5 (Dom)", value=None, step=0.01)

    st.markdown(f"**✈️ {away_team} (Extérieur)**")
    c1, c2, c3, c4 = st.columns(4)
    ag_o05 = c1.number_input("Over 0.5 (Ext)", value=None, step=0.01)
    ag_u05 = c2.number_input("Under 0.5 (Ext)", value=None, step=0.01)
    ag_o15 = c3.number_input("Over 1.5 (Ext)", value=None, step=0.01)
    ag_u15 = c4.number_input("Under 1.5 (Ext)", value=None, step=0.01)

if cat_shots:
  with st.expander("📊 MARCHÉS DES TIRS & TIRS CADRÉS (Lignes ajustables)"):
    st.markdown("### 🏹 Tirs Totaux (Match)")
    c1, c2, c3 = st.columns(3)
    t_shots_line = c1.number_input(
        "Ligne de Tirs Match (ex: 24.5)", value=None, step=0.5
    )
    t_shots_o = c2.number_input(
        "Cote Over Tirs Match", value=None, step=0.01, key="ts_o"
    )
    t_shots_u = c3.number_input(
        "Cote Under Tirs Match", value=None, step=0.01, key="ts_u"
    )

    st.markdown(f"### 🏠 Tirs Totaux Individuels : {home_team}")
    c1, c2, c3 = st.columns(3)
    h_shots_line = c1.number_input(
        f"Ligne Tirs Totaux {home_team}", value=None, step=0.5
    )
    h_shots_o = c2.number_input("Cote Over Tirs Dom", value=None, step=0.01)
    h_shots_u = c3.number_input("Cote Under Tirs Dom", value=None, step=0.01)

    st.markdown(f"### ✈️ Tirs Totaux Individuels : {away_team}")
    c1, c2, c3 = st.columns(3)
    a_shots_line = c1.number_input(
        f"Ligne Tirs Totaux {away_team}", value=None, step=0.5
    )
    a_shots_o = c2.number_input("Cote Over Tirs Ext", value=None, step=0.01)
    a_shots_u = c3.number_input("Cote Under Tirs Ext", value=None, step=0.01)

    st.markdown("### 🎯 Tirs Cadrés Totaux (Match)")
    c1, c2, c3 = st.columns(3)
    t_sot_line = c1.number_input(
        "Ligne Tirs Cadrés Match (ex: 8.5)", value=None, step=0.5
    )
    t_sot_o = c2.number_input("Cote Over SOT Match", value=None, step=0.01)
    t_sot_u = c3.number_input("Cote Under SOT Match", value=None, step=0.01)

    st.markdown(f"### 🏠 Tirs Cadrés Individuels : {home_team}")
    c1, c2, c3 = st.columns(3)
    h_sot_line = c1.number_input(f"Ligne SOT {home_team}", value=None, step=0.5)
    h_sot_o = c2.number_input("Cote Over SOT Dom", value=None, step=0.01)
    h_sot_u = c3.number_input("Cote Under SOT Dom", value=None, step=0.01)

    st.markdown(f"### ✈️ Tirs Cadrés Individuels : {away_team}")
    c1, c2, c3 = st.columns(3)
    a_sot_line = c1.number_input(f"Ligne SOT {away_team}", value=None, step=0.5)
    a_sot_o = c2.number_input("Cote Over SOT Ext", value=None, step=0.01)
    a_sot_u = c3.number_input("Cote Under SOT Ext", value=None, step=0.01)

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
# 11. MOTEUR D'ANALYSE & EXPORT
# ==========================================
if st.button(
    "🚀 Lancer l'Analyse Complète", type="primary", use_container_width=True
):
  if home_team == away_team:
    st.error("Veuillez choisir deux équipes différentes.")
    st.session_state.pop("analysis_data", None)
  elif not (cat_buts_main or cat_buts_team or cat_shots):
    st.warning("Veuillez cocher au moins une catégorie à analyser.")
    st.session_state.pop("analysis_data", None)
  else:
    preds_all = {}
    market_odds = {}

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

      shots_config = [
          (
              "tirs_match",
              t_shots_line,
              lam_h_shots + lam_a_shots,
              var_h_shots + var_a_shots,
              t_shots_o,
              t_shots_u,
          ),
          (
              "tirs_domicile",
              h_shots_line,
              lam_h_shots,
              var_h_shots,
              h_shots_o,
              h_shots_u,
          ),
          (
              "tirs_exterieur",
              a_shots_line,
              lam_a_shots,
              var_a_shots,
              a_shots_o,
              a_shots_u,
          ),
          (
              "sot_match",
              t_sot_line,
              lam_h_sot + lam_a_sot,
              var_h_sot + var_a_sot,
              t_sot_o,
              t_sot_u,
          ),
          (
              "sot_domicile",
              h_sot_line,
              lam_h_sot,
              var_h_sot,
              h_sot_o,
              h_sot_u,
          ),
          (
              "sot_exterieur",
              a_sot_line,
              lam_a_sot,
              var_a_sot,
              a_sot_o,
              a_sot_u,
          ),
      ]

      for prefix, line, lam, var, odd_o, odd_u in shots_config:
        if line is not None:
          prob_u, prob_o = proba_tirs_nbinom(lam, var, line)
          market_key = f"{prefix}_{line}"
          preds_all[market_key] = {"over": prob_o, "under": prob_u}
          market_odds[market_key] = {"over": odd_o, "under": odd_u}

    results = []
    for market, odds in market_odds.items():
      for sel, odd in odds.items():
        if odd is None:
          continue

        c_min = cote_min if cote_min is not None else 1.50
        c_max = cote_max if cote_max is not None else 2.30

        if c_min <= odd <= c_max:
          prob = preds_all[market][sel]
          edge = prob * odd - 1

          if edge > min_edge:
            b = odd - 1
            kelly_quart = (
                max(0.0, (b * prob - (1 - prob)) / b) * 0.25 if b > 0 else 0.0
            )
            results.append(
                ValueBetResult(market, sel, prob, odd, edge, kelly_quart)
            )

    results.sort(key=lambda x: x.edge, reverse=True)

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


# ==========================================
# 12. AFFICHAGE DES RÉSULTATS & EXPORT
# ==========================================
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

  clean_league = (
      league_country.split(" - ")[-1]
      if " - " in str(league_country)
      else league_country
  )

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
          league_name=clean_league,
          home_team=home_team,
          away_team=away_team,
      )
      if success:
        st.success(
            f"✅ {len(results)} Value Bets exportés avec succès vers Google"
            " Sheets !"
        )

    st.divider()

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
            league_name=clean_league,
            home_team=home_team,
            away_team=away_team,
        )
        if success:
          st.success("✅ Pari exporté vers Google Sheets !")

      st.divider()
