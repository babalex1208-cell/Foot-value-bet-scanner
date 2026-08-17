import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import io
import math
import gspread
from google.oauth2.service_account import Credentials
from scipy.optimize import minimize
from scipy.stats import poisson, nbinom
from dataclasses import dataclass

# ==========================================
# 1. CONFIGURATION INITIALE
# ==========================================
st.set_page_config(page_title="Value Bet Scanner Pro", page_icon="⚽", layout="wide")

# Identifiants Google Sheets
SPREADSHEET_ID = "11fcyQntgVPi2xjF0GXIeKAZkXrhRkwq2sfysO325kaI"
SHEET_GID = 475064708

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

# ==========================================
# 2. MODULE PINNACLE & GOOGLE SHEETS (LUNDI)
# ==========================================
def get_gspread_client():
    """Connexion à Google Sheets via st.secrets."""
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scope
    )
    return gspread.authorize(credentials)

def fetch_pinnacle_data_from_fd(season="2526"):
    """Télécharge les données récentes pour toutes les ligues."""
    fd_cache = {}
    for key, info in LEAGUES.items():
        code = info["fd_code"]
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req)
            df = pd.read_csv(io.StringIO(resp.read().decode('utf-8')))
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            fd_cache[code] = df
        except Exception:
            continue
    return fd_cache

def update_clv_in_google_sheet():
    """Lit le Sheet, calcule la CLV Pinnacle et l'écrit en colonne H à partir de la ligne 4."""
    client = get_gspread_client()
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    
    # Sélection de la feuille via GID
    worksheet = None
    for ws in spreadsheet.worksheets():
        if ws.id == SHEET_GID:
            worksheet = ws
            break
    if not worksheet:
        worksheet = spreadsheet.sheet1

    all_rows = worksheet.get_all_values()
    if len(all_rows) < 4:
        return 0, "Le Google Sheet ne contient pas suffisamment de lignes (minimum 4)."

    # Ligne 3 = En-têtes | Lignes 4+ = Données
    headers = [h.strip().lower() for h in all_rows[2]]
    
    # Recherche dynamique ou par défaut des colonnes principales
    def find_col(possible_names, default_idx):
        for name in possible_names:
            if name in headers:
                return headers.index(name)
        return default_idx

    col_code = find_col(['code', 'league_code', 'championnat', 'ligue'], 0)
    col_home = find_col(['hometeam', 'domicile', 'home'], 1)
    col_away = find_col(['awayteam', 'extérieur', 'exterieur', 'away'], 2)
    col_sel  = find_col(['selection', 'pari', 'choix', 'issue'], 3)
    col_odd  = find_col(['cote_prise', 'cote', 'odd'], 4)

    # Récupération des CSV Football-Data
    fd_cache = fetch_pinnacle_data_from_fd()
    
    updates = []
    updated_count = 0

    # Parcours à partir de la ligne 4 (index 3 en Python)
    for row_idx in range(3, len(all_rows)):
        row = all_rows[row_idx]
        real_row_num = row_idx + 1  # Ligne réelle dans Google Sheets

        if len(row) <= max(col_code, col_home, col_away, col_sel, col_odd):
            continue

        league_code = str(row[col_code]).strip()
        home_team = str(row[col_home]).strip()
        away_team = str(row[col_away]).strip()
        selection = str(row[col_sel]).strip().upper()
        
        try:
            odd_taken = float(str(row[col_odd]).replace(',', '.'))
        except ValueError:
            continue

        if league_code in fd_cache:
            df_fd = fd_cache[league_code]
            match = df_fd[(df_fd['HomeTeam'] == home_team) & (df_fd['AwayTeam'] == away_team)]

            if not match.empty:
                m = match.iloc[-1]
                ps_h = m.get('PSH')
                ps_d = m.get('PSD')
                ps_a = m.get('PSA')

                closing_odd = None
                if selection in ['1', 'H', 'HOME', home_team.upper()]:
                    closing_odd = ps_h
                elif selection in ['X', 'D', 'DRAW', 'NUL']:
                    closing_odd = ps_d
                elif selection in ['2', 'A', 'AWAY', away_team.upper()]:
                    closing_odd = ps_a

                if closing_odd and pd.notna(closing_odd) and float(closing_odd) > 0:
                    clv = (odd_taken / float(closing_odd)) - 1.0
                    # Colonne H = Colonne 8
                    updates.append({
                        'range': f'H{real_row_num}',
                        'values': [[round(clv, 4)]]
                    })
                    updated_count += 1

    if updates:
        worksheet.batch_update(updates)

    return updated_count, None

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
    """Calcule les probabilités statistiques avec la Loi Binomiale Négative si surdispersion."""
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
            return x[:n], x[n:2*n], x[2*n], x[2*n+1]

        def neg_log_likelihood(x):
            attack, defense, rho, home_adv = unpack(x)
            lam_home = np.exp(np.clip(attack[home_idx] + defense[away_idx] + home_adv, -20, 20))
            lam_away = np.exp(np.clip(attack[away_idx] + defense[home_idx], -20, 20))
            ll = poisson.logpmf(hg, lam_home) + poisson.logpmf(ag, lam_away)
            
            if home_col == "FTHG":
                tau = np.array([dixon_coles_adjustment(h, a, lh, la, rho) for h, a, lh, la in zip(hg, ag, lam_home, lam_away)])
                ll += np.log(np.clip(tau, 1e-6, None))
            return -np.sum(weights * ll)

        x0 = np.zeros(2*n + 2)
        x0[2*n+1] = 0.2 if home_col == "FTHG" else 1.0
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:n])}]
        result = minimize(neg_log_likelihood, x0, method="SLSQP", constraints=constraints, options={"maxiter": 200})
        
        attack, defense, rho, home_adv = unpack(result.x)
        self.params = {t: {"attack": attack[i], "defense": defense[i]} for i, t in enumerate(self.teams)}
        self.rho = rho
        self.home_advantage = home_adv
        return self

    def get_lambdas(self, home_team, away_team):
        a_h, d_h = self.params[home_team]["attack"], self.params[home_team]["defense"]
        a_a, d_a = self.params[away_team]["attack"], self.params[away_team]["defense"]
        lam_home = np.exp(a_h + d_a + self.home_advantage)
        lam_away = np.exp(a_a + d_h)
        return lam_home, lam_away

    def score_matrix(self, home_team, away_team, max_goals=10):
        lam_home, lam_away = self.get_lambdas(home_team, away_team)
        probs = np.zeros((max_goals+1, max_goals+1))
        for i in range(max_goals+1):
            for j in range(max_goals+1):
                p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away) * dixon_coles_adjustment(i, j, lam_home, lam_away, self.rho)
                probs[i, j] = p
        probs = np.clip(probs, 0, None)
        return probs / probs.sum(), lam_home, lam_away

    def predict_goal_markets(self, home_team, away_team, max_goals=10):
        M, lam_h, lam_a = self.score_matrix(home_team, away_team, max_goals)
        goals = np.arange(max_goals+1)
        
        h_prob = np.tril(M, -1).sum()
        d_prob = np.trace(M)
        a_prob = np.triu(M, 1).sum()
        
        return {
            "1X2": {"H": h_prob, "D": d_prob, "A": a_prob},
            "double_chance": {
                "1X": h_prob + d_prob,
                "12": h_prob + a_prob,
                "X2": d_prob + a_prob
            },
            "over_under_2_5": {"over": M[goals[:, None] + goals[None, :] > 2.5].sum(), "under": M[goals[:, None] + goals[None, :] < 2.5].sum()},
            "btts": {"yes": M[1:, 1:].sum(), "no": 1 - M[1:, 1:].sum()},
            "home_goals": {"over_0_5": M[1:, :].sum(), "under_0_5": M[0, :].sum(), "over_1_5": M[2:, :].sum(), "under_1_5": M[:2, :].sum()},
            "away_goals": {"over_0_5": M[:, 1:].sum(), "under_0_5": M[:, 0].sum(), "over_1_5": M[:, 2:].sum(), "under_1_5": M[:, :2].sum()},
            "expected_goals": {"home": lam_h, "away": lam_a},
        }

# ==========================================
# 4. GESTION DES DONNÉES HISTORIQUES
# ==========================================
@dataclass
class ValueBetResult:
    market: str
    selection: str
    model_prob: float
    bookmaker_odds: float
    edge: float
    kelly_quart: float

@st.cache_data(show_spinner=False)
def load_and_clean_data(league_code):
    dfs = []
    for s in SEASONS:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{league_code}.csv"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req)
            df = pd.read_csv(io.StringIO(resp.read().decode('utf-8')))
            df['Season'] = s
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    
    data = pd.concat(dfs, ignore_index=True)
    cols_to_clean = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
    data = data.dropna(subset=[c for c in cols_to_clean if c in data.columns])
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    
    for col in ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']:
        if col in data.columns:
            data[col] = data[col].astype(int)
    return data

@st.cache_resource(show_spinner=False)
def train_all_models(df):
    goal_model = DixonColesModel().fit(df, "FTHG", "FTAG", halflife_days=TIME_DECAY_HALFLIFE_DAYS)
    shot_model = DixonColesModel().fit(df, "HS", "AS", halflife_days=TIME_DECAY_HALFLIFE_DAYS)
    sot_model = DixonColesModel().fit(df, "HST", "AST", halflife_days=TIME_DECAY_HALFLIFE_DAYS)
    return {"goals": goal_model, "shots": shot_model, "sot": sot_model}

# ==========================================
# 5. INTERFACE STREAMLIT & BARRE LATÉRALE
# ==========================================
st.sidebar.header("🎯 Mode d'Analyse")
match_mode = st.sidebar.radio("Sélectionnez le contexte", ["Avant-match (Statique)", "En direct (Live)"])

st.sidebar.divider()
st.sidebar.header("📅 Routine du Lundi (CLV Pinnacle)")

if st.sidebar.button("🔄 Mettre à jour CLV (Colonne H)", use_container_width=True):
    with st.sidebar.spinner("Calcul des CLV via Pinnacle..."):
        try:
            count, err = update_clv_in_google_sheet()
            if err:
                st.sidebar.error(err)
            elif count > 0:
                st.sidebar.success(f"✅ {count} ligne(s) mise(s) à jour en colonne H !")
            else:
                st.sidebar.info("Aucune nouvelle cote Pinnacle trouvée à synchroniser.")
        except Exception as e:
            st.sidebar.error(f"Erreur de connexion : {e}")

st.sidebar.divider()
st.sidebar.header("⚙️ Stratégie de Paris")
strategie = st.sidebar.selectbox(
    "Filtre de cotes",
    [
        "Afficher Tout (Aucun filtre)",
        "Volume Faible (Cotes 1.50 - 1.85)",
        "Volume Moyen (Cotes 1.70 - 2.10)",
        "Volume Élevé (Cotes 1.85 - 2.30)",
        "Volume Massif (Cotes > 2.00)"
    ]
)

if strategie == "Volume Faible (Cotes 1.50 - 1.85)":
    cote_min, cote_max = 1.50, 1.85
elif strategie == "Volume Moyen (Cotes 1.70 - 2.10)":
    cote_min, cote_max = 1.70, 2.10
elif strategie == "Volume Élevé (Cotes 1.85 - 2.30)":
    cote_min, cote_max = 1.85, 2.30
elif strategie == "Volume Massif (Cotes > 2.00)":
    cote_min, cote_max = 2.00, 100.00
else:
    cote_min, cote_max = 1.01, 100.00

# ==========================================
# 6. SCANNER PRINCIPAL
# ==========================================
st.title("🏆 Scanner de Value Bets Pro (Buts & Tirs)")

league_key = st.selectbox("1️⃣ Choisis un championnat", options=list(LEAGUES.keys()), format_func=lambda x: LEAGUES[x]["country"])

with st.spinner("Calcul et calibration des modèles..."):
    df = load_and_clean_data(LEAGUES[league_key]["fd_code"])
    if df is not None:
        models = train_all_models(df)
        teams_list = models["goals"].teams
        st.success(f"✅ Modèles calibrés avec succès sur {len(df)} matchs historiques !")
    else:
        st.error("Impossible de charger les données.")
        st.stop()

st.divider()

col1, col2 = st.columns(2)
with col1: home_team = st.selectbox("🏠 Équipe à Domicile", options=teams_list)
with col2: away_team = st.selectbox("✈️ Équipe à l'Extérieur", options=teams_list, index=1)

live_minute, live_home_score, live_away_score = 0, 0, 0
if match_mode == "En direct (Live)":
    st.divider()
    st.info("⏱️ **Mode Live Activé** : Indiquez l'état actuel de la rencontre pour réajuster le modèle en temps réel.")
    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        live_minute = st.number_input("Minute du match", min_value=1, max_value=90, value=46)
    with l_col2:
        live_home_score = st.number_input("Score Domicile actuel", min_value=0, max_value=10, value=0)
    with l_col3:
        live_away_score = st.number_input("Score Extérieur actuel", min_value=0, max_value=10, value=1)

st.divider()
st.subheader("2️⃣ Saisissez les cotes du bookmaker")

with st.expander("⚽ MARCHÉS DES BUTS PRINCIPAUX (1X2, DOUBLE CHANCE, O/U 2.5, BTTS)", expanded=True):
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

with st.expander("0️⃣ BUTS PAR ÉQUIPE (Over / Under 0.5 et 1.5)"):
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

with st.expander("📊 MARCHÉS DES TIRS & TIRS CADRÉS (Lignes ajustables)"):
    st.markdown("### 🏹 Tirs Totaux (Match)")
    c1, c2, c3 = st.columns(3)
    t_shots_line = c1.number_input("Ligne de Tirs Match (ex: 24.5)", value=24.5, step=0.5)
    t_shots_o = c2.number_input("Cote Over Tirs Match", value=1.85, step=0.01)
    t_shots_u = c3.number_input("Cote Under Tirs Match", value=1.85, step=0.01)

    st.markdown(f"### 🏠 Tirs Totaux Individuels : {home_team}")
    c1, c2, c3 = st.columns(3)
    h_shots_line = c1.number_input(f"Ligne Tirs Totaux {home_team}", value=13.5, step=0.5)
    h_shots_o = c2.number_input("Cote Over Tirs Dom", value=1.85, step=0.01)
    h_shots_u = c3.number_input("Cote Under Tirs Dom", value=1.85, step=0.01)

    st.markdown(f"### ✈️ Tirs Totaux Individuels : {away_team}")
    c1, c2, c3 = st.columns(3)
    a_shots_line = c1.number_input(f"Ligne Tirs Totaux {away_team}", value=11.5, step=0.5)
    a_shots_o = c2.number_input("Cote Over Tirs Ext", value=1.85, step=0.01)
    a_shots_u = c3.number_input("Cote Under Tirs Ext", value=1.85, step=0.01)

    st.markdown("### 🎯 Tirs Cadrés Totaux (Match)")
    c1, c2, c3 = st.columns(3)
    t_sot_line = c1.number_input("Ligne Tirs Cadrés Match (ex: 8.5)", value=8.5, step=0.5)
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

min_edge = st.slider("Seuil d'Edge minimum (%)", min_value=0.0, max_value=15.0, value=3.0, step=0.5) / 100

if st.button("🚀 Lancer l'Analyse Complète", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Veuillez choisir deux équipes différentes.")
    else:
        # 1. Calculs buts
        base_goals_preds = models["goals"].predict_goal_markets(home_team, away_team)
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
            M = np.zeros((max_goals+1, max_goals+1))
            for i in range(max_goals+1):
                for j in range(max_goals+1):
                    p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) * dixon_coles_adjustment(i, j, lam_h, lam_a, models["goals"].rho)
                    M[i, j] = p
            M = np.clip(M, 0, None)
            M = M / M.sum()
            
            h_prob = np.tril(M, -1).sum()
            d_prob = np.trace(M)
            a_prob = np.triu(M, 1).sum()
            
            buts_manquants = max(0, 3 - score_total_actuel)
            if buts_manquants == 0:
                prob_over_25 = 1.0
            else:
                prob_over_25 = sum((math.exp(-(lam_h + lam_a)) * ((lam_h + lam_a)**k)) / math.factorial(k) for k in range(buts_manquants, 10))
            prob_under_25 = 1.0 - prob_over_25

            preds_goals = {
                "1X2": {"H": h_prob, "D": d_prob, "A": a_prob},
                "double_chance": {"1X": h_prob + d_prob, "12": h_prob + a_prob, "X2": d_prob + a_prob},
                "over_under_2_5": {"over": prob_over_25, "under": prob_under_25},
                "btts": {"yes": M[1:, 1:].sum(), "no": 1 - M[1:, 1:].sum()},
                "home_goals": {"over_0_5": M[1:, :].sum(), "under_0_5": M[0, :].sum(), "over_1_5": M[2:, :].sum(), "under_1_5": M[:2, :].sum()},
                "away_goals": {"over_0_5": M[:, 1:].sum(), "under_0_5": M[:, 0].sum(), "over_1_5": M[:, 2:].sum(), "under_1_5": M[:, :2].sum()},
                "expected_goals": {"home": lam_h, "away": lam_a},
            }
        else:
            preds_goals = base_goals_preds
            lam_h, lam_a = lam_h_base, lam_a_base

        # 2. Tirs (Binomiale Négative)
        lam_h_shots, lam_a_shots = models["shots"].get_lambdas(home_team, away_team)
        h_shots_data = df[df['HomeTeam'] == home_team]['HS']
        a_shots_data = df[df['AwayTeam'] == away_team]['AS']
        var_h_shots = np.var(h_shots_data, ddof=1) if len(h_shots_data) > 1 else lam_h_shots
        var_a_shots = np.var(a_shots_data, ddof=1) if len(a_shots_data) > 1 else lam_a_shots

        if match_mode == "En direct (Live)":
            ratio = (90 - live_minute) / 90.0
            lam_h_shots *= ratio
            lam_a_shots *= ratio
            var_h_shots *= ratio
            var_a_shots *= ratio

        prob_shots_under, prob_shots_over = proba_tirs_nbinom(lam_h_shots + lam_a_shots, var_h_shots + var_a_shots, t_shots_line)
        prob_h_shots_under, prob_h_shots_over = proba_tirs_nbinom(lam_h_shots, var_h_shots, h_shots_line)
        prob_a_shots_under, prob_a_shots_over = proba_tirs_nbinom(lam_a_shots, var_a_shots, a_shots_line)

        # 3. Tirs cadrés (Binomiale Négative)
        lam_h_sot, lam_a_sot = models["sot"].get_lambdas(home_team, away_team)
        h_sot_data = df[df['HomeTeam'] == home_team]['HST']
        a_sot_data = df[df['AwayTeam'] == away_team]['AST']
        var_h_sot = np.var(h_sot_data, ddof=1) if len(h_sot_data) > 1 else lam_h_sot
        var_a_sot = np.var(a_sot_data, ddof=1) if len(a_sot_data) > 1 else lam_a_sot

        if match_mode == "En direct (Live)":
            ratio = (90 - live_minute) / 90.0
            lam_h_sot *= ratio
            lam_a_sot *= ratio
            var_h_sot *= ratio
            var_a_sot *= ratio

        prob_sot_under, prob_sot_over = proba_tirs_nbinom(lam_h_sot + lam_a_sot, var_h_sot + var_a_sot, t_sot_line)
        prob_h_sot_under, prob_h_sot_over = proba_tirs_nbinom(lam_h_sot, var_h_sot, h_sot_line)
        prob_a_sot_under, prob_a_sot_over = proba_tirs_nbinom(lam_a_sot, var_a_sot, a_sot_line)

        # 4. Assemblage
        preds_all = {
            "1X2": preds_goals["1X2"],
            "double_chance": preds_goals["double_chance"],
            "over_under_2_5": preds_goals["over_under_2_5"],
            "btts": preds_goals["btts"],
            "home_goals": preds_goals["home_goals"],      
            "away_goals": preds_goals["away_goals"],      
            f"tirs_match_{t_shots_line}": {"over": prob_shots_over, "under": prob_shots_under},
            f"tirs_domicile_{h_shots_line}": {"over": prob_h_shots_over, "under": prob_h_shots_under},   
            f"tirs_exterieur_{a_shots_line}": {"over": prob_a_shots_over, "under": prob_a_shots_under}, 
            f"sot_match_{t_sot_line}": {"over": prob_sot_over, "under": prob_sot_under},
            f"sot_domicile_{h_sot_line}": {"over": prob_h_sot_over, "under": prob_h_sot_under},
            f"sot_exterieur_{a_sot_line}": {"over": prob_a_sot_over, "under": prob_a_sot_under}
        }

        market_odds = {
            "1X2": {"H": h_odd, "D": d_odd, "A": a_odd},
            "double_chance": {"1X": dc_1x, "12": dc_12, "X2": dc_x2},
            "over_under_2_5": {"over": ou_over, "under": ou_under},
            "btts": {"yes": btts_yes, "no": btts_no},
            "home_goals": {"over_0_5": hg_o05, "under_0_5": hg_u05, "over_1_5": hg_o15, "under_1_5": hg_u15}, 
            "away_goals": {"over_0_5": ag_o05, "under_0_5": ag_u05, "over_1_5": ag_o15, "under_1_5": ag_u15}, 
            f"tirs_match_{t_shots_line}": {"over": t_shots_o, "under": t_shots_u},
            f"tirs_domicile_{h_shots_line}": {"over": h_shots_o, "under": h_shots_u},   
            f"tirs_exterieur_{a_shots_line}": {"over": a_shots_o, "under": a_shots_u}, 
            f"sot_match_{t_sot_line}": {"over": t_sot_o, "under": t_sot_u},
            f"sot_domicile_{h_sot_line}": {"over": h_sot_o, "under": h_sot_u},
            f"sot_exterieur_{a_sot_line}": {"over": a_sot_o, "under": a_sot_u}
        }

        # 5. Détection de Value
        results = []
        for market, odds in market_odds.items():
            for sel, odd in odds.items():
                prob = preds_all[market][sel]
                edge = prob * odd - 1
                
                if edge > min_edge and (cote_min <= odd <= cote_max):
                    b = odd - 1
                    kelly_quart = max(0.0, (b * prob - (1 - prob)) / b) * 0.25 if b > 0 else 0.0
                    results.append(ValueBetResult(market, sel, prob, odd, edge, kelly_quart))
        
        results.sort(key=lambda x: x.edge, reverse=True)

        # 6. Affichage
        st.header(f"📊 Rapport : {home_team} vs {away_team}")
        if match_mode == "En direct (Live)":
            st.caption(f"⚡ Analyse Live à la {live_minute}e minute | Score actuel : {live_home_score} - {live_away_score}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("xG Dom", f"{preds_goals['expected_goals']['home']:.2f}")
        c2.metric("xG Ext", f"{preds_goals['expected_goals']['away']:.2f}")
        c3.metric("Tirs Attendus Dom", f"{lam_h_shots:.1f}")
        c4.metric("Tirs Attendus Ext", f"{lam_a_shots:.1f}")

        st.subheader(f"💸 Value Bets Détectés (Stratégie : {strategie.split('(')[0].strip()})")
        if not results:
            st.info("Aucun Value Bet détecté pour ce match avec vos critères actuels.")
        else:
            for vb in results:
                st.success(f"🎯 **[{vb.market.upper()}] Option : {vb.selection.upper()}**")
                st.write(f"• Probabilité estimée : **{vb.model_prob:.1%}** | Cote saisie : **{vb.bookmaker_odds}**")
                st.write(f"• **EDGE : +{vb.edge:.1%}** | Mise Kelly (1/4) conseillée : **{vb.kelly_quart:.1%}**")
