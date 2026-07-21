import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import io
from scipy.optimize import minimize
from scipy.stats import poisson
from dataclasses import dataclass

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Value Bet Scanner Pro", page_icon="⚽", layout="wide")

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
    "norvege": {"country": "Norvège - Eliteserien", "fd_code": "NOR"},
    "finlande": {"country": "Finlande - Veikkausliiga", "fd_code": "FIN"},
    "suede": {"country": "Suède - Allsvenskan", "fd_code": "SWE"},
}
SEASONS = ["2425", "2526", "2627"]
TIME_DECAY_HALFLIFE_DAYS = 180

# ==========================================
# 2. LOGIQUE MATHÉMATIQUE (DIXON-COLES)
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
        
        if not result.success:
            st.warning(f"Le modèle a eu du mal à converger pour ce championnat.")
            
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
        return {
            "1X2": {"H": np.tril(M, -1).sum(), "D": np.trace(M), "A": np.triu(M, 1).sum()},
            "over_under_2_5": {"over": M[goals[:, None] + goals[None, :] > 2.5].sum(), "under": M[goals[:, None] + goals[None, :] < 2.5].sum()},
            "btts": {"yes": M[1:, 1:].sum(), "no": 1 - M[1:, 1:].sum()},
            "home_goals": {"over_0_5": M[1:, :].sum(), "under_0_5": M[0, :].sum(), "over_1_5": M[2:, :].sum(), "under_1_5": M[:2, :].sum()},
            "away_goals": {"over_0_5": M[:, 1:].sum(), "under_0_5": M[:, 0].sum(), "over_1_5": M[:, 2:].sum(), "under_1_5": M[:, :2].sum()},
            "expected_goals": {"home": lam_h, "away": lam_a},
        }

# ==========================================
# 3. GESTION DES DONNÉES
# ==========================================
@dataclass
class ValueBetResult:
    market: str
    selection: str
    model_prob: float
    bookmaker_odds: float
    edge: float
    kelly_half: float

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
        except:
            continue
    if not dfs: return None
    
    data = pd.concat(dfs, ignore_index=True)
    # On nettoie uniquement ce qui est obligatoire pour les Buts (modèle principal)
    cols_to_check = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    data = data.dropna(subset=cols_to_check)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    
    for col in ['FTHG', 'FTAG']:
        data[col] = data[col].astype(int)
    return data

@st.cache_resource(show_spinner=False)
def train_all_models(df):
    goal_model = DixonColesModel().fit(df, "FTHG", "FTAG", halflife_days=TIME_DECAY_HALFLIFE_DAYS)
    
    # Conditional logic pour les stats de tirs
    shot_model = DixonColesModel().fit(df, "HS", "AS", halflife_days=TIME_DECAY_HALFLIFE_DAYS) if 'HS' in df.columns else None
    sot_model = DixonColesModel().fit(df, "HST", "AST", halflife_days=TIME_DECAY_HALFLIFE_DAYS) if 'HST' in df.columns else None
    
    return {"goals": goal_model, "shots": shot_model, "sot": sot_model}

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================
st.sidebar.header("🎯 Stratégie")
strategie = st.sidebar.selectbox("Filtre de cotes", ["Afficher Tout (Aucun filtre)", "Volume Faible (Cotes 1.50 - 1.85)", "Volume Moyen (Cotes 1.70 - 2.10)", "Volume Élevé (Cotes 1.85 - 2.30)", "Volume Massif (Cotes > 2.00)"])

if strategie == "Volume Faible (Cotes 1.50 - 1.85)": cote_min, cote_max = 1.50, 1.85
elif strategie == "Volume Moyen (Cotes 1.70 - 2.10)": cote_min, cote_max = 1.70, 2.10
elif strategie == "Volume Élevé (Cotes 1.85 - 2.30)": cote_min, cote_max = 1.85, 2.30
elif strategie == "Volume Massif (Cotes > 2.00)": cote_min, cote_max = 2.00, 100.00
else: cote_min, cote_max = 1.01, 100.00

st.title("🏆 Scanner de Value Bets Pro")
league_key = st.selectbox("1️⃣ Choisis un championnat", options=list(LEAGUES.keys()), format_func=lambda x: LEAGUES[x]["country"])

with st.spinner(f"Chargement et calibration..."):
    df = load_and_clean_data(LEAGUES[league_key]["fd_code"])
    if df is not None:
        models = train_all_models(df)
        teams_list = models["goals"].teams
        st.success(f"✅ Prêt !")
    else:
        st.error("Impossible de charger les données.")
        st.stop()

col1, col2 = st.columns(2)
with col1: home_team = st.selectbox("🏠 Domicile", options=teams_list)
with col2: away_team = st.selectbox("✈️ Extérieur", options=teams_list, index=1)

# Interface de saisie des cotes...
st.subheader("2️⃣ Saisissez les cotes")
c1, c2, c3 = st.columns(3)
h_odd = c1.number_input("Cote 1", value=2.00, step=0.01)
d_odd = c2.number_input("Cote X", value=3.40, step=0.01)
a_odd = c3.number_input("Cote 2", value=3.80, step=0.01)

# Analyse
if st.button("🚀 Lancer l'Analyse"):
    preds_goals = models["goals"].predict_goal_markets(home_team, away_team)
    
    st.header(f"📊 Rapport : {home_team} vs {away_team}")
    c1, c2 = st.columns(2)
    c1.metric("xG Dom", f"{preds_goals['expected_goals']['home']:.2f}")
    c2.metric("xG Ext", f"{preds_goals['expected_goals']['away']:.2f}")
    
    if models["shots"] is not None:
        st.info("Modèles de tirs actifs.")
    else:
        st.warning("Données de tirs non disponibles (Analyse limitée aux buts).")
