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
        
        attack, defense, rho, home_adv = unpack(result.x)
        self.params = {t: {"attack": attack[i], "defense": defense[i]} for i, t in enumerate(self.teams)}
        self.rho, self.home_advantage = home_adv
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

def remove_overround(odds):
    implied = {k: 1/v for k, v in odds.items()}
    overround = sum(implied.values())
    return {k: v/overround for k, v in implied.items()}

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
# 4. INTERFACE STREAMLIT & FILTRES
# ==========================================
# --- NOUVEAU : Barre latérale pour la stratégie ---
st.sidebar.header("🎯 Stratégie de Paris")
st.sidebar.markdown("Sélectionnez la tranche de cotes à cibler selon votre volume hebdomadaire.")

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

# Attribution des bornes selon la stratégie choisie
if strategie == "Volume Faible (Cotes 1.50 - 1.85)":
    cote_min, cote_max = 1.50, 1.85
elif strategie == "Volume Moyen (Cotes 1.70 - 2.10)":
    cote_min, cote_max = 1.70, 2.10
elif strategie == "Volume Élevé (Cotes 1.85 - 2.30)":
    cote_min, cote_max = 1.85, 2.30
elif strategie == "Volume Massif (Cotes > 2.00)":
    cote_min, cote_max = 2.00, 100.00 # 100 englobe les très hautes cotes
else:
    cote_min, cote_max = 1.01, 100.00 # Affiche tout par défaut


st.title("🏆 Scanner de Value Bets Pro (Buts & Tirs)")

league_key = st.selectbox("1️⃣ Choisis un championnat", options=list(LEAGUES.keys()), format_func=lambda x: LEAGUES[x]["country"])

with st.spinner(f"Calcul et calibration des modèles..."):
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

st.divider()

st.subheader("2️⃣ Saisissez les cotes du bookmaker")

# Expander 1 : Marchés principaux
with st.expander("⚽ MARCHÉS DES BUTS PRINCIPAUX (1X2, O/U 2.5, BTTS)", expanded=True):
    c1, c2, c3 = st.columns(3)
    h_odd = c1.number_input("Cote 1", value=2.00, step=0.01)
    d_odd = c2.number_input("Cote X", value=3.40, step=0.01)
    a_odd = c3.number_input("Cote 2", value=3.80, step=0.01)

    c1, c2, c3, c4 = st.columns(4)
    ou_over = c1.number_input("Over 2.5 (Buts)", value=1.90, step=0.01)
    ou_under = c2.number_input("Under 2.5 (Buts)", value=1.90, step=0.01)
    btts_yes = c3.number_input("BTTS Oui", value=1.85, step=0.01)
    btts_no = c4.number_input("BTTS Non", value=1.95, step=0.01)

# Expander 2 : Buts individuels par équipe
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

# Expander 3 : Tirs et tirs cadrés
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
        # --- 1. CALCULS PRÉDICTIONS MARCHÉS DES BUTS ---
        preds_goals = models["goals"].predict_goal_markets(home_team, away_team)
        
        # --- 2. CALCULS PRÉDICTIONS MARCHÉS DES TIRS ---
        lam_h_shots, lam_a_shots = models["shots"].get_lambdas(home_team, away_team)
        lam_total_shots = lam_h_shots + lam_a_shots
        prob_shots_under = poisson.cdf(int(np.floor(t_shots_line)), lam_total_shots)
        prob_shots_over = 1.0 - prob_shots_under
        
        # Tirs Individuels
        prob_h_shots_under = poisson.cdf(int(np.floor(h_shots_line)), lam_h_shots)
        prob_h_shots_over = 1.0 - prob_h_shots_under
        prob_a_shots_under = poisson.cdf(int(np.floor(a_shots_line)), lam_a_shots)
        prob_a_shots_over = 1.0 - prob_a_shots_under

        # --- 3. CALCULS PRÉDICTIONS MARCHÉS DES SOT ---
        lam_h_sot, lam_a_sot = models["sot"].get_lambdas(home_team, away_team)
        lam_total_sot = lam_h_sot + lam_a_sot
        
        prob_sot_under = poisson.cdf(int(np.floor(t_sot_line)), lam_total_sot)
        prob_sot_over = 1.0 - prob_sot_under
        
        # SOT Individuels
        prob_h_sot_under = poisson.cdf(int(np.floor(h_sot_line)), lam_h_sot)
        prob_h_sot_over = 1.0 - prob_h_sot_under
        prob_a_sot_under = poisson.cdf(int(np.floor(a_sot_line)), lam_a_sot)
        prob_a_sot_over = 1.0 - prob_a_sot_under

        # --- 4. STRUCTURE DE TOUTES LES PRÉDICTIONS ---
        preds_all = {
            "1X2": preds_goals["1X2"],
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

        # --- 5. STRUCTURE DE TOUTES LES COTES BOOKMAKERS ---
        market_odds = {
            "1X2": {"H": h_odd, "D": d_odd, "A": a_odd},
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

        # --- 6. MOTEUR DU SCANNER DE VALUE (MODIFIÉ AVEC LE FILTRE) ---
        results = []
        for market, odds in market_odds.items():
            for sel, odd in odds.items():
                prob = preds_all[market][sel]
                edge = prob * odd - 1
                
                # NOUVEAU : On vérifie l'Edge minimum ET la stratégie de cotes
                if edge > min_edge and (cote_min <= odd <= cote_max):
                    b = odd - 1
                    kelly_half = max(0.0, (b * prob - (1 - prob)) / b) * 0.5 if b > 0 else 0.0
                    results.append(ValueBetResult(market, sel, prob, odd, edge, kelly_half))
        
        results.sort(key=lambda x: x.edge, reverse=True)

        # --- 7. AFFICHAGE DES RÉSULTATS DANS L'APPLICATION ---
        st.header(f"📊 Rapport : {home_team} vs {away_team}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("xG Dom", f"{preds_goals['expected_goals']['home']:.2f}")
        c2.metric("xG Ext", f"{preds_goals['expected_goals']['away']:.2f}")
        c3.metric("Tirs Attendus Dom", f"{lam_h_shots:.1f}")
        c4.metric("Tirs Attendus Ext", f"{lam_a_shots:.1f}")

        st.subheader(f"💸 Value Bets Détectés (Stratégie : {strategie.split('(')[0].strip()})")
        if not results:
            st.info("Aucun Value Bet détecté pour ce match avec vos critères actuels (Edge ou Cotes hors limites).")
        else:
            for vb in results:
                display_market = vb.market.upper()
                display_selection = vb.selection.upper()
                
                # Formatage propre des titres de marchés
                if vb.market == "home_goals":
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
                st.write(f"• Probabilité modèle : **{vb.model_prob:.1%}** | Cote saisie : **{vb.bookmaker_odds}**")
                st.write(f"• **EDGE : +{vb.edge:.1%}** | Mise Kelly (1/2) conseillée : **{vb.kelly_half:.1%}**")