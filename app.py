import math
import numpy as np
import pandas as pd
import streamlit as st

# Dati degli articoli (valori stimati)
# stagionalita = indici dei mesi da gennaio a dicembre
articoli = {
    "Pallone da calcio F500": {
        "livello": 2400, "trend": 0.0, "variabilita": 0.10,
        "stagionalita": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "costo": 7.5, "costo_ordine": 180, "lead_time": 2.5},

    "Bici elettrica da citta": {
        "livello": 120, "trend": 0.02, "variabilita": 0.10,
        "stagionalita": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "costo": 650, "costo_ordine": 250, "lead_time": 1.5},

    "Borraccia 1 L": {
        "livello": 1500, "trend": 0.0, "variabilita": 0.10,
        "stagionalita": [0.7, 0.7, 0.85, 1.0, 1.2, 1.35, 1.4, 1.3, 1.0, 0.85, 0.75, 0.9],
        "costo": 4, "costo_ordine": 150, "lead_time": 2.5},

    "Tappetino yoga 8 mm": {
        "livello": 1400, "trend": 0.005, "variabilita": 0.25,
        "stagionalita": [1.3, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.85, 1.2, 1.05, 0.95, 1.1],
        "costo": 5.5, "costo_ordine": 150, "lead_time": 2.5},
}

nomi_mesi = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
primo_mese = 0  # lo storico inizia a gennaio 2022 (gennaio = 0)

def genera_domanda(art, seme, var=1.0, trend=None, stag=1.0):
    # domanda = livello * (1 + trend * t) * indice stagionale * (1 + rumore)
    if trend is None:
        trend = art["trend"]
    np.random.seed(seme)
    rumore = np.random.normal(0, 1, 48)
    domanda = []
    for t in range(48):
        mese = (primo_mese + t) % 12
        indice = 1 + stag * (art["stagionalita"][mese] - 1)
        d = art["livello"] * (1 + trend * t) * indice * (1 + art["variabilita"] * var * rumore[t])
        domanda.append(max(0, round(d)))
    return np.array(domanda, dtype=float)


# Metodi di previsione (previsione del mese successivo)

def media_mobile(dati, n):
    return dati[-n:].mean()


def media_ponderata(dati, n):
    pesi = np.arange(1, n + 1)  # il mese piu recente ha il peso maggiore
    return (dati[-n:] * pesi).sum() / pesi.sum()


def esponenziale(dati, alfa):
    prev = dati[0]
    for valore in dati[1:]:
        prev = alfa * valore + (1 - alfa) * prev
    return prev


def prevedi(metodo, dati, n, alfa):
    if metodo == "Media mobile semplice":
        return media_mobile(dati, n)
    if metodo == "Media mobile ponderata":
        return media_ponderata(dati, n)
    return esponenziale(dati, alfa)


metodi = ["Media mobile semplice", "Media mobile ponderata", "Attenuazione esponenziale"]


# Barra laterale: scelta dell'articolo e dei parametri

st.sidebar.title("Parametri")
nome = st.sidebar.selectbox("Articolo", list(articoli.keys()))
art = articoli[nome]

st.sidebar.write("**Domanda**")
scelta_var = st.sidebar.radio("Variabilita della domanda", ["Bassa", "Normale", "Alta"], index=1)
var = {"Bassa": 0.5, "Normale": 1, "Alta": 2}[scelta_var]
trend = st.sidebar.slider("Crescita della domanda (% al mese)", -2.0, 4.0, art["trend"] * 100, 0.5) / 100
scelta_stag = st.sidebar.radio("Stagionalita", ["Nessuna", "Normale", "Forte"], index=1)
stag = {"Nessuna": 0, "Normale": 1, "Forte": 2}[scelta_stag]

st.sidebar.write("**Approvvigionamento**")
lead_time = st.sidebar.slider("Tempo di consegna del fornitore (mesi)", 0.5, 6.0, art["lead_time"], 0.5)
scelta_servizio = st.sidebar.radio("Livello di servizio", ["90%", "95%", "99%"], index=1)
servizio = {"90%": 0.90, "95%": 0.95, "99%": 0.99}[scelta_servizio]


# parametri fissi
capacita = 1.2  # il fornitore consegna al massimo il 120% della domanda media mensile
n = 3           # mesi usati dalle medie mobili
alfa = 0.3      # coefficiente dell'attenuazione esponenziale


# Dati: se i parametri sono quelli di base uso il file CSV, altrimenti rigenero la domanda con i nuovi parametri

storico = pd.read_csv("domanda.csv", index_col="mese", parse_dates=True)
seme = list(articoli.keys()).index(nome) + 1
if var == 1 and stag == 1 and trend == art["trend"]:
    domanda = storico[nome].values.astype(float)
else:
    domanda = genera_domanda(art, seme, var, trend, stag)

date = storico.index
date_future = pd.date_range("2026-01-01", periods=12, freq="MS")


# Confronto dei metodi sugli ultimi 12 mesi. Per ogni mese si fa la previsione usando solo i mesi precedenti

risultati = []
prev_test = {}
reale = domanda[36:]
for m in metodi:
    prev = []
    for t in range(36, 48):
        prev.append(prevedi(m, domanda[:t], n, alfa))
    prev = np.array(prev)
    prev_test[m] = prev
    errore = reale - prev
    risultati.append({
        "Metodo": m,
        "Bias": round(errore.mean(), 1),
        "MAD": round(abs(errore).mean(), 1),
        "MSE": round((errore ** 2).mean()),
        "MAPE %": round((abs(errore) / reale).mean() * 100, 1),
    })
tabella = pd.DataFrame(risultati)
migliore = tabella.loc[tabella["MAD"].idxmin(), "Metodo"]
mse_migliore = tabella.loc[tabella["MAD"].idxmin(), "MSE"]

# previsione per i prossimi 12 mesi con il metodo migliore (valore costante)
previsione_mese = prevedi(migliore, domanda, n, alfa)


# Indici stagionali calcolati dallo storico. Per ogni anno divido la domanda del mese per la media dell'anno, poi faccio la media dei 4 anni
indici = np.zeros(12)
for anno in range(4):
    blocco = domanda[anno * 12:(anno + 1) * 12]
    indici += blocco / blocco.mean()
indici = indici / 4
previsione_stagionale = previsione_mese * indici  # da ottobre a settembre


# Calcolo delle scorte

D = previsione_mese * 12               # domanda annua prevista
d = previsione_mese                    # domanda media mensile
i = 0.25                               # costo di mantenimento annuo (25% del costo)
z = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}[servizio]
sigma = math.sqrt(mse_migliore)        # deviazione standard dell'errore

eoq = math.sqrt(2 * D * art["costo_ordine"] / (i * art["costo"]))
scorta_ciclica = eoq / 2
scorta_sicurezza = z * sigma * math.sqrt(lead_time)
scorta_pipeline = d * lead_time

# scorta anticipata - nei mesi in cui la domanda supera la capacita del fornitore la differenza deve essere prodotta e accumulata nei mesi precedenti
cap = capacita * d
anticipata = [0.0] * 12
for t in range(10, -1, -1):
    anticipata[t] = max(0, anticipata[t + 1] + previsione_stagionale[t + 1] - cap)
scorta_anticipata = sum(anticipata) / 12

totale = scorta_ciclica + scorta_sicurezza + scorta_pipeline + scorta_anticipata


# Pagina principale

st.title("Previsione della domanda e scorte - Decathlon Italia")
st.write("Articolo selezionato:", nome)

st.header("1. Domanda storica e previsione")
grafico = pd.DataFrame(index=date.append(date_future))
grafico["Domanda storica"] = list(domanda) + [None] * 12
grafico["Previsione"] = [None] * 48 + [previsione_mese] * 12
st.line_chart(grafico)
st.write(f"Metodo migliore (MAD piu basso): **{migliore}**. "
         f"Previsione: {previsione_mese:.0f} unita al mese, {D:.0f} unita in un anno.")

st.header("2. Confronto dei metodi (ultimi 12 mesi)")
st.table(tabella)
confronto = pd.DataFrame({"Domanda reale": reale}, index=date[36:])
for m in metodi:
    confronto[m] = prev_test[m]
st.line_chart(confronto)

st.header("3. Scorte")
scorte = pd.DataFrame({
    "Tipo di scorta": ["Ciclica", "Sicurezza", "Anticipata (media)", "Pipeline", "Totale"],
    "Unita": [scorta_ciclica, scorta_sicurezza, scorta_anticipata, scorta_pipeline, totale],
})
scorte["Valore (euro)"] = scorte["Unita"] * art["costo"]
st.table(scorte.round(0).astype({"Unita": int, "Valore (euro)": int}))
st.bar_chart(scorte.iloc[:4].set_index("Tipo di scorta")["Unita"])
st.write(f"Lotto economico EOQ = {eoq:.0f} unita, punto di riordino = {d * lead_time + scorta_sicurezza:.0f} unita.")

st.subheader("Scorta anticipata")
st.write("La previsione viene distribuita sui mesi con gli indici stagionali dello storico. "
         "Quando la domanda di un mese supera la capacita del fornitore serve scorta anticipata.")
mesi_futuri = [f"{t + 1:02d} {nomi_mesi[(primo_mese + t) % 12]}" for t in range(12)]
st.line_chart(pd.DataFrame({"Domanda prevista": previsione_stagionale,
                            "Capacita fornitore": [cap] * 12}, index=mesi_futuri))
st.bar_chart(pd.DataFrame({"Scorta anticipata": anticipata}, index=mesi_futuri))

st.header("4. Confronto tra scenari")
if "scenari" not in st.session_state:
    st.session_state.scenari = []
nome_scenario = st.text_input("Nome scenario", "Base")
if st.button("Salva scenario"):
    st.session_state.scenari.append({
        "Scenario": nome_scenario, "Articolo": nome,
        "Ciclica": round(scorta_ciclica), "Sicurezza": round(scorta_sicurezza),
        "Anticipata": round(scorta_anticipata), "Pipeline": round(scorta_pipeline),
        "Totale": round(totale)})
if st.session_state.scenari:
    tab_scenari = pd.DataFrame(st.session_state.scenari)
    st.table(tab_scenari)
    st.bar_chart(tab_scenari.set_index("Scenario")[["Ciclica", "Sicurezza", "Anticipata", "Pipeline"]])
