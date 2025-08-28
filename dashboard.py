import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Daten laden ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/sea_levels_2015.csv")
    df['Time'] = pd.to_datetime(df['Time']).dt.year
    return df

df = load_data()

# --- Dashboard Intro (zentriert) ---
st.markdown("""
<div style="text-align: center;">
<h1>Global Sea Level Change – Data Dashboard</h1>

<p>Dieses Dashboard analysiert die langfristige Entwicklung des globalen Meeresspiegels auf Basis historischer Beobachtungsdaten seit dem 19. Jahrhundert.<br>
Ziel ist es, zentrale Muster und Trends sichtbar zu machen sowie statistische Verfahren zur Trendabschätzung zu demonstrieren.</p>

<b>Inhalte:</b>
<ul style="list-style-type: none;">
<li>Zeitreihe: Visualisierung der jährlichen Veränderungen des globalen mittleren Meeresspiegels (GMSL).</li>
<li>Lineare Regression: Modellierung des langfristigen Trends zur quantitativen Abschätzung der Steigerungsrate.</li>
<li>Verteilungsanalyse: Histogramm und Boxplot zur Untersuchung der statistischen Eigenschaften der Daten.</li>
</ul>

<b>Datenquelle:</b><br>
Die Daten stammen aus dem Kaggle-Datensatz <i>Sea Level Change</i> (<a href="https://www.kaggle.com/datasets/somesh24/sea-level-change">somesh24/sea-level-change</a>).
</div>
""", unsafe_allow_html=True)

# --- Data Overview ---
st.markdown('<h2 style="text-align:center;">Datenvorschau</h2>', unsafe_allow_html=True)
st.write(df.head())

st.markdown("""  
<div style="text-align: center;">
Wir prüfen nun die Struktur und grundlegende Statistiken des Datensatzes.  

<b>Spalten:</b>
<ul style="list-style-type: none;">
<li><b>Time</b>: enthält monatliche Beobachtungszeitpunkte des Meeresspiegels, beginnend ab 1880.</li>
<li><b>GMSL</b>: <i>Global Mean Sea Level</i> (globaler mittlerer Meeresspiegel), angegeben in Millimetern relativ zu einem Referenzniveau.</li>
<li><b>GMSL Uncertainty</b>: Unsicherheitsbereich der Messung (z. B. aufgrund instrumenteller oder methodischer Limitationen).</li>
</ul>
</div>
""", unsafe_allow_html=True)

# --- Plot 1: Zeitreihe ---
st.markdown('<h2 style="text-align:center;">Meeresspiegel über die Zeit</h2>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=df, x="Time", y="GMSL", ax=ax)
plt.title("Globaler Meeresspiegel (Zeitreihe)")
st.pyplot(fig)

st.markdown("""  
<div style="text-align: center;">
<h3>Zeitreihe des globalen Meeresspiegels</h3>
<p>Die nachfolgende Abbildung zeigt die Entwicklung des <b>globalen mittleren Meeresspiegels (GMSL)</b> seit dem späten 19. Jahrhundert.<br>
Deutlich erkennbar ist ein langfristiger Anstieg, der auf den Einfluss des <b>Klimawandels</b>, insbesondere das Abschmelzen von Gletschern und Eisschilden sowie die thermische Ausdehnung der Ozeane, zurückzuführen ist.</p>

<p>Die Zeitreihe bildet die Grundlage für weitere Analysen, wie die Bestimmung von Trends mittels <b>linearer Regression</b>.</p>
</div>
""", unsafe_allow_html=True)

# --- Plot 2: Lineare Regression ---
st.markdown('<h2 style="text-align:center;">Lineare Regression</h2>', unsafe_allow_html=True)
X = df["Time"].values.reshape(-1,1)
y = df["GMSL"].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X, y, label="Daten", color="blue")
ax.plot(X, y_pred, color="red", label="Lineare Regression")
ax.set_title("Meeresspiegeltrend (Regression)")
ax.set_xlabel("Jahr")
ax.set_ylabel("Meeresspiegel (mm)")
ax.legend()
st.pyplot(fig)

st.markdown("""  
<div style="text-align: center;">
<h3>Lineare Regression</h3>
<p>Um den langfristigen Trend des globalen Meeresspiegels zu quantifizieren,  
wurde eine <b>lineare Regression</b> durchgeführt.</p>

<p>Die Regressionsgerade zeigt den durchschnittlichen jährlichen Anstieg über den gesamten Zeitraum.<br>
Dieser lineare Trend verdeutlicht die <b>kontinuierliche und signifikante Zunahme</b> des Meeresspiegels,  
auch wenn kurzfristige Schwankungen durch natürliche Klimavariabilität auftreten.</p>

<p>Die Analyse liefert damit eine wichtige Grundlage für Prognosen und die Bewertung zukünftiger Risiken.</p>
</div>
""", unsafe_allow_html=True)

# --- Histogramm ---
st.markdown('<h2 style="text-align:center;">Histogramm der Meeresspiegelwerte</h2>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df["GMSL"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

st.markdown("""  
<div style="text-align: center;">
<h3>Histogramm der Meeresspiegelwerte</h3>
<p>Das Histogramm veranschaulicht die <b>Verteilung der gemessenen Meeresspiegelwerte</b>.</p>

<p>Deutlich wird eine Verschiebung hin zu höheren Werten im Zeitverlauf,  
was den in der Zeitreihe erkennbaren Anstieg zusätzlich bestätigt.</p>

<p>Eine solche Verteilung ist hilfreich, um die <b>Variabilität und Streuung</b> der Daten zu verstehen,  
sowie mögliche <b>Ausreißer oder Extremwerte</b> zu identifizieren.</p>
</div>
""", unsafe_allow_html=True)
