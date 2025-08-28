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

st.markdown("""
# Global Sea Level Change – Data Dashboard

Dieses Dashboard analysiert die langfristige Entwicklung des globalen Meeresspiegels auf Basis historischer Beobachtungsdaten seit dem 19. Jahrhundert.  
Ziel ist es, zentrale Muster und Trends sichtbar zu machen sowie statistische Verfahren zur Trendabschätzung zu demonstrieren.  

**Inhalte:**
- Zeitreihe: Visualisierung der jährlichen Veränderungen des globalen mittleren Meeresspiegels (GMSL).  
- Lineare Regression: Modellierung des langfristigen Trends zur quantitativen Abschätzung der Steigerungsrate.  
- Verteilungsanalyse: Histogramm und Boxplot zur Untersuchung der statistischen Eigenschaften der Daten.  

**Datenquelle:**  
Die Daten stammen aus dem Kaggle-Datensatz *Sea Level Change* ([somesh24/sea-level-change](https://www.kaggle.com/datasets/somesh24/sea-level-change)).
""")

# --- Data Overview ---
st.subheader("Datenvorschau")
st.write(df.head())

st.markdown("""  
Wir prüfen nun die Struktur und grundlegende Statistiken des Datensatzes.  

**Spalten:**
- **Time**: enthält monatliche Beobachtungszeitpunkte des Meeresspiegels, beginnend ab 1880.  
- **GMSL**: *Global Mean Sea Level* (globaler mittlerer Meeresspiegel), angegeben in Millimetern relativ zu einem Referenzniveau.  
- **GMSL Uncertainty**: Unsicherheitsbereich der Messung (z. B. aufgrund instrumenteller oder methodischer Limitationen).  
""")

# --- Plot 1: Zeitreihe ---
st.subheader("Meeresspiegel über die Zeit")
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=df, x="Time", y="GMSL", ax=ax)
plt.title("Globaler Meeresspiegel (Zeitreihe)")
st.pyplot(fig)

st.markdown("""  
### Zeitreihe des globalen Meeresspiegels  

Die nachfolgende Abbildung zeigt die Entwicklung des **globalen mittleren Meeresspiegels (GMSL)** seit dem späten 19. Jahrhundert.  
Deutlich erkennbar ist ein langfristiger Anstieg, der auf den Einfluss des **Klimawandels**, insbesondere das Abschmelzen von Gletschern und Eisschilden sowie die thermische Ausdehnung der Ozeane, zurückzuführen ist.  

Die Zeitreihe bildet die Grundlage für weitere Analysen, wie die Bestimmung von Trends mittels **linearer Regression**.  
""")

# --- Plot 2: Lineare Regression ---
st.subheader("Lineare Regression")
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
### Lineare Regression  

Um den langfristigen Trend des globalen Meeresspiegels zu quantifizieren,  
wurde eine **lineare Regression** durchgeführt.  

Die Regressionsgerade zeigt den durchschnittlichen jährlichen Anstieg über den gesamten Zeitraum.  
Dieser lineare Trend verdeutlicht die **kontinuierliche und signifikante Zunahme** des Meeresspiegels,  
auch wenn kurzfristige Schwankungen durch natürliche Klimavariabilität auftreten.  

Die Analyse liefert damit eine wichtige Grundlage für Prognosen und die Bewertung zukünftiger Risiken.  
""")

# --- Histogramm ---
st.subheader("Histogramm der Meeresspiegelwerte")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df["GMSL"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

st.markdown("""  
### Histogramm der Meeresspiegelwerte  

Das Histogramm veranschaulicht die **Verteilung der gemessenen Meeresspiegelwerte**.  

Deutlich wird eine Verschiebung hin zu höheren Werten im Zeitverlauf,  
was den in der Zeitreihe erkennbaren Anstieg zusätzlich bestätigt.  

Eine solche Verteilung ist hilfreich, um die **Variabilität und Streuung** der Daten zu verstehen,  
sowie mögliche **Ausreißer oder Extremwerte** zu identifizieren.  
""")
