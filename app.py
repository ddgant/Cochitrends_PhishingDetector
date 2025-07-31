from flask import Flask, render_template, request
import joblib
import re
import json
import os

# --------------------------
# NUEVO: Para detección híbrida
import pandas as pd
from urllib.parse import urlparse
# --------------------------

# Crear la app Flask
app = Flask(__name__)

# Cargar el modelo de ML entrenado previamente y el vectorizador
model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Inicializamos un conjunto para almacenar URLs maliciosas (blacklist)
blacklisted_urls = set()

# Ruta donde está el archivo JSON con URLs phishing
json_path = 'dataset/links_dataset.json'  # Asegúrate de que la ruta sea correcta

# Verificamos que el archivo exista para cargarlo
if os.path.exists(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            phish_data = json.load(f)

        # Extraer URLs verificadas y online del JSON
        blacklisted_urls = set(
            entry['url'].lower() for entry in phish_data
            if entry.get('verified') == 'yes' and entry.get('online') == 'yes'
        )
        print(f" Cargadas {len(blacklisted_urls)} URLs de phishing desde JSON")

    except Exception as e:
        print(f" Error al cargar URLs de phishing: {e}")

# --------------------------
# NUEVO: Cargar CSV de dominios maliciosos (lista negra)
malicious_domains = set()
try:
    df_domains = pd.read_csv('dataset/your_malicious_domains.csv')  # Usa el nombre correcto
    df_maliciosos = df_domains[~df_domains['type'].str.lower().isin(['benign', 'ham'])]

    for url in df_maliciosos['url']:
        parsed = urlparse(url)
        domain = parsed.netloc if parsed.netloc else parsed.path
        malicious_domains.add(domain.lower())

    print(f" Cargados {len(malicious_domains)} dominios maliciosos desde CSV")
except Exception as e:
    print("Error al cargar dominios maliciosos desde CSV:", e)
# --------------------------

def contiene_url_peligrosa(text):
    """
    Extrae URLs del texto y verifica si alguna está en la lista negra (JSON).
    """
    urls = re.findall(r'https?://[^\s]+', text.lower())
    return any(url in blacklisted_urls for url in urls)

# --------------------------
# NUEVO: Verificar si hay dominios maliciosos conocidos (CSV)
def contiene_dominio_malicioso(texto):
    for dominio in malicious_domains:
        if dominio in texto.lower():
            return dominio
    return None
# --------------------------

def predict_spam(text):
    """
    Clasifica el mensaje como LEGÍTIMO, POSIBLE PHISHING o PHISHING.
    - Si hay URL en JSON => PHISHING (URL detectada)
    - Si hay dominio en CSV => PHISHING (Dominio detectado)
    - Si no, clasifica con modelo y umbral
    """
    if contiene_url_peligrosa(text):
        return 'PHISHING (URL detectada)', 100.0

    dominio = contiene_dominio_malicioso(text)
    if dominio:
        return f'PHISHING (Dominio detectado: {dominio})', 100.0

    # Preprocesar texto
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    vector = vectorizer.transform([cleaned_text])

    # Predecir
    prediction = model.predict(vector)

    try:
        proba = model.predict_proba(vector)[0][1]
    except AttributeError:
        proba = 0.5

    confidence = round(proba * 100, 2)

    if confidence < 30:
        label = 'LEGÍTIMO'
    elif 30 <= confidence <= 80:
        label = 'POSIBLE PHISHING'
    else:
        label = 'PHISHING'

    return label, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        text = request.form.get('text', '')

        if text:
            prediction, confidence = predict_spam(text)
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'text_preview': text[:500] + '...' if len(text) > 500 else text
            }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
