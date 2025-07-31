# --------------------------
# Importación de librerías
# --------------------------

import pandas as pd  # Para manipular datos en tablas
import os  # Para trabajar con archivos y carpetas
import joblib  # Para guardar/cargar el modelo entrenado
import re  # Para usar expresiones regulares (limpieza de texto)
import json  # Para leer archivos JSON (como phishtank.json)
from sklearn.ensemble import RandomForestClassifier  # Modelo de árboles para clasificación
from sklearn.feature_extraction.text import TfidfVectorizer  # Para convertir texto en números (vectores)
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.metrics import accuracy_score, classification_report  # Para medir el rendimiento del modelo
import nltk  # Librería para procesar texto
from nltk.corpus import stopwords  # Lista de palabras comunes que se pueden eliminar
from urllib.parse import urlparse  # Para analizar URLs

# --------------------------
# Descarga las stopwords si no están
# --------------------------

nltk.download('stopwords')  # Solo se descarga una vez

# --------------------------
# Crear carpeta 'model' si no existe
# --------------------------

os.makedirs('model', exist_ok=True)  # Se usa para guardar el modelo entrenado

# --------------------------
# Función para detectar columnas relevantes
# --------------------------

def detect_columns(df):
    # Lista de nombres posibles para la columna que contiene el texto
    text_candidates = ['body', 'content', 'text', 'message', 'email_body', 'url']
    # Lista de nombres posibles para la columna que contiene la etiqueta (spam o no spam)
    label_candidates = ['label', 'class', 'spam', 'type']
    
    # Buscar la primera columna que coincida
    text_col = next((col for col in text_candidates if col in df.columns), None)
    label_col = next((col for col in label_candidates if col in df.columns), None)
    
    return text_col, label_col  # Regresa el nombre de las columnas encontradas

# --------------------------
# Función para limpiar texto
# --------------------------

def clean_text(text):
    text = str(text).lower()  # Convertir todo a minúsculas
    
    # Reemplazar enlaces por un token
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL_TOKEN', text)
    
    # Reemplazar correos electrónicos por un token
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL_TOKEN', text)
    
    # Reemplazar números por un token
    text = re.sub(r'\d+', 'NUM_TOKEN', text)
    
    # Eliminar signos de puntuación
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Quitar espacios dobles
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Eliminar palabras vacías (como "the", "and", "is", etc.)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words and len(word) > 2])

# --------------------------
# Leer archivos CSV del folder 'dataset'
# --------------------------

main_files = [f for f in os.listdir('dataset') if f.endswith('.csv') and '_vectorized' not in f]
dfs = []  # Lista donde se guardarán los DataFrames válidos

# Iterar sobre cada archivo CSV
for file in main_files:
    file_path = os.path.join('dataset', file)
    df_temp = None
    
    try:
        # Intenta abrir el archivo con codificación latin1
        df_temp = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', engine='python')
    except:
        try:
            # Si falla, intenta con codificación utf-8
            df_temp = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')
        except Exception as e:
            print(f" Error en {file}: {str(e)[:50]}")
            continue  # Saltar este archivo si no se pudo leer

    # Verificar que tenga al menos dos columnas
    if df_temp is None or df_temp.shape[1] < 2:
        print(f" {file} no tiene suficientes columnas")
        continue

    # Buscar columnas de texto y etiqueta
    text_col, label_col = detect_columns(df_temp)
    if not text_col or not label_col:
        print(f" Columnas no detectadas en {file}")
        continue

    # Renombrar columnas a 'text' y 'label'
    df_temp = df_temp.rename(columns={text_col: 'text', label_col: 'label'})
    
    # Quedarse solo con esas dos columnas
    df_temp = df_temp[['text', 'label']].copy()

    # Limpiar texto
    df_temp['text'] = df_temp['text'].apply(clean_text)
    
    # Eliminar textos muy cortos (menos de 10 caracteres)
    df_temp = df_temp[df_temp['text'].str.len() > 10]
    
    # Convertir etiquetas: 0 si es 'ham' o 'benign', 1 en cualquier otro caso
    df_temp['label'] = df_temp['label'].apply(lambda x: 0 if str(x).lower() in ['ham', 'benign'] else 1)
    
    # Agregar el DataFrame procesado a la lista
    dfs.append(df_temp)

    print(f" {file}: {len(df_temp)} registros (Cols: {text_col}/{label_col})")

# --------------------------
# Leer archivo JSON de phishtank (URLs de phishing)
# --------------------------

phishtank_json = 'dataset/phishtank.json'
if os.path.exists(phishtank_json):
    try:
        with open(phishtank_json, 'r', encoding='utf-8') as f:
            phish_data = json.load(f)
        
        # Filtrar solo las URLs verificadas y activas
        phishing_urls = [entry['url'] for entry in phish_data if entry.get('verified') == 'yes' and entry.get('online') == 'yes']
        
        # Crear DataFrame
        df_phish_urls = pd.DataFrame(phishing_urls, columns=['text'])
        df_phish_urls['label'] = 1  # Etiquetar como phishing
        df_phish_urls['text'] = df_phish_urls['text'].apply(clean_text)
        
        dfs.append(df_phish_urls)
        print(f" phishtank.json: {len(df_phish_urls)} URLs añadidas como phishing")

    except Exception as e:
        print(f" Error al procesar phishtank.json: {e}")

# --------------------------
# Verificar que haya datos válidos
# --------------------------

if not dfs:
    print(" No se encontraron datos válidos")
    exit()  # Salir del programa si no hay datos

# Unir todos los DataFrames en uno solo
df = pd.concat(dfs, ignore_index=True)
print(f"\nDataset final: {len(df)} registros")
print("Distribución de clases:\n", df['label'].value_counts())

# --------------------------
# Vectorizar el texto con TF-IDF
# --------------------------

vectorizer = TfidfVectorizer(
    max_features=20000,  # Número máximo de palabras únicas
    ngram_range=(1, 2),  # Unigramas y bigramas
    stop_words='english'  # Elimina palabras comunes automáticamente
)

# Convertir texto en vectores numéricos
X = vectorizer.fit_transform(df['text'])  # Matriz de características
y = df['label']  # Etiquetas

# --------------------------
# Dividir datos en entrenamiento y prueba
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# Crear y entrenar el modelo con Random Forest
# --------------------------

model = RandomForestClassifier(
    n_estimators=100,  # Número de árboles en el bosque
    random_state=42,  # Para obtener resultados reproducibles
    n_jobs=-1  # Usar todos los núcleos disponibles
)

model.fit(X_train, y_train)  # Entrenar el modelo con los datos

# --------------------------
# Evaluar el modelo
# --------------------------

y_pred = model.predict(X_test)  # Predecir etiquetas con el modelo

# Calcular la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nExactitud del modelo: {accuracy:.4f}")

# Mostrar métricas detalladas: precisión, recall, F1
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# --------------------------
# Guardar el modelo entrenado y el vectorizador
# --------------------------

joblib.dump(model, 'model/spam_model.pkl')  # Guardar modelo
joblib.dump(vectorizer, 'model/vectorizer.pkl')  # Guardar vectorizador

print("\n Modelo entrenado y guardado exitosamente.")


# --------------------------
# Leer lista de dominios maliciosos desde CSV
# --------------------------

malicious_domains = set()

try:
    df_domains = pd.read_csv('dataset/your_malicious_domains.csv')  # Reemplaza con el nombre correcto
    df_maliciosos = df_domains[~df_domains['type'].str.lower().isin(['benign', 'ham'])]
    
    for url in df_maliciosos['url']:
        parsed = urlparse(url)
        domain = parsed.netloc if parsed.netloc else parsed.path  # netloc es el dominio
        domain = domain.lower()
        malicious_domains.add(domain)

    print(f"\n{len(malicious_domains)} dominios maliciosos cargados para detección directa.")

except Exception as e:
    print("Error al cargar dominios maliciosos:", e)

# --------------------------
# Verificar si texto contiene un dominio malicioso
# --------------------------

def contieneDominioMalicioso(texto):
    for dominio in malicious_domains:
        if dominio in texto.lower():
            return dominio
    return None

# --------------------------
# Clasificación híbrida: modelo + lista negra
# --------------------------

def clasificarMensaje(textoOriginal):
    textoLimpio = clean_text(textoOriginal)
    
    # Paso 1: Revisar dominios maliciosos
    dominioDetectado = contieneDominioMalicioso(textoOriginal)
    if dominioDetectado:
        return {
            'clasificacion': 'PHISHING',
            'confianza': 1.0,
            'motivo': f'Dominio malicioso detectado: {dominioDetectado}'
        }
    
    # Paso 2: Clasificación por modelo
    vectorizado = vectorizer.transform([textoLimpio])
    prediccion = model.predict(vectorizado)[0]
    probabilidad = model.predict_proba(vectorizado)[0][1]  # probabilidad de phishing

    return {
        'clasificacion': 'PHISHING' if prediccion == 1 else 'NORMAL',
        'confianza': round(probabilidad, 2),
        'motivo': 'Clasificado por modelo'
    }
