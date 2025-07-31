
# Detector de Correos Phishing con Machine Learning

Repositorio oficial del proyecto **Cochitrends Phishing Detector**, una herramienta web que detecta correos maliciosos usando Machine Learning y listas negras de dominios.

![Interfaz del detector](/images/Interfaz.png)

---

## Características Principales

- **Clasificación híbrida**: modelo ML + detección de URLs maliciosas por dominio
- **Modelo eficiente** entrenado con Random Forest + TF-IDF
- **Interfaz web con Flask** para análisis en tiempo real
- **Detección avanzada** de patrones comunes de phishing:
  - Lenguaje alarmista
  - Peticiones de verificación
  - Enlaces falsos
  - Correos suplantados

---

## Requisitos

- Python 3.10+
- pip
- Git

---

## Conjunto de datos utilizados:

- Spam Mails Dataset – Kaggle (venky73)
🔗 https://www.kaggle.com/datasets/venky73/spam-mails-dataset
Conjunto de correos clasificados como spam y ham (legítimos), ampliamente usado en modelos de NLP.

- Spam Detector with 98% Accuracy – Kaggle Notebook (sultansagynov)
🔗 https://www.kaggle.com/code/sultansagynov/spam-detector-with-98-accuracy/input
Datos y scripts adicionales para la creación de un detector de spam.

- PhishTank JSON API (phishtank.org)
🔗 https://phishtank.org/developer_info.php
Base de datos en línea colaborativa con URLs confirmadas como phishing. Se utilizó el archivo JSON descargado para entrenar el modelo y alimentar la lista negra.

- Enron Email Dataset – Kaggle (wcukierski)
🔗 https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
Base de correos electrónicos reales obtenidos del caso Enron, útil para análisis de correos legítimos.

---

## ⚙️ Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/ddgant/Cochitrends_PhishingDetector.git
   cd Cochitrends_PhishingDetector
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Asegúrate de tener los siguientes datasets en la carpeta `/dataset`:
   - `spam_datasetOne.csv`, `spam_datasetTwo.csv`, `spam_datasetThree.csv`, etc.
   - `links_dataset.json` (PhishTank u otro)
   - `dominios_datasetFive.csv` (dominios maliciosos con columna `url`)

   - Link de descarga completa "dataset": https://drive.google.com/drive/folders/1HcmZlMrNkq3kt4_DfD2UXPEfgrcXpc0Y?usp=drive_link

4. Entrena el modelo:
   ```bash
   python train_model.py
   ```

5. Inicia la aplicación Flask:
   ```bash
   python app.py
   ```

---

## Uso

1. Abre tu navegador y ve a `http://localhost:5000`
2. Pega el texto del correo electrónico sospechoso
3. Haz clic en **"Analizar"**

### Ejemplo de entrada:
```
Estimado usuario: su cuenta PayPal ha sido suspendida. Ingrese a http://paypal-falso.com/verify
```

### Salida esperada:
```
Clasificación: PHISHING
Confianza: 99.8%
Motivo: Dominio malicioso detectado: paypal-falso.com
```

---

## Estructura del Proyecto

```
Cochitrends_PhishingDetector/
├── app.py                     # Servidor Flask
├── train_model.py             # Entrenamiento del modelo ML
├── model/
│   ├── spam_model.pkl         # Modelo Random Forest
│   └── vectorizer.pkl         # Vectorizador TF-IDF
├── dataset/
│   ├── spam_datasetOne.csv
│   ├── spam_datasetTwo.csv
│   ├── dominios_datasetFive.csv
│   └── links_dataset.json
├── templates/
│   └── index.html             # Interfaz HTML
├── static/
│   └── style.css              # Estilo personalizado (si aplica)
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación del proyecto
```

---

## 🔧 Detalles Técnicos del Modelo

| Elemento           | Configuración                            |
|--------------------|-------------------------------------------|
| Modelo             | Random Forest (`n_estimators=100`)       |
| Vectorización      | TF-IDF (unigramas y bigramas)            |
| Features           | Hasta 20,000 términos                    |
| Entrenamiento      | 80% training, 20% testing                |
| Precisión estimada | ~96-97%                                  |
| Blacklist          | Dominio extraído con `urlparse`          |

---

## API REST (opcional)

**Endpoint `/predict`** — para integración con otras apps:

```
POST /predict HTTP/1.1
Content-Type: text/plain

"Su cuenta ha sido bloqueada. Verifíquela en http://dominio-malicioso.com"
```

**Respuesta esperada:**
```json
{
  "clasificacion": "PHISHING",
  "confianza": 0.98,
  "motivo": "Dominio malicioso detectado: dominio-malicioso.com"
}
```

---

## Solución de Problemas

** Error:** `FileNotFoundError: model/spam_model.pkl`  
** Solución:** Reentrena el modelo:

```bash
python train_model.py
```

---

## Contribuciones

1. Haz fork del repositorio
2. Crea tu branch: `git checkout -b feature/nueva-funcionalidad`
3. Realiza tus cambios y haz commit: `git commit -m "Agrega nueva funcionalidad"`
4. Haz push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request 🚀

---

## Integrantes de proyecto:

Diego De Gante Pérez, 
Joel Alejandro Pérez Yupit, 
Hungman Emmanuel Chong Santiago, 
Hector Javier Raya Romo, 
Cesar Arturo Balam Euan
