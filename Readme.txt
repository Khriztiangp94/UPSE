# Comparativa de Modelos de Aprendizaje Automático y Profundo para Detección de Spam SMS

## Descripción
Este proyecto realiza un **análisis comparativo** entre modelos de aprendizaje automático superficial y profundo aplicados a la detección de mensajes SMS spam en el contexto de telecomunicaciones.

Se implementan y comparan tres modelos:
- **SVM** (Support Vector Machine) — modelo clásico supervisado.
- **MLP** (Perceptrón Multicapa) — red neuronal densa.
- **LSTM** (Long Short-Term Memory) — red recurrente para secuencias de texto.

El objetivo es evaluar el rendimiento de cada modelo y analizar cuál es más adecuado para este tipo de problema.

---

## Dataset
Se utiliza el dataset público **SMS Spam Collection**, disponible en:
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [GitHub](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv)

El dataset contiene **5,574 mensajes** etiquetados como:
- `ham` → mensaje legítimo
- `spam` → mensaje no deseado

---

## Preprocesamiento
1. Conversión a minúsculas.
2. Eliminación de caracteres no alfabéticos.
3. Tokenización con `nltk`.
4. Eliminación de *stopwords*.
5. Vectorización con **TF-IDF** y selección de características (para SVM y MLP).
6. Construcción de vocabulario y *padding* (para LSTM).

---

## Modelos Implementados
### SVM
- Kernel lineal.
- Selección de 1000 características con Chi-cuadrado.

### MLP
- Entrada: 1000 características TF-IDF.
- Varias capas densas con activación ReLU.
- Salida softmax.

### LSTM
- Embedding de dimensión 128.
- LSTM bidireccional con 64 unidades.
- Salida softmax.

---

## Métricas de Evaluación
- Accuracy.
- Precision, Recall, F1-score por clase.
- Matriz de confusión.

---

### Instalacion de dependencias
pip install -r requirements.txt

