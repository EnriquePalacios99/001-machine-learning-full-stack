**Regresión Logística - Teoría**

# Regresión Logística - Teoría

La **Regresión Logística** es un algoritmo de clasificación, a pesar de su nombre que sugiere "regresión". Se utiliza para predecir la probabilidad de que una instancia de datos pertenezca a una clase particular. Es especialmente útil para problemas de clasificación binaria (dos clases), aunque puede extenderse a problemas multiclase.

---

## 1. Concepto Fundamental

A diferencia de la regresión lineal que predice un valor continuo, la regresión logística predice una **probabilidad**. Esta probabilidad se mapea a una de dos clases discretas (por ejemplo, "sí" o "no", "aprobado" o "reprobado", "0" o "1") utilizando una **función de activación**, típicamente la función sigmoide.

---

## 2. La Función Sigmoide (Función Logística)

El corazón de la regresión logística es la **función sigmoide**, también conocida como función logística. Esta función transforma cualquier valor real de entrada en un valor entre 0 y 1, que puede interpretarse como una probabilidad.

La ecuación de la función sigmoide es:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Donde:
* $\sigma(z)$ es la salida de la función sigmoide, un valor entre 0 y 1.
* $e$ es la base del logaritmo natural (aproximadamente 2.71828).
* $z$ es la entrada a la función, que en el contexto de la regresión logística es una combinación lineal de las características de entrada.

---

## 3. El Modelo de Regresión Logística

Similar a la regresión lineal, la regresión logística primero calcula una **combinación lineal ponderada** de las características de entrada. Sin embargo, en lugar de que esta combinación lineal sea la predicción final, se pasa a través de la función sigmoide.

Para una instancia de datos con $n$ características $x_1, x_2, \ldots, x_n$, la entrada $z$ a la función sigmoide se calcula como:

$$z = b_0 + b_1x_1 + b_2x_2 + \ldots + b_nx_n$$

O en forma vectorial:

$$z = \mathbf{w}^T \mathbf{x} + b$$

Donde:
* $\mathbf{w}$ es el **vector de pesos** (coeficientes) del modelo, es decir, $\mathbf{w} = [b_1, b_2, \ldots, b_n]^T$.
* $\mathbf{x}$ es el **vector de características** de entrada, es decir, $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$.
* $b$ es el **sesgo** (término de intercepción), equivalente a $b_0$.

Luego, la probabilidad predicha de que la instancia pertenezca a la clase positiva (clase 1) es:

$$\hat{y} = P(Y=1|\mathbf{x}; \mathbf{w}, b) = \sigma(z) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Donde $\hat{y}$ representa la probabilidad predicha.

La probabilidad de que la instancia pertenezca a la clase negativa (clase 0) es simplemente $1 - \hat{y}$.

---

## 4. Función de Costo (Pérdida)

Para entrenar un modelo de regresión logística, necesitamos una **función de costo** que mida qué tan bien el modelo está realizando sus predicciones. La función de costo para la regresión logística se deriva de la estimación de máxima verosimilitud y se conoce como la función de costo de **entropía cruzada binaria** (Binary Cross-Entropy Loss).

Para una sola instancia de entrenamiento $(x^{(i)}, y^{(i)})$, donde $y^{(i)}$ es la etiqueta verdadera (0 o 1) y $\hat{y}^{(i)}$ es la probabilidad predicha:

Si $y^{(i)} = 1$: El costo es $-\log(\hat{y}^{(i)})$
Si $y^{(i)} = 0$: El costo es $-\log(1 - \hat{y}^{(i)})$

Combinando ambas, la función de costo para una sola instancia es:

$$L(\hat{y}^{(i)}, y^{(i)}) = -y^{(i)}\log(\hat{y}^{(i)}) - (1 - y^{(i)})\log(1 - \hat{y}^{(i)})$$

La función de costo total ($J$) para un conjunto de entrenamiento con $m$ instancias es el promedio de los costos individuales:

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})]$$

El objetivo del entrenamiento es encontrar los valores de $\mathbf{w}$ y $b$ que minimizan esta función de costo.

---

## 5. Optimización: Descenso de Gradiente

Para minimizar la función de costo $J(\mathbf{w}, b)$, se utiliza comúnmente un algoritmo de optimización iterativo como el **Descenso de Gradiente**. El descenso de gradiente actualiza los pesos y el sesgo en la dirección opuesta al gradiente de la función de costo con respecto a los pesos y el sesgo.

Las reglas de actualización para los pesos $w_j$ y el sesgo $b$ en cada iteración son:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

Donde $\alpha$ es la **tasa de aprendizaje** (learning rate), que controla el tamaño de los pasos en cada actualización.

Las derivadas parciales de la función de costo con respecto a los pesos y el sesgo son:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

Estas derivadas se pueden calcular para todas las instancias en el conjunto de entrenamiento (Descenso de Gradiente por lotes) o para subconjuntos (Descenso de Gradiente Estocástico o Mini-Batch).

---

## 6. Límite de Decisión

Una vez que el modelo ha sido entrenado y se han determinado los pesos $\mathbf{w}$ y el sesgo $b$, podemos clasificar nuevas instancias. La clasificación se basa en la probabilidad predicha $\hat{y}$:

* Si $\hat{y} \geq 0.5$, la instancia se clasifica como Clase 1.
* Si $\hat{y} < 0.5$, la instancia se clasifica como Clase 0.

El umbral de 0.5 se puede ajustar según el problema y los requisitos de rendimiento (por ejemplo, para favorecer la precisión sobre el recall o viceversa).

El **límite de decisión** es la frontera que separa las dos clases. Se define por la ecuación $z = 0$, lo que implica:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

Esta ecuación representa una línea (en 2D), un plano (en 3D) o un hiperplano (en dimensiones superiores) que divide el espacio de características.

---

## 7. Supuestos y Limitaciones

* **Linealidad en el log-odds:** La relación entre las características de entrada y el log-odds (logaritmo de las probabilidades) es lineal. Es decir, el log-odds es una combinación lineal de las características.
* **Independencia de errores:** Las observaciones son independientes entre sí.
* **No asume normalidad:** A diferencia de la regresión lineal, la regresión logística no asume que los errores tienen una distribución normal.
* **Sensibilidad a valores atípicos:** Puede ser sensible a valores atípicos, especialmente en las características de entrada.
* **No captura relaciones no lineales:** Por sí misma, la regresión logística no puede capturar relaciones no lineales complejas entre las características y la variable objetivo, a menos que se realicen transformaciones de características.
* **Multicolinealidad:** Si las características de entrada están altamente correlacionadas (multicolinealidad), los coeficientes del modelo pueden volverse inestables y difíciles de interpretar.

---

## 8. Regularización (L1 y L2)

Para prevenir el sobreajuste (overfitting), especialmente cuando se tienen muchas características o un conjunto de datos pequeño, se pueden aplicar técnicas de **regularización**. Las dos formas más comunes son la regularización **L1 (Lasso)** y **L2 (Ridge)**.

### Regularización L2 (Ridge)

Añade un término de penalización a la función de costo que es proporcional al cuadrado de la magnitud de los pesos:

$$J_{L2}(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

Donde $\lambda$ es el parámetro de regularización, que controla la fuerza de la penalización. Un $\lambda$ más grande resultará en pesos más pequeños.

### Regularización L1 (Lasso)

Añade un término de penalización que es proporcional al valor absoluto de la magnitud de los pesos:

$$J_{L1}(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})] + \frac{\lambda}{m} \sum_{j=1}^{n} |w_j|$$

La regularización L1 tiene el efecto de conducir algunos pesos a cero, lo que puede ser útil para la selección de características.

---

## 9. Extensión a la Clasificación Multiclase: Softmax Regression

Cuando se tienen más de dos clases, la regresión logística se puede extender utilizando la función **softmax**. Esto se conoce como **Regresión Softmax** o **Regresión Logística Multinominal**. En lugar de producir una sola probabilidad, la regresión softmax produce una distribución de probabilidad sobre todas las clases.

La probabilidad de que una instancia $\mathbf{x}$ pertenezca a la clase $k$ se calcula como:

$$P(Y=k|\mathbf{x}; \mathbf{W}, \mathbf{b}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

Donde:
* $K$ es el número total de clases.
* $\mathbf{w}_k$ y $b_k$ son los pesos y el sesgo para la clase $k$.
* $\mathbf{W}$ es la matriz de pesos que contiene todos los vectores $\mathbf{w}_k$.
* $\mathbf{b}$ es el vector de sesgos que contiene todos los $b_k$.

La función de costo utilizada para la regresión softmax es la **entropía cruzada categórica**.

---

## 10. Métricas de Evaluación

Para evaluar el rendimiento de un modelo de regresión logística, se utilizan métricas específicas para clasificación:

* **Precisión (Accuracy):** Proporción de predicciones correctas sobre el total de predicciones.
* **Matriz de Confusión:** Tabla que resume los resultados de clasificación (verdaderos positivos, verdaderos negativos, falsos positivos, falsos negativos).
* **Precisión (Precision):** Proporción de verdaderos positivos sobre el total de positivos predichos.
* **Recall (Sensibilidad o Exhaustividad):** Proporción de verdaderos positivos sobre el total de positivos reales.
* **F1-Score:** Media armónica de precisión y recall, útil cuando hay un desequilibrio de clases.
* **Curva ROC y AUC (Area Under the Curve):** La curva ROC (Receiver Operating Characteristic) grafica la tasa de verdaderos positivos contra la tasa de falsos positivos en varios umbrales. El AUC mide el área bajo esta curva, proporcionando una métrica general del rendimiento del clasificador.