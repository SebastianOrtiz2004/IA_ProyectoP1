# Sistema Híbrido de Inteligencia Artificial para la Predicción y Diagnóstico Explicable de la Deserción Estudiantil

Proyecto académico del Primer Parcial que implementa un sistema híbrido experto de Inteligencia Artificial para la detección temprana, diagnóstico causal y explicación transparente de la deserción universitaria en educación superior (Dataset UCI: 4,424 estudiantes).

Desarrollado en **Python puro (100% Librería Estándar)** sin dependencias externas pesadas (`pandas`, `scikit-learn`, `skfuzzy` no requeridas).

---

## 🏛️ Arquitectura Metodológica del Sistema Híbrido

El sistema integra de forma sinérgica cuatro pilares canónicos de la Inteligencia Artificial:

1. **Inducción Simbólica de Reglas Cuantitativas y Causales (PRISM)**
   - Inducción de 18 reglas de inferencia académica difusa con cobertura estadística controlada.
   - Inducción de reglas causales administrativas para la clase `Dropout` (deudas financieras, cuotas impagas, etc.).
   
2. **Minería de Patrones de Comportamiento y Reglas de Asociación (Apriori)**
   - Generación y poda estructurada por fases de ítem-sets frecuentes ($k = 1, 2, 3$).
   - Extracción de reglas de asociación con métricas de Cobertura, Soporte y Confianza.

3. **Optimización Evolutiva de Funciones de Pertenencia (Algoritmo Genético)**
   - Calibración de 19 genes continuos correspondientes a los puntos de corte y vértices geométricos ($a, b, c, d$).
   - Selección por Ruleta de 100 slots proporcionales al fitness.
   - Cruce en 1 punto ($P_c = 0.85$) con **Reparación Geométrica Garantizada** ($0\%$ individuos inviables o muertos).
   - Mutación uniforme por gen ($P_m = 0.05$) y elitismo estricto.
   - Inspección paso a paso de las 60 generaciones sin abreviaciones confusas.

4. **Motor de Inferencia Difusa Mamdani y Diagnóstico Explicable (XAI)**
   - Ejecución canónica de los 4 pasos: Fuzzificación, Inferencia con T-Norma Mínimo, Agregación con S-Norma Máximo y Defuzzificación por Centro de Gravedad (Centroide discreto).
   - Diagnóstico integral caso por caso cruzando el índice difuso con las alertas causales de PRISM y los patrones de Apriori.
   - Evaluación en matriz de confusión sobre 885 estudiantes de prueba independientes (20% nunca vistos).

---

## 📁 Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `datos.py` | Carga de `data.csv`, limpieza y partición estratificada 80% entrenamiento / 20% prueba. |
| `difuso.py` | Funciones de membresía trapezoidales/triangulares, inferencia Mamdani y desglose de los 4 pasos. |
| `prism.py` | Algoritmo PRISM de inducción modular para reglas difusas cuantitativas y reglas causales. |
| `apriori.py` | Algoritmo Apriori estructurado por fases (Cobertura mínima, ítem-sets frecuentes y reglas de asociación). |
| `genetico.py` | Algoritmo Genético (19 genes, ruleta 100 slots, cruce, mutación, reparación y reporte de cortes). |
| `diagnostico.py` | Diagnóstico explicable (XAI) y evaluación con matriz de confusión y métricas estadísticas. |
| `interfaz.py` | Interfaz Gráfica interactiva y didáctica desarrollada con Tkinter (5 pestañas). |
| `proyecto_primer_parcial.py` | Script principal para ejecución completa en consola (CLI). |
| `test_sistema_completo.py` | Suite de pruebas automatizadas que valida todos los módulos y la GUI de punta a punta. |
| `data.csv` | Dataset de deserción estudiantil (UCI Machine Learning Repository). |

---

## 🚀 Instrucciones de Ejecución

### 1. Ejecutar la Interfaz Gráfica (GUI)
```bash
python interfaz.py
```
*En la ventana principal puedes pulsar el botón superior `▶ EJECUTAR TODO EL SISTEMA` para calcular y actualizar todas las pestañas automáticamente.*

### 2. Ejecutar por Consola (CLI)
```bash
python proyecto_primer_parcial.py
```

### 3. Ejecutar la Suite de Pruebas Automatizada
```bash
python test_sistema_completo.py
```

---

## 🔬 Calibración de Puntos de Corte Difusos (19 Genes)

El Algoritmo Genético optimiza los 19 vértices geométricos de las funciones difusas sobre las 3 variables cuantitativas:
- **Variable 1: Nota de Admisión (0 - 200 pts):** Conjuntos `Baja` (Gen 00-01), `Media` (Gen 02-04), `Alta` (Gen 05-06).
- **Variable 2: Materias Aprobadas Semestre 1 (0 - 26 mat):** Conjuntos `Crítica` (Gen 07-08), `Regular` (Gen 09-11), `Completa` (Gen 12-13).
- **Variable 3: Promedio Semestre 1 (0 - 20 pts):** Conjuntos `Deficiente` (Gen 14-15), `Aceptable` (Gen 16-18), `Sobresaliente` (Gen 17-18 acoplados).
