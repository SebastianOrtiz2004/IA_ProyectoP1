# =============================================================================
#  PROYECTO PRIMER PARCIAL: SISTEMA HÍBRIDO DE INTELIGENCIA ARTIFICIAL
#  Predicción y Diagnóstico Explicable de Deserción Estudiantil
#  Dataset: UCI Predict Students' Dropout and Academic Success (ID: 697)
# =============================================================================
#  MÓDULOS DEL PROYECTO:
#   - datos.py        : Carga del dataset CSV y partición 80% entrenamiento / 20% prueba
#   - difuso.py       : Lógica Difusa Mamdani (Fuzzificación, Reglas y Centroide)
#   - prism.py        : Inducción de Reglas PRISM (Empíricas Difusas + Causales de Deserción)
#   - apriori.py      : Minería de Reglas de Asociación No Supervisada
#   - genetico.py     : Algoritmo Genético para calibrar las Funciones de Pertenencia
#   - diagnostico.py  : Diagnóstico Explicable (XAI) y Evaluación Final en Prueba
#
#  Librerías estándar: csv, math, random, time, os, sys (100% Python estándar, sin librerías externas)
# =============================================================================

import os
import sys
import time

# Configuración UTF-8 para consola de Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Importación de los módulos del proyecto
from datos import (
    cargar_dataset,
    partir_dataset
)

from difuso import (
    pertenencia_trapezoidal,
    pertenencia_triangular,
    fuzzificar_entrada,
    REGLAS_DIFUSAS,
    evaluar_motor_difuso
)

from prism import (
    ATRIBUTOS_DISCRETOS_PRISM,
    calcular_metricas_prism,
    ejecutar_prism,
    inducir_reglas_difusas_con_prism
)

from apriori import (
    extraer_items_estudiante_apriori,
    discretizar_para_apriori,
    ejecutar_apriori
)

from genetico import (
    LONGITUD_CROMOSOMA,
    LIMITES_GENES,
    BLOQUES_GENES,
    reparar_geometria,
    cromosoma_a_params,
    obtener_parametros_iniciales,
    calcular_fitness,
    ejecutar_algoritmo_genetico
)

from diagnostico import (
    diagnosticar_estudiante_integral,
    diagnosticar_estudiante,
    evaluar_en_prueba
)


def pausar(mensaje=""):
    """Pausa interactiva si se ejecuta en una terminal abierta, para poder leer paso a paso."""
    if sys.stdin.isatty():
        try:
            input(mensaje)
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print(mensaje.strip())


def main():
    print("\n" + "=" * 70)
    print("  PROYECTO PRIMER PARCIAL - INTELIGENCIA ARTIFICIAL")
    print("  Sistema Híbrido Explicable para Predicción de Deserción Estudiantil")
    print("  Dataset: UCI Machine Learning Repository (ID 697)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # PASO 1: Carga del dataset y división de datos
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PASO 1: CARGA DE DATOS Y PARTICIÓN")
    print("=" * 70)

    ruta_archivo = 'data.csv'
    try:
        lista_estudiantes = cargar_dataset(ruta_archivo)
    except FileNotFoundError:
        carpeta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_archivo = os.path.join(carpeta_actual, 'data.csv')
        lista_estudiantes = cargar_dataset(ruta_archivo)

    total_cargados = len(lista_estudiantes)
    print(f"\n  Total de estudiantes leídos del CSV: {total_cargados}")

    # Contar cuántos estudiantes hay por cada clase
    conteo_clases = {}
    for est in lista_estudiantes:
        clase = est.get('Target', '?')
        conteo_clases[clase] = conteo_clases.get(clase, 0) + 1

    for nombre_clase, cantidad in sorted(conteo_clases.items()):
        porcentaje = (cantidad / total_cargados) * 100
        print(f"    Clase {nombre_clase:>10}: {cantidad:>5} estudiantes ({porcentaje:.1f}%)")

    datos_entrenamiento, datos_prueba = partir_dataset(lista_estudiantes, porcentaje_entrenamiento=0.80)
    print(f"\n  Estudiantes de Entrenamiento (80%): {len(datos_entrenamiento)}")
    print(f"  Estudiantes de Prueba (20%):        {len(datos_prueba)}")

    pausar("\n  Presiona ENTER para iniciar la inducción de reglas con PRISM...")

    # -------------------------------------------------------------------------
    # PASO 2: Algoritmo PRISM
    # -------------------------------------------------------------------------
    # 2A: Inducir empíricamente las reglas cuantitativas para el motor difuso
    reglas_difusas_inducidas, combinaciones_podadas, rep_difuso = inducir_reglas_difusas_con_prism(
        datos_entrenamiento, min_cobertura=15, imprimir=True
    )

    pausar("\n  Presiona ENTER para inducir la regla causal ganadora de PRISM...")

    # 2B: Inducir regla causal paso a paso evaluando todas las 18 variables
    reglas_prism_causales, salida_prism = ejecutar_prism(
        datos_entrenamiento, clase_objetivo='Dropout', semilla=42
    )
    print(salida_prism)

    pausar("\n  Presiona ENTER para iniciar la minería de reglas con Apriori...")

    # -------------------------------------------------------------------------
    # PASO 3: Algoritmo Apriori (Minería de reglas de asociación no supervisada)
    # -------------------------------------------------------------------------
    reglas_apriori, _ = ejecutar_apriori(
        datos_entrenamiento, soporte_minimo=0.25, confianza_minima=0.70, max_reglas=12
    )

    pausar("\n  Presiona ENTER para iniciar el Algoritmo Genético...")

    # -------------------------------------------------------------------------
    # PASO 4: Algoritmo Genético (Calibra funciones de pertenencia difusas)
    # -------------------------------------------------------------------------
    print("\n  El Algoritmo Genético está optimizando los cortes de las funciones de pertenencia...")
    print("  (Esto toma alrededor de 1 minuto usando una muestra estratificada de 400 estudiantes)")

    tiempo_inicio = time.time()
    parametros_calibrados, mejor_fitness, historial, registro_generaciones = ejecutar_algoritmo_genetico(
        datos_entrenamiento,
        N_pob=30,
        numero_generaciones=60,
        sin_mejora_max=60,
        Pc=0.85,
        Pm=0.05,
        reglas_difusas=reglas_difusas_inducidas,
        tam_muestra=400
    )
    segundos_transcurridos = time.time() - tiempo_inicio
    print(f"\n  Tiempo total del Algoritmo Genético: {segundos_transcurridos:.1f} segundos")

    pausar("\n  Presiona ENTER para ver los casos de diagnóstico explicable (XAI)...")

    # -------------------------------------------------------------------------
    # PASO 5: Diagnóstico Explicable Integral (XAI)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PASO 5: DIAGNÓSTICO EXPLICABLE INTEGRAL (XAI)")
    print("  (Cruce de Lógica Difusa + Reglas PRISM + Patrones Apriori)")
    print("=" * 70)

    # Seleccionar 1 estudiante representativo de cada clase para demostrar la explicación
    estudiantes_ejemplo = {'Dropout': None, 'Graduate': None, 'Enrolled': None}
    for est in datos_prueba:
        clase_target = est.get('Target')
        if clase_target in estudiantes_ejemplo and estudiantes_ejemplo[clase_target] is None:
            estudiantes_ejemplo[clase_target] = est

        todos_encontrados = True
        for valor in estudiantes_ejemplo.values():
            if valor is None:
                todos_encontrados = False
                break
        if todos_encontrados:
            break

    for nombre_clase, est in estudiantes_ejemplo.items():
        if est is not None:
            diagnosticar_estudiante_integral(
                est,
                parametros_calibrados,
                reglas_prism=reglas_prism_causales,
                reglas_apriori=reglas_apriori,
                reglas_difusas=reglas_difusas_inducidas
            )

    pausar("\n  Presiona ENTER para la evaluación final en el conjunto de prueba...")

    # -------------------------------------------------------------------------
    # PASO 6: Evaluación final del Sistema Híbrido en los datos de prueba
    # -------------------------------------------------------------------------
    exactitud_final = evaluar_en_prueba(
        datos_prueba,
        parametros_calibrados,
        reglas_prism=reglas_prism_causales,
        reglas_difusas=reglas_difusas_inducidas
    )

    # Comparación de desempeño: Sin optimizar vs. Optimizado por el AG
    parametros_iniciales = obtener_parametros_iniciales()
    aciertos_iniciales = 0

    for est in datos_prueba:
        nota = float(est.get('Admission grade', 100))
        aprobadas = float(est.get('Curricular units 1st sem (approved)', 0))
        promedio = float(est.get('Curricular units 1st sem (grade)', 0))

        riesgo_ini, _ = evaluar_motor_difuso(
            nota, aprobadas, promedio, parametros_iniciales, numero_puntos=50, reglas=reglas_difusas_inducidas
        )

        activa_prism = False
        if reglas_prism_causales:
            for regla in reglas_prism_causales:
                cumple = True
                for atributo, valor in regla['condiciones']:
                    valor_est = est.get(atributo)
                    if valor_est != valor and str(valor_est) != str(valor):
                        cumple = False
                        break
                if cumple:
                    activa_prism = True
                    break

        if riesgo_ini >= 0.5 or activa_prism:
            pred = 'Dropout'
        else:
            pred = 'No Dropout'

        if est.get('Target') == 'Dropout':
            real = 'Dropout'
        else:
            real = 'No Dropout'

        if pred == real:
            aciertos_iniciales += 1

    exactitud_inicial = aciertos_iniciales / len(datos_prueba)

    print("\n  COMPARACIÓN DE CALIBRACIÓN DEL ALGORITMO GENÉTICO:")
    print(f"    - Con funciones 'a ojo' (Sin AG):    Accuracy Híbrido = {exactitud_inicial:.4f} ({exactitud_inicial * 100:.2f}%)")
    print(f"    - Con funciones optimizadas por AG:  Accuracy Híbrido = {exactitud_final:.4f} ({exactitud_final * 100:.2f}%)")
    print("    El Algoritmo Genético ajusta los 19 cortes de las funciones de pertenencia,")
    print("    adaptando la frontera difusa a la distribución real de los estudiantes.")

    print("\n" + "=" * 70)
    print("  EJECUCIÓN DEL PROYECTO COMPLETADA CON ÉXITO")
    print("=" * 70)


if __name__ == '__main__':
    main()
