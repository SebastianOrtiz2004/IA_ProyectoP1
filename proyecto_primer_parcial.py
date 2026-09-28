import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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

    # PASO 1: Carga y partición de datos
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

    conteo_clases = {}
    for estudiante in lista_estudiantes:
        clase_estudiante = estudiante.get('Target', '?')
        conteo_clases[clase_estudiante] = conteo_clases.get(clase_estudiante, 0) + 1

    for nombre_clase, cantidad in sorted(conteo_clases.items()):
        porcentaje = (cantidad / total_cargados) * 100
        print(f"    Clase {nombre_clase:>10}: {cantidad:>5} estudiantes ({porcentaje:.1f}%)")

    datos_entrenamiento, datos_prueba = partir_dataset(lista_estudiantes, porcentaje_entrenamiento=0.80)
    print(f"\n  Estudiantes de Entrenamiento (80%): {len(datos_entrenamiento)}")
    print(f"  Estudiantes de Prueba (20%):        {len(datos_prueba)}")

    pausar("\n  Presiona ENTER para iniciar la inducción de reglas con PRISM...")

    # PASO 2: Inducción de reglas PRISM
    reglas_difusas_inducidas, combinaciones_podadas, rep_difuso = inducir_reglas_difusas_con_prism(
        datos_entrenamiento, min_cobertura=15, imprimir=True
    )

    pausar("\n  Presiona ENTER para inducir la regla causal ganadora de PRISM...")

    reglas_prism_causales, salida_prism = ejecutar_prism(
        datos_entrenamiento, clase_objetivo='Dropout', semilla=42
    )
    print(salida_prism)

    pausar("\n  Presiona ENTER para iniciar la minería de reglas con Apriori...")

    # PASO 3: Minería de reglas Apriori
    reglas_apriori, _ = ejecutar_apriori(
        datos_entrenamiento, soporte_minimo=0.25, confianza_minima=0.70, max_reglas=12
    )

    pausar("\n  Presiona ENTER para iniciar el Algoritmo Genético...")

    # PASO 4: Algoritmo Genético
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

    # PASO 5: Diagnóstico Explicable (XAI)
    print("\n" + "=" * 70)
    print("  PASO 5: DIAGNÓSTICO EXPLICABLE INTEGRAL (XAI)")
    print("  (Cruce de Lógica Difusa + Reglas PRISM + Patrones Apriori)")
    print("=" * 70)

    estudiantes_ejemplo = {'Dropout': None, 'Graduate': None, 'Enrolled': None}
    for estudiante in datos_prueba:
        clase_target = estudiante.get('Target')
        if clase_target in estudiantes_ejemplo and estudiantes_ejemplo[clase_target] is None:
            estudiantes_ejemplo[clase_target] = estudiante

        todos_encontrados = True
        for valor in estudiantes_ejemplo.values():
            if valor is None:
                todos_encontrados = False
                break
        if todos_encontrados:
            break

    for nombre_clase, estudiante_ejemplo in estudiantes_ejemplo.items():
        if estudiante_ejemplo is not None:
            diagnosticar_estudiante_integral(
                estudiante_ejemplo,
                parametros_calibrados,
                reglas_prism=reglas_prism_causales,
                reglas_apriori=reglas_apriori,
                reglas_difusas=reglas_difusas_inducidas
            )

    pausar("\n  Presiona ENTER para la evaluación final en el conjunto de prueba...")

    # PASO 6: Evaluación final en datos de prueba
    exactitud_final = evaluar_en_prueba(
        datos_prueba,
        parametros_calibrados,
        reglas_prism=reglas_prism_causales,
        reglas_difusas=reglas_difusas_inducidas
    )

    parametros_iniciales = obtener_parametros_iniciales()
    aciertos_iniciales = 0

    for estudiante in datos_prueba:
        nota_admision = float(estudiante.get('Admission grade', 100))
        materias_aprobadas = float(estudiante.get('Curricular units 1st sem (approved)', 0))
        promedio_notas = float(estudiante.get('Curricular units 1st sem (grade)', 0))

        riesgo_inicial, _ = evaluar_motor_difuso(
            nota_admision, materias_aprobadas, promedio_notas, parametros_iniciales, numero_puntos=50, reglas=reglas_difusas_inducidas
        )

        activa_prism = False
        if reglas_prism_causales:
            for regla in reglas_prism_causales:
                cumple = True
                for atributo, valor in regla['condiciones']:
                    valor_atributo = estudiante.get(atributo)
                    if valor_atributo != valor and str(valor_atributo) != str(valor):
                        cumple = False
                        break
                if cumple:
                    activa_prism = True
                    break

        if riesgo_inicial >= 0.5 or activa_prism:
            prediccion = 'Dropout'
        else:
            prediccion = 'No Dropout'

        if estudiante.get('Target') == 'Dropout':
            condicion_real = 'Dropout'
        else:
            condicion_real = 'No Dropout'

        if prediccion == condicion_real:
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
