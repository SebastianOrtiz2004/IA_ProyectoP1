import sys
import random
import time
from difuso import evaluar_motor_difuso
from datos import obtener_muestra_estratificada

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


LONGITUD_CROMOSOMA = 19

LIMITES_GENES = [
    (60, 130),
    (90, 160),
    (70, 140),
    (100, 160),
    (120, 190),
    (110, 170),
    (140, 195),
    (0, 2),
    (1, 5),
    (0, 4),
    (2, 6),
    (4, 10),
    (3, 8),
    (5, 12),
    (3, 9),
    (6, 14),
    (5, 11),
    (9, 15),
    (12, 19),
]

BLOQUES_GENES = [
    (0, 1, 'trapecio'),
    (2, 4, 'triangulo'),
    (5, 6, 'trapecio'),
    (7, 8, 'trapecio'),
    (9, 11, 'triangulo'),
    (12, 13, 'trapecio'),
    (14, 15, 'trapecio'),
    (16, 18, 'triangulo'),
]


def reparar_geometria(cromosoma):
    cromosoma_reparado = list(cromosoma)

    for indice_inicio, indice_fin, tipo_figura in BLOQUES_GENES:
        puntos_del_bloque = []
        for indice_gen in range(indice_inicio, indice_fin + 1):
            limite_inferior, limite_superior = LIMITES_GENES[indice_gen]
            valor_actual = cromosoma_reparado[indice_gen]
            if valor_actual < limite_inferior:
                valor_actual = limite_inferior
            elif valor_actual > limite_superior:
                valor_actual = limite_superior
            puntos_del_bloque.append(valor_actual)

        puntos_del_bloque.sort()

        if tipo_figura == 'triangulo':
            separacion_minima = 0.5
            for i in range(1, len(puntos_del_bloque)):
                if puntos_del_bloque[i] <= puntos_del_bloque[i - 1]:
                    puntos_del_bloque[i] = puntos_del_bloque[i - 1] + separacion_minima

            limite_maximo_bloque = LIMITES_GENES[indice_fin][1]
            if puntos_del_bloque[-1] > limite_maximo_bloque:
                puntos_del_bloque[-1] = limite_maximo_bloque
                for i in range(len(puntos_del_bloque) - 2, -1, -1):
                    if puntos_del_bloque[i] >= puntos_del_bloque[i + 1]:
                        puntos_del_bloque[i] = puntos_del_bloque[i + 1] - separacion_minima

        posicion = 0
        for indice_gen in range(indice_inicio, indice_fin + 1):
            cromosoma_reparado[indice_gen] = puntos_del_bloque[posicion]
            posicion += 1

    if cromosoma_reparado[1] < cromosoma_reparado[2] + 4.0:
        cromosoma_reparado[1] = min(LIMITES_GENES[1][1], cromosoma_reparado[2] + 5.0)
        if cromosoma_reparado[1] < cromosoma_reparado[0] + 2.0:
            cromosoma_reparado[0] = max(LIMITES_GENES[0][0], cromosoma_reparado[1] - 4.0)

    if cromosoma_reparado[4] < cromosoma_reparado[5] + 4.0:
        cromosoma_reparado[4] = min(LIMITES_GENES[4][1], cromosoma_reparado[5] + 5.0)
        if cromosoma_reparado[4] < cromosoma_reparado[3] + 2.0:
            cromosoma_reparado[3] = max(LIMITES_GENES[3][0], cromosoma_reparado[4] - 3.0)

    if cromosoma_reparado[8] < cromosoma_reparado[9] + 1.0:
        cromosoma_reparado[8] = min(LIMITES_GENES[8][1], cromosoma_reparado[9] + 1.2)
        if cromosoma_reparado[8] < cromosoma_reparado[7]:
            cromosoma_reparado[7] = max(0.0, cromosoma_reparado[8] - 0.5)

    if cromosoma_reparado[11] < cromosoma_reparado[12] + 1.0:
        cromosoma_reparado[11] = min(LIMITES_GENES[11][1], cromosoma_reparado[12] + 1.2)
        if cromosoma_reparado[11] < cromosoma_reparado[10] + 0.5:
            cromosoma_reparado[10] = max(LIMITES_GENES[10][0], cromosoma_reparado[11] - 0.8)

    if cromosoma_reparado[17] < cromosoma_reparado[16] + 1.5:
        cromosoma_reparado[17] = min(LIMITES_GENES[17][1], cromosoma_reparado[16] + 1.8)
    if cromosoma_reparado[18] < cromosoma_reparado[17] + 1.5:
        cromosoma_reparado[18] = min(LIMITES_GENES[18][1], cromosoma_reparado[17] + 1.8)

    if cromosoma_reparado[15] < cromosoma_reparado[16] + 1.2:
        cromosoma_reparado[15] = min(LIMITES_GENES[15][1], cromosoma_reparado[16] + 1.5)
        if cromosoma_reparado[15] < cromosoma_reparado[14] + 0.8:
            cromosoma_reparado[14] = max(LIMITES_GENES[14][0], cromosoma_reparado[15] - 1.0)

    return cromosoma_reparado


def cromosoma_a_params(cromosoma):
    parametros = {
        'nota_adm_baja':   (0.0, 0.0, cromosoma[0], cromosoma[1]),
        'nota_adm_media':  (cromosoma[2], cromosoma[3], cromosoma[4]),
        'nota_adm_alta':   (cromosoma[5], cromosoma[6], 200.0, 200.0),

        'aprobadas_critica':  (0.0, 0.0, cromosoma[7], cromosoma[8]),
        'aprobadas_regular':  (cromosoma[9], cromosoma[10], cromosoma[11]),
        'aprobadas_completa': (cromosoma[12], cromosoma[13], 26.0, 26.0),

        'nota_sem_deficiente':    (0.0, 0.0, cromosoma[14], cromosoma[15]),
        'nota_sem_aceptable':     (cromosoma[16], cromosoma[17], cromosoma[18]),
        'nota_sem_sobresaliente': (cromosoma[17], min(cromosoma[18] + 1.0, 20.0), 20.0, 20.0),

        'riesgo_bajo':  (0.0, 0.0, 0.15, 0.35),
        'riesgo_medio': (0.25, 0.55, 0.75),
        'riesgo_alto':  (0.60, 0.80, 1.0, 1.0),
    }
    return parametros


def obtener_parametros_iniciales():
    return {
        'nota_adm_baja':          (0.0, 0.0, 95.0, 125.0),
        'nota_adm_media':         (95.0, 127.0, 155.0),
        'nota_adm_alta':          (135.0, 165.0, 200.0, 200.0),
        'aprobadas_critica':      (0.0, 0.0, 0.0, 2.0),
        'aprobadas_regular':      (1.0, 4.0, 7.0),
        'aprobadas_completa':     (5.0, 7.0, 26.0, 26.0),
        'nota_sem_deficiente':    (0.0, 0.0, 6.0, 10.0),
        'nota_sem_aceptable':     (7.0, 12.0, 16.0),
        'nota_sem_sobresaliente': (12.0, 16.0, 20.0, 20.0),
        'riesgo_bajo':            (0.0, 0.0, 0.15, 0.35),
        'riesgo_medio':           (0.25, 0.55, 0.75),
        'riesgo_alto':            (0.60, 0.80, 1.0, 1.0),
    }


def params_a_cromosoma(p):
    return [
        p['nota_adm_baja'][2], p['nota_adm_baja'][3],
        p['nota_adm_media'][0], p['nota_adm_media'][1], p['nota_adm_media'][2],
        p['nota_adm_alta'][0], p['nota_adm_alta'][1],
        p['aprobadas_critica'][2], p['aprobadas_critica'][3],
        p['aprobadas_regular'][0], p['aprobadas_regular'][1], p['aprobadas_regular'][2],
        p['aprobadas_completa'][0], p['aprobadas_completa'][1],
        p['nota_sem_deficiente'][2], p['nota_sem_deficiente'][3],
        p['nota_sem_aceptable'][0], p['nota_sem_aceptable'][1], p['nota_sem_aceptable'][2],
    ]


# FASE 2: Evaluación de aptitud
def calcular_fitness(cromosoma, lista_estudiantes, reglas=None):
    parametros = cromosoma_a_params(cromosoma)
    cantidad_aciertos = 0

    for estudiante in lista_estudiantes:
        try:
            nota = float(estudiante.get('Admission grade', 100))
            aprobadas = float(estudiante.get('Curricular units 1st sem (approved)', 0))
            promedio = float(estudiante.get('Curricular units 1st sem (grade)', 0))
            target_real = estudiante.get('Target', '')

            riesgo, _ = evaluar_motor_difuso(nota, aprobadas, promedio, parametros, numero_puntos=50, reglas=reglas)

            if riesgo >= 0.5:
                prediccion = 'Dropout'
            else:
                prediccion = 'No Dropout'

            if target_real == 'Dropout':
                real = 'Dropout'
            else:
                real = 'No Dropout'

            if prediccion == real:
                cantidad_aciertos += 1
        except Exception:
            pass

    if len(lista_estudiantes) > 0:
        return cantidad_aciertos / len(lista_estudiantes)
    else:
        return 0.0


# FASE 1: Población inicial
def crear_individuo(semilla_aleatoria):
    random.seed(semilla_aleatoria)
    cromosoma = []
    for limite_inferior, limite_superior in LIMITES_GENES:
        valor_aleatorio = random.uniform(limite_inferior, limite_superior)
        cromosoma.append(valor_aleatorio)
    return reparar_geometria(cromosoma)


# FASE 3: Selección por ruleta
def construir_ruleta_100_slots(lista_aptitudes):
    total_individuos = len(lista_aptitudes)
    ruleta = [None] * 100
    suma_total_fitness = sum(lista_aptitudes)

    if suma_total_fitness == 0:
        casillas_por_individuo = 100 // total_individuos
        for indice in range(total_individuos):
            for casilla in range(casillas_por_individuo):
                pos = indice * casillas_por_individuo + casilla
                if pos < 100:
                    ruleta[pos] = indice
        for pos in range(100):
            if ruleta[pos] is None:
                ruleta[pos] = 0
        return ruleta

    probabilidades = []
    for aptitud in lista_aptitudes:
        probabilidades.append(aptitud / suma_total_fitness)

    casillas_asignadas = []
    for prob in probabilidades:
        casillas_asignadas.append(round(prob * 100))

    diferencia = 100 - sum(casillas_asignadas)
    if diferencia > 0:
        indice_mejor = lista_aptitudes.index(max(lista_aptitudes))
        casillas_asignadas[indice_mejor] += diferencia
    elif diferencia < 0:
        for i in range(len(casillas_asignadas) - 1, -1, -1):
            while casillas_asignadas[i] > 0 and diferencia < 0:
                casillas_asignadas[i] -= 1
                diferencia += 1

    posicion = 0
    for indice_ind, cantidad_casillas in enumerate(casillas_asignadas):
        for _ in range(cantidad_casillas):
            if posicion < 100:
                ruleta[posicion] = indice_ind
                posicion += 1

    for pos in range(100):
        if ruleta[pos] is None:
            ruleta[pos] = 0

    return ruleta


def seleccionar_padres(ruleta):
    padre1 = ruleta[random.randint(0, 99)]
    padre2 = ruleta[random.randint(0, 99)]
    intentos = 0
    while padre2 == padre1 and intentos < 20:
        padre2 = ruleta[random.randint(0, 99)]
        intentos += 1
    return padre1, padre2


NOMBRES_GENES = [
    "X1: Nota Admisión Baja (c)",
    "X1: Nota Admisión Baja (d)",
    "X1: Nota Admisión Media (a)",
    "X1: Nota Admisión Media (b)",
    "X1: Nota Admisión Media (c)",
    "X1: Nota Admisión Alta (a)",
    "X1: Nota Admisión Alta (b)",
    "X2: Aprobadas Sem 1 Crítica (c)",
    "X2: Aprobadas Sem 1 Crítica (d)",
    "X2: Aprobadas Sem 1 Regular (a)",
    "X2: Aprobadas Sem 1 Regular (b)",
    "X2: Aprobadas Sem 1 Regular (c)",
    "X2: Aprobadas Sem 1 Completa (a)",
    "X2: Aprobadas Sem 1 Completa (b)",
    "X3: Promedio Sem 1 Deficiente (c)",
    "X3: Promedio Sem 1 Deficiente (d)",
    "X3: Promedio Sem 1 Aceptable (a)",
    "X3: Promedio Sem 1 Aceptable (b)",
    "X3: Promedio Sem 1 Aceptable (c)"
]


# FASE 4: Cruce
def cruzar_y_reparar_detalle(padre1, padre2, prob_cruce=0.85):
    longitud = len(padre1)
    if random.random() <= prob_cruce:
        punto_corte = random.randint(1, longitud - 1)
        hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
        hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
        ocurrio_cruce = True
    else:
        punto_corte = None
        hijo1 = list(padre1)
        hijo2 = list(padre2)
        ocurrio_cruce = False

    hijo1_reparado = reparar_geometria(hijo1)
    hijo2_reparado = reparar_geometria(hijo2)
    return hijo1_reparado, hijo2_reparado, ocurrio_cruce, punto_corte


def cruzar_y_reparar(padre1, padre2, prob_cruce=0.85):
    h1, h2, _, _ = cruzar_y_reparar_detalle(padre1, padre2, prob_cruce)
    return h1, h2


# FASE 5: Mutación
def mutar_y_reparar_detalle(cromosoma, prob_mutacion=0.05):
    cromosoma_mutado = list(cromosoma)
    mutaciones = []
    for indice_gen in range(len(cromosoma_mutado)):
        if random.random() < prob_mutacion:
            lim_inf, lim_sup = LIMITES_GENES[indice_gen]
            rango = lim_sup - lim_inf
            if rango > 0:
                variacion = random.uniform(-rango * 0.15, rango * 0.15)
                val_anterior = cromosoma_mutado[indice_gen]
                nuevo_valor = val_anterior + variacion
                if nuevo_valor < lim_inf:
                    nuevo_valor = lim_inf
                elif nuevo_valor > lim_sup:
                    nuevo_valor = lim_sup
                cromosoma_mutado[indice_gen] = nuevo_valor
                mutaciones.append((indice_gen, NOMBRES_GENES[indice_gen], val_anterior, nuevo_valor))

    return reparar_geometria(cromosoma_mutado), mutaciones


def mutar_y_reparar(cromosoma, prob_mutacion=0.05):
    crom_rep, _ = mutar_y_reparar_detalle(cromosoma, prob_mutacion)
    return crom_rep


def ejecutar_algoritmo_genetico(datos_entrenamiento, N_pob=30, numero_generaciones=60,
                                 sin_mejora_max=60, Pc=0.85, Pm=0.05,
                                 reglas_difusas=None, tam_muestra=400,
                                 callback_progreso=None, N_gen=None):
    if N_gen is not None:
        numero_generaciones = N_gen

    print("\n" + "=" * 70)
    print("  ALGORITMO GENÉTICO - CALIBRACIÓN DE FUNCIONES DE PERTENENCIA")
    print(f"  Cromosoma: {LONGITUD_CROMOSOMA} genes variables")
    print(f"  Población={N_pob}  Generaciones={numero_generaciones}  Pc={Pc}  Pm={Pm}")
    print("  (Reparación geométrica post-cruce y post-mutación: 0% individuos muertos)")
    print("=" * 70)

    muestra_evaluacion, cuotas = obtener_muestra_estratificada(
        datos_entrenamiento, tam_muestra=tam_muestra, semilla=42, retornar_cuotas=True
    )
    print(f"\n  Muestra estratificada para cálculo de fitness: {len(muestra_evaluacion)} estudiantes")
    detalles_estratos = " | ".join([f"{c}: {cnt} ({cnt/len(muestra_evaluacion)*100:.1f}%)" for c, cnt in sorted(cuotas.items())])
    print(f"  Distribución estratificada (Stratified Sampling): {detalles_estratos}")

    print(f"  Generando población inicial de {N_pob} cromosomas...")
    # FASE 1: Población inicial
    poblacion = []
    for i in range(N_pob):
        poblacion.append(crear_individuo(i * 7 + 13))

    print("  Calculando aptitud de la población inicial...")
    # FASE 2: Evaluación de aptitud
    aptitudes = []
    for ind in poblacion:
        aptitudes.append(calcular_fitness(ind, muestra_evaluacion, reglas=reglas_difusas))

    mejor_fitness = max(aptitudes)
    indice_mejor = aptitudes.index(mejor_fitness)
    mejor_cromosoma = list(poblacion[indice_mejor])

    print(f"  Fitness inicial del mejor individuo: {mejor_fitness:.4f} ({mejor_fitness * 100:.2f}%)")

    historial = [mejor_fitness]
    conteo_sin_mejora = 0
    registro_completo_generaciones = []

    print(f"\n  {'Generación':>11}  {'Mejor':>10}  {'Media':>10}  {'Muertos':>8}  {'Estado'}")
    print(f"  {'-' * 60}")

    for generacion in range(1, numero_generaciones + 1):
        # FASE 3: Selección por ruleta
        ruleta = construir_ruleta_100_slots(aptitudes)

        # FASE 6: Elitismo y Reemplazo
        nueva_poblacion = [list(mejor_cromosoma)]
        nuevas_aptitudes = [mejor_fitness]
        origenes_poblacion = [
            {
                'id': 1,
                'origen': "Élite (Mejor de la Generación Anterior)",
                'fitness': mejor_fitness,
                'mutado': False
            }
        ]

        cruces_de_la_generacion = []
        par_reproductivo = 0

        while len(nueva_poblacion) < N_pob:
            par_reproductivo += 1
            # FASE 3: Selección por ruleta
            idx1, idx2 = seleccionar_padres(ruleta)
            padre1 = poblacion[idx1]
            padre2 = poblacion[idx2]

            # FASE 4: Cruce
            h1, h2, ocurrio_cruce, punto_corte = cruzar_y_reparar_detalle(padre1, padre2, Pc)

            # FASE 5: Mutación
            h1, mutaciones_h1 = mutar_y_reparar_detalle(h1, Pm)
            h2, mutaciones_h2 = mutar_y_reparar_detalle(h2, Pm)

            # FASE 2: Evaluación de aptitud
            f1 = calcular_fitness(h1, muestra_evaluacion, reglas=reglas_difusas)
            id_h1 = len(nueva_poblacion) + 1
            nueva_poblacion.append(h1)
            nuevas_aptitudes.append(f1)
            origenes_poblacion.append({
                'id': id_h1,
                'origen': f"Hijo Cruce #{par_reproductivo:02d}" + (" (Mutado)" if mutaciones_h1 else ""),
                'fitness': f1,
                'mutado': bool(mutaciones_h1)
            })

            f2 = None
            id_h2 = None
            if len(nueva_poblacion) < N_pob:
                f2 = calcular_fitness(h2, muestra_evaluacion, reglas=reglas_difusas)
                id_h2 = len(nueva_poblacion) + 1
                nueva_poblacion.append(h2)
                nuevas_aptitudes.append(f2)
                origenes_poblacion.append({
                    'id': id_h2,
                    'origen': f"Hijo Cruce #{par_reproductivo:02d}" + (" (Mutado)" if mutaciones_h2 else ""),
                    'fitness': f2,
                    'mutado': bool(mutaciones_h2)
                })

            cruces_de_la_generacion.append({
                'par_num': par_reproductivo,
                'padre1_id': idx1 + 1,
                'padre1_fitness': aptitudes[idx1],
                'padre2_id': idx2 + 1,
                'padre2_fitness': aptitudes[idx2],
                'cruce_realizado': ocurrio_cruce,
                'punto_corte': punto_corte,
                'hijo1_id': id_h1,
                'hijo1_mutaciones': mutaciones_h1,
                'hijo1_fitness': f1,
                'hijo2_id': id_h2,
                'hijo2_mutaciones': mutaciones_h2,
                'hijo2_fitness': f2
            })

        poblacion = nueva_poblacion
        aptitudes = nuevas_aptitudes

        mejor_de_la_generacion = max(aptitudes)
        media_de_la_generacion = sum(aptitudes) / len(aptitudes)

        if mejor_de_la_generacion > mejor_fitness:
            mejor_fitness = mejor_de_la_generacion
            mejor_cromosoma = list(poblacion[aptitudes.index(mejor_de_la_generacion)])
            conteo_sin_mejora = 0
            estado = "*** MEJORA HISTÓRICA ***"
            hubo_mejora = True
        else:
            conteo_sin_mejora += 1
            estado = f"Estable (sin mejora: {conteo_sin_mejora})"
            hubo_mejora = False

        historial.append(mejor_fitness)

        datos_generacion = {
            'generacion': generacion,
            'total_generaciones': numero_generaciones,
            'elite_anterior_fitness': historial[-2] if len(historial) >= 2 else mejor_fitness,
            'cruces': cruces_de_la_generacion,
            'poblacion_resultante': origenes_poblacion,
            'mejor_fitness_generacion': mejor_de_la_generacion,
            'mejor_fitness_historico': mejor_fitness,
            'fitness_promedio_generacion': media_de_la_generacion,
            'mejor_fitness_gen': mejor_de_la_generacion,
            'fitness_promedio_gen': media_de_la_generacion,
            'hubo_mejora': hubo_mejora,
            'estado': estado,
            'conteo_sin_mejora': conteo_sin_mejora
        }
        registro_completo_generaciones.append(datos_generacion)

        if generacion % 5 == 0 or generacion == 1 or hubo_mejora:
            print(f"  {generacion:>11}  {mejor_fitness:>10.4f}  {media_de_la_generacion:>10.4f}  {0:>8}  {estado}")

        if callback_progreso:
            callback_progreso(generacion, numero_generaciones, mejor_fitness, media_de_la_generacion, datos_generacion)

        if conteo_sin_mejora >= sin_mejora_max:
            print(f"\n  Parada temprana en generación {generacion}: {conteo_sin_mejora} generaciones sin mejora.")
            break

    print(f"\n  Algoritmo Genético completado. Mejor Fitness: {mejor_fitness:.4f} ({mejor_fitness * 100:.2f}%)")

    parametros_calibrados = cromosoma_a_params(mejor_cromosoma)
    print(f"\n  Parámetros difusos calibrados por el AG:")
    for nombre_param, valores in parametros_calibrados.items():
        if 'riesgo' not in nombre_param:
            valores_redondeados = tuple(round(v, 2) for v in valores)
            print(f"    {nombre_param}: {valores_redondeados}")

    return parametros_calibrados, mejor_fitness, historial, registro_completo_generaciones


def formatear_detalle_generacion(datos_generacion):
    generacion = datos_generacion['generacion']
    total_generaciones = datos_generacion['total_generaciones']
    cruces = datos_generacion['cruces']
    pob = datos_generacion['poblacion_resultante']

    lineas = []
    lineas.append("=" * 100)
    lineas.append(f"                 GENERACIÓN {generacion:02d} / {total_generaciones:02d} - REGISTRO COMPLETO PASO A PASO")
    lineas.append("=" * 100)

    lineas.append("\n[1] PRESERVACIÓN POR ELITISMO:")
    lineas.append(f"    El mejor cromosoma de la generación anterior pasa directo a la posición #01 sin alteraciones.")
    lineas.append(f"    • Individuo #01 (Élite): Fitness = {datos_generacion['elite_anterior_fitness']*100:5.2f}%\n")

    total_cruces_efectivos = sum(1 for c in cruces if c['cruce_realizado'])
    total_mutaciones = sum(len(c['hijo1_mutaciones']) + (len(c['hijo2_mutaciones']) if c['hijo2_id'] else 0) for c in cruces)

    lineas.append(f"[2] CICLOS DE SELECCIÓN, CRUCE Y MUTACIÓN ({len(cruces)} Pares Reproductivos | {total_cruces_efectivos} Cruces | {total_mutaciones} Mutaciones en Genes):")
    lineas.append("-" * 100)

    for c in cruces:
        p_num = c['par_num']
        lineas.append(f"  >>> PAR REPRODUCTIVO #{p_num:02d}:")
        lineas.append(f"      • Selección de Padres por Ruleta (100 Casillas proporcionales a Fitness):")
        lineas.append(f"          - Padre 1: Individuo #{c['padre1_id']:02d}  (Fitness: {c['padre1_fitness']*100:5.2f}%)")
        lineas.append(f"          - Padre 2: Individuo #{c['padre2_id']:02d}  (Fitness: {c['padre2_fitness']*100:5.2f}%)")

        if c['cruce_realizado']:
            k = c['punto_corte']
            lineas.append(f"      • Operador de Cruce (Probabilidad Pc = 0.85):")
            lineas.append(f"          - Estado: CRUCE EJECUTADO con éxito en Punto de Corte k = {k} (de 19 genes del cromosoma)")
            lineas.append(f"          - Hijo 1: Hereda genes [0 .. {k-1}] de Padre 1 y [{k} .. 18] de Padre 2")
            lineas.append(f"          - Hijo 2: Hereda genes [0 .. {k-1}] de Padre 2 y [{k} .. 18] de Padre 1")
        else:
            lineas.append(f"      • Operador de Cruce (Probabilidad Pc = 0.85):")
            lineas.append(f"          - Estado: SIN CRUCE (Probabilidad aleatoria > Pc). Los hijos heredan copias íntegras.")

        lineas.append(f"      • Reparación Geométrica: Corrección de límites y orden de vértices aplicada (0% inviables).")

        lineas.append(f"      • Operador de Mutación (Probabilidad Pm = 0.05 por gen del cromosoma):")
        muts1 = c['hijo1_mutaciones']
        if muts1:
            lineas.append(f"          - Hijo 1 (Individuo #{c['hijo1_id']:02d}): SÍ MUTÓ ({len(muts1)} gen(es) del cromosoma alterado(s)):")
            for _, nom, v_ant, v_nue in muts1:
                lineas.append(f"              * {nom}: {v_ant:.2f} -> {v_nue:.2f}")
        else:
            lineas.append(f"          - Hijo 1 (Individuo #{c['hijo1_id']:02d}): NO MUTÓ (0 genes del cromosoma alterados)")

        if c['hijo2_id']:
            muts2 = c['hijo2_mutaciones']
            if muts2:
                lineas.append(f"          - Hijo 2 (Individuo #{c['hijo2_id']:02d}): SÍ MUTÓ ({len(muts2)} gen(es) del cromosoma alterado(s)):")
                for _, nom, v_ant, v_nue in muts2:
                    lineas.append(f"              * {nom}: {v_ant:.2f} -> {v_nue:.2f}")
            else:
                lineas.append(f"          - Hijo 2 (Individuo #{c['hijo2_id']:02d}): NO MUTÓ (0 genes del cromosoma alterados)")

        lineas.append(f"      • Evaluación de Aptitud Resultante:")
        lineas.append(f"          - Individuo #{c['hijo1_id']:02d} (Hijo 1): Fitness = {c['hijo1_fitness']*100:5.2f}%")
        if c['hijo2_id']:
            lineas.append(f"          - Individuo #{c['hijo2_id']:02d} (Hijo 2): Fitness = {c['hijo2_fitness']*100:5.2f}%")
        lineas.append("")

    lineas.append(f"[3] SELECCIÓN DE INDIVIDUOS QUE CONFORMAN LA NUEVA POBLACIÓN ({len(pob)} Individuos):")
    lineas.append("-" * 100)
    lineas.append(f"   #   | Origen Genético                               | Fitness (%) | Condición")
    lineas.append("-" * 100)
    for ind in pob:
        condicion = "Élite Preservado" if ind['id'] == 1 else ("Mutado" if ind['mutado'] else "Normal")
        lineas.append(f"  #{ind['id']:02d}  | {ind['origen']:<45} |     {ind['fitness']*100:5.2f}% | {condicion}")
    lineas.append("-" * 100)

    mejor_fit_g = datos_generacion.get('mejor_fitness_generacion', datos_generacion.get('mejor_fitness_gen', 0.0))
    prom_fit_g = datos_generacion.get('fitness_promedio_generacion', datos_generacion.get('fitness_promedio_gen', 0.0))
    lineas.append(f"\n[4] ESTADÍSTICAS Y BALANCE DE LA GENERACIÓN {generacion:02d}:")
    lineas.append(f"    • Mejor Fitness de esta Generación:   {mejor_fit_g*100:5.2f}%")
    lineas.append(f"    • Mejor Fitness Histórico Acumulado:  {datos_generacion['mejor_fitness_historico']*100:5.2f}%")
    lineas.append(f"    • Fitness Promedio de la Población:   {prom_fit_g*100:5.2f}%")
    lineas.append(f"    • Cruces Efectivos Ejecutados:        {total_cruces_efectivos} de {len(cruces)}")
    lineas.append(f"    • Mutaciones en Genes del Cromosoma:  {total_mutaciones} alteraciones")
    lineas.append(f"    • Estado Evolutivo:                   {datos_generacion['estado']}")
    lineas.append("=" * 100 + "\n")

    return "\n".join(lineas)


def formatear_resumen_general(registro_completo):
    lineas = []
    lineas.append("=" * 115)
    lineas.append("       RESUMEN EVOLUTIVO GENERAL DE TODAS LAS GENERACIONES DEL ALGORITMO GENÉTICO")
    lineas.append("=" * 115)
    lineas.append(f"{'Generación':<16} | {'Mejor Generación':>18} | {'Mejor Histórico':>16} | {'Media Generación':>18} | {'Cruces OK':>10} | {'Mutaciones':>11} | {'Estado'}")
    lineas.append("-" * 115)

    for g in registro_completo:
        num_generacion = g['generacion']
        m_generacion = g.get('mejor_fitness_generacion', g.get('mejor_fitness_gen', 0.0)) * 100
        m_historico = g['mejor_fitness_historico'] * 100
        media_generacion = g.get('fitness_promedio_generacion', g.get('fitness_promedio_gen', 0.0)) * 100
        cruces_ok = sum(1 for c in g['cruces'] if c['cruce_realizado'])
        tot_mut = sum(len(c['hijo1_mutaciones']) + (len(c['hijo2_mutaciones']) if c['hijo2_id'] else 0) for c in g['cruces'])
        estado = g['estado']
        etiqueta_gen = f"Generación {num_generacion:02d}"
        lineas.append(f"{etiqueta_gen:<16} | {m_generacion:>17.2f}% | {m_historico:>15.2f}% | {media_generacion:>17.2f}% | {cruces_ok:>5}/{len(g['cruces']):<3} | {tot_mut:>11} | {estado}")

    lineas.append("-" * 115)
    lineas.append(f" Total de generaciones ejecutadas y registradas: {len(registro_completo)}")
    if registro_completo:
        f_inicial = registro_completo[0].get('mejor_fitness_generacion', registro_completo[0].get('mejor_fitness_gen', 0.0)) * 100
        f_final = registro_completo[-1]['mejor_fitness_historico'] * 100
        lineas.append(f" Evolución de la Calidad: De {f_inicial:.2f}% (Generación 01) a {f_final:.2f}% (Generación {len(registro_completo):02d}) [Ganancia: +{f_final - f_inicial:.2f}%]")
    lineas.append("=" * 115 + "\n")
    return "\n".join(lineas)


def formatear_todas_las_generaciones_completo(registro_completo):
    partes = [formatear_resumen_general(registro_completo)]
    for g in registro_completo:
        partes.append(formatear_detalle_generacion(g))
    return "\n\n".join(partes)


METADATOS_GENES = [
    {
        'indice': 0,
        'nombre_gen': "Gen 00",
        'variable_corta': "X1: Nota Admisión",
        'conjunto': "Baja",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "c (Fin Meseta)",
        'rango': (60, 130),
        'clave_param': 'nota_adm_baja',
        'pos_tupla': 2,
        'significado': "Puntaje máximo de admisión con certeza total (μ=1.0) de Nota Baja."
    },
    {
        'indice': 1,
        'nombre_gen': "Gen 01",
        'variable_corta': "X1: Nota Admisión",
        'conjunto': "Baja",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "d (Fin Rampa)",
        'rango': (90, 160),
        'clave_param': 'nota_adm_baja',
        'pos_tupla': 3,
        'significado': "Puntaje donde la pertenencia al conjunto Nota Baja se extingue (μ=0.0)."
    },
    {
        'indice': 2,
        'nombre_gen': "Gen 02",
        'variable_corta': "X1: Nota Admisión",
        'conjunto': "Media",
        'figura': "Triángulo [a,b,c]",
        'vertice_corto': "a (Inicio Rampa)",
        'rango': (70, 140),
        'clave_param': 'nota_adm_media',
        'pos_tupla': 0,
        'significado': "Puntaje donde el estudiante empieza a pertenecer al conjunto Nota Media."
    },
    {
        'indice': 3,
        'nombre_gen': "Gen 03",
        'variable_corta': "X1: Nota Admisión",
        'conjunto': "Media",
        'figura': "Triángulo [a,b,c]",
        'vertice_corto': "b (Cúspide μ=1)",
        'rango': (100, 160),
        'clave_param': 'nota_adm_media',
        'pos_tupla': 1,
        'significado': "Puntaje de certeza absoluta (μ=1.0) representativo de Nota Media."
    },
    {
        'indice': 4,
        'nombre_gen': "Gen 04",
        'variable_corta': "X1: Nota Admisión",
        'conjunto': "Media",
        'figura': "Triángulo [a,b,c]",
        'vertice_corto': "c (Fin Rampa)",
        'rango': (120, 190),
        'clave_param': 'nota_adm_media',
        'pos_tupla': 2,
        'significado': "Puntaje donde culmina la pertenencia al conjunto de Nota Media."
    },
    {
        'indice': 5,
        'nombre_gen': "Gen 05",
        'variable_corta': "X1: Nota Admisión",
        'conjunto': "Alta",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "a (Inicio Rampa)",
        'rango': (110, 170),
        'clave_param': 'nota_adm_alta',
        'pos_tupla': 0,
        'significado': "Puntaje a partir del cual el estudiante empieza a pertenecer a Nota Alta."
    },
    {
        'indice': 6,
        'nombre_gen': "Gen 06",
        'variable_corta': "X1: Nota Admisión",
        'conjunto': "Alta",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "b (Inicio Meseta)",
        'rango': (140, 195),
        'clave_param': 'nota_adm_alta',
        'pos_tupla': 1,
        'significado': "Puntaje desde el cual el estudiante tiene certeza 100% de Nota Alta hasta 200 pts."
    },
    {
        'indice': 7,
        'nombre_gen': "Gen 07",
        'variable_corta': "X2: Aprobadas S1",
        'conjunto': "Crítica",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "c (Fin Meseta)",
        'rango': (0, 2),
        'clave_param': 'aprobadas_critica',
        'pos_tupla': 2,
        'significado': "Cantidad de materias aprobadas donde el rezago académico es 100% crítico."
    },
    {
        'indice': 8,
        'nombre_gen': "Gen 08",
        'variable_corta': "X2: Aprobadas S1",
        'conjunto': "Crítica",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "d (Fin Rampa)",
        'rango': (1, 5),
        'clave_param': 'aprobadas_critica',
        'pos_tupla': 3,
        'significado': "Límite superior donde el rezago crítico deja de tener efecto (μ=0.0)."
    },
    {
        'indice': 9,
        'nombre_gen': "Gen 09",
        'variable_corta': "X2: Aprobadas S1",
        'conjunto': "Regular",
        'figura': "Triángulo [a,b,c]",
        'vertice_corto': "a (Inicio Rampa)",
        'rango': (0, 4),
        'clave_param': 'aprobadas_regular',
        'pos_tupla': 0,
        'significado': "Materias aprobadas donde empieza la categoría de avance regular."
    },
    {
        'indice': 10,
        'nombre_gen': "Gen 10",
        'variable_corta': "X2: Aprobadas S1",
        'conjunto': "Regular",
        'figura': "Triángulo [a,b,c]",
        'vertice_corto': "b (Cúspide μ=1)",
        'rango': (2, 6),
        'clave_param': 'aprobadas_regular',
        'pos_tupla': 1,
        'significado': "Número de materias aprobadas típico con certeza máxima (100%) de avance regular."
    },
    {
        'indice': 11,
        'nombre_gen': "Gen 11",
        'variable_corta': "X2: Aprobadas S1",
        'conjunto': "Regular",
        'figura': "Triángulo [a,b,c]",
        'vertice_corto': "c (Fin Rampa)",
        'rango': (4, 10),
        'clave_param': 'aprobadas_regular',
        'pos_tupla': 2,
        'significado': "Materias aprobadas donde culmina el avance regular y transiciona a completo."
    },
    {
        'indice': 12,
        'nombre_gen': "Gen 12",
        'variable_corta': "X2: Aprobadas S1",
        'conjunto': "Completa",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "a (Inicio Rampa)",
        'rango': (3, 8),
        'clave_param': 'aprobadas_completa',
        'pos_tupla': 0,
        'significado': "Materias aprobadas desde las cuales se empieza a pertenecer a avance completo."
    },
    {
        'indice': 13,
        'nombre_gen': "Gen 13",
        'variable_corta': "X2: Aprobadas S1",
        'conjunto': "Completa",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "b (Inicio Meseta)",
        'rango': (5, 12),
        'clave_param': 'aprobadas_completa',
        'pos_tupla': 1,
        'significado': "Materias aprobadas donde el avance completo es pleno (100%) hasta 26 materias."
    },
    {
        'indice': 14,
        'nombre_gen': "Gen 14",
        'variable_corta': "X3: Promedio S1",
        'conjunto': "Deficiente",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "c (Fin Meseta)",
        'rango': (3, 9),
        'clave_param': 'nota_sem_deficiente',
        'pos_tupla': 2,
        'significado': "Promedio semestral límite bajo el cual el rendimiento es 100% deficiente."
    },
    {
        'indice': 15,
        'nombre_gen': "Gen 15",
        'variable_corta': "X3: Promedio S1",
        'conjunto': "Deficiente",
        'figura': "Trapecio [a,b,c,d]",
        'vertice_corto': "d (Fin Rampa)",
        'rango': (6, 14),
        'clave_param': 'nota_sem_deficiente',
        'pos_tupla': 3,
        'significado': "Promedio donde el rendimiento deja de considerarse deficiente (μ=0.0)."
    },
    {
        'indice': 16,
        'nombre_gen': "Gen 16",
        'variable_corta': "X3: Promedio S1",
        'conjunto': "Aceptable",
        'figura': "Triángulo [a,b,c]",
        'vertice_corto': "a (Inicio Rampa)",
        'rango': (5, 11),
        'clave_param': 'nota_sem_aceptable',
        'pos_tupla': 0,
        'significado': "Promedio semestral donde comienza el conjunto de calificaciones aceptables."
    },
    {
        'indice': 17,
        'nombre_gen': "Gen 17",
        'variable_corta': "X3: Promedio S1",
        'conjunto': "Aceptable",
        'figura': "Triángulo [a,b,c]",
        'vertice_corto': "b (Cúspide μ=1)",
        'rango': (9, 15),
        'clave_param': 'nota_sem_aceptable',
        'pos_tupla': 1,
        'significado': "Promedio representativo donde la certeza de nota aceptable es máxima (100%)."
    },
    {
        'indice': 18,
        'nombre_gen': "Gen 18",
        'variable_corta': "X3: Promedio S1",
        'conjunto': "Aceptable",
        'figura': "Triángulo [a,b,c]",
        'vertice_corto': "c (Fin Rampa)",
        'rango': (12, 19),
        'clave_param': 'nota_sem_aceptable',
        'pos_tupla': 2,
        'significado': "Promedio donde culmina la nota aceptable y transiciona a sobresaliente."
    }
]


def formatear_reporte_puntos_de_corte_calibrados(parametros_calibrados, mejor_fitness=None, parametros_iniciales=None):
    if parametros_iniciales is None:
        parametros_iniciales = obtener_parametros_iniciales()

    cromosoma_calibrado = params_a_cromosoma(parametros_calibrados)
    cromosoma_inicial = params_a_cromosoma(parametros_iniciales)

    lineas = []
    lineas.append("=" * 115)
    lineas.append("  DEMOSTRACIÓN DE LOS 19 PUNTOS DE CORTE RESULTANTES TRAS EL ALGORITMO GENÉTICO (MEJOR FIT)")
    lineas.append("  Mapeo Exhaustivo por Variable, Función de Pertenencia Difusa y Vértice Geométrico")
    lineas.append("=" * 115)

    if mejor_fitness is not None:
        lineas.append(f"\n  • Calidad del Mejor Individuo (Fitness / Exactitud): {mejor_fitness * 100:.2f}% de aciertos")
    lineas.append(f"  • Total de Puntos de Corte Optimizados:               19 genes del cromosoma")
    lineas.append(f"  • Estrategia de Calibración:                         Evolución en 60 Generaciones con Población de 30")
    lineas.append(f"  • Restricción Aplicada:                              Reparación Geométrica Garantizada (0% Individuos Inválidos)\n")

    lineas.append("[1] TABLA DETALLADA DE LOS 19 PUNTOS DE CORTE OPTIMIZADOS (GEN A GEN):")
    lineas.append("-" * 115)
    lineas.append(f"{'#Gen':<7} | {'Variable de Entrada':<22} | {'Conjunto':<10} | {'Vértice':<17} | {'Inicial':>8} | {'Rango':>10} | {'Calibrado':>9} | {'Variación (Δ)':>13}")
    lineas.append("-" * 115)

    for meta in METADATOS_GENES:
        idx = meta['indice']
        v_ini = cromosoma_inicial[idx]
        v_cal = cromosoma_calibrado[idx]
        delta = v_cal - v_ini
        signo = "+" if delta >= 0 else ""
        delta_str = f"{signo}{delta:.2f}"
        rango_str = f"[{meta['rango'][0]}, {meta['rango'][1]}]"

        lineas.append(
            f"{meta['nombre_gen']:<7} | {meta['variable_corta']:<22} | {meta['conjunto']:<10} | "
            f"{meta['vertice_corto']:<17} | {v_ini:>8.2f} | {rango_str:>10} | {v_cal:>9.2f} | {delta_str:>13}"
        )

    lineas.append("-" * 115)

    lineas.append("\n[2] RECONSTRUCCIÓN DE LAS 9 FUNCIONES DE PERTENENCIA CALIBRADAS POR VARIABLE:")
    lineas.append("=" * 115)

    p = parametros_calibrados
    p0 = parametros_iniciales

    lineas.append("\n• VARIABLE 1: NOTA DE ADMISIÓN (Dominio Global: 0.0 a 200.0 puntos):")
    lineas.append("  " + "-" * 105)
    baja = p['nota_adm_baja']
    lineas.append(f"  1. Conjunto 'Baja' (Trapecio [a, b, c, d] = [{baja[0]:.1f}, {baja[1]:.1f}, {baja[2]:.2f}, {baja[3]:.2f}]):")
    lineas.append(f"     - [0.0 a {baja[2]:.2f} pts] : Meseta de certeza total (μ = 1.0). El estudiante tiene certeza 100% de Nota Baja.")
    lineas.append(f"     - [{baja[2]:.2f} a {baja[3]:.2f} pts] : Rampa descendente donde μ decrece linealmente de 1.0 a 0.0.")
    lineas.append(f"     - Ajuste por AG: El fin de meseta cambió de {p0['nota_adm_baja'][2]:.1f} a {baja[2]:.2f} pts y el pie de rampa de {p0['nota_adm_baja'][3]:.1f} a {baja[3]:.2f} pts.")

    media = p['nota_adm_media']
    lineas.append(f"\n  2. Conjunto 'Media' (Triángulo [a, b, c] = [{media[0]:.2f}, {media[1]:.2f}, {media[2]:.2f}]):")
    lineas.append(f"     - Inicio rampa ascendente (μ = 0.0) : {media[0]:.2f} pts (antes: {p0['nota_adm_media'][0]:.1f} pts).")
    lineas.append(f"     - Cúspide de máxima certeza (μ = 1.0) : {media[1]:.2f} pts (antes: {p0['nota_adm_media'][1]:.1f} pts).")
    lineas.append(f"     - Fin rampa descendente   (μ = 0.0) : {media[2]:.2f} pts (antes: {p0['nota_adm_media'][2]:.1f} pts).")

    alta = p['nota_adm_alta']
    lineas.append(f"\n  3. Conjunto 'Alta' (Trapecio [a, b, c, d] = [{alta[0]:.2f}, {alta[1]:.2f}, {alta[2]:.1f}, {alta[3]:.1f}]):")
    lineas.append(f"     - [{alta[0]:.2f} a {alta[1]:.2f} pts] : Rampa ascendente donde μ sube de 0.0 a 1.0.")
    lineas.append(f"     - [{alta[1]:.2f} a 200.0 pts] : Meseta de certeza total (μ = 1.0). Estudiantes con alto desempeño garantizado.")

    lineas.append("\n• VARIABLE 2: MATERIAS APROBADAS 1er SEMESTRE (Dominio Global: 0 a 26 materias):")
    lineas.append("  " + "-" * 105)
    crit = p['aprobadas_critica']
    lineas.append(f"  1. Conjunto 'Crítica' (Trapecio [a, b, c, d] = [{crit[0]:.1f}, {crit[1]:.1f}, {crit[2]:.2f}, {crit[3]:.2f}]):")
    lineas.append(f"     - [0 a {crit[2]:.2f} mat] : Certeza absoluta de rezago crítico (μ = 1.0). Típico de desertores tempranos.")
    lineas.append(f"     - [{crit[2]:.2f} a {crit[3]:.2f} mat] : Transición gradual de salida del rezago crítico.")

    reg = p['aprobadas_regular']
    lineas.append(f"\n  2. Conjunto 'Regular' (Triángulo [a, b, c] = [{reg[0]:.2f}, {reg[1]:.2f}, {reg[2]:.2f}]):")
    lineas.append(f"     - Inicia en {reg[0]:.2f} mat, alcanza su cúspide más típica en {reg[1]:.2f} mat (μ = 1.0) y termina en {reg[2]:.2f} mat.")

    comp = p['aprobadas_completa']
    lineas.append(f"\n  3. Conjunto 'Completa' (Trapecio [a, b, c, d] = [{comp[0]:.2f}, {comp[1]:.2f}, {comp[2]:.1f}, {comp[3]:.1f}]):")
    lineas.append(f"     - A partir de {comp[1]:.2f} materias aprobadas, el estudiante es 100% clasificado en Avance Completo.")

    lineas.append("\n• VARIABLE 3: PROMEDIO DE NOTAS 1er SEMESTRE (Dominio Global: 0.0 a 20.0 puntos):")
    lineas.append("  " + "-" * 105)
    defn = p['nota_sem_deficiente']
    lineas.append(f"  1. Conjunto 'Deficiente' (Trapecio [a, b, c, d] = [{defn[0]:.1f}, {defn[1]:.1f}, {defn[2]:.2f}, {defn[3]:.2f}]):")
    lineas.append(f"     - De 0.0 a {defn[2]:.2f} pts: Certeza absoluta de promedio deficiente (μ = 1.0).")
    lineas.append(f"     - De {defn[2]:.2f} a {defn[3]:.2f} pts: Zona de transición difusa hacia notas aceptables.")

    acep = p['nota_sem_aceptable']
    lineas.append(f"\n  2. Conjunto 'Aceptable' (Triángulo [a, b, c] = [{acep[0]:.2f}, {acep[1]:.2f}, {acep[2]:.2f}]):")
    lineas.append(f"     - Inicia en {acep[0]:.2f} pts, alcanza máxima representatividad en {acep[1]:.2f} pts y culmina en {acep[2]:.2f} pts.")

    sob = p['nota_sem_sobresaliente']
    lineas.append(f"\n  3. Conjunto 'Sobresaliente' (Trapecio [a, b, c, d] = [{sob[0]:.2f}, {sob[1]:.2f}, {sob[2]:.1f}, {sob[3]:.1f}]):")
    lineas.append(f"     - A partir de {sob[1]:.2f} pts hasta 20.0 pts: Estudiante con promedio sobresaliente garantizado.")

    lineas.append("\n[3] JUSTIFICACIÓN ACADÉMICA Y EFECTO DEL ALGORITMO GENÉTICO:")
    lineas.append("-" * 115)
    lineas.append("  • El Algoritmo Genético descubrió que en el dataset UCI los desertores no se separan de manera rígida;")
    lineas.append("    al ajustar estos 19 puntos de corte, adaptó las fronteras difusas a la densidad real de estudiantes,")
    lineas.append("    evitando etiquetar erróneamente a alumnos que aprueban pocas materias con notas decentes.")
    lineas.append("=" * 115 + "\n")

    return "\n".join(lineas)
