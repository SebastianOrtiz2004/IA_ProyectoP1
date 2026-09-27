# =============================================================================
# ARCHIVO: prism.py
# Descripción: Algoritmo PRISM de Aprendizaje Inductivo de Reglas (Académico)
# Métodos:
#   1. PRISM Cuantitativo (Difuso): Evalúa las 27 combinaciones candidatas desde
#      el inicio y poda las reglas con soporte insuficiente o nulo.
#   2. PRISM Cualitativo (Causal): Construcción iterativa paso a paso con tabla
#      de candidatas y regla compuesta final de múltiples variables.
# =============================================================================

import sys
import random
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Diccionario de traducción y correspondencia Inglés -> Español para atributos
DICCIONARIO_TRADUCCION_ATRIBUTOS = {
    'Marital status': 'Estado_Civil',
    'Application mode': 'Modo_Admision',
    'Application order': 'Orden_Preferencia',
    'Course': 'Carrera',
    'Daytime/evening attendance': 'Horario_Estudio',
    'Previous qualification': 'Titulacion_Previa',
    'Nacionality': 'Nacionalidad',
    "Mother's qualification": 'Escolaridad_Madre',
    "Father's qualification": 'Escolaridad_Padre',
    "Mother's occupation": 'Ocupacion_Madre',
    "Father's occupation": 'Ocupacion_Padre',
    'Displaced': 'Estudiante_Foraneo',
    'Educational special needs': 'Necesidades_Especiales',
    'Debtor': 'Tiene_Deudas',
    'Tuition fees up to date': 'Matricula_Al_Dia',
    'Gender': 'Genero',
    'Scholarship holder': 'Tiene_Beca',
    'International': 'Estudiante_Extranjero',
}

# Lista ordenada de los 18 atributos discretos analizados
ATRIBUTOS_DISCRETOS_PRISM = list(DICCIONARIO_TRADUCCION_ATRIBUTOS.keys())


def traducir_atributo(nombre_ingles):
    """Devuelve el nombre en español del atributo."""
    return DICCIONARIO_TRADUCCION_ATRIBUTOS.get(nombre_ingles, nombre_ingles)


def calcular_metricas_prism(datos_completos, lista_condiciones, clase_objetivo='Dropout'):
    """
    Calcula métricas evaluando siempre sobre los N datos completos.
      |A|     = casos que cumplen el antecedente
      |A & B| = casos que cumplen el antecedente Y son de la clase objetivo
      Confianza = |A & B| / |A|
      Soporte   = |A & B| / N
      Lift      = Confianza / P(B)
    """
    total_estudiantes = len(datos_completos)
    estudiantes_con_antecedente = 0
    estudiantes_con_interseccion = 0
    total_estudiantes_clase = 0

    for estudiante in datos_completos:
        es_de_la_clase = (estudiante.get('Target') == clase_objetivo)
        if es_de_la_clase:
            total_estudiantes_clase += 1

        cumple_todas = True
        for atributo, valor_esperado in lista_condiciones:
            valor_real = estudiante.get(atributo)
            if valor_real != valor_esperado and str(valor_real) != str(valor_esperado):
                cumple_todas = False
                break

        if cumple_todas:
            estudiantes_con_antecedente += 1
            if es_de_la_clase:
                estudiantes_con_interseccion += 1

    confianza = estudiantes_con_interseccion / estudiantes_con_antecedente if estudiantes_con_antecedente > 0 else 0.0
    soporte = estudiantes_con_interseccion / total_estudiantes if total_estudiantes > 0 else 0.0
    probabilidad_clase = total_estudiantes_clase / total_estudiantes if total_estudiantes > 0 else 0.0
    lift = confianza / probabilidad_clase if (probabilidad_clase > 0 and confianza > 0) else 0.0

    return estudiantes_con_interseccion, estudiantes_con_antecedente, confianza, soporte, lift


def generar_encabezado_prism(ruta_archivo, N, conteo_B, prob_B):
    """Genera el encabezado formal académico del informe PRISM."""
    lineas = []
    lineas.append("=" * 95)
    lineas.append("           INFORME DE EJECUCIÓN: ALGORITMO PRISM ADAPTADO (ACADÉMICO)")
    lineas.append("=" * 95)
    lineas.append(f" Archivo de datos:             '{ruta_archivo}'")
    lineas.append(f" Total de Observaciones (N):   {N}")
    lineas.append(f" Columna Objetivo (Clase):     'Deserción' (Target)")
    lineas.append(f" Valor Positivo de la Clase:   'Sí' (Dropout)")
    lineas.append(f" Conteo Clase Positiva (|B|):  {conteo_B}")
    lineas.append(f" Proporción Base P(B):         {prob_B:.6f} ({prob_B * 100:.2f}%)")
    lineas.append("")
    lineas.append(f" Variables discretas evaluadas en cada iteración ({len(ATRIBUTOS_DISCRETOS_PRISM)} en total):")
    for i, attr_ing in enumerate(ATRIBUTOS_DISCRETOS_PRISM, 1):
        attr_esp = traducir_atributo(attr_ing)
        lineas.append(f"   {i:02d}. {attr_ing} ({attr_esp})")
    lineas.append("=" * 95)
    return "\n".join(lineas) + "\n"


def seleccionar_regla_ganadora_prism(candidatas, semilla=42):
    """
    Selecciona la regla ganadora entre las candidatas según el criterio académico:
      1° Máxima Confianza (redondeada a 8 decimales)
      2° Máxima Cobertura (|A & B|) / Soporte
      3° Máximo Lift (redondeado a 8 decimales)
    En caso de empate técnico perfecto en las 3 métricas, aplica desempate aleatorio con semilla.
    """
    def funcion_ordenamiento(candidata):
        return (round(candidata['confianza'], 8), candidata['cobertura'], round(candidata['lift'], 8))

    candidatas_ordenadas = sorted(candidatas, key=funcion_ordenamiento, reverse=True)
    mejor_puntaje = funcion_ordenamiento(candidatas_ordenadas[0])

    empates_top = [candidata for candidata in candidatas_ordenadas if funcion_ordenamiento(candidata) == mejor_puntaje]
    hubo_empate = len(empates_top) > 1

    if hubo_empate:
        random.seed(semilla)
        regla_ganadora = random.choice(empates_top)
    else:
        regla_ganadora = candidatas_ordenadas[0]

    return regla_ganadora, hubo_empate, len(empates_top), candidatas_ordenadas


def ejecutar_prism(datos_entrenamiento, clase_objetivo='Dropout', semilla=42, min_cobertura=10, **kwargs):
    """
    Ejecuta el Algoritmo PRISM en su formulación algorítmica canónica:
    Construye UNA SOLA REGLA COMPUESTA GANADORA a través de iteraciones sucesivas
    (1 variable añadida por iteración), evaluando todas las candidatas y seleccionando
    el mejor par atributo-valor con [*].
    Condición de parada del MIENTRAS:
      MIENTRAS (regla cubre algún ejemplo negativo AND Atributos ≠ ∅)
      En el momento en que la regla alcanza Confianza = 100.0% (0 ejemplos negativos cubiertos),
      la condición del MIENTRAS se evalúa como FALSO y el algoritmo concluye devolviendo la regla.
    """
    N = len(datos_entrenamiento)
    conteo_B = sum(1 for estudiante in datos_entrenamiento if estudiante.get('Target') == clase_objetivo)
    prob_B = conteo_B / N if N > 0 else 0.0

    salida_texto = []
    encabezado = generar_encabezado_prism('data.csv', N, conteo_B, prob_B)
    salida_texto.append(encabezado)

    condiciones_actuales = []
    columnas_restantes = list(ATRIBUTOS_DISCRETOS_PRISM)
    historial_iteraciones = []
    total_iteraciones_max = len(ATRIBUTOS_DISCRETOS_PRISM)

    for iteracion in range(1, total_iteraciones_max + 1):
        candidatas = []

        for atributo in columnas_restantes:
            valores_unicos = set()
            for estudiante in datos_entrenamiento:
                if atributo in estudiante:
                    valores_unicos.add(estudiante[atributo])

            for valor in valores_unicos:
                condiciones_prueba = condiciones_actuales + [(atributo, valor)]
                cobertura_interseccion, casos_antecedente, confianza_regla, soporte_regla, lift_regla = calcular_metricas_prism(
                    datos_entrenamiento, condiciones_prueba, clase_objetivo
                )

                # Umbral mínimo de casos para evitar reglas espurias de 1 caso
                umbral = min_cobertura if iteracion == 1 else max(5, min_cobertura // 2)
                if cobertura_interseccion < umbral:
                    continue

                partes_regla = [f"{traducir_atributo(nombre_attr)} = '{valor_attr}'" for nombre_attr, valor_attr in condiciones_prueba]
                texto_antecedente = " AND ".join(partes_regla)
                texto_completo = f"SI {texto_antecedente} ENTONCES Deserción = 'Sí'"

                candidatas.append({
                    'atributo': atributo,
                    'valor': valor,
                    'condiciones': condiciones_prueba,
                    'texto_regla': texto_completo,
                    'casos_A': casos_antecedente,
                    'cobertura': cobertura_interseccion,
                    'confianza': confianza_regla,
                    'soporte': soporte_regla,
                    'lift': lift_regla
                })

        if not candidatas:
            msg = f"\n  Iteración {iteracion}: No hay más variables candidatas que cumplan el umbral de cobertura.\n"
            salida_texto.append(msg)
            break

        regla_ganadora, hubo_empate, numero_empates, candidatas_ordenadas = seleccionar_regla_ganadora_prism(
            candidatas, semilla=semilla + iteracion
        )

        # Ancho dinámico para la tabla de candidatas
        max_longitud = max(len(candidata['texto_regla']) for candidata in candidatas_ordenadas)
        ancho_col = max(55, min(95, max_longitud + 2))
        ancho_tabla = ancho_col + 65

        iter_encabezado = f"\n>>> ITERACIÓN {iteracion} / {total_iteraciones_max} (Evaluando {len(candidatas)} Reglas Candidatas Probadas)\n"
        iter_encabezado += "-" * ancho_tabla + "\n"
        iter_encabezado += f"{'#':<3} | {'Regla Candidata Completa':<{ancho_col}} | {'|A|':<6} | {'|A&B|':<6} | {'Confianza':<10} | {'Soporte':<10} | {'Lift':<8} |\n"
        iter_encabezado += "-" * ancho_tabla + "\n"

        mostrar_top = min(10, len(candidatas_ordenadas))
        iter_cuerpo = ""
        for indice, candidata in enumerate(candidatas_ordenadas[:mostrar_top], 1):
            marca = " [*]" if candidata == regla_ganadora else "    "
            texto_candidata = (candidata['texto_regla'][:ancho_col-3] + '...') if len(candidata['texto_regla']) > ancho_col else candidata['texto_regla']
            iter_cuerpo += f"{indice:<3}{marca}| {texto_candidata:<{ancho_col}} | {candidata['casos_A']:<6} | {candidata['cobertura']:<6} | {candidata['confianza']:<10.6f} | {candidata['soporte']:<10.6f} | {candidata['lift']:<8.4f} |\n"

        iter_pie = "-" * ancho_tabla + "\n"
        if hubo_empate:
            iter_pie += f" NOTA: ¡Empate técnico detectado entre {numero_empates} reglas top! Se aplicó desempate aleatorio (semilla={semilla}).\n"

        anuncio = (
            f"\n  ===================================================================================================\n"
            f"  LA REGLA QUE PASA ES (Iteración {iteracion}):\n"
            f"  {regla_ganadora['texto_regla']}\n"
            f"  -> Confianza: {regla_ganadora['confianza']:.6f} ({regla_ganadora['confianza']*100:.2f}%) | "
            f"Cobertura (|A & B|): {regla_ganadora['cobertura']} | Soporte: {regla_ganadora['soporte']*100:.2f}% | Lift: {regla_ganadora['lift']:.4f}\n"
            f"  ===================================================================================================\n"
        )

        bloque_iteracion = iter_encabezado + iter_cuerpo + iter_pie + anuncio
        salida_texto.append(bloque_iteracion)

        # Incorporar la condición ganadora al antecedente
        condiciones_actuales.append((regla_ganadora['atributo'], regla_ganadora['valor']))
        columnas_restantes.remove(regla_ganadora['atributo'])
        historial_iteraciones.append(regla_ganadora)

        # CONDICIÓN DE PARADA OFICIAL DE PRISM:
        # MIENTRAS (regla cubre algún ejemplo negativo AND Atributos ≠ ∅) => FALSO
        if regla_ganadora['confianza'] >= 1.0:
            msg_parada = (
                f"\n  [!] CONDICIÓN DE PARADA OFICIAL DE PRISM:\n"
                f"      La regla ya no cubre ningún ejemplo negativo (Confianza = 100.0%).\n"
                f"      La condición del MIENTRAS se evalúa como FALSO. Concluye la inducción de la regla.\n"
            )
            salida_texto.append(msg_parada)
            break

    if not condiciones_actuales:
        return [], "".join(salida_texto)

    # RESUMEN FINAL DE LA REGLA COMPLETA INDUCIDA (CON LAS 18 VARIABLES)
    cobertura_final, casos_antecedente_final, confianza_final, soporte_final, lift_final = calcular_metricas_prism(
        datos_entrenamiento, condiciones_actuales, clase_objetivo
    )

    partes_finales = [f"{traducir_atributo(nombre_attr)} = '{valor_attr}'" for nombre_attr, valor_attr in condiciones_actuales]
    texto_final = f"SI {' AND '.join(partes_finales)} ENTONCES Deserción = 'Sí'"

    resumen_txt = "\n" + "=" * 95 + "\n"
    resumen_txt += "                 RESUMEN FINAL DE LA REGLA INDUCIDA (ACADÉMICO)\n"
    resumen_txt += "=" * 95 + "\n"
    resumen_txt += f" REGLA COMPLETA GANADORA ({len(condiciones_actuales)} Variables Compuestas en el Antecedente):\n"
    resumen_txt += f"   {texto_final}\n\n"
    resumen_txt += f" MÉTRICAS FINALES:\n"
    resumen_txt += f"   - Casos A (|A|):                   {casos_antecedente_final}\n"
    resumen_txt += f"   - Cobertura (|A & B|):              {cobertura_final}\n"
    resumen_txt += f"   - Confianza (Precisión Local):     {confianza_final:.6f} ({confianza_final * 100:.2f}%)\n"
    resumen_txt += f"   - Soporte (Proporción Global N):   {soporte_final:.6f} ({soporte_final * 100:.2f}%)\n"
    resumen_txt += f"   - Lift (Fuerza de Correlación):    {lift_final:.4f}\n"
    resumen_txt += "=" * 95 + "\n"

    salida_texto.append(resumen_txt)

    regla_dict = {
        'condiciones': condiciones_actuales,
        'texto_regla': texto_final,
        'historial': [(h['condiciones'][-1], h['confianza'], h['cobertura']) for h in historial_iteraciones],
        'confianza': confianza_final,
        'cobertura': cobertura_final,
        'soporte': soporte_final,
        'lift': lift_final
    }

    lista_reglas_finales = [regla_dict]
    # Si la regla compuesta acumuló múltiples condiciones, conservar también la regla
    # general de la primera iteración si posee alta confianza (>= 80%) y amplia cobertura
    for paso_iteracion in historial_iteraciones[:-1]:
        if paso_iteracion['confianza'] >= 0.80 and paso_iteracion['cobertura'] >= 15:
            conds_paso = paso_iteracion['condiciones']
            partes_paso = [f"{traducir_atributo(attr)} = '{val}'" for attr, val in conds_paso]
            regla_base = {
                'condiciones': conds_paso,
                'texto_regla': f"SI {' AND '.join(partes_paso)} ENTONCES Deserción = 'Sí'",
                'historial': [],
                'confianza': paso_iteracion['confianza'],
                'cobertura': paso_iteracion['cobertura'],
                'soporte': paso_iteracion['soporte'],
                'lift': paso_iteracion['lift']
            }
            lista_reglas_finales.append(regla_base)

    salida_completa = "".join(salida_texto)
    return lista_reglas_finales, salida_completa


def inducir_reglas_difusas_con_prism(datos_entrenamiento, min_cobertura=15, imprimir=False):
    """
    PRISM DIFUSO (CUANTITATIVO):
    Evalúa las 27 combinaciones teóricas completas (3 x 3 x 3) desde el inicio.
    Genera tabla de todas las candidatas probadas, indicando cuáles fueron
    APROBADAS (consecuente asignado) y cuáles fueron PODADAS con su motivo.
    """
    lineas_difuso = []
    encabezado = "\n" + "=" * 95 + "\n"
    encabezado += "  PASO 2A: PRISM DIFUSO - INDUCCIÓN DE REGLAS MAMDANI DESDE EL 100% DE DATOS\n"
    encabezado += "=" * 95 + "\n"
    encabezado += " Espacio combinatorio: 27 combinaciones teóricas (3x3x3)\n"
    encabezado += " Variables: X1_Nota_Admision (Baja/Media/Alta),\n"
    encabezado += "            X2_Materias_Aprobadas (Critica/Regular/Completa),\n"
    encabezado += "            X3_Promedio_Notas (Deficiente/Aceptable/Sobresaliente)\n"
    encabezado += "=" * 95 + "\n"
    lineas_difuso.append(encabezado)

    conteos_combinaciones = {}
    for estudiante in datos_entrenamiento:
        try:
            nota_admision = float(estudiante.get('Admission grade', 100))
            materias_aprobadas = float(estudiante.get('Curricular units 1st sem (approved)', 0))
            promedio_notas = float(estudiante.get('Curricular units 1st sem (grade)', 0))
            estado_estudiante = estudiante.get('Target', '')
        except (ValueError, TypeError):
            continue

        etiqueta_admision = 'X1_Baja' if nota_admision < 115 else ('X1_Alta' if nota_admision >= 145 else 'X1_Media')
        etiqueta_aprobadas = 'X2_Critica' if materias_aprobadas <= 2 else ('X2_Completa' if materias_aprobadas >= 6 else 'X2_Regular')
        etiqueta_promedio = 'X3_Deficiente' if promedio_notas < 10.0 else ('X3_Sobresaliente' if promedio_notas >= 14.0 else 'X3_Aceptable')

        clave_combinacion = (etiqueta_admision, etiqueta_aprobadas, etiqueta_promedio)
        if clave_combinacion not in conteos_combinaciones:
            conteos_combinaciones[clave_combinacion] = {'total_casos': 0, 'casos_desercion': 0}
        conteos_combinaciones[clave_combinacion]['total_casos'] += 1
        if estado_estudiante == 'Dropout':
            conteos_combinaciones[clave_combinacion]['casos_desercion'] += 1

    # Agregar combinaciones inexistentes (0 casos)
    etiquetas_admision = ['X1_Baja', 'X1_Media', 'X1_Alta']
    etiquetas_aprobadas = ['X2_Critica', 'X2_Regular', 'X2_Completa']
    etiquetas_promedio = ['X3_Deficiente', 'X3_Aceptable', 'X3_Sobresaliente']
    for etiqueta_1 in etiquetas_admision:
        for etiqueta_2 in etiquetas_aprobadas:
            for etiqueta_3 in etiquetas_promedio:
                combinacion_tupla = (etiqueta_1, etiqueta_2, etiqueta_3)
                if combinacion_tupla not in conteos_combinaciones:
                    conteos_combinaciones[combinacion_tupla] = {'total_casos': 0, 'casos_desercion': 0}

    # Ordenar las 27 combinaciones por frecuencia
    def criterio_total(elemento_combinacion):
        return elemento_combinacion[1]['total_casos']

    combinaciones_ordenadas = sorted(conteos_combinaciones.items(), key=criterio_total, reverse=True)

    reglas_inducidas = []
    combinaciones_podadas = []

    tabla = "\n>>> EVALUACIÓN DE LAS 27 COMBINACIONES CANDIDATAS PROBADAS DESDE EL INICIO:\n"
    tabla += "-" * 95 + "\n"
    tabla += f"{'#':<3} | {'Regla Candidata (Antecedente)':<42} | {'Casos':<6} | {'Dropout':<7} | {'Tasa Des':<9} | {'Consecuente':<13} | {'Estado':<10} |\n"
    tabla += "-" * 95 + "\n"

    for indice, (combinacion, estadisticas) in enumerate(combinaciones_ordenadas, 1):
        total_casos = estadisticas['total_casos']
        casos_desercion = estadisticas['casos_desercion']
        tasa_desercion = casos_desercion / total_casos if total_casos > 0 else 0.0
        texto_antecedente = " AND ".join(combinacion)

        if total_casos >= min_cobertura:
            if tasa_desercion >= 0.50:
                consecuente = 'Riesgo_Alto'
            elif tasa_desercion <= 0.25:
                consecuente = 'Riesgo_Bajo'
            else:
                consecuente = 'Riesgo_Medio'
            estado = "APROBADA"
            reglas_inducidas.append((list(combinacion), consecuente, total_casos, tasa_desercion))
        else:
            consecuente = "---"
            estado = "PODADA"
            motivo = f"Baja cobertura ({total_casos} < {min_cobertura})" if total_casos > 0 else "0 casos (inexistente)"
            combinaciones_podadas.append((list(combinacion), total_casos, tasa_desercion, motivo))

        marca = "[X]" if estado == "APROBADA" else "   "
        tabla += f"{indice:<3} {marca}| {texto_antecedente:<42} | {total_casos:<6} | {casos_desercion:<7} | {tasa_desercion*100:6.1f}%  | {consecuente:<13} | {estado:<10} |\n"

    tabla += "-" * 95 + "\n"
    resumen_corte = (
        f"\nRESUMEN DE INDUCCIÓN DIFUSA:\n"
        f"  - Total Reglas Candidatas Probadas: 27\n"
        f"  - Reglas Aprobadas (Cobertura >= {min_cobertura}): {len(reglas_inducidas)}\n"
        f"  - Reglas Podadas por Ruido o Inexistencia:     {len(combinaciones_podadas)}\n"
        f"=" * 95 + "\n"
    )

    lineas_difuso.append(tabla)
    lineas_difuso.append(resumen_corte)
    reporte_difuso_completo = "".join(lineas_difuso)
    if imprimir:
        print(reporte_difuso_completo)

    reglas_formato = [(ant, cons) for ant, cons, _, _ in reglas_inducidas]
    return reglas_formato, combinaciones_podadas, reporte_difuso_completo
