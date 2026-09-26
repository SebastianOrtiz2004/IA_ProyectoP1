# =============================================================================
# ARCHIVO: difuso.py
# Descripción: Lógica Difusa Mamdani paso a paso
# Fases:
#   1. Fuzzificación: Convertir valores numéricos reales en grados de pertenencia (0.0 a 1.0)
#   2. Evaluación de Reglas: Operador Y (AND) usando el mínimo (min)
#   3. Agregación: Operador O (OR) usando el máximo (max)
#   4. Defuzzificación: Método del Centroide discreto para obtener el riesgo final
# =============================================================================

def pertenencia_trapezoidal(valor_x, a, b, c, d):
    """
    Función de pertenencia trapezoidal.
    - Entre b y c la certeza es total (1.0).
    - Menor que a o mayor que d está fuera (0.0).
    - Entre a y b sube en rampa.
    - Entre c y d baja en rampa.
    """
    if a == b == c == d:
        if valor_x == a:
            return 1.0
        else:
            return 0.0

    # Meseta superior (grado 1.0)
    if b <= valor_x <= c:
        return 1.0

    # Fuera del soporte (grado 0.0)
    if valor_x <= a or valor_x >= d:
        return 0.0

    # Rampa de subida
    if a < valor_x < b:
        if b != a:
            return (valor_x - a) / (b - a)
        else:
            return 1.0

    # Rampa de bajada
    if c < valor_x < d:
        if d != c:
            return (d - valor_x) / (d - c)
        else:
            return 1.0

    return 0.0


def pertenencia_triangular(valor_x, a, b, c):
    """
    Función de pertenencia triangular.
    - El pico máximo está exactamente en b (grado 1.0).
    - Menor que a o mayor que c es 0.0.
    """
    if b <= a or c <= b:
        return 0.0

    if valor_x <= a or valor_x >= c:
        return 0.0

    if a < valor_x <= b:
        return (valor_x - a) / (b - a)

    if b < valor_x < c:
        return (c - valor_x) / (c - b)

    return 0.0


def fuzzificar_entrada(nota_admision, materias_aprobadas, promedio_notas, parametros):
    """
    Convierte las 3 notas reales del estudiante a grados lingüísticos difusos.
    """
    grados_de_pertenencia = {}

    # Variable 1: Nota de admisión (escala 0 a 200)
    p_baja = parametros['nota_adm_baja']
    p_media = parametros['nota_adm_media']
    p_alta = parametros['nota_adm_alta']

    grados_de_pertenencia['X1_Baja'] = pertenencia_trapezoidal(nota_admision, p_baja[0], p_baja[1], p_baja[2], p_baja[3])
    grados_de_pertenencia['X1_Media'] = pertenencia_triangular(nota_admision, p_media[0], p_media[1], p_media[2])
    grados_de_pertenencia['X1_Alta'] = pertenencia_trapezoidal(nota_admision, p_alta[0], p_alta[1], p_alta[2], p_alta[3])

    # Variable 2: Materias aprobadas en primer semestre (0 a 26)
    c_critica = parametros['aprobadas_critica']
    c_regular = parametros['aprobadas_regular']
    c_completa = parametros['aprobadas_completa']

    grados_de_pertenencia['X2_Critica'] = pertenencia_trapezoidal(materias_aprobadas, c_critica[0], c_critica[1], c_critica[2], c_critica[3])
    grados_de_pertenencia['X2_Regular'] = pertenencia_triangular(materias_aprobadas, c_regular[0], c_regular[1], c_regular[2])
    grados_de_pertenencia['X2_Completa'] = pertenencia_trapezoidal(materias_aprobadas, c_completa[0], c_completa[1], c_completa[2], c_completa[3])

    # Variable 3: Promedio de notas en primer semestre (0 a 20)
    n_deficiente = parametros['nota_sem_deficiente']
    n_aceptable = parametros['nota_sem_aceptable']
    n_sobresaliente = parametros['nota_sem_sobresaliente']

    grados_de_pertenencia['X3_Deficiente'] = pertenencia_trapezoidal(promedio_notas, n_deficiente[0], n_deficiente[1], n_deficiente[2], n_deficiente[3])
    grados_de_pertenencia['X3_Aceptable'] = pertenencia_triangular(promedio_notas, n_aceptable[0], n_aceptable[1], n_aceptable[2])
    grados_de_pertenencia['X3_Sobresaliente'] = pertenencia_trapezoidal(promedio_notas, n_sobresaliente[0], n_sobresaliente[1], n_sobresaliente[2], n_sobresaliente[3])

    return grados_de_pertenencia


# Base de reglas difusas por defecto (referencia si no se pasan reglas de PRISM)
REGLAS_DIFUSAS = [
    # Riesgo Alto (deserción muy probable)
    (['X1_Baja', 'X2_Critica'],                       'Riesgo_Alto'),
    (['X2_Critica', 'X3_Deficiente'],                  'Riesgo_Alto'),
    (['X1_Baja', 'X3_Deficiente'],                     'Riesgo_Alto'),
    (['X1_Media', 'X2_Critica'],                       'Riesgo_Alto'),
    (['X1_Alta', 'X2_Critica', 'X3_Deficiente'],       'Riesgo_Alto'),
    # Riesgo Medio (alerta moderada)
    (['X1_Media', 'X2_Regular'],                       'Riesgo_Medio'),
    (['X2_Regular', 'X3_Aceptable'],                   'Riesgo_Medio'),
    (['X3_Deficiente', 'X2_Regular'],                  'Riesgo_Medio'),
    (['X1_Media', 'X3_Aceptable'],                     'Riesgo_Medio'),
    (['X1_Baja', 'X2_Regular'],                        'Riesgo_Medio'),
    # Riesgo Bajo (buen desempeño académico)
    (['X2_Completa', 'X3_Sobresaliente'],              'Riesgo_Bajo'),
    (['X1_Alta', 'X2_Completa'],                       'Riesgo_Bajo'),
    (['X1_Alta', 'X2_Regular', 'X3_Sobresaliente'],    'Riesgo_Bajo'),
    (['X3_Sobresaliente', 'X2_Regular'],               'Riesgo_Bajo'),
    (['X1_Media', 'X2_Completa'],                      'Riesgo_Bajo'),
    (['X1_Media', 'X2_Regular', 'X3_Sobresaliente'],   'Riesgo_Bajo'),
    (['X1_Alta', 'X3_Aceptable', 'X2_Completa'],       'Riesgo_Bajo'),
]


def evaluar_motor_difuso(nota_admision, materias_aprobadas, promedio_notas, parametros, numero_puntos=200, reglas=None):
    """
    Ejecuta el sistema difuso Mamdani completo para un estudiante:
      Paso 1: Fuzzificación
      Paso 2: Inferencia de reglas (mínimo)
      Paso 3: Agregación de consecuentes (máximo)
      Paso 4: Defuzzificación por Centroide para calcular el valor numérico de riesgo [0 a 1]
    """
    if reglas is None:
        reglas = REGLAS_DIFUSAS

    # FASE 1: Fuzzificación
    grados_pertenencia = fuzzificar_entrada(nota_admision, materias_aprobadas, promedio_notas, parametros)

    # FASE 2: Inferencia con reglas (operador AND = mínimo)
    fuerzas_activacion = {
        'Riesgo_Bajo': 0.0,
        'Riesgo_Medio': 0.0,
        'Riesgo_Alto': 0.0
    }
    detalle_reglas = []

    for condiciones_antecedente, consecuente in reglas:
        fuerza_regla = 1.0
        for condicion in condiciones_antecedente:
            grado_condicion = grados_pertenencia.get(condicion, 0.0)
            if grado_condicion < fuerza_regla:
                fuerza_regla = grado_condicion

        detalle_reglas.append((condiciones_antecedente, consecuente, fuerza_regla))

        # FASE 3: Agregación (operador OR = máximo)
        if fuerza_regla > fuerzas_activacion[consecuente]:
            fuerzas_activacion[consecuente] = fuerza_regla

    # FASE 4: Defuzzificación mediante Centroide discreto
    paso = 1.0 / numero_puntos
    suma_numerador_centroide = 0.0
    suma_denominador_centroide = 0.0

    parametros_riesgo_bajo = parametros['riesgo_bajo']
    parametros_riesgo_medio = parametros['riesgo_medio']
    parametros_riesgo_alto = parametros['riesgo_alto']

    for indice in range(numero_puntos + 1):
        punto_y = indice * paso

        pertenencia_bajo = pertenencia_trapezoidal(
            punto_y,
            parametros_riesgo_bajo[0], parametros_riesgo_bajo[1],
            parametros_riesgo_bajo[2], parametros_riesgo_bajo[3]
        )
        pertenencia_medio = pertenencia_triangular(
            punto_y,
            parametros_riesgo_medio[0], parametros_riesgo_medio[1],
            parametros_riesgo_medio[2]
        )
        pertenencia_alto = pertenencia_trapezoidal(
            punto_y,
            parametros_riesgo_alto[0], parametros_riesgo_alto[1],
            parametros_riesgo_alto[2], parametros_riesgo_alto[3]
        )

        corte_regla_bajo = min(fuerzas_activacion['Riesgo_Bajo'], pertenencia_bajo)
        corte_regla_medio = min(fuerzas_activacion['Riesgo_Medio'], pertenencia_medio)
        corte_regla_alto = min(fuerzas_activacion['Riesgo_Alto'], pertenencia_alto)

        pertenencia_agregada = max(corte_regla_bajo, corte_regla_medio, corte_regla_alto)

        suma_numerador_centroide += punto_y * pertenencia_agregada
        suma_denominador_centroide += pertenencia_agregada

    if suma_denominador_centroide > 0:
        riesgo_calculado = suma_numerador_centroide / suma_denominador_centroide
    else:
        riesgo_calculado = 0.5

    detalle = {
        'grados': grados_pertenencia,
        'detalle_reglas': detalle_reglas,
        'fuerzas': fuerzas_activacion,
        'suma_numerador': suma_numerador_centroide,
        'suma_denominador': suma_denominador_centroide,
        'riesgo': riesgo_calculado
    }
    return riesgo_calculado, detalle


def barra_grado(grado, ancho=12):
    """Genera una barra de caracteres para representar visualmente el grado de pertenencia."""
    llenos = int(round(grado * ancho))
    return "#" * llenos + "-" * (ancho - llenos)


def describir_los_4_pasos_difusos(nota_admision, materias_aprobadas, promedio_notas, parametros, detalle_motor=None, reglas=None):
    """
    Genera el desglose exhaustivo y pedagógico de los 4 PASOS CANÓNICOS de la Lógica Difusa Mamdani:
      Paso 1: Fuzzificación (Entrada continua -> Grados de pertenencia mu en [0, 1]).
      Paso 2: Evaluación de Reglas e Inferencia Mamdani (Operador T-norma = MÍNIMO).
      Paso 3: Agregación de Consecuentes (Operador S-norma = MÁXIMO).
      Paso 4: Defuzzificación por Centro de Gravedad (Centroide discreto).
    """
    if detalle_motor is None:
        riesgo_calc, detalle = evaluar_motor_difuso(
            nota_admision, materias_aprobadas, promedio_notas, parametros,
            numero_puntos=100, reglas=reglas
        )
    else:
        detalle = detalle_motor
        riesgo_calc = detalle['riesgo']

    grados = detalle['grados']
    detalle_reglas = detalle['detalle_reglas']
    fuerzas = detalle['fuerzas']
    num = detalle.get('suma_numerador', 0.0)
    den = detalle.get('suma_denominador', 0.0)

    # Identificar reglas activadas
    reglas_activas = [r for r in detalle_reglas if r[2] > 0.0001]
    reglas_activas.sort(key=lambda r: r[2], reverse=True)

    lineas = []
    lineas.append("=" * 95)
    lineas.append("        DESGLOSE METODOLÓGICO: LOS 4 PASOS FUNDAMENTALES DEL MOTOR DIFUSO MAMDANI")
    lineas.append("=" * 95)

    # PASO 1: FUZZIFICACIÓN
    lineas.append("\n[PASO 1: FUZZIFICACIÓN DE LAS VARIABLES NUMÉRICAS DE ENTRADA]")
    lineas.append("-" * 95)
    lineas.append("  Traduce las mediciones reales del estudiante a grados de membresía lingüística μ ∈ [0.0, 1.0]:\n")

    p_baja = parametros['nota_adm_baja']
    p_med = parametros['nota_adm_media']
    p_alt = parametros['nota_adm_alta']
    lineas.append(f"  • Variable 1: Nota de Admisión (X1 = {nota_admision:.1f} / 200 puntos)")
    lineas.append(f"      - Conjunto 'Baja'   (Trapecio [{p_baja[0]:.1f}, {p_baja[1]:.1f}, {p_baja[2]:.1f}, {p_baja[3]:.1f}]): μ = {grados['X1_Baja']:.3f} [{barra_grado(grados['X1_Baja'])}]")
    lineas.append(f"      - Conjunto 'Media'  (Triángulo [{p_med[0]:.1f}, {p_med[1]:.1f}, {p_med[2]:.1f}]):       μ = {grados['X1_Media']:.3f} [{barra_grado(grados['X1_Media'])}]")
    lineas.append(f"      - Conjunto 'Alta'   (Trapecio [{p_alt[0]:.1f}, {p_alt[1]:.1f}, {p_alt[2]:.1f}, {p_alt[3]:.1f}]): μ = {grados['X1_Alta']:.3f} [{barra_grado(grados['X1_Alta'])}]\n")

    c_crit = parametros['aprobadas_critica']
    c_reg = parametros['aprobadas_regular']
    c_comp = parametros['aprobadas_completa']
    lineas.append(f"  • Variable 2: Materias Aprobadas 1er Semestre (X2 = {materias_aprobadas:.0f} materias)")
    lineas.append(f"      - Conjunto 'Crítica'  (Trapecio [{c_crit[0]:.1f}, {c_crit[1]:.1f}, {c_crit[2]:.1f}, {c_crit[3]:.1f}]): μ = {grados['X2_Critica']:.3f} [{barra_grado(grados['X2_Critica'])}]")
    lineas.append(f"      - Conjunto 'Regular'  (Triángulo [{c_reg[0]:.1f}, {c_reg[1]:.1f}, {c_reg[2]:.1f}]):       μ = {grados['X2_Regular']:.3f} [{barra_grado(grados['X2_Regular'])}]")
    lineas.append(f"      - Conjunto 'Completa' (Trapecio [{c_comp[0]:.1f}, {c_comp[1]:.1f}, {c_comp[2]:.1f}, {c_comp[3]:.1f}]): μ = {grados['X2_Completa']:.3f} [{barra_grado(grados['X2_Completa'])}]\n")

    n_def = parametros['nota_sem_deficiente']
    n_acep = parametros['nota_sem_aceptable']
    n_sob = parametros['nota_sem_sobresaliente']
    lineas.append(f"  • Variable 3: Promedio de Notas 1er Semestre (X3 = {promedio_notas:.2f} / 20 puntos)")
    lineas.append(f"      - Conjunto 'Deficiente'    (Trapecio [{n_def[0]:.1f}, {n_def[1]:.1f}, {n_def[2]:.1f}, {n_def[3]:.1f}]): μ = {grados['X3_Deficiente']:.3f} [{barra_grado(grados['X3_Deficiente'])}]")
    lineas.append(f"      - Conjunto 'Aceptable'     (Triángulo [{n_acep[0]:.1f}, {n_acep[1]:.1f}, {n_acep[2]:.1f}]):       μ = {grados['X3_Aceptable']:.3f} [{barra_grado(grados['X3_Aceptable'])}]")
    lineas.append(f"      - Conjunto 'Sobresaliente' (Trapecio [{n_sob[0]:.1f}, {n_sob[1]:.1f}, {n_sob[2]:.1f}, {n_sob[3]:.1f}]): μ = {grados['X3_Sobresaliente']:.3f} [{barra_grado(grados['X3_Sobresaliente'])}]")

    # PASO 2: EVALUACIÓN DE REGLAS E INFERENCIA MAMDANI
    lineas.append("\n[PASO 2: EVALUACIÓN DE REGLAS E INFERENCIA MAMDANI (OPERADOR T-NORMA = MÍNIMO)]")
    lineas.append("-" * 95)
    lineas.append("  Para cada regla 'SI A AND B ENTONCES C', se evalúa la conjunción usando el operador MÍNIMO:")
    lineas.append("  Fuerza de activación α = min(μ_A, μ_B, ...)\n")
    lineas.append(f"  Total de reglas evaluadas en la base de conocimiento: {len(detalle_reglas)}")
    lineas.append(f"  Reglas activadas con fuerza α > 0.000: {len(reglas_activas)} regla(s):\n")

    if reglas_activas:
        for idx, (antecedente, consecuente, fuerza) in enumerate(reglas_activas, 1):
            desglose_conds = [f"μ({c})={grados.get(c, 0.0):.3f}" for c in antecedente]
            texto_ant = " AND ".join(antecedente)
            lineas.append(f"    #{idx:02d} | SI {texto_ant} ENTONCES {consecuente}")
            lineas.append(f"         Operación: min({', '.join(desglose_conds)}) => Fuerza α = {fuerza:.4f}")
    else:
        lineas.append("    (Ninguna regla cuantitativa superó el umbral de activación mínima; se activa regla neutral).")

    # PASO 3: AGREGACIÓN DE CONSECUENTES
    lineas.append("\n[PASO 3: AGREGACIÓN DE CONSECUENTES (OPERADOR S-NORMA = MÁXIMO)]")
    lineas.append("-" * 95)
    lineas.append("  Combina las conclusiones de todas las reglas activadas agrupándolas por su consecuente:")
    lineas.append("  Fuerza Agregada = max(α_1, α_2, ...) para cada nivel de riesgo:\n")
    lineas.append(f"  • Nivel 'Riesgo_Alto' : α_max = {fuerzas['Riesgo_Alto']:.4f}  [{barra_grado(fuerzas['Riesgo_Alto'])}]  (Mayor alerta de abandono)")
    lineas.append(f"  • Nivel 'Riesgo_Medio': α_max = {fuerzas['Riesgo_Medio']:.4f}  [{barra_grado(fuerzas['Riesgo_Medio'])}]  (Alerta moderada)")
    lineas.append(f"  • Nivel 'Riesgo_Bajo' : α_max = {fuerzas['Riesgo_Bajo']:.4f}  [{barra_grado(fuerzas['Riesgo_Bajo'])}]  (Estudiante académicamente seguro)\n")
    lineas.append("  Interpretación geométrica: Cada conjunto difuso de salida es truncado en su meseta superior")
    lineas.append("  al nivel α_max correspondiente y unido con el operador MÁXIMO, formando la curva difusa combinada.")

    # PASO 4: DEFUZZIFICACIÓN POR CENTROIDE
    lineas.append("\n[PASO 4: DEFUZZIFICACIÓN POR EL MÉTODO DEL CENTROIDE (CENTRO DE GRAVEDAD - COG)]")
    lineas.append("-" * 95)
    lineas.append("  Convierte el área difusa agregada en un único valor numérico puntual (crisp) de riesgo:")
    lineas.append("            Σ [ y_j · μ_agregada(y_j) ]     Momento de Área Total")
    lineas.append("      y* = ----------------------------- = -----------------------")
    lineas.append("                 Σ [ μ_agregada(y_j) ]          Área Total Agregada\n")

    if den > 0:
        lineas.append(f"  Cálculo sobre el dominio discreto de riesgo [0.0 a 1.0]:")
        lineas.append(f"    • Momento de Área Total (Numerador Σ y·μ)  = {num:.6f}")
        lineas.append(f"    • Área Total Agregada (Denominador Σ μ)   = {den:.6f}")
        lineas.append(f"    • Índice de Riesgo Cuantitativo Final y*  = {num:.6f} / {den:.6f} = {riesgo_calc:.4f} ({riesgo_calc*100:.2f}%)")
    else:
        lineas.append(f"    • Índice de Riesgo Cuantitativo por Defecto y* = {riesgo_calc:.4f} ({riesgo_calc*100:.2f}%)")

    veredicto = "ALTO RIESGO DE DESERCIÓN ('Dropout')" if riesgo_calc >= 0.5 else "BAJO RIESGO DE DESERCIÓN ('No Dropout')"
    lineas.append(f"\n  Criterio de Clasificación:")
    lineas.append(f"    - Si y* >= 0.50 (50.0%) -> Dropout (Estudiante en peligro de deserción)")
    lineas.append(f"    - Si y* <  0.50 (50.0%) -> No Dropout (Estudiante con probabilidad de continuidad)")
    lineas.append(f"\n  >>> VEREDICTO EXCLUSIVO DEL MOTOR DIFUSO: {veredicto} (Riesgo: {riesgo_calc*100:.2f}%)")
    lineas.append("=" * 95 + "\n")

    return "\n".join(lineas)

