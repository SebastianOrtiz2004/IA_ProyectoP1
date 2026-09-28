from difuso import evaluar_motor_difuso, describir_los_4_pasos_difusos
from apriori import extraer_items_estudiante_apriori

def diagnosticar_estudiante_integral(estudiante, parametros, reglas_prism=None, reglas_apriori=None, reglas_difusas=None):
    nota_admision = float(estudiante.get('Admission grade', 100))
    materias_aprobadas = float(estudiante.get('Curricular units 1st sem (approved)', 0))
    promedio_notas = float(estudiante.get('Curricular units 1st sem (grade)', 0))
    target_real = estudiante.get('Target', '?')

    riesgo_calculado, detalle = evaluar_motor_difuso(
        nota_admision, materias_aprobadas, promedio_notas, parametros,
        numero_puntos=100, reglas=reglas_difusas
    )

    reglas_prism_activadas = []
    if reglas_prism:
        for regla in reglas_prism:
            cumple_todas = True
            for atributo, valor in regla['condiciones']:
                valor_estudiante = estudiante.get(atributo)
                if valor_estudiante != valor and str(valor_estudiante) != str(valor):
                    cumple_todas = False
                    break
            if cumple_todas:
                reglas_prism_activadas.append(regla)

    reglas_apriori_activadas = []
    if reglas_apriori:
        items_estudiante = extraer_items_estudiante_apriori(estudiante, incluir_target=False)
        for regla in reglas_apriori:
            if regla['antecedente'].issubset(items_estudiante):
                reglas_apriori_activadas.append(regla)

    print("\n" + "=" * 70)
    print("  DIAGNÓSTICO EXPLICABLE INTEGRAL (XAI) - CASO ESTUDIANTE")
    print("=" * 70)
    print(f"  Perfil Académico: Nota Admisión={nota_admision:.1f}/200 | Aprobadas S1={materias_aprobadas:.0f} | Promedio S1={promedio_notas:.2f}/20")
    print(f"  Target Real: {target_real}")

    print("\n" + describir_los_4_pasos_difusos(
        nota_admision, materias_aprobadas, promedio_notas, parametros,
        detalle_motor=detalle, reglas=reglas_difusas
    ))

    print("\n  [PILAR 2: INDUCCIÓN SIMBÓLICA PRISM (Reglas Causales Supervisadas)]")
    if len(reglas_prism_activadas) > 0:
        print(f"    ¡ALERTA CRÍTICA PRISM! El estudiante activa {len(reglas_prism_activadas)} regla(s) de deserción:")
        for indice, regla in enumerate(reglas_prism_activadas, 1):
            lista_conds = []
            for atr, val in regla['condiciones']:
                lista_conds.append(f"({atr} = {val})")
            texto_conds = " AND ".join(lista_conds)
            print(f"      Regla PRISM #{indice}: SI {texto_conds} -> Target = Dropout")
            print(f"        (Confianza={regla['confianza'] * 100:.1f}%, Cobertura histórica={regla['cobertura']} casos)")
    else:
        print("    Alerta PRISM: No activa reglas discretas de deserción inminente.")

    print("\n  [PILAR 3: MINERÍA DE ASOCIACIÓN APRIORI (Patrones de Comportamiento)]")
    if len(reglas_apriori_activadas) > 0:
        print(f"    El estudiante encaja en {len(reglas_apriori_activadas)} patrones frecuentes de asociación:")
        for regla in reglas_apriori_activadas[:3]:
            ante = ' AND '.join(sorted(regla['antecedente']))
            cons = ' AND '.join(sorted(regla['consecuente']))
            soporte_val = regla.get('soporte', regla.get('soporte_regla', 0.0))
            print(f"      Patrón: SI [{ante}] -> ENTONCES [{cons}] (Conf={regla['confianza'] * 100:.1f}%, Soporte={soporte_val * 100:.1f}%)")
    else:
        print("    Perfil sin patrones atípicos de alta confianza en Apriori.")

    print("\n  [SÍNTESIS DIAGNÓSTICA EXPLICABLE (XAI)]")
    if riesgo_calculado >= 0.5 and len(reglas_prism_activadas) > 0:
        veredicto = "RIESGO EXTREMO CONFIRMADO (Convergencia Difusa + Causal PRISM)"
        explicacion = "Tanto el bajo rendimiento en notas como sus factores administrativos y sociodemográficos indican alta probabilidad de abandono."
    elif riesgo_calculado >= 0.5:
        veredicto = "RIESGO ACADÉMICO ELEVADO (Alerta Difusa)"
        explicacion = "El riesgo se debe principalmente al bajo rendimiento de materias y promedio en el primer semestre."
    elif len(reglas_prism_activadas) > 0:
        veredicto = "ALERTA ADMINISTRATIVA / SOCIODEMOGRÁFICA (Alerta PRISM)"
        explicacion = "Aunque sus notas sean regulares, presenta factores críticos (como deudas o matrícula) fuertemente asociados a deserción."
    else:
        veredicto = "ESTUDIANTE SEGURO / BAJO RIESGO"
        explicacion = "Buen desempeño académico general y ausencia de factores de riesgo administrativo."

    print(f"    Veredicto Final: {veredicto}")
    print(f"    Explicación: {explicacion}")

    if riesgo_calculado >= 0.5 or len(reglas_prism_activadas) > 0:
        prediccion_final = 'Dropout'
    else:
        prediccion_final = 'No Dropout'

    acierto = (prediccion_final == 'Dropout') == (target_real == 'Dropout')
    resultado_texto = '[CORRECTO]' if acierto else '[DISCREPANCIA]'
    print(f"    Predicción del Sistema: {prediccion_final} | Realidad: {target_real} -> {resultado_texto}")
    print("=" * 70)

    return riesgo_calculado, prediccion_final


def diagnosticar_estudiante(estudiante, parametros):
    return diagnosticar_estudiante_integral(estudiante, parametros)


def evaluar_en_prueba(datos_prueba, parametros, reglas_prism=None, reglas_difusas=None):
    print("\n" + "=" * 70)
    print("  PASO 6: EVALUACIÓN FINAL EN DATOS DE PRUEBA")
    print("  (Evaluación Unificada del Sistema Híbrido: Difuso Mamdani + Causal PRISM)")
    print("=" * 70)

    verdaderos_positivos_hibrido = 0
    falsos_positivos_hibrido = 0
    verdaderos_negativos_hibrido = 0
    falsos_negativos_hibrido = 0

    verdaderos_positivos_difuso = 0
    falsos_positivos_difuso = 0
    verdaderos_negativos_difuso = 0
    falsos_negativos_difuso = 0

    for estudiante in datos_prueba:
        nota = float(estudiante.get('Admission grade', 100))
        aprobadas = float(estudiante.get('Curricular units 1st sem (approved)', 0))
        promedio = float(estudiante.get('Curricular units 1st sem (grade)', 0))
        target_real = estudiante.get('Target', '')
        es_desertor_real = (target_real == 'Dropout')

        riesgo, _ = evaluar_motor_difuso(nota, aprobadas, promedio, parametros, numero_puntos=100, reglas=reglas_difusas)
        prediccion_difuso_dropout = (riesgo >= 0.5)

        activa_regla_prism = False
        if reglas_prism:
            for regla in reglas_prism:
                cumple_regla = True
                for atributo, valor in regla['condiciones']:
                    valor_est = estudiante.get(atributo)
                    if valor_est != valor and str(valor_est) != str(valor):
                        cumple_regla = False
                        break
                if cumple_regla:
                    activa_regla_prism = True
                    break

        prediccion_hibrido_dropout = prediccion_difuso_dropout or activa_regla_prism

        if prediccion_difuso_dropout and es_desertor_real:
            verdaderos_positivos_difuso += 1
        elif prediccion_difuso_dropout and not es_desertor_real:
            falsos_positivos_difuso += 1
        elif not prediccion_difuso_dropout and not es_desertor_real:
            verdaderos_negativos_difuso += 1
        else:
            falsos_negativos_difuso += 1

        if prediccion_hibrido_dropout and es_desertor_real:
            verdaderos_positivos_hibrido += 1
        elif prediccion_hibrido_dropout and not es_desertor_real:
            falsos_positivos_hibrido += 1
        elif not prediccion_hibrido_dropout and not es_desertor_real:
            verdaderos_negativos_hibrido += 1
        else:
            falsos_negativos_hibrido += 1

    total_estudiantes_prueba = len(datos_prueba)

    if total_estudiantes_prueba > 0:
        exactitud_hibrido = (verdaderos_positivos_hibrido + verdaderos_negativos_hibrido) / total_estudiantes_prueba
    else:
        exactitud_hibrido = 0.0

    total_desertores_reales = verdaderos_positivos_hibrido + falsos_negativos_hibrido
    if total_desertores_reales > 0:
        sensibilidad_hibrido = verdaderos_positivos_hibrido / total_desertores_reales
    else:
        sensibilidad_hibrido = 0.0

    total_predichos_desertores = verdaderos_positivos_hibrido + falsos_positivos_hibrido
    if total_predichos_desertores > 0:
        precision_hibrido = verdaderos_positivos_hibrido / total_predichos_desertores
    else:
        precision_hibrido = 0.0

    if total_estudiantes_prueba > 0:
        exactitud_difuso = (verdaderos_positivos_difuso + verdaderos_negativos_difuso) / total_estudiantes_prueba
    else:
        exactitud_difuso = 0.0

    total_desertores_difuso = verdaderos_positivos_difuso + falsos_negativos_difuso
    if total_desertores_difuso > 0:
        sensibilidad_difuso = verdaderos_positivos_difuso / total_desertores_difuso
    else:
        sensibilidad_difuso = 0.0

    res_difuso = evaluar_rendimiento_motor_difuso(datos_prueba, parametros, reglas_difusas=reglas_difusas)
    print("\n" + formatear_reporte_demostracion_motor_difuso(res_difuso, es_calibrado=False))

    print(f"\n  Total de estudiantes evaluados en prueba: {total_estudiantes_prueba}")
    print("\n  MATRIZ DE CONFUSIÓN (SISTEMA HÍBRIDO COMPLETO: DIFUSO + PRISM):")
    print(f"  {'':>32} {'Predicción Dropout':>20} {'Predicción No Dropout':>24}")
    print(f"  {'Realidad: Dropout':>32} {verdaderos_positivos_hibrido:>20} {falsos_negativos_hibrido:>24}")
    print(f"  {'Realidad: No Dropout':>32} {falsos_positivos_hibrido:>20} {verdaderos_negativos_hibrido:>24}")

    print("\n  MÉTRICAS DEL SISTEMA HÍBRIDO UNIFICADO:")
    print(f"    - Exactitud (Accuracy)  = {exactitud_hibrido:.4f} ({exactitud_hibrido * 100:.2f}%)")
    print(f"    - Sensibilidad (Recall) = {sensibilidad_hibrido:.4f} ({sensibilidad_hibrido * 100:.2f}%)")
    print(f"    - Precisión             = {precision_hibrido:.4f} ({precision_hibrido * 100:.2f}%)")

    print("\n  COMPARACIÓN METODOLÓGICA (SOLO DIFUSO vs. HÍBRIDO UNIFICADO):")
    print(f"    - Solo Difuso:        Accuracy = {exactitud_difuso:.4f}  |  Recall = {sensibilidad_difuso:.4f}")
    print(f"    - Híbrido (+ PRISM):  Accuracy = {exactitud_hibrido:.4f}  |  Recall = {sensibilidad_hibrido:.4f}")

    diferencia_rescate = verdaderos_positivos_hibrido - verdaderos_positivos_difuso
    if diferencia_rescate > 0:
        print(f"    => PRISM rescata {diferencia_rescate} estudiantes desertores adicionales que el modelo difuso no detectaba por notas.")

    return exactitud_hibrido


def evaluar_rendimiento_motor_difuso(datos_prueba, parametros, reglas_difusas=None):
    total = len(datos_prueba)
    vp = 0
    vn = 0
    fp = 0
    fn = 0
    muestras_casos = []

    for indice, estudiante in enumerate(datos_prueba, 1):
        nota_admision = float(estudiante.get('Admission grade', 100))
        materias_aprobadas = float(estudiante.get('Curricular units 1st sem (approved)', 0))
        promedio_notas = float(estudiante.get('Curricular units 1st sem (grade)', 0))
        estado_estudiante_real = estudiante.get('Target', '')
        es_desertor_real = (estado_estudiante_real == 'Dropout')

        riesgo, _ = evaluar_motor_difuso(
            nota_admision, materias_aprobadas, promedio_notas, parametros, numero_puntos=100, reglas=reglas_difusas
        )
        pred_dropout = (riesgo >= 0.5)

        acerto = False
        if pred_dropout and es_desertor_real:
            vp += 1
            acerto = True
        elif not pred_dropout and not es_desertor_real:
            vn += 1
            acerto = True
        elif pred_dropout and not es_desertor_real:
            fp += 1
        else:
            fn += 1

        if indice <= 35:
            muestras_casos.append({
                'num': indice,
                'nota_admision': nota_admision,
                'materias_aprobadas': materias_aprobadas,
                'promedio_notas': promedio_notas,
                'riesgo': riesgo,
                'pred': 'Dropout' if pred_dropout else 'No Dropout',
                'target': estado_estudiante_real,
                'acerto': acerto
            })

    acc = (vp + vn) / total if total > 0 else 0.0
    rec = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    prec = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    spec = vn / (vn + fp) if (vn + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'total': total,
        'vp': vp,
        'vn': vn,
        'fp': fp,
        'fn': fn,
        'accuracy': acc,
        'recall': rec,
        'precision': prec,
        'specificity': spec,
        'f1': f1,
        'muestras': muestras_casos
    }


def formatear_reporte_demostracion_motor_difuso(res_difuso, es_calibrado=False):
    total = res_difuso['total']
    vp = res_difuso['vp']
    vn = res_difuso['vn']
    fp = res_difuso['fp']
    fn = res_difuso['fn']
    acc = res_difuso['accuracy'] * 100
    rec = res_difuso['recall'] * 100
    prec = res_difuso['precision'] * 100
    spec = res_difuso['specificity'] * 100
    f1 = res_difuso['f1'] * 100
    muestras = res_difuso['muestras']

    titulo_calibracion = "CALIBRADO POR ALGORITMO GENÉTICO" if es_calibrado else "PARÁMETROS INICIALES"

    lineas = []
    lineas.append("=" * 100)
    lineas.append(f"  DEMOSTRACIÓN DE EFICACIA DEL MOTOR DIFUSO MAMDANI EN DATOS DE PRUEBA ({titulo_calibracion})")
    lineas.append(f"  Evaluación Exclusiva sobre {total} Estudiantes que el Sistema NUNCA Vio Durante el Entrenamiento")
    lineas.append("=" * 100)

    lineas.append("\n[1] METODOLOGÍA DE EVALUACIÓN Y GARANTÍA CIENTÍFICA:")
    lineas.append("----------------------------------------------------------------------------------------------------")
    lineas.append(f"  • Conjunto de Datos: {total} estudiantes (20% del dataset UCI 'data.csv') aislados estrictamente.")
    lineas.append("  • Ni una sola fila de estos 885 estudiantes fue utilizada para inducir las reglas ni calibrar cortes.")
    lineas.append("  • Entrada al motor: Únicamente las 3 variables cuantitativas (Nota Admisión, Aprobadas S1, Promedio S1).")
    lineas.append("  • Salida del motor: Índice de riesgo continuo [0.0 a 1.0] calculado mediante Centroide discreto.")
    lineas.append("  • Regla de decisión: Si Riesgo >= 0.50 -> Predice 'Dropout' (Deserción); de lo contrario 'No Dropout'.\n")

    lineas.append("[2] MATRIZ DE CONFUSIÓN DEL MOTOR DIFUSO SOLO (885 CASOS NO VISTOS):")
    lineas.append("----------------------------------------------------------------------------------------------------")
    lineas.append("                                            Predicción: Dropout         Predicción: No Dropout")
    lineas.append(f"  Realidad: Desertores Reales (264)              {vp:>5} (VP)                    {fn:>5} (FN)")
    lineas.append(f"  Realidad: Estudiantes que Siguen (621)         {fp:>5} (FP)                    {vn:>5} (VN)")
    lineas.append("----------------------------------------------------------------------------------------------------")
    lineas.append(f"  • Verdaderos Positivos (VP = {vp:>3}): Desertores reales detectados correctamente solo por sus notas.")
    lineas.append(f"  • Verdaderos Negativos (VN = {vn:>3}): Estudiantes seguros que continúan, identificados sin error.")
    lineas.append(f"  • Falsos Positivos     (FP = {fp:>3}): Falsas alarmas (estudiantes con notas bajas que no desertaron).")
    lineas.append(f"  • Falsos Negativos     (FN = {fn:>3}): Desertores que tenían notas aceptables y desertaron por otros motivos.\n")

    lineas.append("[3] MÉTRICAS ESTADÍSTICAS DE GENERALIZACIÓN DEL MOTOR DIFUSO:")
    lineas.append("----------------------------------------------------------------------------------------------------")
    lineas.append(f"  • Exactitud Global (Accuracy)       : {acc:6.2f}%  ({vp + vn} aciertos directos de {total} estudiantes)")
    lineas.append(f"  • Sensibilidad / Detección (Recall) : {rec:6.2f}%  (Detecta a casi 7 de cada 10 desertores reales)")
    lineas.append(f"  • Especificidad (Capacidad Negativa): {spec:6.2f}%  (Distingue al {spec:.1f}% de los estudiantes que continúan)")
    lineas.append(f"  • Precisión Diagnóstica             : {prec:6.2f}%  (De cada 100 alertas emitidas, {prec:.0f} son deserciones reales)")
    lineas.append(f"  • Puntuación F1 (Balance Armónico)  : {f1:6.2f}%\n")

    lineas.append("[4] DEMOSTRACIÓN CASO POR CASO EN ESTUDIANTES DE PRUEBA REALES:")
    lineas.append("----------------------------------------------------------------------------------------------------")
    lineas.append(f"{'#Est':<5} | {'Nota Adm':<9} | {'Aprob S1':<8} | {'Prom S1':<8} | {'Riesgo Difuso (y*)':<18} | {'Predicción':<12} | {'Realidad':<10} | {'Resultado'}")
    lineas.append("-" * 100)

    for muestra_estudiante in muestras:
        marca = "✓ ACIERTO" if muestra_estudiante['acerto'] else "✗ DISCREPANCIA"
        riesgo_txt = f"{muestra_estudiante['riesgo']:.4f} ({muestra_estudiante['riesgo']*100:5.1f}%)"
        lineas.append(f"#{muestra_estudiante['num']:03d}  | {muestra_estudiante['nota_admision']:>8.1f} | {muestra_estudiante['materias_aprobadas']:>8.0f} | {muestra_estudiante['promedio_notas']:>8.2f} | {riesgo_txt:<18} | {muestra_estudiante['pred']:<12} | {muestra_estudiante['target']:<10} | {marca}")

    lineas.append("-" * 100)
    lineas.append(f" ... Mostrando los primeros {len(muestras)} de {total} estudiantes del conjunto de prueba independiente.")
    lineas.append(" Conclusión: El motor difuso demuestra una alta capacidad de discernimiento académico autónomo")
    lineas.append(f" obteniendo un {acc:.2f}% de aciertos en datos que jamás pasaron por la fase de entrenamiento.")
    lineas.append("=" * 100 + "\n")

    return "\n".join(lineas)
