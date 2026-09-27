# =============================================================================
# ARCHIVO: apriori.py
# Descripción: Algoritmo Apriori de Reglas de Asociación (Estructurado por Fases)
# Fases metodológicas:
#   - FASE 0: Cálculo de la Cobertura Mínima
#   - FASE 1: Generación y Poda de Ítem-Sets Frecuentes (k = 1, 2, 3) con Tablas
#   - FASE 2: Extracción y Evaluación de Reglas de Asociación (A -> B) con Tablas
#   - RESUMEN FINAL: Reglas Perfectas (1.0) y Reglas de Alta Confianza
# 100% Python estándar (sin librerías externas)
# =============================================================================
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def extraer_items_estudiante_apriori(estudiante, incluir_target=False):
    """
    Convierte el perfil de un estudiante a un conjunto de ítems en español 'Atributo=Valor'.
    """
    items = set()

    # Variables institucionales binarias
    columnas_binarias = [
        ('Displaced', 'Estudiante_Foraneo'),
        ('Educational special needs', 'Necesidades_Especiales'),
        ('Debtor', 'Tiene_Deudas'),
        ('Tuition fees up to date', 'Matricula_Al_Dia'),
        ('Scholarship holder', 'Tiene_Beca'),
        ('International', 'Estudiante_Extranjero'),
    ]

    for col, nombre_esp in columnas_binarias:
        val = estudiante.get(col)
        if val is not None:
            try:
                items.add(f"{nombre_esp}={int(val)}")
            except (ValueError, TypeError):
                pass

    # Género
    val_gen = estudiante.get('Gender')
    if val_gen is not None:
        try:
            items.add("Genero=Masculino" if int(val_gen) == 1 else "Genero=Femenino")
        except (ValueError, TypeError):
            pass

    # Horario de estudio
    val_hor = estudiante.get('Daytime/evening attendance')
    if val_hor is not None:
        try:
            items.add("Horario=Diurno" if int(val_hor) == 1 else "Horario=Nocturno")
        except (ValueError, TypeError):
            pass

    # Target (opcional, para minería global)
    if incluir_target:
        valor_objetivo = estudiante.get('Target', '')
        if valor_objetivo:
            desercion_esp = "Si" if valor_objetivo == 'Dropout' else "No"
            items.add(f"Desercion={desercion_esp}")

    # Nota de admisión discretizada
    try:
        nota_admision = float(estudiante.get('Admission grade', 0))
        if nota_admision < 115:
            items.add("Nota_Admision=Baja")
        elif nota_admision < 145:
            items.add("Nota_Admision=Media")
        else:
            items.add("Nota_Admision=Alta")
    except (ValueError, TypeError):
        pass

    # Materias aprobadas en 1er semestre
    try:
        aprobadas_primer_semestre = float(estudiante.get('Curricular units 1st sem (approved)', 0))
        if aprobadas_primer_semestre == 0:
            items.add("Aprobadas_S1=Cero")
        elif aprobadas_primer_semestre <= 3:
            items.add("Aprobadas_S1=Pocas")
        elif aprobadas_primer_semestre <= 6:
            items.add("Aprobadas_S1=Regular")
        else:
            items.add("Aprobadas_S1=Muchas")
    except (ValueError, TypeError):
        pass

    # Promedio de notas en 1er semestre
    try:
        promedio_primer_semestre = float(estudiante.get('Curricular units 1st sem (grade)', 0))
        if promedio_primer_semestre < 10.0:
            items.add("Promedio_S1=Deficiente")
        elif promedio_primer_semestre < 14.0:
            items.add("Promedio_S1=Aceptable")
        else:
            items.add("Promedio_S1=Sobresaliente")
    except (ValueError, TypeError):
        pass

    # Materias aprobadas en 2do semestre
    try:
        aprobadas_segundo_semestre = float(estudiante.get('Curricular units 2nd sem (approved)', 0))
        if aprobadas_segundo_semestre == 0:
            items.add("Aprobadas_S2=Cero")
        elif aprobadas_segundo_semestre <= 3:
            items.add("Aprobadas_S2=Pocas")
        elif aprobadas_segundo_semestre <= 6:
            items.add("Aprobadas_S2=Regular")
        else:
            items.add("Aprobadas_S2=Muchas")
    except (ValueError, TypeError):
        pass

    return items


def discretizar_para_apriori(lista_estudiantes):
    """Convierte la lista de estudiantes en una lista de transacciones (conjuntos de ítems)."""
    transacciones = []
    for estudiante in lista_estudiantes:
        items = extraer_items_estudiante_apriori(estudiante, incluir_target=False)
        transacciones.append(frozenset(items))
    return transacciones


def extraer_nombre_atributo(item):
    """Extrae el nombre de la variable antes del igual (ej. 'Tiene_Deudas=1' -> 'Tiene_Deudas')."""
    return item.split('=')[0]


def tiene_contradiccion(conjunto_items):
    """Verifica que un ítem-set no contenga dos valores diferentes del mismo atributo."""
    atributos_vistos = set()
    for item in conjunto_items:
        atributo = extraer_nombre_atributo(item)
        if atributo in atributos_vistos:
            return True
        atributos_vistos.add(atributo)
    return False


def calcular_cobertura(itemset, transacciones):
    """Cuenta cuántas transacciones contienen el conjunto de ítems."""
    conteo = 0
    for t in transacciones:
        if itemset.issubset(t):
            conteo += 1
    return conteo


def generar_combinaciones(elementos, r):
    """Genera combinaciones de tamaño r sin itertools."""
    elem = list(elementos)
    n = len(elem)
    if r == 1:
        return [(elem[i],) for i in range(n)]
    elif r == 2:
        return [(elem[i], elem[j]) for i in range(n) for j in range(i + 1, n)]
    elif r == 3:
        return [(elem[i], elem[j], elem[k]) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    return []


def ejecutar_apriori(datos_entrenamiento, soporte_minimo=0.25, confianza_minima=0.70, max_reglas=15):
    """
    Ejecuta el Algoritmo Apriori estructurado exactamente por Fases:
      - Encabezado formal
      - FASE 0: Cobertura Mínima
      - FASE 1: Generación y Poda de Ítem-Sets Frecuentes (k = 1, 2, 3) con Tablas
      - FASE 2: Extracción y Evaluación de Reglas de Asociación con Tablas
      - Resumen de Mejores Reglas
    """
    N = len(datos_entrenamiento)
    cobertura_minima = math.ceil(N * soporte_minimo)

    transacciones = discretizar_para_apriori(datos_entrenamiento)

    # Identificar todos los ítems únicos disponibles
    todos_items_posibles = set()
    for t in transacciones:
        todos_items_posibles.update(t)

    lineas_salida = []

    # ENCABEZADO FORMAL
    encabezado = "=" * 95 + "\n"
    encabezado += "         INFORME DE EJECUCIÓN: EXTRACCIÓN DE REGLAS DE ASOCIACIÓN (APRIORI)\n"
    encabezado += "                  Análisis Paso a Paso de Frecuencia y Confianza\n"
    encabezado += "=" * 95 + "\n"
    encabezado += f" Base de datos analizada:             'data.csv'\n"
    encabezado += f" Registros totales en BD (N):         {N} transacciones\n"
    encabezado += f" Soporte Mínimo requerido:            {soporte_minimo:.2f} ({soporte_minimo * 100:.0f}%)\n"
    encabezado += f" Confianza Mínima requerida:          {confianza_minima:.2f} ({confianza_minima * 100:.0f}%)\n"
    encabezado += f" Ítems Atributo=Valor Disponibles:    {len(todos_items_posibles)} ítems discretizados en español\n"
    encabezado += "=" * 95 + "\n"
    lineas_salida.append(encabezado)
    print(encabezado)

    # FASE 0: DETERMINANDO LA COBERTURA MÍNIMA
    fase0 = "\n" + "=" * 95 + "\n"
    fase0 += "             FASE 0: DETERMINANDO LA COBERTURA MÍNIMA\n"
    fase0 += "=" * 95 + "\n"
    fase0 += f" Traducción del soporte estadístico a un umbral absoluto de registros:\n"
    fase0 += f"   {N} (Registros totales) x {soporte_minimo:.2f} (Soporte mínimo) = {cobertura_minima} (Cobertura Mínima)\n"
    fase0 += f" Para que una regla se considere suficientemente frecuente, el número de ejemplos\n"
    fase0 += f" que cumplen la regla (antecedente y consecuente) debe ser de al menos {cobertura_minima}.\n"
    fase0 += "=" * 95 + "\n"
    lineas_salida.append(fase0)
    print(fase0)

    # FASE 1: GENERACIÓN Y PODA DE ÍTEM-SETS FRECUENTES
    fase1_encabezado = "\n" + "=" * 95 + "\n"
    fase1_encabezado += "         FASE 1: GENERACIÓN Y PODA DE ÍTEM-SETS FRECUENTES (k = 1, 2, 3)\n"
    fase1_encabezado += "=" * 95 + "\n"
    lineas_salida.append(fase1_encabezado)
    print(fase1_encabezado)

    itemsets_frecuentes_por_nivel = {}
    reporte_fase1 = {}

    # Nivel k = 1
    itemsets_k1 = {}
    reporte_k1 = []
    conteo_items = {}
    for t in transacciones:
        for it in t:
            conteo_items[it] = conteo_items.get(it, 0) + 1

    for it, cob in conteo_items.items():
        c_set = frozenset([it])
        aprobado = (cob >= cobertura_minima)
        reporte_k1.append({
            'itemset': c_set,
            'texto': f"{{{it}}}",
            'cobertura': cob,
            'soporte': cob / N,
            'estado': "APROBADO" if aprobado else "DESCARTADO"
        })
        if aprobado:
            itemsets_k1[c_set] = cob

    itemsets_frecuentes_por_nivel[1] = itemsets_k1
    reporte_fase1[1] = sorted(reporte_k1, key=lambda x: x['cobertura'], reverse=True)

    # Nivel k = 2 y k = 3
    items_aprobados_k1 = list(itemsets_k1.keys())

    for k in [2, 3]:
        candidatos_k = set()
        if k == 2:
            prev = list(itemsets_frecuentes_por_nivel[1].keys())
            for i in range(len(prev)):
                for j in range(i + 1, len(prev)):
                    union_set = prev[i] | prev[j]
                    if len(union_set) == 2 and not tiene_contradiccion(union_set):
                        candidatos_k.add(union_set)
        elif k == 3:
            prev_k2 = list(itemsets_frecuentes_por_nivel.get(2, {}).keys())
            for i in range(len(prev_k2)):
                for j in range(i + 1, len(prev_k2)):
                    union_set = prev_k2[i] | prev_k2[j]
                    if len(union_set) == 3 and not tiene_contradiccion(union_set):
                        # Poda apriori: todos los subconjuntos de 2 deben estar en L2
                        items_lista = list(union_set)
                        todos_frec = True
                        for idx_sub in range(3):
                            sub = frozenset(items_lista[:idx_sub] + items_lista[idx_sub + 1:])
                            if sub not in itemsets_frecuentes_por_nivel.get(2, {}):
                                todos_frec = False
                                break
                        if todos_frec:
                            candidatos_k.add(union_set)

        itemsets_k = {}
        reporte_k = []

        for cand in candidatos_k:
            cob = calcular_cobertura(cand, transacciones)
            aprobado = (cob >= cobertura_minima)
            texto_cand = " AND ".join(sorted(list(cand)))
            reporte_k.append({
                'itemset': cand,
                'texto': f"{{{texto_cand}}}",
                'cobertura': cob,
                'soporte': cob / N,
                'estado': "APROBADO" if aprobado else "DESCARTADO"
            })
            if aprobado:
                itemsets_k[cand] = cob

        reporte_fase1[k] = sorted(reporte_k, key=lambda x: x['cobertura'], reverse=True)
        itemsets_frecuentes_por_nivel[k] = itemsets_k

    # Imprimir tablas de Fase 1 para k = 1, 2, 3 (MOSTRANDO TODOS LOS ÍTEM-SETS ANALIZADOS)
    for nivel in [1, 2, 3]:
        rep = reporte_fase1.get(nivel, [])
        aprobados_c = sum(1 for x in rep if x['estado'] == "APROBADO")
        descartados_c = len(rep) - aprobados_c

        if not rep:
            continue

        longitud_maxima = max(len(item['texto']) for item in rep)
        ancho_col = max(50, longitud_maxima + 2)
        ancho_tabla = ancho_col + 45

        subtitulos_nivel = {
            1: "Fase 1 (Iteración k=1): Filtrado de Ítems Individuales",
            2: "Fase 1 (Iteración k=2): Combinación de Pares Atributo-Valor",
            3: "Fase 1 (Iteración k=3) y Condición de Parada: Tríos Atributo-Valor"
        }
        subtitulo = subtitulos_nivel.get(nivel, f"Fase 1 (Iteración k={nivel})")
        encabezado_tabla = f"\n>>> {subtitulo} (Total: {len(rep)} | Aprobados: {aprobados_c} | Descartados: {descartados_c})\n"
        encabezado_tabla += "-" * ancho_tabla + "\n"
        encabezado_tabla += f"{'#':<4} | {'Ítem-set (Atributos = Valor)':<{ancho_col}} | {'Cobertura':<10} | {'Soporte':<10} | {'Estado (>= ' + str(cobertura_minima) + ')':<15} |\n"
        encabezado_tabla += "-" * ancho_tabla + "\n"

        cuerpo_tabla = ""
        for idx, r in enumerate(rep, 1):
            marca = " [X]" if r['estado'] == "APROBADO" else "    "
            cuerpo_tabla += f"{idx:<4}{marca}| {r['texto']:<{ancho_col}} | {r['cobertura']:<10} | {r['soporte']:<10.6f} | {r['estado']:<15} |\n"

        pie_tabla = "-" * ancho_tabla + "\n"
        bloque_k = encabezado_tabla + cuerpo_tabla + pie_tabla
        lineas_salida.append(bloque_k)
        print(bloque_k)

    # FASE 2: REGLAS DE ASOCIACIÓN
    reglas_evaluadas = []

    for nivel in [2, 3]:
        itemsets_nivel = itemsets_frecuentes_por_nivel.get(nivel, {})
        for conjunto_items, cob_conjunta in itemsets_nivel.items():
            elementos = list(conjunto_items)
            for r in range(1, len(elementos)):
                for tupla_ant in generar_combinaciones(elementos, r):
                    antecedente = frozenset(tupla_ant)
                    consecuente = conjunto_items - antecedente
                    if len(consecuente) == 0:
                        continue

                    # Cobertura del antecedente
                    cob_ant = calcular_cobertura(antecedente, transacciones)
                    confianza = cob_conjunta / cob_ant if cob_ant > 0 else 0.0
                    aprobado = (confianza >= confianza_minima)

                    texto_ant = " AND ".join(sorted(list(antecedente)))
                    texto_cons = " AND ".join(sorted(list(consecuente)))
                    texto_regla = f"SI {texto_ant} ENTONCES {texto_cons}"

                    reglas_evaluadas.append({
                        'conjunto_origen': conjunto_items,
                        'antecedente': antecedente,
                        'consecuente': consecuente,
                        'texto_regla': texto_regla,
                        'cobertura_conjunta': cob_conjunta,
                        'cobertura_antecedente': cob_ant,
                        'confianza': confianza,
                        'soporte': cob_conjunta / N,
                        'soporte_regla': cob_conjunta / N,
                        'aprobado': aprobado,
                        'estado': "VÁLIDA" if aprobado else "DESCARTADA"
                    })

    # Ordenar reglas por confianza y cobertura conjunta
    reglas_evaluadas.sort(
        key=lambda r: (round(r['confianza'], 6), r['cobertura_conjunta']),
        reverse=True
    )

    reglas_validas = [r for r in reglas_evaluadas if r['aprobado']]
    reglas_descartadas = [r for r in reglas_evaluadas if not r['aprobado']]

    fase2_enc = "\n" + "=" * 95 + "\n"
    fase2_enc += "   FASE 2: LA ANATOMÍA DE LA CONFIANZA Y EVALUACIÓN DE REGLAS\n"
    fase2_enc += "=" * 95 + "\n"
    fase2_enc += " Fórmula: Confianza = Cobertura total de la regla (|A & B|) / Cobertura del antecedente (|A|)\n"
    fase2_enc += f" Total de Reglas Generadas y Evaluadas: {len(reglas_evaluadas)}\n"
    fase2_enc += f" Reglas Válidas (Confianza >= {confianza_minima:.2f}): {len(reglas_validas)}\n"
    fase2_enc += f" Reglas Descartadas (< {confianza_minima:.2f}):          {len(reglas_descartadas)}\n"
    fase2_enc += "=" * 95 + "\n"
    lineas_salida.append(fase2_enc)
    print(fase2_enc)

    longitud_maxima_r = max(len(r['texto_regla']) for r in reglas_evaluadas) if reglas_evaluadas else 55
    ancho_col_regla = max(55, longitud_maxima_r + 2)
    ancho_tabla_r = ancho_col_regla + 55

    tabla_reglas = f"\n>>> TABLA DE EVALUACIÓN DE REGLAS DE ASOCIACIÓN (Fase 2 - Confianza Mínima = {confianza_minima:.2f})\n"
    tabla_reglas += "-" * ancho_tabla_r + "\n"
    tabla_reglas += f"{'#':<4} | {'Regla de Asociación (A -> B)':<{ancho_col_regla}} | {'|A&B|':<6} | {'|A|':<6} | {'Confianza':<10} | {'Veredicto':<12} |\n"
    tabla_reglas += "-" * ancho_tabla_r + "\n"

    cuerpo_reglas = ""
    # Mostrar todas las reglas evaluadas en su totalidad sin ningún truncamiento
    for idx, r in enumerate(reglas_evaluadas, 1):
        marca = " [X]" if r['aprobado'] else "    "
        cuerpo_reglas += f"{idx:<4}{marca}| {r['texto_regla']:<{ancho_col_regla}} | {r['cobertura_conjunta']:<6} | {r['cobertura_antecedente']:<6} | {r['confianza']:<10.6f} | {r['estado']:<12} |\n"

    pie_reglas = "-" * ancho_tabla_r + "\n"

    bloque_fase2 = tabla_reglas + cuerpo_reglas + pie_reglas
    lineas_salida.append(bloque_fase2)
    print(bloque_fase2)

    # RESUMEN FINAL: EL PRODUCTO FINAL Y ANÁLISIS DE SENSIBILIDAD
    reglas_perfectas = [r for r in reglas_validas if round(r['confianza'], 4) >= 1.0]
    reglas_altas = [r for r in reglas_validas if round(r['confianza'], 4) < 1.0]

    resumen = "\n" + "=" * 95 + "\n"
    resumen += "      EL PRODUCTO FINAL: CONJUNTO DE REGLAS DE ASOCIACIÓN\n"
    resumen += f"     Las mejores reglas descubiertas (Soporte >= {soporte_minimo:.2f}, Confianza >= {confianza_minima:.2f})\n"
    resumen += "=" * 95 + "\n"

    if reglas_perfectas:
        resumen += f"\n [☆] CONFIANZA ABSOLUTA = 1.0 (Reglas Perfectas) ({len(reglas_perfectas)} reglas):\n"
        for r in reglas_perfectas:
            resumen += f"   - {r['texto_regla']}\n"
            resumen += f"     -> Cobertura total (|A & B|): {r['cobertura_conjunta']} | Cobertura antecedente (|A|): {r['cobertura_antecedente']} | Confianza = {r['cobertura_conjunta']}/{r['cobertura_antecedente']} = 1.00\n"

    if reglas_altas:
        resumen += f"\n [⬡] CONFIANZA FUERTE >= {confianza_minima:.2f} ({len(reglas_altas)} reglas):\n"
        for r in reglas_altas:
            resumen += f"   - {r['texto_regla']}\n"
            resumen += f"     -> Cobertura total (|A & B|): {r['cobertura_conjunta']} | Cobertura antecedente (|A|): {r['cobertura_antecedente']} | Confianza = {r['cobertura_conjunta']}/{r['cobertura_antecedente']} = {r['confianza']:.4f}\n"

    resumen += "\n" + "-" * 95 + "\n"
    resumen += " [!] NOTA DE ANÁLISIS DE SENSIBILIDAD:\n"
    resumen += "     La modificación del umbral de confianza no altera la generación de ítem-sets (Fase 1),\n"
    resumen += "     pero actúa como un filtro selectivo final en la Fase 2, ajustando la rigidez del sistema experto.\n"
    resumen += "=" * 95 + "\n"
    lineas_salida.append(resumen)
    print(resumen)

    salida_texto_completa = "".join(lineas_salida)

    # Eliminar duplicados conservando el formato para el sistema híbrido
    reglas_unicas = []
    pares_vistos = set()
    for r in reglas_validas:
        par = (r['antecedente'], r['consecuente'])
        if par not in pares_vistos:
            pares_vistos.add(par)
            reglas_unicas.append(r)

    return reglas_unicas, salida_texto_completa
