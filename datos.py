# =============================================================================
# ARCHIVO: datos.py
# Descripción: Lectura del archivo CSV y partición en entrenamiento y prueba
# =============================================================================
import csv
import random

def cargar_dataset(ruta_archivo):
    """
    Lee el archivo CSV de estudiantes separado por punto y coma (;).
    Convierte cada fila en un diccionario con sus nombres de columnas.
    Convierte los números a float o int para poder operar con ellos.
    """
    lista_estudiantes = []
    with open(ruta_archivo, encoding='utf-8-sig') as archivo:
        lector_csv = csv.DictReader(archivo, delimiter=';')
        for fila in lector_csv:
            estudiante = {}
            for columna, valor in fila.items():
                columna_limpia = columna.strip().replace('\t', '').replace('\ufeff', '')
                valor_limpio = valor.strip()
                try:
                    if '.' in valor_limpio:
                        estudiante[columna_limpia] = float(valor_limpio)
                    else:
                        estudiante[columna_limpia] = int(valor_limpio)
                except (ValueError, AttributeError):
                    estudiante[columna_limpia] = valor_limpio
            lista_estudiantes.append(estudiante)
    return lista_estudiantes


def partir_dataset(lista_estudiantes, porcentaje_entrenamiento=0.80, semilla=42):
    """
    Divide los estudiantes en dos grupos:
      - 80% para entrenamiento (aprender reglas y calibrar funciones)
      - 20% para prueba (evaluar aciertos finales)
    Usa una semilla fija (42) para que siempre salgan los mismos datos.
    """
    random.seed(semilla)
    indices = list(range(len(lista_estudiantes)))
    random.shuffle(indices)

    punto_corte = int(len(lista_estudiantes) * porcentaje_entrenamiento)
    indices_entrenamiento = indices[:punto_corte]
    indices_prueba = indices[punto_corte:]

    datos_entrenamiento = []
    for indice in indices_entrenamiento:
        datos_entrenamiento.append(lista_estudiantes[indice])

    datos_prueba = []
    for indice in indices_prueba:
        datos_prueba.append(lista_estudiantes[indice])

    return datos_entrenamiento, datos_prueba


def obtener_muestra_estratificada(lista_estudiantes, tam_muestra=400, columna_clase='Target', semilla=42, retornar_cuotas=False):
    """
    Extrae una submuestra representativa mediante Muestreo Estratificado (Stratified Sampling).
    Garantiza que la proporción de cada clase ('Dropout', 'Graduate', 'Enrolled') en la
    submuestra sea exactamente proporcional a su distribución en la población original,
    evitando sesgos de muestreo en el cálculo de aptitud (fitness) del Algoritmo Genético.

    Parámetros:
      - lista_estudiantes: lista completa de estudiantes (ej. conjunto de entrenamiento).
      - tam_muestra: tamaño objetivo de la muestra (ej. 400).
      - columna_clase: nombre de la columna objetivo ('Target').
      - semilla: semilla aleatoria para reproducibilidad experimental.
      - retornar_cuotas: booleano; si es True retorna tupla (muestra, dict_cuotas).
    """
    if tam_muestra >= len(lista_estudiantes):
        cuotas = {}
        for estudiante in lista_estudiantes:
            clase_estudiante = estudiante.get(columna_clase, 'Desconocido')
            cuotas[clase_estudiante] = cuotas.get(clase_estudiante, 0) + 1
        return (list(lista_estudiantes), cuotas) if retornar_cuotas else list(lista_estudiantes)

    rng = random.Random(semilla)
    estratos = {}
    for estudiante in lista_estudiantes:
        clase = estudiante.get(columna_clase, 'Desconocido')
        estratos.setdefault(clase, []).append(estudiante)

    total_poblacion = len(lista_estudiantes)
    cuotas = {}
    for clase, grupo in estratos.items():
        proporcion = len(grupo) / total_poblacion
        cuotas[clase] = int(round(proporcion * tam_muestra))

    # Ajuste por redondeo si la suma difiere ligeramente de tam_muestra
    diferencia = tam_muestra - sum(cuotas.values())
    if diferencia != 0:
        clase_mayoritaria = max(estratos.keys(), key=lambda clase_nombre: len(estratos[clase_nombre]))
        cuotas[clase_mayoritaria] += diferencia

    muestra = []
    for clase, grupo in estratos.items():
        grupo_copia = list(grupo)
        rng.shuffle(grupo_copia)
        muestra.extend(grupo_copia[:cuotas[clase]])

    rng.shuffle(muestra)
    if retornar_cuotas:
        return muestra, cuotas
    return muestra
