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
    for i in indices_entrenamiento:
        datos_entrenamiento.append(lista_estudiantes[i])

    datos_prueba = []
    for i in indices_prueba:
        datos_prueba.append(lista_estudiantes[i])

    return datos_entrenamiento, datos_prueba
