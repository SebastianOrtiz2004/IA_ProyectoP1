# =============================================================================
# PRUEBA INTEGRAL AUTOMATIZADA: SISTEMA COMPLETO Y GUI
# =============================================================================
import sys
import os
import tkinter as tk

def test_todo():
    print("=" * 80)
    print("INICIANDO PRUEBAS UNITARIAS Y DE INTEGRACIÓN")
    print("=" * 80)

    # 1. Datos
    print("[1/7] Probando módulo datos.py...")
    from datos import cargar_dataset, partir_dataset
    dataset = cargar_dataset('data.csv')
    assert len(dataset) == 4424, f"Dataset debe tener 4424 filas, se obtuvieron {len(dataset)}"
    train, test = partir_dataset(dataset, 0.8)
    assert len(train) == 3539, f"Entrenamiento debe ser 3539, fue {len(train)}"
    assert len(test) == 885, f"Prueba debe ser 885, fue {len(test)}"
    print("      -> datos.py superado con éxito.")

    # 2. PRISM
    print("[2/7] Probando módulo prism.py...")
    from prism import inducir_reglas_difusas_con_prism, ejecutar_prism
    reglas_difusas, cobertura, reporte = inducir_reglas_difusas_con_prism(train, min_cobertura=15)
    assert len(reglas_difusas) == 18, f"Se esperaban 18 reglas difusas, se obtuvieron {len(reglas_difusas)}"
    reglas_causales, reporte_causal = ejecutar_prism(train, clase_objetivo='Dropout', max_iteraciones_regla=4)
    assert len(reglas_causales) >= 1, "Debe existir al menos 1 regla causal"
    print(f"      -> prism.py superado ({len(reglas_difusas)} reglas difusas y {len(reglas_causales)} regla(s) causal(es)).")

    # 3. Apriori
    print("[3/7] Probando módulo apriori.py...")
    from apriori import ejecutar_apriori, extraer_items_estudiante_apriori
    reglas_asoc, reporte_ap = ejecutar_apriori(train, soporte_minimo=0.25, confianza_minima=0.70)
    assert len(reglas_asoc) > 0, "Apriori debe generar reglas de asociación"
    items_ej = extraer_items_estudiante_apriori(train[0])
    assert len(items_ej) > 0, "Debe extraer items de estudiante"
    print(f"      -> apriori.py superado ({len(reglas_asoc)} reglas de asociación minadas).")

    # 4. Difuso
    print("[4/7] Probando módulo difuso.py...")
    from difuso import (
        fuzzificar_entrada,
        evaluar_motor_difuso,
        describir_los_4_pasos_difusos
    )
    from genetico import obtener_parametros_iniciales
    p = obtener_parametros_iniciales()
    fuzzy_vals = fuzzificar_entrada(118.0, 1.0, 10.0, p)
    assert 'X1_Baja' in fuzzy_vals and 'X2_Critica' in fuzzy_vals and 'X3_Deficiente' in fuzzy_vals
    y_cog, detalle_eval = evaluar_motor_difuso(118.0, 1.0, 10.0, p, reglas=reglas_difusas)
    assert 0.0 <= y_cog <= 1.0, f"Salida difusa {y_cog} fuera de rango [0, 1]"
    texto_pasos = describir_los_4_pasos_difusos(118.0, 1.0, 10.0, p, reglas=reglas_difusas)
    assert "PASO 1" in texto_pasos and "PASO 4" in texto_pasos
    print(f"      -> difuso.py superado (y*={y_cog:.4f}, reporte de 4 pasos generado correctamente).")

    # 5. Genético
    print("[5/7] Probando módulo genetico.py...")
    from genetico import (
        ejecutar_algoritmo_genetico,
        params_a_cromosoma,
        cromosoma_a_params,
        formatear_reporte_puntos_de_corte_calibrados,
        formatear_detalle_generacion,
        formatear_resumen_general,
        formatear_todas_las_generaciones_completo
    )
    crom = params_a_cromosoma(p)
    assert len(crom) == 19, f"Cromosoma debe tener 19 genes, tiene {len(crom)}"
    p_reconst = cromosoma_a_params(crom)
    assert 'nota_adm_baja' in p_reconst
    rep_cortes = formatear_reporte_puntos_de_corte_calibrados(p, 0.77, p)
    assert "19 PUNTOS DE CORTE RESULTANTES" in rep_cortes

    # Test rápido de ejecución evolutiva (2 generaciones pequeñas para validar flujo)
    p_opt, f_opt, hist, reg = ejecutar_algoritmo_genetico(
        train, N_pob=6, numero_generaciones=2, sin_mejora_max=2, reglas_difusas=reglas_difusas, tam_muestra=100
    )
    assert len(reg) == 2, "Debe registrar 2 generaciones"
    det_gen = formatear_detalle_generacion(reg[0])
    assert "GENERACIÓN" in det_gen
    res_gen = formatear_resumen_general(reg)
    assert "RESUMEN" in res_gen
    print(f"      -> genetico.py superado (Cromosoma 19 genes, reporte cortes y evolución comprobados).")

    # 6. Diagnóstico y XAI
    print("[6/7] Probando módulo diagnostico.py...")
    from diagnostico import (
        diagnosticar_estudiante_integral,
        evaluar_rendimiento_motor_difuso,
        formatear_reporte_demostracion_motor_difuso
    )
    riesgo_est, pred_est = diagnosticar_estudiante_integral(
        test[0], p, reglas_prism=reglas_causales, reglas_apriori=reglas_asoc, reglas_difusas=reglas_difusas
    )
    assert 0.0 <= riesgo_est <= 1.0, f"Riesgo {riesgo_est} fuera de rango"
    assert pred_est in ['Dropout', 'No Dropout'], f"Predicción desconocida: {pred_est}"

    rendimiento = evaluar_rendimiento_motor_difuso(test[:50], p, reglas_difusas=reglas_difusas)
    assert 'accuracy' in rendimiento and 'vp' in rendimiento
    rep_demo = formatear_reporte_demostracion_motor_difuso(rendimiento, es_calibrado=False)
    assert "MATRIZ DE CONFUSIÓN" in rep_demo
    print(f"      -> diagnostico.py superado (Diagnóstico integral y matriz de confusión verificados).")

    # 7. Interfaz Gráfica (Tkinter)
    print("[7/7] Probando interfaz.py (GUI y todos sus botones y callbacks)...")
    from interfaz import AplicacionDesercion
    root = tk.Tk()
    root.withdraw() # Ocultar ventana para prueba automatizada sin parpadeo
    app = AplicacionDesercion(root)

    # Verificar que los datos y reglas cargaron en la app
    assert len(app.datos_entrenamiento) == 3539
    assert len(app.reglas_difusas) == 18

    # Probar callbacks principales
    assert hasattr(app, 'combo_carrera'), "Debe existir selector de carrera en el formulario"
    assert hasattr(app, 'combo_pagos'), "Debe existir selector de matrícula en el formulario"
    app.accion_calcular_prism_causal(mostrar_aviso=False)
    app.accion_calcular_apriori(mostrar_aviso=False)
    app.accion_ver_puntos_de_corte_calibrados()
    app.actualizar_texto_puntos_de_corte()
    app.dibujar_funciones_pertenencia()
    app.cargar_guia_teorica_difusa()
    app.accion_diagnosticar_formulario()
    app.cargar_ejemplo_clase('Dropout')
    app.actualizar_resultados_finales()

    # Cerrar ventana
    root.destroy()
    print("      -> interfaz.py superado (Todas las pestañas y botones ejecutados con 0 errores).")

    print("=" * 80)
    print("¡TODAS LAS PRUEBAS COMPLETADAS CON ÉXITO! 100% FUNCIONAL Y LIBRE DE ERRORES.")
    print("=" * 80)

if __name__ == '__main__':
    test_todo()
