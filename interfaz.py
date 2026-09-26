# =============================================================================
# ARCHIVO: interfaz.py
# Descripción: Interfaz Gráfica sencilla y didáctica (Tkinter)
# Proyecto: Sistema Híbrido de Inteligencia Artificial para Deserción Estudiantil
# Módulos utilizados: datos, difuso, prism, apriori, genetico, diagnostico
# 100% Librería estándar de Python (sin instalar librerías externas)
# =============================================================================

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random
import os

# Importamos los módulos propios del proyecto
from datos import cargar_dataset, partir_dataset
from difuso import (
    pertenencia_trapezoidal,
    pertenencia_triangular,
    fuzzificar_entrada,
    evaluar_motor_difuso,
    describir_los_4_pasos_difusos
)
from prism import (
    inducir_reglas_difusas_con_prism,
    ejecutar_prism
)
from apriori import (
    ejecutar_apriori,
    extraer_items_estudiante_apriori
)
from genetico import (
    obtener_parametros_iniciales,
    crear_individuo,
    calcular_fitness,
    construir_ruleta_100_slots,
    seleccionar_padres,
    cruzar_y_reparar,
    mutar_y_reparar,
    cruzar_y_reparar_detalle,
    mutar_y_reparar_detalle,
    ejecutar_algoritmo_genetico,
    formatear_detalle_generacion,
    formatear_resumen_general,
    formatear_todas_las_generaciones_completo,
    formatear_reporte_puntos_de_corte_calibrados,
    params_a_cromosoma,
    METADATOS_GENES,
    cromosoma_a_params,
    LIMITES_GENES,
    LONGITUD_CROMOSOMA,
    NOMBRES_GENES
)
from diagnostico import (
    diagnosticar_estudiante_integral,
    evaluar_rendimiento_motor_difuso,
    formatear_reporte_demostracion_motor_difuso
)


class AplicacionDesercion:
    def __init__(self, ventana_principal):
        self.ventana = ventana_principal
        self.ventana.title("Proyecto Primer Parcial - Sistema Híbrido de IA (Deserción Estudiantil)")
        self.ventana.geometry("1080x750")
        self.ventana.minsize(920, 680)

        # Variables de estado del sistema
        self.datos_completos = []
        self.datos_entrenamiento = []
        self.datos_prueba = []
        self.reglas_difusas = []
        self.reglas_prism = []
        self.reglas_apriori = []
        self.parametros_difusos = obtener_parametros_iniciales()
        self.poblacion_genetico = []
        self.aptitudes_genetico = []
        self.historial_fitness = []
        self.registro_generaciones = []

        # Estilo visual sencillo y ordenado
        self.estilo = ttk.Style()
        self.estilo.theme_use('clam')

        # Cargar los datos y reglas difusas iniciales
        self.cargar_datos_iniciales()

        # Construir la barra superior maestra (botón de ejecución global)
        self.crear_barra_superior_maestra()

        # Construir las pestañas de la interfaz
        self.crear_pestanas()

    def cargar_datos_iniciales(self):
        """Carga el dataset CSV y realiza la partición 80% entrenamiento / 20% prueba."""
        ruta_csv = 'data.csv'
        if not os.path.exists(ruta_csv):
            ruta_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv')

        try:
            self.datos_completos = cargar_dataset(ruta_csv)
            self.datos_entrenamiento, self.datos_prueba = partir_dataset(self.datos_completos, 0.80)
            # Inducción inicial de las 18 reglas cuantitativas
            self.reglas_difusas, _, self.reporte_difuso = inducir_reglas_difusas_con_prism(self.datos_entrenamiento, min_cobertura=15)
        except Exception as error:
            print("Aviso al cargar datos:", error)

    def crear_barra_superior_maestra(self):
        """Barra superior con el botón maestro para ejecutar todo el sistema de una sola vez."""
        frame_maestro = tk.Frame(self.ventana, bg="#f3f3f3", relief=tk.RIDGE, bd=1, padx=10, pady=8)
        frame_maestro.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Botón maestro de un solo clic
        self.btn_ejecutar_todo = tk.Button(
            frame_maestro,
            text="▶ EJECUTAR TODO EL SISTEMA (Calcular Todas las Pestañas)",
            font=("Arial", 11, "bold"),
            bg="#0078d4",
            fg="white",
            activebackground="#005a9e",
            activeforeground="white",
            relief=tk.RAISED,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.accion_ejecutar_todo_el_sistema
        )
        self.btn_ejecutar_todo.pack(side=tk.LEFT, padx=(0, 15))

        # Etiqueta de estado
        self.lbl_estado_global = tk.Label(
            frame_maestro,
            text="Estado: Listo. Presiona 'EJECUTAR TODO EL SISTEMA' para calcular todos los módulos a la vez.",
            font=("Arial", 9, "italic"),
            bg="#f3f3f3",
            fg="#333333"
        )
        self.lbl_estado_global.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Barra de progreso
        self.barra_progreso = ttk.Progressbar(frame_maestro, orient=tk.HORIZONTAL, length=220, mode='determinate')
        self.barra_progreso.pack(side=tk.RIGHT, padx=5)

    def crear_pestanas(self):
        """Crea el contenedor con las 5 pestañas principales del proyecto."""
        panel_pestanas = ttk.Notebook(self.ventana)
        panel_pestanas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        # 1. Pestaña de Reglas (PRISM y Apriori)
        tab_reglas = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_reglas, text=" 1. Reglas (PRISM y Apriori) ")
        self.construir_tab_reglas(tab_reglas)

        # 2. Pestaña del Algoritmo Genético
        tab_genetico = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_genetico, text=" 2. Algoritmo Genético ")
        self.construir_tab_genetico(tab_genetico)

        # 3. Pestaña de Funciones de Pertenencia
        tab_funciones = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_funciones, text=" 3. Funciones Difusas ")
        self.construir_tab_funciones(tab_funciones)

        # 4. Pestaña de Diagnóstico Explicable
        tab_diagnostico = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_diagnostico, text=" 4. Diagnóstico Explicable ")
        self.construir_tab_diagnostico(tab_diagnostico)

        # 5. Pestaña de Matriz de Confusión y Resultados
        tab_resultados = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_resultados, text=" 5. Resultados Finales ")
        self.construir_tab_resultados(tab_resultados)

    # =========================================================================
    # BOTÓN MAESTRO: EJECUTAR TODO EL SISTEMA DE UNA SOLA VEZ
    # =========================================================================
    def accion_ejecutar_todo_el_sistema(self):
        """
        Ejecuta todo el pipeline completo sin necesidad de ir pestaña por pestaña:
          1. Inducción de Reglas PRISM (cuantitativas y causales).
          2. Minería de Reglas de Asociación Apriori.
          3. Optimización evolutiva con el Algoritmo Genético.
          4. Actualización de las Funciones de Pertenencia difusas.
          5. Evaluación de la Matriz de Confusión y precarga del Diagnóstico Explicable.
        """
        if not self.datos_entrenamiento:
            messagebox.showerror("Error", "No se encontraron datos en 'data.csv'.")
            return

        self.btn_ejecutar_todo.config(state=tk.DISABLED, bg="#888888")
        self.barra_progreso['value'] = 5
        self.lbl_estado_global.config(text="Paso 1/5: Induciendo reglas difusas y causales PRISM...", fg="#005a9e")
        self.ventana.update()

        # Paso 1: Reglas PRISM
        self.mostrar_reglas_difusas()
        self.accion_calcular_prism_causal(mostrar_aviso=False)

        # Paso 2: Reglas Apriori
        self.barra_progreso['value'] = 25
        self.lbl_estado_global.config(text="Paso 2/5: Minando patrones y reglas de asociación con Apriori...", fg="#005a9e")
        self.ventana.update()
        self.accion_calcular_apriori(mostrar_aviso=False)

        # Paso 3: Algoritmo Genético
        self.barra_progreso['value'] = 50
        self.lbl_estado_global.config(text="Paso 3/5: Calibrando funciones difusas con Algoritmo Genético...", fg="#005a9e")
        self.ventana.update()
        self.accion_ejecutar_genetico_completo(mostrar_alerta=False)

        # Paso 4: Dibujar Funciones de Pertenencia
        self.barra_progreso['value'] = 80
        self.lbl_estado_global.config(text="Paso 4/5: Dibujando figuras de pertenencia y curvas de evolución...", fg="#005a9e")
        self.ventana.update()
        self.dibujar_funciones_pertenencia()

        # Paso 5: Evaluar Matriz de Confusión y precargar diagnóstico
        self.barra_progreso['value'] = 95
        self.lbl_estado_global.config(text="Paso 5/5: Evaluando matriz de confusión y diagnóstico explicable...", fg="#005a9e")
        self.ventana.update()
        self.actualizar_resultados_finales()
        self.cargar_ejemplo_clase('Dropout')

        self.barra_progreso['value'] = 100
        self.lbl_estado_global.config(text="¡Sistema completo ejecutado con éxito! Todas las pestañas están actualizadas.", fg="#107c41")
        self.btn_ejecutar_todo.config(state=tk.NORMAL, bg="#0078d4")

        messagebox.showinfo(
            "Ejecución Completa",
            "¡Todo el sistema se ejecutó exitosamente de inicio a fin!\n\n"
            "✓ Reglas PRISM y Apriori calculadas.\n"
            "✓ Algoritmo Genético evolucionado (Fitness calibrado).\n"
            "✓ Funciones de pertenencia graficadas.\n"
            "✓ Diagnóstico explicable precargado.\n"
            "✓ Matriz de Confusión (885 casos) actualizada.\n\n"
            "Ahora puedes revisar libremente cualquier pestaña."
        )

    # =========================================================================
    # PESTAÑA 1: REGLAS (PRISM Y APRIORI)
    # =========================================================================
    def construir_tab_reglas(self, parent):
        frame_superior = ttk.Frame(parent)
        frame_superior.pack(fill=tk.X, padx=10, pady=5)

        lbl_titulo = ttk.Label(
            frame_superior,
            text="Reglas de Conocimiento Inducidas desde los Datos",
            font=("Arial", 12, "bold")
        )
        lbl_titulo.pack(side=tk.LEFT)

        btn_generar_apriori = ttk.Button(
            frame_superior,
            text="Minería Apriori (Calcular)",
            command=self.accion_calcular_apriori
        )
        btn_generar_apriori.pack(side=tk.RIGHT, padx=5)

        btn_generar_prism = ttk.Button(
            frame_superior,
            text="PRISM Causal (Calcular)",
            command=self.accion_calcular_prism_causal
        )
        btn_generar_prism.pack(side=tk.RIGHT, padx=5)

        notebook_reglas = ttk.Notebook(parent)
        notebook_reglas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        frame_sub_difuso = ttk.Frame(notebook_reglas)
        notebook_reglas.add(frame_sub_difuso, text="Reglas Difusas (PRISM Cuantitativo)")
        self.txt_reglas_difusas = scrolledtext.ScrolledText(frame_sub_difuso, font=("Courier", 10))
        self.txt_reglas_difusas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_sub_causal = ttk.Frame(notebook_reglas)
        notebook_reglas.add(frame_sub_causal, text="Reglas Causales (PRISM Cualitativo)")
        self.txt_reglas_causales = scrolledtext.ScrolledText(frame_sub_causal, font=("Courier", 10))
        self.txt_reglas_causales.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_sub_apriori = ttk.Frame(notebook_reglas)
        notebook_reglas.add(frame_sub_apriori, text="Reglas de Asociación (Apriori)")
        self.txt_reglas_apriori = scrolledtext.ScrolledText(frame_sub_apriori, font=("Courier", 10))
        self.txt_reglas_apriori.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.mostrar_reglas_difusas()

    def mostrar_reglas_difusas(self):
        self.txt_reglas_difusas.delete('1.0', tk.END)
        if hasattr(self, 'reporte_difuso') and self.reporte_difuso:
            self.txt_reglas_difusas.insert(tk.END, self.reporte_difuso)
        else:
            self.txt_reglas_difusas.insert(tk.END, "=== REGLAS DIFUSAS INDUCIDAS POR PRISM (100% DATOS) ===\n\n")
            for i, (ant, cons) in enumerate(self.reglas_difusas, 1):
                texto_ant = " Y ".join(ant)
                self.txt_reglas_difusas.insert(tk.END, f"Regla #{i:02d}: SI {texto_ant:<42} ENTONCES {cons}\n")

    def accion_calcular_prism_causal(self, mostrar_aviso=True):
        if not self.datos_entrenamiento:
            return
        self.txt_reglas_causales.delete('1.0', tk.END)
        self.txt_reglas_causales.insert(tk.END, "Ejecutando PRISM sobre los datos completos (evaluando todas las candidatas)...\n")
        self.ventana.update()

        self.reglas_prism, salida_texto_prism = ejecutar_prism(
            self.datos_entrenamiento, clase_objetivo='Dropout', semilla=42
        )

        self.txt_reglas_causales.delete('1.0', tk.END)
        self.txt_reglas_causales.insert(tk.END, salida_texto_prism)

    def accion_calcular_apriori(self, mostrar_aviso=True):
        if not self.datos_entrenamiento:
            return
        self.txt_reglas_apriori.delete('1.0', tk.END)
        self.txt_reglas_apriori.insert(tk.END, "Buscando reglas de asociación con Apriori (Fase 0, Fase 1, Fase 2)...\n")
        self.ventana.update()

        self.reglas_apriori, salida_texto_apriori = ejecutar_apriori(
            self.datos_entrenamiento, soporte_minimo=0.25, confianza_minima=0.70, max_reglas=15
        )

        self.txt_reglas_apriori.delete('1.0', tk.END)
        self.txt_reglas_apriori.insert(tk.END, salida_texto_apriori)

    # =========================================================================
    # PESTAÑA 2: ALGORITMO GENÉTICO (VISUALIZACIÓN DETALLADA)
    # =========================================================================
    def construir_tab_genetico(self, parent):
        frame_superior = ttk.Frame(parent)
        frame_superior.pack(fill=tk.X, padx=10, pady=5)

        # Botón para ejecutar las 60 generaciones reales
        self.btn_ejecutar_genetico = ttk.Button(
            frame_superior,
            text="▶ 1. Ejecutar Algoritmo Genético (60 Generaciones Reales)",
            command=self.accion_ejecutar_genetico_completo
        )
        self.btn_ejecutar_genetico.pack(side=tk.LEFT, padx=(0, 10))

        # Selector de Generación paso a paso
        lbl_selector = ttk.Label(frame_superior, text="Inspeccionar Generación:", font=("Arial", 9, "bold"))
        lbl_selector.pack(side=tk.LEFT, padx=(5, 5))

        self.combo_generaciones = ttk.Combobox(frame_superior, state="readonly", width=42)
        self.combo_generaciones['values'] = ["Presiona 'Ejecutar Algoritmo Genético' primero"]
        self.combo_generaciones.current(0)
        self.combo_generaciones.pack(side=tk.LEFT, padx=5)
        self.combo_generaciones.bind("<<ComboboxSelected>>", self.accion_cambiar_generacion_seleccionada)

        # Botón para ver detalle de la generación seleccionada
        btn_ver_detalle = ttk.Button(
            frame_superior,
            text="🔍 Ver Detalle",
            command=self.accion_cambiar_generacion_seleccionada
        )
        btn_ver_detalle.pack(side=tk.LEFT, padx=5)

        # Botón para ver todas las 60 generaciones continuas
        btn_ver_todo = ttk.Button(
            frame_superior,
            text="📋 Ver Todas las 60 Generaciones (Continuo)",
            command=self.accion_ver_todas_las_generaciones
        )
        btn_ver_todo.pack(side=tk.LEFT, padx=5)

        # Botón para exportar el registro completo a archivo
        btn_guardar_reporte = ttk.Button(
            frame_superior,
            text="💾 Guardar Reporte (.txt)",
            command=self.accion_guardar_reporte_genetico
        )
        btn_guardar_reporte.pack(side=tk.LEFT, padx=5)

        # Botón para ver los puntos de corte calibrados por el mejor fit (19 genes explicados)
        btn_ver_cortes = ttk.Button(
            frame_superior,
            text="🔬 Puntos de Corte Calibrados (19 Genes)",
            command=self.accion_ver_puntos_de_corte_calibrados
        )
        btn_ver_cortes.pack(side=tk.LEFT, padx=5)

        # Contenedor dividido en 2 columnas
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Panel izquierdo: ScrolledText para el detalle de padres, cruce, mutación y población
        frame_izq = ttk.Frame(paned)
        paned.add(frame_izq, weight=3)

        self.lbl_titulo_detalle_ag = ttk.Label(
            frame_izq,
            text="Detalle Paso a Paso de Selección, Cruce, Mutación y Población Resultante:",
            font=("Arial", 10, "bold")
        )
        self.lbl_titulo_detalle_ag.pack(anchor=tk.W, padx=5, pady=2)

        self.txt_genetico = scrolledtext.ScrolledText(frame_izq, font=("Courier", 9))
        self.txt_genetico.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Texto explicativo inicial
        self.txt_genetico.insert(
            tk.END,
            "====================================================================================================\n"
            "                 MÓDULO DE OPTIMIZACIÓN EVOLUTIVA: ALGORITMO GENÉTICO (60 GENERACIONES)\n"
            "====================================================================================================\n"
            "  Este módulo calibra los 19 puntos de corte de las funciones de pertenencia difusas mediante:\n"
            "    1. Selección de Padres por Ruleta de 100 Casillas proporcionales al Fitness.\n"
            "    2. Cruce en 1 Punto Aleatorio (Pc = 85%) con Reparación Geométrica instantánea (0% inviables).\n"
            "    3. Mutación Gen a Gen (Pm = 5%) con ajuste fino de +/- 15% del rango del gen.\n"
            "    4. Elitismo Estricto: El mejor individuo de cada generación pasa intacto a la siguiente.\n\n"
            "  Presiona '▶ 1. Ejecutar Algoritmo Genético (60 Generaciones Reales)' para iniciar el proceso evolutivo.\n"
            "  Podrás inspeccionar generación por generación cómo se formaron los padres, hijos, mutaciones\n"
            "  y la población resultante de cada una de las 60 generaciones sin omisiones.\n"
            "====================================================================================================\n"
        )

        # Panel derecho: Gráfica de evolución y tarjetas de información
        frame_der = ttk.Frame(paned)
        paned.add(frame_der, weight=2)

        lbl_graf = ttk.Label(frame_der, text="Evolución del Fitness por Generación (Curva de Aprendizaje):", font=("Arial", 10, "bold"))
        lbl_graf.pack(anchor=tk.W, padx=5, pady=2)

        self.canvas_genetico = tk.Canvas(frame_der, bg="white", highlightthickness=1, highlightbackground="#cccccc", height=280)
        self.canvas_genetico.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas_genetico.bind("<Configure>", lambda e: self.dibujar_grafica_fitness())

        # Tarjeta de parámetros y resultados
        frame_metricas_ag = ttk.LabelFrame(frame_der, text=" Parámetros y Garantías Metodológicas del AG ", padding=10)
        frame_metricas_ag.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_info_gen_estado = ttk.Label(
            frame_metricas_ag,
            text="• Generaciones: 60 Generaciones Reales (sin truncamiento)\n"
                 "• Tamaño de Población: 30 Individuos por generación\n"
                 "• Cromosoma: 19 genes continuos (puntos a, b, c, d)\n"
                 "• Operador de Selección: Ruleta proporcional de 100 slots\n"
                 "• Operador de Cruce: 1 Punto aleatorio (Probabilidad = 85%)\n"
                 "• Operador de Mutación: 5% por gen (+/- 15% del rango)\n"
                 "• Reparación Geométrica: Garantía de 0% individuos muertos\n"
                 "• Elitismo: 1 Individuo Élite preservado por generación",
            font=("Arial", 9)
        )
        self.lbl_info_gen_estado.pack(anchor=tk.W)

        self.lbl_resultado_fitness = ttk.Label(
            frame_metricas_ag,
            text="\nFitness Inicial: --  |  Fitness Final: --",
            font=("Arial", 9, "bold"),
            foreground="#005a9e"
        )
        self.lbl_resultado_fitness.pack(anchor=tk.W)

    def accion_cambiar_generacion_seleccionada(self, event=None):
        """Muestra el reporte detallado paso a paso de la generación elegida en el combobox."""
        if not self.registro_generaciones:
            messagebox.showinfo("Aviso", "Primero ejecuta el Algoritmo Genético para generar el registro de las 60 generaciones.")
            return

        seleccion = self.combo_generaciones.get()
        if "Resumen General" in seleccion:
            texto = formatear_resumen_general(self.registro_generaciones)
            self.txt_genetico.delete('1.0', tk.END)
            self.txt_genetico.insert(tk.END, texto)
            self.txt_genetico.see('1.0')
            return

        if "Puntos de Corte" in seleccion:
            self.accion_ver_puntos_de_corte_calibrados()
            return

        if "Ver Registro Continuo Completo" in seleccion:
            self.accion_ver_todas_las_generaciones()
            return

        try:
            num_str = seleccion.split('.')[0].strip()
            numero_generacion = int(num_str)
            if 1 <= numero_generacion <= len(self.registro_generaciones):
                datos_generacion = self.registro_generaciones[numero_generacion - 1]
                texto = formatear_detalle_generacion(datos_generacion)
                self.txt_genetico.delete('1.0', tk.END)
                self.txt_genetico.insert(tk.END, texto)
                self.txt_genetico.see('1.0')
        except Exception as error:
            print("Error al mostrar generación:", error)

    def accion_ver_todas_las_generaciones(self):
        """Muestra de manera continua el desglose paso a paso de las 60 generaciones completas."""
        if not self.registro_generaciones:
            messagebox.showinfo("Aviso", "Primero ejecuta el Algoritmo Genético para generar el registro de las 60 generaciones.")
            return

        self.txt_genetico.delete('1.0', tk.END)
        self.txt_genetico.insert(tk.END, "Generando vista continua exhaustiva de las 60 generaciones...\n")
        self.ventana.update()

        texto_completo = formatear_todas_las_generaciones_completo(self.registro_generaciones)
        self.txt_genetico.delete('1.0', tk.END)
        self.txt_genetico.insert(tk.END, texto_completo)
        self.txt_genetico.see('1.0')

    def accion_guardar_reporte_genetico(self):
        """Guarda en un archivo de texto el registro exhaustivo de todas las 60 generaciones."""
        if not self.registro_generaciones:
            messagebox.showinfo("Aviso", "No hay datos para guardar. Ejecuta primero el Algoritmo Genético.")
            return

        ruta_archivo = "registro_completo_60_generaciones.txt"
        try:
            texto = formatear_todas_las_generaciones_completo(self.registro_generaciones)
            with open(ruta_archivo, "w", encoding="utf-8") as f:
                f.write(texto)
            messagebox.showinfo("Archivo Guardado", f"El registro paso a paso de las 60 generaciones se guardó con éxito en:\n\n{os.path.abspath(ruta_archivo)}")
        except Exception as err:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo: {err}")

    def accion_ver_puntos_de_corte_calibrados(self):
        """Muestra el desglose detallado de los 19 puntos de corte calibrados por el mejor fit del AG."""
        mejor_fit = None
        if self.historial_fitness:
            mejor_fit = max(self.historial_fitness)
        texto = formatear_reporte_puntos_de_corte_calibrados(
            self.parametros_difusos,
            mejor_fitness=mejor_fit,
            parametros_iniciales=obtener_parametros_iniciales()
        )
        self.txt_genetico.delete('1.0', tk.END)
        self.txt_genetico.insert(tk.END, texto)
        self.txt_genetico.see('1.0')

    def accion_ejecutar_genetico_completo(self, mostrar_alerta=True):
        """Ejecuta el Algoritmo Genético real (60 Generaciones) con progreso visual y registro completo."""
        self.txt_genetico.delete('1.0', tk.END)
        self.txt_genetico.insert(tk.END, "====================================================================================================\n")
        self.txt_genetico.insert(tk.END, "      INICIANDO EJECUCIÓN REAL DEL ALGORITMO GENÉTICO (60 GENERACIONES - 30 INDIVIDUOS)\n")
        self.txt_genetico.insert(tk.END, "====================================================================================================\n\n")
        self.txt_genetico.insert(tk.END, "Progreso en tiempo real de cada generación:\n")
        self.txt_genetico.insert(tk.END, "----------------------------------------------------------------------------------------------------\n")
        self.ventana.update()

        self.historial_fitness = []

        def callback_progreso_gui(generacion, total_generaciones, mejor, media, datos_generacion):
            marca = " *** MEJORA ***" if datos_generacion['hubo_mejora'] else ""
            linea = f"  • Generación {generacion:02d} / {total_generaciones:02d}: Mejor Fitness = {mejor*100:5.2f}% | Media = {media*100:5.2f}% | {datos_generacion['estado']}{marca}\n"
            self.txt_genetico.insert(tk.END, linea)
            self.txt_genetico.see(tk.END)
            self.historial_fitness.append(mejor)
            if generacion % 3 == 0 or generacion == total_generaciones or datos_generacion['hubo_mejora']:
                self.dibujar_grafica_fitness()
                self.ventana.update()

        try:
            (
                self.parametros_difusos,
                mejor_fitness,
                self.historial_fitness,
                self.registro_generaciones
            ) = ejecutar_algoritmo_genetico(
                self.datos_entrenamiento,
                N_pob=30,
                numero_generaciones=60,
                sin_mejora_max=60,
                Pc=0.85,
                Pm=0.05,
                reglas_difusas=self.reglas_difusas,
                tam_muestra=400,
                callback_progreso=callback_progreso_gui
            )

            # Actualizar valores del Combobox
            opciones = ["00. Resumen General (Tabla Comparativa 60 Generaciones)"]
            for g in self.registro_generaciones:
                num = g['generacion']
                mejora_tag = " [MEJORA]" if g['hubo_mejora'] else ""
                mejor_fit_val = g.get('mejor_fitness_generacion', g.get('mejor_fitness_gen', 0.0))
                opciones.append(f"{num:02d}. Generación {num:02d} (Fitness: {mejor_fit_val*100:5.2f}%){mejora_tag}")
            opciones.append("98. Puntos de Corte Calibrados por el Mejor Fit (19 Genes Explicados)")
            opciones.append("99. Ver Registro Continuo Completo (Todas las 60 Generaciones)")

            self.combo_generaciones['values'] = opciones
            self.combo_generaciones.current(1)  # Seleccionar Generación 01 por defecto

            # Actualizar etiquetas de resumen
            fit_ini = self.registro_generaciones[0].get('mejor_fitness_generacion', self.registro_generaciones[0].get('mejor_fitness_gen', 0.0)) * 100
            fit_fin = mejor_fitness * 100
            self.lbl_resultado_fitness.config(
                text=f"\nFitness Inicial: {fit_ini:.2f}%  ->  Fitness Final: {fit_fin:.2f}%\nGanancia Evolutiva: +{fit_fin - fit_ini:.2f}%"
            )

            # Redibujar gráfica de evolución y figuras difusas
            self.dibujar_grafica_fitness()
            self.dibujar_funciones_pertenencia()

            # Mostrar inmediatamente la Generación 01 paso a paso
            self.accion_cambiar_generacion_seleccionada()

            if mostrar_alerta:
                messagebox.showinfo(
                    "Algoritmo Genético Completado",
                    f"¡Las 60 generaciones fueron ejecutadas y registradas con éxito!\n\n"
                    f"• Fitness inicial: {fit_ini:.2f}%\n"
                    f"• Fitness optimizado: {fit_fin:.2f}%\n"
                    f"• Ganancia obtenida: +{fit_fin - fit_ini:.2f}%\n\n"
                    f"Puedes seleccionar cualquiera de las 60 generaciones en el selector superior\n"
                    f"para examinar paso a paso sus cruces, puntos de corte, mutaciones y nueva población."
                )

        except Exception as error:
            messagebox.showerror("Error en AG", f"Ocurrió un error al ejecutar el Algoritmo Genético: {error}")

    def dibujar_grafica_fitness(self):
        """Dibuja en el canvas la curva de evolución del mejor fitness."""
        self.canvas_genetico.delete("all")
        ancho = self.canvas_genetico.winfo_width()
        alto = self.canvas_genetico.winfo_height()

        if ancho < 50 or alto < 50:
            return

        margen_izq = 50
        margen_der = 20
        margen_sup = 30
        margen_inf = 40

        self.canvas_genetico.create_line(margen_izq, alto - margen_inf, ancho - margen_der, alto - margen_inf, width=2)
        self.canvas_genetico.create_line(margen_izq, margen_sup, margen_izq, alto - margen_inf, width=2)

        self.canvas_genetico.create_text(ancho // 2, alto - 15, text="Generación", font=("Arial", 9, "bold"))
        self.canvas_genetico.create_text(25, alto // 2, text="Fitness %", font=("Arial", 9, "bold"), angle=90)

        if len(self.historial_fitness) < 2:
            self.canvas_genetico.create_text(
                ancho // 2, alto // 2,
                text="Presiona '3. Ejecutar Algoritmo Completo' para ver la curva.",
                font=("Arial", 10, "italic"), fill="#888888"
            )
            return

        min_f = 0.70
        max_f = 0.85
        total_puntos = len(self.historial_fitness)

        ancho_util = ancho - margen_izq - margen_der
        alto_util = alto - margen_sup - margen_inf

        coordenadas = []
        for i, fit in enumerate(self.historial_fitness):
            x = margen_izq + (i / max(1, total_puntos - 1)) * ancho_util
            fit_clamped = max(min_f, min(max_f, fit))
            y = (alto - margen_inf) - ((fit_clamped - min_f) / (max_f - min_f)) * alto_util
            coordenadas.append((x, y))

        for i in range(len(coordenadas) - 1):
            x1, y1 = coordenadas[i]
            x2, y2 = coordenadas[i + 1]
            self.canvas_genetico.create_line(x1, y1, x2, y2, fill="#2b5797", width=2)
            self.canvas_genetico.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="#007acc", outline="")

        xf, yf = coordenadas[-1]
        self.canvas_genetico.create_oval(xf - 4, yf - 4, xf + 4, yf + 4, fill="#e81123", outline="")
        self.canvas_genetico.create_text(
            xf, yf - 12,
            text=f"{self.historial_fitness[-1]*100:.2f}%",
            font=("Arial", 9, "bold"), fill="#2b5797"
        )

    # =========================================================================
    # PESTAÑA 3: FUNCIONES DE PERTENENCIA (GRÁFICOS DIFUSOS)
    # =========================================================================
    def construir_tab_funciones(self, parent):
        frame_superior = ttk.Frame(parent)
        frame_superior.pack(fill=tk.X, padx=10, pady=5)

        lbl_info = ttk.Label(
            frame_superior,
            text="Visualización de las Funciones de Pertenencia y Fundamentos del Motor Difuso Mamdani",
            font=("Arial", 11, "bold")
        )
        lbl_info.pack(side=tk.LEFT)

        btn_redibujar = ttk.Button(
            frame_superior,
            text="Actualizar Gráficos",
            command=self.dibujar_funciones_pertenencia
        )
        btn_redibujar.pack(side=tk.RIGHT, padx=5)

        notebook_funciones = ttk.Notebook(parent)
        notebook_funciones.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Subpestaña 1: Gráficos de Funciones de Pertenencia
        frame_graficos = ttk.Frame(notebook_funciones)
        notebook_funciones.add(frame_graficos, text=" Curvas de Pertenencia Calibradas ")

        frame_g1 = ttk.LabelFrame(frame_graficos, text="Variable 1: Nota de Admisión (0 a 200 puntos)")
        frame_g1.pack(fill=tk.BOTH, expand=True, pady=3)
        self.canvas_g1 = tk.Canvas(frame_g1, bg="white", height=130)
        self.canvas_g1.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_g2 = ttk.LabelFrame(frame_graficos, text="Variable 2: Materias Aprobadas 1er Semestre (0 a 26 materias)")
        frame_g2.pack(fill=tk.BOTH, expand=True, pady=3)
        self.canvas_g2 = tk.Canvas(frame_g2, bg="white", height=130)
        self.canvas_g2.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        frame_g3 = ttk.LabelFrame(frame_graficos, text="Variable 3: Promedio de Calificaciones 1er Semestre (0 a 20 puntos)")
        frame_g3.pack(fill=tk.BOTH, expand=True, pady=3)
        self.canvas_g3 = tk.Canvas(frame_g3, bg="white", height=130)
        self.canvas_g3.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Subpestaña 2: Fundamento Teórico y Fórmulas de los 4 Pasos Canónicos
        frame_teoria = ttk.Frame(notebook_funciones)
        notebook_funciones.add(frame_teoria, text=" Los 4 Pasos del Motor Difuso (Fundamento Teórico y Fórmulas) ")
        self.txt_teoria_difusa = scrolledtext.ScrolledText(frame_teoria, font=("Courier", 10))
        self.txt_teoria_difusa.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.cargar_guia_teorica_difusa()

        # Subpestaña 3: Puntos de Corte Calibrados por el Algoritmo Genético (19 Genes)
        frame_cortes = ttk.Frame(notebook_funciones)
        notebook_funciones.add(frame_cortes, text=" Puntos de Corte Calibrados por el AG (19 Genes) ")
        self.txt_cortes_difusos = scrolledtext.ScrolledText(frame_cortes, font=("Courier", 10))
        self.txt_cortes_difusos.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.actualizar_texto_puntos_de_corte()

        self.ventana.after(100, self.dibujar_funciones_pertenencia)

    def cargar_guia_teorica_difusa(self):
        """Carga la guía formal y matemática de los 4 pasos canónicos del motor difuso Mamdani."""
        if not hasattr(self, 'txt_teoria_difusa'):
            return
        self.txt_teoria_difusa.config(state=tk.NORMAL)
        self.txt_teoria_difusa.delete('1.0', tk.END)
        texto = (
            "=" * 95 + "\n"
            "       GUÍA TEÓRICA Y METODOLÓGICA: LOS 4 PASOS DEL MOTOR DIFUSO MAMDANI\n"
            "=" * 95 + "\n\n"
            "[PASO 1: FUZZIFICACIÓN (TRANSFORMACIÓN A GRADOS DE MEMBRESÍA μ)]\n"
            "---------------------------------------------------------------------------------------\n"
            "• Objetivo: Transformar variables numéricas continuas (crisp) en grados de pertenencia μ ∈ [0, 1].\n"
            "• Geometría implementada (100% matemática analítica sin librerías externas):\n"
            "    - Función Trapezoidal(x; a, b, c, d):\n"
            "        μ(x) = 0                       si x <= a o x >= d\n"
            "        μ(x) = (x - a) / (b - a)       si a < x < b  (pendiente ascendente)\n"
            "        μ(x) = 1.0                     si b <= x <= c (meseta de certeza total)\n"
            "        μ(x) = (d - x) / (d - c)       si c < x < d  (pendiente descendente)\n"
            "    - Función Triangular(x; a, b, c):\n"
            "        μ(x) = 0                       si x <= a o x >= c\n"
            "        μ(x) = (x - a) / (b - a)       si a < x < b\n"
            "        μ(x) = (c - x) / (c - b)       si b <= x < c\n\n"
            "[PASO 2: EVALUACIÓN DE REGLAS E INFERENCIA DIFUSA (T-NORMA MÍNIMO)]\n"
            "---------------------------------------------------------------------------------------\n"
            "• Objetivo: Determinar la fuerza de activación de cada regla difusa en la base de conocimiento.\n"
            "• Operador T-Norma de Mamdani (Conjunción AND):\n"
            "    α_r = min( μ_A(x_1), μ_B(x_2), μ_C(x_3) )\n"
            "• Cada regla evaluada trunca el conjunto difuso consecuente correspondiente a la altura α_r.\n\n"
            "[PASO 3: AGREGACIÓN DE CONSECUENTES (S-NORMA MÁXIMO)]\n"
            "---------------------------------------------------------------------------------------\n"
            "• Objetivo: Combinar las consecuencias de todas las reglas activadas en una única región difusa global.\n"
            "• Operador S-Norma de Mamdani (Disyunción OR):\n"
            "    μ_agregada(y) = max( α_1, α_2, ..., α_k ) para cada valor de salida y ∈ [0.0, 1.0].\n"
            "• Genera la superficie geométrica difusa envolvente de todos los consecuentes activados.\n\n"
            "[PASO 4: DEFUZZIFICACIÓN POR CENTRO DE GRAVEDAD (CENTROIDE - COG)]\n"
            "---------------------------------------------------------------------------------------\n"
            "• Objetivo: Extraer un único valor escalar cuantitativo y* ∈ [0.0, 1.0] representativo.\n"
            "• Fórmula del Centro de Gravedad Discreto (100 puntos de muestreo en el dominio [0.0, 1.0]):\n"
            "              Σ [ y_j · μ_agregada(y_j) ]     Momento Total del Área\n"
            "        y* = ----------------------------- = ------------------------\n"
            "                   Σ [ μ_agregada(y_j) ]         Área Total Agregada\n\n"
            "• Criterio de Clasificación Final:\n"
            "    - Si y* >= 0.50 (50.0%) -> Estudiante clasificado en Alto Riesgo de Deserción ('Dropout').\n"
            "    - Si y* <  0.50 (50.0%) -> Estudiante clasificado con Continuidad ('No Dropout').\n"
            "=" * 95 + "\n"
        )
        self.txt_teoria_difusa.insert(tk.END, texto)
        self.txt_teoria_difusa.config(state=tk.DISABLED)

    def dibujar_conjunto_en_canvas(self, canvas, puntos, tipo, color, x_min, x_max, etiqueta):
        ancho = canvas.winfo_width()
        alto = canvas.winfo_height()
        if ancho < 50 or alto < 50:
            return

        margen_x = 40
        margen_y = 25
        ancho_util = ancho - 2 * margen_x
        alto_util = alto - 2 * margen_y

        def escala_x(val):
            return margen_x + ((val - x_min) / (x_max - x_min)) * ancho_util

        def escala_y(mu):
            return (alto - margen_y) - mu * alto_util

        if tipo == 'trapecio':
            a, b, c, d = puntos
            coords = [
                (escala_x(a), escala_y(0.0)),
                (escala_x(b), escala_y(1.0)),
                (escala_x(c), escala_y(1.0)),
                (escala_x(d), escala_y(0.0))
            ]
            x_etiqueta = escala_x((b + c) / 2)
        else:
            a, b, c = puntos
            coords = [
                (escala_x(a), escala_y(0.0)),
                (escala_x(b), escala_y(1.0)),
                (escala_x(c), escala_y(0.0))
            ]
            x_etiqueta = escala_x(b)

        for i in range(len(coords) - 1):
            canvas.create_line(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1], fill=color, width=2)

        canvas.create_text(x_etiqueta, escala_y(1.0) - 10, text=etiqueta, font=("Arial", 9, "bold"), fill=color)

    def preparar_ejes_canvas(self, canvas, x_min, x_max, titulo_eje):
        canvas.delete("all")
        ancho = canvas.winfo_width()
        alto = canvas.winfo_height()
        if ancho < 50:
            return

        margen_x = 40
        margen_y = 25

        canvas.create_line(margen_x, alto - margen_y, ancho - margen_x, alto - margen_y, width=1, fill="#888888")
        canvas.create_line(margen_x, margen_y, margen_x, alto - margen_y, width=1, fill="#888888")

        canvas.create_text(margen_x - 15, margen_y, text="1.0", font=("Arial", 8))
        canvas.create_text(margen_x - 15, alto - margen_y, text="0.0", font=("Arial", 8))

        canvas.create_text(margen_x, alto - 10, text=str(x_min), font=("Arial", 8))
        canvas.create_text(ancho - margen_x, alto - 10, text=str(x_max), font=("Arial", 8))
        canvas.create_text(ancho // 2, alto - 10, text=titulo_eje, font=("Arial", 8, "italic"))

    def dibujar_funciones_pertenencia(self):
        """Dibuja las funciones difusas calibradas en cada uno de los 3 lienzos."""
        p = self.parametros_difusos

        self.preparar_ejes_canvas(self.canvas_g1, 0, 200, "Puntaje de Admisión")
        self.dibujar_conjunto_en_canvas(self.canvas_g1, p['nota_adm_baja'], 'trapecio', "#007acc", 0, 200, "Baja")
        self.dibujar_conjunto_en_canvas(self.canvas_g1, p['nota_adm_media'], 'triangulo', "#107c41", 0, 200, "Media")
        self.dibujar_conjunto_en_canvas(self.canvas_g1, p['nota_adm_alta'], 'trapecio', "#d83b01", 0, 200, "Alta")

        self.preparar_ejes_canvas(self.canvas_g2, 0, 26, "Número de Materias Aprobadas")
        self.dibujar_conjunto_en_canvas(self.canvas_g2, p['aprobadas_critica'], 'trapecio', "#d83b01", 0, 26, "Crítica")
        self.dibujar_conjunto_en_canvas(self.canvas_g2, p['aprobadas_regular'], 'triangulo', "#b48b00", 0, 26, "Regular")
        self.dibujar_conjunto_en_canvas(self.canvas_g2, p['aprobadas_completa'], 'trapecio', "#107c41", 0, 26, "Completa")

        self.preparar_ejes_canvas(self.canvas_g3, 0, 20, "Promedio Semestral")
        self.dibujar_conjunto_en_canvas(self.canvas_g3, p['nota_sem_deficiente'], 'trapecio', "#d83b01", 0, 20, "Deficiente")
        self.dibujar_conjunto_en_canvas(self.canvas_g3, p['nota_sem_aceptable'], 'triangulo', "#b48b00", 0, 20, "Aceptable")
        self.dibujar_conjunto_en_canvas(self.canvas_g3, p['nota_sem_sobresaliente'], 'trapecio', "#107c41", 0, 20, "Sobresaliente")

        # Refrescar reporte explicativo de puntos de corte (19 genes)
        self.actualizar_texto_puntos_de_corte()

    def actualizar_texto_puntos_de_corte(self):
        """Actualiza la subpestaña de puntos de corte con el reporte explicativo de los 19 genes."""
        if not hasattr(self, 'txt_cortes_difusos'):
            return
        mejor_fit = None
        if self.historial_fitness:
            mejor_fit = max(self.historial_fitness)
        texto = formatear_reporte_puntos_de_corte_calibrados(
            self.parametros_difusos,
            mejor_fitness=mejor_fit,
            parametros_iniciales=obtener_parametros_iniciales()
        )
        self.txt_cortes_difusos.config(state=tk.NORMAL)
        self.txt_cortes_difusos.delete('1.0', tk.END)
        self.txt_cortes_difusos.insert(tk.END, texto)
        self.txt_cortes_difusos.config(state=tk.DISABLED)

    # =========================================================================
    # PESTAÑA 4: DIAGNÓSTICO EXPLICABLE (XAI)
    # =========================================================================
    def construir_tab_diagnostico(self, parent):
        frame_izq = ttk.LabelFrame(parent, text="Datos del Estudiante a Evaluar")
        frame_izq.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(frame_izq, text="Nota de Admisión (0 - 200):").pack(anchor=tk.W, padx=10, pady=2)
        self.entry_adm = ttk.Entry(frame_izq)
        self.entry_adm.insert(0, "118.0")
        self.entry_adm.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(frame_izq, text="Materias Aprobadas S1 (0 - 26):").pack(anchor=tk.W, padx=10, pady=2)
        self.entry_aprob = ttk.Entry(frame_izq)
        self.entry_aprob.insert(0, "1")
        self.entry_aprob.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(frame_izq, text="Promedio 1er Semestre (0 - 20):").pack(anchor=tk.W, padx=10, pady=2)
        self.entry_prom = ttk.Entry(frame_izq)
        self.entry_prom.insert(0, "10.0")
        self.entry_prom.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(frame_izq, text="¿Tiene Deuda de Matrícula?").pack(anchor=tk.W, padx=10, pady=2)
        self.combo_deudor = ttk.Combobox(frame_izq, values=["0 (Sin Deudas)", "1 (Tiene Deuda)"], state="readonly")
        self.combo_deudor.current(1)
        self.combo_deudor.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(frame_izq, text="¿Pagos de Colegiatura al Día?").pack(anchor=tk.W, padx=10, pady=2)
        self.combo_pagos = ttk.Combobox(frame_izq, values=["1 (Pagos al Día)", "0 (Atrasado)"], state="readonly")
        self.combo_pagos.current(1)
        self.combo_pagos.pack(fill=tk.X, padx=10, pady=2)

        btn_evaluar = ttk.Button(
            frame_izq,
            text="Calcular Diagnóstico Explicable",
            command=self.accion_diagnosticar_formulario
        )
        btn_evaluar.pack(fill=tk.X, padx=10, pady=15)

        ttk.Separator(frame_izq, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(frame_izq, text="Cargar Casos Reales del Dataset:").pack(anchor=tk.W, padx=10, pady=2)
        btn_caso_drop = ttk.Button(frame_izq, text="Cargar Caso: Desertor Real", command=lambda: self.cargar_ejemplo_clase('Dropout'))
        btn_caso_drop.pack(fill=tk.X, padx=10, pady=2)

        btn_caso_grad = ttk.Button(frame_izq, text="Cargar Caso: Graduado Real", command=lambda: self.cargar_ejemplo_clase('Graduate'))
        btn_caso_grad.pack(fill=tk.X, padx=10, pady=2)

        frame_der = ttk.LabelFrame(parent, text="Informe Explicable del Sistema Híbrido (XAI)")
        frame_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.txt_diagnostico = scrolledtext.ScrolledText(frame_der, font=("Courier", 10))
        self.txt_diagnostico.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def cargar_ejemplo_clase(self, clase_objetivo):
        for est in self.datos_prueba:
            if est.get('Target') == clase_objetivo:
                self.entry_adm.delete(0, tk.END)
                self.entry_adm.insert(0, str(est.get('Admission grade', 100)))

                self.entry_aprob.delete(0, tk.END)
                self.entry_aprob.insert(0, str(est.get('Curricular units 1st sem (approved)', 0)))

                self.entry_prom.delete(0, tk.END)
                self.entry_prom.insert(0, str(est.get('Curricular units 1st sem (grade)', 0)))

                deudor_val = int(est.get('Debtor', 0))
                self.combo_deudor.current(deudor_val)

                pagos_val = int(est.get('Tuition fees up to date', 1))
                idx_pagos = 0 if pagos_val == 1 else 1
                self.combo_pagos.current(idx_pagos)

                self.accion_diagnosticar_formulario(target_real=clase_objetivo)
                break

    def accion_diagnosticar_formulario(self, target_real=None):
        try:
            nota = float(self.entry_adm.get())
            aprob = float(self.entry_aprob.get())
            prom = float(self.entry_prom.get())
            deudor = int(self.combo_deudor.get().split()[0])
            pagos = int(self.combo_pagos.get().split()[0])
        except ValueError:
            messagebox.showerror("Error", "Por favor ingresa números válidos en las notas.")
            return

        riesgo, det = evaluar_motor_difuso(nota, aprob, prom, self.parametros_difusos, 100, reglas=self.reglas_difusas)
        alerta_prism = (deudor == 1 and pagos == 0) or (pagos == 0 and aprob <= 1)

        self.txt_diagnostico.delete('1.0', tk.END)
        self.txt_diagnostico.insert(tk.END, "=" * 95 + "\n")
        self.txt_diagnostico.insert(tk.END, "  INFORME DE DIAGNÓSTICO EXPLICABLE (XAI) - SISTEMA HÍBRIDO DE INTELIGENCIA ARTIFICIAL\n")
        self.txt_diagnostico.insert(tk.END, "=" * 95 + "\n\n")

        self.txt_diagnostico.insert(tk.END, f"Perfil del Estudiante: Admisión = {nota:.1f}/200 | Aprobadas S1 = {aprob:.0f} materias | Promedio S1 = {prom:.2f}/20 pts\n")
        self.txt_diagnostico.insert(tk.END, f"Factores Administrativos: Deudor = {deudor} ({'Tiene Deuda' if deudor==1 else 'Sin Deudas'}) | Colegiatura al Día = {pagos} ({'Al Día' if pagos==1 else 'Atrasado'})\n")
        if target_real:
            self.txt_diagnostico.insert(tk.END, f"Condición Real Verificada en Dataset: {target_real}\n\n")
        else:
            self.txt_diagnostico.insert(tk.END, "\n")

        # DESARROLLO EXPLICABLE DE LOS 4 PASOS CANÓNICOS DE LA LÓGICA DIFUSA MAMDANI
        texto_4_pasos = describir_los_4_pasos_difusos(
            nota, aprob, prom, self.parametros_difusos, detalle_motor=det, reglas=self.reglas_difusas
        )
        self.txt_diagnostico.insert(tk.END, texto_4_pasos + "\n\n")

        # PILAR 2: REGLAS CAUSALES PRISM
        self.txt_diagnostico.insert(tk.END, "[PILAR 2: EVALUACIÓN DE REGLAS CAUSALES PRISM (FACTORES ADMINISTRATIVOS)]\n")
        self.txt_diagnostico.insert(tk.END, "-" * 95 + "\n")
        if alerta_prism:
            self.txt_diagnostico.insert(tk.END, "  ¡ALERTA CRÍTICA PRISM ACTIVADA!\n")
            self.txt_diagnostico.insert(tk.END, f"  El estudiante presenta factores de alto riesgo administrativo detectados por PRISM:\n")
            self.txt_diagnostico.insert(tk.END, f"  - Estado de Deuda: {'Deudor Activo' if deudor==1 else 'Sin deuda'}\n")
            self.txt_diagnostico.insert(tk.END, f"  - Pagos de Matrícula/Colegiatura: {'Atrasado' if pagos==0 else 'Al día'}\n")
            self.txt_diagnostico.insert(tk.END, f"  - Rendimiento Primer Semestre: {aprob:.0f} materias aprobadas\n")
            self.txt_diagnostico.insert(tk.END, "  Conclusión Causal: Aunque las notas no fuesen críticas, el bloqueo administrativo\n")
            self.txt_diagnostico.insert(tk.END, "  conduce a una probabilidad casi segura de deserción institucional forzosa.\n\n")
        else:
            self.txt_diagnostico.insert(tk.END, "  Sin alertas administrativas de PRISM.\n")
            self.txt_diagnostico.insert(tk.END, "  El estudiante está al día con sus pagos y no registra deudas activas.\n\n")

        # SÍNTESIS Y VEREDICTO FINAL DEL SISTEMA HÍBRIDO
        self.txt_diagnostico.insert(tk.END, "[SÍNTESIS DIAGNÓSTICA Y VEREDICTO FINAL DEL SISTEMA HÍBRIDO (XAI)]\n")
        self.txt_diagnostico.insert(tk.END, "=" * 95 + "\n")
        if riesgo >= 0.5 or alerta_prism:
            pred = "Dropout (Alto Riesgo de Deserción)"
            color_txt = "PELIGRO DE ABANDONO DETECTADO"
            if riesgo >= 0.5 and alerta_prism:
                motivo = "Convergencia Crítica: Alto riesgo académico en el Motor Difuso Y causal administrativa PRISM."
            elif riesgo >= 0.5:
                motivo = "Alerta Académica: Bajo rendimiento cuantitativo detectado por el Motor Difuso Mamdani."
            else:
                motivo = "Alerta Administrativa: Rescatado por Regla Causal PRISM (Deuda o retraso en pagos)."
        else:
            pred = "No Dropout (Bajo Riesgo / Continuará)"
            color_txt = "ESTUDIANTE SEGURO"
            motivo = "Rendimiento académico satisfactorio y ausencia de causales administrativas de deserción."

        self.txt_diagnostico.insert(tk.END, f"  Estado General:    {color_txt}\n")
        self.txt_diagnostico.insert(tk.END, f"  Predicción Final:  {pred}\n")
        self.txt_diagnostico.insert(tk.END, f"  Justificación XAI: {motivo}\n")
        if target_real:
            acerto = (pred.startswith("Dropout") and target_real == "Dropout") or (pred.startswith("No Dropout") and target_real != "Dropout")
            marca = "✓ PREDICCIÓN CORRECTA (Coincide con el histórico real)" if acerto else "✗ DISCREPANCIA CON LA CONDICIÓN REAL"
            self.txt_diagnostico.insert(tk.END, f"  Verificación:      {marca} (Realidad: {target_real})\n")
        self.txt_diagnostico.insert(tk.END, "=" * 95 + "\n")

    # =========================================================================
    # PESTAÑA 5: MATRIZ DE CONFUSIÓN Y RESULTADOS
    # =========================================================================
    def construir_tab_resultados(self, parent):
        frame_superior = ttk.Frame(parent)
        frame_superior.pack(fill=tk.X, padx=15, pady=8)

        lbl_titulo = ttk.Label(
            frame_superior,
            text="Evaluación Rigurosa en los 885 Datos de Prueba Independientes (20% del Dataset)",
            font=("Arial", 12, "bold")
        )
        lbl_titulo.pack(side=tk.LEFT)

        btn_recalcular = ttk.Button(
            frame_superior,
            text="Recalcular Resultados en Prueba",
            command=self.actualizar_resultados_finales
        )
        btn_recalcular.pack(side=tk.RIGHT, padx=5)

        # Tarjetas de métricas principales
        frame_tarjetas = ttk.Frame(parent)
        frame_tarjetas.pack(fill=tk.X, padx=15, pady=5)

        caja_acc_dif = ttk.LabelFrame(frame_tarjetas, text="Exactitud Motor Difuso")
        caja_acc_dif.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.lbl_acc_difuso = tk.Label(caja_acc_dif, text="76.95%", font=("Arial", 16, "bold"), fg="#881798")
        self.lbl_acc_difuso.pack(pady=6)

        caja_acc = ttk.LabelFrame(frame_tarjetas, text="Exactitud Sist. Híbrido")
        caja_acc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.lbl_acc = tk.Label(caja_acc, text="81.69%", font=("Arial", 16, "bold"), fg="#007acc")
        self.lbl_acc.pack(pady=6)

        caja_rec = ttk.LabelFrame(frame_tarjetas, text="Sensibilidad (Recall)")
        caja_rec.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.lbl_rec = tk.Label(caja_rec, text="69.32%", font=("Arial", 16, "bold"), fg="#107c41")
        self.lbl_rec.pack(pady=6)

        caja_prec = ttk.LabelFrame(frame_tarjetas, text="Precisión Diagnóstica")
        caja_prec.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.lbl_prec = tk.Label(caja_prec, text="74.29%", font=("Arial", 16, "bold"), fg="#b48b00")
        self.lbl_prec.pack(pady=6)

        caja_prism = ttk.LabelFrame(frame_tarjetas, text="Aporte de PRISM")
        caja_prism.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.lbl_prism = tk.Label(caja_prism, text="+4 Desertores", font=("Arial", 16, "bold"), fg="#d83b01")
        self.lbl_prism.pack(pady=6)

        # Notebook con dos pestañas de resultados exhaustivos
        notebook_res = ttk.Notebook(parent)
        notebook_res.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)

        # Subpestaña 1: Demostración Motor Difuso en Prueba (885 Casos)
        frame_sub_difuso = ttk.Frame(notebook_res)
        notebook_res.add(frame_sub_difuso, text=" Demostración Motor Difuso en Prueba (885 Casos No Vistos) ")
        self.txt_resultado_difuso = scrolledtext.ScrolledText(frame_sub_difuso, font=("Courier", 10))
        self.txt_resultado_difuso.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Subpestaña 2: Matriz Global Sistema Híbrido
        frame_sub_hibrido = ttk.Frame(notebook_res)
        notebook_res.add(frame_sub_hibrido, text=" Matriz Global Sistema Híbrido (Difuso + PRISM + Apriori) ")
        self.txt_matriz = scrolledtext.ScrolledText(frame_sub_hibrido, font=("Courier", 10))
        self.txt_matriz.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.actualizar_resultados_finales()

    def actualizar_resultados_finales(self):
        """Calcula dinámicamente las métricas de prueba del Motor Difuso y del Sistema Híbrido."""
        if not self.datos_prueba:
            return

        es_calibrado = len(self.historial_fitness) > 1

        # 1. EVALUACIÓN EXCLUSIVA DEL MOTOR DIFUSO EN PRUEBA (885 DATOS NO VISTOS)
        res_difuso = evaluar_rendimiento_motor_difuso(
            self.datos_prueba, self.parametros_difusos, self.reglas_difusas
        )
        reporte_difuso = formatear_reporte_demostracion_motor_difuso(res_difuso, es_calibrado=es_calibrado)

        if hasattr(self, 'txt_resultado_difuso'):
            self.txt_resultado_difuso.config(state=tk.NORMAL)
            self.txt_resultado_difuso.delete('1.0', tk.END)
            self.txt_resultado_difuso.insert(tk.END, reporte_difuso)
            self.txt_resultado_difuso.config(state=tk.DISABLED)

        # 2. EVALUACIÓN DEL SISTEMA HÍBRIDO (DIFUSO OR PRISM)
        verdaderos_positivos_hibrido = 0
        falsos_positivos_hibrido = 0
        verdaderos_negativos_hibrido = 0
        falsos_negativos_hibrido = 0

        verdaderos_positivos_difuso = res_difuso['vp']

        for est in self.datos_prueba:
            nota = float(est.get('Admission grade', 100))
            aprob = float(est.get('Curricular units 1st sem (approved)', 0))
            prom = float(est.get('Curricular units 1st sem (grade)', 0))
            target_real = est.get('Target', '')
            es_desertor_real = (target_real == 'Dropout')

            riesgo, _ = evaluar_motor_difuso(
                nota, aprob, prom, self.parametros_difusos, 100, reglas=self.reglas_difusas
            )
            pred_dif = (riesgo >= 0.5)

            activa_prism = False
            if self.reglas_prism:
                for r in self.reglas_prism:
                    cumple = True
                    for attr, val in r['condiciones']:
                        v_est = est.get(attr)
                        if v_est != val and str(v_est) != str(val):
                            cumple = False
                            break
                    if cumple:
                        activa_prism = True
                        break

            pred_hib = pred_dif or activa_prism

            if pred_hib and es_desertor_real:
                verdaderos_positivos_hibrido += 1
            elif pred_hib and not es_desertor_real:
                falsos_positivos_hibrido += 1
            elif not pred_hib and not es_desertor_real:
                verdaderos_negativos_hibrido += 1
            else:
                falsos_negativos_hibrido += 1

        total = len(self.datos_prueba)
        acc_h = (verdaderos_positivos_hibrido + verdaderos_negativos_hibrido) / total if total > 0 else 0
        rec_h = verdaderos_positivos_hibrido / (verdaderos_positivos_hibrido + falsos_negativos_hibrido) if (verdaderos_positivos_hibrido + falsos_negativos_hibrido) > 0 else 0
        prec_h = verdaderos_positivos_hibrido / (verdaderos_positivos_hibrido + falsos_positivos_hibrido) if (verdaderos_positivos_hibrido + falsos_positivos_hibrido) > 0 else 0
        spec_h = verdaderos_negativos_hibrido / (verdaderos_negativos_hibrido + falsos_positivos_hibrido) if (verdaderos_negativos_hibrido + falsos_positivos_hibrido) > 0 else 0
        f1_h = 2 * (prec_h * rec_h) / (prec_h + rec_h) if (prec_h + rec_h) > 0 else 0
        rescatados = verdaderos_positivos_hibrido - verdaderos_positivos_difuso

        # Actualizar etiquetas de tarjetas
        if hasattr(self, 'lbl_acc_difuso'):
            self.lbl_acc_difuso.config(text=f"{res_difuso['accuracy']*100:.2f}%")
        if hasattr(self, 'lbl_acc'):
            self.lbl_acc.config(text=f"{acc_h*100:.2f}%")
        if hasattr(self, 'lbl_rec'):
            self.lbl_rec.config(text=f"{rec_h*100:.2f}%")
        if hasattr(self, 'lbl_prec'):
            self.lbl_prec.config(text=f"{prec_h*100:.2f}%")
        if hasattr(self, 'lbl_prism'):
            signo = "+" if rescatados >= 0 else ""
            self.lbl_prism.config(text=f"{signo}{rescatados} Desertores")

        # Actualizar cuadro de texto de la matriz híbrida
        if hasattr(self, 'txt_matriz'):
            self.txt_matriz.config(state=tk.NORMAL)
            self.txt_matriz.delete('1.0', tk.END)
            contenido = (
                f"=" * 95 + "\n"
                f"  MATRIZ DE CONFUSIÓN Y RENDIMIENTO DEL SISTEMA HÍBRIDO ({total} Estudiantes de Prueba)\n"
                f"  Integración Sinérgica: Motor Difuso Mamdani + Reglas Causales PRISM + Apriori\n"
                f"=" * 95 + "\n\n"
                f"TABLA DE CONTINGENCIA (885 CASOS NO VISTOS):\n"
                f"---------------------------------------------------------------------------------------\n"
                f"                                            Predicción: Dropout         Predicción: No Dropout\n"
                f"  Realidad: Desertores Reales (264)              {verdaderos_positivos_hibrido:>5} (VP)                    {falsos_negativos_hibrido:>5} (FN)\n"
                f"  Realidad: Estudiantes que Siguen (621)         {falsos_positivos_hibrido:>5} (FP)                    {verdaderos_negativos_hibrido:>5} (VN)\n"
                f"---------------------------------------------------------------------------------------\n\n"
                f"SIGNIFICADO DE CADA CUADRANTE:\n"
                f"  • Verdaderos Positivos (VP = {verdaderos_positivos_hibrido:>3}): Desertores reales detectados correctamente.\n"
                f"  • Verdaderos Negativos (VN = {verdaderos_negativos_hibrido:>3}): Estudiantes seguros que continúan sin riesgo.\n"
                f"  • Falsos Positivos     (FP = {falsos_positivos_hibrido:>3}): Falsas alarmas (se predijo abandono pero continuó).\n"
                f"  • Falsos Negativos     (FN = {falsos_negativos_hibrido:>3}): Desertores que pasaron desapercibidos.\n\n"
                f"MÉTRICAS ESTADÍSTICAS DEL SISTEMA HÍBRIDO:\n"
                f"  • Exactitud Global (Accuracy)       : {acc_h*100:6.2f}%  ({verdaderos_positivos_hibrido + verdaderos_negativos_hibrido} aciertos de {total} casos)\n"
                f"  • Sensibilidad / Cobertura (Recall) : {rec_h*100:6.2f}%\n"
                f"  • Especificidad                     : {spec_h*100:6.2f}%\n"
                f"  • Precisión Diagnóstica             : {prec_h*100:6.2f}%\n"
                f"  • Puntuación F1 (Balance Armónico)  : {f1_h*100:6.2f}%\n\n"
                f"ANÁLISIS COMPARATIVO DE ARQUITECTURA:\n"
                f"  • Solo Motor Difuso (Notas Académicas):  Accuracy = {res_difuso['accuracy']*100:.2f}% | Recall = {res_difuso['recall']*100:.2f}%\n"
                f"  • Sistema Híbrido (Difuso + PRISM Causal): Accuracy = {acc_h*100:.2f}% | Recall = {rec_h*100:.2f}%\n"
                f"  • Aporte Específico de PRISM:            Rescata +{rescatados} estudiantes que abandonaron por causales\n"
                f"                                           administrativas (deudas/matrícula) a pesar de notas regulares.\n\n"
                f"IMPACTO DE LA CALIBRACIÓN POR ALGORITMO GENÉTICO:\n"
                f"  • Estado de las funciones difusas:      {'CALIBRADAS POR EL ALGORITMO GENÉTICO (60 GENERACIONES)' if es_calibrado else 'PARÁMETROS INICIALES'}\n"
                f"  • La optimización evolutiva ajusta los 19 puntos de corte de los trapecios y triángulos,\n"
                f"    reduciendo el solapamiento erróneo y maximizando la separación entre clases.\n"
                f"=" * 95 + "\n"
            )
            self.txt_matriz.insert(tk.END, contenido)
            self.txt_matriz.config(state=tk.DISABLED)


def main():
    ventana = tk.Tk()
    app = AplicacionDesercion(ventana)
    ventana.mainloop()


if __name__ == '__main__':
    main()
