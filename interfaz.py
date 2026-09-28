
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random
import os

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
    ejecutar_prism,
    traducir_atributo
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

DICCIONARIO_CARRERAS = {
    9991: "9991 - Gestión / Administración (Nocturno)",
    9500: "9500 - Enfermería",
    9147: "9147 - Gestión / Administración (Diurno)",
    9238: "9238 - Servicio Social (Diurno)",
    9085: "9085 - Enfermería Veterinaria",
    9119: "9119 - Ingeniería Informática",
    9254: "9254 - Turismo",
    9070: "9070 - Diseño de Comunicación",
    9670: "9670 - Gestión de Publicidad y Marketing",
    9773: "9773 - Periodismo y Comunicación",
    9003: "9003 - Agronomía",
    9853: "9853 - Educación Básica",
    9556: "9556 - Higiene Bucodental",
    8014: "8014 - Servicio Social (Nocturno)",
    171: "171 - Diseño de Animación y Multimedia",
    9130: "9130 - Equinocultura",
    33: "33 - Tecnologías de Producción de Biocombustibles"
}


class AplicacionDesercion:
    def __init__(self, ventana_principal):
        self.ventana = ventana_principal
        self.ventana.title("Proyecto Primer Parcial - Sistema Híbrido de IA (Deserción Estudiantil)")
        self.ventana.geometry("1080x750")
        self.ventana.minsize(920, 680)

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

        self.estilo = ttk.Style()
        self.estilo.theme_use('clam')

        self.cargar_datos_iniciales()

        self.crear_barra_superior_maestra()

        self.crear_pestanas()

    def cargar_datos_iniciales(self):
        ruta_csv = 'data.csv'
        if not os.path.exists(ruta_csv):
            ruta_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv')

        try:
            self.datos_completos = cargar_dataset(ruta_csv)
            self.datos_entrenamiento, self.datos_prueba = partir_dataset(self.datos_completos, 0.80)
            self.reglas_difusas, _, self.reporte_difuso = inducir_reglas_difusas_con_prism(self.datos_entrenamiento, min_cobertura=15)
            self.reglas_prism, _ = ejecutar_prism(self.datos_entrenamiento, clase_objetivo='Dropout', semilla=42)
        except Exception as error:
            print("Aviso al cargar datos:", error)

    def crear_barra_superior_maestra(self):
        frame_maestro = tk.Frame(self.ventana, bg="#f3f3f3", relief=tk.RIDGE, bd=1, padx=10, pady=8)
        frame_maestro.pack(fill=tk.X, padx=10, pady=(10, 5))

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

        self.lbl_estado_global = tk.Label(
            frame_maestro,
            text="Estado: Listo. Presiona 'EJECUTAR TODO EL SISTEMA' para calcular todos los módulos a la vez.",
            font=("Arial", 9, "italic"),
            bg="#f3f3f3",
            fg="#333333"
        )
        self.lbl_estado_global.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.barra_progreso = ttk.Progressbar(frame_maestro, orient=tk.HORIZONTAL, length=220, mode='determinate')
        self.barra_progreso.pack(side=tk.RIGHT, padx=5)

    def crear_pestanas(self):
        panel_pestanas = ttk.Notebook(self.ventana)
        panel_pestanas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        tab_reglas = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_reglas, text=" 1. Reglas (PRISM y Apriori) ")
        self.construir_tab_reglas(tab_reglas)

        tab_genetico = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_genetico, text=" 2. Algoritmo Genético ")
        self.construir_tab_genetico(tab_genetico)

        tab_funciones = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_funciones, text=" 3. Funciones Difusas ")
        self.construir_tab_funciones(tab_funciones)

        tab_diagnostico = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_diagnostico, text=" 4. Diagnóstico Explicable ")
        self.construir_tab_diagnostico(tab_diagnostico)

        tab_resultados = ttk.Frame(panel_pestanas)
        panel_pestanas.add(tab_resultados, text=" 5. Resultados Finales ")
        self.construir_tab_resultados(tab_resultados)

    def accion_ejecutar_todo_el_sistema(self):
        if not self.datos_entrenamiento:
            messagebox.showerror("Error", "No se encontraron datos en 'data.csv'.")
            return

        self.btn_ejecutar_todo.config(state=tk.DISABLED, bg="#888888")
        self.barra_progreso['value'] = 5
        self.lbl_estado_global.config(text="Paso 1/5: Induciendo reglas difusas y causales PRISM...", fg="#005a9e")
        self.ventana.update()

        self.mostrar_reglas_difusas()
        self.accion_calcular_prism_causal(mostrar_aviso=False)

        self.barra_progreso['value'] = 25
        self.lbl_estado_global.config(text="Paso 2/5: Minando patrones y reglas de asociación con Apriori...", fg="#005a9e")
        self.ventana.update()
        self.accion_calcular_apriori(mostrar_aviso=False)

        self.barra_progreso['value'] = 50
        self.lbl_estado_global.config(text="Paso 3/5: Calibrando funciones difusas con Algoritmo Genético...", fg="#005a9e")
        self.ventana.update()
        self.accion_ejecutar_genetico_completo(mostrar_alerta=False)

        self.barra_progreso['value'] = 80
        self.lbl_estado_global.config(text="Paso 4/5: Dibujando figuras de pertenencia y curvas de evolución...", fg="#005a9e")
        self.ventana.update()
        self.dibujar_funciones_pertenencia()

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

    def construir_tab_genetico(self, parent):
        frame_superior = ttk.Frame(parent)
        frame_superior.pack(fill=tk.X, padx=10, pady=5)

        self.btn_ejecutar_genetico = ttk.Button(
            frame_superior,
            text="▶ 1. Ejecutar Algoritmo Genético (60 Generaciones Reales)",
            command=self.accion_ejecutar_genetico_completo
        )
        self.btn_ejecutar_genetico.pack(side=tk.LEFT, padx=(0, 10))

        lbl_selector = ttk.Label(frame_superior, text="Inspeccionar Generación:", font=("Arial", 9, "bold"))
        lbl_selector.pack(side=tk.LEFT, padx=(5, 5))

        self.combo_generaciones = ttk.Combobox(frame_superior, state="readonly", width=42)
        self.combo_generaciones['values'] = ["Presiona 'Ejecutar Algoritmo Genético' primero"]
        self.combo_generaciones.current(0)
        self.combo_generaciones.pack(side=tk.LEFT, padx=5)
        self.combo_generaciones.bind("<<ComboboxSelected>>", self.accion_cambiar_generacion_seleccionada)

        btn_ver_detalle = ttk.Button(
            frame_superior,
            text="🔍 Ver Detalle",
            command=self.accion_cambiar_generacion_seleccionada
        )
        btn_ver_detalle.pack(side=tk.LEFT, padx=5)

        btn_ver_todo = ttk.Button(
            frame_superior,
            text="📋 Ver Todas las 60 Generaciones (Continuo)",
            command=self.accion_ver_todas_las_generaciones
        )
        btn_ver_todo.pack(side=tk.LEFT, padx=5)

        btn_guardar_reporte = ttk.Button(
            frame_superior,
            text="💾 Guardar Reporte (.txt)",
            command=self.accion_guardar_reporte_genetico
        )
        btn_guardar_reporte.pack(side=tk.LEFT, padx=5)

        btn_ver_cortes = ttk.Button(
            frame_superior,
            text="🔬 Puntos de Corte Calibrados (19 Genes)",
            command=self.accion_ver_puntos_de_corte_calibrados
        )
        btn_ver_cortes.pack(side=tk.LEFT, padx=5)

        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

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

        frame_der = ttk.Frame(paned)
        paned.add(frame_der, weight=2)

        lbl_graf = ttk.Label(frame_der, text="Evolución del Fitness por Generación (Curva de Aprendizaje):", font=("Arial", 10, "bold"))
        lbl_graf.pack(anchor=tk.W, padx=5, pady=2)

        self.canvas_genetico = tk.Canvas(frame_der, bg="white", highlightthickness=1, highlightbackground="#cccccc", height=280)
        self.canvas_genetico.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas_genetico.bind("<Configure>", lambda e: self.dibujar_grafica_fitness())

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

            fit_ini = self.registro_generaciones[0].get('mejor_fitness_generacion', self.registro_generaciones[0].get('mejor_fitness_gen', 0.0)) * 100
            fit_fin = mejor_fitness * 100
            self.lbl_resultado_fitness.config(
                text=f"\nFitness Inicial: {fit_ini:.2f}%  ->  Fitness Final: {fit_fin:.2f}%\nGanancia Evolutiva: +{fit_fin - fit_ini:.2f}%"
            )

            self.dibujar_grafica_fitness()
            self.dibujar_funciones_pertenencia()

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

        frame_graficos = ttk.Frame(notebook_funciones)
        notebook_funciones.add(frame_graficos, text=" Curvas de Pertenencia Calibradas ")

        frame_g1 = ttk.LabelFrame(frame_graficos, text="Variable 1: Nota de Admisión (0 a 200 puntos)")
        frame_g1.pack(fill=tk.BOTH, expand=True, pady=3)
        self.canvas_g1 = tk.Canvas(frame_g1, bg="white", height=140)
        self.canvas_g1.pack(fill=tk.BOTH, expand=True, padx=5, pady=4)
        self.canvas_g1.bind("<Configure>", lambda e: self.dibujar_funciones_pertenencia())

        frame_g2 = ttk.LabelFrame(frame_graficos, text="Variable 2: Materias Aprobadas 1er Semestre (0 a 26 materias)")
        frame_g2.pack(fill=tk.BOTH, expand=True, pady=3)
        self.canvas_g2 = tk.Canvas(frame_g2, bg="white", height=140)
        self.canvas_g2.pack(fill=tk.BOTH, expand=True, padx=5, pady=4)
        self.canvas_g2.bind("<Configure>", lambda e: self.dibujar_funciones_pertenencia())

        frame_g3 = ttk.LabelFrame(frame_graficos, text="Variable 3: Promedio de Calificaciones 1er Semestre (0 a 20 puntos)")
        frame_g3.pack(fill=tk.BOTH, expand=True, pady=3)
        self.canvas_g3 = tk.Canvas(frame_g3, bg="white", height=140)
        self.canvas_g3.pack(fill=tk.BOTH, expand=True, padx=5, pady=4)
        self.canvas_g3.bind("<Configure>", lambda e: self.dibujar_funciones_pertenencia())

        frame_cortes = ttk.Frame(notebook_funciones)
        notebook_funciones.add(frame_cortes, text=" Puntos de Corte Calibrados por el AG (19 Genes) ")
        self.txt_cortes_difusos = scrolledtext.ScrolledText(frame_cortes, font=("Courier", 10))
        self.txt_cortes_difusos.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.actualizar_texto_puntos_de_corte()

        self.ventana.after(100, self.dibujar_funciones_pertenencia)

    def cargar_guia_teorica_difusa(self):
        self.cargar_ejemplo_clase('Dropout')

    def mostrar_demostracion_caso_real(self, clase_objetivo='Dropout'):
        self.cargar_ejemplo_clase(clase_objetivo)

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

    def dibujar_variable_difusa_completa(self, canvas, x_min, x_max, titulo_eje, conjuntos):
        canvas.delete("all")
        ancho = canvas.winfo_width()
        alto = canvas.winfo_height()
        if ancho < 60 or alto < 50:
            return

        margen_izq = 55
        margen_der = 45
        margen_arr = 26
        margen_abj = 38
        ancho_util = ancho - margen_izq - margen_der
        alto_util = alto - margen_arr - margen_abj
        if ancho_util <= 10 or alto_util <= 10:
            return

        def escala_x(val):
            return margen_izq + ((val - x_min) / float(x_max - x_min)) * ancho_util

        def escala_y(mu):
            return (alto - margen_abj) - mu * alto_util

        canvas.create_rectangle(0, 0, ancho, alto, fill="#ffffff", outline="")

        canvas.create_line(margen_izq, escala_y(0.5), ancho - margen_der, escala_y(0.5), fill="#f1f5f9", dash=(2, 4))
        canvas.create_line(margen_izq, escala_y(1.0), ancho - margen_der, escala_y(1.0), fill="#e2e8f0", dash=(2, 4))

        canvas.create_line(margen_izq, escala_y(0.0), ancho - margen_der, escala_y(0.0), fill="#475569", width=1.5)
        canvas.create_line(margen_izq, margen_arr, margen_izq, escala_y(0.0), fill="#475569", width=1.5)

        canvas.create_text(margen_izq - 12, escala_y(1.0), text="1.0", font=("Arial", 7), fill="#64748b")
        canvas.create_text(margen_izq - 12, escala_y(0.5), text="0.5", font=("Arial", 7), fill="#94a3b8")
        canvas.create_text(margen_izq - 12, escala_y(0.0), text="0.0", font=("Arial", 7), fill="#64748b")
        canvas.create_text(margen_izq - 32, margen_arr + alto_util // 2, text="μ", font=("Arial", 8, "bold"), fill="#1e293b")

        canvas.create_text(margen_izq + ancho_util // 2, alto - 8, text=titulo_eje, font=("Arial", 8, "italic"), fill="#334155")

        canvas.create_text(margen_izq, escala_y(0.0) + 11, text=str(x_min), font=("Arial", 8, "bold"), fill="#1e293b")
        canvas.create_text(ancho - margen_der, escala_y(0.0) + 11, text=str(x_max), font=("Arial", 8, "bold"), fill="#1e293b")

        puntos_corte_raw = []
        resumen_texto_partes = []

        for puntos, tipo, color, etiqueta in conjuntos:
            if tipo == 'trapecio':
                a, b, c, d = puntos
                coords = [
                    (escala_x(a), escala_y(0.0)),
                    (escala_x(b), escala_y(1.0)),
                    (escala_x(c), escala_y(1.0)),
                    (escala_x(d), escala_y(0.0))
                ]
                x_etiqueta = escala_x((b + c) / 2.0)
                if b > x_min + 0.001:
                    puntos_corte_raw.append((b, color, 1.0, f"{etiqueta}_b"))
                if c < x_max - 0.001:
                    puntos_corte_raw.append((c, color, 1.0, f"{etiqueta}_c"))
                if d < x_max - 0.001 and d > x_min + 0.001:
                    puntos_corte_raw.append((d, color, 0.0, f"{etiqueta}_d"))
                if a > x_min + 0.001:
                    puntos_corte_raw.append((a, color, 0.0, f"{etiqueta}_a"))

                fmt_vals = f"[{puntos[0]:.1f}, {puntos[1]:.1f}, {puntos[2]:.1f}, {puntos[3]:.1f}]"
                resumen_texto_partes.append(f"{etiqueta}: {fmt_vals}")

            else:  # triangulo
                a, b, c = puntos
                coords = [
                    (escala_x(a), escala_y(0.0)),
                    (escala_x(b), escala_y(1.0)),
                    (escala_x(c), escala_y(0.0))
                ]
                x_etiqueta = escala_x(b)
                if a > x_min + 0.001:
                    puntos_corte_raw.append((a, color, 0.0, f"{etiqueta}_a"))
                if x_min + 0.001 < b < x_max - 0.001:
                    puntos_corte_raw.append((b, color, 1.0, f"{etiqueta}_b"))
                if c < x_max - 0.001:
                    puntos_corte_raw.append((c, color, 0.0, f"{etiqueta}_c"))

                fmt_vals = f"[{puntos[0]:.1f}, {puntos[1]:.1f}, {puntos[2]:.1f}]"
                resumen_texto_partes.append(f"{etiqueta}: {fmt_vals}")

            for i in range(len(coords) - 1):
                canvas.create_line(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1], fill=color, width=2.5)

            canvas.create_text(x_etiqueta, escala_y(1.0) - 10, text=etiqueta, font=("Arial", 9, "bold"), fill=color)

            for pt_x_val, pt_y_mu in [(puntos[i], 1.0 if (tipo=='trapecio' and i in (1,2)) or (tipo=='triangulo' and i==1) else 0.0)
                                      for i in range(len(puntos))]:
                if x_min + 0.001 < pt_x_val < x_max - 0.001:
                    px = escala_x(pt_x_val)
                    py = escala_y(pt_y_mu)
                    canvas.create_line(px, py, px, escala_y(0.0), fill=color, dash=(2, 3), width=1.2)
                    canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill=color, outline="#ffffff", width=1.5)
                    if pt_y_mu == 1.0:
                        canvas.create_text(px, py - 9, text=f"{pt_x_val:.1f}", font=("Arial", 7, "bold"), fill=color)

        validos = [p for p in puntos_corte_raw if (x_min + 0.001 < p[0] < x_max - 0.001)]
        validos.sort(key=lambda p: p[0])

        cortes_unicos = []
        for p in validos:
            if not cortes_unicos or abs(p[0] - cortes_unicos[-1]['val']) > 0.15:
                cortes_unicos.append({'val': p[0], 'colores': [p[1]], 'roles': [p[3]]})
            else:
                if p[1] not in cortes_unicos[-1]['colores']:
                    cortes_unicos[-1]['colores'].append(p[1])
                cortes_unicos[-1]['roles'].append(p[3])

        prev_x_pix = -999
        prev_level = 0
        for corte in cortes_unicos:
            val = corte['val']
            x_pix = escala_x(val)
            color_corte = corte['colores'][0] if len(corte['colores']) == 1 else "#334155"

            if abs(x_pix - prev_x_pix) < 36:
                level = 1 if prev_level == 0 else 0
            else:
                level = 0
            prev_x_pix = x_pix
            prev_level = level

            y_base = escala_y(0.0)
            if level == 0:
                canvas.create_line(x_pix, y_base - 2, x_pix, y_base + 5, fill=color_corte, width=1.5)
                y_num = y_base + 11
            else:
                canvas.create_line(x_pix, y_base - 2, x_pix, y_base + 16, fill=color_corte, width=1.2, dash=(2, 2))
                y_num = y_base + 22

            texto_corte = f"{int(val)}" if (val == int(val) and x_max <= 26) else f"{val:.1f}"
            canvas.create_text(x_pix, y_num, text=texto_corte, font=("Arial", 8, "bold"), fill=color_corte)

        texto_resumen = "  |  ".join(resumen_texto_partes)
        canvas.create_text(ancho - margen_der, 11, anchor=tk.E, text=texto_resumen, font=("Courier", 8, "bold"), fill="#475569")

    def dibujar_funciones_pertenencia(self):
        p = self.parametros_difusos

        conjuntos_v1 = [
            (p['nota_adm_baja'], 'trapecio', "#007acc", "Baja"),
            (p['nota_adm_media'], 'triangulo', "#107c41", "Media"),
            (p['nota_adm_alta'], 'trapecio', "#d83b01", "Alta")
        ]
        self.dibujar_variable_difusa_completa(self.canvas_g1, 0, 200, "Puntaje de Admisión", conjuntos_v1)

        conjuntos_v2 = [
            (p['aprobadas_critica'], 'trapecio', "#d83b01", "Crítica"),
            (p['aprobadas_regular'], 'triangulo', "#b48b00", "Regular"),
            (p['aprobadas_completa'], 'trapecio', "#107c41", "Completa")
        ]
        self.dibujar_variable_difusa_completa(self.canvas_g2, 0, 26, "Número de Materias Aprobadas (1er Semestre)", conjuntos_v2)

        conjuntos_v3 = [
            (p['nota_sem_deficiente'], 'trapecio', "#d83b01", "Deficiente"),
            (p['nota_sem_aceptable'], 'triangulo', "#b48b00", "Aceptable"),
            (p['nota_sem_sobresaliente'], 'trapecio', "#107c41", "Sobresaliente")
        ]
        self.dibujar_variable_difusa_completa(self.canvas_g3, 0, 20, "Promedio Semestral (1er Semestre)", conjuntos_v3)

        self.actualizar_texto_puntos_de_corte()

    def actualizar_texto_puntos_de_corte(self):
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

    def dibujar_proceso_difuso_4_paneles(self, canvas, nota_admision, materias_aprobadas, promedio_notas, parametros, fuerzas, riesgo_calculado, target_real=None, det=None):
        ancho = canvas.winfo_width()
        alto = canvas.winfo_height()
        if ancho <= 1:
            try:
                ancho = int(canvas.cget("width"))
                alto = int(canvas.cget("height"))
            except Exception:
                pass
        if ancho < 120 or alto < 60:
            return

        canvas.delete("all")
        canvas.create_rectangle(0, 0, ancho, alto, fill="#ffffff", outline="")

        m_izq = 8
        m_der = 8
        gap = 20
        w_total_util = ancho - m_izq - m_der - (3 * gap)
        if w_total_util < 120:
            return
        w_panel = w_total_util / 4.0

        pad_l = 24  # Para etiquetas 0.0, 1.0 y cortes mu en eje Y
        pad_r = 6   # Margen derecho
        pad_t = 22  # Para títulos y marcadores superiores X1, X2, X3, Centroid
        pad_b = 26  # Para ticks del eje X y cajas de valor inferiores

        y_top = pad_t
        y_bottom = alto - pad_b
        h_box = y_bottom - y_top
        if h_box < 25:
            return

        def get_panel_box(idx):
            xp0 = m_izq + idx * (w_panel + gap)
            xp1 = xp0 + w_panel
            bx0 = xp0 + pad_l
            bx1 = xp1 - pad_r
            return xp0, xp1, bx0, bx1

        for i in range(3):
            _, xp1, _, _ = get_panel_box(i)
            next_xp0, _, _, _ = get_panel_box(i + 1)
            y_mid = (y_top + y_bottom) / 2
            canvas.create_line(
                xp1 + 2, y_mid, next_xp0 - 2, y_mid,
                fill="#d32f2f", width=2.5, arrow=tk.LAST, arrowshape=(6, 8, 3)
            )

        def sombreado_trapecio(px_fn, py_fn, a, b, c, d, mu_corte, x_min_val, x_max_val):
            if mu_corte <= 0.005:
                return
            x_s = max(x_min_val, min(a, b))
            x_e = min(x_max_val, max(c, d))
            if x_e <= x_s:
                return
            pasos = 30
            pts = [px_fn(x_s), py_fn(0.0)]
            for step in range(pasos + 1):
                xv = x_s + (x_e - x_s) * (step / pasos)
                mv = pertenencia_trapezoidal(xv, a, b, c, d)
                pts.extend([px_fn(xv), py_fn(min(mv, mu_corte))])
            pts.extend([px_fn(x_e), py_fn(0.0)])
            canvas.create_polygon(pts, fill="#ffeb3b", outline="")

        def sombreado_triangulo(px_fn, py_fn, a, b, c, mu_corte, x_min_val, x_max_val):
            if mu_corte <= 0.005:
                return
            x_s = max(x_min_val, a)
            x_e = min(x_max_val, c)
            if x_e <= x_s:
                return
            pasos = 30
            pts = [px_fn(x_s), py_fn(0.0)]
            for step in range(pasos + 1):
                xv = x_s + (x_e - x_s) * (step / pasos)
                mv = pertenencia_triangular(xv, a, b, c)
                pts.extend([px_fn(xv), py_fn(min(mv, mu_corte))])
            pts.extend([px_fn(x_e), py_fn(0.0)])
            canvas.create_polygon(pts, fill="#ffeb3b", outline="")

        def dibujar_flechas_mu(bx0, bx1, py_fn, lista_mus):
            mus = sorted([m for m in lista_mus if m > 0.01])
            prev_y = None
            for mu_val in mus:
                y_mu = py_fn(mu_val)
                canvas.create_line(bx0, y_mu, bx1, y_mu, fill="#c2185b", width=1.3, arrow=tk.LAST, arrowshape=(5, 7, 3))
                y_txt = y_mu
                if prev_y is not None and abs(y_txt - prev_y) < 10:
                    y_txt = prev_y - 10
                canvas.create_text(bx0 - 10, y_txt, text=f"{mu_val:.2f}", font=("Arial", 7, "bold"), fill="#c2185b")
                prev_y = y_txt

        def dibujar_caja_input(x_pix, y_bot, texto_val):
            bw = max(26, len(texto_val) * 7 + 4)
            y_box = y_bot + 12
            canvas.create_rectangle(x_pix - bw / 2, y_box - 6, x_pix + bw / 2, y_box + 6, outline="#c2185b", fill="#ffffff", width=1.5)
            canvas.create_text(x_pix, y_box, text=texto_val, font=("Arial", 7, "bold"), fill="#c2185b")

        def py_coord(mu_val):
            c = max(0.0, min(1.0, mu_val))
            return y_bottom - c * h_box

        _, _, bx0_0, bx1_0 = get_panel_box(0)
        bw_0 = bx1_0 - bx0_0
        def px0(x):
            c = max(0.0, min(200.0, x))
            return bx0_0 + (c / 200.0) * bw_0

        p_adm_b = parametros.get('nota_adm_baja', [0, 0, 100, 140])
        p_adm_m = parametros.get('nota_adm_media', [110, 135, 160])
        p_adm_a = parametros.get('nota_adm_alta', [145, 170, 200, 200])

        mu_adm_b = pertenencia_trapezoidal(nota_admision, p_adm_b[0], p_adm_b[1], p_adm_b[2], p_adm_b[3])
        mu_adm_m = pertenencia_triangular(nota_admision, p_adm_m[0], p_adm_m[1], p_adm_m[2])
        mu_adm_a = pertenencia_trapezoidal(nota_admision, p_adm_a[0], p_adm_a[1], p_adm_a[2], p_adm_a[3])

        sombreado_trapecio(px0, py_coord, p_adm_b[0], p_adm_b[1], p_adm_b[2], p_adm_b[3], mu_adm_b, 0, 200)
        sombreado_triangulo(px0, py_coord, p_adm_m[0], p_adm_m[1], p_adm_m[2], mu_adm_m, 0, 200)
        sombreado_trapecio(px0, py_coord, p_adm_a[0], p_adm_a[1], p_adm_a[2], p_adm_a[3], mu_adm_a, 0, 200)

        canvas.create_line(px0(0), py_coord(1.0), px0(p_adm_b[2]), py_coord(1.0), px0(p_adm_b[3]), py_coord(0.0), fill="#0d47a1", width=1.6)
        canvas.create_line(px0(p_adm_m[0]), py_coord(0.0), px0(p_adm_m[1]), py_coord(1.0), px0(p_adm_m[2]), py_coord(0.0), fill="#ff8f00", width=1.6)
        canvas.create_line(px0(p_adm_a[0]), py_coord(0.0), px0(p_adm_a[1]), py_coord(1.0), px0(200), py_coord(1.0), fill="#1b5e20", width=1.6)

        canvas.create_rectangle(bx0_0, y_top, bx1_0, y_bottom, outline="#000000", width=1.5)
        canvas.create_text((bx0_0 + bx1_0) / 2, y_top - 11, text="1. Input: Admisión", font=("Arial", 8, "bold"), fill="#000000")
        canvas.create_text(bx0_0 - 9, py_coord(0.0), text="0.0", font=("Arial", 7), fill="#000000")
        canvas.create_text(bx0_0 - 9, py_coord(1.0), text="1.0", font=("Arial", 7), fill="#000000")
        canvas.create_text(px0(0), y_bottom + 8, text="0", font=("Arial", 7), fill="#000000")
        canvas.create_text(px0(200), y_bottom + 8, text="200", font=("Arial", 7), fill="#000000")

        x1_pix = px0(nota_admision)
        canvas.create_line(x1_pix, y_top, x1_pix, y_bottom, fill="#c2185b", dash=(2, 2), width=1.4)
        canvas.create_text(x1_pix, y_top - 6, text="X1", font=("Arial", 7, "bold"), fill="#c2185b")
        dibujar_flechas_mu(bx0_0, bx1_0, py_coord, [mu_adm_b, mu_adm_m, mu_adm_a])
        dibujar_caja_input(x1_pix, y_bottom, f"{nota_admision:.1f}")

        _, _, bx0_1, bx1_1 = get_panel_box(1)
        bw_1 = bx1_1 - bx0_1
        def px1(x):
            c = max(0.0, min(20.0, x))
            return bx0_1 + (c / 20.0) * bw_1

        p_apr_c = parametros.get('aprobadas_critica', [0, 0, 2, 4])
        p_apr_r = parametros.get('aprobadas_regular', [2, 4, 6])
        p_apr_a = parametros.get('aprobadas_completa', [5, 6, 26, 26])

        mu_apr_c = pertenencia_trapezoidal(materias_aprobadas, p_apr_c[0], p_apr_c[1], p_apr_c[2], p_apr_c[3])
        mu_apr_r = pertenencia_triangular(materias_aprobadas, p_apr_r[0], p_apr_r[1], p_apr_r[2])
        mu_apr_a = pertenencia_trapezoidal(materias_aprobadas, p_apr_a[0], p_apr_a[1], p_apr_a[2], p_apr_a[3])

        sombreado_trapecio(px1, py_coord, p_apr_c[0], p_apr_c[1], p_apr_c[2], p_apr_c[3], mu_apr_c, 0, 20)
        sombreado_triangulo(px1, py_coord, p_apr_r[0], p_apr_r[1], p_apr_r[2], mu_apr_r, 0, 20)
        sombreado_trapecio(px1, py_coord, p_apr_a[0], p_apr_a[1], p_apr_a[2], p_apr_a[3], mu_apr_a, 0, 20)

        canvas.create_line(px1(0), py_coord(1.0), px1(p_apr_c[2]), py_coord(1.0), px1(p_apr_c[3]), py_coord(0.0), fill="#c62828", width=1.6)
        canvas.create_line(px1(p_apr_r[0]), py_coord(0.0), px1(p_apr_r[1]), py_coord(1.0), px1(p_apr_r[2]), py_coord(0.0), fill="#ff8f00", width=1.6)
        canvas.create_line(px1(p_apr_a[0]), py_coord(0.0), px1(p_apr_a[1]), py_coord(1.0), px1(20), py_coord(1.0), fill="#1b5e20", width=1.6)

        canvas.create_rectangle(bx0_1, y_top, bx1_1, y_bottom, outline="#000000", width=1.5)
        canvas.create_text((bx0_1 + bx1_1) / 2, y_top - 11, text="2. Input: Aprobadas", font=("Arial", 8, "bold"), fill="#000000")
        canvas.create_text(bx0_1 - 9, py_coord(0.0), text="0.0", font=("Arial", 7), fill="#000000")
        canvas.create_text(bx0_1 - 9, py_coord(1.0), text="1.0", font=("Arial", 7), fill="#000000")
        canvas.create_text(px1(0), y_bottom + 8, text="0", font=("Arial", 7), fill="#000000")
        canvas.create_text(px1(20), y_bottom + 8, text="20", font=("Arial", 7), fill="#000000")

        x2_pix = px1(materias_aprobadas)
        canvas.create_line(x2_pix, y_top, x2_pix, y_bottom, fill="#c2185b", dash=(2, 2), width=1.4)
        canvas.create_text(x2_pix, y_top - 6, text="X2", font=("Arial", 7, "bold"), fill="#c2185b")
        dibujar_flechas_mu(bx0_1, bx1_1, py_coord, [mu_apr_c, mu_apr_r, mu_apr_a])
        texto_aprobadas = f"{materias_aprobadas:.1f}" if materias_aprobadas != int(materias_aprobadas) else f"{int(materias_aprobadas)}"
        dibujar_caja_input(x2_pix, y_bottom, texto_aprobadas)

        _, _, bx0_2, bx1_2 = get_panel_box(2)
        bw_2 = bx1_2 - bx0_2
        def px2(x):
            c = max(0.0, min(20.0, x))
            return bx0_2 + (c / 20.0) * bw_2

        p_prom_d = parametros.get('nota_sem_deficiente', [0, 0, 8, 11])
        p_prom_a = parametros.get('nota_sem_aceptable', [9.5, 12.5, 15.5])
        p_prom_s = parametros.get('nota_sem_sobresaliente', [14.5, 17.0, 20.0, 20.0])

        mu_prom_d = pertenencia_trapezoidal(promedio_notas, p_prom_d[0], p_prom_d[1], p_prom_d[2], p_prom_d[3])
        mu_prom_a = pertenencia_triangular(promedio_notas, p_prom_a[0], p_prom_a[1], p_prom_a[2])
        mu_prom_s = pertenencia_trapezoidal(promedio_notas, p_prom_s[0], p_prom_s[1], p_prom_s[2], p_prom_s[3])

        sombreado_trapecio(px2, py_coord, p_prom_d[0], p_prom_d[1], p_prom_d[2], p_prom_d[3], mu_prom_d, 0, 20)
        sombreado_triangulo(px2, py_coord, p_prom_a[0], p_prom_a[1], p_prom_a[2], mu_prom_a, 0, 20)
        sombreado_trapecio(px2, py_coord, p_prom_s[0], p_prom_s[1], p_prom_s[2], p_prom_s[3], mu_prom_s, 0, 20)

        canvas.create_line(px2(0), py_coord(1.0), px2(p_prom_d[2]), py_coord(1.0), px2(p_prom_d[3]), py_coord(0.0), fill="#c62828", width=1.6)
        canvas.create_line(px2(p_prom_a[0]), py_coord(0.0), px2(p_prom_a[1]), py_coord(1.0), px2(p_prom_a[2]), py_coord(0.0), fill="#ff8f00", width=1.6)
        canvas.create_line(px2(p_prom_s[0]), py_coord(0.0), px2(p_prom_s[1]), py_coord(1.0), px2(20), py_coord(1.0), fill="#1b5e20", width=1.6)

        canvas.create_rectangle(bx0_2, y_top, bx1_2, y_bottom, outline="#000000", width=1.5)
        canvas.create_text((bx0_2 + bx1_2) / 2, y_top - 11, text="3. Input: Promedio", font=("Arial", 8, "bold"), fill="#000000")
        canvas.create_text(bx0_2 - 9, py_coord(0.0), text="0.0", font=("Arial", 7), fill="#000000")
        canvas.create_text(bx0_2 - 9, py_coord(1.0), text="1.0", font=("Arial", 7), fill="#000000")
        canvas.create_text(px2(0), y_bottom + 8, text="0", font=("Arial", 7), fill="#000000")
        canvas.create_text(px2(20), y_bottom + 8, text="20", font=("Arial", 7), fill="#000000")

        x3_pix = px2(promedio_notas)
        canvas.create_line(x3_pix, y_top, x3_pix, y_bottom, fill="#c2185b", dash=(2, 2), width=1.4)
        canvas.create_text(x3_pix, y_top - 6, text="X3", font=("Arial", 7, "bold"), fill="#c2185b")
        dibujar_flechas_mu(bx0_2, bx1_2, py_coord, [mu_prom_d, mu_prom_a, mu_prom_s])
        dibujar_caja_input(x3_pix, y_bottom, f"{promedio_notas:.1f}")

        _, _, bx0_3, bx1_3 = get_panel_box(3)
        bw_3 = bx1_3 - bx0_3
        def px3(y_val):
            c = max(0.0, min(1.0, y_val))
            return bx0_3 + c * bw_3

        p_rie_b = parametros.get('riesgo_bajo', [0.0, 0.0, 0.25, 0.45])
        p_rie_m = parametros.get('riesgo_medio', [0.3, 0.5, 0.7])
        p_rie_a = parametros.get('riesgo_alto', [0.55, 0.75, 1.0, 1.0])

        canvas.create_line(px3(0.0), py_coord(1.0), px3(p_rie_b[2]), py_coord(1.0), px3(p_rie_b[3]), py_coord(0.0), fill="#bdbdbd", width=1.2)
        canvas.create_line(px3(p_rie_m[0]), py_coord(0.0), px3(p_rie_m[1]), py_coord(1.0), px3(p_rie_m[2]), py_coord(0.0), fill="#bdbdbd", width=1.2)
        canvas.create_line(px3(p_rie_a[0]), py_coord(0.0), px3(p_rie_a[1]), py_coord(1.0), px3(1.0), py_coord(1.0), fill="#bdbdbd", width=1.2)

        alpha_b = fuerzas.get('Riesgo_Bajo', 0.0)
        alpha_m = fuerzas.get('Riesgo_Medio', 0.0)
        alpha_a = fuerzas.get('Riesgo_Alto', 0.0)

        poly_agg = [px3(0.0), py_coord(0.0)]
        hay_masa = False
        pasos_agg = 80
        for s in range(pasos_agg + 1):
            y_v = s / float(pasos_agg)
            mb = pertenencia_trapezoidal(y_v, p_rie_b[0], p_rie_b[1], p_rie_b[2], p_rie_b[3])
            mm = pertenencia_triangular(y_v, p_rie_m[0], p_rie_m[1], p_rie_m[2])
            ma = pertenencia_trapezoidal(y_v, p_rie_a[0], p_rie_a[1], p_rie_a[2], p_rie_a[3])
            c_b = min(alpha_b, mb)
            c_m = min(alpha_m, mm)
            c_a = min(alpha_a, ma)
            mu_agg = max(c_b, c_m, c_a)
            if mu_agg > 0.001:
                hay_masa = True
            poly_agg.extend([px3(y_v), py_coord(mu_agg)])
        poly_agg.extend([px3(1.0), py_coord(0.0)])

        if hay_masa:
            canvas.create_polygon(poly_agg, fill="#80deea", outline="#00897b", width=1.6)

        canvas.create_rectangle(bx0_3, y_top, bx1_3, y_bottom, outline="#000000", width=1.5)
        canvas.create_text((bx0_3 + bx1_3) / 2, y_top - 11, text="4. Aggregation & Defuzzify", font=("Arial", 8, "bold"), fill="#000000")
        canvas.create_text(bx0_3 - 9, py_coord(0.0), text="0.0", font=("Arial", 7), fill="#000000")
        canvas.create_text(bx0_3 - 9, py_coord(1.0), text="1.0", font=("Arial", 7), fill="#000000")
        canvas.create_text(px3(0.0), y_bottom + 8, text="0", font=("Arial", 7), fill="#000000")
        canvas.create_text(px3(1.0), y_bottom + 8, text="1", font=("Arial", 7), fill="#000000")

        x_cog = px3(riesgo_calculado)
        canvas.create_line(x_cog, y_top - 2, x_cog, y_bottom, fill="#7b1fa2", width=2.5, arrow=tk.LAST, arrowshape=(6, 8, 3))
        canvas.create_text(x_cog, y_top - 8, text="Centroid", font=("Arial", 7, "bold"), fill="#7b1fa2")

        txt_out = f"Output = {riesgo_calculado * 100:.1f}%"
        bw_out = 66
        x_box_out = max(bx0_3 + bw_out / 2 + 1, min(bx1_3 - bw_out / 2 - 1, x_cog))
        y_box_out = y_bottom + 12
        canvas.create_rectangle(x_box_out - bw_out / 2, y_box_out - 6, x_box_out + bw_out / 2, y_box_out + 6, outline="#7b1fa2", fill="#ffffff", width=1.5)
        canvas.create_text(x_box_out, y_box_out, text=txt_out, font=("Arial", 6, "bold"), fill="#7b1fa2")

    def dibujar_masa_difusa_detallada_en_canvas(self, canvas, fuerzas, parametros, riesgo_calculado, target_real=None, titulo_extra="", detalles_calc=None):
        ancho = canvas.winfo_width()
        alto = canvas.winfo_height()
        if ancho <= 1:
            try:
                ancho = int(canvas.cget("width"))
                alto = int(canvas.cget("height"))
            except Exception:
                pass
        if ancho < 80 or alto < 50:
            return

        canvas.delete("all")

        margen_izq = 55
        margen_der = 50
        margen_arr = 26
        margen_abj = 36
        ancho_util = ancho - margen_izq - margen_der
        alto_util = alto - margen_arr - margen_abj
        if ancho_util <= 10 or alto_util <= 10:
            return

        def escala_x(val_y):
            return margen_izq + val_y * ancho_util

        def escala_y(val_mu):
            return (alto - margen_abj) - val_mu * alto_util

        canvas.create_rectangle(0, 0, ancho, alto, fill="#ffffff", outline="")

        for mu_val in [0.25, 0.50, 0.75, 1.0]:
            y_pix = escala_y(mu_val)
            canvas.create_line(margen_izq, y_pix, ancho - margen_der, y_pix, fill="#f0f3f6", dash=(2, 4))
            canvas.create_text(margen_izq - 10, y_pix, text=f"{mu_val:.2f}", font=("Arial", 7), fill="#777777")

        for y_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            x_pix = escala_x(y_val)
            canvas.create_line(x_pix, margen_arr, x_pix, alto - margen_abj, fill="#f8fafc", dash=(2, 4))
            canvas.create_line(x_pix, alto - margen_abj, x_pix, alto - margen_abj + 4, fill="#888888")
            canvas.create_text(x_pix, alto - margen_abj + 12, text=f"{y_val:.1f}", font=("Arial", 7), fill="#555555")

        canvas.create_line(margen_izq, escala_y(0.0), ancho - margen_der, escala_y(0.0), fill="#444444", width=1.5)
        canvas.create_line(margen_izq, margen_arr, margen_izq, escala_y(0.0), fill="#444444", width=1.5)
        canvas.create_text(margen_izq - 10, escala_y(0.0), text="0.00", font=("Arial", 7), fill="#777777")
        canvas.create_text(margen_izq - 32, margen_arr + alto_util // 2, text="μ(y)", font=("Arial", 8, "bold"), fill="#222222")
        canvas.create_text(margen_izq + ancho_util // 2, alto - 8, text="Universo de Discurso y ∈ [0.0, 1.0] (Riesgo Continuo de Deserción)", font=("Arial", 8, "italic"), fill="#333333")

        p_bajo = parametros.get('riesgo_bajo', [0.0, 0.0, 0.25, 0.45])
        p_medio = parametros.get('riesgo_medio', [0.3, 0.5, 0.7])
        p_alto = parametros.get('riesgo_alto', [0.55, 0.75, 1.0, 1.0])

        pts_b = [
            (escala_x(p_bajo[0]), escala_y(0.0)),
            (escala_x(p_bajo[1]), escala_y(1.0)),
            (escala_x(p_bajo[2]), escala_y(1.0)),
            (escala_x(p_bajo[3]), escala_y(0.0))
        ]
        canvas.create_line([c for pt in pts_b for c in pt], fill="#90caf9", dash=(3, 3), width=1)
        canvas.create_text(escala_x((p_bajo[1] + p_bajo[2]) / 2), escala_y(1.0) + 12, text="Bajo", font=("Arial", 8, "italic"), fill="#1976d2")

        pts_m = [
            (escala_x(p_medio[0]), escala_y(0.0)),
            (escala_x(p_medio[1]), escala_y(1.0)),
            (escala_x(p_medio[2]), escala_y(0.0))
        ]
        canvas.create_line([c for pt in pts_m for c in pt], fill="#ffe082", dash=(3, 3), width=1)
        canvas.create_text(escala_x(p_medio[1]), escala_y(1.0) + 12, text="Medio", font=("Arial", 8, "italic"), fill="#f57c00")

        pts_a = [
            (escala_x(p_alto[0]), escala_y(0.0)),
            (escala_x(p_alto[1]), escala_y(1.0)),
            (escala_x(p_alto[2]), escala_y(1.0)),
            (escala_x(p_alto[3]), escala_y(0.0))
        ]
        canvas.create_line([c for pt in pts_a for c in pt], fill="#ef9a9a", dash=(3, 3), width=1)
        canvas.create_text(escala_x((p_alto[1] + p_alto[2]) / 2), escala_y(1.0) + 12, text="Alto", font=("Arial", 8, "italic"), fill="#d32f2f")

        alpha_b = fuerzas.get('Riesgo_Bajo', 0.0)
        alpha_m = fuerzas.get('Riesgo_Medio', 0.0)
        alpha_a = fuerzas.get('Riesgo_Alto', 0.0)

        if alpha_b > 0.005:
            canvas.create_line(escala_x(p_bajo[0]), escala_y(alpha_b), escala_x(p_bajo[3]), escala_y(alpha_b), fill="#1565c0", dash=(4, 2), width=1.5)
            canvas.create_text(escala_x(p_bajo[0]) + 28, escala_y(alpha_b) - 7, text=f"α_Bajo={alpha_b:.2f}", font=("Arial", 7, "bold"), fill="#1565c0")

        if alpha_m > 0.005:
            canvas.create_line(escala_x(p_medio[0]), escala_y(alpha_m), escala_x(p_medio[2]), escala_y(alpha_m), fill="#e65100", dash=(4, 2), width=1.5)
            canvas.create_text(escala_x(p_medio[1]), escala_y(alpha_m) - 7, text=f"α_Medio={alpha_m:.2f}", font=("Arial", 7, "bold"), fill="#e65100")

        if alpha_a > 0.005:
            canvas.create_line(escala_x(p_alto[0]), escala_y(alpha_a), escala_x(p_alto[3]), escala_y(alpha_a), fill="#c62828", dash=(4, 2), width=1.5)
            canvas.create_text(escala_x(p_alto[3]) - 28, escala_y(alpha_a) - 7, text=f"α_Alto={alpha_a:.2f}", font=("Arial", 7, "bold"), fill="#c62828")

        num_pasos = 160
        poly_coords = [escala_x(0.0), escala_y(0.0)]
        hay_masa = False

        for i in range(num_pasos + 1):
            y_val = i / float(num_pasos)
            mb = pertenencia_trapezoidal(y_val, p_bajo[0], p_bajo[1], p_bajo[2], p_bajo[3])
            mm = pertenencia_triangular(y_val, p_medio[0], p_medio[1], p_medio[2])
            ma = pertenencia_trapezoidal(y_val, p_alto[0], p_alto[1], p_alto[2], p_alto[3])
            c_b = min(alpha_b, mb)
            c_m = min(alpha_m, mm)
            c_a = min(alpha_a, ma)
            mu_agg = max(c_b, c_m, c_a)
            if mu_agg > 0.0001:
                hay_masa = True
            poly_coords.extend([escala_x(y_val), escala_y(mu_agg)])

        poly_coords.extend([escala_x(1.0), escala_y(0.0)])

        if hay_masa:
            canvas.create_polygon(poly_coords, fill="#bbdefb", outline="#0d47a1", width=2)
        else:
            canvas.create_text(
                margen_izq + ancho_util // 2, margen_arr + alto_util // 2,
                text="⚠️ Sin activación en reglas (Masa = 0). Salvaguarda neutral y* = 0.50",
                font=("Arial", 9, "bold"), fill="#e65100"
            )

        x_umbral = escala_x(0.50)
        canvas.create_line(x_umbral, margen_arr, x_umbral, escala_y(0.0), fill="#616161", dash=(4, 3), width=1.5)
        canvas.create_text(x_umbral, margen_arr - 6, text="Umbral = 0.50", font=("Arial", 8, "bold"), fill="#424242")
        canvas.create_text(escala_x(0.22), margen_arr + 8, text="◄ NO DESERCIÓN (Bajo Riesgo)", font=("Arial", 7, "bold"), fill="#2e7d32")
        canvas.create_text(escala_x(0.78), margen_arr + 8, text="DESERCIÓN (Alto Riesgo) ►", font=("Arial", 7, "bold"), fill="#c62828")

        x_cog = escala_x(riesgo_calculado)
        color_cog = "#d32f2f" if riesgo_calculado >= 0.5 else "#107c41"

        canvas.create_line(x_cog, margen_arr + 14, x_cog, escala_y(0.0), fill=color_cog, width=2.5)

        canvas.create_polygon(
            x_cog - 7, escala_y(0.0) + 11,
            x_cog + 7, escala_y(0.0) + 11,
            x_cog, escala_y(0.0),
            fill=color_cog, outline="#222222"
        )
        canvas.create_text(x_cog, escala_y(0.0) + 20, text=f"▲ Fulcro y*={riesgo_calculado:.4f}", font=("Arial", 8, "bold"), fill=color_cog)

        tag_diag = "DROPOUT" if riesgo_calculado >= 0.5 else "NO DROPOUT"
        tag_texto = f" y* = {riesgo_calculado:.4f} [{tag_diag}] "
        x_tag = max(margen_izq + 65, min(ancho - margen_der - 65, x_cog))
        canvas.create_rectangle(x_tag - 65, margen_arr - 2, x_tag + 65, margen_arr + 14, fill=color_cog, outline="#ffffff")
        canvas.create_text(x_tag, margen_arr + 6, text=tag_texto, font=("Arial", 8, "bold"), fill="#ffffff")

        titulo = "FUNCIÓN DE MASA DIFUSA AGREGADA Y CENTROIDE (Paso 3 y 4 Mamdani)"
        if target_real:
            titulo += f" | Estudiante Dataset: '{target_real}'"
        canvas.create_text(margen_izq, 10, anchor=tk.W, text=titulo, font=("Arial", 9, "bold"), fill="#0d47a1")

        if detalles_calc:
            den_val = detalles_calc.get('suma_denominador', 0.0)
            num_val = detalles_calc.get('suma_numerador', 0.0)
            integral_masa = den_val * 0.01
            integral_momento = num_val * 0.01
            info_masa = f"Área (Masa) = {integral_masa:.3f} | Momento = {integral_momento:.3f} | y* = {riesgo_calculado:.4f}"
            canvas.create_text(ancho - margen_der, 10, anchor=tk.E, text=info_masa, font=("Courier", 8, "bold"), fill="#37474f")

    def dibujar_masa_difusa_en_canvas(self, canvas, fuerzas, parametros, riesgo_calculado, target_real=None, titulo_extra="", detalles_calc=None, nota_admision=None, materias_aprobadas=None, promedio_notas=None):
        modo = getattr(self, 'modo_vista_grafico', None)
        modo_val = modo.get() if modo else "4_paneles"
        if modo_val == "detallada":
            self.dibujar_masa_difusa_detallada_en_canvas(
                canvas, fuerzas, parametros, riesgo_calculado, target_real=target_real,
                titulo_extra=titulo_extra, detalles_calc=detalles_calc
            )
        else:
            if nota_admision is None and hasattr(self, 'ultimo_caso_diagnostico'):
                caso_guardado = self.ultimo_caso_diagnostico
                nota_admision = caso_guardado.get('nota_admision', 118.0)
                materias_aprobadas = caso_guardado.get('materias_aprobadas', 1.0)
                promedio_notas = caso_guardado.get('promedio_notas', 10.0)
            elif nota_admision is None:
                nota_admision, materias_aprobadas, promedio_notas = 118.0, 1.0, 10.0
            self.dibujar_proceso_difuso_4_paneles(
                canvas, nota_admision, materias_aprobadas, promedio_notas, parametros, fuerzas, riesgo_calculado,
                target_real=target_real, det=detalles_calc
            )

    def redibujar_masa_diagnostico(self):
        if hasattr(self, 'ultimo_caso_diagnostico') and hasattr(self, 'canvas_masa_diagnostico'):
            caso_actual = self.ultimo_caso_diagnostico
            modo = getattr(self, 'modo_vista_grafico', None)
            modo_val = modo.get() if modo else "4_paneles"

            if modo_val == "detallada":
                self.dibujar_masa_difusa_detallada_en_canvas(
                    self.canvas_masa_diagnostico,
                    caso_actual.get('fuerzas', {}),
                    caso_actual.get('parametros', self.parametros_difusos),
                    caso_actual.get('riesgo', 0.5),
                    target_real=caso_actual.get('target_real'),
                    detalles_calc=caso_actual.get('det')
                )
            else:
                self.dibujar_proceso_difuso_4_paneles(
                    self.canvas_masa_diagnostico,
                    caso_actual.get('nota_admision', 118.0),
                    caso_actual.get('materias_aprobadas', 1.0),
                    caso_actual.get('promedio_notas', 10.0),
                    caso_actual.get('parametros', self.parametros_difusos),
                    caso_actual.get('fuerzas', {}),
                    caso_actual.get('riesgo', 0.5),
                    target_real=caso_actual.get('target_real'),
                    det=caso_actual.get('det')
                )

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

        ttk.Label(frame_izq, text="Carrera Universitaria (Course):").pack(anchor=tk.W, padx=10, pady=2)
        valores_carreras = list(DICCIONARIO_CARRERAS.values())
        self.combo_carrera = ttk.Combobox(frame_izq, values=valores_carreras, state="readonly")
        self.combo_carrera.set(DICCIONARIO_CARRERAS[9991])
        self.combo_carrera.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(frame_izq, text="¿Cuotas de Matrícula al Día? (Tuition):").pack(anchor=tk.W, padx=10, pady=2)
        self.combo_pagos = ttk.Combobox(
            frame_izq,
            values=["1 (Matrícula al Día / Sin Atrasos)", "0 (Atrasado / Con Cuotas Pendientes)"],
            state="readonly"
        )
        self.combo_pagos.current(1)
        self.combo_pagos.pack(fill=tk.X, padx=10, pady=2)

        btn_evaluar = ttk.Button(
            frame_izq,
            text="Calcular Diagnóstico Explicable",
            command=self.accion_diagnosticar_formulario
        )
        btn_evaluar.pack(fill=tk.X, padx=10, pady=15)

        ttk.Separator(frame_izq, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(frame_izq, text="Cargar Casos Reales (Clases Binarias):").pack(anchor=tk.W, padx=10, pady=(6, 2))
        btn_caso_drop = ttk.Button(frame_izq, text="🚨 Cargar Caso Real: Desertor (Dropout)", command=lambda: self.cargar_ejemplo_clase('Dropout'))
        btn_caso_drop.pack(fill=tk.X, padx=10, pady=3)

        btn_caso_grad = ttk.Button(frame_izq, text="🎓 Cargar Caso Real: No Desertor (Graduate)", command=lambda: self.cargar_ejemplo_clase('Graduate'))
        btn_caso_grad.pack(fill=tk.X, padx=10, pady=3)

        frame_der = ttk.LabelFrame(parent, text="Informe Explicable del Sistema Híbrido (XAI)")
        frame_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_grafico_diag = ttk.LabelFrame(
            frame_der,
            text="Visualización Gráfica de Inferencia Mamdani"
        )
        frame_grafico_diag.pack(fill=tk.X, padx=5, pady=3)

        frame_selector_vista = ttk.Frame(frame_grafico_diag)
        frame_selector_vista.pack(fill=tk.X, padx=6, pady=(1, 3))

        ttk.Label(frame_selector_vista, text="Modo de Visualización:", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=(2, 6))

        self.modo_vista_grafico = tk.StringVar(value="4_paneles")

        rb_4p = ttk.Radiobutton(
            frame_selector_vista,
            text="📊 Pipeline de 4 Paneles (Inputs X1, X2, X3 ➔ Masa & Centroide)",
            variable=self.modo_vista_grafico,
            value="4_paneles",
            command=self.redibujar_masa_diagnostico
        )
        rb_4p.pack(side=tk.LEFT, padx=6)

        rb_det = ttk.Radiobutton(
            frame_selector_vista,
            text="⚖️ Función de Masa Detallada (Balanza Fulcro COG, Integrales y Umbral)",
            variable=self.modo_vista_grafico,
            value="detallada",
            command=self.redibujar_masa_diagnostico
        )
        rb_det.pack(side=tk.LEFT, padx=6)

        self.canvas_masa_diagnostico = tk.Canvas(frame_grafico_diag, bg="white", height=185)
        self.canvas_masa_diagnostico.pack(fill=tk.BOTH, expand=False, padx=4, pady=2)
        self.canvas_masa_diagnostico.bind("<Configure>", lambda e: self.redibujar_masa_diagnostico())

        self.frame_resumen_hibrido = tk.Frame(frame_der, bg="#f1f3f5", relief=tk.GROOVE, bd=1, padx=6, pady=4)
        self.frame_resumen_hibrido.pack(fill=tk.X, padx=5, pady=(3, 3))

        self.lbl_tarjeta_difuso = tk.Label(
            self.frame_resumen_hibrido,
            text="📚 Riesgo Académico (Mamdani): --",
            font=("Arial", 9, "bold"),
            bg="#f1f3f5",
            fg="#222222"
        )
        self.lbl_tarjeta_difuso.pack(side=tk.LEFT, padx=(4, 10))

        self.lbl_tarjeta_prism = tk.Label(
            self.frame_resumen_hibrido,
            text="🏛️ Causal Administrativa (PRISM): --",
            font=("Arial", 9, "bold"),
            bg="#f1f3f5",
            fg="#222222"
        )
        self.lbl_tarjeta_prism.pack(side=tk.LEFT, padx=(4, 10))

        self.lbl_tarjeta_veredicto = tk.Label(
            self.frame_resumen_hibrido,
            text="Diagnóstico: --",
            font=("Arial", 9, "bold"),
            bg="#f1f3f5",
            fg="#222222"
        )
        self.lbl_tarjeta_veredicto.pack(side=tk.RIGHT, padx=6)

        self.txt_diagnostico = scrolledtext.ScrolledText(frame_der, font=("Courier", 10))
        self.txt_diagnostico.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ventana.after(200, lambda: self.accion_diagnosticar_formulario())

    def cargar_ejemplo_clase(self, clase_objetivo):
        caso_seleccionado = None
        for estudiante in self.datos_prueba:
            if estudiante.get('Target') == clase_objetivo:
                materias_aprobadas_val = float(estudiante.get('Curricular units 1st sem (approved)', 0))
                promedio_notas_val = float(estudiante.get('Curricular units 1st sem (grade)', 0))

                if clase_objetivo == 'Dropout':
                    if estudiante.get('Course') == 9991 and estudiante.get('Tuition fees up to date') == 0:
                        caso_seleccionado = estudiante
                        break
                    elif materias_aprobadas_val <= 1 and promedio_notas_val <= 8.0:
                        if caso_seleccionado is None:
                            caso_seleccionado = estudiante
                    continue

                if clase_objetivo == 'Graduate':
                    if materias_aprobadas_val >= 5 and promedio_notas_val >= 12.0:
                        caso_seleccionado = estudiante
                        break
                    continue

        if caso_seleccionado is None:
            for estudiante in self.datos_prueba:
                if estudiante.get('Target') == clase_objetivo:
                    caso_seleccionado = estudiante
                    break

        if caso_seleccionado:
            self.entry_adm.delete(0, tk.END)
            self.entry_adm.insert(0, str(caso_seleccionado.get('Admission grade', 100)))

            self.entry_aprob.delete(0, tk.END)
            self.entry_aprob.insert(0, str(caso_seleccionado.get('Curricular units 1st sem (approved)', 0)))

            self.entry_prom.delete(0, tk.END)
            self.entry_prom.insert(0, str(caso_seleccionado.get('Curricular units 1st sem (grade)', 0)))

            carrera_id = int(caso_seleccionado.get('Course', 9991))
            texto_carrera = DICCIONARIO_CARRERAS.get(carrera_id, f"{carrera_id} - Carrera {carrera_id}")
            self.combo_carrera.set(texto_carrera)

            pagos_val = int(caso_seleccionado.get('Tuition fees up to date', 1))
            idx_pagos = 0 if pagos_val == 1 else 1
            self.combo_pagos.current(idx_pagos)

            self.accion_diagnosticar_formulario(target_real=clase_objetivo)

    def accion_diagnosticar_formulario(self, target_real=None):
        try:
            nota_admision = float(self.entry_adm.get())
            materias_aprobadas = float(self.entry_aprob.get())
            promedio_notas = float(self.entry_prom.get())

            carrera_str = self.combo_carrera.get().strip()
            if not carrera_str:
                carrera_str = DICCIONARIO_CARRERAS[9991]
                self.combo_carrera.set(carrera_str)
            carrera_codigo = int(carrera_str.split()[0])

            pagos_str = self.combo_pagos.get().strip()
            if not pagos_str:
                pagos_str = "1"
            pagos_val = int(pagos_str.split()[0])
        except ValueError:
            messagebox.showerror("Error", "Por favor ingresa números válidos en las notas.")
            return

        estudiante_eval = {
            'Admission grade': nota_admision,
            'Curricular units 1st sem (approved)': materias_aprobadas,
            'Curricular units 1st sem (grade)': promedio_notas,
            'Course': carrera_codigo,
            'Tuition fees up to date': pagos_val,
            'Debtor': 1 if pagos_val == 0 else 0
        }

        riesgo, det = evaluar_motor_difuso(
            nota_admision, materias_aprobadas, promedio_notas, self.parametros_difusos, 100, reglas=self.reglas_difusas
        )

        if not self.reglas_prism and self.datos_entrenamiento:
            self.reglas_prism, _ = ejecutar_prism(self.datos_entrenamiento, clase_objetivo='Dropout', semilla=42)

        reglas_prism_disparadas = []
        if self.reglas_prism:
            for regla in self.reglas_prism:
                cumple_todas = True
                for atributo, valor_esperado in regla['condiciones']:
                    valor_estudiante = estudiante_eval.get(atributo)
                    if valor_estudiante != valor_esperado and str(valor_estudiante) != str(valor_esperado):
                        cumple_todas = False
                        break
                if cumple_todas:
                    reglas_prism_disparadas.append(regla)

        alerta_prism = len(reglas_prism_disparadas) > 0

        reglas_apriori_disparadas = []
        if self.reglas_apriori:
            items_estudiante = extraer_items_estudiante_apriori(estudiante_eval, incluir_target=False)
            for regla in self.reglas_apriori:
                if regla['antecedente'].issubset(items_estudiante):
                    reglas_apriori_disparadas.append(regla)

        nombre_carrera = DICCIONARIO_CARRERAS.get(carrera_codigo, f"Carrera {carrera_codigo}")
        estado_pagos_txt = "Al Día (Sin Cuotas Vencidas)" if pagos_val == 1 else "Atrasado (Cuotas Pendientes de Matrícula)"

        if hasattr(self, 'lbl_tarjeta_difuso'):
            color_difuso = "#d13438" if riesgo >= 0.5 else "#107c41"
            estado_dif_txt = "Alto Riesgo Académico" if riesgo >= 0.5 else "Bajo Riesgo Académico"
            self.lbl_tarjeta_difuso.config(
                text=f"📚 Riesgo Académico (Mamdani): {riesgo*100:.2f}% ({estado_dif_txt})",
                fg=color_difuso
            )

        if hasattr(self, 'lbl_tarjeta_prism'):
            if alerta_prism:
                top_conf_pct = max(r['confianza'] for r in reglas_prism_disparadas) * 100
                self.lbl_tarjeta_prism.config(
                    text=f"⚠️ Causal PRISM: Matrícula Atrasada ({top_conf_pct:.1f}% probabilidad de abandono)",
                    fg="#d13438"
                )
            else:
                self.lbl_tarjeta_prism.config(
                    text="✓ Causal PRISM: Matrícula al Día (Solvente)",
                    fg="#107c41"
                )

        if hasattr(self, 'lbl_tarjeta_veredicto'):
            if riesgo >= 0.5 and alerta_prism:
                txt_v = "🚨 CONVERGENCIA CRÍTICA (Académica + Financiera)"
                col_v = "#b71c1c"
            elif riesgo >= 0.5:
                txt_v = "⚠️ ALERTA ACADÉMICA PURA (Notas bajas, solvente)"
                col_v = "#d13438"
            elif alerta_prism:
                txt_v = "⚠️ ALERTA FINANCIERA (Rescatado por PRISM)"
                col_v = "#8a1c14"
            else:
                txt_v = "🛡️ ESTUDIANTE SEGURO (Bajo Riesgo)"
                col_v = "#107c41"
            self.lbl_tarjeta_veredicto.config(text=f"Diagnóstico: {txt_v}", fg=col_v)

        self.txt_diagnostico.delete('1.0', tk.END)
        self.txt_diagnostico.insert(tk.END, "=" * 95 + "\n")
        self.txt_diagnostico.insert(tk.END, "  INFORME DE DIAGNÓSTICO EXPLICABLE (XAI) - SISTEMA HÍBRIDO DE INTELIGENCIA ARTIFICIAL\n")
        self.txt_diagnostico.insert(tk.END, "=" * 95 + "\n\n")

        self.txt_diagnostico.insert(tk.END, f"Perfil del Estudiante Evaluado:\n")
        self.txt_diagnostico.insert(tk.END, f"  - Nota de Admisión:            {nota_admision:.1f} / 200 pts\n")
        self.txt_diagnostico.insert(tk.END, f"  - Materias Aprobadas S1:       {materias_aprobadas:.0f} asignaturas\n")
        self.txt_diagnostico.insert(tk.END, f"  - Promedio de Notas S1:        {promedio_notas:.2f} / 20 pts\n")
        self.txt_diagnostico.insert(tk.END, f"  - Carrera Universitaria:       {nombre_carrera}\n")
        self.txt_diagnostico.insert(tk.END, f"  - Situación de Matrícula:      {estado_pagos_txt}\n")
        if target_real:
            self.txt_diagnostico.insert(tk.END, f"  - Condición Real Verificada:   {target_real}\n\n")
        else:
            self.txt_diagnostico.insert(tk.END, "\n")

        texto_4_pasos = describir_los_4_pasos_difusos(
            nota_admision, materias_aprobadas, promedio_notas, self.parametros_difusos, detalle_motor=det, reglas=self.reglas_difusas
        )
        self.txt_diagnostico.insert(tk.END, texto_4_pasos + "\n\n")

        self.txt_diagnostico.insert(tk.END, "[PILAR 2: EVALUACIÓN DE REGLAS CAUSALES PRISM (EVALUACIÓN DINÁMICA DE CONOCIMIENTO)]\n")
        self.txt_diagnostico.insert(tk.END, "-" * 95 + "\n")
        if alerta_prism:
            self.txt_diagnostico.insert(tk.END, f"  ¡ALERTA CRÍTICA PRISM ACTIVADA! (Se activaron {len(reglas_prism_disparadas)} regla(s) de conocimiento):\n")
            for idx_regla, r_disp in enumerate(reglas_prism_disparadas, 1):
                partes_c = [f"({traducir_atributo(a)} = '{v}')" for a, v in r_disp['condiciones']]
                formula_c = " AND ".join(partes_c)
                conf_pct = r_disp['confianza'] * 100
                sop_pct = r_disp['soporte'] * 100
                cob_casos = r_disp['cobertura']
                lift_val = r_disp['lift']
                self.txt_diagnostico.insert(tk.END, f"  -> Regla PRISM #{idx_regla}: SI {formula_c} ENTONCES Deserción = 'Sí'\n")
                self.txt_diagnostico.insert(tk.END, f"     Métricas de Inducción: Confianza = {conf_pct:.2f}% | Cobertura Histórica = {cob_casos} casos | Soporte = {sop_pct:.2f}% | Lift = {lift_val:.4f}\n")
            top_conf = max(r['confianza'] for r in reglas_prism_disparadas) * 100
            self.txt_diagnostico.insert(tk.END, f"\n  Explicación Causal e Institucional:\n")
            self.txt_diagnostico.insert(tk.END, f"  El estudiante cursa la carrera de {nombre_carrera} con estado de matrícula '{estado_pagos_txt}'.\n")
            self.txt_diagnostico.insert(tk.END, f"  El algoritmo PRISM demostró matemáticamente sobre el conjunto de entrenamiento que el {top_conf:.2f}% de los\n")
            self.txt_diagnostico.insert(tk.END, f"  estudiantes bajo este patrón causal de atraso financiero abandonaron la universidad.\n\n")
        else:
            self.txt_diagnostico.insert(tk.END, "  Sin alertas críticas en el módulo causal PRISM.\n")
            self.txt_diagnostico.insert(tk.END, f"  El perfil del estudiante ({nombre_carrera}, Matrícula: {estado_pagos_txt}) no coincide con ninguna\n")
            self.txt_diagnostico.insert(tk.END, "  de las reglas causales determinísticas de abandono obligatorio inducidas por el algoritmo PRISM.\n\n")

        self.txt_diagnostico.insert(tk.END, "[PILAR 3: MINERÍA DE ASOCIACIÓN APRIORI (PATRONES FRECUENTES DE ASOCIACIÓN)]\n")
        self.txt_diagnostico.insert(tk.END, "-" * 95 + "\n")
        if reglas_apriori_disparadas:
            self.txt_diagnostico.insert(tk.END, f"  El estudiante activa {len(reglas_apriori_disparadas)} regla(s) de asociación en Apriori:\n")
            for idx_ap, r_ap in enumerate(reglas_apriori_disparadas[:3], 1):
                ant_ap = " AND ".join(sorted(r_ap['antecedente']))
                cons_ap = " AND ".join(sorted(r_ap['consecuente']))
                conf_ap = r_ap['confianza'] * 100
                sop_ap = r_ap.get('soporte', r_ap.get('soporte_regla', 0.0)) * 100
                self.txt_diagnostico.insert(tk.END, f"  -> Patrón #{idx_ap}: SI [{ant_ap}] ENTONCES [{cons_ap}] (Conf={conf_ap:.1f}%, Sop={sop_ap:.1f}%)\n")
            self.txt_diagnostico.insert(tk.END, "\n")
        else:
            self.txt_diagnostico.insert(tk.END, "  No se registran patrones atípicos de alta confianza en Apriori para este perfil individual.\n")
            self.txt_diagnostico.insert(tk.END, "  (Nota: Para enriquecer los patrones globales, ejecuta 'Minería Apriori' en la Pestaña 1).\n\n")

        self.txt_diagnostico.insert(tk.END, "[SÍNTESIS DIAGNÓSTICA Y VEREDICTO FINAL DEL SISTEMA HÍBRIDO (XAI)]\n")
        self.txt_diagnostico.insert(tk.END, "=" * 95 + "\n")
        if riesgo >= 0.5 or alerta_prism:
            pred = "Dropout (Alto Riesgo de Deserción)"
            color_txt = "PELIGRO DE ABANDONO DETECTADO"
            if riesgo >= 0.5 and alerta_prism:
                motivo = "Convergencia Crítica: Alto riesgo académico continuo en Motor Difuso Y causal administrativo-institucional PRISM."
            elif riesgo >= 0.5:
                motivo = "Alerta Académica Pura: Bajo rendimiento cuantitativo en asignaturas y calificaciones según Inferencia Difusa Mamdani (El estudiante está al día financieramente, pero reprueba académicamente)."
            else:
                top_conf_pct = max(r['confianza'] for r in reglas_prism_disparadas) * 100
                motivo = f"Alerta Administrativo-Institucional: Activación de Regla Causal PRISM ({top_conf_pct:.1f}% de certeza histórica por matrícula impaga, a pesar de tener notas aprobatorias)."
        else:
            pred = "No Dropout (Bajo Riesgo / Continuará)"
            color_txt = "ESTUDIANTE SEGURO"
            motivo = "Rendimiento académico satisfactorio y ausencia de causales administrativas o institucionales de deserción."

        self.txt_diagnostico.insert(tk.END, f"  Estado General:    {color_txt}\n")
        self.txt_diagnostico.insert(tk.END, f"  Predicción Final:  {pred}\n")
        self.txt_diagnostico.insert(tk.END, f"  Justificación XAI: {motivo}\n")
        if target_real:
            acerto = (pred.startswith("Dropout") and target_real == "Dropout") or (pred.startswith("No Dropout") and target_real != "Dropout")
            marca = "✓ PREDICCIÓN CORRECTA (Coincide con el histórico real)" if acerto else "✗ DISCREPANCIA CON LA CONDICIÓN REAL"
            self.txt_diagnostico.insert(tk.END, f"  Verificación:      {marca} (Realidad: {target_real})\n")
        self.txt_diagnostico.insert(tk.END, "=" * 95 + "\n")

        self.ultimo_caso_diagnostico = {
            'nota_admision': nota_admision,
            'materias_aprobadas': materias_aprobadas,
            'promedio_notas': promedio_notas,
            'carrera': carrera_codigo,
            'pagos': pagos_val,
            'fuerzas': det['fuerzas'],
            'parametros': self.parametros_difusos,
            'riesgo': riesgo,
            'target_real': target_real,
            'det': det
        }
        if hasattr(self, 'canvas_masa_diagnostico'):
            self.redibujar_masa_diagnostico()

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

        notebook_res = ttk.Notebook(parent)
        notebook_res.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)

        frame_sub_difuso = ttk.Frame(notebook_res)
        notebook_res.add(frame_sub_difuso, text=" Demostración Motor Difuso en Prueba (885 Casos No Vistos) ")
        self.txt_resultado_difuso = scrolledtext.ScrolledText(frame_sub_difuso, font=("Courier", 10))
        self.txt_resultado_difuso.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        frame_sub_hibrido = ttk.Frame(notebook_res)
        notebook_res.add(frame_sub_hibrido, text=" Matriz Global Sistema Híbrido (Difuso + PRISM + Apriori) ")
        self.txt_matriz = scrolledtext.ScrolledText(frame_sub_hibrido, font=("Courier", 10))
        self.txt_matriz.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.actualizar_resultados_finales()

    def actualizar_resultados_finales(self):
        if not self.datos_prueba:
            return

        es_calibrado = len(self.historial_fitness) > 1

        res_difuso = evaluar_rendimiento_motor_difuso(
            self.datos_prueba, self.parametros_difusos, self.reglas_difusas
        )
        reporte_difuso = formatear_reporte_demostracion_motor_difuso(res_difuso, es_calibrado=es_calibrado)

        if hasattr(self, 'txt_resultado_difuso'):
            self.txt_resultado_difuso.config(state=tk.NORMAL)
            self.txt_resultado_difuso.delete('1.0', tk.END)
            self.txt_resultado_difuso.insert(tk.END, reporte_difuso)
            self.txt_resultado_difuso.config(state=tk.DISABLED)

        verdaderos_positivos_hibrido = 0
        falsos_positivos_hibrido = 0
        verdaderos_negativos_hibrido = 0
        falsos_negativos_hibrido = 0

        verdaderos_positivos_difuso = res_difuso['vp']

        for estudiante in self.datos_prueba:
            nota_admision = float(estudiante.get('Admission grade', 100))
            materias_aprobadas = float(estudiante.get('Curricular units 1st sem (approved)', 0))
            promedio_notas = float(estudiante.get('Curricular units 1st sem (grade)', 0))
            condicion_real_estudiante = estudiante.get('Target', '')
            es_desertor_real = (condicion_real_estudiante == 'Dropout')

            riesgo, _ = evaluar_motor_difuso(
                nota_admision, materias_aprobadas, promedio_notas, self.parametros_difusos, 100, reglas=self.reglas_difusas
            )
            prediccion_difusa = (riesgo >= 0.5)

            activa_prism = False
            if self.reglas_prism:
                for regla in self.reglas_prism:
                    cumple = True
                    for atributo, valor in regla['condiciones']:
                        valor_estudiante = estudiante.get(atributo)
                        if valor_estudiante != valor and str(valor_estudiante) != str(valor):
                            cumple = False
                            break
                    if cumple:
                        activa_prism = True
                        break

            prediccion_hibrida = prediccion_difusa or activa_prism

            if prediccion_hibrida and es_desertor_real:
                verdaderos_positivos_hibrido += 1
            elif prediccion_hibrida and not es_desertor_real:
                falsos_positivos_hibrido += 1
            elif not prediccion_hibrida and not es_desertor_real:
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
