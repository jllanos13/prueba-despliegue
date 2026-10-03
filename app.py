import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de Aprobación", layout="wide")
st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza una predicción utilizando un modelo de Bagging pre-entrenado.")

# Función para procesar y predecir sobre un DataFrame
def procesar_y_predecir(df_input):
    df_procesado = df_input.copy()
    
    # Asegurar que las columnas requeridas existan o se manejen
    if 'Felder' not in df_procesado.columns or 'Examen_admisión' not in df_procesado.columns:
        raise ValueError("El conjunto de datos debe contener las columnas 'Felder' y 'Examen_admisión'.")

    # 1. Cargar y aplicar el transformador/columnas de One-Hot para la variable Felder
    one_hot_transformer = joblib.load('one_hot_columns.joblib')

    if isinstance(one_hot_transformer, list):
        si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
        for col_name in si_columnas_one_hot:
            valor_esperado = col_name.replace('Felder_', '')
            df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(int)
    else:
        df_encoded = pd.get_dummies(df_procesado[['Felder']])
        df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
        si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]

    # Eliminar la variable original Felder
    df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')

    # Asegurar que existan todas las columnas que el modelo espera
    if isinstance(one_hot_transformer, list):
        for col in si_columnas_one_hot:
            if col not in df_procesado.columns:
                df_procesado[col] = 0

    # 2. Normalizar la variable Examen_admisión con 'min_max_scaler.joblib'
    scaler = joblib.load('min_max_scaler.joblib')
    df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
    df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')

    # Reordenar las columnas conforme lo espera el modelo
    columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
    df_procesado = df_procesado[columnas_ordenadas]

    # 3. Predicción con 'bagging_optimizado.joblib'
    model = joblib.load('bagging_optimizado.joblib')
    predicciones = model.predict(df_procesado)
    
    return df_procesado, predicciones

# Crear pestañas para separar las opciones de entrada
tab1, tab2 = st.tabs(["Predicción Individual (Manual)", "Predicción por Archivo (Lote)"])

with tab1:
    st.header("Datos de Entrada Individual")
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder, key="manual_felder")
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01, key="manual_examen")

    if st.button("Realizar Predicción Individual", key="btn_manual"):
        try:
            df_single = pd.DataFrame({'Felder': [felder_input], 'Examen_admisión': [examen_input]})
            df_proc, pred = procesar_y_predecir(df_single)
            
            st.subheader("Datos Procesados")
            st.dataframe(df_proc)
            st.success(f"La predicción del modelo (Nota Final Estimada) es: {pred[0]:.4f}")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

with tab2:
    st.header("Cargar Archivo para Predicción")
    st.write("Sube un archivo Excel (.xlsx) o CSV (.csv) que contenga las columnas `Felder` y `Examen_admisión`.")
    
    archivo_subido = st.file_uploader("Selecciona tu archivo:", type=["xlsx", "csv"])
    
    if archivo_subido is not None:
        try:
            if archivo_subido.name.endswith('.xlsx'):
                df_file = pd.read_excel(archivo_subido)
            else:
                df_file = pd.read_csv(archivo_subido)
            
            st.subheader("Vista previa del archivo cargado")
            st.dataframe(df_file.head())
            
            if st.button("Procesar y Predecir Archivo", key="btn_archivo"):
                df_proc, predicciones = procesar_y_predecir(df_file)
                
                # Agregar predicciones al DataFrame original para mostrar al usuario
                df_resultado = df_file.copy()
                df_resultado['Nota_final_Predicha'] = predicciones
                
                st.subheader("Resultados de las Predicciones")
                st.dataframe(df_resultado)
                
                # Permitir descargar el resultado
                csv = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar Predicciones (CSV)",
                    data=csv,
                    file_name="predicciones_resultados.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
