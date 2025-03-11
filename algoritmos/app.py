import streamlit as st
import pandas as pd
import joblib

class MaintainabilityPredictorWeb:
    def __init__(self, model_path):
        """
        Initialize the Maintainability Predictor web application.
        
        :param model_path: Path to the trained machine learning model
        """
        # Load the pre-trained model
        self.model = joblib.load(model_path)
        
        # Define column names for prediction
        self.columnas = [
            'code_smells', 'major_violations', 'blocker_violations','maintainability_issues', 'sqale_debt_ratio',
            'duplicated_files', 'minor_violations', 'files', 'duplicated_lines_density',
            'critical_violations', 'effort_to_reach_maintainability_rating_a',
            'uncovered_lines', 'classes', 'forks', 'total_contributors',
            'total_files', 'stars', 'size_kb', 'total_pull_requests',
            'created_year', 'created_month', 'created_day', 'sqale_rating'
        ]

    def run(self):
        """
        Create the Streamlit web application interface.
        """
        # Set page configuration
        st.set_page_config(
            page_title="Predictor de Mantenibilidad de Proyectos",
            page_icon="🔍",
            layout="wide"
        )

        # Title and description
        st.title("🖥️ Predictor de Mantenibilidad de Proyectos")
        st.markdown("""
        ### Predice la mantenibilidad de tu proyecto de software
        Ingresa los datos de tu proyecto para obtener una predicción de su mantenibilidad.
        """)

        # Create a sidebar for input
        st.sidebar.header("Parámetros de Entrada del Proyecto")

        # Create input fields
        input_data = {}
        for columna in self.columnas:
            input_data[columna] = st.sidebar.number_input(
                label=columna.replace('_', ' ').title(), 
                value=0.0, 
                step=0.01,
                help=f"Ingrese el valor para {columna}"
            )

        # Prediction button
        if st.sidebar.button("Predecir Mantenibilidad"):
            # Create DataFrame
            df = pd.DataFrame([input_data])
            
            try:
                # Make prediction
                prediccion = self.model.predict(df)
                
                # Map prediction to message
                mensajes_mantenibilidad = {
                    0: ("🟢 Mantenibilidad del Proyecto: ALTA", "success"),
                    1: ("🔴 Mantenibilidad del Proyecto: BAJA", "error"),
                    2: ("🟡 Mantenibilidad del Proyecto: MEDIA", "warning")
                }

                mensaje, tipo_alerta = mensajes_mantenibilidad.get(prediccion[0], ("⚠️ Valor no reconocido", "warning"))

                # Display results
                getattr(st, tipo_alerta)(mensaje)

            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")

        # Additional information section
        st.sidebar.markdown("### Información")
        st.sidebar.info(
            "Esta herramienta utiliza un modelo de machine learning "
            "para predecir la mantenibilidad de proyectos de software."
        )

def main():
    # Path to the model
    MODEL_PATH = './modelo/modelo_mantenibilidad.pkl'
    
    # Create and run the predictor
    predictor = MaintainabilityPredictorWeb(MODEL_PATH)
    predictor.run()

if __name__ == "__main__":
    main()
