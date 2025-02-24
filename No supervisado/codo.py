import matplotlib
matplotlib.use('Agg')  # Usar backend sin interfaz gráfica

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import os

def elbow_method(data, max_clusters=10):
    # Preprocesamiento: Escalado de características con RobustScaler
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Lista para almacenar los valores de inercia
    inertias = []
    
    # Probar diferentes números de clusters
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        inertias.append(kmeans.inertia_)
    
    # Determinar el número óptimo de clusters utilizando el método de la distancia máxima
    x_points = np.arange(1, max_clusters + 1)
    y_points = np.array(inertias)
    
    # Línea entre los extremos (k=1 y k=max_clusters)
    line_start = np.array([x_points[0], y_points[0]])
    line_end = np.array([x_points[-1], y_points[-1]])
    
    # Calcular la distancia perpendicular de cada punto a la línea
    distances = []
    for i in range(len(x_points)):
        point = np.array([x_points[i], y_points[i]])
        dist = np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        distances.append(dist)
    
    # Encontrar el índice del punto con la distancia máxima
    optimal_k = np.argmax(distances) + 1  # +1 porque los índices comienzan en 0

    # Graficar el método del codo
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o', label='Inercia')
    plt.xticks(range(1, max_clusters + 1))  # Asegurar ticks claros del 1 al 10
    plt.plot(optimal_k, inertias[optimal_k - 1], 'ro', markersize=10, label='Número Óptimo de Clusters')
    
    plt.title('Método del Codo (Elbow Method) para K-means')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Guardar la gráfica en un archivo
    output_path = 'Curva_codo.png'
    plt.savefig(output_path)
    plt.close()  # Cerrar la figura para liberar memoria
    
    print(f"Gráfica del codo guardada en {os.path.abspath(output_path)}")
    
    return inertias, optimal_k

# Función principal para ejecutar el script
def main():
    """
    Función principal para ejecutar el método del codo.
    """
    try:
        # Reemplaza 'tu_archivo.csv' con la ruta a tu archivo de datos
        data = pd.read_csv('./data/Datos_Escalados.csv')
    except FileNotFoundError:
        print("Error: Archivo no encontrado. Verifica la ruta del archivo CSV.")
        return
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
        return
    
    # Eliminar columnas no numéricas si las hay
    data_numeric = data.select_dtypes(include=[np.number])
    
    if data_numeric.empty:
        print("Error: No hay columnas numéricas en el conjunto de datos.")
        return
    
    # Ejecutar método del codo
    inertias, optimal_k = elbow_method(data_numeric)
    
    # Imprimir valores de inercia y el número óptimo de clusters
    print("\nValores de Inercia:")
    for k, inercia in enumerate(inertias, 1):
        print(f"Clusters {k}: {inercia}")
    
    print(f"\nEl número óptimo de clusters es: {optimal_k}")

if __name__ == "__main__":
    main()
