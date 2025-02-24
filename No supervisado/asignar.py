import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo

# Función para analizar los clusters de mantenibilidad
def analyze_maintainability_clusters(df):
    # Identificar métricas donde valores más altos indican PEOR mantenibilidad
    negative_metrics = [
        'code_smells',
        'major_violations',
        'blocker_violations',
        'maintainability_issues',
        'sqale_debt_ratio',
        'duplicated_files',
        'duplicated_lines_density',
        'critical_violations',
        'uncovered_lines'
    ]
    
    # Calcular score de mantenibilidad por cluster
    cluster_scores = {}
    
    for cluster in df['mantenibilidad'].unique():
        cluster_data = df[df['mantenibilidad'] == cluster]
        
        # Calcular promedio de métricas negativas
        negative_score = np.mean([cluster_data[metric].mean() for metric in negative_metrics])
        
        cluster_scores[cluster] = {
            'negative_score': negative_score,
            'size': len(cluster_data),
            'metrics': {
                metric: cluster_data[metric].mean()
                for metric in df.columns if metric != 'mantenibilidad'
            }
        }
    
    # Ordenar clusters por score (mayor score = peor mantenibilidad)
    ranked_clusters = sorted(
        cluster_scores.items(),
        key=lambda x: x[1]['negative_score'],
        reverse=True
    )
    
    # Asignar niveles de mantenibilidad
    maintainability_levels = {
        ranked_clusters[0][0]: "Baja Mantenibilidad",
        ranked_clusters[1][0]: "Mantenibilidad Media",
        ranked_clusters[2][0]: "Alta Mantenibilidad"
    }
    
    # Imprimir análisis detallado
    print("\nANÁLISIS DE CLUSTERS DE MANTENIBILIDAD")
    print("=" * 50)
    
    for cluster, level in maintainability_levels.items():
        score = cluster_scores[cluster]
        print(f"\nCluster {cluster} - {level}")
        print(f"Número de proyectos: {score['size']}")
        print("\nCaracterísticas principales:")
        print("-" * 30)
        
        # Mostrar las métricas más significativas
        metrics = score['metrics']
        for metric in negative_metrics:
            print(f"{metric}: {metrics[metric]:.2f}")
            
        print("\nJustificación:")
        if level == "Baja Mantenibilidad":
            print("Este cluster muestra los valores más altos en métricas negativas como:")
        elif level == "Mantenibilidad Media":
            print("Este cluster presenta valores intermedios en la mayoría de las métricas:")
        else:
            print("Este cluster exhibe los valores más bajos en métricas negativas como:")
            
        # Mostrar las 3 métricas más distintivas
        distinctive_metrics = sorted(
            negative_metrics,
            key=lambda m: abs(metrics[m] - np.mean([cluster_scores[c]['metrics'][m] for c in cluster_scores if c != cluster])),
            reverse=True
        )[:3]
        
        for metric in distinctive_metrics:
            value = metrics[metric]
            other_clusters_mean = np.mean([cluster_scores[c]['metrics'][metric] for c in cluster_scores if c != cluster])
            print(f"- {metric}: {value:.2f} (promedio otros clusters: {other_clusters_mean:.2f})")
    
    return maintainability_levels

# Función para visualizar los clusters con un gráfico de dispersión
def plot_cluster_scatter(df, maintainability_levels):
    # Usar dos métricas clave para el gráfico: 'code_smells' y 'maintainability_issues'
    key_metrics = ['code_smells', 'maintainability_issues']
    
    # Crear un gráfico de dispersión
    plt.figure(figsize=(10, 6))
    
    # Asignar colores por cluster
    palette = {
        "Baja Mantenibilidad": "red",
        "Mantenibilidad Media": "yellow",
        "Alta Mantenibilidad": "green"
    }
    
    # Graficar los puntos de cada proyecto
    for cluster in df['mantenibilidad'].unique():
        cluster_data = df[df['mantenibilidad'] == cluster]
        cluster_label = maintainability_levels[cluster]
        plt.scatter(
            cluster_data[key_metrics[0]], 
            cluster_data[key_metrics[1]], 
            label=cluster_label,
            color=palette[cluster_label],
            alpha=0.6
        )
    
    # Añadir título y etiquetas
    plt.title("Distribución de Proyectos según Clusters de Mantenibilidad")
    plt.xlabel("Code Smells")
    plt.ylabel("Maintainability Issues")
    plt.legend(title="Cluster de Mantenibilidad", loc='upper right')
    
    # Guardar la gráfica como un archivo
    plt.tight_layout()
    plt.savefig('cluster_plot.png')  # Guardar la imagen como 'cluster_plot.png'
    print("Gráfico guardado como 'cluster_plot.png'")
    plt.close()  # Cerrar la figura después de guardarla para evitar problemas con la memoria

# Uso del código
# Asegúrate de que el archivo CSV esté cargado correctamente
file_path = './data/DATASET_FINAL.csv'  # Ajusta la ruta si es necesario
df = pd.read_csv(file_path)

# Llamar a las funciones de análisis y visualización
maintainability_levels = analyze_maintainability_clusters(df)
plot_cluster_scatter(df, maintainability_levels)
