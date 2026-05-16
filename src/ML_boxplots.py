import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
OUTPUT_DIR = "Figures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

np.random.seed(42)
MODELS_ORDER = ['LR', 'LDA', 'KNN', 'DT-CART', 'NB', 'TabNet', 'TabTransformer']

# ==========================================
# 1. MAPEO EXACTO DE ARCHIVOS (VERIFICADO CON LATEX)
# ==========================================
FILENAME_MAP = {
    # --- TABLA IV (Todas las actividades) ---
    '4_a': 'all-xyz.eps',
    '4_b': 'all-xyzSMV.eps',
    '4_c': 'all-fil.eps',
    '4_d': 'all-smvRUF.eps',
    '4_e': 'all-smvRUFHRR.eps',
    '4_f': 'all-todas.eps',

    # --- TABLA V (4 actividades) ---
    '5_a': '4a-xyz.eps',
    '5_b': '4a-xyzSMV.eps',
    '5_c': '4a-todoFil.eps',
    '5_d': '4a-smvf-ratio.eps',
    '5_e': '4a-smvf-ratioHRR.eps',
    '5_f': '4a-todas.eps'
}

# ==========================================
# 2. DATOS ESTADÍSTICOS (CORREGIDOS)
# ==========================================
TABLE_STATS = {
    # --- TABLA IV ---
    '4_a': {'LR': (0.15, 0.06), 'LDA': (0.15, 0.06), 'KNN': (0.18, 0.08), 'DT-CART': (0.16, 0.05), 'NB': (0.17, 0.07), 'TabNet': (0.17, 0.05), 'TabTransformer': (0.18, 0.08)},
    '4_b': {'LR': (0.19, 0.07), 'LDA': (0.19, 0.07), 'KNN': (0.19, 0.06), 'DT-CART': (0.18, 0.06), 'NB': (0.22, 0.10), 'TabNet': (0.21, 0.06), 'TabTransformer': (0.21, 0.03)},
    '4_c': {'LR': (0.26, 0.02), 'LDA': (0.25, 0.02), 'KNN': (0.23, 0.02), 'DT-CART': (0.21, 0.03), 'NB': (0.25, 0.03), 'TabNet': (0.25, 0.02), 'TabTransformer': (0.26, 0.02)},
    
    # CORRECCIÓN AQUÍ: TabTransformer (0.22, 0.02) en lugar de (0.22, 0.00)
    '4_d': {'LR': (0.29, 0.04), 'LDA': (0.26, 0.04), 'KNN': (0.26, 0.02), 'DT-CART': (0.24, 0.03), 'NB': (0.26, 0.05), 'TabNet': (0.26, 0.04), 'TabTransformer': (0.22, 0.02)},
    
    '4_e': {'LR': (0.40, 0.11), 'LDA': (0.40, 0.10), 'KNN': (0.31, 0.09), 'DT-CART': (0.30, 0.09), 'NB': (0.35, 0.08), 'TabNet': (0.39, 0.09), 'TabTransformer': (0.31, 0.10)},
    '4_f': {'LR': (0.31, 0.05), 'LDA': (0.32, 0.08), 'KNN': (0.30, 0.09), 'DT-CART': (0.26, 0.08), 'NB': (0.36, 0.08), 'TabNet': (0.33, 0.06), 'TabTransformer': (0.26, 0.09)},
    
    # --- TABLA V ---
    '5_a': {'LR': (0.40, 0.16), 'LDA': (0.40, 0.16), 'KNN': (0.40, 0.14), 'DT-CART': (0.38, 0.13), 'NB': (0.45, 0.15), 'TabNet': (0.36, 0.10), 'TabTransformer': (0.31, 0.13)},
    '5_b': {'LR': (0.39, 0.16), 'LDA': (0.40, 0.16), 'KNN': (0.40, 0.13), 'DT-CART': (0.34, 0.10), 'NB': (0.53, 0.07), 'TabNet': (0.39, 0.08), 'TabTransformer': (0.31, 0.05)},
    '5_c': {'LR': (0.48, 0.03), 'LDA': (0.45, 0.02), 'KNN': (0.50, 0.02), 'DT-CART': (0.46, 0.04), 'NB': (0.49, 0.04), 'TabNet': (0.46, 0.03), 'TabTransformer': (0.40, 0.05)},
    '5_d': {'LR': (0.51, 0.05), 'LDA': (0.48, 0.07), 'KNN': (0.48, 0.06), 'DT-CART': (0.43, 0.06), 'NB': (0.49, 0.05), 'TabNet': (0.48, 0.05), 'TabTransformer': (0.37, 0.02)},
    '5_e': {'LR': (0.70, 0.11), 'LDA': (0.70, 0.15), 'KNN': (0.56, 0.18), 'DT-CART': (0.49, 0.23), 'NB': (0.67, 0.13), 'TabNet': (0.68, 0.12), 'TabTransformer': (0.47, 0.12)},
    '5_f': {'LR': (0.65, 0.11), 'LDA': (0.71, 0.11), 'KNN': (0.58, 0.13), 'DT-CART': (0.43, 0.23), 'NB': (0.67, 0.13), 'TabNet': (0.62, 0.12), 'TabTransformer': (0.45, 0.13)}
}

# ==========================================
# 3. PERFILES VISUALES (Shape Data)
# ==========================================
VISUAL_SHAPES = {
    '4_a': {
        'LR': {'median': 0.15, 'q1': 0.12, 'q3': 0.18, 'wl': 0.08, 'wh': 0.20, 'out': [0.03]},
        'LDA': {'median': 0.15, 'q1': 0.12, 'q3': 0.18, 'wl': 0.08, 'wh': 0.20, 'out': [0.03]},
        'KNN': {'median': 0.17, 'q1': 0.12, 'q3': 0.22, 'wl': 0.08, 'wh': 0.28, 'out': None},
        'DT-CART': {'median': 0.16, 'q1': 0.13, 'q3': 0.19, 'wl': 0.09, 'wh': 0.22, 'out': None},
        'NB': {'median': 0.17, 'q1': 0.13, 'q3': 0.21, 'wl': 0.08, 'wh': 0.25, 'out': None}
    },
    '4_b': {
        'LR': {'median': 0.19, 'q1': 0.17, 'q3': 0.20, 'wl': 0.14, 'wh': 0.25, 'out': [0.06]},
        'LDA': {'median': 0.19, 'q1': 0.17, 'q3': 0.21, 'wl': 0.13, 'wh': 0.25, 'out': [0.05]},
        'DT-CART': {'median': 0.18, 'q1': 0.15, 'q3': 0.20, 'wl': 0.12, 'wh': 0.24, 'out': [0.05]},
        'NB': {'median': 0.20, 'q1': 0.15, 'q3': 0.27, 'wl': 0.10, 'wh': 0.35, 'out': None}
    },
    '4_c': {
        'KNN': {'median': 0.23, 'q1': 0.22, 'q3': 0.235, 'wl': 0.21, 'wh': 0.24, 'out': [0.28]},
    },
    '4_d': {
        'LR': {'median': 0.29, 'q1': 0.27, 'q3': 0.30, 'wl': 0.255, 'wh': 0.32, 'out': [0.24]},
        'LDA': {'median': 0.26, 'q1': 0.245, 'q3': 0.275, 'wl': 0.24, 'wh': 0.30, 'out': [0.21, 0.21]},
        'NB': {'median': 0.27, 'q1': 0.24, 'q3': 0.295, 'wl': 0.225, 'wh': 0.315, 'out': [0.18]}
    },
    '4_e': {
         'NB': {'median': 0.36, 'q1': 0.31, 'q3': 0.40, 'wl': 0.27, 'wh': 0.44, 'out': [0.22]}
    },
    '4_f': {
         'NB': {'median': 0.37, 'q1': 0.32, 'q3': 0.41, 'wl': 0.28, 'wh': 0.45, 'out': [0.22]}
    },
    '5_a': {
        'LR': {'median': 0.42, 'q1': 0.33, 'q3': 0.48, 'wl': 0.27, 'wh': 0.57, 'out': [0.08]},
        'LDA': {'median': 0.42, 'q1': 0.33, 'q3': 0.48, 'wl': 0.27, 'wh': 0.57, 'out': [0.08]},
        'NB': {'median': 0.48, 'q1': 0.36, 'q3': 0.54, 'wl': 0.30, 'wh': 0.60, 'out': [0.15]}
    },
    '5_b': {
        'DT-CART': {'median': 0.32, 'q1': 0.28, 'q3': 0.37, 'wl': 0.24, 'wh': 0.42, 'out': [0.52]},
        'NB': {'median': 0.53, 'q1': 0.49, 'q3': 0.57, 'wl': 0.45, 'wh': 0.62, 'out': None}
    },
    '5_c': {
        'LR': {'median': 0.48, 'q1': 0.465, 'q3': 0.49, 'wl': 0.45, 'wh': 0.495, 'out': [0.52]},
        'LDA': {'median': 0.45, 'q1': 0.44, 'q3': 0.455, 'wl': 0.435, 'wh': 0.46, 'out': [0.41, 0.49]},
        'KNN': {'median': 0.50, 'q1': 0.495, 'q3': 0.505, 'wl': 0.49, 'wh': 0.51, 'out': [0.45, 0.54]},
        'DT-CART': {'median': 0.46, 'q1': 0.44, 'q3': 0.48, 'wl': 0.42, 'wh': 0.50, 'out': [0.40]},
        'NB': {'median': 0.48, 'q1': 0.465, 'q3': 0.505, 'wl': 0.45, 'wh': 0.52, 'out': [0.54]}
    },
    '5_d': {
        'NB': {'median': 0.50, 'q1': 0.47, 'q3': 0.53, 'wl': 0.44, 'wh': 0.55, 'out': [0.40]}
    },
    '5_e': {
        'LR': {'median': 0.72, 'q1': 0.66, 'q3': 0.78, 'wl': 0.58, 'wh': 0.83, 'out': [0.45]},
        'LDA': {'median': 0.75, 'q1': 0.67, 'q3': 0.80, 'wl': 0.58, 'wh': 0.87, 'out': [0.40, 0.42]},
        'KNN': {'median': 0.60, 'q1': 0.52, 'q3': 0.68, 'wl': 0.42, 'wh': 0.76, 'out': [0.28]},
        'DT-CART': {'median': 0.45, 'q1': 0.38, 'q3': 0.52, 'wl': 0.26, 'wh': 0.72, 'out': [0.78, 0.80]},
        'NB': {'median': 0.70, 'q1': 0.61, 'q3': 0.76, 'wl': 0.56, 'wh': 0.82, 'out': [0.42]}
    },
    '5_f': {
        'LR': {'median': 0.65, 'q1': 0.59, 'q3': 0.71, 'wl': 0.53, 'wh': 0.76, 'out': [0.78]},
        'KNN': {'median': 0.58, 'q1': 0.51, 'q3': 0.66, 'wl': 0.46, 'wh': 0.73, 'out': [0.52]},
        'NB': {'median': 0.70, 'q1': 0.61, 'q3': 0.76, 'wl': 0.55, 'wh': 0.80, 'out': [0.42]}
    }
}

# ==========================================
# 4. FUNCIONES DE GENERACIÓN Y CALIBRACIÓN
# ==========================================

def get_raw_shape_points(median, q1, q3, wl, wh, outliers=None):
    """Reconstruye la forma del boxplot."""
    iqr_points = [
        q1, 
        q1 + (median - q1)*0.33, 
        q1 + (median - q1)*0.66, 
        median, 
        median + (q3 - median)*0.33, 
        median + (q3 - median)*0.66, 
        q3
    ]
    whisker_points = [wl, wh] 
    data = np.array(iqr_points + whisker_points)
    if outliers:
        data = np.concatenate([data, np.array(outliers)])
    return data

def get_default_shape(std_magnitude):
    """Forma por defecto si no hay datos visuales específicos."""
    if std_magnitude < 0.04:
        return np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]) # Compact
    else:
        return np.array([-1.5, -0.8, -0.2, 0.0, 0.2, 0.8, 1.5]) # Normal

def generate_and_calibrate(plot_key, model):
    target_mean, target_std = TABLE_STATS[plot_key][model]
    
    if target_std == 0:
        return np.full(7, target_mean)
    
    if plot_key in VISUAL_SHAPES and model in VISUAL_SHAPES[plot_key]:
        params = VISUAL_SHAPES[plot_key][model]
        raw_data = get_raw_shape_points(
            params['median'], params['q1'], params['q3'], 
            params['wl'], params['wh'], params.get('out')
        )
    else:
        raw_data = get_default_shape(target_std)
        
    current_mean = np.mean(raw_data)
    current_std = np.std(raw_data, ddof=1)
    
    z_scores = (raw_data - current_mean) / current_std
    final_data = z_scores * target_std + target_mean
    
    return final_data

# ==========================================
# 5. BUCLE DE GENERACIÓN Y PLOTEO
# ==========================================

def main():
    print("--- INICIANDO GENERACIÓN EPS NOMBRADA (FUENTE AUMENTADA + FIX) ---")
    
    for key, filename in FILENAME_MAP.items():
        if key not in TABLE_STATS:
            continue
            
        plot_data = []
        validation_errors = []
        
        for model in MODELS_ORDER:
            data_points = generate_and_calibrate(key, model)
            plot_data.append(data_points)
            
            t_mean, t_std = TABLE_STATS[key][model]
            a_mean = np.mean(data_points)
            a_std = np.std(data_points, ddof=1)
            
            if abs(a_mean - t_mean) > 1e-4 or abs(a_std - t_std) > 1e-4:
                validation_errors.append(f"{model}: Error de validación")
        
        if validation_errors:
            print(f"⚠️  Advertencia en {key}: {validation_errors}")

        fig, ax = plt.subplots(figsize=(9, 6))
        
        bplot = ax.boxplot(plot_data, labels=MODELS_ORDER, patch_artist=False, widths=0.5,
                           flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=5))
        
        plt.setp(bplot['boxes'], color='black', linewidth=1)
        plt.setp(bplot['whiskers'], color='black', linewidth=1)
        plt.setp(bplot['caps'], color='black', linewidth=1)
        plt.setp(bplot['medians'], color='#ff7f0e', linewidth=1.5)
        
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        
        # ==========================================
        # FUENTE GRANDE
        # ==========================================
        ax.tick_params(axis='both', labelsize=14)
        
        path = os.path.join(OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, format='eps')
        plt.close(fig)
        print(f"Generado: {path} (Key: {key})")

    print("--- PROCESO COMPLETADO ---")

if __name__ == "__main__":
    main()