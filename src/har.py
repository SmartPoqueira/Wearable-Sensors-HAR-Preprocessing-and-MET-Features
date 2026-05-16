import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker # Importar el módulo ticker

data = [
    [1.000, -0.208, 0.566, -0.038, -0.391, 0.418, -0.110, 0.001, -0.220, -0.036, -0.038, -0.251],
    [-0.208, 1.000, -0.543, 0.700, 0.457, -0.096, 0.367, -0.001, 0.508, -0.001, 0.700, 0.626],
    [0.566, -0.543, 1.000, -0.448, -0.170, 0.000, -0.001, 0.270, -0.460, -0.031, -0.448, -0.536],
    [-0.038, 0.700, -0.448, 1.000, 0.204, 0.000, 0.000, 0.000, 0.621, -0.021, 1.000, 0.775],
    [-0.391, 0.457, -0.170, 0.204, 1.000, -0.727, 0.711, 0.090, 0.299, 0.002, 0.204, 0.199],
    [0.418, -0.096, 0.000, 0.000, -0.727, 1.000, -0.268, -0.006, -0.105, 0.000, 0.000, 0.000],
    [-0.110, 0.367, -0.001, 0.000, 0.711, -0.268, 1.000, -0.004, 0.051, 0.000, 0.000, 0.000],
    [0.001, -0.001, 0.270, 0.000, 0.090, -0.006, -0.004, 1.000, 0.008, 0.000, 0.000, 0.000],
    [-0.220, 0.508, -0.460, 0.621, 0.299, -0.105, 0.051, 0.008, 1.000, -0.027, 0.621, 0.749],
    [-0.036, -0.001, -0.031, -0.021, 0.002, 0.000, 0.000, 0.000, -0.027, 1.000, -0.021, -0.032],
    [-0.038, 0.700, -0.448, 1.000, 0.204, 0.000, 0.000, 0.000, 0.621, -0.021, 1.000, 0.775],
    [-0.251, 0.626, -0.536, 0.775, 0.199, 0.000, 0.000, 0.000, 0.749, -0.032, 0.775, 1.000]
]
labels = ['x', 'y', 'z', 'HR', 'SVMg', 'xfil', 'yfil', 'zfil', 'SVMgfil', 'RUF', '%HRR', 'activity']
df_corr = pd.DataFrame(data, columns=labels, index=labels)

annot_df = df_corr.copy().astype(object)
for col in df_corr.columns:
    for idx in df_corr.index:
        val = df_corr.loc[idx, col]
        if abs(val - 1.0) < 1e-9:
            annot_df.loc[idx, col] = "1"
        else:
            annot_df.loc[idx, col] = "{:.3f}".format(val)

sns.set_theme(style="white")

FONT_SIZE_NUMEROS = 16
FONT_SIZE_EJES = 16.5

fig = plt.figure(figsize=(14, 13))

gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.06)

ax_heatmap = plt.subplot(gs[0])
ax_cbar = plt.subplot(gs[1])

sns.heatmap(df_corr,
            annot=annot_df,
            fmt="",
            cmap="coolwarm",
            center=0,
            vmin=-0.75,
            vmax=1.0,
            square=True,
            linewidths=0,
            annot_kws={"size": FONT_SIZE_NUMEROS},
            ax=ax_heatmap,
            cbar_ax=ax_cbar
            )

ax_heatmap.tick_params(axis='x', labelsize=FONT_SIZE_EJES)

ax_heatmap.tick_params(axis='y', labelsize=FONT_SIZE_EJES, rotation=90)

# *** MODIFICACIÓN PARA FORMATO DE 1 DECIMAL EN LA BARRA DE COLOR ***
ax_cbar.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax_cbar.tick_params(labelsize=FONT_SIZE_EJES)
# ******************************************************************

plt.tight_layout()

nombre_archivo = "Fig3_M008corrmatrix.eps"
plt.savefig(nombre_archivo, format='eps', dpi=300, bbox_inches='tight')

print(f"Gráfico generado con etiquetas del eje Y giradas y formato de 1 decimal en la barra de color. Guardado como '{nombre_archivo}'")
