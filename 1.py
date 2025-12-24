import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.patches import Rectangle

# --- Вхідні параметри ---
n_manning = 0.013
phi = 0.95
# Словник інтенсивностей: {Q20 (л/с*га) : q (м/с)}
q_scenarios = {
    104.0: 0.0000104,
    166.7: 0.00001667
}

# ДБН параметри
mr = 143
gamma_exp = 1.82
P = 5
n_exp = 0.71

# Геометрія
L_drain = 6.0
i_trans = 0.025
W_list = np.arange(4.5, 18.0, 1.0) 
i_long_list = np.arange(0.0, 4.1, 0.1) / 100 
span_lengths = [40, 80, 120]
delta_i_deflection = 0.005 # Зменшення ухилу через прогин (0.5%)

# --- Допоміжні функції ---
def get_dbn_params(q20_val, P_val, L_val, W_val):
    t_con = 2.0
    v_avg_surface = 0.5
    t_can = (L_val / v_avg_surface) / 60
    tr = t_con + t_can
    A = q20_val * (20**n_exp) * (1 + np.log10(P_val) / np.log10(mr))**gamma_exp
    
    if A <= 300: z_mid = 0.33
    elif A <= 400: z_mid = 0.31
    elif A <= 500: z_mid = 0.30
    elif A <= 600: z_mid = 0.29
    elif A <= 700: z_mid = 0.28
    elif A <= 800: z_mid = 0.27
    elif A <= 1000: z_mid = 0.26
    elif A <= 1200: z_mid = 0.25
    else: z_mid = 0.24
    
    F_ha = (L_val * W_val) / 10000
    gamma_dist = 1.0 if F_ha < 500 else 0.95
    return A, z_mid, gamma_dist, tr, n_exp

def calculate_max_depth_manning(q_int, W, L, i_long):
    i_safe = max(i_long, 0.00001) 
    Q = phi * q_int * L * W
    h = (Q * n_manning / (W * (i_safe**0.5)))**0.6
    return max(h, 0)

def calculate_max_depth_dbn(q20_val, W, L, i_long):
    A, z_mid, gamma_dist, tr, n_val = get_dbn_params(q20_val, P, L, W)
    F_ha_total = (L * W) / 10000
    qr_peak = z_mid * A * F_ha_total * gamma_dist / (tr**n_val)
    Q_peak = qr_peak / 1000
    i_safe = max(i_long, 0.00001)
    h = (Q_peak * n_manning / (W * (i_safe**0.5)))**0.6
    return max(h, 0)

def calculate_depth_visualization(X, Y, q_manning_val, q20_val, W_val, L_val, i_l, method='manning'):
    """
    Розраховує профіль глибини для візуалізації (без врахування i_trans).
    """
    i_safe = np.maximum(i_l, 0.00001)
    
    if method == 'manning':
        Q_at_x = phi * q_manning_val * X * W_val 
        h_x = (Q_at_x * n_manning / (W_val * (i_safe**0.5)))**0.6
    else: # dbn
        A, z_mid, gamma_dist, tr, n_val = get_dbn_params(q20_val, P, L_val, W_val)
        F_ha = (L_val * W_val) / 10000
        qr_peak = z_mid * A * F_ha * gamma_dist / (tr**n_val)
        Q_peak = qr_peak / 1000
        Q_at_x = Q_peak * (X / L_val)
        h_x = (Q_at_x * n_manning / (W_val * (i_safe**0.5)))**0.6
    
    return np.maximum(h_x, 0)

# --- Генерація Таблиці (Excel) ---
print("Розрахунок повної таблиці...")
results_data = []

for q20_val, q_man_val in q_scenarios.items():
    for L_span in span_lengths:
        for W in W_list:
            for i_long_design in i_long_list:
                # Unloaded
                i_eff_unloaded = i_long_design
                h_man = calculate_max_depth_manning(q_man_val, W, L_drain, i_eff_unloaded) * 1000
                h_dbn = calculate_max_depth_dbn(q20_val, W, L_drain, i_eff_unloaded) * 1000
                results_data.append({
                    "Q": q20_val, "Span": L_span, "W": W, "State": "Unloaded",
                    "i_des%": round(i_long_design*100, 2), "i_eff%": round(i_eff_unloaded*100, 2),
                    "h_man": h_man, "h_dbn": h_dbn
                })
                # Loaded
                i_eff_loaded = max(i_long_design - delta_i_deflection, 0.00001)
                h_man = calculate_max_depth_manning(q_man_val, W, L_drain, i_eff_loaded) * 1000
                h_dbn = calculate_max_depth_dbn(q20_val, W, L_drain, i_eff_loaded) * 1000
                results_data.append({
                    "Q": q20_val, "Span": L_span, "W": W, "State": "Loaded",
                    "i_des%": round(i_long_design*100, 2), "i_eff%": round(i_eff_loaded*100, 2),
                    "h_man": h_man, "h_dbn": h_dbn
                })

df = pd.DataFrame(results_data)
df.to_excel("bridge_drainage_results.xlsx", index=False)
print("Таблицю збережено: bridge_drainage_results.xlsx")

# --- Генерація Графіків ---
print("Генерація графіків (3D та 2D)...")

# Ваш оновлений список ухилів
i_plots_pct = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

W_plot = 15.0 
L_plot_geom = L_drain

# Сітка
x = np.linspace(0, L_plot_geom, 50)
y = np.linspace(0, W_plot, 50)
X, Y = np.meshgrid(x, y)

# *** ГОЛОВНИЙ ЦИКЛ ПО ІНТЕНСИВНОСТЯХ (Q=104 та Q=166.7) ***
for q20_key, q_man_val in q_scenarios.items():
    print(f"Обробка сценарію Q={q20_key}...")
    
    for i_val in i_plots_pct:
        i_des = i_val / 100.0
        # Навантажений стан:
        i_eff = max(i_des - delta_i_deflection, 0.00001)
        
        # Розрахунок (мм)
        Z_man = calculate_depth_visualization(X, Y, q_man_val, q20_key, W_plot, L_plot_geom, i_eff, 'manning') * 1000
        Z_dbn = calculate_depth_visualization(X, Y, q_man_val, q20_key, W_plot, L_plot_geom, i_eff, 'dbn') * 1000
        
        # === 1. Побудова 3D Графіка (Суцільні кольори) ===
        fig_3d = plt.figure(figsize=(12, 9))
        ax = fig_3d.add_subplot(111, projection='3d')
        
        surf1 = ax.plot_surface(X, Y, Z_man, color='deepskyblue', alpha=0.6, shade=True)
        surf2 = ax.plot_surface(X, Y, Z_dbn, color='salmon', alpha=0.6, shade=True)
        
        ax.set_xlabel('Length of drain L (m)', fontsize=11, labelpad=10)
        ax.set_ylabel('Width of bridge W (m)', fontsize=11, labelpad=10)
        ax.set_zlabel('Depth (mm)', fontsize=11, labelpad=10)
        
        ax.set_title(f'3D Comparison: Manning vs DBN (Q={q20_key})\n'
                     f'Design Slope i={i_val}%  ->  Effective Slope i={i_eff*100:.2f}% (Loaded)', fontsize=14)
        
        p1 = Rectangle((0, 0), 1, 1, fc='deepskyblue', alpha=0.6)
        p2 = Rectangle((0, 0), 1, 1, fc='salmon', alpha=0.6)
        ax.legend([p1, p2], ['Manning Method', 'DBN Method'], loc='upper left', fontsize=12)
        
        # Ім'я файлу включає Q
        fname_3d = f'3D_Solid_Q{int(q20_key)}_i_{i_val}pct.png'
        plt.savefig(fname_3d)
        plt.close()

        # === 2. Побудова 2D Графіків (Профілі) ===
        h_man_prof = Z_man[0, :]
        h_dbn_prof = Z_dbn[0, :]
        x_prof = x
        
        fig_2d, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(x_prof, h_man_prof, color='deepskyblue', linewidth=2.5, label='Manning')
        ax1.plot(x_prof, h_dbn_prof, color='salmon', linestyle='--', linewidth=2.5, label='DBN')
        ax1.set_ylabel('Depth h (mm)', fontsize=12)
        ax1.set_xlabel('Drain Length L (m)', fontsize=12)
        ax1.set_title(f'2D Hydraulic Profile (Q={q20_key}, Design i={i_val}%, Eff i={i_eff*100:.2f}%)', fontsize=13)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.legend(fontsize=11)
        
        delta = h_dbn_prof - h_man_prof
        ax2.plot(x_prof, delta, color='black', linewidth=1.5)
        ax2.fill_between(x_prof, delta, 0, where=(delta>0), color='salmon', alpha=0.3, label='DBN > Manning')
        ax2.fill_between(x_prof, delta, 0, where=(delta<0), color='deepskyblue', alpha=0.3, label='Manning > DBN')
        ax2.set_ylabel('Difference (mm)', fontsize=12)
        ax2.set_xlabel('Drain Length L (m)', fontsize=12)
        ax2.set_title('Difference (H_DBN - H_Manning)', fontsize=13)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        fname_2d = f'2D_Profile_Q{int(q20_key)}_i_{i_val}pct.png'
        plt.savefig(fname_2d)
        plt.close()
        
    print(f"Графіки для Q={q20_key} завершено.")