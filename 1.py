import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

# --- Налаштування сторінки ---
st.set_page_config(page_title="Гідравлічний розрахунок", layout="wide")

st.title("Гідравлічний розрахунок водовідведення")
st.markdown("Порівняння методів: **Формула Маннінга** vs **Методика ДБН**")

# --- Вхідні параметри (Sidebar) ---
st.sidebar.header("Вхідні дані")

# 1. Інтенсивність опадів
# Словник інтенсивностей: {Відображення : (Q20, q)}
# Ви можете розширити цей список у майбутньому
precip_options = {
    "Варіант 1 (Q20=104 л/с*га)": (104.0, 0.0000104),
    "Варіант 2 (Q20=166.7 л/с*га)": (166.7, 0.00001667)
}

selected_precip = st.sidebar.selectbox("Інтенсивність опадів", list(precip_options.keys()))
q20_val, q_man_val = precip_options[selected_precip]

# 2. Геометрія
st.sidebar.subheader("Геометрія водозбору")
L_drain = st.sidebar.number_input("Довжина водозбору L (м)", value=6.0, step=0.5, min_value=0.1)
W_bridge = st.sidebar.number_input("Ширина водозбору W (м)", value=15.0, step=0.5, min_value=0.1)
i_long_pct = st.sidebar.number_input("Поздовжній ухил (%)", value=2.0, step=0.1, min_value=0.0)
i_trans_pct = st.sidebar.number_input("Поперечний ухил (%)", value=2.5, step=0.1, min_value=0.0)

# Переведення відсотків у частки
i_long = i_long_pct / 100.0
i_trans = i_trans_pct / 100.0

# Константи
n_manning = 0.013
phi = 0.95
mr = 143
gamma_exp = 1.82
P = 5
n_exp = 0.71

# --- Функції розрахунку ---
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

def calculate_max_depth_manning(q_int, W, L, i_l):
    i_safe = max(i_l, 0.00001) 
    Q = phi * q_int * L * W
    h = (Q * n_manning / (W * (i_safe**0.5)))**0.6
    return max(h, 0)

def calculate_max_depth_dbn(q20_val, W, L, i_l):
    A, z_mid, gamma_dist, tr, n_val = get_dbn_params(q20_val, P, L, W)
    F_ha_total = (L * W) / 10000
    qr_peak = z_mid * A * F_ha_total * gamma_dist / (tr**n_val)
    Q_peak = qr_peak / 1000
    i_safe = max(i_l, 0.00001)
    h = (Q_peak * n_manning / (W * (i_safe**0.5)))**0.6
    return max(h, 0)

def calculate_depth_array(X, q_val, q20_val, W_val, L_val, i_l, method='manning'):
    """Розрахунок масиву глибин для графіків"""
    i_safe = np.maximum(i_l, 0.00001)
    if method == 'manning':
        Q_at_x = phi * q_val * X * W_val 
        h_x = (Q_at_x * n_manning / (W_val * (i_safe**0.5)))**0.6
    else: # dbn
        A, z_mid, gamma_dist, tr, n_val = get_dbn_params(q20_val, P, L_val, W_val)
        F_ha = (L_val * W_val) / 10000
        qr_peak = z_mid * A * F_ha * gamma_dist / (tr**n_val)
        Q_peak = qr_peak / 1000
        Q_at_x = Q_peak * (X / L_val)
        h_x = (Q_at_x * n_manning / (W_val * (i_safe**0.5)))**0.6
    return np.maximum(h_x, 0)

# --- Основна логіка ---

if st.sidebar.button("Розрахувати", type="primary"):
    
    # 1. Розрахунок максимальних глибин
    h_max_man = calculate_max_depth_manning(q_man_val, W_bridge, L_drain, i_long) * 1000
    h_max_dbn = calculate_max_depth_dbn(q20_val, W_bridge, L_drain, i_long) * 1000
    
    # 2. Виведення числових результатів
    st.subheader("Результати розрахунку (максимальна глибина)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="За Маннінгом", value=f"{h_max_man:.2f} мм")
    with col2:
        st.metric(label="За ДБН", value=f"{h_max_dbn:.2f} мм")
        delta = h_max_dbn - h_max_man
    with col3:
        st.metric(label="Різниця (ДБН - Маннінг)", value=f"{delta:.2f} мм", delta_color="off")

    st.markdown("---")

    # 3. Підготовка даних для графіків
    x_plot = np.linspace(0, L_drain, 50)
    y_plot = np.linspace(0, W_bridge, 50)
    X, Y = np.meshgrid(x_plot, y_plot)
    
    # Розрахунок поверхонь
    Z_man = calculate_depth_array(X, q_man_val, q20_val, W_bridge, L_drain, i_long, 'manning') * 1000
    Z_dbn = calculate_depth_array(X, q_man_val, q20_val, W_bridge, L_drain, i_long, 'dbn') * 1000

    # 4. Графіки
    tab1, tab2 = st.tabs(["2D Профіль", "3D Візуалізація"])

    with tab1:
        st.subheader(f"Профіль глибини вздовж довжини L={L_drain}м")
        
        # Беремо профіль (перший рядок масиву, оскільки вздовж X глибина змінюється, вздовж Y у цій моделі - константа для перерізу)
        h_man_prof = Z_man[0, :]
        h_dbn_prof = Z_dbn[0, :]
        
        fig_2d, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(x_plot, h_man_prof, color='deepskyblue', linewidth=2.5, label='Manning')
        ax1.plot(x_plot, h_dbn_prof, color='salmon', linestyle='--', linewidth=2.5, label='DBN')
        
        ax1.set_ylabel('Глибина h (мм)')
        ax1.set_xlabel('Довжина водозбору L (м)')
        ax1.set_title(f'Гідравлічний профіль (Ухил {i_long_pct}%)')
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.legend()
        
        st.pyplot(fig_2d)

    with tab2:
        st.subheader("3D Порівняння поверхонь води")
        fig_3d = plt.figure(figsize=(10, 8))
        ax = fig_3d.add_subplot(111, projection='3d')
        
        surf1 = ax.plot_surface(X, Y, Z_man, color='deepskyblue', alpha=0.6, shade=True)
        surf2 = ax.plot_surface(X, Y, Z_dbn, color='salmon', alpha=0.6, shade=True)
        
        ax.set_xlabel('Довжина L (м)')
        ax.set_ylabel('Ширина W (м)')
        ax.set_zlabel('Глибина (мм)')
        
        p1 = Rectangle((0, 0), 1, 1, fc='deepskyblue', alpha=0.6)
        p2 = Rectangle((0, 0), 1, 1, fc='salmon', alpha=0.6)
        ax.legend([p1, p2], ['Manning', 'DBN'], loc='upper left')
        
        st.pyplot(fig_3d)

else:
    st.info("Введіть параметри зліва та натисніть 'Розрахувати'")