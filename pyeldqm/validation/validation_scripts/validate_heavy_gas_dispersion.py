import numpy as np
import math
from scipy.special import gamma
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker  # <--- 1. YEH IMPORT ADD KIYA

# ==============================================================================
#  1. CONSTANTS & COEFFICIENTS
# ==============================================================================

HEAVY_GAS_PARAMS = {
    'A': {'n': 0.108, 'sy1': 0.22, 'sy2': 0.0001, 'sz1': 0.20, 'sz2': 0.0, 'sz3': 0.0},
    'B': {'n': 0.112, 'sy1': 0.16, 'sy2': 0.0001, 'sz1': 0.12, 'sz2': 0.0, 'sz3': 0.0},
    'C': {'n': 0.120, 'sy1': 0.11, 'sy2': 0.0001, 'sz1': 0.08, 'sz2': 0.0002, 'sz3': -0.5},
    'D': {'n': 0.142, 'sy1': 0.08, 'sy2': 0.0001, 'sz1': 0.06, 'sz2': 0.0015, 'sz3': -0.5},
    'E': {'n': 0.203, 'sy1': 0.06, 'sy2': 0.0001, 'sz1': 0.03, 'sz2': 0.0003, 'sz3': -1.0},
    'F': {'n': 0.253, 'sy1': 0.04, 'sy2': 0.0001, 'sz1': 0.016, 'sz2': 0.0003, 'sz3': -1.0}
}

# Physical Constants
VON_KARMAN = 0.41      
G = 9.81               
R_GAS_CONST = 8.314    
TAMB = 298.15          
MW_AIR = 29       
MW_GAS = 64.06        # Chlorine
TEMP_GAS_INIT = 239.15 
CE_CONST = 1.15        
DELTA_L = 1.0          
H_GND_TRANSFER = 20.0  
CP_GAS = 480     
Release_time = 60 # minute

# ==============================================================================
#  2. HELPER FUNCTIONS
# ==============================================================================

def calc_rho(temp, mw):
    pressure = 101325.0  
    return (pressure * mw * 1e-3) / (R_GAS_CONST * temp)

def calc_phi(Ri_prime):
    if Ri_prime < 0: return 1.0
    return 0.88 + 0.099 * (Ri_prime**1.04) + 1.4e-25 * (Ri_prime**5.7)

def get_ustar(u_ref, z_ref, z0):
    return (VON_KARMAN * u_ref) / math.log(z_ref / z0)

def calc_sigma_w(u_star, Ri_T):
    ratio = math.sqrt(1 + Ri_T**(2/3))
    return u_star * ratio

def calc_Sy_passive(x, params):
    denom = math.sqrt(1 + params['sy2'] * x)
    Sy_base = (params['sy1'] * x) / denom
    time_factor = (Release_time / 5.0)**0.2
    return Sy_base * time_factor

# ==============================================================================
#  3. SECONDARY SOURCE BLANKET
# ==============================================================================

def calc_secondary_source(Q_source, u_star, U_ref, z_ref, n_exp, source_type, source_dims):
    rho_source = calc_rho(TEMP_GAS_INIT, MW_GAS)
    rho_a = calc_rho(TAMB, MW_AIR)
    g_prime = G * (rho_source - rho_a) / rho_a
    U_10 = U_ref * (10.0 / z_ref)**n_exp 
    
    if source_type == 'instantaneous':
        V0 = source_dims.get('volume', 1.0)
        A0 = source_dims.get('area', 1.0)
        Hb = V0 / A0
        Rb = math.sqrt(A0 / math.pi)
    elif source_type == 'puddle':
        D = source_dims.get('diameter', 10.0)
        Hb = Q_source / (rho_source * U_10 * D)
        Rb = D / 2.0
    else: 
        Hb = math.sqrt((Q_source * math.pi) / (4 * rho_source * U_10))
        Rb = 0.0 

    Hb = max(Hb, 0.1)

    Ri_star = (g_prime * Hb) / (u_star**2)
    term_temp = (TAMB - TEMP_GAS_INIT) / TEMP_GAS_INIT
    Ri_T = G * term_temp * (Hb / (u_star * U_ref)) * ((z_ref / Hb)**n_exp)
    sigma_w = calc_sigma_w(u_star, Ri_T)
    ratio_sq = (u_star / sigma_w)**2 if sigma_w > 0 else 1.0
    Ri_prime = Ri_star * ratio_sq 
    phi = calc_phi(Ri_prime)
    
    Erosion_Flux = (rho_a * VON_KARMAN * sigma_w * (1 + n_exp)) / phi
    
    if source_type == 'continuous':
        if Erosion_Flux <= 1e-6:
            Rb = 50.0 
        else:
            Area = Q_source / Erosion_Flux
            Rb = math.sqrt(Area / math.pi)
    elif source_type == 'puddle':
        calc_Area = Q_source / Erosion_Flux if Erosion_Flux > 1e-6 else 99999
        phys_Area = math.pi * (Rb**2)
        if calc_Area > phys_Area:
             Rb = math.sqrt(calc_Area / math.pi)

    return Rb, Hb

# ==============================================================================
#  4. DISPERSION PHYSICS ENGINE
# ==============================================================================

def run_heavy_gas_model(stab_class, Q_kg_s, U_ref, z_ref, z0, source_cfg):
    params = HEAVY_GAS_PARAMS[stab_class]
    n_exp = params['n']
    gamma_term = gamma(1.0 / (1.0 + n_exp))
    rho_a = calc_rho(TAMB, MW_AIR)
    u_star = get_ustar(U_ref, z_ref, z0)
    
    Rb, Hb = calc_secondary_source(
        Q_kg_s, u_star, U_ref, z_ref, n_exp, 
        source_cfg['type'], source_cfg['dims']
    )
    
    def derivatives(x, state):
        Sz, Beff, Tc, Flux = state
        Sz = max(Sz, 0.1)
        Beff = max(Beff, 0.1)
        Tc = min(Tc, TAMB) 
        Flux = max(Flux, 1e-6)

        w_c = Q_kg_s / Flux 
        if w_c > 1.0: w_c = 1.0
        
        inv_MW_mix = (w_c / MW_GAS) + ((1 - w_c) / MW_AIR)
        MW_mix = 1.0 / inv_MW_mix
        rho_c = calc_rho(Tc, MW_mix) 

        Heff = (Sz / (1.0 + n_exp)) * gamma_term
        g_prime = G * (rho_c - rho_a) / rho_a
        U_eff = (U_ref / gamma_term) * ((Sz / z_ref)**n_exp)
        
        Ri_star = (g_prime * Heff) / (u_star**2)
        term_temp = max(0, (TAMB - Tc) / Tc)
        Ri_T = G * term_temp * (Heff / (u_star * U_ref)) * ((z_ref / Heff)**n_exp)
        sigma_w = calc_sigma_w(u_star, Ri_T)
        Ri_prime = Ri_star * (u_star / sigma_w)**2
        phi = calc_phi(max(0.0, Ri_prime))
        
        scaling_factor = (1.0 + n_exp) / gamma_term
        dSz_dx = (scaling_factor * VON_KARMAN * u_star) / (U_eff * phi)
        
        term_gravity = 0.0
        if g_prime > 0 and Ri_prime > 1.0:
            part_a = CE_CONST * gamma_term * ((z_ref / Sz)**n_exp)
            part_b = math.sqrt(g_prime * Heff) / U_ref
            term_gravity = part_a * part_b
            
        denom = math.sqrt(1 + params['sy2'] * x)
        dSy_dx = (params['sy1'] / denom) - (params['sy1'] * x * params['sy2']) / (2 * denom**3)
        term_passive = (math.sqrt(math.pi) / 2.0) * dSy_dx
        dBeff_dx = term_gravity + term_passive
        
        entrainment_rate = (rho_a * VON_KARMAN * sigma_w * (1 + n_exp)) / phi
        dMassFlux_dx = entrainment_rate * (2 * Beff)
        
        F_H = H_GND_TRANSFER * (TAMB - Tc) 
        dT_dilution = (dMassFlux_dx / Flux) * (TAMB - Tc)
        dT_heating = ((F_H * (2 * Beff)) / DELTA_L) / (Flux * CP_GAS)
        dTc_dx = dT_dilution + dT_heating
        
        return [dSz_dx, dBeff_dx, dTc_dx, dMassFlux_dx]

    Sz_init = Hb * (1.0 + n_exp) / gamma_term
    
    y0 = [Sz_init, Rb, TEMP_GAS_INIT, Q_kg_s]
    sol = solve_ivp(fun=derivatives, t_span=(Rb, 12000), y0=y0, method='RK45', max_step=10.0)
    
    return sol, n_exp, gamma_term, U_ref, z_ref

# ==============================================================================
#  5. ALOHA-STYLE 2D PLOTTING (WITH TEXT REPORT & BLACK GRIDS)
# ==============================================================================

if __name__ == "__main__":
    # --- Settings ---
    STABILITY = 'D'        
    WIND_SPEED = 5.0       
    Z_REF = 3.0           
    Z0 = 0.03              
    SOURCE_RATE = 3.0      # kg/s SO2
    
    source_config = {'type': 'continuous', 'dims': {'diameter': 0.1}}
    
    print("Running Physics Engine (Calculations in Meters)...")
    sol, n, gamma_val, U_ref, z_ref = run_heavy_gas_model(
        STABILITY, SOURCE_RATE, WIND_SPEED, Z_REF, Z0, source_config
    )
    
    print("Analyzing Threat Zones...")
    
    # --- CALCULATE EXACT DISTANCES FOR TEXT SUMMARY ---
    METERS_TO_MILES = 0.000691371
    
    # AEGL Limits for chlorine (ppm)
    limits = {'AEGL-3': 30, 'AEGL-2': 0.75, 'AEGL-1': 0.2}
    zone_distances = {}

    # Check centerline concentration to find max distance
    x_dist = sol.t
    centerline_ppm = []
    
    for i, x in enumerate(x_dist):
        Sz = sol.y[0][i]
        Beff = sol.y[1][i]
        Tc = sol.y[2][i]
        Flux = sol.y[3][i]
        
        Heff = (Sz / (1.0 + n)) * gamma_val
        U_eff = (U_ref / gamma_val) * ((Sz / z_ref)**n)
        
        w_c = SOURCE_RATE / Flux 
        if w_c > 1.0: w_c = 1.0
        inv_MW_mix = (w_c / MW_GAS) + ((1 - w_c) / MW_AIR)
        MW_mix = 1.0 / inv_MW_mix
        rho_mix = calc_rho(Tc, MW_mix)
        
        conc_kgm3 = SOURCE_RATE / (2 * Beff * Heff * U_eff)
        ppm = (conc_kgm3 / rho_mix) * (MW_AIR / MW_GAS) * 1e6
        centerline_ppm.append(ppm)
    
    centerline_ppm = np.array(centerline_ppm)
    
    # Find cutoff distance for each limit
    print("\n" + "="*40)
    print("   ALOHA THREAT ZONE TEXT SUMMARY")
    print("="*40)
    print(f"Chemical: Chlorine (MW: {MW_GAS})")
    print(f"Wind: {WIND_SPEED} m/s | Stability: {STABILITY}")
    print("-" * 40)
    
    for name, limit in limits.items():
        # Find index where ppm drops below limit
        indices = np.where(centerline_ppm > limit)[0]
        if len(indices) > 0:
            max_idx = indices[-1]
            dist_meters = x_dist[max_idx]
            dist_miles = dist_meters * METERS_TO_MILES
            print(f"{name} ({limit} ppm) : {dist_miles:.2f} miles ({dist_meters:.0f} m)")
        else:
            print(f"{name} ({limit} ppm) : < 10 meters")
            
    print("="*40 + "\n")

    # --- PLOTTING ---
    print("Generating 2D Footprint...")
    x_line = sol.t
    y_max_plot = 1000 
    y_line = np.linspace(-y_max_plot, y_max_plot, 500) 
    X_grid, Y_grid = np.meshgrid(x_line, y_line)
    Z_grid = np.zeros_like(X_grid)
    
    params = HEAVY_GAS_PARAMS[STABILITY]
    
    for i, x in enumerate(x_line):
        Sz = sol.y[0][i]
        Beff = sol.y[1][i]
        Tc = sol.y[2][i]
        Flux = sol.y[3][i]
        Heff = (Sz / (1.0 + n)) * gamma_val
        U_eff = (U_ref / gamma_val) * ((Sz / z_ref)**n)
        w_c = SOURCE_RATE / Flux 
        if w_c > 1.0: w_c = 1.0
        inv_MW_mix = (w_c / MW_GAS) + ((1 - w_c) / MW_AIR)
        MW_mix = 1.0 / inv_MW_mix
        rho_mix = calc_rho(Tc, MW_mix)
        conc_kgm3_center = SOURCE_RATE / (2 * Beff * Heff * U_eff)
        ppm_center = (conc_kgm3_center / rho_mix) * (MW_AIR / MW_GAS) * 1e6
        
        Sy_passive = calc_Sy_passive(x, params)
        b_core = Beff - (math.sqrt(math.pi) / 2.0) * Sy_passive
        if b_core < 0: b_core = 0
        
        for j, y in enumerate(y_line):
            abs_y = abs(y)
            if abs_y <= b_core:
                Z_grid[j, i] = ppm_center
            else:
                if Sy_passive > 0:
                    exponent = -((abs_y - b_core) / Sy_passive)**2
                    Z_grid[j, i] = ppm_center * math.exp(exponent)
                else:
                    Z_grid[j, i] = 0

    # --- PLOTTING IN MILES ---
    X_grid_miles = X_grid * METERS_TO_MILES
    Y_grid_miles = Y_grid * METERS_TO_MILES
    
    plt.figure(figsize=(10, 6))
    
    levels = [0.2, 0.75, 30, 1000000] 
    colors = ['yellow', 'orange', 'red']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    
    contour = plt.contourf(X_grid_miles, Y_grid_miles, Z_grid, levels=levels, cmap=cmap, norm=norm, extend='max')
    
    red_patch = mpatches.Patch(color='red', label='AEGL-3 (>= 30 ppm)')
    org_patch = mpatches.Patch(color='orange', label='AEGL-2 (>= 0.75 ppm)')
    yel_patch = mpatches.Patch(color='yellow', label='AEGL-1 (>= 0.2 ppm)')
    
    # --- AXIS FORMATTING (Grid & Ticks) ---
    
    # 1. Y-Axis Positive Only
    ax = plt.gca()
    def abs_formatter(x, pos):
        return f"{abs(x):.1f}"
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(abs_formatter))

    # 2. X-Axis Grid har 2 Miles baad
    plt.xticks(np.arange(0, 10, 2))  # <--- 2. TICKS HAR 2 MILES PAR
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)

    plt.title(f" Threat Zone Estimate\nSource: {SOURCE_RATE} kg/s Sulfur dioxide | Wind: {WIND_SPEED} m/s | Class: {STABILITY}")
    plt.xlabel("miles")
    plt.ylabel("miles")
    
    # Limits in Miles
    plt.ylim(-3.0, 3.0)
    plt.xlim(0, 8.0) 
    
    plt.legend(handles=[red_patch, org_patch, yel_patch], loc='upper right')
    
    # 3. BLACK GRID LINES
    plt.grid(True, linestyle='-', alpha=0.5, color='black') # <--- 3. BLACK GRID
    
    plt.tight_layout()
    plt.show()

    print("Plot generated successfully")
