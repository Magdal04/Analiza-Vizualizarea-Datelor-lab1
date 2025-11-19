import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurare pentru calitate profesională
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'

# Creare directoare necesare
Path("lab1/graphs").mkdir(parents=True, exist_ok=True)

def load_data():
    """Încarcă și pregătește datele inițiale"""
    print("=== ÎNCĂRCARE DATE ===")
    
    try:
        df = pd.read_csv("lab1/raw_energy_data.csv", parse_dates=["date"])
        print(f"Date încărcate: {df.shape[0]} observații, {df.shape[1]} variabile")
        return df
    except FileNotFoundError:
        print("✗ Eroare: Fișierul 'lab1/raw_energy_data.csv' nu a fost găsit")
        print("  Asigură-te că fișierul există în directorul corect")
        exit()
    except Exception as e:
        print(f"✗ Eroare la încărcare: {e}")
        exit()

def clean_data(df):
    """Curăță și prelucrează datele"""
    print("\n=== CURĂȚARE DATE ===")
    
    # Verificare valori lipsă
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Înlocuire {missing.sum()} valori lipsă prin interpolare")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate()
    
    # Eliminare duplicate
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Eliminare {duplicates} duplicate")
        df = df.drop_duplicates()
    
    return df

def engineer_features(df):
    """Creează variabile noi pentru analiză"""
    print("Creare variabile derivate...")
    
    # Variabile temporale
    df['an'] = df["date"].dt.year
    df['luna'] = df["date"].dt.month
    df['ziua'] = df['date'].dt.day
    df['ora'] = df["date"].dt.hour
    df['ziua_saptamanii'] = df["date"].dt.day_name()
    df['weekend'] = df["date"].dt.day_of_week >= 5
    df['trimestru'] = df['date'].dt.quarter
    
    # Variabile energetice
    df['consum_acoperit'] = (df['productie'] / df["consum"]).round(3)
    df['total_regenerabila'] = df[['hidro', 'fotovolt', 'eolian', 'biomasa']].sum(axis=1)
    df['procent_regenerabila'] = (df['total_regenerabila'] / df['productie'] * 100).round(2)
    df['sold_absolut'] = df['sold'].abs()
    
    return df

def create_visualizations(df):
    """Generează toate vizualizările"""
    print("\n=== GENERARE VIZUALIZĂRI ===")
    
    # 1. Producție solară pe ore
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.scatter(df['ora'], df['fotovolt'], alpha=0.6, s=20)
    ax.set_xlabel("Ora zilei", fontsize=12)
    ax.set_ylabel("Energie Solară (MW)", fontsize=12)
    ax.set_title("Distribuția Producției Solare pe Ore", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.savefig("lab1/graphs/solar_energy_distribution.png")
    plt.tight_layout()
    plt.show()

    # 2. Heatmap producție pe intervale orare
    df['interval_3h'] = (df['ora'] // 3) * 3
    surse_energie = ['carbune', 'hidro', 'hidrocarburi', 'nuclear', 'eolian', 'fotovolt', 'biomasa']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=100)
    
    prod_interval = df.groupby('interval_3h')[surse_energie].mean()
    sns.heatmap(prod_interval.T, annot=True, cmap='YlOrRd', fmt='.0f', ax=ax1, cbar_kws={'label': 'MW'})
    ax1.set_title("Producție Medie pe Intervale de 3 Ore", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Interval Orar")
    ax1.set_ylabel("Sursă de Energie")
    
    prod_ora = df.groupby('ora')[surse_energie].mean()
    sns.heatmap(prod_ora.T, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'MW'})
    ax2.set_title("Producție Medie pe Ore", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Ora")
    ax2.set_ylabel("Sursă de Energie")
    
    plt.savefig("lab1/graphs/heatmap_production_patterns.png")
    plt.tight_layout()
    plt.show()

    # 3. Producție lunară pe surse
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    prod_lunara = df.groupby(['an','luna'])[surse_energie].mean()
    prod_lunara.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="Sursă Energie", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    luni_abrv = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun', 'Iul', 'Aug', 'Sep', 'Oct', 'Noi', 'Dec']
    labels = [f"{an}-{luni_abrv[luna-1]}" for an, luna in prod_lunara.index]
    
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel("Perioadă (An-Lună)", fontsize=12)
    ax.set_ylabel("Producție Medie (MW)", fontsize=12)
    ax.set_title("Compoziția Producției Energetice Lunare", fontsize=14, fontweight='bold')
    plt.savefig("lab1/graphs/monthly_production_composition.png")
    plt.tight_layout()
    plt.show()

    # 4. Sold energetic zilnic comparativ
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    sold_zilnic = df.groupby(['an', 'ziua'])['sold'].mean().reset_index()
    
    for an in [2024, 2025]:
        date_an = sold_zilnic[sold_zilnic['an'] == an]
        ax.plot(date_an['ziua'], date_an['sold'], label=f'An {an}', marker='o', linewidth=2, markersize=4)
    
    ax.set_xlabel('Ziua Lunii', fontsize=12)
    ax.set_ylabel('Sold Energetic (MW)', fontsize=12)
    ax.set_title('Soldul Energetic Zilnic Mediu: 2024 vs 2025', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig("lab1/graphs/daily_energy_balance_comparison.png")
    plt.tight_layout()
    plt.show()

    # 5. Seria temporală a soldului (netezită)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    df_sorted = df.sort_values('date')
    
    ax.plot(df_sorted['date'], df_sorted['sold'], linewidth=1.5, alpha=0.8, color='tab:blue')
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Sold Energetic (MW)', fontsize=12)
    ax.set_title('Evoluția Soldului Energetic (Serie Netezită)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    ax.axhline(y=0, color='red')
    plt.savefig("lab1/graphs/energy_balance_timeseries.png")
    plt.tight_layout()
    plt.show()

    # 6. Profilul producției pe ore
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    prod_ora = df.groupby('ora')['productie'].mean()
    
    ax.plot(prod_ora.index, prod_ora.values, marker='o', linewidth=2.5, markersize=6, 
            color='crimson', markerfacecolor='white', markeredgewidth=2)
    ax.set_xlabel('Ora Zilei', fontsize=12)
    ax.set_ylabel('Producție Medie (MW)', fontsize=12)
    ax.set_title('Profilul Orar al Producției Energetice', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    plt.savefig("lab1/graphs/hourly_production_profile.png")
    plt.tight_layout()
    plt.show()

    # 7. Consum pe zilele săptămânii
    fig, ax = plt.subplots(figsize=(10, 6))
    zile_ordine = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    zile_ro = ['Luni', 'Marți', 'Miercuri', 'Joi', 'Vineri', 'Sâmbătă', 'Duminică']
    
    consum_zile = df.groupby('ziua_saptamanii')['consum'].mean().reindex(zile_ordine)
    
    bars = ax.bar(zile_ro, consum_zile.values, color='skyblue', edgecolor='navy', alpha=0.8)
    ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=10)
    
    ax.set_xlabel('Ziua Săptămânii', fontsize=12)
    ax.set_ylabel('Consum Mediu (MW)', fontsize=12)
    ax.set_title('Consumul Energetic Mediu pe Zilele Săptămânii', fontsize=14, fontweight='bold')
    ax.set_ylim(5000, None)
    plt.xticks(rotation=45)
    plt.savefig("lab1/graphs/weekly_consumption_pattern.png")
    plt.tight_layout()
    plt.show()

    # 8. Comparație lunară consum vs producție
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Zilnic
    comp_zilnic = df.groupby('ziua')[['consum', 'productie']].mean()
    ax1.plot(comp_zilnic.index, comp_zilnic['consum'], label='Consum', linewidth=2.5)
    ax1.plot(comp_zilnic.index, comp_zilnic['productie'], label='Producție', linewidth=2.5)
    ax1.set_xlabel('Ziua Lunii', fontsize=12)
    ax1.set_ylabel('Valoare Medie (MW)', fontsize=12)
    ax1.set_ylim(5500, None)
    ax1.set_title('Comparație Zilnică: Consum vs Producție', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lunar
    comp_lunar = df.groupby('luna')[['consum', 'productie']].mean()
    luni_ro = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun', 'Iul', 'Aug', 'Sep', 'Oct', 'Noi', 'Dec']
    
    x_pos = np.arange(len(luni_ro))
    ax2.bar(x_pos - 0.2, comp_lunar['consum'], width=0.4, label='Consum', alpha=0.8)
    ax2.bar(x_pos + 0.2, comp_lunar['productie'], width=0.4, label='Producție', alpha=0.8)
    
    ax2.set_xlabel('Lună', fontsize=12)
    ax2.set_ylabel('Valoare Medie (MW)', fontsize=12)
    ax2.set_title('Comparație Lunară: Consum vs Producție', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(luni_ro)
    ax2.set_ylim(4500, None)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.savefig("lab1/graphs/consumption_production_comparison.png")
    plt.tight_layout()
    plt.show()

def generate_analysis_report(df):
    """Generează un raport de analiză"""
    print("\n=== RAPORT ANALIZĂ ===")
    
    print(f"Perioadă analizată: {df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Observații totale: {len(df):,}")
    
    print("\n--- STATISTICI CHEIE ---")
    stats = df[['consum', 'productie', 'sold', 'procent_regenerabila']].describe()
    print(stats.round(2))
    
    print("\n--- PERFORMANȚĂ ENERGETICĂ ---")
    sold_mediu = df['sold'].mean()
    regenerabila_medie = df['procent_regenerabila'].mean()
    print(f"Sold mediu: {sold_mediu:+.1f} MW")
    print(f"Pondere energie regenerabilă: {regenerabila_medie:.1f}%")
    
    consum_max_zi = df.groupby('ziua_saptamanii')['consum'].mean().idxmax()
    print(f"Zi cu consum maxim: {consum_max_zi}")

def main():
    """Funcția principală"""
    # Încărcare și preparare date
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    
    # Analiză și vizualizare
    generate_analysis_report(df)
    create_visualizations(df)
    
    print("\nANALIZĂ FINALIZATĂ CU SUCCES!")
    print("Toate graficele salvate în directorul 'lab1/graphs/'")

if __name__ == "__main__":
    main()
