import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setări pentru afișare mai frumoasă
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. CITIREA DATELOR DIN CSV
print("=== CITIREA ȘI ÎNCĂRCAREA DATELOR ===")

try:
    df = pd.read_csv("Data Analysis/lab1/raw_energy_data.csv", parse_dates=["date"] )
    print(f"Fișier încărcat cu succes! Dimensiune: {df.shape}")
except FileNotFoundError:
    print("Fișierul nu a fost găsit.")
    exit()
except Exception as e:
    print(f"Eroare la încărcarea fișierului: {e}")
    exit()

print("\nPrimele 5 rânduri din date:")
print(df.head())

# 2. CURĂȚAREA DATELOR
print("\n=== CURĂȚAREA DATELOR ===")

print("Valori lipsă pe coloane:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() > 0:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    df[numeric_cols] = df[numeric_cols].interpolate()
    print("Valori lipsă înlocuite prin interpolare.")
else:
    print("Nu s-au găsit valori lipsă.")

duplicate_count = df.duplicated().sum()
print(f"\nNumăr de duplicate găsite: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate eliminate.")
    print(f"Dimensiune nouă: {df.shape}")

# 3. TRANSFORMĂRI ȘI ENGINEERING FEATURES
print("\n=== TRANSFORMĂRI ȘI CREARE VARIABILE NOI ===")

df['an'] = df["date"].dt.year
df['luna'] = df["date"].dt.month
df['ziua'] = df['date'].dt.day
df['ora'] = df["date"].dt.hour
df['minut'] = df["date"].dt.minute
df['ziua_saptamanii'] = df["date"].dt.day_name()
df['weekend'] = df["date"].dt.day_of_week >= 5
df['trimester'] = df['date'].dt.quarter

df['consum_acoperit'] = df['productie'] / df["consum"]

df['total_regenerabila'] = df['hidro'] + df['fotovolt'] + df['eolian'] + df['biomasa']

df['procent_regenerabila'] = df['total_regenerabila'] / df['productie']

df['sold_absolut'] = df['sold'].abs()

print("Variabile noi create cu succes!")

# 4. EXPLORARE INIȚIALĂ ȘI STATISTICI
print("\n=== EXPLORARE INIȚIALĂ ȘI STATISTICI ===")

print(f"\nPerioada acoperită: {df['date'].min()} - {df['date'].max()}")
print(f"Număr total de observații: {len(df)}")

print(f"\n--- STATISTICI SURSE DE ENERGIE ---")
print(df.describe())

print("\nMedie consum pe lună:")
print(df.groupby('luna')['consum'].mean())

print("\nZile ale săptămânii cu cel mai mare consum de energie:")
print(df.groupby('ziua_saptamanii')['consum'].mean().sort_values(ascending=False))

# 5. ANALIZA CORELAȚIILOR
print("\n=== VIZUALIZĂRI ===")
fig, ax = plt.subplots(figsize = (10,6))

ax.scatter(df['ora'], df['fotovolt'])

ax.set_xlabel("Hour")
ax.set_ylabel("Solar Energy (MW)")

plt.show()

fig, ax2 = plt.subplots(2, figsize = (10,6))

df['interval'] = (df['ora'] // 3) * 3
corelations1 = df.groupby('interval')[['carbune', 'hidro', 'hidrocarburi',  'nuclear', 'eolian', 'fotovolt', 'biomasa']].mean()
corelations2 = df.groupby('ora')[['carbune', 'hidro', 'hidrocarburi',  'nuclear', 'eolian', 'fotovolt', 'biomasa']].mean()

sns.heatmap(corelations1, annot=True, cmap='viridis', ax=ax2[0])
sns.heatmap(corelations2, cmap='viridis', ax=ax2[1])

plt.show()

#TODO give labels about this graph
fig, ax2 = plt.subplots(2, figsize = (10,6))

corelations3 = df.groupby('luna')[['carbune', 'hidro', 'hidrocarburi',  'nuclear', 'eolian', 'fotovolt', 'biomasa']].mean()
sns.heatmap(corelations3, annot=True, cmap='viridis', ax=ax2[1])
plt.show()

fig, ax3 = plt.subplots(figsize = (10,6))

prod_lunara = df.groupby(['an','luna'])[['carbune', 'hidro', 'hidrocarburi',  'nuclear', 'eolian', 'fotovolt', 'biomasa']].mean()
#prod_lunara['an_luna'] = prod_lunara['an'].astype(str) + '-' + prod_lunara['luna'].astype(str).str.zfill(2)

prod_lunara.plot(kind='bar', stacked=True, ax=ax3, colormap='viridis')

handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles[::-1], labels[::-1], title="Tip energie", bbox_to_anchor=(1,0.95), loc='upper left')


month_abrv = {
    1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
    7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec',
}
labels = [f"{year}-{month_abrv[month_i]}" for year, month_i in prod_lunara.index]

ax3.set_xticklabels(labels, rotation = 45)
plt.tight_layout()
plt.show()
# TASK 2: Soldul zilnic pentru 2024 și 2025 #TODO Make it smoother. 365 values instead of 30
print("\n--- TASK 2: Soldul zilnic pentru 2024 și 2025 ---")
fig, ax = plt.subplots(figsize=(12, 6))
sold_zilnic = df.groupby(['an', 'ziua'])['sold'].mean().reset_index()

for year in [2024, 2025]:
    data_an = sold_zilnic[sold_zilnic['an'] == year]
    ax.plot(data_an['ziua'], data_an['sold'], label=f'An {year}', marker='o')

ax.set_xlabel('Ziua lunii')
ax.set_ylabel('Sold')
ax.set_title('Soldul zilnic pentru 2024 și 2025')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# TASK 3: Seria temporară pe sold  #TODO Smooth this graph, too much distortion from value to value
print("\n--- TASK 3: Seria temporară pe sold ---")
fig, ax = plt.subplots(figsize=(12, 6))
df_sorted = df.sort_values('date')
ax.plot(df_sorted['date'], df_sorted['sold'], linewidth=1, alpha=0.7)
ax.set_xlabel('Data')
ax.set_ylabel('Sold')
ax.set_title('Seria temporală a soldului energetic')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# TASK 4: Peek-ul producției pe ore
print("\n--- TASK 4: Peek-ul producției pe ore ---")
fig, ax = plt.subplots(figsize=(12, 6))
prod_pe_ora = df.groupby('ora')['productie'].mean()
ax.plot(prod_pe_ora.index, prod_pe_ora.values, marker='o', linewidth=2, markersize=6)
ax.set_xlabel('Ora zilei')
ax.set_ylabel('Producție medie')
ax.set_title('Profilul producției energetice pe ore')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# TASK 5: Consumul mediu pe zilele săptămânii  #TODO make it more zoomed in, starting from 5k
print("\n--- TASK 5: Consumul mediu pe zilele săptămânii ---")
fig, ax = plt.subplots(figsize=(10, 6))
consum_zile = df.groupby('ziua_saptamanii')['consum'].mean()

# Reordonăm zilele săptămânii
zile_ordine = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
consum_zile = consum_zile.reindex(zile_ordine)

ax.bar(consum_zile.index, consum_zile.values, color='skyblue', alpha=0.7)
ax.set_xlabel('Ziua săptămânii')
ax.set_ylabel('Consum mediu')
ax.set_title('Consumul mediu pe zilele săptămânii')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# TASK 6: Producția medie lunară pentru 2024 și 2025 
print("\n--- TASK 6: Producția medie lunară pentru 2024 și 2025 ---")
fig, ax = plt.subplots(figsize=(12, 6))
prod_lunara = df.groupby(['an', 'luna'])['productie'].mean().reset_index()

luni = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun', 
        'Iul', 'Aug', 'Sep', 'Oct', 'Noi', 'Dec']

for year in [2024, 2025]:
    data_an = prod_lunara[prod_lunara['an'] == year]
    ax.plot(data_an['luna'], data_an['productie'], 
            label=f'An {year}', marker='s', linewidth=2)

ax.set_xlabel('Lună')
ax.set_ylabel('Producție medie')
ax.set_title('Producția medie lunară pentru 2024 și 2025')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(luni)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# TASK 7: Comparare grafică între consum și producție  #TODO Smooth it out, need more
print("\n--- TASK 7: Comparare grafică între consum și producție ---")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Comparație zilnică
comparatie_zilnica = df.groupby('ziua')[['consum', 'productie']].mean()
ax1.plot(comparatie_zilnica.index, comparatie_zilnica['consum'], 
         label='Consum', marker='o', linewidth=2)
ax1.plot(comparatie_zilnica.index, comparatie_zilnica['productie'], 
         label='Producție', marker='s', linewidth=2)
ax1.set_xlabel('Ziua lunii')
ax1.set_ylabel('Valoare')
ax1.set_title('Comparație consum vs producție - Zilnic')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Comparație lunară
comparatie_lunara = df.groupby('luna')[['consum', 'productie']].mean()
ax2.bar(comparatie_lunara.index - 0.2, comparatie_lunara['consum'], 
        width=0.4, label='Consum', alpha=0.7)
ax2.bar(comparatie_lunara.index + 0.2, comparatie_lunara['productie'], 
        width=0.4, label='Producție', alpha=0.7)
ax2.set_xlabel('Lună')
ax2.set_ylabel('Valoare medie')
ax2.set_title('Comparație consum vs producție - Lunar')   #TODO make it zoom in at 5000
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(luni)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# TASK 8: Comparare grafică pentru anii 2024 și 2025
print("\n--- TASK 8: Comparare grafică pentru anii 2024 și 2025 ---")
import pandas as pd

# --- Aggregate main yearly stats --- 
summary = df.groupby('an').agg({
    'consum': 'mean',
    'productie': 'mean',
    'procent_regenerabila': lambda x: x.mean() * 100,
    'sold': 'mean'
}).reset_index()

# --- Extract numeric values ---
c2024 = summary.loc[summary['an'] == 2024, 'consum'].values[0]
c2025 = summary.loc[summary['an'] == 2025, 'consum'].values[0]

# --- Calculate % change manually ---
summary['Δ_consum_%'] = np.nan  # make column first
summary.loc[summary['an'] == 2025, 'Δ_consum_%'] = (c2025 - c2024) / c2024 * 100

# --- Or use pct_change for others directly ---
summary['Δ_productie_%'] = summary['productie'].pct_change() * 100
summary['Δ_sold_%'] = summary['sold'].pct_change() * 100

print(summary)


summary.plot(x='an', y=['Δ_consum_%', 'Δ_productie_%', 'Δ_sold_%'], 
             kind='bar', figsize=(10,6), colormap='coolwarm' )
plt.title('Diferențe procentuale între ani')
plt.ylabel('Schimbare (%)')
plt.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()


print("\n=== ANALIZA FINALIZATĂ CU SUCCES ===")
