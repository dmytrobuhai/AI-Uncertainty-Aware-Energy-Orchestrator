import pandas as pd
from pathlib import Path
import tqdm

BASE = Path(__file__).parent.parent
OUT_DIR = Path.joinpath(BASE, Path("data"))
ALERTS_SUBDIR = (Path("alerts"))
DATA_DIR = Path.joinpath(OUT_DIR, ALERTS_SUBDIR)
OUTPUT_FILE = Path.joinpath(OUT_DIR, "alerts_merged_sorted.csv")

DATE_COL = "Оголошено о"
END_COL  = "Закінчено о"

DROP_COLUMNS = [
    "Район/Громада/Місто",
]

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise RuntimeError("CSV файли не знайдено")

dataframes = {}
columns_sets = {}

for file in tqdm.tqdm(csv_files):
    df = pd.read_csv(file)
    dataframes[file.name] = df
    columns_sets[file.name] = set(df.columns)

all_columns = set.union(*columns_sets.values())

inconsistent = False
for fname, cols in tqdm.tqdm(columns_sets.items()):
    diff = all_columns.symmetric_difference(cols)
    if diff:
        inconsistent = True
        print(f"\n⚠ Різниця у стовпцях: {fname}")
        print("   Відмінності:", diff)

if inconsistent:
    print("\n❗ Виявлено розбіжності у структурі CSV")
else:
    print("✓ Усі CSV мають однакові стовпці")

df = pd.concat(dataframes.values(), ignore_index=True)

print(f"Загальна кількість рядків: {len(df)}")

df = df[
    df[END_COL].notna() &
    (df[END_COL].astype(str).str.strip() != "")
]


print(f"Після фільтрації 'Закінчено о': {len(df)}")

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    format="%d.%m.%Y, %H:%M:%S",
    errors="coerce"
)

invalid_dates = df[DATE_COL].isna().sum()
if invalid_dates:
    print(f"Непарсованих дат: {invalid_dates}")

df = df.sort_values(by=DATE_COL, ascending=True)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\nГотово. Збережено у {OUTPUT_FILE}")
