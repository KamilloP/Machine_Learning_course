import pandas as pd
import os


# Ścieżki do plików
input_file = os.path.expanduser('~/data/wikihowSep.csv')
output_file = os.path.expanduser('~/data/wikihowSep_clean.csv')


# Wczytanie CSV
df = pd.read_csv(input_file)


# Sprawdź kolumny, które chcesz zachować (te same co w wikihowAll.csv)
columns_to_keep = ['headline', 'title', 'text']


# Zachowaj tylko wybrane kolumny
df_clean = df[columns_to_keep]


# Zapisz do nowego pliku
df_clean.to_csv(output_file, index=False)


print(f'Wyczyszczony plik zapisany do {output_file}')
print('Kolumny w nowym pliku:', df_clean.columns.tolist())
