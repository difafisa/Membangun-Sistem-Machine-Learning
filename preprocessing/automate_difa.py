import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

input_file = 'transactions_dataset/metaverse_transactions_dataset.csv' 
output_file = 'preprocessing/transactions_preprocessing/metaverse_clean.csv'

output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

# load data
try:
    data = pd.read_csv(input_file)
    print(f"Berhasil memuat data dari {input_file}")
except FileNotFoundError:
    print(f"ERROR: File tidak ditemukan di {input_file}")
    exit(1) 

# drop kolom yang tidak relevan
columns_to_drop = ['timestamp', 'sending_address', 'receiving_address', 'risk_score']
data.drop(columns=columns_to_drop, inplace=True, errors='ignore') # errors='ignore' agar tidak error jika kolom sudah tidak ada

# drop missing & duplicate
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# encode categorical
cols = data.select_dtypes(include=['object']).columns
for col in cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# save processed data
data.to_csv(output_file, index=False)

print(f"Preprocessing selesai. File disimpan di {output_file}")