---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: '1.11.5'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Klasifikasi untuk Baju Thrifting 
### Latar Belakang

<p style="text-indent: 50px; text-align: justify;">Di zaman sekarang ini fashion semakin hari semakin perkembang pesat. Terutama fashion anak kuliah atau lingkungan kampus, dengn begitu muncul baju thrif. Baju thrift merupakan pakaian atau barang bekas yang masih bagus dan layak diapaki dengan harga jual yang murah. Bahkan baju thrift masih ada yang baru dan masih berbandrol. Dengan begitu anak kuliah atau masyarakat bisa mengikuti trend atau mengikuti outfit pada zaman sekarang tanpa mahal-mahal membeli baju yang baru. baju thrift memiliki fisik yang masih bagus dan layak digunakan.Saya mengambil data ini dengan tujuan mengenalkan Perusahaan Fashion Campus pada E-Commerce di kalangan mahasiswa atau anak muda. Memberikan strategi pemasaran untuk penjualan pakaian bekas atau thrift. Mengembangkan strategis bisnis untuk mengenalkan dan mengembangkan model dan trend-trend thrift atau pakaian bekas di kalangan anak muda. Menganalisis data untuk menentukan faktor-faktor yang mempengaruhi prediksi perkembangan Fashion Campus.<p>

### Rumusan Masalah 
<p style="text-indent: 50px; text-align: justify;">1. Bagaimana mengklasifikasikan barang atau baju thrift?
2. Model apa saja yang digunakan?<p>

### Tujuan
<p style="text-indent: 50px; text-align: justify;">Pada pekerjaan kali ini, saya akan melakukan klasifikasi untuk baju thrifting setiap tahun. Tujuan pekerjaan ini adalah mengenalkan Perusahan Fashion Campus pada E-Commerce di kalangan mahasiswa atau anak muda di zaman sekarang ini. Dataset yang saya gunakan adalah dataset "Fashion Campus"<p>

#### a. Data Understanding
<p style="text-indent: 50px; text-align: justify;">Menampilkan Data<p>
```{code-cell} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```
```{code-cell} python
df = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/click_stream_new.csv')
df.head()
```
```{code-cell} python
import numpy as np  # Import library numpy
import pandas as pd  # Import library pandas
df = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/click_stream_new.csv')  
# Membaca file dataset.csv dan menyimpannya ke dalam DataFrame df
df.shape  # Menampilkan bentuk (jumlah baris dan kolom) dari DataFrame df
```
```{code-cell} python
data_df.plot()
```
#### b. EDA
```{code-cell} python
data_df.info()  # Menampilkan informasi umum mengenai DataFrame, termasuk jumlah entri, jumlah kolom, tipe data setiap kolom, dan apakah ada nilai kosong (non-null).
print('Ukuran data ', data_df.shape)  # Menampilkan ukuran data dalam format tuple (jumlah baris, jumlah kolom).
data_df.dtypes  # Menampilkan tipe data dari setiap kolom dalam DataFrame.
```
```{code-cell} python
# Mencari Missing Value
data_df.isnull().sum()
```
```{code-cell} python
import seaborn as sns
# Menghitung korelasi antar fitur untuk subset yang diinginkan
features = data_df[['session_id', 'event_name', 'event_time', 'event_id', 'traffic_source', 'product_id','quantity','item_price','payment_status','search_keywords','promo_code','promo_amount']]
correlation_matrix = features.corr()

# Menampilkan matriks korelasi
print("Matriks Korelasi:")
print(correlation_matrix)

# Menggambar heatmap untuk visualisasi korelasi
plt.figure(figsize=(10, 6))
plt.title("Heatmap Korelasi antar Fitur")
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
```
```{code-cell} python
# Deskripsi Statistik
data_df.describe()
```
#### c. Prepocessing
<p style="text-indent: 50px; text-align: justify;"> sliding windows<p>
```{code-cell} python
# Fungsi untuk membuat sliding windows
def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Menyiapkan data untuk sliding windows
window_size = 3  # Ukuran window
data_values = data_df[['session_id', 'event_name']].values  # Menggunakan kolom 'Close' dan 'High' untuk prediksi
X, y = create_sliding_windows(data_values, window_size)

# Membuat DataFrame untuk hasil sliding windows
sliding_window_df = pd.DataFrame(X.reshape(X.shape[0], -1), columns=[f'session_id_t-{window_size-i}' for i in range(window_size)] + [f'event_name_t-{window_size-i}' for i in range(window_size)])
sliding_window_df['Target_session_id_t'] = y[:, 0]  # Target Close
sliding_window_df['Target_event_id_t'] = y[:, 1]  # Target High

# Menampilkan hasil sliding windows
print(sliding_window_df.head())
```
#### d. Normalisasi
```{code-cell} python
# Inisialisasi scaler untuk fitur (input) dan target (output)
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalisasi fitur (Close_t-3, Close_t-2, Close_t-1, High_t-3, High_t-2, High_t-1)
df_features_normalized = pd.DataFrame(
    scaler_features.fit_transform(sliding_window_df.iloc[:, :-2]),  # Ambil semua kolom kecuali target
    columns=sliding_window_df.columns[:-2],  # Nama kolom tanpa target
    index=sliding_window_df.index
)

# Normalisasi target (Target_Close_t dan Target_High_t)
df_target_normalized = pd.DataFrame(
    scaler_target.fit_transform(sliding_window_df[['Target_Close_t', 'Target_High_t']]),
    columns=['Target_Close_t', 'Target_High_t'],
    index=sliding_window_df.index
)

# Gabungkan kembali dataframe yang sudah dinormalisasi
df_normalized = pd.concat([df_features_normalized, df_target_normalized], axis=1)

# Menampilkan hasil normalisasi
print(df_normalized.head())

# Mengatur fitur (X) dan target (y)
X = df_normalized[['Close_t-3', 'Close_t-2', 'Close_t-1', 'High_t-3', 'High_t-2', 'High_t-1']]
y = df_normalized[['Target_Close_t', 'Target_High_t']]  # Target adalah harga yang dinormalisasi

# Membagi data menjadi training dan testing (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print('===== Data Train =====')
print(X_train)

print('===== Data Testing ====')
print(X_test)

# Mengambil nilai tanggal dari indeks X_train dan X_test
dates_train = X_train.index
dates_test = X_test.index

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Plot data Close dan High dengan format tanggal di sumbu x
plt.figure(figsize=(14, 7))

# Plot Close
plt.plot(data_df.index, data_df['Close'], label='Close', linestyle='-', color='blue')

# Plot High
plt.plot(data_df.index, data_df['High'], label='High', linestyle='--', color='orange')

# Format sumbu x agar menampilkan tanggal
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Menampilkan label tanggal per bulan

plt.title('Close and High Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)  # Putar label tanggal agar tidak tumpang tindih
plt.tight_layout()
plt.show()
```


