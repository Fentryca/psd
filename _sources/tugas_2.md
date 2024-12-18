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

# Prediksi Harga Daging Ayam

### Latar Belakang

<p style="text-indent: 50px; text-align: justify;"> Harga daging ayam merupakan peran penting dalam perekonomian masyarakat, terutama dalam aspek pangan dan juga peternakan. Daging ayam merupakan sumber protein utama yang banyak peminatnya untuk dikonsumsi sehari-hari. Prediksi ini sangat bermanfaat untuk peternak dan juga konsumen untuk memantau harga daging ayam. Saya mengambil data Prediksi Harga Daging Ayam dari website https://www.bi.go.id/hargapangan/TabelHarga/ PasarModernDaerah# yang selanjutnya akan saya uji sebagai proyek dari matakuliah saya, yaitu Proyek Sains Data. Untuk lebih jelasnya dapat disimak langkah-langkah dalam mengerjakan proyek ini.<p>

### Rumusan Masalah

<p style="text-indent: 50px; text-align: justify;"> 1. Bagaimana cara untuk melihat harga daging ayam per minggu?
2. Bagaimana peternak bisa memprediksi harga ayam per minggu? <p>

### Tujuan

<p style="text-indent: 50px; text-align: justify;">Pada pekerjaan kali ini, saya akan melalukan klasifikasi untuk prediksi harga daging ayam. Tujuan pekerjaan ini adalah untuk membantu prediksi harga daging ayam per minggu pada tahun 2021. Dataset yang digunakan dataset "prediksi harga daging ayam" yang saya ambil dari Pusat Informasi Harga Pangan Strategis Nasional (PIHPS).<p>

#### a. Data undersanding
```{code-cell} python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px

color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")

import pandas as pd

# URL raw dari file Excel di GitHub (URL yang benar)
url = "https://github.com/Fentryca/Proyek-Sains-Data/raw/main/PSD/Daging_Ayam.xlsx"

# Membaca file Excel dari URL
df = pd.read_excel(url)

# Mengurutkan ulang data jika diperlukan (misalnya dari yang terbaru ke terlama)
df = df.iloc[::-1].reset_index(drop=True)

# Menampilkan 5 baris pertama dari data
df.head()
```
<p style="text-indent: 50px; text-align: justify;">
	Day	Daging Ayam
0	11/ 10/ 2024	44,400
1	10/ 10/ 2024	44,400
2	09/ 10/ 2024	46,050
3	08/ 10/ 2024	44,150
4	07/ 10/ 2024	44,150
<p>

#### Mengubah format kolom dan tanggal menjadi datatime
```{code-cell} python
# Ubah format kolom 'Day' menjadi datetime dengan dayfirst=True
df['Day'] = pd.to_datetime(df['Day'], dayfirst=True)

# Menampilkan 5 baris pertama untuk verifikasi
df.head()
```
<p style="text-indent: 50px; text-align: justify;">
Day	Daging Ayam
0	2024-10-11	44,400
1	2024-10-10	44,400
2	2024-10-09	46,050
3	2024-10-08	44,150
4	2024-10-07	44,150
<p>
```{code-cell} python
# Melihat semua nama kolom dalam data
df.columns
```
<p style="text-indent: 50px; text-align: justify;">
# Melihat semua nama kolom dalam data df.columns<p>

#### b. Plotting Data
```{code-cell} python
import matplotlib.pyplot as plt

# Membuat visualisasi harga daging ayam dari waktu ke waktu
plt.figure(figsize=(20,6))

# Plot data tanggal dan harga daging ayam
plt.plot(df["Day"], df['Daging Ayam'], lw=2)

# Pengaturan label sumbu dan judul
plt.xlabel("Day", fontsize=16)
plt.ylabel("Daging Ayam", fontsize=16)
#plt.title("Daging Ayam Ras Segar", fontsize=16)

# Menampilkan plot
plt.show()
```
```{code-cell} python
df.describe()
```
<p style="text-indent: 50px; text-align: justify;">
Day
count	205
mean	2024-05-22 00:00:00
min	2024-01-01 00:00:00
25%	2024-03-12 00:00:00
50%	2024-05-22 00:00:00
75%	2024-08-01 00:00:00
max	2024-10-11 00:00:00
<p>

#### c. Ektrasi Data
```{code-cell} python
# Membuat pergeseran data pada kolom 'Daging Ayam'
df_slide['xt-3'] = df_slide['Daging Ayam'].shift(-3)
df_slide['xt-2'] = df_slide['Daging Ayam'].shift(-2)
df_slide['xt-1'] = df_slide['Daging Ayam'].shift(-1)
df_slide['xt'] = df_slide['Daging Ayam']

# Menghapus kolom asli 'Daging Ayam Ras Segar'
df_slide = df_slide.drop(columns=['Daging Ayam'])

# Menampilkan 5 baris pertama dari dataframe yang sudah diubah
df_slide.head()
```
<p style="text-indent: 50px; text-align: justify;">

Day	xt-3	xt-2	xt-1	xt
0	2024-10-11	44,150	46,050	44,400	44,400
1	2024-10-10	44,150	44,150	46,050	44,400
2	2024-10-09	44,150	44,150	44,150	46,050
3	2024-10-08	44,150	44,150	44,150	44,150
4	2024-10-07	44,150	44,150	44,150	44,150
<p>

```{code-cell} python
df_slide.dtypes
```
<p style="text-indent: 50px; text-align: justify;">
0
Day	datetime64[ns]
xt-3	object
xt-2	object
xt-1	object
xt	object

dtype: object<p>
```{code-cell} python
df_slide = df_slide.set_index('Day')
```
#### d. Visualisasi Data
```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

# Mengambil semua kolom fitur dari dataframe df
features = list(df.columns)  # Ambil semua kolom dari df

# Menghitung jumlah subplot yang diperlukan
num_features = len(features)
num_cols = 2  # Membuat subplot dengan 2 kolom
num_rows = (num_features + 1) // num_cols  # Pembulatan ke atas untuk jumlah baris

# Membuat subplot sesuai dengan jumlah fitur
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 4))

# Mengubah axs menjadi array 2D jika hanya ada satu baris
if num_features == 1:
    axs = np.array([[axs]])
# Melakukan plotting untuk setiap fitur
for i, feature in enumerate(features):
    row = i // num_cols  # Menentukan baris subplot
    col = i % num_cols   # Menentukan kolom subplot

    # Pastikan axs memiliki dimensi yang sesuai (misalnya jika hanya satu fitur, axs bisa berupa satu sumbu)
    ax = axs[row, col] if num_features > 1 else axs
    ax.hist(df[feature], bins=20, color='blue', alpha=0.5)  # Plot histogram
    ax.set_title(f'Distribusi dari {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frekuensi')

# Menghapus subplot yang tidak terpakai jika jumlah fitur ganjil
if num_features % num_cols != 0:
    fig.delaxes(axs[-1, -1])  # Menghapus subplot terakhir yang tidak terpakai

# Menyesuaikan layout agar lebih rapi
plt.tight_layout()

# Menampilkan plot
plt.show()
```
```{code-cell} python
print(df_slide.isnull().sum())
```
<p style="text-indent: 50px; text-align: justify;">
xt-3    3
xt-2    2
xt-1    1
xt      0
dtype: int64<p>

```{code-cell} python
# Mengambil semua kolom fitur dari dataframe df (kecuali 'Tanggal')
features = df.columns.drop('Day')  # Ambil semua kolom kecuali kolom tanggal
colors = plt.get_cmap('tab10', len(features))  # Mengambil colormap untuk warna
plt.figure(figsize=(15, 5))

# Melakukan plotting untuk setiap fitur
for i, feature in enumerate(features):
    plt.plot(df['Day'], df[feature], linestyle='', marker='.', color=colors(i), label=feature)

plt.title('Harga Daging Ayam')
plt.xlabel('Day')
plt.ylabel('Harga')
plt.legend()
plt.xticks(rotation=90)  # Memutar label pada sumbu x agar lebih jelas
plt.show()
```
```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
features = ['xt-3', 'xt-2', 'xt-1', 'xt']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,2,i+1)
  sb.boxplot(df_slide[col])
plt.show()
```
#### e. Preprocessing Data
<p style="text-indent: 50px; text-align: justify;">Missing value<p>

```{code-cell} python
df_slide_cleaned = df_slide.dropna()
df_slide_cleaned.isna().sum()
```
#### f. Normalisasi
```{code-cell} python
# Cek apakah kolom selain 'Day' memang berisi NaN sebelum normalisasi
print("Data sebelum normalisasi:")
print(df_slide_cleaned.head())
```
```{code-cell} python
from sklearn.preprocessing import MinMaxScaler

data = {
    'Day': ['2024-10-11', '2024-10-10', '2024-10-09', '2024-10-08', '2024-10-07'],
    'xt-3': [44150, 44150 , 44150, 44150, 44150],
    'xt-2': [46050, 44150, 44150, 44150, 44150],
    'xt-1': [44400, 46050, 44150, 44150 , 44150],
    'xt': [44400, 44400, 46050, 44150, 44150 ]
}

df_slide_cleaned = pd.DataFrame(data)
df_slide_cleaned['Day'] = pd.to_datetime(df_slide_cleaned['Day'])  # Pastikan kolom Day berbentuk datetime jika dibutuhkan

# Memisahkan kolom 'Day' dari data yang akan dinormalisasi
Day_column = df_slide_cleaned['Day']
data_to_normalize = df_slide_cleaned.drop(columns=['Day'])

# Normalisasi Min-Max pada data selain 'Day'
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_to_normalize), columns=data_to_normalize.columns)

# Menambahkan kembali kolom 'Day' yang asli
df_normalized = pd.concat([Day_column.reset_index(drop=True), data_normalized], axis=1)

# Tampilkan hasil normalisasi yang sudah dirapikan
print(df_normalized.head())
```

#### g. Modelling

<p style="text-indent: 50px; text-align: justify;">Modelling Data Regresi Linear<p>
```{code-cell} python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Gunakan 'xt' sebagai target
target_column = 'xt'  # Kolom target

# Pisahkan fitur (X) dan target (y)
X = df_slide_cleaned[['xt-3', 'xt-2', 'xt-1']]  # Fitur (kolom lain)
y = df_slide_cleaned[target_column]  # Kolom target ('xt')

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Buat model regresi linier
model_lr = LinearRegression()

# Latih model dengan data training
model_lr.fit(X_train, y_train)

# Prediksi dengan data testing
y_pred = model_lr.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
```
```{code-cell} python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Misalkan 'df_slide_cleaned' sudah ada, dan ini adalah bagian fitur dan target
X = df_slide_cleaned[['xt-3', 'xt-2', 'xt-1']]
y = df_slide_cleaned['xt']

# Pisahkan salah satu fitur untuk visualisasi sederhana (misalnya xt-1)
X_single = X['xt-1']

# Buat plot untuk regresi linier
plt.figure(figsize=(10,6))

# Plot scatter plot data asli
sns.scatterplot(x=X_single, y=y, label="Data Asli")

# Plot garis regresi
sns.regplot(x=X_single, y=y, line_kws={"color": "red"}, label="Garis Regresi")

# Tambahkan judul dan label sumbu
plt.title('Model Regresi Linier untuk Prediksi xt berdasarkan xt-1')
plt.xlabel('xt-1')
plt.ylabel('xt')

# Tampilkan legenda
plt.legend()

# Tampilkan plot
plt.show()
```
<p style="text-indent: 50px; text-align: justify;">Modelling Random Forest<p>

```{code-cell} python
# Import library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Misalkan data kamu ada dalam variabel X dan y
# X adalah fitur (misal data input), y adalah target (misal label/output yang ingin diprediksi)
# Kamu perlu mendefinisikan X dan y sesuai dengan dataset kamu, misalnya:
# X = data_fitur
# y = data_target

# Membagi data menjadi data pelatihan (train) dan data uji (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Membuat instance RandomForestRegressor dengan 100 pohon dan seed untuk reproduksibilitas.
rf_model.fit(X_train, y_train)  # Melatih model dengan data pelatihan.

# Prediksi pada data uji
y_pred_rf = rf_model.predict(X_test)  # Menggunakan model untuk memprediksi nilai pada data uji.

# Evaluasi model
mse_rf = mean_squared_error(y_test, y_pred_rf)  # Menghitung Mean Squared Error (MSE) untuk prediksi Random Forest.
r2_rf = r2_score(y_test, y_pred_rf)  # Menghitung R-squared (R²) untuk mengevaluasi model.

# Menghitung RMSE
rmse_rf = np.sqrt(mse_rf)  # Menghitung Root Mean Squared Error (RMSE) dari MSE.

# Menghitung MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100  # Menghitung Mean Absolute Percentage Error (MAPE).

# Menampilkan hasil evaluasi
print(f'Root Mean Squared Error (Random Forest): {rmse_rf}')  # Mencetak RMSE untuk model Random Forest.
print(f'Mean Squared Error (Random Forest): {mse_rf}')  # Mencetak MSE untuk model Random Forest.
print(f'R-squared (Random Forest): {r2_rf}')  # Mencetak R² untuk model Random Forest.
print(f'MAPE (Random Forest): {mape}%')  # Mencetak MAPE untuk model Random Forest.
```
```{code-cell} python
# Mengambil tanggal dari DataFrame asli sesuai dengan index y_test
dates = df.loc[y_test.index, 'Day']  # Mengambil hari dari DataFrame asli berdasarkan index dari y_test.

# Create a DataFrame to hold the actual vs predicted results for Random Forest with date index
results_random_forest = pd.DataFrame({'Date': dates, 'Actual': y_test, 'Predicted': y_pred_rf})
# Membuat DataFrame baru untuk menyimpan hasil aktual dan prediksi dari model Random Forest dengan kolom tanggal.

results_random_forest.set_index('Date', inplace=True)  # Mengatur kolom 'Date' sebagai index DataFrame.

# Plotting the Actual vs Predicted values with dashed lines
plt.figure(figsize=(15, 6))  # Mengatur ukuran figure untuk plot.
plt.plot(results_random_forest.index, results_random_forest['Actual'], label='Actual', marker='o', linestyle='--', color='blue')
# Menggambar garis untuk nilai aktual dengan marker bulat dan garis putus-putus.

plt.plot(results_random_forest.index, results_random_forest['Predicted'], label='Predicted', marker='x', linestyle='--', color='orange')
# Menggambar garis untuk nilai prediksi dengan marker silang dan garis putus-putus.

plt.title('Actual vs Predicted Values (Random Forest)')  # Menambahkan judul pada plot.
plt.xlabel('Index')  # Menambahkan label sumbu x.
plt.ylabel('Values')  # Menambahkan label sumbu y.
plt.legend()  # Menampilkan legenda untuk membedakan antara nilai aktual dan prediksi.
plt.grid(True)  # Menambahkan grid pada plot.

# Display the plot
plt.xticks(rotation=45)  # Memutar label sumbu x untuk meningkatkan keterbacaan.
plt.tight_layout()  # Menyesuaikan layout untuk menghindari tumpang tindih label.
plt.show()  # Menampilkan plot.
```