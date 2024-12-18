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

# Prediksi Penyakit Diabetes

### Latar Belakang

<p style="text-indent: 50px; text-align: justify;">Pada pekerjaan kali ini, saya akan melakukan klasifikasi untuk prediksi penyakit diabetes. Tujuan dari pekerjaan ini adalah untuk membantu dalam memprediksi adanya peyakit diabetes tahap awal pada individu berdasarkan berbagai fitur yang tersedia. Dataset yang kami gunakan adalah dataset "Early Stage Diabetes Risk Prediction" yang kami ambil dari UCI Machine Learning Respository . Dataset ini berasal dari Rumah Sakit di Sylhet, Bangladesh dan disetujui oleh dokter.
Langkah pertama yang dilakukan adalah mengumpulkan data. Data tersebut berada di aiven.com, sehingga data perlu ditarik dari sumber tersebut. Dataset ini terdiri dari** 520 baris dan **16 fitur, yaitu Age, Sex, Polyuria, Polydipsia, Sudden weight loss, Weakness, Polyphagia, Genital thrush, Visual blurring, Itching, Irritability, Delayed healing, Partial paresis, Muscle, Alopecia, Obseit, Class.
Terdapat 2 type data dalam dataset ini, yakni catagorial dan integer :
Tipe data categorical, juga dikenal sebagai tipe data katagorikal, merujuk pada variabe yang menggambarkan kategori atau kelompok yang berbeda
Tipe data integer merujuk pada nilai-nilai yang terdiri dari angka.<p>

### Rumusan Masalah

<p style="text-indent: 50px; text-align: justify;">Berdasarkan latar belakang diatas, maka rumusan masalahnya adalah sebagai berikut : 
Bagaimana mengidentifikasi faktor risiko utama yang mempengaruhi perkembangan diabetes pada tahap awal?
Bagaimana membangun model prediksi yang akurat untuk mendeteksi risiko diabetes pada tahap awal?<p>

### Tujuan
<p style="text-indent: 50px; text-align: justify;">Tujuan dari pekerjaan ini adalah untuk membantu dalam memprediksi adanya peyakit diabetes tahap awal pada individu berdasarkan berbagai fitur yang tersedia. Dataset yang kami gunakan adalah dataset "Early Stage Diabetes Risk Prediction" yang kami ambil dari UCI Machine Learning Respository . Dataset ini berasal dari Rumah Sakit di Sylhet, Bangladesh dan disetujui oleh dokter.<p>

#### a. Data Understanding (Memahami Data)

<p style="text-indent: 50px; text-align: justify;"> Menampilkan Data <p>
```{code-cell} python
# Import library yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
```
```{code-cell} python
df = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/diabetes_data_upload.csv')
df.head()
```
```{code-cell} python
import numpy as np  # Import library numpy
import pandas as pd  # Import library pandas
df = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/diabetes_data_upload.csv')  # Membaca file dataset.csv dan menyimpannya ke dalam DataFrame df
df.shape  # Menampilkan bentuk (jumlah baris dan kolom) dari DataFrame df
```
#### Menentukan Missing Value
```{code-cell} python
import numpy as np
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/diabetes_data_upload.csv')
df.isnull().sum()
```
```{code-cell} python
df.info()
print("Shape of data:")
print(df.shape)
```
```{code-cell} python
print("Jumlah data duplicated:", df.duplicated().sum(), end="")
df.isna().sum()
```
#### Mengekplorasi Data(Numerik)
```{code-cell}
import numpy as np  # Import library numpy
import pandas as pd  # Import library pandas
df = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/integer.csv')  # Membaca file dataset.csv
df.describe() #hanya berlaku untuk type data integer
```
#### d. Prepocessing
<p style="text-indent: 50px; text-align: justify;">Menghapus Outlier<p>
```{code-cell} python
# Mengimpor pustaka pandas dengan alias 'pd'.
import pandas as pd

# Mengimpor kelas LocalOutlierFactor dari modul neighbors di pustaka Scikit-learn (sklearn).
from sklearn.neighbors import LocalOutlierFactor

# Membaca dataset dari file CSV ("dataset.csv") ke dalam DataFrame X menggunakan Pandas.
X = pd.read_csv("https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/data_diabet.csv")

# Membuat objek LocalOutlierFactor (LOF).
# n_neighbors=5 menentukan jumlah tetangga yang akan digunakan dalam perhitungan LOF.
# contamination=0.1 menentukan tingkat kontaminasi atau persentase outlier yang diharapkan dalam data.
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)

# Menggunakan metode fit_predict() dari objek LOF untuk menentukan status outlier (outlier atau bukan) untuk setiap sampel dalam data.
# Hasilnya akan berupa array yang berisi prediksi status outlier untuk setiap sampel.
y_pred = lof.fit_predict(X)

# Cari indeks dari nilai -1 dalam array y_pred
outlier_indices = [index for index, value in enumerate(y_pred) if value == -1]

# Cetak indeks dari nilai -1 untuk mengetahui data ke berapa yang dianggap sebagai outlier
print("Data outlier terdapat pada indeks:", outlier_indices)

# Menghapus baris yang mengandung outlier dari DataFrame
X_cleaned = X.drop(outlier_indices)

# Menyimpan DataFrame yang telah dibersihkan ke file CSV baru
X_cleaned.to_csv("dataset_tanpa_outlier.csv", index=False)

# Menampilkan jumlah baris asli dan jumlah baris setelah outlier dihapus
print("Jumlah baris asli:", len(X))
print("Jumlah baris setelah outlier dihapus:", len(X_cleaned))
print("Dataset tanpa outlier telah disimpan ke 'dataset_tanpa_outlier.csv'")
```

<p style="text-indent: 50px; text-align: justify;"> Menyimpan Data Training dan Data Testing <p>
```{code-cell}python
import pandas as pd
from sklearn.model_selection import train_test_split

# URL raw untuk file CSV
file_path = 'https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/dataset_tanpa_outlier.csv'

# Membaca data dari file CSV
df = pd.read_csv(file_path)

# Membagi data menjadi 80% untuk training dan 20% untuk testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Menyimpan data training ke file CSV
train_file_path = 'hd_training_80.csv'
train_df.to_csv(train_file_path, index=False)

# Menyimpan data testing ke file CSV
test_file_path = 'hd_testing_20.csv'
test_df.to_csv(test_file_path, index=False)

# Output hasil penyimpanan
print(f"Data training telah disimpan ke {train_file_path}")
print(f"Data testing telah disimpan ke {test_file_path}")
```
#### Pemodelan
<p style="text-indent: 50px; text-align: justify;">Prediksi Penyakit jantung Menggunakan Klasifikasi K-Nearest Neighbor (KNN)<p>
```{code-cell} python
import numpy as np  # Import library numpy
import pandas as pd  # Import library pandas
df = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/dataset_tanpa_outlier.csv')  # Membaca file dataset.csv dan menyimpannya ke dalam DataFrame df
df.shape  # Menampilkan bentuk (jumlah baris dan kolom) dari DataFrame df
```
```{code-cell} python
import numpy as np  # Import library numpy
import pandas as pd  # Import library pandas
df = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/dataset_tanpa_outlier.csv')  # Membaca file dataset.csv dan menyimpannya ke dalam DataFrame df
df.shape  # Menampilkan bentuk (jumlah baris dan kolom) dari DataFrame df
```
<p style="text-indent: 50px; text-align: justify;">Menentukan nilai K<p>
```{code-cell} python
# Import library yang dibutuhkan
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Membaca data dari file CSV
data = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/dataset_tanpa_outlier.csv')

# Memisahkan data menjadi fitur (X) dan target (y)
X = data.drop('class', axis=1)  # Jika targetnya disebut 'target'
y = data['class']

# Normalisasi data menggunakan MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Menyimpan scaler ke file pickle
scaler_filename = "preprocessing.pickle"
pickle.dump(scaler, open(scaler_filename, "wb"))

# Membagi data menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Menentukan kisaran nilai K yang ingin diuji
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}

# Inisialisasi model KNN
knn = KNeighborsClassifier()

# Inisialisasi Grid Search Cross-Validation
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)

# Melatih model menggunakan Grid Search Cross-Validation
grid_search.fit(X_train, y_train)

# Menampilkan hasil Grid Search
print("K optimal: ", grid_search.best_params_)

# Menggunakan model terbaik untuk prediksi pada data uji
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# Menghitung dan menampilkan akurasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi pada data uji: ", accuracy)

# Menyimpan model yang sudah dilatih ke file pickle
model_filename = "knn_model.pickle"
pickle.dump(best_knn, open(model_filename, "wb"))
```
<p style="text-indent: 50px; text-align: justify;">Menghitung KNN<p>
```{code-cell} python
# Import library yang dibutuhkan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Membaca data dari file CSV
data = pd.read_csv('https://raw.githubusercontent.com/Fentryca/Proyek-Sains-Data/main/dataset_tanpa_outlier.csv')

# Memisahkan data menjadi fitur (X) dan target (y)
X = data.drop('class', axis=1)  # Jika targetnya disebut 'target'
y = data['class']

# Normalisasi data menggunakan MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Membagi data menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Menginisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Melatih model KNN
knn.fit(X_train, y_train)

# Melakukan prediksi dengan data pengujian
y_pred = knn.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi model KNN:", accuracy)
```
