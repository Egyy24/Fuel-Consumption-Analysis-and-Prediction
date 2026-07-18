# Fuel Consumption Analysis and Prediction

## Anggota Kelompok

- Alif Zuhair (24523213)
- Qanzul Arays (23523228)
- Muhammad Ibnu Rasyid (23523127)

## Deskripsi Proyek

Proyek ini merupakan tugas akhir mata kuliah Fundamental Data Science yang terdiri dari dua studi kasus menggunakan dataset yang sama, yaitu data konsumsi bahan bakar kendaraan:

1. **Supervised Learning (Regresi)** - Memprediksi konsumsi bahan bakar gabungan kendaraan (`FUELCONSUMPTION_COMB`) berdasarkan spesifikasi kendaraan seperti ukuran mesin, jumlah silinder, kelas kendaraan, jenis transmisi, dan jenis bahan bakar. Model terbaik dari studi kasus ini kemudian di-deploy sebagai aplikasi web interaktif menggunakan Gradio.
2. **Unsupervised Learning (Clustering)** - Mengelompokkan kendaraan ke dalam beberapa segmen berdasarkan karakteristik mesin dan tingkat emisi CO2 menggunakan algoritma K-Means, untuk mengidentifikasi kategori kendaraan ramah lingkungan hingga kendaraan dengan emisi tinggi.

## Struktur File

```
PROJECT/
├── sk1-supervised learning/
│   ├── myenv/                                    # Virtual environment (tidak perlu di-commit)
│   ├── app.py                                    # Aplikasi Gradio untuk demo model prediksi
│   ├── fuel_consumption_prediction_model.pkl     # Model hasil training (model, preprocessor, nama fitur)
│   ├── FuelConsumptionSupervised.ipynb           # Notebook analisis dan pemodelan regresi
│   └── requirements.txt                          # Daftar library yang dibutuhkan
└── sk2-unsupervised learning/
    ├── venv/                                     # Virtual environment (tidak perlu di-commit)
    └── FuelConsumptionUnsupervised.ipynb         # Notebook analisis clustering (K-Means)
```

| File / Folder | Deskripsi |
|---|---|
| `sk1-supervised learning/FuelConsumptionSupervised.ipynb` | Notebook analisis dan pemodelan regresi (Linear Regression dan Random Forest Regressor) |
| `sk1-supervised learning/app.py` | Aplikasi Gradio untuk mendemonstrasikan model prediksi konsumsi bahan bakar |
| `sk1-supervised learning/fuel_consumption_prediction_model.pkl` | File model hasil training (model, preprocessor, dan nama fitur) yang dihasilkan dari notebook supervised |
| `sk1-supervised learning/requirements.txt` | Daftar dependency Python yang dibutuhkan untuk menjalankan `app.py` |
| `sk1-supervised learning/myenv/` | Folder virtual environment untuk studi kasus supervised |
| `sk2-unsupervised learning/FuelConsumptionUnsupervised.ipynb` | Notebook analisis clustering (K-Means) untuk segmentasi kendaraan |
| `sk2-unsupervised learning/venv/` | Folder virtual environment untuk studi kasus unsupervised |

## Sumber Dataset

Dataset yang digunakan berjudul **Fuel Consumption**, berisi data konsumsi bahan bakar berbagai jenis kendaraan tahun 2014, dengan total 1.067 baris data dan 13 kolom (fitur).

- **Sumber asli**: Kaggle, dataset "Fuel Consumption" oleh Sarita:
  `https://www.kaggle.com/datasets/sarita19/fuel-consumption`
- **Notebook Supervised**: dataset dimuat dari file lokal `FuelConsumption.csv` yang diunggah ke Google Colab (`/content/FuelConsumption.csv`).
- **Notebook Unsupervised**: dataset yang sama dimuat langsung dari repositori GitHub berikut:
  `https://raw.githubusercontent.com/alifzuhair/TUGAS-AKHIR-UNSUPERVISED-FSD/refs/heads/main/FuelConsumption.csv`

## Studi Kasus 1: Supervised Learning (Prediksi Konsumsi Bahan Bakar)

**Fitur Input (X):**
- `ENGINESIZE` - ukuran mesin (Liter)
- `CYLINDERS` - jumlah silinder
- `VEHICLECLASS` - kelas kendaraan
- `TRANSMISSION` - jenis transmisi
- `FUELTYPE` - jenis bahan bakar (X=Reguler, Z=Premium, D=Diesel, E=Ethanol)

**Target Prediksi (y):**
- `FUELCONSUMPTION_COMB` - konsumsi bahan bakar gabungan kota & tol (L/100km)

**Kolom yang dikecualikan:** `MODELYEAR` (tidak bervariasi, semua tahun 2014), `MAKE` dan `MODEL` (kardinalitas terlalu tinggi), serta `FUELCONSUMPTION_CITY`, `FUELCONSUMPTION_HWY`, `FUELCONSUMPTION_COMB_MPG`, dan `CO2EMISSIONS` (berpotensi menyebabkan data leakage karena sangat berkorelasi langsung dengan target).

**Tahapan analisis:**
1. Deskripsi dataset dan pemilihan fitur
2. Exploratory Data Analysis (distribusi data, hubungan antar fitur, korelasi)
3. Preprocessing (pengecekan duplikat dan outlier, pengelompokan kategori jarang menjadi "Other", Standard Scaler untuk fitur numerik, One-Hot Encoding untuk fitur kategorikal)
4. Pemilihan dan perbandingan algoritma: Linear Regression (baseline) dan Random Forest Regressor
5. Training dan evaluasi model menggunakan MAE, RMSE, dan R² Score
6. Analisis feature importance
7. Penyimpanan model terlatih (Random Forest) ke dalam file `.pkl` menggunakan joblib

**Hasil:** Random Forest Regressor terbukti memberikan performa terbaik (R² sekitar 0.99) dibandingkan Linear Regression, sehingga dipilih sebagai model final yang di-deploy pada `app.py`.

## Studi Kasus 2: Unsupervised Learning (Segmentasi Kendaraan)

Analisis ini bertujuan mengelompokkan kendaraan berdasarkan efisiensi bahan bakar dan tingkat emisi CO2 menggunakan algoritma **K-Means Clustering**.

**Fitur yang digunakan:**
- `ENGINESIZE`, `CYLINDERS`, `FUELCONSUMPTION_COMB`, `CO2EMISSIONS`

**Tahapan analisis:**
1. Standardisasi fitur numerik menggunakan StandardScaler
2. Penentuan jumlah cluster optimal menggunakan Elbow Method (WCSS)
3. Clustering dengan K-Means (k=3)
4. Visualisasi hasil clustering dan interpretasi karakteristik tiap cluster

**Hasil segmentasi menghasilkan 3 kelompok kendaraan:**
- **Cluster 0** - Kendaraan Kelas Menengah (Standard / Family Cars): ukuran mesin 2.5–4.0 L, emisi CO2 moderat (200–350 g/km)
- **Cluster 1** - Kendaraan Ekonomis / Ramah Lingkungan (Eco-Friendly): ukuran mesin 1.0–2.5 L, emisi CO2 terendah (mayoritas di bawah 250 g/km)
- **Cluster 2** - Kendaraan Performa Tinggi / Berbobot Berat (High Performance / Heavy Duty): ukuran mesin 4.0 L ke atas, emisi CO2 tertinggi (di atas 300 hingga 500 g/km)

## Aplikasi Prediksi (app.py)

`app.py` adalah aplikasi web sederhana berbasis **Gradio** yang menggunakan model Random Forest Regressor hasil training dari `FuelConsumptionSupervised.ipynb`. Pengguna dapat memasukkan spesifikasi kendaraan (ukuran mesin, jumlah silinder, kelas kendaraan, jenis transmisi, dan jenis bahan bakar) melalui antarmuka interaktif, dan aplikasi akan menampilkan hasil prediksi konsumsi bahan bakar gabungan dalam satuan L/100km.

## Cara Menjalankan Notebook

### 1. sk1-supervised learning/FuelConsumptionSupervised.ipynb

Notebook ini dirancang untuk dijalankan di **Google Colab** karena menggunakan modul `google.colab`.

1. Buka notebook `sk1-supervised learning/FuelConsumptionSupervised.ipynb` di Google Colab.
2. Siapkan file `FuelConsumption.csv` dan unggah ke sesi Colab (path yang digunakan: `/content/FuelConsumption.csv`). Bisa diunggah manual melalui panel file di Colab, atau menyesuaikan kode pembacaan data jika ingin membaca dari sumber lain.
3. Jalankan seluruh cell secara berurutan dari atas ke bawah (Runtime > Run all).
4. Pada bagian akhir notebook, model akan disimpan sebagai file `fuel_consumption_prediction_model.pkl` dan otomatis diunduh ke komputer melalui `files.download()`.
5. Pindahkan file `fuel_consumption_prediction_model.pkl` hasil download ke folder `sk1-supervised learning/` (menggantikan file lama) agar `app.py` menggunakan model terbaru.

Jika ingin menjalankan di lingkungan lokal (VS Code, Jupyter lokal, dll), hapus atau sesuaikan baris `from google.colab import files` dan `files.download(filename)`, karena fungsi tersebut hanya tersedia di Google Colab.

### 2. sk2-unsupervised learning/FuelConsumptionUnsupervised.ipynb

Notebook ini dapat dijalankan baik di Google Colab maupun lingkungan Jupyter lokal, karena dataset dimuat langsung dari URL GitHub (tidak perlu upload file manual).

1. Buka notebook `sk2-unsupervised learning/FuelConsumptionUnsupervised.ipynb` di Google Colab atau Jupyter/VS Code lokal.
2. Pastikan komputer terhubung ke internet karena dataset diambil langsung dari URL raw GitHub.
3. Jalankan seluruh cell secara berurutan dari atas ke bawah.

### 3. Menjalankan app.py (Aplikasi Gradio)

1. Masuk ke folder `sk1-supervised learning/`:
   ```
   cd "sk1-supervised learning"
   ```
2. Pastikan file `fuel_consumption_prediction_model.pkl` (hasil dari notebook supervised) berada pada folder yang sama dengan `app.py`.
3. (Opsional) Buat dan aktifkan virtual environment terlebih dahulu, misalnya menggunakan folder `myenv` yang sudah ada:
   ```
   python -m venv myenv
   myenv\Scripts\activate      # Windows
   source myenv/bin/activate   # macOS/Linux
   ```
4. Install library yang dibutuhkan melalui `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
   Atau, jika belum memiliki `requirements.txt`, install manual:
   ```
   pip install gradio joblib pandas scikit-learn
   ```
5. Jalankan aplikasi melalui terminal:
   ```
   python app.py
   ```
6. Setelah berjalan, Gradio akan menampilkan tautan lokal (dan tautan publik sementara jika `share=True` diaktifkan) yang dapat dibuka di browser untuk mencoba prediksi konsumsi bahan bakar secara interaktif.

## Catatan Kompatibilitas

Jika terjadi error terkait versi scikit-learn saat memuat file `.pkl` (misalnya perbedaan versi antara Google Colab dan lingkungan lokal), pastikan versi scikit-learn yang digunakan untuk memuat model (di `app.py`) sama dengan versi yang digunakan saat melatih dan menyimpan model di Colab. Versi dapat dicek dengan:
```
pip show scikit-learn
```
dan disesuaikan menggunakan:
```
pip install scikit-learn==<versi_yang_sesuai>
```

## Library yang Digunakan

- pandas, numpy
- matplotlib, seaborn
- scikit-learn (model_selection, linear_model, ensemble, preprocessing, compose, metrics, pipeline, cluster)
- joblib
- gradio
