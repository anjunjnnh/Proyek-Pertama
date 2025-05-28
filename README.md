# Laporan Proyek Machine Learning - Anju Anjannah
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Pendidikan**, dengan judul **Student Dropout Prediction Using Predictive Analytics**  

### Latar Belakang

![foto mahasiswa](https://news.unair.ac.id/wp-content/uploads/2019/07/Ilustrasi-oleh-DiginationID.jpg)

Mahasiswa memiliki pengaruh yang cukup tinggi dalam akreditasi, sedangkan tidak sedikit
Perguruan Tinggi yang menerapkan drop out untuk mengatasi permasalahan mahasiswa,
misalnya IPK rendah, kurang serius dalam perkuliahan, hingga lama lulus. Hal ini yang membuat
banyak peneliti yang melakukan penelitian terhadap faktor yang menyebabkan mahasiswa drop
out. Penelitian yang sejenis dengan menggunakan algoritma data mining diterapkan pada solusi
prediksi mahasiswa drop out dengan melihat dari sisi SKS perkuliahan, IPK dan jumlah semester
yang telah dilalui (Utari et al., 2020). Dengan melakukan prediksi mahasiswa drop out, program
studi dapat memantau mahasiswanya yang terprediksi drop out sehingga mahasiswa
mendapatkan bimbingan secepatnya dan mahasiswa yang bersangkutan dapat lulus tepat waktu
(Putra, 2017). [[1]](https://download.garuda.kemdikbud.go.id/article.php?article=2259821&val=15965&title=Deteksi%20Dini%20Mahasiswa%20Drop%20Out%20Menggunakan%20C50)

Berdasarkan data, terdapat 602.208 mahasiswa di Indonesia yang memutuskan untuk berhenti kuliah dari total 8.483.213 mahasiswa yang terdaftar. Menariknya, perguruan tinggi swasta (PTS) menjadi tempat di mana fenomena ini paling sering terjadi. Pada tahun 2020, tercatat sebanyak 478.826 mahasiswa atau 79,5% dari total kasus drop out berasal dari PTS. Sementara itu, 101.758 orang berasal dari perguruan tinggi negeri (PTN), 18.284 orang dari perguruan tinggi agama (PTA), dan sisanya 3.395 orang dari perguruan tinggi kedinasan (PTK).[[2]](https://opendata.jabarprov.go.id/id/infografik/alasan-mahasiswa-drop-out-atau-putus-kuliah,-apakah-gara-gara-skripsi-susah)

Dalam era data saat ini, analisis prediktif menawarkan potensi besar untuk mengatasi tantangan ini. Dengan memanfaatkan data historis mahasiswa, kita dapat membangun model machine learning yang mampu mengidentifikasi pola dan faktor-faktor yang terkait dengan dropout. Model prediktif ini dapat memberikan wawasan berharga kepada institusi pendidikan, memungkinkan mereka untuk mengambil tindakan proaktif dalam mendukung mahasiswa yang berisiko, seperti menawarkan konseling, bimbingan akademik, atau bantuan finansial. Studi analitik prediktif untuk menentukan mahasiswa sarjana yang berisiko dropout telah menunjukkan akurasi yang bervariasi tergantung pada klasifikasi kriteria yang digunakan.

## Business Understanding 
Institusi pendidikan tinggi menghadapi tantangan signifikan dalam mempertahankan mahasiswa, yang berdampak pada pendapatan, reputasi, dan efisiensi operasional. Tingkat dropout yang tinggi menunjukkan perlunya pendekatan yang lebih proaktif dalam mengidentifikasi mahasiswa berisiko. Proses identifikasi manual saat ini tidak efisien dan akurat. Proyek ini bertujuan menyediakan model prediktif berbasis machine learning yang dapat diandalkan untuk mengidentifikasi mahasiswa berisiko dropout secara dini, memungkinkan institusi untuk melakukan intervensi tepat waktu, meningkatkan retensi, dan mengoptimalkan penggunaan sumber daya.
### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
-  Algoritma machine learning apa (misalnya Gaussian Naive Bayes, Logistic Regression, dan Random Forest) yang memberikan kinerja terbaik dalam memprediksi dropout mahasiswa pada dataset yang digunakan?
-  Fitur-fitur apa dari dataset yang paling berkorelasi dengan status dropout mahasiswa dan dapat dijadikan indikator risiko dropout?
-  Bagaimana metrik evaluasi model (seperti accuracy, precision, recall, F1-score, dan confusion matrix) menunjukkan kinerja masing-masing model dalam mengidentifikasi mahasiswa yang berisiko dropout dan yang tidak berisiko dropout?

### Goals
Tujuan utama dari proyek ini adalah untuk mengembangkan, melatih, dan mengevaluasi beberapa model machine learning yang mampu memprediksi status dropout mahasiswa secara akurat. Proyek ini bertujuan untuk mengidentifikasi algoritma machine learning yang paling efektif dalam memprediksi dropout berdasarkan data yang tersedia, serta memberikan wawasan tentang faktor-faktor kunci yang berkontribusi terhadap risiko dropout.

### Solution Statement 
Proyek ini mengusulkan solusi berbasis machine learning untuk mengatasi tantangan identifikasi dropout mahasiswa. Solusi ini melibatkan pembangunan model klasifikasi menggunakan algoritma seperti Gaussian Naive Bayes, Logistic Regression, dan Random Forest pada dataset historis mahasiswa. Data akan melalui proses preprocessing termasuk encoding dan standarisasi. Kinerja model akan dievaluasi secara komprehensif menggunakan metrik standar seperti accuracy, precision, recall, dan F1-score, serta visualisasi confusion matrix. Hasil evaluasi ini akan digunakan untuk membandingkan kinerja model dan mengidentifikasi model terbaik untuk memprediksi dropout mahasiswa. Solusi ini bertujuan untuk memberikan institusi pendidikan alat prediktif yang berbasis data untuk mengidentifikasi mahasiswa berisiko dropout secara proaktif, memungkinkan intervensi yang tepat waktu dan meningkatkan retensi mahasiswa.

## Data Understanding

Tahap *Data Understanding* bertujuan untuk memahami struktur, kualitas, dan karakteristik dataset yang digunakan. Proses ini dilakukan melalui analisis eksploratif data (EDA) untuk mendapatkan insight awal dan mempersiapkan strategi pemrosesan lebih lanjut.

### 1. Memahami Data

#### a. Memuat Dataset  
Dataset berkaitan dengan prediksi dropout dan kesuksesan akademik siswa.

#### b. Kondisi Dataset  
Sebelum analisis lebih lanjut, dilakukan pemeriksaan terhadap kondisi dataset:
- **Nilai Hilang**: Memeriksa dan menangani kolom/baris kosong.
- **Duplikasi**: Menghapus baris duplikat jika ditemukan.

### 2. Analisis Data Eksploratif (EDA)

EDA dilakukan untuk:
- Memahami distribusi dan karakteristik fitur
- Menilai keseimbangan kelas target
- Menemukan hubungan antar variabel

Visualisasi yang digunakan antara lain:
- Histogram, piechart, countplot
- Korelasi fitur menggunakan heatmap

### 3. Insight dari EDA

Beberapa temuan penting dari eksplorasi awal:

### Fitur Akademik Sangat Mempengaruhi Dropout
Beberapa fitur akademik seperti:

- *Curricular units 2nd sem (approved)*  
- *Curricular units 2nd sem (grade)*  
- *Curricular units 2nd sem (enrolled)*  

memiliki korelasi negatif terhadap dropout. Hal ini menunjukkan bahwa siswa dengan performa akademik yang baik cenderung bertahan.

### Faktor Sosial dan Ekonomi Juga Berperan
Fitur-fitur berikut mengindikasikan bahwa kemampuan finansial siswa dapat memengaruhi keputusan untuk melanjutkan studi:

- *Tuition fees up to date*  
- *Scholarship holder*  
- *Debtor*  

### Fitur Demografis Kurang Relevan
Beberapa fitur seperti *Marital Status*, *Mother's/Father's occupation*, dan *Nationality* menunjukkan korelasi rendah terhadap target dropout, sehingga perlu dipertimbangkan kembali kegunaannya dalam pemodelan.

## Data Preparation

Persiapan data adalah langkah krusial sebelum membangun model machine learning. Tujuan tahap ini adalah membersihkan, memformat, dan menyesuaikan data agar dapat digunakan secara optimal dalam proses pelatihan dan evaluasi model.

### Langkah-langkah Persiapan Data

1. **Menangani Data yang Tidak Sesuai**
   - Data yang memiliki nilai target tidak relevan akan dihapus. Dalam konteks proyek ini, hanya nilai `target = 0` dan `target = 2` yang valid.
   - Dihapus semua baris yang memiliki nilai target selain 0 dan 2 (jika ada).
   - Selanjutnya, dibuat kolom baru bernama `Dropout` dengan ketentuan:
     - `target == 0` berarti **siswa dropout** → `Dropout = 1`
     - `target == 2` berarti **siswa tidak dropout** → `Non-Dropout = 0`

2. **Standarisasi Data**
   - Fitur numerik dinormalisasi menggunakan **StandardScaler**, yaitu teknik transformasi yang mengubah distribusi fitur menjadi memiliki rata-rata 0 dan standar deviasi 1.
   - Standarisasi ini penting karena banyak algoritma machine learning sensitif terhadap skala data.
4. **Membagi Data Latih dan Uji**
   - Dataset dibagi menjadi **data latih (training set)** dan **data uji (testing set)** menggunakan rasio 80:20.
   - Pembagian ini dilakukan untuk mengevaluasi kinerja model terhadap data yang belum pernah dilihat saat pelatihan, sehingga hasil evaluasi lebih objektif.


## Modeling
Tiga algoritma klasifikasi digunakan untuk membangun model prediksi:
1. [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
2. [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
3. [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Langkah-langkah Modeling

1. **Persiapan Data**: pembersihan, encoding, dan pembagian data (train-test split)
2. **Pelatihan Model**: masing-masing algoritma dilatih dengan data latih
3. **Evaluasi Model**: menggunakan metrik klasifikasi
   - Akurasi
   - Presisi
   - Recall
   - F1-score
4. **Visualisasi**: perbandingan metrik antar model, confusion matrix, dll.

## Hasil Evaluasi

| Model                    | Akurasi | Precision | Recall | F1-Score |
|--------------------------|---------|-----------|--------|----------|
| Gaussian Naive Bayes     | 0.85    | 0.85      | 0.85   | 0.85     |
| Logistic Regression      | 0.91    | 0.91      | 0.91   | 0.91     |
| Random Forest Classifier | 0.92    | 0.92      | 0.92   | 0.92     |

Random Forest Classifier menunjukkan performa terbaik.
