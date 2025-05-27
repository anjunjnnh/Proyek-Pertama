# Laporan Proyek Machine Learning - Anju Anjannah
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Pendidikan**, dengan judul **Student Dropout Prediction Using Predictive Analytics**  

### Latar Belakang

![foto mahasiswa](https://news.unair.ac.id/wp-content/uploads/2019/07/Ilustrasi-oleh-DiginationID.jpg)

Angka dropout mahasiswa merupakan salah satu tantangan terbesar yang dihadapi oleh institusi pendidikan tinggi di seluruh dunia. Mahasiswa yang tidak menyelesaikan pendidikannya tidak hanya mengalami kerugian pribadi dalam hal pencapaian akademik dan prospek karier, tetapi juga memberikan dampak negatif pada institusi dalam bentuk penurunan tingkat kelulusan, reputasi, dan efisiensi sumber daya. Mengidentifikasi mahasiswa yang berisiko dropout pada tahap awal adalah krusial untuk memberikan dukungan yang tepat waktu dan efektif. Penelitian menunjukkan bahwa penggunaan analisis prediktif dalam pengaturan pendidikan memberikan wawasan praktis untuk pengembangan kebijakan dan intervensi yang bertujuan mengurangi tingkat dropout [2].

Saat ini, identifikasi risiko dropout sering kali mengandalkan pendekatan manual yang subjektif, memakan waktu, dan seringkali tidak akurat. Keterbatasan ini dapat mengakibatkan terlambatnya intervensi, sehingga peluang untuk mempertahankan mahasiswa menjadi berkurang.

Dalam era data saat ini, analisis prediktif menawarkan potensi besar untuk mengatasi tantangan ini. Dengan memanfaatkan data historis mahasiswa, kita dapat membangun model machine learning yang mampu mengidentifikasi pola dan faktor-faktor yang terkait dengan dropout. Model prediktif ini dapat memberikan wawasan berharga kepada institusi pendidikan, memungkinkan mereka untuk mengambil tindakan proaktif dalam mendukung mahasiswa yang berisiko, seperti menawarkan konseling, bimbingan akademik, atau bantuan finansial. Studi analitik prediktif untuk menentukan mahasiswa sarjana yang berisiko dropout telah menunjukkan akurasi yang bervariasi tergantung pada klasifikasi kriteria yang digunakan [1].

## Business Understanding 
Institusi pendidikan tinggi menghadapi tantangan signifikan dalam mempertahankan mahasiswa, yang berdampak pada pendapatan, reputasi, dan efisiensi operasional. Tingkat dropout yang tinggi menunjukkan perlunya pendekatan yang lebih proaktif dalam mengidentifikasi mahasiswa berisiko. Proses identifikasi manual saat ini tidak efisien dan akurat. Proyek ini bertujuan menyediakan model prediktif berbasis machine learning yang dapat diandalkan untuk mengidentifikasi mahasiswa berisiko dropout secara dini, memungkinkan institusi untuk melakukan intervensi tepat waktu, meningkatkan retensi, dan mengoptimalkan penggunaan sumber daya.
### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
-  Algoritma machine learning apa (misalnya, Naive Bayes, Logistic Regression, Random Forest) yang memberikan kinerja terbaik dalam memprediksi dropout mahasiswa pada dataset yang digunakan?
-  Fitur-fitur apa dari dataset yang paling berkorelasi dengan status dropout mahasiswa dan dapat dijadikan indikator risiko dropout?
-  Bagaimana metrik evaluasi model (seperti accuracy, precision, recall, F1-score, dan confusion matrix) menunjukkan kinerja masing-masing model dalam mengidentifikasi mahasiswa yang berisiko dropout dan yang tidak berisiko dropout?

### Goals
Tujuan utama dari proyek ini adalah untuk mengembangkan, melatih, dan mengevaluasi beberapa model machine learning yang mampu memprediksi status dropout mahasiswa secara akurat. Proyek ini bertujuan untuk mengidentifikasi algoritma machine learning yang paling efektif dalam memprediksi dropout berdasarkan data yang tersedia, serta memberikan wawasan tentang faktor-faktor kunci yang berkontribusi terhadap risiko dropout.

### Solution Statement 
Proyek ini mengusulkan solusi berbasis machine learning untuk mengatasi tantangan identifikasi dropout mahasiswa. Solusi ini melibatkan pembangunan model klasifikasi menggunakan algoritma seperti Gaussian Naive Bayes, Logistic Regression, dan Random Forest pada dataset historis mahasiswa. Data akan melalui proses preprocessing termasuk encoding dan standarisasi. Kinerja model akan dievaluasi secara komprehensif menggunakan metrik standar seperti accuracy, precision, recall, dan F1-score, serta visualisasi confusion matrix. Hasil evaluasi ini akan digunakan untuk membandingkan kinerja model dan mengidentifikasi model terbaik untuk memprediksi dropout mahasiswa. Solusi ini bertujuan untuk memberikan institusi pendidikan alat prediktif yang berbasis data untuk mengidentifikasi mahasiswa berisiko dropout secara proaktif, memungkinkan intervensi yang tepat waktu dan meningkatkan retensi mahasiswa.

## Data Understanding
Data Understanding yang mencakup analisis eksploratif data (EDA) dengan penjelasan mengenai kondisi dataset, serta visualisasi yang relevan untuk mendukung insight yang disampaikan:
1. Memahami Data <br>
   Memahami data adalah langkah awal yang krusial untuk menganalisis informasi dan kualitas data. Pada bagian ini, kita akan memuat dataset, menjelaskan kondisi dataset, serta melakukan analisis eksploratif untuk mendapatkan wawasan yang lebih dalam.<br>
a. Memuat Data<br>
Dataset yang digunakan dalam proyek ini adalah mengenai gaji karyawan berdasarkan  pengalaman kerja. Dataset dapat diakses melalui tautan ini (https://www.kaggle.com/datasets/adilshamim8/predict-students-dropout-and-academic-success/code).<br>
b.  Kondisi Dataset <br>
Sebelum melakukan analisis lebih lanjut, penting untuk memahami kondisi dataset. Kita akan memeriksa adanya nilai hilang, duplikasi, dan outlier. <br>
    - Nilai Hilang: Memeriksa jumlah nilai hilang dalam dataset.
    - Duplikasi: Memeriksa apakah terdapat baris yang duplikat.
c.  Analisis Data Eksploratif (EDA) <br>
Analisis data eksploratif adalah proses untuk menganalisis karakteristik data, menemukan pola, dan memeriksa asumsi.<br>
d. Insight dari EDA <br>
Visualisasi awal dan eksplorasi data menggunakan distribusi frekuensi dan heatmap korelasi membantu memahami hubungan antar fitur.<br>

## Data Preparation
Persiapan data adalah langkah penting untuk mempersiapkan data sebelum membangun model machine learning. Pada bagian ini, kita akan fokus pada beberapa proses utama yang diperlukan untuk memastikan data siap digunakan dalam pemodelan.
Langkah-langkah yang diambil dalam proses persiapan data sebagai berikut:
1. Menangani data yang kurang sesuai
- Menghapus semua baris dari DataFrame data yang memiliki nilai 1 pada kolom target.
- Membuat kolom baru bernama 'Dropout' di DataFrame data, berdasarkan nilai dari kolom 'target' dengan asumsi :
  - target == 0 berarti orang tersebut dropout, sehingga Dropout = 1.
  - target == 1 berarti tidak dropout, sehingga Dropout = 0.
3. Mengatasi Outlier
  - Outlier dapat mempengaruhi hasil analisis dan model yang dibangun. Oleh karena itu, kita perlu mengidentifikasi dan menangani outlier dalam dataset. Dalam proyek ini, kita menggunakan metode Z-score untuk mendeteksi outlier. Baris yang memiliki Z-score di atas threshold yang ditentukan akan dihapus dari dataset.
3. Pembagian Data Latih dan Uji
  - Setelah menangani missing value dan outlier, langkah selanjutnya adalah membagi dataset menjadi data latih dan data uji. Pembagian ini penting untuk memastikan bahwa model dapat dievaluasi dengan baik. Dalam proyek ini, kita menggunakan 10% dari data sebagai data uji.
4. Standarisasi Data
  - Standarisasi data adalah langkah penting untuk memastikan bahwa fitur-fitur dalam dataset memiliki skala yang sama. Hal ini dapat membantu algoritma machine learning dalam proses pelatihan. Kita akan melakukan standarisasi pada fitur numerik menggunakan StandardScaler.
