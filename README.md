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
