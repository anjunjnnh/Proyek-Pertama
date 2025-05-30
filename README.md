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
-  Algoritma machine learning apa (misalnya Gaussian Naive Bayes, Logistic Regression, dan Random Forest Classifier) yang memberikan kinerja terbaik dalam memprediksi dropout mahasiswa pada dataset yang digunakan?
-  Fitur-fitur apa dari dataset yang paling berkorelasi dengan status dropout mahasiswa dan dapat dijadikan indikator risiko dropout?
-  Bagaimana metrik evaluasi model (seperti accuracy, precision, recall, F1-score, dan confusion matrix) menunjukkan kinerja masing-masing model dalam mengidentifikasi mahasiswa yang berisiko dropout dan yang tidak berisiko dropout?

### Goals
Tujuan utama dari proyek ini adalah untuk mengembangkan, melatih, dan mengevaluasi beberapa model machine learning yang mampu memprediksi status dropout mahasiswa secara akurat. Proyek ini bertujuan untuk mengidentifikasi algoritma machine learning yang paling efektif dalam memprediksi dropout berdasarkan data yang tersedia, serta memberikan wawasan tentang faktor-faktor kunci yang berkontribusi terhadap risiko dropout.

### Solution Statement 
Proyek ini mengusulkan solusi berbasis machine learning untuk mengatasi tantangan identifikasi dropout mahasiswa. Solusi ini melibatkan pembangunan model klasifikasi menggunakan algoritma seperti Gaussian Naive Bayes, Logistic Regression, dan Random Forest Classifier pada dataset historis mahasiswa. Data akan melalui proses preprocessing termasuk encoding dan standarisasi. Kinerja model akan dievaluasi secara komprehensif menggunakan metrik standar seperti accuracy, precision, recall, dan F1-score, serta visualisasi confusion matrix. Hasil evaluasi ini akan digunakan untuk membandingkan kinerja model dan mengidentifikasi model terbaik untuk memprediksi dropout mahasiswa. Solusi ini bertujuan untuk memberikan institusi pendidikan alat prediktif yang berbasis data untuk mengidentifikasi mahasiswa berisiko dropout secara proaktif, memungkinkan intervensi yang tepat waktu dan meningkatkan retensi mahasiswa.

## Data Understanding

Tahap ini berfokus pada pemahaman mendalam terhadap dataset sebelum memulai pemrosesan dan pemodelan. Ini merupakan langkah krusial untuk memastikan data siap digunakan dan untuk mendapatkan wawasan awal.

Dataset ini bersumber dari: **https://www.kaggle.com/datasets/adilshamim8/predict-students-dropout-and-academic-success/**. Dataset ini berisi informasi tentang berbagai faktor yang dapat mempengaruhi status akademik mahasiswa.

Setelah data berhasil dimuat, pemeriksaan awal menunjukkan dataset memiliki **4424 baris dan 37 kolom**.

Pemeriksaan tipe data dan non-null count menggunakan `.info()` memberikan gambaran tentang jenis data yang terkandung dalam setiap kolom dan keberadaan nilai yang hilang.`.isnull().sum()` secara spesifik mengkonfirmasi tidak ada nilai yang hilang di dataset ini. Analisis statistik deskriptif menggunakan `.describe().T` merangkum distribusi fitur numerik seperti nilai rata-rata, standar deviasi, nilai minimum dan maksimum, serta kuartil, memberikan wawasan tentang rentang dan penyebaran data.

Eksplorasi Data (EDA) lebih lanjut dilakukan untuk memvisualisasikan distribusi variabel target (`target`) dan fitur-fitur penting lainnya menggunakan histogram, countplot, dan pie chart. Visualisasi ini membantu memahami sebaran status mahasiswa (Graduate, Dropout, Enrolled) dan distribusi fitur kategorikal seperti Gender. Analisis korelasi menggunakan heatmap juga dilakukan untuk mengidentifikasi hubungan linear antar fitur dan antara fitur dengan variabel target.

Beberapa fitur penting yang terdapat dalam dataset awal meliputi:
- **Marital Status**: Status pernikahan mahasiswa.
- **Application mode**: Mode aplikasi pendaftaran mahasiswa.
- **Course**: Program studi yang diambil mahasiswa.
- **Daytime/evening attendance**: Status kehadiran (siang/malam).
- **Previous qualification**: Kualifikasi pendidikan sebelumnya.
- **Nacionality**: Kebangsaan mahasiswa.
- **Mother's qualification**: Kualifikasi pendidikan ibu.
- **Father's qualification**: Kualifikasi pendidikan ayah.
- **Mother's occupation**: Pekerjaan ibu.
- **Father's occupation**: Pekerjaan ayah.
- **Educational special needs**: Kebutuhan khusus pendidikan.
- **Displaced**: Apakah mahasiswa mengungsi.
- **Debtor**: Apakah mahasiswa memiliki utang biaya pendidikan.
- **Tuition fees up to date**: Status pembayaran biaya kuliah.
- **Gender**: Jenis kelamin mahasiswa.
- **Scholarship holder**: Status penerima beasiswa.
- **Age at enrollment**: Usia saat pendaftaran.
- **International**: Apakah mahasiswa internasional.
- **Curricular units 1st sem (credited)**: Unit kurikuler semester 1 yang dikreditkan.
- **Curricular units 1st sem (enrolled)**: Unit kurikuler semester 1 yang didaftarkan.
- **Curricular units 1st sem (evaluations)**: Evaluasi unit kurikuler semester 1.
- **Curricular units 1st sem (approved)**: Unit kurikuler semester 1 yang disetujui/lulus.
- **Curricular units 1st sem (grade)**: Nilai rata-rata unit kurikuler semester 1.
- **Curricular units 1st sem (without evaluations)**: Unit kurikuler semester 1 tanpa evaluasi.
- **Curricular units 2nd sem (credited)**: Unit kurikuler semester 2 yang dikreditkan.
- **Curricular units 2nd sem (enrolled)**: Unit kurikuler semester 2 yang didaftarkan.
- **Curricular units 2nd sem (evaluations)**: Evaluasi unit kurikuler semester 2.
- **Curricular units 2nd sem (approved)**: Unit kurikuler semester 2 yang disetujui/lulus.
- **Curricular units 2nd sem (grade)**: Nilai rata-rata unit kurikuler semester 2.
- **Curricular units 2nd sem (without evaluations)**: Unit kurikuler semester 2 tanpa evaluasi.
- **Unemployment rate**: Tingkat pengangguran (indikator eksternal).
- **Inflation rate**: Tingkat inflasi (indikator eksternal).
- **GDP**: Produk Domestik Bruto (indikator eksternal).
- **Target**: Status akademik mahasiswa (Graduate, Dropout, Enrolled).

Temuan utama dari tahap ini adalah pemahaman mengenai struktur data, kualitas data, distribusi variabel target yang tidak seimbang (yang akan ditangani di tahap selanjutnya), serta identifikasi beberapa fitur yang menunjukkan korelasi dengan variabel target, seperti `Tuition fees up to date`, `Curricular units 1st sem (approved)`, `Curricular units 1st sem (grade)`, `Curricular units 2nd sem (approved)`, dan `Curricular units 2nd sem (grade)`. Informasi ini menjadi dasar untuk langkah-langkah persiapan data berikutnya.

## Data Preparation

Tahap ini mempersiapkan data untuk pembangunan model machine learning. Proses ini meliputi langkah-langkah sebagai berikut:

1.  **Encoding Variabel Target**: Sebelum pembersihan lebih lanjut, variabel target kategorikal (`target`) diubah menjadi representasi numerik menggunakan `LabelEncoder`. Langkah ini dilakukan untuk mengubah nilai-nilai seperti 'Graduate', 'Dropout', dan 'Enrolled' menjadi angka.
2.  **Menangani Data yang Tidak Sesuai**: Data yang memiliki nilai target yang di-encode dan tidak relevan dengan masalah klasifikasi biner antara "Dropout" dan "Non-Dropout" dihapus. Dalam notebook, ini secara spesifik dilakukan dengan menghapus baris di mana nilai `target` yang sudah di-encode adalah 1 (yang sebelumnya diidentifikasi sebagai status 'Enrolled').
3.  **Membuat Variabel Target Biner**: Setelah membersihkan data, dibuat kolom baru bernama `Dropout` sebagai variabel target biner. Nilai `target` yang sudah di-encode 0 (Dropout) dipetakan menjadi 1 (Dropout), sementara nilai `target` yang sudah di-encode 2 (Graduate) dipetakan menjadi 0 (Non-Dropout). Ini mengubah masalah menjadi klasifikasi biner yang akan digunakan untuk pemodelan.
4.  **Standarisasi Fitur**: Fitur numerik pada dataset distandarisasi menggunakan `StandardScaler`. Teknik ini mengubah distribusi fitur menjadi memiliki rata-rata 0 dan standar deviasi 1, yang penting untuk performa algoritma machine learning yang sensitif terhadap skala data.
5.  **Membagi Data Latih dan Uji**: Dataset kemudian dibagi menjadi set pelatihan (training set) dan set pengujian (testing set) menggunakan rasio 80:20. Pembagian ini krusial untuk mengevaluasi kinerja model secara objektif pada data yang belum pernah dilihat selama proses pelatihan.

## Modeling
Tiga algoritma klasifikasi digunakan untuk membangun model prediksi:
1.  [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) <br>
Gaussian Naive Bayes adalah algoritma klasifikasi yang didasarkan pada teorema Bayes dengan asumsi "naive" bahwa fitur-fitur adalah independen satu sama lain ketika diberikan kelas target. Model ini memperkirakan probabilitas bahwa suatu instance data termasuk dalam kelas tertentu berdasarkan probabilitas fitur-fiturnya, dengan menganggap distribusi setiap fitur dalam setiap kelas mengikuti distribusi Gaussian (normal). Keunggulannya terletak pada kesederhanaan dan efisiensinya, terutama pada dataset dengan dimensi tinggi. <br>
Untuk model ini, digunakan **parameter default** dari pustaka scikit-learn. Parameter default ini meliputi:
      - `priors`: None (probabilitas prior kelas dipelajari dari data).
      - `var_smoothing`: 1e-9 (nilai kecil untuk stabilitas numerik).

2. [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) <br>
Logistic Regression adalah algoritma klasifikasi linier yang digunakan untuk memprediksi probabilitas kelas biner. Meskipun dinamakan "regresi", ini adalah model klasifikasi yang menggunakan fungsi logistik (sigmoid) untuk memetakan output linier ke dalam probabilitas antara 0 dan 1. Model ini mempelajari hubungan linier antara fitur-fitur input dan log-odds dari kelas target. Keunggulan Logistic Regression adalah interpretasi yang relatif mudah dan efisiensi komputasi. <br>
Model Logistic Regression juga menggunakan **parameter default** scikit-learn. Beberapa parameter default kunci meliputi:
   - `penalty`: 'l2' (penalti regularisasi L2).
   - `C`: 1.0 (invers kekuatan regularisasi).
   - `solver`: 'lbfgs' (algoritma untuk optimasi).
   - `max_iter`: 100 (jumlah iterasi maksimum untuk konvergensi).

3. [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) <br>
Random Forest Classifier adalah algoritma ensemble yang membangun banyak pohon keputusan (decision trees) selama pelatihan dan menghasilkan prediksi kelas yang merupakan mode (kelas yang paling sering muncul) dari prediksi pohon-pohon individu. Setiap pohon dalam hutan dilatih pada subset data pelatihan yang diambil secara acak (bagging), dan pada setiap node pemisahan, hanya subset fitur yang dipilih secara acak yang dipertimbangkan (feature randomness). Kombinasi dari banyak pohon yang dilatih secara independen ini membantu mengurangi overfitting dan meningkatkan robustnes serta akurasi model. <br>
Untuk Random Forest Classifier, model dikonfigurasi dengan parameter spesifik:
   - `n_estimators`: **500**. Ini menentukan jumlah pohon yang akan dibangun dalam hutan. Penggunaan 500 pohon umumnya menghasilkan model yang lebih robust.
   - `criterion`: **'entropy'**. Ini adalah fungsi untuk mengukur kualitas pemisahan pada setiap node pohon. 'Entropy' mengukur ketidakmurnian informasi atau pengacakan.

Parameter lain yang tidak disebutkan secara eksplisit menggunakan nilai **default** dari scikit-learn, seperti `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `bootstrap=True`, dan `random_state=None`.

# 4. Model Development dan Evaluation

Pada tahap ini, dilakukan pembangunan dan evaluasi terhadap tiga model klasifikasi machine learning yang berbeda: Gaussian Naive Bayes, Logistic Regression, dan Random Forest Classifier. Tujuan utama adalah untuk memprediksi status putus studi mahasiswa berdasarkan data yang telah dipersiapkan.

Langkah-langkah utama yang dilakukan dalam tahap ini meliputi:
1.  **Pemilihan Model**: Memilih algoritma klasifikasi yang akan digunakan. Dalam proyek ini, dipilih Gaussian Naive Bayes, Logistic Regression, dan Random Forest Classifier karena karakteristik dan efisiensi komputasi mereka.
2.  **Pelatihan Model**: Masing-masing model dilatih menggunakan data pelatihan (x_train, y_train) yang telah dipisahkan di tahap persiapan data. Proses pelatihan ini memungkinkan model untuk mempelajari pola hubungan antara fitur-fitur input dan variabel target ('Dropout').
3.  **Prediksi pada Data Uji**: Setelah dilatih, masing-masing model digunakan untuk membuat prediksi terhadap data pengujian (x_test). Hasil prediksi ini akan dibandingkan dengan nilai target sebenarnya (y_test) untuk mengevaluasi kinerja model.
4.  **Evaluasi Kinerja Model**: Kinerja setiap model dievaluasi menggunakan berbagai metrik klasifikasi standar, yang dikumpulkan dalam fungsi `perform`. Metrik ini meliputi:
    -   **Akurasi (Accuracy)**: Proporsi prediksi yang benar dari total prediksi.
    -   **Presisi (Precision)**: Kemampuan model untuk tidak mengklasifikasikan instance negatif sebagai positif.
    -   **Recall (Sensitivity)**: Kemampuan model untuk menemukan semua instance positif.
    -   **F1-Score**: Rata-rata harmonis dari Presisi dan Recall, memberikan keseimbangan antara keduanya.
    -   **Confusion Matrix**: Tabel yang menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas.
    -   **Classification Report**: Ringkasan tekstual dari metrik Presisi, Recall, F1-Score, dan Support untuk setiap kelas.

Berikut adalah detail evaluasi untuk setiap model:

### a. Gaussian Naive Bayes

Setelah pelatihan dan prediksi, evaluasi model Gaussian Naive Bayes pada data uji menghasilkan metrik yang ditunjukkan pada output berikut :
![Screenshot 2025-05-30 230740](https://github.com/user-attachments/assets/9f56b069-4ef4-4fa1-8391-f6bf22a58f32)


### b. Logistic Regression

Hasil evaluasi model Logistic Regression pada data uji ditampilkan pada output berikut.
![Screenshot 2025-05-30 230348](https://github.com/user-attachments/assets/f3d3eb62-0a13-4e1d-9fc6-23672faf2c58)


### c. Random Forest Classifier

Evaluasi model Random Forest Classifier pada data uji menghasilkan metrik yang ditampilkan pada output berikut.
![Screenshot 2025-05-30 230932](https://github.com/user-attachments/assets/0dd48d0e-70bb-4407-966e-bf982120911c)


**Insight :** <br>

Pada tahap ini, tiga model klasifikasi machine learning dibangun dan dievaluasi: Gaussian Naive Bayes, Logistic Regression, dan Random Forest Classifier. Setiap model dilatih pada set data pelatihan dan kemudian digunakan untuk membuat prediksi pada set data pengujian. Evaluasi dilakukan menggunakan fungsi `perform` yang menghitung dan menampilkan berbagai metrik kinerja. **Temuan utama** dari tahap ini adalah metrik kinerja individual untuk setiap model pada set data pengujian, yang ditampilkan setelah eksekusi kode masing-masing model. Metrik ini menjadi dasar perbandingan di tahap selanjutnya.

# 5. Comparison of Results

Tahap ini membandingkan kinerja dari ketiga model yang telah dilatih dan dievaluasi, dengan fokus utama pada metrik Akurasi sebagai indikator utama kinerja prediksi.

### Hasil Perbandingan Akurasi

Hasil Akurasi dari setiap model yang dihitung pada data uji dirangkum sebagai berikut:

| Model                    | Akurasi |
|--------------------------|---------|
| Gaussian Naive Bayes     | ~0.846  |
| Logistic Regression      | ~0.910  |
| Random Forest Classifier | ~0.923  |

*Catatan: Nilai akurasi diambil dari output eksekusi kode.*

### Visualisasi Perbandingan

Perbandingan akurasi ketiga model divisualisasikan menggunakan plot bar horizontal. Visualisasi ini memudahkan identifikasi model dengan kinerja akurasi tertinggi.
![download](https://github.com/user-attachments/assets/8bd571b6-2bb2-4c27-9a24-42fd5ec9a28c)

Selain perbandingan akurasi, Confusion Matrix untuk setiap model juga telah divisualisasikan pada tahap evaluasi sebelumnya. Confusion Matrix memberikan detail mengenai jumlah True Positives, True Negatives, False Positives, dan False Negatives, yang sangat penting untuk memahami jenis kesalahan yang dibuat oleh setiap model.

Tahap ini membandingkan kinerja ketiga model berdasarkan metrik akurasi dan visualisasi. Hasil perbandingan akurasi menunjukkan bahwa model **Random Forest Classifier** mencapai akurasi tertinggi pada set data pengujian. Plot bar horizontal secara visual menegaskan temuan ini. Meskipun akurasi adalah metrik utama untuk perbandingan di sini, metrik lain seperti Precision, Recall, dan F1-Score memberikan gambaran yang lebih lengkap tentang performa masing-masing model. Berdasarkan evaluasi ini, Random Forest menjadi model yang paling menjanjikan untuk tugas prediksi putus studi ini.

### Dampak terhadap Business Understanding

Model prediksi dropout yang dibangun memiliki dampak signifikan bagi institusi pendidikan:
-   **Identifikasi Dini**: Memungkinkan identifikasi mahasiswa berisiko tinggi putus studi pada tahap awal.
-   **Intervensi Tepat Sasaran**: Informasi ini memungkinkan institusi untuk merancang dan memberikan intervensi yang lebih tepat waktu dan sesuai, baik itu dukungan akademik, finansial, atau konseling.
-   **Peningkatan Retensi Mahasiswa**: Dengan mengidentifikasi dan mendukung mahasiswa berisiko, institusi dapat meningkatkan tingkat retensi mahasiswa.
-   **Peningkatan Efisiensi Sumber Daya**: Intervensi yang ditargetkan lebih efisien dalam penggunaan sumber daya dibandingkan pendekatan umum.

## Referensi 
[1] https://download.garuda.kemdikbud.go.id/article.php?article=2259821&val=15965&title=Deteksi%20Dini%20Mahasiswa%20Drop%20Out%20Menggunakan%20C50

[2] https://opendata.jabarprov.go.id/id/infografik/alasan-mahasiswa-drop-out-atau-putus-kuliah,-apakah-gara-gara-skripsi-susah

[3] https://proceeding.unindra.ac.id/index.php/semnasristek/article/viewFile/5814/1426
