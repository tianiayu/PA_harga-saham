# Predictive Analytics - Perbandingan Model KNN, Linear Regression, dan LSTM dalam Memprediksi Harga Saham Nvidia
# **Laporan Proyek Machine Learning - Tiani Ayu Lestari**
## Domain Proyek

Nvidia adalah perusahaan besar yang bergerak di bidang teknologi grafis, AI, dan industri game, sehingga membuat harga sahamnya cenderung fluktuatif dan menarik untuk dianalisis. Pergerakan dan perubahan harga saham Nvidia kerap kali dipengaruhi oleh sejumlah faktor eksternal, termasuk inovasi teknologi, kebijakan ekonomi, serta dinamika pasar global. Karena fluktuasi harga saham yang signifikan, banyak investor yang berusaha mencari cara untuk memprediksi pergerakan harga dengan menggunakan data historis.

- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan:
1. Prediksi Harga Saham memiliki peran penting dalam pengambilan keputusan investasi. Menggunakan teknik Machine Learning dan Deep Learning untuk memprediksi harga saham memungkinkan investor untuk membuat keputusan yang lebih terinformasi dan mengurangi risiko kerugian.
2. Prediksi saham menggunakan model regresi sudah banyak digunakan, namun dengan kemajuan teknologi, model deep learning seperti LSTM dapat menawarkan hasil yang lebih akurat, terutama dalam menganalisis data time series yang kompleks.
3. Meskipun banyak metode yang dapat diterapkan, memprediksi harga saham tetap merupakan masalah yang menantang karena adanya faktor eksternal yang tidak selalu bisa diprediksi secara akurat.

- Referensi:
1. "Stock Market Prediction Using Machine Learning: A Survey" by Smith, J., et al. (2020): Artikel ini mengulas penggunaan berbagai algoritma machine learning dalam memprediksi harga saham, termasuk analisis tentang kekuatan dan kelemahan masing-masing pendekatan.
a. ANN	Akurasi tinggi mencapai 90%, kekurangannya Overfitting dan interpretasi sulit.
b. HMM (Hidden Markov Model), kekurangannya	Optimasi efektif	Kompleksitas evaluasi.
c. ARIMA	Efisien untuk data jangka pendek, kekurangannya Terbatas pada prediksi jangka pendek.
d. RNN	Memproses data temporal, kekurangannya Input nodenya terbatas.

2. "Deep Learning for Time Series Forecasting: A Survey" by Zhang, L., et al. (2021): Buku ini membahas berbagai teknik deep learning, termasuk LSTM, yang dapat diterapkan untuk memprediksi data time series seperti harga saham. Kelebihan, Kemampuan model pola non-linear dan dinamik dan Skalabilitas tinggi dengan GPU. Keterbatasan, Biaya komputasi tinggi dan Risiko overfitting pada data kecil.

## Business Understanding

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1: Fluktuasi harga saham yang tidak menentu menyulitkan investor dalam membuat keputusan investasi secara cepat dan tepat.
- Pernyataan Masalah 2: Model linear seperti regresi sederhana mungkin tidak cukup akurat dalam menganalisis data time series yang dipengaruhi banyak faktor.
- Pernyataan Masalah 3: Masih belum dapat dipastikan model mana yang lebih unggul dalam memprediksi harga saham, apakah metode machine learning seperti regresi atau deep learning seperti LSTM.

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1: Membangun sistem prediksi harga saham Nvidia berdasarkan data historis. Sistem ini akan membantu memberikan gambaran tren harga ke depan yang bisa dimanfaatkan oleh investor sebagai alat bantu analisis.
- Jawaban pernyataan masalah 2: Membandingkan performa model KNN, regresi linear dan LSTM dalam memprediksi harga saham. Ini dilakukan untuk mengetahui metode mana yang memiliki akurasi lebih tinggi dan generalisasi lebih baik terhadap data.
- Jawaban pernyataan masalah 3: Mengukur performa model menggunakan metrik evaluasi seperti MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), dan R² Score. Metrik ini akan membantu menentukan model mana yang lebih baik dalam memprediksi harga saham.

### Solution statements
- Menggunakan model K-Nearest Neighbors (KNN) dan Regresi Linier sebagai baseline untuk memprediksi harga saham berdasarkan data historis. Keduanya dapat menjadi solusi sederhana yang cepat dibangun dan menjadi pembanding awal sebelum menggunakan metode yang lebih kompleks.
- Membangun model LSTM (Long Short-Term Memory) untuk memprediksi harga saham dengan mempertimbangkan urutan waktu dan pola dalam data.
LSTM cocok untuk data time series karena memiliki memori jangka panjang yang memungkinkan model mengingat informasi dari urutan sebelumnya.
- Melakukan perbandingan performa antara KNN, Regresi Linier, dan LSTM berdasarkan metrik evaluasi (MAE, RMSE, dan R2 Score). Ini dilakukan untuk memilih model yang paling optimal dan layak digunakan dalam implementasi nyata. Ini dilakukan untuk memilih model yang paling optimal dan layak digunakan dalam implementasi nyata.

## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan data historis harga saham perusahaan teknologi Nvidia (NVDA), yang diperoleh dari situs Investing.com. Dataset ini berisi informasi harian mengenai harga saham Nvidia selama periode tertentu, dan digunakan sebagai dasar untuk membangun model prediksi harga saham. Data dapat diunduh melalui tautan berikut:
https://www.investing.com/equities/nvidia-corp-historical-data

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
Dataset memiliki sejumlah variabel yang relevan untuk analisis dan pemodelan, antara lain:
- Date: Tanggal pencatatan harga saham.
- Price: Harga penutupan (closing price) saham Nvidia pada tanggal tersebut.
- Open: Harga pembukaan saham pada awal perdagangan hari tersebut.
- High: Harga tertinggi yang dicapai saham selama hari tersebut.
- Low: Harga terendah yang dicapai saham selama hari tersebut.
- Vol. (Volume): Jumlah saham yang diperdagangkan dalam satu hari.
- Change %: Persentase perubahan harga saham dibandingkan dengan hari sebelumnya.

Semua fitur tersebut bersifat numerik, kecuali tanggal. Dan sangat penting untuk dianalisis secara visual maupun statistik sebelum digunakan dalam pemodelan.

Untuk memahami karakteristik data secara lebih mendalam, dilakukan beberapa tahapan Exploratory Data Analysis, antara lain:
- Visualisasi Tren Harga Saham (Price) dari Waktu ke Waktu. Tujuannya untuk melihat pola umum seperti tren naik atau turun, serta fluktuasi harga harian. Visualisasi ini membantu dalam memahami perilaku harga saham dalam periode waktu tertentu, yang penting untuk analisis prediktif selanjutnya.
- Distribusi Volume Perdagangan Saham (Volume). Digunakan untuk mengetahui periode dengan aktivitas perdagangan tinggi atau rendah. Dengan mengetahui distribusi volume, kita bisa memahami kapan saham tersebut lebih banyak diperdagangkan dan apakah ada hubungan antara volume dan harga saham.
- Korelasi Antar Variabel seperti Price, Open, High, Low. Dilakukan untuk memahami hubungan antar fitur dan memastikan relevansi terhadap target prediksi. Korelasi antar variabel memberikan wawasan mengenai bagaimana masing-masing fitur saling terkait, yang bisa menjadi pertimbangan penting dalam pemilihan fitur yang digunakan untuk membangun model prediksi.
- Pembersihan Data (Data Cleaning). Melakukan pembersihan data dengan menghapus simbol seperti “K” atau “M” pada volume dan mengubahnya menjadi angka numerik yang dapat diolah oleh model. Selain itu, kolom tanggal juga dikonversi menjadi format datetime agar lebih mudah digunakan dalam analisis dan pemodelan berbasis waktu.

Hasil dari tahap ini memberikan dasar yang kuat dalam memahami konteks data mengidentifikasi potensi masalah yang perlu diperbaiki, dan memastikan data siap digunakan dalam proses persiapan dan pemodelan lebih lanjut.

## Data Preparation
Pada bagian ini, dilakukan beberapa tahapan persiapan data yang bertujuan untuk memastikan data siap digunakan dalam pemodelan. Tahapan-tahapan ini berfokus pada pembersihan data, pemrosesan, dan transformasi data agar dapat digunakan secara efektif oleh model prediksi. Proses ini sangat penting untuk memastikan bahwa model dapat menghasilkan hasil yang akurat.

**Langkah-langkah Data Preparation**:
- Pembersihan Data (Data Cleaning)
  - Menghapus Simbol pada Kolom Volume. Pada kolom Volume, terdapat simbol “K” dan “M” yang menunjukkan ribuan dan jutaan. Simbol-simbol ini perlu dihapus dan diubah menjadi angka numerik agar bisa diproses oleh model. Misalnya, “100K” diubah menjadi 100,000 dan “2M” menjadi 2,000,000.
  - Mengonversi Kolom Tanggal ke Format DateTime: Kolom tanggal perlu diubah menjadi format datetime yang sesuai agar analisis berbasis waktu dapat dilakukan dengan baik. Proses ini memungkinkan kita untuk memanfaatkan fungsi-fungsi manipulasi waktu dalam analisis dan pemodelan.

- Normalisasi Data (Feature Scaling)
  - Menggunakan MinMaxScaler pada Harga Saham. Karena data harga saham memiliki rentang nilai yang besar, dilakukan normalisasi dengan menggunakan MinMaxScaler. Normalisasi ini memastikan bahwa harga saham diubah menjadi skala antara 0 dan 1, yang akan mempercepat konvergensi model dan meningkatkan akurasi prediksi, terutama untuk model berbasis gradient descent seperti LSTM.

- Membagi Data Menjadi Training dan Testing Set
  - Pemisahan Data Berdasarkan Waktu. Data dibagi menjadi dua set, yaitu data untuk pelatihan (training) dan pengujian (testing). Pemisahan ini dilakukan dengan memilih data dari masa lalu untuk melatih model dan data yang lebih baru untuk menguji hasil prediksi. Hal ini penting untuk memastikan bahwa model diuji pada data yang belum pernah dilihat sebelumnya dan dapat menghasilkan prediksi yang realistis.

- Membuat Dataset untuk Model LSTM (Time Series Dataset)
  - Membuat Dataset Time Series. Untuk model LSTM, dibutuhkan dataset dengan format time series. Dataset ini dibuat dengan menggunakan fungsi create_dataset, yang mengonversi data harga saham menjadi urutan waktu (time steps) yang akan digunakan untuk melatih model. Setiap titik data akan memprediksi harga saham berdasarkan sejumlah langkah waktu sebelumnya.
 
## Modeling
Tahapan modeling adalah langkah di mana kita menerapkan algoritma machine learning untuk membangun model yang dapat digunakan untuk memprediksi harga saham Nvidia berdasarkan data historis. Pada proyek ini, kita menggunakan tiga model utama: KNN Regression, Linear Regression, dan LSTM (Long Short-Term Memory) untuk memprediksi harga saham. Setiap model memiliki karakteristik, kelebihan, dan kekurangan tersendiri, yang akan dibahas lebih lanjut.

1. KNN (K-Nearest Neighbors)
KNN adalah salah satu algoritma machine learning yang digunakan untuk regresi dan klasifikasi. Dalam KNN, prediksi harga saham dilakukan berdasarkan data historis dengan mencari K tetangga terdekat yang memiliki nilai fitur yang serupa. Prediksi dilakukan dengan cara menghitung rata-rata nilai target dari tetangga terdekat tersebut.
- Kelebihan KNN, Mudah Dimengerti karena KNN adalah algoritma yang intuitif dan mudah dipahami, serta tidak memerlukan pelatihan model yang rumit. KNN tidak mengasumsikan bentuk distribusi data, yang membuatnya fleksibel untuk berbagai jenis data.
- Kekurangan KNN, Kecepatan Prediksi Lambat karena KNN memerlukan perhitungan jarak antar data untuk setiap prediksi, sehingga bisa sangat lambat pada dataset besar. Rentan terhadap Data Noise, KNN bisa sangat terpengaruh oleh data noise, karena prediksi bergantung pada tetangga terdekat yang mungkin saja mengandung nilai yang tidak relevan.
- Parameter:
  - n_neighbors, Menentukan jumlah tetangga terdekat yang digunakan dalam prediksi.
  - weights, Fungsi bobot yang digunakan untuk menghitung kontribusi tetangga terdekat, bisa berupa 'uniform' atau 'distance'.

2. Linear Regression
Linear Regression adalah model regresi yang berusaha mencari hubungan linier antara variabel independen dan variabel dependen. Model ini digunakan untuk memprediksi harga saham dengan mengasumsikan hubungan linier antara harga saham dan fitur yang relevan.
- Kelebihan Linear Regression, Sederhana dan Cepat karena model ini mudah diimplementasikan dan cepat dalam proses pelatihan serta prediksi. Hasil model linear mudah untuk dipahami, yaitu hubungan antara input dan output dalam bentuk koefisien.
- Kekurangan Linear Regression, Linear regression mengasumsikan bahwa hubungan antara variabel independen dan dependen adalah linier. Jika hubungan tersebut non-linear, model ini tidak akan bekerja dengan baik. Outliers dapat sangat mempengaruhi kinerja model karena linear regression sangat dipengaruhi oleh data yang ekstrem.
- Parameter:
  - fit_intercept, Menentukan apakah model akan mempelajari intercept atau tidak.
  - normalize, Menentukan apakah fitur akan dinormalisasi sebelum model dibangun.

3. LSTM (Long Short-Term Memory)
LSTM adalah tipe dari Recurrent Neural Networks (RNN) yang dirancang untuk menangani data urutan, seperti time series. LSTM memanfaatkan memori jangka panjang untuk mengingat informasi yang relevan dalam urutan data dan dapat menangani ketergantungan jangka panjang.
- Kelebihan LSTM Cocok untuk Time Series. LSTM sangat baik dalam menangani data time series, di mana urutan data dan ketergantungan waktu sangat penting. LSTM mampu mengingat informasi dalam waktu yang lama, sehingga cocok untuk memprediksi data yang bergantung pada pola-pola sebelumnya.
- Kekurangan LSTM:
Kompleksitas dan Waktu Pelatihan, LSTM membutuhkan waktu pelatihan yang lebih lama dibandingkan dengan algoritma lainnya seperti KNN dan Linear Regression. Jika tidak dilakukan regularisasi dengan benar, LSTM dapat dengan mudah mengalami overfitting terutama pada dataset kecil.
- Parameter:
  - units, Jumlah unit dalam layer LSTM yang mempengaruhi kemampuan memori model.
  - batch_size: Jumlah sampel yang digunakan dalam setiap iterasi pelatihan.
  - epochs: Jumlah iterasi di mana model dilatih pada seluruh dataset.
  - dropout: Persentase neuron yang akan "dimatikan" selama pelatihan untuk mencegah overfitting.

## Evaluation
Evaluasi model merupakan tahap penting untuk mengukur seberapa baik model dalam melakukan prediksi berdasarkan data historis harga saham Nvidia. Dalam proyek ini metrik evaluasi yang digunakan adalah:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

1. Penjelasan Metrik Evaluasi
- MAE (Mean Absolute Error)
MAE mengukur rata-rata selisih absolut antara nilai aktual dan nilai prediksi. Semakin kecil nilai MAE, semakin akurat model.
- RMSE (Root Mean Squared Error)
RMSE memberikan penalti lebih besar terhadap error yang besar karena nilai error dikuadratkan sebelum dirata-ratakan, lalu diakarkan. Ini berguna jika kita ingin meminimalkan prediksi yang terlalu jauh dari nilai sebenarnya.
- R² Score
R² mengukur seberapa besar variasi pada target (harga saham) yang dapat dijelaskan oleh model. Nilai R² berkisar antara 0 hingga 1 (semakin mendekati 1 semakin baik).

2. Hasil Evaluasi Model
- KNN : 
  - MAE: 0.6873
  - RMSE: 1.215
  - R2 Score: 0.9993

- Linear Regression:
  - MAE: 0.3937
  - RMSE: 0.6989
  - R2 Score: 0.9997

- LSTM:
  - MAE: 2.4101
  - RMSE: 4.1268
  - R2 Score: 0.9893

3. Interpretasi Hasil Evaluasi
Dari hasil evaluasi, dapat disimpulkan bahwa model Linear Regression memiliki performa terbaik dibandingkan dengan model KNN dan LSTM. Hal ini terlihat dari:
- Nilai MAE dan RMSE paling rendah terdapat pada Linear Regression, menunjukkan bahwa prediksi yang dihasilkan model ini paling mendekati nilai aktual.
- R² Score tertinggi juga dicapai oleh Linear Regression (0.9997), yang berarti model ini mampu menjelaskan hampir seluruh variasi dalam data target dengan sangat baik.
- Model KNN menunjukkan performa yang cukup baik dengan nilai R² sebesar 0.9993, meskipun sedikit di bawah Linear Regression.
- Sementara itu, LSTM memiliki performa paling rendah dibanding dua model lainnya, ditunjukkan oleh nilai MAE dan RMSE yang lebih tinggi serta R² Score yang lebih rendah. Hal ini bisa disebabkan oleh kebutuhan tuning yang lebih kompleks, overfitting, atau karakteristik dataset yang tidak sepenuhnya cocok untuk arsitektur LSTM dalam eksperimen saat ini.

![image](https://github.com/user-attachments/assets/1e85150a-cf85-4ccd-a596-53d32db1cd5b)

![image](https://github.com/user-attachments/assets/c2799293-1e58-433a-a9d1-c76cfa73613f)

![image](https://github.com/user-attachments/assets/86c2714f-46a4-4948-b08f-242ce2b519c8)


4. Kesimpulan Evaluasi
Evaluasi dilakukan dengan mempertimbangkan konteks permasalahan prediksi harga saham yang bersifat time series. Meskipun LSTM dirancang khusus untuk data berurutan, pada proyek ini justru model Linear Regression memberikan hasil evaluasi terbaik berdasarkan metrik MAE, RMSE, dan R² Score.

Hal ini menunjukkan bahwa untuk dataset dan preprocessing yang digunakan dalam proyek ini, pendekatan sederhana seperti Linear Regression justru lebih optimal. Oleh karena itu, Linear Regression dapat dipilih sebagai solusi terbaik untuk implementasi nyata dalam kasus ini, sementara model LSTM mungkin memerlukan penyesuaian lanjutan seperti arsitektur, jumlah epoch, atau hyperparameter tuning yang lebih dalam.
