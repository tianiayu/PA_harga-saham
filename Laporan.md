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
Dataset yang digunakan dalam proyek ini merupakan data historis harga saham perusahaan teknologi Nvidia (NVDA), yang diperoleh dari situs Investing.com. Dataset ini berisi informasi harian mengenai harga saham Nvidia selama periode tertentu, dan digunakan sebagai dasar untuk membangun model prediksi harga saham. Data dapat diunduh melalui tautan berikut (https://www.investing.com/equities/nvidia-corp-historical-data). Jumlah Dataset terdiri dari 1.256 baris dan 7 kolom. Kondisi data, tidak ditemukan nilai kosong pada seluruh kolom. Dan tidak ditemukan data duplikat.

### Variabel-variabel pada NVIDIA Corporation (NVDA) dataset adalah sebagai berikut:
- Date: Tanggal pencatatan harga saham.
- Price: Harga penutupan (closing price) saham Nvidia pada tanggal tersebut.
- Open: Harga pembukaan saham pada awal perdagangan hari tersebut.
- High: Harga tertinggi yang dicapai saham selama hari tersebut.
- Low: Harga terendah yang dicapai saham selama hari tersebut.
- Vol. (Volume): Jumlah saham yang diperdagangkan dalam satu hari.
- Change %: Persentase perubahan harga saham dibandingkan dengan hari sebelumnya.

Semua fitur tersebut bersifat numerik, kecuali tanggal. Dan sangat penting untuk dianalisis secara visual maupun statistik sebelum digunakan dalam pemodelan.

Untuk memahami karakteristik data secara lebih mendalam, dilakukan beberapa tahapan Exploratory Data Analysis, antara lain:
- Visualisasi Tren Harga Saham (Price) dari Waktu ke Waktu.
![image](https://github.com/user-attachments/assets/2b2ea3db-3f1a-4771-98e1-4e28385bdf7d)
Visualisasi tren Price dari waktu ke waktu memperlihatkan fluktuasi yang signifikan dan tren pertumbuhan jangka panjang. Pola ini memperlihatkan adanya periode tertentu dengan kenaikan atau penurunan harga saham secara konsisten, yang relevan untuk pendekatan prediktif pada model.
- Visualisasi Heatmap Korelasi Antar Fitur
![image](https://github.com/user-attachments/assets/f87b117c-8877-4996-8f3e-f9175f3633c7)
Visualisasi heatmap menunjukkan korelasi sangat kuat antara fitur Price, Open, High, dan Low, yang semuanya memiliki nilai korelasi mendekati 1. Hal ini menunjukkan bahwa pergerakan harga dalam dataset ini konsisten antar fitur tersebut. Sementara itu, Volume memiliki korelasi negatif lemah dengan fitur harga, yang berarti peningkatan volume perdagangan tidak selalu diikuti oleh kenaikan harga. Fitur Change % menunjukkan hubungan yang sangat lemah terhadap semua fitur lainnya, mengindikasikan bahwa persentase perubahan harga harian tidak banyak dipengaruhi oleh harga maupun volume.

## Data Preparation
Pada tahap ini, dilakukan serangkaian proses untuk mempersiapkan data sebelum digunakan dalam pemodelan. Teknik data preparation berikut diterapkan secara berurutan:

**Langkah-langkah Data Preparation**:
- Konversi Kolom Tanggal
Kolom Date dikonversi ke dalam format datetime. Format datetime diperlukan agar data dapat diurutkan berdasarkan waktu dan dianalisis.Pembersihan Data (Data Cleaning)
- Pembersihan Format Kolom Volume
Kolom Vol awalnya berisi simbol seperti “K” dan “M” (contoh: “5.2M”, “300K”). Data ini dibersihkan dan dikonversi ke nilai numerik. Simbol non-numerik tidak dapat diproses oleh model machine learning. Konversi ke tipe numerik memungkinkan analisis statistik dan proses pelatihan model.
- Pemilihan Fitur dan Target
Setelah data dibersihkan, proses dilanjutkan dengan pemisahan fitur (independen variabel) dan target (dependent variabel). Fitur yang digunakan untuk pelatihan model adalah kolom: Open, High, Low, dan Vol., sedangkan target yang ingin diprediksi adalah Price.
- Normalisasi Fitur
Karena model seperti KNN dan beberapa algoritma lain sensitif terhadap skala fitur, maka dilakukan normalisasi fitur menggunakan Min-Max Scaler untuk mengubah skala data ke rentang 0 hingga 1. Ini memastikan setiap fitur memiliki kontribusi yang setara dalam proses pelatihan model.

Tahapan-tahapan tersebut dilakukan untuk memastikan bahwa data terstruktur dan sesuai untuk digunakan dalam pemodelan prediktif, khususnya pada konteks time-series forecasting untuk harga saham NVIDIA. Langkah-langkah ini sangat penting agar model dapat dilatih dengan data yang valid dan akurat.
 
## Modeling
Tahapan modeling merupakan proses di mana algoritma machine learning diterapkan untuk membangun model prediktif berdasarkan data historis harga saham Nvidia. Dalam proyek ini, kami menggunakan tiga pendekatan model utama: KNN Regression, Linear Regression, dan LSTM (Long Short-Term Memory). Masing-masing model memiliki kelebihan dan keterbatasan, serta digunakan dengan parameter yang disesuaikan untuk mencapai performa terbaik.

1. KNN (K-Nearest Neighbors)
KNN digunakan untuk memprediksi harga saham dengan cara mengidentifikasi sejumlah K tetangga terdekat berdasarkan kemiripan fitur, lalu menghitung rata-rata nilai target dari tetangga-tetangga tersebut.
- Kelebihan KNN yaitu; sederhana dan mudah dipahami, tidak memerlukan pelatihan model eksplisit, dan tidak mengasumsikan bentuk distribusi data tertentu.
- Kekurangan KNN yaitu; proses prediksi lambat pada dataset besar karena menghitung jarak setiap kali prediksi dilakukan dan rentan terhadap noise dalam data.
- Parameter:
  - n_neighbors=5: Menggunakan 5 tetangga terdekat dalam prediksi.
  - weights='distance': Memberikan bobot lebih besar pada tetangga yang lebih dekat.
  - algorithm='auto': Memilih algoritma pencarian tetangga terbaik secara otomatis.

2. Linear Regression
Linear Regression digunakan untuk memodelkan hubungan linier antara variabel input dan harga saham. Model ini diasumsikan bekerja baik jika terdapat korelasi linier dalam data.
- Kelebihan Linear Regression yaitu; proses pelatihan cepat dan efisien dan interpretasi model mudah (berbasis koefisien).
- Kekurangan Linear Regression yaitu; tidak cocok untuk data dengan pola non-linier dan sangat terpengaruh oleh outlier.
- Parameter:
  - fit_intercept=True: Model mempelajari nilai intersep.
  - normalize=False: Tidak dilakukan normalisasi fitur karena preprocessing dilakukan sebelumnya.
  - Implementasi menggunakan sklearn.linear_model.LinearRegression.

3. LSTM (Long Short-Term Memory)
LSTM merupakan jenis jaringan saraf tiruan (neural network) yang dirancang untuk data sequential, seperti time series. Model ini sangat sesuai untuk memprediksi harga saham berdasarkan urutan waktu karena kemampuannya untuk mengingat pola historis jangka panjang.
- Kelebihan LSTM yaitu; mampu menangkap ketergantungan waktu dalam data danocok untuk data time series yang kompleks.
- Kekurangan LSTM yaitu; struktur model kompleks dan pelatihan memakan waktu lama dan risiko overfitting, terutama pada dataset kecil jika tidak dilakukan regularisasi.
- Parameter:
  - units=50: Jumlah neuron dalam layer LSTM.
  - batch_size=32: Jumlah data per iterasi pelatihan.
  - epochs=100: Jumlah iterasi pelatihan pada seluruh dataset.
  - dropout=0.2: Menonaktifkan 20% neuron secara acak selama pelatihan untuk mencegah overfitting.
  - optimizer='adam': Optimizer yang digunakan untuk mempercepat konvergensi.
  - loss='mean_squared_error': Fungsi loss yang digunakan untuk menghitung kesalahan prediksi.
  - Model diimplementasikan menggunakan TensorFlow/Keras

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
