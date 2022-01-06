# Laporan Proyek Machine Learning - Taufiq Adjie Sanjaya
---
## Domain Proyek
---
Seorang investor properti sukses dari Indonesia berencana ingin menambah jumlah aset propertinya di Australia lebih tepatnya Sydney. Tentu, ia harus mengetahui properti seperti apa yang cocok dengan warga disana.

Untuk mendatangkan profit di kemudian hari serta penjualan yang cepat kita perlu sebuah data tentang penjualan properti yang ada di Sydney tersebut. Kita akan memprediksi harga properti berdasarkan tipe tertentu. Lalu, sang investor dapat mencari dan menentukan properti mana yang memiliki harga dibawah pasaran.

Hal yang diperlukan untuk memecahkan masalah diatas adalah sebuah model algoritma machine learning. Berikutnya kita akan membahas alur untuk mengembangkan model algoritma machine learning untuk memprediksi harga pasar dari sebuah sehingga dapat menjawab permasalahan diatas untuk membantu sang investor menemukan properti dengan harga di bawah pasaran.

## Business Understanding
---
Pada bagian ini, sang investor perlu menguraikan masalah pada latar belakang serta jawaban dari masalah tersebut.
### Problem Statements
Berdasarkan pernyataan diatas, sang investor akan membuat sebuah sistem prediksi harga properti untuk menjawab permasalahan tersebut.
* Berdasarkan pada beberapa fitur yang ada, fitur apa yang paling berpengaruh pada harga properti?
* Berapa harga pasar properti dengan karakteristik tertentu?
* Tipe properti seperti apa yang lebih banyak terjual?

### Goals
Untuk menjawab masalah latar belakang pada statement diatas, kita akan membuat predictive modelling dengan tujuan sebagai berikut :
* Mengetahui fitur mana yang memiliki korelasi tinggi terhadap fitur target (harga)
* Membuat model machine learning yang dapat memprediksi harga properti seakurat mungkin berdasarkan fitur yang ada
* Melakukan analisa data satu persatu 


### Solution Statements
Berikut cara untuk meraih goals diatas :
  * Melalui pendekatan machine learning dengan teknik multivariate analysis, kita akan mengetahui hubungan antar dua atau lebih variabel sehingga kita dapat mengetahui fitur mana yang memiliki korelasi tinggi terhadap fitur target berdasarkan pola persebaran datanya
  * Kita akan menerapkan tiga jenis algoritma model machine learning untuk mengetahui model mana yang menghasilkan nilai prediksi mendekati nilai sebenarnya.
  * Melalui pendekatan machine learning dengan teknik univariate analysis, kita dapat menngetahui tipe properti apa yang paling banyak terjual.

# Data Understanding
---
Data yang akan kita gunakan berikut bersumber dari Kaggle. Berikut data yang akan kita gunakan : [Sydney House Price](https://github.com/cumapemula/dataset/tree/main/SydneyHousePrices.csv). Data tersebut berisi jumlah total 199504 baris dan 9 kolom. Data tersebut berisi informasi tentang penjualan properti dengan karakteristik tertentu di kota Sydney. Karakteristik tersebut akan dijelaskan melalui variabel dibawah ini.

**Variabel pada dataset Sydney House Price adalah sebagai berikut :**
* Date       : yaitu tanggal properti tersebut terjual
* Id         : yaitu urutan data
* Suburb     : yaitu letak pinggir kota properti tersebut
* postalCode : yaitu kodepos properti
* sellPrice  : yaitu harga jual properti
* bed        : yaitu jumlah ruang kamar tidur pada properti
* bath       : yaitu jumlah ruang kamar mandi pada properti
* car        : yaitu jumlah mobil yang dapat dimasukkan ke dalam garasi
* propType   : yaitu tipe properti tersebut


Untuk lebih memahami data, kita akan menggunakan beberapa tahapan diantaranya :
  * Data loading
  * Exploratory Data Analysis :
    * Deskripsi Variabel
    * Univariate Analysis
    * Multivariate Analaysis

Berikut uraian dari tahapan tersebut :
* Data Loading 

![](https://raw.githubusercontent.com/cumapemula/dataset/main/1.png)

Gambar diatas adalah output dari kode yang kita jalankan untuk mengimport dataset. Dapat kita ketahui bahwa data pada tabel tersebut memiliki total 199504 baris serta 9 kolom. 

* Exploratory Data Analysis

  * Deskripsi Variabel  

![](https://github.com/cumapemula/dataset/blob/main/22.png?raw=true)

Gambar diatas menunjukkan info bahwa pada data terdapat enam kolom tipe numerik dan tiga kolom tipe kategorikal. Coba kita perhatikan, jumlah data pada kolom bed dan car tidak sama seperti kolom lain. Ini menunjukkan bahwa kolom tersebut terdapat nilai yang kosong (missing value), ini akan kita bahas pada tahap selanjutnya.

![](https://github.com/cumapemula/dataset/blob/main/23.png?raw=true)

Gambar diatas menginformasikan kita nilai ukuran tendensi sentral pada data.
<br>
<br>
Keterangan : 
* count : yaitu jumlah data keseluruhan
* mean : yaitu nilai rata-rata pada kolom tertentu
* std : yaitu standar deviasi pada kolom tertentu
* min : yaitu nilai terkecil pada kolom tertentu
* 25% : yaitu kuartil bawah data pada kolom tertentu
* 50% : yaitu median atau nilai tengah data pada kolom tertentu
* 75% : yaitu kuartil bawah data pada kolom tertentu
* max : yaitu nilai terbesar pada kolom tertentu


  * Univariate Analysis

Teknik ini digunakan untuk menganalisa data satu persatu. Sebagai contoh berikut :

![](https://github.com/cumapemula/dataset/blob/main/9.png?raw=true)

Pada grafik diatas kita ketahui bahwa properti dengan tipe house paling banyak terjual. Dengan ini, point ketiga dari problem statements di awal sudah terjawab.


  * Multivariate Analysis

Berbeda dengan teknik sebelumnya, pada teknik ini menunjukkan hubungan antara dua atau lebih variabel pada data. Sebagai contoh berikut :

![](https://github.com/cumapemula/dataset/blob/main/10.png?raw=true)

Pada pola sebaran data grafik diatas, fitur bedrooms,bathrooms, dan garage membentuk sebuah pola pada fitur target price yang berarti ketiga kolom tersebut memiliki korelasi terhadap fitur target. Untuk mengetahui fitur mana yang memiliki korelasi tinggi terhadap target, kita evaluasi skor korelasi tersebut sebagai berikut :

![](https://github.com/cumapemula/dataset/blob/main/11.png?raw=true)

Koefisien korelasi berkisar antara -1 (korelasi negatif) dan +1 (korelasi positif). Jika skor mendekati 0, maka semakin lemah korelasinya. Bila terdapat korelasi yang lemah, fitur tersebut dapat kita drop.

# Data Preparation
---
Pada bagian ini kita akan melakukan empat tahap persiapan data, yaitu :
* Drop Columns
* Missing Value & Outliers
* Encoding 
* Reduksi Dimensi
* Train Test Split
* Standarisasi


Berikut Penjelasannya :
* Drop Columns

Kita akan mengeliminasi kolom Date, Id, dan postalCode karena dianggap kurang relevan dengan kasus ini.
 
![](https://raw.githubusercontent.com/cumapemula/dataset/main/2.png)
 
Kita sudah berhasil mengeliminasi kolom tersebut serta merubah nama kolom agar terlihat rapih dan mudah dimengerti. Terlihat bahwa jumlah kolom sudah berkurang menjadi 6 kolom.

* Missing Value & Outliers

Berikut perintah kode serta output untuk mengetahui apakah terdapat missing value pada data.

![](https://github.com/cumapemula/dataset/blob/main/5.png?raw=true)

Terlihat bahwa cukup banyak missing value yang terdapat pada kolom bedrooms dan garage. Kita akan mengganti nilai tersebut dengan nilai mean. Jalankan perintah kode berikut serta cek kembali apakah masih terdapat missing value atau tidak.

![](https://github.com/cumapemula/dataset/blob/main/6.png?raw=true)

Setelah data kita pastikan sudah tidak ada nilai yang kosong, selanjutnya kita pastikan apakah ada suatu pengamatan yang berada di luar lingkungan pengamatan lainnya atau disebut outliers. Sebagai contoh kita gunakan kolom garage.

![](https://github.com/cumapemula/dataset/blob/main/7.png?raw=true)

Titik sampel yang berada diluar garis batas merupakan outliers. Kita akan menggunakan teknik IQR Method untuk menangani outliers tersebut lalu cek kembali pada kolom garage apakah masih terdapat outliers atau tidak.

![](https://github.com/cumapemula/dataset/blob/main/8.png?raw=true)

* Encoding

Teknik encoding yang kita gunakan yaitu _One Hot Encoding_. Teknik ini berfungsi untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili variabel kategori.

![](https://github.com/cumapemula/dataset/blob/main/12.png?raw=true)


* Reduksi Dimensi

Teknik reduksi (pengurangan) dimensi adalah prosedur yang mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data. Teknik pengurangan dimensi yang paling populer adalah _Principal Component Analysis_ atau disingkat menjadi PCA. Ia adalah teknik untuk mereduksi dimensi, mengekstraksi fitur, dan mentransformasi data dari “n-dimensional space” ke dalam sistem berkoordinat baru dengan dimensi m, di mana m lebih kecil dari n.

![](https://github.com/cumapemula/dataset/blob/main/13.png?raw=true)

![](https://github.com/cumapemula/dataset/blob/main/14.png?raw=true)

* Train Test Split

Pada tahap ini kita akan membagi dataset menjadi data latih dan data uji. Tahap ini diperlukan untuk mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Berikut perintah kode serta output untuk pembagian dataset.

![](https://github.com/cumapemula/dataset/blob/main/15.png?raw=true)

* Standarisasi

Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

![](https://github.com/cumapemula/dataset/blob/main/16.png?raw=true)

![](https://github.com/cumapemula/dataset/blob/main/17.png?raw=true)

# Modeling
---
Pada tahap ini, kita akan mengembangkan tiga model dengan algoritma machine learning yang berbeda untuk mendapatkan hasil prediksi yang paling akurat. Algoritma tersebut antara lain K-Nearest Neighbor, Random Forest, dan Boosting Algorithm.
* K-Nearest Neighbor

Kelebihan dari algoritma ini yaitu relatif sederhana dibandingkan dengan algoritma lain. Namun memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar sering disebut sebagai _curse of dimensionality_ (kutukan dimensi).
* Random Forest

Kelebihan dari algoritma ini selain cukup sederhana juga memiliki tingkat keberhasilan lebih tinggi karena algoritma ini termasuk ke dalam model kategori ensemble  (group) learning yaitu model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Kekurangan dari algoritma ini yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data.
* Boosting Algorithm

Kelebihan dari boosting algorithm yaitu pada algoritma ini sangat powerful dalam meningkatkan akurasi prediksi. 

# Evaluation
---
Pada kasus regresi ini metrik yang akan kita gunakan adalah Mean Squared Error(MSE). Cara kerja metrik MSE adalah dengan menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi, yang dirumuskan sebagai berikut :

![](https://github.com/cumapemula/dataset/blob/main/18.png?raw=true)
* Keterangan :

  * N = Jumlah Dataset

  * yi = Nilai Sebenarnya

  * y_pred = Nilai prediksi


Berikut adalah hasil evaluasi metrik pada ketiga algoritma model kita :

![](https://github.com/cumapemula/dataset/blob/main/19.png?raw=true)

![](https://github.com/cumapemula/dataset/blob/main/20.png?raw=true)

Pada data dan grafik diatas diketahui bahwa pada model K-Nearest Neighbor menghasilkan nilai error yang paling kecil dibanding algoritma yang lain. Maka, model K-Nearest Neighbor yang akan kita pilih untuk prediksi harga properti. Lalu kita akan menguji model menggunakan beberapa harga dari data test dan didapat hasil sebagai berikut :

![](https://github.com/cumapemula/dataset/blob/main/21.png?raw=true)

Dapat dilihat bahwa pada hasil prediksi algoritma K-Nearest Neighbor mendekati nilai sebenarnya. Maka, model ini yang akan kita berikan kepada sang investor untuk memprediksi harga properti di Sydney.
