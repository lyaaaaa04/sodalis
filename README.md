# Sodalis - Sistem _P2P Lending_ Mahasiswa Menggunakan _Deep Learning_ untuk Prediksi Kelayakan Peminjaman

<img width="6000" height="3375" alt="SODALIS" src="https://github.com/user-attachments/assets/2bc9d14c-7183-469d-8b2a-6c33ca4f9d50" />

## Project Overview

SODALIS adalah platform _Peer-to-Peer_ (P2P) _Lending_ yang dirancang khusus untuk mahasiswa. Platform ini hadir sebagai ekosistem keuangan digital "dari mahasiswa, oleh mahasiswa, dan untuk mahasiswa" yang mempertemukan mahasiswa yang membutuhkan pendanaan dengan mahasiswa lain yang ingin mengalokasikan dana secara lebih produktif. Mahasiswa sering menghadapi kebutuhan dana mendesak seperti pembayaran UKT, biaya penelitian, kebutuhan akademik, hingga pendanaan kompetisi. Namun, solusi pinjaman yang tersedia saat ini memiliki keterbatasan, seperti proses peminjaman yang rumit, bunga yang berubah-ubah, hingga risiko kebocoran data pribadi pada layanan pinjaman.

SODALIS mengatasi permasalahan tersebut dengan mengintegrasikan teknologi _Internet Computer Protocol_ (ICP) untuk meningkatkan keamanan dan transparansi transaksi. Selain itu, platform ini menerapkan sistem _fixed rate_ agar pengguna memperoleh kepastian biaya pinjaman sejak awal. Untuk meningkatkan kualitas keputusan pendanaan, SODALIS juga mengimplementasikan sistem _AI Credit Scoring_ berbasis _Deep Learning_ menggunakan **TensorFlow Functional API**. Model ini akan memanfaatkan _custom layer, custom loss function_, dan _custom callback_ untuk memprediksi kelayakan peminjaman mahasiswa lebih akurat.

## Business Understanding

### Problem Statements

Beberapa permasalahan utama yang menjadi dasar pengembangan sistem SODALIS:
- Mahasiswa sering membutuhkan dana cepat untuk kebutuhan akademik maupun pribadi, namun layanan pinjaman yang terjadi umumnya memiliki proses yang rumit dan kurang ramah terhadap mahasiswa.
- Banyak layanan pinjaman digital memiliki bunga yang berubah-ubah sehingga meningkatkan ketidakpastian dan risiko finansial bagi peminjam.
- Belum tersedia platform internal mahasiswa yang aman, transparan, dan mampu melakukan penilaian kelayakan peminjaman secara objektif.
- Penilaian kelayakan pinjaman mahasiswa membutuhkan pendekatan yang lebih cerdas karena mahasiswa umumnya belum memiliki riwayat kredit formal.
  
Berdasarkan kondisi tersebut, diperlukan sistem berbasis _Artificial Intelligence_ yang mampu mendukung pengambilan keputusan kelayakan pinjaman secara lebih akurat. Untuk menjawab permasalahan tersebut, akan fokus pada pertanyaan berikut:
- Bagaimana akurasi model _Deep Learning Credit Scoring_ berbasis _TensorFlow Functional API_ dengan _custom Layer, Loss Function_, dan _Callback_ dalam memprediksi kelayakan peminjaman mahasiswa?
  
### Goals

Untuk menjawab pertanyaan tersebut, maka akan dibuat sebuah sistem prediksi kelayakan dengan tujuan atau goals sebagai berikut:

- Membangun sistem _AI Credit Scoring berbasis Deep Learning_ untuk membantu memprediksi kelayakan peminjaman mahasiswa.
- Meningkatkan akurasi penilaian risiko melalui pendekatan _custom Deep Learning architecture_ menggunakan TensorFlow Functional API.
    
### Solution Statements

Berdasarkan tujuan atau goals yang sudah dijelaskan sebelumnya, dirancang sebuah alur solusi agar sistem mampu menyediakan proses peminjaman mahasiswa didukung oleh sistem AI Credit Scoring untuk membantu proses penilaian kelayakan pinjaman.

- **Data Understanding**. Tahap awal untuk memahami data yang dimiliki. Pada tahap ini dilakukan beberapa proses untuk memahami karakteristik data mahasiswa, seperti struktur data, distribusi fitur, hubungan antar fitur, dan pola yang dapat memengaruhi kelayakan pinjaman.
- **Data Loading**. Tahapan awal dilakukan dengan memuat dataset mahasiswa yang mencakup informasi akademik, ekonomi, riwayat pembayaran, dan atribut lain yang digunakan sebagai dasar sistem Credit Scoring.
- **Exploratory Data Analysis**. Pada tahap ini dilakukan analisis eksploratif untuk memahami karakteristik data, seperti distribusi fitur numerik, analisis fitur kategorikal, ketidakseimbangan kelas, serta hubungan antar fitur menggunakan visualisasi.
- **Data Preprocessing**. Ini merupakan tahap persiapan data sebelum digunakan pada proses selanjutnya. Pada tahap ini dilakukan transformasi data untuk meningkatkan performa model, seperti feature engineering, pembuatan fitur *Debt Service Ratio (DSR)*, transformasi logaritma, dan penghapusan fitur yang tidak digunakan.
- **Data Preparation**. Pada tahap ini data dipersiapkan dengan beberapa teknik seperti data splitting, identifikasi tipe fitur, penanganan outlier, encoding fitur kategorikal, feature scaling, serta pembuatan data pipeline.
- **Modeling**. Pada proses pengembangan model digunakan pendekatan _Deep Learning Credit Scoring_ menggunakan TensorFlow Functional API dengan implementasi komponen kustom sebagai berikut:
  - **Model Development dengan Wide & Deep Learning Architecture**. Model digunakan untuk mempelajari pola linear dan non-linear dari data mahasiswa dalam memprediksi kelayakan pinjaman melalui kombinasi _Wide Branch_ dan _Deep Branch_.
  - **Custom Component Development**. Pada tahap ini diterapkan _Custom Layer, Custom Loss Function_, dan _Custom Callback _untuk meningkatkan kemampuan model dalam melakukan ekstraksi fitur, menangani ketidakseimbangan data, dan mengoptimalkan proses pelatihan.
  - **Custom Training Loop**. Proses pelatihan model diimplementasikan menggunakan _tf.GradientTape_ untuk memberikan fleksibilitas dalam pengelolaan proses training dan evaluasi.
- **Inference Model**. Setelah model selesai dilatih, dilakukan proses inference untuk memprediksi tingkat kelayakan peminjaman mahasiswa berdasarkan data baru menggunakan model yang telah diekspor dalam format TensorFlow.
- **REST API Development**. Model yang telah dilatih diintegrasikan ke dalam REST API menggunakan _FastAPI_ agar sistem dapat melayani proses prediksi Credit Scoring secara real-time dan mendukung integrasi dengan aplikasi SODALIS.
- **Evaluation**. Tahap untuk mengukur kinerja model dan menilai sejauh mana model berhasil mencapai tujuannya. Pada tahap ini digunakan metrik berupa _Accuracy, Precision, Recall, F1-Score, AUC_, dan _MAE_ untuk mengevaluasi performa model Credit Scoring. Selain itu digunakan _TensorBoard_ untuk memantau dan memvisualisasikan proses pelatihan model.
