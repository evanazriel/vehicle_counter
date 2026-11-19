# Real-Time Traffic Vehicle Detection & PostgreSQL Database Integration

Sistem pemantauan dan deteksi kendaraan lalu lintas secara *real-time* berbasis **Deep Learning (Ultralytics YOLOv8)** dan **Computer Vision (OpenCV)** yang terhubung langsung ke **Live Feed CCTV Publik DKI Jakarta (Balitower)** serta terintegrasi dengan **Basis Data Relasional PostgreSQL**.

Sistem ini secara otomatis mendeteksi objek kendaraan (**Mobil**, **Motor**, **Bus**, dan **Truk**), menampilkan bounding box dan persentase akurasi (*confidence score*), serta mencatat data deteksi ke database PostgreSQL secara berkala dengan logika pencegahan data duplikat.

## Fitur Utama
* **Integrasi Live CCTV Publik:** Mengakses feed transmisi HLS (`.m3u8`) CCTV jalan raya Kota Jakarta (Satpol PP / Balitower) tanpa jeda (*zero buffering lag*).
* **Klasifikasi Kendaraan Otomatis:** Mendeteksi 4 kategori utama transportasi jalan raya:
  *  **Mobil (*Car*)**
  *  **Sepeda Motor (*Motorcycle*)**
  *  **Bus (*Bus*)**
  *  **Truk (*Truck*)**
* **Bounding Box & Confidence Overlay:** Menampilkan kotak batas deteksi beserta nilai persentase keyakinan model secara dinamis pada video.
* **Logging Transaksional PostgreSQL:** Menyimpan jenis kendaraan dan tingkat akurasi ke PostgreSQL.
* **Logika Anti-Duplikasi (Pencegahan Data Kembar):** Jika kendaraan dengan nilai akurasi (*conf score*) yang sama terdeteksi kembali, sistem **hanya memperbarui waktu (*timestamp*)** data tersebut alih-alih membuat baris baru yang memenuhi database.

# Requirements
* opencv-python 
* scikit-learn 
* pandas 
* scikit-image 
* matplotlib 
* ultralytics
* psycopg2-binary