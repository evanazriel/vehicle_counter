# Real-Time Traffic Vehicle Detection & PostgreSQL Database Integration

Sistem pemantauan dan deteksi kendaraan lalu lintas secara *real-time* berbasis **Deep Learning (Ultralytics YOLOv8)** dan **Computer Vision (OpenCV)** yang terhubung langsung ke **Live Feed CCTV Publik DKI Jakarta (Balitower)** serta terintegrasi dengan **Basis Data Relasional PostgreSQL**. 

Sistem ini secara otomatis mendeteksi objek kendaraan (**Mobil**, **Motor**, **Bus**, dan **Truk**), menampilkan *bounding box* dan persentase akurasi (*confidence score*), mencatat data deteksi ke database PostgreSQL secara berkala dengan logika pencegahan data duplikat, serta menyajikan **Dashboard Grafik Analitik** untuk evaluasi performa model dan statistik lalu lintas.

---

## Fitur Utama
* **Integrasi Live CCTV Publik:** Mengakses feed transmisi HLS (`.m3u8`) CCTV jalan raya Kota Jakarta (Satpol PP / Balitower) tanpa jeda (*zero buffering lag*).
* **Klasifikasi Kendaraan Otomatis:** Mendeteksi 4 kategori utama transportasi jalan raya:
  * 🚗 **Mobil (*Car*)**
  * 🏍️ **Sepeda Motor (*Motorcycle*)**
  * 🚌 **Bus (*Bus*)**
  * 🚚 **Truk (*Truck*)**
* **Bounding Box & Confidence Overlay:** Menampilkan kotak batas deteksi beserta nilai persentase keyakinan model secara dinamis pada video.
* **Logging Transaksional PostgreSQL:** Menyimpan jenis kendaraan dan tingkat akurasi ke PostgreSQL.
* **Logika Anti-Duplikasi (Pencegahan Data Kembar):** Jika kendaraan dengan nilai akurasi (*conf score*) yang sama terdeteksi kembali, sistem **hanya memperbarui waktu (*timestamp*)** data tersebut alih-alih membuat baris baru yang memenuhi database.
* **Dashboard Grafik Analitik (Data Visualization):** Menghasilkan 4 grafik komprehensif langsung di bawah sel Jupyter Notebook untuk evaluasi hasil penelitian (Bab 4 Skripsi/Tugas Akhir).

---

## Skema Basis Data PostgreSQL

Tabel: **`traffic_detection_logs`**

Hanya menyimpan identitas transaksi, jenis kendaraan yang terdeteksi, nilai akurasi, dan waktu pencatatan:

| Nama Kolom | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| **`id`** | `SERIAL PRIMARY KEY` | Nomor ID unik transaksi log (Auto Increment) |
| **`vehicle_type`** | `VARCHAR(50) NOT NULL` | Jenis kendaraan (`Mobil`, `Motor`, `Bus`, `Truk`) |
| **`confidence_score`** | `REAL NOT NULL` | Nilai persentase keyakinan model (misal: `94.2`) |
| **`timestamp`** | `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` | Waktu saat kendaraan terdeteksi / diperbarui |

---

## Penjelasan Rinci Kode Program

Berikut penjelasan langkah demi langkah dari script utama yang digunakan:

### 1. Koneksi Basis Data & Auto-Create Table
```python
conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT", "5432")
)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS traffic_detection_logs (
        id SERIAL PRIMARY KEY,
        vehicle_type VARCHAR(50) NOT NULL,
        confidence_score REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")
conn.commit()
```
* **`CREATE TABLE IF NOT EXISTS`**: Mencegah error `UndefinedTable`. Program secara cerdas akan mengecek apakah tabel `traffic_detection_logs` sudah ada; jika belum, tabel akan dibuat otomatis tanpa perlu membuka pgAdmin.

---

### 2. Pemuatan Model YOLOv8 & Live CCTV Feed
```python
model = YOLO("yolov8n.pt")

video_path = 'https://dki-jkt.balitower.co.id:7028/502506_JKP_SATPOL-PP_JL.-GERBANG-PEMUDA-C04_CCTV-01/tracks-v1/index.fmp4.m3u8' 
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```
* **`YOLO("yolov8n.pt")`**: Menggunakan model *Deep Learning* YOLOv8 varian *nano* yang sudah terlatih pada dataset COCO.
* **`video_path`**: Menggunakan link transmisi HLS (`.m3u8`) langsung dari CCTV Publik Dishub/Satpol PP DKI Jakarta (Jl. Gerbang Pemuda, Senayan).
* **`cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`**: Mengatur antrean memori kamera menjadi 1 frame. Ini adalah kunci agar tayangan video **selalu *real-time*** dan tidak mengalami delay/keterlambatan dari kondisi jalan aslinya.

---

### 3. Deteksi Objek & Visualisasi Akurasi
```python
results = model(frame, classes=[2, 3, 5, 7], verbose=False)
annotated_frame = results[0].plot()
```
* **`classes=[2, 3, 5, 7]`**: Membatasi deteksi hanya pada target kendaraan (2: *car*, 3: *motorcycle*, 5: *bus*, 7: *truck*) dan mengabaikan objek lain (seperti orang, tiang, tas) agar beban komputasi ringan dan cepat.
* **`results[0].plot()`**: Fungsi bawaan Ultralytics yang secara otomatis menggambar *bounding box* warna-warni pada kendaraan lengkap dengan nama kelas dan persentase akurasinya (misal: `car 94%`).

---

### 4. Logika Penyimpanan & Anti-Duplikasi Data
```python
if cursor and frame_count % 30 == 0:
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = round(float(box.conf[0]) * 100, 1)
        vehicle_name = class_map.get(cls_id, "Lainnya")

        # 1. CEK: Apakah data dengan jenis dan conf score yang sama sudah tercatat?
        check_query = """
            SELECT id FROM traffic_detection_logs 
            WHERE vehicle_type = %s AND ROUND(confidence_score::numeric, 1) = %s
            ORDER BY id DESC LIMIT 1;
        """
        cursor.execute(check_query, (vehicle_name, conf))
        existing_row = cursor.fetchone()

        if existing_row:
            # JIKA SUDAH ADA: Update waktu terakhir terlihat
            cursor.execute("""
                UPDATE traffic_detection_logs 
                SET timestamp = CURRENT_TIMESTAMP 
                WHERE id = %s;
            """, (existing_row[0],))
        else:
            # JIKA BELUM ADA: Baru lakukan Insert data baru
            cursor.execute("""
                INSERT INTO traffic_detection_logs (vehicle_type, confidence_score)
                VALUES (%s, %s);
            """, (vehicle_name, conf))

    conn.commit()
```
* **`frame_count % 30 == 0`**: Menyimpan data setiap 30 frame sekali ($\approx$ 1 detik sekali), bukan setiap milidetik, agar database tidak menerima banjir *request*.
* **Logika Anti-Duplikat**:
  * Program memeriksa apakah kendaraan dengan jenis dan skor akurasi yang sama sudah ada di database.
  * Jika **sudah ada**, program menjalankan `UPDATE` untuk memperbarui kolom `timestamp` ke detik saat ini.
  * Jika **baru pertama kali terlihat**, program menjalankan `INSERT` untuk mencatat baris baru.

---

### 5. Pembersihan Resource
```python
if cursor:
    cursor.close()
    conn.close()

cap.release()
cv2.destroyAllWindows()
```
* Menutup kursor dan koneksi database dengan aman.
* Melepaskan akses stream kamera dan menutup seluruh jendela tampilan video OpenCV saat pengguna menekan tombol **`q`**.

---

##  Dashboard Visualisasi & Analisis Data (Bab 4 Evaluasi)

Data yang tersimpan di PostgreSQL diekstrak menggunakan **Pandas** dan divisualisasikan dengan **Matplotlib** & **Seaborn**:

### Informasi yang Dihasilkan dari Grafik:
1. **Donut Chart (Komposisi %):** Menggambarkan proporsi dominasi jenis kendaraan di jalan (misal: 65% Mobil, 25% Motor, 7% Truk, 3% Bus).
2. **Bar Chart (Total Unit):** Mengukur volume riil beban jalan raya yang melintas.
3. **Bar Chart (Akurasi %):** Mengevaluasi keandalan model *Deep Learning* (misal: Mobil rata-rata 92%, Truk 95%, Motor 78%).
4. **Grafik Garis Tren Waktu:** Menganalisis fluktuasi kepadatan lalu lintas berdasarkan sumbu X (Jam:Menit:Detik) dan sumbu Y (Jumlah Unit Kendaraan).

---

## Konfigurasi Keamanan Lingkungan (`.env`)

Untuk mencegah kebocoran *password* database ke repositori publik GitHub:

1. Buat file **`.env`** di folder proyek Anda:
   ```env
   DB_HOST=localhost
   DB_NAME=nama_database_anda
   DB_USER=postgres
   DB_PASSWORD=password_rahasia_anda
   DB_PORT=5432
   ```

2. Pastikan file **`.env`** sudah terdaftar di dalam **`.gitignore`**:
   ```gitignore
   .env
   __pycache__/
   *.pyc
   ```

---

## Panduan Instalasi & Cara Menjalankan

### 1. Persiapan Lingkungan & Dependensi
Pastikan Python 3.10+ dan PostgreSQL sudah terpasang di komputer Anda.

Install pustaka pendukung via terminal:
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Program
1. Buka file notebook di VS Code atau Jupyter Notebook.
2. Jalankan sel kode deteksi CCTV. Jendela *live stream* CCTV Jl. Gerbang Pemuda Jakarta akan terbuka menampilkan kotak deteksi objek secara *real-time*.
3. Tekan tombol **`q`** pada jendela video untuk menghentikan program.
4. Jalankan sel visualisasi untuk langsung melihat 4 grafik analitik di bawah sel notebook.

---

## Requirements
* `opencv-python`
* `ultralytics`
* `psycopg2-binary`
* `pandas`
* `matplotlib`
* `seaborn`
* `python-dotenv`
* `scikit-learn`
* `pillow`

---
