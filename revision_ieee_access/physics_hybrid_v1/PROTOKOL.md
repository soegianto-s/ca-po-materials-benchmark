# Eksperimen bertahap fisika–AI dan komposisi–struktur

Protokol disusun sebelum pelatihan utama dan pembacaan skor uji varian baru.
Pilot tiga epoch hanya memeriksa implementasi dan waktu eksekusi; bukan hasil
yang dipakai dalam perbandingan utama.

## Pertanyaan dan varian

Apakah batas keluaran fisik dan informasi komposisi memperbaiki generalisasi
lintas komposisi? Lima varian graf memisahkan perubahan berikut:

| Varian | Band gap | Loss densitas | Cabang komposisi |
|---|---|---|---|
| graph_baseline | bebas | aktif | tidak |
| graph_gap | nonnegatif | aktif | tidak |
| graph_nodensity | bebas | nonaktif | tidak |
| graph_physics | nonnegatif | nonaktif | tidak |
| graph_hybrid | nonnegatif | nonaktif | aktif |

Kontras utama: graph_gap − graph_baseline; graph_nodensity − graph_baseline;
graph_physics − graph_nodensity; graph_physics − graph_gap; dan graph_hybrid
− graph_physics. Dua varian RF (biasa/class-weighted) menjadi pembanding.
Semua hasil akan dilaporkan, tanpa menghapus seed atau fold yang buruk.

## Data dan pembagian yang dibekukan

Gunakan tepat 15 manifest fit/validasi/uji eksperimen periodic_benchmark_v1:
seed partisi 42, 2024, 7, masing-masing lima fold. Semua komposisi tereduksi
terpisah antarset. Masing-masing model dilatih dengan seed inisialisasi
42, 2024, dan 7 pada setiap fold: 225 pelatihan graf dan 45 kelompok RF.
Setiap kelompok RF mencakup dua classifier dan tiga regresor.

Graf lama tetap menjadi referensi historis. Baseline graf dilatih ulang
karena eksperimen baru memakai kriteria seleksi epoch yang seragam dan
berbagai seed inisialisasi; checkpoint lama tidak dianggap hasil protokol baru.
Konfigurasi tetap; tidak ada pencarian hiperparameter berdasarkan outer-test.

## Implementasi fisika dan fusi

Band gap terikat dihitung sebagai ReLU dari keluaran dalam satuan eV,
setelah inverse standardization. Ini mengizinkan nol tepat untuk keluaran
metallic. Gradien ReLU nol pada wilayah negatif merupakan keterbatasan
parameterisasi dan bukan bukti akurasi. Pembatasan adalah bagian forward
pass selama pelatihan, bukan clipping yang ditambahkan setelah melihat skor.

Densitas operasional seluruh varian dihitung secara analitis dari struktur,
termasuk pembanding RF. Jangan menyebut nilai R² densitas analitis sebagai
peningkatan AI. Prediksi head densitas hanya dilaporkan sebagai auxiliary
jika loss-nya aktif. Head empat keluaran dipertahankan agar penghapusan loss
tidak sekaligus mengubah ukuran model; head densitas yang tidak dilatih
tidak digunakan sebagai keluaran operasional.

Encoder graf mempertahankan radius 5 Å, delapan radial basis, dua lapisan
bergating lebar 32, LayerNorm, serta mean pooling situs. Cabang komposisi
memakai 90 deskriptor, distandardisasi dengan mean/SD fit saja, lalu MLP
90→32→32. Embedding komposisi dan graf digabung sebelum head 64→32→4.
Penambahan cabang menambah parameter; hasilnya bukan isolasi sempurna
pengaruh informasi dari kapasitas. Kontrol kapasitas tetap merupakan batas.

## Pelatihan dan seleksi

AdamW, learning rate 0,001, weight decay 10⁻⁵, batch 16, dropout 0,1,
gradient clipping 5, maksimum 150 epoch, early stopping patience 25.
Loss training = weighted BCE + 0,5 MSE gap + MSE energi + 0,5 MSE densitas
jika aktif. Seluruh MSE memakai standardisasi target dari fit saja.
Bobot positif BCE adalah rasio negatif/positif pada fit.

Kriteria pemilihan epoch dan scheduler **sama untuk semua varian**:
weighted BCE + 0,5 MSE gap + MSE energi pada validasi, tanpa densitas.
Scheduler membagi learning rate dua setelah patience 10. Kriteria ini
mencegah perbandingan nilai loss dengan jumlah target berbeda.

Jika suatu konfigurasi terpilih dilaporkan, pemilihannya dilakukan secara
terpisah di setiap pasangan outer-fold/seed, hanya berdasarkan kriteria
validasi bersama. Tidak memilih satu varian global memakai nilai uji OOF.
Prediksi semua konfigurasi tetap disimpan untuk perbandingan terkontrol.

## Kalibrasi dan keputusan

Probabilitas mentah dan terkalibrasi dinilai terpisah. Kalibrator Platt
monoton p_cal = sigmoid(a logit(p) + b) di-fit hanya pada validasi; a dibatasi
0–20, b −20–20, dengan regularisasi tetap 10⁻⁴. Probabilitas di-clip ke
[10⁻⁶,1−10⁻⁶] hanya untuk operasi logit/log-loss.

Threshold 0,5 dan threshold F2 terpilih di validasi dilaporkan. Grid threshold
0,05–0,95, langkah 0,025, tie-break nilai tertinggi. Penggunaan validasi yang
sama untuk early stopping, kalibrasi, dan threshold dapat meningkatkan
ketidakstabilan pada sampel kecil, tetapi outer-test tidak pernah dipakai
untuk fitting/seleksi tersebut. Seluruh evaluasi akhir memakai outer-test.

## Metrik dan interpretasi

Klasifikasi: AP, ROC-AUC, precision, recall, F1/F2, MCC, balanced accuracy.
Kalibrasi: Brier score, log-loss, ECE sepuluh bin tetap dan reliability plot;
ECE bukan ukuran tunggal yang definitif pada sampel kecil.
Regresi: MAE, RMSE, R² band gap dan energi pembentukan. Pelanggaran fisika:
jumlah/rate prediksi band gap negatif. Audit densitas analitis dipisahkan.

Gabungkan lima fold menjadi 511 prediksi OOF untuk setiap pasangan seed
partisi/inisialisasi: sembilan set OOF. Pisahkan variasi inisialisasi dari
variasi partisi. Jangan memperlakukan 4.599 prediksi sebagai 4.599 material
independen. Selisih model dihitung berpasangan pada seed dan fold yang sama;
SD deskriptif tidak dianggap confidence interval independen.

Generalisasi yang diukur tetap generalisasi komposisi dalam satu snapshot,
bukan validasi eksternal atau keluarga kimia independen. Batas keluaran
nonnegatif, graf periodik, dan fusi fitur tidak merupakan PINN berbasis PDE
atau bukti konsistensi termodinamika convex hull. Penelitian tidak mengklaim
validasi scaffold atau jaminan penerimaan jurnal.

## Reproduksi

Kode utama: `scripts/physics_hybrid_experiment.py`.
Pengujian: `scripts/test_physics_hybrid_experiment.py` (tujuh pengujian lulus).
Manifest menyimpan konfigurasi, hash data/kode/graf, dan split sumber.
Checkpoint menyimpan optimizer, scheduler, RNG, statistik fit, best epoch,
riwayat loss dan waktu pelatihan. Menjalankan ulang nama eksperimen yang sama
melanjutkan checkpoint hanya jika fingerprint cocok.
