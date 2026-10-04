# Uji sensitivitas cakupan deskriptor

Protokol post hoc ditetapkan 27 September 2026 sebelum menghitung hasil sensitivitas. Hasil utama dan kode pelatihannya tetap dibekukan. Tujuan: menilai apakah perbandingan hibrid, RF, graf fisika, dan kontrol jumlah parameter bergantung pada tiga rekaman dengan unsur di luar tabel komposisi historis.

Eksklusi tetap: mp-1214043 (Pd), mp-1214415 (Pd), mp-677016 (Eu), ditentukan hanya dari cakupan fitur, bukan kesalahan prediksi. Tidak mengimputasi atribut atau menambahkan fitur sesudah melihat skor. Hasil berlaku pada domain unsur yang didukung; bukan perbaikan prediksi Pd/Eu.

1. Sensitivitas populasi evaluasi: hapus ketiga ID dari prediksi OOF lama semua model, tanpa melatih ulang atau mengubah kalibrator/threshold. Bandingkan 511 vs 508 sampel pada sembilan pasangan seed.
2. Sensitivitas pelatihan: keluarkan ID dari fit, validation, test pada seluruh 15 split lama tanpa membagi ulang sampel lain. Hitung ulang statistik fit, bobot kelas, kalibrasi, dan threshold sesuai protokol lama. Ulangi tiga seed inisialisasi untuk graf fisika, hibrid, kontrol graf (135 fit graf) dan 45 kelompok RF (RF biasa dan berbobot). Semua konfigurasi tetap, CPU deterministik, 150 epoch, patience 25. Semua hasil dilaporkan. Lima ablation lama tidak semuanya dilatih ulang; kesimpulan retraining dibatasi pada empat keluarga pembanding ini.
3. Perbandingan berpasangan: hasil retraining 508 melawan model lama yang dinilai pada 508 ID yang sama. Laporkan AP mentah, MAE gap/energi, Brier terkalibrasi, pelanggaran domain, sembilan selisih, dan variasi antarpartisi/inisialisasi. Jangan menafsirkan sembilan run sebagai eksperimen independen atau mengklaim signifikansi.
4. Jika peringkat atau arah efek berubah, laporkan apa adanya; tidak memilih konfigurasi ulang dari hasil test. Uji transfer eksternal yang telah selesai tetap berasal dari model utama 511; tidak disebut sebagai hasil retraining 508.

Split baru, ID, hash input, statistik fit, checkpoint, prediksi, kalibrator, dan sejarah validasi disimpan. Tabel literatur memberi perbandingan ruang lingkup/metode; angka dari korpus berbeda bukan perbandingan akurasi langsung. Ini bukan klaim prioritas "pertama" atau penelusuran sistematis lengkap.
