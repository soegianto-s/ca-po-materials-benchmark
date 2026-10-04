# Kontrol graf dengan jumlah parameter setara

Ekstensi post hoc tanggal 27 September 2026, setelah hasil utama terlihat; tidak diklaim sebagai preregistrasi. Arsitektur kontrol ditetapkan sebelum hasil kontrol diperiksa. Tidak ada pencarian konfigurasi berdasar skor test.

Graf kontrol dan hibrid masing-masing mempunyai 15.940 parameter. Encoder graf identik. Hibrid memasukkan 90 deskriptor komposisi ke MLP 90–32–32. Kontrol memasukkan 90 statistik representasi simpul graf: 32 mean, 32 momen kedua, dan 26 kanal maksimum pertama. Pemilihan 26 kanal mengikuti urutan tetap arsitektur, bukan performa. Keduanya menggabungkan cabang32 dengan mean graf32 ke head64–32–4. Tidak ada parameter padding tambahan hanya untuk menyamakan jumlah. Keduanya mempertahankan readout densitas33 parameter yang tidak masuk loss, sama seperti eksperimen asal; cabang tambahan kontrol/hibrid tetap aktif. Seluruh input kontrol berasal dari graf; tidak memakai tabel komposisi eksternal.

Perbedaan pooling tersebut harus dinyatakan: ini kontrol jumlah parameter dan masukan cabang, bukan penyamaan seluruh inductive bias atau kapasitas efektif. Kesetaraan jumlah parameter tidak menjamin kesetaraan optimisasi.

Gunakan tepat 15 fold lama ×3 seed, 150 epoch maksimum, patience25, pembobotan loss dan validasi identik, CPU deterministik. Nonnegative gap dan penghapusan density loss sama dengan graph_physics dan graph_hybrid. Pembandingan utama: hybrid versus kontrol; pembandingan konteks: kontrol versus graph_physics. Laporkan semua45 hasil, bukan subset yang menang. Pemilihan epoch dan kalibrasi hanya validasi. Hasil ini tidak mengubah prosedur validation-selected graph lama.

Uji implementasi: jumlah parameter sama, prediksi tidak bergantung pada input komposisi, invariansi supercell, gradien tersedia pada cabang aktif. Script utama dan input dibekukan dengan hash dalam experiment.json. Run lama dipertahankan.
