LLM Fine-Tuning for Task-Oriented Instruction Generation


1. Open-Source LLM Selection
Model yang digunakan adalah google/flan-t5-small dari Hugging Face.
Alasan pemilihan:

Ukuran model kecil dan ringan, cocok untuk fine-tuning dengan sumber daya terbatas.

Sudah terlatih pada berbagai tugas instruksi, sehingga lebih adaptif pada task-oriented instruction generation.

Mendukung framework Transformers dan mudah diintegrasikan.


2. Dataset Design and Preparation
Data Type & Format
Dataset berupa JSON list dengan setiap entry berisi:

instruction: kalimat natural language yang merepresentasikan user intent.

output: list langkah instruksi terstruktur (step-by-step).


``bash
{
  "instruction": "How do I reset my password in the xx app?",
  "output": [
    "Open the app",
    "Go to the login page",
    "Tap on the 'Forgot Password' link",
    "Enter your registered email address",
    "Tap 'Submit' to request a reset link",
    "Check your email and verify the reset request",
    "Follow the link to set a new password"
  ]
}

Data Collection & Annotation
Kumpulkan data dari dokumentasi resmi, tutorial, FAQ, dan feedback pengguna.

Anotator menulis langkah instruksi sesuai dengan intent.

Review data untuk memastikan akurasi dan konsistensi.


Preprocessing
Input dan output distandardisasi dalam format teks. Output berupa string dengan langkah-langkah digabung menggunakan newline \n.

Tokenisasi menggunakan tokenizer flan-t5-small dengan truncation dan padding.

Konversi dataset ke format TensorDataset untuk PyTorch DataLoader.


Handling Edge Cases
Oversampling pada intent yang jarang muncul untuk mengurangi imbalance.

Filtering konten sensitif dan data berulang.

Menambahkan data dari berbagai domain dan kasus untuk meningkatkan generalisasi.


3. Fine-Tuning Strategy
Pendekatan: Full fine-tuning model flan-t5-small.

Alasan: Model kecil memudahkan full fine-tuning tanpa memerlukan teknik PEFT. Cocok untuk task spesifik dengan dataset terbatas.

Hyperparameter Utama:

Epoch: 100 (monitor loss untuk menghindari overfitting)

Learning rate: 5e-5

Batch size: 4 (disesuaikan dengan GPU memory)

Tantangan:

Resource GPU terbatas → solusi: batch kecil, mixed precision bila perlu.

Overfitting → solusi: early stopping & monitoring loss.

Catastrophic forgetting → bisa ditangani dengan training data campuran jika tersedia.



4. Evaluation and Benchmarking
Metode Evaluasi:

Loss selama training sebagai indikator konvergensi.

BLEU/ROUGE untuk mengukur kesamaan output model dengan ground truth.

Evaluasi manual (human evaluation) untuk validasi kualitas dan relevansi instruksi.

Benchmark:

Bandingkan model fine-tuned dengan model dasar flan-t5-small tanpa fine-tuning.

Bandingkan dengan instruksi manual sebagai gold standard.

Evaluasi Kualitatif:

Uji kasus nyata (misal reset password).

Review kelengkapan dan urutan langkah instruksi.


5. Practical Implementation Sketch
Struktur Kode
src/finetune.py — melakukan:

Load data

Tokenisasi

Fine-tuning dengan loop epoch & monitoring loss

Simpan model terlatih di models/final_model

src/inference.py — melakukan:

Load model hasil fine-tuning

Fungsi generate instruksi langkah demi langkah dari input natural language



Contoh Penggunaan Inference
Input:

```bash
Copy code
How do I reset my password in the e-commerce "xx" mobile app?
```

Output:

Open the app Go to 'Settings' Select a picture