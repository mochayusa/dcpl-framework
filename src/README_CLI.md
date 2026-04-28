# CLI Usage Guide — Baseline & DCPL Experiments

Dokumen ini menjelaskan cara menjalankan **baseline models** dan **DCPL framework**
menggunakan Command Line Interface (CLI) melalui file:

src/project_main.py

Eksperimen dirancang untuk **AIware performance modelling** dengan:
- split data **80/20**
- **random seed berbeda** di setiap run
- **multi-run evaluation** (misalnya 30 kali)
- **cross-fitting (OOF)** untuk DCPL

---

## 1. Prasyarat

Pastikan Anda berada di root project dan environment sudah aktif:

```bash
conda activate dcpl
```

Struktur direktori minimal:

project_root/
├── src/
│   ├── project_main.py
│   ├── dcpl/
│   │   ├── framework.py
│   │   ├── models.py
│   │   └── ...
│   └── experiments/
├── data/
│   └── llm_pilot_data/
│       └── raw_data/
│           └── per_model/
│               ├── modelA.csv
│               ├── modelB.csv
│               └── ...
└── results/

## 2. Format Umum Perintah CLI
```bash
python src/project_main.py <command> [arguments] [options]
```

Command utama yang tersedia:

* baseline → menjalankan baseline models

* dcpl → menjalankan DCPL framework

## 3. Menjalankan Baseline Experiments
### 3.1 Menjalankan SELURUH baseline (30 kali)
```bash
python src/project_main.py baseline all 30
```
Baseline yang dijalankan:

* lr — Linear Regression
* ridge — Ridge Regression
* rf — Random Forest
* nn — Neural Network (MLP)
* llm_pilot — XGBoost (LLM-Pilot style)


Setiap baseline:
* dijalankan 30 kali
* menggunakan split 80/20
* menggunakan seed berbeda di setiap run

### 3.2 Menjalankan SATU baseline saja

Contoh menjalankan Neural Network 30 kali:
3.2 Menjalankan SATU baseline saja

Contoh menjalankan Neural Network 30 kali:
```bash
python src/project_main.py baseline nn 30
```

Baseline valid lainnya:
```bash
lr | ridge | rf_light | nn | llm_pilot
```
### 3.3 Opsi Tambahan untuk Baseline
```bash
python src/project_main.py baseline all 30 \
  --base-seed 42 \
  --seed-stride 1000 \
  --test-size 0.20 \
  --target Target_throughput_tokens_per_sec
```

## 4. Menjalankan DCPL Experiments
### 4.1 Menjalankan DCPL 30 kali
```bash
python src/project_main.py dcpl 30
```

DCPL akan:

* menjalankan 30 eksperimen
* setiap eksperimen memakai seed berbeda
* melakukan cross-fitting (OOF) pada training set
* menghasilkan distribusi performa (mean/std)

### 4.2 Memilih Gate Model

Gate yang tersedia:
```bash
lr | ridge | nn | rf
```

Contoh menggunakan gate Neural Network:
```bash
python src/project_main.py dcpl 30 --gate-kind nn
```
### 4.3 Mengatur Jumlah Inner Cross-Validation (Cross-Fitting)
```bash
python src/project_main.py dcpl 30 --inner-splits 5
```

Artinya:

* training set dibagi menjadi 5 fold
* digunakan untuk menghasilkan OOF predictions saat melatih gate

### 4.4 Opsi Tambahan untuk DCPL
```bash
python src/project_main.py dcpl 30 \
  --gate-kind ridge \
  --inner-splits 5 \
  --base-seed 42 \
  --seed-stride 1000 \
  --test-size 0.20 \
  --target Target_latency_ms
```

## 5. Struktur Output & Hasil Eksperimen
### 5.1 Output Baseline
results/runs/
└── 30x_baselines/
    └── baseline_split80__MULTI_BASELINES__30x_base42/
        ├── baseline_split80_permodel_ALL_learners_ALL_runs_stacked.csv
        ├── baseline_split80_permodel_mean_across_runs.csv
        └── baseline_split80_permodel_std_across_runs.csv

### 5.2 Output DCPL
results/runs/
└── 30x_dcpl/
    └── dcpl_split80__multirun_30x_base42/
        ├── dcpl_split80_permodel_ALL_runs_stacked.csv
        ├── dcpl_split80_permodel_mean_across_runs.csv
        └── dcpl_split80_permodel_std_across_runs.csv

### 5.3 Output per Run Individual

Setiap run memiliki folder sendiri yang berisi:

* predictions/*.csv — hasil prediksi test set

* summary.csv — metrik performa per model

* split_mask.csv — informasi data train/test

* manifest.json — metadata eksperimen

## 6. Prinsip Reproducibility

Setiap eksperimen:

* menggunakan random_state = base_seed + i * seed_stride

* seed dipropagasikan ke:

* train/test split

* Random Forest experts

* Neural Network

* XGBoost

* gate model

* DCPL menggunakan cross-fitting (out-of-fold prediction) untuk menghindari data leakage

## 7. Contoh Workflow Eksperimen Lengkap
### 1. Jalankan semua baseline
```bash
python src/project_main.py baseline all 30
```
### 2. Jalankan DCPL dengan gate Ridge
```bash
python src/project_main.py dcpl 30 --gate-kind ridge
```

### 3. Bandingkan mean/std performa dari CSV hasil

## 8. Catatan Penting

Gunakan jumlah run yang sama untuk baseline dan DCPL agar perbandingan fair.

Pastikan inner_splits <= n_train untuk dataset kecil.

Gunakan split_mask.csv untuk analisis error per-sample atau debugging.

## 9. Bantuan CLI

Untuk melihat semua opsi:
```bash
python src/project_main.py --help
```

Atau bantuan spesifik:
```bash
python src/project_main.py baseline --help

python src/project_main.py dcpl --help
```