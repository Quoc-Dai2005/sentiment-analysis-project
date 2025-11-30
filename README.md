# 📘 Phân Tích Cảm Xúc Bình Luận Tiếng Việt (Vietnamese Sentiment Analysis)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PhoBERT](https://img.shields.io/badge/Model-PhoBERT%20Large-yellow?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge)

<<<<<<< Updated upstream
Dự án xây dựng hệ thống AI tự động phân loại cảm xúc từ văn bản tiếng Việt sử dụng mô hình ngôn ngữ lớn **PhoBERT**.

## 📂 1. Cấu Trúc & Mô Tả Module Mã Nguồn

Dự án bao gồm các thành phần mã nguồn chính sau đây:

### 🛠️ Module 1: Huấn luyện Mô hình (`trainer-ai.ipynb`)
Đây là module nòng cốt (Core Engine), chịu trách nhiệm "dạy" cho AI học từ dữ liệu.
* **Chức năng:**
    1.  **Data Loading:** Đọc dữ liệu từ file `data/comments.csv`.
    2.  **Preprocessing:** Sử dụng thư viện `Underthesea` để tách từ tiếng Việt (Word Segmentation).
    3.  **Tokenization:** Mã hóa văn bản thành dạng số sử dụng `AutoTokenizer` của PhoBERT.
    4.  **Training:** Huấn luyện mô hình `vinai/phobert-large` với kỹ thuật *Mixed Precision (FP16)* và *Gradient Accumulation*.
    5.  **Evaluation:** Đánh giá độ chính xác, vẽ biểu đồ Loss và Confusion Matrix.
    6.  **Export:** Lưu model đã huấn luyện ra thư mục `my_phobert_sentiment`.

### 🌐 Module 2: Ứng dụng Web (`app.py`)
Đây là module giao diện người dùng (User Interface).
* **Chức năng:**
    1.  **Model Loading:** Tải model từ thư mục `my_phobert_sentiment`.
    2.  **Inference:** Nhận văn bản -> Tách từ -> Dự đoán cảm xúc (Tích cực/Tiêu cực/Trung lập).
    3.  **UI:** Hiển thị giao diện web chat bằng `Gradio`.

### 📦 Module 3: Quản lý Thư viện (`requirements.txt`)
* **Chức năng:** Liệt kê danh sách các thư viện Python cần thiết (Torch, Transformers, Gradio, Scikit-learn...).

---

## ⚙️ 2. Hướng Dẫn Cài Đặt (Installation)

**Bước 1: Clone dự án về máy**
```bash
git clone https://github.com/Quoc-Dai2005/sentiment-analysis-project.git
cd sentiment-analysis-project
```
**Bước 2: Tạo môi trường ảo (Khuyến nghị)**
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```
**Bước 3: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```
## 🚀 3. Hướng Dẫn Sử Dụng
* **Cách 1: Huấn luyện lại mô hình (Training)**
    1. Để file dữ liệu tại `data/comments.csv`.
    2. Mở file `trainer-ai.ipynb` trong VS Code.
    3. Chọn Run All để chạy toàn bộ quá trình train.
    4. Model mới sẽ được lưu vào `my_phobert_sentiment`.
* **Cách 2: Chạy ứng dụng Demo (Web App)**
    1. Mở Terminal tại thư mục dự án.
    2. Chạy lệnh:
    ```bash
    python app.py
    ```
    3. Truy cập link `http://127.0.0.1:7860` trên trình duyệt.
## ⚠️ Lưu ý kỹ thuật
* **Yêu cầu GPU:** Nên train trên GPU (NVIDIA) hoặc Kaggle/Colab để đạt tốc độ tốt nhất.
* **Git LFS:** File model PhoBERT rất nặng (>1GB), không được upload trực tiếp lên GitHub mà nên được lưu cục bộ hoặc dùng Git LFS.


=======
---

## 📋 Thông Tin Dự Án

- Môn học: MAT3508 – Nhập môn Trí tuệ Nhân tạo
- Học kỳ: Học kỳ 1 – Năm học 2025-2026
- Trường: VNU-HUS (ĐHQG Hà Nội – Trường Đại học Khoa học Tự nhiên)
- Tiêu đề dự án: Phân tích cảm xúc đánh giá tiếng Việt sử dụng mô hình PhoBERT
- Ngày nộp: 30/11/2025

- Báo cáo PDF: `bao_cao_AI.pdf`
- Slide 1: `Slide_AI.pdf`
- Slide 2: `Green-Modern-Simple-Cybersecurity-Presentation.pdf`

Thành viên nhóm:

| Họ và tên        | Mã sinh viên | GitHub         | Đóng góp chính                            |
|------------------|-------------|----------------|-------------------------------------------|
| Đồng Quốc Đại    | 23001513    | Quoc-Dai2005   | Mô hình PhoBERT, huấn luyện, tối ưu.     |
| Chu Thành Dũng   | 23001506    | ChuThanhDung   | Dữ liệu, NLP, viết báo cáo.              |
| Nguyễn Mạnh Dũng | 23001507    | mdunglittleboi | Trực quan, slide, demo Gradio, kiểm thử. |

---

## 📂 1. Cấu Trúc & Mô Tả Module Mã Nguồn

.
├── app.py # Web demo Gradio dùng model đã huấn luyện
├── trainer-ai.ipynb # Notebook huấn luyện PhoBERT trên dữ liệu Tiki
├── bao_cao_AI.pdf # Báo cáo chính
├── Green-Modern-Simple-Cybersecurity-Presentation.pdf
├── Slide_AI.pdf
├── requirements.txt
├── data/
│ └── comments.csv # Dữ liệu đánh giá (1–5 sao)
├── my_phobert_sentiment/ # Thư mục model đã fine-tune (tự sinh sau khi train, gửi trong file pdf gửi trên canvas)
│ ├── config.json
│ ├── pytorch_model.bin
│ ├── tokenizer.json
│ └── ...
└── README.md

text

### 🛠️ `trainer-ai.ipynb` – Huấn luyện mô hình

- Đọc dữ liệu từ `data/comments.csv`.
- Tiền xử lý: chuẩn hóa Unicode, làm sạch văn bản.
- Tách từ tiếng Việt bằng `underthesea.word_tokenize`.
- Mã hóa bằng `AutoTokenizer` của `vinai/phobert-large`.
- Chia train/test, tạo `Dataset` HuggingFace, cấu hình `TrainingArguments`.
- Huấn luyện với FP16, Gradient Accumulation, LR = 1e-5, 4 epoch.
- Đánh giá bằng Accuracy, F1-macro, Confusion Matrix, ROC-AUC.
- Lưu model + tokenizer vào `./my_phobert_sentiment`.

### 🌐 `app.py` – Ứng dụng Web Gradio

- Tải model và tokenizer từ `./my_phobert_sentiment`.
- Hàm xử lý:
  - Nhận câu tiếng Việt thô.
  - Tách từ bằng `underthesea` (đưa về dạng “giao_hàng nhanh”).
  - Tokenize bằng PhoBERT, đưa qua model.
  - Trả về nhãn cảm xúc và xác suất 3 lớp.
- Tạo giao diện Gradio với:
  - Ô nhập text.
  - Hiển thị nhãn + xác suất từng lớp.
  - Một số ví dụ mẫu (ví dụ câu khen/chê/trung lập).
- Chạy server Gradio trên `http://127.0.0.1:7860`.

### 📦 `requirements.txt`

Chứa các thư viện chính:

- torch, torchvision, torchaudio
- transformers, datasets
- underthesea
- scikit-learn
- matplotlib, seaborn
- gradio

---

## ⚙️ 2. Hướng Dẫn Cài Đặt

1. Clone dự án:

git clone https://github.com/Quoc-Dai2005/sentiment-analysis-project.git
cd sentiment-analysis-project

text

2. Tạo và kích hoạt môi trường ảo (khuyến nghị):

python -m venv venv

Windows:
.\venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

text

3. Cài đặt thư viện:

pip install -r requirements.txt

text

---

## 🚀 3. Hướng Dẫn Sử Dụng

### Cách 1 – Huấn luyện lại PhoBERT

1. Chuẩn bị dữ liệu tại `data/comments.csv` (gồm text + rating 1–5).  
2. Mở `trainer-ai.ipynb` trong VS Code / Jupyter.  
3. Chạy toàn bộ notebook.  
4. Sau khi chạy xong, thư mục `my_phobert_sentiment/` sẽ được tạo với model đã fine-tune.

### Cách 2 – Chạy demo web Gradio

1. Đảm bảo đã có thư mục `my_phobert_sentiment/` (từ bước train hoặc copy từ nơi khác).  
2. Mở Terminal tại thư mục dự án, chạy:

python app.py

text

3. Mở trình duyệt và truy cập:

http://127.0.0.1:7860

text

4. Nhập câu bình luận tiếng Việt để hệ thống dự đoán cảm xúc.

---

## ⚠️ 4. Lưu Ý Kỹ Thuật

- Nên huấn luyện trên GPU (NVIDIA) hoặc Google Colab/Kaggle để tránh thời gian training quá lâu.  
- PhoBERT-Large và model fine-tune khá nặng, dễ bị lỗi Out-of-Memory nếu GPU yếu; hãy:
  - Giữ `max_length` hợp lý (128).
  - Dùng batch size nhỏ + Gradient Accumulation.
  - Bật FP16 như trong notebook.  
- File model đã train có thể >1GB – không upload trực tiếp lên GitHub, đã gửi link google drive trong file pdf gửi trên canvas

---

## 📚 5. Tài Liệu Tham Khảo

- PhoBERT – Vietnamese BERT-based Language Models.  
- HuggingFace Transformers & Datasets docs.  
- Gradio – Build ML web apps in Python.  
>>>>>>> Stashed changes
