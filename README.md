# 📘 Phân Tích Cảm Xúc Bình Luận Tiếng Việt (Vietnamese Sentiment Analysis)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PhoBERT](https://img.shields.io/badge/Model-PhoBERT%20Large-yellow?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge)

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


