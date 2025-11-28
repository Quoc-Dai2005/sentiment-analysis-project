# 📘 Phân Tích Cảm Xúc Bình Luận (Sentiment Analysis Project)

Dự án xây dựng hệ thống AI tự động phân loại cảm xúc từ văn bản tiếng Việt sử dụng mô hình ngôn ngữ lớn **PhoBERT**.

## 📂 1. Cấu Trúc & Mô Tả Module Mã Nguồn
Dự án bao gồm các thành phần mã nguồn chính sau đây:

### 🛠️ Module 1: Huấn luyện Mô hình (`trainer-ai.ipynb`)
Đây là module nòng cốt (Core Engine), chịu trách nhiệm "dạy" cho AI học từ dữ liệu.
* **Chức năng:**
    1.  **Data Loading:** Đọc dữ liệu từ file `data/comments.csv`, xử lý giá trị thiếu (null).
    2.  **Preprocessing:** Sử dụng thư viện `Underthesea` để tách từ tiếng Việt (Word Segmentation), chuẩn hóa nhãn (Label Encoding).
    3.  **Tokenization:** Mã hóa văn bản thành dạng số sử dụng `AutoTokenizer` của PhoBERT.
    4.  **Training Loop:** Cấu hình và huấn luyện mô hình `vinai/phobert-large` thông qua `Trainer API` của HuggingFace. Sử dụng kỹ thuật *Mixed Precision (FP16)* và *Gradient Accumulation* để tối ưu bộ nhớ.
    5.  **Evaluation:** Đánh giá độ chính xác, vẽ biểu đồ Loss, Confusion Matrix và ROC Curve.
    6.  **Export:** Lưu model đã huấn luyện ra thư mục `my_phobert_sentiment`.

### 🌐 Module 2: Ứng dụng Web (`app.py`)
Đây là module giao diện người dùng (User Interface), giúp tương tác với mô hình đã huấn luyện.
* **Chức năng:**
    1.  **Model Loading:** Tải model và tokenizer từ thư mục `my_phobert_sentiment`.
    2.  **Inference Logic:** Nhận văn bản đầu vào từ người dùng -> Tách từ -> Đưa qua Model -> Tính toán xác suất (Softmax).
    3.  **UI Rendering:** Khởi tạo giao diện web bằng `Gradio`, hiển thị kết quả dự đoán (Tích cực/Tiêu cực/Trung lập) và độ tin cậy.

### 📦 Module 3: Quản lý Thư viện (`requirements.txt`)
* **Chức năng:** Liệt kê danh sách các thư viện Python và phiên bản cụ thể cần thiết để chạy dự án (Torch, Transformers, Gradio, Scikit-learn...).

---

## ⚙️ 2. Hướng Dẫn Cài Đặt (Installation)

**Bước 1: Clone dự án**
```bash
git clone [https://github.com/Quoc-Dai2005/sentiment-analysis-project.git](https://github.com/Quoc-Dai2005/sentiment-analysis-project.git)
cd sentiment-analysis-project
