# 📘 Phân Tích Cảm Xúc Bình Luận Tiếng Việt (Vietnamese Sentiment Analysis)

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
* **Chức năng:** Liệt kê danh sách các thư viện Python và phiên bản cụ thể cần thiết để chạy dự án (Torch, Transformers, Gradio, Scikit-learn, Underthesea...).

---

## ⚙️ 2. Hướng Dẫn Cài Đặt (Installation)

**Bước 1: Clone dự án về máy**
```bash
git clone [https://github.com/Quoc-Dai2005/sentiment-analysis-project.git](https://github.com/Quoc-Dai2005/sentiment-analysis-project.git)
cd sentiment-analysis-project
Bước 2: Tạo môi trường ảo (Khuyến dùng)

Bash

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
Bước 3: Cài đặt thư viện

Bash

pip install -r requirements.txt
🚀 3. Hướng Dẫn Sử Dụng (Usage)
Cách 1: Huấn luyện lại mô hình (Training)
Nếu bạn muốn train lại từ đầu với dữ liệu mới:

Đảm bảo file dữ liệu nằm tại data/comments.csv.

Mở file trainer-ai.ipynb bằng Jupyter Notebook hoặc VS Code.

Chọn Run All để chạy toàn bộ quá trình.

Sau khi xong, model mới sẽ được lưu tự động tại thư mục my_phobert_sentiment.

Cách 2: Chạy ứng dụng Demo (Web App)
Để mở giao diện web chat:

Mở Terminal tại thư mục gốc dự án.

Chạy lệnh:

Bash

python app.py
Truy cập đường link hiển thị trên màn hình (thường là http://127.0.0.1:7860) trên trình duyệt web.

⚠️ Lưu ý kỹ thuật
GPU: Quá trình train yêu cầu GPU (NVIDIA T4/P100 trở lên) để đạt tốc độ tốt nhất. Nếu chạy CPU sẽ rất chậm.

Dữ liệu: File CSV đầu vào cần có 2 cột chính: rating (số sao) và content (nội dung bình luận).

Model: Model PhoBERT-Large sau khi train có dung lượng >1GB, nên không được upload trực tiếp lên GitHub mà phải lưu cục bộ hoặc dùng Git LFS.

Tác giả: Quốc Đại (VNU-HUS)
