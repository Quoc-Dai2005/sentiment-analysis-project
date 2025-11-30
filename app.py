"""
Ứng dụng web phân tích cảm xúc tiếng Việt dùng PhoBERT và Gradio.

- Tải mô hình đã fine-tune từ thư mục `my_phobert_sentiment`.
- Cung cấp hàm dự đoán cảm xúc cho một câu tiếng Việt.
- Xây dựng giao diện web đơn giản bằng Gradio để demo cho người dùng cuối.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
import gradio as gr

# Đường dẫn tới thư mục chứa model đã fine-tune (được trainer-ai.ipynb lưu ra)
MODEL_DIR = "./my_phobert_sentiment"

# Thiết bị suy luận: ưu tiên GPU nếu có, nếu không thì dùng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải tokenizer và model PhoBERT đã fine-tune
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

# Ánh xạ chỉ số lớp -> nhãn cảm xúc
LABEL_MAPPING = {
    0: "Tiêu cực",
    1: "Trung lập",
    2: "Tích cực"
}


def preprocess_text(text: str) -> str:
    """
    Tiền xử lý câu tiếng Việt đầu vào trước khi đưa vào PhoBERT.

    Thao tác:
    - Tách từ tiếng Việt bằng underthesea theo dạng 'giao_hàng nhanh'.
    - Có thể mở rộng thêm các bước làm sạch nếu cần.

    Args:
        text (str): Câu/bình luận tiếng Việt thô do người dùng nhập.

    Returns:
        str: Câu đã được tách từ, sẵn sàng để tokenize bởi PhoBERT.
    """
    # Tách từ: chuyển "Giao hàng nhanh" -> "Giao_hàng nhanh"
    text_seg = word_tokenize(text, format="text")
    return text_seg


def predict_sentiment(text: str):
    """
    Dự đoán cảm xúc của một câu tiếng Việt sử dụng mô hình PhoBERT đã fine-tune.

    Quy trình:
    - Tiền xử lý + tách từ tiếng Việt.
    - Tokenize bằng tokenizer của PhoBERT.
    - Chạy forward pass qua model (không cần gradient).
    - Lấy chỉ số lớp có xác suất cao nhất và tính xác suất softmax.

    Args:
        text (str): Câu/bình luận tiếng Việt thô.

    Returns:
        tuple[str, dict]:
            - Nhãn cảm xúc dự đoán (Tiêu cực / Trung lập / Tích cực).
            - Dictionary chứa xác suất của từng lớp, dùng để hiển thị trên UI.
    """
    if not text or text.strip() == "":
        return "Không hợp lệ", {
            "Tiêu cực": 0.0,
            "Trung lập": 0.0,
            "Tích cực": 0.0,
        }

    # 1. Tiền xử lý & tách từ
    text_seg = preprocess_text(text)

    # 2. Mã hóa câu sang tensor cho PhoBERT
    inputs = tokenizer(
        text_seg,
        return_tensors="pt",
        truncation=True,
        max_length=128,        # Giữ chiều dài cố định để tránh tràn RAM
        padding=True
    ).to(device)

    # 3. Suy luận (không cần tính gradient)
    with torch.no_grad():
        logits = model(**inputs).logits

    # 4. Tính xác suất softmax
    probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

    # 5. Lấy lớp có xác suất cao nhất
    pred_idx = int(torch.argmax(logits, dim=1).item())
    label = LABEL_MAPPING.get(pred_idx, "Không xác định")

    # Chuẩn hóa thành dict cho Gradio hiển thị dạng bar chart
    prob_dict = {
        "Tiêu cực": float(probs[0]),
        "Trung lập": float(probs[1]),
        "Tích cực": float(probs[2]),
    }

    return label, prob_dict


def gradio_interface(text: str):
    """
    Hàm wrapper dành riêng cho Gradio.

    Nhận input là câu tiếng Việt, trả về:
    - Nhãn cảm xúc (chuỗi).
    - Dictionary xác suất 3 lớp (để Gradio vẽ bar chart).

    Args:
        text (str): Câu/bình luận tiếng Việt.

    Returns:
        tuple[str, dict]: (label, prob_dict) như hàm predict_sentiment.
    """
    label, prob_dict = predict_sentiment(text)
    # Gộp nhãn dự đoán và xác suất để hiển thị đẹp hơn trên UI nếu muốn
    return f"Cảm xúc dự đoán: {label}", prob_dict


# --------------- CẤU HÌNH GIAO DIỆN GRADIO ----------------

# Tạo component nhập văn bản
input_box = gr.Textbox(
    lines=3,
    label="Nhập bình luận tiếng Việt",
    placeholder="Ví dụ: Giao hàng nhanh, đóng gói cẩn thận, rất hài lòng."
)

# Output gồm:
# - Text: nhãn cảm xúc
# - Label: phân bố xác suất 3 lớp
outputs = [
    gr.Textbox(label="Kết quả"),
    gr.Label(label="Xác suất từng lớp cảm xúc")
]

# Một số ví dụ mẫu để người dùng thử nhanh
examples = [
    ["Giao hàng nhanh, đóng gói kỹ, rất hài lòng."],
    ["Sách bìa đẹp nhưng nội dung thì rỗng tuếch."],
    ["Cũng tạm được, không có gì đặc biệt."],
    ["Shop làm ăn chán, treo đầu dê bán thịt chó."]
]

demo = gr.Interface(
    fn=gradio_interface,
    inputs=input_box,
    outputs=outputs,
    title="AI Phân Tích Cảm Xúc Tiếng Việt (PhoBERT Large)",
    description="Nhập một câu bình luận tiếng Việt, mô hình sẽ đoán đó là Tích cực, Trung lập hay Tiêu cực.",
    examples=examples
)


if __name__ == "__main__":
    # share=True giúp tạo link public (nếu chạy trên Colab), debug=True để dễ bắt lỗi
    print("Đang khởi động ứng dụng web Gradio...")
    demo.launch(share=False, debug=True)
