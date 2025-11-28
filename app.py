import os
import torch
import gradio as gr
from underthesea import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "./my_phobert_sentiment"

if not os.path.exists(MODEL_PATH):
    print(f"LỖI: Không tìm thấy thư mục model tại '{MODEL_PATH}'.")
    print("Vui lòng chạy file train trước để lưu model, hoặc chỉnh lại đường dẫn.")
    exit()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Đang chạy ứng dụng trên thiết bị: {device}")

print("Đang tải model và tokenizer... Vui lòng chờ!")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    print("✅ Tải model thành công!")
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    exit()

def predict_sentiment(text):
    if not text:
        return None
    
    text_seg = word_tokenize(text, format="text")
    
    inputs = tokenizer(
        text_seg, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128, 
        padding="max_length"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
    
    return {
        "Tiêu cực 😡": float(probs[0]),
        "Trung lập 😐": float(probs[1]),
        "Tích cực 😍": float(probs[2])
    }

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Nhập bình luận vào đây (Ví dụ: Hàng đẹp nhưng giao hơi chậm)...", 
        label="📝 Nội dung bình luận"
    ),
    outputs=gr.Label(num_top_classes=3, label="📊 Kết quả phân tích"),
    title="🤖 AI PHÂN TÍCH CẢM XÚC (PHOBERT)",
    description="""
    **Mô hình:** PhoBERT Large (Fine-tuned)
    **Chức năng:** Dự đoán cảm xúc của bình luận tiếng Việt.
    **Nhãn:** Tích cực (Positive), Trung lập (Neutral), Tiêu cực (Negative).
    """,
    examples=[
        ["Giao hàng siêu nhanh, đóng gói cẩn thận, sách rất đẹp!"],
        ["Chất lượng sản phẩm quá tệ, không bao giờ quay lại."],
        ["Hàng tạm ổn, giá hơi cao so với chất lượng."],
        ["Shop treo đầu dê bán thịt chó, lừa đảo."],
        ["Mọi thứ đều ổn, shipper thân thiện."]
    ],
    theme="default"
)

if __name__ == "__main__":
    print("🌐 Đang khởi động Web Server...")
    iface.launch(share=False, inbrowser=True)