import os
import uuid
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

# Import module gợi ý (sẽ tải PhoBERT + cache khi khởi động)
from yolo_recipe_xai import recommend_recipes

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ==========================================
# CẤU HÌNH
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ANNOTATED_FOLDER = os.path.join(BASE_DIR, 'static', 'annotated')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# Class names từ data.yaml (61 classes)
CLASS_NAMES = [
    '', '-', 'táo', 'măng tây', 'bơ', 'chuối', 'ớt chuông', 'mướp đắng',
    'cải chíp', 'bông cải xanh', 'bắp cải', 'cà rốt', 'hạt điều', 'bông cải trắng',
    'su su', 'ức gà', 'đùi gà', 'cánh gà', 'ớt', 'dừa', 'dừa', 'ngô', 'cua',
    'dưa chuột', 'trứng', 'trứng_', 'cà tím', 'tỏi', 'gừng', 'nho', 'chanh',
    'xà lách', 'đuôi tôm hùm', 'xoài', 'dưa lưới', 'sữa', 'hành tây', 'cam',
    'hàu', 'đu đủ', 'đậu phộng', 'đậu Hà Lan', 'hạt tiêu', 'tiêu', 'dứa',
    'thịt ba chỉ', 'khoai tây', 'bí đỏ', 'củ cải', 'gạo lứt đỏ', 'cá hồi',
    'sò điệp', 'tôm', 'cải bó xôi', 'dâu tây', 'khoai lang', 'tempeh',
    'đậu hũ', 'cà chua', 'cá ngừ', 'gạo trắng'
]

# Chuẩn hoá class: chỉ fix alias/lỗi, giữ nguyên tên gốc để XAI rules
# có thể match đúng với tên món (ví dụ: "Ức gà xào hành tây")
NORMALIZE_MAP = {
    'trứng_': 'trứng',   # alias trùng
    '':       None,       # class rỗng → bỏ qua
    '-':      None,       # class '-' → bỏ qua
}

# Màu sắc cho bounding box (HSL palette đẹp)
BOX_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
]

# ==========================================
# TẢI YOLO MODEL
# ==========================================
print(" Đang tải YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print(" YOLO tải xong!")

# ==========================================
# ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/annotated/<filename>')
def annotated_image(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Không có file ảnh được gửi lên'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    # Lưu ảnh tạm
    ext = os.path.splitext(file.filename)[1].lower() or '.jpg'
    unique_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(upload_path)

    try:
        # ---- BƯỚC 1: YOLO nhận diện ----
        results = yolo_model.predict(source=upload_path, conf=0.25, verbose=False)
        result = results[0]

        # Lấy danh sách nguyên liệu phát hiện được
        detected_raw = []
        boxes_data = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                if class_id < len(CLASS_NAMES):
                    raw_name = CLASS_NAMES[class_id]
                    norm_name = NORMALIZE_MAP.get(raw_name, raw_name)
                    if norm_name:  # Bỏ qua class rỗng / '-'
                        detected_raw.append(norm_name)
                        xyxy = box.xyxy[0].tolist()
                        boxes_data.append({
                            'x1': int(xyxy[0]), 'y1': int(xyxy[1]),
                            'x2': int(xyxy[2]), 'y2': int(xyxy[3]),
                            'label': norm_name,
                            'confidence': round(confidence * 100, 1)
                        })

        # Loại bỏ trùng lặp, giữ thứ tự
        yolo_ingredients = list(dict.fromkeys(detected_raw))

        # ---- BƯỚC 2: Vẽ bounding box lên ảnh ----
        img = Image.open(upload_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        for i, box in enumerate(boxes_data):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            # Vẽ hộp
            draw.rectangle([box['x1'], box['y1'], box['x2'], box['y2']],
                            outline=color, width=3)
            # Vẽ label background
            label_text = f"{box['label']} {box['confidence']}%"
            draw.rectangle([box['x1'], box['y1'] - 22, box['x1'] + len(label_text) * 8, box['y1']],
                            fill=color)
            draw.text((box['x1'] + 4, box['y1'] - 20), label_text, fill='white')

        annotated_name = f"ann_{unique_name}"
        annotated_path = os.path.join(ANNOTATED_FOLDER, annotated_name)
        img.save(annotated_path)

        # ---- BƯỚC 3: Gợi ý món ăn ----
        if not yolo_ingredients:
            return jsonify({
                'ingredients': [],
                'annotated_image': f'/static/annotated/{annotated_name}',
                'recipes': [],
                'message': 'Không phát hiện nguyên liệu nào trong ảnh. Vui lòng thử ảnh khác!'
            })

        recipes, cold_start_type = recommend_recipes(yolo_ingredients, top_k=10)

        return jsonify({
            'ingredients': yolo_ingredients,
            'annotated_image': f'/static/annotated/{annotated_name}',
            'recipes': recipes,
            'cold_start_type': cold_start_type,
            'message': f'Phát hiện {len(yolo_ingredients)} nguyên liệu!'
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500
    finally:
        # Dọn ảnh gốc tạm
        if os.path.exists(upload_path):
            os.remove(upload_path)


if __name__ == '__main__':
    print("\n🌐 Web demo đang chạy tại: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
