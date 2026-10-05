import json
import re
import torch
import numpy as np
import os
import pickle
import gc
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Đảm bảo in ra tiếng Việt không bị lỗi encoding trên terminal Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. CẤU HÌNH VÀ TẢI MÔ HÌNH PHO-BERT
# ==========================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(" Đang tải mô hình PhoBERT (Vui lòng đợi)...")
model_name = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
phobert_model = AutoModel.from_pretrained(model_name)
print(" Tải AI hoàn tất!")

def get_embedding(text):
    if not text.strip():
        return np.zeros(768)
    segmented_text = word_tokenize(text, format="text")
    encoded_input = tokenizer(segmented_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        features = phobert_model(**encoded_input)
    sentence_embedding = features.last_hidden_state[0, 0, :].numpy()
    del encoded_input, features
    return sentence_embedding

# ==========================================
# 2. XỬ LÝ DỮ LIỆU VÀ TÍNH VECTOR (CÓ CACHE)
# ==========================================
DATASET_PATH = os.path.join(_BASE_DIR, 'datasets', 'cookpad_dataset_cleaned.json')
CACHE_PATH   = os.path.join(_BASE_DIR, 'cache', 'recipe_vectors_cache.pkl')
META_PATH    = os.path.join(_BASE_DIR, 'cache', 'recipe_vectors_cache.meta.json')
BATCH_SIZE   = 50   # Nhúng 50 món/lần, tự lưu checkpoint

print("📂 Đang tải Database JSON...")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    raw_dishes = json.load(f)

# Loại bỏ công thức trùng lặp (giữ lượt thích cao nhất)
clean_recipes = {}
for recipe in raw_dishes:
    title = recipe['title'].strip().lower()
    if title not in clean_recipes:
        clean_recipes[title] = recipe
    else:
        clean_recipes[title]['likes_count'] += recipe['likes_count']
        if clean_recipes[title]['cook_time'] == 0 and recipe['cook_time'] > 0:
            clean_recipes[title]['cook_time'] = recipe['cook_time']

dishes = list(clean_recipes.values())
print(f"   ✅ Sau khi lọc trùng: {len(dishes)} công thức duy nhất (từ {len(raw_dishes)} bản gốc)")

# ---- Kiểm tra cache có hợp lệ không ----
def _cache_is_valid():
    """Trả về True nếu cache tồn tại, đúng kích thước và metadata khớp."""
    if not os.path.exists(CACHE_PATH) or not os.path.exists(META_PATH):
        return False
    try:
        with open(META_PATH, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return meta.get('num_dishes') == len(dishes) and os.path.normcase(meta.get('dataset_path', '')) == os.path.normcase(DATASET_PATH)
    except Exception:
        return False

# ---- Load hoặc build vectors ----
if _cache_is_valid():
    print("⚡ Đang nạp Vector từ cache (hợp lệ)...")
    with open(CACHE_PATH, 'rb') as f:
        database_vectors = pickle.load(f)
    print(f"   ✅ Nạp xong {len(database_vectors)} vectors.")
else:
    # Xóa cache cũ nếu tồn tại nhưng không hợp lệ
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print("🗑️  Đã xóa cache cũ không hợp lệ.")

    total = len(dishes)
    print(f"🔄 Bắt đầu nhúng {total} công thức thành Vector...")
    print(f"   (Mỗi batch {BATCH_SIZE} món — có thể mất 30-90 phút lần đầu)")

    database_vectors_list = []
    for i, dish in enumerate(dishes):
        ingredients_str = " ".join(dish.get('ingredients', []))
        context_text = f"{dish.get('title', '')} {ingredients_str}"
        vec = get_embedding(context_text)
        database_vectors_list.append(vec)

        # Lưu checkpoint mỗi BATCH_SIZE món
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"   [{i+1:>5}/{total}] {pct:5.1f}% — Đang lưu checkpoint...")
            tmp_arr = np.array(database_vectors_list)
            with open(CACHE_PATH + '.tmp', 'wb') as f:
                pickle.dump(tmp_arr, f)
            gc.collect()

    # Hoàn tất: rename tmp → chính thức
    database_vectors = np.array(database_vectors_list)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(database_vectors, f)
    if os.path.exists(CACHE_PATH + '.tmp'):
        os.remove(CACHE_PATH + '.tmp')

    # Lưu metadata
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump({'num_dishes': len(dishes), 'dataset_path': DATASET_PATH}, f, ensure_ascii=False)

    print(f"✅ Đã lưu cache {len(database_vectors)} vectors vào: {CACHE_PATH}")

database_vectors = np.array(database_vectors)

# ==========================================
# 3. TẬP LUẬT XAI (GIẢI THÍCH LÝ DO - RULE ENGINE)
# ==========================================

# ── Tất cả class YOLO từ data.yaml (nc=61) ──
# Chỉ dùng khi cần tham chiếu đúng tên class gốc
ALL_YOLO_CLASSES = [
    'táo', 'măng tây', 'bơ', 'chuối', 'ớt chuông', 'mướp đắng',
    'cải chíp', 'bông cải xanh', 'bắp cải', 'cà rốt', 'hạt điều',
    'bông cải trắng', 'su su', 'ức gà', 'đùi gà', 'cánh gà', 'ớt',
    'dừa', 'ngô', 'cua', 'dưa chuột', 'trứng', 'cà tím', 'tỏi',
    'gừng', 'nho', 'chanh', 'xà lách', 'đuôi tôm hùm', 'xoài',
    'dưa lưới', 'sữa', 'hành tây', 'cam', 'hàu', 'đu đủ',
    'đậu phộng', 'đậu Hà Lan', 'hạt tiêu', 'tiêu', 'dứa',
    'thịt ba chỉ', 'khoai tây', 'bí đỏ', 'củ cải', 'gạo lứt đỏ',
    'cá hồi', 'sò điệp', 'tôm', 'cải bó xôi', 'dâu tây',
    'khoai lang', 'tempeh', 'đậu hũ', 'cà chua', 'cá ngừ', 'gạo trắng',
]

# ── CLASS_SYNONYMS: Đã xóa hoàn toàn theo yêu cầu của user ──
# Không map "ba rọi" -> "thịt ba chỉ" nữa. Đầu vào là gì thì phải khớp y chang chữ đó.
CLASS_SYNONYMS = {}

# ── NON_YOLO_MAIN_INGREDIENTS: Các nguyên liệu chính không có trong YOLO ──
NON_YOLO_MAIN_INGREDIENTS = [
    "thịt", "xá xíu", "xíu mại", "thịt băm", "thịt bằm", "thịt xay", "thịt nạc", "cháo", "bún", "miến", "mì", "phở", "hủ tiếu", "bánh canh",
    "thịt bò", "bò", "mực", "bạch tuộc", "lạp xưởng", "xúc xích", "chả", "giò", "sườn",
    "nấm", "ốc", "hến", "ngao", "lươn", "ếch", "vịt", "dê", "ghẹ", "cơm", "xôi", "nui", "cút", "trứng cút",
    "chân gà", "mề gà", "lòng gà", "sụn gà", "cổ gà", "đầu gà",
    "thịt gà", "gà", "thịt cá", "cá", "thịt heo", "thịt lợn", "heo", "tôm hùm", "tôm", "đậu",
    "ba rọi", "thịt ba rọi"
]

# Danh sách tất cả các marker có thể xuất hiện trong tên món
ALL_TITLE_MARKERS = sorted(ALL_YOLO_CLASSES + NON_YOLO_MAIN_INGREDIENTS, key=len, reverse=True)

CONDIMENTS_ONLY = [
    # Danh sách ban đầu của bạn
    "mắm", "muối", "đường", "bột ngọt", "hạt nêm", "dầu ăn", "dầu mè", "dầu hào",
    "tương", "xì dầu", "nước tương", "giấm", "bơ", "sữa đặc", "nước cốt dừa",
    "bột năng", "bột mì", "ngò", "sả", "lá chanh", "mật ong", "sa tế", "mì chính",
    "rượu", "kem", "phô mai", "gia vị", "hành lá", "hành tím", "hành khô", "hành phi",
    "ớt", "tương ớt", "chanh", "rau răm", "ngò rí", "bột tỏi", "đường phèn", "nước",
    "mayonnaise",
    
    # Bổ sung thêm các loại gia vị mặn, điều vị & dầu
    "bột canh", "nước cốt chanh", "tỏi", "gừng", "dầu olive", "dầu phộng", "dầu màu điều",
    "hắc xì dầu", "giấm táo", "giấm gạo", 
    
    # Bổ sung thêm các loại đường, si rô & chất tạo ngọt
    "đường thốt nốt", "đường mía thô", "đường đen", "đường vàng", "đường ngô",
    "si rô phong", "si rô bắp", "vani",
    
    # Bổ sung thêm các loại nước sốt, mè & lá thơm
    "tương cà", "ketchup", "sốt mè rang", "sốt hummus", "sốt spaghetti", 
    "muối vừng", "mè rang", "lá thơm", "oregano", "basil", "húng tây", 
    "thyme", "xạ hương", "parsley", "mùi tây"
]
YOLO_FLAVOR_CLASSES = {
    "tỏi", "gừng", "hành tây", "hạt tiêu", "tiêu", "hành",
}
IGNORE_MISSING = CONDIMENTS_ONLY + list(YOLO_FLAVOR_CLASSES)

def is_condiment_only(ingredients: list) -> bool:
    if not ingredients:
        return False
    cleaned = {clean_word(i) for i in ingredients}
    return cleaned.issubset(YOLO_FLAVOR_CLASSES)

def clean_word(word: str) -> str:
    return word.lower().strip()

def _word_in(marker: str, text: str) -> bool:
    pattern = r'(?<!\w)' + re.escape(marker) + r'(?!\w)'
    return bool(re.search(pattern, text, re.UNICODE))

def scan_title_markers(title_lower: str):
    found = []
    temp_title = title_lower
    for marker in ALL_TITLE_MARKERS:
        if _word_in(marker, temp_title):
            found.append(marker)
            pattern = r'(?<!\w)' + re.escape(marker) + r'(?!\w)'
            temp_title = re.sub(pattern, ' ', temp_title, flags=re.UNICODE)
    return found

def apply_xai_rules(yolo_ingredients, dish, semantic_score):
    rule_score = 0.0
    explanations = []

    title_lower = dish.get('title', '').lower()
    recipe_ingredients = []
    for ing in dish.get('ingredients', []):
        recipe_ingredients.extend([x.strip().lower() for x in ing.split(',') if x.strip()])
    likes_count = dish.get('likes_count', 0)
    cook_time = dish.get('cook_time', 0)

    yolo_cleaned = [clean_word(y) for y in yolo_ingredients]
    yolo_set = set(yolo_cleaned)

    # ═══════════════════════════════════════════════════════════════
    # LUẬT 1: KHỚP DANH SÁCH NGUYÊN LIỆU
    # Nguyên liệu trùng khớp → cộng điểm (Thưởng combo)
    # Nguyên liệu có trong công thức nhưng không có trong đầu vào → trừ điểm
    # ═══════════════════════════════════════════════════════════════
    matched_yolo = []
    for y_clean, y_orig in zip(yolo_cleaned, yolo_ingredients):
        if any(y_clean in r_ing for r_ing in recipe_ingredients):
            if y_orig not in matched_yolo:
                matched_yolo.append(y_orig)

    # Thưởng combo: 1 khớp→+5.0 | 2 khớp→+12.5 | 3 khớp→+23.75
    if matched_yolo:
        n = len(matched_yolo)
        rule_score += sum(5.0 * (1.5 ** i) for i in range(n))
        
        labels = list(dict.fromkeys(matched_yolo))
        explanations.append(
            f"✅ Tuyệt vời! {n} nguyên liệu [{', '.join(labels)}] "
            f"có trong công thức!"
        )

    # Phạt LUÔN: -3.0 mỗi nguyên liệu người dùng có mà không dùng đến
    unused_yolo = []
    for y in yolo_cleaned:
        if not any(y in r for r in recipe_ingredients):
            unused_yolo.append(y)
            
    if unused_yolo:
        rule_score -= 3.0 * len(unused_yolo)
        explanations.append(
            f"⚠️ Bỏ phí nguyên liệu: Không dùng đến [{', '.join(unused_yolo)}]."
        )

    # Trừ điểm cho nguyên liệu xuất hiện trong công thức nhưng người dùng không có
    not_in_input = [
        r_ing for r_ing in recipe_ingredients
        if not any(m in r_ing for m in yolo_set) and not any(sp in r_ing for sp in IGNORE_MISSING)
    ]
    if not_in_input:
        rule_score -= 0.3 * len(not_in_input)
        explanations.append(
            f"➖ {len(not_in_input)} nguyên liệu trong công thức bạn không có."
        )

    # ═══════════════════════════════════════════════════════════════
    # LUẬT 3: THIẾU NGUYÊN LIỆU PHỤ
    # ═══════════════════════════════════════════════════════════════
    all_recipe_non_basic = [
        r for r in recipe_ingredients
        if not any(sp in r for sp in IGNORE_MISSING)
    ]
    matched_count = sum(
        1 for r in all_recipe_non_basic
        if any(y_c in r for y_c in yolo_cleaned)
    )
    missing_extra = [
        r.split(',')[0].strip()
        for r in all_recipe_non_basic
        if not any(y_c in r for y_c in yolo_cleaned)
    ]

    if missing_extra:
        # Phạt nặng khi thiếu nguyên liệu KHÔNG phải gia vị (-1.5 thay vì -0.3)
        rule_score -= 1.5 * len(missing_extra)
        total_nb = len(all_recipe_non_basic)
        coverage = matched_count / total_nb if total_nb > 0 else 1.0
        
        # Nếu chỉ đáp ứng được < 50% nguyên liệu chính và có từ 3 nguyên liệu trở lên -> phạt thêm rất nặng
        if coverage < 0.5 and total_nb >= 3:
            rule_score -= 5.0
            
        if len(missing_extra) > 1:
            explanations.append(
                f"🛒 Thiếu {len(missing_extra)} nguyên liệu chính "
                f"(bạn có {matched_count}/{total_nb}): [{missing_extra[0]}]..."
            )
        else:
            explanations.append(f"🛒 Gần đủ! Cần thêm nguyên liệu chính: [{missing_extra[0]}].")

    # ═══════════════════════════════════════════════════════════════
    # LUẬT 4 & 5: LIKES + THỜI GIAN → CHỈ LÀ TIEBREAKER (micro-bonus)
    # ═══════════════════════════════════════════════════════════════
    tiebreak = 0.0
    if likes_count > 20:
        tiebreak += 0.002
        explanations.append(f"❤️ {likes_count} lượt thích — cộng đồng đã thử và khen!")
    elif likes_count > 5:
        tiebreak += 0.001
        explanations.append(f"👍 {likes_count} lượt thích.")

    if 0 < cook_time <= 30:
        tiebreak += 0.002
        explanations.append(f"⏱️ Nhanh: {cook_time} phút.")
    elif 0 < cook_time <= 90:
        tiebreak += 0.001
        explanations.append(f"⏱️ {cook_time} phút.")

    rule_score += tiebreak
    return rule_score, explanations

# ==========================================
# 4. HỆ THỐNG GỢI Ý (RECOMMENDATION ENGINE)
# ==========================================
def recommend_recipes(yolo_ingredients, top_k=5):
    """
    Xếp hạng theo:
      1. rule_score  — ingredient match (PRIMARY)
      2. semantic    — PhoBERT similarity (TIEBREAKER, nhân 0.05)
    Likes & thời gian chỉ phá tie khi rule_score bằng nhau (micro-bonus 0.001-0.002).
    Returns: (recipes, cold_start_type)
      cold_start_type: None | "condiment_only" | "low_score"
    """
    # CS-2: Toàn bộ là gia vị → không gợi ý, hướng dẫn user thêm nguyên liệu chính
    if is_condiment_only(yolo_ingredients):
        return [], "condiment_only"

    query_text = " ".join(yolo_ingredients)
    query_vector = get_embedding(query_text)
    cosine_scores = cosine_similarity([query_vector], database_vectors)[0]

    final_scores = []
    for idx, dish in enumerate(dishes):
        if idx >= len(cosine_scores):
            break
        semantic_score = float(cosine_scores[idx])
        if not dish.get('title') or not dish.get('ingredients'):
            continue
        rule_bonus, explanations = apply_xai_rules(yolo_ingredients, dish, semantic_score)

        # total = ingredient match (chính) + semantic (chỉ tiebreaker)
        total_score = rule_bonus + 0.05 * semantic_score

        final_scores.append({
            'title':          dish.get('title', ''),
            'likes_count':    dish.get('likes_count', 0),
            'cook_time':      dish.get('cook_time', 0),
            'ingredients':    dish.get('ingredients', []),
            'steps':          dish.get('steps', []),
            'image_url':      dish.get('image_url', ''),
            'total_score':    round(total_score, 4),
            'semantic_score': round(semantic_score, 4),
            'rule_bonus':     round(rule_bonus, 4),
            'explanations':   explanations,
        })

    final_scores.sort(key=lambda x: x['total_score'], reverse=True)
    top = final_scores[:top_k]

    # CS-3: Score tốt nhất vẫn âm → kết quả không khớp tốt
    cold_start_type = "low_score" if (top and top[0]['total_score'] < 0) else None
    return top, cold_start_type


# ==========================================
# CHẠY THỬ NGHIỆM (Chỉ khi chạy trực tiếp)
# ==========================================
if __name__ == "__main__":
    results, cold_start_type = recommend_recipes(["bắp cải", "cà rốt"], top_k=10)
    for i, r in enumerate(results):
        print(f"\n{i+1}. {r['title'].upper()}")
        print(f"   📊 Điểm: {r['total_score']} | ⏱️ {r['cook_time']} phút | ❤️ {r['likes_count']}")
        for exp in r['explanations']: 
            print(f"   {exp}")
