import cloudscraper
from bs4 import BeautifulSoup
import json
import time
import re
import urllib.parse
import os

# ==========================================
# 1. CẤU HÌNH VÀ DANH SÁCH NGUYÊN LIỆU
# ==========================================
raw_names = ['', '-', 'táo', 'măng tây', 'bơ', 'chuối', 'ớt chuông', 'mướp đắng', 'cải chíp', 'bông cải xanh', 'bắp cải', 'cà rốt', 'hạt điều', 'bông cải trắng', 'su su', 'ức gà', 'đùi gà', 'cánh gà', 'ớt', 'dừa', 'dừa', 'ngô', 'cua', 'dưa chuột', 'trứng', 'trứng_', 'cà tím', 'tỏi', 'gừng', 'nho', 'chanh', 'xà lách', 'đuôi tôm hùm', 'xoài', 'dưa lưới', 'sữa', 'hành tây', 'cam', 'hàu', 'đu đủ', 'đậu phộng', 'đậu Hà Lan', 'hạt tiêu', 'tiêu', 'dứa', 'thịt ba chỉ', 'khoai tây', 'bí đỏ', 'củ cải', 'gạo lứt đỏ', 'cá hồi', 'sò điệp', 'tôm', 'cải bó xôi', 'dâu tây', 'khoai lang', 'tempeh', 'đậu hũ', 'cà chua', 'cá ngừ', 'gạo trắng']

def get_clean_ingredients(names_list):
    # Loại bỏ các nhãn rác từ YOLO
    trash_labels = ['', '-', "''''", 'trứng_']
    clean = set([x.strip() for x in names_list if x and x not in trash_labels])
    return list(clean)

INGREDIENTS_LIST = get_clean_ingredients(raw_names)
BASE_URL = "https://cookpad.com"

# Khởi tạo Cloudscraper để vượt qua tường lửa của Cookpad
session = cloudscraper.create_scraper()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
})

# ==========================================
# 2. CÁC HÀM CÀO DỮ LIỆU
# ==========================================
def get_recipe_links(ingredient):
    links = []
    query = urllib.parse.quote(ingredient)
    page = 1
    MAX_PAGES = 10 # Giới hạn số trang tối đa muốn cào cho mỗi nguyên liệu
    
    print(f"► Đang tìm kiếm liên kết cho: {ingredient}")
    
    # THAY ĐỔI Ở ĐÂY: Thay `while True` bằng điều kiện `page <= MAX_PAGES`
    while page <= MAX_PAGES:
        search_url = f"{BASE_URL}/vn/tim-kiem/{query}?page={page}"
        try:
            response = session.get(search_url, timeout=10)
            if response.status_code != 200:
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            new_links_found = 0
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if '/vn/cong-thuc/' in href and 'tao-moi' not in href:
                    full_link = BASE_URL + href
                    if full_link not in links:
                        links.append(full_link)
                        new_links_found += 1
                        
            # Dừng nếu trang này không có thêm link mới nào
            if new_links_found == 0:
                break
                
            print(f"   ↳ Lấy được {new_links_found} link ở trang {page}/{MAX_PAGES}...")
            time.sleep(0.5) # Chống block IP
            
            # TĂNG SỐ TRANG LÊN ĐỂ LẬT TRANG TIẾP THEO
            page += 1 
            
        except Exception as e:
            print(f"  [!] Lỗi tìm kiếm '{ingredient}' trang {page}: {e}")
            break
            
    return links

def extract_details(url):
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Tiêu đề
        title = soup.find('h1').text.strip() if soup.find('h1') else "N/A"
        
        # Hình ảnh
        img_tag = soup.find('picture').find('img') if soup.find('picture') else None
        image_url = img_tag.get('src', '') if img_tag else ""
        
        # Thời gian nấu
        cook_time = 0
        time_text = soup.find(string=re.compile(r'\d+\s*(phút|giờ)'))
        if time_text:
            nums = re.findall(r'\d+', time_text)
            if nums:
                cook_time = int(nums[0])
                if 'giờ' in time_text: cook_time *= 60
                
        # Bắt khối Nguyên liệu
        ingredients = []
        ing_div = soup.find('div', class_=lambda c: c and 'ingredient-list' in c)
        if ing_div:
            for li in ing_div.find_all('li'):
                txt = re.sub(r'\s+', ' ', li.text.strip())
                if len(txt) > 0:
                    ingredients.append(txt)
        
        # Fallback Nguyên liệu
        if not ingredients:
            ing_header = soup.find(lambda tag: tag.name in ["h2", "h3", "h4", "div", "span"] and tag.text and "nguyên liệu" in tag.text.lower())
            if ing_header:
                container = ing_header.find_next_sibling(['ul', 'ol', 'div'])
                if container:
                    ingredients = [li.text.strip() for li in container.find_all(['li', 'div']) if len(li.text.strip()) > 2]
                    
        # Bắt khối Các bước làm
        steps = [re.sub(r'\s+', ' ', s.text.strip()) for s in soup.find_all(attrs={"itemprop": "recipeInstructions"})]
        if not steps:
            steps_container = soup.find(id='steps')
            if steps_container:
                steps = [re.sub(r'\s+', ' ', p.text.strip()) for p in steps_container.find_all('p') if len(p.text.strip()) > 2]
            else:
                step_div = soup.find('div', class_=lambda c: c and 'step-list' in c)
                if step_div:
                    steps = [re.sub(r'\s+', ' ', li.text.strip()) for li in step_div.find_all(['li', 'div', 'p']) if len(li.text.strip()) > 2]
        
        # Lượt yêu thích
        likes = 0
        likes_tag = soup.find(string=re.compile(r'^\d+$'))
        if likes_tag: 
            try:
                likes = int(likes_tag.strip())
            except:
                pass

        if len(ingredients) > 0:
            return {
                "title": title,
                "likes_count": likes,
                "cook_time": cook_time,
                "ingredients": ingredients,
                "steps": steps,
                "image_url": image_url
            }
        else:
            print(f"  [-] Bỏ qua '{title}': Không tìm thấy khối nguyên liệu.")
            return None
            
    except Exception as e:
        print(f"  [!] Lỗi khi vào link: {e}")
        return None

# ==========================================
# 3. KỊCH BẢN CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    final_data = []
    
    print("="*60)
    print(f"🚀 BẮT ĐẦU CÀO DỮ LIỆU TỪ COOKPAD CHO {len(INGREDIENTS_LIST)} NGUYÊN LIỆU")
    print("="*60)

    start_time = time.time()

    for ing in INGREDIENTS_LIST:
        links = get_recipe_links(ing)
        
        if not links:
            print("  [-] Không lấy được link nào.")
            continue
            
        success_count = 0
        for link in links:
            data = extract_details(link)
            if data:
                final_data.append(data)
                success_count += 1
                print(f"  [+] Đã lưu: {data['title']}")
                
            time.sleep(1.5) # Nghỉ 1.5s giữa mỗi link để không làm server Cookpad quá tải
            
        print(f"✅ Hoàn thành nguyên liệu [{ing}]: Lưu {success_count} công thức.\n")

    # Lưu toàn bộ dữ liệu ra file JSON
    output_file = 'cookpad_dataset1.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    end_time = time.time()
    
    print("="*60)
    print(f"🎉 HOÀN TẤT TRỌN VẸN QUÁ TRÌNH CÀO DỮ LIỆU!")
    print(f"📁 Tổng số công thức lưu được: {len(final_data)}")
    print(f"⏱️ Tổng thời gian chạy: {round(end_time - start_time, 2)} giây")
    print(f"💾 File lưu tại: {os.path.abspath(output_file)}")
    print("="*60)