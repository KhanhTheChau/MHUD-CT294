import requests
from bs4 import BeautifulSoup
import os

def download_images(search_term, num_images=5):
    f = "./static/image/"
    # URL tìm kiếm Google Images
    search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={search_term}"
    
    # Gửi yêu cầu GET tới Google Images
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Tìm tất cả các thẻ <img>
    img_tags = soup.find_all("img")

    # Tải hình ảnh
    count = 0
    for img in img_tags:
        if count >= num_images:
            break
        try:
            img_url = img["src"]
            img_data = requests.get(img_url).content
            with open(os.path.join(f, f"image_{count + 1}.jpg"), "wb") as img_file:
                img_file.write(img_data)
                print(f"Đã tải hình ảnh: image_{count + 1}.jpg")
                count += 1
        except Exception as e:
            print(f"Lỗi khi tải hình ảnh: {e}")

# Ví dụ sử dụng
search_term = "Chlorophyllum-molybdites"  # Từ khóa tìm kiếm
download_images(search_term, num_images=4)  # Tải 5 hình ảnh
