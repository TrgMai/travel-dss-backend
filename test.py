import os
import json

def merge_hotel_files(input_dir="data/hotels", output_file="data/all_hotels.json"):
    all_hotels = []

    if not os.path.exists(input_dir):
        print(f"[!] Không tìm thấy thư mục: {input_dir}")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_hotels.extend(data)  # nối danh sách khách sạn
                    else:
                        print(f"[!] ⚠ File {filename} không phải list => bỏ qua")
            except Exception as e:
                print(f"[!] ❌ Lỗi đọc {filename}: {e}")

    # Ghi ra file tổng
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_hotels, f, ensure_ascii=False, indent=2)

    print(f"[✅] Đã gộp {len(all_hotels)} khách sạn vào '{output_file}'")

if __name__ == "__main__":
    merge_hotel_files()
