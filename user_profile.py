def build_user_profile(data):
    return {
        "ngan_sach": data.get("ngan_sach", 10000000),
        "so_ngay": data.get("so_ngay", 3),
        "so_thich": data.get("so_thich", [])
    }
