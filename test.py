from recommend_tours import recommend_for_user

user_profile = {
    "ngan_sach": 7000000,
    "so_thich": ["biển", "thư giãn", "ẩm thực"]
}

result = recommend_for_user(user_profile)
print(result)
