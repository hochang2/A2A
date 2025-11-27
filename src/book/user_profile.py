import os
import csv
import json
from collections import Counter, defaultdict

# ---------------------------------------------------------------------
# 📌 파일 경로
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MY_RATINGS_PATH = os.path.join(BASE_DIR, "data", "goodbooks-10k", "my_ratings.csv")
BOOK_GENRES_PATH = os.path.join(BASE_DIR, "data", "goodbooks-10k", "book_genres.json")


# ---------------------------------------------------------------------
# 📌 book_genres.json 로드
# ---------------------------------------------------------------------
def load_book_genres():
    if not os.path.exists(BOOK_GENRES_PATH):
        print(f"[WARN] book_genres.json 없음: {BOOK_GENRES_PATH}")
        return {}
    with open(BOOK_GENRES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)   # { "1": ["fantasy", "sci-fi", ...], ... }


# ---------------------------------------------------------------------
# 📌 my_ratings.csv → 해당 유저가 본 책들 로딩
# ---------------------------------------------------------------------
def load_seen_books(user_id: int):
    """
    my_ratings.csv에서 user_id가 평가한 book_id 목록을 반환한다.
    """
    seen = []
    if not os.path.exists(MY_RATINGS_PATH):
        return seen

    with open(MY_RATINGS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["user_id"]) == user_id:
                seen.append(int(row["book_id"]))
    return seen


# ---------------------------------------------------------------------
# 📌 선호 장르 계산 (본 책들의 장르 빈도)
# ---------------------------------------------------------------------
def compute_top_genres(seen_books, book_genres, top_k=5):
    """
    seen_books: [249042, 17803, ...]
    book_genres: { "1": ["fantasy", "children"], ... }

    반환:
    ["fantasy", "horror", ...]
    """
    counter = Counter()

    for bid in seen_books:
        genres = book_genres.get(str(bid), [])
        counter.update(genres)

    # 상위 top_k 반환
    return [genre for genre, _ in counter.most_common(top_k)]


# ---------------------------------------------------------------------
# 📌 최종 User Profile 생성
# ---------------------------------------------------------------------
def get_user_profile(user_id: int):
    """
    반환 구조:
    {
        "seen_books": [...],
        "top_genres": [...]
    }
    """
    book_genres = load_book_genres()
    seen_books = load_seen_books(user_id=user_id)
    top_genres = compute_top_genres(seen_books, book_genres)

    return {
        "seen_books": seen_books,
        "top_genres": top_genres
    }


# ---------------------------------------------------------------------
# ✔ 테스트 실행
# ---------------------------------------------------------------------
if __name__ == "__main__":
    uid = 1234
    profile = get_user_profile(uid)
    print(profile)
