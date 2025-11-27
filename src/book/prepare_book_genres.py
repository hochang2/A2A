import os
import csv
import json
from collections import defaultdict

# -------------------------------------------------------
# 📌 파일 경로 설정
# -------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

BOOKS_PATH = os.path.join(BASE_DIR, "data", "goodbooks-10k", "books.csv")
BOOK_TAGS_PATH = os.path.join(BASE_DIR, "data", "goodbooks-10k", "book_tags.csv")
TAGS_PATH = os.path.join(BASE_DIR, "data", "goodbooks-10k", "tags.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "goodbooks-10k", "book_genres.json")


# -------------------------------------------------------
# 📌 1. Genre Normalization 규칙 정의
# -------------------------------------------------------
def normalize_genre(tag: str) -> str | None:
    """
    Goodreads 태그는 messy하기 때문에, 공통된 장르 이름으로 정규화한다.
    - None 리턴 시 '장르로 취급하지 않음'
    """
    tag = tag.lower()

    # children/childrens/children-s
    if "children" in tag:
        return "children"

    # young adult
    if tag.startswith("ya") or "young adult" in tag:
        return "young-adult"

    # sci-fi
    if "sci-fi" in tag or "science fiction" in tag:
        return "sci-fi"

    # fantasy
    if "fantasy" in tag:
        return "fantasy"

    # romance
    if "romance" in tag:
        return "romance"

    # horror
    if "horror" in tag or "scary" in tag or "ghost" in tag:
        return "horror"

    # mystery / thriller
    if "mystery" in tag or "thriller" in tag or "suspense" in tag:
        return "mystery"

    # adventure
    if "adventure" in tag:
        return "adventure"

    # history / historical
    if "history" in tag or "historical" in tag:
        return "history"

    # classics
    if "classic" in tag:
        return "classic"

    # nonfiction
    if "nonfiction" in tag:
        return "nonfiction"

    # poetry
    if "poetry" in tag:
        return "poetry"

    # religion / spirituality
    if "religion" in tag or "spiritual" in tag:
        return "spirituality"

    return None  # 🔥 잡다한 태그는 버림


# -------------------------------------------------------
# 📌 2. CSV 로드
# -------------------------------------------------------
def load_tags():
    tag_id_to_name = {}
    with open(TAGS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_id_to_name[int(row["tag_id"])] = row["tag_name"]
    return tag_id_to_name


def load_book_tags():
    book_to_tags = defaultdict(list)
    with open(BOOK_TAGS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            book_id = int(row["goodreads_book_id"])
            tag_id = int(row["tag_id"])
            count = int(row["count"])
            book_to_tags[book_id].append((tag_id, count))
    return book_to_tags


# -------------------------------------------------------
# 📌 3. 장르 추출
# -------------------------------------------------------
def extract_genres(tag_id_to_name, book_to_tags, top_k=5):
    book_genres = {}

    for book_id, tag_list in book_to_tags.items():
        # count 기준 내림차순 정렬
        sorted_tags = sorted(tag_list, key=lambda x: x[1], reverse=True)

        genre_list = []
        for (tag_id, count) in sorted_tags[:20]:  # 상위 태그 20개 중에서 필터링
            raw_tag = tag_id_to_name.get(tag_id, "").lower().strip()
            genre = normalize_genre(raw_tag)
            if genre:
                genre_list.append(genre)

        # unique
        genre_list = list(dict.fromkeys(genre_list))

        # 최대 top_k 개만 저장
        book_genres[book_id] = genre_list[:top_k]

    return book_genres


# -------------------------------------------------------
# 📌 4. 메인 실행
# -------------------------------------------------------
def main():
    print("📘 태그 로딩 중...")
    tag_id_to_name = load_tags()

    print("📗 book_tags 로딩 중...")
    book_to_tags = load_book_tags()

    print("📙 장르 추출 중...")
    result = extract_genres(tag_id_to_name, book_to_tags, top_k=5)

    # JSON 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 완료! book_genres.json 생성됨 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
