# File: build_sentiment_features.py
# Pre-computes daily financial sentiment scores using FinBERT from a news dataset.
# Handles NaN/missing text as Neutral (0.0), aggregates scores by Date,
# and exports the clean features CSV to the sentiments/ directory.

import os
from typing import Optional
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ── CONSTANTS & CONFIGURATION ────────────────────────────────────────────────
# 1. Input / Output File Paths
INPUT_CSV_PATH = "sentiments/aapl_news_yahoo.csv"  # Thay đường dẫn file tin tức đầu vào của bạn tại đây
OUTPUT_DIR = "sentiments"
OUTPUT_FILENAME = "aapl_sen_23-07-2012_27-01-2020.csv"  # Tên file xuất ra mặc định

# 2. DataFrame Column Mapping
DATE_COL = "Date"  # Tên cột chứa ngày
TEXT_COL = "title"  # Cột text dùng cho FinBERT ('title' hoặc 'description')
TICKER_COL: Optional[str] = (
    None  # Set Tên cột nếu là multi-ticker, hoặc None nếu single stock
)

# 3. Model & Processing Options
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32  # Tăng/giảm tùy vào GPU/CPU của máy
DEFAULT_NEUTRAL_SCORE = 0.0  # Giá trị gán cho NaN / Tin bị trống
FILL_MISSING_DATES = (
    True  # Tự động điền 0.0 cho các ngày giao dịch/ngày lịch không có tin
)
DATE_FREQ = "D"  # 'D' (Calendar days) hoặc 'B' (Business days - ngày giao dịch)

# ── MAPPING CONFIGURATION ───────────────────────────────────────────────────
LABEL_MAP = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}


def load_sentiment_pipeline():
    """Load FinBERT model & tokenizer from local cache if available."""
    print(f"[FinBERT] Loading model '{MODEL_NAME}'...")
    try:
        # Thử load từ local cache trước
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, local_files_only=True
        )
    except Exception:
        print("[FinBERT] Local cache not found. Downloading weights...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    nlp = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
    )
    return nlp


def calculate_headline_scores(
    texts: list[str], nlp_pipeline
) -> np.ndarray:
    """Batch-compute sentiment scores for a list of text headlines."""
    scores = []
    # Xử lý theo từng batch để tối ưu tốc độ và bộ nhớ
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        # Thay thế các chuỗi rỗng/NaN bằng khoảng trắng an toàn
        safe_batch = [
            str(t).strip() if (pd.notna(t) and str(t).strip()) else "Neutral"
            for t in batch
        ]

        results = nlp_pipeline(safe_batch)
        for original_text, res in zip(batch, results):
            if not pd.notna(original_text) or not str(original_text).strip():
                # Trường hợp NaN/Empty -> ép về Neutral
                scores.append(DEFAULT_NEUTRAL_SCORE)
            else:
                label = res["label"].upper()
                confidence = res["score"]
                score = LABEL_MAP.get(label, 0.0) * confidence
                scores.append(score)

    return np.array(scores)


def main():
    # 1. Kiểm tra & Chuẩn bị môi trường
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(
            f"Input file not found at '{INPUT_CSV_PATH}'. Check CONSTANTS config."
        )

    print(f"[Data] Loading news dataset from '{INPUT_CSV_PATH}'...")
    df = pd.read_csv(INPUT_CSV_PATH)

    # 2. Tiền xử lý cột Ngày
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # 3. Load Model & Tính Sentiment Scores
    nlp_pipeline = load_sentiment_pipeline()
    print(
        f"[FinBERT] Scoring {len(df)} headlines on column '{TEXT_COL}' (Batch Size = {BATCH_SIZE})..."
    )

    headlines = df[TEXT_COL].tolist()
    df["headline_sentiment"] = calculate_headline_scores(
        headlines, nlp_pipeline
    )

    # 4. Groupby theo Ngày (Daily Aggregation)
    print("[Processing] Aggregating scores by date...")
    group_cols = [DATE_COL]
    if TICKER_COL and TICKER_COL in df.columns:
        group_cols.append(TICKER_COL)

    daily_df = (
        df.groupby(group_cols)
        .agg(
            sentiment_score=("headline_sentiment", "mean"),
            news_count=("headline_sentiment", "count"),
        )
        .reset_index()
    )

    # 5. Xử lý các ngày không có tin tức trong khoảng time-series (Optional)
    if FILL_MISSING_DATES and len(daily_df) > 0:
        print(
            f"[Processing] Filling missing dates ({DATE_FREQ}) with Neutral ({DEFAULT_NEUTRAL_SCORE})..."
        )
        min_date = daily_df[DATE_COL].min()
        max_date = daily_df[DATE_COL].max()

        if TICKER_COL and TICKER_COL in daily_df.columns:
            # Multi-ticker date filling
            all_tickers = daily_df[TICKER_COL].unique()
            full_idx = pd.MultiIndex.from_product(
                [
                    pd.date_range(min_date, max_date, freq=DATE_FREQ),
                    all_tickers,
                ],
                names=[DATE_COL, TICKER_COL],
            )
            daily_df = (
                daily_df.set_index([DATE_COL, TICKER_COL])
                .reindex(full_idx)
                .reset_index()
            )
        else:
            # Single-ticker date filling
            full_idx = pd.date_range(min_date, max_date, freq=DATE_FREQ)
            daily_df = (
                daily_df.set_index(DATE_COL)
                .reindex(full_idx)
                .rename_axis(DATE_COL)
                .reset_index()
            )

        # Điền 0.0 cho sentiment_score và 0 cho news_count ngày không có tin
        daily_df["sentiment_score"] = daily_df["sentiment_score"].fillna(
            DEFAULT_NEUTRAL_SCORE
        )
        daily_df["news_count"] = daily_df["news_count"].fillna(0).astype(int)

    # 6. Xuất kết quả ra CSV
    daily_df.to_csv(output_path, index=False)
    print("\n========================================================")
    print(f" SUCCESS: Daily sentiment features saved to '{output_path}'")
    print(f" Total daily records: {len(daily_df)}")
    print(
        f" Date Range: {daily_df[DATE_COL].min().strftime('%Y-%m-%d')} -> {daily_df[DATE_COL].max().strftime('%Y-%m-%d')}"
    )
    print("========================================================\n")
    print(daily_df.head(10))


if __name__ == "__main__":
    main()