# Patch Notes

## 2026-05-24

Fixed Streamlit Cloud CSV upload error:

```text
TypeError at src/collectors.py line 103
df["_combined_text"] = df.astype(str).agg(" | ".join, axis=1)
```

Changes:
- Replaced fragile pandas `agg(" | ".join, axis=1)` fallback with row-wise safe combiner.
- Added `safe_cell()` to handle NaN, None, numeric, date, and mixed cell types.
- Added more Korean column aliases: 댓글, 의견, 언론사, 매체, 등록일, 주소.
- Added CSV cp949 fallback in `app.py` for Korean CSV files.
