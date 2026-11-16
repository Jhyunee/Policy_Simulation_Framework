# Unique education_level values from Nemotron-Personas-Japan, frequency-sorted descending.
# Use as the LLM-facing enum so scope_definer's picks always match a parquet row.

EDUCATION_LEVELS = [
    '高校卒',  # n=386,715
    '大学卒 文系',  # n=168,203
    '短大卒',  # n=135,663
    '中学卒',  # n=103,588
    '高専卒',  # n=90,050
    '大学卒 理系',  # n=72,319
    '大学院卒 理系',  # n=23,296
    '大学院卒 文系',  # n=10,017
    '大学 文系',  # n=5,460
    '大学 理系',  # n=2,402
    '大学院 理系',  # n=1,544
    '大学院 文系',  # n=743
]
