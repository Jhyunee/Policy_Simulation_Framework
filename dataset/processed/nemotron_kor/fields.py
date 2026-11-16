# Per-country persona projection for Korea Nemotron parquet.
#
# Each entry is a DuckDB SELECT expression. Plain column names project as-is;
# use ``source AS target`` to rename a country-specific column into the canonical
# downstream field name. The trailing token after ``AS`` (or the bare name) is
# what shows up as a key in the persona dict consumed by refiner / Stage 2.

PERSONA_FIELDS = [
    "professional_persona",
    "skills_and_expertise",
    "career_goals_and_ambitions",
    "sex",
    "age",
    "marital_status",
    "military_status",
    "education_level",
    "occupation",
    "district",
    "province",
    "bachelors_field",
]
