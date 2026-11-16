# Per-country persona projection for Japan Nemotron parquet.
#
# Japan Nemotron lacks bachelors_field / family_persona / military_status /
# family_type / housing_type. ``area`` substitutes for ``district`` (smaller
# admin unit), ``prefecture`` substitutes for ``province`` (broader admin
# region). No good analog exists for ``bachelors_field`` so it is dropped.

PERSONA_FIELDS = [
    "professional_persona",
    "skills_and_expertise",
    "career_goals_and_ambitions",
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    "area AS district",
    "prefecture AS province",
]
