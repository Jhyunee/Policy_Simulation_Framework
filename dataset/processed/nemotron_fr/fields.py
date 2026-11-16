# Per-country persona projection for France Nemotron parquet.
#
# France Nemotron lacks bachelors_field / family_persona / military_status /
# family_type / housing_type. ``commune`` substitutes for ``district`` (smaller
# admin unit), ``departement`` substitutes for ``province`` (broader admin
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
    "commune AS district",
    "departement AS province",
]
