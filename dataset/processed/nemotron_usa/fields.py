# Per-country persona projection for USA Nemotron parquet.
#
# USA Nemotron lacks family_persona / military_status / family_type / housing_type.
# ``city`` substitutes for ``district`` (city is the equivalent admin subdivision),
# ``state`` substitutes for ``province``. ``bachelors_field`` is present natively.

PERSONA_FIELDS = [
    "professional_persona",
    "skills_and_expertise",
    "career_goals_and_ambitions",
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    "city AS district",
    "state AS province",
    "bachelors_field",
]
