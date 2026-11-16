# Per-country persona projection for Singapore Nemotron parquet.
#
# Singapore Nemotron lacks family_persona / military_status / family_type /
# bachelors_field / province / housing_type. ``planning_area`` substitutes for
# ``district``; ``industry`` (employment sector) substitutes for the academic
# ``bachelors_field`` slot so diversity_penalty keys remain meaningful.

PERSONA_FIELDS = [
    "professional_persona",
    "skills_and_expertise",
    "career_goals_and_ambitions",
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    "planning_area AS district",
    "industry AS bachelors_field",
]
