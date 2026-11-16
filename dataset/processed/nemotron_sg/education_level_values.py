# Unique education_level values from Nemotron-Personas-Singapore, frequency-sorted descending.
# Use as the LLM-facing enum so scope_definer's picks always match a parquet row.

EDUCATION_LEVELS = [
    'University',  # n=52,147
    'Secondary',  # n=21,392
    'Polytechnic',  # n=17,923
    'Post Secondary (Non-Tertiary)',  # n=17,037
    'Other Diploma',  # n=14,249
    'Primary',  # n=8,709
    'No Qualification',  # n=8,360
    'Lower Secondary',  # n=8,183
]
