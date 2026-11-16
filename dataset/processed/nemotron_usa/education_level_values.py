# Unique education_level values from Nemotron-Personas-USA, frequency-sorted descending.
# Use as the LLM-facing enum so scope_definer's picks always match a parquet row.

EDUCATION_LEVELS = [
    'less_than_9th',  # n=251,677
    'high_school',  # n=212,908
    'some_college',  # n=165,909
    'bachelors',  # n=158,158
    'graduate',  # n=95,110
    'associates',  # n=66,209
    '9th_12th_no_diploma',  # n=50,029
]
