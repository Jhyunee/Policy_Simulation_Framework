# Unique education_level values from Nemotron-Personas-France, frequency-sorted descending.
# Use as the LLM-facing enum so scope_definer's picks always match a parquet row.

EDUCATION_LEVELS = [
    'CAP ou BEP',           # n=233,675
    'Baccalauréat',         # n=192,249
    'Sans diplôme ou CEP',  # n=188,437
    'Bac+2',                # n=113,959
    'Bac+5 ou plus',        # n=107,503
    'Bac+3 ou Bac+4',       # n=96,056
    'Brevet',               # n=68,121
]
