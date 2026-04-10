TENX_WINDOW_PROMPT = """
Analyze this 1024x1024 pathology image at 10x magnification.
Output a JSON object with the following keys:
- "tumor_polygon": list of [x, y] coordinates (list of lists) defining the tumor boundary in image coordinates (0-1023). If no tumor, output empty list.
- "subtype_scores": list of 7 floats (clear_cell, papillary, chromophobe, ccprct, medullary, tfe3, fh) summing to 1.
- "boundary_type": either "pushing" or "infiltrative".
- "grade4_prob": probability of sarcomatoid/rhabdoid features (0-1).
Only output the JSON object inside <answer> tags. Keep it concise.
"""