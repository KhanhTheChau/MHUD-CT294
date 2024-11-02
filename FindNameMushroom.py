api_key = "AIzaSyDA68zSuHD0IL4AEm06XYSPwBWZs8Wu4_Y"

import google.generativeai as genai

genai.configure(api_key=api_key)

prompt_dict = {
    "cap-shape": {
        "b": "bell",
        "c": "conical",
        "x": "convex",
        "x": "flat",
        "k": "knobbed",
        "s": "sunken"
    },
    "stalk-surface-above-ring": {
        "f": "fibrous",
        "y": "scaly",
        "k": "silky",
        "s": "smooth"
    },
    "veil-color": {
      "n": "brown",
      "o": "orange",
      "w": "white",
      "y": "yellow"  
    },
    "habitat": {
        "g": "grasses",
        "l": "leaves",
        "m": "meadows",
        "p": "paths",
        "u": "urban",
        "w": "waste",
        "d": "woods"
    },
    "population": {
        "a": "abundant",
        "c": "clustered",
        "n": "numerous",
        "s": "scattered",
        "v": "several"
    }
}


def FindName(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    print(response.text)
    
    return response.text

# print(FindName())