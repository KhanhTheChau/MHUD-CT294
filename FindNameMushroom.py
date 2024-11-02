api_key = "AIzaSyA9kHBTKC_4-mcuNHSKfZIS75oS-QvIve0"

import google.generativeai as genai

genai.configure(api_key=api_key)

prompt_dict = {
    "cap-shape": {
        "b": "bell",
        "c": "conical",
        "x": "convex",
        "f": "flat",
        "k": "knobbed",
        "s": "sunken"
    },
    "cap-surface": {
        "f": "fibrous",
        "g": "grooves",
        "y": "scaly",
        "s": "smooth"
    },
    "cap-color": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "r": "green",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "bruises": {
        "t": "bruises",
        "f": "no bruises"
    },
    "odor": {
        "a": "almond",
        "l": "anise",
        "c": "creosote",
        "y": "fishy",
        "f": "foul",
        "m": "musty",
        "n": "none",
        "p": "pungent",
        "s": "spicy"
    },
    "gill-attachment": {
        "a": "attached",
        "f": "free"
    },
    "veil-color": {
        "n": "brown",
        "o": "orange",
        "w": "white",
        "y": "yellow"
    },
    "ring-number": {
        "n": "none",
        "o": "one",
        "t": "two"
    },
    "ring-type": {
        "c": "cobwebby",
        "e": "evanescent",
        "f": "flaring",
        "l": "large",
        "n": "none",
        "p": "pendant",
        "s": "sheathing",
        "z": "zone"
    },
    "spore-print-color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "r": "green",
        "o": "orange",
        "u": "purple",
        "w": "white",
        "y": "yellow"
    },
    "population": {
        "a": "abundant",
        "c": "clustered",
        "n": "numerous",
        "s": "scattered",
        "v": "several",
        "y": "solitary"
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
    "gill-spacing": {
        "c": "close",
        "w": "crowded"
    },
    "gill-size": {
        "b": "broad",
        "n": "narrow"
    },
    "gill-color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "g": "gray",
        "r": "green",
        "o": "orange",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "stalk-shape": {
        "e": "enlarging",
        "t": "tapering"
    },
    "stalk-root": {
        "b": "bulbous",
        "c": "club",
        "e": "equal",
        "r": "rooted"
    },
    "stalk-surface-above-ring": {
        "f": "fibrous",
        "k": "silky",
        "s": "smooth",
        "y": "scaly"
    },
    "stalk-surface-below-ring": {
        "f": "fibrous",
        "k": "silky",
        "s": "smooth",
        "y": "scaly"
    },
    "stalk-color-above-ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "stalk-color-below-ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "veil-type": {
        "p": "partial",
        "u": "universal"
    }
}


def FindName(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    print(response.text)
    
    return response.text

# print(FindName())