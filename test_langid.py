"""Quick test for translator module."""
from app.nlp.translator import to_english, from_english

tests = [
    ("मुझे 2 दिन से बुखार है और सिर में दर्द है", "hi"),
    ("மூன்று நாட்களாக காய்ச்சல் இருக்கிறது", "ta"),
    ("मला 3 दिवसांपासून ताप आहे", "mr"),
]

for text, lang in tests:
    eng = to_english(text, lang)
    print(f"  [{lang}] '{text}' -> '{eng}'")

print()
print("  English -> Hindi:", from_english("You have fever symptoms. Please visit the nearest PHC.", "hi"))
