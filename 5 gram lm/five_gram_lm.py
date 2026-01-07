import re
import random
from collections import defaultdict, Counter

# text preprocessing (remove punctuation, lowercase, tokenize)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

# build 5-gram model({(w1,w2,w3,w4):{w5:count,w6:count,...},...})
def build_ngram_model(tokens, n=5):
    model = defaultdict(Counter)
    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i:i+n-1])
        next_word = tokens[i+n-1]
        model[context][next_word] += 1
    return model

# text generation
def generate_text(model, seed, length=30):
    output = seed.lower().split()

    if len(output) < 4:
        return "Please enter at least 4 words."

    for _ in range(length):
        context = tuple(output[-4:])
        if context not in model:
            break
        next_word = random.choices(
            list(model[context].keys()),
            weights=model[context].values()
        )[0]
        output.append(next_word)

    return " ".join(output)

if __name__ == "__main__":
    with open("pride_and_prejudice.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = preprocess_text(text)
    model = build_ngram_model(tokens)

    print("5-gram Language Model (Jane Austen Style)")
    print("Type 'exit' to quit\n")

    while True:
        seed = input("Enter at least 4-word seed text: ")

        if seed.lower() == "exit":
            break

        output = generate_text(model, seed)
        print("\nGenerated Text:")#maximum length of output text is 30words
        print(output)