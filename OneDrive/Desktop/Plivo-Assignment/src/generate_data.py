import json
import random

PERSON_NAMES = ["Alice", "Bob", "Ramesh Sharma", "John Doe", "Priya Kumar", "Anil Singh", "Sara Khan", "Vikram Patel", "Maya Rao", "Karan Mehta"]
CITIES = ["Chennai", "Mumbai", "New York", "Berlin", "Paris", "Tokyo", "London", "Sydney", "Delhi", "San Francisco"]
LOCATIONS = ["Central Park", "Marina Beach", "Eiffel Tower", "Louvre Museum", "Golden Gate Bridge", "Red Fort", "Colosseum", "Times Square", "Hyde Park", "Sydney Opera House"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "example.com", "outlook.com", "hotmail.com"]
DATES = ["12/03/2020", "01-01-2023", "July 4, 2022", "15-08-2021", "23/11/2019", "31-12-2022", "05/05/2020", "10-10-2021", "02/02/2023", "20/07/2022"]
CREDIT_CARDS = ["4242 4242 4242 4242", "5555 5555 5555 4444", "4111 1111 1111 1111", "3782 822463 10005", "6011 1111 1111 1117",
                "3530 111333 00000", "5105 1051 0510 5100", "6011 0009 9013 9424", "3530 111333 000000", "5555 4444 3333 1111"]
PHONES = ["9876543210", "1234567890", "9988776655", "1112223333", "7778889999",
          "6665554444", "9998887777", "5556667777", "4443332222", "3332221111"]

TEMPLATES = {
    "CREDIT_CARD": "My credit card number is {val}",
    "EMAIL": "You can email me at {val}",
    "PHONE": "Call me on {val}",
    "PERSON_NAME": "My name is {val}",
    "DATE": "My birthday is {val}",
    "CITY": "I live in {val}",
    "LOCATION": "I want to visit {val} next year",
}

POOLS = {
    "CREDIT_CARD": CREDIT_CARDS,
    "EMAIL": [f"{name.lower().replace(' ', '.')}@{random.choice(EMAIL_DOMAINS)}" for name in PERSON_NAMES],
    "PHONE": PHONES,
    "PERSON_NAME": PERSON_NAMES,
    "DATE": DATES,
    "CITY": CITIES,
    "LOCATION": LOCATIONS,
}

def apply_stt_noise(text, entity_spans):
    """Add mild STT-like noise without breaking offsets"""
    noisy_text = ""
    last_idx = 0
    for start, end in sorted(entity_spans):
        # text before entity
        segment = text[last_idx:start]
        segment = segment.replace("@", " at ").replace(".", " dot ")
        # optionally replace digits with words (safe)
        digit_map = {"0":"zero","1":"one","2":"two","3":"three","4":"four",
                     "5":"five","6":"six","7":"seven","8":"eight","9":"nine"}
        for d in "0123456789":
            if random.random() < 0.2:  # lower chance
                segment = segment.replace(d, digit_map[d])
        noisy_text += segment
        # entity remains intact
        noisy_text += text[start:end]
        last_idx = end
    # after last entity
    segment = text[last_idx:]
    segment = segment.replace("@", " at ").replace(".", " dot ")
    for d in "0123456789":
        if random.random() < 0.2:
            segment = segment.replace(d, digit_map[d])
    noisy_text += segment
    return noisy_text

def generate_example(labels_in_sentence):
    entities = []
    sentence_parts = []
    char_idx = 0
    for label in labels_in_sentence:
        val = random.choice(POOLS[label])
        template = TEMPLATES[label]
        text_piece = template.format(val=val)
        start = char_idx + text_piece.find(val)
        end = start + len(val)
        entities.append({"start": start, "end": end, "label": label})
        sentence_parts.append(text_piece)
        char_idx += len(text_piece) + 1  # space between parts
    text = " ".join(sentence_parts)
    noisy_text = apply_stt_noise(text, [(e["start"], e["end"]) for e in entities])
    return noisy_text, entities

def generate_dataset(out_file="data/train.jsonl", total_examples=1000, start_id=1):
    labels = list(POOLS.keys())
    data = []
    curr_id = start_id

    for label in labels:
        for _ in range(10):
            chosen_labels = random.sample(labels, k=random.randint(1,3))
            text, entities = generate_example(chosen_labels)
            data.append({"id": f"utt_{curr_id:04d}", "text": text, "entities": entities})
            curr_id += 1

    while len(data) < total_examples:
        chosen_labels = random.sample(labels, k=random.randint(1,3))
        text, entities = generate_example(chosen_labels)
        data.append({"id": f"utt_{curr_id:04d}", "text": text, "entities": entities})
        curr_id += 1

    random.shuffle(data)
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    print(f"Generated {len(data)} examples in {out_file}")
    return curr_id

def generate_dev_set(out_file="data/dev.jsonl", total_examples=200, start_id=1001):
    labels = list(POOLS.keys())
    data = []
    curr_id = start_id
    while len(data) < total_examples:
        chosen_labels = random.sample(labels, k=random.randint(1,3))
        text, entities = generate_example(chosen_labels)
        data.append({"id": f"utt_{curr_id:04d}", "text": text, "entities": entities})
        curr_id += 1

    random.shuffle(data)
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    print(f"Generated {len(data)} examples in {out_file}")

if __name__ == "__main__":
    random.seed(42)
    next_id = generate_dataset(out_file="data/train.jsonl", total_examples=1000, start_id=1)
    generate_dev_set(out_file="data/dev.jsonl", total_examples=200, start_id=next_id)
