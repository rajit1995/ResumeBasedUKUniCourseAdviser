"""
generate_dataset.py

Generates a large synthetic dataset of UK university courses, in the same
shape as the sample data provided:

    {"university": "...", "course": "...", "focus": ["...", "...", "..."]}

By default it generates 100,000 records and writes them to
`courses_dataset.json` (a single JSON array) and also to
`courses_dataset.jsonl` (one JSON object per line, easier to stream for
large files / RAG ingestion).

Usage:
    python generate_dataset.py
    python generate_dataset.py --count 100000 --seed 42 --out courses_dataset
"""

import argparse
import json
import random

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

UNIVERSITIES = [
    "University of Cambridge", "University of Oxford", "Imperial College London",
    "University College London", "University of Edinburgh", "King's College London",
    "London School of Economics", "University of Manchester", "University of Bristol",
    "University of Warwick", "University of Glasgow", "University of Birmingham",
    "University of Leeds", "University of Sheffield", "University of Southampton",
    "University of Nottingham", "Durham University", "University of York",
    "Queen Mary University of London", "University of St Andrews", "Lancaster University",
    "University of Exeter", "Cardiff University", "Newcastle University",
    "Queen's University Belfast", "University of Bath", "University of Liverpool",
    "Loughborough University", "University of Surrey", "University of Sussex",
    "University of Leicester", "University of Reading", "Royal Holloway, University of London",
    "Aston University", "University of East Anglia", "Heriot-Watt University",
    "University of Aberdeen", "University of Dundee", "Brunel University London",
    "City, University of London", "University of Strathclyde", "Swansea University",
    "University of Stirling", "University of Essex", "University of Kent",
    "Birkbeck, University of London", "Coventry University", "University of Portsmouth",
    "University of the West of England", "Northumbria University",
]

# Degree prefixes, weighted towards taught masters which dominate UK postgrad listings
DEGREE_TYPES = [
    "MSc", "MSc", "MSc", "MA", "MRes", "MPhil", "MEng", "PGDip", "PGCert", "MBA",
]

# Each subject maps to: a list of natural-sounding course-title variants
# and a pool of "focus" tags relevant to that subject.
SUBJECTS = {
    "Artificial Intelligence": {
        "variants": [
            "Artificial Intelligence",
            "Advanced Computer Science (Artificial Intelligence)",
            "Artificial Intelligence and Machine Learning",
            "Applied Artificial Intelligence",
        ],
        "focus_pool": ["AI", "ML", "Research", "Robotics", "Applied ML", "Computer Vision", "NLP"],
    },
    "Machine Learning": {
        "variants": [
            "Machine Learning",
            "Machine Learning and Machine Intelligence",
            "Statistical Machine Learning",
            "Machine Learning Systems",
        ],
        "focus_pool": ["ML", "AI", "Data Science", "NLP", "Deep Learning", "Statistics", "Research"],
    },
    "Data Science": {
        "variants": [
            "Data Science",
            "Data Science and Analytics",
            "Data Science and Artificial Intelligence",
            "Big Data Science",
        ],
        "focus_pool": ["Data Science", "ML", "Big Data", "Analytics", "Statistics", "AI", "Cloud Computing"],
    },
    "Cybersecurity": {
        "variants": [
            "Cyber Security",
            "Information Security",
            "Cyber Security and Networks",
            "Cyber Security Management",
        ],
        "focus_pool": ["Cybersecurity", "Networks", "Cryptography", "Cloud Computing", "Risk Management", "Ethical Hacking"],
    },
    "Robotics": {
        "variants": [
            "Robotics",
            "Robotics and Autonomous Systems",
            "Advanced Robotics",
            "Robotics and Artificial Intelligence",
        ],
        "focus_pool": ["Robotics", "AI", "Control Systems", "Mechatronics", "Applied ML", "Computer Vision"],
    },
    "Software Engineering": {
        "variants": [
            "Software Engineering",
            "Software Engineering and Cloud Computing",
            "Advanced Software Engineering",
            "Software Systems Engineering",
        ],
        "focus_pool": ["Software Engineering", "Cloud Computing", "DevOps", "Systems Design", "Agile", "Distributed Systems"],
    },
    "Business Analytics": {
        "variants": [
            "Business Analytics",
            "Business Analytics and Data Science",
            "Management with Business Analytics",
            "Business Analytics and Consulting",
        ],
        "focus_pool": ["Business", "Data Science", "Analytics", "Finance", "Strategy", "ML"],
    },
    "Finance": {
        "variants": [
            "Finance",
            "Finance and Investment",
            "Financial Technology (FinTech)",
            "Mathematical Finance",
        ],
        "focus_pool": ["Finance", "Economics", "Risk Management", "Investment", "FinTech", "Data Science"],
    },
    "Biomedical Engineering": {
        "variants": [
            "Biomedical Engineering",
            "Biomedical Engineering and Healthcare Technology",
            "Medical Robotics and Image-Guided Intervention",
            "Biomedical Data Science",
        ],
        "focus_pool": ["Healthcare", "Engineering", "Biotech", "Medical Devices", "Research", "Data Science"],
    },
    "Computational Biology": {
        "variants": [
            "Computational Biology",
            "Bioinformatics and Computational Biology",
            "Computational Genomics",
            "Health Data Science",
        ],
        "focus_pool": ["Bioinformatics", "Data Science", "Healthcare", "ML", "Research", "Genomics"],
    },
    "Human-Computer Interaction": {
        "variants": [
            "Human-Computer Interaction",
            "User Experience Design",
            "Human-Centred Computer Systems",
            "Interaction Design",
        ],
        "focus_pool": ["UX Design", "AI", "Cognitive Science", "Software Engineering", "Research", "Design"],
    },
    "Cloud Computing": {
        "variants": [
            "Cloud Computing",
            "Cloud Computing and Networks",
            "Cloud and Distributed Systems",
            "Cloud Computing for Data Science",
        ],
        "focus_pool": ["Cloud Computing", "DevOps", "Networks", "Cybersecurity", "Systems Design", "Data Science"],
    },
    "Natural Language Processing": {
        "variants": [
            "Natural Language Processing",
            "Speech and Language Processing",
            "Computational Linguistics",
            "Natural Language Processing and Machine Learning",
        ],
        "focus_pool": ["NLP", "AI", "ML", "Linguistics", "Research", "Deep Learning"],
    },
    "Computer Vision": {
        "variants": [
            "Computer Vision",
            "Computer Vision and Machine Learning",
            "Computer Vision and Robotics",
            "Advanced Computer Vision",
        ],
        "focus_pool": ["Computer Vision", "AI", "ML", "Robotics", "Applied ML", "Deep Learning"],
    },
    "FinTech": {
        "variants": [
            "Financial Technology",
            "FinTech and Data Science",
            "FinTech with Artificial Intelligence",
            "Banking and Digital Finance",
        ],
        "focus_pool": ["Finance", "AI", "Blockchain", "Data Science", "Software Engineering", "FinTech"],
    },
    "Renewable Energy Engineering": {
        "variants": [
            "Renewable Energy Engineering",
            "Sustainable Energy Systems",
            "Energy and Environmental Engineering",
            "Clean Energy Technologies",
        ],
        "focus_pool": ["Sustainability", "Engineering", "Energy Systems", "Research", "Policy", "Climate"],
    },
    "Civil Engineering": {
        "variants": [
            "Civil Engineering",
            "Structural Engineering",
            "Civil Engineering with Sustainability",
            "Infrastructure Engineering and Management",
        ],
        "focus_pool": ["Engineering", "Construction", "Sustainability", "Project Management", "Infrastructure"],
    },
    "Marketing Analytics": {
        "variants": [
            "Marketing Analytics",
            "Digital Marketing and Analytics",
            "Marketing with Data Analytics",
            "Consumer Analytics",
        ],
        "focus_pool": ["Marketing", "Data Science", "Analytics", "Business", "Strategy", "Digital Media"],
    },
    "Public Health": {
        "variants": [
            "Public Health",
            "Global Public Health",
            "Public Health Data Science",
            "Epidemiology and Public Health",
        ],
        "focus_pool": ["Healthcare", "Policy", "Epidemiology", "Data Science", "Research", "Global Health"],
    },
    "Quantum Computing": {
        "variants": [
            "Quantum Technology",
            "Quantum Computing",
            "Quantum Science and Technology",
            "Quantum Engineering",
        ],
        "focus_pool": ["Quantum Computing", "Physics", "Research", "Cryptography", "Algorithms"],
    },
}

SUBJECT_NAMES = list(SUBJECTS.keys())


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def generate_record(rng: random.Random) -> dict:
    """Generate a single synthetic course record."""
    university = rng.choice(UNIVERSITIES)
    subject_name = rng.choice(SUBJECT_NAMES)
    subject = SUBJECTS[subject_name]

    degree = rng.choice(DEGREE_TYPES)
    variant = rng.choice(subject["variants"])
    course = f"{degree} {variant}"

    focus_pool = subject["focus_pool"]
    # pick 2-4 focus tags, but never more than the pool size
    k = rng.randint(2, min(4, len(focus_pool)))
    focus = rng.sample(focus_pool, k)

    return {"university": university, "course": course, "focus": focus}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic UK course dataset")
    parser.add_argument("--count", type=int, default=100_000, help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="courses_dataset", help="Output file base name (no extension)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    records = [generate_record(rng) for _ in range(args.count)]

    json_path = f"{args.out}.json"
    jsonl_path = f"{args.out}.jsonl"

    # Single JSON array (matches the shape of the sample data given)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # JSON Lines (one record per line) - easier to stream for large datasets / RAG ingestion
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Generated {len(records):,} records")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {jsonl_path}")


if __name__ == "__main__":
    main()
