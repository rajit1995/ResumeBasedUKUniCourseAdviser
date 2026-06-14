import argparse
import csv
import json
import random

# ---------------------------------------------------------------------------
# Departments: each maps to a pool of course titles (>= 10) and a pool of
# "focus" tags relevant to that department.
# ---------------------------------------------------------------------------

DEPARTMENTS = {
    "Science": {
        "courses": [
            "BSc Physics", "BSc Chemistry", "BSc Biology", "BSc Mathematics",
            "MSc Astrophysics", "BSc Biochemistry", "MSc Neuroscience",
            "BSc Materials Science", "MSc Applied Mathematics", "BSc Genetics",
            "MSc Theoretical Physics", "BSc Forensic Science",
            "BSc Molecular Biology", "MSc Computational Science",
        ],
        "focus_pool": [
            "Physics", "Chemistry", "Biology", "Mathematics", "Research",
            "Laboratory Work", "Data Analysis", "Genetics",
        ],
    },
    "Technology & Engineering": {
        "courses": [
            "BEng Computer Science", "MSc Artificial Intelligence",
            "BEng Electrical Engineering", "MSc Robotics",
            "BEng Mechanical Engineering", "MSc Cybersecurity",
            "BEng Civil Engineering", "MSc Data Science",
            "BEng Aerospace Engineering", "MSc Software Engineering",
            "BEng Chemical Engineering", "MSc Cloud Computing",
            "BEng Electronic Engineering", "MSc Machine Learning",
        ],
        "focus_pool": [
            "AI", "Machine Learning", "Software Engineering", "Cybersecurity",
            "Robotics", "Cloud Computing", "Data Science", "Engineering",
        ],
    },
    "Commerce & Business": {
        "courses": [
            "BSc Business Administration", "MSc Finance",
            "BSc Accounting and Finance", "MSc Marketing",
            "BSc Economics", "MBA Business Management",
            "BSc International Business", "MSc Human Resource Management",
            "BSc Entrepreneurship", "MSc Supply Chain Management",
            "BSc Banking and Finance", "MSc Business Analytics",
            "BSc Management with Marketing", "MSc International Management",
        ],
        "focus_pool": [
            "Finance", "Business", "Economics", "Marketing", "Management",
            "Analytics", "Strategy", "Entrepreneurship",
        ],
    },
    "Arts & Humanities": {
        "courses": [
            "BA Philosophy", "BA History", "BA Fine Art",
            "BA Theology and Religious Studies", "BA Classics",
            "BA Archaeology", "MA Art History", "BA Cultural Studies",
            "BA Anthropology", "MA Philosophy and Ethics",
            "BA Museum Studies", "BA Ancient History",
            "BA Religious Studies", "MA History of Art",
        ],
        "focus_pool": [
            "History", "Philosophy", "Art", "Theology", "Culture",
            "Anthropology", "Research", "Critical Thinking",
        ],
    },
    "Geography & Environment": {
        "courses": [
            "BSc Geography", "BSc Environmental Science",
            "MSc Urban Planning", "BSc Geology", "MSc Climate Change",
            "BSc Earth Sciences", "MSc Environmental Management",
            "BSc Oceanography", "MSc Sustainable Development",
            "BSc Human Geography", "MSc Geographic Information Systems",
            "BSc Physical Geography", "BSc Environmental Geoscience",
            "MSc Geospatial Science",
        ],
        "focus_pool": [
            "Geography", "Environmental Science", "Sustainability",
            "Climate", "GIS", "Urban Planning", "Earth Sciences", "Research",
        ],
    },
    "Sports": {
        "courses": [
            "BSc Sports Science", "BSc Sports Coaching",
            "MSc Sports Management", "BSc Physical Education",
            "BSc Exercise and Health Science", "MSc Strength and Conditioning",
            "BSc Sports Therapy", "BSc Sports Journalism",
            "MSc Sport and Exercise Psychology", "BSc Sports Nutrition",
            "BSc Sport Development", "MSc Sports Performance Analysis",
            "BSc Sports Rehabilitation", "BSc Sport Business Management",
        ],
        "focus_pool": [
            "Sports Science", "Coaching", "Fitness", "Health", "Nutrition",
            "Psychology", "Management", "Performance Analysis",
        ],
    },
    "Literature": {
        "courses": [
            "BA English Literature", "BA Creative Writing",
            "MA Comparative Literature", "BA English Language and Literature",
            "MA Creative Writing", "BA Literature and Film",
            "BA Children's Literature", "MA Victorian Literature",
            "BA American Literature", "MA Modern Literature",
            "BA Drama and Literature", "BA World Literature",
            "BA Postcolonial Literature", "MA English Studies",
        ],
        "focus_pool": [
            "Literature", "Creative Writing", "Critical Analysis", "Drama",
            "Cultural Studies", "Research", "Linguistics",
        ],
    },
    "Languages & Linguistics": {
        "courses": [
            "BA Modern Languages (French and Spanish)", "BA Linguistics",
            "MA Translation Studies", "BA German Studies",
            "BA Italian Studies", "BA Chinese Studies",
            "MA Applied Linguistics", "BA Japanese Studies",
            "BA Russian Studies", "BA Arabic Studies", "MA TESOL",
            "BA French and Linguistics", "BA Hispanic Studies",
            "MA Language and Communication",
        ],
        "focus_pool": [
            "Languages", "Linguistics", "Translation", "Communication",
            "Culture", "TESOL", "Research",
        ],
    },
    "Medicine & Health Sciences": {
        "courses": [
            "MBBS Medicine", "BSc Nursing", "MSc Public Health",
            "BSc Pharmacy", "BSc Biomedical Science",
            "MSc Healthcare Management", "BSc Physiotherapy",
            "MSc Nutrition and Dietetics", "BSc Midwifery",
            "MSc Epidemiology", "BSc Dentistry", "MSc Mental Health Nursing",
            "BSc Paramedic Science", "MSc Clinical Research",
        ],
        "focus_pool": [
            "Healthcare", "Medicine", "Clinical Practice", "Public Health",
            "Nutrition", "Nursing", "Research",
        ],
    },
    "Law": {
        "courses": [
            "LLB Law", "LLM International Law", "LLB Law with Criminology",
            "LLM Human Rights Law", "LLB European Law",
            "LLM Commercial Law", "LLB Law and Politics",
            "LLM Intellectual Property Law", "LLB Business Law",
            "LLM International Business Law", "LLB Law with French Law",
            "LLM Environmental Law", "LLB Law with Spanish Law",
            "LLM Maritime Law",
        ],
        "focus_pool": [
            "Law", "Human Rights", "Policy", "International Law",
            "Commercial Law", "Criminology", "Research",
        ],
    },
    "Social Sciences": {
        "courses": [
            "BSc Psychology", "BA Sociology", "BA Politics",
            "MA International Relations", "BA Social Policy",
            "BSc Criminology", "BA Politics and Economics",
            "MA Development Studies", "BSc Cognitive Science",
            "BA Anthropology and Sociology", "MA Political Science",
            "BSc Behavioural Science", "BA Social Work", "MA Global Studies",
        ],
        "focus_pool": [
            "Psychology", "Sociology", "Politics", "Policy",
            "International Relations", "Social Work", "Research",
        ],
    },
}

# Minimum number of courses every university must offer per department.
MIN_COURSES_PER_DEPARTMENT = 10


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_universities(csv_path):
    universities = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("University_name") or "").strip()
            if not name:
                continue
            region = (row.get("Region") or "").strip()
            founded_raw = (row.get("Founded_year") or "").strip()
            motto = (row.get("Motto") or "").strip()
            rank_raw = (row.get("World_rank") or row.get("UK_rank") or "").strip()

            founded_year = int(founded_raw) if founded_raw.isdigit() else None
            uk_rank = int(rank_raw) if rank_raw.isdigit() else None

            universities.append({
                "university": name,
                "region": region,
                "founded_year": founded_year,
                "motto": motto if motto and motto.upper() != "NA" else None,
                "uk_rank": uk_rank,
            })

    if not universities:
        raise SystemExit(f"No universities found in '{csv_path}'.")

    return universities


# ---------------------------------------------------------------------------
# Synthetic admissions statistics (derived from UK rank)
# ---------------------------------------------------------------------------

def admissions_stats(uk_rank, total_unis, rng: random.Random):
    """Generate plausible acceptance-rate / competitiveness figures.

    Lower UK rank (closer to 1) -> more selective -> lower acceptance
    rates and "Very High" competitiveness. Higher-ranked-number
    (less selective) universities tend to have higher acceptance rates
    and are generally reported as more accessible / welcoming to
    international (including Indian) applicants.
    """
    if uk_rank is None:
        percentile = 0.5
    else:
        percentile = (uk_rank - 1) / max(total_unis - 1, 1)  # 0.0 .. 1.0

    base_acceptance = 5 + percentile * 70  # ~5% .. ~75%
    acceptance_rate = base_acceptance + rng.uniform(-5, 5)
    acceptance_rate = max(2.0, min(85.0, acceptance_rate))

    intl_delta = rng.uniform(-3, 8)
    international_acceptance_rate = max(2.0, min(90.0, acceptance_rate + intl_delta))

    if percentile < 0.10:
        india_friendly_score = rng.randint(5, 7)
        competitiveness = "Very High"
    elif percentile < 0.30:
        india_friendly_score = rng.randint(6, 8)
        competitiveness = "High"
    elif percentile < 0.60:
        india_friendly_score = rng.randint(7, 9)
        competitiveness = "Medium"
    else:
        india_friendly_score = rng.randint(8, 10)
        competitiveness = "Low to Medium"

    return {
        "acceptance_rate_pct": round(acceptance_rate, 1),
        "international_acceptance_rate_pct": round(international_acceptance_rate, 1),
        "india_friendly_score": india_friendly_score,
        "competitiveness": competitiveness,
    }


# ---------------------------------------------------------------------------
# Course generation
# ---------------------------------------------------------------------------

def generate_courses_for_university(rng: random.Random):
    """Return a list of {course, department, focus} dicts covering every
    department, with at least MIN_COURSES_PER_DEPARTMENT courses each."""
    courses = []
    for department, info in DEPARTMENTS.items():
        pool = info["courses"]
        focus_pool = info["focus_pool"]

        # Pick MIN_COURSES_PER_DEPARTMENT..len(pool) courses (without
        # replacement) so every university has at least the minimum,
        # with some natural variation between universities.
        k = rng.randint(MIN_COURSES_PER_DEPARTMENT, len(pool))
        chosen = rng.sample(pool, k)

        for course in chosen:
            focus_k = rng.randint(2, min(4, len(focus_pool)))
            focus = rng.sample(focus_pool, focus_k)
            courses.append({"course": course, "department": department, "focus": focus})

    return courses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate UK university + course dataset from a CSV of universities"
    )
    parser.add_argument("--csv", type=str, default="uk_universities.csv",
                         help="Path to the input CSV of UK universities")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="courses_dataset",
                         help="Output file base name for the flattened course dataset (no extension)")
    parser.add_argument("--uni-out", type=str, default="universities_dataset.json",
                         help="Output path for the nested per-university dataset")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    universities = load_universities(args.csv)
    total_unis = len(universities)

    uni_records = []
    flat_records = []

    for uni in universities:
        stats = admissions_stats(uni["uk_rank"], total_unis, rng)
        courses = generate_courses_for_university(rng)

        record = {
            "university": uni["university"],
            "region": uni["region"],
            "founded_year": uni["founded_year"],
            "motto": uni["motto"],
            "uk_rank": uni["uk_rank"],
            **stats,
            "courses": courses,
        }
        uni_records.append(record)

        for c in courses:
            flat_records.append({
                "university": uni["university"],
                "region": uni["region"],
                "founded_year": uni["founded_year"],
                "motto": uni["motto"],
                "uk_rank": uni["uk_rank"],
                **stats,
                "course": c["course"],
                "department": c["department"],
                "focus": c["focus"],
            })

    # Nested per-university dataset
    with open(args.uni_out, "w", encoding="utf-8") as f:
        json.dump(uni_records, f, indent=2)

    # Flattened per-course dataset (used for RAG / vector store ingestion)
    jsonl_path = f"{args.out}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in flat_records:
            f.write(json.dumps(rec) + "\n")

    # Also a single JSON array, for convenience / on-the-fly index builds
    json_path = f"{args.out}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(flat_records, f, indent=2)

    print(f"Loaded {total_unis:,} universities from '{args.csv}'.")
    print(f"Wrote nested university dataset: {args.uni_out} ({len(uni_records):,} universities)")
    print(f"Wrote flattened course dataset: {jsonl_path} ({len(flat_records):,} course records)")
    print(f"Wrote flattened course dataset: {json_path} ({len(flat_records):,} course records)")


if __name__ == "__main__":
    main()
