import json
import sqlite3
from pathlib import Path

from reasoning_generator_agent import ReasoningGeneratorAgent


RESULT_PATH = Path(
    "/gpfs/work/aac/jiahuiwang24/"
    "reasoning_generator_3b/results/conditional_tests_result.json"
)


DDL_SCHEMA = """
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    major TEXT
);

CREATE TABLE scores (
    score_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    course TEXT,
    score REAL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);
""".strip()


DATABASE_SETUP = """
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    major TEXT
);

CREATE TABLE scores (
    score_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    course TEXT,
    score REAL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

INSERT INTO students VALUES
    (1, 'Alice', 22, 'Computer Science'),
    (2, 'Bob', 21, 'Mathematics'),
    (3, 'Carol', 20, 'Computer Science'),
    (4, 'David', 25, 'Computer Science');

INSERT INTO scores VALUES
    (1, 1, 'Databases', 95.0),
    (2, 1, 'AI', 88.0),
    (3, 2, 'Databases', 91.0),
    (4, 4, 'AI', 70.0);
""".strip()


TESTS = [
    {
        "id": "T1_single_table_filter",
        "category": "single_table_filter_order",
        "question": (
            "List the names of students who are older than 20 "
            "and major in Computer Science. "
            "Order the names alphabetically."
        ),
        "schema_links": [
            "students.name",
            "students.age",
            "students.major",
        ],
        "expected_rows": [
            ["Alice"],
            ["David"],
        ],
    },
    {
        "id": "T2_required_join",
        "category": "required_join_filter",
        "question": (
            "List the student name and course name for every "
            "score that is at least 90. "
            "Order the results by score from highest to lowest."
        ),
        "schema_links": [
            "students.student_id",
            "students.name",
            "scores.student_id",
            "scores.course",
            "scores.score",
        ],
        "expected_rows": [
            ["Alice", "Databases"],
            ["Bob", "Databases"],
        ],
    },
    {
        "id": "T3_aggregation",
        "category": "join_group_by_average",
        "question": (
            "For each course, show the course name and its "
            "average score. Order the results by course name."
        ),
        "schema_links": [
            "scores.course",
            "scores.score",
        ],
        "expected_rows": [
            ["AI", 79.0],
            ["Databases", 93.0],
        ],
    },
    {
        "id": "T4_not_exists",
        "category": "not_exists_subquery",
        "question": (
            "List the names of students who have no records "
            "in the scores table. Order the names alphabetically."
        ),
        "schema_links": [
            "students.student_id",
            "students.name",
            "scores.student_id",
        ],
        "expected_rows": [
            ["Carol"],
        ],
    },
    {
        "id": "T5_group_by_having",
        "category": "join_group_by_having",
        "question": (
            "List the names of students who have more than one "
            "score record, together with the number of their "
            "score records. Order the results by student name."
        ),
        "schema_links": [
            "students.student_id",
            "students.name",
            "scores.student_id",
            "scores.score_id",
        ],
        "expected_rows": [
            ["Alice", 2],
        ],
    },
]


def normalize_value(value):
    if isinstance(value, float):
        return round(value, 4)
    return value


def normalize_rows(rows):
    return [
        [
            normalize_value(value)
            for value in row
        ]
        for row in rows
    ]


def execute_and_evaluate(sql, expected_rows):
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.executescript(DATABASE_SETUP)

    try:
        cursor.execute(sql)
        actual_rows = normalize_rows(cursor.fetchall())
        execution_success = True
        execution_error = None
    except sqlite3.Error as error:
        actual_rows = []
        execution_success = False
        execution_error = str(error)

    connection.close()

    normalized_expected = normalize_rows(expected_rows)

    return {
        "execution_success": execution_success,
        "execution_error": execution_error,
        "actual_rows": actual_rows,
        "expected_rows": normalized_expected,
        "result_match": actual_rows == normalized_expected,
    }


def main():
    print("Loading the Reasoning Generator Agent once...")
    agent = ReasoningGeneratorAgent(
        max_new_tokens=512,
    )

    all_results = []

    for index, test in enumerate(TESTS, start=1):
        print()
        print("=" * 70)
        print(f"Running {index}/{len(TESTS)}: {test['id']}")
        print("Category:", test["category"])
        print("Question:", test["question"])

        generation_result = agent.generate(
            question=test["question"],
            ddl_schema=DDL_SCHEMA,
            schema_links=test["schema_links"],
            evidence=None,
            dialect="sqlite",
            candidate_count=1,
        )

        candidate = generation_result["candidates"][0]

        if candidate["parse_success"]:
            evaluation = execute_and_evaluate(
                candidate["sql"],
                test["expected_rows"],
            )
        else:
            evaluation = {
                "execution_success": False,
                "execution_error": "SQL extraction failed",
                "actual_rows": [],
                "expected_rows": test["expected_rows"],
                "result_match": False,
            }

        item = {
            "id": test["id"],
            "category": test["category"],
            "question": test["question"],
            "schema_links": test["schema_links"],
            "raw_response": candidate["raw_response"],
            "sql": candidate["sql"],
            "parse_success": candidate["parse_success"],
            "parse_method": candidate["parse_method"],
            "generation_seconds": generation_result[
                "metadata"
            ]["generation_seconds"],
            "evaluation": evaluation,
        }

        all_results.append(item)

        print("Generated SQL:")
        print(candidate["sql"])
        print("Parse success:", candidate["parse_success"])
        print(
            "Execution success:",
            evaluation["execution_success"],
        )
        print("Actual rows:", evaluation["actual_rows"])
        print("Expected rows:", evaluation["expected_rows"])
        print("Result match:", evaluation["result_match"])

    total = len(all_results)
    parse_count = sum(
        item["parse_success"]
        for item in all_results
    )
    execution_count = sum(
        item["evaluation"]["execution_success"]
        for item in all_results
    )
    match_count = sum(
        item["evaluation"]["result_match"]
        for item in all_results
    )

    summary = {
        "model": "Qwen2.5-Coder-3B-Instruct",
        "prompt_version": "v2_minimum_tables",
        "gpu": "NVIDIA A40",
        "total_tests": total,
        "parse_success_count": parse_count,
        "execution_success_count": execution_count,
        "result_match_count": match_count,
        "parse_success_rate": parse_count / total,
        "execution_success_rate": execution_count / total,
        "result_match_rate": match_count / total,
    }

    output = {
        "summary": summary,
        "tests": all_results,
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print(json.dumps(summary, indent=2))
    print("Saved to:", RESULT_PATH)


if __name__ == "__main__":
    main()
