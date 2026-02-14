# sqlite.py
from __future__ import annotations

import os
import sqlite3
import random
from datetime import date, timedelta
from pathlib import Path


DB_NAME = os.environ.get("SQLITE_DB", "school.db")
SEED = int(os.environ.get("SQLITE_SEED", "42"))

# Scale knobs (keep modest for fast demo)
NUM_STUDENTS = int(os.environ.get("NUM_STUDENTS", "120"))
NUM_COURSES = int(os.environ.get("NUM_COURSES", "14"))
SEMESTERS = ["2024-Fall", "2025-Spring", "2025-Fall"]  # change freely


FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ishaan", "Krishna",
    "Ananya", "Aadhya", "Diya", "Ira", "Meera", "Saanvi", "Myra", "Aarohi", "Riya",
    "Rahul", "Kiran", "Suresh", "Priya", "Neha", "Vikram", "Nikhil", "Sneha", "Pooja",
]
LAST_NAMES = [
    "Verma", "Patel", "Gupta", "Mehta", "Singh",
    "Kumar", "Das", "Roy", "Bose", "Chowdhury",
]

PROGRAMS = ["Computer Science", "Data Science", "AI & ML", "Information Systems", "Cybersecurity"]
SECTIONS = ["A", "B", "C", "D"]

DEPARTMENTS = ["CS", "DS", "AI", "IS", "CY"]
COURSE_TITLES = [
    "Database Systems", "Operating Systems", "Computer Networks", "Machine Learning",
    "Deep Learning", "Data Structures", "Algorithms", "Cloud Computing",
    "NLP Fundamentals", "Information Security", "Software Engineering",
    "Data Visualization", "MLOps Foundations", "Graph Databases",
    "Statistics for Data Science", "Ethical AI",
]

GRADE_BANDS = [
    ("A", 90, 100),
    ("B", 80, 89),
    ("C", 70, 79),
    ("D", 60, 69),
    ("F", 0, 59),
]


def make_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def grade_from_score(score: float) -> str:
    for letter, lo, hi in GRADE_BANDS:
        if lo <= score <= hi:
            return letter
    return "F"


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys = ON;")
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA synchronous = NORMAL;")
    return con


def recreate_schema(con: sqlite3.Connection) -> None:
    cur = con.cursor()

    # Drop in FK-safe order
    cur.executescript(
        """
        DROP TABLE IF EXISTS attendance;
        DROP TABLE IF EXISTS enrollments;
        DROP TABLE IF EXISTS courses;
        DROP TABLE IF EXISTS students;
        """
    )

    cur.executescript(
        """
        CREATE TABLE students (
            student_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            program      TEXT NOT NULL,
            section      TEXT NOT NULL,
            year         INTEGER NOT NULL CHECK (year BETWEEN 1 AND 4)
        );

        CREATE TABLE courses (
            course_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            course_code  TEXT NOT NULL UNIQUE,
            course_name  TEXT NOT NULL,
            department   TEXT NOT NULL,
            credits      INTEGER NOT NULL CHECK (credits BETWEEN 1 AND 6)
        );

        CREATE TABLE enrollments (
            enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id    INTEGER NOT NULL,
            course_id     INTEGER NOT NULL,
            semester      TEXT NOT NULL,
            score         REAL NOT NULL CHECK (score BETWEEN 0 AND 100),
            grade         TEXT NOT NULL CHECK (grade IN ('A','B','C','D','F')),
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
            FOREIGN KEY (course_id)  REFERENCES courses(course_id)  ON DELETE CASCADE,
            UNIQUE(student_id, course_id, semester)
        );

        CREATE TABLE attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id    INTEGER NOT NULL,
            course_id     INTEGER NOT NULL,
            semester      TEXT NOT NULL,
            class_date    TEXT NOT NULL,
            present       INTEGER NOT NULL CHECK (present IN (0,1)),
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
            FOREIGN KEY (course_id)  REFERENCES courses(course_id)  ON DELETE CASCADE
        );

        CREATE INDEX idx_enrollments_student ON enrollments(student_id);
        CREATE INDEX idx_enrollments_course  ON enrollments(course_id);
        CREATE INDEX idx_enrollments_sem     ON enrollments(semester);

        CREATE INDEX idx_att_student_course  ON attendance(student_id, course_id);
        CREATE INDEX idx_att_semester        ON attendance(semester);
        CREATE INDEX idx_att_date            ON attendance(class_date);
        """
    )

    con.commit()


def seed_students(con: sqlite3.Connection, rng: random.Random) -> None:
    cur = con.cursor()
    rows = []
    for _ in range(NUM_STUDENTS):
        rows.append(
            (
                make_name(rng),
                rng.choice(PROGRAMS),
                rng.choice(SECTIONS),
                rng.randint(1, 4),
            )
        )
    cur.executemany(
        "INSERT INTO students(name, program, section, year) VALUES (?,?,?,?)",
        rows,
    )
    con.commit()


def seed_courses(con: sqlite3.Connection, rng: random.Random) -> None:
    cur = con.cursor()
    titles = COURSE_TITLES[:]
    rng.shuffle(titles)
    titles = titles[:NUM_COURSES]

    rows = []
    for i, title in enumerate(titles, start=1):
        dept = rng.choice(DEPARTMENTS)
        code = f"{dept}{100 + i}"
        credits = rng.choice([2, 3, 3, 4])
        rows.append((code, title, dept, credits))

    cur.executemany(
        "INSERT INTO courses(course_code, course_name, department, credits) VALUES (?,?,?,?)",
        rows,
    )
    con.commit()


def seed_enrollments_and_attendance(con: sqlite3.Connection, rng: random.Random) -> None:
    cur = con.cursor()

    student_ids = [r[0] for r in cur.execute("SELECT student_id FROM students").fetchall()]
    course_ids = [r[0] for r in cur.execute("SELECT course_id FROM courses").fetchall()]

    enrollment_rows = []
    attendance_rows = []

    # Build a small calendar per semester (10 class dates)
    sem_start = {
        "2024-Fall": date(2024, 9, 1),
        "2025-Spring": date(2025, 2, 1),
        "2025-Fall": date(2025, 9, 1),
    }

    for sem in SEMESTERS:
        start = sem_start.get(sem, date(2025, 1, 1))
        class_dates = [(start + timedelta(days=7 * i)).isoformat() for i in range(10)]

        for sid in student_ids:
            # each semester: 3-5 courses
            chosen = rng.sample(course_ids, k=rng.randint(3, 5))
            for cid in chosen:
                # score distribution: mostly 60-95
                base = rng.gauss(mu=78, sigma=10)
                score = max(0, min(100, round(base, 1)))
                grade = grade_from_score(score)

                enrollment_rows.append((sid, cid, sem, score, grade))

                # attendance probability correlates loosely with score
                # higher score => slightly higher attendance
                p_present = min(0.98, max(0.60, 0.70 + (score - 70) / 100))
                for d in class_dates:
                    present = 1 if rng.random() < p_present else 0
                    attendance_rows.append((sid, cid, sem, d, present))

    cur.executemany(
        "INSERT OR IGNORE INTO enrollments(student_id, course_id, semester, score, grade) VALUES (?,?,?,?,?)",
        enrollment_rows,
    )
    cur.executemany(
        "INSERT INTO attendance(student_id, course_id, semester, class_date, present) VALUES (?,?,?,?,?)",
        attendance_rows,
    )
    con.commit()


def create_views(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    cur.executescript(
        """
        DROP VIEW IF EXISTS student_performance;

        CREATE VIEW student_performance AS
        SELECT
            s.student_id,
            s.name,
            s.program,
            s.section,
            e.semester,
            ROUND(AVG(e.score), 2) AS avg_score,
            SUM(CASE WHEN e.grade = 'A' THEN 1 ELSE 0 END) AS num_A,
            COUNT(*) AS num_courses
        FROM students s
        JOIN enrollments e ON e.student_id = s.student_id
        GROUP BY s.student_id, e.semester;
        """
    )
    con.commit()


def print_summary(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()
    print("Tables:", [t[0] for t in tables])

    for t in ["students", "courses", "enrollments", "attendance"]:
        n = cur.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
        print(f"{t}: {n}")

    # A couple example queries
    print("\nExample: Top 5 students by avg score (latest semester)")
    latest = cur.execute("SELECT semester FROM enrollments ORDER BY semester DESC LIMIT 1;").fetchone()[0]
    rows = cur.execute(
        """
        SELECT s.name, s.program, ROUND(AVG(e.score), 2) AS avg_score
        FROM students s
        JOIN enrollments e ON e.student_id = s.student_id
        WHERE e.semester = ?
        GROUP BY s.student_id
        ORDER BY avg_score DESC
        LIMIT 5;
        """,
        (latest,),
    ).fetchall()
    for r in rows:
        print(r)


def main() -> None:
    rng = random.Random(SEED)
    db_path = Path(DB_NAME).resolve()

    con = connect(db_path)
    try:
        recreate_schema(con)
        seed_students(con, rng)
        seed_courses(con, rng)
        seed_enrollments_and_attendance(con, rng)
        create_views(con)
        print(f"Created DB: {db_path}")
        print_summary(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
