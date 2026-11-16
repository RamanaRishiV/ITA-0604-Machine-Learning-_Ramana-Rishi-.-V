
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(
    BASE_DIR, "instance", "attrisense.db"
)

if not os.path.exists(DATABASE_PATH):
    print("Database not found. Check the instance folder.")
    raise SystemExit(1)

connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()

try:
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in cursor.fetchall()]

    if "employee_name" not in columns:
        cursor.execute(
            "ALTER TABLE predictions "
            "ADD COLUMN employee_name VARCHAR(100)"
        )
        print("Added employee_name column.")

    if "employee_id" not in columns:
        cursor.execute(
            "ALTER TABLE predictions "
            "ADD COLUMN employee_id VARCHAR(50)"
        )
        print("Added employee_id column.")

    # Preserve old prediction records with identifiable placeholders.
    cursor.execute("""
        UPDATE predictions
        SET employee_name = 'Existing Employee'
        WHERE employee_name IS NULL
    """)

    cursor.execute("""
        UPDATE predictions
        SET employee_id = 'LEGACY-' || id
        WHERE employee_id IS NULL
    """)

    # Enforce employee ID uniqueness within each user's account.
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        unique_employee_id_per_user
        ON predictions (user_id, employee_id)
    """)

    connection.commit()

    print("Database migration completed successfully!")

except Exception as error:
    connection.rollback()
    print("Migration failed:", error)

finally:
    connection.close()
