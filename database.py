import sqlite3

conn = sqlite3.connect(
    "toxishield.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""

               CREATE TABLE IF NOT EXISTS history(

                                                     id INTEGER PRIMARY KEY AUTOINCREMENT,

                                                     comment TEXT,

                                                     prediction TEXT,

                                                     score REAL,

                                                     emotion TEXT

               )

               """)

conn.commit()