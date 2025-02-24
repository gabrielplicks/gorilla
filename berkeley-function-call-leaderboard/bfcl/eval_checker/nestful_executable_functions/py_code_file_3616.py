import sqlite3



def insert_user(name: str, email: str) -> int:

    """Inserts a new user into the database and returns the new user ID.

    Args:

        name: The name of the user.

        email: The email of the user.

    """

    conn = sqlite3.connect('users.db')

    cursor = conn.cursor()

    cursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))

    conn.commit()

    cursor.execute('SELECT last_insert_rowid()')

    user_id = cursor.fetchone()[0]

    conn.close()



    return user_id

