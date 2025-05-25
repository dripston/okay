import psycopg2

db_config = {
    "dbname": "weather_db",
    "user": "weather_user",
    "password": "your_secure_password",  # Use the same password as above
    "host": "localhost",
    "port": "5432"
}

try:
    conn = psycopg2.connect(**db_config)
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")