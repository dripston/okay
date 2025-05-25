from data_processor import BangaloreWeatherProcessor
import schedule
import time

db_config = {
    "dbname": "weather_db",
    "user": "weather_user",
    "password": "your_secure_password",  # Use the password you set in the terminal
    "host": "localhost",
    "port": "5432"
}

def daily_job():
    processor = BangaloreWeatherProcessor(db_config)
    processor.process_daily_update()

if __name__ == "__main__":
    # Schedule daily updates at 1 AM
    schedule.every().day.at("01:00").do(daily_job)
    
    while True:
        schedule.run_pending()
        time.sleep(60)