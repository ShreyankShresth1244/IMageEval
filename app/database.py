import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from app.config import DATABASE_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.

    Returns:
        conn (psycopg2.extensions.connection): The database connection object.
    """
    try:
        conn = psycopg2.connect(
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
            dbname=DATABASE_CONFIG["database"],
            cursor_factory=RealDictCursor
        )
        logging.info("Database connection established.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        raise


def initialize_tables():
    """
    Create necessary tables for the tool in the database if they do not already exist.
    """
    create_products_table_query = """
    CREATE TABLE IF NOT EXISTS products (
        id SERIAL PRIMARY KEY,
        image_url1 TEXT,
        image_url2 TEXT,
        image_url3 TEXT,
        image_url4 TEXT,
        image_url5 TEXT
    );
    """

    create_enhanced_images_table_query = """
    CREATE TABLE IF NOT EXISTS enhanced_images (
        id SERIAL PRIMARY KEY,
        original_image_url TEXT,
        enhanced_image_url TEXT,
        evaluation_status TEXT,
        issues_detected JSON
    );
    """

    conn = None
    try:
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(create_products_table_query)
                cursor.execute(create_enhanced_images_table_query)
                logging.info("Tables initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing tables: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


def test_database_setup():
    """
    Test the database setup by checking the connection and listing existing tables.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
                """)
                tables = cursor.fetchall()
                logging.info("Database setup verified. Existing tables:")
                for table in tables:
                    logging.info(f"- {table['table_name']}")
    except Exception as e:
        logging.error(f"Error testing database setup: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


if __name__ == "__main__":
    # Initialize tables and test database setup
    initialize_tables()
    test_database_setup()
