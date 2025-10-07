"""
Database Configuration and Initial Setup
AI Warehouse Management System
"""

import os
import sqlite3
from warehouse_ai import WarehouseAI

def initialize_database():
    """Initialize the database with proper schema"""
    print("🗄️ Initializing database schema...")
    
    # Initialize the AI system (this will create tables)
    ai_system = WarehouseAI()
    
    print("✅ Database schema created successfully!")
    return ai_system

def check_database_exists():
    """Check if database file exists"""
    return os.path.exists('warehouse.db')

def get_table_info():
    """Get information about existing tables"""
    conn = sqlite3.connect('warehouse.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    table_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        table_info[table_name] = count
    
    conn.close()
    return table_info

if __name__ == '__main__':
    if not check_database_exists():
        print("Database not found. Creating new database...")
        initialize_database()
    else:
        print("Database exists. Table information:")
        tables = get_table_info()
        for table, count in tables.items():
            print(f"  {table}: {count} records")