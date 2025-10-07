"""
Database Setup and Sample Data Generator
AI Warehouse Management System
Date: October 2025

This script creates sample data for the warehouse management system
including products, inventory levels, transactions, and historical data
for AI model training.
"""

import sqlite3
import random
from datetime import datetime, timedelta
import json

class DatabaseSetup:
    def __init__(self, db_path='warehouse.db'):
        self.db_path = db_path
    
    def create_sample_data(self):
        """Create comprehensive sample data for the warehouse system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("🗄️ Creating sample database with realistic warehouse data...")
        
        # Sample product categories and their typical products
        sample_products = [
            # Grains & Cereals
            ('RICE001', 'Basmati Rice Premium', 'Grains', 45.50, 100, 5000, 'Golden Harvest Ltd', 5),
            ('RICE002', 'Jasmine Rice', 'Grains', 52.00, 80, 4000, 'Asian Grains Co.', 7),
            ('WHEAT001', 'Wheat Flour Organic', 'Grains', 35.00, 200, 10000, 'Organic Mills', 3),
            ('WHEAT002', 'Whole Wheat Flour', 'Grains', 32.50, 150, 8000, 'Natural Foods', 4),
            ('OATS001', 'Rolled Oats Premium', 'Grains', 68.00, 50, 2000, 'Health First', 6),
            ('CORN001', 'Corn Flour Yellow', 'Grains', 28.00, 100, 3000, 'Farm Fresh', 4),
            
            # Pulses & Legumes
            ('PULSE001', 'Red Lentils Premium', 'Pulses', 75.00, 100, 3000, 'Protein Source Ltd', 5),
            ('PULSE002', 'Green Lentils Organic', 'Pulses', 82.00, 80, 2500, 'Organic Pulses', 6),
            ('PULSE003', 'Chickpeas Kabuli', 'Pulses', 95.00, 60, 2000, 'Delhi Traders', 4),
            ('PULSE004', 'Black Beans', 'Pulses', 110.00, 40, 1500, 'International Foods', 8),
            ('PULSE005', 'Kidney Beans Red', 'Pulses', 105.00, 50, 1800, 'Premium Beans', 7),
            
            # Oils & Fats
            ('OIL001', 'Sunflower Oil Refined', 'Oils', 85.00, 50, 2000, 'Sun Oil Industries', 5),
            ('OIL002', 'Olive Oil Extra Virgin', 'Oils', 450.00, 20, 500, 'Mediterranean Oils', 12),
            ('OIL003', 'Coconut Oil Virgin', 'Oils', 280.00, 30, 800, 'Tropical Oils', 8),
            ('OIL004', 'Mustard Oil Pure', 'Oils', 120.00, 40, 1200, 'Traditional Oils', 6),
            ('OIL005', 'Sesame Oil', 'Oils', 380.00, 15, 400, 'Specialty Oils', 10),
            
            # Sweeteners
            ('SUGAR001', 'White Sugar Refined', 'Sweeteners', 42.00, 150, 8000, 'Sweet Industries', 3),
            ('SUGAR002', 'Brown Sugar Organic', 'Sweeteners', 65.00, 80, 3000, 'Natural Sweet', 5),
            ('HONEY001', 'Raw Honey Multiflora', 'Sweeteners', 320.00, 25, 600, 'Bee Happy', 7),
            ('JAGGER001', 'Jaggery Blocks', 'Sweeteners', 55.00, 60, 2000, 'Traditional Sweet', 4),
            
            # Spices & Seasonings
            ('SPICE001', 'Turmeric Powder', 'Spices', 120.00, 25, 500, 'Spice Master', 8),
            ('SPICE002', 'Red Chili Powder', 'Spices', 180.00, 30, 600, 'Hot Spices Ltd', 6),
            ('SPICE003', 'Cumin Seeds', 'Spices', 350.00, 15, 300, 'Aromatic Spices', 10),
            ('SPICE004', 'Coriander Seeds', 'Spices', 280.00, 20, 400, 'Fresh Spices', 7),
            ('SPICE005', 'Black Pepper Whole', 'Spices', 680.00, 10, 200, 'Premium Spices', 12),
            ('SPICE006', 'Cardamom Green', 'Spices', 1200.00, 5, 100, 'Royal Spices', 15),
            
            # Beverages
            ('TEA001', 'Green Tea Leaves', 'Beverages', 250.00, 30, 1000, 'Tea Gardens', 8),
            ('TEA002', 'Black Tea Premium', 'Beverages', 180.00, 50, 1500, 'Darjeeling Tea', 6),
            ('COFFEE001', 'Arabica Coffee Beans', 'Beverages', 480.00, 20, 500, 'Coffee Roasters', 10),
            ('COFFEE002', 'Instant Coffee', 'Beverages', 220.00, 40, 800, 'Quick Brew', 5),
            
            # Seasonings & Additives
            ('SALT001', 'Sea Salt Natural', 'Seasonings', 25.00, 200, 5000, 'Ocean Salt', 3),
            ('SALT002', 'Rock Salt Pink', 'Seasonings', 45.00, 100, 2000, 'Himalayan Salt', 5),
            ('VINEGAR001', 'Apple Cider Vinegar', 'Seasonings', 85.00, 30, 600, 'Health Vinegar', 7),
            
            # Nuts & Dry Fruits
            ('NUTS001', 'Almonds Premium', 'Nuts', 650.00, 20, 400, 'Nutty Delights', 8),
            ('NUTS002', 'Cashews Whole', 'Nuts', 580.00, 25, 500, 'Cashew King', 6),
            ('NUTS003', 'Walnuts Halves', 'Nuts', 720.00, 15, 300, 'Brain Food', 10),
            ('RAISINS001', 'Golden Raisins', 'Nuts', 380.00, 30, 600, 'Dried Fruits Co', 5),
            
            # Frozen & Preserved
            ('FROZEN001', 'Mixed Vegetables', 'Frozen', 85.00, 40, 800, 'Freeze Fresh', 2),
            ('FROZEN002', 'Green Peas Frozen', 'Frozen', 65.00, 60, 1200, 'Garden Frozen', 2),
            ('PICKLE001', 'Mango Pickle Traditional', 'Preserved', 120.00, 25, 500, 'Grandma Recipe', 12),
        ]
        
        # Insert products
        product_ids = {}
        for sku, name, category, price, min_stock, max_stock, supplier, lead_time in sample_products:
            try:
                cursor.execute('''
                    INSERT INTO products (sku, name, category, unit_price, min_stock_level, max_stock_level, supplier, lead_time_days)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (sku, name, category, price, min_stock, max_stock, supplier, lead_time))
                
                product_id = cursor.lastrowid
                product_ids[sku] = product_id
                
                # Initialize inventory with realistic stock levels
                initial_stock = random.randint(min_stock * 2, max_stock // 2)
                cursor.execute('''
                    INSERT INTO inventory (product_id, quantity, location, zone)
                    VALUES (?, ?, ?, ?)
                ''', (product_id, initial_stock, 'MAIN', random.choice(['A', 'B', 'C', 'D'])))
                
                print(f"   ✅ Added {name} with {initial_stock} units")
                
            except sqlite3.IntegrityError:
                print(f"   ⚠️ Product {sku} already exists, skipping...")
                # Get existing product ID
                cursor.execute('SELECT id FROM products WHERE sku = ?', (sku,))
                result = cursor.fetchone()
                if result:
                    product_ids[sku] = result[0]
        
        # Create realistic transaction history for the past 90 days
        print("\n📊 Generating transaction history for AI training...")
        
        transaction_types = ['IN', 'OUT', 'ADJUSTMENT']
        locations = ['MAIN', 'WAREHOUSE_A', 'WAREHOUSE_B']
        
        for product_id in product_ids.values():
            # Get product details for realistic transaction generation
            cursor.execute('SELECT min_stock_level, max_stock_level, unit_price FROM products WHERE id = ?', (product_id,))
            min_stock, max_stock, unit_price = cursor.fetchone()
            
            # Generate transactions for past 90 days
            for day in range(90):
                date = datetime.now() - timedelta(days=day)
                
                # Create 1-5 transactions per product per day (more realistic)
                num_transactions = random.choices([0, 1, 2, 3], weights=[30, 40, 20, 10])[0]
                
                for _ in range(num_transactions):
                    transaction_type = random.choices(transaction_types, weights=[20, 70, 10])[0]
                    
                    if transaction_type == 'IN':
                        # Stock receiving - larger quantities
                        quantity = random.randint(min_stock, max_stock // 4)
                        price = unit_price * random.uniform(0.95, 1.05)  # Price variation
                    elif transaction_type == 'OUT':
                        # Sales/consumption - smaller quantities
                        quantity = random.randint(1, min_stock)
                        price = unit_price * random.uniform(1.0, 1.1)  # Selling price
                    else:  # ADJUSTMENT
                        # Stock adjustments
                        quantity = random.randint(-50, 50)
                        price = unit_price
                    
                    # Add some seasonality and day-of-week patterns
                    if date.weekday() in [5, 6]:  # Weekend - less activity
                        quantity = int(quantity * 0.7)
                    
                    if date.month in [11, 12]:  # Festival season - more activity
                        quantity = int(quantity * 1.3)
                    
                    cursor.execute('''
                        INSERT INTO transactions (product_id, transaction_type, quantity, price, 
                                                location, timestamp, reference_number, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        product_id, 
                        transaction_type, 
                        abs(quantity), 
                        price,
                        random.choice(locations),
                        date.strftime('%Y-%m-%d %H:%M:%S'),
                        f"REF{random.randint(10000, 99999)}",
                        f"Auto-generated sample transaction"
                    ))
        
        # Create some alerts based on current stock levels
        print("\n🚨 Generating sample alerts...")
        
        cursor.execute('''
            SELECT p.id, p.name, p.sku, i.quantity, p.min_stock_level, p.max_stock_level
            FROM products p
            JOIN inventory i ON p.id = i.product_id
        ''')
        
        for product_id, name, sku, quantity, min_stock, max_stock in cursor.fetchall():
            if quantity <= min_stock * 0.5:  # Very low stock
                cursor.execute('''
                    INSERT INTO alerts (product_id, alert_type, message, severity)
                    VALUES (?, ?, ?, ?)
                ''', (product_id, 'LOW_STOCK', f"URGENT: {name} ({sku}) critically low - only {quantity} units left!", 'HIGH'))
                
            elif quantity <= min_stock:  # Low stock
                cursor.execute('''
                    INSERT INTO alerts (product_id, alert_type, message, severity)
                    VALUES (?, ?, ?, ?)
                ''', (product_id, 'LOW_STOCK', f"Low stock: {name} ({sku}) has {quantity} units (min: {min_stock})", 'MEDIUM'))
                
            elif quantity >= max_stock * 0.9:  # Near overstock
                cursor.execute('''
                    INSERT INTO alerts (product_id, alert_type, message, severity)
                    VALUES (?, ?, ?, ?)
                ''', (product_id, 'OVERSTOCK', f"High inventory: {name} ({sku}) has {quantity} units (approaching max: {max_stock})", 'LOW'))
        
        # Generate some sample forecasts
        print("\n🔮 Creating sample forecast data...")
        
        forecast_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 31)]
        
        for product_id in list(product_ids.values())[:10]:  # Only for first 10 products
            for date in forecast_dates:
                predicted_demand = random.randint(5, 50)
                confidence = random.uniform(0.6, 0.95)
                
                cursor.execute('''
                    INSERT INTO forecasts (product_id, forecast_date, predicted_demand, confidence_score)
                    VALUES (?, ?, ?, ?)
                ''', (product_id, date, predicted_demand, confidence))
        
        # Create model performance records
        print("\n📈 Adding model performance tracking...")
        
        for product_id in list(product_ids.values())[:15]:
            cursor.execute('''
                INSERT INTO model_performance (model_type, product_id, accuracy_score, mae_score, data_points)
                VALUES (?, ?, ?, ?, ?)
            ''', ('demand_forecast', product_id, random.uniform(0.7, 0.95), random.uniform(2.5, 8.0), random.randint(30, 90)))
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ Database setup complete!")
        print(f"📦 Added {len(sample_products)} products across multiple categories")
        print(f"📊 Generated ~{len(product_ids) * 90 * 1.5:.0f} historical transactions")
        print(f"🚨 Created realistic alerts based on stock levels")
        print(f"🔮 Added 30-day forecasts for top products")
        print(f"📈 Included model performance tracking")
        print(f"\n🎯 Your AI system now has rich data for training and recommendations!")
        
        return True

    def reset_database(self):
        """Reset database by deleting all data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tables = ['alerts', 'forecasts', 'model_performance', 'transactions', 'inventory', 'products']
        
        for table in tables:
            cursor.execute(f'DELETE FROM {table}')
            print(f"   🗑️ Cleared {table} table")
        
        conn.commit()
        conn.close()
        print("✅ Database reset complete")

    def get_database_stats(self):
        """Get current database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        tables = ['products', 'inventory', 'transactions', 'alerts', 'forecasts', 'model_performance']
        
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            stats[table] = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(quantity * unit_price) FROM inventory i JOIN products p ON i.product_id = p.id')
        total_value = cursor.fetchone()[0] or 0
        stats['total_inventory_value'] = total_value
        
        conn.close()
        return stats

if __name__ == '__main__':
    print("🚀 AI Warehouse Database Setup & Sample Data Generator")
    print("=" * 60)
    
    db_setup = DatabaseSetup()
    
    # Show current stats
    print("\n📊 Current Database Status:")
    try:
        current_stats = db_setup.get_database_stats()
        for table, count in current_stats.items():
            if table != 'total_inventory_value':
                print(f"   {table}: {count} records")
            else:
                print(f"   Total Inventory Value: ₹{count:,.2f}")
    except:
        print("   Database not initialized yet")
    
    print("\nOptions:")
    print("1. Create sample data (recommended for first run)")
    print("2. Reset database and create fresh sample data")
    print("3. Just show current stats")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        db_setup.create_sample_data()
    elif choice == '2':
        confirm = input("⚠️ This will delete all existing data. Continue? (yes/no): ").strip().lower()
        if confirm == 'yes':
            db_setup.reset_database()
            db_setup.create_sample_data()
        else:
            print("Operation cancelled")
    elif choice == '3':
        stats = db_setup.get_database_stats()
        print("\n📊 Current Database Statistics:")
        for table, count in stats.items():
            if table != 'total_inventory_value':
                print(f"   {table}: {count} records")
            else:
                print(f"   Total Inventory Value: ₹{count:,.2f}")
    else:
        print("Invalid choice")
    
    print("\n🎉 Ready to run your AI Warehouse Management System!")
    print("💡 Run 'python app.py' to start the application")