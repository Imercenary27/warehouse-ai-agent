"""
AI Warehouse Stock & Commodities Management Agent
Main Flask Application
Author: AI Warehouse Management System
Date: October 2025
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Import our AI warehouse management system
from warehouse_ai import WarehouseAI

# Initialize the AI system
warehouse_ai = WarehouseAI()

@app.route('/')
def index():
    """Main dashboard route"""
    return render_template('dashboard.html')

@app.route('/api/products', methods=['GET', 'POST'])
def handle_products():
    """Handle product management - GET all products or POST new product"""
    if request.method == 'POST':
        try:
            data = request.json
            result = warehouse_ai.add_product(
                data['sku'], 
                data['name'], 
                data['category'],
                data['unit_price'], 
                data.get('min_stock', 10),
                data.get('max_stock', 1000)
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400
    
    # GET all products with current stock
    try:
        conn = sqlite3.connect(warehouse_ai.db_path)
        df = pd.read_sql_query('''
            SELECT p.*, i.quantity as current_stock
            FROM products p
            LEFT JOIN inventory i ON p.id = i.product_id
            ORDER BY p.created_at DESC
        ''', conn)
        conn.close()
        
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/inventory/<int:product_id>', methods=['PUT'])
def update_inventory(product_id):
    """Update inventory levels for a specific product"""
    try:
        data = request.json
        result = warehouse_ai.update_stock(
            product_id, 
            data['quantity'],
            data.get('transaction_type', 'ADJUSTMENT'),
            data.get('price')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/forecast/<int:product_id>')
def get_forecast(product_id):
    """Generate AI demand forecast for a product"""
    try:
        days_ahead = request.args.get('days', 30, type=int)
        result = warehouse_ai.predict_demand(product_id, days_ahead)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get all active alerts"""
    try:
        conn = sqlite3.connect(warehouse_ai.db_path)
        df = pd.read_sql_query('''
            SELECT a.*, p.name as product_name, p.sku
            FROM alerts a
            JOIN products p ON a.product_id = p.id
            WHERE a.status = 'ACTIVE'
            ORDER BY a.created_at DESC
            LIMIT 50
        ''', conn)
        conn.close()
        
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations')
def get_recommendations():
    """Get AI-powered reorder recommendations"""
    try:
        recommendations = warehouse_ai.get_reorder_recommendations()
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics')
def get_analytics():
    """Get dashboard analytics and statistics"""
    try:
        conn = sqlite3.connect(warehouse_ai.db_path)
        
        # Get summary statistics
        stats = {}
        cursor = conn.cursor()
        
        # Total products
        cursor.execute('SELECT COUNT(*) FROM products')
        stats['total_products'] = cursor.fetchone()[0]
        
        # Total inventory value
        cursor.execute('''
            SELECT COALESCE(SUM(p.unit_price * i.quantity), 0) as total_value
            FROM products p
            JOIN inventory i ON p.id = i.product_id
        ''')
        stats['total_inventory_value'] = cursor.fetchone()[0]
        
        # Low stock items
        cursor.execute('''
            SELECT COUNT(*) FROM products p
            JOIN inventory i ON p.id = i.product_id
            WHERE i.quantity <= p.min_stock_level
        ''')
        stats['low_stock_items'] = cursor.fetchone()[0]
        
        # Active alerts count
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE status = 'ACTIVE'")
        stats['active_alerts'] = cursor.fetchone()[0]
        
        # Recent transactions for chart
        df_transactions = pd.read_sql_query('''
            SELECT DATE(timestamp) as date, transaction_type, COUNT(*) as count
            FROM transactions
            WHERE timestamp >= date('now', '-30 days')
            GROUP BY DATE(timestamp), transaction_type
            ORDER BY date DESC
        ''', conn)
        
        conn.close()
        
        return jsonify({
            'stats': stats,
            'recent_transactions': df_transactions.to_dict('records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/transactions')
def get_transactions():
    """Get recent transactions"""
    try:
        limit = request.args.get('limit', 100, type=int)
        conn = sqlite3.connect(warehouse_ai.db_path)
        
        df = pd.read_sql_query('''
            SELECT t.*, p.name as product_name, p.sku
            FROM transactions t
            JOIN products p ON t.product_id = p.id
            ORDER BY t.timestamp DESC
            LIMIT ?
        ''', conn, params=(limit,))
        
        conn.close()
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/categories')
def get_categories():
    """Get all product categories"""
    try:
        conn = sqlite3.connect(warehouse_ai.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT category FROM products ORDER BY category')
        categories = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return jsonify(categories)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Initialize sample data on first run
    print("🚀 Starting AI Warehouse Management System...")
    print("📊 Setting up database and sample data...")
    
    # Add sample products for demonstration
    sample_products = [
        ('RICE001', 'Basmati Rice Premium', 'Grains', 45.50, 100, 5000),
        ('WHEAT002', 'Wheat Flour Organic', 'Grains', 35.00, 200, 10000),
        ('SUGAR003', 'Sugar White Refined', 'Sweeteners', 42.00, 150, 8000),
        ('OIL004', 'Cooking Oil Sunflower', 'Oils', 85.00, 50, 2000),
        ('PULSE005', 'Red Lentils Premium', 'Pulses', 75.00, 100, 3000),
        ('SPICE006', 'Turmeric Powder', 'Spices', 120.00, 25, 500),
        ('TEA007', 'Green Tea Leaves', 'Beverages', 250.00, 30, 1000),
        ('SALT008', 'Sea Salt Natural', 'Seasonings', 25.00, 200, 5000)
    ]
    
    # Check if products already exist
    conn = sqlite3.connect(warehouse_ai.db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM products')
    existing_products = cursor.fetchone()[0]
    conn.close()
    
    if existing_products == 0:
        print("📦 Adding sample products...")
        for sku, name, category, price, min_stock, max_stock in sample_products:
            result = warehouse_ai.add_product(sku, name, category, price, min_stock, max_stock)
            if result.get('success'):
                # Add some initial stock
                initial_stock = min_stock * 3  # Start with 3x minimum stock
                warehouse_ai.update_stock(result['product_id'], initial_stock, 'IN', price)
                print(f"   ✅ Added {name} with {initial_stock} units")
                
                # Add some sample transactions for forecasting
                import random
                for i in range(random.randint(5, 15)):
                    days_ago = random.randint(1, 30)
                    sale_qty = random.randint(5, min_stock // 2)
                    # Simulate past transactions
                    conn = sqlite3.connect(warehouse_ai.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO transactions (product_id, transaction_type, quantity, price, timestamp)
                        VALUES (?, 'OUT', ?, ?, datetime('now', '-{} days'))
                    '''.format(days_ago), (result['product_id'], sale_qty, price))
                    conn.commit()
                    conn.close()
    else:
        print(f"📦 Found {existing_products} existing products in database")
    
    print("\n🌐 Dashboard will be available at: http://localhost:5000")
    print("📚 API documentation at: http://localhost:5000/health")
    print("🤖 AI features ready for demand forecasting and recommendations!")
    print("\n" + "="*60)
    
    # Start the Flask application
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True
    )