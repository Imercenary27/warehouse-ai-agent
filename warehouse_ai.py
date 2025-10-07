"""
AI Warehouse Management System - Core AI Logic
Author: AI Warehouse Management System
Date: October 2025

This module contains the main AI warehouse management class with machine learning
capabilities for demand forecasting, stock optimization, and automated alerts.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class WarehouseAI:
    """
    AI-powered warehouse management system with machine learning capabilities.
    
    Features:
    - Demand forecasting using Random Forest
    - Automated stock alerts
    - Reorder recommendations
    - Inventory optimization
    - Transaction tracking
    """
    
    def __init__(self, db_path='./database/warehouse.db'):
        """Initialize the AI warehouse management system"""
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.demand_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=10
        )
        self.price_model = LinearRegression()
        self.setup_database()
    
    def setup_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Products table - core product information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                unit_price REAL NOT NULL,
                min_stock_level INTEGER DEFAULT 10,
                max_stock_level INTEGER DEFAULT 1000,
                supplier TEXT DEFAULT '',
                lead_time_days INTEGER DEFAULT 7,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Inventory table - current stock levels
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                quantity INTEGER NOT NULL DEFAULT 0,
                reserved_quantity INTEGER DEFAULT 0,
                location TEXT DEFAULT 'MAIN',
                zone TEXT DEFAULT 'A',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        # Transactions table - all stock movements
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                transaction_type TEXT NOT NULL CHECK (transaction_type IN ('IN', 'OUT', 'ADJUSTMENT', 'TRANSFER')),
                quantity INTEGER NOT NULL,
                price REAL,
                reference_number TEXT,
                location TEXT DEFAULT 'MAIN',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                user_id TEXT DEFAULT 'system',
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        # Forecasts table - AI predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                forecast_date DATE,
                predicted_demand INTEGER,
                confidence_score REAL,
                model_version TEXT DEFAULT 'v1.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        # Alerts table - automated notifications
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                alert_type TEXT NOT NULL CHECK (alert_type IN ('LOW_STOCK', 'OVERSTOCK', 'REORDER', 'EXPIRED', 'CRITICAL')),
                message TEXT NOT NULL,
                severity TEXT DEFAULT 'MEDIUM' CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                status TEXT DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'RESOLVED', 'DISMISSED')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        # Performance metrics table - model accuracy tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                product_id INTEGER,
                accuracy_score REAL,
                mae_score REAL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_points INTEGER,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def add_product(self, sku, name, category, unit_price, min_stock=10, max_stock=1000, supplier='', lead_time=7):
        """Add a new product to the system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO products (sku, name, category, unit_price, min_stock_level, max_stock_level, supplier, lead_time_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (sku, name, category, unit_price, min_stock, max_stock, supplier, lead_time))
            
            product_id = cursor.lastrowid
            
            # Initialize inventory record
            cursor.execute('''
                INSERT INTO inventory (product_id, quantity)
                VALUES (?, ?)
            ''', (product_id, 0))
            
            conn.commit()
            return {"success": True, "product_id": product_id, "message": f"Product {name} added successfully"}
            
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                return {"success": False, "error": f"SKU '{sku}' already exists"}
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Database error: {str(e)}"}
        finally:
            conn.close()
    
    def update_stock(self, product_id, quantity, transaction_type="ADJUSTMENT", price=None, reference_number=None, notes=None):
        """Update stock levels and record transaction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get current stock first
            cursor.execute('SELECT quantity FROM inventory WHERE product_id = ?', (product_id,))
            current_stock_result = cursor.fetchone()
            
            if not current_stock_result:
                return {"success": False, "error": "Product not found in inventory"}
            
            current_stock = current_stock_result[0]
            
            # Calculate new stock based on transaction type
            if transaction_type == "IN":
                new_quantity = current_stock + quantity
                cursor.execute('''
                    UPDATE inventory SET quantity = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE product_id = ?
                ''', (new_quantity, product_id))
                
            elif transaction_type == "OUT":
                if current_stock < quantity:
                    return {"success": False, "error": f"Insufficient stock. Available: {current_stock}, Requested: {quantity}"}
                new_quantity = current_stock - quantity
                cursor.execute('''
                    UPDATE inventory SET quantity = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE product_id = ?
                ''', (new_quantity, product_id))
                
            else:  # ADJUSTMENT or TRANSFER
                cursor.execute('''
                    UPDATE inventory SET quantity = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE product_id = ?
                ''', (quantity, product_id))
                new_quantity = quantity
            
            # Record transaction
            cursor.execute('''
                INSERT INTO transactions (product_id, transaction_type, quantity, price, reference_number, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (product_id, transaction_type, quantity, price, reference_number, notes))
            
            conn.commit()
            
            # Check for alerts after stock update
            self.check_stock_alerts(product_id)
            
            return {
                "success": True, 
                "message": f"Stock updated successfully. New quantity: {new_quantity}",
                "new_quantity": new_quantity
            }
            
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": f"Transaction failed: {str(e)}"}
        finally:
            conn.close()
    
    def check_stock_alerts(self, product_id):
        """Check and generate stock alerts for a product"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT p.name, p.sku, p.min_stock_level, p.max_stock_level, i.quantity
                FROM products p
                JOIN inventory i ON p.id = i.product_id
                WHERE p.id = ?
            ''', (product_id,))
            
            result = cursor.fetchone()
            if not result:
                return
                
            name, sku, min_stock, max_stock, current_quantity = result
            
            # Clear existing active alerts for this product
            cursor.execute('''
                UPDATE alerts SET status = 'RESOLVED', resolved_at = CURRENT_TIMESTAMP 
                WHERE product_id = ? AND status = 'ACTIVE'
            ''', (product_id,))
            
            # Check for critical stock level (0 stock)
            if current_quantity == 0:
                cursor.execute('''
                    INSERT INTO alerts (product_id, alert_type, message, severity)
                    VALUES (?, 'CRITICAL', ?, 'CRITICAL')
                ''', (product_id, f"CRITICAL: {name} ({sku}) is out of stock!"))
                
            # Check for low stock
            elif current_quantity <= min_stock:
                urgency = "HIGH" if current_quantity <= min_stock * 0.5 else "MEDIUM"
                cursor.execute('''
                    INSERT INTO alerts (product_id, alert_type, message, severity)
                    VALUES (?, 'LOW_STOCK', ?, ?)
                ''', (product_id, f"Low stock alert: {name} ({sku}) has only {current_quantity} units left (min: {min_stock})", urgency))
                
                # Also create reorder recommendation
                cursor.execute('''
                    INSERT INTO alerts (product_id, alert_type, message, severity)
                    VALUES (?, 'REORDER', ?, 'MEDIUM')
                ''', (product_id, f"Reorder recommended for {name} ({sku}). Consider ordering {max_stock - current_quantity} units"))
            
            # Check for overstock
            elif current_quantity >= max_stock:
                cursor.execute('''
                    INSERT INTO alerts (product_id, alert_type, message, severity)
                    VALUES (?, 'OVERSTOCK', ?, 'LOW')
                ''', (product_id, f"Overstock alert: {name} ({sku}) has {current_quantity} units (max recommended: {max_stock})"))
            
            conn.commit()
            
        except Exception as e:
            print(f"Error checking alerts: {e}")
        finally:
            conn.close()
    
    def predict_demand(self, product_id, days_ahead=30):
        """Predict demand for a product using machine learning"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get historical transaction data (outbound transactions only)
            df = pd.read_sql_query('''
                SELECT DATE(timestamp) as date, SUM(quantity) as daily_demand
                FROM transactions 
                WHERE product_id = ? AND transaction_type = 'OUT'
                AND timestamp >= date('now', '-90 days')
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp)
            ''', conn, params=(product_id,))
            
            if len(df) < 7:  # Need at least a week of data
                return {
                    "error": "Insufficient historical data for prediction. Need at least 7 days of transaction history.",
                    "data_points": len(df)
                }
            
            # Prepare features for machine learning
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Create lag features for better prediction
            df['demand_lag_1'] = df['daily_demand'].shift(1)
            df['demand_lag_7'] = df['daily_demand'].shift(7)
            df['demand_rolling_7'] = df['daily_demand'].rolling(window=7, min_periods=1).mean()
            
            # Remove rows with NaN values
            df = df.dropna()
            
            if len(df) < 5:
                return {"error": "Insufficient data after preprocessing"}
            
            # Features for training
            feature_columns = ['day_of_week', 'day_of_month', 'month', 'is_weekend', 'demand_lag_1', 'demand_rolling_7']
            X = df[feature_columns].values
            y = df['daily_demand'].values
            
            # Train the model
            self.demand_model.fit(X, y)
            
            # Calculate model accuracy
            predictions = self.demand_model.predict(X)
            mae = mean_absolute_error(y, predictions)
            accuracy = max(0, 1 - (mae / (np.mean(y) + 1e-6)))
            
            # Store model performance
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_performance (model_type, product_id, accuracy_score, mae_score, data_points)
                VALUES (?, ?, ?, ?, ?)
            ''', ('demand_forecast', product_id, accuracy, mae, len(df)))
            
            # Generate future predictions
            future_predictions = []
            future_dates = []
            
            last_date = df['date'].max()
            last_demand = df['daily_demand'].iloc[-1]
            rolling_avg = df['demand_rolling_7'].iloc[-1]
            
            for i in range(1, days_ahead + 1):
                future_date = last_date + timedelta(days=i)
                future_dates.append(future_date)
                
                # Create features for future prediction
                future_features = [
                    future_date.dayofweek,
                    future_date.day,
                    future_date.month,
                    1 if future_date.dayofweek in [5, 6] else 0,
                    last_demand,  # Use last known demand as lag feature
                    rolling_avg   # Use rolling average
                ]
                
                pred = self.demand_model.predict([future_features])[0]
                pred = max(0, int(pred))  # Ensure non-negative integer
                future_predictions.append(pred)
                
                # Update lag features for next prediction
                last_demand = pred
                if len(future_predictions) >= 7:
                    rolling_avg = np.mean(future_predictions[-7:])
            
            # Store predictions in database
            for date, pred in zip(future_dates, future_predictions):
                cursor.execute('''
                    INSERT OR REPLACE INTO forecasts 
                    (product_id, forecast_date, predicted_demand, confidence_score, model_version)
                    VALUES (?, ?, ?, ?, ?)
                ''', (product_id, date.strftime('%Y-%m-%d'), pred, accuracy, 'v1.1'))
            
            conn.commit()
            
            return {
                "success": True,
                "predictions": [
                    {"date": date.strftime('%Y-%m-%d'), "predicted_demand": pred}
                    for date, pred in zip(future_dates, future_predictions)
                ],
                "total_predicted_demand": sum(future_predictions),
                "average_daily_demand": np.mean(future_predictions),
                "model_accuracy": round(accuracy * 100, 2),
                "mae_score": round(mae, 2),
                "training_data_points": len(df)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
        finally:
            conn.close()
    
    def get_reorder_recommendations(self):
        """Get AI-powered reorder recommendations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get products that need reordering
            cursor.execute('''
                SELECT p.id, p.name, p.sku, p.category, i.quantity, p.min_stock_level, 
                       p.max_stock_level, p.lead_time_days, p.unit_price
                FROM products p
                JOIN inventory i ON p.id = i.product_id
                WHERE i.quantity <= p.min_stock_level * 1.5
                ORDER BY (i.quantity / CAST(p.min_stock_level AS FLOAT)) ASC
            ''')
            
            recommendations = []
            for row in cursor.fetchall():
                product_id, name, sku, category, current_qty, min_stock, max_stock, lead_time, unit_price = row
                
                # Get demand forecast for this product
                forecast = self.predict_demand(product_id, 30)
                
                # Calculate recommended order quantity
                if "error" not in forecast:
                    predicted_30_day = forecast.get("total_predicted_demand", 0)
                    avg_daily_demand = forecast.get("average_daily_demand", 1)
                    
                    # Safety stock = lead time demand + buffer
                    safety_stock = int(avg_daily_demand * lead_time * 1.5)
                    
                    # Recommended order = max stock - current + predicted consumption during lead time
                    lead_time_demand = int(avg_daily_demand * lead_time)
                    recommended_order = max(0, max_stock - current_qty + lead_time_demand)
                    
                    # Calculate urgency based on stock level
                    stock_ratio = current_qty / max(min_stock, 1)
                    if current_qty == 0:
                        urgency = "CRITICAL"
                        priority = 1
                    elif stock_ratio <= 0.5:
                        urgency = "HIGH"
                        priority = 2
                    elif stock_ratio <= 1.0:
                        urgency = "MEDIUM" 
                        priority = 3
                    else:
                        urgency = "LOW"
                        priority = 4
                    
                    recommendations.append({
                        "product_id": product_id,
                        "sku": sku,
                        "name": name,
                        "category": category,
                        "current_quantity": current_qty,
                        "min_stock_level": min_stock,
                        "recommended_order_quantity": recommended_order,
                        "predicted_30_day_demand": predicted_30_day,
                        "avg_daily_demand": round(avg_daily_demand, 2),
                        "safety_stock": safety_stock,
                        "lead_time_days": lead_time,
                        "urgency": urgency,
                        "priority": priority,
                        "estimated_cost": round(recommended_order * unit_price, 2),
                        "stock_ratio": round(stock_ratio, 2),
                        "days_until_stockout": max(1, int(current_qty / max(avg_daily_demand, 0.1)))
                    })
                else:
                    # Fallback recommendation without forecast
                    recommended_order = max_stock - current_qty
                    urgency = "HIGH" if current_qty == 0 else "MEDIUM"
                    
                    recommendations.append({
                        "product_id": product_id,
                        "sku": sku,
                        "name": name,
                        "category": category,
                        "current_quantity": current_qty,
                        "min_stock_level": min_stock,
                        "recommended_order_quantity": recommended_order,
                        "predicted_30_day_demand": "Insufficient data",
                        "urgency": urgency,
                        "priority": 2 if urgency == "HIGH" else 3,
                        "estimated_cost": round(recommended_order * unit_price, 2),
                        "note": "Recommendation based on min/max levels only"
                    })
            
            # Sort by priority (most urgent first)
            recommendations.sort(key=lambda x: x.get('priority', 5))
            
            return recommendations
            
        except Exception as e:
            return {"error": f"Failed to generate recommendations: {str(e)}"}
        finally:
            conn.close()
    
    def get_inventory_summary(self):
        """Get comprehensive inventory summary"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            summary = pd.read_sql_query('''
                SELECT 
                    p.category,
                    COUNT(*) as product_count,
                    SUM(i.quantity) as total_units,
                    SUM(i.quantity * p.unit_price) as total_value,
                    AVG(i.quantity) as avg_stock_level,
                    SUM(CASE WHEN i.quantity <= p.min_stock_level THEN 1 ELSE 0 END) as low_stock_count
                FROM products p
                JOIN inventory i ON p.id = i.product_id
                GROUP BY p.category
                ORDER BY total_value DESC
            ''', conn)
            
            return summary.to_dict('records')
            
        except Exception as e:
            return {"error": f"Failed to generate summary: {str(e)}"}
        finally:
            conn.close()
    
    def get_product_performance(self, days=30):
        """Get product performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            performance = pd.read_sql_query('''
                SELECT 
                    p.sku,
                    p.name,
                    p.category,
                    SUM(CASE WHEN t.transaction_type = 'OUT' THEN t.quantity ELSE 0 END) as units_sold,
                    SUM(CASE WHEN t.transaction_type = 'OUT' THEN t.quantity * COALESCE(t.price, p.unit_price) ELSE 0 END) as revenue,
                    COUNT(CASE WHEN t.transaction_type = 'OUT' THEN 1 END) as transaction_count,
                    i.quantity as current_stock
                FROM products p
                LEFT JOIN transactions t ON p.id = t.product_id 
                    AND t.timestamp >= date('now', '-{} days')
                JOIN inventory i ON p.id = i.product_id
                GROUP BY p.id, p.sku, p.name, p.category, i.quantity
                ORDER BY units_sold DESC
            '''.format(days), conn)
            
            return performance.to_dict('records')
            
        except Exception as e:
            return {"error": f"Failed to get performance data: {str(e)}"}
        finally:
            conn.close()
    
    def cleanup_old_data(self, days_to_keep=365):
        """Clean up old transactions and forecasts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete old transactions
            cursor.execute('''
                DELETE FROM transactions 
                WHERE timestamp < date('now', '-{} days')
            '''.format(days_to_keep))
            
            # Delete old forecasts
            cursor.execute('''
                DELETE FROM forecasts 
                WHERE created_at < date('now', '-{} days')
            '''.format(days_to_keep // 2))
            
            # Delete resolved alerts older than 30 days
            cursor.execute('''
                DELETE FROM alerts 
                WHERE status = 'RESOLVED' AND resolved_at < date('now', '-30 days')
            ''')
            
            conn.commit()
            return {"success": True, "message": "Old data cleaned up successfully"}
            
        except Exception as e:
            return {"error": f"Cleanup failed: {str(e)}"}
        finally:
            conn.close()