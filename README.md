# AI Warehouse Stock & Commodities Management System

A beginner-friendly AI-powered warehouse management prototype. Enables inventory tracking, demand forecasting, smart reorder recommendations, and real-time analytics via a Flask API and modern dashboard.

## 🚀 Features

- **Product Management**: Add, view, and update products with SKUs, categories, pricing, and stock levels.
- **Inventory Tracking**: Record stock IN/OUT/adjustments and view current quantities.
- **Demand Forecasting**: 30-day AI predictions using historical transaction data and Random Forest.
- **Reorder Recommendations**: Intelligent suggestions based on forecast, safety stock, and lead times.
- **Automated Alerts**: Low stock, overstock, and critical shortage notifications.
- **Analytics Dashboard**: Real-time KPIs, transaction trends, and product tables.

## 📂 Project Structure

```bash
twarehouse-ai/
├── app.py                 # Main Flask application and API routes
├── warehouse_ai.py        # Core AI logic and database schema
├── warehouse.db           # SQLite database (auto-generated)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── database/
│   ├── database_config.py # DB schema initializer
│   └── setup_database.py  # Sample data generator
└── templates/
    └── dashboard.html     # Web dashboard UI
```

## 🛠️ Prerequisites

- Python 3.8+
- pip

## ⚡ Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize database schema**
   ```bash
   python database/database_config.py
   ```

3. **Populate sample data**
   ```bash
   python database/setup_database.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Visit dashboard**
   Open your browser and go to: http://localhost:5000

6. **API health check**
   ```bash
   curl http://localhost:5000/health
   ```

## 🔎 API Endpoints

- **GET /**
  - Serves the dashboard HTML.
- **GET /api/products**
  - Fetch all products with current stock.
- **POST /api/products**
  - Add a new product. JSON body: `sku, name, category, unit_price, min_stock, max_stock`.
- **PUT /api/inventory/<product_id>**
  - Update stock. JSON body: `quantity, transaction_type, price`.
- **GET /api/forecast/<product_id>?days=30**
  - Generate demand forecast.
- **GET /api/alerts**
  - Retrieve active stock alerts.
- **GET /api/recommendations**
  - Get AI reorder suggestions.
- **GET /api/analytics**
  - Get dashboard statistics and recent transactions.
- **GET /api/transactions?limit=100**
  - Fetch recent transactions.
- **GET /api/categories**
  - List all product categories.
- **GET /health**
  - Health check endpoint.

## 🔧 Configuration

- **Database Path**: Default is `warehouse.db` in project root.
- **ML Parameters**: Adjust in `warehouse_ai.py` (e.g., `n_estimators`, model depth).
- **Port & Host**: Change `app.run()` arguments in `app.py` as needed.

## 🎯 Next Steps

- Swap SQLite for PostgreSQL and serve via Gunicorn/Nginx.
- Add authentication (JWT/OAuth2) and multi-user support.
- Implement caching (Redis) and async tasks (Celery) for heavy ML jobs.
- Enhance AI: LSTM/Prophet forecasting and external commodity price integration.
- Add unit tests (pytest) and CI/CD pipeline.

---

Made with ❤️ by Rajat
