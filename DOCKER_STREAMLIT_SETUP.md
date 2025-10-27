# 🐳 Docker + Streamlit Setup Guide

This guide shows you how to run your Premier League Match Predictor using Docker with a beautiful Streamlit interface.

## 📋 Prerequisites

- Docker installed on your system
- Docker Compose installed
- Your trained model files in the `models/` directory

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and run both services:**
   ```bash
   docker-compose up --build
   ```

2. **Access the applications:**
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs

### Option 2: Manual Docker Setup

1. **Build the API container:**
   ```bash
   docker build -t premier-league-api .
   ```

2. **Run the API:**
   ```bash
   docker run -p 8000:8000 \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/data:/app/data \
     premier-league-api
   ```

3. **Build the Streamlit container:**
   ```bash
   docker build -f Dockerfile.streamlit -t premier-league-ui .
   ```

4. **Run Streamlit:**
   ```bash
   docker run -p 8501:8501 \
     -e API_URL=http://host.docker.internal:8000 \
     premier-league-ui
   ```

## 🎨 Streamlit Features

Your Streamlit interface includes:

- **🏠 Team Selection**: Choose home and away teams from dropdown
- **📅 Date Picker**: Select match date
- **🎯 Real-time Predictions**: Get instant predictions from your ML model
- **📊 Interactive Charts**: Visualize probabilities and confidence
- **📱 Responsive Design**: Works on desktop and mobile
- **🔄 Live API Connection**: Real-time connection to your FastAPI backend

## 🛠️ Development Mode

For development with hot reload:

```bash
# Terminal 1: Run API
python api/main.py

# Terminal 2: Run Streamlit
cd streamlit
streamlit run app.py
```

## 🧪 Testing the Setup

1. **Check API health:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test prediction:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{
          "home_team": "Arsenal",
          "away_team": "Chelsea",
          "match_date": "2024-01-15"
        }'
   ```

3. **Visit Streamlit UI:**
   Open http://localhost:8501 in your browser

## 📁 Project Structure

```
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── predictor.py
│   ├── schemas.py
│   └── requirements.txt
├── streamlit/              # Streamlit frontend
│   ├── app.py
│   └── requirements.txt
├── models/                 # ML model files
│   ├── rf_model.pkl
│   ├── label_encoder.pkl
│   └── feature_names.json
├── data/                   # Training data
│   └── results.csv
├── Dockerfile              # API container
├── Dockerfile.streamlit    # UI container
└── docker-compose.yml      # Orchestration
```

## 🚀 Production Deployment

For production deployment:

1. **Use environment variables:**
   ```yaml
   environment:
     - API_URL=https://your-api-domain.com
   ```

2. **Add SSL/TLS:**
   ```yaml
   - "443:8000"  # HTTPS
   ```

3. **Configure resource limits:**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 512M
         cpus: '0.5'
   ```

## 🐛 Troubleshooting

**API not connecting:**
- Check if containers are running: `docker ps`
- Check logs: `docker-compose logs api`

**Model not loading:**
- Ensure model files exist in `models/` directory
- Check file permissions

**Port already in use:**
```bash
# Kill existing processes
docker-compose down
# Or change ports in docker-compose.yml
```

## 🎯 Next Steps

- Deploy to cloud platforms (AWS, GCP, Azure)
- Add authentication and user management
- Implement caching for better performance
- Add more visualization features
- Create mobile app using the API
