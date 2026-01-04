# Bark Drying Chamber Backend

Python Flask + MongoDB backend for Cinnamon Bark Drying Monitoring System.

## Features
- Manual Fan & Heater control from mobile app
- ML decision (DRY / NOT_DRY) from Raspberry Pi
- Latest bark image URL
- Real-time status

## API Endpoints
- GET  /api/chamber          → Get latest status
- PATCH /api/chamber         → Update fan/heater (from app)
- POST /api/chamber/update   → Send ML result + image (from Pi)

## Run
```bash
pip install -r requirements.txt
python app.py