# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)  # Mobile app එකට connect වෙන්න allow කරනවා

# MongoDB Connection
client = MongoClient(os.getenv("MONGODB_URI"))
db = client.get_database()
collection = db.chamber_status

LATEST_DOC_ID = "latest_chamber"

@app.route('/api/chamber', methods=['GET'])
def get_chamber_status():
    doc = collection.find_one({"_id": LATEST_DOC_ID})
    
    if not doc:
        default = {
            "fan_status": "OFF",
            "heater_status": "OFF",
            "ml_decision": "NOT_DRY",
            "confidence": 0,
            "image_url": None,
            "timestamp": datetime.utcnow().isoformat()
        }
        return jsonify(default)
    
    doc.pop("_id", None)
    return jsonify(doc)

@app.route('/api/chamber', methods=['PATCH'])
def update_chamber_status():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    allowed = ["fan_status", "heater_status"]
    update_data = {k: v.upper() for k, v in data.items() if k in allowed}
    if not update_data:
        return jsonify({"error": "Invalid fields"}), 400
    
    update_data["timestamp"] = datetime.utcnow().isoformat()
    
    collection.update_one(
        {"_id": LATEST_DOC_ID},
        {"$set": update_data},
        upsert=True
    )
    
    return jsonify({"success": True})

@app.route('/api/chamber/update', methods=['POST'])
def pi_update():
    data = request.get_json()
    required = ["ml_decision", "confidence", "image_url"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400
    
    update_data = {
        "ml_decision": data["ml_decision"].upper(),
        "confidence": float(data["confidence"]),
        "image_url": data["image_url"],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    current = collection.find_one({"_id": LATEST_DOC_ID})
    if current:
        update_data["fan_status"] = current.get("fan_status", "OFF")
        update_data["heater_status"] = current.get("heater_status", "OFF")
    
    collection.update_one(
        {"_id": LATEST_DOC_ID},
        {"$set": update_data},
        upsert=True
    )
    
    return jsonify({"success": True})

@app.route('/')
def home():
    return "Bark Drying Backend Running! 🌿"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)