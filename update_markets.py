import requests
import json
import os
from datetime import datetime

# --- CONFIGURATION ---
MARKET_SLUG = "israel-strikes-iran-by-january-31-2026"
API_URL = "https://gamma-api.polymarket.com/events"
STATE_FILE = "bot_state.json"

class PredictionEngine:
    def __init__(self):
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {
            "epoch": 0,
            "history": [],
            "weights": {"momentum": 0.3, "learning_bias": 0.0},
            "last_prediction": None
        }

    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def fetch_data(self):
        """Fetches live data from Polymarket."""
        try:
            params = {"slug": MARKET_SLUG}
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # --- FIX IS HERE ---
            # The API returns a LIST of events: [ {event_data} ]
            if isinstance(data, list) and len(data) > 0:
                event = data # Grab the first event
                market = event['markets'] # Grab the first market within the event
                
                # Prices are stored as a string list: "[\"0.53\", \"0.47\"]"
                prices = json.loads(market['outcomePrices'])
                
                return {
                    "question": market['question'],
                    "price": float(prices), # Price of "Yes"
                    "volume": float(market.get('volume', 0))
                }
            else:
                print(f"API returned empty data for slug: {MARKET_SLUG}")
                return None
        except Exception as e:
            print(f"API Error details: {e}")
            return None

    def run_algorithm(self, live_data):
        current_price = live_data['price']
        
        # 1. Error Correction (Learning)
        if self.state['last_prediction'] is not None:
            error = current_price - self.state['last_prediction']
            self.state['weights']['learning_bias'] += (error * 0.1)

        # 2. Momentum
        history = self.state['history'][-5:]
        if history:
            avg_price = sum([h['price'] for h in history]) / len(history)
            momentum = (current_price - avg_price) * self.state['weights']['momentum']
        else:
            momentum = 0

        # 3. Prediction
        smart_pred = current_price + momentum + self.state['weights']['learning_bias']
        smart_pred = max(0.01, min(0.99, smart_pred))

        # Update State
        self.state['epoch'] += 1
        self.state['last_prediction'] = smart_pred
        self.state['history'].append({
            "time": datetime.utcnow().strftime("%H:%M"),
            "price": current_price,
            "pred": smart_pred,
            "bias": self.state['weights']['learning_bias']
        })
        self.state['history'] = self.state['history'][-20:]
        
        return smart_pred

    def generate_html(self, data, prediction):
        prev_pred = self.state['history'][-2]['pred'] if len(self.state['history']) > 1 else prediction
        color = "#00ff00" if prediction >= prev_pred else "#ff4444"
        rows = ""
        for h in reversed(self.state['history']):
            rows += f"<tr><td>{h['time']}</td><td>{h['price']:.3f}</td><td>{h['pred']:.3f}</td><td>{h['bias']:.4f}</td></tr>"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Bot</title>
            <meta http-equiv="refresh" content="300">
            <style>
                body {{ background: #111; color: #eee; font-family: monospace; padding: 20px; max-width: 800px; margin: 0 auto; }}
                h1 {{ border-bottom: 2px solid #555; }}
              .card {{ background: #222; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
              .big {{ font-size: 3em; font-weight: bold; color: {color}; }}
                table {{ width: 100%; border-collapse: collapse; }}
                td, th {{ border-bottom: 1px solid #333; padding: 8px; text-align: left; }}
            </style>
        </head>
        <body>
            <h1>PREDICTION BOT // LIVE</h1>
            <p>Target: {data['question']}</p>
            
            <div class="card">
                <div>SMART PREDICTION</div>
                <div class="big">{prediction:.1%}</div>
                <p>Market Price: {data['price']:.1%} | Bias: {self.state['weights']['learning_bias']:.4f}</p>
            </div>

            <div class="card">
                <h3>LEARNING LOG (LAST 20 EPOCHS)</h3>
                <table>
                    <tr><th>Time</th><th>Market</th><th>Bot Pred</th><th>Bias</th></tr>
                    {rows}
                </table>
            </div>
            <p style="text-align:center; color: #555;">Epoch: {self.state['epoch']}</p>
        </body>
        </html>
        """
        with open("index.html", "w") as f:
            f.write(html)

if __name__ == "__main__":
    bot = PredictionEngine()
    data = bot.fetch_data()
    
    if data:
        prediction = bot.run_algorithm(data)
        bot.generate_html(data, prediction)
        bot.save_state()
        print(f"Success! Prediction: {prediction}")
    else:
        print("Failed to fetch data.")