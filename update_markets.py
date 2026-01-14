import requests
import json
import os
from datetime import datetime

# --- CONFIGURATION ---
MARKET_SLUG = "israel-strikes-iran-by-january-31-2026"
API_URL = "https://gamma-api.polymarket.com/events"
STATE_FILE = "bot_state.json"  # This file stores the bot's "brain"

class PredictionEngine:
    def __init__(self):
        self.state = self.load_state()

    def load_state(self):
        """Loads the hidden states (memory) from the previous run."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass # If file is corrupted, start over
        
        # DEFAULT STATE (Epoch 0)
        return {
            "epoch": 0,
            "history": [], 
            "weights": {
                "momentum": 0.3,       # Short-term trend sensitivity
                "learning_bias": 0.0   # Long-term correction (starts neutral)
            },
            "last_prediction": None
        }

    def save_state(self):
        """Saves the learned weights back to the repo."""
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def fetch_data(self):
        """Gets live data from Polymarket."""
        try:
            params = {"slug": MARKET_SLUG}
            response = requests.get(API_URL, params=params)
            data = response.json()
            if not data or not data.get('markets'): return None
            
            market = data['markets']
            prices = json.loads(market['outcomePrices'])
            
            return {
                "question": market['question'],
                "price": float(prices), # "Yes" Price
                "volume": float(data.get('volume', 0))
            }
        except Exception as e:
            print(f"API Error: {e}")
            return None

    def run_algorithm(self, live_data):
        current_price = live_data['price']
        
        # --- 1. LONG-TERM LEARNING (Error Correction) ---
        # If we made a prediction last time, check how wrong we were
        if self.state['last_prediction'] is not None:
            # Error = Real Price - Our Last Prediction
            error = current_price - self.state['last_prediction']
            
            # Update Bias: If we were too high, lower the bias. If too low, raise it.
            # 0.1 is the 'Learning Rate'
            self.state['weights']['learning_bias'] += (error * 0.1)

        # --- 2. SHORT-TERM ALGO (Momentum) ---
        # Compare current price to the average of the last 5 runs
        history = self.state['history'][-5:]
        if history:
            avg_price = sum([h['price'] for h in history]) / len(history)
            momentum_score = (current_price - avg_price) * self.state['weights']['momentum']
        else:
            momentum_score = 0

        # --- 3. FINAL PREDICTION ---
        # Base Price + Momentum + Learned Bias
        smart_pred = current_price + momentum_score + self.state['weights']['learning_bias']
        
        # Clamp result between 1% and 99%
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
        
        # Keep history short (last 20 items) to save space
        self.state['history'] = self.state['history'][-20:]
        
        return smart_pred

    def generate_html(self, data, prediction):
        """Creates the visual dashboard."""
        
        # Trend Indicator
        prev_pred = self.state['history'][-2]['pred'] if len(self.state['history']) > 1 else prediction
        trend = "UP" if prediction > prev_pred else "DOWN"
        color = "#00ff00" if prediction > prev_pred else "#ff4444"

        # Table Rows
        rows = ""
        for h in reversed(self.state['history']):
            rows += f"<tr><td>{h['time']}</td><td>{h['price']:.3f}</td><td>{h['pred']:.3f}</td><td>{h['bias']:.4f}</td></tr>"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Algorithmic Forecaster</title>
            <meta http-equiv="refresh" content="300">
            <style>
                body {{ background: #111; color: #eee; font-family: sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }}
                h1 {{ border-bottom: 2px solid #555; }}
               .card {{ background: #222; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
               .big-num {{ font-size: 3em; font-weight: bold; color: {color}; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
                th, td {{ border-bottom: 1px solid #333; padding: 8px; text-align: left; }}
                th {{ color: #888; }}
            </style>
        </head>
        <body>
            <h1>Market Prediction Bot</h1>
            <p>Target: {data['question']}</p>
            
            <div class="card">
                <div>SMART PREDICTION ({trend})</div>
                <div class="big-num">{prediction:.1%}</div>
                <div style="font-size: 0.8em; color: #888;">
                    Raw Market Price: {data['price']:.1%} | 
                    Learned Bias: {self.state['weights']['learning_bias']:.4f}
                </div>
            </div>

            <div class="card">
                <h3>Algorithm Memory (Last 20 Epochs)</h3>
                <table>
                    <tr><th>Time (UTC)</th><th>Real Price</th><th>Bot Prediction</th><th>Internal Bias</th></tr>
                    {rows}
                </table>
            </div>
            
            <p style="text-align:center; font-size: 0.7em; color: #555;">
                Epoch: {self.state['epoch']} | Powered by GitHub Actions
            </p>
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
        print(f"Epoch {bot.state['epoch']} Complete. Prediction: {prediction}")
    else:
        print("Failed to fetch data.")