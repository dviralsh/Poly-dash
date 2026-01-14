import requests
import json
import os
import math
from datetime import datetime

# --- CONFIGURATION ---
MARKET_SLUG = "israel-strikes-iran-by-january-31-2026"
API_URL = "https://gamma-api.polymarket.com/events"
STATE_FILE = "bot_memory.json" # Stores the "brain" of the bot

class MarketIntelligence:
    def __init__(self):
        self.state = self.load_state()
    
    def load_state(self):
        """Loads the hidden states (memory) from the previous run."""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        else:
            # Initialize fresh state (Epoch 0)
            return {
                "epoch": 0,
                "history": { 
                    "momentum": 0.5, 
                    "insider_sensitivity": 1.0, 
                    "adaptive_bias": 0.0 # This changes over time (Learning)
                },
                "last_prediction": None,
                "learning_rate": 0.05
            }

    def save_state(self):
        """Saves the current state back to the repo."""
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def fetch_live_data(self):
        """Fetches raw data from Polymarket."""
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
                "volume": float(data.get('volume', 0)),
                "liquidity": float(market.get('liquidity', 0))
            }
        except Exception as e:
            print(f"API Error: {e}")
            return None

    def calculate_smart_prediction(self, live_data):
        """
        The Core Algorithm:
        1. Short-Term: Momentum (Are we diverging from the 10-epoch average?)
        2. Heuristic: Insider Volatility (High volume + Low price move = Accumulation)
        3. Long-Term: Adaptive Bias (Self-correction from previous errors)
        """
        current_price = live_data['price']
        
        # A. LONG-TERM LEARNING (Self-Correction)
        # If we made a prediction last time, how wrong were we?
        if self.state['last_prediction'] is not None:
            # Error = Real Price - What we predicted
            error = current_price - self.state['last_prediction']
            
            # Update the bias weight (Gradient Descent-lite)
            # If we were too high, lower the bias. If too low, raise it.
            self.state['weights']['adaptive_bias'] += (error * self.state['learning_rate'])

        # B. SHORT-TERM MOMENTUM
        # Calculate Moving Average of last 10 runs
        history_prices = [h['price'] for h in self.state['history'][-10:]]
        if history_prices:
            avg_price = sum(history_prices) / len(history_prices)
            momentum = (current_price - avg_price) * self.state['weights']['momentum']
        else:
            momentum = 0

        # C. INSIDER HEURISTIC
        # Detect volume spikes relative to recent history
        insider_boost = 0.0
        if self.state['history']:
            last_vol = self.state['history'][-1]['volume']
            vol_change = (live_data['volume'] - last_vol) / (last_vol + 1)
            # If volume spiked > 5% but price moved < 2%, assume "Stealth Accumulation"
            if vol_change > 0.05 and abs(momentum) < 0.02:
                insider_boost = 0.05 # Add 5% probability

        # D. FINAL CALCULATION
        smart_prob = current_price + momentum + insider_boost + self.state['weights']['adaptive_bias']
        
        # Clamp result between 1% and 99%
        return max(0.01, min(0.99, smart_prob))

    def run_epoch(self):
        """Runs one cycle of the bot."""
        data = self.fetch_live_data()
        if not data: return
        
        # Generate Prediction
        smart_prob = self.calculate_smart_prediction(data)
        
        # Update Memory
        self.state['epoch'] += 1
        self.state['last_prediction'] = smart_prob
        self.state['history'].append({
            "timestamp": datetime.utcnow().strftime("%H:%M"),
            "price": data['price'],
            "volume": data['volume'],
            "predicted": smart_prob
        })
        
        # Keep history manageable (last 50 runs)
        self.state['history'] = self.state['history'][-50:]
        
        self.save_state()
        self.generate_dashboard(data, smart_prob)

    def generate_dashboard(self, data, smart_prob):
        """Writes the HTML dashboard."""
        
        # Determine Trend direction
        if len(self.state['history']) > 1:
            prev = self.state['history'][-2]['predicted']
            trend_arrow = "↑" if smart_prob > prev else "↓"
            trend_color = "#00ff00" if smart_prob > prev else "#ff0000"
        else:
            trend_arrow = "-"
            trend_color = "#cccccc"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Automated Intelligence Unit</title>
            <meta http-equiv="refresh" content="300"> 
            <style>
                body {{ background: #111; color: #eee; font-family: monospace; padding: 20px; }}
               .box {{ border: 1px solid #444; padding: 20px; margin: 20px 0; border-radius: 5px; }}
               .big-text {{ font-size: 3em; font-weight: bold; color: {trend_color}; }}
               .label {{ color: #888; font-size: 0.8em; text-transform: uppercase; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                td, th {{ border: 1px solid #333; padding: 8px; text-align: left; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>INTELLIGENCE UNIT // AUTONOMOUS MODE</h1>
            <div class="label">TARGET MARKET: {data['question']}</div>
            <div class="label">EPOCH: {self.state['epoch']} | LEARNING RATE: {self.state['learning_rate']}</div>

            <div class="box">
                <div class="label">SMART PREDICTION (ADJUSTED)</div>
                <div class="big-text">{smart_prob:.1%} {trend_arrow}</div>
                <p>Base Market Price: {data['price']:.1%}</p>
                <p>Learned Bias Weight: {self.state['weights']['adaptive_bias']:.4f}</p>
            </div>

            <div class="box">
                <div class="label">DECISION LOG (LAST 5 EPOCHS)</div>
                <table>
                    <tr><th>Time</th><th>Market Price</th><th>Bot Prediction</th><th>Volume</th></tr>
                    {''.join([f"<tr><td>{h['timestamp']}</td><td>{h['price']:.2f}</td><td>{h['predicted']:.2f}</td><td>${h['volume']:,.0f}</td></tr>" for h in reversed(self.state['history'][-5:])])}
                </table>
            </div>
            
            <p style="text-align: center; color: #555; font-size: 0.7em;">
                Generated by GitHub Actions | Updates every 15 minutes
            </p>
        </body>
        </html>
        """
        with open("index.html", "w") as f:
            f.write(html)
        print("Dashboard updated successfully.")

if __name__ == "__main__":
    bot = MarketIntelligence()
    bot.run_epoch()