import requests
import json
import numpy as np
import time
import os
import math
from textblob import TextBlob

# --- CONFIGURATION ---
GAMMA_URL = "https://gamma-api.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
DATA_FILE = 'data.json'
TRADERS_FILE = 'traders.json'
MODEL_STATE_FILE = 'model_state.json'
PREDICTION_LOG_FILE = 'prediction_log.json'

TOPICS = {
    'GAZA': ['gaza', 'hamas', 'rafah', 'sinwar', 'hostage', 'israel', 'idf'],
    'NORTH': ['lebanon', 'hezbollah', 'beirut', 'nasrallah'],
    'IRAN': ['iran', 'tehran', 'nuclear', 'khamenei', 'irgc'],
    'US_POL': ['biden', 'trump', 'election', 'harris'],
    'UKRAINE': ['ukraine', 'russia', 'putin', 'zelensky', 'kursk']
}

class MetaLearner:
    """
    The RL Brain. Observes the Alpha Signal and learns to weight it.
    """
    def __init__(self):
        self.learning_rate = 0.05
        self.state = self.load_state()
        self.pending_predictions = self.load_predictions()

    def load_state(self):
        if os.path.exists(MODEL_STATE_FILE):
            try: 
                with open(MODEL_STATE_FILE, 'r') as f: return json.load(f)
            except: pass
        return { "weights": {"signal_w": 0.5, "bias": 0.0}, "epoch": 0 }

    def load_predictions(self):
        if os.path.exists(PREDICTION_LOG_FILE):
            try: 
                with open(PREDICTION_LOG_FILE, 'r') as f: return json.load(f)
            except: pass
        return {}

    def save_state(self):
        with open(MODEL_STATE_FILE, 'w') as f: json.dump(self.state, f, indent=2)
        with open(PREDICTION_LOG_FILE, 'w') as f: json.dump(self.pending_predictions, f, indent=2)

    def predict_delta(self, signal):
        w = self.state['weights']
        return (w['signal_w'] * signal) + w['bias']

    def learn(self, current_prices_map):
        updated_log = {}
        updates = 0
        total_error = 0
        
        for m_id, log in self.pending_predictions.items():
            if m_id in current_prices_map:
                actual_price = current_prices_map[m_id]
                past_price = log['price_at_prediction']
                pred_delta = log['predicted_delta']
                signal = log['features']['signal']
                
                # Error Calculation
                actual_delta = actual_price - past_price
                error = pred_delta - actual_delta
                
                # Weight Update
                w = self.state['weights']
                w['signal_w'] -= self.learning_rate * error * signal
                w['bias'] -= self.learning_rate * error
                
                total_error += abs(error)
                updates += 1
            else:
                updated_log[m_id] = log
                
        self.pending_predictions = updated_log
        if updates > 0:
            print(f"🎓 RL Learned from {updates} events. New Signal Weight: {self.state['weights']['signal_w']:.3f}")

    def log_prediction(self, m_id, price, signal, delta):
        self.pending_predictions[m_id] = {
            "timestamp": time.time(),
            "price_at_prediction": price,
            "features": {"signal": signal},
            "predicted_delta": delta
        }

class HiveMind:
    def __init__(self):
        self.traders = self.load_traders()
        self.learner = MetaLearner()

    def load_traders(self):
        if os.path.exists(TRADERS_FILE):
            try: 
                with open(TRADERS_FILE, 'r') as f: return json.load(f)
            except: pass
        return {}

    def save_traders(self):
        with open(TRADERS_FILE, 'w') as f: json.dump(self.traders, f, indent=2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def safe_parse(self, data):
        if isinstance(data, list): return data
        if isinstance(data, str):
            try: return json.loads(data)
            except: return []
        return []

    def fetch_markets(self):
        print("--- 🧠 Starting Hybrid Scan ---")
        markets = []
        try:
            params = {'limit': 100, 'closed': 'false', 'order': 'volume24hr', 'ascending': 'false'}
            resp = requests.get(f"{GAMMA_URL}/markets", params=params)
            data = resp.json()
            for m in data:
                if m.get('archived'): continue
                text = (str(m.get('question')) + str(m.get('description'))).lower()
                for tags in TOPICS.values():
                    if any(t in text for t in tags):
                        markets.append(m); break
        except Exception as e: print(f"Fetch Error: {e}")
        print(f"✅ Found {len(markets)} markets.")
        return markets

    def analyze_crowd_wisdom(self, market_id, target_outcome_idx, current_price):
        """
        STABLE ALGORITHM (With Trend Correction Logic)
        """
        net_vote_strength = 0
        participant_count = 0
        
        try:
            params = {'market': market_id, 'limit': 1000}
            trades = requests.get(f"{DATA_API_URL}/trades", params=params).json()
            if not isinstance(trades, list): return 0.0, "API Limit"

            for t in reversed(trades):
                user = t.get('taker_address')
                side = t.get('side')
                size = float(t.get('size', 0))
                price_at_trade = float(t.get('price', 0))
                
                raw_idx = t.get('outcome_index', t.get('outcomeIndex'))
                if raw_idx is None: continue
                trade_idx = int(raw_idx)
                
                if user not in self.traders: self.traders[user] = {'score': 1.0, 'trades': 0}
                trader = self.traders[user]

                # Direction Logic
                is_bullish = False
                if trade_idx == target_outcome_idx: is_bullish = (side == "BUY")
                else: is_bullish = (side == "SELL")

                # Score Update
                price_improved = False
                if is_bullish and current_price > price_at_trade: price_improved = True
                if not is_bullish and current_price < price_at_trade: price_improved = True
                
                if price_improved: trader['score'] *= 1.02
                else: trader['score'] *= 0.98
                trader['score'] = max(0.2, min(5.0, trader['score']))
                self.traders[user] = trader

                # Stable Math
                vector = 1.0 if is_bullish else -1.0
                reliability = self.sigmoid(trader['score'] - 1)
                vote_weight = np.log1p(size) * reliability
                
                net_vote_strength += (vote_weight * vector)
                participant_count += 1
                        
        except Exception as e: pass
        
        # Tanh Squashing
        impact = np.tanh(net_vote_strength / 20.0) * 0.15 
        
        narrative = ""
        if participant_count > 0:
            if net_vote_strength > 2: narrative = f"Strong Buy ({participant_count} trds)"
            elif net_vote_strength < -2: narrative = f"Strong Sell ({participant_count} trds)"
            else: narrative = f"Neutral ({participant_count} trds)"
            
        return impact, narrative

    def run(self):
        markets = self.fetch_markets()
        if not markets: return []

        # 1. Learn
        current_map = {}
        for m in markets:
            try:
                prices = self.safe_parse(m.get('outcomePrices'))
                outcomes = self.safe_parse(m.get('outcomes'))
                if not prices: continue
                t_idx = outcomes.index("Yes") if "Yes" in outcomes else np.argmax([float(p) for p in prices])
                current_map[m['id']] = float(prices[t_idx])
            except: continue
        self.learner.learn(current_map)
        
        results = []
        print("🧠 Generating Predictions...")
        for m in markets:
            try:
                prices = self.safe_parse(m.get('outcomePrices'))
                outcomes = self.safe_parse(m.get('outcomes'))
                if not prices or not outcomes: continue
                
                t_idx = outcomes.index("Yes") if "Yes" in outcomes else np.argmax([float(p) for p in prices])
                price = float(prices[t_idx])
                name = outcomes[t_idx]
                if price < 0.01: continue
                
                # Fetch Trend (24h change)
                change_24h = float(m.get('oneDayPriceChange') or 0)

                # --- 1. ALPHA SIGNAL ---
                crowd_impact, crowd_txt = self.analyze_crowd_wisdom(m['id'], t_idx, price)
                
                # --- 2. ALPHA PREDICTION (Fixed + Momentum) ---
                # Fix: Add Trend to avoid "Bearish Bias" from profit taking
                alpha_pred = price + crowd_impact + (change_24h * 0.1)
                alpha_pred = max(0.01, min(0.99, alpha_pred))
                
                # --- 3. RL PREDICTION (Learned) ---
                rl_delta = self.learner.predict_delta(crowd_impact)
                rl_pred = price + rl_delta
                rl_pred = max(0.01, min(0.99, rl_pred))
                
                self.learner.log_prediction(m['id'], price, crowd_impact, rl_delta)
                
                # Format
                score = abs(alpha_pred - price) * 100
                hue = 120 * (1 - min(1, score/20))
                w = self.learner.state['weights']
                narrative = f"Analyzed <b>{name}</b>.<br>" \
                            f"👥 <b>Crowd:</b> {crowd_txt} (Imp {crowd_impact:+.3f})<br>" \
                            f"⚡ <b>Alpha:</b> {alpha_pred:.2f} (Trend {change_24h:+.1%})<br>" \
                            f"🧠 <b>RL:</b> {rl_pred:.2f} (Weight {w['signal_w']:.2f})"

                results.append({
                    "question": m['question'],
                    "outcome": name,
                    "price": f"{price:.2f}",
                    "alpha_pred": f"{alpha_pred:.2f}",
                    "rl_pred": f"{rl_pred:.2f}",
                    "score_float": score,
                    "color": f"hsl({hue:.0f}, 90%, 45%)",
                    "narrative": narrative,
                    "url": f"https://polymarket.com/market/{m.get('slug')}",
                    "raw_data": json.dumps(m)
                })
            except Exception as e: continue
            
        self.save_traders()
        self.learner.save_state()
        results.sort(key=lambda x: x['score_float'], reverse=True)
        print(f"🎉 Analysis Complete. Saved {len(results)} markets.")
        return results

if __name__ == "__main__":
    mind = HiveMind()
    data = mind.run()
    with open(DATA_FILE, 'w') as f: json.dump(data, f, indent=2)
    print("Done.")