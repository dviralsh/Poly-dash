import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import time
import os
import math

# --- CONFIGURATION ---
GAMMA_URL = "https://gamma-api.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
DATA_FILE = 'data.json'
TRADERS_FILE = 'traders.json'

TOPICS = {
    'GAZA': ['gaza', 'hamas', 'rafah', 'sinwar', 'hostage'],
    'NORTH': ['lebanon', 'hezbollah', 'beirut', 'nasrallah'],
    'IRAN': ['iran', 'tehran', 'nuclear', 'khamenei', 'irgc'],
    'US_POL': ['biden', 'trump', 'election', 'harris'],
    'UKRAINE': ['ukraine', 'russia', 'putin', 'zelensky', 'kursk']
}

class HiveMind:
    def __init__(self):
        self.traders = self.load_traders()

    def load_traders(self):
        if os.path.exists(TRADERS_FILE):
            try:
                with open(TRADERS_FILE, 'r') as f:
                    return json.load(f)
            except: return {}
        return {}

    def save_traders(self):
        with open(TRADERS_FILE, 'w') as f:
            json.dump(self.traders, f, indent=2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fetch_markets(self):
        print("--- 🧠 Starting Corrected Hive Mind Scan ---")
        all_markets = []
        limit = 50
        offset = 0
        
        for _ in range(10): 
            try:
                params = {'limit': limit, 'offset': offset, 'closed': 'false', 'order': 'volume24hr', 'ascending': 'false'}
                resp = requests.get(f"{GAMMA_URL}/markets", params=params).json()
                if not resp: break
                
                for m in resp:
                    if m.get('archived'): continue
                    text = (str(m.get('question')) + str(m.get('description'))).lower()
                    relevant = False
                    for topic_tags in TOPICS.values():
                        if any(t in text for t in topic_tags):
                            relevant = True
                            break
                    if relevant:
                        all_markets.append(m)
                
                offset += limit
                time.sleep(0.1)
            except: break
        
        print(f"Captured {len(all_markets)} markets.")
        return all_markets

    def analyze_sentiment(self, market_id):
        try:
            params = {'marketID': market_id, 'limit': 20}
            resp = requests.get(f"{GAMMA_URL}/comments", params=params).json()
            if not resp: return 0, "No Comments"
            
            polarities = [TextBlob(c.get('body', '')).sentiment.polarity for c in resp]
            return np.mean(polarities), "Mixed Discussion"
        except: return 0, "No Comments"

    def analyze_crowd_wisdom(self, market_id, target_outcome_idx, current_price):
        """
        CORRECTED LOGIC: Checks outcomeIndex to determine if trade is Bullish or Bearish.
        """
        net_vote_strength = 0
        participant_count = 0
        
        try:
            # Fetch recent trades
            params = {'market': market_id, 'limit': 1000}
            trades = requests.get(f"{DATA_API_URL}/trades", params=params).json()
            
            for t in trades:
                user = t.get('taker_address')
                side = t.get('side') # "BUY" or "SELL"
                size = float(t.get('size', 0))
                price_at_trade = float(t.get('price', 0))
                # API returns outcomeIndex as string often
                trade_outcome_idx = int(t.get('outcome_index', t.get('outcomeIndex', -1)))
                
                # --- 1. Continuous Learning (Update Score) ---
                if user not in self.traders:
                    self.traders[user] = {'score': 1.0, 'trades': 0}
                
                trader = self.traders[user]
                
                # Did they win?
                # If they BOUGHT the target, and price is now higher -> WIN
                # If they BOUGHT the OPPONENT, and price is now higher -> LOSS (because opponent dropped)
                
                is_bullish_trade = False
                if side == "BUY":
                    if trade_outcome_idx == target_outcome_idx: is_bullish_trade = True
                    else: is_bullish_trade = False # They bought the enemy
                elif side == "SELL":
                    if trade_outcome_idx == target_outcome_idx: is_bullish_trade = False # Sold target
                    else: is_bullish_trade = True # Sold enemy (Bullish for target)

                # Score update logic
                # We use sigmoid normalization later, so we let raw score float naturally
                if is_bullish_trade:
                    if current_price > price_at_trade: trader['score'] *= 1.02 # Good prediction
                    else: trader['score'] *= 0.98
                else:
                    # Bearish trade (e.g. Bought No). If Price Dropped, they won.
                    if current_price < price_at_trade: trader['score'] *= 1.02
                    else: trader['score'] *= 0.98

                trader['trades'] += 1
                self.traders[user] = trader
                
                # --- 2. Calculate Vote Impact ---
                # Determine vector direction relative to OUR target
                vector = 0
                if trade_outcome_idx == target_outcome_idx:
                    vector = 1 if side == "BUY" else -1
                else:
                    # Trading the opponent has inverse effect
                    vector = -1 if side == "BUY" else 1
                
                # Weighted Vote: Log(Size) * Sigmoid(Score)
                # Sigmoid maps score 0..inf -> 0.5..1.0 mostly
                # We center sigmoid at 1.0 so bad traders (<1) have weight <0.5
                reliability = self.sigmoid(trader['score'] - 1) # Center around 1.0
                
                vote_weight = np.log1p(size) * reliability
                net_vote_strength += (vote_weight * vector)
                participant_count += 1
                    
        except Exception as e:
            pass
        
        # Normalize Net Vote (-1 to 1)
        # Using Tanh to squash the infinite sum into a predictable range
        impact = np.tanh(net_vote_strength / 20.0) * 0.15 # Max 15% price impact
        
        narrative = ""
        if participant_count > 0:
            if net_vote_strength > 2: narrative = f"Strong Buy Consensus ({participant_count} traders)"
            elif net_vote_strength < -2: narrative = f"Strong Sell Consensus ({participant_count} traders)"
            else: narrative = f"Neutral Flow ({participant_count} traders)"
            
        return impact, narrative

    def run_analysis(self):
        markets = self.fetch_markets()
        results = []
        
        print("🧠 Processing Corrected Crowd Wisdom...")
        
        for m in markets:
            try:
                outcomes = json.loads(m.get('outcomes', '[]'))
                prices = json.loads(m.get('outcomePrices', '[]'))
                if not outcomes or not prices: continue
                
                # Select Target (Favorite or Yes)
                target_idx = 0
                if "Yes" in outcomes: target_idx = outcomes.index("Yes")
                else: target_idx = np.argmax([float(p) for p in prices])
                
                price = float(prices[target_idx])
                name = outcomes[target_idx]
                
                # 1. Sentiment
                sent_score, sent_txt = self.analyze_sentiment(m['id'])
                
                # 2. Crowd Wisdom (Now unbiased)
                crowd_impact, crowd_txt = self.analyze_crowd_wisdom(m['id'], target_idx, price)
                
                # 3. Prediction
                pred_price = price + (sent_score * 0.02) + crowd_impact
                pred_price = max(0.01, min(0.99, pred_price))
                
                # Score
                gap = abs(pred_price - price)
                score = gap * 100 
                hue = 120 * (1 - min(1, score/15))
                
                narrative = f"Analyzed <b>{name}</b>. "
                narrative += f"👥 <b>Crowd:</b> {crowd_txt}. "
                
                if pred_price > price + 0.01:
                    narrative += f"🚀 <b>Target:</b> {pred_price:.2f} (Undervalued)"
                elif pred_price < price - 0.01:
                    narrative += f"🔻 <b>Target:</b> {pred_price:.2f} (Overvalued)"
                else:
                    narrative += "⚖️ Fair Value."

                results.append({
                    "question": m['question'],
                    "outcome": name,
                    "price": f"{price:.2f}",
                    "predicted": f"{pred_price:.2f}",
                    "delta": pred_price - price,
                    "score_float": score,
                    "color": f"hsl({hue:.0f}, 90%, 45%)",
                    "narrative": narrative,
                    "url": f"https://polymarket.com/market/{m.get('slug')}",
                    "raw_data": json.dumps(m)
                })
                
            except Exception as e: continue
            
        self.save_traders()
        results.sort(key=lambda x: x['score_float'], reverse=True)
        return results

if __name__ == "__main__":
    mind = HiveMind()
    data = mind.run_analysis()
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print("Done.")