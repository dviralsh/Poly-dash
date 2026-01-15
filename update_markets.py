import requests
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
BASE_URL = "https://gamma-api.polymarket.com/markets"
HISTORY_FILE = 'market_history_log.json'
DATA_FILE = 'data.json'

# Expanded Keywords (Lower case)
KEYWORDS = [
    'middle east', 'israel', 'gaza', 'hamas', 'hezbollah', 'idf',
    'iran', 'lebanon', 'syria', 'yemen', 'houthi', 'palestin',
    'saudi', 'uae', 'qatar', 'red sea', 'netanyahu', 'sinwar', 'ceasefire',
    'tehran', 'beirut', 'jerusalem', 'tel aviv'
]

class HybridIntelligence:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fetch_all_relevant_markets(self):
        """
        Fetches MANY markets using pagination to ensure we don't miss active topics.
        """
        all_markets = []
        cursor = "" # Polymarket uses cursor based pagination or offset
        offset = 0
        limit = 100
        max_pages = 10 # Fetch up to 1000 markets to be safe

        print("Scanning Polymarket Deep Data...")

        for _ in range(max_pages):
            try:
                params = {
                    'limit': limit, 
                    'offset': offset, 
                    'closed': 'false'
                }
                resp = requests.get(BASE_URL, params=params).json()
                
                if not resp: break
                
                # Filter locally
                for m in resp:
                    if m.get('archived'): continue
                    
                    # Robust Keyword Check
                    q_text = (m.get('question') or "").lower()
                    desc_text = (m.get('description') or "").lower()
                    full_text = f"{q_text} {desc_text}"
                    
                    if any(k in full_text for k in KEYWORDS):
                        all_markets.append(m)

                offset += limit
                time.sleep(0.2) # Be nice to API

            except Exception as e:
                print(f"Fetch error: {e}")
                break
        
        print(f"Found {len(all_markets)} relevant Middle East markets.")
        return all_markets

    def calculate_hybrid_score(self, df):
        """
        Combines ML Anomaly Score with Hard Rules (Whales).
        """
        # 1. Base ML Score (Anomaly Detection)
        # If we have very little data, skip ML and use heuristics
        if len(df) > 5:
            features = ['change_abs', 'whale_index', 'spread', 'cluster_momentum']
            X = df[features].fillna(0)
            
            # Contamination=auto lets the model decide how many are outliers
            iso = IsolationForest(contamination='auto', random_state=42)
            iso.fit(X)
            
            # Convert decision_function to 0-1 score
            raw_scores = iso.decision_function(X) * -1 
            df['ml_score'] = self.scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()
        else:
            df['ml_score'] = 0.0

        # 2. Heuristic Score (Hard Rules)
        # Whale Index is already Log scale. 
        # > 1.0 is high. > 2.0 is extreme.
        # We clamp it to 0-1 range for scoring.
        df['whale_score'] = df['whale_index'].apply(lambda x: min(x / 2.0, 1.0))

        # 3. Final Weighted Score
        # We take the MAXIMUM of ML or Whale. 
        # This prevents the "0% Signal" bug when a Whale is present.
        df['final_score'] = df[['ml_score', 'whale_score']].max(axis=1)
        
        return df

    def get_color(self, score):
        # 0.0 -> Green (120)
        # 1.0 -> Red (0)
        hue = 120 * (1 - score)
        return f"hsl({hue:.0f}, 90%, 45%)" # Increased saturation for better visibility

def main():
    bot = HybridIntelligence()
    
    # 1. Fetch
    raw_markets = bot.fetch_all_relevant_markets()
    if not raw_markets:
        print("No markets found. Check keywords or API.")
        return

    # 2. Parse into DataFrame
    parsed_data = []
    for m in raw_markets:
        try:
            tokens = m.get('tokens', [])
            price = float(tokens[0].get('price', 0.5)) if tokens else 0.5
            vol = float(m.get('volume24hr') or 0)
            liq = float(m.get('liquidityNum') or 0)
            change_24h = float(m.get('oneDayPriceChange', 0) or 0)
            
            # Whale Index Calculation
            whale_idx = 0
            if liq > 10: # Avoid division by zero on dead markets
                whale_idx = np.log1p(vol) / (np.log1p(liq))
                # Boost index if volume is explicitly massive (>100k)
                if vol > 100000: whale_idx *= 1.2

            parsed_data.append({
                'id': m['id'],
                'question': m['question'],
                'price': price,
                'change_abs': abs(change_24h),
                'whale_index': whale_idx,
                'spread': float(m.get('bestAsk',0)) - float(m.get('bestBid',0)),
                'cluster_momentum': 0, # Placeholder
                'slug': m.get('slug'),
                'volume': vol
            })
        except: continue
        
    df = pd.DataFrame(parsed_data)
    if df.empty: return

    # 3. Process
    # (Simple cluster logic for now to save compute time)
    df['cluster_momentum'] = 0.0 
    
    # Run Hybrid Scoring
    df = bot.calculate_hybrid_score(df)
    
    # 4. Output
    output = []
    for _, row in df.iterrows():
        score = float(row['final_score'])
        
        # Smart Tagging
        reasons = []
        if row['whale_score'] > 0.5: reasons.append("Whale Activity 🐋")
        if row['change_abs'] > 0.05: reasons.append("Price Volatility 📈")
        if row['ml_score'] > 0.8: reasons.append("Anomaly Detected 🤖")
        if not reasons: reasons.append("Stable")
        
        output.append({
            "question": row['question'],
            "price": f"{row['price']:.2f}",
            "score_float": score,
            "color": bot.get_color(score),
            "analysis": " + ".join(reasons),
            "url": f"https://polymarket.com/market/{row['slug']}",
            "volume": row['volume']
        })

    # Sort by Score (Desc) so hottest news is first
    output.sort(key=lambda x: x['score_float'], reverse=True)
    
    with open(DATA_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(output)} markets.")

if __name__ == "__main__":
    main()