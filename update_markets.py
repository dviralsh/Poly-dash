import requests
import json
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
BASE_URL = "https://gamma-api.polymarket.com/markets"
HISTORY_FILE = 'market_history_log.json'  # We keep a rolling window of data points
DATA_FILE = 'data.json'
MAX_HISTORY_POINTS = 5000  # Keep last 5000 data points for training

# Filter Keywords
KEYWORDS = [
    'middle east', 'israel', 'gaza', 'hamas', 'hezbollah', 'idf',
    'iran', 'lebanon', 'syria', 'yemen', 'houthi', 'palestin',
    'saudi', 'uae', 'qatar', 'red sea', 'netanyahu', 'sinwar', 'ceasefire'
]

class ContinuousLearner:
    def __init__(self):
        self.history = self.load_history()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.scaler = MinMaxScaler()
        
    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                return pd.read_json(HISTORY_FILE)
            except:
                return pd.DataFrame()
        return pd.DataFrame()

    def save_history(self, new_df):
        # Combine old history with new snapshot
        combined = pd.concat([self.history, new_df], ignore_index=True)
        
        # Rolling Window: Drop old data to keep training fast and relevant
        if len(combined) > MAX_HISTORY_POINTS:
            combined = combined.iloc[-MAX_HISTORY_POINTS:]
            
        combined.to_json(HISTORY_FILE, orient='records', date_format='iso')
        self.history = combined

    def calculate_cluster_momentum(self, current_df):
        """
        Sophisticated: Detects if 'Related Topics' are moving together.
        If 'Iran' markets move, and 'Nuclear' markets move, both get a boost.
        """
        if current_df.empty: return current_df
        
        # 1. Turn questions into math vectors
        tfidf_matrix = self.vectorizer.fit_transform(current_df['question'])
        
        # 2. Calculate Similarity Matrix (How related is Market A to Market B?)
        # Result is a square matrix of 0.0 to 1.0 similarities
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # 3. Calculate "Cluster Heat"
        # For each market, sum the 'change_24h' of highly similar markets
        cluster_momentum = []
        prices = current_df['change_abs'].values
        
        for i in range(len(current_df)):
            # Get indices of similar markets (similarity > 0.3)
            # We assume markets with >30% text similarity are "connected"
            sim_scores = similarity_matrix[i]
            related_indices = np.where(sim_scores > 0.3)[0]
            
            # Average movement of the cluster (excluding self)
            if len(related_indices) > 1:
                cluster_move = np.mean(prices[related_indices])
            else:
                cluster_move = 0
            
            cluster_momentum.append(cluster_move)
            
        current_df['cluster_momentum'] = cluster_momentum
        return current_df

    def detect_anomalies(self, df):
        """
        The ML Core: Isolation Forest.
        """
        # FIX: Initialize the column with 0.0 (Normal) first. 
        # This prevents the KeyError if we return early.
        df['anomaly_score'] = 0.0

        # We need at least 2 points to compare things.
        if len(df) < 2: return df 
        
        try:
            # Features for the ML model
            features = ['change_abs', 'whale_index', 'spread', 'cluster_momentum']
            X = df[features].fillna(0)
            
            # Train the model on THIS snapshot
            # Contamination is how "sensitive" it is. 0.1 = Top 10% are alerts.
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(X)
            
            # decision_function returns a continuous score.
            raw_scores = iso_forest.decision_function(X)
            
            # Invert and Normalize to 0-1 scale
            # Reshape is needed because scaler expects a 2D array
            scores_reshaped = raw_scores.reshape(-1, 1) * -1
            anomaly_score = self.scaler.fit_transform(scores_reshaped).flatten()
            
            df['anomaly_score'] = anomaly_score
            
        except Exception as e:
            print(f"ML Calculation skipped: {e}")
            # We keep the default 0.0 scores we set at the start
            
        return df

    def get_color(self, score):
        """
        Returns a CSS HSL color string.
        0.0 (Boring) -> Green/Blue
        0.5 (Active) -> Yellow/Orange
        1.0 (Critical) -> Red/Glowing
        """
        # Interpolate Hue: 120 (Green) to 0 (Red)
        hue = 120 * (1 - score)
        
        # Saturation: 50% to 100% (More exciting = More vivid)
        sat = 50 + (50 * score)
        
        return f"hsl({hue:.0f}, {sat:.0f}%, 50%)"

def fetch_and_process():
    learner = ContinuousLearner()
    
    print("Fetching live market data...")
    markets = []
    
    # Fetch Data
    params = {'limit': 150, 'closed': 'false'}
    try:
        resp = requests.get(BASE_URL, params=params).json()
    except Exception as e:
        print(e)
        return

    # 1. Pre-Processing to create a DataFrame
    parsed_data = []
    for m in resp:
        if m.get('archived'): continue
        q = m.get('question', '').lower()
        if not any(k in q for k in KEYWORDS): continue
        
        # Extract numerics
        try:
            tokens = m.get('tokens', [])
            price = float(tokens[0].get('price', 0.5)) if tokens else 0.5
            vol = float(m.get('volume24hr') or 0)
            liq = float(m.get('liquidityNum') or 0)
            
            # Spread
            bid = float(m.get('bestBid') or 0)
            ask = float(m.get('bestAsk') or 0)
            spread = ask - bid
            
            # Derived ML Features
            change_24h = float(m.get('oneDayPriceChange', 0) or 0)
            
            # Whale Index (Continuous)
            whale_idx = np.log1p(vol) / (np.log1p(liq) + 1) # Avoid div/0
            
            parsed_data.append({
                'id': m['id'],
                'question': m['question'],
                'price': price,
                'volume': vol,
                'liquidity': liq,
                'change_raw': change_24h,
                'change_abs': abs(change_24h), # Magnitude matters for anomaly
                'spread': spread,
                'whale_index': whale_idx,
                'slug': m.get('slug')
            })
        except: continue
    
    if not parsed_data: return

    df = pd.DataFrame(parsed_data)
    
    # 2. NLP & Connection Analysis
    print("Analyzing topic connections...")
    df = learner.calculate_cluster_momentum(df)
    
    # 3. Unsupervised ML (Anomaly Detection)
    print("Running Isolation Forest...")
    df = learner.detect_anomalies(df)
    
    # 4. Save History (Continuous Learning)
    # We save this snapshot so the model has more data next time
    learner.save_history(df[['change_abs', 'whale_index', 'spread', 'cluster_momentum']])
    
    # 5. Format for Frontend
    output = []
    for _, row in df.iterrows():
        score = row['anomaly_score']
        
        # Create a text summary of WHY it's interesting
        reasons = []
        if row['whale_index'] > 0.8: reasons.append("Whale Activity")
        if row['cluster_momentum'] > 5: reasons.append("Sector Trend")
        if row['change_abs'] > 5: reasons.append("Price Spike")
        
        reason_text = " + ".join(reasons) if reasons else "Quiet"
        
        output.append({
            "id": row['id'],
            "question": row['question'],
            "price": f"{row['price']:.2f}",
            "score_float": float(score), # 0.0 to 1.0
            "color": learner.get_color(score), # HSL String
            "analysis": reason_text,
            "url": f"https://polymarket.com/market/{row['slug']}",
            # Keep these for debugging/display if needed
            "volume": row['volume'],
            "liquidity": row['liquidity']
        })
        
    # Sort by Anomaly Score (Most interesting first)
    output.sort(key=lambda x: x['score_float'], reverse=True)
    
    with open(DATA_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Done. Processed {len(output)} markets.")

if __name__ == "__main__":
    fetch_and_process()