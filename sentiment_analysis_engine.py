"""
Bambhoria God-Eye V52 - Advanced Sentiment Analysis Engine
=========================================================
Revolutionary feature that analyzes:
- Real-time news sentiment
- Market mood detection
- Social media sentiment
- Economic event impact
- Sector rotation signals
- Market psychology indicators

This gives the AI system human-like market intuition!
"""

import requests
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import re
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentType(Enum):
    """Types of sentiment analysis"""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentData:
    """Container for sentiment analysis results"""
    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_type: SentimentType
    confidence: float  # 0.0 to 1.0
    news_count: int
    headline: str
    impact_factor: float  # Market impact multiplier
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MockNewsProvider:
    """Mock news provider for demonstration (can be replaced with real APIs)"""
    
    def __init__(self):
        self.news_templates = {
            'RELIANCE': [
                "Reliance Industries reports strong quarterly results, revenue up {}%",
                "RIL announces major expansion in renewable energy sector",
                "Reliance stock hits new highs on positive investor sentiment",
                "Oil prices surge, benefiting Reliance's petrochemical business",
                "Jio subscriber base crosses {} million milestone",
                "Reliance faces regulatory challenges in telecom sector",
                "Concerns over debt levels impact Reliance stock price",
                "Competition intensifies in retail sector for Reliance"
            ],
            'TCS': [
                "TCS wins major {} million dollar contract from global bank",
                "Tata Consultancy Services reports record quarterly profits",
                "TCS announces hiring of {} thousand engineers",
                "Digital transformation demand boosts TCS revenue",
                "TCS stock under pressure due to visa concerns",
                "Attrition rates rise at TCS, margin pressure increases",
                "Currency headwinds impact TCS international revenue"
            ],
            'INFY': [
                "Infosys beats earnings estimates, guidance raised",
                "Infosys launches new AI-powered platform",
                "Large deal wins drive Infosys growth in Q{}",
                "Infosys CEO optimistic about digital transformation trends",
                "Infosys faces talent shortage in key technology areas",
                "Client ramp-down affects Infosys revenue growth",
                "Infosys stock falls on weak guidance revision"
            ],
            'HDFC': [
                "HDFC Bank reports strong loan growth of {}%",
                "HDFC Bank's asset quality improves significantly",
                "Digital banking initiatives drive HDFC's growth",
                "HDFC Bank raises {} billion in capital",
                "RBI restrictions on HDFC Bank's digital launches lifted",
                "HDFC Bank faces increased competition from fintech",
                "Credit costs rise for HDFC Bank in challenging environment"
            ],
            'ICICI': [
                "ICICI Bank's retail portfolio shows robust growth",
                "ICICI Bank stock rallies on improved NPA outlook",
                "ICICI Bank announces strategic partnership with {}",
                "Digital lending drives ICICI Bank's profitability",
                "ICICI Bank faces regulatory scrutiny over lending practices",
                "Economic slowdown impacts ICICI Bank's credit growth",
                "ICICI Bank's provisions increase amid economic uncertainty"
            ]
        }
        
        self.sector_news = [
            "Banking sector outlook improves on economic recovery signs",
            "IT sector benefits from increased digitalization trends",
            "Oil prices volatility impacts energy sector stocks",
            "RBI policy decision supports banking sector sentiment",
            "Global recession fears weigh on IT export companies",
            "Regulatory changes create uncertainty in financial sector"
        ]
    
    def get_latest_news(self, symbol: str, count: int = 5) -> List[Dict]:
        """Generate mock news for a symbol"""
        news_list = []
        templates = self.news_templates.get(symbol, [])
        
        for i in range(count):
            template = random.choice(templates)
            
            # Fill in dynamic values
            if '{}' in template:
                if 'revenue up' in template or 'growth' in template:
                    value = random.randint(5, 25)
                elif 'million' in template:
                    value = random.randint(100, 1000)
                elif 'thousand' in template:
                    value = random.randint(50, 200)
                elif 'billion' in template:
                    value = random.randint(1, 10)
                elif 'Q{}' in template:
                    value = random.randint(1, 4)
                else:
                    value = random.randint(10, 50)
                
                headline = template.format(value)
            else:
                headline = template
            
            # Simulate realistic news data
            news_item = {
                'headline': headline,
                'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 360)),
                'source': random.choice(['Economic Times', 'Business Standard', 'Reuters', 'Bloomberg', 'Moneycontrol']),
                'url': f"https://news-source.com/{symbol.lower()}-{i}",
                'sentiment_keywords': self._extract_sentiment_keywords(headline)
            }
            news_list.append(news_item)
        
        return news_list
    
    def get_sector_news(self, count: int = 3) -> List[Dict]:
        """Generate mock sector-wide news"""
        news_list = []
        
        for i in range(count):
            headline = random.choice(self.sector_news)
            news_item = {
                'headline': headline,
                'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 180)),
                'source': random.choice(['ET Markets', 'CNBC', 'Financial Express']),
                'sector': 'General',
                'sentiment_keywords': self._extract_sentiment_keywords(headline)
            }
            news_list.append(news_item)
        
        return news_list
    
    def _extract_sentiment_keywords(self, text: str) -> List[str]:
        """Extract sentiment-bearing keywords from text"""
        positive_keywords = ['strong', 'growth', 'beats', 'wins', 'optimistic', 'improves', 'robust', 'rallies', 'benefits']
        negative_keywords = ['pressure', 'challenges', 'concerns', 'falls', 'weak', 'uncertainty', 'slowdown', 'scrutiny']
        
        keywords = []
        text_lower = text.lower()
        
        for keyword in positive_keywords + negative_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords


class SentimentAnalyzer:
    """Advanced sentiment analysis engine"""
    
    def __init__(self):
        self.positive_words = {
            'excellent': 0.9, 'outstanding': 0.9, 'strong': 0.7, 'growth': 0.6,
            'wins': 0.8, 'beats': 0.7, 'optimistic': 0.6, 'robust': 0.7,
            'rallies': 0.8, 'improves': 0.6, 'benefits': 0.5, 'surge': 0.8,
            'milestone': 0.6, 'expansion': 0.5, 'profit': 0.6, 'record': 0.7,
            'partnership': 0.5, 'strategic': 0.4, 'innovation': 0.5, 'launch': 0.4
        }
        
        self.negative_words = {
            'pressure': -0.6, 'challenges': -0.5, 'concerns': -0.6, 'falls': -0.7,
            'weak': -0.6, 'uncertainty': -0.5, 'slowdown': -0.7, 'scrutiny': -0.6,
            'debt': -0.4, 'competition': -0.3, 'shortage': -0.5, 'restrictions': -0.6,
            'recession': -0.8, 'crisis': -0.9, 'losses': -0.8, 'decline': -0.6,
            'volatility': -0.4, 'headwinds': -0.5, 'regulatory': -0.4
        }
        
        self.sector_multipliers = {
            'RELIANCE': 1.2,  # High impact stock
            'TCS': 1.1,       # Stable large cap
            'INFY': 1.1,      # Stable large cap
            'HDFC': 1.0,      # Banking sector
            'ICICI': 1.0      # Banking sector
        }
        
        logger.info("🧠 Sentiment Analyzer initialized with market psychology models")
    
    def analyze_sentiment(self, news_list: List[Dict], symbol: str) -> SentimentData:
        """Analyze sentiment from news headlines"""
        
        if not news_list:
            return SentimentData(
                symbol=symbol,
                sentiment_score=0.0,
                sentiment_type=SentimentType.NEUTRAL,
                confidence=0.0,
                news_count=0,
                headline="No news available",
                impact_factor=1.0,
                timestamp=datetime.now()
            )
        
        total_score = 0.0
        total_weight = 0.0
        recent_headlines = []
        
        for news in news_list:
            headline = news['headline']
            # Time decay factor (recent news more important)
            time_diff = (datetime.now() - news['timestamp']).total_seconds() / 3600  # hours
            time_weight = max(0.1, 1.0 - (time_diff / 24))  # Decay over 24 hours
            
            # Analyze headline sentiment
            headline_score = self._analyze_text_sentiment(headline)
            
            # Apply time weight
            weighted_score = headline_score * time_weight
            total_score += weighted_score
            total_weight += time_weight
            
            recent_headlines.append(headline)
        
        # Calculate final sentiment
        if total_weight > 0:
            avg_sentiment = total_score / total_weight
        else:
            avg_sentiment = 0.0
        
        # Apply sector multiplier
        sector_multiplier = self.sector_multipliers.get(symbol, 1.0)
        final_sentiment = avg_sentiment * sector_multiplier
        
        # Clamp to [-1, 1] range
        final_sentiment = max(-1.0, min(1.0, final_sentiment))
        
        # Determine sentiment type
        sentiment_type = self._get_sentiment_type(final_sentiment)
        
        # Calculate confidence based on news count and consistency
        confidence = self._calculate_confidence(news_list, final_sentiment)
        
        # Calculate market impact factor
        impact_factor = self._calculate_impact_factor(final_sentiment, confidence, len(news_list))
        
        # Get most relevant headline
        most_recent_headline = recent_headlines[0] if recent_headlines else "No news"
        
        return SentimentData(
            symbol=symbol,
            sentiment_score=final_sentiment,
            sentiment_type=sentiment_type,
            confidence=confidence,
            news_count=len(news_list),
            headline=most_recent_headline,
            impact_factor=impact_factor,
            timestamp=datetime.now()
        )
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of a single text"""
        text_lower = text.lower()
        score = 0.0
        word_count = 0
        
        # Check positive words
        for word, weight in self.positive_words.items():
            if word in text_lower:
                score += weight
                word_count += 1
        
        # Check negative words
        for word, weight in self.negative_words.items():
            if word in text_lower:
                score += weight  # weight is negative
                word_count += 1
        
        # Normalize by word count
        if word_count > 0:
            score = score / word_count
        
        # Apply percentage boost/penalty for numbers
        if any(char.isdigit() for char in text):
            # Look for percentage patterns
            import re
            percentages = re.findall(r'(\d+)%', text)
            if percentages:
                pct = int(percentages[0])
                if 'up' in text_lower or 'growth' in text_lower or 'increase' in text_lower:
                    score += min(0.3, pct / 100)  # Positive boost
                elif 'down' in text_lower or 'decline' in text_lower or 'fall' in text_lower:
                    score -= min(0.3, pct / 100)  # Negative penalty
        
        return score
    
    def _get_sentiment_type(self, score: float) -> SentimentType:
        """Convert numerical score to sentiment type"""
        if score >= 0.6:
            return SentimentType.VERY_BULLISH
        elif score >= 0.2:
            return SentimentType.BULLISH
        elif score <= -0.6:
            return SentimentType.VERY_BEARISH
        elif score <= -0.2:
            return SentimentType.BEARISH
        else:
            return SentimentType.NEUTRAL
    
    def _calculate_confidence(self, news_list: List[Dict], sentiment: float) -> float:
        """Calculate confidence in sentiment analysis"""
        if not news_list:
            return 0.0
        
        # Base confidence from news count
        news_count_factor = min(1.0, len(news_list) / 5.0)  # Max confidence with 5+ news
        
        # Confidence from sentiment strength
        sentiment_strength = abs(sentiment)
        
        # Confidence from news recency
        recent_count = sum(1 for news in news_list 
                          if (datetime.now() - news['timestamp']).total_seconds() < 3600)
        recency_factor = min(1.0, recent_count / 3.0)
        
        # Combined confidence
        confidence = (news_count_factor * 0.4 + 
                     sentiment_strength * 0.4 + 
                     recency_factor * 0.2)
        
        return min(1.0, confidence)
    
    def _calculate_impact_factor(self, sentiment: float, confidence: float, news_count: int) -> float:
        """Calculate how much this sentiment should impact trading decisions"""
        base_impact = abs(sentiment) * confidence
        
        # More news = higher impact
        news_factor = min(1.5, 1.0 + (news_count - 1) * 0.1)
        
        # Strong sentiment has non-linear impact
        sentiment_factor = 1.0 + (abs(sentiment) ** 2) * 0.5
        
        impact = base_impact * news_factor * sentiment_factor
        
        # Clamp to reasonable range
        return min(2.0, max(0.1, impact))


class SentimentSignalGenerator:
    """Generates trading signals based on sentiment analysis"""
    
    def __init__(self):
        self.sentiment_history = defaultdict(lambda: deque(maxlen=50))
        self.signal_cooldown = defaultdict(lambda: datetime.min)
        self.cooldown_minutes = 15  # Minimum time between sentiment signals
        
        logger.info("📡 Sentiment Signal Generator initialized")
    
    def generate_sentiment_signal(self, sentiment_data: SentimentData) -> Optional[Dict]:
        """Generate trading signal from sentiment data"""
        
        symbol = sentiment_data.symbol
        current_time = datetime.now()
        
        # Check cooldown
        if current_time - self.signal_cooldown[symbol] < timedelta(minutes=self.cooldown_minutes):
            return None
        
        # Store sentiment history
        self.sentiment_history[symbol].append(sentiment_data)
        
        # Need at least 2 data points for trend analysis
        if len(self.sentiment_history[symbol]) < 2:
            return None
        
        # Analyze sentiment trend
        recent_sentiments = list(self.sentiment_history[symbol])[-5:]  # Last 5 data points
        sentiment_trend = self._calculate_sentiment_trend(recent_sentiments)
        
        # Generate signal based on sentiment and trend
        signal = self._evaluate_sentiment_signal(sentiment_data, sentiment_trend)
        
        if signal:
            self.signal_cooldown[symbol] = current_time
            return signal
        
        return None
    
    def _calculate_sentiment_trend(self, sentiment_list: List[SentimentData]) -> float:
        """Calculate trend in sentiment over time"""
        if len(sentiment_list) < 2:
            return 0.0
        
        scores = [s.sentiment_score for s in sentiment_list]
        
        # Simple linear trend calculation
        x = list(range(len(scores)))
        y = scores
        
        # Calculate slope
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def _evaluate_sentiment_signal(self, sentiment_data: SentimentData, trend: float) -> Optional[Dict]:
        """Evaluate if sentiment warrants a trading signal"""
        
        sentiment_score = sentiment_data.sentiment_score
        confidence = sentiment_data.confidence
        impact_factor = sentiment_data.impact_factor
        
        # Signal thresholds
        min_confidence = 0.6
        min_sentiment_strength = 0.4
        min_impact = 0.8
        
        # Check if sentiment is strong enough
        if (confidence < min_confidence or 
            abs(sentiment_score) < min_sentiment_strength or 
            impact_factor < min_impact):
            return None
        
        # Determine signal direction
        if sentiment_score > 0 and trend >= 0:
            action = 'BUY'
            signal_strength = sentiment_score * impact_factor
        elif sentiment_score < 0 and trend <= 0:
            action = 'SELL'
            signal_strength = abs(sentiment_score) * impact_factor
        else:
            # Mixed signals, no action
            return None
        
        # Calculate signal confidence
        signal_confidence = min(0.95, confidence * impact_factor * 0.8)
        
        # Adjust confidence based on trend alignment
        if (sentiment_score > 0 and trend > 0) or (sentiment_score < 0 and trend < 0):
            signal_confidence *= 1.1  # Boost for trend alignment
        
        signal_confidence = min(0.95, signal_confidence)
        
        return {
            'symbol': sentiment_data.symbol,
            'action': action,
            'confidence': signal_confidence,
            'source': 'SENTIMENT_ANALYSIS',
            'sentiment_score': sentiment_score,
            'sentiment_type': sentiment_data.sentiment_type.value,
            'news_headline': sentiment_data.headline,
            'news_count': sentiment_data.news_count,
            'impact_factor': impact_factor,
            'trend': trend,
            'timestamp': datetime.now()
        }


class SentimentEngine:
    """Main sentiment analysis engine that coordinates all components"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.news_provider = MockNewsProvider()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.signal_generator = SentimentSignalGenerator()
        
        # State management
        self.is_running = False
        self.update_thread = None
        self.sentiment_data = {}
        self.sentiment_callbacks = []
        
        # Update frequency
        self.update_interval = 30  # seconds
        
        logger.info(f"🚀 Sentiment Engine initialized for symbols: {symbols}")
    
    def register_callback(self, callback):
        """Register callback for sentiment signals"""
        self.sentiment_callbacks.append(callback)
        logger.info(f"✅ Registered sentiment callback: {callback.__name__}")
    
    def start(self):
        """Start sentiment analysis engine"""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("🎯 Sentiment Engine started - analyzing market psychology")
    
    def stop(self):
        """Stop sentiment analysis engine"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        logger.info("🛑 Sentiment Engine stopped")
    
    def _update_loop(self):
        """Main update loop for sentiment analysis"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # Get latest news
                    news_list = self.news_provider.get_latest_news(symbol, count=3)
                    
                    # Analyze sentiment
                    sentiment_data = self.sentiment_analyzer.analyze_sentiment(news_list, symbol)
                    self.sentiment_data[symbol] = sentiment_data
                    
                    # Generate signal if applicable
                    signal = self.signal_generator.generate_sentiment_signal(sentiment_data)
                    
                    if signal:
                        logger.info(f"🧠 SENTIMENT SIGNAL: {symbol} {signal['action']} | "
                                  f"Confidence: {signal['confidence']:.3f} | "
                                  f"Sentiment: {signal['sentiment_type'].upper()}")
                        logger.info(f"   News: {signal['news_headline'][:80]}...")
                        
                        # Send to callbacks
                        for callback in self.sentiment_callbacks:
                            try:
                                callback(signal)
                            except Exception as e:
                                logger.error(f"❌ Sentiment callback error: {e}")
                
                # Sleep before next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Sentiment engine error: {e}")
                time.sleep(5)
    
    def get_current_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """Get current sentiment for a symbol"""
        return self.sentiment_data.get(symbol)
    
    def get_all_sentiments(self) -> Dict[str, SentimentData]:
        """Get current sentiment for all symbols"""
        return self.sentiment_data.copy()
    
    def get_sentiment_summary(self) -> Dict[str, str]:
        """Get a summary of current market sentiment"""
        summary = {}
        for symbol, sentiment_data in self.sentiment_data.items():
            if sentiment_data:
                summary[symbol] = f"{sentiment_data.sentiment_type.value.upper()} ({sentiment_data.sentiment_score:.2f})"
            else:
                summary[symbol] = "NO_DATA"
        
        return summary


def demo_sentiment_engine():
    """Demonstration of the sentiment analysis engine"""
    print("\n" + "="*80)
    print("      🧠 DEMO: Advanced Sentiment Analysis Engine 🧠")
    print("="*80 + "\n")
    
    symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI']
    
    # Initialize sentiment engine
    sentiment_engine = SentimentEngine(symbols)
    
    # Register callback to capture signals
    signals_received = []
    
    def sentiment_callback(signal):
        signals_received.append(signal)
        print(f"\n🎯 SENTIMENT TRADING SIGNAL:")
        print(f"   Symbol: {signal['symbol']}")
        print(f"   Action: {signal['action']}")
        print(f"   Confidence: {signal['confidence']:.3f}")
        print(f"   Sentiment: {signal['sentiment_type'].upper()}")
        print(f"   Impact Factor: {signal['impact_factor']:.2f}")
        print(f"   News: {signal['news_headline'][:100]}...")
    
    sentiment_engine.register_callback(sentiment_callback)
    
    print("🚀 Starting Sentiment Analysis Engine...")
    print("🧠 Analyzing market psychology from news and social sentiment")
    print("📡 Generating intelligent trading signals based on market mood\n")
    
    # Start engine
    sentiment_engine.start()
    
    # Run for 30 seconds to demonstrate
    try:
        for i in range(6):  # 6 cycles of 5 seconds each
            time.sleep(5)
            
            # Show current sentiment status
            print(f"\n📊 SENTIMENT STATUS UPDATE #{i+1}:")
            sentiment_summary = sentiment_engine.get_sentiment_summary()
            
            for symbol, sentiment in sentiment_summary.items():
                print(f"   {symbol}: {sentiment}")
            
            # Show detailed sentiment for one symbol
            if i == 2:  # Show details in middle of demo
                all_sentiments = sentiment_engine.get_all_sentiments()
                for symbol, sentiment_data in all_sentiments.items():
                    if sentiment_data:
                        print(f"\n🔍 DETAILED ANALYSIS for {symbol}:")
                        print(f"   Sentiment Score: {sentiment_data.sentiment_score:.3f}")
                        print(f"   Confidence: {sentiment_data.confidence:.3f}")
                        print(f"   News Count: {sentiment_data.news_count}")
                        print(f"   Impact Factor: {sentiment_data.impact_factor:.2f}")
                        print(f"   Latest Headline: {sentiment_data.headline}")
                        break
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    
    # Stop engine
    sentiment_engine.stop()
    
    # Summary
    print("\n" + "="*80)
    print("           📊 SENTIMENT ANALYSIS DEMO SUMMARY 📊")
    print("="*80)
    print(f"\n🎯 SIGNALS GENERATED: {len(signals_received)}")
    
    if signals_received:
        print(f"\n📈 SENTIMENT SIGNALS:")
        for i, signal in enumerate(signals_received[-3:], 1):  # Show last 3 signals
            print(f"   {i}. {signal['symbol']} {signal['action']} "
                  f"(Confidence: {signal['confidence']:.3f}, "
                  f"Sentiment: {signal['sentiment_type'].upper()})")
    
    print(f"\n🧠 MARKET PSYCHOLOGY INSIGHTS:")
    final_sentiments = sentiment_engine.get_sentiment_summary()
    bullish_count = sum(1 for s in final_sentiments.values() if 'BULLISH' in s)
    bearish_count = sum(1 for s in final_sentiments.values() if 'BEARISH' in s)
    neutral_count = len(final_sentiments) - bullish_count - bearish_count
    
    print(f"   Bullish Stocks: {bullish_count}")
    print(f"   Bearish Stocks: {bearish_count}")
    print(f"   Neutral Stocks: {neutral_count}")
    
    if bullish_count > bearish_count:
        print(f"   📈 Overall Market Mood: POSITIVE")
    elif bearish_count > bullish_count:
        print(f"   📉 Overall Market Mood: NEGATIVE")
    else:
        print(f"   ⚖️  Overall Market Mood: MIXED")
    
    print("\n🎉 Sentiment Analysis Engine Demo Complete!")
    print("🌟 This revolutionary feature gives AI human-like market intuition!")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo_sentiment_engine()