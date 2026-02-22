# Chapter 7: Core Vol Strategies

## The Foundation of Volatility Trading

Every sophisticated volatility trader begins with mastering the core strategies that form the building blocks of volatility trading. These fundamental approaches—long volatility strategies like straddles and strangles, short volatility strategies like iron condors and credit spreads, and dynamic strategies like gamma scalping—provide the essential toolkit for expressing volatility views in the markets.

This chapter explores each core strategy in detail: their construction, risk characteristics, optimal market conditions, and practical implementation considerations. Understanding these strategies deeply is crucial because they form the foundation upon which more complex volatility trading approaches are built.

## Long Volatility Strategies: Betting on Movement

Long volatility strategies profit when the underlying asset moves more than the market expects, regardless of direction. These strategies are the preferred tools for traders who believe implied volatility is too low or who need portfolio protection against unexpected events.

### Long Straddles: The Pure Volatility Play

The long straddle is the most fundamental volatility strategy, providing pure exposure to volatility changes without directional bias.

**Construction**:
```
Long Straddle = Buy Call + Buy Put (same strike, same expiration)
Typically executed at-the-money (ATM)
```

**Example: SPX Long Straddle**:
```
SPX Price: 4,000
Buy 4,000 Call @ 65
Buy 4,000 Put @ 65
Total Cost: 130 points ($13,000 per straddle)
```

**Payoff Characteristics**:
```
Maximum Loss: Premium paid ($13,000)
Breakeven Points: Strike ± Premium (3,870 and 4,130)
Maximum Profit: Unlimited
Profit Zone: |Stock Move| > Premium paid
```

**Greeks Profile**:
- **Delta**: 0 (initially delta-neutral)
- **Gamma**: High positive gamma
- **Vega**: High positive vega
- **Theta**: High negative theta

### When to Use Long Straddles

**Ideal Market Conditions**:
1. **Low implied volatility**: Cheaper premium, better risk/reward
2. **Expected volatility increase**: Earnings, FOMC meetings, news events
3. **Uncertain direction**: High conviction on movement, no directional bias
4. **Mean reversion setup**: Volatility significantly below historical averages

**Timing Considerations**:
```
Entry Timing:
- 30-45 days before known events
- When VIX is below 20th percentile
- After sustained low volatility periods

Exit Timing:  
- Volatility spike (profit taking)
- 50% of time decay passed
- Breakeven approached with limited time
```

### Long Straddle Management

**Delta Management**:
As the underlying moves, the straddle becomes directionally exposed:
```
If SPX moves to 4,100:
Call delta increases to ~0.70
Put delta decreases to ~-0.30
Net delta: +0.40 (long exposure)

Management options:
1. Sell stock to rebalance to neutral
2. Close winning side, keep losing side
3. Roll losing side closer to ATM
```

**Volatility Management**:
```
Vol Increase Scenarios:
- Take profits on 50-100% vol increase
- Partial profit-taking to reduce risk
- Roll to next expiration if time remains

Vol Decrease Scenarios:
- Stop loss at 50% of premium
- Convert to calendar spread
- Wait for mean reversion if early
```

### Long Strangles: Asymmetric Volatility Exposure

Long strangles modify the straddle concept by using different strikes, typically out-of-the-money options.

**Construction**:
```
Long Strangle = Buy OTM Call + Buy OTM Put
Strikes equidistant from current price
```

**Example: SPX Long Strangle**:
```
SPX Price: 4,000
Buy 4,100 Call @ 35
Buy 3,900 Put @ 35
Total Cost: 70 points ($7,000 per strangle)
```

**Advantages vs. Straddles**:
- **Lower cost**: OTM options are cheaper
- **Wider breakeven zone**: Requires larger move but cheaper entry
- **Better risk/reward**: Higher percentage returns possible
- **Skew benefits**: Can exploit volatility skew differences

**Disadvantages**:
- **Higher movement threshold**: Needs bigger move to profit
- **Time decay**: Faster decay if price stays between strikes
- **Gamma risk**: Less gamma to benefit from volatility

### Strangle Strike Selection

The choice of strikes significantly affects the strategy's behavior:

**Narrow Strangles** (strikes close to ATM):
- Higher cost, lower movement requirement
- More gamma exposure
- Better for moderate volatility increases

**Wide Strangles** (strikes far from ATM):
- Lower cost, higher movement requirement  
- Less gamma exposure
- Better for extreme volatility events

**Optimization Framework**:
```
Strike Selection = f(Implied Vol, Historical Vol, Time to Expiration, 
                     Expected Move, Risk Budget, Skew)

Common approaches:
- 16-delta strikes (1 standard deviation)
- 25-delta strikes (more conservative)
- Fixed percentage OTM (5%, 10%, 15%)
```

### Earnings Straddles and Strangles

Earnings announcements represent one of the most common applications for long volatility strategies:

**Earnings Vol Dynamics**:
```
Pre-Earnings: High implied vol (uncertainty premium)
Post-Earnings: Vol crush (uncertainty resolved)
Typical vol drop: 20-40% overnight
```

**Strategy Considerations**:
- **Buy 2-4 weeks before**: Avoid last-minute premium
- **Expected move calculation**: Use straddle price as guide
- **Historical analysis**: Compare actual vs. expected moves
- **Sector effects**: Some sectors show more consistent patterns

**Risk Management**:
```
Position Sizing: Limited to 2-5% of portfolio per position
Stop Loss: 50% of premium if vol decreases before earnings
Profit Taking: 100%+ gains if vol increases before announcement
```

## Short Volatility Strategies: Harvesting the Vol Premium

Short volatility strategies profit from the tendency of implied volatility to exceed realized volatility over time. These strategies harvest the "volatility risk premium" but require careful risk management due to their negative convexity.

### Short Straddles: The Volatility Seller's Tool

Short straddles are the mirror image of long straddles, profiting when the underlying remains near the strike price.

**Construction**:
```
Short Straddle = Sell Call + Sell Put (same strike, same expiration)
```

**Risk Profile**:
```
Maximum Profit: Premium received
Maximum Loss: Unlimited (theoretically)
Profit Zone: |Stock Move| < Premium received
Risk Management: Essential due to unlimited risk
```

**Greeks Profile**:
- **Delta**: 0 (initially neutral)
- **Gamma**: High negative gamma (accelerating losses)
- **Vega**: High negative vega (profits from vol decrease)
- **Theta**: High positive theta (time decay benefit)

### When to Use Short Straddles

**Optimal Conditions**:
1. **High implied volatility**: Premium rich environment
2. **Expected vol decrease**: After events, vol spikes
3. **Range-bound expectations**: Low expected movement
4. **Contango environment**: VIX futures above spot

**Timing Framework**:
```
Entry Signals:
- VIX above 80th percentile
- Post-event vol crush opportunities
- Overbought volatility indicators
- Strong contango in vol term structure

Exit Signals:
- 75% of maximum profit achieved
- Vol spike threatening position
- Approaching expiration with losses
- Technical breakdown in underlying
```

### Short Straddle Management

**Delta Hedging**:
```python
def delta_hedge_short_straddle(position_delta, hedge_threshold=500):
    if abs(position_delta) > hedge_threshold:
        hedge_shares = -position_delta
        return hedge_shares
    return 0

# Example hedging decision
position_delta = -1200  # Short 1200 delta
hedge_shares = delta_hedge_short_straddle(position_delta, 500)
# Result: Buy 1200 shares to neutralize delta
```

**Volatility Stop Loss**:
```
Vol Stop Guidelines:
- Exit if vol increases >30% from entry
- Daily vol increase >5% for 2 consecutive days
- VIX spike >25% in single session
- Technical breakdown in underlying support/resistance
```

### Iron Condors: Limited Risk Vol Selling

Iron condors provide a defined-risk alternative to naked short straddles by adding long options to cap potential losses.

**Construction**:
```
Iron Condor = Short Strangle + Long Strangle (wider strikes)

Components:
- Sell OTM Call
- Buy Further OTM Call
- Sell OTM Put  
- Buy Further OTM Put
```

**Example: SPX Iron Condor**:
```
SPX Price: 4,000

Sell 4,100 Call @ 35
Buy 4,200 Call @ 15
Sell 3,900 Put @ 35
Buy 3,800 Put @ 15

Net Credit: 40 points ($4,000)
Max Risk: 100 - 40 = 60 points ($6,000)
Max Reward: 40 points ($4,000)
Risk/Reward Ratio: 1.5:1
```

**Strike Selection Strategies**:

**Standard Iron Condor** (16-delta short strikes):
- Probability of success: ~68%
- Requires movement <1 standard deviation
- Balanced risk/reward profile

**Wide Iron Condor** (10-delta short strikes):
- Higher probability of success: ~80%
- Lower premium collected
- Conservative approach

**Narrow Iron Condor** (25-delta short strikes):
- Lower probability of success: ~50%
- Higher premium collected
- Aggressive approach

### Credit Spreads: Directional Vol Selling

Credit spreads combine volatility selling with mild directional bias, offering another tool for harvesting volatility premium.

**Bull Put Spread Construction**:
```
Bull Put Spread = Sell Higher Strike Put + Buy Lower Strike Put
Bullish bias with volatility selling component
```

**Example: SPX Bull Put Spread**:
```
SPX Price: 4,000

Sell 3,950 Put @ 45
Buy 3,900 Put @ 30
Net Credit: 15 points ($1,500)
Max Risk: 50 - 15 = 35 points ($3,500)
Max Reward: 15 points ($1,500)
Breakeven: 3,935
```

**Bear Call Spread Construction**:
```
Bear Call Spread = Sell Lower Strike Call + Buy Higher Strike Call
Bearish bias with volatility selling component
```

### Advanced Short Vol Techniques

**Rolling Strategies**:
```
Rolling Up: Move strikes higher as underlying rises
Rolling Down: Move strikes lower as underlying falls
Rolling Out: Extend expiration to collect more time premium

Rolling Decision Framework:
- Roll for additional credit
- Maintain probability of success >60%
- Avoid rolling into low-probability scenarios
```

**Ratio Spreads**:
```
Put Ratio Spread = Buy 1 ATM Put + Sell 2 OTM Puts
Call Ratio Spread = Buy 1 ATM Call + Sell 2 OTM Calls

Benefits:
- Positive theta from extra short option
- Profit from sideways movement
- Lower initial cost

Risks:
- Unlimited risk beyond short strikes
- Negative gamma exposure
- Complex risk management
```

## Gamma Scalping: The Dynamic Vol Strategy

Gamma scalping represents the active approach to volatility trading, attempting to profit from the difference between implied and realized volatility through dynamic hedging.

### Gamma Scalping Fundamentals

**Concept**:
Buy options (positive gamma) and hedge the delta continuously, profiting from the rebalancing process when realized volatility exceeds implied volatility.

**Theoretical Framework**:
```
Gamma P&L = 0.5 × Gamma × (Stock Move)²
Realized Vol = √(252 × Average(Daily Returns²))

Profit Condition: Realized Vol > Implied Vol - Transaction Costs
```

**Basic Process**:
1. Establish positive gamma position (long options)
2. Delta hedge by trading underlying
3. Rebalance hedge as delta changes
4. Profit from buying low/selling high through rebalancing
5. Close before theta erosion exceeds gamma gains

### Gamma Scalping Execution

**Position Setup**:
```python
def setup_gamma_scalp(underlying_price, target_gamma, days_to_expiry):
    # Select optimal strike (usually ATM)
    strike = round_to_strike(underlying_price)
    
    # Calculate required option quantity
    option_gamma = calculate_gamma(strike, days_to_expiry)
    quantity = target_gamma / option_gamma
    
    # Initial delta hedge
    option_delta = calculate_delta(strike, days_to_expiry)
    hedge_shares = -quantity * option_delta
    
    return quantity, hedge_shares
```

**Rebalancing Rules**:
```python
def rebalancing_rules(current_delta, last_hedge_delta, threshold=0.10):
    delta_change = abs(current_delta - last_hedge_delta)
    
    # Rebalance triggers
    if delta_change > threshold:
        return True
    elif time_since_last_hedge > max_time_interval:
        return True
    elif volatility_spike_detected():
        return True
    else:
        return False
```

### Gamma Scalping Parameters

**Rebalancing Frequency**:
```
High Frequency (every 0.05 delta move):
- Better gamma capture
- Higher transaction costs
- Requires tight spreads

Low Frequency (every 0.20 delta move):
- Lower transaction costs
- Reduced gamma capture
- Risk of missing moves
```

**Optimal Strike Selection**:
- **ATM options**: Maximum gamma, highest theta
- **Slightly OTM**: Lower cost, decent gamma
- **Multiple strikes**: Diversified gamma exposure

**Time to Expiration**:
```
Short-term (1-2 weeks):
- High gamma, high theta
- Requires active management
- Best for high vol periods

Medium-term (4-8 weeks):
- Balanced gamma/theta trade-off
- Less management intensive
- Good for consistent strategies

Long-term (>8 weeks):
- Low gamma, low theta
- Stable positions
- Capital intensive
```

### Advanced Gamma Scalping

**Volatility Forecasting Integration**:
```python
def dynamic_rebalancing_threshold(forecasted_vol, current_vol):
    vol_ratio = forecasted_vol / current_vol
    
    if vol_ratio > 1.2:  # Expect higher vol
        return 0.05  # More frequent rebalancing
    elif vol_ratio < 0.8:  # Expect lower vol
        return 0.15  # Less frequent rebalancing
    else:
        return 0.10  # Standard threshold
```

**Multi-Asset Gamma Scalping**:
- Scale gamma across multiple underlyings
- Diversification benefits
- Correlation risk management
- Portfolio-level optimization

**Time-of-Day Effects**:
```
Market Open (9:30-10:30 AM):
- High volatility, wide spreads
- Reduced rebalancing frequency
- Wait for market settle

Mid-Day (10:30 AM - 3:00 PM):
- Normal volatility, tight spreads
- Standard rebalancing rules
- Optimal gamma scalping period

Market Close (3:00-4:00 PM):
- Increased volatility
- Position management focus
- Overnight risk considerations
```

## Delta-Hedging Strategies

Delta hedging forms the foundation of most professional volatility trading, transforming directional positions into pure volatility plays.

### Delta-Hedging Fundamentals

**Objective**: Eliminate directional risk to isolate volatility exposure

**Basic Process**:
```
1. Calculate position delta
2. Trade underlying to neutralize delta
3. Monitor and rebalance as delta changes
4. Profit from pure volatility exposure
```

**Delta Calculation**:
```python
def calculate_portfolio_delta(positions):
    total_delta = 0
    for position in positions:
        option_delta = get_option_delta(position.strike, position.expiry)
        total_delta += position.quantity * option_delta
    return total_delta
```

### Static vs. Dynamic Hedging

**Static Hedging**:
- Hedge set at position initiation
- No rebalancing unless major changes
- Lower transaction costs
- Accepts some directional risk

**Dynamic Hedging**:
- Continuous delta monitoring and rebalancing
- Attempts to maintain delta neutrality
- Higher transaction costs
- Better volatility isolation

### Hedging Frequency Optimization

**Threshold-Based Hedging**:
```python
def should_rebalance(current_delta, target_delta=0, threshold=1000):
    delta_deviation = abs(current_delta - target_delta)
    return delta_deviation > threshold
```

**Time-Based Hedging**:
```python
def time_based_rebalance(last_hedge_time, rebalance_interval='1H'):
    time_since_hedge = current_time - last_hedge_time
    return time_since_hedge >= rebalance_interval
```

**Volatility-Based Hedging**:
```python
def vol_based_rebalance(current_vol, last_vol, vol_threshold=0.02):
    vol_change = abs(current_vol - last_vol) / last_vol
    return vol_change > vol_threshold
```

### Transaction Cost Management

**Cost Components**:
```
Total Cost = Bid-Ask Spread + Commission + Market Impact + Slippage

Bid-Ask Cost: 0.01-0.05% for liquid stocks
Commission: $0.50-1.00 per contract
Market Impact: Function of size and liquidity
Slippage: 0.01-0.10% depending on execution
```

**Cost Optimization**:
```python
def optimize_hedge_size(required_hedge, min_trade_size, cost_per_share):
    # Only hedge if cost-benefit is positive
    expected_benefit = calculate_risk_reduction(required_hedge)
    total_cost = abs(required_hedge) * cost_per_share
    
    if expected_benefit > total_cost and abs(required_hedge) >= min_trade_size:
        return required_hedge
    else:
        return 0  # Skip rebalancing
```

## Strategy Selection Framework

### Market Regime Analysis

**Low Volatility Regime** (VIX < 15):
- **Preferred strategies**: Long straddles/strangles, gamma scalping
- **Avoid**: Short volatility strategies
- **Rationale**: Cheap volatility, mean reversion opportunity

**Normal Volatility Regime** (VIX 15-25):
- **Preferred strategies**: Iron condors, credit spreads, selective long vol
- **Balanced approach**: Mix of long and short vol strategies
- **Rationale**: Diverse opportunity set

**High Volatility Regime** (VIX > 25):
- **Preferred strategies**: Short straddles, iron condors, volatility selling
- **Avoid**: Long volatility strategies
- **Rationale**: Rich premiums, mean reversion expectations

### Risk Management Integration

**Position Sizing**:
```python
def calculate_position_size(strategy_type, account_size, max_risk_pct):
    if strategy_type == 'long_vol':
        # Limited risk strategies can be sized larger
        return account_size * max_risk_pct * 1.5
    elif strategy_type == 'short_vol':
        # Unlimited risk strategies require smaller sizing
        return account_size * max_risk_pct * 0.5
    else:
        return account_size * max_risk_pct
```

**Portfolio Correlation**:
```python
def adjust_for_correlation(base_position_size, existing_positions, correlation):
    correlation_adjustment = 1 - (correlation * position_overlap_factor)
    adjusted_size = base_position_size * correlation_adjustment
    return max(adjusted_size, minimum_position_size)
```

## Key Takeaways

1. **Long volatility strategies** provide portfolio protection and profit from volatility increases but suffer from time decay
2. **Short volatility strategies** harvest the volatility risk premium but require sophisticated risk management
3. **Gamma scalping** bridges the gap between implied and realized volatility through dynamic hedging
4. **Delta hedging** is essential for isolating volatility exposure from directional market moves
5. **Strategy selection** should be based on volatility regime, market conditions, and risk management capabilities
6. **Transaction costs** can significantly impact strategy profitability and must be carefully managed
7. **Risk management** is paramount, especially for strategies with unlimited loss potential

These core strategies form the foundation of volatility trading. Mastering their construction, management, and optimization is essential before progressing to more advanced volatility techniques. In the next chapter, we'll explore sophisticated strategies that build upon these fundamentals.

---

*"Core volatility strategies are like musical scales—master them completely, and you have the foundation to create any complex composition in the volatility markets."*