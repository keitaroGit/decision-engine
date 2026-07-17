import os, requests, json, time, re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

ALPHA_KEY  = os.environ.get('ALPHA_VANTAGE_KEY', '')
FRED_KEY   = os.environ.get('FRED_API_KEY', '')
CLAUDE_KEY = os.environ.get('CLAUDE_API_KEY', '')
NEWS_KEY  = os.environ.get('NEWS_API_KEY', '')

client = Anthropic(api_key=CLAUDE_KEY)

cache = {}
CACHE_TTL = 900

def cached(key, fn):
    now = time.time()
    if key in cache and now - cache[key]['ts'] < CACHE_TTL:
        return cache[key]['data']
    data = fn()
    cache[key] = {'data': data, 'ts': now}
    return data

def av(params):
    params['apikey'] = ALPHA_KEY
    try:
        r = requests.get('https://www.alphavantage.co/query', params=params, timeout=10)
        return r.json()
    except:
        return {}

def get_quote(ticker):
    def fn():
        d = av({'function': 'GLOBAL_QUOTE', 'symbol': ticker})
        q = d.get('Global Quote', {})
        return {
            'price':      q.get('05. price', 'N/A'),
            'change_pct': q.get('10. change percent', 'N/A'),
        }
    return cached('quote_' + ticker, fn)

def get_overview(ticker):
    def fn():
        d = av({'function': 'OVERVIEW', 'symbol': ticker})
        return {
            'name':           d.get('Name', ticker),
            'sector':         d.get('Sector', 'N/A'),
            'pe_ratio':       d.get('PERatio', 'N/A'),
            'pb_ratio':       d.get('PriceToBookRatio', 'N/A'),
            'ps_ratio':       d.get('PriceToSalesRatioTTM', 'N/A'),
            'profit_margin':  d.get('ProfitMargin', 'N/A'),
            'op_margin':      d.get('OperatingMarginTTM', 'N/A'),
            'revenue_growth': d.get('QuarterlyRevenueGrowthYOY', 'N/A'),
            'eps_growth':     d.get('QuarterlyEarningsGrowthYOY', 'N/A'),
            'eps_ttm':        d.get('EPS', 'N/A'),
            'beta':           d.get('Beta', 'N/A'),
            'mkt_cap':        d.get('MarketCapitalization', 'N/A'),
            '52w_high':       d.get('52WeekHigh', 'N/A'),
            '52w_low':        d.get('52WeekLow', 'N/A'),
            'analyst_target': d.get('AnalystTargetPrice', 'N/A'),
        }
    return cached('overview_' + ticker, fn)

def get_earnings(ticker):
    def fn():
        d = av({'function': 'EARNINGS', 'symbol': ticker})
        quarters = d.get('quarterlyEarnings', [])[:3]
        return [{'date': q.get('fiscalDateEnding',''), 'surprise': q.get('surprisePercentage','')} for q in quarters]
    return cached('earnings_' + ticker, fn)

def fred_val(series_id):
    def fn():
        try:
            r = requests.get('https://api.stlouisfed.org/fred/series/observations',
                params={'series_id': series_id, 'api_key': FRED_KEY, 'file_type': 'json', 'limit': 1, 'sort_order': 'desc'},
                timeout=8)
            obs = r.json().get('observations', [])
            return obs[0].get('value', 'N/A') if obs else 'N/A'
        except:
            return 'N/A'
    return cached('fred_' + series_id, fn)

def get_macro():
    return {
        'us10y':  fred_val('DGS10'),
        'oil':    fred_val('DCOILWTICO'),
        'usdyen': fred_val('DEXJPUS'),
    }

# Japanese translation tables
VERDICT_JA = {
    'STRONG BUY': '強い買い',
    'BUY': '買い',
    'WATCH': '様子見',
    'PASS': '見送り',
    'STRONG PASS': '強い 見送り',
}

SIGNAL1_JA = {'TAILWIND': '追い風', 'NEUTRAL': '中立', 'HEADWIND': '向かい風'}
SIGNAL2_JA = {'STRONG': '強い', 'NEUTRAL': '中立', 'WEAK': '弱い'}
SIGNAL3_JA = {'UNDERVALUED': '割安', 'FAIR': '適正', 'OVERVALUED': '割高'}


def get_news(ticker, company_name):
    """Fetch latest news for ticker from NewsAPI"""
    def fn():
        try:
            # Search by company name for better results
            query = company_name if company_name and company_name != ticker else ticker
            r = requests.get('https://newsapi.org/v2/everything', params={
                'q': query,
                'apiKey': NEWS_KEY,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 5,
            }, timeout=8)
            articles = r.json().get('articles', [])
            news = []
            for a in articles:
                title = a.get('title', '')
                source = a.get('source', {}).get('name', '')
                date = a.get('publishedAt', '')[:10]
                if title and '[Removed]' not in title:
                    news.append(date + ' [' + source + '] ' + title)
            return news[:5]
        except:
            return []
    return cached('news_' + ticker, fn)

SYSTEM_PROMPT = """You are an investment analyst. Output ONLY a JSON object with NO Japanese text anywhere.
All values must use ASCII characters only. No Unicode, no special chars, no curly quotes, no em dashes.

Required JSON structure (fill in real analysis):
{
  "verdict": "BUY",
  "confidence": 72,
  "summary_en": "Short plain English summary under 150 chars",
  "layer1": {
    "score": 6,
    "signal": "NEUTRAL",
    "points": ["point one under 80 chars", "point two under 80 chars"]
  },
  "layer2": {
    "score": 7,
    "signal": "STRONG",
    "points": ["point one", "point two"]
  },
  "layer3": {
    "score": 5,
    "signal": "FAIR",
    "points": ["point one", "point two"]
  },
  "distortion_found": false,
  "distortion_en": null,
  "risks": ["risk one", "risk two"],
  "catalysts": ["catalyst one", "catalyst two"]
}

verdict must be one of: STRONG BUY, BUY, WATCH, PASS, STRONG PASS
layer1 signal: TAILWIND, NEUTRAL, or HEADWIND
layer2 signal: STRONG, NEUTRAL, or WEAK
layer3 signal: UNDERVALUED, FAIR, or OVERVALUED
Output ONLY the JSON. No text before or after."""

def run_analysis(ticker, overview, quote, earnings, macro, horizon='mid', lang='en', news=None):
    horizon_map = {
        'short': 'SHORT TERM 1-3 months: weight technicals and near-term catalysts most',
        'mid':   'MID TERM 3-12 months: weight earnings momentum and margins most',
        'long':  'LONG TERM 1-3 years: weight competitive moat and macro cycle most',
    }
    htext = horizon_map.get(horizon, horizon_map['mid'])

    prompt = "\n".join([
        "TICKER: " + ticker + " | HORIZON: " + htext,
        "Sector: " + str(overview.get('sector','')),
        "Price: $" + str(quote.get('price','')) + " | Change: " + str(quote.get('change_pct','')),
        "52W: High=" + str(overview.get('52w_high','')) + " Low=" + str(overview.get('52w_low','')) + " AnalystTarget=" + str(overview.get('analyst_target','')),
        "Valuation: PE=" + str(overview.get('pe_ratio','')) + " PB=" + str(overview.get('pb_ratio','')) + " PS=" + str(overview.get('ps_ratio','')),
        "Fundamentals: EPS=" + str(overview.get('eps_ttm','')) + " EPSgrowth=" + str(overview.get('eps_growth','')) + " RevGrowth=" + str(overview.get('revenue_growth','')),
        "Margins: OpMargin=" + str(overview.get('op_margin','')) + " ProfitMargin=" + str(overview.get('profit_margin','')),
        "Beta=" + str(overview.get('beta','')) + " MktCap=" + str(overview.get('mkt_cap','')),
        "Recent EPS surprises: " + str(earnings),
        "MACRO: US10Y=" + str(macro.get('us10y','')) + "% Oil=$" + str(macro.get('oil','')) + " USD/JPY=" + str(macro.get('usdyen','')),
        "",
        ("LATEST NEWS (use this for current competitive landscape and recent events):\n" + "\n".join(news) + "\n" if news else "") +
        "Output JSON only." + (" IMPORTANT: Write ALL text values (summary_en, all points array items, risks, catalysts, distortion_en) in JAPANESE language. Keep JSON keys in English." if lang=="ja" else " Use English for all text values."),
    ])

    try:
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = msg.content[0].text.strip()

        # Extract JSON boundaries
        start = raw.find('{')
        end = raw.rfind('}')
        if start == -1 or end == -1:
            return {'error': 'No JSON in response'}
        text = raw[start:end+1]

        # Force ASCII only
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Remove control chars
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        data = json.loads(text)

        # Add Japanese translations server-side
        verdict = data.get('verdict', 'WATCH')
        data['verdict_ja'] = VERDICT_JA.get(verdict, verdict)

        l1 = data.get('layer1', {})
        l2 = data.get('layer2', {})
        l3 = data.get('layer3', {})

        l1['signal_ja'] = SIGNAL1_JA.get(l1.get('signal',''), l1.get('signal',''))
        l2['signal_ja'] = SIGNAL2_JA.get(l2.get('signal',''), l2.get('signal',''))
        l3['signal_ja'] = SIGNAL3_JA.get(l3.get('signal',''), l3.get('signal',''))

        return data

    except json.JSONDecodeError as e:
        return {'error': 'JSON parse error: ' + str(e)}
    except Exception as e:
        return {'error': str(e)}


def translate_to_japanese(data):
    """Translate analysis text fields to Japanese using Claude"""
    fields_to_translate = {
        'summary': data.get('summary_en', ''),
        'layer1_points': data.get('layer1', {}).get('points', []),
        'layer2_points': data.get('layer2', {}).get('points', []),
        'layer3_points': data.get('layer3', {}).get('points', []),
        'risks': data.get('risks', []),
        'catalysts': data.get('catalysts', []),
        'distortion': data.get('distortion_en', ''),
    }

    prompt = "Translate these investment analysis texts to natural Japanese. Return ONLY a JSON object with the same keys. Keep numbers, tickers, and percentages as-is.\n\n" + json.dumps(fields_to_translate, ensure_ascii=True)

    try:
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=1000,
            system="You are a financial translator. Translate English investment analysis to Japanese. Return ONLY a valid JSON object with the same structure as input. No other text.",
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = msg.content[0].text.strip()
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            translated = json.loads(raw[start:end+1])
            # Apply translations
            if 'summary' in translated:
                data['summary_ja'] = translated['summary']
            if 'layer1_points' in translated:
                data['layer1']['points_ja'] = translated['layer1_points']
            if 'layer2_points' in translated:
                data['layer2']['points_ja'] = translated['layer2_points']
            if 'layer3_points' in translated:
                data['layer3']['points_ja'] = translated['layer3_points']
            if 'risks' in translated:
                data['risks_ja'] = translated['risks']
            if 'catalysts' in translated:
                data['catalysts_ja'] = translated['catalysts']
            if 'distortion' in translated and translated['distortion']:
                data['distortion_ja'] = translated['distortion']
    except Exception as e:
        pass  # Keep English if translation fails
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data    = request.get_json()
    ticker  = data.get('ticker', '').upper().strip()
    horizon = data.get('horizon', 'mid')
    lang    = data.get('lang', 'en')

    if not ticker:
        return jsonify({'error': 'Ticker required'}), 400

    overview = get_overview(ticker)
    quote    = get_quote(ticker)
    earnings = get_earnings(ticker)
    macro    = get_macro()

    # Only reject if Alpha Vantage returns completely empty data
    if not overview.get('name') or overview.get('name') == '':
        return jsonify({'error': 'Ticker not found: ' + ticker}), 404

    news     = get_news(ticker, overview.get('name', ticker))
    result = run_analysis(ticker, overview, quote, earnings, macro, horizon, lang, news)

    if 'error' in result:
        return jsonify(result), 500

    # Translate to Japanese if requested
    if lang == 'ja':
        result = translate_to_japanese(result)

    return jsonify({
        'ticker':   ticker,
        'name':     overview.get('name'),
        'price':    quote.get('price'),
        'change':   quote.get('change_pct'),
        'horizon':  horizon,
        'analysis': result,
        'raw': {
            'news': news,
            'overview': overview,
            'quote':    quote,
            'earnings': earnings,
            'macro':    macro,
        }
    })

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
