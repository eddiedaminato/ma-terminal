"""
MA Terminal - Strategic Financial Intelligence

Professional equity research automation tool for comprehensive financial analysis.

Features:
- Automated data extraction from Yahoo Finance
- Advanced financial metrics (Valuation, Growth, Health Score)
- Performance visualization with professional charts
- PDF report generation with executive summary

Author: Eddie
Version: 1.2.5
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from typing import Optional, List, Dict, Tuple
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# 1. DATA EXTRACTION & PROCESSING

def get_financial_data(ticker_symbol: str, years: int = 10) -> Optional[pd.DataFrame]:

    try:
        ticker = yf.Ticker(ticker_symbol)

        # Fetch extended historical data
        inc_q = ticker.quarterly_financials
        bs_q = ticker.quarterly_balance_sheet
        cf_q = ticker.quarterly_cashflow

        # Use annual data as primary source
        inc = ticker.financials
        bs = ticker.balance_sheet
        cf = ticker.cashflow

        # Validate data availability
        if inc.empty or bs.empty or cf.empty:
            return None

        def fetch_row(df: pd.DataFrame, names: List[str]) -> pd.Series:

            for name in names:
                if name in df.index:
                    return df.loc[name]
            return pd.Series(0, index=df.columns)

        # Extract key financial metrics
        data_dict = {
            "Revenue": fetch_row(inc, ['Total Revenue', 'Operating Revenue']),
            "EBIT": fetch_row(inc, ['EBIT', 'Operating Income']),
            "Net_Income": fetch_row(inc, ['Net Income']),
            "Tax_Provision": fetch_row(inc, ['Tax Provision', 'Income Tax Expense']),
            "Interest_Expense": fetch_row(inc, ['Interest Expense']),
            "Capex": fetch_row(cf, ['Capital Expenditure']),
            "Depreciation": fetch_row(cf, ['Depreciation And Amortization']),
            "Operating_Cash_Flow": fetch_row(cf, ['Operating Cash Flow']),
            "Total_Debt": fetch_row(bs, ['Total Debt']),
            "Cash": fetch_row(bs, [
                'Cash Cash Equivalents And Short Term Investments',
                'Cash And Cash Equivalents'
            ]),
            "Total_Assets": fetch_row(bs, ['Total Assets']),
            "Total_Equity": fetch_row(bs, ['Stockholders Equity', 'Total Equity Gross Minority Interest']),
            "Current_Assets": fetch_row(bs, ['Current Assets']),
            "Current_Liabilities": fetch_row(bs, ['Current Liabilities'])
        }

        df = pd.DataFrame(data_dict)

        # Process dates: extract year only
        df.index = pd.to_datetime(df.index).year.astype(str)

        # Remove duplicates (keep most recent data for each year)
        df = df[~df.index.duplicated(keep='first')]

        # Remove years with insufficient data
        df = df[df['Revenue'].notna() & (df['Revenue'] != 0)]

        # Limit to requested number of years
        df = df.tail(years)

        # Calculate derived metrics
        df['EBIT_Margin (%)'] = (df['EBIT'] / df['Revenue']) * 100
        df['Net_Margin (%)'] = (df['Net_Income'] / df['Revenue']) * 100
        df['FCF'] = df['Operating_Cash_Flow'] + df['Capex']  # Capex is negative
        df['Net_Debt'] = df['Total_Debt'] - df['Cash']
        df['ROE (%)'] = (df['Net_Income'] / df['Total_Equity']) * 100
        df['Debt_to_Equity'] = df['Total_Debt'] / df['Total_Equity']
        df['Current_Ratio'] = df['Current_Assets'] / df['Current_Liabilities']

        # Sort chronologically (oldest to newest)
        return df.sort_index(ascending=True).round(2)

    except Exception as e:
        print(f"[ERROR] Failed to retrieve data for {ticker_symbol}: {str(e)}")
        return None


def get_market_data(ticker_symbol: str) -> Dict[str, float]:

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        return {
            'market_cap': info.get('marketCap', 0),
            'shares_outstanding': info.get('sharesOutstanding', 0),
            'current_price': info.get('currentPrice', 0),
            'enterprise_value': info.get('enterpriseValue', 0)
        }
    except:
        return {'market_cap': 0, 'shares_outstanding': 0, 'current_price': 0, 'enterprise_value': 0}


# 2. ADVANCED ANALYTICS

def calculate_growth_metrics(df: pd.DataFrame) -> Dict[str, float]:

    metrics = {}

    # Calculate CAGR for Revenue (if we have enough data)
    if len(df) >= 2:
        years = len(df) - 1
        revenue_start = df['Revenue'].iloc[0]
        revenue_end = df['Revenue'].iloc[-1]

        if revenue_start > 0:
            metrics['Revenue_CAGR'] = ((revenue_end / revenue_start) ** (1 / years) - 1) * 100
        else:
            metrics['Revenue_CAGR'] = 0

        # YoY Revenue Growth (most recent year)
        if len(df) >= 2:
            metrics['Revenue_YoY'] = ((df['Revenue'].iloc[-1] / df['Revenue'].iloc[-2]) - 1) * 100

        # EBIT Growth
        ebit_start = df['EBIT'].iloc[0]
        ebit_end = df['EBIT'].iloc[-1]
        if ebit_start > 0:
            metrics['EBIT_CAGR'] = ((ebit_end / ebit_start) ** (1 / years) - 1) * 100
        else:
            metrics['EBIT_CAGR'] = 0

    return metrics


def calculate_health_score(df: pd.DataFrame) -> Tuple[str, float, Dict[str, str]]:

    latest = df.iloc[-1]
    score = 0
    breakdown = {}

    # 1. Profitability (30 points)
    ebit_margin = latest['EBIT_Margin (%)']
    if ebit_margin > 20:
        score += 30
        breakdown['Profitability'] = 'A'
    elif ebit_margin > 15:
        score += 25
        breakdown['Profitability'] = 'B'
    elif ebit_margin > 10:
        score += 20
        breakdown['Profitability'] = 'C'
    elif ebit_margin > 5:
        score += 15
        breakdown['Profitability'] = 'D'
    else:
        score += 10
        breakdown['Profitability'] = 'F'

    # 2. Leverage (25 points)
    debt_to_equity = latest['Debt_to_Equity']
    if debt_to_equity < 0.5:
        score += 25
        breakdown['Leverage'] = 'A'
    elif debt_to_equity < 1.0:
        score += 20
        breakdown['Leverage'] = 'B'
    elif debt_to_equity < 2.0:
        score += 15
        breakdown['Leverage'] = 'C'
    elif debt_to_equity < 3.0:
        score += 10
        breakdown['Leverage'] = 'D'
    else:
        score += 5
        breakdown['Leverage'] = 'F'

    # 3. Liquidity (20 points)
    current_ratio = latest['Current_Ratio']
    if current_ratio > 2.0:
        score += 20
        breakdown['Liquidity'] = 'A'
    elif current_ratio > 1.5:
        score += 16
        breakdown['Liquidity'] = 'B'
    elif current_ratio > 1.0:
        score += 12
        breakdown['Liquidity'] = 'C'
    elif current_ratio > 0.75:
        score += 8
        breakdown['Liquidity'] = 'D'
    else:
        score += 4
        breakdown['Liquidity'] = 'F'

    # 4. Returns (25 points)
    roe = latest['ROE (%)']
    if roe > 20:
        score += 25
        breakdown['Returns'] = 'A'
    elif roe > 15:
        score += 20
        breakdown['Returns'] = 'B'
    elif roe > 10:
        score += 15
        breakdown['Returns'] = 'C'
    elif roe > 5:
        score += 10
        breakdown['Returns'] = 'D'
    else:
        score += 5
        breakdown['Returns'] = 'F'

    # Convert to grade
    if score >= 90:
        grade = 'A'
    elif score >= 80:
        grade = 'B'
    elif score >= 70:
        grade = 'C'
    elif score >= 60:
        grade = 'D'
    else:
        grade = 'F'

    return grade, score, breakdown


def calculate_valuation_metrics(df: pd.DataFrame, market_data: Dict[str, float]) -> Dict[str, float]:

    latest = df.iloc[-1]
    metrics = {}

    # P/E Ratio
    if latest['Net_Income'] > 0 and market_data['market_cap'] > 0:
        metrics['P/E'] = market_data['market_cap'] / latest['Net_Income']
    else:
        metrics['P/E'] = 0

    # EV/EBITDA
    ebitda = latest['EBIT'] + latest['Depreciation']
    if ebitda > 0 and market_data['enterprise_value'] > 0:
        metrics['EV/EBITDA'] = market_data['enterprise_value'] / ebitda
    else:
        metrics['EV/EBITDA'] = 0

    # P/B Ratio
    if latest['Total_Equity'] > 0 and market_data['market_cap'] > 0:
        metrics['P/B'] = market_data['market_cap'] / latest['Total_Equity']
    else:
        metrics['P/B'] = 0

    # FCF Yield
    if market_data['market_cap'] > 0 and latest['FCF'] != 0:
        metrics['FCF_Yield (%)'] = (latest['FCF'] / market_data['market_cap']) * 100
    else:
        metrics['FCF_Yield (%)'] = 0

    return metrics


# 3. ENHANCED VISUALIZATION

def plot_comprehensive_analysis(df: pd.DataFrame, ticker: str, growth_metrics: Dict, health_score: Tuple) -> None:

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Revenue & EBIT Trends
    ax1 = fig.add_subplot(gs[0, :])
    plot_df = df[['Revenue', 'EBIT']] / 1e9
    x = range(len(df))
    width = 0.35

    ax1.bar([i - width/2 for i in x], plot_df['Revenue'], width, label='Revenue', color='#1d3557', edgecolor='white', linewidth=0.7)
    ax1.bar([i + width/2 for i in x], plot_df['EBIT'], width, label='EBIT', color='#a8dadc', edgecolor='white', linewidth=0.7)

    ax1.set_title(f'{ticker} - Revenue & EBIT Performance', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Value (Billions)', fontsize=11)
    ax1.set_xlabel('Fiscal Year', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.index, rotation=0)
    ax1.legend(frameon=False, loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Margins Evolution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df.index, df['EBIT_Margin (%)'], marker='o', linewidth=2.5, color='#457b9d', label='EBIT Margin')
    ax2.plot(df.index, df['Net_Margin (%)'], marker='s', linewidth=2.5, color='#e63946', label='Net Margin')
    ax2.set_title('Profitability Margins', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Margin (%)', fontsize=10)
    ax2.legend(frameon=False)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. Free Cash Flow Trend
    ax3 = fig.add_subplot(gs[1, 1])
    fcf_data = df['FCF'] / 1e9
    colors = ['#2a9d8f' if x > 0 else '#e76f51' for x in fcf_data]
    ax3.bar(df.index, fcf_data, color=colors, edgecolor='white', linewidth=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_title('Free Cash Flow', fontsize=12, fontweight='bold')
    ax3.set_ylabel('FCF (Billions)', fontsize=10)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 4. Financial Health Score
    ax4 = fig.add_subplot(gs[2, 0])
    grade, score, breakdown = health_score
    categories = list(breakdown.keys())
    grades_map = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
    values = [grades_map[breakdown[cat]] for cat in categories]
    colors_grade = ['#2a9d8f' if v >= 4 else '#f4a261' if v >= 3 else '#e76f51' for v in values]

    ax4.barh(categories, values, color=colors_grade, edgecolor='white', linewidth=0.7)
    ax4.set_xlim(0, 5)
    ax4.set_title(f'Financial Health Score: {grade} ({score}/100)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Grade (A=5, F=1)', fontsize=10)
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax4.text(val + 0.1, i, breakdown[cat], va='center', fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # 5. Growth Metrics Summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    growth_text = f"""
    GROWTH ANALYSIS
    {'='*35}

    Revenue CAGR:        {growth_metrics.get('Revenue_CAGR', 0):.2f}%
    Revenue YoY:         {growth_metrics.get('Revenue_YoY', 0):.2f}%
    EBIT CAGR:           {growth_metrics.get('EBIT_CAGR', 0):.2f}%

    LATEST RATIOS
    {'='*35}

    ROE:                 {df.iloc[-1]['ROE (%)']:.2f}%
    Debt/Equity:         {df.iloc[-1]['Debt_to_Equity']:.2f}x
    Current Ratio:       {df.iloc[-1]['Current_Ratio']:.2f}x
    """

    ax5.text(0.1, 0.5, growth_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#f1faee', alpha=0.8))

    plt.savefig("temp_chart.png", dpi=300, bbox_inches='tight')
    plt.show()


# 4. ENHANCED PDF REPORT


def export_to_pdf(df: pd.DataFrame, ticker: str, filename: str,
                  growth_metrics: Dict, valuation: Dict, health_score: Tuple) -> None:

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 12, f"Equity Research Report: {ticker}", ln=True, align='C')
    pdf.set_font("Arial", 'I', 11)
    pdf.cell(0, 8, "MA Terminal - Strategic Financial Intelligence", ln=True, align='C')
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 6, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align='C')
    pdf.ln(5)

    # Executive Summary Box
    grade, score, breakdown = health_score
    pdf.set_fill_color(29, 53, 87)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "EXECUTIVE SUMMARY", ln=True, fill=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=9)

    summary_items = [
        f"Financial Health Grade: {grade} ({score}/100)",
        f"Revenue CAGR ({len(df)}Y): {growth_metrics.get('Revenue_CAGR', 0):.2f}%",
        f"Latest EBIT Margin: {df.iloc[-1]['EBIT_Margin (%)']:.2f}%",
        f"ROE: {df.iloc[-1]['ROE (%)']:.2f}%"
    ]

    for item in summary_items:
        pdf.cell(0, 6, f"  - {item}", ln=True)
    pdf.ln(5)

    # Charts
    pdf.image("temp_chart.png", x=5, y=pdf.get_y(), w=200)
    pdf.ln(140)

    # Valuation Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "VALUATION METRICS", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.ln(2)

    val_data = [
        ["P/E Ratio", f"{valuation.get('P/E', 0):.2f}x"],
        ["EV/EBITDA", f"{valuation.get('EV/EBITDA', 0):.2f}x"],
        ["Price/Book", f"{valuation.get('P/B', 0):.2f}x"],
        ["FCF Yield", f"{valuation.get('FCF_Yield (%)', 0):.2f}%"]
    ]

    for row in val_data:
        pdf.cell(95, 6, row[0], border=1)
        pdf.cell(95, 6, row[1], border=1, align='R')
        pdf.ln()

    pdf.ln(5)

    # Financial Summary Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "FINANCIAL SUMMARY (Last 5 Years)", ln=True)
    pdf.ln(2)

    # Select key metrics for summary
    summary_cols = ['Revenue', 'EBIT', 'Net_Income', 'FCF', 'EBIT_Margin (%)', 'ROE (%)']
    summary_df = df[summary_cols].tail(5)

    pdf.set_font("Arial", 'B', 8)
    col_width = 30
    pdf.cell(35, 7, "Metric", border=1, align='C')
    for year in summary_df.index:
        pdf.cell(col_width, 7, str(year), border=1, align='C')
    pdf.ln()

    pdf.set_font("Arial", size=7)
    for col in summary_cols:
        pdf.cell(35, 6, col, border=1)
        for val in summary_df[col]:
            if "%" in col or "Ratio" in col:
                label = f"{val:.2f}%"
            else:
                label = f"{val / 1e9:.2f}B"
            pdf.cell(col_width, 6, label, border=1, align='R')
        pdf.ln()

    # Footer
    pdf.ln(8)
    pdf.set_font("Arial", 'I', 7)
    pdf.cell(0, 4, "Disclaimer: This report is for informational purposes only. All values in local currency (Billions).", ln=True)
    pdf.cell(0, 4, f"Data Source: Yahoo Finance API | Analysis Engine: MA Terminal v1.2.5", ln=True)

    pdf.output(filename)
    print(f"✔ Enhanced PDF Report exported: {filename}")


# 5. MAIN EXECUTION


def main(ticker_symbol: str) -> None:

    print(f"\n{'='*70}")
    print(f"MA TERMINAL | Initializing Analysis for: {ticker_symbol}")
    print(f"{'='*70}\n")

    # Step 1: Extract financial data
    print(f"[1/5] Extracting financial data...")
    df = get_financial_data(ticker_symbol)

    if df is None:
        print(f"\nERROR: Unable to retrieve data for ticker '{ticker_symbol}'")
        print("Possible reasons: Invalid ticker, delisted company, or API unavailable.\n")
        return

    print(f"✔ Data extracted successfully ({len(df)} fiscal years)\n")

    # Step 2: Get market data for valuation
    print(f"[2/5] Fetching market data...")
    market_data = get_market_data(ticker_symbol)
    print(f"✔ Market data retrieved\n")

    # Step 3: Calculate analytics
    print(f"[3/5] Computing advanced analytics...")
    growth_metrics = calculate_growth_metrics(df)
    health_score = calculate_health_score(df)
    valuation = calculate_valuation_metrics(df, market_data)
    print(f"✔ Analytics computed\n")

    # Step 4: Generate visualization
    print(f"[4/5] Generating comprehensive charts...")
    plot_comprehensive_analysis(df, ticker_symbol, growth_metrics, health_score)
    print(f"✔ Charts generated\n")

    # Step 5: Export PDF
    print(f"[5/5] Exporting enhanced PDF report...")
    report_name = f"Report_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf"
    export_to_pdf(df, ticker_symbol, report_name, growth_metrics, valuation, health_score)

    # Console output
    print(f"\n{'='*70}")
    print(f"INVESTMENT ANALYSIS SUMMARY | TICKER: {ticker_symbol}")
    print(f"{'='*70}\n")

    grade, score, breakdown = health_score
    print(f"Financial Health Score: {grade} ({score}/100)")
    print(f"  -> Profitability: {breakdown['Profitability']} | Leverage: {breakdown['Leverage']}")
    print(f"  -> Liquidity: {breakdown['Liquidity']} | Returns: {breakdown['Returns']}\n")

    print(f"Growth Metrics:")
    print(f"  -> Revenue CAGR: {growth_metrics.get('Revenue_CAGR', 0):.2f}%")
    print(f"  -> Revenue YoY: {growth_metrics.get('Revenue_YoY', 0):.2f}%")
    print(f"  -> EBIT CAGR: {growth_metrics.get('EBIT_CAGR', 0):.2f}%\n")

    print(f"Valuation Multiples:")
    print(f"  -> P/E: {valuation.get('P/E', 0):.2f}x | EV/EBITDA: {valuation.get('EV/EBITDA', 0):.2f}x")
    print(f"  -> P/B: {valuation.get('P/B', 0):.2f}x | FCF Yield: {valuation.get('FCF_Yield (%)', 0):.2f}%\n")

    print(f"{'='*70}")
    print("Analysis completed successfully | MA Terminal v1.2.5")
    print(f"{'='*70}\n")


# ENTRY POINT

if __name__ == "__main__":

    print("MA TERMINAL - Strategic Financial Intelligence Platform")
    print("="*70)
    print("\nSupported Markets: NYSE, NASDAQ, LSE, TSE, B3, and 100+ global exchanges")
    print("\nExamples:")
    print("  - US Stocks: AAPL, MSFT, GOOGL, TSLA")
    print("  - Brazil (B3): PETR4.SA, VALE3.SA, ITUB4.SA")
    print("  - UK (LSE): HSBA.L, BP.L, VOD.L")
    print("  - Japan (TSE): 7203.T (Toyota), 9984.T (SoftBank)")
    print("  - Europe: ASML.AS (Netherlands), SAP.DE (Germany)")
    print("="*70 + "\n")

    # Interactive ticker input with validation
    while True:
        ticker_input = input("Enter ticker symbol (or 'exit' to quit): ").strip().upper()

        if ticker_input.lower() == 'exit':
            print("\nExiting MA Terminal. Goodbye!\n")
            break

        if not ticker_input:
            print(" Error: Ticker cannot be empty. Please try again.\n")
            continue

        if len(ticker_input) > 10:
            print(" Error: Invalid ticker format. Please try again.\n")
            continue

        # Execute analysis
        main(ticker_input)

        # Option to analyze another ticker
        print("\n")
        continue_analysis = input("Analyze another ticker? (y/n): ").strip().lower()
        if continue_analysis != 'y':
            print("\nThank you for using MA Terminal. See you soon.\n")
            break
