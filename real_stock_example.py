"""
ตัวอย่างการนำเข้าข้อมูลหุ้นจริง
Example of importing real stock data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from stock_analysis_system import StockAnalysisSystem

def get_real_stock_data(symbols, start_date='2020-01-01', end_date='2024-12-31'):
    """
    ดึงข้อมูลหุ้นจริงจาก Yahoo Finance
    """
    print(f"กำลังดึงข้อมูลหุ้น: {', '.join(symbols)}")

    # ดึงข้อมูลราคาปิด
    data = yf.download(symbols, start=start_date, end=end_date)['Close']

    # ลบข้อมูลที่ขาดหาย
    data = data.dropna()

    print(f"✓ ได้รับข้อมูล {len(data)} วัน สำหรับ {len(symbols)} หุ้น")
    print(f"ช่วงเวลา: {data.index[0].date()} ถึง {data.index[-1].date()}")

    return data

def get_fundamentals(symbols):
    """
    ดึงข้อมูลพื้นฐานของหุ้น (Market Cap, Book-to-Market)
    """
    fundamentals = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            market_cap = info.get('marketCap', np.nan)
            book_value = info.get('bookValue', np.nan)
            total_revenue = info.get('totalRevenue', np.nan)

            # คำนวณ Book-to-Market ratio
            if book_value and total_revenue:
                book_to_market = book_value / total_revenue
            else:
                book_to_market = np.nan

            fundamentals[symbol] = {
                'market_cap': market_cap,
                'book_to_market': book_to_market
            }

        except Exception as e:
            print(f"ไม่สามารถดึงข้อมูล {symbol}: {e}")
            fundamentals[symbol] = {
                'market_cap': np.nan,
                'book_to_market': np.nan
            }

    return pd.DataFrame(fundamentals).T

def main():
    """
    ตัวอย่างการวิเคราะห์หุ้นจริง
    """
    print("=" * 70)
    print("📈 ตัวอย่างการวิเคราะห์หุ้นจริง")
    print("=" * 70)

    # เลือกหุ้นที่ต้องการวิเคราะห์ (ใช้ symbol จริง)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # หุ้นเทคโนโลยีชั้นนำ

    # ขั้นตอนที่ 1: ดึงข้อมูลราคาหุ้น
    print("\n[1/4] กำลังดึงข้อมูลราคาหุ้น...")
    prices = get_real_stock_data(symbols)

    # ขั้นตอนที่ 2: ดึงข้อมูลพื้นฐาน
    print("\n[2/4] กำลังดึงข้อมูลพื้นฐาน...")
    fundamentals = get_fundamentals(symbols)

    # ขั้นตอนที่ 3: สร้างระบบวิเคราะห์
    print("\n[3/4] กำลังสร้างระบบวิเคราะห์...")
    system = StockAnalysisSystem()

    # ขั้นตอนที่ 4: วิเคราะห์หุ้น
    print("\n[4/4] กำลังวิเคราะห์...")
    results = system.analyze_stock(
        prices,
        fundamentals['market_cap'],
        fundamentals['book_to_market']
    )

    # แสดงผลลัพธ์
    print("\n" + "=" * 70)
    print("📊 ผลการวิเคราะห์หุ้นจริง")
    print("=" * 70)

    # ผลตอบแทนและความเสี่ยง
    print("\n1️⃣  ผลตอบแทนและความเสี่ยงรายตัว (Annualized):")
    print("-" * 70)
    log_returns = results['preprocessed_data']['log_returns']

    for col in log_returns.columns:
        annual_return = log_returns[col].mean() * 252 * 100
        annual_vol = log_returns[col].std() * np.sqrt(252) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        print(f"\n  {col}:")
        print(f"    Expected Return: {annual_return:>7.2f}%")
        print(f"    Volatility:      {annual_vol:>7.2f}%")
        print(f"    Sharpe Ratio:    {sharpe:>7.3f}")

    print("\n" + "=" * 70)
    print("💡 หมายเหตุ:")
    print("  - ข้อมูลนี้มาจาก Yahoo Finance จริง")
    print("  - คุณสามารถเปลี่ยน symbols เป็นหุ้นที่สนใจได้")
    print("  - ระบบจะวิเคราะห์ตามหลักการทางการเงินเช่นเดียวกับข้อมูลจำลอง")
    print("=" * 70)

if __name__ == "__main__":
    main()