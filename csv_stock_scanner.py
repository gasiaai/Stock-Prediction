"""
ตัวอย่างการสแกนหุ้นจาก CSV ไฟล์
Example of scanning stocks from CSV file
"""

import yfinance as yf
import pandas as pd
import numpy as np
from stock_analysis_system import StockAnalysisSystem
import os

def load_symbols_from_csv(csv_path):
    """
    โหลดรายชื่อหุ้นจาก CSV ไฟล์
    Load stock symbols from CSV file
    """
    try:
        df = pd.read_csv(csv_path)
        symbols = df['Symbol'].dropna().tolist()
        print(f"✓ โหลดได้ {len(symbols)} หุ้นจาก {csv_path}")
        return symbols
    except Exception as e:
        print(f"❌ ไม่สามารถโหลด CSV: {e}")
        return []

def get_real_stock_data(symbols, start_date='2020-01-01', end_date='2024-12-31', max_symbols=50):
    """
    ดึงข้อมูลหุ้นจริงจาก Yahoo Finance
    Limit to max_symbols to avoid API limits
    """
    # จำกัดจำนวนหุ้นเพื่อไม่ให้เกินขีดจำกัด API
    if len(symbols) > max_symbols:
        symbols = symbols[:max_symbols]
        print(f"⚠️  จำกัดการวิเคราะห์เป็น {max_symbols} หุ้นแรก")

    print(f"กำลังดึงข้อมูลหุ้น: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

    # ดึงข้อมูลราคาปิด
    data = yf.download(symbols, start=start_date, end=end_date)['Close']

    # ลบข้อมูลที่ขาดหาย
    data = data.dropna(axis=1, how='all')  # ลบคอลัมน์ที่ไม่มีข้อมูลเลย

    if data.empty:
        print("❌ ไม่มีข้อมูลหุ้นที่ถูกต้อง")
        return pd.DataFrame()

    print(f"✓ ได้รับข้อมูล {len(data)} วัน สำหรับ {len(data.columns)} หุ้น")
    print(f"ช่วงเวลา: {data.index[0].date()} ถึง {data.index[-1].date()}")

    return data

def get_fundamentals(symbols):
    """
    ดึงข้อมูลพื้นฐานของหุ้น (Market Cap, Book-to-Market)
    """
    fundamentals = {}

    for i, symbol in enumerate(symbols):
        if (i + 1) % 10 == 0:
            print(f"กำลังดึงข้อมูลพื้นฐาน... {i+1}/{len(symbols)}")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            market_cap = info.get('marketCap', np.nan)
            book_value = info.get('bookValue', np.nan)
            total_revenue = info.get('totalRevenue', np.nan)

            # คำนวณ Book-to-Market ratio
            if book_value and total_revenue and total_revenue > 0:
                book_to_market = book_value / total_revenue
            else:
                book_to_market = np.nan

            fundamentals[symbol] = {
                'market_cap': market_cap,
                'book_to_market': book_to_market
            }

        except Exception as e:
            # print(f"ไม่สามารถดึงข้อมูล {symbol}: {e}")
            fundamentals[symbol] = {
                'market_cap': np.nan,
                'book_to_market': np.nan
            }

    return pd.DataFrame(fundamentals).T

def main():
    """
    สแกนหุ้นจาก CSV และวิเคราะห์
    """
    print("=" * 80)
    print("📈 สแกนหุ้นจาก CSV ไฟล์")
    print("=" * 80)

    # เส้นทางไปยังไฟล์ CSV
    csv_path = r"G:\download\code practice\Research\MIT2\nasdaqmedtomeg.csv"

    if not os.path.exists(csv_path):
        print(f"❌ ไม่พบไฟล์ CSV: {csv_path}")
        return

    # ขั้นตอนที่ 1: โหลดรายชื่อหุ้นจาก CSV
    print("\n[1/5] กำลังโหลดรายชื่อหุ้นจาก CSV...")
    symbols = load_symbols_from_csv(csv_path)

    if not symbols:
        return

    # ขั้นตอนที่ 2: ดึงข้อมูลราคาหุ้น
    print("\n[2/5] กำลังดึงข้อมูลราคาหุ้น...")
    prices = get_real_stock_data(symbols, max_symbols=20)  # ทดสอบกับ 20 หุ้นแรก

    if prices.empty:
        return

    # ขั้นตอนที่ 3: ดึงข้อมูลพื้นฐาน
    print("\n[3/5] กำลังดึงข้อมูลพื้นฐาน...")
    fundamentals = get_fundamentals(prices.columns.tolist())

    # ขั้นตอนที่ 4: สร้างระบบวิเคราะห์
    print("\n[4/5] กำลังสร้างระบบวิเคราะห์...")
    system = StockAnalysisSystem()

    # ขั้นตอนที่ 5: วิเคราะห์หุ้น
    print("\n[5/5] กำลังวิเคราะห์...")
    results = system.analyze_stock(
        prices,
        fundamentals['market_cap'],
        fundamentals['book_to_market']
    )

    # แสดงผลลัพธ์
    print("\n" + "=" * 80)
    print("📊 ผลการสแกนหุ้นจาก CSV")
    print("=" * 80)

    # สรุปผลการวิเคราะห์
    print(f"\n📊 สรุป: วิเคราะห์ได้ {len(prices.columns)} หุ้น จาก CSV")

    # แสดงหุ้นที่มีผลตอบแทนสูงสุด
    log_returns = results['preprocessed_data']['log_returns']
    annual_returns = log_returns.mean() * 252 * 100

    print("\n🏆 หุ้นที่มีผลตอบแทนสูงสุด (Top 5):")
    top_performers = annual_returns.nlargest(5)
    for symbol, ret in top_performers.items():
        print(f"  {symbol}: {ret:.2f}%")

    print("\n📉 หุ้นที่มีความเสี่ยงต่ำสุด (Top 5 by Sharpe Ratio):")
    annual_vols = log_returns.std() * np.sqrt(252) * 100
    sharpe_ratios = annual_returns / annual_vols.replace(0, np.nan)
    top_sharpe = sharpe_ratios.nlargest(5)
    for symbol, sharpe in top_sharpe.items():
        ret = annual_returns[symbol]
        vol = annual_vols[symbol]
        print(f"  {symbol}: Sharpe={sharpe:.3f}, Return={ret:.2f}%, Vol={vol:.2f}%")

    print("\n" + "=" * 80)
    print("💡 หมายเหตุ:")
    print("  - ข้อมูลนี้มาจาก Yahoo Finance จริง")
    print("  - สแกนจาก CSV ไฟล์ NASDAQ")
    print("  - คุณสามารถปรับ max_symbols เพื่อวิเคราะห์หุ้นมากขึ้น")
    print("=" * 80)

if __name__ == "__main__":
    main()