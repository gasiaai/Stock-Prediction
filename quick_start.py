"""
Quick Start Guide - เริ่มต้นใช้งานอย่างรวดเร็ว
ตัวอย่างที่ง่ายที่สุดสำหรับการใช้งาน Stock Analysis System
"""

import numpy as np
import pandas as pd
from stock_analysis_system import StockAnalysisSystem

# ตั้งค่า random seed
np.random.seed(42)


def create_sample_stock_data():
    """
    สร้างข้อมูลหุ้นตัวอย่าง 3 ตัว
    """
    # สร้างวันที่ 1 ปี (252 วันซื้อขาย)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')

    # สร้างราคาหุ้นจำลอง
    np.random.seed(42)

    # หุ้น A: Tech stock (ผันผวนสูง)
    price_A = [100]
    for _ in range(251):
        change = np.random.normal(0.001, 0.025)  # +0.1% mean, 2.5% std
        price_A.append(price_A[-1] * (1 + change))

    # หุ้น B: Stable stock (ผันผวนต่ำ)
    price_B = [100]
    for _ in range(251):
        change = np.random.normal(0.0005, 0.01)  # +0.05% mean, 1% std
        price_B.append(price_B[-1] * (1 + change))

    # หุ้น C: Growth stock (ผันผวนปานกลาง)
    price_C = [100]
    for _ in range(251):
        change = np.random.normal(0.0015, 0.018)  # +0.15% mean, 1.8% std
        price_C.append(price_C[-1] * (1 + change))

    # สร้าง DataFrame
    prices = pd.DataFrame({
        'TECH': price_A,
        'STABLE': price_B,
        'GROWTH': price_C
    }, index=dates)

    return prices


def main():
    """
    ตัวอย่างการใช้งานแบบง่าย
    """
    print("=" * 70)
    print("🚀 Quick Start - Stock Analysis System")
    print("=" * 70)

    # ขั้นตอนที่ 1: สร้างข้อมูลตัวอย่าง
    print("\n[1/3] กำลังสร้างข้อมูลตัวอย่าง...")
    prices = create_sample_stock_data()
    print(f"✓ สร้างข้อมูลหุ้น 3 ตัว สำเร็จ ({len(prices)} วัน)")
    print(f"\n  หุ้นที่วิเคราะห์: {', '.join(prices.columns)}")
    print(f"\n  ราคาปัจจุบัน:")
    for col in prices.columns:
        print(f"    {col:10s}: ${prices[col].iloc[-1]:,.2f}")

    # ขั้นตอนที่ 2: สร้างระบบวิเคราะห์
    print("\n[2/3] กำลังสร้างระบบวิเคราะห์...")
    system = StockAnalysisSystem()
    print("✓ ระบบพร้อมใช้งาน")

    # ขั้นตอนที่ 3: วิเคราะห์หุ้น
    print("\n[3/3] กำลังวิเคราะห์...")
    results = system.analyze_stock(prices)

    # แสดงผลลัพธ์หลัก
    print("\n" + "=" * 70)
    print("📊 ผลการวิเคราะห์")
    print("=" * 70)

    # 1. ผลตอบแทนและความเสี่ยงของแต่ละหุ้น
    print("\n1️⃣  ผลตอบแทนและความเสี่ยงรายตัว (Annualized):")
    print("-" * 70)
    log_returns = results['preprocessed_data']['log_returns']

    for col in log_returns.columns:
        annual_return = log_returns[col].mean() * 252 * 100
        annual_vol = log_returns[col].std() * np.sqrt(252) * 100
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        var_99 = results['risk_analysis']['var'][col]['VaR'] * 100

        print(f"\n  {col}:")
        print(f"    Expected Return: {annual_return:>7.2f}%")
        print(f"    Volatility:      {annual_vol:>7.2f}%")
        print(f"    Sharpe Ratio:    {sharpe:>7.3f}")
        print(f"    VaR (99%):       {var_99:>7.2f}% (daily)")

    # 2. พอร์ตโฟลิโอที่เหมาะสม
    print("\n2️⃣  พอร์ตโฟลิโอที่แนะนำ:")
    print("-" * 70)

    mv_result = results['portfolio_optimization']['mean_variance']
    print(f"\n  📈 Mean-Variance Optimal Portfolio:")
    print(f"    Expected Return: {mv_result['expected_return']*100:>7.2f}%")
    print(f"    Volatility:      {mv_result['volatility']*100:>7.2f}%")
    print(f"    Sharpe Ratio:    {mv_result['sharpe_ratio']:>7.3f}")
    print(f"\n    สัดส่วนการลงทุน:")
    for i, (stock, weight) in enumerate(zip(prices.columns, mv_result['weights'])):
        print(f"      {stock:10s}: {weight*100:>6.2f}%")

    rp_weights = results['portfolio_optimization']['risk_parity_weights']
    print(f"\n  ⚖️  Risk Parity Portfolio:")
    print(f"    สัดส่วนการลงทุน:")
    for i, (stock, weight) in enumerate(zip(prices.columns, rp_weights)):
        print(f"      {stock:10s}: {weight*100:>6.2f}%")

    # 3. ความสัมพันธ์ระหว่างหุ้น
    print("\n3️⃣  ความสัมพันธ์ระหว่างหุ้น:")
    print("-" * 70)
    corr_matrix = results['risk_analysis']['correlation']['correlation_matrix']
    print(f"\n{corr_matrix.to_string()}")

    # 4. การคาดการณ์ด้วย Monte Carlo
    print("\n4️⃣  การคาดการณ์ราคา 1 ปีข้างหน้า (Monte Carlo):")
    print("-" * 70)

    for stock, mc_result in results['monte_carlo'].items():
        current_price = prices[stock].iloc[-1]
        expected_price = mc_result['mean_final_price']
        p5 = mc_result['percentile_5']
        p95 = mc_result['percentile_95']

        print(f"\n  {stock}:")
        print(f"    ราคาปัจจุบัน:     ${current_price:>8.2f}")
        print(f"    ราคาคาดหวัง:      ${expected_price:>8.2f}")
        print(f"    ช่วง 90% CI:      ${p5:>8.2f} - ${p95:>8.2f}")
        print(f"    ผลตอบแทนคาดหวัง: {(expected_price/current_price-1)*100:>7.2f}%")

    # คำแนะนำการลงทุน
    print("\n" + "=" * 70)
    print("💡 คำแนะนำ")
    print("=" * 70)

    # หาหุ้นที่มี Sharpe Ratio สูงสุด
    sharpes = {}
    for col in log_returns.columns:
        annual_return = log_returns[col].mean() * 252
        annual_vol = log_returns[col].std() * np.sqrt(252)
        sharpes[col] = annual_return / annual_vol if annual_vol > 0 else 0

    best_stock = max(sharpes, key=sharpes.get)
    worst_risk = max([(col, results['risk_analysis']['var'][col]['VaR'])
                     for col in prices.columns],
                    key=lambda x: x[1])

    print(f"\n  ✅ หุ้นที่มี Risk-adjusted return ดีที่สุด: {best_stock}")
    print(f"     (Sharpe Ratio: {sharpes[best_stock]:.3f})")

    print(f"\n  ⚠️  หุ้นที่มีความเสี่ยงสูงสุด: {worst_risk[0]}")
    print(f"     (VaR 99%: {worst_risk[1]*100:.2f}% daily)")

    best_weight_idx = np.argmax(mv_result['weights'])
    recommended_stock = prices.columns[best_weight_idx]
    print(f"\n  🎯 พอร์ตโฟลิโอที่เหมาะสมแนะนำให้เน้นน้ำหนักที่: {recommended_stock}")
    print(f"     (สัดส่วน: {mv_result['weights'][best_weight_idx]*100:.1f}%)")

    # สรุป
    print("\n" + "=" * 70)
    print("✅ การวิเคราะห์เสร็จสมบูรณ์!")
    print("=" * 70)
    print("\n📚 ต้องการข้อมูลเพิ่มเติม?")
    print("  - รันไฟล์ example_usage.py เพื่อดูตัวอย่างที่ละเอียดยิ่งขึ้น")
    print("  - อ่าน README.md สำหรับเอกสารประกอบฉบับเต็ม")
    print("  - ดู stock_analysis_system.py สำหรับรายละเอียดทางเทคนิค")
    print("\n⚠️  คำเตือน: นี่เป็นเครื่องมือทางการศึกษา ไม่ใช่คำแนะนำในการลงทุน")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
