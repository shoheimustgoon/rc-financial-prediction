# 🚌 Bus Fleet Reliability & Maintenance Impact Analysis

> **[🇯🇵 日本語の説明はこちら (Click here for Japanese Description)](#japanese-description)**

## 📖 Overview

A reliability engineering framework for analyzing **bus fleet engine failures**, tracking **MTBF (Mean Time Between Failures)** with mileage correction, and evaluating **maintenance effectiveness** using bathtub curve analysis and survival models.

This project uses a **"Bus Company" analogy** to demonstrate techniques directly applicable to any fleet management, manufacturing equipment, or industrial reliability scenario.

### Key Methods

| Method | Purpose |
|---|---|
| **Exposure-Adjusted MTBF** | Fair comparison across different mileage buses |
| **Bathtub Curve (Aging)** | Classify phase: Early failure / Stable / Wear-out |
| **Monthly Error Rate Tracking** | Trend monitoring with count, rate, and MTBF |
| **Kaplan-Meier Survival** | Non-parametric failure-time estimation |
| **Cox PH / Weibull AFT** | Hazard ratio and lifespan quantification |
| **Right-Censoring** | Handles buses still running at observation end |
| **Breakpoint Detection** | Identifies when aging phase transitions occur |

---

## 🚌 The Analogy

- **Bus Company** operates 100 buses across 3 depots
- **Engine failures** are tracked monthly
- **Maintenance types:** Oil change, tire replacement, engine overhaul
- **Challenge:** Bus A runs 5,000 km/month, Bus B runs 1,000 km/month → simple MTBF is unfair
- **Solution:** Normalize by **mileage (exposure)**, not calendar time

### Bathtub Curve

```
Failure
 Rate
  ↑
  |╲                          ╱
  | ╲   Early    Stable     ╱ Wear-out
  |  ╲  Failure  Period    ╱  (IFR)
  |   ╲  (DFR)          ╱
  |    ╲_______________╱
  |
  └──────────────────────────→ Mileage (km)
```

---

## 📊 Analysis Pipeline

```
Failure Log → Mileage-Adjusted Rate → Monthly KPI (Count/Rate/MTBF)
                                    → Bathtub Phase Classification
                                    → Survival Analysis (KM/Cox/Weibull)
                                    → Maintenance Impact Evaluation
```

---

## 📦 Key Libraries

```
numpy, pandas, scipy, lifelines, matplotlib, openpyxl
```

---

## 📚 References

1. Kaplan & Meier (1958) "Nonparametric estimation from incomplete observations"
2. Cox (1972) "Regression models and life-tables"
3. Abernethy (2004) "The New Weibull Handbook"
4. ReliaSoft — Bathtub curve and reliability engineering
5. MIL-HDBK-217F — Reliability prediction of electronic equipment

---

## 👨‍💻 Author

**Go Sato** — Data Scientist | Reliability Engineering & Survival Analysis

---

---

<a name="japanese-description"></a>

# 🚌 バス車両の信頼性・メンテナンス影響度分析

## 📖 概要

バス車両の**エンジン故障**を追跡し、走行距離補正の**MTBF（平均故障間隔）** を用いて**メンテナンスの効果**を評価する信頼性工学フレームワークです。

**「バス会社」のたとえ話**を用いて、車両管理・製造装置・産業用機器の信頼性分析に直接応用可能な手法を実証しています。

### 主要手法

| 手法 | 目的 |
|---|---|
| **Exposure補正MTBF** | 走行距離の異なるバスを公平に比較 |
| **バスタブ曲線** | フェーズ分類：初期故障/安定期/摩耗故障 |
| **月次エラーレート追跡** | 件数・レート・MTBFのトレンド監視 |
| **KM / Cox PH / Weibull AFT** | 生存曲線、ハザード比、寿命推定 |
| **右打ち切り** | 観測終了時に稼働中のバスの扱い |
| **ブレークポイント検出** | エージングフェーズ遷移点の特定 |

---

## 🚌 たとえ話

- **バス会社** が3つの営業所で100台のバスを運行
- **エンジン故障** を月次で追跡
- **メンテナンス:** オイル交換、タイヤ交換、エンジンオーバーホール
- **課題:** 月5,000km走るバスと月1,000kmのバスを「時間」で比較するのは不公平
- **解決策:** カレンダー時間ではなく**走行距離（Exposure）**で正規化

---

## 📚 主要文献

1. Kaplan & Meier (1958) JASA
2. Cox (1972) J. Royal Statistical Society
3. Abernethy (2004) "The New Weibull Handbook"
4. MIL-HDBK-217F

---

## 👨‍💻 Author

**佐藤 剛 (Go Sato)** — データサイエンティスト | 信頼性工学・生存時間分析
