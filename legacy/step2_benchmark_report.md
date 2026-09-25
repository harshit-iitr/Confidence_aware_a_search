# Step 2: Full Benchmark Evaluation Report

**Hardware Platform:** xpu [Intel(R) Arc(TM) 130V GPU (8GB)]  
**Test Set:** 50 clean instances (strictly disjoint from training set)  
**Search Timeout:** 10.0 seconds per instance (Max expansions: 15000)  
**Total Run Duration:** 10.31 minutes

## Comparative Benchmark Summary Table

| Algorithm | Solve Rate (%) | Mean Expansions | Median Expansions | Mean Path Cost | Mean Search Time (ms) | Deadlock Resilience |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Classical A*** | **86.0%** (43/50) | 3777.2 | 2312.0 | 22.12 | 169.6 ms | Optimal / Zero deadlocks (High expansions) |
| **Paper Learned A*** | **76.0%** (38/50) | 188.2 | 127.0 | 20.97 | 2320.6 ms | Susceptible to uncalibrated deadlocks |
| **Paper Learned GBFS** | **92.0%** (46/50) | 127.0 | 28.0 | 25.07 | 1331.6 ms | Susceptible to uncalibrated deadlocks |
| **Fixed 50/50 Hybrid (Ablation)** | **86.0%** (43/50) | 173.8 | 106.0 | 24.51 | 2313.4 ms | Moderate fallback (Static blend) |
| **Confidence-Aware Hybrid A* (Ours)** | **94.0%** (47/50) | 125.6 | 70.0 | 25.02 | 2068.4 ms | Robust dynamic recovery via MC Dropout |
