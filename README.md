# Automated Document Scanner & Enhancer 🚀

A production-ready, highly modular Computer Vision application designed to take poorly lit, angled smartphone photos of documents and transform them into clean, orthogonal, print-ready, high-contrast monochrome formats. 

Developed as part of the **Week 1 Portfolio Sprint**.

---

## 🏗️ Architecture & Project Structure

The project follows strict modular engineering guidelines, separating configuration, core image processing pipelines, benchmarks, and the user interface.

```text
document-scanner/
├── config.py          # ScannerConfig containing 17 production parameters
├── scanner.py         # DocumentScanner core engine & ScanResult dataclass
├── app.py             # Web UI demonstration built with Streamlit
├── benchmark.py       # Comprehensive MLOps performance profiling script
├── requirements.txt   # Python dependencies (optimized for headless envs)
├── Dockerfile         # Multi-stage secure production container definition
└── README.md          # Project documentation
⚡ Performance Profiling & SLA VerificationThe core processing pipeline includes an isolated warm-up phase and detailed tail-latency profiling across three distinct configuration presets (default, strict, fast).Empirical Results (810×1080 px, 100 Iterations, Windows 11 Host)Config PresetMean Latencyp50 Latencyp95 Latencyp99 LatencySLA Status (< 200ms)default45.1 ms44.9 ms48.7 ms51.6 ms✅ PASSstrict35.0 ms34.9 ms39.1 ms40.0 ms✅ PASSNote: All configurations pass the strict production SLA barrier of 200ms with a substantial 4x safety headroom.🛠️ Installation & Local Execution1. Environment SetupClone the repository and set up a virtual environment using Python 3.14+ (or your local equivalent):Bash# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Force UTF-8 environment variable for Windows file-paths compatibility
export PYTHONUTF8=1       # On Windows PowerShell use: $env:PYTHONUTF8=1

# Install requirements
pip install -r requirements.txt
2. Run Latency BenchmarksRun the automated regression and profiling script, which evaluates pipelines and exports telemetry data:Bashpython benchmark.py --compare --export report.json
3. Launch Web UILaunch the interactive Streamlit dashboard to process documents in real-time:Bashstreamlit run app.py
🐳 Docker Deployment (CI/CD Ready)The project includes a production-grade, multi-stage Dockerfile that ensures minimum image sizes, implements a non-root security context, and integrates container health checks.Build Image:Bashdocker build -t doc-scanner-app .
Run Automated Regression Test (CI Gate):Bashdocker run --rm doc-scanner-app python benchmark.py --compare
Run Web Server:Bashdocker run -p 8501:8501 doc-scanner-app
📈 Known Improvements & Future RoadmapCV Optimization: Fine-tune cv2.adaptiveThreshold window sizing to minimize salt-and-pepper noise artifacts under uneven illumination gradients.Contour Reliability: Expand cv2.approxPolyDP heuristics to handle low-contrast edges on complex background textures.
