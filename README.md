# 🎯 YOLO AI Analyzer

AI-powered YOLO model performance analyzer with actionable insights.

## 📋 Özellikler

- ✅ **YOLO11 sonuçlarını upload et** (CSV, YAML, PNG)
- 🤖 **AI-powered analiz** (Claude & GPT entegrasyonu)
- 📊 **Otomatik metric extraction** ve görselleştirme
- 🎯 **Aksiyon önerileri** (IoU, LR, augmentation vb.)
- 🔄 **Farklı eğitimleri karşılaştır**
- 📄 **PDF/HTML rapor export**

---

## 🚀 Hızlı Başlangıç

### 1. Proje Kurulumu

```bash
# Repo'yu clone'la
git clone https://github.com/your-username/yolo-ai-analyzer.git
cd yolo-ai-analyzer

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# .env dosyası oluştur
cp .env.example .env
# API key'lerini .env'e ekle

# Frontend setup
cd ../frontend
npm install
```

### 2. Çalıştırma

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn app.main:app --reload
# http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# http://localhost:5173
```

---

## 📁 Proje Yapısı

```
yolo-ai-analyzer/
├── backend/
│   ├── app/
│   │   ├── main.py              ✅ FastAPI app (HAZIR)
│   │   ├── models.py            🔨 Pydantic models
│   │   ├── parsers/
│   │   │   ├── yolo_parser.py   🔨 YOLO CSV/YAML parser
│   │   │   └── metrics_extractor.py
│   │   ├── analyzers/
│   │   │   ├── llm_analyzer.py  🔨 Claude/GPT entegrasyonu
│   │   │   └── rule_based.py    🔨 Fallback kurallar
│   │   └── utils/
│   │       ├── file_handler.py
│   │       └── visualization.py
│   ├── uploads/
│   ├── reports/
│   ├── requirements.txt         ✅ HAZIR
│   └── .env
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ✅ Main component (HAZIR)
│   │   ├── components/
│   │   │   ├── FileUploader.jsx     🔨 Dosya upload UI
│   │   │   ├── MetricsDisplay.jsx   🔨 Metric kartları
│   │   │   ├── AIAnalysis.jsx       🔨 AI analiz gösterimi
│   │   │   └── ComparisonView.jsx   🔨 Karşılaştırma
│   │   ├── services/
│   │   │   └── api.js           🔨 Backend API calls
│   │   └── App.css              🔨 Styling
│   ├── package.json             ✅ HAZIR
│   └── vite.config.js
│
├── README.md                    ✅ HAZIR
└── .gitignore                   ✅ HAZIR
```

**Legend:**  
✅ = Iskelet hazır (genişletilecek)  
🔨 = Cursor/Codex ile oluşturulacak

---

## 🛠️ Cursor/Codex ile Geliştirme Adımları

### **Adım 1: YOLO Parser (Backend)**

**Dosya:** `backend/app/parsers/yolo_parser.py`

```python
# -*- coding: utf-8 -*-
import pandas as pd
import yaml
from pathlib import Path

class YOLOResultParser:
    """
    YOLO11 eğitim sonuçlarını parse et
    
    Input files:
    - runs/segment/train/results.csv
    - runs/segment/train/args.yaml
    """
    
    def __init__(self, csv_path: str, yaml_path: str = None):
        self.csv_path = csv_path
        self.yaml_path = yaml_path
    
    def parse_metrics(self) -> dict:
        """
        CSV'den son epoch metriklerini çıkar
        
        Returns:
            {
                'precision': float,
                'recall': float,
                'map50': float,
                'map50_95': float,
                'box_loss': float,
                'seg_loss': float,
                ...
            }
        """
        # TODO: Pandas ile CSV oku
        # TODO: Son satırı al (final epoch)
        # TODO: Metrikleri dict'e çevir
        pass
    
    def parse_config(self) -> dict:
        """YAML config'i parse et"""
        # TODO: YAML dosyasını oku
        # TODO: Training parametrelerini çıkar
        pass
    
    def get_full_analysis(self) -> dict:
        """Tüm bilgileri birleştir"""
        return {
            'metrics': self.parse_metrics(),
            'config': self.parse_config(),
            'training_curves': self.extract_curves()
        }
```

**Cursor Prompt:**
```
@yolo_parser.py içindeki parse_metrics() fonksiyonunu tamamla. 
YOLO11 results.csv formatını biliyorsun. Son epoch'taki precision, 
recall, mAP@0.5, mAP@0.5:0.95 değerlerini çıkar. UTF-8 encoding kullan.
```

---

### **Adım 2: LLM Analyzer (Backend)**

**Dosya:** `backend/app/analyzers/llm_analyzer.py`

```python
# -*- coding: utf-8 -*-
import os
from anthropic import Anthropic
from openai import OpenAI
from typing import Dict, Literal

class LLMAnalyzer:
    """Claude ve GPT ile model analizi"""
    
    def __init__(self, provider: Literal["claude", "openai"] = "claude"):
        self.provider = provider
        
        if provider == "claude":
            self.client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def analyze(self, metrics: Dict, config: Dict) -> Dict:
        """
        Metrics ve config'i analiz et
        
        Returns:
            {
                'summary': str,
                'strengths': List[str],
                'weaknesses': List[str],
                'action_items': List[str],
                'risk_level': str
            }
        """
        prompt = self._build_prompt(metrics, config)
        
        if self.provider == "claude":
            return self._analyze_with_claude(prompt)
        else:
            return self._analyze_with_gpt(prompt)
    
    def _build_prompt(self, metrics: Dict, config: Dict) -> str:
        """
        LLM için prompt oluştur
        
        Senin FKT projendeki deneyimlerini de ekle:
        - Recall düşükse → IoU azalt önerisi
        - Precision düşükse → Confidence artır
        - mAP düşükse → Augmentation artır
        """
        return f"""
        Sen bir YOLO expert'isin. Aşağıdaki model sonuçlarını analiz et:
        
        **Metrics:**
        - Precision: {metrics.get('precision', 'N/A')}
        - Recall: {metrics.get('recall', 'N/A')}
        - mAP@0.5: {metrics.get('map50', 'N/A')}
        - mAP@0.5:0.95: {metrics.get('map50_95', 'N/A')}
        
        **Training Config:**
        - Epochs: {config.get('epochs', 'N/A')}
        - Batch Size: {config.get('batch', 'N/A')}
        - Learning Rate: {config.get('lr0', 'N/A')}
        - IoU Threshold: {config.get('iou', 'N/A')}
        
        Lütfen şu formatta cevapla:
        1. ÖZET: Genel değerlendirme (2-3 cümle)
        2. GÜÇLÜ YÖNLER: Liste halinde
        3. ZAYIF YÖNLER: Liste halinde
        4. AKSİYON ÖNERİLERİ: Somut, uygulanabilir öneriler
        5. RİSK SEVİYESİ: low/medium/high
        """
    
    def _analyze_with_claude(self, prompt: str) -> Dict:
        # TODO: Claude API call
        pass
    
    def _analyze_with_gpt(self, prompt: str) -> Dict:
        # TODO: OpenAI API call
        pass
```

**Cursor Prompt:**
```
@llm_analyzer.py içindeki _analyze_with_claude() ve _analyze_with_gpt() 
fonksiyonlarını tamamla. Response'u parse edip structured dict'e çevir.
Error handling ekle. Timeout 30 saniye olsun.
```

---

### **Adım 3: Frontend API Service**

**Dosya:** `frontend/src/services/api.js`

```javascript
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export const api = {
  // Upload results
  uploadResults: async (files) => {
    const formData = new FormData();
    if (files.csv) formData.append('results_csv', files.csv);
    if (files.yaml) formData.append('config_yaml', files.yaml);
    if (files.graphs) {
      files.graphs.forEach(g => formData.append('graphs', g));
    }
    
    const response = await axios.post(`${API_BASE}/upload/results`, formData);
    return response.data;
  },
  
  // Get AI analysis
  analyzeMetrics: async (metrics) => {
    const response = await axios.post(`${API_BASE}/analyze/metrics`, metrics);
    return response.data;
  },
  
  // Compare runs
  compareRuns: async (runIds) => {
    const response = await axios.post(`${API_BASE}/compare`, { run_ids: runIds });
    return response.data;
  },
  
  // Get history
  getHistory: async () => {
    const response = await axios.get(`${API_BASE}/history`);
    return response.data;
  }
};
```

---

### **Adım 4: Styling**

**Dosya:** `frontend/src/App.css`

```css
/* Modern, clean design */
:root {
  --primary: #4f46e5;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --bg: #f9fafb;
  --card-bg: #ffffff;
  --text: #1f2937;
  --text-secondary: #6b7280;
}

.app-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
  background: var(--bg);
  min-height: 100vh;
}

.metric-card {
  background: var(--card-bg);
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  transition: transform 0.2s;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* TODO: Daha fazla styling Cursor ile */
```

---

## 🎯 Geliştirme Roadmap

### **Faz 1: MVP (1-2 hafta)**
- [x] Backend iskelet
- [x] Frontend iskelet
- [ ] YOLO CSV parser
- [ ] Claude/GPT entegrasyonu
- [ ] Basit UI
- [ ] Localhost deployment

### **Faz 2: Özellikler (2-3 hafta)**
- [ ] Grafik görselleştirme
- [ ] Çoklu eğitim karşılaştırma
- [ ] Database entegrasyonu (SQLite)
- [ ] PDF rapor export
- [ ] Gelişmiş UI/UX

### **Faz 3: Production (1 hafta)**
- [ ] Docker containerization
- [ ] Error handling & logging
- [ ] Unit tests
- [ ] Masaüstü app (Electron/PyInstaller)

---

## 💡 Cursor/Codex Kullanım İpuçları

### **Etkili Promptlar:**

```
# Parser için:
@yolo_parser.py results.csv dosyasını pandas ile oku. 
Son epoch'taki tüm metrikleri dict'e çevir. Column isimleri: 
epoch, train/box_loss, train/seg_loss, metrics/precision(B), 
metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)

# LLM için:
@llm_analyzer.py Claude'a metrics gönder, JSON response al. 
Schema: {summary: str, strengths: str[], weaknesses: str[], 
action_items: str[], risk_level: "low"|"medium"|"high"}

# Frontend için:
@FileUploader.jsx güzel drag&drop interface ekle. 
Desteklenen formatlar: .csv, .yaml, .png. Preview göster.
```

### **Debugging:**
```
@main.py bu endpoint'te 500 error alıyorum. Detaylı logging ekle 
ve hata mesajını debug et.
```

---

## 🤝 Katkıda Bulunma

1. Fork et
2. Feature branch oluştur (`git checkout -b feature/amazing-feature`)
3. Commit yap (`git commit -m 'Add amazing feature'`)
4. Push yap (`git push origin feature/amazing-feature`)
5. Pull Request aç

---

## 📝 Lisans

MIT License - FKT AI Projects

---

## 📧 İletişim

Sorularınız için: [your-email]

**Built with ❤️ for FKT & future AI projects**
