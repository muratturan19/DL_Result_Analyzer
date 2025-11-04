# 🎯 DL_Result_Analyzer

Derin öğrenme modeli sonuçlarını analiz etmek için tasarlanmış güçlü ve esnek bir platform.

## 📋 Özellikler

- ✅ **YOLO11 sonuçlarını upload et** (CSV, YAML, PNG)
- 🤖 **AI-powered analiz** (Claude & GPT entegrasyonu)
- 📊 **Otomatik metric extraction** ve görselleştirme
- 🎯 **Aksiyon önerileri** (IoU, LR, augmentation vb.)
- 🗃️ **Veri seti özeti** (args.yaml'dan eğitim/val/test görsel adetleri ve sınıf isimleri)
- 💬 **Rapor sonrası Q/A** (oluşturulan rapor üzerinden LLM'e soru sor)
- 🔄 **Farklı eğitimleri karşılaştır**
- 📄 **PDF/HTML rapor export**

---

## 🆕 UI Revamp

Rapor HTML çıktıları modern bir dashboard görünümüyle güncellendi. Yeni tema sistemi, erişilebilir bileşenler ve yazdırma optimizasyonu sayesinde hem ekranda hem PDF çıktısında tutarlı bir deneyim sunulur.

### Öne çıkanlar

- 🌗 **Koyu/Aydınlık tema**: Varsayılan koyu tema, sistem tercihini algılar ve kullanıcı seçimi `localStorage` ile kalıcı hale gelir.
- 📊 **Stat kartları**: Precision, Recall, mAP ve Loss metrikleri için otomatik renklendirme (hedef/tolerans eşikleri JS içinde tek noktadan yönetiliyor).
- ⚠️ **Risk çipleri**: JSON benzeri risk listeleri okunaklı çiplere dönüştürülür, renkler risk seviyesine göre değişir.
- 📱 **Mobil & tablet uyumu**: Stat grid, tablolar ve aksiyon listeleri küçük ekranlarda yatay kaydırma veya stack düzeni ile görünür kalır.
- 🖨️ **A4 yazdırma modu**: `web/print.css` koyu arka planı kapatır, kenar boşluklarını ayarlar ve kartları sayfa bölünmelerine karşı korur.

### Nasıl devreye alınır?

1. Repo kökündeki `web/` klasöründe bulunan `report-theme.css`, `report-ui.js`, `print.css` ve `icons.svg` dosyaları otomatik olarak HTML çıktısına inline eklenir. Ek build adımı gerekmez.
2. Backend'i güncel kodla başlatıp yeni bir rapor ürettiğinizde `_generate_html_report` fonksiyonu yeni şablonu kullanır.
3. Önizleme için `/api/report/{report_id}/export?format=html` uç noktasını çağırarak tarayıcıda açabileceğiniz tek dosyalı raporu indirin.

### Tema & Yazdırma

- Header'daki **Tema** butonu renk paletini değiştirir; seçim tarayıcıda saklanır ve `prefers-color-scheme` değişimlerini dinler.
- **Yazdır** butonu `window.print()` çağırır ve ekran modunda görünür, baskıda otomatik gizlenir.
- Tabloların sticky başlıkları ve kart gölgeleri yazdırma modunda temizlenerek kurumsal bir PDF çıktısı alınır.

### Eski raporları güncelleme (Quick Guide)

1. Çalışma dizininde `git pull` ile bu sürümü alın ve backend servislerini yeniden başlatın.
2. Her rapor için `GET /api/report/{report_id}/export?format=html` çağrısı yaparak yeni şablonu kullanan HTML dosyasını indirin.
3. Arşivde manuel tutulmuş HTML'leriniz varsa, başlığa Tailwind CDN betiğini ekleyin ve `web/report-theme.css`, `web/print.css`, `web/report-ui.js`, `web/icons.svg` içeriklerini sırasıyla `<style>`, `<style media="print">`, `<script>` ve `<svg>` blokları olarak inline yerleştirin.

---

## 🚀 Hızlı Başlangıç

### Veri Seti Özeti & Rapor Asistanı

- Upload sonrasında arayüzde **Veri Seti Özeti** kartı, args.yaml içindeki `data_dict`/`dataset_info` alanlarını parse ederek eğitim/val/test görsel sayılarını ve sınıf isimlerini gösterir.
- Backend tarafında bu bilgiler `config.dataset` alanına eklenir; LLM prompt'u da bu sayıları değerlendirmeye zorlar.
- Aynı yükleme işlemiyle birlikte benzersiz bir `report_id` üretilir. Frontend'deki **Rapor Asistanı** paneli bu ID'yi kullanarak raporla ilgili ek soruları LLM'e iletir.
- API üzerinden soru sormak için:

```bash
curl -X POST "http://localhost:8000/api/report/<REPORT_ID>/qa" \
  -H "Content-Type: application/json" \
  -d '{"question": "Eğitimde kaç görsel var?", "llm_provider": "claude"}'
```

Yanıt yapısı:

```json
{
  "status": "success",
  "report_id": "...",
  "qa": {
    "question": "...",
    "answer": "...",
    "references": ["results.csv → metrik özeti"],
    "follow_up_questions": ["..."]
  }
}
```

LLM erişimi yoksa backend kurallı bir yanıt üretir ve referansları yine paylaşır.

### 1. Proje Kurulumu

```bash
# Repo'yu clone'la
git clone https://github.com/muratturan19/DL_Result_Analyzer.git
cd DL_Result_Analyzer

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# .env dosyası oluştur
cp .env.example .env
# .env dosyasını düzenle ve API key'lerini ekle:
# - CLAUDE_API_KEY=sk-ant-...
# - OPENAI_API_KEY=sk-...
# - LLM_PROVIDER=claude  # veya openai

# Frontend setup
cd ../frontend
npm install
```

### 1.5. Sample Data ile Test (Opsiyonel)

```bash
# Backend'i başlat
cd backend
uvicorn app.main:app --reload

# Başka bir terminal'de frontend'i başlat
cd frontend
npm run dev

# Browser'da: http://localhost:5173
# Upload: examples/sample_results.csv + examples/sample_args.yaml
```

### 2. Çalıştırma

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # Eğer venv kullanıyorsanız
uvicorn app.main:app --reload
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# Frontend: http://localhost:5173
```

### 3. Test Suite Çalıştırma

```bash
cd backend
pytest tests/ -v
# 23/24 test geçmeli (OpenAI API key yoksa 1 test fail)

# Coverage raporu için:
pytest tests/ -v --cov=app --cov-report=html
```

---

## 📁 Proje Yapısı

```
DL_Result_Analyzer/
├── backend/
│   ├── app/
│   │   ├── main.py                  ✅ FastAPI app (COMPLETE)
│   │   ├── parsers/
│   │   │   └── yolo_parser.py       ✅ YOLO CSV/YAML parser (COMPLETE)
│   │   └── analyzers/
│   │       └── llm_analyzer.py      ✅ Claude/GPT integration (COMPLETE)
│   ├── tests/                       ✅ Pytest test suite (23/24 passing)
│   │   ├── test_yolo_parser.py
│   │   ├── test_llm_analyzer.py
│   │   └── test_api.py
│   ├── uploads/                     📁 Auto-created on upload
│   ├── requirements.txt             ✅ COMPLETE
│   └── .env.example                 ✅ COMPLETE
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  ✅ Full implementation (COMPLETE)
│   │   │   ├── FileUploader         ✅ Multi-file upload
│   │   │   ├── MetricsDisplay       ✅ Metric cards with status
│   │   │   └── AIAnalysis           ✅ LLM analysis display
│   │   ├── App.css                  ✅ Modern styling (COMPLETE)
│   │   └── main.jsx
│   ├── package.json                 ✅ COMPLETE
│   └── node_modules/                ✅ Installed
│
├── examples/                        ✅ Sample data for testing
│   ├── sample_results.csv           ✅ 100-epoch YOLO11 results
│   ├── sample_args.yaml             ✅ Training configuration
│   ├── sample_data.yaml             ✅ Dataset definition
│   └── README.md                    ✅ Usage instructions
│
├── TEST_REPORT.md                   ✅ Comprehensive test report
├── .env.example                     ✅ Environment variables template
├── README.md                        ✅ This file
└── .gitignore                       ✅ COMPLETE
```

**Legend:**
✅ = Fully implemented and tested
📁 = Auto-generated directory
🔨 = To be developed (future features)

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

### **Faz 1: MVP (TAMAMLANDI ✅)**
- [x] Backend iskelet
- [x] Frontend iskelet
- [x] YOLO CSV parser (Full implementation)
- [x] Claude/GPT entegrasyonu (Full implementation)
- [x] Modern UI (Responsive design)
- [x] Localhost deployment
- [x] Sample data ve test suite
- [x] Comprehensive test report

### **Faz 2: Özellikler (Devam Ediyor 🔨)**
- [x] Error handling & logging (Backend)
- [ ] Grafik görselleştirme (Recharts integration)
- [ ] Çoklu eğitim karşılaştırma
- [ ] Database entegrasyonu (SQLite)
- [ ] PDF rapor export
- [ ] Gelişmiş UI/UX (Dark mode, animations)

### **Faz 3: Production (Planlanıyor 📋)**
- [ ] Docker containerization
- [ ] E2E tests (Playwright/Cypress)
- [ ] CI/CD pipeline
- [ ] Masaüstü app (Electron/PyInstaller)
- [ ] Performance optimization

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
