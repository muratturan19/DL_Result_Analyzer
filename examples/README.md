# Sample Data - YOLO11 Training Results

Bu dizinde DL_Result_Analyzer'ı test etmek için gerçekçi sample data bulunmaktadır.

## 📁 Dosyalar

### `sample_results.csv`
- **Açıklama:** 100 epoch YOLO11 detection eğitim sonuçları
- **Format:** YOLO11 results.csv formatı
- **Epoch Sayısı:** 100
- **Metrikler:**
  - Precision: 0.58 → 0.79 (son epoch)
  - Recall: 0.65 → 0.82 (son epoch)
  - mAP50: 0.62 → 0.86 (son epoch)
  - mAP50-95: 0.35 → 0.40 (son epoch)
  - Loss: Giderek azalan (1.23 → 0.73)

### `sample_args.yaml`
- **Açıklama:** YOLO11 training konfigürasyonu
- **Model:** yolo11l.pt
- **Proje:** FKT Potluk (Leather Seat Dent) Detection
- **Parametreler:**
  - Epochs: 100
  - Batch: 16
  - Image Size: 640
  - Learning Rate: 0.01
  - Augmentations: hsv, flip, rotate, translate
  - NMS IoU: 0.7
  - Confidence: 0.25

### `sample_data.yaml`
- **Açıklama:** Dataset tanımı
- **Classes:** 2 (potluk, temiz)
- **Dataset:** FKT Potluk Detection Dataset
- **Split:** Train/Val/Test

## 🧪 Kullanım

### Backend Test (Manuel)

```bash
cd backend
python -c "
from app.parsers.yolo_parser import YOLOResultParser

parser = YOLOResultParser('../examples/sample_results.csv', '../examples/sample_args.yaml')
metrics = parser.parse_metrics()
config = parser.parse_config()

print('Metrics:', metrics)
print('Config:', config)
"
```

### Backend Test (API)

```bash
# Terminal 1: Backend başlat
cd backend
uvicorn app.main:app --reload

# Terminal 2: cURL ile test
curl -X POST http://localhost:8000/api/upload/results \
  -F "results_csv=@examples/sample_results.csv" \
  -F "config_yaml=@examples/sample_args.yaml"
```

### Frontend Test

1. Backend'i başlat:
```bash
cd backend
uvicorn app.main:app --reload
```

2. Frontend'i başlat:
```bash
cd frontend
npm install  # İlk seferinde
npm run dev
```

3. Browser'da `http://localhost:5173` aç

4. Sample dosyaları upload et:
   - results.csv → `examples/sample_results.csv`
   - args.yaml → `examples/sample_args.yaml`

5. "Analiz Et" butonuna tıkla

## 📊 Beklenen Sonuçlar

### Metrics (Last Epoch - Epoch 99)
```
Precision: 79.01%
Recall: 81.95%
mAP@0.5: 85.55%
mAP@0.5:0.95: 39.99%
Loss: 0.7325
```

### Config
```
Epochs: 100
Batch Size: 16
Learning Rate: 0.01
IoU Threshold: 0.7
Confidence Threshold: 0.25
```

### LLM Analysis (Örnek)
LLM analizi gerçek API key ile çalışır. Örnek çıktı:

**ÖZET:**
Model performansı iyi seviyede. Precision ve Recall dengeli, mAP50 yüksek ancak mAP50-95 düşük.

**GÜÇLÜ YÖNLER:**
- Yüksek recall (0.82) - Az false negative
- Dengeli precision/recall oranı
- mAP50 oldukça yüksek (0.86)

**ZAYIF YÖNLER:**
- mAP50-95 düşük (0.40) - Farklı IoU threshold'larda performans düşüyor
- Loss hala azalma eğiliminde - Daha fazla epoch gerekebilir

**AKSİYON ÖNERİLERİ:**
1. **BUGÜN:** Validation IoU threshold'unu test et (0.3, 0.4, 0.5, 0.6)
2. **YARIN:** Epoch sayısını 150-200'e çıkar
3. **BU HAFTA:** Dataset'i 600-800 görsel'e çıkar
4. **2-3 HAFTA:** Augmentation parametrelerini optimize et

## 🎯 Test Senaryoları

### Scenario 1: Başarılı Upload
- CSV + YAML upload
- Metrics parse edildi ✅
- LLM analizi döndü ✅

### Scenario 2: Sadece CSV
- Sadece CSV upload
- Metrics parse edildi ✅
- Config bilgisi yok ⚠️

### Scenario 3: Hatalı Dosya
- Boş CSV
- Parser hatası ❌
- Error message gösterildi ✅

## 📝 Notlar

- Sample data FKT projesi karakteristiklerine göre oluşturulmuştur
- Gerçek bir 100 epoch eğitimini simüle eder
- Metrikler gerçekçi değerlerle ilerlemektedir
- LLM prompt'ları FKT domain knowledge içerir

---

**Oluşturulma Tarihi:** 2025-11-01
**Versiyon:** 1.0
