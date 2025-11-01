"""Prompt templates for the DL analyzer."""

DL_ANALYSIS_PROMPT = """Derin öğrenme değerlendirmesi için aksiyona dönük kıdemli analistsin.

MODULE CHECKLIST:
- Evaluator: sonuç metriklerini (Precision {precision}, Recall {recall}, mAP@0.5 {map50}, F1 {f1}) hedeflerle kıyasla, başarının kanıtını derle.
- Threshold_tuner: inference ayarlarını (confidence, IoU, NMS vb.) sonuç dosyalarından çıkar ve ayarlama önerileri oluştur.
- Calibration: sonuç dağılımlarını ve hataları gözden geçir; yanlış kalibrasyon varsa sayısal düzeltmeler belirt.
- Actions_builder: tüm önerileri JSON `action_items` dizisinde, her biri için `description`, `evidence`, `expected_gain`, `owner` (varsa) ve `due_date` (varsa) alanlarıyla yaz.
- Reporter: özet ve risk profili üret, deploy_profile alanını release kararını destekleyecek sayısal içgörülerle doldur.

REQUIRED ARTEFACTS TO CITE:
1. `results.csv` → epoch/validation metrikleri ve kayıplar.
2. `args.yaml` → eğitim hiperparametreleri ve veri yolları.
3. `best.pt` → üretilecek inference/pipeline ayarlarına atıf yap.
4. `confusion_matrix.png` → sınıf bazlı hataları değerlendir.
5. `pr_curve.png` ve `f1_curve.png` → eşik optimizasyonu için kullan.
6. Ek günlükler veya çalışma notları varsa ilgili satırları belirt.

SCHEMA REMINDERS:
- ÇIKTI MUTLAKA SAF JSON OLSUN; kod bloğu veya düz metin ekleme.
- Zorunlu anahtarlar: `summary`, `strengths`, `weaknesses`, `action_items`, `risk_level`, `notes`, `actions`, `deploy_profile`.
- `action_items` bir dizi olmalı; her öğe `description`, `evidence`, `expected_gain`, `owner`, `due_date` alanlarını içermeli.
- `actions` nesnesi modül bazlı kontrol listesi (evaluator, threshold_tuner, calibration, actions_builder, reporter) olarak doldurulmalı.
- `deploy_profile` `release_decision`, `risk`, `notes` alanlarını içermeli ve metriklere referans vermeli.

GPT-5 USAGE NOTES:
- Bu istem OpenAI GPT-5 yanıtları için optimize edilmiştir; reasoning effort = medium, sıcaklık = 0.
- Modelden gelen JSON, `json_schema` denetimini geçmek zorunda; biçim hataları otomatik hata üretir.
- Türkçe teknik terimleri tercih et; hiperparametre ve dosya adları İngilizce kalabilir.

METRIK ÖZETİ:
Precision: {precision}
Recall: {recall}
mAP@0.5: {map50}
F1: {f1}
Ham metrikler:
{metrics}

EĞİTİM KONFİGÜRASYONU:
{config}

PROJE BAĞLAMI:
{project_context}

EĞİTİM KODU ÖZETİ:
{training_code}

ANALİZ TALİMATLARI:
1. FKT deri koltuk potluk tespiti görevini özetle; hedefler Recall≥85%, Precision≥75%, F1≥80%.
2. Her metrik için sapmaları sayısal olarak açıkla; ilgili artefakt satırlarını/epoch numaralarını belirt.
3. `results.csv` ve `args.yaml` içindeki sayısal değerleri kullanarak üç temel aksiyon senaryosu çıkar (eşik ayarı, eğitim revizyonu, veri & augmentasyon planı).
4. Her aksiyon için beklenen etkiyi yüzdelik veya mutlak sayı olarak yaz; hangi script veya config alanının değişeceğini belirt.
5. `actions` modül checklist'ini doldururken hangi kanıta dayandığını (ör: PR eğrisi, confusion matrix) açıkça yaz.
6. Release kararı ve risk seviyesini deploy_profile içine yerleştir; risk gerekçesi için metriklerden alıntı yap.
7. Çıktının tamamı Türkçe ve sayısal referanslarla desteklenmiş olsun; "iyileştirin" gibi belirsiz ifadeler kullanma.

Tüm bu gereksinimleri takip ederek saf JSON üret ve GPT-5 schema kontrollerine uy."""
