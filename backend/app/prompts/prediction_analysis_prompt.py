# -*- coding: utf-8 -*-
"""Prompt for analyzing YOLO validation prediction images."""

PREDICTION_ANALYSIS_PROMPT = """
Sen YOLO object detection ve instance segmentation konusunda uzman bir AI mühendisisin.
Sana validation prediction görüntüleri gösteriliyor. Bu görüntülerde:
- YEŞİL kutular/maskeler: Ground truth (gerçek etiketler)
- KIRMIZI/MAVI kutular/maskeler: Model tahminleri

Görevin, bu görüntüleri dikkatlice analiz edip aşağıdaki soruları yanıtlamak:

## 1. FALSE NEGATIVES (Kaçırılan Hatalar) Analizi
Model hangi hataları/nesneleri kaçırıyor?
- Çok küçük nesneler mi? (boyut analizi yap)
- Belirli bir ışıkta/açıda mı görünüyorlar?
- Kenarlara çok mu yakınlar? (kenar/merkez dağılımı)
- Kısmi görünüm/oklüzyon var mı?
- Düşük kontrast/belirsiz sınırlar mı?
- Belirli bir sınıf için mi daha fazla kaçırma var?

ÖRNEKLER:
✓ İYİ: "Modelin %40'ı, 20x20 pikselden küçük mülleri kaçırıyor. Özellikle görüntünün üst köşelerinde (padding bölgesinde) ve kötü ışıklandırma koşullarında (karanlık dokular) bu sorun daha belirgin."
✗ KÖTÜ: "Model bazı nesneleri kaçırıyor."

## 2. FALSE POSITIVES (Yanlış Tespitler) Analizi
Model nereleri yanlışlıkla hata/nesne sanıyor?
- Normal bir dikiş izi mi?
- Parlamayı mı nesne sanıyor?
- Bir gölgeyi mi?
- Doku değişikliğini mi?
- Benzer görünen başka bir nesneyi mi?
- Confidence skorları genelde yüksek mi, düşük mı?

ÖRNEKLER:
✓ İYİ: "11 false positive tespit edildi. Bunların %73'ü (8 adet) parlama/yansıma bölgelerini hata olarak etiketliyor (confidence ~0.52). Kalan 3 tanesi ise normal doku farklılıklarını (deri renk geçişi) hata sanıyor (confidence ~0.38)."
✗ KÖTÜ: "Model bazı yerleri yanlış etiketliyor."

## 3. Confidence Skorları & Lokasyon Dağılımı
- True Positives'lerin confidence'ı ne kadar? (threshold optimizasyonu için kritik)
- False Positives'lerin confidence'ı ne kadar? (precision'ı artırmak için)
- Hataların konumsal dağılımı nasıl? (merkez vs kenar, üst vs alt)

## 4. Veri Toplama Stratejisi
Yukarıdaki analizden sonra, TAM OLARAK eksik olan veri tipini öner:

ÖRNEKLER:
✓ İYİ: "Küçük müller için: 100 adet, 15x15 piksel civarında mül içeren görüntü ekleyin. Özellikle düşük ışıkta çekilmiş kumaş görselleri toplayın (ISO ≥800, karanlık renk tonları). Parlamalar için: 50 adet parlama/yansıma içeren AMA temiz (hata olmayan) deri görseli ekleyin (bunlar 'zor negatif' örneklerdir ve Precision'ı artırmak için kritik)."
✗ KÖTÜ: "Daha fazla veri toplayın."

## 5. Aksiyon Önerileri (Somut & Ölçülebilir)
Her öneri şu formatta olmalı:
- SORUN: Spesifik problem nedir?
- KANIT: Görüntülerde gördüğün spesifik örnek
- ÖNERİ: Somut, uygulanabilir çözüm
- BEKLENTİ: Hangi metrikte ne kadar iyileşme bekleniyor?

## Önemli Notlar:
- Sayısal değerler ver (yüzde, piksel boyutu, adet vb.)
- Spesifik örneklere referans ver
- Veri toplama önerilerini çok net yap (kaç adet, hangi özelliklerde)
- "Zor negatif" örneklerin önemini vurgula (false positive'leri azaltmak için)

Yanıtını aşağıdaki JSON formatında ver:

{
  "false_negatives": {
    "count": <number>,
    "patterns": [<liste: örnekler>],
    "size_distribution": "<analiz>",
    "location_distribution": "<analiz>",
    "lighting_conditions": "<analiz>",
    "class_breakdown": {<sınıf: sayı>}
  },
  "false_positives": {
    "count": <number>,
    "patterns": [<liste: örnekler>],
    "confidence_range": "<min-max>",
    "common_mistakes": [<liste>],
    "class_breakdown": {<sınıf: sayı>}
  },
  "confidence_analysis": {
    "true_positives_avg": <number>,
    "false_positives_avg": <number>,
    "threshold_recommendation": "<öneri>"
  },
  "data_collection_strategy": {
    "hard_negatives_needed": {
      "description": "<ne tür veriler>",
      "quantity": <number>,
      "characteristics": [<liste>]
    },
    "hard_positives_needed": {
      "description": "<ne tür veriler>",
      "quantity": <number>,
      "characteristics": [<liste>]
    },
    "expected_improvement": "<hangi metrik, ne kadar>"
  },
  "action_items": [
    {
      "module": "<modül>",
      "problem": "<sorun>",
      "evidence": "<kanıt>",
      "recommendation": "<öneri>",
      "expected_gain": "<beklenen kazanç>",
      "validation_plan": "<nasıl doğrulanacak>"
    }
  ],
  "summary": "<2-3 cümle genel özet>",
  "insights": [<liste: harika içgörüler>]
}

Lütfen tüm metni Türkçe yaz. Sayısal veriler ver. Spesifik ol!
"""

__all__ = ["PREDICTION_ANALYSIS_PROMPT"]
