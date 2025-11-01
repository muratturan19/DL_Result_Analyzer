import React, { useState } from 'react';
import './App.css';

// =============================================================================
// COMPONENTS (Ayrı dosyalarda olacak)
// =============================================================================

const FileUploader = ({ onUpload, isLoading, llmProvider, setLlmProvider }) => {
  const [files, setFiles] = useState({
    csv: null,
    yaml: null,
    graphs: []
  });

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Immediately notify parent that upload has started
    onUpload({ _loading: true });

    const formData = new FormData();
    if (files.csv) formData.append('results_csv', files.csv);
    if (files.yaml) formData.append('config_yaml', files.yaml);
    files.graphs.forEach(graph => formData.append('graphs', graph));
    formData.append('llm_provider', llmProvider);

    try {
      const response = await fetch('http://localhost:8000/api/upload/results', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      onUpload(data);
    } catch (error) {
      console.error('Upload failed:', error);
      onUpload({ error: error.message });
    }
  };

  return (
    <div className="uploader-card">
      <h2>📤 YOLO Sonuçlarını Yükle</h2>
      <form onSubmit={handleSubmit}>
        <div className="file-input">
          <label>results.csv:</label>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFiles({...files, csv: e.target.files[0]})}
            disabled={isLoading}
          />
        </div>

        <div className="file-input">
          <label>args.yaml (opsiyonel):</label>
          <input
            type="file"
            accept=".yaml,.yml"
            onChange={(e) => setFiles({...files, yaml: e.target.files[0]})}
            disabled={isLoading}
          />
        </div>

        <div className="file-input">
          <label>Grafikler (opsiyonel):</label>
          <input
            type="file"
            accept=".png,.jpg"
            multiple
            onChange={(e) => setFiles({...files, graphs: Array.from(e.target.files)})}
            disabled={isLoading}
          />
        </div>

        <div className="llm-provider-section">
          <label className="llm-provider-label">🤖 AI Sağlayıcı:</label>
          <div className="llm-provider-options">
            <label className={`provider-option ${llmProvider === 'claude' ? 'selected' : ''}`}>
              <input
                type="radio"
                name="llm_provider"
                value="claude"
                checked={llmProvider === 'claude'}
                onChange={(e) => setLlmProvider(e.target.value)}
                disabled={isLoading}
              />
              <span className="provider-badge claude">
                <span className="provider-icon">🧠</span>
                Claude
              </span>
            </label>
            <label className={`provider-option ${llmProvider === 'openai' ? 'selected' : ''}`}>
              <input
                type="radio"
                name="llm_provider"
                value="openai"
                checked={llmProvider === 'openai'}
                onChange={(e) => setLlmProvider(e.target.value)}
                disabled={isLoading}
              />
              <span className="provider-badge openai">
                <span className="provider-icon">⚡</span>
                OpenAI
              </span>
            </label>
          </div>
        </div>

        <button type="submit" className="btn-primary" disabled={isLoading || !files.csv}>
          {isLoading ? '⏳ Analiz Ediliyor...' : 'Analiz Et 🚀'}
        </button>
      </form>
    </div>
  );
};

const MetricsDisplay = ({ metrics }) => {
  if (!metrics) return null;

  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <h3>Precision</h3>
        <div className="metric-value">{(metrics.precision * 100).toFixed(1)}%</div>
        <div className={`metric-status ${metrics.precision > 0.7 ? 'good' : 'warning'}`}>
          {metrics.precision > 0.7 ? '✅ İyi' : '⚠️ Düşük'}
        </div>
      </div>
      
      <div className="metric-card">
        <h3>Recall</h3>
        <div className="metric-value">{(metrics.recall * 100).toFixed(1)}%</div>
        <div className={`metric-status ${metrics.recall > 0.7 ? 'good' : 'warning'}`}>
          {metrics.recall > 0.7 ? '✅ İyi' : '⚠️ Düşük'}
        </div>
      </div>
      
      <div className="metric-card">
        <h3>mAP@0.5</h3>
        <div className="metric-value">{(metrics.map50 * 100).toFixed(1)}%</div>
        <div className={`metric-status ${metrics.map50 > 0.6 ? 'good' : 'warning'}`}>
          {metrics.map50 > 0.6 ? '✅ İyi' : '⚠️ Düşük'}
        </div>
      </div>
      
      <div className="metric-card">
        <h3>mAP@0.5:0.95</h3>
        <div className="metric-value">{(metrics.map50_95 * 100).toFixed(1)}%</div>
      </div>
    </div>
  );
};

const LoadingStatus = ({ status }) => {
  if (!status) return null;

  const statusConfig = {
    uploading: {
      icon: '📤',
      text: 'Dosyalar yükleniyor...',
      class: 'status-uploading'
    },
    parsing: {
      icon: '📊',
      text: 'Metrikler analiz ediliyor...',
      class: 'status-parsing'
    },
    analyzing: {
      icon: '🤖',
      text: 'AI analizi yapılıyor...',
      class: 'status-analyzing'
    },
    complete: {
      icon: '✅',
      text: 'Analiz tamamlandı!',
      class: 'status-complete'
    },
    error: {
      icon: '❌',
      text: 'Bir hata oluştu',
      class: 'status-error'
    }
  };

  const config = statusConfig[status] || statusConfig.analyzing;

  return (
    <div className={`loading-status ${config.class}`}>
      <div className="loading-spinner">
        <div className="spinner-icon">{config.icon}</div>
        <div className="spinner-animation"></div>
      </div>
      <div className="loading-text">{config.text}</div>
      <div className="loading-bar">
        <div className="loading-bar-fill"></div>
      </div>
    </div>
  );
};

const AIAnalysis = ({ analysis, isLoading }) => {
  if (isLoading) {
    return <LoadingStatus status="analyzing" />;
  }

  if (!analysis) return null;

  return (
    <div className="analysis-panel">
      <h2>🤖 AI Analiz Sonuçları</h2>
      
      <div className="analysis-section">
        <h3>📊 Özet</h3>
        <p>{analysis.summary}</p>
      </div>
      
      <div className="analysis-columns">
        <div className="analysis-section strengths">
          <h3>✅ Güçlü Yönler</h3>
          <ul>
            {analysis.strengths.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>
        
        <div className="analysis-section weaknesses">
          <h3>⚠️ Zayıf Yönler</h3>
          <ul>
            {analysis.weaknesses.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
      
      <div className="analysis-section action-items">
        <h3>🎯 Aksiyon Önerileri</h3>
        <div className="actions-list">
          {analysis.action_items.map((action, idx) => (
            <div key={idx} className="action-item">
              <span className="action-number">{idx + 1}</span>
              <span className="action-text">{action}</span>
            </div>
          ))}
        </div>
      </div>
      
      <div className={`risk-badge risk-${analysis.risk_level}`}>
        Risk Seviyesi: {analysis.risk_level.toUpperCase()}
      </div>
    </div>
  );
};

// =============================================================================
// MAIN APP
// =============================================================================

function App() {
  const [metrics, setMetrics] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(null);
  const [error, setError] = useState(null);
  const [llmProvider, setLlmProvider] = useState('claude');

  const toNumber = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const handleUpload = async (uploadResponse) => {
    console.log('Upload response:', uploadResponse);

    if (!uploadResponse) {
      setLoading(false);
      setLoadingStatus(null);
      return;
    }

    // Handle loading start signal
    if (uploadResponse._loading) {
      setLoading(true);
      setLoadingStatus('uploading');
      return;
    }

    // Check for upload errors
    if (uploadResponse.error) {
      setError(uploadResponse.error);
      setLoading(false);
      setLoadingStatus('error');
      setTimeout(() => setLoadingStatus(null), 3000);
      return;
    }

    setError(null);
    setLoading(true);
    setLoadingStatus('parsing');

    const { metrics: responseMetrics, analysis: responseAnalysis, config: responseConfig } = uploadResponse;

    if (responseMetrics) {
      setMetrics(responseMetrics);
    }

    if (responseAnalysis) {
      setAnalysis(responseAnalysis);
      setLoadingStatus('complete');
      setTimeout(() => {
        setLoading(false);
        setLoadingStatus(null);
      }, 1500);
      return;
    }

    setAnalysis(null);

    if (!responseMetrics) {
      setLoading(false);
      setLoadingStatus(null);
      return;
    }

    const metricsPayload = {
      precision: responseMetrics.precision ?? 0,
      recall: responseMetrics.recall ?? 0,
      map50: responseMetrics.map50 ?? 0,
      map50_95: responseMetrics.map50_95 ?? 0,
      loss: responseMetrics.loss ?? 0,
      epochs: toNumber(responseConfig?.epochs, 0),
      batch_size: toNumber(responseConfig?.batch, 0),
      learning_rate: toNumber(responseConfig?.lr0, 0),
      iou_threshold: toNumber(responseConfig?.iou, 0.5),
      conf_threshold: toNumber(responseConfig?.conf, 0.5)
    };

    setLoadingStatus('analyzing');
    try {
      const response = await fetch('http://localhost:8000/api/analyze/metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metricsPayload)
      });
      const responseData = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = responseData?.detail || 'Analiz isteği başarısız oldu.';
        throw new Error(Array.isArray(message) ? message[0]?.msg || 'Analiz isteği başarısız oldu.' : message);
      }
      const aiAnalysis = responseData;
      setAnalysis(aiAnalysis);
      setLoadingStatus('complete');
      setTimeout(() => {
        setLoading(false);
        setLoadingStatus(null);
      }, 1500);
    } catch (error) {
      console.error('Analysis failed:', error);
      setAnalysis(null);
      setError(error.message || 'Analiz sırasında bir hata oluştu.');
      setLoadingStatus('error');
      setTimeout(() => {
        setLoading(false);
        setLoadingStatus(null);
      }, 3000);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🎯 DL_Result_Analyzer</h1>
        <p>Model performansını analiz et ve AI önerileri al</p>
      </header>

      <main className="app-main">
        <FileUploader
          onUpload={handleUpload}
          isLoading={loading}
          llmProvider={llmProvider}
          setLlmProvider={setLlmProvider}
        />

        {loadingStatus && <LoadingStatus status={loadingStatus} />}

        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}

        {metrics && (
          <>
            <MetricsDisplay metrics={metrics} />
            <AIAnalysis analysis={analysis} isLoading={loading} />
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>FKT AI Projects © 2025</p>
      </footer>
    </div>
  );
}

export default App;
