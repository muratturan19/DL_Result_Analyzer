import React, { useState } from 'react';
import './App.css';

// =============================================================================
// COMPONENTS (Ayrı dosyalarda olacak)
// =============================================================================

const FileUploader = ({ onUpload }) => {
  const [files, setFiles] = useState({
    csv: null,
    yaml: null,
    graphs: []
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    if (files.csv) formData.append('results_csv', files.csv);
    if (files.yaml) formData.append('config_yaml', files.yaml);
    files.graphs.forEach(graph => formData.append('graphs', graph));

    try {
      const response = await fetch('http://localhost:8000/api/upload/results', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      onUpload(data);
    } catch (error) {
      console.error('Upload failed:', error);
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
          />
        </div>
        
        <div className="file-input">
          <label>args.yaml (opsiyonel):</label>
          <input 
            type="file" 
            accept=".yaml,.yml"
            onChange={(e) => setFiles({...files, yaml: e.target.files[0]})}
          />
        </div>
        
        <div className="file-input">
          <label>Grafikler (opsiyonel):</label>
          <input 
            type="file" 
            accept=".png,.jpg"
            multiple
            onChange={(e) => setFiles({...files, graphs: Array.from(e.target.files)})}
          />
        </div>
        
        <button type="submit" className="btn-primary">
          Analiz Et 🚀
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

const AIAnalysis = ({ analysis, isLoading }) => {
  if (isLoading) {
    return <div className="loading">🤖 AI analiz yapıyor...</div>;
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

  const handleUpload = async (uploadResponse) => {
    console.log('Upload response:', uploadResponse);

    if (!uploadResponse) {
      return;
    }

    const { metrics: responseMetrics, analysis: responseAnalysis } = uploadResponse;

    if (responseMetrics) {
      setMetrics(responseMetrics);
    }

    if (responseAnalysis) {
      setAnalysis(responseAnalysis);
      setLoading(false);
      return;
    }

    setAnalysis(null);

    if (!responseMetrics) {
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/analyze/metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(responseMetrics)
      });
      const aiAnalysis = await response.json();
      setAnalysis(aiAnalysis);
    } catch (error) {
      console.error('Analysis failed:', error);
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🎯 DL_Result_Analyzer</h1>
        <p>Model performansını analiz et ve AI önerileri al</p>
      </header>
      
      <main className="app-main">
        <FileUploader onUpload={handleUpload} />
        
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
