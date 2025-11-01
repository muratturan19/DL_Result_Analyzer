import React, { useEffect, useMemo, useState } from 'react';
import './App.css';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  AreaChart,
  Area
} from 'recharts';

const formatPercent = (value, digits = 1) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'N/A';
  return `${(numeric * 100).toFixed(digits)}%`;
};

const calculateTrend = (series = []) => {
  if (!series || series.length < 2) return 0;
  const last = Number(series[series.length - 1]);
  const previous = Number(series[series.length - 2]);
  if (!Number.isFinite(last) || !Number.isFinite(previous)) return 0;
  return last - previous;
};

const MetricProgressBar = ({ value, target }) => {
  const percent = Math.max(0, Math.min(100, value * 100));
  const targetPercent = target * 100;
  const isOnTrack = value >= target;

  return (
    <div className="metric-progress">
      <div className="metric-progress-track">
        <div
          className={`metric-progress-fill ${isOnTrack ? 'good' : 'warning'}`}
          style={{ width: `${percent}%` }}
        />
        <div
          className="metric-progress-target"
          style={{ left: `${targetPercent}%` }}
        />
      </div>
      <div className="metric-progress-labels">
        <span>0%</span>
        <span>{(target * 100).toFixed(0)}% hedef</span>
        <span>100%</span>
      </div>
    </div>
  );
};

const MetricCard = ({ title, value, target, trend }) => {
  const status = value >= target ? 'good' : 'warning';
  const trendArrow = trend > 0 ? '↑' : trend < 0 ? '↓' : '→';
  const trendClass = trend > 0 ? 'trend-up' : trend < 0 ? 'trend-down' : 'trend-flat';

  return (
    <div className={`metric-card enhanced ${status}`}>
      <div className="metric-card-header">
        <h3>{title}</h3>
        <span className={`metric-trend ${trendClass}`}>{trendArrow} {(trend * 100).toFixed(1)}%</span>
      </div>
      <div className="metric-value">{formatPercent(value)}</div>
      <MetricProgressBar value={value} target={target} />
    </div>
  );
};

const ProductionScoreCard = ({ precision, recall, map50 }) => {
  const precisionScore = Number.isFinite(precision) ? precision : 0;
  const recallScore = Number.isFinite(recall) ? recall : 0;
  const mapScore = Number.isFinite(map50) ? map50 : 0;
  const score = (recallScore * 0.5 + precisionScore * 0.3 + mapScore * 0.2) * 100;
  const status = score >= 85 ? 'ready' : score >= 70 ? 'medium' : 'low';

  return (
    <div className={`production-score-card status-${status}`}>
      <div className="production-score-title">Production Hazırlık Skoru</div>
      <div className="production-score-value">{score.toFixed(1)}</div>
      <div className="production-score-bar">
        <div style={{ width: `${Math.min(score, 100)}%` }} />
      </div>
      <p className="production-score-note">Recall ağırlıklı birleşik skor (hedef ≥ 85)</p>
    </div>
  );
};

const MetricsDisplay = ({ metrics, history }) => {
  if (!metrics) return null;

  const trends = useMemo(() => ({
    precision: calculateTrend(history?.precision),
    recall: calculateTrend(history?.recall),
    map50: calculateTrend(history?.map50),
  }), [history]);

  const f1 = metrics.precision && metrics.recall
    ? (2 * metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
    : 0;

  return (
    <div className="metrics-section">
      <div className="metrics-grid enhanced">
        <MetricCard title="Precision" value={metrics.precision ?? 0} target={0.75} trend={trends.precision ?? 0} />
        <MetricCard title="Recall" value={metrics.recall ?? 0} target={0.85} trend={trends.recall ?? 0} />
        <MetricCard title="mAP@0.5" value={metrics.map50 ?? 0} target={0.8} trend={trends.map50 ?? 0} />
        <div className="metric-card enhanced neutral">
          <div className="metric-card-header">
            <h3>F1 Skoru</h3>
            <span className="metric-info">Hedef ≥ 0.80</span>
          </div>
          <div className="metric-value">{formatPercent(f1)}</div>
          <MetricProgressBar value={f1} target={0.8} />
        </div>
      </div>
      <ProductionScoreCard
        precision={metrics.precision ?? 0}
        recall={metrics.recall ?? 0}
        map50={metrics.map50 ?? 0}
      />
    </div>
  );
};

const TrainingCharts = ({ history }) => {
  if (!history || !history.epochs || history.epochs.length === 0) return null;

  const epochData = useMemo(() => {
    return history.epochs.map((epoch, idx) => ({
      epoch,
      train_loss: history.train_box_loss?.[idx],
      val_loss: history.val_box_loss?.[idx],
      precision: history.precision?.[idx],
      recall: history.recall?.[idx],
      map50: history.map50?.[idx],
      map50_95: history.map50_95?.[idx]
    }));
  }, [history]);

  return (
    <div className="charts-grid">
      <div className="chart-card">
        <h3>Epoch Loss Eğrileri</h3>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={epochData} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" />
            <YAxis />
            <Tooltip formatter={(value) => value?.toFixed?.(4)} />
            <Legend />
            <Line type="monotone" dataKey="train_loss" stroke="#4f46e5" name="Train Loss" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="val_loss" stroke="#10b981" name="Val Loss" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-card">
        <h3>Precision / Recall Trendleri</h3>
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={epochData} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorPrecision" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.7} />
                <stop offset="95%" stopColor="#4f46e5" stopOpacity={0.1} />
              </linearGradient>
              <linearGradient id="colorRecall" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.7} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" />
            <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
            <Tooltip formatter={(value) => formatPercent(value, 2)} />
            <Legend />
            <Area type="monotone" dataKey="precision" stroke="#4f46e5" fill="url(#colorPrecision)" name="Precision" />
            <Area type="monotone" dataKey="recall" stroke="#f59e0b" fill="url(#colorRecall)" name="Recall" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-card">
        <h3>mAP Eğrileri</h3>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={epochData} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" />
            <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
            <Tooltip formatter={(value) => formatPercent(value, 2)} />
            <Legend />
            <Line type="monotone" dataKey="map50" stroke="#0ea5e9" name="mAP@0.5" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="map50_95" stroke="#6366f1" name="mAP@0.5:0.95" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const ConfusionMatrix = ({ metrics }) => {
  if (!metrics?.precision || !metrics?.recall) return null;

  const positiveSamples = 100;
  const tp = Math.round(metrics.recall * positiveSamples);
  const fn = positiveSamples - tp;
  const fp = Math.max(0, Math.round(tp * ((1 - metrics.precision) / Math.max(metrics.precision, 1e-6))));
  const tn = Math.max(0, 100 - fp);

  const matrix = [
    { label: 'Gerçek Potluk', values: [tp, fn], rowLabel: 'TP', colLabel: 'FN' },
    { label: 'Gerçek Temiz', values: [fp, tn], rowLabel: 'FP', colLabel: 'TN' }
  ];

  const formatCell = (value) => `${value}`;

  return (
    <div className="confusion-card">
      <h3>Confusion Matrix (Tahmini)</h3>
      <table className="confusion-table">
        <thead>
          <tr>
            <th></th>
            <th>Tahmin Potluk</th>
            <th>Tahmin Temiz</th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row) => (
            <tr key={row.label}>
              <th>{row.label}</th>
              {row.values.map((value, idx) => (
                <td key={idx}>
                  <div className="confusion-cell">
                    <span className="confusion-value">{formatCell(value)}</span>
                    <span className="confusion-tag">{idx === 0 ? row.rowLabel : row.colLabel}</span>
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const HeatmapGrid = ({ title, metric, data, xValues, yValues }) => {
  if (!data || !xValues || !yValues) return null;

  const maxValue = Math.max(...data.map((item) => item[metric] ?? 0));
  const minValue = Math.min(...data.map((item) => item[metric] ?? 0));

  const colorForValue = (value) => {
    if (!Number.isFinite(value)) return '#e5e7eb';
    const ratio = maxValue === minValue ? 0.5 : (value - minValue) / (maxValue - minValue);
    const opacity = 0.15 + ratio * 0.75;
    return `rgba(79, 70, 229, ${opacity})`;
  };

  return (
    <div className="heatmap-card">
      <h4>{title}</h4>
      <div className="heatmap-grid">
        <div className="heatmap-axis heatmap-axis-y">
          {yValues.map((value) => (
            <div key={value} className="heatmap-axis-label">IoU {value.toFixed(2)}</div>
          ))}
        </div>
        <div className="heatmap-body">
          <div className="heatmap-axis heatmap-axis-x">
            {xValues.map((value) => (
              <div key={value} className="heatmap-axis-label">Conf {value.toFixed(2)}</div>
            ))}
          </div>
          <div className="heatmap-cells" style={{ gridTemplateColumns: `repeat(${xValues.length}, 1fr)` }}>
            {yValues.map((iou) =>
              xValues.map((conf) => {
                const cell = data.find((item) => Math.abs(item.iou - iou) < 1e-6 && Math.abs(item.confidence - conf) < 1e-6);
                const value = cell?.[metric] ?? 0;
                return (
                  <div
                    key={`${iou}-${conf}-${metric}`}
                    className="heatmap-cell"
                    style={{ backgroundColor: colorForValue(value) }}
                  >
                    <span>{(value * 100).toFixed(1)}%</span>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const ThresholdOptimizer = ({ initialArtifacts }) => {
  const [bestFile, setBestFile] = useState(initialArtifacts?.best || null);
  const [dataFile, setDataFile] = useState(initialArtifacts?.yaml || null);
  const [iouRange, setIouRange] = useState({ start: 0.3, end: 0.7, step: 0.05 });
  const [confRange, setConfRange] = useState({ start: 0.1, end: 0.5, step: 0.05 });
  const [result, setResult] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (initialArtifacts?.best && (!bestFile || initialArtifacts.best.name !== bestFile.name)) {
      setBestFile(initialArtifacts.best);
    }
    if (initialArtifacts?.yaml && (!dataFile || initialArtifacts.yaml.name !== dataFile.name)) {
      setDataFile(initialArtifacts.yaml);
    }
  }, [initialArtifacts]);

  const handleOptimize = async () => {
    if (!bestFile || !dataFile) {
      setError('best.pt ve data.yaml dosyalarını seçmelisiniz.');
      return;
    }

    setError(null);
    setResult(null);
    setIsRunning(true);

    const formData = new FormData();
    formData.append('best_model', bestFile);
    formData.append('data_yaml', dataFile);
    formData.append('iou_range', JSON.stringify(iouRange));
    formData.append('conf_range', JSON.stringify(confRange));

    try {
      const response = await fetch('http://localhost:8000/api/optimize/thresholds', {
        method: 'POST',
        body: formData
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.detail || 'Optimizasyon başarısız oldu');
      }

      setResult(payload);
    } catch (err) {
      console.error('Threshold optimization failed:', err);
      setResult(null);
      setError(
        err.message ||
          'Optimizasyon sırasında hata oluştu. Backend şu anda gerçek YOLO değerlendirmesi sunmuyor olabilir.'
      );
      setError(err.message || 'Optimizasyon sırasında hata oluştu.');
    } finally {
      setIsRunning(false);
    }
  };

  const downloadConfig = () => {
    if (!result?.production_config?.base64) return;
    const link = document.createElement('a');
    link.href = `data:text/yaml;base64,${result.production_config.base64}`;
    link.download = result.production_config.filename || 'production_config.yaml';
    link.click();
  };

  return (
    <div className="optimizer-card">
      <h2>🎛️ Threshold Optimizer</h2>
      <p className="optimizer-subtitle">IoU ve Confidence kombinasyonlarını grid search ile tarayın, production için en iyi eşikleri belirleyin.</p>

      <div className="optimizer-grid">
        <div className="optimizer-inputs">
          <div className="file-input">
            <label>best.pt:</label>
            <input
              type="file"
              accept=".pt"
              onChange={(e) => setBestFile(e.target.files?.[0] || null)}
            />
            {bestFile && <span className="file-hint">Seçildi: {bestFile.name}</span>}
          </div>

          <div className="file-input">
            <label>data.yaml:</label>
            <input
              type="file"
              accept=".yaml,.yml"
              onChange={(e) => setDataFile(e.target.files?.[0] || null)}
            />
            {dataFile && <span className="file-hint">Seçildi: {dataFile.name}</span>}
          </div>

          <div className="range-input-group">
            <label>IoU Aralığı</label>
            <div className="range-inputs">
              <input
                type="number"
                step="0.01"
                min="0.1"
                max="0.9"
                value={iouRange.start}
                onChange={(e) => setIouRange({ ...iouRange, start: Number(e.target.value) })}
              />
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.01"
                value={iouRange.start}
                onChange={(e) => setIouRange({ ...iouRange, start: Number(e.target.value) })}
              />
              <input
                type="range"
                min={iouRange.start + 0.01}
                max="0.95"
                step="0.01"
                value={iouRange.end}
                onChange={(e) => setIouRange({ ...iouRange, end: Number(e.target.value) })}
              />
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="0.5"
                value={iouRange.end}
                onChange={(e) => setIouRange({ ...iouRange, end: Number(e.target.value) })}
              />
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="0.3"
                value={iouRange.step}
                onChange={(e) => setIouRange({ ...iouRange, step: Number(e.target.value) })}
              />
            </div>
          </div>

          <div className="range-input-group">
            <label>Confidence Aralığı</label>
            <div className="range-inputs">
              <input
                type="number"
                step="0.01"
                min="0.05"
                max="0.8"
                value={confRange.start}
                onChange={(e) => setConfRange({ ...confRange, start: Number(e.target.value) })}
              />
              <input
                type="range"
                min="0.05"
                max="0.8"
                step="0.01"
                value={confRange.start}
                onChange={(e) => setConfRange({ ...confRange, start: Number(e.target.value) })}
              />
              <input
                type="range"
                min={confRange.start + 0.01}
                max="0.95"
                step="0.01"
                value={confRange.end}
                onChange={(e) => setConfRange({ ...confRange, end: Number(e.target.value) })}
              />
              <input
                type="number"
                step="0.01"
                min="0.1"
                max="0.95"
                value={confRange.end}
                onChange={(e) => setConfRange({ ...confRange, end: Number(e.target.value) })}
              />
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="0.3"
                value={confRange.step}
                onChange={(e) => setConfRange({ ...confRange, step: Number(e.target.value) })}
              />
            </div>
          </div>

          <button className="btn-primary" type="button" onClick={handleOptimize} disabled={isRunning}>
            {isRunning ? '⏳ Optimizasyon...' : 'Optimizasyonu Başlat'}
          </button>

          {error && <div className="optimizer-error">{error}</div>}
        </div>

        {result && (
          <div className="optimizer-results">
            <div className="optimizer-best">
              <h3>En İyi Eşikler</h3>
              <p><strong>Confidence:</strong> {result.best.confidence.toFixed(2)}</p>
              <p><strong>IoU:</strong> {result.best.iou.toFixed(2)}</p>
              <p><strong>Recall:</strong> {formatPercent(result.best.recall, 2)}</p>
              <p><strong>Precision:</strong> {formatPercent(result.best.precision, 2)}</p>
              <p><strong>F1:</strong> {formatPercent(result.best.f1, 2)}</p>
              <button className="btn-secondary" type="button" onClick={downloadConfig}>
                production_config.yaml indir
              </button>
            </div>

            <div className="heatmap-wrapper">
              <HeatmapGrid
                title="Recall Heatmap"
                metric="recall"
                data={result.heatmap.values}
                xValues={result.heatmap.confidence_values}
                yValues={result.heatmap.iou_values}
              />
              <HeatmapGrid
                title="Precision Heatmap"
                metric="precision"
                data={result.heatmap.values}
                xValues={result.heatmap.confidence_values}
                yValues={result.heatmap.iou_values}
              />
              <HeatmapGrid
                title="F1 Heatmap"
                metric="f1"
                data={result.heatmap.values}
                xValues={result.heatmap.confidence_values}
                yValues={result.heatmap.iou_values}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const FileUploader = ({ onUpload, onArtifactsUpdate, isLoading, llmProvider, setLlmProvider }) => {
  const [files, setFiles] = useState({
    csv: null,
    yaml: null,
    graphs: [],
    best: null
  });

  const updateFiles = (patch) => {
    const updated = { ...files, ...patch };
    setFiles(updated);
    onArtifactsUpdate?.(updated);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    onUpload({ _loading: true });

    const formData = new FormData();
    if (files.csv) formData.append('results_csv', files.csv);
    if (files.yaml) formData.append('config_yaml', files.yaml);
    if (files.best) formData.append('best_model', files.best);
    files.graphs.forEach((graph) => formData.append('graphs', graph));
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
            onChange={(e) => updateFiles({ csv: e.target.files?.[0] || null })}
            disabled={isLoading}
          />
        </div>

        <div className="file-input">
          <label>args.yaml (opsiyonel):</label>
          <input
            type="file"
            accept=".yaml,.yml"
            onChange={(e) => updateFiles({ yaml: e.target.files?.[0] || null })}
            disabled={isLoading}
          />
        </div>

        <div className="file-input">
          <label>best.pt (opsiyonel):</label>
          <input
            type="file"
            accept=".pt"
            onChange={(e) => updateFiles({ best: e.target.files?.[0] || null })}
            disabled={isLoading}
          />
        </div>

        <div className="file-input">
          <label>Grafikler (opsiyonel):</label>
          <input
            type="file"
            accept=".png,.jpg"
            multiple
            onChange={(e) => updateFiles({ graphs: Array.from(e.target.files || []) })}
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
  if (isLoading && !analysis) {
    return <LoadingStatus status="analyzing" />;
  }

  if (!analysis) return null;

  const actionItems = Array.isArray(analysis.action_items) ? analysis.action_items : [];

  const renderActionText = (item) => {
    if (typeof item === 'string') return item;
    if (item?.description) return item.description;
    return JSON.stringify(item);
  };

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
            {analysis.strengths?.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>

        <div className="analysis-section weaknesses">
          <h3>⚠️ Zayıf Yönler</h3>
          <ul>
            {analysis.weaknesses?.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="analysis-section action-items">
        <h3>🎯 Aksiyon Önerileri</h3>
        <div className="actions-list">
          {actionItems.map((action, idx) => (
            <div key={idx} className="action-item">
              <span className="action-number">{idx + 1}</span>
              <span className="action-text">{renderActionText(action)}</span>
            </div>
          ))}
        </div>
      </div>

      <div className={`risk-badge risk-${analysis.risk_level}`}>
        Risk Seviyesi: {analysis.risk_level?.toUpperCase?.()}
      </div>

      {analysis.notes && (
        <div className="analysis-section">
          <h3>📝 Notlar</h3>
          <p>{analysis.notes}</p>
        </div>
      )}
    </div>
  );
};

function App() {
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(null);
  const [error, setError] = useState(null);
  const [llmProvider, setLlmProvider] = useState('claude');
  const [artifacts, setArtifacts] = useState({});
  const [activePage, setActivePage] = useState('dashboard');

  const navigationItems = [
    { id: 'dashboard', label: 'Model Özeti', icon: '📊' },
    { id: 'threshold', label: 'Threshold Optimizer', icon: '🎛️' }
  ];

  const toNumber = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const handleUpload = async (uploadResponse) => {
    if (!uploadResponse) {
      setLoading(false);
      setLoadingStatus(null);
      return;
    }

    if (uploadResponse._loading) {
      setLoading(true);
      setLoadingStatus('uploading');
      setError(null);
      return;
    }

    if (uploadResponse.error) {
      setError(uploadResponse.error);
      setLoading(false);
      setLoadingStatus('error');
      setTimeout(() => setLoadingStatus(null), 3000);
      return;
    }

    const { metrics: responseMetrics, analysis: responseAnalysis, config: responseConfig, history: responseHistory } = uploadResponse;

    setMetrics(responseMetrics || null);
    setHistory(responseHistory || null);
    setConfig(responseConfig || null);
    setAnalysis(null);

    setLoading(true);
    setLoadingStatus('parsing');

    if (uploadResponse.files) {
      setArtifacts((prev) => ({ ...prev, server: uploadResponse.files }));
    }

    if (responseAnalysis) {
      setAnalysis(responseAnalysis);
      setLoading(false);
      setLoadingStatus('complete');
      setTimeout(() => setLoadingStatus(null), 1500);
      return;
    }

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
      setAnalysis(responseData);
      setLoadingStatus('complete');
      setTimeout(() => {
        setLoading(false);
        setLoadingStatus(null);
      }, 1500);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(err.message || 'Analiz sırasında bir hata oluştu.');
      setLoadingStatus('error');
      setTimeout(() => {
        setLoading(false);
        setLoadingStatus(null);
      }, 3000);
    }
  };

  const renderDashboard = () => (
    <>
      <header className="app-header">
        <h1>🎯 DL_Result_Analyzer</h1>
        <p>YOLO11 modelinizi değerlendirin, recall odaklı aksiyon planları çıkarın.</p>
      </header>

      <main className="app-main">
        <FileUploader
          onUpload={handleUpload}
          onArtifactsUpdate={(payload) => setArtifacts((prev) => ({ ...prev, client: payload }))}
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
            <MetricsDisplay metrics={metrics} history={history} />
            <TrainingCharts history={history} />
            <ConfusionMatrix metrics={metrics} />
            <AIAnalysis analysis={analysis} isLoading={loading} />
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>FKT AI Projects © 2025</p>
      </footer>
    </>
  );

  const renderThresholdPage = () => (
    <>
      <header className="app-header threshold">
        <h1>🎛️ Threshold Optimizer</h1>
        <p>IoU ve confidence eşiklerini ayrı bir çalışma alanında optimize edin.</p>
      </header>

      <main className="app-main threshold-main">
        <ThresholdOptimizer
          initialArtifacts={{
            best: artifacts?.client?.best || artifacts?.server?.best || null,
            yaml: artifacts?.client?.yaml || artifacts?.server?.yaml || null
          }}
        />
      </main>

      <footer className="app-footer">
        <p>FKT AI Projects © 2025</p>
      </footer>
    </>
  );

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="sidebar-brand">
          <span className="brand-icon">🧠</span>
          <span className="brand-title">DL Analyzer</span>
        </div>
        <nav className="sidebar-nav">
          {navigationItems.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`sidebar-nav-item ${activePage === item.id ? 'active' : ''}`}
              onClick={() => setActivePage(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <div className="app-content">
        <div className="app-container">
          {activePage === 'dashboard' ? renderDashboard() : renderThresholdPage()}
        </div>
      </div>
    </div>
  );
}

export default App;
