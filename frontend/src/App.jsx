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

const DatasetSummary = ({ dataset }) => {
  if (!dataset || Object.keys(dataset).length === 0) return null;

  const formatCount = (value) => {
    if (value === null || value === undefined) return null;
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return numeric.toLocaleString('tr-TR');
    }
    if (typeof value === 'string') {
      return value;
    }
    return String(value);
  };

  const countItems = [
    { label: 'Eğitim', value: formatCount(dataset.train_images) },
    { label: 'Doğrulama', value: formatCount(dataset.val_images) },
    { label: 'Test', value: formatCount(dataset.test_images) },
    { label: 'Toplam', value: formatCount(dataset.total_images) }
  ].filter((item) => item.value);

  const pathItems = [
    { label: 'data.yaml', value: dataset.config_path },
    { label: 'Train', value: dataset.train_path },
    { label: 'Val', value: dataset.val_path },
    { label: 'Test', value: dataset.test_path }
  ].filter((item) => item.value);

  const classNames = Array.isArray(dataset.class_names) ? dataset.class_names : [];

  return (
    <div className="dataset-summary-card">
      <h3>🗃️ Veri Seti Özeti</h3>
      {countItems.length > 0 && (
        <div className="dataset-summary-grid">
          {countItems.map((item) => (
            <div key={item.label} className="dataset-count">
              <span className="dataset-count-label">{item.label}</span>
              <span className="dataset-count-value">{item.value}</span>
              <span className="dataset-count-unit">görsel</span>
            </div>
          ))}
        </div>
      )}

      <div className="dataset-summary-meta">
        {classNames.length > 0 && (
          <div className="dataset-classes">
            <span className="dataset-meta-label">Sınıflar</span>
            <div className="dataset-class-list">
              {classNames.map((name, idx) => (
                <span key={`${name}-${idx}`} className="dataset-class-chip">{name}</span>
              ))}
            </div>
          </div>
        )}
        {typeof dataset.class_count === 'number' && (
          <div className="dataset-meta-item">
            <span className="dataset-meta-label">Sınıf Sayısı</span>
            <span className="dataset-meta-value">{dataset.class_count}</span>
          </div>
        )}
        {pathItems.length > 0 && (
          <div className="dataset-paths">
            {pathItems.map((item) => (
              <div key={item.label} className="dataset-meta-item">
                <span className="dataset-meta-label">{item.label}</span>
                <span className="dataset-meta-value dataset-path-value">{item.value}</span>
              </div>
            ))}
          </div>
        )}
      </div>
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
  const [projectInfo, setProjectInfo] = useState({
    projectName: '',
    shortDescription: '',
    classCount: '',
    trainingMethod: '',
    projectFocus: '',
    trainingCode: null
  });

  const methodOptions = [
    { value: 'yolov8-s', label: 'YOLOv8-S' },
    { value: 'yolov8-m', label: 'YOLOv8-M' },
    { value: 'yolov8-l', label: 'YOLOv8-L' },
    { value: 'yolov8-seg', label: 'YOLOv8-Seg' },
    { value: 'yolov11-n', label: 'YOLO11-N' },
    { value: 'yolov11-s', label: 'YOLO11-S' },
    { value: 'yolov11-m', label: 'YOLO11-M' },
    { value: 'yolov11-l', label: 'YOLO11-L' },
    { value: 'yolov11-seg', label: 'YOLO11-Seg' }
  ];

  const focusOptions = [
    { value: 'recall', label: 'Recall Öncelikli' },
    { value: 'precision', label: 'Precision Öncelikli' },
    { value: 'f1', label: 'F1 Dengesi' },
    { value: 'map50', label: 'mAP@0.5 Maksimizasyonu' },
    { value: 'latency', label: 'Çıkarım Hızı / Gecikme' }
  ];

  const updateFiles = (patch) => {
    const updated = { ...files, ...patch };
    setFiles(updated);
    onArtifactsUpdate?.(updated);
  };

  const updateProjectInfo = (key, value) => {
    setProjectInfo((prev) => ({
      ...prev,
      [key]: value
    }));
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

    if (projectInfo.projectName.trim()) {
      formData.append('project_name', projectInfo.projectName.trim());
    }
    if (projectInfo.shortDescription.trim()) {
      formData.append('short_description', projectInfo.shortDescription.trim());
    }
    if (projectInfo.classCount) {
      formData.append('class_count', projectInfo.classCount);
    }
    if (projectInfo.trainingMethod) {
      formData.append('training_method', projectInfo.trainingMethod);
    }
    if (projectInfo.projectFocus) {
      formData.append('project_focus', projectInfo.projectFocus);
    }
    if (projectInfo.trainingCode) {
      formData.append('training_code', projectInfo.trainingCode);
    }

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
        <div className="uploader-grid">
          <div className="uploader-column">
            <div className="file-input">
              <label htmlFor="resultsCsv">results.csv:</label>
              <input
                id="resultsCsv"
                type="file"
                accept=".csv"
                onChange={(e) => updateFiles({ csv: e.target.files?.[0] || null })}
                disabled={isLoading}
              />
            </div>

            <div className="file-input">
              <label htmlFor="argsYaml">args.yaml (opsiyonel):</label>
              <input
                id="argsYaml"
                type="file"
                accept=".yaml,.yml"
                onChange={(e) => updateFiles({ yaml: e.target.files?.[0] || null })}
                disabled={isLoading}
              />
            </div>

            <div className="file-input">
              <label htmlFor="bestModel">best.pt (opsiyonel):</label>
              <input
                id="bestModel"
                type="file"
                accept=".pt"
                onChange={(e) => updateFiles({ best: e.target.files?.[0] || null })}
                disabled={isLoading}
              />
            </div>

            <div className="file-input">
              <label htmlFor="graphs">Grafikler (opsiyonel):</label>
              <input
                id="graphs"
                type="file"
                accept=".png,.jpg"
                multiple
                onChange={(e) => updateFiles({ graphs: Array.from(e.target.files || []) })}
                disabled={isLoading}
              />
            </div>
          </div>

          <div className="uploader-column uploader-metadata">
            <div className="input-group">
              <label htmlFor="projectName">Proje Adı</label>
              <input
                id="projectName"
                type="text"
                placeholder="Örn. FKT Potluk Tespiti"
                value={projectInfo.projectName}
                onChange={(e) => updateProjectInfo('projectName', e.target.value)}
                disabled={isLoading}
              />
            </div>

            <div className="input-group">
              <label htmlFor="shortDescription">Kısa Tanım</label>
              <textarea
                id="shortDescription"
                placeholder="Modelin odağını ve veri setini kısaca anlatın"
                value={projectInfo.shortDescription}
                onChange={(e) => updateProjectInfo('shortDescription', e.target.value)}
                disabled={isLoading}
                rows={3}
              />
            </div>

            <div className="input-row">
              <div className="input-group">
                <label htmlFor="classCount">Class Sayısı</label>
                <input
                  id="classCount"
                  type="number"
                  min="1"
                  placeholder="Örn. 2"
                  value={projectInfo.classCount}
                  onChange={(e) => updateProjectInfo('classCount', e.target.value)}
                  disabled={isLoading}
                />
              </div>

              <div className="input-group">
                <label htmlFor="projectFocus">Proje Odağı</label>
                <select
                  id="projectFocus"
                  value={projectInfo.projectFocus}
                  onChange={(e) => updateProjectInfo('projectFocus', e.target.value)}
                  disabled={isLoading}
                >
                  <option value="">Seçiniz</option>
                  {focusOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="input-row">
              <div className="input-group">
                <label htmlFor="trainingMethod">Kullanılan Metot</label>
                <select
                  id="trainingMethod"
                  value={projectInfo.trainingMethod}
                  onChange={(e) => updateProjectInfo('trainingMethod', e.target.value)}
                  disabled={isLoading}
                >
                  <option value="">Seçiniz</option>
                  {methodOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="file-input compact">
                <label htmlFor="trainingCode">Eğitim Kodu (opsiyonel):</label>
                <input
                  id="trainingCode"
                  type="file"
                  accept=".py,.ipynb,.txt"
                  onChange={(e) => updateProjectInfo('trainingCode', e.target.files?.[0] || null)}
                  disabled={isLoading}
                />
                {projectInfo.trainingCode && (
                  <span className="file-hint">Seçildi: {projectInfo.trainingCode.name}</span>
                )}
              </div>
            </div>
          </div>
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

  const actions = Array.isArray(analysis.actions) ? analysis.actions : [];
  const strengths = Array.isArray(analysis.strengths)
    ? analysis.strengths
    : analysis.strengths
      ? [analysis.strengths]
      : [];
  const weaknesses = Array.isArray(analysis.weaknesses)
    ? analysis.weaknesses
    : analysis.weaknesses
      ? [analysis.weaknesses]
      : [];
  const deployProfile = analysis.deploy_profile || {};
  const calibration = analysis.calibration;

  const rawRisk = analysis?.risk_level
    || (typeof analysis?.risk === 'object' && analysis?.risk?.level)
    || (typeof analysis?.risk === 'object' && analysis?.risk?.LEVEL)
    || analysis?.risk;
  const riskString = rawRisk != null ? String(rawRisk).trim() : '';
  const riskMatch = riskString.match(/(yüksek|yuksek|orta|düşük|dusuk|high|medium|low)/i);
  const normalizedRisk = riskMatch
    ? riskMatch[0]
        .toLowerCase()
        .replace('yüksek', 'yuksek')
        .replace('düşük', 'dusuk')
    : '';
  const riskDisplay = riskMatch ? riskMatch[0].toUpperCase() : (riskString ? riskString.toUpperCase() : '');
  const riskClassName = normalizedRisk ? `risk-${normalizedRisk}` : '';

  const summaryHighlights = useMemo(() => {
    const summarySource = Array.isArray(analysis?.summary)
      ? analysis.summary.join(' ')
      : analysis?.summary;
    if (!summarySource) return [];
    return summarySource
      .split(/\n+/)
      .flatMap((line) => line.split(/(?<=[.!?])\s+(?=[A-ZİĞÖŞÜ0-9])/u))
      .map((item) => item.replace(/^[-•\d\)\(]+\s*/u, '').trim())
      .filter(Boolean);
  }, [analysis?.summary]);

  const notesHighlights = useMemo(() => {
    const notesSource = Array.isArray(analysis?.notes)
      ? analysis.notes.join('\n')
      : analysis?.notes;
    if (!notesSource || typeof notesSource !== 'string') return [];
    return notesSource
      .split(/\n+/)
      .map((item) => item.replace(/^[-•]+\s*/u, '').trim())
      .filter(Boolean);
  }, [analysis?.notes]);

  const renderAction = (action, idx) => {
    if (!action || typeof action !== 'object') {
      return (
        <div key={idx} className="action-card">
          <div className="action-card-header">
            <span className="action-index">{idx + 1}</span>
            <div className="action-card-title">{String(action ?? 'Belirtilmedi')}</div>
          </div>
        </div>
      );
    }

    const {
      module,
      problem,
      evidence,
      recommendation,
      expected_gain,
      validation_plan,
    } = action;

    const formatValue = (value) => {
      if (value == null) return '';
      if (typeof value === 'string') return value;
      if (Array.isArray(value)) return value.map((item) => String(item)).join('\n• ');
      if (typeof value === 'object') return JSON.stringify(value);
      return String(value);
    };

    const fields = [
      { label: 'Problem', value: formatValue(problem) },
      { label: 'Kanıt', value: formatValue(evidence) },
      { label: 'Öneri', value: formatValue(recommendation) },
      { label: 'Beklenen Kazanç', value: formatValue(expected_gain) },
      { label: 'Doğrulama Planı', value: formatValue(validation_plan) },
    ].filter(({ value }) => value);

    return (
      <div key={idx} className="action-card">
        <div className="action-card-header">
          <span className="action-index">{idx + 1}</span>
          <div>
            <div className="action-card-subtitle">Modül</div>
            <div className="action-card-title">{module || 'Belirtilmedi'}</div>
          </div>
        </div>
        <div className="action-card-body">
          {fields.length > 0 ? (
            <dl className="action-fields">
              {fields.map(({ label, value }) => (
                <div key={label} className="action-field">
                  <dt>{label}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="action-empty">Ek detay bulunamadı.</p>
          )}
        </div>
      </div>
    );
  };

  const deployProfileEntries = Object.entries(deployProfile).filter(([, value]) =>
    value !== null && value !== undefined && value !== ''
  );

  const labelMap = {
    release_decision: 'Yayın Kararı',
    rollout_strategy: 'Yayın Planı',
    monitoring_plan: 'İzleme Planı',
    owner: 'Sorumlu',
    notes: 'Notlar',
  };

  const renderDeployValue = (key, value) => {
    if (value === null || value === undefined) return null;
    const displayLabel = labelMap[key] || key.replace(/_/g, ' ');
    return (
      <div key={key} className="deploy-card">
        <div className="deploy-label">{displayLabel}</div>
        <div className="deploy-value">{typeof value === 'string' ? value : JSON.stringify(value)}</div>
      </div>
    );
  };

  return (
    <div className="analysis-panel">
      <div className="analysis-panel-header">
        <h2>🤖 AI Analiz Sonuçları</h2>
        {riskString && (
          <span className={`risk-chip ${riskClassName}`}>
            Risk: {riskDisplay || 'BİLİNMİYOR'}
          </span>
        )}
      </div>

      <div className="analysis-section">
        <h3>📊 Özet</h3>
        {summaryHighlights.length > 1 ? (
          <div className="summary-grid">
            {summaryHighlights.map((item, idx) => (
              <div key={idx} className="summary-card">
                <span>{item}</span>
              </div>
            ))}
          </div>
        ) : (
          <p>{analysis.summary}</p>
        )}
      </div>

      <div className="analysis-columns">
        <div className="analysis-section strengths">
          <h3>✅ Güçlü Yönler</h3>
          <ul>
            {strengths.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>

        <div className="analysis-section weaknesses">
          <h3>⚠️ Zayıf Yönler</h3>
          <ul>
            {weaknesses.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="analysis-section action-items">
        <h3>🎯 Aksiyon Önerileri</h3>
        <div className="actions-list">
          {actions.length > 0 ? (
            actions.map((action, idx) => renderAction(action, idx))
          ) : (
            <p>Henüz aksiyon önerisi bulunmuyor.</p>
          )}
        </div>
      </div>

      {notesHighlights.length > 0 && (
        <div className="analysis-note-callout">
          <span className="note-icon">📝</span>
          <div className="note-content">
            <h4>Notlar</h4>
            <ul>
              {notesHighlights.map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {deployProfileEntries.length > 0 && (
        <div className="analysis-section">
          <h3>🚀 Yayın Profili</h3>
          <div className="deploy-grid">
            {deployProfileEntries.map(([key, value]) => renderDeployValue(key, value))}
          </div>
        </div>
      )}

      {calibration && (
        <div className="analysis-section">
          <h3>🎯 Kalibrasyon Bulguları</h3>
          <pre className="calibration-block">{JSON.stringify(calibration, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

const ReportExportActions = ({ reportId, projectName }) => {
  const [isExporting, setIsExporting] = useState(false);
  const [error, setError] = useState(null);

  const slugify = (value) => {
    if (!value) return 'dl-result-report';
    return value
      .toString()
      .toLowerCase()
      .replace(/[^a-z0-9]+/gi, '-')
      .replace(/^-+|-+$/g, '')
      .replace(/-{2,}/g, '-') || 'dl-result-report';
  };

  const handleExport = async (format) => {
    if (!reportId) return;

    setIsExporting(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:8000/api/report/${reportId}/export?format=${format}`);
      if (!response.ok) {
        let message = 'Rapor dışa aktarılamadı.';
        try {
          const payload = await response.json();
          if (payload?.detail) {
            message = Array.isArray(payload.detail) ? payload.detail[0]?.msg || message : payload.detail;
          }
        } catch (err) {
          console.debug('Export error payload parse failed', err);
        }
        throw new Error(message);
      }

      const blob = await response.blob();
      const extension = format === 'pdf' ? 'pdf' : 'html';
      const slug = slugify(projectName);
      const filename = `${slug}-${reportId.slice(0, 8)}.${extension}`;

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Rapor dışa aktarılamadı.';
      setError(message);
    } finally {
      setIsExporting(false);
    }
  };

  if (!reportId) {
    return null;
  }

  return (
    <div className="export-card">
      <div>
        <h2>📄 Raporu Dışa Aktar</h2>
        <p>LLM analizi ve metrikleri PDF veya HTML olarak kaydedin.</p>
      </div>
      <div className="export-actions">
        <button
          type="button"
          className="export-button"
          disabled={isExporting}
          onClick={() => handleExport('pdf')}
        >
          PDF indir
        </button>
        <button
          type="button"
          className="export-button secondary"
          disabled={isExporting}
          onClick={() => handleExport('html')}
        >
          HTML indir
        </button>
      </div>
      {isExporting && <p className="export-status">Rapor hazırlanıyor…</p>}
      {error && <p className="export-error">{error}</p>}
    </div>
  );
};

const ReportAssistant = ({ reportId, qaHistory, onAsk, isLoading, error, dataset }) => {
  const [question, setQuestion] = useState('');
  const [localError, setLocalError] = useState(null);

  const datasetCounts = useMemo(() => {
    if (!dataset) return [];
    const entries = [
      { label: 'Eğitim', value: dataset.train_images },
      { label: 'Doğrulama', value: dataset.val_images },
      { label: 'Test', value: dataset.test_images },
      { label: 'Toplam', value: dataset.total_images }
    ];
    return entries
      .map(({ label, value }) => {
        if (value === null || value === undefined) return null;
        const numeric = Number(value);
        if (Number.isFinite(numeric)) {
          return `${label}: ${numeric.toLocaleString('tr-TR')} görsel`;
        }
        if (typeof value === 'string' && value.trim()) {
          return `${label}: ${value}`;
        }
        return null;
      })
      .filter(Boolean);
  }, [dataset]);

  const orderedHistory = useMemo(() => {
    if (!Array.isArray(qaHistory)) return [];
    return [...qaHistory].reverse();
  }, [qaHistory]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setLocalError('Lütfen bir soru yazın.');
      return;
    }

    try {
      setLocalError(null);
      await onAsk(trimmed);
      setQuestion('');
    } catch (err) {
      setLocalError(err.message || 'Soru gönderilemedi.');
    }
  };

  const renderReferences = (references = []) => {
    if (!references || references.length === 0) return null;
    return (
      <div className="qa-section">
        <span className="qa-section-title">Referanslar</span>
        <ul>
          {references.map((ref, idx) => (
            <li key={`${ref}-${idx}`}>{ref}</li>
          ))}
        </ul>
      </div>
    );
  };

  const renderFollowUps = (followUps = []) => {
    if (!followUps || followUps.length === 0) return null;
    return (
      <div className="qa-section">
        <span className="qa-section-title">Önerilen sonraki sorular</span>
        <ul>
          {followUps.map((item, idx) => (
            <li key={`${item}-${idx}`}>{item}</li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    <div className="qa-card">
      <div className="qa-card-header">
        <div>
          <h2>💬 Rapor Asistanı</h2>
          <p>Hazırlanan raporla ilgili ek sorular sorup açıklama alın.</p>
        </div>
        {reportId && <span className="qa-report-id">ID: {reportId.slice(0, 8)}…</span>}
      </div>

      {datasetCounts.length > 0 && (
        <div className="qa-dataset-callout">
          <span className="qa-dataset-label">Veri seti görsel sayıları:</span>
          <span className="qa-dataset-values">{datasetCounts.join(' • ')}</span>
        </div>
      )}

      <form className="qa-form" onSubmit={handleSubmit}>
        <textarea
          placeholder="Örn. Eğitimde toplam kaç görsel kullanıldı, recall neden hedefin altında?"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          disabled={isLoading}
          rows={3}
        />
        {(localError || error) && (
          <div className="qa-error">{localError || error}</div>
        )}
        <div className="qa-form-actions">
          <button type="submit" className="btn-primary" disabled={isLoading || !question.trim()}>
            {isLoading ? '🤖 Yanıt aranıyor…' : 'Soruyu Gönder'}
          </button>
        </div>
      </form>

      <div className="qa-history">
        {orderedHistory.length === 0 ? (
          <p className="qa-empty">Henüz soru sorulmadı. İlk soruyu sen sor!</p>
        ) : (
          orderedHistory.map((entry, idx) => (
            <div key={entry.timestamp || idx} className="qa-entry">
              <div className="qa-entry-header">
                <span className="qa-entry-index">Soru #{orderedHistory.length - idx}</span>
                {entry.timestamp && <span className="qa-entry-timestamp">{entry.timestamp}</span>}
              </div>
              <div className="qa-question">
                <span className="qa-section-title">Soru</span>
                <p>{entry.question}</p>
              </div>
              <div className="qa-answer">
                <span className="qa-section-title">Yanıt</span>
                <p style={{ whiteSpace: 'pre-line' }}>{entry.answer || 'Yanıt bulunamadı.'}</p>
              </div>
              {renderReferences(entry.references)}
              {renderFollowUps(entry.follow_up_questions)}
              {entry.notes && (
                <div className="qa-section">
                  <span className="qa-section-title">Not</span>
                  <p>{entry.notes}</p>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

function App() {
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [config, setConfig] = useState(null);
  const [project, setProject] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(null);
  const [error, setError] = useState(null);
  const [llmProvider, setLlmProvider] = useState('claude');
  const [artifacts, setArtifacts] = useState({});
  const [activePage, setActivePage] = useState('dashboard');
  const [reportId, setReportId] = useState(null);
  const [qaHistory, setQaHistory] = useState([]);
  const [qaLoading, setQaLoading] = useState(false);
  const [qaError, setQaError] = useState(null);

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
      setReportId(null);
      setQaHistory([]);
      setQaError(null);
      setQaLoading(false);
      setProject(null);
      return;
    }

    if (uploadResponse._loading) {
      setLoading(true);
      setLoadingStatus('uploading');
      setError(null);
      setReportId(null);
      setQaHistory([]);
      setQaError(null);
      setQaLoading(false);
      setProject(null);
      return;
    }

    if (uploadResponse.error) {
      setError(uploadResponse.error);
      setLoading(false);
      setLoadingStatus('error');
      setReportId(null);
      setQaHistory([]);
      setQaError(null);
      setQaLoading(false);
      setProject(null);
      setTimeout(() => setLoadingStatus(null), 3000);
      return;
    }

    const { metrics: responseMetrics, analysis: responseAnalysis, config: responseConfig, history: responseHistory } = uploadResponse;

    setMetrics(responseMetrics || null);
    setHistory(responseHistory || null);
    setConfig(responseConfig || null);
    setProject(uploadResponse.project || responseConfig?.project_context || null);
    setAnalysis(null);
    setReportId(uploadResponse.report_id || null);
    setQaHistory(Array.isArray(uploadResponse.qa_history) ? uploadResponse.qa_history : []);
    setQaError(null);
    setQaLoading(false);

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

  const handleAskQuestion = async (questionText) => {
    if (!reportId) {
      throw new Error('Aktif bir rapor bulunamadı.');
    }

    setQaLoading(true);
    setQaError(null);

    try {
      const response = await fetch(`http://localhost:8000/api/report/${reportId}/qa`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: questionText, llm_provider: llmProvider })
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = data?.detail || 'Soru yanıtlanamadı.';
        throw new Error(Array.isArray(message) ? message[0]?.msg || message : message);
      }

      setQaHistory((prev) => {
        if (Array.isArray(data?.qa_history)) {
          return data.qa_history;
        }
        if (data?.qa) {
          return [...prev, data.qa];
        }
        return prev;
      });

      if (data?.report_id && data.report_id !== reportId) {
        setReportId(data.report_id);
      }
    } catch (err) {
      console.error('QA request failed:', err);
      const message = err instanceof Error ? err.message : String(err);
      setQaError(message);
      throw new Error(message);
    } finally {
      setQaLoading(false);
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
            <DatasetSummary dataset={config?.dataset} />
            <TrainingCharts history={history} />
            <ConfusionMatrix metrics={metrics} />
            <AIAnalysis analysis={analysis} isLoading={loading} />
            {reportId && (
              <ReportExportActions
                reportId={reportId}
                projectName={
                  (project && (project.project_name || project.name))
                  || config?.project_context?.project_name
                  || config?.project_context?.name
                  || config?.project?.project_name
                  || config?.project?.name
                }
              />
            )}
            {reportId && (
              <ReportAssistant
                reportId={reportId}
                qaHistory={qaHistory}
                onAsk={handleAskQuestion}
                isLoading={qaLoading}
                error={qaError}
                dataset={config?.dataset}
              />
            )}
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
