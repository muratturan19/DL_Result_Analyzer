import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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

const formatDuration = (seconds) => {
  const numeric = Number(seconds);
  if (!Number.isFinite(numeric) || numeric < 0) return '—';
  const totalSeconds = Math.floor(numeric);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;
  const parts = [
    hours > 0 ? String(hours).padStart(2, '0') : null,
    String(hours > 0 ? minutes : Math.max(minutes, 0)).padStart(2, '0'),
    String(secs).padStart(2, '0')
  ].filter(Boolean);
  return parts.join(':');
};

const buildRangeValues = (range) => {
  if (!range) return [];
  const start = Number(range.start);
  const end = Number(range.end);
  const step = Number(range.step);
  if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(step) || step <= 0) {
    return [];
  }
  const values = [];
  for (let current = start; current <= end + 1e-9; current += step) {
    values.push(Number(current.toFixed(6)));
  }
  if (values.length === 0 || values[values.length - 1] < Number(end.toFixed(6))) {
    values.push(Number(end.toFixed(6)));
  }
  const unique = Array.from(new Set(values.map((value) => Number(value.toFixed(6)))));
  unique.sort((a, b) => a - b);
  return unique;
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
  const [bestSource, setBestSource] = useState(() => {
    const artifact = initialArtifacts?.best;
    if (artifact instanceof File) {
      return { kind: 'upload', file: artifact };
    }
    if (typeof artifact === 'string' && artifact) {
      return { kind: 'server', filename: artifact };
    }
    return null;
  });
  const [dataSource, setDataSource] = useState(() => {
    const artifact = initialArtifacts?.yaml;
    if (artifact instanceof File) {
      return { kind: 'upload', file: artifact };
    }
    if (typeof artifact === 'string' && artifact) {
      return { kind: 'server', filename: artifact };
    }
    return null;
  });
  const [split, setSplit] = useState('test');
  const [iouRange, setIouRange] = useState({ start: 0.3, end: 0.7, step: 0.05 });
  const [confRange, setConfRange] = useState({ start: 0.1, end: 0.5, step: 0.05 });
  const [result, setResult] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [datasetRoot, setDatasetRoot] = useState('');
  const [progressInfo, setProgressInfo] = useState(null);
  const [progressError, setProgressError] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [reportId, setReportId] = useState(null);
  const [isExporting, setIsExporting] = useState(false);
  const [savedReports, setSavedReports] = useState([]);
  const [showReportsList, setShowReportsList] = useState(false);
  const [isLoadingReports, setIsLoadingReports] = useState(false);
  const progressPollRef = useRef(null);
  const elapsedTimerRef = useRef(null);

  useEffect(() => {
    if (!initialArtifacts) return;

    const bestArtifact = initialArtifacts.best;
    if (bestArtifact instanceof File) {
      setBestSource((prev) => {
        if (prev?.kind === 'upload' && prev.file === bestArtifact) {
          return prev;
        }
        return { kind: 'upload', file: bestArtifact };
      });
    } else if (typeof bestArtifact === 'string' && bestArtifact) {
      setBestSource((prev) => {
        if (prev?.kind === 'upload') {
          return prev;
        }
        if (prev?.kind === 'server' && prev.filename === bestArtifact) {
          return prev;
        }
        return { kind: 'server', filename: bestArtifact };
      });
    }

    const yamlArtifact = initialArtifacts.yaml;
    if (yamlArtifact instanceof File) {
      setDataSource((prev) => {
        if (prev?.kind === 'upload' && prev.file === yamlArtifact) {
          return prev;
        }
        return { kind: 'upload', file: yamlArtifact };
      });
    } else if (typeof yamlArtifact === 'string' && yamlArtifact) {
      setDataSource((prev) => {
        if (prev?.kind === 'upload') {
          return prev;
        }
        if (prev?.kind === 'server' && prev.filename === yamlArtifact) {
          return prev;
        }
        return { kind: 'server', filename: yamlArtifact };
      });
    }
  }, [initialArtifacts]);

  const estimatedIouValues = useMemo(() => buildRangeValues(iouRange), [iouRange]);
  const estimatedConfValues = useMemo(() => buildRangeValues(confRange), [confRange]);
  const estimatedTotalCycles = estimatedIouValues.length * estimatedConfValues.length;

  const serverBestName = useMemo(() => {
    if (initialArtifacts?.serverBest) return initialArtifacts.serverBest;
    if (typeof initialArtifacts?.best === 'string') return initialArtifacts.best;
    return null;
  }, [initialArtifacts]);

  const serverYamlName = useMemo(() => {
    if (initialArtifacts?.serverYaml) return initialArtifacts.serverYaml;
    if (typeof initialArtifacts?.yaml === 'string') return initialArtifacts.yaml;
    return null;
  }, [initialArtifacts]);

  const combinationsTested = useMemo(() => {
    if (!result?.heatmap?.values) return 0;
    return result.heatmap.values.length;
  }, [result]);

  const stopProgressPolling = useCallback(() => {
    if (progressPollRef.current) {
      clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }
  }, []);

  const stopElapsedTimer = useCallback(() => {
    if (elapsedTimerRef.current) {
      clearInterval(elapsedTimerRef.current);
      elapsedTimerRef.current = null;
    }
  }, []);

  const clearProgressTimers = useCallback(() => {
    stopProgressPolling();
    stopElapsedTimer();
  }, [stopProgressPolling, stopElapsedTimer]);

  useEffect(() => () => clearProgressTimers(), [clearProgressTimers]);

  const fetchProgressSnapshot = useCallback(
    async (id) => {
      if (!id) return false;
      try {
        const response = await fetch(`http://localhost:8000/api/optimize/thresholds/status/${id}`);
        if (!response.ok) {
          if (response.status === 404) {
            return false;
          }
          throw new Error(`Status ${response.status}`);
        }
        const payload = await response.json();
        setProgressInfo(payload);
        setProgressError(null);
        return true;
      } catch (err) {
        console.error('Progress polling failed:', err);
        setProgressError('İlerleme bilgisi alınamadı. Bağlantıyı kontrol edin.');
        return false;
      }
    },
    []
  );

  const startProgressPolling = useCallback(
    (id) => {
      if (!id) return;
      stopProgressPolling();
      const poll = () => {
        void fetchProgressSnapshot(id);
      };
      poll();
      progressPollRef.current = setInterval(poll, 2000);
    },
    [stopProgressPolling, fetchProgressSnapshot]
  );

  const startElapsedTimer = useCallback(() => {
    stopElapsedTimer();
    const startedAt = Date.now();
    setElapsedSeconds(0);
    elapsedTimerRef.current = setInterval(() => {
      const diff = Math.floor((Date.now() - startedAt) / 1000);
      setElapsedSeconds(diff);
    }, 1000);
  }, [stopElapsedTimer]);

  const updateIouRange = (key, value) => {
    const numeric = Number(value);
    setIouRange((prev) => {
      if (!Number.isFinite(numeric)) {
        return prev;
      }
      if (key === 'start') {
        const start = Math.min(Math.max(numeric, 0.1), 0.9);
        const end = Math.min(0.95, Math.max(prev.end, start + 0.01));
        return { ...prev, start, end };
      }
      if (key === 'end') {
        const end = Math.min(Math.max(numeric, prev.start + 0.01), 0.95);
        return { ...prev, end };
      }
      if (key === 'step') {
        const step = Math.min(Math.max(numeric, 0.01), 0.3);
        return { ...prev, step };
      }
      return prev;
    });
  };

  const updateConfRange = (key, value) => {
    const numeric = Number(value);
    setConfRange((prev) => {
      if (!Number.isFinite(numeric)) {
        return prev;
      }
      if (key === 'start') {
        const start = Math.min(Math.max(numeric, 0.05), 0.8);
        const end = Math.min(0.95, Math.max(prev.end, start + 0.01));
        return { ...prev, start, end };
      }
      if (key === 'end') {
        const end = Math.min(Math.max(numeric, prev.start + 0.01), 0.95);
        return { ...prev, end };
      }
      if (key === 'step') {
        const step = Math.min(Math.max(numeric, 0.01), 0.3);
        return { ...prev, step };
      }
      return prev;
    });
  };

  const handleOptimize = async () => {
    if (!bestSource) {
      setError('best.pt dosyası gerekli. Sunucuda kayıtlı değilse dosya seçin.');
      return;
    }
    if (!dataSource) {
      setError('data.yaml dosyası gerekli. Sunucuda kayıtlı değilse dosya seçin.');
      return;
    }

    setError(null);
    setResult(null);
    setProgressError(null);
    stopProgressPolling();
    stopElapsedTimer();

    const newProgressId =
      typeof crypto !== 'undefined' && crypto?.randomUUID
        ? crypto.randomUUID()
        : `progress-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const startedAtIso = new Date().toISOString();
    setProgressInfo({
      progress_id: newProgressId,
      status: 'running',
      total_cycles: estimatedTotalCycles,
      completed_cycles: 0,
      progress: estimatedTotalCycles > 0 ? 0 : null,
      current_iou: null,
      current_confidence: null,
      started_at: startedAtIso,
      updated_at: startedAtIso,
      elapsed_seconds: 0,
      estimated_remaining_seconds: null
    });
    setElapsedSeconds(0);
    startElapsedTimer();
    startProgressPolling(newProgressId);

    setIsRunning(true);

    const formData = new FormData();
    if (bestSource.kind === 'upload') {
      formData.append('best_model', bestSource.file);
    } else if (bestSource.kind === 'server') {
      formData.append('best_model_filename', bestSource.filename);
    }

    if (dataSource.kind === 'upload') {
      formData.append('data_yaml', dataSource.file);
    } else if (dataSource.kind === 'server') {
      formData.append('data_yaml_filename', dataSource.filename);
    }

    const trimmedDatasetRoot = datasetRoot.trim();
    if (trimmedDatasetRoot) {
      formData.append('dataset_root', trimmedDatasetRoot);
    }

    formData.append(
      'ranges',
      JSON.stringify({
        iou: iouRange,
        confidence: confRange
      })
    );
    formData.append('split', split);
    formData.append('progress_id', newProgressId);

    try {
      const response = await fetch('http://localhost:8000/api/optimize/thresholds', {
        method: 'POST',
        body: formData
      });

      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const message =
          payload?.detail ||
          'Optimizasyon başarısız oldu. Model ve data dosyalarının erişilebilir olduğundan emin olun.';
        throw new Error(message);
      }

      setResult(payload);
      setReportId(payload.report_id || null);
      await fetchProgressSnapshot(newProgressId);
    } catch (err) {
      console.error('Threshold optimization failed:', err);
      setResult(null);
      setError(
        err instanceof Error
          ? err.message
          : 'Optimizasyon sırasında hata oluştu. Backend şu anda gerçek YOLO değerlendirmesi sunmuyor olabilir.'
      );
      const snapshotReceived = await fetchProgressSnapshot(newProgressId);
      if (!snapshotReceived) {
        setProgressInfo((prev) =>
          prev
            ? {
                ...prev,
                status: 'error'
              }
            : prev
        );
      }
    } finally {
      stopProgressPolling();
      stopElapsedTimer();
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

  const exportReport = async (format) => {
    if (!reportId) {
      setError('Henüz kaydedilmiş bir rapor yok.');
      return;
    }

    setIsExporting(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:8000/api/optimize/thresholds/reports/${reportId}/export?format=${format}`);
      if (!response.ok) {
        let message = 'Rapor dışa aktarılamadı.';
        try {
          const payload = await response.json();
          if (payload?.detail) {
            message = payload.detail;
          }
        } catch (err) {
          console.debug('Export error payload parse failed', err);
        }
        throw new Error(message);
      }

      const blob = await response.blob();
      const extension = format === 'pdf' ? 'pdf' : 'html';
      const filename = `threshold-report-${reportId.slice(0, 8)}.${extension}`;

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

  const loadSavedReports = async () => {
    setIsLoadingReports(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/optimize/thresholds/reports');
      if (!response.ok) {
        throw new Error('Kaydedilmiş raporlar yüklenemedi.');
      }

      const data = await response.json();
      setSavedReports(data.reports || []);
      setShowReportsList(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Raporlar yüklenemedi.';
      setError(message);
    } finally {
      setIsLoadingReports(false);
    }
  };

  const loadReport = async (id) => {
    setError(null);

    try {
      const response = await fetch(`http://localhost:8000/api/optimize/thresholds/reports/${id}`);
      if (!response.ok) {
        throw new Error('Rapor yüklenemedi.');
      }

      const data = await response.json();
      setResult(data);
      setReportId(id);
      setShowReportsList(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Rapor yüklenemedi.';
      setError(message);
    }
  };

  const useServerBest = () => {
    if (serverBestName) {
      setBestSource({ kind: 'server', filename: serverBestName });
    }
  };

  const useServerYaml = () => {
    if (serverYamlName) {
      setDataSource({ kind: 'server', filename: serverYamlName });
    }
  };

  const renderArtifactHint = (source, fallbackLabel, serverName, onUseServer) => {
    if (!source) {
      if (serverName) {
        return (
          <button type="button" className="link-button" onClick={onUseServer}>
            Sunucudaki {serverName} dosyasını kullan
          </button>
        );
      }
      return <span className="file-hint">{fallbackLabel}</span>;
    }

    if (source.kind === 'upload') {
      return (
        <div className="artifact-pill upload">
          <span>Yüklenecek: {source.file.name}</span>
          {serverName && (
            <button type="button" className="link-button" onClick={onUseServer}>
              Sunucu versiyonunu kullan
            </button>
          )}
        </div>
      );
    }

    return (
      <div className="artifact-pill server">
        <span>Sunucudan kullanılacak: {source.filename}</span>
      </div>
    );
  };

  const bestMetrics = result?.best;

  const totalCycles = Number(progressInfo?.total_cycles) > 0 ? progressInfo.total_cycles : estimatedTotalCycles;
  const completedCycles = Number(progressInfo?.completed_cycles) > 0 ? progressInfo.completed_cycles : 0;
  const rawProgressRatio = (() => {
    if (progressInfo && Number.isFinite(progressInfo.progress)) {
      return progressInfo.progress;
    }
    if (totalCycles) {
      return completedCycles / totalCycles;
    }
    return 0;
  })();
  const clampedProgressRatio = Math.max(0, Math.min(1, rawProgressRatio || 0));
  const progressPercent = Math.round(clampedProgressRatio * 100);
  const elapsedDisplaySeconds = progressInfo?.elapsed_seconds ?? elapsedSeconds;
  const remainingSeconds = progressInfo?.estimated_remaining_seconds;
  const currentStepLabel = progressInfo?.current_iou != null && progressInfo?.current_confidence != null
    ? `IoU ${Number(progressInfo.current_iou).toFixed(2)} · Conf ${Number(progressInfo.current_confidence).toFixed(2)}`
    : isRunning
      ? 'Hazırlanıyor'
      : '—';
  const progressStatus = progressInfo?.status || (isRunning ? 'running' : null);
  const progressStatusLabel = progressStatus === 'success'
    ? 'Optimizasyon tamamlandı'
    : progressStatus === 'error'
      ? 'Optimizasyon hata verdi'
      : 'Optimizasyon devam ediyor';
  const completedCycleLabel = totalCycles
    ? `${Math.min(completedCycles, totalCycles).toLocaleString('tr-TR')} / ${totalCycles.toLocaleString('tr-TR')} cycle`
    : completedCycles
      ? `${completedCycles.toLocaleString('tr-TR')} cycle`
      : 'Cycle bilgisi hazırlanıyor';
  const formattedProgressPercent = progressPercent.toLocaleString('tr-TR');

  return (
    <div className="optimizer-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h2 style={{ margin: 0 }}>🎛️ Threshold Optimizer</h2>
        <button
          type="button"
          className="btn-secondary"
          onClick={loadSavedReports}
          disabled={isLoadingReports}
          style={{ padding: '8px 16px', fontSize: '14px' }}
        >
          {isLoadingReports ? '⏳ Yükleniyor...' : '📂 Kaydedilmiş Raporlar'}
        </button>
      </div>
      <p className="optimizer-subtitle">
        IoU ve confidence eşiklerini grid search ile tarayın, daha önce yüklediğiniz best.pt dosyasını tekrar kullanarak
        production için en iyi kombinasyonu bulun.
      </p>

      {showReportsList && (
        <div className="saved-reports-modal" style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'white',
            borderRadius: '16px',
            padding: '24px',
            maxWidth: '800px',
            maxHeight: '80vh',
            overflow: 'auto',
            width: '90%'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ margin: 0 }}>📂 Kaydedilmiş Threshold Raporları</h3>
              <button
                type="button"
                onClick={() => setShowReportsList(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '24px',
                  cursor: 'pointer',
                  padding: '4px 8px'
                }}
              >
                ✕
              </button>
            </div>
            {savedReports.length === 0 ? (
              <p>Henüz kaydedilmiş rapor yok.</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {savedReports.map((report) => (
                  <div
                    key={report.report_id}
                    style={{
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      padding: '16px',
                      cursor: 'pointer',
                      transition: 'background 0.2s'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = '#f9fafb'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'white'}
                    onClick={() => loadReport(report.report_id)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <strong>{report.model_filename || 'Model'}</strong>
                      <span style={{ fontSize: '12px', color: '#6b7280' }}>
                        {new Date(report.timestamp).toLocaleString('tr-TR')}
                      </span>
                    </div>
                    <div style={{ fontSize: '14px', color: '#4b5563' }}>
                      <div>Split: {report.split} • Kombinasyon: {report.total_combinations}</div>
                      <div>
                        En İyi: IoU {report.best_iou?.toFixed(2)} • Conf {report.best_confidence?.toFixed(2)} • F1 {formatPercent(report.best_f1)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      <div className="optimizer-grid">
        <div className="optimizer-inputs">
          <div className="file-input">
            <label>best.pt:</label>
            <input
              type="file"
              accept=".pt"
              onChange={(e) => {
                const file = e.target.files?.[0] || null;
                setError(null);
                if (file) {
                  setBestSource({ kind: 'upload', file });
                } else if (serverBestName) {
                  setBestSource({ kind: 'server', filename: serverBestName });
                } else {
                  setBestSource(null);
                }
              }}
            />
            {renderArtifactHint(
              bestSource,
              'Sunucuda kayıtlı best.pt yoksa buradan yükleyin.',
              serverBestName,
              useServerBest
            )}
          </div>

          <div className="file-input">
            <label>data.yaml:</label>
            <input
              type="file"
              accept=".yaml,.yml"
              onChange={(e) => {
                const file = e.target.files?.[0] || null;
                setError(null);
                if (file) {
                  setDataSource({ kind: 'upload', file });
                } else if (serverYamlName) {
                  setDataSource({ kind: 'server', filename: serverYamlName });
                } else {
                  setDataSource(null);
                }
              }}
            />
            {renderArtifactHint(
              dataSource,
              'Veri kümesi config dosyası gereklidir.',
              serverYamlName,
              useServerYaml
            )}
          </div>

          <div className="dataset-root-input">
            <label>Veri seti kök klasörü</label>
            <input
              type="text"
              placeholder={dataSource ? "data.yaml dosyasının bulunduğu klasörü girin" : "Önce data.yaml dosyası seçin"}
              value={datasetRoot}
              onChange={(e) => setDatasetRoot(e.target.value)}
            />
            <span className="input-hint">
              {dataSource
                ? "📁 Seçtiğiniz data.yaml dosyasının bulunduğu tam klasör yolunu girin. Örn: C:\\Projeler\\ModelEgitimi veya /home/user/datasets/boxing"
                : "⚠️ Önce yukarıdan data.yaml dosyasını seçin, sonra o dosyanın bulunduğu klasörü buraya yazın."}
            </span>
          </div>

          <div className="split-select">
            <label>Değerlendirilecek split</label>
            <select value={split} onChange={(e) => setSplit(e.target.value)}>
              <option value="test">Test</option>
              <option value="val">Validation</option>
              <option value="train">Train</option>
            </select>
            <span className="split-hint">Grid search bu split üzerinde çalıştırılır.</span>
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
                onChange={(e) => updateIouRange('start', e.target.value)}
              />
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.01"
                value={iouRange.start}
                onChange={(e) => updateIouRange('start', e.target.value)}
              />
              <input
                type="range"
                min={Math.min(0.95, Math.max(0.01, iouRange.start + 0.01))}
                max="0.95"
                step="0.01"
                value={iouRange.end}
                onChange={(e) => updateIouRange('end', e.target.value)}
              />
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="0.95"
                value={iouRange.end}
                onChange={(e) => updateIouRange('end', e.target.value)}
              />
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="0.3"
                value={iouRange.step}
                onChange={(e) => updateIouRange('step', e.target.value)}
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
                onChange={(e) => updateConfRange('start', e.target.value)}
              />
              <input
                type="range"
                min="0.05"
                max="0.8"
                step="0.01"
                value={confRange.start}
                onChange={(e) => updateConfRange('start', e.target.value)}
              />
              <input
                type="range"
                min={Math.min(0.95, Math.max(0.01, confRange.start + 0.01))}
                max="0.95"
                step="0.01"
                value={confRange.end}
                onChange={(e) => updateConfRange('end', e.target.value)}
              />
              <input
                type="number"
                step="0.01"
                min="0.1"
                max="0.95"
                value={confRange.end}
                onChange={(e) => updateConfRange('end', e.target.value)}
              />
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="0.3"
                value={confRange.step}
                onChange={(e) => updateConfRange('step', e.target.value)}
              />
            </div>
          </div>

          <button className="btn-primary" type="button" onClick={handleOptimize} disabled={isRunning}>
            {isRunning ? '⏳ Optimizasyon...' : 'Optimizasyonu Başlat'}
          </button>

          {(isRunning || progressInfo) && (
            <div className={`optimizer-progress-card ${progressStatus || 'running'}`}>
              <div className="optimizer-progress-top">
                <span className="progress-status-badge">{progressStatusLabel}</span>
                <span className="progress-cycle-count">{completedCycleLabel}</span>
              </div>
              <div className="optimizer-progress-bar">
                <div style={{ width: `${progressPercent}%` }} />
              </div>
              <div className="optimizer-progress-meta">
                <div className="progress-meta-item">
                  <span>Geçen Süre</span>
                  <strong>{formatDuration(elapsedDisplaySeconds)}</strong>
                </div>
                <div className="progress-meta-item">
                  <span>Tahmini Kalan</span>
                  <strong>{Number.isFinite(remainingSeconds) ? formatDuration(remainingSeconds) : '—'}</strong>
                </div>
                <div className="progress-meta-item">
                  <span>Şu Anki Kombinasyon</span>
                  <strong>{currentStepLabel}</strong>
                </div>
                <div className="progress-meta-item">
                  <span>İlerleme</span>
                  <strong>%{formattedProgressPercent}</strong>
                </div>
              </div>
              {progressInfo?.detail && (
                <div className="optimizer-progress-detail">{progressInfo.detail}</div>
              )}
            </div>
          )}

          {progressError && <div className="optimizer-progress-warning">{progressError}</div>}

          {error && <div className="optimizer-error">{error}</div>}
        </div>

        {result && bestMetrics && (
          <div className="optimizer-results">
            <div className="optimizer-summary">
              <div className="summary-header">
                <span className="summary-badge">En iyi kombinasyon</span>
                <h3>
                  Conf {bestMetrics.confidence.toFixed(2)} · IoU {bestMetrics.iou.toFixed(2)}
                </h3>
                <p>
                  {split.toUpperCase()} split üzerinde {combinationsTested || 'birden çok'} kombinasyon tarandı.
                </p>
              </div>
              <div className="summary-metric-grid">
                <div className="summary-metric">
                  <span>Recall</span>
                  <strong>{formatPercent(bestMetrics.recall, 2)}</strong>
                </div>
                <div className="summary-metric">
                  <span>Precision</span>
                  <strong>{formatPercent(bestMetrics.precision, 2)}</strong>
                </div>
                <div className="summary-metric">
                  <span>F1 Skoru</span>
                  <strong>{formatPercent(bestMetrics.f1, 2)}</strong>
                </div>
                <div className="summary-metric">
                  <span>mAP@0.5</span>
                  <strong>{formatPercent(bestMetrics.map50, 2)}</strong>
                </div>
                <div className="summary-metric">
                  <span>mAP@0.75</span>
                  <strong>{formatPercent(bestMetrics.map75, 2)}</strong>
                </div>
                <div className="summary-metric">
                  <span>mAP@0.5:0.95</span>
                  <strong>{formatPercent(bestMetrics.map5095, 2)}</strong>
                </div>
              </div>
              <div className="summary-actions">
                <button className="btn-secondary" type="button" onClick={downloadConfig}>
                  production_config.yaml indir
                </button>
                {reportId && (
                  <>
                    <button className="btn-secondary" type="button" onClick={() => exportReport('pdf')} disabled={isExporting}>
                      {isExporting ? '⏳ Export ediliyor...' : 'PDF olarak indir'}
                    </button>
                    <button className="btn-secondary" type="button" onClick={() => exportReport('html')} disabled={isExporting}>
                      HTML olarak indir
                    </button>
                  </>
                )}
                <span className="summary-note">
                  {reportId
                    ? `Rapor kaydedildi (ID: ${reportId.slice(0, 8)}...) • Konfigürasyon dosyası production için hazır.`
                    : 'Konfigürasyon dosyası production için hazır.'}
                </span>
              </div>
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
    trainingCode: null,
    projectType: '',
    datasetTotals: {
      train: '',
      val: '',
      test: '',
      total: ''
    },
    splitRatios: {
      train: '',
      val: '',
      test: ''
    },
    folderDistributions: {
      train: '',
      val: '',
      test: ''
    }
  });
  const [formErrors, setFormErrors] = useState([]);

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

  const projectTypeOptions = [
    { value: 'classification', label: 'Sınıflandırma' },
    { value: 'object_detection', label: 'Nesne Tespiti' },
    { value: 'instance_segmentation', label: 'Instance Segmentasyon' },
    { value: 'semantic_segmentation', label: 'Semantik Segmentasyon' },
    { value: 'pose_estimation', label: 'Pose Tahmini' },
    { value: 'other', label: 'Diğer' }
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

  const updateNestedProjectInfo = (section, key, value) => {
    setProjectInfo((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
  };

  const parseIntegerField = (value, fieldLabel, errors) => {
    const trimmed = (value ?? '').toString().trim();
    if (!trimmed) return null;
    const numeric = Number(trimmed);
    if (!Number.isFinite(numeric) || numeric < 0) {
      errors.push(`${fieldLabel} için 0 veya daha büyük bir değer giriniz.`);
      return null;
    }
    if (!Number.isInteger(numeric)) {
      errors.push(`${fieldLabel} tam sayı olmalıdır.`);
      return null;
    }
    return numeric;
  };

  const parsePercentageField = (value, fieldLabel, errors) => {
    const trimmed = (value ?? '').toString().trim();
    if (!trimmed) return null;
    const numeric = Number(trimmed);
    if (!Number.isFinite(numeric)) {
      errors.push(`${fieldLabel} için geçerli bir yüzde değeri giriniz.`);
      return null;
    }
    if (numeric < 0 || numeric > 100) {
      errors.push(`${fieldLabel} 0 ile 100 arasında olmalıdır.`);
      return null;
    }
    return Number(numeric.toFixed(2));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    console.log('=== DOSYA YÜKLEME BAŞLADI ===');
    console.log('Timestamp:', new Date().toISOString());

    const errors = [];

    const parsedTotals = {};
    Object.entries(projectInfo.datasetTotals || {}).forEach(([key, value]) => {
      const labelMap = {
        train: 'Train görsel sayısı',
        val: 'Val görsel sayısı',
        test: 'Test görsel sayısı',
        total: 'Toplam görsel sayısı'
      };
      const parsed = parseIntegerField(value, labelMap[key] || key, errors);
      if (parsed !== null) {
        parsedTotals[key] = parsed;
      }
    });

    const parsedSplitRatios = {};
    let splitSum = 0;
    let splitCount = 0;
    Object.entries(projectInfo.splitRatios || {}).forEach(([key, value]) => {
      const labelMap = {
        train: 'Train yüzdesi',
        val: 'Val yüzdesi',
        test: 'Test yüzdesi'
      };
      const parsed = parsePercentageField(value, labelMap[key] || key, errors);
      if (parsed !== null) {
        parsedSplitRatios[key] = parsed;
        splitSum += parsed;
        splitCount += 1;
      }
    });

    if (splitCount > 0 && Math.abs(splitSum - 100) > 0.5) {
      errors.push('Train/Val/Test yüzdelerinin toplamı %100 olmalıdır.');
    }

    const folderDistributionPayload = {};
    Object.entries(projectInfo.folderDistributions || {}).forEach(([key, value]) => {
      const trimmed = (value ?? '').toString().trim();
      if (trimmed) {
        folderDistributionPayload[key] = trimmed;
      }
    });

    if (errors.length > 0) {
      console.warn('Form doğrulaması başarısız:', errors);
      setFormErrors(errors);
      return;
    }

    setFormErrors([]);

    // Log file details
    const fileDetails = {
      csv: files.csv ? { name: files.csv.name, size: files.csv.size, type: files.csv.type } : null,
      yaml: files.yaml ? { name: files.yaml.name, size: files.yaml.size, type: files.yaml.type } : null,
      best: files.best ? { name: files.best.name, size: files.best.size, type: files.best.type } : null,
      graphs: files.graphs.map(g => ({ name: g.name, size: g.size, type: g.type })),
      trainingCode: projectInfo.trainingCode ? { name: projectInfo.trainingCode.name, size: projectInfo.trainingCode.size } : null
    };
    console.log('Yüklenecek dosyalar:', fileDetails);

    // Calculate total upload size
    const totalSize = [
      files.csv?.size || 0,
      files.yaml?.size || 0,
      files.best?.size || 0,
      ...files.graphs.map(g => g.size),
      projectInfo.trainingCode?.size || 0
    ].reduce((sum, size) => sum + size, 0);
    console.log('Toplam dosya boyutu:', (totalSize / 1024 / 1024).toFixed(2), 'MB');
    console.log('Doğrulanmış veri seti toplamları:', parsedTotals);
    console.log('Doğrulanmış split yüzdeleri:', parsedSplitRatios);
    console.log('Klasör dağılımı notları:', folderDistributionPayload);

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
    if (projectInfo.projectType) {
      formData.append('project_type', projectInfo.projectType);
    }
    if (Object.keys(parsedTotals).length > 0) {
      formData.append('dataset_totals', JSON.stringify(parsedTotals));
    }
    if (Object.keys(parsedSplitRatios).length > 0) {
      formData.append('split_ratios', JSON.stringify(parsedSplitRatios));
    }
    if (Object.keys(folderDistributionPayload).length > 0) {
      formData.append('folder_distributions', JSON.stringify(folderDistributionPayload));
    }

    const uploadStartTime = Date.now();
    console.log('FormData hazırlandı, sunucuya gönderiliyor...');

    try {
      console.log('Fetch isteği başlatılıyor: POST http://localhost:8000/api/upload/results');

      const response = await fetch('http://localhost:8000/api/upload/results', {
        method: 'POST',
        body: formData
      });

      const uploadDuration = ((Date.now() - uploadStartTime) / 1000).toFixed(2);
      console.log(`Sunucu yanıtı alındı (${uploadDuration}s)`);
      console.log('Response status:', response.status, response.statusText);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        console.error('Sunucu hata yanıtı döndü:', response.status);
        const errorText = await response.text();
        console.error('Hata detayı:', errorText);
        throw new Error(`Upload başarısız: ${response.status} ${response.statusText}`);
      }

      console.log('JSON yanıtı parse ediliyor...');
      const data = await response.json();
      console.log('Yanıt başarıyla alındı:', {
        status: data.status,
        hasMetrics: !!data.metrics,
        hasAnalysis: !!data.analysis,
        hasConfig: !!data.config,
        reportId: data.report_id
      });
      console.log('=== DOSYA YÜKLEME TAMAMLANDI ===\n');

      onUpload(data);
    } catch (error) {
      const uploadDuration = ((Date.now() - uploadStartTime) / 1000).toFixed(2);
      console.error('=== DOSYA YÜKLEME HATASI ===');
      console.error('Süre:', uploadDuration, 'saniye');
      console.error('Hata türü:', error.name);
      console.error('Hata mesajı:', error.message);
      console.error('Stack trace:', error.stack);

      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        console.error('Network hatası: Sunucuya bağlanılamıyor. Backend çalışıyor mu?');
      } else if (error.name === 'AbortError') {
        console.error('İstek iptal edildi veya timeout oluştu');
      }

      console.error('=========================\n');
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

            <div className="input-group">
              <label htmlFor="projectType">Proje Türü</label>
              <select
                id="projectType"
                value={projectInfo.projectType}
                onChange={(e) => updateProjectInfo('projectType', e.target.value)}
                disabled={isLoading}
              >
                <option value="">Seçiniz</option>
                {projectTypeOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <span className="form-hint">Modelin çözdüğü bilgisayar görüşü görevini seçin.</span>
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

            <div className="metadata-section">
              <div className="metadata-header">
                <h4>Veri Seti Toplamları</h4>
                <span className="form-hint">Her bölüme ait görsel sayısını giriniz.</span>
              </div>
              <div className="nested-input-grid">
                {['train', 'val', 'test', 'total'].map((key) => {
                  const labelMap = {
                    train: 'Train',
                    val: 'Val',
                    test: 'Test',
                    total: 'Toplam'
                  };
                  return (
                    <div key={key} className="nested-input-group">
                      <label htmlFor={`dataset-total-${key}`}>{labelMap[key]}</label>
                      <input
                        id={`dataset-total-${key}`}
                        type="number"
                        min="0"
                        step="1"
                        inputMode="numeric"
                        placeholder="Örn. 1200"
                        value={projectInfo.datasetTotals?.[key] ?? ''}
                        onChange={(e) => updateNestedProjectInfo('datasetTotals', key, e.target.value)}
                        disabled={isLoading}
                      />
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="metadata-section">
              <div className="metadata-header">
                <h4>Veri Seti Bölünmeleri (%)</h4>
                <span className="form-hint">Train/Val/Test yüzdeleri toplamı 100 olmalıdır.</span>
              </div>
              <div className="nested-input-grid">
                {['train', 'val', 'test'].map((key) => {
                  const labelMap = {
                    train: 'Train %',
                    val: 'Val %',
                    test: 'Test %'
                  };
                  return (
                    <div key={key} className="nested-input-group">
                      <label htmlFor={`split-ratio-${key}`}>{labelMap[key]}</label>
                      <input
                        id={`split-ratio-${key}`}
                        type="number"
                        min="0"
                        max="100"
                        step="0.1"
                        inputMode="decimal"
                        placeholder="Örn. 70"
                        value={projectInfo.splitRatios?.[key] ?? ''}
                        onChange={(e) => updateNestedProjectInfo('splitRatios', key, e.target.value)}
                        disabled={isLoading}
                      />
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="metadata-section">
              <div className="metadata-header">
                <h4>Klasör İçi Oranlar</h4>
                <span className="form-hint">Her klasördeki sınıf dağılımını yüzde veya oran olarak belirtin (örn. Potluk %60 / Temiz %40).</span>
              </div>
              <div className="nested-input-grid">
                {['train', 'val', 'test'].map((key) => {
                  const labelMap = {
                    train: 'Train klasörü',
                    val: 'Val klasörü',
                    test: 'Test klasörü'
                  };
                  return (
                    <div key={key} className="nested-input-group">
                      <label htmlFor={`folder-dist-${key}`}>{labelMap[key]}</label>
                      <input
                        id={`folder-dist-${key}`}
                        type="text"
                        placeholder="Örn. Potluk %60 / Temiz %40"
                        value={projectInfo.folderDistributions?.[key] ?? ''}
                        onChange={(e) => updateNestedProjectInfo('folderDistributions', key, e.target.value)}
                        disabled={isLoading}
                      />
                    </div>
                  );
                })}
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

        {formErrors.length > 0 && (
          <div className="form-error-list">
            <h4>Form Hataları</h4>
            <ul>
              {formErrors.map((error, idx) => (
                <li key={idx}>{error}</li>
              ))}
            </ul>
          </div>
        )}

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
    console.log('handleUpload çağrıldı:', uploadResponse ? Object.keys(uploadResponse) : 'null');

    if (!uploadResponse) {
      console.log('Upload response boş, state sıfırlanıyor');
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
      console.log('Loading durumu başlatılıyor (uploading status)');
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
      console.error('Upload response hata içeriyor:', uploadResponse.error);
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

    console.log('Upload response yapısı:', {
      hasMetrics: !!responseMetrics,
      hasAnalysis: !!responseAnalysis,
      hasConfig: !!responseConfig,
      hasHistory: !!responseHistory,
      reportId: uploadResponse.report_id,
      hasFiles: !!uploadResponse.files
    });

    setMetrics(responseMetrics || null);
    setHistory(responseHistory || null);
    setConfig(responseConfig || null);
    setProject(uploadResponse.project || responseConfig?.project_context || null);
    setAnalysis(null);
    setReportId(uploadResponse.report_id || null);
    setQaHistory(Array.isArray(uploadResponse.qa_history) ? uploadResponse.qa_history : []);
    setQaError(null);
    setQaLoading(false);

    console.log('State güncellendi, parsing durumuna geçiliyor');
    setLoading(true);
    setLoadingStatus('parsing');

    if (uploadResponse.files) {
      console.log('Artifact dosyaları kaydediliyor:', uploadResponse.files);
      setArtifacts({ server: uploadResponse.files });
    }

    if (responseAnalysis) {
      console.log('Analiz zaten sunucu tarafında yapılmış, tamamlandı olarak işaretleniyor');
      setAnalysis(responseAnalysis);
      setLoading(false);
      setLoadingStatus('complete');
      setTimeout(() => setLoadingStatus(null), 1500);
      return;
    }

    if (!responseMetrics) {
      console.warn('Response metrics bulunamadı, işlem sonlandırılıyor');
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

    console.log('Analyzing durumuna geçiliyor, metrics payload hazırlanıyor');
    setLoadingStatus('analyzing');

    try {
      console.log('Metrics analizi için istek gönderiliyor:', metricsPayload);
      const analysisStartTime = Date.now();

      const response = await fetch('http://localhost:8000/api/analyze/metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metricsPayload)
      });

      const analysisDuration = ((Date.now() - analysisStartTime) / 1000).toFixed(2);
      console.log(`Analiz yanıtı alındı (${analysisDuration}s), status:`, response.status);

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        const message = responseData?.detail || 'Analiz isteği başarısız oldu.';
        console.error('Analiz isteği başarısız:', message);
        throw new Error(Array.isArray(message) ? message[0]?.msg || 'Analiz isteği başarısız oldu.' : message);
      }

      console.log('Analiz başarıyla tamamlandı');
      setAnalysis(responseData);
      setLoadingStatus('complete');
      setTimeout(() => {
        console.log('Loading durumu temizleniyor');
        setLoading(false);
        setLoadingStatus(null);
      }, 1500);
    } catch (err) {
      console.error('=== ANALİZ HATASI ===');
      console.error('Hata türü:', err.name);
      console.error('Hata mesajı:', err.message);
      console.error('Stack trace:', err.stack);
      console.error('==================\n');

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
        <p>Murat Turan AI Projects © 2025</p>
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
            best: artifacts?.server?.best_model || artifacts?.client?.best || null,
            yaml: artifacts?.server?.yaml || artifacts?.client?.yaml || null,
            serverBest: artifacts?.server?.best_model || null,
            serverYaml: artifacts?.server?.yaml || null
          }}
        />
      </main>

      <footer className="app-footer">
        <p>Murat Turan AI Projects © 2025</p>
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
        <div className={`app-container ${activePage === 'threshold' ? 'threshold-container' : ''}`}>
          {activePage === 'dashboard' ? renderDashboard() : renderThresholdPage()}
        </div>
      </div>
    </div>
  );
}

export default App;
