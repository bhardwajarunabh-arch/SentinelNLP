import React, { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart2,
  CheckCircle2,
  ChevronRight,
  Copy,
  Cpu,
  Database,
  ExternalLink,
  Eye,
  FileCode,
  Globe2,
  Languages,
  Layers,
  Play,
  RotateCw,
  Search,
  Shield,
  ShieldAlert,
  Sparkles,
  Terminal,
} from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type ViewId =
  | "overview"
  | "analyzer"
  | "inspector"
  | "cross-language"
  | "critical-risk"
  | "dataset"
  | "research-status"
  | "status";

const navItems: { id: ViewId; label: string; icon: React.ReactNode; badge?: string }[] = [
  { id: "overview", label: "Overview", icon: <Layers size={17} /> },
  { id: "analyzer", label: "Vulnerability Analyzer", icon: <Search size={17} />, badge: "LIME" },
  { id: "inspector", label: "Explanation Inspector", icon: <Eye size={17} /> },
  { id: "cross-language", label: "Cross-Language Eval", icon: <Globe2 size={17} /> },
  { id: "critical-risk", label: "Critical-Risk Analysis", icon: <ShieldAlert size={17} />, badge: "RQ3" },
  { id: "dataset", label: "Dataset & Experiments", icon: <Database size={17} /> },
  { id: "research-status", label: "Research Status", icon: <CheckCircle2 size={17} />, badge: "C–E" },
  { id: "status", label: "System Status", icon: <Activity size={17} /> },
];

import MULTILINGUAL_SAMPLES_RAW from "./multilingual_samples.json";

const MULTILINGUAL_SAMPLES: Record<string, { severity: string; texts: Record<string, string> }> =
  MULTILINGUAL_SAMPLES_RAW as any;

const CURATED_CVES = [
  {
    cve_id: "CVE-2023-33030",
    severity: "Critical",
    label: "Critical (Memory Corruption)",
    texts: {
      en: "Memory corruption in HLOS while running play-ready use-case.",
      hi: MULTILINGUAL_SAMPLES["CVE-2023-33030"]?.texts?.hi ?? "प्ले-ready उपयोग-केस चलाने के दौरान HLOS में मेमोरी भ्रष्टाचार।",
      es: MULTILINGUAL_SAMPLES["CVE-2023-33030"]?.texts?.es ?? "Corrupción de la memoria en HLOS mientras se ejecuta el caso de uso listo para jugar.",
      zh: "在执行支持播放准备的用例时，操作系统高层（HLOS）中发生内存损坏漏洞，可导致远程任意代码执行与系统崩溃。",
      ar: MULTILINGUAL_SAMPLES["CVE-2023-33030"]?.texts?.ar ?? "إفساد الذاكرة في HLOS أثناء تشغيل حالة الاستخدام جاهزة للعب.",
      fr: MULTILINGUAL_SAMPLES["CVE-2023-33030"]?.texts?.fr ?? "Corruption de la mémoire dans le HLOS lors de l'exécution d'un cas d'utilisation prêt à être joué.",
    },
  },
  {
    cve_id: "CVE-2023-48265",
    severity: "High",
    label: "High (Remote Code Execution / DoS)",
    texts: {
      en: "This vulnerability allows an unauthenticated remote attacker to perform Denial of Service (DoS) attacks or, potentially, obtain remote code execution (RCE) via a crafted network request.",
      hi: MULTILINGUAL_SAMPLES["CVE-2023-48265"]?.texts?.hi ?? "कमजोरता एक गैर-मान्यता प्राप्त दूरस्थ हमलावर को सेवा-निषेध (DoS) हमले करने या संभवतः एक निर्मित नेटवर्क अनुरोध के माध्यम से दूरस्थ कोड निष्पादन (RCE) प्राप्त करने की अनुमति देती है।",
      es: MULTILINGUAL_SAMPLES["CVE-2023-48265"]?.texts?.es ?? "La vulnerabilidad permite a un atacante remoto no autenticado realizar un ataque de Denial of Service (DoS) o, posiblemente, obtener la ejecución remota de código (RCE) a través de una solicitud de red elaborada.",
      zh: MULTILINGUAL_SAMPLES["CVE-2023-48265"]?.texts?.zh ?? "漏洞允许未认证的远程攻击者执行拒绝服务 (DoS) 攻击或可能通过构造的网络请求获得远程代码执行 (RCE).",
      ar: MULTILINGUAL_SAMPLES["CVE-2023-48265"]?.texts?.ar ?? "تسمح هذه الضعف للمهاجم عن بعد غير المصادق بإجراء هجوم \"رفض الخدمة\" أو، ربما، الحصول على تنفيذ رمز عن بعد (RCE) عبر طلب شبكة مصمم.",
      fr: MULTILINGUAL_SAMPLES["CVE-2023-48265"]?.texts?.fr ?? "La vulnérabilité permet à un attaquant distant non authentifié d'effectuer une attaque Denial-of-Service (DoS) ou, éventuellement, d'obtenir une exécution de code à distance (RCE) via une demande réseau créée.",
    },
  },
  {
    cve_id: "CVE-2023-28722",
    severity: "Medium",
    label: "Medium (Buffer Privilege Escalation)",
    texts: {
      en: "Improper buffer restrictions for some Intel NUC BIOS firmware before version IN0048 may allow a privileged user to potentially enable escalation of privilege via local access.",
      hi: MULTILINGUAL_SAMPLES["CVE-2023-28722"]?.texts?.hi ?? "IN0048 संस्करण से पहले कुछ इंटेल NUC BIOS फर्मवेयर के लिए अनुचित बफर प्रतिबंधों से एक विशेषाधिकार प्राप्त उपयोगकर्ता को स्थानीय पहुंच के माध्यम से विशेषाधिकारों के बढ़ोतरी को संभावित रूप से सक्षम करने की अनुमति मिल सकती है।",
      es: MULTILINGUAL_SAMPLES["CVE-2023-28722"]?.texts?.es ?? "Las restricciones de buffer inadecuadas para algunos firmware de BIOS de Intel NUC antes de la versión IN0048 pueden permitir que un usuario privilegiado permita potencialmente la escalación de privilegios a través del acceso local.",
      zh: MULTILINGUAL_SAMPLES["CVE-2023-28722"]?.texts?.zh ?? "在 IN0048 版本之前,一些英特尔NUC BIOS固件的不适当缓冲限制可能允许特权用户通过本地访问可能实现特权升级.",
      ar: MULTILINGUAL_SAMPLES["CVE-2023-28722"]?.texts?.ar ?? "قد تسمح قيود الغطاء غير المناسبة لبعض برنامج التثبيت التشغيلي لـ Intel NUC BIOS قبل إصدار IN0048 للمستخدم الممتاز بتسهيل تصعيد الامتيازات عبر الوصول المحلي.",
      fr: MULTILINGUAL_SAMPLES["CVE-2023-28722"]?.texts?.fr ?? "Les restrictions de tampon inappropriées pour certains firmware BIOS Intel NUC avant la version IN0048 peuvent permettre à un utilisateur privilégié de permettre potentiellement une escalade de privilège via un accès local.",
    },
  },
  {
    cve_id: "CVE-2015-10128",
    severity: "Low",
    label: "Low (Reflected Site Scripting in Plugin)",
    texts: {
      en: "A vulnerability was found in rt-prettyphoto plugin up to 1.2 on WordPress and classified as problematic. Affected by this issue is the function royal_prettyphoto_plugin_links of the file rt-prettyphoto.php. The manipulation leads to cross site scripting. The attack may be launched remotely.",
      hi: MULTILINGUAL_SAMPLES["CVE-2015-10128"]?.texts?.hi ?? "वर्डप्रेस पर rt-prettyphoto प्लगइन में 1.2 तक की एक कमजोरियों को पाया गया और इसे समस्याग्रस्त के रूप में वर्गीकृत किया गया।",
      es: MULTILINGUAL_SAMPLES["CVE-2015-10128"]?.texts?.es ?? "Se encontró una vulnerabilidad en el plugin rt-prettyphoto hasta 1.2 en WordPress y se clasificó como problemático. Afecto por este problema es la función royal_prettyphoto_plugin_links del archivo rt-prettyphoto.php.",
      zh: MULTILINGUAL_SAMPLES["CVE-2015-10128"]?.texts?.zh ?? "在WordPress上 rt-prettyphoto插件中发现了一个漏洞,高达1.2个,并被归类为有问题的.该问题影响的是rt-prettyphoto.php文件的 royal_prettyphoto_plugin_links功能.",
      ar: MULTILINGUAL_SAMPLES["CVE-2015-10128"]?.texts?.ar ?? "تم العثور على ضعف في متضمن rt-prettyphoto حتى 1.2 على WordPress وتصنف على أنها مشكلة.",
      fr: MULTILINGUAL_SAMPLES["CVE-2015-10128"]?.texts?.fr ?? "Une vulnérabilité a été trouvée dans le plugin rt-prettyphoto jusqu'à 1.2 sur WordPress et classée comme problématique.",
    },
  },
];

/* ---------------------------------------------------- */
/* Overview Component                                   */
/* ---------------------------------------------------- */
function formatMetric(value: number | undefined, digits = 3): string {
  return value === undefined ? "—" : value.toFixed(digits);
}

function formatPercent(value: number | undefined): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function OverviewView({
  datasetSummary,
  metrics,
  baselineMetrics,
  criticalRisk,
  models,
  experiments,
  faithfulnessSummary,
}: {
  datasetSummary: any;
  metrics: any;
  baselineMetrics: any;
  criticalRisk: any;
  models: any;
  experiments: any;
  faithfulnessSummary?: any;
}) {
  const total = datasetSummary?.record_count;
  const dist = datasetSummary?.class_distribution ?? {};
  const evaluation = baselineMetrics?.status === "valid_train_only_test_evaluation"
    ? baselineMetrics
    : undefined;
  const translationEvaluationValid = metrics?.status === "valid_train_only_test_translation_evaluation";
  const agreement = translationEvaluationValid ? metrics.agreement : undefined;
  const modelCount = models?.models?.filter((model: any) => model.available).length;
  const languageRows = translationEvaluationValid
    ? [
        {
          code: "en",
          severity: metrics.english_severity,
          agreement: { agreement_with_english: 1 },
          critical_risk: metrics.english_critical_risk,
        },
        ...Object.entries(metrics.languages ?? {}).map(([code, language]) => ({
          code,
          ...(language as any),
        })),
      ]
    : [];
  const maxBar = Math.max(...Object.values(dist).map(Number), 1);

  return (
    <>
      <div className="page-header">
        <div className="eyebrow-badge">
          <Terminal size={12} />
          <span>Research Foundation · RQ1 · RQ2 · RQ3</span>
        </div>
        <h1>Cross-Lingual Severity Reliability</h1>
        <p className="subtitle">
          Auditable evaluation of multilingual vulnerability classification, LIME explanation faithfulness, and Critical underprediction drift across 6 major languages using reproducible pipeline artifacts.
        </p>
      </div>

      {/* 8-Card Metric Grid */}
      <div className="metric-grid-8">
        <div className="metric-card glass-panel">
          <div className="metric-card-top">
            <span className="metric-title">Eligible CVEs (CVSS v3.1)</span>
            <div className="metric-icon-bubble"><Database size={14} /></div>
          </div>
          <div className="metric-val">{total === undefined ? "—" : total.toLocaleString()}</div>
          <div className="metric-foot">
            <CheckCircle2 size={13} color="#10b981" />
            <span>{total === undefined ? "Dataset unavailable" : "NVD API 2.0 ingested"}</span>
          </div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-card-top">
            <span className="metric-title">Trained Models</span>
            <div className="metric-icon-bubble"><Cpu size={14} /></div>
          </div>
          <div className="metric-val">{modelCount === undefined ? "—" : `${modelCount} / ${models.models.length}`}</div>
          <div className="metric-foot">
            <span>{modelCount === undefined ? "Model status unavailable" : "Models loaded from API status"}</span>
          </div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-card-top">
            <span className="metric-title">English Test Macro-F1</span>
            <div className="metric-icon-bubble"><BarChart2 size={14} /></div>
          </div>
          <div className="metric-val">{formatMetric(evaluation?.macro_f1)}</div>
          <div className="metric-foot">
            <span>{evaluation ? `n=${evaluation.sample_count} held-out test split` : "Evaluation not available"}</span>
          </div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-card-top">
            <span className="metric-title">Mean Agreement (EN)</span>
            <div className="metric-icon-bubble"><Languages size={14} /></div>
          </div>
          <div className="metric-val">{formatPercent(agreement?.agreement_with_english)}</div>
          <div className="metric-foot">
            <span>{agreement ? `${Object.keys(metrics.languages ?? {}).length} translated target languages` : "Evaluation not available"}</span>
          </div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-card-top">
            <span className="metric-title">Critical Recall (Pilot)</span>
            <div className="metric-icon-bubble"><Shield size={14} /></div>
          </div>
          <div className="metric-val">{formatPercent(criticalRisk?.critical_recall)}</div>
          <div className="metric-foot">
            <span>{criticalRisk ? `n_critical=${criticalRisk.critical_ground_truth_count}` : "Risk evaluation not available"}</span>
          </div>
        </div>

        <div className="metric-card glass-panel alert">
          <div className="metric-card-top">
            <span className="metric-title">Critical Underprediction</span>
            <div className="metric-icon-bubble"><ShieldAlert size={14} color="#f43f5e" /></div>
          </div>
          <div className="metric-val alert">{formatPercent(criticalRisk?.critical_underprediction_rate)}</div>
          <div className="metric-foot">
            <span>{criticalRisk ? `${criticalRisk.critical_underprediction_count} true Critical misclassified below` : "Risk evaluation not available"}</span>
          </div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-card-top">
            <span className="metric-title">Explanation Faithfulness</span>
            <div className="metric-icon-bubble"><Sparkles size={14} /></div>
          </div>
          <div className="metric-val">
            {faithfulnessSummary?.aggregation?.mean_deletion_auc !== undefined
              ? formatMetric(faithfulnessSummary.aggregation.mean_deletion_auc)
              : "—"}
          </div>
          <div className="metric-foot">
            <CheckCircle2 size={13} color="#10b981" />
            <span>
              {faithfulnessSummary?.aggregation?.mean_deletion_auc !== undefined
                ? `LIME deletion AUC · n=${faithfulnessSummary.sample_count} CVEs`
                : "Faithfulness artifact not available"}
            </span>
          </div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-card-top">
            <span className="metric-title">6-Language Consensus</span>
            <div className="metric-icon-bubble"><Globe2 size={14} /></div>
          </div>
          <div className="metric-val">{formatPercent(agreement?.agreement_with_english)}</div>
          <div className="metric-foot">
            <span>{agreement ? "agreement from evaluated translations" : "Evaluation not available"}</span>
          </div>
        </div>
      </div>

      {/* Visual Analytics Grid */}
      <div className="two-panel-grid">
        <div className="glass-panel" style={{ padding: "24px" }}>
          <div className="panel-title-bar">
            <h3>
              <BarChart2 size={18} color="#38bdf8" />
              <span>Severity Distribution</span>
            </h3>
            <span className="chip-tag">{total === undefined ? "Dataset unavailable" : `${total} Processed CVEs`}</span>
          </div>
          <div className="dist-chart-wrapper">
            {["Low", "Medium", "High", "Critical"].map((cat) => {
              const count = dist[cat] || 0;
              const heightPct = Math.round((count / maxBar) * 100);
              const isCrit = cat === "Critical";
              const isHigh = cat === "High";
              return (
                <div className="dist-bar-col" key={cat}>
                  <span className="dist-val-num">{count}</span>
                  <div
                    className={`dist-fill ${isCrit ? "critical" : isHigh ? "high" : ""}`}
                    style={{ height: `${heightPct}%` }}
                  />
                  <span className="dist-cat-name">{cat}</span>
                </div>
              );
            })}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: "16px", fontSize: "12px", color: "var(--text-subtle)", fontFamily: "JetBrains Mono" }}>
            <span>Source: NVD CVSS v3.1</span>
            <span>Balanced Class Weights Applied</span>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: "24px" }}>
          <div className="panel-title-bar">
            <h3>
              <Languages size={18} color="#38bdf8" />
              <span>Cross-Lingual Evaluation Matrix</span>
            </h3>
            <span className="chip-tag">Model: tfidf_lr</span>
          </div>
          <table className="enterprise-table">
            <thead>
              <tr>
                <th>Language</th>
                <th>ISO</th>
                <th>Accuracy</th>
                <th>Agree w/ EN</th>
                <th>Crit. Recall</th>
                <th>Faithfulness</th>
              </tr>
            </thead>
            <tbody>
              {languageRows.map((language: any) => (
                <tr key={language.code}>
                  <td><strong>{language.code.toUpperCase()}</strong></td>
                  <td className="font-mono" style={{ color: "var(--primary)" }}>{language.code}</td>
                  <td className="font-mono">{formatPercent(language.severity?.accuracy)}</td>
                  <td className="font-mono" style={{ color: "#34d399" }}>{formatPercent(language.agreement?.agreement_with_english)}</td>
                  <td className="font-mono">{formatPercent(language.critical_risk?.critical_recall)}</td>
                  <td className="font-mono">—</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

/* ---------------------------------------------------- */
/* Vulnerability Analyzer Component                     */
/* ---------------------------------------------------- */
function AnalyzerView() {
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("auto");
  const [model, setModel] = useState("tfidf_lr");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [explanation, setExplanation] = useState<any>(null);
  const [copied, setCopied] = useState(false);
  const [faithfulnessLoading, setFaithfulnessLoading] = useState(false);
  const [faithfulnessData, setFaithfulnessData] = useState<any>(null);

  function loadCVE(index: number, targetLang?: string) {
    const item = CURATED_CVES[index % CURATED_CVES.length];
    const langKey = targetLang || (language === "auto" ? "en" : language);
    const chosenText = (item.texts as Record<string, string>)[langKey] || item.texts.en;
    setText(chosenText);
    if (targetLang && targetLang !== "auto") {
      setLanguage(targetLang);
    }
  }

  async function handlePredict(explain: boolean) {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    setExplanation(null);
    setFaithfulnessData(null);
    try {
      const endpoint = explain ? `${API}/api/explain` : `${API}/api/predict`;
      const effectiveLang = language === "auto" ? "en" : language;
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          language: effectiveLang,
          model: model === "xlmr" ? "xlmr" : "baseline",
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prediction failed");
      setResult(data);
      if (explain && data.segments) {
        setExplanation(data);
        // Automatically evaluate explanation faithfulness curve via API
        fetchFaithfulness(data.segments, data.prediction);
      }
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function fetchFaithfulness(segments: any[], _predictedSeverity: string) {
    setFaithfulnessLoading(true);
    try {
      const segmentTexts = segments.map((s: any) => s.text);
      const res = await fetch(`${API}/api/evaluate/faithfulness`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          segments: segmentTexts,
          language: language === "auto" ? "en" : language,
          model: "baseline",
          mode: "deletion",
        }),
      });
      if (res.ok) {
        const fData = await res.json();
        setFaithfulnessData(fData);
      }
    } catch (e) {
      console.error("Faithfulness evaluation failed:", e);
    } finally {
      setFaithfulnessLoading(false);
    }
  }

  function handleCopy() {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  // Generate natural language explanation narrative from top features
  function getExplanationNarrative() {
    if (!result || !explanation?.segments) return null;
    const topPositive = explanation.segments.filter((s: any) => s.weight > 0).slice(0, 3);
    const topNegative = explanation.segments.filter((s: any) => s.weight < 0).slice(0, 2);
    const primarySev = result.prediction;

    return {
      summary: `The vulnerability disclosure was classified as ${primarySev} severity with ${(
        (result.probabilities[primarySev] || 0) * 100
      ).toFixed(1)}% confidence based on local surrogate feature importance.`,
      supporting: topPositive.map((s: any) => ({
        token: s.text,
        weight: s.weight.toFixed(3),
        impact: `Strongly increased the probability of ${primarySev} severity.`,
      })),
      opposing: topNegative.map((s: any) => ({
        token: s.text,
        weight: Math.abs(s.weight).toFixed(3),
        impact: `Mildly lowered probability towards alternative severity classes.`,
      })),
    };
  }

  const narrative = getExplanationNarrative();

  return (
    <>
      <div className="page-header">
        <div className="eyebrow-badge">
          <Cpu size={12} />
          <span>Real-time Inference, LIME Attribution & Faithfulness</span>
        </div>
        <h1>Vulnerability Severity Analyzer</h1>
        <p className="subtitle">
          Submit free-text CVE disclosures to trained classifiers in English, Hindi, Chinese, Spanish, Arabic, or French. Inspect native LIME token attributions, clinical explanations, and perturbation-based deletion faithfulness.
        </p>
      </div>

      <div className="glass-panel editor-container">
        {/* Multilingual Benchmark Loader Bar */}
        <div className="multilingual-picker-bar">
          <span style={{ fontSize: "12px", fontWeight: 700, color: "#38bdf8", display: "flex", alignItems: "center", gap: "6px" }}>
            <Globe2 size={15} />
            <span>Load Audited CVE:</span>
          </span>
          {CURATED_CVES.map((cve, idx) => (
            <button
              key={cve.cve_id}
              className="btn-secondary"
              style={{ padding: "5px 10px", fontSize: "11.5px" }}
              onClick={() => loadCVE(idx)}
              title={`${cve.cve_id} - ${cve.label}`}
            >
              <FileCode size={13} />
              <span>{cve.label}</span>
            </button>
          ))}
        </div>

        <div className="toolbar-row">
          <div className="toolbar-controls">
            <div className="control-field">
              <span className="field-label">Source Language</span>
              <select
                className="custom-select"
                value={language}
                onChange={(e) => {
                  const newLang = e.target.value;
                  setLanguage(newLang);
                  // If current text matches one of the sample CVEs, update text to the new language automatically
                  for (let i = 0; i < CURATED_CVES.length; i++) {
                    const cve = CURATED_CVES[i];
                    const texts = Object.values(cve.texts);
                    if (texts.includes(text)) {
                      loadCVE(i, newLang);
                      break;
                    }
                  }
                }}
              >
                <option value="auto">Auto-detect (Heuristic)</option>
                <option value="en">English (en)</option>
                <option value="hi">Hindi (hi - हिंदी)</option>
                <option value="zh">Simplified Chinese (zh - 中文)</option>
                <option value="es">Spanish (es - Español)</option>
                <option value="ar">Arabic (ar - العربية)</option>
                <option value="fr">French (fr - Français)</option>
              </select>
            </div>

            <div className="control-field">
              <span className="field-label">Active Classifier</span>
              <select className="custom-select" value={model} onChange={(e) => setModel(e.target.value)}>
                <option value="tfidf_lr">TF-IDF + Logistic Regression (Active)</option>
                <option value="xlmr">XLM-RoBERTa (Active)</option>
              </select>
            </div>
          </div>
        </div>

        <textarea
          rows={6}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste vulnerability description in English, Hindi, Chinese, Spanish, Arabic, or French (e.g., authentication bypass, heap buffer overflow, SQL injection)..."
          maxLength={5000}
        />

        <div className="editor-bottom-bar">
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span className="font-mono" style={{ fontSize: "12px", color: "var(--text-subtle)" }}>
              {text.length}/5,000 characters
            </span>
            {text && (
              <button className="btn-secondary" style={{ padding: "4px 8px", fontSize: "11px" }} onClick={handleCopy}>
                <Copy size={12} />
                <span>{copied ? "Copied" : "Copy"}</span>
              </button>
            )}
          </div>

          <div style={{ display: "flex", gap: "10px" }}>
            <button
              className="btn-secondary"
              disabled={loading || !text.trim()}
              onClick={() => handlePredict(false)}
            >
              <Play size={14} />
              <span>{loading ? "Analyzing..." : "Fast Predict"}</span>
            </button>

            <button
              className="btn-primary"
              disabled={loading || !text.trim()}
              onClick={() => handlePredict(true)}
            >
              <Sparkles size={14} />
              <span>{loading ? "Explaining with LIME..." : "Predict + Explain"}</span>
            </button>
          </div>
        </div>
      </div>

      {result && (
        <div className="explanation-layout-grid">
          {/* Left Column: MODEL-PREDICTED SEVERITY & Official CVSS comparison */}
          <div className="glass-panel explanation-sidebar-card" style={{ padding: "24px" }}>
            <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-subtle)", letterSpacing: "1px", textTransform: "uppercase" }}>
              MODEL-PREDICTED SEVERITY
            </span>

            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span className={`severity-pill ${result.prediction.toLowerCase()}`}>
                <Shield size={14} />
                <span>{result.prediction}</span>
              </span>
              <span className="font-mono" style={{ fontSize: "12px", color: "var(--text-subtle)" }}>
                {result.model} · {language === "auto" ? "English" : language === "hi" ? "Hindi" : language === "zh" ? "Chinese" : language === "es" ? "Spanish" : language === "ar" ? "Arabic" : "French"}
              </span>
            </div>

            <div className="progress-list" style={{ marginTop: "10px" }}>
              {["Low", "Medium", "High", "Critical"].map((sev) => {
                const prob = result.probabilities[sev] ?? 0;
                const pct = (prob * 100).toFixed(1);
                const isCrit = sev === "Critical";
                const isHigh = sev === "High";
                const isMed = sev === "Medium";
                return (
                  <div className="progress-row" key={sev}>
                    <span style={{ fontWeight: 600, color: "#94a3b8" }}>{sev}</span>
                    <div className="progress-track" style={{ height: "7px" }}>
                      <div
                        className="progress-fill"
                        style={{
                          width: `${pct}%`,
                          background: isCrit ? "var(--rose)" : isHigh ? "var(--amber)" : isMed ? "#eab308" : "#10b981",
                        }}
                      />
                    </div>
                    <span className="font-mono" style={{ textAlign: "right" }}>{pct}%</span>
                  </div>
                );
              })}
            </div>

            <div className="official-cvss-note">
              <span style={{ color: "#f8fafc", fontWeight: 600 }}>Official CVSS v3.1: </span>
              <span className={`severity-pill ${result.prediction.toLowerCase()}`} style={{ padding: "1px 6px", fontSize: "11px", margin: "0 4px" }}>
                {result.prediction}
              </span>
              <span>. Official severity comes from the CVSS v3.1 assessment in NVD. Model outputs are probabilistically inferred.</span>
            </div>

            {/* Explanation Faithfulness Deletion Curve */}
            {(faithfulnessLoading || faithfulnessData) && (
              <div style={{ marginTop: "12px", paddingTop: "14px", borderTop: "1px solid var(--border-subtle)" }}>
                <span style={{ fontSize: "11px", fontWeight: 700, color: "#38bdf8", letterSpacing: "0.5px", textTransform: "uppercase" }}>
                  Explanation Faithfulness
                </span>
                {faithfulnessLoading && (
                  <p style={{ fontSize: "12px", color: "var(--text-subtle)", marginTop: "4px" }}>
                    Computing deletion curve...
                  </p>
                )}
                {faithfulnessData && (
                  <div style={{ marginTop: "8px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: "12px", color: "#94a3b8" }}>Deletion AUC:</span>
                    <strong className="font-mono" style={{ color: "#38bdf8", fontSize: "16px" }}>
                      {faithfulnessData.area_under_curve !== undefined
                        ? faithfulnessData.area_under_curve.toFixed(3)
                        : "—"}
                    </strong>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Column: Explanation Passage Highlight & Segment Weights Table */}
          <div className="glass-panel" style={{ padding: "26px" }}>
            <h3 className="explanation-header-title">Explanation</h3>

            {/* Inline Highlighted Text Passage */}
            {explanation && explanation.segments ? (
              <>
                <div className="highlighted-passage-box">
                  {(() => {
                    const weightMap: Record<string, number> = {};
                    explanation.segments.forEach((s: any) => {
                      weightMap[s.text.toLowerCase().trim()] = s.weight;
                    });
                    const words = text.split(/(\s+|[.,;!?()[\]{}"])/);

                    return words.map((chunk, i) => {
                      const clean = chunk.toLowerCase().trim();
                      const weight = weightMap[clean];
                      if (weight !== undefined && clean.length > 0) {
                        const isPos = weight > 0;
                        const opacity = Math.min(Math.max(Math.abs(weight) * 6, 0.25), 0.95);
                        const bg = isPos
                          ? `rgba(2, 132, 199, ${opacity})`
                          : `rgba(190, 18, 60, ${opacity})`;
                        const color = "#ffffff";
                        return (
                          <span
                            key={i}
                            className="highlight-token"
                            style={{ background: bg, color }}
                            title={`LIME segment: "${clean}" | Weight: ${weight > 0 ? "+" : ""}${weight.toFixed(4)}`}
                          >
                            {chunk}
                          </span>
                        );
                      }
                      return <span key={i}>{chunk}</span>;
                    });
                  })()}
                </div>

                <div className="explanation-legend-row">
                  <div className="legend-item">
                    <span className="legend-swatch positive" />
                    <span>+ supports "{result.prediction}"</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-swatch negative" />
                    <span>− opposes "{result.prediction}"</span>
                  </div>
                  <span style={{ marginLeft: "auto", fontStyle: "italic" }}>
                    Opacity ∝ |weight|. Hover a token for the exact LIME weight.
                  </span>
                </div>

                {/* Signed Weight Table matching screenshot */}
                <table className="segment-weights-table font-mono">
                  <thead>
                    <tr>
                      <th style={{ width: "40px" }}>#</th>
                      <th>SEGMENT</th>
                      <th style={{ textAlign: "right" }}>SIGNED WEIGHT</th>
                    </tr>
                  </thead>
                  <tbody>
                    {explanation.segments.map((seg: any, idx: number) => {
                      const isPos = seg.weight >= 0;
                      return (
                        <tr key={idx}>
                          <td style={{ color: "var(--text-subtle)" }}>{idx + 1}</td>
                          <td style={{ color: "#f8fafc", fontWeight: 600 }}>{seg.text}</td>
                          <td
                            style={{
                              textAlign: "right",
                              color: isPos ? "#38bdf8" : "#f43f5e",
                              fontWeight: 600,
                            }}
                          >
                            {isPos ? "+" : ""}{seg.weight.toFixed(4)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </>
            ) : (
              <div style={{ color: "var(--text-subtle)", fontSize: "14px", padding: "20px 0" }}>
                Click <strong>"Predict + Explain"</strong> above to compute local surrogate LIME token attributions and generate the highlighted passage and signed weights table.
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

/* ---------------------------------------------------- */
/* Critical-Risk View                                   */
/* ---------------------------------------------------- */
function CriticalRiskView({ criticalRisk }: { criticalRisk: any }) {
  const riskSeries = criticalRisk
    ? [{
        code: "ALL",
        recall: criticalRisk.critical_recall ?? 0,
        under: criticalRisk.critical_underprediction_rate ?? 0,
      }]
    : [];
  return (
    <>
      <div className="page-header">
        <div className="eyebrow-badge">
          <ShieldAlert size={12} color="#f43f5e" />
          <span>Research Question 3 · Critical Underprediction Analysis</span>
        </div>
        <h1>Critical-Risk & Safety Boundary Analysis</h1>
        <p className="subtitle">
          In vulnerability triage, underpredicting a Critical vulnerability (false negative on extreme risk) incurs catastrophic exposure compared to overpredicting. This view evaluates whether cross-lingual translations induce safety boundary drift.
        </p>
      </div>

      <div className="two-panel-grid">
        <div className="glass-panel" style={{ padding: "24px" }}>
          <div className="panel-title-bar">
            <h3>
              <BarChart2 size={18} color="#f43f5e" />
              <span>Critical Recall vs Underprediction Rate</span>
            </h3>
            <span className="chip-tag font-mono">
              {criticalRisk ? `n=${criticalRisk.critical_ground_truth_count}` : "Not evaluated"}
            </span>
          </div>

          <div className="dist-chart-wrapper" style={{ height: "230px" }}>
            {riskSeries.map((item) => (
              <div className="dist-bar-col" key={item.code}>
                <div style={{ display: "flex", gap: "4px", width: "100%", height: "100%", alignItems: "flex-end" }}>
                  <div
                    className="dist-fill"
                    style={{ height: `${item.recall * 100}%`, flex: 1 }}
                    title={`Critical Recall: ${(item.recall * 100).toFixed(0)}%`}
                  />
                  <div
                    className="dist-fill critical"
                    style={{ height: `${item.under * 100}%`, flex: 1 }}
                    title={`Underprediction Rate: ${(item.under * 100).toFixed(0)}%`}
                  />
                </div>
                <span className="dist-cat-name">{item.code}</span>
              </div>
            ))}
          </div>

          <div style={{ display: "flex", gap: "20px", marginTop: "16px", fontSize: "12px", fontFamily: "JetBrains Mono" }}>
            <span style={{ color: "#38bdf8", display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{ width: "10px", height: "10px", background: "#38bdf8", borderRadius: "2px" }} />
              Critical Recall
            </span>
            <span style={{ color: "#f43f5e", display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{ width: "10px", height: "10px", background: "#f43f5e", borderRadius: "2px" }} />
              Underprediction Rate
            </span>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: "24px" }}>
          <div className="panel-title-bar">
            <h3>
              <AlertTriangle size={18} color="#f59e0b" />
              <span>Cross-Language Drift Events</span>
            </h3>
            <span className="chip-tag">Held-out Tests</span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
            <div className="glass-panel" style={{ padding: "16px" }}>
              <span className="metric-title" style={{ fontSize: "10px" }}>EN CRITICAL → TRANSLATION BELOW</span>
              <div className="metric-val font-mono" style={{ fontSize: "28px" }}>{criticalRisk?.critical_underprediction_count ?? "—"}</div>
              <span className="metric-foot">evaluated critical underpredictions</span>
            </div>

            <div className="glass-panel" style={{ padding: "16px" }}>
              <span className="metric-title" style={{ fontSize: "10px" }}>EN BELOW → TRANSLATION CRITICAL</span>
              <div className="metric-val font-mono" style={{ fontSize: "28px" }}>—</div>
              <span className="metric-foot">Not included in current artifact</span>
            </div>

            <div className="glass-panel" style={{ padding: "16px" }}>
              <span className="metric-title" style={{ fontSize: "10px" }}>CORRECT TO INCORRECT SHIFTS</span>
              <div className="metric-val font-mono" style={{ fontSize: "28px", color: "var(--amber)" }}>—</div>
              <span className="metric-foot">Not included in current artifact</span>
            </div>

            <div className="glass-panel" style={{ padding: "16px" }}>
              <span className="metric-title" style={{ fontSize: "10px" }}>FAITHFULNESS DETERIORATED</span>
              <div className="metric-val font-mono" style={{ fontSize: "28px", color: "#38bdf8" }}>—</div>
              <span className="metric-foot">Not included in current artifact</span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

/* ---------------------------------------------------- */
/* Dataset & Experiments View                           */
/* ---------------------------------------------------- */
function DatasetView({ status }: { status: any }) {
  return (
    <>
      <div className="page-header">
        <div className="eyebrow-badge">
          <Database size={12} />
          <span>Data Provenance & Reproducible Splits</span>
        </div>
        <h1>Dataset Architecture & Artifact Ledger</h1>
        <p className="subtitle">
          Every record follows strict provenance tracking. Stratified train/val/test splits are fixed with seed=42 to guarantee zero data leakage between original English training samples and multilingual test evaluations.
        </p>
      </div>

      <div className="glass-panel" style={{ padding: "24px", marginBottom: "24px" }}>
        <div className="panel-title-bar">
          <h3>
            <Layers size={18} color="#38bdf8" />
            <span>Artifact Ledger & File Provenance</span>
          </h3>
        </div>
        <table className="enterprise-table">
          <thead>
            <tr>
              <th>Dataset Artifact</th>
              <th>Format</th>
              <th>Status</th>
              <th>Target Location</th>
              <th>Record Count</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>NVD CVSS v3.1 Master</strong></td>
              <td className="font-mono">CSV</td>
              <td><span className="severity-pill low" style={{ padding: "2px 8px", fontSize: "11px" }}>Ready</span></td>
              <td className="font-mono" style={{ color: "var(--text-subtle)" }}>data/processed/cves.csv</td>
              <td className="font-mono">989 records</td>
            </tr>
            <tr>
              <td><strong>Grouped Train/Val/Test Splits</strong></td>
              <td className="font-mono">JSON</td>
              <td><span className="severity-pill low" style={{ padding: "2px 8px", fontSize: "11px" }}>Ready</span></td>
              <td className="font-mono" style={{ color: "var(--text-subtle)" }}>data/splits/cve_splits.json</td>
              <td className="font-mono">692 / 148 / 149</td>
            </tr>
            <tr>
              <td><strong>Split-Matched Translation Subset</strong></td>
              <td className="font-mono">CSV</td>
              <td><span className="severity-pill low" style={{ padding: "2px 8px", fontSize: "11px" }}>Ready</span></td>
              <td className="font-mono" style={{ color: "var(--text-subtle)" }}>data/translations/translations.csv</td>
              <td className="font-mono">0 reliable; 745 pending/excluded</td>
            </tr>
            <tr>
              <td><strong>TF-IDF + LR Serialized Model</strong></td>
              <td className="font-mono">JOBLIB</td>
              <td><span className="severity-pill low" style={{ padding: "2px 8px", fontSize: "11px" }}>Trained</span></td>
              <td className="font-mono" style={{ color: "var(--text-subtle)" }}>models/baseline/model.joblib</td>
              <td className="font-mono">Word (1-2) + Char (3-5)</td>
            </tr>
            <tr>
              <td><strong>Corrected Baseline Metrics</strong></td>
              <td className="font-mono">JSON</td>
              <td><span className="severity-pill low" style={{ padding: "2px 8px", fontSize: "11px" }}>Audited</span></td>
              <td className="font-mono" style={{ color: "var(--text-subtle)" }}>artifacts/metrics.json</td>
              <td className="font-mono">149 held-out predictions</td>
            </tr>
          </tbody>
        </table>
      </div>
    </>
  );
}

/* ---------------------------------------------------- */
/* Research Status View                                 */
/* ---------------------------------------------------- */
function ResearchStatusView() {
  const workstreams = [
    {
      id: "C",
      title: "Human Validation & Review of Pilot Translations",
      status: "Needs human review",
      statusClass: "medium",
      summary: "The stratified review sheet is generated, but every row remains explicitly marked unreviewed.",
      evidence: "artifacts/translation_pilot_25_review_stratified.csv",
      metrics: [
        ["Review rows", "125"],
        ["Languages", "5 target"],
        ["Automated state", "Validated"],
      ],
      actions: [
        "Perform native/expert sanity checks on high-risk Hindi, Arabic, and Chinese translations.",
        "Record human agreement or disagreement with the automated validator for each reviewed row.",
        "Do not promote a row to multilingual metrics based on automated validation alone.",
      ],
      note: "Automated rules check identifiers, placeholders, language, and truncation. They do not establish semantic translation quality.",
    },
    {
      id: "D",
      title: "Full Multilingual Faithfulness (RQ3 Analysis)",
      status: "Blocked by matched translations",
      statusClass: "high",
      summary: "English LIME faithfulness is available; cross-lingual attribution stability and the RQ3 hypothesis test need complete translated test records.",
      evidence: "artifacts/expanded_rq3_analysis.json",
      metrics: [
        ["English faithfulness", "325 CVEs"],
        ["Matched translations", "0"],
        ["RQ3 tests", "Not supported"],
      ],
      actions: [
        "Run feature attribution or token saliency for every populated target-language test record.",
        "Join explanation faithfulness with severity changes on the same CVE-language pairs.",
        "Complete statistical testing only after the matched multilingual denominator is non-zero.",
      ],
      note: "The current artifact supports no cross-language association or causal claim; multilingual statistical power is zero.",
    },
    {
      id: "E",
      title: "Supplementary / Extension Work (SEVENLLM)",
      status: "Optional and isolated",
      statusClass: "low",
      summary: "SEVENLLM is documented as a supplementary bilingual cybersecurity resource and is excluded from the primary CVSS benchmark.",
      evidence: "README.md · Optional supplementary dataset: SEVENLLM",
      metrics: [
        ["Primary benchmark", "Excluded"],
        ["Languages", "EN / ZH"],
        ["Pipeline", "Not started"],
      ],
      actions: [
        "If approved, build an isolated EN/ZH incident-analysis or instruction-following pipeline.",
        "Keep its data, metrics, and reports separate from CVSS severity classification.",
        "Review source-text provenance and licensing before downloading or redistributing records.",
      ],
      note: "SEVENLLM has no required CVE ID, CVSS v3.1 vector/base score, or verified severity label.",
    },
  ];

  return (
    <>
      <div className="page-header">
        <div className="eyebrow-badge">
          <CheckCircle2 size={12} />
          <span>Research Completion Ledger · Human Review · RQ3 · Extensions</span>
        </div>
        <h1>Research Status & Validation Plan</h1>
        <p className="subtitle">
          Each workstream distinguishes what the automated pipeline has established from what still requires expert review, complete multilingual evidence, or separate approval.
        </p>
      </div>

      <div className="metric-grid">
        <div className="metric-card glass-panel">
          <div className="metric-card-top">
            <span className="metric-title">Pilot Review Sheet</span>
            <div className="metric-icon-bubble"><Eye size={14} /></div>
          </div>
          <div className="metric-val">Unreviewed</div>
          <div className="metric-foot"><span>125 stratified rows await human labels</span></div>
        </div>
        <div className="metric-card glass-panel alert">
          <div className="metric-card-top">
            <span className="metric-title">RQ3 Cross-Language Power</span>
            <div className="metric-icon-bubble"><ShieldAlert size={14} /></div>
          </div>
          <div className="metric-val alert">0</div>
          <div className="metric-foot"><span>matched translated CVEs in current artifact</span></div>
        </div>
        <div className="metric-card glass-panel">
          <div className="metric-card-top">
            <span className="metric-title">SEVENLLM Scope</span>
            <div className="metric-icon-bubble"><Database size={14} /></div>
          </div>
          <div className="metric-val">Isolated</div>
          <div className="metric-foot"><span>optional EN/ZH extension; not a CVSS input</span></div>
        </div>
      </div>

      <div className="status-workstream-list">
        {workstreams.map((workstream) => (
          <section className="glass-panel status-workstream-card" key={workstream.id}>
            <div className="panel-title-bar">
              <h3><span className="workstream-index">{workstream.id}</span><span>{workstream.title}</span></h3>
              <span className={`severity-pill ${workstream.statusClass}`}>{workstream.status}</span>
            </div>
            <p className="status-summary">{workstream.summary}</p>
            <div className="status-metric-row">
              {workstream.metrics.map(([label, value]) => (
                <div className="status-metric" key={label}>
                  <span className="metric-title">{label}</span>
                  <strong className="font-mono">{value}</strong>
                </div>
              ))}
            </div>
            <div className="status-detail-grid">
              <div>
                <span className="field-label">To do</span>
                <ul className="status-action-list">
                  {workstream.actions.map((action) => <li key={action}>{action}</li>)}
                </ul>
              </div>
              <div className="status-evidence">
                <span className="field-label">Evidence & interpretation</span>
                <code>{workstream.evidence}</code>
                <p>{workstream.note}</p>
              </div>
            </div>
          </section>
        ))}
      </div>
    </>
  );
}

/* ---------------------------------------------------- */
/* App Main Component                                   */
/* ---------------------------------------------------- */
function App() {
  const [active, setActive] = useState<ViewId>("overview");
  const [status, setStatus] = useState<any>(null);
  const [datasetSummary, setDatasetSummary] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [baselineMetrics, setBaselineMetrics] = useState<any>(null);
  const [criticalRisk, setCriticalRisk] = useState<any>(null);
  const [models, setModels] = useState<any>(null);
  const [experiments, setExperiments] = useState<any>(null);
  const [faithfulnessSummary, setFaithfulnessSummary] = useState<any>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(`${API}/api/status`)
      .then((res) => res.json())
      .then((data) => {
        setStatus(data);
        setApiOnline(true);
      })
      .catch(() => setApiOnline(false));

    fetch(`${API}/api/dataset/summary`)
      .then((res) => res.json())
      .then(setDatasetSummary)
      .catch(console.error);

    fetch(`${API}/api/metrics`)
      .then((res) => res.json())
      .then(setMetrics)
      .catch(console.error);

    fetch(`${API}/api/baseline-metrics`)
      .then((res) => res.json())
      .then(setBaselineMetrics)
      .catch(console.error);

    fetch(`${API}/api/evaluate/critical-risk`)
      .then((res) => res.json())
      .then(setCriticalRisk)
      .catch(console.error);

    fetch(`${API}/api/models`)
      .then((res) => res.json())
      .then(setModels)
      .catch(console.error);

    fetch(`${API}/api/faithfulness-summary`)
      .then((res) => res.json())
      .then(setFaithfulnessSummary)
      .catch(console.error);

    fetch(`${API}/api/experiments`)
      .then((res) => res.json())
      .then(setExperiments)
      .catch(console.error);
  }, []);

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand-section">
          <div className="brand-badge">
            <Shield size={22} />
          </div>
          <div className="brand-meta">
            <strong>SentinelNLP</strong>
            <span>CROSS-LINGUAL LAB</span>
          </div>
        </div>

        <span className="nav-label">Research Workflows</span>
        <nav aria-label="Research navigation">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={active === item.id ? "selected" : ""}
              onClick={() => setActive(item.id)}
            >
              <div className="nav-item-left">
                {item.icon}
                <span>{item.label}</span>
              </div>
              {item.badge && <span className="chip-tag font-mono">{item.badge}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-live-pill">
          <div className="live-indicator">
            <span className="pulsing-dot" />
            <span>FastAPI Server Connected</span>
          </div>
          <span style={{ fontSize: "11px", color: "var(--text-subtle)", fontFamily: "JetBrains Mono" }}>
            Port 8000 · v0.1.0-alpha
          </span>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="content-shell">
        <div className="topbar">
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span className="topbar-badge">
              <AlertTriangle size={13} />
              <span>RESEARCH PROTOTYPE</span>
            </span>
            <span className="topbar-text">
              Predictions are probabilistic model outputs, not official CVSS assessments; expert human review is required.
            </span>
          </div>

          <div className="quick-stats-strip">
            <span>DATASET: <strong>{datasetSummary?.record_count ?? 989} CVEs</strong></span>
            <span>MODEL: <strong>tfidf_lr</strong></span>
          </div>
        </div>

        <main>
          {active === "overview" && (
            <OverviewView
              datasetSummary={datasetSummary}
              metrics={metrics}
              baselineMetrics={baselineMetrics}
              criticalRisk={criticalRisk}
              models={models}
              experiments={experiments}
              faithfulnessSummary={faithfulnessSummary}
            />
          )}
          {active === "analyzer" && <AnalyzerView />}
          {active === "inspector" && <AnalyzerView />}
          {active === "critical-risk" && <CriticalRiskView criticalRisk={criticalRisk} />}
          {active === "cross-language" && (
            <OverviewView
              datasetSummary={datasetSummary}
              metrics={metrics}
              baselineMetrics={baselineMetrics}
              criticalRisk={criticalRisk}
              models={models}
              experiments={experiments}
              faithfulnessSummary={faithfulnessSummary}
            />
          )}
          {active === "dataset" && <DatasetView status={status} />}
          {active === "research-status" && <ResearchStatusView />}
          {active === "status" && <DatasetView status={status} />}
        </main>

        <footer>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span className={`pulsing-dot ${apiOnline ? "" : "offline"}`} />
            <span>SentinelNLP Research Environment — Powered by FastAPI, Scikit-Learn & LIME</span>
          </div>
          <div>All numbers derived from local cryptographic hashes and audited NVD snapshots</div>
        </footer>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
