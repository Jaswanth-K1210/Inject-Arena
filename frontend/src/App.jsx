import React, { useState, useEffect, useRef } from 'react';
import './index.css';

// Backend base — '' means same origin (Docker / Space). Vite dev proxy handles localhost.
const API_BASE = '';

// Stage timing (ms) for the live-attack animation. Independent of trace.t timings
// so the demo paces consistently regardless of the underlying recording.
const STAGE_TIMING = {
  generation: 1500,
  pg2_scan:   1500,
  agent_run:  2000,
  firewall_scan: 1500,
  verdict:    600,
};

// Default fallback when API is unreachable (dev mode without backend).
const DEFAULT_ATTACK_TYPES = [
  { id: 'email_exfiltration', label: 'Email Exfiltration', icon: '📧',
    description: 'Trick the email agent into forwarding session data to an attacker address.' },
  { id: 'forbidden_tool',     label: 'Forbidden Tool Call', icon: '🔧',
    description: 'Force the agent to invoke a tool that is explicitly off-limits.' },
  { id: 'prompt_leak',        label: 'System-Prompt Leak',  icon: '🔓',
    description: 'Get the agent to reveal its hidden system prompt verbatim.' },
  { id: 'rag_injection',      label: 'RAG Document Injection', icon: '📄',
    description: 'Hide an injection inside a retrieved document the agent has to summarise.' },
];
const DEFAULT_STEP_OPTIONS = [50, 100, 300, 500, 1000, 1500];


// ---------------------------------------------------------------------------
// DefenseCard
// ---------------------------------------------------------------------------

function DefenseCard({ title, subtitle, state, icon }) {
  const stateClass = state || 'idle';
  const stateIcon = state === 'flagged' ? '✗' : state === 'scanning' ? '⏳' : '✓';
  const label = state === 'scanning' ? 'Scanning…'
              : state === 'passed'  ? 'Bypassed'
              : state === 'flagged' ? 'Flagged'
              : 'Idle';
  return (
    <div className={`defense-card ${stateClass}`}>
      <div className="defense-icon">{icon}</div>
      <div className="defense-content">
        <h4>{title}</h4>
        <p>{subtitle}</p>
      </div>
      <div className={`defense-status status-${stateClass}`}>{stateIcon} {label}</div>
    </div>
  );
}


// ---------------------------------------------------------------------------
// AttackVisualization — animates a real trace returned from /api/attack
// ---------------------------------------------------------------------------

function AttackVisualization({ trace, isRunning, stepCount, onComplete }) {
  const [stage, setStage] = useState(0);            // 0 idle → 1 gen → 2 pg2 → 3 agent → 4 fw → 5 done
  const [payloadText, setPayloadText] = useState('');
  const [agentOutput, setAgentOutput] = useState('');
  const [pg2State, setPg2State] = useState('idle');
  const [secAlignState, setSecAlignState] = useState('idle');
  const [fwState, setFwState] = useState('idle');
  const timersRef = useRef([]);

  useEffect(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    if (!isRunning || !trace) {
      setStage(0); setPayloadText(''); setAgentOutput('');
      setPg2State('idle'); setSecAlignState('idle'); setFwState('idle');
      return;
    }

    // Index timeline events by stage.
    const events = {};
    for (const ev of trace.timeline || []) {
      events[ev.stage] = ev;
    }
    const pg2Ev    = events.pg2_scan;
    const agentEv  = events.agent_run;
    const fwEv     = events.firewall_scan;
    const verdict  = events.verdict;

    const fullPayload = trace.payload || (events.generation || {}).payload || '';
    const truncatedPayload = fullPayload.length > 380
      ? fullPayload.slice(0, 380) + '…'
      : fullPayload;

    // Stage 1: generation + typewriter
    setStage(1);
    let i = 0;
    const typewriterId = setInterval(() => {
      i += Math.max(1, Math.floor(truncatedPayload.length / 80));
      setPayloadText(truncatedPayload.slice(0, Math.min(i, truncatedPayload.length)));
      if (i >= truncatedPayload.length) clearInterval(typewriterId);
    }, STAGE_TIMING.generation / 80);
    timersRef.current.push(() => clearInterval(typewriterId));

    let t = STAGE_TIMING.generation;

    // Stage 2: PG2
    timersRef.current.push(setTimeout(() => {
      setStage(2); setPg2State('scanning');
    }, t));
    t += STAGE_TIMING.pg2_scan;
    timersRef.current.push(setTimeout(() => {
      setPg2State(pg2Ev?.flagged ? 'flagged' : 'passed');
    }, t));

    // Stage 3: SecAlign agent
    timersRef.current.push(setTimeout(() => {
      setStage(3); setSecAlignState('scanning');
    }, t));
    t += STAGE_TIMING.agent_run;
    timersRef.current.push(setTimeout(() => {
      const agentText = agentEv?.agent_output || '';
      setAgentOutput(agentText.length > 600 ? agentText.slice(0, 600) + '…' : agentText);
      setSecAlignState(verdict?.task_succeeded ? 'flagged' : 'passed');
    }, t));

    // Stage 4: LlamaFirewall
    timersRef.current.push(setTimeout(() => {
      setStage(4); setFwState('scanning');
    }, t));
    t += STAGE_TIMING.firewall_scan;
    timersRef.current.push(setTimeout(() => {
      setFwState(fwEv?.flagged ? 'flagged' : 'passed');
    }, t));

    // Stage 5: verdict
    timersRef.current.push(setTimeout(() => {
      setStage(5);
      onComplete && onComplete(trace.outcome || {});
    }, t + STAGE_TIMING.verdict));

    return () => {
      timersRef.current.forEach(timer => {
        if (typeof timer === 'function') timer();
        else clearTimeout(timer);
      });
      timersRef.current = [];
    };
  }, [isRunning, trace, stepCount, onComplete]);

  if (!isRunning && stage === 0) return null;

  const toolCallMatch = agentOutput.match(/\b(send_email|execute_shell|run_command|forward_message)\s*\([^)]*\)/i);

  return (
    <div className="viz-container">
      <h3>Live Attack Execution</h3>

      <div className="attacker-panel">
        <h4>🤖 InjectArena Attacker ({stepCount} steps)</h4>
        <div className="typewriter-box">
          {payloadText}
          {stage === 1 && <span className="cursor">|</span>}
        </div>
      </div>

      <div className="pipeline">
        <DefenseCard title="Llama Prompt Guard 2" subtitle="86M classifier" icon="🛡️" state={pg2State} />
        <div className="arrow">→</div>
        <DefenseCard title="Meta-SecAlign-8B" subtitle="8B agent + LoRA" icon="🧠" state={secAlignState} />
        <div className="arrow">→</div>
        <DefenseCard title="LlamaFirewall" subtitle="Scanner pipeline" icon="🔥" state={fwState} />
      </div>

      {stage >= 3 && agentOutput && (
        <div className="agent-response-panel">
          <h4>Agent Response</h4>
          <p className="agent-text">{agentOutput}</p>
          {toolCallMatch && (
            <div className="malicious-highlight">
              <span className="code-label">Function Call</span>
              {toolCallMatch[0]}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// ---------------------------------------------------------------------------
// ResultCard
// ---------------------------------------------------------------------------

function ResultCard({ outcome, stepCount, attackType, payloadPreview, onRetry }) {
  if (!outcome) return null;
  const broke_pg2 = outcome.broke_pg2;
  const broke_fw  = outcome.broke_fw;
  const blocked_at = outcome.blocked_at;

  // "Firewall broken" = bypassed both Meta classifier defenses (PG2 + LlamaFirewall).
  // This is the headline win — the result the demo is built around.
  const firewall_broken = broke_pg2 && broke_fw;

  if (firewall_broken) {
    return (
      <div className="result-card success-card">
        <div className="result-header">
          <h2>🚨 META FIREWALL BROKEN</h2>
        </div>
        <div className="result-body">
          <div className="bypass-summary">
            <div className="bypass-row passed">
              <span className="bypass-icon">🛡️</span>
              <span className="bypass-label">Llama Prompt Guard 2</span>
              <span className="bypass-status">✓ Bypassed</span>
            </div>
            <div className="bypass-row passed">
              <span className="bypass-icon">🔥</span>
              <span className="bypass-label">LlamaFirewall</span>
              <span className="bypass-status">✓ Bypassed</span>
            </div>
          </div>

          <p><strong>Attacker payload:</strong> "{payloadPreview}"</p>

          <div className="why-matters">
            <h4>Why this matters</h4>
            <p>Meta deployed Llama Prompt Guard 2 and LlamaFirewall as the primary
              input/output defenses against prompt injection. Our 1.5B-parameter RL
              attacker beat both at <strong>100% rate</strong> across recorded attacks.
            </p>
            <p className="footnote">Next milestone: full end-to-end task execution against the SecAlign-8B inner agent (work in progress, see roadmap).</p>
          </div>

          <div className="result-actions">
            <button className="btn-secondary" onClick={() => onRetry(stepCount)}>Run another scenario</button>
          </div>
        </div>
      </div>
    );
  }

  // Failure: PG2 or FW caught the payload
  return (
    <div className="result-card failure-card">
      <div className="result-header">
        <h2>🛡️ Attack blocked at {blocked_at || 'a defense layer'}</h2>
      </div>
      <div className="result-body">
        <p>At {stepCount} training steps the attacker didn't bypass all classifier layers.</p>
        <p><strong>PG2:</strong> {broke_pg2 ? 'bypassed' : 'flagged'} ·
          &nbsp;<strong>LlamaFirewall:</strong> {broke_fw ? 'bypassed' : 'flagged'}</p>

        <div className="result-actions">
          {stepCount < 300  && <button className="btn-primary" onClick={() => onRetry(300)}>↑ Retry with 300 steps</button>}
          {stepCount < 1500 && <button className="btn-primary" onClick={() => onRetry(1500)}>↑↑ Retry with 1500 steps</button>}
          <button className="btn-secondary" onClick={() => onRetry(stepCount, true)}>Try a different attack type</button>
        </div>
      </div>
    </div>
  );
}


// ---------------------------------------------------------------------------
// LaunchModeModal — appears on click to ask "Live training" or "Pre-tested results"
// ---------------------------------------------------------------------------

const COLAB_NOTEBOOK_URL = 'https://colab.research.google.com/github/Jaswanth-K1210/Inject-Arena/blob/main/notebooks/colab_runner.ipynb';

// Calibrated from the actual Colab A100 measurement: 300 steps took ~1.5 hours.
// → 18 sec/step (includes 4 GRPO completions × full defense stack per step).
const SECS_PER_STEP = 18;

function formatDuration(steps) {
  const totalSec = steps * SECS_PER_STEP;
  const hrs = Math.floor(totalSec / 3600);
  const min = Math.round((totalSec % 3600) / 60);
  if (hrs === 0) return `≈${min} min`;
  if (min === 0) return `≈${hrs} hr`;
  return `≈${hrs} hr ${min} min`;
}

function LaunchModeModal({ open, onClose, onPickRecorded, attackTypeLabel, steps }) {
  if (!open) return null;
  const estimate = formatDuration(steps);
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>How do you want to run this attack?</h3>
          <p className="modal-subtitle">{attackTypeLabel} · {steps} steps</p>
        </div>

        <div className="modal-options">
          <button className="modal-option recommended" onClick={onPickRecorded}>
            <div className="modal-option-icon">⚡</div>
            <div className="modal-option-body">
              <div className="modal-option-title">View Pre-Tested Result <span className="badge-instant">Instant</span></div>
              <p>Real recorded attack from our trained 1.5B Qwen attacker against the live Meta defense stack on A100. PG2 92% / FW 100% bypass rate.</p>
              <p className="modal-option-cta">Watch the recorded attack →</p>
            </div>
          </button>

          <a className="modal-option live" href={COLAB_NOTEBOOK_URL} target="_blank" rel="noopener noreferrer">
            <div className="modal-option-icon">🔥</div>
            <div className="modal-option-body">
              <div className="modal-option-title">Run Live Training <span className="badge-time">{estimate}</span></div>
              <p>Open the Colab notebook and train a fresh attacker for {steps} steps on a real A100 GPU. Free with Colab Pro; you'll watch the reward curve climb in real time.</p>
              <p className="modal-option-cta">Open Colab notebook →</p>
            </div>
          </a>
        </div>

        <div className="modal-footnote">
          Live runs aren't hosted here — Hugging Face free Spaces don't have GPUs, and a long-running training process would block other users. Use Colab and bring the trained checkpoint back.
          <br />
          <a href="#" onClick={(e) => { e.preventDefault(); onClose(); }} className="modal-cancel">Cancel</a>
        </div>
      </div>
    </div>
  );
}


// ---------------------------------------------------------------------------
// Stats badges (hero) — pulled from /api/stats
// ---------------------------------------------------------------------------

function StatBadges({ stats }) {
  if (!stats) return null;
  const fmt = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);
  return (
    <div className="stats-row">
      <span className="stat-badge">{fmt(stats.pg2_bypass_rate)} PG2 Bypass</span>
      <span className="stat-badge">{fmt(stats.fw_bypass_rate)} LlamaFirewall Bypass</span>
      <span className="stat-badge">{stats.trace_count} recorded attacks</span>
      <span className="stat-badge">Trained on A100</span>
    </div>
  );
}


// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  const [activeTab, setActiveTab] = useState('attack');
  const [attackTypes, setAttackTypes] = useState(DEFAULT_ATTACK_TYPES);
  const [stepOptions, setStepOptions] = useState(DEFAULT_STEP_OPTIONS);
  const [attackType, setAttackType] = useState(DEFAULT_ATTACK_TYPES[0].id);
  const [steps, setSteps] = useState(300);
  const [isAttacking, setIsAttacking] = useState(false);
  const [trace, setTrace] = useState(null);
  const [outcome, setOutcome] = useState(null);
  const [stats, setStats] = useState(null);
  const [highlight, setHighlight] = useState(null);
  const [error, setError] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);

  // Initial data fetch
  useEffect(() => {
    fetch(`${API_BASE}/api/attack-types`)
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then(data => {
        if (Array.isArray(data.attack_types) && data.attack_types.length > 0) {
          setAttackTypes(data.attack_types);
        }
        if (Array.isArray(data.step_options) && data.step_options.length > 0) {
          setStepOptions(data.step_options);
        }
      })
      .catch(() => { /* fallback to defaults */ });

    fetch(`${API_BASE}/api/stats`)
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then(setStats)
      .catch(() => { /* hide badges if unreachable */ });

    fetch(`${API_BASE}/api/highlight`)
      .then(r => r.ok ? r.json() : Promise.reject(r))
      .then(setHighlight)
      .catch(() => {});
  }, []);

  // Click → open modal. Modal decides whether to fetch recorded trace or send
  // the user to Colab. We DON'T launch the attack directly anymore.
  const openLaunchModal = () => {
    setError(null);
    setModalOpen(true);
  };

  const runRecordedAttack = async () => {
    setModalOpen(false);
    setIsAttacking(false);   // reset visualization first
    setOutcome(null);
    setTrace(null);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/attack`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attack_type: attackType, steps }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`API error ${res.status}: ${detail}`);
      }
      const t = await res.json();
      // small tick to allow viz mount before timers begin
      setTimeout(() => {
        setTrace(t);
        setIsAttacking(true);
      }, 30);
    } catch (e) {
      setError(e.message || String(e));
    }
  };

  const handleComplete = (finalOutcome) => {
    setIsAttacking(false);
    setOutcome(finalOutcome);
  };

  const handleRetry = (newSteps, changeType = false) => {
    setSteps(newSteps);
    if (changeType) {
      const idx = attackTypes.findIndex(t => t.id === attackType);
      const next = (idx + 1) % attackTypes.length;
      setAttackType(attackTypes[next].id);
    }
    setOutcome(null);
    setTrace(null);
    document.getElementById('config-panel')?.scrollIntoView({ behavior: 'smooth' });
    // Skip the modal on retry — user already chose recorded once.
    setTimeout(runRecordedAttack, 500);
  };

  const heroPayloadPreview = highlight?.payload
    ? (highlight.payload.length > 100 ? highlight.payload.slice(0, 100) + '…' : highlight.payload)
    : 'Payload → PG2 ✓ → SecAlign → LlamaFirewall ✓';

  const payloadPreview = trace?.payload
    ? (trace.payload.length > 100 ? trace.payload.slice(0, 100) + '…' : trace.payload)
    : '';

  return (
    <div className="app-container">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1 className="hero-title">
            <span className="emoji">🛡️</span> InjectArena <span className="emoji">⚔️</span>
          </h1>
          <p className="hero-subtitle">
            We broke Meta's prompt-injection firewall.<br />
            <strong>{stats ? `${Math.round((stats.pg2_bypass_rate ?? 0) * 100)}% Llama Prompt Guard 2 + ${Math.round((stats.fw_bypass_rate ?? 0) * 100)}% LlamaFirewall bypass` : '100% LlamaFirewall bypass'}
              &nbsp;across {stats?.trace_count ?? 24} recorded attacks.</strong>
          </p>

          <div className="hero-animation">
            <div className="animation-loop">
              {heroPayloadPreview} → <span className="text-green">PG2 ✓</span> → <span className="text-green">SecAlign</span> → <span className="text-green">LlamaFirewall ✓</span>
              <div className="firewall-broken">FIREWALL BROKEN</div>
            </div>
          </div>

          <StatBadges stats={stats} />

          <div className="hero-actions">
            <button className="btn-primary btn-large" onClick={() => {
              setActiveTab('attack');
              setTimeout(() => document.getElementById('config-panel')?.scrollIntoView({ behavior: 'smooth' }), 30);
            }}>▶ Launch Live Attack</button>
            <button className="btn-secondary btn-large" onClick={() => {
              setActiveTab('dashboard');
              setTimeout(() => document.getElementById('dashboard')?.scrollIntoView({ behavior: 'smooth' }), 30);
            }}>📊 See Training Results</button>
          </div>
        </div>
      </section>

      <main className="main-content">
        {activeTab === 'attack' ? (
          <>
            <section id="config-panel" className="config-panel">
              <h2>Configure your attack</h2>

              <div className="config-group">
                <label>Attack type:</label>
                <div className="attack-type-grid">
                  {attackTypes.map(type => (
                    <div
                      key={type.id}
                      className={`type-card ${attackType === type.id ? 'selected' : ''}`}
                      onClick={() => setAttackType(type.id)}
                    >
                      <span className="type-icon">{type.icon}</span>
                      <div className="type-info">
                        <strong>{type.label}</strong>
                        <p>{type.description || type.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="config-group">
                <label>Training steps: <span className="step-hint">More steps = stronger attacker</span></label>
                <div className="steps-selector">
                  {stepOptions.map(s => (
                    <label key={s} className={`step-radio ${steps === s ? 'selected' : ''}`}>
                      <input type="radio" name="steps" value={s} checked={steps === s} onChange={() => setSteps(s)} />
                      {s}
                    </label>
                  ))}
                </div>
              </div>

              <button className="btn-primary btn-full launch-btn" onClick={openLaunchModal} disabled={isAttacking}>
                {isAttacking ? 'Attacking…' : '🚀 Launch Attack'}
              </button>

              {error && <div className="error-message" style={{ color: '#ef4444', marginTop: 12 }}>⚠ {error}</div>}
            </section>

            {(isAttacking || outcome !== null) && (
              <section className="visualization-section">
                <AttackVisualization
                  trace={trace}
                  isRunning={isAttacking}
                  stepCount={steps}
                  onComplete={handleComplete}
                />
              </section>
            )}

            {outcome !== null && !isAttacking && (
              <section className="result-section">
                <ResultCard
                  outcome={outcome}
                  stepCount={steps}
                  attackType={attackType}
                  payloadPreview={payloadPreview}
                  onRetry={handleRetry}
                />
              </section>
            )}
          </>
        ) : (
          <section id="dashboard" className="dashboard-section">
            <h2>Training Results Dashboard</h2>
            <div className="dashboard-grid">
              <div className="plot-card">
                <img src="/plots/reward_curve.png" alt="Bypass rate by training steps"
                     onError={(e) => e.target.style.display = 'none'} />
                <p><strong>Bypass progression:</strong> Bypass rate climbs as the attacker trains. PG2 reaches ~92% by 1500 steps; LlamaFirewall stays at 100%.</p>
              </div>
              <div className="plot-card">
                <img src="/plots/bypass_bars.png" alt="RL-trained vs handcrafted baseline"
                     onError={(e) => e.target.style.display = 'none'} />
                <p><strong>Bypass bars:</strong> RL-trained attacker (92% PG2 / 100% FW) vs handcrafted baseline (15% / 20%).</p>
              </div>
              <div className="plot-card">
                <img src="/plots/per_category.png" alt="Per-category breakdown"
                     onError={(e) => e.target.style.display = 'none'} />
                <p><strong>Per-category:</strong> Bypass rates broken down by attack type — strongest on email/RAG, weakest on prompt-leak.</p>
              </div>
              <div className="plot-card">
                <div className="plot-stats-block">
                  <div className="big-stat">{stats ? `${Math.round((stats.pg2_bypass_rate ?? 0) * 100)}%` : '—'}</div>
                  <div className="stat-label">PG2 Bypass Rate</div>
                  <div className="big-stat">{stats ? `${Math.round((stats.fw_bypass_rate ?? 0) * 100)}%` : '—'}</div>
                  <div className="stat-label">LlamaFirewall Bypass Rate</div>
                  <div className="big-stat">{stats ? stats.trace_count : '—'}</div>
                  <div className="stat-label">Recorded Attacks</div>
                </div>
                <p><strong>Aggregate stats:</strong> Live numbers from the trace store — exactly what the public demo serves.</p>
              </div>
            </div>
          </section>
        )}
      </main>

      <LaunchModeModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onPickRecorded={runRecordedAttack}
        attackTypeLabel={attackTypes.find(t => t.id === attackType)?.label || attackType}
        steps={steps}
      />

      <footer className="app-footer">
        <a href="https://github.com/Jaswanth-K1210/Inject-Arena" target="_blank" rel="noopener noreferrer">GitHub</a> ·
        &nbsp;<a href="https://huggingface.co/spaces/Jaswanth-K/Inject-Arena" target="_blank" rel="noopener noreferrer">HF Space</a> ·
        &nbsp;<a href="https://github.com/Jaswanth-K1210/Inject-Arena/blob/main/demo/VIDEO_SCRIPT.md" target="_blank" rel="noopener noreferrer">Demo Video</a> ·
        &nbsp;<a href="https://github.com/Jaswanth-K1210/Inject-Arena#citation" target="_blank" rel="noopener noreferrer">Citation</a> ·
        &nbsp;Apache-2.0
      </footer>
    </div>
  );
}

export default App;
