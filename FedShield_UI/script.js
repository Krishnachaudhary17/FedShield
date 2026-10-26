// FedShield Dashboard — Real federated training via Flask API
// Clicking "Start Training" triggers ACTUAL PyTorch FedAvg on the backend.
// Live logs and metrics are streamed back via polling /train/status.

const API = 'http://127.0.0.1:5000';
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));

function log(t, good = false) {
  let d = document.createElement('div');
  d.textContent = t;
  if (good) d.className = 'good';
  $('log').appendChild(d);
  $('log').scrollTop = $('log').scrollHeight;
}

function setMetrics(m) {
  if (m.accuracy)  $('accuracy').textContent  = m.accuracy;
  if (m.precision) $('precision').textContent = m.precision;
  if (m.recall)    $('recall').textContent    = m.recall;
  if (m.f1)        $('f1').textContent        = m.f1;
  if (m.epsilon)   $('epsilon').textContent   = m.epsilon;
  if (m.round)     $('round').textContent     = m.round;
}

function animateClients(progress) {
  ['c1','c2','c3'].forEach(id => {
    const bar = $(id).querySelector('.bar i');
    const em  = $(id).querySelector('em');
    bar.style.width = progress + '%';
    em.textContent  = progress >= 100 ? 'Update ready' : 'Training';
  });
}

let _currentRounds = 4;

async function initDashboard() {
  try {
    const res = await fetch(`${API}/model_info`);
    if (res.ok) {
      const data = await res.json();
      if (data.total_rounds) {
        _currentRounds = data.total_rounds;
        $('round').textContent = data.total_rounds;
        $('startBtn').textContent = `Train Round ${data.total_rounds + 1}`;
        if (data.metrics && Object.keys(data.metrics).length > 0) {
          setMetrics(data.metrics);
        }
      }
    }
  } catch (e) {
    // API not running yet
  }
}
initDashboard();

async function pollStatus(btn) {
  while (true) {
    await sleep(800);
    try {
      const res  = await fetch(`${API}/train/status`);
      const data = await res.json();

      // Append only new log lines
      const newLogs = data.logs.slice(_lastLogCount);
      newLogs.forEach(line => {
        const isGood = line.includes('done') || line.includes('complete') || line.includes('updated');
        log(line, isGood);
      });
      _lastLogCount = data.logs.length;

      // Update metrics live if available
      if (data.metrics && data.metrics.round) {
        _currentRounds = data.metrics.round;
        setMetrics(data.metrics);
        // Animate bars proportionally to round progress
        const pct = Math.min(30 + (data.metrics.round % 5 + 1) * 20, 95);
        animateClients(pct);
      }

      if (data.done) {
        animateClients(100);
        if (data.error) {
          log('ERROR: ' + data.error);
          btn.textContent = `Train Round ${_currentRounds + 1}`;
          btn.disabled    = false;
        } else {
          const finalRound = (data.metrics && data.metrics.round) || _currentRounds;
          $('round').textContent = finalRound;
          log(`Federated training complete! Global model updated to Round ${finalRound}. Fraud Analysis page now uses the new weights.`, true);
          btn.textContent = `Train Round ${finalRound + 1}`;
          btn.disabled    = false;
        }
        break;
      }
    } catch (e) {
      log('Lost connection to API server: ' + e.message);
      btn.disabled    = false;
      btn.textContent = `Train Round ${_currentRounds + 1}`;
      break;
    }
  }
}

$('startBtn').onclick = async () => {
  const btn = $('startBtn');

  // Check API is reachable first
  try {
    await fetch(`${API}/health`);
  } catch (e) {
    $('log').innerHTML = '';
    log('Cannot reach API server. Run: python api_server.py');
    return;
  }

  btn.disabled    = true;
  btn.textContent = `Training Round ${_currentRounds + 1} in progress...`;
  $('log').innerHTML = '';
  _lastLogCount   = 0;
  animateClients(0);
  ['c1','c2','c3'].forEach(id => $(id).querySelector('em').textContent = 'Ready');

  log('Sending training request to FedShield API...');
  log(`Real PyTorch FedAvg training starting for Round ${_currentRounds + 1}...`);

  try {
    const res = await fetch(`${API}/train`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ num_rounds: 1 })
    });
    const data = await res.json();
    if (data.error) {
      log('Error: ' + data.error);
      btn.disabled    = false;
      btn.textContent = `Train Round ${_currentRounds + 1}`;
      return;
    }
    log('Backend training thread started. Streaming live PyTorch logs...');
    await pollStatus(btn);
  } catch (e) {
    log('Failed to start training: ' + e.message);
    btn.disabled    = false;
    btn.textContent = `Train Round ${_currentRounds + 1}`;
  }
};
