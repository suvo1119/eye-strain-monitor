const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('overlay');
const canvasCtx = canvasElement.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');

const blinksTotalEl = document.getElementById('blinksTotal');
const blinksPerMinEl = document.getElementById('blinksPerMin');
const perclosEl = document.getElementById('perclos');
const strainEl = document.getElementById('strain');

let camera = null;
let running = false;
let sessionStart = null;
let sessionEnd = null;

// indices based on MediaPipe face mesh
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

const WINDOW_SECONDS = 60;
const BLINK_EAR_THRESH = 0.25;
const PERCLOS_EAR_THRESH = 0.25;

let framesBuffer = []; // {t, ear, closed}
let blinkEvents = []; // {start, end, dur}
let currentBlinkStart = null;
let blinkCount = 0;

// basic feature checks
const cameraSelect = document.getElementById('cameraSelect');
const refreshCams = document.getElementById('refreshCams');
const errorMsg = document.getElementById('errorMsg');

if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
  statusEl.textContent = 'Camera not available in this browser. Use Chrome/Edge on HTTPS or localhost.';
  startBtn.disabled = true;
  cameraSelect.disabled = true;
  refreshCams.disabled = true;
}

async function enumerateCameras(){
  try{
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d=>d.kind === 'videoinput');
    cameraSelect.innerHTML = '';
    videoInputs.forEach((d, i)=>{
      const label = d.label || `Camera ${i+1}`;
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = label;
      cameraSelect.appendChild(opt);
    });
    if(videoInputs.length === 0){
      const opt = document.createElement('option'); opt.textContent = 'No cameras found'; opt.disabled = true; cameraSelect.appendChild(opt);
    }
  }catch(err){
    console.error('enumerateDevices failed', err);
    errorMsg.textContent = 'Could not enumerate cameras: ' + (err.message || err);
  }
}

refreshCams.addEventListener('click', async ()=>{ errorMsg.textContent = ''; await enumerateCameras(); });

cameraSelect.addEventListener('change', async ()=>{
  // If running, restart stream with chosen device
  if(running){
    stop();
    await start();
  }
});

// Request permission button to explicitly prompt for camera access
const requestPermBtn = document.getElementById('requestPerm');
requestPermBtn.addEventListener('click', async ()=>{
  errorMsg.textContent = '';
  try{
    const s = await navigator.mediaDevices.getUserMedia({video:true});
    // immediately stop tracks (we only wanted permission)
    s.getTracks().forEach(t=>t.stop());
    errorMsg.textContent = 'Camera permission granted. Click Start.';
    await enumerateCameras();
  }catch(err){
    console.error('Permission request failed', err);
    errorMsg.textContent = 'Permission denied or error. Check site camera settings.';
  }
});

async function checkPermissionState(){
  if(!navigator.permissions || !navigator.permissions.query) return;
  try{
    const status = await navigator.permissions.query({name:'camera'});
    if(status.state === 'denied'){
      errorMsg.textContent = 'Camera access is blocked. Open site settings to allow camera.';
      startBtn.disabled = true;
    }else if(status.state === 'granted'){
      errorMsg.textContent = '';
      startBtn.disabled = false;
    }
    status.onchange = ()=>{ checkPermissionState(); };
  }catch(e){
    // ignore if permission API not available for camera
  }
}

// Try to populate camera list up front (labels may be empty until permission granted)
enumerateCameras();
checkPermissionState();


// elements for report
const reportPanel = document.getElementById('reportPanel');
const reportSummary = document.getElementById('reportSummary');
const reportActions = document.getElementById('reportActions');
const downloadTxtBtn = document.getElementById('downloadTxt');
const downloadCsvBtn = document.getElementById('downloadCsv');
const copyReportBtn = document.getElementById('copyReport');
const rawReportEl = document.getElementById('rawReport');
let chart = null;

function nowSec(){return Date.now()/1000.0}

function purgeOld(now){
  const cutoff = now - WINDOW_SECONDS;
  framesBuffer = framesBuffer.filter(x=>x.t>=cutoff);
  blinkEvents = blinkEvents.filter(x=>x.start>=cutoff);
}

function EAR(landmarks, eye){
  const p1 = landmarks[eye[0]];
  const p2 = landmarks[eye[1]];
  const p3 = landmarks[eye[2]];
  const p4 = landmarks[eye[3]];
  const p5 = landmarks[eye[4]];
  const p6 = landmarks[eye[5]];

  const vertical = Math.hypot(p2.x - p6.x, p2.y - p6.y);
  const horizontal = Math.hypot(p1.x - p4.x, p1.y - p4.y);
  return horizontal>0 ? vertical/horizontal : 0;
}

function computeMetrics(){
  const now = nowSec();
  purgeOld(now);
  const framesLen = framesBuffer.length;
  if(framesLen===0) return {blinksPerMin:0, perclos:0, avgEar:0, avgBlinkMs:0, blinksInWindow:0};
  const perclos = 100.0 * framesBuffer.filter(f=>f.closed).length / framesLen;
  const avgEar = framesBuffer.reduce((s,f)=>s+f.ear,0)/framesLen;
  const blinksInWindow = blinkEvents.length;
  const avgBlinkMs = blinksInWindow>0 ? blinkEvents.reduce((s,b)=>s+b.dur,0)/blinksInWindow : 0;
  const blinksPerMin = blinksInWindow * (60.0 / WINDOW_SECONDS);
  return {blinksPerMin, perclos, avgEar, avgBlinkMs, blinksInWindow};
}

function strainLevel(metrics){
  const bpm = metrics.blinksPerMin;
  const per = metrics.perclos;
  const dur = metrics.avgBlinkMs;
  if(bpm>=12 && per<10 && dur<300) return ['Low','green'];
  if((bpm>=8 && bpm<12) || (per>=10 && per<20) || (dur>=300 && dur<400)) return ['Mild','orange'];
  if(bpm<8 || per>=20 || dur>=400) return ['High','red'];
  return ['Moderate','yellow'];
}

function onResults(results){
  // resize canvas to match video
  if(videoElement.videoWidth && videoElement.videoHeight){
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  }

  canvasCtx.save();
  canvasCtx.clearRect(0,0,canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if(results.multiFaceLandmarks && results.multiFaceLandmarks.length>0){
    const lm = results.multiFaceLandmarks[0];
    // draw points (sparse)
    canvasCtx.fillStyle = 'rgba(0,180,120,0.8)';
    for(let i=0;i<lm.length;i+=4){
      const x = lm[i].x*canvasElement.width;
      const y = lm[i].y*canvasElement.height;
      canvasCtx.fillRect(x-1,y-1,2,2);
    }

    const earLeft = EAR(lm, LEFT_EYE);
    const earRight = EAR(lm, RIGHT_EYE);
    const ear = (earLeft+earRight)/2.0;
    const isClosed = ear < PERCLOS_EAR_THRESH;
    const t = nowSec();
    framesBuffer.push({t, ear, closed:isClosed});

    if(ear < BLINK_EAR_THRESH){
      if(currentBlinkStart===null) currentBlinkStart = t;
    } else {
      if(currentBlinkStart!==null){
        const dur = (t-currentBlinkStart)*1000.0;
        blinkEvents.push({start:currentBlinkStart, end:t, dur});
        blinkCount += 1;
        currentBlinkStart = null;
      }
    }

    const metrics = computeMetrics();
    const [level,labelColor] = strainLevel(metrics);

    // update DOM
    blinksTotalEl.textContent = blinkCount;
    blinksPerMinEl.textContent = metrics.blinksPerMin.toFixed(1);
    perclosEl.textContent = metrics.perclos.toFixed(1);
    strainEl.textContent = level;
    strainEl.style.color = labelColor;

    // overlay a small info box
    const boxW = Math.min(260, canvasElement.width * 0.4);
    canvasCtx.fillStyle = 'rgba(2,6,23,0.6)';
    canvasCtx.fillRect(12,12,boxW,96);
    canvasCtx.fillStyle = '#fff';
    canvasCtx.font = '16px Inter, Arial';
    canvasCtx.fillText(`Blinks: ${blinkCount}`, 22, 36);
    canvasCtx.fillText(`Blinks/min: ${metrics.blinksPerMin.toFixed(1)}`, 22, 56);
    canvasCtx.fillText(`PERCLOS: ${metrics.perclos.toFixed(1)}%`, 22, 76);
    canvasCtx.fillText(`Strain: ${level}`, 22, 96);

  } else {
    canvasCtx.fillStyle = 'rgba(200,0,0,0.08)';
    canvasCtx.fillRect(0,0,canvasElement.width, canvasElement.height);
  }

  canvasCtx.restore();
}

const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
faceMesh.onResults(onResults);

function resetSession(){
  framesBuffer = [];
  blinkEvents = [];
  currentBlinkStart = null;
  blinkCount = 0;
}

let cameraStream = null;
let frameLoopRequest = null;

async function start(){
  startBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = 'Status: starting...';
  reportPanel.setAttribute('aria-hidden', 'true');
  reportActions.style.display = 'none';
  rawReportEl.textContent = '';

  try{
    // request the stream and attach it to the video element
    let stream;
    try{
      stream = await navigator.mediaDevices.getUserMedia({video:{facingMode: 'user'}});
    }catch(err){
      console.warn('getUserMedia with facingMode failed, trying default video constraint', err);
      stream = await navigator.mediaDevices.getUserMedia({video:true});
    }
    cameraStream = stream;
    videoElement.srcObject = stream;
    await videoElement.play();

    running = true;
    sessionStart = nowSec();
    resetSession();
    statusEl.textContent = 'Status: running';

    // frame loop: send video frames to faceMesh
    const frameLoop = async () => {
      if(!running) return;
      try{
        await faceMesh.send({image: videoElement});
      }catch(err){
        // faceMesh may throw if frame not ready; log for debugging
        console.error('faceMesh send error', err);
      }
      frameLoopRequest = requestAnimationFrame(frameLoop);
    };
    frameLoop();
  }catch(e){
    console.error('start error', e);
    statusEl.textContent = 'Status: camera error (see console)';
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

function aggregatePerMinute(startSec, endSec){
  const minutes = Math.max(1, Math.ceil((endSec-startSec)/60.0));
  const rows = [];
  for(let i=0;i<minutes;i++){
    const s = startSec + i*60;
    const e = s + 60;
    const frames_i = framesBuffer.filter(f=>f.t>=s && f.t<e);
    const blinks_i = blinkEvents.filter(b=>b.start>=s && b.start<e);
    const frames_count = frames_i.length;
    const blinks_in_min = blinks_i.length;
    const perclos_i = frames_count>0 ? 100.0 * frames_i.filter(f=>f.closed).length / frames_count : 0.0;
    const avg_blink_ms_i = blinks_in_min>0 ? blinks_i.reduce((ss,b)=>ss+b.dur,0)/blinks_in_min : 0.0;
    rows.push({minute:i, start_iso:new Date(s*1000).toISOString(), frames:frames_count, blinks_in_min, perclos:perclos_i, avg_blink_ms:avg_blink_ms_i});
  }
  return rows;
}

function genTxtReport(startSec, endSec, rows, summary){
  let s = 'Eye Strain Session Report\n';
  s += `Start: ${new Date(startSec*1000).toISOString()}\n`;
  s += `End: ${new Date(endSec*1000).toISOString()}\n`;
  s += `Duration (s): ${(endSec-startSec).toFixed(1)}\n\n`;
  s += `Total frames: ${summary.total_frames}\n`;
  s += `Total blinks: ${summary.total_blinks}\n`;
  s += `Blinks/min: ${summary.blinks_per_min.toFixed(2)}\n`;
  s += `Avg blink duration (ms): ${summary.avg_blink_duration_ms.toFixed(1)}\n`;
  s += `PERCLOS (%): ${summary.overall_perclos.toFixed(2)}\n`;
  s += `Avg EAR: ${summary.avg_ear.toFixed(4)}\n`;
  s += `Final strain level: ${summary.final_level}\n\n`;
  s += 'Minute-by-minute:\n';
  s += 'minute_index,start_iso,frames,blinks_in_min,blinks_per_min,perclos,avg_blink_ms\n';
  rows.forEach(r=>{
    s += `${r.minute},${r.start_iso},${r.frames},${r.blinks_in_min},${r.blinks_in_min.toFixed(1)},${r.perclos.toFixed(1)},${r.avg_blink_ms.toFixed(1)}\n`;
  });
  return s;
}

function genCsv(rows){
  let s = 'minute_index,start_iso,frames,blinks_in_min,blinks_per_min,perclos,avg_blink_ms\n';
  rows.forEach(r=>{
    s += `${r.minute},${r.start_iso},${r.frames},${r.blinks_in_min},${r.blinks_in_min.toFixed(1)},${r.perclos.toFixed(1)},${r.avg_blink_ms.toFixed(1)}\n`;
  });
  return s;
}

function showReport(startSec, endSec){
  sessionEnd = endSec;
  const rows = aggregatePerMinute(startSec, endSec);
  const total_frames = framesBuffer.length;
  const total_blinks = blinkEvents.length;
  const avg_blink_duration_ms = total_blinks>0 ? blinkEvents.reduce((s,b)=>s+b.dur,0)/total_blinks : 0.0;
  const overall_perclos = total_frames>0 ? 100.0 * framesBuffer.filter(f=>f.closed).length / total_frames : 0.0;
  const avg_ear = total_frames>0 ? framesBuffer.reduce((s,f)=>s+f.ear,0)/total_frames : 0.0;
  const blinks_per_min = total_blinks / Math.max(1, (endSec-startSec)/60.0);
  const duration_sec = Math.round(endSec - startSec);

  const final_metrics = {blinks_per_min:blinks_per_min, perclos:overall_perclos, avg_ear:avg_ear, avg_blink_duration_ms:avg_blink_duration_ms};
  const [final_level, final_color] = strainLevel(final_metrics);

  const summary = {
    total_frames, total_blinks, avg_blink_duration_ms, overall_perclos, avg_ear, blinks_per_min, final_level
  };

  const txt = genTxtReport(startSec, endSec, rows, {total_frames, total_blinks, avg_blink_duration_ms, overall_perclos, avg_ear, blinks_per_min, final_level});
  const csv = genCsv(rows);

  // Simplified report display
  const reportHtml = `
    <div style="padding: 16px; border-radius: 12px; background: rgba(100, 150, 255, 0.08); margin-bottom: 16px;">
      <div style="font-size: 1.5rem; font-weight: 600; color: ${final_color}; margin-bottom: 8px;">${final_level} Strain Level</div>
      <div style="font-size: 0.95rem; color: #b4c4e8; margin-bottom: 12px;">Session Duration: ${Math.floor(duration_sec/60)}m ${duration_sec%60}s</div>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
        <div style="padding: 8px; background: rgba(0,0,0,0.2); border-radius: 8px;">
          <div style="font-size: 0.8rem; color: #8898c8;">Blinks</div>
          <div style="font-size: 1.3rem; font-weight: 600; color: #fff;">${total_blinks}</div>
        </div>
        <div style="padding: 8px; background: rgba(0,0,0,0.2); border-radius: 8px;">
          <div style="font-size: 0.8rem; color: #8898c8;">Blinks/min</div>
          <div style="font-size: 1.3rem; font-weight: 600; color: #60a5fa;">${blinks_per_min.toFixed(1)}</div>
        </div>
        <div style="padding: 8px; background: rgba(0,0,0,0.2); border-radius: 8px;">
          <div style="font-size: 0.8rem; color: #8898c8;">Eye Closure (%)</div>
          <div style="font-size: 1.3rem; font-weight: 600; color: #06b6d4;">${overall_perclos.toFixed(1)}%</div>
        </div>
        <div style="padding: 8px; background: rgba(0,0,0,0.2); border-radius: 8px;">
          <div style="font-size: 0.8rem; color: #8898c8;">Avg Blink (ms)</div>
          <div style="font-size: 1.3rem; font-weight: 600; color: #fff;">${avg_blink_duration_ms.toFixed(0)}</div>
        </div>
      </div>

      <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(100,120,160,0.2); font-size: 0.85rem; color: #8898c8;">
        <strong>What this means:</strong><br>
        ${final_level === 'Low' ? '✅ Your eyes are healthy. Keep up good habits!' : ''}
        ${final_level === 'Mild' ? '⚠️ Some eye strain detected. Take breaks every 20 minutes.' : ''}
        ${final_level === 'Moderate' ? '⚠️ Moderate strain. Follow the 20-20-20 rule: every 20 min, look 20 ft away for 20 sec.' : ''}
        ${final_level === 'High' ? '🔴 High eye strain! Take a break now. Rest your eyes and follow 20-20-20 rule.' : ''}
      </div>
    </div>
  `;

  reportSummary.innerHTML = reportHtml;
  rawReportEl.textContent = txt;
  reportActions.style.display = 'flex';
  reportPanel.setAttribute('aria-hidden', 'false');

  // Set up download functionality for TXT
  downloadTxtBtn.onclick = function(){
    const txtBlob = new Blob([txt], {type:'text/plain'});
    const url = URL.createObjectURL(txtBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session_${new Date(startSec*1000).toISOString().replace(/[:.]/g,'')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Set up download functionality for CSV
  downloadCsvBtn.onclick = function(){
    const csvBlob = new Blob([csv], {type:'text/csv'});
    const url = URL.createObjectURL(csvBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session_${new Date(startSec*1000).toISOString().replace(/[:.]/g,'')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // copy to clipboard
  if(copyReportBtn){
    copyReportBtn.onclick = async ()=>{ 
      await navigator.clipboard.writeText(txt);
      alert('Report copied to clipboard'); 
    };
  }

  // Remove chart for simpler UI
  const chartContainer = document.getElementById('chart');
  if(chartContainer) chartContainer.style.display = 'none';
}

function stop(){
  // stop frame loop
  if(frameLoopRequest) cancelAnimationFrame(frameLoopRequest);
  frameLoopRequest = null;
  // stop media stream
  if(cameraStream){
    cameraStream.getTracks().forEach(t=>t.stop());
    cameraStream = null;
  }
  running = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = 'Status: stopped';
  const end = nowSec();
  showReport(sessionStart, end);
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);

