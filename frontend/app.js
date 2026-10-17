/**
 * T225 Railway Safety Surveillance Console - Main Frontend Application
 */

const API_BASE = window.location.origin;

// Application State
const state = {
  activeTab: 'tab-surveillance',
  barrierState: 'UNKNOWN',
  processingStatus: 'IDLE',
  feedMode: 'raw', // 'raw' or 'detection'
  latestOutputVideo: null,
  zones: [],
  selectedZoneIndex: 0,
  activeNewPoints: [],
  availableVideos: [],
  selectedVideoPath: '',
  canvasFrameImg: null,
  pollInterval: null,
  liveFeedInterval: null,
  lastCompletedFrame: null
};

// DOM Elements
const elements = {
  tabs: document.querySelectorAll('.tab-btn'),
  tabPanels: document.querySelectorAll('.tab-panel'),
  teleStatus: document.getElementById('tele-status'),
  teleHealth: document.getElementById('tele-health'),
  barrierBtns: document.querySelectorAll('.barrier-btn'),
  
  // Metrics
  metricActiveAlerts: document.getElementById('metric-active-alerts'),
  metricIntrusions: document.getElementById('metric-intrusions'),
  metricCrossings: document.getElementById('metric-crossings'),
  metricBarrierNote: document.getElementById('metric-barrier-note'),
  metricDetections: document.getElementById('metric-detections'),
  badgeEventCount: document.getElementById('badge-event-count'),
  badgeEvidenceCount: document.getElementById('badge-evidence-count'),

  // Surveillance Viewport
  videoSelector: document.getElementById('video-selector'),
  videoUploadInput: document.getElementById('video-upload-input'),
  btnBrowseUpload: document.getElementById('btn-browse-upload'),
  surveillancePlayer: document.getElementById('surveillance-player'),
  liveProcessingFeed: document.getElementById('live-processing-feed'),
  liveFeedBadge: document.getElementById('live-feed-badge'),
  videoEmptyState: document.getElementById('video-empty-state'),
  btnFeedRaw: document.getElementById('btn-feed-raw'),
  btnFeedDetection: document.getElementById('btn-feed-detection'),
  feedResolutionBadge: document.getElementById('feed-resolution-badge'),
  feedFpsBadge: document.getElementById('feed-fps-badge'),
  btnDlAnnotatedVideo: document.getElementById('btn-dl-annotated-video'),

  // Controls
  modeSelector: document.getElementById('mode-selector'),
  confSlider: document.getElementById('conf-slider'),
  confValDisplay: document.getElementById('conf-val-display'),
  checkEvidence: document.getElementById('check-evidence'),
  btnStartProcess: document.getElementById('btn-start-process'),
  btnStopProcess: document.getElementById('btn-stop-process'),
  progressText: document.getElementById('progress-text'),
  progressSpeed: document.getElementById('progress-speed'),
  progressFill: document.getElementById('progress-fill'),
  statusAlertBox: document.getElementById('status-alert-box'),
  btnDlCsv: document.getElementById('btn-dl-csv'),
  btnDlEventsCsv: document.getElementById('btn-dl-events-csv'),

  // Zone Studio
  zoneCanvas: document.getElementById('zone-canvas'),
  btnFetchFrame: document.getElementById('btn-fetch-frame'),
  btnClearCurrentZone: document.getElementById('btn-clear-current-zone'),
  zoneListContainer: document.getElementById('zone-list-container'),
  zoneNameInput: document.getElementById('zone-name-input'),
  zoneTypeSelect: document.getElementById('zone-type-select'),
  zoneDemoCheckbox: document.getElementById('zone-demo-checkbox'),
  zoneColorInput: document.getElementById('zone-color-input'),
  btnAddNewZone: document.getElementById('btn-add-new-zone'),
  btnSaveZones: document.getElementById('btn-save-zones'),

  // Events & Evidence
  eventsTableBody: document.getElementById('events-table-body'),
  filterSeverity: document.getElementById('filter-severity'),
  filterType: document.getElementById('filter-type'),
  btnRefreshEvents: document.getElementById('btn-refresh-events'),
  evidenceGrid: document.getElementById('evidence-grid'),
  btnRefreshEvidence: document.getElementById('btn-refresh-evidence'),
  masterDetectionBanner: document.getElementById('master-detection-banner'),
  masterDetectionName: document.getElementById('master-detection-name'),
  btnPlayMaster: document.getElementById('btn-play-master'),
  btnDlMaster: document.getElementById('btn-dl-master'),

  // Diagnostics
  diagModelPath: document.getElementById('diag-model-path'),
  diagModelExists: document.getElementById('diag-model-exists'),
  diagModelSize: document.getElementById('diag-model-size'),
  diagModelParams: document.getElementById('diag-model-params'),
  diagDevice: document.getElementById('diag-device'),
  diagCuda: document.getElementById('diag-cuda'),
  diagArch: document.getElementById('diag-arch'),
  diagClassesContainer: document.getElementById('diag-classes-container'),

  // Modal
  evidenceModal: document.getElementById('evidence-modal'),
  modalTitle: document.getElementById('modal-title'),
  modalVideoPlayer: document.getElementById('modal-video-player'),
  modalMetadata: document.getElementById('modal-metadata'),
  btnDownloadClip: document.getElementById('btn-download-clip'),
  btnCloseModal: document.getElementById('btn-close-modal'),
  btnDismissModal: document.getElementById('btn-dismiss-modal')
};

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await loadInitialData();
  startStatusPolling();
  await checkLatestOutputVideo();
  if (state.latestOutputVideo) {
    loadVideoPreview(null, 'detection');
  }
});

async function loadInitialData() {
  await Promise.all([
    fetchVideos(),
    fetchZones(),
    fetchEvents(),
    fetchEvidence(),
    fetchDiagnostics()
  ]);
}

function setupEventListeners() {
  // Tab Switching
  elements.tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.tabs.forEach(b => b.classList.remove('active'));
      elements.tabPanels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');
      state.activeTab = tabId;

      if (tabId === 'tab-zones') {
        renderZoneCanvas();
      }
    });
  });

  // Video Feed Mode Switcher (Raw vs Detection)
  elements.btnFeedRaw.addEventListener('click', () => {
    loadVideoPreview(state.selectedVideoPath, 'raw');
  });

  elements.btnFeedDetection.addEventListener('click', () => {
    loadVideoPreview(null, 'detection');
  });

  // Master Detection Video from Evidence Vault
  if (elements.btnPlayMaster) {
    elements.btnPlayMaster.addEventListener('click', () => {
      // Switch to surveillance tab and play detection video
      elements.tabs.forEach(b => b.classList.remove('active'));
      elements.tabPanels.forEach(p => p.classList.remove('active'));
      const survTabBtn = document.querySelector('[data-tab="tab-surveillance"]');
      if (survTabBtn) survTabBtn.classList.add('active');
      document.getElementById('tab-surveillance').classList.add('active');
      state.activeTab = 'tab-surveillance';
      loadVideoPreview(null, 'detection');
    });
  }

  // Player metadata updates (resolution, duration)
  elements.surveillancePlayer.addEventListener('loadedmetadata', updatePlayerMetadata);
  elements.surveillancePlayer.addEventListener('error', () => {
    console.warn('Surveillance Player Error Code:', elements.surveillancePlayer.error ? elements.surveillancePlayer.error.code : 'unknown');
  });

  // Barrier Toggle
  elements.barrierBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const targetState = btn.getAttribute('data-state');
      await updateBarrierState(targetState);
    });
  });

  // Confidence Slider
  elements.confSlider.addEventListener('input', (e) => {
    elements.confValDisplay.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Video Selection from dropdown
  elements.videoSelector.addEventListener('change', (e) => {
    state.selectedVideoPath = e.target.value;
    if (state.selectedVideoPath) {
      loadVideoPreview(state.selectedVideoPath, 'raw');
      fetchSnapshotFrame();
    }
  });

  // Video Upload
  elements.btnBrowseUpload.addEventListener('click', () => elements.videoUploadInput.click());
  elements.videoUploadInput.addEventListener('change', handleVideoUpload);

  // Processing Actions
  elements.btnStartProcess.addEventListener('click', startVideoProcessing);
  elements.btnStopProcess.addEventListener('click', stopVideoProcessing);

  // CSV Downloads
  elements.btnDlCsv.addEventListener('click', () => window.open(`${API_BASE}/api/outputs/csv`, '_blank'));
  elements.btnDlEventsCsv.addEventListener('click', () => window.open(`${API_BASE}/api/outputs/events-csv`, '_blank'));

  // Zone Studio
  elements.zoneCanvas.addEventListener('click', handleCanvasClick);
  elements.btnFetchFrame.addEventListener('click', fetchSnapshotFrame);
  elements.btnClearCurrentZone.addEventListener('click', clearActiveZonePoints);
  elements.btnAddNewZone.addEventListener('click', createNewZone);
  elements.btnSaveZones.addEventListener('click', saveZonesToBackend);
  elements.zoneNameInput.addEventListener('input', updateActiveZoneForm);
  elements.zoneTypeSelect.addEventListener('change', updateActiveZoneForm);
  elements.zoneDemoCheckbox.addEventListener('change', updateActiveZoneForm);
  elements.zoneColorInput.addEventListener('input', updateActiveZoneForm);

  // Filters & Refresh
  elements.filterSeverity.addEventListener('change', fetchEvents);
  elements.filterType.addEventListener('change', fetchEvents);
  elements.btnRefreshEvents.addEventListener('click', fetchEvents);
  elements.btnRefreshEvidence.addEventListener('click', fetchEvidence);

  // Modal Close
  elements.btnCloseModal.addEventListener('click', closeModal);
  elements.btnDismissModal.addEventListener('click', closeModal);
  elements.evidenceModal.querySelector('.modal-backdrop').addEventListener('click', closeModal);
}

// -------------------------------------------------------------
// Video Loading & Preview Management
// -------------------------------------------------------------

function loadVideoPreview(videoPath, mode = 'raw') {
  state.feedMode = mode;

  // Hide live processing elements if active
  if (elements.liveProcessingFeed) elements.liveProcessingFeed.classList.add('hidden');
  if (elements.liveFeedBadge) elements.liveFeedBadge.classList.add('hidden');
  elements.surveillancePlayer.classList.remove('hidden');

  if (mode === 'raw') {
    elements.btnFeedRaw.classList.add('active');
    elements.btnFeedDetection.classList.remove('active');

    const target = videoPath || state.selectedVideoPath;
    if (!target) {
      elements.videoEmptyState.classList.remove('hidden');
      return;
    }

    elements.videoEmptyState.classList.add('hidden');
    const matched = state.availableVideos.find(v => v.path === target || v.name === target);
    const filename = target.split(/[\\/]/).pop();
    const streamUrl = (matched && matched.stream_url) 
      ? `${API_BASE}${matched.stream_url}` 
      : `${API_BASE}/api/videos/stream?name=${encodeURIComponent(filename)}`;

    // Set poster snapshot preview from frame 0
    elements.surveillancePlayer.poster = `${API_BASE}/api/video-frame?video_path=${encodeURIComponent(target)}&frame_number=0`;

    if (elements.surveillancePlayer.getAttribute('data-loaded-src') !== streamUrl) {
      elements.surveillancePlayer.src = streamUrl;
      elements.surveillancePlayer.setAttribute('data-loaded-src', streamUrl);
      elements.surveillancePlayer.load();
    }
  } else if (mode === 'detection') {
    elements.btnFeedRaw.classList.remove('active');
    elements.btnFeedDetection.classList.add('active');

    if (!state.latestOutputVideo) {
      showAlert('No annotated detection video found yet. Run analysis to generate full detection recording.', 'info');
      return;
    }

    elements.videoEmptyState.classList.add('hidden');
    elements.surveillancePlayer.removeAttribute('poster');
    const detectionUrl = `${API_BASE}/api/outputs/video/${encodeURIComponent(state.latestOutputVideo)}`;
    if (elements.surveillancePlayer.getAttribute('data-loaded-src') !== detectionUrl) {
      elements.surveillancePlayer.src = detectionUrl;
      elements.surveillancePlayer.setAttribute('data-loaded-src', detectionUrl);
      elements.surveillancePlayer.load();
      elements.surveillancePlayer.play().catch(e => console.log('Autoplay notice:', e));
    }
  }
}

function updatePlayerMetadata() {
  const p = elements.surveillancePlayer;
  if (p.videoWidth && p.videoHeight) {
    elements.feedResolutionBadge.textContent = `Resolution: ${p.videoWidth}x${p.videoHeight}`;
  }
  if (p.duration && !isNaN(p.duration)) {
    const mins = Math.floor(p.duration / 60);
    const secs = Math.floor(p.duration % 60);
    elements.feedFpsBadge.textContent = `Duration: ${mins}:${secs.toString().padStart(2, '0')}`;
  }
}

async function checkLatestOutputVideo() {
  try {
    const res = await fetch(`${API_BASE}/api/outputs/latest-video`);
    if (res.ok) {
      const data = await res.json();
      if (data.filename) {
        state.latestOutputVideo = data.filename;
        if (elements.btnDlAnnotatedVideo) {
          elements.btnDlAnnotatedVideo.href = `${API_BASE}/api/outputs/video/${data.filename}`;
          elements.btnDlAnnotatedVideo.classList.remove('hidden');
        }
        if (elements.masterDetectionBanner) {
          elements.masterDetectionBanner.classList.remove('hidden');
          elements.masterDetectionName.textContent = data.filename;
          elements.btnDlMaster.href = `${API_BASE}/api/outputs/video/${data.filename}`;
        }
      }
    }
  } catch (e) {
    console.warn('Check latest output video error:', e);
  }
}

// -------------------------------------------------------------
// Live Status Polling
// -------------------------------------------------------------

function startStatusPolling() {
  if (state.pollInterval) clearInterval(state.pollInterval);
  state.pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/processing/status`);
      if (!res.ok) return;
      const data = await res.json();
      updateProcessingUI(data);
    } catch (e) {
      console.warn('Status poll error:', e);
    }
  }, 1000);
}

function updateProcessingUI(data) {
  state.processingStatus = data.status;
  elements.teleStatus.textContent = data.status;

  if (data.status === 'PROCESSING') {
    elements.teleStatus.className = 'tele-value badge-alert';
    elements.btnStartProcess.disabled = true;
    elements.btnStopProcess.disabled = false;
  } else {
    elements.teleStatus.className = 'tele-value badge-neutral';
    elements.btnStartProcess.disabled = false;
    elements.btnStopProcess.disabled = true;
  }

  // Progress Bar
  elements.progressFill.style.width = `${data.progress_percent}%`;
  elements.progressText.textContent = `Inference Progress: ${data.progress_percent}% (Frame ${data.current_frame} / ${data.total_frames})`;
  elements.progressSpeed.textContent = `Speed: ${data.fps} fps`;

  // Metrics
  elements.metricActiveAlerts.textContent = data.active_alerts_count;
  elements.metricDetections.textContent = data.total_detections_count;

  if (data.status === 'COMPLETED' && state.lastCompletedFrame !== data.total_frames) {
    state.lastCompletedFrame = data.total_frames;
    
    // Stop live frame feed
    if (state.liveFeedInterval) {
      clearInterval(state.liveFeedInterval);
      state.liveFeedInterval = null;
    }
    if (elements.liveProcessingFeed) elements.liveProcessingFeed.classList.add('hidden');
    if (elements.liveFeedBadge) elements.liveFeedBadge.classList.add('hidden');
    elements.surveillancePlayer.classList.remove('hidden');

    if (data.output_video_path) {
      state.latestOutputVideo = data.output_video_path;
      if (elements.btnDlAnnotatedVideo) {
        elements.btnDlAnnotatedVideo.href = `${API_BASE}/api/outputs/video/${data.output_video_path}`;
        elements.btnDlAnnotatedVideo.classList.remove('hidden');
      }
      if (elements.masterDetectionBanner) {
        elements.masterDetectionBanner.classList.remove('hidden');
        elements.masterDetectionName.textContent = data.output_video_path;
        elements.btnDlMaster.href = `${API_BASE}/api/outputs/video/${data.output_video_path}`;
      }

      // Automatically switch player to the full detection video and play it
      loadVideoPreview(null, 'detection');
      showAlert('Processing complete! Displaying Full Annotated Detection Video with bounding boxes.', 'success');
    } else {
      showAlert('Processing complete! Output CSVs and evidence generated.', 'success');
    }

    fetchEvents();
    fetchEvidence();
  } else if (data.status === 'ERROR') {
    if (state.liveFeedInterval) {
      clearInterval(state.liveFeedInterval);
      state.liveFeedInterval = null;
    }
    if (elements.liveProcessingFeed) elements.liveProcessingFeed.classList.add('hidden');
    if (elements.liveFeedBadge) elements.liveFeedBadge.classList.add('hidden');
    elements.surveillancePlayer.classList.remove('hidden');
    showAlert(`Processing error: ${data.error_message}`, 'error');
  }
}

// -------------------------------------------------------------
// Barrier State Management
// -------------------------------------------------------------

async function updateBarrierState(targetState) {
  try {
    const res = await fetch(`${API_BASE}/api/barrier-state?state=${targetState}`, { method: 'PUT' });
    if (!res.ok) throw new Error('Failed to update barrier state');
    state.barrierState = targetState;
    
    elements.barrierBtns.forEach(btn => {
      if (btn.getAttribute('data-state') === targetState) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    elements.metricBarrierNote.textContent = `Barrier: ${targetState}`;
  } catch (err) {
    showAlert(err.message, 'error');
  }
}

// -------------------------------------------------------------
// Video Pipeline Handlers
// -------------------------------------------------------------

async function fetchVideos() {
  try {
    const res = await fetch(`${API_BASE}/api/videos`);
    const data = await res.json();
    state.availableVideos = data.videos || [];
    
    elements.videoSelector.innerHTML = '';
    if (state.availableVideos.length === 0) {
      elements.videoSelector.innerHTML = '<option value="">No videos found</option>';
      return;
    }

    state.availableVideos.forEach((v) => {
      const opt = document.createElement('option');
      opt.value = v.path;
      opt.textContent = `${v.name} (${v.size_mb} MB) [${v.source}]`;
      elements.videoSelector.appendChild(opt);
    });

    if (!state.selectedVideoPath) {
      state.selectedVideoPath = state.availableVideos[0].path;
    }
    elements.videoSelector.value = state.selectedVideoPath;
    loadVideoPreview(state.selectedVideoPath, 'raw');
    fetchSnapshotFrame();
  } catch (e) {
    console.error('Error fetching videos:', e);
  }
}

async function handleVideoUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  showAlert(`Uploading video ${file.name}...`, 'info');
  try {
    const res = await fetch(`${API_BASE}/api/videos/upload`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    const data = await res.json();
    showAlert(`Successfully uploaded ${data.filename}! Video preview ready.`, 'success');
    state.selectedVideoPath = data.path;
    await fetchVideos();
    elements.videoSelector.value = data.path;
    loadVideoPreview(data.path, 'raw');
    fetchSnapshotFrame();
  } catch (err) {
    showAlert(`Upload error: ${err.message}`, 'error');
  }
}

async function startVideoProcessing() {
  if (!state.selectedVideoPath) {
    showAlert('Please select or upload a video file first.', 'error');
    return;
  }

  const payload = {
    video_path: state.selectedVideoPath,
    conf_threshold: parseFloat(elements.confSlider.value),
    barrier_state: state.barrierState,
    detection_mode: elements.modeSelector.value,
    generate_evidence: elements.checkEvidence.checked
  };

  try {
    const res = await fetch(`${API_BASE}/api/processing/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start processing');
    }
    showAlert('Inference pipeline launched in background.', 'info');

    // Show live inference feed
    elements.videoEmptyState.classList.add('hidden');
    elements.surveillancePlayer.classList.add('hidden');
    elements.liveProcessingFeed.classList.remove('hidden');
    elements.liveFeedBadge.classList.remove('hidden');

    if (state.liveFeedInterval) clearInterval(state.liveFeedInterval);
    state.liveFeedInterval = setInterval(() => {
      if (state.processingStatus === 'PROCESSING') {
        elements.liveProcessingFeed.src = `${API_BASE}/api/processing/live-frame?t=${Date.now()}`;
      } else {
        clearInterval(state.liveFeedInterval);
        state.liveFeedInterval = null;
      }
    }, 250);

  } catch (err) {
    showAlert(err.message, 'error');
  }
}

async function stopVideoProcessing() {
  try {
    await fetch(`${API_BASE}/api/processing/stop`, { method: 'POST' });
    if (state.liveFeedInterval) {
      clearInterval(state.liveFeedInterval);
      state.liveFeedInterval = null;
    }
    showAlert('Stop signal sent to video processor.', 'info');
  } catch (err) {
    showAlert(err.message, 'error');
  }
}

// -------------------------------------------------------------
// Interactive Zone Configuration Studio
// -------------------------------------------------------------

async function fetchZones() {
  try {
    const res = await fetch(`${API_BASE}/api/zones`);
    state.zones = await res.json();
    renderZoneList();
    loadZoneIntoForm(0);
  } catch (e) {
    console.error('Error fetching zones:', e);
  }
}

function renderZoneList() {
  elements.zoneListContainer.innerHTML = '';
  state.zones.forEach((z, idx) => {
    const item = document.createElement('div');
    item.className = `zone-item ${idx === state.selectedZoneIndex ? 'active' : ''}`;
    item.innerHTML = `
      <div>
        <strong style="color: ${z.color}">● ${z.name}</strong>
        <div style="font-size: 0.72rem; color: #94a3b8;">${z.zone_type} (${z.points.length} pts)</div>
      </div>
      ${z.is_demo ? '<span class="zone-badge-demo">Demo</span>' : ''}
    `;
    item.addEventListener('click', () => {
      state.selectedZoneIndex = idx;
      renderZoneList();
      loadZoneIntoForm(idx);
      renderZoneCanvas();
    });
    elements.zoneListContainer.appendChild(item);
  });
}

function loadZoneIntoForm(idx) {
  if (!state.zones[idx]) return;
  const z = state.zones[idx];
  elements.zoneNameInput.value = z.name;
  elements.zoneTypeSelect.value = z.zone_type;
  elements.zoneDemoCheckbox.checked = z.is_demo;
  elements.zoneColorInput.value = z.color;
}

function updateActiveZoneForm() {
  if (!state.zones[state.selectedZoneIndex]) return;
  const z = state.zones[state.selectedZoneIndex];
  z.name = elements.zoneNameInput.value;
  z.zone_type = elements.zoneTypeSelect.value;
  z.is_demo = elements.zoneDemoCheckbox.checked;
  z.color = elements.zoneColorInput.value;
  renderZoneList();
  renderZoneCanvas();
}

function createNewZone() {
  const newZone = {
    id: `zone-${Date.now()}`,
    name: `New Restricted Zone ${state.zones.length + 1}`,
    zone_type: 'TRACK_INTRUSION',
    points: [],
    is_demo: false,
    color: '#06b6d4'
  };
  state.zones.push(newZone);
  state.selectedZoneIndex = state.zones.length - 1;
  renderZoneList();
  loadZoneIntoForm(state.selectedZoneIndex);
  renderZoneCanvas();
}

function clearActiveZonePoints() {
  if (!state.zones[state.selectedZoneIndex]) return;
  state.zones[state.selectedZoneIndex].points = [];
  renderZoneCanvas();
}

async function saveZonesToBackend() {
  try {
    for (const z of state.zones) {
      if (z.points.length < 3) {
        throw new Error(`Zone '${z.name}' must have at least 3 polygon points.`);
      }
    }
    const res = await fetch(`${API_BASE}/api/zones`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zones: state.zones })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Save failed');
    }
    showAlert('Zone configuration saved successfully!', 'success');
  } catch (err) {
    showAlert(err.message, 'error');
  }
}

async function fetchSnapshotFrame() {
  try {
    const videoParam = state.selectedVideoPath ? `?video_path=${encodeURIComponent(state.selectedVideoPath)}` : '';
    const imgUrl = `${API_BASE}/api/video-frame${videoParam}&frame_number=60`;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      state.canvasFrameImg = img;
      renderZoneCanvas();
    };
    img.src = imgUrl;
  } catch (e) {
    console.error('Frame fetch error:', e);
  }
}

function handleCanvasClick(e) {
  const rect = elements.zoneCanvas.getBoundingClientRect();
  const scaleX = elements.zoneCanvas.width / rect.width;
  const scaleY = elements.zoneCanvas.height / rect.height;

  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  // Normalized coordinates (0.0 to 1.0)
  const normX = Math.max(0.0, Math.min(1.0, clickX / elements.zoneCanvas.width));
  const normY = Math.max(0.0, Math.min(1.0, clickY / elements.zoneCanvas.height));

  if (!state.zones[state.selectedZoneIndex]) {
    createNewZone();
  }

  state.zones[state.selectedZoneIndex].points.push({ x: normX, y: normY });
  renderZoneCanvas();
}

function renderZoneCanvas() {
  const canvas = elements.zoneCanvas;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // 1. Draw Background Frame
  if (state.canvasFrameImg) {
    ctx.drawImage(state.canvasFrameImg, 0, 0, w, h);
  } else {
    // Dark grid background
    ctx.fillStyle = '#0a0e17';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
  }

  // 2. Draw all configured zones
  state.zones.forEach((z, idx) => {
    if (!z.points || z.points.length === 0) return;
    const isSelected = (idx === state.selectedZoneIndex);

    ctx.beginPath();
    ctx.moveTo(z.points[0].x * w, z.points[0].y * h);
    for (let i = 1; i < z.points.length; i++) {
      ctx.lineTo(z.points[i].x * w, z.points[i].y * h);
    }
    if (z.points.length >= 3) {
      ctx.closePath();
      ctx.fillStyle = `${z.color}33`; // 20% opacity fill
      ctx.fill();
    }

    ctx.strokeStyle = z.color;
    ctx.lineWidth = isSelected ? 3 : 1.5;
    if (z.is_demo) {
      ctx.setLineDash([6, 6]);
    } else {
      ctx.setLineDash([]);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw vertex dots
    z.points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, isSelected ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? '#ffffff' : z.color;
      ctx.fill();
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Label Zone Name
    if (z.points.length > 0) {
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 13px JetBrains Mono';
      ctx.fillText(`${z.name} [${z.zone_type}]`, z.points[0].x * w + 8, z.points[0].y * h - 8);
    }
  });
}

// -------------------------------------------------------------
// Events & Violation Log
// -------------------------------------------------------------

async function fetchEvents() {
  try {
    const sev = elements.filterSeverity.value;
    const type = elements.filterType.value;
    let url = `${API_BASE}/api/events?limit=100`;
    if (sev) url += `&severity=${sev}`;
    if (type) url += `&event_type=${type}`;

    const res = await fetch(url);
    const data = await res.json();
    const events = data.events || [];

    renderEventsTable(events);
    updateEventMetrics(events);
  } catch (e) {
    console.error('Error fetching events:', e);
  }
}

function renderEventsTable(events) {
  elements.eventsTableBody.innerHTML = '';
  if (events.length === 0) {
    elements.eventsTableBody.innerHTML = `
      <tr><td colspan="10" class="text-center text-muted">No events recorded. Start inference to log detections and violations.</td></tr>
    `;
    return;
  }

  events.forEach(ev => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono-text">${ev.event_id}</td>
      <td class="mono-text">${ev.video_timestamp_str}</td>
      <td><strong>${ev.event_type}</strong></td>
      <td>${ev.object_class}</td>
      <td class="mono-text">${ev.tracking_id !== null ? ev.tracking_id : '--'}</td>
      <td class="mono-text">${(ev.confidence * 100).toFixed(1)}%</td>
      <td>${ev.zone_name}</td>
      <td><span class="mono-text">${ev.barrier_state}</span></td>
      <td><span class="badge-sev ${ev.severity}">${ev.severity}</span></td>
      <td>
        ${ev.evidence_path ? 
          `<button class="btn btn-primary btn-sm" onclick="playEvidenceModal('${ev.evidence_path}', '${ev.event_id}', '${ev.event_type}', '${ev.severity}')">▶ View</button>` : 
          '<span class="text-dim">Pending</span>'
        }
      </td>
    `;
    elements.eventsTableBody.appendChild(tr);
  });
}

function updateEventMetrics(events) {
  elements.badgeEventCount.textContent = events.length;
  let intrusions = 0;
  let crossings = 0;

  events.forEach(ev => {
    if (ev.event_type === 'TRACK_INTRUSION') intrusions++;
    if (ev.event_type.includes('CROSSING')) crossings++;
  });

  elements.metricIntrusions.textContent = intrusions;
  elements.metricCrossings.textContent = crossings;
}

// -------------------------------------------------------------
// Evidence Vault
// -------------------------------------------------------------

async function fetchEvidence() {
  try {
    const res = await fetch(`${API_BASE}/api/evidence`);
    const data = await res.json();
    const clips = data.evidence_clips || [];
    elements.badgeEvidenceCount.textContent = clips.length;

    renderEvidenceGrid(clips);
  } catch (e) {
    console.error('Error fetching evidence:', e);
  }
}

function renderEvidenceGrid(clips) {
  elements.evidenceGrid.innerHTML = '';
  if (clips.length === 0) {
    elements.evidenceGrid.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🗄️</div>
        <h3>No Evidence Clips Generated Yet</h3>
        <p>Evidence clips are automatically created when confirmed intrusions or violations meet temporal and cooldown thresholds.</p>
      </div>
    `;
    return;
  }

  clips.forEach(clip => {
    const card = document.createElement('div');
    card.className = 'evidence-card';
    const videoUrl = `${API_BASE}/api/evidence/${clip.filename}`;
    
    card.innerHTML = `
      <div class="evidence-thumb-wrap">
        <video class="evidence-thumb-video" preload="metadata" muted>
          <source src="${videoUrl}" type="video/mp4">
        </video>
      </div>
      <div class="evidence-card-content">
        <div class="evidence-title">${clip.filename}</div>
        <div class="evidence-meta">Size: ${clip.size_kb} KB • H.264 Verified</div>
        <button class="btn btn-secondary btn-full" onclick="playEvidenceModal('${clip.filename}', '${clip.filename}', 'CONFIRMED_EVIDENCE', 'HIGH')">
          ▶ Play Clip
        </button>
      </div>
    `;
    elements.evidenceGrid.appendChild(card);
  });
}

window.playEvidenceModal = function(filename, eventId, eventType, severity) {
  const videoUrl = `${API_BASE}/api/evidence/${filename}`;
  elements.modalTitle.textContent = `Evidence Record: ${eventId}`;
  elements.modalVideoPlayer.src = videoUrl;
  elements.btnDownloadClip.href = videoUrl;
  elements.btnDownloadClip.download = filename;

  elements.modalMetadata.innerHTML = `
    <div style="display: flex; gap: 1rem; font-size: 0.82rem; font-family: var(--font-mono); color: #94a3b8;">
      <span>TYPE: <strong>${eventType}</strong></span>
      <span>SEVERITY: <strong class="badge-sev ${severity}">${severity}</strong></span>
      <span>FORMAT: <strong>H.264 / yuv420p</strong></span>
    </div>
  `;

  elements.evidenceModal.classList.remove('hidden');
  elements.modalVideoPlayer.play().catch(() => {});
};

function closeModal() {
  elements.evidenceModal.classList.add('hidden');
  elements.modalVideoPlayer.pause();
  elements.modalVideoPlayer.src = '';
}

// -------------------------------------------------------------
// Diagnostics
// -------------------------------------------------------------

async function fetchDiagnostics() {
  try {
    const res = await fetch(`${API_BASE}/api/model-info`);
    const data = await res.json();

    elements.diagModelPath.textContent = data.model_path;
    elements.diagModelExists.textContent = data.model_exists ? 'YES (Verified on Disk)' : 'NO';
    elements.diagModelSize.textContent = `${data.file_size_mb} MB`;
    elements.diagModelParams.textContent = data.parameter_count ? data.parameter_count.toLocaleString() : 'N/A';
    elements.diagDevice.textContent = data.device.toUpperCase();
    elements.diagCuda.textContent = data.cuda_available ? 'Available' : 'CPU Execution';
    elements.diagArch.textContent = data.is_custom_checkpoint ? 'YOLO11n (Ultralytics)' : 'Pretrained YOLO';

    elements.diagClassesContainer.innerHTML = '';
    const classNames = data.model_names || {};
    Object.entries(classNames).forEach(([id, name]) => {
      const pill = document.createElement('div');
      pill.className = 'class-pill';
      pill.textContent = `ID ${id}: ${name}`;
      elements.diagClassesContainer.appendChild(pill);
    });
  } catch (e) {
    console.error('Error fetching diagnostics:', e);
  }
}

// -------------------------------------------------------------
// Notification Banner Helper
// -------------------------------------------------------------

function showAlert(message, type = 'info') {
  elements.statusAlertBox.className = `alert-banner ${type}`;
  elements.statusAlertBox.textContent = message;
  elements.statusAlertBox.classList.remove('hidden');

  setTimeout(() => {
    elements.statusAlertBox.classList.add('hidden');
  }, 5000);
}
