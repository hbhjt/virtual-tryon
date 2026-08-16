const $ = (selector) => document.querySelector(selector);
const elements = {
  camera: $("#camera"), canvas: $("#captureCanvas"), photoPreview: $("#photoPreview"),
  emptyCamera: $("#emptyCamera"), startCamera: $("#startCamera"), shutter: $("#shutter"),
  liveBadge: $("#liveBadge"), qualityCard: $("#qualityCard"), qualityRing: $("#qualityRing"),
  qualityValue: $("#qualityValue"), qualityHint: $("#qualityHint"), qualityDetail: $("#qualityDetail"),
  autoCapture: $("#autoCapture"), processing: $("#processing"), processingTitle: $("#processingTitle"),
  processingText: $("#processingText"), aiProgress: $("#aiProgress"), aiProgressFill: $("#aiProgressFill"),
  aiProgressPercent: $("#aiProgressPercent"), garmentList: $("#garmentList"), garmentUpload: $("#garmentUpload"),
  resultPlaceholder: $("#resultPlaceholder"), resultImage: $("#resultImage"), resultMessage: $("#resultMessage"),
  retryButton: $("#retryButton"), downloadButton: $("#downloadButton"), systemStatus: $("#systemStatus"),
  aiMode: $("#aiMode"), aiModeHint: $("#aiModeHint"), toast: $("#toast"),
  scaleCard: $("#scaleCard"), scaleDown: $("#scaleDown"), scaleUp: $("#scaleUp"),
  garmentScaleValue: $("#garmentScaleValue"), scaleTrackFill: $("#scaleTrackFill"), scaleHint: $("#scaleHint"),
  renderModes: [...document.querySelectorAll('input[name="renderMode"]')],
  uploads: [$("#photoUpload"), $("#photoUploadSecondary")],
};

const state = {
  stream: null, selectedGarment: null, analyzing: false, processing: false,
  stableFrames: 0, previousLandmarks: null, analyzeTimer: null, lastPhotoUrl: null,
  garmentScale: 1, lastFastResultId: null, lastResultGarmentId: null,
};

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => elements.toast.classList.remove("show"), 3600);
}

function currentRenderMode() {
  return elements.renderModes.find((input) => input.checked)?.value || "fast";
}

function createAIJobId() {
  if (crypto.randomUUID) return crypto.randomUUID().replaceAll("-", "");
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function setAIProgress(progress, message) {
  const value = Math.max(0, Math.min(100, Number(progress) || 0));
  elements.aiProgressFill.style.width = `${value}%`;
  elements.aiProgressPercent.textContent = `${Math.round(value)}%`;
  if (message) elements.processingText.textContent = message;
}

function startAIProgressPolling(jobId) {
  let stopped = false, timeoutId = null;
  elements.aiProgress.hidden = false;
  setAIProgress(1, "照片正在提交到本地 AI");
  const poll = async () => {
    if (stopped) return;
    try {
      const progress = await apiJson(`/api/tryon/progress/${jobId}`);
      setAIProgress(progress.progress, progress.message);
      if (progress.status === "completed" || progress.status === "error") return;
    } catch (_) { /* The main request will surface a useful error. */ }
    timeoutId = setTimeout(poll, 850);
  };
  poll();
  return () => { stopped = true; clearTimeout(timeoutId); };
}

function updateScaleControl() {
  const minimum = 0.8, maximum = 1.6;
  const aiMode = currentRenderMode() === "ai";
  elements.garmentScaleValue.textContent = `${Math.round(state.garmentScale * 100)}%`;
  elements.scaleTrackFill.style.width = `${((state.garmentScale - minimum) / (maximum - minimum)) * 100}%`;
  elements.scaleCard.classList.toggle("unavailable", aiMode);
  elements.scaleHint.textContent = aiMode ? "仅快速预览支持手动缩放" : "生成后围绕固定中心缩放，每次 5%";
  // Keep the buttons clickable in AI mode so a click can explain why the
  // operation is unavailable instead of appearing to do nothing.
  elements.scaleDown.disabled = state.processing || (!aiMode && state.garmentScale <= minimum);
  elements.scaleUp.disabled = state.processing || (!aiMode && state.garmentScale >= maximum);
}

async function changeGarmentScale(delta) {
  if (state.processing) return;
  if (currentRenderMode() !== "fast") {
    showToast("衣服缩放只用于快速预览，请先选择“快速预览”模式");
    return;
  }
  const next = Math.round((state.garmentScale + delta) * 20) / 20;
  state.garmentScale = Math.max(0.8, Math.min(1.6, next));
  updateScaleControl();
  if (!elements.resultImage.hidden) {
    if (state.lastFastResultId && state.lastResultGarmentId === state.selectedGarment?.id) {
      await resizeGeneratedGarment();
    } else {
      showToast("请先用当前衣服生成一次快速预览，之后即可固定中心缩放");
    }
  }
}

async function resizeGeneratedGarment() {
  const requestedScale = state.garmentScale;
  state.processing = true; updateScaleControl(); elements.processing.hidden = false; elements.aiProgress.hidden = true;
  elements.processingTitle.textContent = "正在调整衣服大小";
  elements.processingText.textContent = "位置和中心已锁定，不会重新识别人物";
  try {
    const form = new FormData();
    form.append("result_id", state.lastFastResultId);
    form.append("garment_scale", state.garmentScale.toFixed(2));
    const result = await apiJson("/api/tryon/scale", { method: "POST", body: form });
    elements.resultImage.src = `${result.image_url}?t=${Date.now()}`;
    elements.downloadButton.href = result.image_url;
    if (Math.abs(result.garment_scale - requestedScale) > 0.001) {
      state.garmentScale = result.garment_scale;
      updateScaleControl();
      showToast(`后台仍是旧版本，目前最多 ${Math.round(result.garment_scale * 100)}%；请重启项目服务`);
    } else {
      showToast(`已锁定中心缩放到 ${Math.round(result.garment_scale * 100)}%，用时 ${(result.elapsed_ms / 1000).toFixed(1)} 秒`);
    }
  } catch (error) {
    if (error.message.includes("失效")) state.lastFastResultId = null;
    showToast(error.message);
  } finally {
    state.processing = false; updateScaleControl(); elements.processing.hidden = true;
  }
}

async function apiJson(url, options) {
  const response = await fetch(url, options);
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* empty body */ }
  if (!response.ok) throw new Error(payload.detail || `请求失败（${response.status}）`);
  return payload;
}

async function checkHealth() {
  try {
    const health = await apiJson("/api/health");
    elements.systemStatus.className = health.pose_model_available ? "system-status ready" : "system-status warning";
    elements.systemStatus.lastElementChild.textContent = health.pose_model_available ? "本地姿态模型已就绪" : "标准站姿模式";
    if (!health.ai?.installed) {
      elements.aiMode.disabled = true;
      elements.aiModeHint.textContent = "AI 环境未安装，快速模式仍可用";
    } else if (health.ai.model_cached) {
      elements.aiModeHint.textContent = "约 3–5 分钟 · 需完整上半身 · 模型已缓存";
    } else {
      elements.aiModeHint.textContent = "首次使用需下载约 5GB 模型 · 需完整上半身";
    }
  } catch (_) {
    elements.systemStatus.className = "system-status warning";
    elements.systemStatus.lastElementChild.textContent = "本地服务异常";
  }
}

function selectGarment(garment) {
  state.selectedGarment = garment;
  document.querySelectorAll(".garment-card").forEach((card) => card.classList.toggle("selected", card.dataset.id === garment.id));
}

async function loadGarments(selectId = null) {
  const data = await apiJson("/api/garments");
  elements.garmentList.innerHTML = "";
  data.items.forEach((garment, index) => {
    const card = document.createElement("button");
    card.type = "button"; card.className = "garment-card"; card.dataset.id = garment.id;
    card.innerHTML = `<img src="${garment.image_url}" alt="${garment.name}"><span>${garment.name}</span>`;
    card.addEventListener("click", () => selectGarment(garment));
    elements.garmentList.appendChild(card);
    if (garment.id === selectId || (!state.selectedGarment && index === 0)) selectGarment(garment);
  });
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return showToast("当前浏览器不支持摄像头，请上传照片");
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    elements.camera.srcObject = state.stream;
    await elements.camera.play();
    elements.camera.hidden = false; elements.photoPreview.hidden = true; elements.emptyCamera.hidden = true;
    elements.liveBadge.hidden = false; elements.qualityCard.hidden = false; elements.shutter.disabled = false;
    scheduleAnalysis(200);
  } catch (_) { showToast("无法打开摄像头，请允许权限或上传照片"); }
}

function drawMirroredFrame(maxSide = null) {
  let width = elements.camera.videoWidth, height = elements.camera.videoHeight;
  if (!width || !height) return null;
  if (maxSide && Math.max(width, height) > maxSide) {
    const scale = maxSide / Math.max(width, height); width = Math.round(width * scale); height = Math.round(height * scale);
  }
  elements.canvas.width = width; elements.canvas.height = height;
  const context = elements.canvas.getContext("2d", { alpha: false });
  context.save(); context.translate(width, 0); context.scale(-1, 1);
  context.drawImage(elements.camera, 0, 0, width, height); context.restore();
  return elements.canvas;
}

const canvasBlob = (canvas, quality = 0.9) => new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));

function landmarkMotion(current) {
  if (!state.previousLandmarks || !current) return 1;
  const names = Object.keys(current);
  return names.reduce((sum, name) => {
    const previous = state.previousLandmarks[name];
    return sum + (previous ? Math.hypot(current[name].x - previous.x, current[name].y - previous.y) : 1);
  }, 0) / Math.max(names.length, 1);
}

function updateQuality(data) {
  const metrics = data.metrics, score = Math.round(metrics.score * 100);
  elements.qualityRing.style.setProperty("--score", `${score * 3.6}deg`);
  elements.qualityValue.textContent = score; elements.qualityHint.textContent = metrics.guidance;
  elements.qualityDetail.textContent = `清晰度 ${Math.round(metrics.sharpness * 100)} · 光线 ${Math.round(metrics.lighting * 100)}`;
  const landmarks = data.pose.detected ? data.pose.landmarks : null;
  state.stableFrames = metrics.ready && landmarkMotion(landmarks) < 0.022 ? state.stableFrames + 1 : 0;
  state.previousLandmarks = landmarks;
  if (state.stableFrames > 0) elements.qualityDetail.textContent = `请保持不动 · ${Math.min(state.stableFrames, 3)}/3`;
  if (state.stableFrames >= 3 && elements.autoCapture.checked && !state.processing) { state.stableFrames = 0; captureFromCamera(); }
}

function scheduleAnalysis(delay = 420) {
  clearTimeout(state.analyzeTimer);
  if (state.stream && !state.processing) state.analyzeTimer = setTimeout(analyzeFrame, delay);
}

async function analyzeFrame() {
  if (state.analyzing || state.processing || !state.stream) return scheduleAnalysis();
  const canvas = drawMirroredFrame(520); if (!canvas) return scheduleAnalysis();
  state.analyzing = true;
  try {
    const form = new FormData(); form.append("image", await canvasBlob(canvas, 0.76), "preview.jpg");
    updateQuality(await apiJson("/api/analyze", { method: "POST", body: form }));
  } catch (_) {
    elements.qualityHint.textContent = "姿态分析暂时不可用"; elements.qualityDetail.textContent = "仍可点击拍摄按钮";
  } finally { state.analyzing = false; scheduleAnalysis(); }
}

function displayPhoto(url) {
  if (state.lastPhotoUrl?.startsWith("blob:")) URL.revokeObjectURL(state.lastPhotoUrl);
  state.lastPhotoUrl = url; elements.photoPreview.src = url; elements.photoPreview.hidden = false;
  elements.camera.hidden = true; elements.liveBadge.hidden = true; elements.qualityCard.hidden = true;
}

async function captureFromCamera() {
  if (state.processing || !state.stream) return;
  state.processing = true; clearTimeout(state.analyzeTimer); elements.processing.hidden = false;
  const burst = [];
  for (let index = 0; index < 3; index += 1) {
    const canvas = drawMirroredFrame(1600); if (!canvas) break;
    burst.push(await canvasBlob(canvas, 0.94));
    if (index < 2) await new Promise((resolve) => setTimeout(resolve, 110));
  }
  if (!burst.length) { state.processing = false; elements.processing.hidden = true; return; }
  displayPhoto(URL.createObjectURL(burst.at(-1))); await runTryon(burst);
}

async function processUploadedPhoto(file) {
  if (!file?.type.startsWith("image/")) return showToast("请选择 JPG 或 PNG 图片");
  displayPhoto(URL.createObjectURL(file)); elements.emptyCamera.hidden = true; elements.shutter.disabled = true;
  await runTryon(file);
}

async function runTryon(imageInput) {
  if (!state.selectedGarment) return showToast("请先选择一件上衣");
  const mode = currentRenderMode();
  state.processing = true; updateScaleControl(); elements.processing.hidden = false; clearTimeout(state.analyzeTimer);
  elements.processingTitle.textContent = mode === "ai" ? "AI 正在生成真实试穿" : "正在生成快速预览";
  elements.processingText.textContent = mode === "ai" ? "本地 CPU 约需 3–5 分钟，请保持页面开启" : "本地 CPU 处理中，请稍候";
  elements.aiProgress.hidden = mode !== "ai";
  let stopProgressPolling = null;
  try {
    const form = new FormData(); form.append("garment_id", state.selectedGarment.id); form.append("mode", mode);
    form.append("garment_scale", state.garmentScale.toFixed(2));
    const jobId = mode === "ai" ? createAIJobId() : null;
    if (jobId) form.append("job_id", jobId);
    const isBurst = Array.isArray(imageInput);
    if (isBurst) imageInput.forEach((blob, index) => form.append("images", blob, `burst-${index}.jpg`));
    else form.append("image", imageInput, "person.jpg");
    const resultPromise = apiJson(isBurst ? "/api/tryon/burst" : "/api/tryon", { method: "POST", body: form });
    if (jobId) stopProgressPolling = startAIProgressPolling(jobId);
    const result = await resultPromise;
    if (mode === "ai") setAIProgress(100, "AI 高质量试穿完成");
    elements.resultImage.src = `${result.image_url}?t=${Date.now()}`; elements.resultImage.hidden = false;
    elements.resultPlaceholder.hidden = true; elements.downloadButton.href = result.image_url;
    elements.downloadButton.classList.remove("disabled"); elements.retryButton.disabled = false;
    elements.resultMessage.hidden = !result.warning; if (result.warning) elements.resultMessage.textContent = result.warning;
    state.lastFastResultId = result.mode === "fast" ? result.id : null;
    state.lastResultGarmentId = result.mode === "fast" ? state.selectedGarment.id : null;
    const burstNote = result.burst_frames ? `，已从 ${result.burst_frames} 张中择优` : "";
    const scaleNote = result.mode === "fast" ? `，衣服 ${Math.round(result.garment_scale * 100)}%` : "";
    showToast(`${result.mode === "ai" ? "AI 高质量" : "快速预览"}完成${burstNote}${scaleNote}，用时 ${(result.elapsed_ms / 1000).toFixed(1)} 秒`);
  } catch (error) { showToast(error.message); }
  finally {
    stopProgressPolling?.();
    state.processing = false; updateScaleControl(); elements.processing.hidden = true; elements.aiProgress.hidden = true;
  }
}

function retry() {
  state.lastFastResultId = null; state.lastResultGarmentId = null;
  elements.resultImage.hidden = true; elements.resultPlaceholder.hidden = false; elements.resultMessage.hidden = true;
  elements.downloadButton.classList.add("disabled"); elements.retryButton.disabled = true;
  if (state.stream) {
    elements.photoPreview.hidden = true; elements.camera.hidden = false; elements.liveBadge.hidden = false;
    elements.qualityCard.hidden = false; elements.shutter.disabled = false; state.previousLandmarks = null; state.stableFrames = 0;
    scheduleAnalysis(100);
  } else { elements.photoPreview.hidden = true; elements.emptyCamera.hidden = false; }
}

elements.startCamera.addEventListener("click", startCamera);
elements.shutter.addEventListener("click", captureFromCamera);
elements.retryButton.addEventListener("click", retry);
elements.scaleDown.addEventListener("click", () => changeGarmentScale(-0.05));
elements.scaleUp.addEventListener("click", () => changeGarmentScale(0.05));
elements.uploads.forEach((input) => input.addEventListener("change", () => processUploadedPhoto(input.files[0])));
elements.renderModes.forEach((input) => input.addEventListener("change", () => {
  updateScaleControl();
  if (input.checked && input.value === "ai") showToast("AI 高质量模式会在本机 CPU 上生成，约需 3–5 分钟");
}));
elements.garmentUpload.addEventListener("change", async () => {
  const file = elements.garmentUpload.files[0]; if (!file) return;
  const form = new FormData(); form.append("name", file.name.replace(/\.[^.]+$/, "")); form.append("image", file, file.name);
  try {
    showToast("正在处理服装图片……"); const garment = await apiJson("/api/garments", { method: "POST", body: form });
    await loadGarments(garment.id); showToast("服装已加入衣橱");
  } catch (error) { showToast(error.message); }
  finally { elements.garmentUpload.value = ""; }
});

window.addEventListener("beforeunload", () => state.stream?.getTracks().forEach((track) => track.stop()));
updateScaleControl();
Promise.all([checkHealth(), loadGarments()]).catch((error) => showToast(error.message));
