from __future__ import annotations

import argparse
import faulthandler
import base64
import hashlib
import cgi
import io
import json
import os
import re
import threading
from types import SimpleNamespace
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import cv2 as cv
import numpy as np
import torch

faulthandler.enable()
from PIL import Image

from .adapters import batch_to_tensors
from .atari import AtariHintConfig, AtariHintGenerator
from .conditions import ColorizationCondition, ColorizationMode, ModalityDropout, TaskSampler
from .dataset import UnifiedCollator
from .openniji import OpenNijiParquetColorizationDataset, OpenNijiParquetRecord, deform_reference_rgb
from .reference_conditioning import ReferenceConditioningBuilder, ReferenceConditioningConfig
from .smoke_train_flux_klein import WD_PROJECTOR_FILE, build_condition_images, build_prompt_text, build_wd_projector, generate_sample, tensor_to_pil

DEFAULT_LORA_DIR = Path(
    "/data/shasegawa/adeleine/outputs/flux2_klein_openniji_sketchkeras_wip_xdog_fallback_deformed_ref_text_empty_10k/lora"
)
DEFAULT_HOLDOUT = Path(
    "/data/shasegawa/adeleine/outputs/flux2_klein_openniji_full_epoch_atari_focus_holdout1k_512/holdout_1000.jsonl"
)
DEFAULT_TRAIN_LOG = Path("/data/shasegawa/adeleine/logs/flux2_klein_train_gpu1_full_epoch_atari_focus.log")


INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adeleine v2 Color Lab</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&family=M+PLUS+Rounded+1c:wght@400;500;700;800&display=swap" rel="stylesheet">
  <style>

    :root {
      --pink: #ff6fb5;
      --hot-pink: #ff3d94;
      --yellow: #ffd93d;
      --sky: #4fc3f7;
      --purple: #a66dff;
      --mint: #4ce0b3;
      --ink: #3a2b4d;
      --muted: #7c6690;
      --panel: rgba(255,255,255,.9);
      --line: #ffd3ea;
      --soft: #fff4fb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: 'M PLUS Rounded 1c', 'Baloo 2', ui-rounded, system-ui, sans-serif;
      color: var(--ink);
      background: linear-gradient(120deg, #ffe6f5 0%, #ffe9c7 25%, #dff6ff 55%, #eee0ff 100%);
      background-size: 200% 200%;
      animation: bgshift 18s ease-in-out infinite;
      overflow-x: hidden;
    }
    @keyframes bgshift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
    .floaties { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
    .floaties span { position: absolute; left: var(--x); top: -12%; font-size: 1.75rem; animation: fall 16s linear infinite; animation-delay: var(--d); opacity: .75; }
    @keyframes fall { 0% { transform: translateY(-10vh) rotate(0deg); opacity: 0; } 12%,88% { opacity: .8; } 100% { transform: translateY(112vh) rotate(360deg); opacity: 0; } }
    .shell { max-width: 1480px; margin: 0 auto; padding: 26px 22px 40px; position: relative; z-index: 1; }
    header { text-align: center; padding: 28px 0 20px; }
    h1 { margin: 0; font-family: 'Baloo 2', sans-serif; font-weight: 800; font-size: clamp(42px, 7vw, 82px); line-height: .9; letter-spacing: 0; }
    .pop-title { background: linear-gradient(90deg, var(--hot-pink), var(--purple), var(--sky)); -webkit-background-clip: text; background-clip: text; color: transparent; text-shadow: 3px 3px 0 rgba(255,255,255,.62); }
    .sparkle { display: inline-block; animation: spin 3s linear infinite; font-size: clamp(26px, 4vw, 46px); vertical-align: top; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .tagline { margin: 10px auto 0; max-width: 760px; color: #6b5480; font-size: 1rem; font-weight: 700; }
    .tag { display: inline-flex; align-items: center; justify-content: center; margin-top: 14px; padding: 9px 16px; color: #fff; font-weight: 800; border-radius: 999px; background: linear-gradient(135deg, var(--sky), var(--mint)); box-shadow: 0 6px 0 rgba(43,155,199,.35); }
    .layout { display: grid; grid-template-columns: minmax(230px, 300px) minmax(0, 1fr); gap: 20px; align-items: start; }
    .panel { background: var(--panel); border: 3px solid #fff; border-radius: 28px; box-shadow: 0 12px 0 rgba(255,111,181,.22), 0 22px 42px rgba(166,109,255,.16); overflow: hidden; backdrop-filter: blur(8px); }
    .panel h2 { margin: 0; padding: 18px 20px 10px; font-family: 'Baloo 2', sans-serif; font-size: 1.5rem; color: var(--hot-pink); line-height: 1.1; }
    .panel-body { padding: 18px; }
    .focus-row > .panel { height: 100%; }
    .presets { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; max-height: 76vh; overflow: auto; padding: 3px 4px 8px; }
    .preset { border: 0; border-radius: 20px; background: #fffaff; padding: 8px; cursor: pointer; text-align: left; overflow: hidden; box-shadow: 0 7px 16px rgba(166,109,255,.16); transition: transform .14s, box-shadow .14s; }
    .preset:hover { transform: translateY(-3px); box-shadow: 0 12px 22px rgba(166,109,255,.22); }
    .preset.active { outline: 4px solid rgba(255,61,148,.35); background: #fff4fb; }
    .preset img { display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 14px; background: var(--soft); }
    .preset span { display: block; padding: 8px 4px 2px; font-size: 12px; font-weight: 800; color: #6b4b8a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .work { display: grid; gap: 20px; min-width: 0; }
    .focus-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 20px; align-items: stretch; }
    .paint-panel, .result { min-height: min(76vh, 820px); }
    .controls { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 14px; }
    .field { display: grid; gap: 7px; min-width: 0; }
    label { font-size: 12px; font-weight: 900; color: #6b4b8a; }
    input, textarea, select, button { width: 100%; border: 2px solid var(--line); border-radius: 16px; background: #fffaff; color: var(--ink); font: inherit; padding: 11px 13px; min-height: 44px; transition: border-color .18s, box-shadow .18s, transform .12s; }
    input:focus, textarea:focus, select:focus { outline: none; border-color: var(--sky); box-shadow: 0 0 0 4px rgba(79,195,247,.22); }
    textarea { min-height: 86px; resize: vertical; }
    input[type="checkbox"] { width: 20px; min-height: 20px; height: 20px; padding: 0; accent-color: var(--hot-pink); }
    input[type="color"] { padding: 4px; }
    .checkrow { display: flex; align-items: center; gap: 9px; min-height: 44px; padding: 0 12px; border-radius: 16px; background: #fff4fb; font-weight: 800; color: #6b5480; }
    .wide { grid-column: span 2; }
    .xwide { grid-column: span 3; }
    .full { grid-column: 1 / -1; }
    .actions { display: grid; grid-template-columns: 220px 180px minmax(0, 1fr); gap: 12px; align-items: center; }
    .painter-wrap { display: grid; grid-template-rows: minmax(0, 1fr) auto; gap: 14px; align-items: start; background: linear-gradient(135deg, rgba(255,244,251,.95), rgba(223,246,255,.85)); border-radius: 24px; padding: 8px 18px 18px; height: calc(100% - 54px); }
    .painter-tools { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .painter-canvas { width: 100%; max-width: min(100%, 72vh); aspect-ratio: 1 / 1; border: 4px solid var(--yellow); border-radius: 22px; background: #fff; touch-action: none; box-shadow: 0 10px 24px rgba(0,0,0,.12); justify-self: center; align-self: start; }
    button { font-family: 'Baloo 2', sans-serif; font-weight: 800; cursor: pointer; border: 0; color: #fff; background: linear-gradient(135deg, var(--hot-pink), var(--purple)); box-shadow: 0 6px 0 #c23f8e; }
    button:hover { transform: translateY(-2px); }
    button:active { transform: translateY(3px); box-shadow: 0 2px 0 #c23f8e; }
    button.primary { background: linear-gradient(135deg, var(--hot-pink), var(--purple)); }
    button.secondary { background: linear-gradient(135deg, var(--sky), var(--mint)); box-shadow: 0 6px 0 #2b9bc7; }
    button.secondary:active { box-shadow: 0 2px 0 #2b9bc7; }
    button:disabled { opacity: .58; cursor: wait; transform: none; box-shadow: none; }
    .status { padding: 12px 14px; border-radius: 16px; background: #fff4fb; font-size: 13px; color: var(--muted); font-weight: 800; overflow-wrap: anywhere; }
    .grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
    .tile { border: 3px solid #fff; border-radius: 22px; overflow: hidden; background: #fffaff; min-width: 0; box-shadow: 0 8px 18px rgba(166,109,255,.12); }
    .tile h3 { margin: 0; padding: 10px 12px; font-family: 'Baloo 2', sans-serif; font-size: 14px; color: #6b4b8a; background: #fff4fb; }
    .tile img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; display: block; background: #fff; }
    .result { position: sticky; top: 18px; align-self: start; }
    .result h2 { color: var(--mint); }
    .result img { display: block; width: calc(100% - 36px); max-width: min(100%, 72vh); aspect-ratio: 1 / 1; object-fit: contain; margin: 8px auto 18px; border-radius: 22px; background: #fff; box-shadow: 0 10px 28px rgba(0,0,0,.12); }
    .paint-panel .painter-wrap, .result img { margin-top: 8px; }
    .paint-panel .painter-wrap { height: calc(100% - 62px); }
    @media (max-width: 1260px) { .layout { grid-template-columns: 1fr; } .focus-row { grid-template-columns: 1fr 1fr; } .result { position: static; } .presets { grid-template-columns: repeat(4, 1fr); max-height: none; } }
    @media (max-width: 1100px) { .focus-row { grid-template-columns: 1fr; } .paint-panel, .result { min-height: 360px; } }
    @media (max-width: 760px) { .shell { padding: 14px; } .controls { grid-template-columns: 1fr 1fr; } .wide, .xwide { grid-column: 1 / -1; } .grid { grid-template-columns: 1fr 1fr; } .presets { grid-template-columns: 1fr 1fr; } .actions, .painter-tools { grid-template-columns: 1fr; } }

  </style>
</head>
<body>
  <div class="floaties" aria-hidden="true">
    <span style="--d:0s;--x:6%">*</span>
    <span style="--d:2s;--x:18%">+</span>
    <span style="--d:4s;--x:32%">o</span>
    <span style="--d:1s;--x:52%">*</span>
    <span style="--d:3s;--x:68%">+</span>
    <span style="--d:5s;--x:80%">o</span>
    <span style="--d:2.5s;--x:92%">*</span>
  </div>
  <div class="shell">
    <header>
      <h1><span class="pop-title">Adeleine</span><span class="sparkle">*</span><br><span class="pop-title">Color Lab</span></h1>
      <p class="tagline">Line art colorization with optional Atari hints, references, and text.</p>
      <div class="tag" id="health">loading</div>
    </header>
    <main class="layout">
      <section class="panel"><h2>Preset Gallery</h2><div class="panel-body"><div class="presets" id="presets"></div></div></section>
      <section class="work">
        <div class="focus-row">
          <section class="panel paint-panel"><h2>Atari Canvas</h2><div class="panel-body painter-wrap">
            <canvas id="atariCanvas" class="painter-canvas" width="512" height="512"></canvas>
            <div class="painter-tools">
              <div class="field"><label>Tool</label><select id="atariTool"><option value="dot">dot</option><option value="free">freehand</option><option value="line">line</option></select></div>
              <div class="field"><label>Color</label><input id="atariColor" type="color" value="#ff5f5f"></div>
              <div class="field"><label>Brush</label><input id="atariBrush" type="number" value="9" min="1" max="48"></div>
              <div class="field"><label>Canvas</label><button id="clearAtari" class="secondary" type="button">Clear</button></div>
            </div>
          </div></section>
          <section class="panel result"><h2>Result</h2><img id="resultImage"></section>
        </div>
        <section class="panel"><h2>Options</h2><div class="panel-body controls">
          <div class="field wide"><label>Line art</label><input id="lineartFile" type="file" accept="image/*"></div>
          <div class="field wide"><label>Atari image</label><input id="atariFile" type="file" accept="image/*"></div>
          <div class="field wide"><label>Reference</label><input id="referenceFile" type="file" accept="image/*"></div>
          <div class="field"><label>Atari</label><div class="checkrow"><input id="includeAtari" type="checkbox"><span>use</span></div></div>
          <div class="field"><label>Reference</label><div class="checkrow"><input id="includeReference" type="checkbox"><span>use</span></div></div>
          <div class="field"><label>Text</label><div class="checkrow"><input id="includeText" type="checkbox"><span>use</span></div></div>
          <div class="field"><label>Clean lines</label><div class="checkrow"><input id="cleanLineart" type="checkbox" checked><span>on</span></div></div>
          <div class="field"><label>Mode</label><select id="mode"><option value="render">render</option><option value="diverse">diverse</option><option value="flat">flat</option></select></div>
          <div class="field"><label>Seed</label><input id="seed" type="number" value="3170"></div>
          <div class="field"><label>Steps</label><input id="steps" type="number" value="12" min="1" max="40"></div>
          <div class="field"><label>Guidance</label><input id="guidance" type="number" value="3.0" min="0" max="12" step="0.1"></div>
          <div class="field xwide"><label>Text prompt</label><textarea id="prompt"></textarea></div>
          <div class="field full actions"><button id="generate" class="primary">Generate</button><button id="clear" class="secondary" type="button">Clear Uploads</button><div class="status" id="status">Pick a preset or upload a line art.</div></div>
        </div></section>
        <section class="panel"><h2>Preview Board</h2><div class="panel-body grid">
          <div class="tile"><h3>line art</h3><img id="lineartPreview"></div>
          <div class="tile"><h3>atari</h3><img id="atariPreview"></div>
          <div class="tile"><h3>mask</h3><img id="maskPreview"></div>
          <div class="tile"><h3>reference</h3><img id="referencePreview"></div>
          <div class="tile"><h3>target</h3><img id="targetPreview"></div>
        </div></section>
      </section>
    </main>
  </div>
<script>
let selectedPreset = null;
let atariHasPaint = false;
let atariDrawing = false;
let atariStart = null;
let atariBase = null;
const $ = (id) => document.getElementById(id);
const atariCanvas = $('atariCanvas');
const atariCtx = atariCanvas.getContext('2d');
function setStatus(text) { $('status').textContent = text; }
function setImg(id, src) { const img = $(id); img.src = src || ''; }
function drawBlankAtari() { atariCtx.fillStyle = '#ffffff'; atariCtx.fillRect(0, 0, 512, 512); atariBase = atariCtx.getImageData(0, 0, 512, 512); atariHasPaint = false; }
async function resetAtariCanvas(src) {
  if (!src) { drawBlankAtari(); return; }
  try {
    const resp = await fetch(src, { cache: 'no-store' });
    if (!resp.ok) throw new Error(resp.statusText);
    const blob = await resp.blob();
    const bitmap = await createImageBitmap(blob);
    atariCtx.fillStyle = '#ffffff';
    atariCtx.fillRect(0, 0, 512, 512);
    atariCtx.drawImage(bitmap, 0, 0, 512, 512);
    atariBase = atariCtx.getImageData(0, 0, 512, 512);
    atariHasPaint = false;
    setImg('atariPreview', atariCanvas.toDataURL('image/png'));
  } catch (err) {
    drawBlankAtari();
    setStatus(`Could not load line art into Atari canvas: ${err.message || err}`);
  }
}
function restoreAtariBase() { if (atariBase) atariCtx.putImageData(atariBase, 0, 0); else drawBlankAtari(); atariHasPaint = false; setImg('atariPreview', atariCanvas.toDataURL('image/png')); }
function canvasPoint(ev) { const r = atariCanvas.getBoundingClientRect(); const p = ev.touches ? ev.touches[0] : ev; return { x: (p.clientX - r.left) * 512 / r.width, y: (p.clientY - r.top) * 512 / r.height }; }
function paintDot(pt) { atariCtx.fillStyle = $('atariColor').value; atariCtx.beginPath(); atariCtx.arc(pt.x, pt.y, Math.max(1, Number($('atariBrush').value) || 9), 0, Math.PI * 2); atariCtx.fill(); atariHasPaint = true; setImg('atariPreview', atariCanvas.toDataURL('image/png')); }
function paintSegment(a, b) { atariCtx.strokeStyle = $('atariColor').value; atariCtx.lineWidth = Math.max(1, Number($('atariBrush').value) || 9); atariCtx.lineCap = 'round'; atariCtx.lineJoin = 'round'; atariCtx.beginPath(); atariCtx.moveTo(a.x, a.y); atariCtx.lineTo(b.x, b.y); atariCtx.stroke(); atariHasPaint = true; setImg('atariPreview', atariCanvas.toDataURL('image/png')); }
function startAtari(ev) { ev.preventDefault(); $('includeAtari').checked = true; atariDrawing = true; atariStart = canvasPoint(ev); if ($('atariTool').value === 'dot') { paintDot(atariStart); atariDrawing = false; } }
function moveAtari(ev) { if (!atariDrawing || $('atariTool').value !== 'free') return; ev.preventDefault(); const pt = canvasPoint(ev); paintSegment(atariStart, pt); atariStart = pt; }
function endAtari(ev) { if (!atariDrawing) return; ev.preventDefault(); const pt = canvasPoint(ev); if ($('atariTool').value === 'line') paintSegment(atariStart, pt); atariDrawing = false; }
async function refreshHealth() {
  try { const data = await (await fetch('/api/health')).json(); $('health').textContent = data.model_loaded ? 'model loaded' : 'lazy model'; }
  catch { $('health').textContent = 'offline'; }
}
async function loadPresets() {
  const data = await (await fetch('/api/presets?limit=36')).json();
  const root = $('presets'); root.innerHTML = '';
  data.presets.forEach((p) => {
    const b = document.createElement('button'); b.className = 'preset';
    b.innerHTML = `<img src="${p.thumbnail}"><span>#${p.id} ${p.title || p.style || 'preset'}</span>`;
    b.onclick = () => selectPreset(p.id, b); root.appendChild(b);
  });
  if (data.presets.length) selectPreset(data.presets[0].id, root.querySelector('.preset'));
}
async function selectPreset(id, button) {
  selectedPreset = id;
  document.querySelectorAll('.preset').forEach(x => x.classList.remove('active'));
  if (button) button.classList.add('active');
  const meta = await (await fetch(`/api/preset/${id}`)).json();
  $('prompt').value = meta.prompt || '';
  setImg('targetPreview', `/api/preset/${id}/image?kind=target&t=${Date.now()}`);
  const lineUrl = `/api/preset/${id}/image?kind=lineart&t=${Date.now()}`;
  setImg('lineartPreview', lineUrl);
  resetAtariCanvas(lineUrl);
  setImg('atariPreview', `/api/preset/${id}/image?kind=atari&t=${Date.now()}`);
  setImg('maskPreview', `/api/preset/${id}/image?kind=mask&t=${Date.now()}`);
  setImg('referencePreview', `/api/preset/${id}/image?kind=reference&t=${Date.now()}`);
  setStatus(`Preset #${id} selected.`);
}
function filePreview(inputId, imageId) { const file = $(inputId).files[0]; if (!file) return null; const url = URL.createObjectURL(file); setImg(imageId, url); return url; }
['lineartFile','atariFile','referenceFile'].forEach(id => $(id).addEventListener('change', () => {
  if (id === 'lineartFile') { const url = filePreview(id, 'lineartPreview'); resetAtariCanvas(url); }
  if (id === 'atariFile') { filePreview(id, 'atariPreview'); $('includeAtari').checked = true; atariHasPaint = false; }
  if (id === 'referenceFile') { filePreview(id, 'referencePreview'); $('includeReference').checked = true; }
}));
atariCanvas.addEventListener('mousedown', startAtari);
atariCanvas.addEventListener('mousemove', moveAtari);
window.addEventListener('mouseup', endAtari);
atariCanvas.addEventListener('touchstart', startAtari, { passive: false });
atariCanvas.addEventListener('touchmove', moveAtari, { passive: false });
atariCanvas.addEventListener('touchend', endAtari, { passive: false });
$('clearAtari').onclick = restoreAtariBase;
$('clear').onclick = () => { ['lineartFile','atariFile','referenceFile'].forEach(id => $(id).value = ''); if (selectedPreset !== null) selectPreset(selectedPreset, document.querySelector('.preset.active')); else restoreAtariBase(); };
$('generate').onclick = async () => {
  const fd = new FormData();
  if (selectedPreset !== null) fd.append('preset_id', selectedPreset);
  fd.append('include_atari', $('includeAtari').checked ? '1' : '0');
  fd.append('include_reference', $('includeReference').checked ? '1' : '0');
  fd.append('include_text', $('includeText').checked ? '1' : '0');
  fd.append('clean_lineart', $('cleanLineart').checked ? '1' : '0');
  fd.append('mode', $('mode').value); fd.append('prompt', $('prompt').value); fd.append('seed', $('seed').value); fd.append('steps', $('steps').value); fd.append('guidance', $('guidance').value);
  ['lineartFile','atariFile','referenceFile'].forEach(id => { if ($(id).files[0]) fd.append(id, $(id).files[0]); });
  if ($('includeAtari').checked && atariHasPaint) fd.append('atari_canvas', atariCanvas.toDataURL('image/png'));
  $('generate').disabled = true; setStatus('Generating...');
  try {
    const resp = await fetch('/api/colorize', { method: 'POST', body: fd });
    const data = await resp.json(); if (!resp.ok) throw new Error(data.error || resp.statusText);
    setImg('resultImage', data.generated); setImg('lineartPreview', data.previews.lineart); setImg('atariPreview', data.previews.atari); setImg('maskPreview', data.previews.mask); setImg('referencePreview', data.previews.reference); setStatus(data.summary);
  } catch (err) { setStatus(String(err.message || err)); }
  finally { $('generate').disabled = false; refreshHealth(); }
};
drawBlankAtari(); refreshHealth(); loadPresets().catch(err => setStatus(String(err))); setInterval(refreshHealth, 8000);
</script>
</body>
</html>'''


@dataclass
class FormImage:
    filename: str
    data: bytes


@dataclass
class WebArgs:
    host: str
    port: int
    model_id: str
    lora_dir: Path
    hf_home: Path
    holdout_manifest: Path
    train_log: Path
    sketch_root: Optional[Path]
    image_size: int
    preset_count: int
    device: str
    local_files_only: bool
    cpu_offload: bool
    preload: bool
    max_condition_images: int
    spatial_hint_mode: str
    spatial_condition_id_mode: str
    reference_conditioning: str
    reference_condition_mode: str
    reference_tag_cache_root: Optional[Path]
    reference_mask_root: Optional[Path]
    reference_mask_fallback: str
    skytnt_repo: Optional[Path]
    skytnt_model_id: str
    skytnt_ckpt: Optional[Path]
    skytnt_net: str
    skytnt_image_size: int
    skytnt_device: str
    skytnt_fp32: bool
    skytnt_local_files_only: bool
    reference_cache_generated_masks: bool
    wd_tagger_model: Optional[Path]
    wd_tagger_labels: Optional[Path]
    wd_tagger_threshold: float
    reference_tag_max: int
    append_reference_tags: bool
    reference_tag_prefix: str
    reference_wd_max_refs: int
    reference_wd_tokens_per_ref: int
    reference_wd_embed_dim: int
    max_sequence_length: int
    default_steps: int
    default_guidance: float
    seed: int


class AdeleineWebApp:
    def __init__(self, args: WebArgs):
        self.args = args
        self.collator = UnifiedCollator()
        self.reference_conditioner = ReferenceConditioningBuilder(
            ReferenceConditioningConfig(
                mode=args.reference_conditioning,
                tag_cache_root=args.reference_tag_cache_root,
                mask_root=args.reference_mask_root,
                mask_fallback=args.reference_mask_fallback,
                skytnt_repo=args.skytnt_repo,
                skytnt_model_id=args.skytnt_model_id,
                skytnt_ckpt=args.skytnt_ckpt,
                skytnt_net=args.skytnt_net,
                skytnt_image_size=args.skytnt_image_size,
                skytnt_device=args.skytnt_device,
                skytnt_fp32=args.skytnt_fp32,
                skytnt_local_files_only=args.skytnt_local_files_only,
                cache_generated_masks=args.reference_cache_generated_masks,
                wd_tagger_model=args.wd_tagger_model,
                wd_tagger_labels=args.wd_tagger_labels,
                wd_tagger_threshold=args.wd_tagger_threshold,
                max_tags=args.reference_tag_max,
            )
        )
        self._pipe = None
        self._wd_projector = None
        self._pipe_lock = threading.Lock()
        self._dataset = None
        self._dataset_lock = threading.Lock()
        self._preset_lock = threading.Lock()
        self._preset_cache: dict[int, ColorizationCondition] = {}
        self._target_cache: dict[int, np.ndarray] = {}
        self._records = self._read_holdout_records(args.holdout_manifest, 0)
        if args.preload:
            self.load_pipeline()

    @property
    def model_loaded(self) -> bool:
        return self._pipe is not None

    def health(self) -> dict[str, Any]:
        return {"ok": True, "model_loaded": self.model_loaded, "lora_dir": str(self.args.lora_dir), "device": self.args.device, "cpu_offload": self.args.cpu_offload, "preset_count": len(self._records), "train_log": str(self.args.train_log), "latest_training": self._tail_training_line()}

    def list_presets(self, limit: int) -> list[dict[str, Any]]:
        total = len(self._records)
        limit = min(max(limit, 0), total)
        if limit == 0:
            return []
        if limit >= total:
            indices = list(range(total))
        else:
            indices = []
            seen = set()
            for value in np.linspace(0, total - 1, limit):
                idx = int(round(float(value)))
                if idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
            cursor = 0
            while len(indices) < limit and cursor < total:
                if cursor not in seen:
                    indices.append(cursor)
                    seen.add(cursor)
                cursor += 1
        presets = []
        for idx in indices:
            record = self._records[idx]
            title = compact_prompt_label(record.prompt, record.style)
            presets.append({"id": idx, "style": record.style, "prompt": record.prompt, "title": title, "thumbnail": f"/api/preset/{idx}/image?kind=target"})
        return presets

    def preset_meta(self, preset_id: int) -> dict[str, Any]:
        record = self._records[preset_id]
        return {"id": preset_id, "prompt": record.prompt, "style": record.style, "url": record.url, "parquet_path": str(record.parquet_path), "row_group": record.row_group, "row_in_group": record.row_in_group}

    def preset_image(self, preset_id: int, kind: str) -> Image.Image:
        if kind == "target":
            return Image.fromarray(self.get_preset_target(preset_id))
        cond = self.get_preset_condition(preset_id)
        if kind == "lineart":
            return Image.fromarray(cond.lineart)
        if kind == "atari" and cond.atari_rgb is not None:
            return Image.fromarray(cond.atari_rgb)
        if kind == "mask" and cond.atari_mask is not None:
            return mask_to_preview(cond.atari_mask)
        if kind == "reference" and cond.references:
            return Image.fromarray(cond.references[0])
        return blank_image(self.args.image_size)

    def get_preset_target(self, preset_id: int) -> np.ndarray:
        with self._preset_lock:
            cached = self._target_cache.get(preset_id)
            if cached is not None:
                return cached
            dataset = self.load_dataset()
            record = self._records[preset_id]
            row = dataset._read_record(record)
            color_bgr = dataset._decode_image(row["image"]["bytes"])
            color_bgr = dataset._resize_square(color_bgr)
            color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)
            self._target_cache[preset_id] = color_rgb
            return color_rgb

    def get_preset_condition(self, preset_id: int) -> ColorizationCondition:
        with self._preset_lock:
            cached = self._preset_cache.get(preset_id)
            if cached is not None:
                return cached
            dataset = self.load_dataset()
            state = np.random.get_state()
            np.random.seed(self.args.seed + preset_id)
            try:
                cond = dataset[preset_id]
            finally:
                np.random.set_state(state)
            record = self._records[preset_id]
            cond.lineart = self._sketchkeras_lineart_for_record(record, cond.lineart)
            self._preset_cache[preset_id] = cond
            if cond.target is not None:
                self._target_cache[preset_id] = cond.target
            return cond

    def _sketchkeras_lineart_for_record(self, record: OpenNijiParquetRecord, fallback: np.ndarray) -> np.ndarray:
        if self.args.sketch_root is None:
            return fallback
        digest = hashlib.sha256(record.url.encode("utf-8")).hexdigest()
        path = self.args.sketch_root / f"{digest}.png"
        line_bgr = cv.imread(str(path), cv.IMREAD_COLOR)
        if line_bgr is None:
            return fallback
        line_bgr = cv.resize(line_bgr, (self.args.image_size, self.args.image_size), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(line_bgr, cv.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 127.0:
            gray = 255 - gray
        rgb = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
        return rgb.astype(np.uint8)

    def colorize(self, form: dict[str, Any], files: dict[str, FormImage]) -> dict[str, Any]:
        start = time.time()
        preset_id = parse_optional_int(form.get("preset_id"))
        preset = self.get_preset_condition(preset_id) if preset_id is not None and 0 <= preset_id < len(self._records) else None
        clean_lineart = truthy(form.get("clean_lineart"), default=True)
        include_atari = truthy(form.get("include_atari"), default=False)
        include_reference = truthy(form.get("include_reference"), default=False)
        include_text = truthy(form.get("include_text"), default=False)
        mode = parse_mode(form.get("mode", "render"))
        seed = parse_int(form.get("seed"), self.args.seed)
        steps = int(np.clip(parse_int(form.get("steps"), self.args.default_steps), 1, 80))
        guidance = float(np.clip(parse_float(form.get("guidance"), self.args.default_guidance), 0.0, 20.0))
        prompt = str(form.get("prompt") or "")
        lineart = decode_form_image(files.get("lineartFile"), self.args.image_size)
        if lineart is None and preset is not None:
            lineart = preset.lineart.copy()
        if lineart is None:
            raise ValueError("Line art is required. Upload a line art image or select a hold-out preset.")
        if clean_lineart:
            lineart = clean_lineart_rgb(lineart)
        atari_rgb = None
        atari_mask = None
        if include_atari:
            atari_rgb = decode_data_url_image(form.get("atari_canvas"), self.args.image_size)
            if atari_rgb is None:
                atari_rgb = decode_form_image(files.get("atariFile"), self.args.image_size)
            if atari_rgb is None and preset is not None:
                atari_rgb = preset.atari_rgb.copy() if preset.atari_rgb is not None else None
                atari_mask = preset.atari_mask.copy() if preset.atari_mask is not None else None
            if atari_rgb is not None and atari_mask is None:
                atari_mask = infer_atari_mask(atari_rgb, lineart)
        reference = None
        if include_reference:
            reference = decode_form_image(files.get("referenceFile"), self.args.image_size)
            if reference is None and preset is not None and preset.target is not None:
                state = np.random.get_state()
                np.random.seed(seed + 7001)
                try:
                    reference = deform_reference_rgb(preset.target.copy())
                finally:
                    np.random.set_state(state)
            elif reference is None and preset is not None and preset.references:
                reference = preset.references[0].copy()
        text = prompt if include_text else ""
        if include_text and not text and preset is not None:
            text = preset.text
        references = [reference] if reference is not None else []
        ref_cond = self.reference_conditioner.build(references)
        cond = ColorizationCondition(
            lineart=lineart,
            target=preset.target.copy() if preset is not None and preset.target is not None else None,
            atari_rgb=atari_rgb,
            atari_mask=atari_mask,
            references=references,
            reference_foregrounds=ref_cond.foregrounds,
            reference_backgrounds=ref_cond.backgrounds,
            reference_masks=ref_cond.masks,
            reference_tags=ref_cond.tags,
            reference_wd_indices=ref_cond.wd_indices,
            reference_wd_scores=ref_cond.wd_scores,
            text=text,
            mode=mode,
            metadata={"source": "web", "preset_id": preset_id, "reference_tags": ref_cond.tags},
        )
        generated, prompt_text, labels = self.run_generation(cond, seed=seed, steps=steps, guidance=guidance)
        elapsed = time.time() - start
        return {"generated": image_to_data_url(generated), "prompt": prompt_text, "condition_labels": labels, "summary": f"Generated in {elapsed:.1f}s with {', '.join(labels)}.", "previews": {"lineart": image_to_data_url(Image.fromarray(lineart)), "atari": image_to_data_url(Image.fromarray(atari_rgb)) if atari_rgb is not None else image_to_data_url(blank_image(self.args.image_size)), "mask": image_to_data_url(mask_to_preview(atari_mask)) if atari_mask is not None else image_to_data_url(blank_image(self.args.image_size)), "reference": image_to_data_url(Image.fromarray(reference)) if reference is not None else image_to_data_url(blank_image(self.args.image_size))}}

    def run_generation(self, cond: ColorizationCondition, seed: int, steps: int, guidance: float) -> tuple[Image.Image, str, list[str]]:
        pipe = self.load_pipeline()
        batch = self.collator([cond])
        tensors = batch_to_tensors(batch, device="cpu")
        condition_images_cpu, labels, _ = build_condition_images(tensors, False, self.args.max_condition_images, self.args.spatial_hint_mode, self.args.reference_condition_mode)
        prompt_text = build_prompt_text(
            tensors.text,
            batch.mode,
            batch.presence,
            False,
            reference_tags=tensors.reference_tags,
            append_reference_tags=self.args.append_reference_tags,
            reference_tag_prefix=self.args.reference_tag_prefix,
        )[0]
        sample_args = SimpleNamespace(
            device=self.args.device if self.args.device.startswith("cuda") else "cpu",
            image_size=self.args.image_size,
            sample_inference_steps=steps,
            sample_guidance_scale=guidance,
            max_sequence_length=self.args.max_sequence_length,
            spatial_condition_id_mode=self.args.spatial_condition_id_mode,
            reference_wd_context=self._wd_projector is not None,
        )
        result = generate_sample(pipe, tensors, condition_images_cpu, labels, [prompt_text], sample_args, seed, self._wd_projector)
        return result, prompt_text, labels

    def load_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        with self._pipe_lock:
            if self._pipe is not None:
                return self._pipe
            os.environ.setdefault("HF_HOME", str(self.args.hf_home))
            os.environ.setdefault("HF_HUB_CACHE", str(self.args.hf_home / "hub"))
            from diffusers import Flux2KleinPipeline
            from peft import PeftModel
            pipe = Flux2KleinPipeline.from_pretrained(self.args.model_id, cache_dir=str(self.args.hf_home / "hub"), torch_dtype=torch.bfloat16, local_files_only=self.args.local_files_only)
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, self.args.lora_dir)
            if self.args.cpu_offload:
                pipe.enable_model_cpu_offload(self.args.device)
            else:
                pipe.to(self.args.device)
            pipe.transformer.eval()
            pipe.set_progress_bar_config(disable=True)
            if self.args.reference_conditioning in {"split_wd", "split_wd_tags"}:
                if (self.args.lora_dir / WD_PROJECTOR_FILE).exists():
                    self._wd_projector = build_wd_projector(
                        pipe,
                        self.args.wd_tagger_labels,
                        self.args.reference_wd_embed_dim,
                        self.args.reference_wd_max_refs,
                        self.args.reference_wd_tokens_per_ref,
                        self.args.device,
                        next(pipe.transformer.parameters()).dtype,
                        state_dir=self.args.lora_dir,
                    ).eval()
                else:
                    print(f"warning: {self.args.lora_dir / WD_PROJECTOR_FILE} missing; serving without WD context", flush=True)
            self._pipe = pipe
            return pipe

    def load_dataset(self) -> OpenNijiParquetColorizationDataset:
        if self._dataset is not None:
            return self._dataset
        with self._dataset_lock:
            if self._dataset is not None:
                return self._dataset
            line_methods = ("pencil",) if self.args.sketch_root is not None else ("xdog",)
            dataset = OpenNijiParquetColorizationDataset(
                repo_id="all",
                hf_home=self.args.hf_home,
                parquet_pattern="data/*.parquet",
                sketch_root=self.args.sketch_root,
                image_size=self.args.image_size,
                max_records=1,
                line_methods=line_methods,
                dropout=ModalityDropout(TaskSampler(weights={"all": 1.0})),
                reference_policy="deformed_self",
                reference_conditioning=self.args.reference_conditioning,
                reference_tag_cache_root=self.args.reference_tag_cache_root,
                reference_mask_root=self.args.reference_mask_root,
                reference_mask_fallback=self.args.reference_mask_fallback,
                skytnt_repo=self.args.skytnt_repo,
                skytnt_model_id=self.args.skytnt_model_id,
                skytnt_ckpt=self.args.skytnt_ckpt,
                skytnt_net=self.args.skytnt_net,
                skytnt_image_size=self.args.skytnt_image_size,
                skytnt_device=self.args.skytnt_device,
                skytnt_fp32=self.args.skytnt_fp32,
                skytnt_local_files_only=self.args.skytnt_local_files_only,
                reference_cache_generated_masks=self.args.reference_cache_generated_masks,
                wd_tagger_model=self.args.wd_tagger_model,
                wd_tagger_labels=self.args.wd_tagger_labels,
                wd_tagger_threshold=self.args.wd_tagger_threshold,
                reference_tag_max=self.args.reference_tag_max,
            )
            dataset.records = self._records
            dataset.groups = dataset._build_groups(self._records)
            dataset.atari = AtariHintGenerator(AtariHintConfig(mode="dot"))
            dataset.lineart.config.morphology_prob = 0.0
            dataset.lineart.config.color_variant_prob = 0.0
            self._dataset = dataset
            return dataset

    def _tail_training_line(self) -> str:
        path = self.args.train_log
        if not path.exists():
            return ""
        try:
            data = path.read_bytes()[-20000:].decode("utf-8", errors="ignore")
        except Exception:
            return ""
        lines = [line.strip() for line in re.split(r"\r|\n", data) if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _read_holdout_records(path: Path, limit: int) -> list[OpenNijiParquetRecord]:
        records: list[OpenNijiParquetRecord] = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if limit > 0 and len(records) >= limit:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                records.append(OpenNijiParquetRecord(parquet_path=Path(obj["parquet_path"]), row_group=int(obj["row_group"]), row_in_group=int(obj["row_in_group"]), prompt=obj.get("prompt", ""), style=obj.get("style", ""), url=obj.get("url", ""), group_key=obj.get("group_key", "")))
        return records


class AdeleineRequestHandler(BaseHTTPRequestHandler):
    app: AdeleineWebApp

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/" or parsed.path == "/index.html":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/health":
                self.send_json(self.app.health())
                return
            if parsed.path == "/api/presets":
                limit = parse_int(first(parse_qs(parsed.query).get("limit")), 12)
                self.send_json({"presets": self.app.list_presets(limit)})
                return
            match = re.fullmatch(r"/api/preset/(\d+)", parsed.path)
            if match:
                self.send_json(self.app.preset_meta(int(match.group(1))))
                return
            match = re.fullmatch(r"/api/preset/(\d+)/image", parsed.path)
            if match:
                kind = first(parse_qs(parsed.query).get("kind")) or "target"
                self.send_image(self.app.preset_image(int(match.group(1)), kind))
                return
            self.send_error_json(404, "Not found")
        except Exception as exc:
            self.send_error_json(500, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/colorize":
                form, files = self.parse_multipart()
                self.send_json(self.app.colorize(form, files))
                return
            self.send_error_json(404, "Not found")
        except Exception as exc:
            self.send_error_json(500, str(exc))

    def parse_multipart(self) -> tuple[dict[str, str], dict[str, FormImage]]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            values = parse_qs(raw)
            return {key: first(value) or "" for key, value in values.items()}, {}
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type})
        fields: dict[str, str] = {}
        files: dict[str, FormImage] = {}
        for key in form.keys():
            item = form[key]
            if isinstance(item, list):
                item = item[0]
            if getattr(item, "filename", None):
                data = item.file.read()
                if data:
                    files[key] = FormImage(filename=item.filename or key, data=data)
            else:
                fields[key] = item.value
        return fields, files

    def send_json(self, payload: Any, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def send_image(self, image: Image.Image) -> None:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        self.send_bytes(buf.getvalue(), "image/png")

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def first(values: Optional[list[str]]) -> Optional[str]:
    if not values:
        return None
    return values[0]


def truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on", "use"}


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_mode(value: Any) -> ColorizationMode:
    try:
        return ColorizationMode(str(value))
    except Exception:
        return ColorizationMode.RENDER


def blank_image(size: int) -> Image.Image:
    return Image.new("RGB", (size, size), "white")


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def decode_form_image(file: Optional[FormImage], size: int) -> Optional[np.ndarray]:
    if file is None:
        return None
    image = Image.open(io.BytesIO(file.data)).convert("RGB")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def decode_data_url_image(value: Any, size: int) -> Optional[np.ndarray]:
    if not value:
        return None
    text = str(value)
    if not text.startswith("data:image/") or "," not in text:
        return None
    payload = text.split(",", 1)[1]
    try:
        data = base64.b64decode(payload)
    except Exception:
        return None
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def compact_prompt_label(prompt: str, style: str = "", max_len: int = 42) -> str:
    text = re.sub(r"<@!?\d+>", "", prompt or "")
    text = re.sub(r"-?\s*Image\s*#?\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    if style and style != "V5-Default":
        text = f"{style}: {text}" if text else style
    if not text:
        text = style or "holdout preset"
    return text[: max_len - 1].rstrip() + "…" if len(text) > max_len else text


def clean_lineart_rgb(rgb: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    if float(np.mean(gray)) < 127.0:
        gray = 255 - gray
    out = np.full_like(gray, 255, dtype=np.uint8)
    out[gray < 210] = 0
    return cv.cvtColor(out, cv.COLOR_GRAY2RGB)


def infer_atari_mask(atari_rgb: np.ndarray, lineart_rgb: np.ndarray) -> np.ndarray:
    nonwhite = np.any(atari_rgb < 245, axis=2)
    diff = np.max(np.abs(atari_rgb.astype(np.int16) - lineart_rgb.astype(np.int16)), axis=2) > 25
    candidates = []
    for mask in [diff, nonwhite]:
        ratio = float(np.mean(mask))
        if ratio > 0.00002:
            candidates.append((ratio, mask))
    if not candidates:
        mask = np.zeros(atari_rgb.shape[:2], dtype=np.bool_)
    else:
        mask = min(candidates, key=lambda item: item[0])[1]
    return (mask.astype(np.uint8) * 255)[..., None]


def mask_to_preview(mask: Optional[np.ndarray]) -> Image.Image:
    if mask is None:
        return Image.new("RGB", (1, 1), "black")
    m = mask.squeeze()
    rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    rgb[m > 0] = 255
    return Image.fromarray(rgb)


def parse_args() -> WebArgs:
    parser = argparse.ArgumentParser(description="Adeleine v2 local colorization web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--lora_dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--hf_home", type=Path, default=Path("/data/shasegawa/adeleine/huggingface"))
    parser.add_argument("--holdout_manifest", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--train_log", type=Path, default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--sketch_root", type=Path, default=Path("/data/shasegawa/adeleine/openniji/sketchkeras"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--preset_count", type=int, default=48)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--max_condition_images", type=int, default=4)
    parser.add_argument("--spatial_hint_mode", choices=["separate", "fused", "fused_masked"], default="separate")
    parser.add_argument("--spatial_condition_id_mode", choices=["default", "hint_to_output", "line_hint_to_output"], default="default")
    parser.add_argument("--reference_conditioning", choices=["none", "split", "split_tags", "split_wd", "split_wd_tags"], default="none")
    parser.add_argument("--reference_condition_mode", choices=["full", "foreground", "background", "split", "split_full"], default="full")
    parser.add_argument("--reference_tag_cache_root", type=Path)
    parser.add_argument("--reference_mask_root", type=Path)
    parser.add_argument("--reference_mask_fallback", choices=["skytnt", "grabcut", "ellipse", "whole", "skip"], default="grabcut")
    parser.add_argument("--skytnt_repo", type=Path)
    parser.add_argument("--skytnt_model_id", default="skytnt/anime-seg")
    parser.add_argument("--skytnt_ckpt", type=Path)
    parser.add_argument("--skytnt_net", default="isnet_is")
    parser.add_argument("--skytnt_image_size", type=int, default=1024)
    parser.add_argument("--skytnt_device", default="cuda:2")
    parser.add_argument("--skytnt_fp32", action="store_true")
    parser.add_argument("--skytnt_local_files_only", action="store_true")
    parser.add_argument("--no_reference_cache_generated_masks", action="store_true")
    parser.add_argument("--wd_tagger_model", type=Path)
    parser.add_argument("--wd_tagger_labels", type=Path)
    parser.add_argument("--wd_tagger_threshold", type=float, default=0.35)
    parser.add_argument("--reference_tag_max", type=int, default=24)
    parser.add_argument("--append_reference_tags", action="store_true")
    parser.add_argument("--reference_tag_prefix", default="reference attributes")
    parser.add_argument("--reference_wd_max_refs", type=int, default=2, help="Must match the training run")
    parser.add_argument("--reference_wd_tokens_per_ref", type=int, default=16, help="Must match the training run")
    parser.add_argument("--reference_wd_embed_dim", type=int, default=768, help="Must match the training run")
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--default_steps", type=int, default=12)
    parser.add_argument("--default_guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=3170)
    ns = parser.parse_args()
    if ns.reference_conditioning in {"split_tags", "split_wd_tags"}:
        ns.append_reference_tags = True
    sketch_root = ns.sketch_root if ns.sketch_root and ns.sketch_root.exists() else None
    return WebArgs(
        host=ns.host,
        port=ns.port,
        model_id=ns.model_id,
        lora_dir=ns.lora_dir,
        hf_home=ns.hf_home,
        holdout_manifest=ns.holdout_manifest,
        train_log=ns.train_log,
        sketch_root=sketch_root,
        image_size=ns.image_size,
        preset_count=ns.preset_count,
        device=ns.device,
        local_files_only=ns.local_files_only,
        cpu_offload=ns.cpu_offload,
        preload=ns.preload,
        max_condition_images=ns.max_condition_images,
        spatial_hint_mode=ns.spatial_hint_mode,
        spatial_condition_id_mode=ns.spatial_condition_id_mode,
        reference_conditioning=ns.reference_conditioning,
        reference_condition_mode=ns.reference_condition_mode,
        reference_tag_cache_root=ns.reference_tag_cache_root,
        reference_mask_root=ns.reference_mask_root,
        reference_mask_fallback=ns.reference_mask_fallback,
        skytnt_repo=ns.skytnt_repo,
        skytnt_model_id=ns.skytnt_model_id,
        skytnt_ckpt=ns.skytnt_ckpt,
        skytnt_net=ns.skytnt_net,
        skytnt_image_size=ns.skytnt_image_size,
        skytnt_device=ns.skytnt_device,
        skytnt_fp32=ns.skytnt_fp32,
        skytnt_local_files_only=ns.skytnt_local_files_only,
        reference_cache_generated_masks=not ns.no_reference_cache_generated_masks,
        wd_tagger_model=ns.wd_tagger_model,
        wd_tagger_labels=ns.wd_tagger_labels,
        wd_tagger_threshold=ns.wd_tagger_threshold,
        reference_tag_max=ns.reference_tag_max,
        append_reference_tags=ns.append_reference_tags,
        reference_tag_prefix=ns.reference_tag_prefix,
        reference_wd_max_refs=ns.reference_wd_max_refs,
        reference_wd_tokens_per_ref=ns.reference_wd_tokens_per_ref,
        reference_wd_embed_dim=ns.reference_wd_embed_dim,
        max_sequence_length=ns.max_sequence_length,
        default_steps=ns.default_steps,
        default_guidance=ns.default_guidance,
        seed=ns.seed,
    )


def main() -> None:
    args = parse_args()
    app = AdeleineWebApp(args)
    AdeleineRequestHandler.app = app
    server = HTTPServer((args.host, args.port), AdeleineRequestHandler)
    print(f"Adeleine v2 web server: http://{args.host}:{args.port}")
    print(json.dumps(app.health(), ensure_ascii=False, indent=2))
    server.serve_forever()


if __name__ == "__main__":
    main()
