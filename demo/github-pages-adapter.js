(function () {
  "use strict";

  window.PIANOFLOW_GITHUB_PAGES = true;
  const SETTINGS_KEY = "pianoflow-saved-video-settings";
  const nativeFetch = window.fetch.bind(window);

  window.fetch = async function (input, init) {
    let path = "";
    try { path = new URL(typeof input === "string" ? input : input.url, location.href).pathname; } catch {}
    if (path.endsWith("/capabilities")) {
      return new Response(JSON.stringify({ aiMelody: false, analysisVersion: null }), {
        status: 200, headers: { "Content-Type": "application/json" }
      });
    }
    if (path.endsWith("/video-settings")) {
      if ((init?.method || input?.method || "GET").toUpperCase() === "POST") {
        try { localStorage.setItem(SETTINGS_KEY, String(init?.body || "{}")); }
        catch { return new Response(JSON.stringify({ error: "ブラウザーの保存領域を利用できません" }), { status: 507 }); }
        return new Response(null, { status: 204 });
      }
      try {
        const saved = localStorage.getItem(SETTINGS_KEY);
        return saved
          ? new Response(saved, { status: 200, headers: { "Content-Type": "application/json" } })
          : new Response(null, { status: 404 });
      } catch { return new Response(null, { status: 404 }); }
    }
    if (["/analyze-melody", "/align-reference", "/mux-piano-mp4"].some(route => path.endsWith(route))) {
      return new Response(JSON.stringify({ error: "GitHub Pages版ではサーバー処理を利用できません" }), {
        status: 501, headers: { "Content-Type": "application/json" }
      });
    }
    return nativeFetch(input, init);
  };

  function encodeVariableLength(value) {
    let buffer = value & 0x7f;
    while ((value >>= 7)) buffer = (buffer << 8) | ((value & 0x7f) | 0x80);
    const result = [];
    while (true) {
      result.push(buffer & 0xff);
      if (buffer & 0x80) buffer >>= 8;
      else break;
    }
    return result;
  }

  function makeSampleMidi() {
    const events = [];
    const pitches = [60, 64, 67, 72, 71, 69, 67, 64, 62, 60, 55, 60];
    pitches.forEach((pitch, index) => {
      const start = index * 480;
      events.push({ tick: start, bytes: [0x90, pitch, 88] });
      events.push({ tick: start + 360, bytes: [0x80, pitch, 0] });
    });
    events.sort((a, b) => a.tick - b.tick || a.bytes[0] - b.bytes[0]);

    const track = [0x00, 0xff, 0x03, 0x0b, ...new TextEncoder().encode("Sample MIDI"),
      0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20];
    let previousTick = 0;
    for (const event of events) {
      track.push(...encodeVariableLength(event.tick - previousTick), ...event.bytes);
      previousTick = event.tick;
    }
    const endTick = pitches.length * 480;
    track.push(...encodeVariableLength(endTick - previousTick), 0xff, 0x2f, 0x00);
    const header = [0x00, 0x01, 0x00, 0x01, 0x01, 0xe0];
    const chunk = bytes => [0x4d, 0x54, 0x72, 0x6b, (bytes.length >>> 24) & 255,
      (bytes.length >>> 16) & 255, (bytes.length >>> 8) & 255, bytes.length & 255, ...bytes];
    const midi = new Uint8Array([0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, ...header, ...chunk(track)]);
    return new File([midi], "PianoFlow-オリジナルサンプル.mid", { type: "audio/midi" });
  }

  function loadSampleMidi() {
    const input = document.getElementById("midiFile");
    if (!input || typeof DataTransfer === "undefined") return;
    const transfer = new DataTransfer();
    transfer.items.add(makeSampleMidi());
    input.files = transfer.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function localizeSaveMessages() {
    const detail = document.getElementById("detail");
    const label = document.getElementById("videoSettingsSavedLabel");
    const exportButton = document.getElementById("export");
    const replace = () => {
      if (detail && /共通設定として保存|HTML直接起動版とWindows版/.test(detail.textContent)) {
        detail.textContent = "設定はこのブラウザーに保存されます。別の端末には同期されません。";
      }
      if (label && label.textContent.includes("共通設定を復元済み")) label.textContent = "このブラウザーから復元済み";
      if (exportButton && /MP4/.test(exportButton.textContent)) exportButton.textContent = "↓ 保存設定からWebMを作成";
    };
    new MutationObserver(replace).observe(document.body, { subtree: true, childList: true, characterData: true });
    replace();
  }

  document.addEventListener("DOMContentLoaded", function () {
    const topPicker = document.getElementById("pickTop");
    const midiPicker = document.getElementById("pickMidi");
    const midiInput = document.getElementById("midiFile");
    if (topPicker && midiInput) topPicker.onclick = () => midiInput.click();
    if (midiPicker) midiPicker.textContent = "♪ MIDIファイルを選ぶ";

    const summary = document.querySelector("details.settings-group summary");
    if (summary) summary.textContent = "MIDIを読み込む";
    if (summary?.parentElement) summary.parentElement.open = true;
    const performanceSummary = Array.from(document.querySelectorAll("details.settings-group summary")).find(node => node.textContent.includes("解析・演奏"));
    if (performanceSummary) performanceSummary.textContent = "MIDIの設定";
    const analysisSettings = document.querySelectorAll("#bpm, #lyrics, #sensitivity");
    analysisSettings.forEach(control => control.closest("label.setting")?.remove());

    const resolution = document.getElementById("exportResolution");
    if (resolution) resolution.innerHTML = '<option value="1080p" selected>標準（1080p）</option>';
    const format = document.getElementById("outputHint");
    if (format) format.textContent = "動画はこのブラウザー内でWebM形式に書き出します。MP4書き出しには対応していません。";
    const status = document.getElementById("status");
    const detail = document.getElementById("detail");
    if (status) status.textContent = "サンプルMIDIを試すか、ご自身のMIDIを選んでください";
    if (detail) detail.textContent = "MIDIはこの端末内で処理します。音声AI解析は含みません。";

    const outputPanel = document.querySelector(".panel.output");
    if (outputPanel && !document.getElementById("pianoSampleAttribution")) {
      const attribution = document.createElement("p");
      attribution.id = "pianoSampleAttribution";
      attribution.className = "pages-demo-note";
      attribution.append("ピアノ音源：");
      const source = document.createElement("a");
      source.href = "https://github.com/Tonejs/audio/tree/master/salamander";
      source.textContent = "Salamander Grand Piano V3";
      attribution.append(source, " by Alexander Holm（");
      const license = document.createElement("a");
      license.href = "https://creativecommons.org/licenses/by/3.0/";
      license.textContent = "CC BY 3.0";
      attribution.append(license, "）。サンプルを再生時に移調します。");
      outputPanel.append(attribution);
    }

    if (midiPicker && !document.getElementById("loadSampleMidi")) {
      const button = document.createElement("button");
      button.type = "button";
      button.id = "loadSampleMidi";
      button.className = "pages-sample-button";
      button.textContent = "▶ オリジナルサンプルで試す";
      button.addEventListener("click", loadSampleMidi);
      midiPicker.insertAdjacentElement("afterend", button);
      const note = document.createElement("p");
      note.className = "pages-demo-note";
      note.textContent = "MIDIを読み込むと、再生・配色・歌詞の編集を試せます。選んだファイルはこの端末のブラウザー内で処理します。";
      button.insertAdjacentElement("afterend", note);
    }
    localizeSaveMessages();
    if (new URLSearchParams(location.search).get("sample") === "1") setTimeout(loadSampleMidi, 120);
  });
})();
