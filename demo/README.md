# PianoFlow v126 — GitHub Pages MIDI demo

This is a static, browser-only MIDI demonstration copied from the v126 app. It does not include the local AI analysis server.

Supported here: MIDI import, playback, note-bar and keyboard styling, lyric editing/timing, local settings storage, and browser WebM recording. AI audio-to-piano analysis and server-generated MP4 are not available in this Pages build. MIDI files and settings are kept in the visitor's browser; they are not uploaded.

The demo uses the included original sample MIDI button if a visitor does not have a MIDI file. For WebM recording, use a current Chrome or Edge browser.

Piano samples: Salamander Grand Piano V3 by Alexander Holm, distributed via [Tone.js](https://github.com/Tonejs/audio/tree/master/salamander), licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The app embeds the sample subset for offline playback and transposes samples at playback time. See the repository's `THIRD_PARTY_NOTICES.md` for full attribution.

`piano_samples_embedded.js` is intentionally included here as a copy of the root file so this demo can load its samples using a local relative path.
