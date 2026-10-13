//! AI video generation (Clip ▸ AI Video…): describe an edit, get a generated clip on top of the
//! selection.
//!
//! | command | does |
//! |---|---|
//! | `clip.aiVideo` | nest the selection (unless it already holds a generated result), render it flattened to H.264 and run the AI video provider on it in a background job; the result is imported and placed on a new top video track of the nest when the job finishes |
//! | `ai.providers` | the available providers (the stub ships; real models plug in here) |
//!
//! **The nest is the safe slot** (design decision, 2026-10-13): the selection is nested first
//! (`clip.nest`, same position and length), so the timeline slot is taken before any work starts
//! and the generated result can never cover clips on tracks above the selection — inside the nest
//! the top tracks are empty. The original stays untouched on the bottom track.
//!
//! **Flattened source render.** The first generation renders the nest (which holds exactly what
//! was selected) to `<name>-ai-src.mp4` (H.264, no audio, the nest's frame size and rate, a high
//! bitrate — models re-encode the input, so input artifacts compound). Regenerating a nest that
//! already holds a result reuses that file as the provider input instead of re-rendering, and
//! stacks the new result on another new top track, so the nest keeps the generation history.
//!
//! **Recognition** is via `ProjectItem.metadata`: generated media items carry `"AI Generate"` (the
//! prompt) and `"AI Source"` (the flattened source render they came from). A nest whose topmost
//! generated-tagged clip exists is a generated nest. No project schema changes.
//!
//! **Applying finished jobs.** Like proxies ([`crate::proxies::poll`]), finished jobs are applied
//! on the session thread by [`poll`] (from `Session::poll_persistence`): import, tag, place. A
//! nest deleted while generating leaves the job to end with an error; nothing on the timeline is
//! touched and the imported item (if any) stays in the Project panel.
//!
//! Files go where narrations go (`dir` param, else Scratch Disks ▸ Captured, else next to the
//! project, else `AI Video` in the data or temporary directory), never into the repository.

use std::path::Path;
use std::sync::PoisonError;

use filmcraft_export::{ExportSettings, Format, Progress, Report};
use filmcraft_project::{ItemId, ItemKind, Track, TrackId, TrackKind};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p, has_selection, str_p};
use crate::{EngineError, Result, Session};

/// Metadata key of generated media items (the prompt).
pub const META_KEY: &str = "AI Generate";
/// Metadata key of the flattened source render a generated item came from.
pub(crate) const META_SOURCE: &str = "AI Source";
/// Prompt size cap (bytes): hostile input, and the prompt is stored in the project.
const MAX_PROMPT_BYTES: usize = 10_000;
/// The text-layer name marking the "AI is generating" overlay clips (the marker lives on the
/// layer because the overlay's graphic canvas item is shared with ordinary graphics).
pub(crate) const OVERLAY_LAYER: &str = "AI Generating";
/// The default overlay text (the UI passes a localized one as `pendingText`).
const DEFAULT_PENDING: &str = "AI is generating…";

/// An AI video provider: edits `input` (an H.264/mp4 file of `source_seconds` length) per
/// `prompt`, writing the result to `out`. `api_key` is the provider's credential (empty = none
/// configured: the provider answers with a clean error). Progress updates the job's status line.
pub trait AiVideoProvider: Send + Sync {
    fn id(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn edit(
        &self,
        services: &dyn crate::Services,
        api_key: &str,
        input: &Path,
        prompt: &str,
        source_seconds: f64,
        out: &Path,
        progress: &Progress,
    ) -> std::result::Result<(), String>;
}

/// The provider a new session uses: MiniMax when compiled in, the stub otherwise.
pub fn default_provider() -> std::sync::Arc<dyn AiVideoProvider> {
    #[cfg(feature = "minimax")]
    {
        std::sync::Arc::new(crate::ai_video_minimax::MiniMaxProvider)
    }
    #[cfg(not(feature = "minimax"))]
    {
        std::sync::Arc::new(StubProvider)
    }
}

/// The built-in stand-in: simulates the stages and copies the input to the output, so the whole
/// pipeline (nest, render, job, import, place) can be built and tested before a real model is
/// plugged in. The copy is renamed `-ai`, so the result is visibly distinct from the source.
pub struct StubProvider;

impl AiVideoProvider for StubProvider {
    fn id(&self) -> &'static str {
        "stub"
    }
    fn label(&self) -> &'static str {
        "Stub (copies the input)"
    }
    fn edit(
        &self,
        services: &dyn crate::Services,
        _api_key: &str,
        input: &Path,
        _prompt: &str,
        _source_seconds: f64,
        out: &Path,
        progress: &Progress,
    ) -> std::result::Result<(), String> {
        use std::sync::atomic::Ordering;
        let input = input.to_string_lossy().into_owned();
        let out = out.to_string_lossy().into_owned();
        *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Uploading…".into();
        let bytes = services.read_file(&input).map_err(|e| format!("{input}: {e}"))?;
        progress.total.store(bytes.len().max(1) as u64, Ordering::Relaxed);
        progress.done.store(0, Ordering::Relaxed);
        if progress.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Generating…".into();
        progress.done.store(bytes.len() as u64 / 2, Ordering::Relaxed);
        *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Downloading…".into();
        services.write_file(&out, &bytes).map_err(|e| format!("{out}: {e}"))?;
        progress.done.store(bytes.len().max(1) as u64, Ordering::Relaxed);
        Ok(())
    }
}

/// A generation job whose result still has to be imported and placed into the nest.
pub struct PendingAi {
    pub job: u64,
    /// The nest the result goes into (a new top video track).
    pub nest: ItemId,
    pub prompt: String,
    /// The provider's output file (imported by [`poll`]).
    pub out: String,
    /// The flattened source render, recorded on the placed item for regenerations.
    pub source: String,
}

/// The selected clip's nest (None unless the first selected clip references a sequence).
fn selected_nest(s: &Session) -> Option<ItemId> {
    let seq = s.active_sequence()?;
    let clip = s.state.selection.first().copied()?;
    let item = seq.find_item(clip).map(|(_, it)| it.item)?;
    matches!(s.project.items.get(&item)?.kind, ItemKind::Sequence(_)).then_some(item)
}

/// Whether a nest holds a generated result (a video-track clip whose item carries [`META_KEY`]).
fn is_generated(s: &Session, nest: ItemId) -> bool {
    s.project.sequence(nest).is_some_and(|q| {
        q.video_tracks.iter().any(|t| t.items.iter().any(|it| s.project.items.get(&it.item).is_some_and(|x| x.metadata.contains_key(META_KEY))))
    })
}

/// A file name stem from an item name: path separators and control characters become `-`.
fn stem_of(name: &str) -> String {
    let mut out: String = name.chars().take(80).map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '-' }).collect();
    out = out.trim().trim_matches('-').to_string();
    if out.is_empty() {
        out = "AI Video".into();
    }
    out
}

/// The flattened source render of a generated nest, from its generated items' [`META_SOURCE`].
fn source_render(s: &Session, nest: ItemId) -> Option<String> {
    s.project.sequence(nest)?.video_tracks.iter().flat_map(|t| t.items.iter()).find_map(|it| {
        let x = s.project.items.get(&it.item)?;
        x.metadata.contains_key(META_KEY).then(|| x.metadata.get(META_SOURCE).cloned()).flatten()
    })
}

fn generate(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "clip.aiVideo";
    let prompt = str_p(p, "prompt").map(str::trim).filter(|t| !t.is_empty()).ok_or_else(|| bad(cmd, "`prompt` is required"))?;
    if prompt.len() > MAX_PROMPT_BYTES {
        return Err(bad(cmd, format!("`prompt` is longer than {MAX_PROMPT_BYTES} bytes")));
    }
    let prompt = prompt.to_string();
    let wait = bool_p(p, "wait").unwrap_or(false);
    let dir = crate::voiceover::media_dir(s, p, "AI Video");

    // the selection must hold picture: audio alone has nothing to generate from
    if let Some(seq) = s.active_sequence() {
        let on_video = |c: filmcraft_project::ClipId| seq.find_item(c).is_some_and(|(t, _)| seq.video_tracks.iter().any(|vt| vt.id == t));
        if !s.state.selection.iter().copied().any(on_video) {
            return Err(bad(cmd, "select a clip with video"));
        }
    }

    // Nest-or-reuse: a generated nest regenerates from its flattened source render; anything else
    // is nested first, so the nest we render holds exactly what was selected.
    let (nest, regen) = match selected_nest(s) {
        Some(n) if is_generated(s, n) => (n, true),
        _ => {
            let r = s.execute("clip.nest", json!({"name": crate::commands::next_nested_name(s)}))?;
            let n = r["sequence"].as_u64().map(ItemId).ok_or_else(|| EngineError::Other("nesting did not return a sequence".into()))?;
            (n, false)
        }
    };

    let (source, stem) = if regen {
        let src = source_render(s, nest).ok_or_else(|| bad(cmd, "the nest's source render is not recorded; delete the generated clips and generate again"))?;
        if s.services.file_size(&src).is_err() {
            return Err(bad(cmd, format!("the source render {} is gone; delete the generated clips and generate again", Path::new(&src).display())));
        }
        let name = s.project.items.get(&nest).map(|i| i.name.clone()).unwrap_or_default();
        (src, stem_of(&name))
    } else {
        let name = s.project.items.get(&nest).map(|i| i.name.clone()).unwrap_or_default();
        let stem = stem_of(&name);
        let out = proxies_unique_path(&dir, &stem, "-ai-src");
        (out, stem)
    };
    let out = proxies_unique_path(&dir, &stem, "-ai");

    if !filmcraft_export::available(Format::H264) {
        return Err(EngineError::Other("H.264 export is not available (no encoder registered)".into()));
    }
    let q = s.project.sequence(nest).ok_or(EngineError::NoSequence)?;
    let (width, height) = (q.settings.width, q.settings.height);
    // the nest's length: the model's target duration (clamped into its own range by the provider)
    let nest_len = q.video_tracks.iter().chain(q.audio_tracks.iter()).flat_map(|t| t.items.iter()).map(|i| i.end()).max().unwrap_or_default();
    let source_seconds = nest_len.seconds();
    let settings = ExportSettings {
        format: Format::H264,
        path: source.clone(),
        scale: 1.0,
        include_audio: false,
        quality: 85,
        // high on purpose: the model re-encodes the input, so input artifacts compound
        bitrate_kbps: (f64::from(width) * f64::from(height) / 100.0).max(2_000.0) as u32,
        part_of_batch: true,
        ..Default::default()
    };

    // the "AI is generating" overlay covers the nest while the job runs (removed by place()/poll())
    let pending_text = str_p(p, "pendingText").map(str::trim).filter(|t| !t.is_empty()).unwrap_or(DEFAULT_PENDING);
    add_pending_overlay(s, nest, &pending_text.chars().take(200).collect::<String>())?;

    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    let job = crate::Job { id, label: format!("AI Video: {}", stem), progress: Default::default(), result: Default::default() };
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let project = s.project.clone();
    let provider = s.media.full_res_provider(project.clone(), s.services.clone());
    let ai = s.ai_provider.clone();
    let services = s.services.clone();
    let api_key = s.prefs.ai_video.minimax_api_key.clone();
    let pending = PendingAi { job: id, nest, prompt: prompt.clone(), out: out.clone(), source: source.clone() };
    let run = move || {
        use std::sync::atomic::Ordering;
        let t0 = web_time::Instant::now();
        let r = (|| -> std::result::Result<String, String> {
            if !regen {
                *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = "Rendering the selection…".into();
                if let Some(d) = Path::new(&source).parent() {
                    let _ = std::fs::create_dir_all(d);
                }
                filmcraft_export::export(&project, nest, &settings, &provider, &prog).map_err(|e| e.to_string())?;
            }
            if prog.cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Generating ({})…", ai.label());
            ai.edit(&*services, &api_key, Path::new(&source), &prompt, source_seconds, Path::new(&out), &prog)?;
            if prog.cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            Ok(out.clone())
        })();
        let secs = t0.elapsed().as_secs_f64();
        let r = match r {
            Ok(path) => {
                *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Done in {secs:.1}s");
                Ok(Report { path, frames: 0, seconds: secs, bytes: 0, render_fps: 0.0, extra_files: Vec::new() })
            }
            Err(e) => {
                let _ = std::fs::remove_file(&out);
                *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = e.clone();
                *prog.error.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.clone());
                Err(e)
            }
        };
        prog.finished.store(true, Ordering::Relaxed);
        *res.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
    };
    let run = crate::export_tools::guard_job(job.progress.clone(), job.result.clone(), run);
    s.jobs.push(job);
    s.ai_jobs.push(pending);
    if wait || cfg!(target_arch = "wasm32") {
        run();
        poll(s);
    } else {
        std::thread::Builder::new().name("filmcraft-ai".into()).spawn(run).map_err(|e| EngineError::Other(e.to_string()))?;
    }
    Ok(json!({"job": id, "sequence": nest.0, "mode": if regen { "regen" } else { "first" }}))
}

/// Whether a clip is the "AI is generating" overlay (its text layer carries [`OVERLAY_LAYER`]).
pub(crate) fn is_overlay_clip(it: &filmcraft_project::TrackItem) -> bool {
    it.effects.iter().any(|e| {
        e.effect == filmcraft_project::graphic::TEXT_LAYER
            && matches!(e.param("name").map(|p| p.value_at(Tick::ZERO)), Some(filmcraft_project::ParamValue::Text(t)) if t == OVERLAY_LAYER)
    })
}

/// Add the "AI is generating" overlay: a new top video track in the nest holding a centred text
/// graphic for the nest's whole length. [`remove_pending_pr`] takes it out again when the
/// generation finishes (or fails), so the overlay covers exactly the waiting time.
pub(crate) fn add_pending_overlay(s: &mut Session, nest: ItemId, text: &str) -> Result<()> {
    let (w, h, rate, dur) = {
        let q = s.project.sequence(nest).ok_or(EngineError::NoSequence)?;
        let dur = q.video_tracks.iter().chain(q.audio_tracks.iter()).flat_map(|t| t.items.iter()).map(|i| i.end()).max().unwrap_or_default();
        (q.settings.width, q.settings.height, q.settings.frame_rate, dur)
    };
    if dur <= Tick::ZERO {
        return Ok(());
    }
    s.edit("AI Generating Overlay", |pr, _| {
        let src = crate::graphics::graphic_source(pr, w, h, rate);
        let mut layer = filmcraft_project::graphic::new_text_layer(text, filmcraft_geom::Vec2::new(0.0, 0.0), (f64::from(h) / 12.0).max(12.0));
        // an unset position auto-centres; a background strip keeps the text readable
        layer.params.remove("position");
        layer.params.insert("name".into(), filmcraft_project::Param::new(filmcraft_project::ParamValue::Text(OVERLAY_LAYER.into())));
        layer.params.insert("background".into(), filmcraft_project::Param::new(filmcraft_project::ParamValue::Bool(true)));
        let track_id = TrackId(pr.alloc_id());
        let mut ti = pr
            .make_track_item(src, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, dur), rate)
            .ok_or_else(|| EngineError::Other("the overlay cannot be placed".into()))?;
        ti.name = "AI Generating…".into();
        ti.effects.push(layer);
        let seq = pr.sequence_mut(nest).ok_or(EngineError::NoSequence)?;
        let index = seq.video_tracks.len();
        seq.video_tracks.push(Track::new(track_id, TrackKind::Video, format!("Video {}", index + 1)));
        seq.video_tracks[index].items.push(ti);
        seq.check().map_err(EngineError::Other)
    })
}

/// Remove the overlay clips (and the tracks they emptied) from the nest.
fn remove_pending_pr(pr: &mut filmcraft_project::Project, nest: ItemId) -> Result<()> {
    let Some(seq) = pr.sequence_mut(nest) else { return Ok(()) };
    let mut emptied: Vec<TrackId> = Vec::new();
    for t in &mut seq.video_tracks {
        if t.items.iter().any(is_overlay_clip) {
            t.items.retain(|i| !is_overlay_clip(i));
            if t.items.is_empty() {
                emptied.push(t.id);
            }
        }
    }
    seq.video_tracks.retain(|t| !emptied.contains(&t.id));
    seq.check().map_err(EngineError::Other)
}

/// Take the overlay out again after a failed generation (best effort: a nest deleted while
/// generating has nothing left to clean up).
pub(crate) fn remove_pending(s: &mut Session, nest: ItemId) {
    if s.project.sequence(nest).is_none_or(|q| !q.video_tracks.iter().any(|t| t.items.iter().any(is_overlay_clip))) {
        return;
    }
    if let Err(e) = s.edit("AI Generating Overlay", |pr, _| remove_pending_pr(pr, nest)) {
        s.error_toast("ai.video", e.to_string());
    }
}

/// `<dir>/<stem><suffix>.mp4`, numbered when taken (thin wrapper over [`crate::proxies::unique_path`]).
fn proxies_unique_path(dir: &str, stem: &str, suffix: &str) -> String {
    crate::proxies::unique_path(Path::new(dir), stem, suffix, "mp4", &[]).to_string_lossy().into_owned()
}

/// Apply finished generation jobs: import the result, tag it, place it on a new top video track of
/// the nest. Failed jobs are left alone (their error is on the job); a placement error is a toast,
/// never a crash — the imported item stays in the Project panel either way.
pub fn poll(s: &mut Session) {
    use std::sync::atomic::Ordering;
    let finished: Vec<usize> = s
        .ai_jobs
        .iter()
        .enumerate()
        .filter(|(_, pj)| s.jobs.iter().find(|j| j.id == pj.job).is_none_or(|j| j.progress.finished.load(Ordering::Relaxed)))
        .map(|(i, _)| i)
        .collect();
    for i in finished.into_iter().rev() {
        let pj = s.ai_jobs.remove(i);
        let ok =
            s.jobs.iter().find(|j| j.id == pj.job).and_then(|j| j.result.lock().unwrap_or_else(PoisonError::into_inner).clone()).is_some_and(|r| r.is_ok());
        if !ok {
            remove_pending(s, pj.nest);
            continue;
        }
        if let Err(e) = place(s, &pj) {
            s.error_toast("ai.video", e.to_string());
        }
    }
}

/// Import `pj.out` and place it: new top video track in the nest, start aligned to the nest's
/// start, longer than the nest trimmed to it (the original shows through where it is shorter).
fn place(s: &mut Session, pj: &PendingAi) -> Result<()> {
    let r = s.execute("file.import", json!({"paths": [pj.out]}))?;
    let item = ItemId(
        r["items"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(Value::as_u64)
            .ok_or_else(|| EngineError::Other("importing the result returned no item".into()))?,
    );
    let (prompt, source, nest) = (pj.prompt.clone(), pj.source.clone(), pj.nest);
    let active = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    s.edit("AI Video Result", |pr, st| {
        let dur = pr.item(item).and_then(|i| i.as_media()).map(|m| m.info.duration).ok_or_else(|| EngineError::Other("the result is not media".into()))?;
        let rate = pr.sequence(nest).ok_or(EngineError::NoSequence)?.settings.frame_rate;
        let track_id = TrackId(pr.alloc_id());
        let mut ti = pr
            .make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, dur), rate)
            .ok_or_else(|| EngineError::Other("the result cannot be placed".into()))?;
        let it = pr.item_mut(item).ok_or_else(|| EngineError::Other("the result disappeared".into()))?;
        it.metadata.insert(META_KEY.into(), prompt);
        it.metadata.insert(META_SOURCE.into(), source);
        let seq = pr.sequence_mut(nest).ok_or(EngineError::NoSequence)?;
        let len = seq.video_tracks.iter().chain(seq.audio_tracks.iter()).flat_map(|t| t.items.iter()).map(|i| i.end()).max().unwrap_or_default();
        ti.duration = ti.duration.min(len);
        let index = seq.video_tracks.len();
        seq.video_tracks.push(Track::new(track_id, TrackKind::Video, format!("Video {}", index + 1)));
        seq.video_tracks[index].items.push(ti);
        // the waiting is over: take the "AI is generating" overlay out again (it re-checks the nest)
        remove_pending_pr(pr, nest)?;
        // select the nest clip in the active sequence, so pressing the shortcut again regenerates
        st.selection = pr
            .sequence(active)
            .map(|q| q.video_tracks.iter().flat_map(|t| t.items.iter()).filter(|i| i.item == nest).map(|i| i.id).collect())
            .unwrap_or_default();
        Ok(())
    })?;
    Ok(())
}

/// `ai.providers`: the available AI video providers.
fn providers(s: &mut Session, _p: &Value) -> Result<Value> {
    Ok(json!({"default": s.ai_provider.id(), "providers": [{"id": s.ai_provider.id(), "label": s.ai_provider.label()}]}))
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "clip.aiVideo",
            label: "AI Video…",
            menu: &["Clip"],
            shortcut: Some("Cmd+Alt+G"),
            params: r#"{"prompt":str,"dir":str?,"wait":bool=false,"pendingText":str?}"#,
            enabled: has_selection,
            run: generate,
            journal: true,
        },
        CommandSpec {
            id: "ai.providers",
            label: "AI Video Providers",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: crate::commands::always,
            run: providers,
            journal: false,
        },
    ]
}
