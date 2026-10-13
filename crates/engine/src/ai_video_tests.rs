//! Tests of [`crate::ai_video`]: the stub chain nests the selection, flattens it to H.264, places
//! the generated result on a new top track of the nest, and regenerations stack above it. Failed
//! and hostile runs end as errors, never panics.

use super::ai_video::{META_KEY, META_SOURCE, PendingAi, StubProvider, add_pending_overlay, is_overlay_clip};
use crate::{Job, Session};
use filmcraft_export::Report;
use filmcraft_media::DemoScene;
use filmcraft_project::{ItemId, ItemKind};
use filmcraft_time::Tick;
use serde_json::json;
use std::path::PathBuf;

fn fixture(name: &str) -> (Session, ItemId, String) {
    let movie_dir = crate::media_test_util::tmp_dir("ai-video");
    let path: PathBuf = movie_dir.join(format!("{name}.mp4"));
    crate::media_test_util::make_movie(&path, DemoScene::Aurora, 64, 64, 6);
    // each test generates into its own directory, so parallel runs cannot collide on file names
    let out_dir = crate::media_test_util::tmp_dir("ai-video-out").to_string_lossy().into_owned();
    let (mut s, _items, seq) = crate::media_test_util::session_with(&[&path]);
    // the tests exercise the stub chain; a session may default to a real provider (MiniMax)
    s.ai_provider = std::sync::Arc::new(StubProvider);
    (s, seq, out_dir)
}

/// The first clip of the sequence's first video track (what the tests select).
fn first_clip(s: &Session, seq: ItemId) -> filmcraft_project::TrackItem {
    s.project.sequence(seq).unwrap().video_tracks[0].items.first().unwrap().clone()
}

/// The clip of `seq` that references a nested sequence (None until one is nested).
fn nest_clip(s: &Session, seq: ItemId) -> Option<filmcraft_project::TrackItem> {
    s.project
        .sequence(seq)
        .unwrap()
        .video_tracks
        .iter()
        .flat_map(|t| t.items.iter())
        .find_map(|i| matches!(s.project.items.get(&i.item).map(|x| &x.kind), Some(ItemKind::Sequence(_))).then(|| i.clone()))
}

/// The top track's last clip of the nest, with its `AI Generate` and `AI Source` metadata.
fn top_generated(s: &Session, nest: ItemId) -> (filmcraft_project::TrackItem, String, String) {
    let track = s.project.sequence(nest).unwrap().video_tracks.last().unwrap();
    let it = track.items.last().unwrap();
    let item = s.project.items.get(&it.item).unwrap();
    (it.clone(), item.metadata.get(META_KEY).cloned().unwrap_or_default(), item.metadata.get(META_SOURCE).cloned().unwrap_or_default())
}

fn is_sequence_item(s: &Session, item: ItemId) -> bool {
    matches!(s.project.items.get(&item).map(|x| &x.kind), Some(ItemKind::Sequence(_)))
}

#[test]
fn stub_chain_nests_renders_and_places() {
    let (mut s, seq, dir) = fixture("chain");
    let original = first_clip(&s, seq);
    s.state.selection = vec![original.id];
    let r = s.execute("clip.aiVideo", json!({"prompt": "make it golden", "dir": dir, "wait": true})).unwrap();
    assert_eq!(r["mode"], "first");
    let nest = ItemId(r["sequence"].as_u64().unwrap());

    // the nest clip replaced the original in place: same start, same duration
    let placed = nest_clip(&s, seq).unwrap();
    assert_eq!(placed.start, original.start);
    assert_eq!(placed.duration, original.duration);

    // the nest holds the original below and the result on a new top track, tagged; the "AI is
    // generating" overlay is gone again
    let q = s.project.sequence(nest).unwrap();
    assert_eq!(q.video_tracks.len(), 2);
    assert!(!q.video_tracks.iter().flat_map(|t| t.items.iter()).any(is_overlay_clip), "the overlay is removed with the result");
    assert!(q.video_tracks[0].items.iter().all(|i| !is_sequence_item(&s, i.item)), "the original stays on the bottom track");
    let (top, prompt, source) = top_generated(&s, nest);
    assert_eq!(prompt, "make it golden");
    assert!(source.ends_with("-ai-src.mp4"), "{source}");
    assert_eq!(top.start, Tick::ZERO);
    assert!(top.duration <= placed.duration, "the result is trimmed to the nest's length");

    // the nest clip is selected again, so the next invocation regenerates
    assert_eq!(s.state.selection, vec![placed.id]);
}

#[test]
fn regeneration_stacks_on_top_of_the_previous_result() {
    let (mut s, seq, dir) = fixture("regen");
    s.state.selection = vec![first_clip(&s, seq).id];
    let r1 = s.execute("clip.aiVideo", json!({"prompt": "first pass", "dir": dir, "wait": true})).unwrap();
    let nest = ItemId(r1["sequence"].as_u64().unwrap());
    let first = top_generated(&s, nest).0;

    // the nest clip is still selected (place() selects it): this is a regeneration
    let r2 = s.execute("clip.aiVideo", json!({"prompt": "second pass", "dir": dir, "wait": true})).unwrap();
    assert_eq!(r2["mode"], "regen");
    assert_eq!(r2["sequence"], json!(nest.0));

    // a third track holds the new result; the first result and the original stay
    let q = s.project.sequence(nest).unwrap();
    assert_eq!(q.video_tracks.len(), 3);
    assert!(!q.video_tracks.iter().flat_map(|t| t.items.iter()).any(is_overlay_clip), "the overlay is removed with the result");
    let (second, prompt, _) = top_generated(&s, nest);
    assert_eq!(prompt, "second pass");
    assert_ne!(second.item, first.item);
    assert!(q.video_tracks.iter().flat_map(|t| t.items.iter()).any(|i| i.id == first.id), "the first result is kept");
}

#[test]
fn hostile_params_are_errors() {
    let (mut s, seq, _dir) = fixture("hostile");
    s.state.selection = vec![first_clip(&s, seq).id];

    // a missing or empty prompt is rejected
    assert!(s.execute("clip.aiVideo", json!({})).is_err());
    assert!(s.execute("clip.aiVideo", json!({"prompt": "   "})).is_err());
    // an over-long prompt is rejected, not truncated
    let long = "x".repeat(META_KEY.len() + 10_000);
    assert!(s.execute("clip.aiVideo", json!({"prompt": long, "wait": true})).is_err());
    // the nest did not happen: the fixture's own sequence is still the only sequence item
    assert_eq!(s.project.items.values().filter(|i| matches!(i.kind, ItemKind::Sequence(_))).count(), 1);

    // audio only: nothing to generate from
    let audio_only = s.project.sequence(seq).unwrap().audio_tracks[0].items[0].id;
    s.state.selection = vec![audio_only];
    assert!(s.execute("clip.aiVideo", json!({"prompt": "x", "wait": true})).is_err());

    // no selection: the command cannot even start
    s.state.selection.clear();
    assert!(s.execute("clip.aiVideo", json!({"prompt": "x", "wait": true})).is_err());
}

#[test]
fn a_result_that_cannot_be_placed_ends_as_an_error_not_a_crash() {
    let (mut s, _seq, _dir) = fixture("gone");
    // a finished job whose nest is gone: poll must clear the pending entry without panicking
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    let job = Job { id, label: "AI Video".into(), progress: Default::default(), result: Default::default() };
    *job.result.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(Ok(Report { path: String::new(), frames: 0, seconds: 0.0, bytes: 0, render_fps: 0.0, extra_files: Vec::new() }));
    job.progress.finished.store(true, std::sync::atomic::Ordering::Relaxed);
    s.jobs.push(job);
    s.ai_jobs.push(PendingAi { job: id, nest: ItemId(999_999), prompt: "p".into(), out: "/nonexistent/ai.mp4".into(), source: "/nonexistent/src.mp4".into() });
    super::ai_video::poll(&mut s);
    assert!(s.ai_jobs.is_empty());
}

#[test]
fn the_overlay_is_removed_when_the_job_fails() {
    let (mut s, seq, dir) = fixture("overlay-fail");
    s.state.selection = vec![first_clip(&s, seq).id];
    let r = s.execute("clip.aiVideo", json!({"prompt": "first pass", "dir": dir, "wait": true})).unwrap();
    let nest = ItemId(r["sequence"].as_u64().unwrap());

    // a regeneration starts: the overlay goes on top of the nest while the job runs
    add_pending_overlay(&mut s, nest, "AI is generating…").unwrap();
    assert!(s.project.sequence(nest).unwrap().video_tracks.iter().any(|t| t.items.iter().any(is_overlay_clip)));

    // the job fails (no balance, no network…): the overlay comes off, the result stays
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    let job = Job { id, label: "AI Video".into(), progress: Default::default(), result: Default::default() };
    *job.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(Err("insufficient balance".into()));
    job.progress.finished.store(true, std::sync::atomic::Ordering::Relaxed);
    s.jobs.push(job);
    s.ai_jobs.push(PendingAi { job: id, nest, prompt: "p".into(), out: "/nonexistent/ai.mp4".into(), source: "/nonexistent/src.mp4".into() });
    super::ai_video::poll(&mut s);

    let q = s.project.sequence(nest).unwrap();
    assert!(!q.video_tracks.iter().any(|t| t.items.iter().any(is_overlay_clip)), "the overlay is gone");
    assert_eq!(top_generated(&s, nest).1, "first pass", "the previous result is untouched");
}

#[test]
fn a_deleted_source_render_is_a_clean_error() {
    let (mut s, seq, dir) = fixture("src-gone");
    s.state.selection = vec![first_clip(&s, seq).id];
    let r = s.execute("clip.aiVideo", json!({"prompt": "pass one", "dir": dir, "wait": true})).unwrap();
    let nest = ItemId(r["sequence"].as_u64().unwrap());
    let (_, _, source) = top_generated(&s, nest);

    // the flattened source render disappears (cleanup, another machine…)
    let _ = std::fs::remove_file(&source);
    let r2 = s.execute("clip.aiVideo", json!({"prompt": "pass two", "dir": dir, "wait": true}));
    assert!(r2.is_err(), "{r2:?}");
    // the first result is untouched
    assert_eq!(top_generated(&s, nest).1, "pass one");
}
