//! The MiniMax AI video provider (Clip ▸ AI Video): video-to-video through the cloud H3 models
//! (https://platform.minimax.cn). Compiled only with the `minimax` feature (HTTPS via ureq +
//! rustls; off in wasm builds).
//!
//! [`MiniMaxProvider::edit`] runs the four stages of the
//! [video generation V2](https://platform.minimax.cn/docs/api-reference/video-generation-v2-create)
//! flow on the job thread:
//!
//! 1. **Upload** the flattened source render to `v1/files/upload` with
//!    `purpose=video_generation_input` (mp4, ≤ 50 MB, valid for 7 days).
//! 2. **Create** a `v2/video_generation` task that references the upload in its content as
//!    `mm_file://{file_id}` with `role=reference_video` (multimodal-reference generation) plus the
//!    prompt, `resolution=768P` and the source length clamped into the model's 4–15 s range.
//! 3. **Poll** `v2/query/video_generation/{task_id}` every 5 s (up to 15 minutes) until the task
//!    succeeds, fails or is cancelled; every poll checks the job's cancel flag.
//! 4. **Download** the resulting mp4 from `task.content.url` into `out`.
//!
//! API errors (HTTP status or `base_resp.status_code`) surface as the job's error string, which
//! the UI shows on the job — never a crash, and nothing on the timeline is touched.

use std::io::Read;
use std::path::Path;
use std::sync::PoisonError;
use std::time::Duration;

use serde_json::{Value, json};
use web_time::Instant;

use crate::ai_video::AiVideoProvider;
use filmcraft_export::Progress;

/// The MiniMax open platform's API root.
const BASE: &str = "https://api.minimax.cn";
/// The model used for generation (the standard H3; Max is faster but caps at 768P anyway).
const MODEL: &str = "MiniMax-H3";
/// Output resolution (H3 also offers 2K; 768P keeps the prototype fast and cheap).
const RESOLUTION: &str = "768P";
/// The upload endpoint's `video_generation_input` purpose caps a reference video at 50 MB.
const MAX_UPLOAD_BYTES: usize = 50 << 20;
/// The model's duration range (seconds): the source length is clamped into it.
const MIN_DURATION: i64 = 4;
const MAX_DURATION: i64 = 15;
/// Query cadence and overall budget for a generation task.
const POLL_EVERY: Duration = Duration::from_secs(5);
const POLL_BUDGET: Duration = Duration::from_secs(15 * 60);

/// The MiniMax H3 provider: needs an API key (Settings ▸ AI Video ▸ MiniMax API Key).
pub struct MiniMaxProvider;

impl AiVideoProvider for MiniMaxProvider {
    fn id(&self) -> &'static str {
        "minimax"
    }
    fn label(&self) -> &'static str {
        "MiniMax H3 (cloud)"
    }
    fn edit(
        &self,
        services: &dyn crate::Services,
        api_key: &str,
        input: &Path,
        prompt: &str,
        source_seconds: f64,
        out: &Path,
        progress: &Progress,
    ) -> std::result::Result<(), String> {
        use std::sync::atomic::Ordering;
        let key = api_key.trim();
        if key.is_empty() {
            return Err("no MiniMax API key: set it in Settings ▸ AI Video".into());
        }
        let agent = agent();

        // 1. Upload the source render (mp4, purpose=video_generation_input).
        *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Uploading…".into();
        let input = input.to_string_lossy().into_owned();
        let bytes = services.read_file(&input).map_err(|e| format!("{input}: {e}"))?;
        if bytes.len() > MAX_UPLOAD_BYTES {
            return Err(format!(
                "the source render is {} MB; MiniMax accepts up to 50 MB per reference video (shorten the selection)",
                bytes.len() / (1 << 20)
            ));
        }
        progress.total.store(bytes.len().max(1) as u64, Ordering::Relaxed);
        progress.done.store(0, Ordering::Relaxed);
        if progress.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let file_id = parse_upload(&post_multipart(&agent, key, &format!("{BASE}/v1/files/upload"), &bytes)?)?;
        progress.done.store(bytes.len() as u64, Ordering::Relaxed);

        // 2. Create the generation task referencing the upload.
        if progress.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Generating ({MODEL})…");
        let task_id = parse_created(&post_json(&agent, key, &format!("{BASE}/v2/video_generation"), &task_body(prompt, &file_id, source_seconds))?)?;

        // 3. Poll until the task leaves queued/running (and keep the job cancellable).
        let deadline = Instant::now() + POLL_BUDGET;
        let url = loop {
            if progress.cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let v = get_json(&agent, key, &format!("{BASE}/v2/query/video_generation/{task_id}"))?;
            match parse_task(&v)? {
                TaskResult::Done(url) => break url,
                TaskResult::Waiting(status) => {
                    *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Generating ({MODEL}), {status}…");
                }
                TaskResult::Failed(e) => return Err(e),
            }
            if Instant::now() >= deadline {
                return Err(format!("the MiniMax task {task_id} did not finish in {} minutes", POLL_BUDGET.as_secs() / 60));
            }
            let until = Instant::now() + POLL_EVERY;
            while Instant::now() < until {
                if progress.cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                std::thread::sleep(Duration::from_millis(250).min(until.saturating_duration_since(Instant::now())));
            }
        };

        // 4. Download the result into `out` (via services, so a web host could too).
        if progress.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Downloading…".into();
        let video = get_bytes(&agent, &url)?;
        progress.total.store(video.len().max(1) as u64, Ordering::Relaxed);
        progress.done.store(0, Ordering::Relaxed);
        let out = out.to_string_lossy().into_owned();
        services.write_file(&out, &video).map_err(|e| format!("{out}: {e}"))?;
        progress.done.store(video.len().max(1) as u64, Ordering::Relaxed);
        Ok(())
    }
}

// ------------------------------------------------------------------ HTTP

fn agent() -> ureq::Agent {
    let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .unversioned_rustls_crypto_provider(provider)
                .build(),
        )
        .timeout_connect(Some(Duration::from_secs(30)))
        // HTTP errors come back as responses: MiniMax's real reason is in the JSON body
        .http_status_as_error(false)
        .build()
        .into()
}

fn auth(key: &str) -> String {
    format!("Bearer {key}")
}

/// Response body as JSON, with the HTTP status checked. Every request and response is logged
/// (`filmcraft.log`), so a server-side error can be matched against the console's request log.
fn json_of(resp: ureq::http::Response<ureq::Body>, url: &str) -> Result<Value, String> {
    let status = resp.status().as_u16();
    let mut body = resp.into_body();
    let mut text = String::new();
    body.as_reader().read_to_string(&mut text).map_err(|e| format!("MiniMax: reading the response failed: {e}"))?;
    log::info!("MiniMax {url} → HTTP {status} {}", truncate_str(&text));
    if !(200..300).contains(&status) {
        // the body usually carries `base_resp` with the real reason; otherwise show it raw
        if let Ok(v) = serde_json::from_str::<Value>(&text)
            && let Some(msg) = base_message(&v)
        {
            return Err(format!("MiniMax (HTTP {status}): {msg}"));
        }
        return Err(format!("MiniMax: HTTP {status}: {}", truncate_str(&text)));
    }
    serde_json::from_str(&text).map_err(|e| format!("MiniMax: the response is not JSON: {e}"))
}

fn get_json(agent: &ureq::Agent, key: &str, url: &str) -> Result<Value, String> {
    let resp = agent.get(url).header("Authorization", auth(key)).call().map_err(|e| format!("MiniMax: {e}"))?;
    json_of(resp, url)
}

fn post_json(agent: &ureq::Agent, key: &str, url: &str, body: &Value) -> Result<Value, String> {
    log::info!("MiniMax POST {url} {}", body);
    // send_json sets `Content-Type: application/json` itself
    let resp = agent.post(url).header("Authorization", auth(key)).send_json(body.clone()).map_err(|e| format!("MiniMax: {e}"))?;
    json_of(resp, url)
}

/// `multipart/form-data` with the `purpose` field and the file (built by hand: one boundary, two
/// parts — ureq has no multipart support of its own).
fn post_multipart(agent: &ureq::Agent, key: &str, url: &str, file: &[u8]) -> Result<Value, String> {
    const BOUNDARY: &str = "filmcraft-ai-video-7f3a";
    log::info!("MiniMax POST {url} ({} bytes, purpose=video_generation_input)", file.len());
    let mut body = Vec::with_capacity(file.len() + 512);
    let push = |body: &mut Vec<u8>, part: &str| body.extend_from_slice(part.as_bytes());
    push(&mut body, &format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nvideo_generation_input\r\n"));
    push(&mut body, &format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"source.mp4\"\r\nContent-Type: video/mp4\r\n\r\n"));
    body.extend_from_slice(file);
    push(&mut body, &format!("\r\n--{BOUNDARY}--\r\n"));
    let resp = agent
        .post(url)
        .header("Authorization", auth(key))
        .header("Content-Type", format!("multipart/form-data; boundary={BOUNDARY}"))
        .send(body)
        .map_err(|e| format!("MiniMax: {e}"))?;
    json_of(resp, url)
}

fn get_bytes(agent: &ureq::Agent, url: &str) -> Result<Vec<u8>, String> {
    let resp = agent.get(url).call().map_err(|e| format!("MiniMax: downloading the result: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("MiniMax: downloading the result returned HTTP {}", resp.status().as_u16()));
    }
    let mut body = resp.into_body();
    let mut out = Vec::new();
    body.as_reader().read_to_end(&mut out).map_err(|e| format!("MiniMax: downloading the result: {e}"))?;
    log::info!("MiniMax downloaded the result ({} bytes)", out.len());
    Ok(out)
}

// ------------------------------------------------------------------ responses

/// `base_resp.status_msg` when `base_resp.status_code` says the call failed (MiniMax's v1 shape).
fn base_message(v: &Value) -> Option<String> {
    let code = v.pointer("/base_resp/status_code").and_then(Value::as_i64).unwrap_or(0);
    (code != 0).then(|| v.pointer("/base_resp/status_msg").and_then(Value::as_str).unwrap_or("unknown error").to_string())
}

/// The uploaded file's id (`file.file_id`) from the upload response. The server has been seen
/// returning the id as a JSON number (the docs show a string), so both are accepted.
fn parse_upload(v: &Value) -> Result<String, String> {
    if let Some(msg) = base_message(v) {
        return Err(format!("MiniMax upload failed: {msg}"));
    }
    id_of(&v.pointer("/file/file_id").cloned().unwrap_or_default()).ok_or_else(|| format!("MiniMax upload returned no file id: {}", truncate(v)))
}

/// The task id from the create response (string or number).
fn parse_created(v: &Value) -> Result<String, String> {
    if let Some(msg) = base_message(v) {
        return Err(format!("MiniMax task creation failed: {msg}"));
    }
    id_of(&v.get("task_id").cloned().unwrap_or_default()).ok_or_else(|| format!("MiniMax task creation returned no task id: {}", truncate(v)))
}

/// An id field as a string: the server returns ids as JSON strings or numbers.
fn id_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// The response JSON, truncated for an error line (hostile input: the body is untrusted).
fn truncate(v: &Value) -> String {
    truncate_str(&v.to_string())
}

/// Any response text, truncated to 300 characters.
fn truncate_str(s: &str) -> String {
    if s.chars().count() <= 300 {
        return s.to_string();
    }
    format!("{}…", s.chars().take(300).collect::<String>())
}

/// What a query response says about the task.
#[derive(Debug)]
enum TaskResult {
    /// The generation succeeded; the result video's URL.
    Done(String),
    /// Still queued or running; the status word for the job line.
    Waiting(&'static str),
    /// The task failed (or its status is unknown): the error for the job.
    Failed(String),
}

fn parse_task(v: &Value) -> Result<TaskResult, String> {
    if let Some(msg) = base_message(v) {
        return Err(format!("MiniMax task query failed: {msg}"));
    }
    let Some(task) = v.get("task") else { return Err("MiniMax task query returned no task".into()) };
    match task.get("status").and_then(Value::as_str).unwrap_or_default() {
        "succeeded" => {
            let url = task.pointer("/content/url").and_then(Value::as_str).map(str::to_string).filter(|u| !u.is_empty());
            Ok(match url {
                Some(url) => TaskResult::Done(url),
                None => TaskResult::Failed("the MiniMax task succeeded but returned no video URL".into()),
            })
        }
        "failed" | "cancelled" => {
            let word = task.get("status").and_then(Value::as_str).unwrap_or("failed");
            let reason = ["fail_reason", "error", "message"].iter().find_map(|k| task.get(k)).and_then(Value::as_str).unwrap_or("no reason given");
            Err(format!("the MiniMax task {word}: {reason}"))
        }
        "queued" => Ok(TaskResult::Waiting("queued")),
        "running" => Ok(TaskResult::Waiting("running")),
        other => Err(format!("the MiniMax task has an unknown status `{other}`")),
    }
}

/// The create-task request body: prompt + the uploaded source as the reference video.
/// `source_seconds` is clamped into the model's 4–15 s duration range.
fn task_body(prompt: &str, file_id: &str, source_seconds: f64) -> Value {
    let duration =
        if source_seconds.is_finite() && source_seconds > 0.0 { (source_seconds.round() as i64).clamp(MIN_DURATION, MAX_DURATION) } else { MIN_DURATION };
    json!({
        "model": MODEL,
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "role": "reference_video", "video_url": {"url": format!("mm_file://{file_id}")}},
        ],
        "resolution": RESOLUTION,
        "duration": duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_response_gives_the_file_id_and_reports_errors() {
        let ok = json!({"file": {"file_id": "f-123", "bytes": 7}, "base_resp": {"status_code": 0, "status_msg": "success"}});
        assert_eq!(parse_upload(&ok).as_deref(), Ok("f-123"));
        // the server has been seen returning the id as a number
        let numeric = json!({"file": {"file_id": 451881386889603u64}, "base_resp": {"status_code": 0, "status_msg": "success"}});
        assert_eq!(parse_upload(&numeric).as_deref(), Ok("451881386889603"));
        let bad = json!({"base_resp": {"status_code": 1004, "status_msg": "invalid file"}});
        assert_eq!(parse_upload(&bad).unwrap_err(), "MiniMax upload failed: invalid file");
        let empty = json!({"base_resp": {"status_code": 0}, "file": {}});
        assert_eq!(parse_upload(&empty).unwrap_err(), format!("MiniMax upload returned no file id: {empty}"));
    }

    #[test]
    fn created_response_gives_the_task_id() {
        let ok = json!({"task_id": "424010985738629"});
        assert_eq!(parse_created(&ok).as_deref(), Ok("424010985738629"));
        let numeric = json!({"task_id": 424010985738629u64});
        assert_eq!(parse_created(&numeric).as_deref(), Ok("424010985738629"));
        let bad = json!({"base_resp": {"status_code": 1008, "status_msg": "file expired"}});
        assert_eq!(parse_created(&bad).unwrap_err(), "MiniMax task creation failed: file expired");
        assert_eq!(parse_created(&json!({})).unwrap_err(), "MiniMax task creation returned no task id: {}");
    }

    #[test]
    fn task_states_map_to_waiting_done_and_errors() {
        let run = json!({"task": {"id": "t", "status": "running"}});
        assert!(matches!(parse_task(&run), Ok(TaskResult::Waiting("running"))));
        let queued = json!({"task": {"id": "t", "status": "queued"}});
        assert!(matches!(parse_task(&queued), Ok(TaskResult::Waiting("queued"))));
        let done = json!({"task": {"status": "succeeded", "content": {"url": "https://cdn/x.mp4"}}});
        assert!(matches!(parse_task(&done), Ok(TaskResult::Done(u)) if u == "https://cdn/x.mp4"));
        let no_url = json!({"task": {"status": "succeeded"}});
        assert!(matches!(parse_task(&no_url), Ok(TaskResult::Failed(_))));
        let failed = json!({"task": {"status": "failed", "fail_reason": "content policy"}});
        assert_eq!(parse_task(&failed).unwrap_err(), "the MiniMax task failed: content policy");
        let unknown = json!({"task": {"status": "warping"}});
        assert_eq!(parse_task(&unknown).unwrap_err(), "the MiniMax task has an unknown status `warping`");
        let no_task = json!({"base_resp": {"status_code": 0}});
        assert_eq!(parse_task(&no_task).unwrap_err(), "MiniMax task query returned no task");
    }

    #[test]
    fn the_task_body_references_the_upload_and_clamps_the_duration() {
        let v = task_body("make it stormy", "f-9", 7.4);
        assert_eq!(v["model"], MODEL);
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "make it stormy");
        assert_eq!(v["content"][1]["type"], "video_url");
        assert_eq!(v["content"][1]["role"], "reference_video");
        assert_eq!(v["content"][1]["video_url"]["url"], "mm_file://f-9");
        assert_eq!(v["duration"], 7);
        assert_eq!(task_body("p", "f", 0.5)["duration"], MIN_DURATION);
        assert_eq!(task_body("p", "f", 600.0)["duration"], MAX_DURATION);
        assert_eq!(task_body("p", "f", f64::NAN)["duration"], MIN_DURATION);
        assert_eq!(v["resolution"], "768P");
    }
}
