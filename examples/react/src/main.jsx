import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const statusLabels = {
  idle: "대기",
  preparing: "모델 준비 중",
  warming_up: "배경 학습 중",
  running: "분석 중",
  stopping: "종료 중",
  stopped: "종료됨",
  error: "오류",
};
const phaseLabels = {
  before: "이전",
  start: "움직임 시작",
  during: "진행",
  after: "이후",
};
const analysisLabels = {
  ready: "Gemini 연결 대기",
  queued: "분석 대기",
  analyzing: "Gemini 분석 중",
  done: "분석 완료",
  failed: "분석 실패",
};

function App() {
  const [backend, setBackend] = useState(
    import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
  );
  const [state, setState] = useState({ status: "idle" });
  const [records, setRecords] = useState([]);
  const [camera, setCamera] = useState(0);
  const [objects, setObjects] = useState("person");
  const [overlay, setOverlay] = useState(false);
  const [mirror, setMirror] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [originalVideo, setOriginalVideo] = useState(null);
  const api = backend.replace(/\/$/, "");
  const active = ["preparing", "warming_up", "running", "stopping"].includes(
    state.status,
  );

  async function request(path, options = {}) {
    const response = await fetch(api + path, {
      ...options,
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(
        typeof body.detail === "string"
          ? body.detail
          : `요청 실패 (${response.status})`,
      );
    }
    return response.json();
  }

  useEffect(() => {
    let disposed = false,
      loading = false;
    const poll = async () => {
      if (loading) return;
      loading = true;
      try {
        const response = await fetch(api + "/motion/status", {
          signal: AbortSignal.timeout(3000),
        });
        if (!response.ok) throw new Error("상태 조회 실패");
        const next = await response.json();
        const events = next.run_id
          ? await fetch(`${api}/motion/runs/${next.run_id}/events`, {
              signal: AbortSignal.timeout(3000),
            }).then((r) => {
              if (!r.ok) throw new Error("사건 조회 실패");
              return r.json();
            })
          : [];
        if (!disposed) {
          setState(next);
          setRecords(events);
          setConnectionError("");
        }
      } catch (error) {
        if (!disposed)
          setConnectionError(`FastAPI 연결을 확인하세요: ${error.message}`);
      } finally {
        loading = false;
      }
    };
    poll();
    const timer = setInterval(poll, 500);
    return () => {
      disposed = true;
      clearInterval(timer);
    };
  }, [api]);

  useEffect(
    () => () => {
      if (originalVideo) URL.revokeObjectURL(originalVideo);
    },
    [originalVideo],
  );

  async function start(source = "webcam", videoId = null) {
    setBusy(true);
    setMessage("");
    try {
      await request("/motion/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source,
          camera: Number(camera),
          objects: objects
            .split(",")
            .map((v) => v.trim())
            .filter(Boolean),
          mirror: source === "webcam" && mirror,
          overlay,
          video_id: videoId,
        }),
      });
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function upload(event) {
    const file = event.target.files[0];
    if (!file) return;
    setBusy(true);
    setMessage("");
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(api + "/motion/videos", {
        method: "POST",
        body,
        signal: AbortSignal.timeout(120000),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "업로드 실패");
      setOriginalVideo(URL.createObjectURL(file));
      await start("video", data.video_id);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }

  const groups = Object.groupBy(records, (record) => record.event_id);
  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">LOCAL CAMERA / MOTION CAPTURE</p>
          <h1>움직임을 사건으로 모으기</h1>
          <p>
            같은 PC의 Python이 카메라를 읽고, 화면과 사건 사진을 공유합니다.
          </p>
        </div>
        <span className={`status ${state.status}`}>
          {statusLabels[state.status] || state.status}
        </span>
      </header>
      <section className="controls" aria-label="분석 설정">
        <label>
          FastAPI 주소
          <input value={backend} onChange={(e) => setBackend(e.target.value)} />
        </label>
        <label>
          카메라 번호
          <input
            type="number"
            min="0"
            value={camera}
            onChange={(e) => setCamera(e.target.value)}
          />
        </label>
        <label>
          탐지 종류
          <input
            value={objects}
            onChange={(e) => setObjects(e.target.value)}
            placeholder="person,dog"
          />
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={mirror}
            onChange={(e) => setMirror(e.target.checked)}
          />
          미리보기 좌우 반전
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={overlay}
            onChange={(e) => setOverlay(e.target.checked)}
          />
          최근 분석 박스 표시
        </label>
        <div className="actions">
          <button
            disabled={busy || active || !!connectionError}
            onClick={() => start()}
          >
            카메라 시작
          </button>
          <button
            className="secondary"
            disabled={!active || busy}
            onClick={async () => {
              try {
                await request(`/motion/runs/${state.run_id}/stop`, {
                  method: "POST",
                });
              } catch (e) {
                setMessage(e.message);
              }
            }}
          >
            중지
          </button>
          <label className={`upload ${active || busy ? "disabled" : ""}`}>
            동영상 분석
            <input
              type="file"
              accept="video/*"
              disabled={active || busy}
              onChange={upload}
            />
          </label>
        </div>
      </section>
      {(message || connectionError || state.error || state.preview_error) && (
        <p role="alert" className="error">
          {message || connectionError || state.error || state.preview_error}
        </p>
      )}
      <div className="workspace">
        <section className="live">
          <div className="section-title">
            <h2>카메라 · 분석 화면</h2>
            <span>{state.event_active ? "움직임 수집 중" : "사건 대기"}</span>
          </div>
          <div className="preview">
            {active && state.run_id && !connectionError ? (
              <img
                src={`${api}/motion/runs/${state.run_id}/preview.mjpg`}
                alt="같은 PC 카메라의 최신 미리보기"
              />
            ) : (
              <div className="empty">
                {state.status === "stopped"
                  ? "분석이 종료되었습니다."
                  : "카메라를 시작하거나 동영상을 선택하세요."}
              </div>
            )}
          </div>
          <div className="metrics">
            <span>
              처리 <strong>{state.frames_processed || 0}</strong> 프레임
            </span>
            <span>
              사건 <strong>{state.episode_count || 0}</strong>
            </span>
            <span>
              묶음 <strong>{state.bundle_count || 0}</strong>
            </span>
            <span>
              <strong>{(state.processing_fps || 0).toFixed(1)}</strong> FPS
            </span>
          </div>
          {originalVideo && (
            <details>
              <summary>업로드한 원본 동영상 재생</summary>
              <video src={originalVideo} controls />
            </details>
          )}
        </section>
        <section className="events">
          <div className="section-title">
            <h2>사건 사진 묶음</h2>
            <span>{Object.keys(groups).length}건</span>
          </div>
          {records.length === 0 ? (
            <div className="empty">
              움직임 구간이 끝나면 사진 묶음이 표시됩니다.
            </div>
          ) : (
            Object.entries(groups)
              .reverse()
              .map(([id, parts]) => (
                <article key={id} className="event">
                  <h3>사건 {id.slice(-6)}</h3>
                  {parts.map((record) => (
                    <div key={record.part_index} className="part">
                      <div className="part-heading">
                        <strong>묶음 {record.part_index}</strong>
                        <span>
                          {record.start_sec.toFixed(1)}–
                          {record.end_sec.toFixed(1)}초
                        </span>
                      </div>
                      <p className="tags">
                        {record.end_reason} ·{" "}
                        {record.episode_end ? "사건 종료" : "이어서 수집"} ·{" "}
                        {analysisLabels[record.analysis_status]}
                      </p>
                      {!record.context_complete && (
                        <p className="quality">
                          맥락 부족 또는 관측 누락 · 사전{" "}
                          {record.pre_context_complete ? "충분" : "부족"} / 사후{" "}
                          {record.post_context_complete ? "충분" : "부족"}
                        </p>
                      )}
                      {!record.evidence_complete && (
                        <p className="quality">
                          일부 움직임 증거 누락 (
                          {record.omitted_moving_objects.length}개 추적 키)
                        </p>
                      )}
                      <div className="photos">
                        {record.images.map((image, index) => (
                          <a
                            key={image.frame_index}
                            href={api + image.url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            <img
                              loading="lazy"
                              src={api + image.url}
                              alt={`묶음 ${record.part_index} 사진 ${index + 1}`}
                            />
                            <span>
                              {phaseLabels[image.phase]} ·{" "}
                              {image.timestamp_sec.toFixed(1)}초
                            </span>
                          </a>
                        ))}
                      </div>
                      {record.analysis_result != null && (
                        <pre>
                          {typeof record.analysis_result === "string"
                            ? record.analysis_result
                            : JSON.stringify(record.analysis_result, null, 2)}
                        </pre>
                      )}
                      {record.analysis_error && (
                        <p className="error">{record.analysis_error}</p>
                      )}
                    </div>
                  ))}
                </article>
              ))
          )}
        </section>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
