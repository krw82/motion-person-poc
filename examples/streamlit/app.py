"""Streamlit is a viewer of the same FastAPI run, never another camera reader."""

import requests
import streamlit as st

st.set_page_config(page_title="Motion capture", layout="wide")
st.title("카메라 모션 · 사건 사진")
backend = st.sidebar.text_input("FastAPI 주소", "http://127.0.0.1:8000").rstrip("/")
camera = st.sidebar.number_input("카메라 번호", min_value=0, value=0)
try:
    if st.sidebar.button("카메라 시작"):
        response = requests.post(
            backend + "/motion/runs", json={"camera": camera}, timeout=10
        )
        response.raise_for_status()
    if st.sidebar.button("중지"):
        state = requests.get(backend + "/motion/status", timeout=3).json()
        if state["run_id"]:
            requests.post(
                f"{backend}/motion/runs/{state['run_id']}/stop", timeout=3
            ).raise_for_status()

    video = st.sidebar.file_uploader("동영상", type=["mp4", "mov", "avi", "mkv"])
    if video is not None and st.sidebar.button("동영상 분석"):
        upload = requests.post(
            backend + "/motion/videos",
            files={"file": (video.name, video.getvalue())},
            timeout=60,
        )
        upload.raise_for_status()
        response = requests.post(
            backend + "/motion/runs",
            json={"source": "video", **upload.json(), "mirror": False},
            timeout=10,
        )
        response.raise_for_status()

except requests.RequestException as exc:
    st.sidebar.error(f"FastAPI 요청을 처리하지 못했습니다: {exc}")


@st.fragment(run_every=0.3)
def viewer():
    try:
        response = requests.get(backend + "/motion/status", timeout=3)
        response.raise_for_status()
        state = response.json()
        st.caption(
            f"상태: {state['status']} · 처리 {state.get('frames_processed', 0)}프레임"
        )
        if state.get("error"):
            st.error(state["error"])
        run_id = state["run_id"]
        if not run_id:
            st.info("같은 PC의 FastAPI에서 카메라를 시작하세요.")
            return
        if state["status"] not in ("stopped", "error"):
            image = requests.get(
                f"{backend}/motion/runs/{run_id}/preview.jpg", timeout=2
            )
            if image.ok:
                st.image(image.content, caption="현재 카메라", width=640)
        response = requests.get(f"{backend}/motion/runs/{run_id}/events", timeout=3)
        response.raise_for_status()
        groups = {}
        for record in response.json():
            groups.setdefault(record["event_id"], []).append(record)
        for event_id, parts in reversed(groups.items()):
            st.subheader(f"사건 {event_id[-6:]}")
            for record in parts:
                st.markdown(f"**묶음 {record['part_index']}**")
                st.caption(
                    f"{record['start_sec']:.1f}–{record['end_sec']:.1f}초 · {record['end_reason']} · 분석 {record['analysis_status']}"
                )
                if not record["context_complete"]:
                    st.caption("사전·사후 맥락이 부족하거나 관측 누락이 있습니다.")
                if not record["evidence_complete"]:
                    st.caption("일부 움직임 증거가 사진에 포함되지 않았습니다.")
                st.image(
                    [backend + image["url"] for image in record["images"]], width=180
                )
                if record["analysis_result"] is not None:
                    st.write(record["analysis_result"])
                if record["analysis_error"]:
                    st.warning(record["analysis_error"])
    except requests.RequestException as exc:
        st.warning(f"FastAPI 연결을 확인하세요: {exc}")


viewer()
