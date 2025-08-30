async function submitComment() {
  const input = document.getElementById("commentInput");
  const text = input.value.trim();
  if (!text) return;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    // 1) HTTP 에러 방어
    if (!res.ok) {
      const raw = await res.text().catch(() => "");
      console.error("HTTP error:", res.status, raw);
      // 서버가 죽었거나 일시 에러 → 사용자 경험 우선: 댓글은 등록해줌(데모/실사용 취향에 따라 경고로 바꿔도 됨)
      postComment(text);
      input.value = "";
      return;
    }

    // 2) JSON 파싱
    let data;
    try {
      data = await res.json();
    } catch (e) {
      console.error("JSON parse error:", e);
      // JSON이 아니어도 fail-open
      postComment(text);
      input.value = "";
      return;
    }

    // 3) 서버측 로직 에러 (ok=false) 처리
    if (data && data.ok === false) {
      console.warn("Predict failed on server:", data.error);
      // 서버 추론 실패 시에도 일반 댓글은 등록(원한다면 alert로 바꾸세요)
      postComment(text);
      input.value = "";
      return;
    }

    // 4) 정상 흐름
    const color = data?.confidence_color || null;

    if (color === "red") {
      showPopup("danger");
    } else if (color === "orange") {
      showPopup("warning");
    } else {
      postComment(text);
      input.value = ""; // 일반 댓글도 작성 후 초기화!
    }
  } catch (err) {
    // 네트워크 예외 방어
    console.error("Network error:", err);
    postComment(text); // 네트워크 불안정 시에도 댓글은 등록(필요시 alert로 교체)
    input.value = "";
  }
}

function showPopup(level) {
  const popup = document.getElementById("popup");
  const icon = document.getElementById("popupIcon");
  const title = document.getElementById("popupTitle");
  const text = document.getElementById("popupText");

  popup.classList.remove("hidden");
  popup.classList.remove("danger", "warning");
  popup.classList.add(level);

  if (level === "danger") {
    title.textContent = "부정적인 표현이 많습니다";
    text.textContent =
      "해당 댓글은 다른 사용자에게 불쾌감을 줄 수 있어요. 게시 전 다시 확인해 주세요.";
    icon.src = "/static/warning-red.png";
  } else {
    title.textContent = "권장되지 않는 표현이 포함되어 있어요";
    text.textContent =
      "해당 댓글은 표현이 부적절할 수 있어요. 다시 한번 검토해 주세요.";
    icon.src = "/static/warning-yellow.png";
  }
}

function rewriteComment() {
  document.getElementById("popup").classList.add("hidden");
}

function postAnyway() {
  const input = document.getElementById("commentInput");
  const text = input.value.trim();
  if (!text) return;
  postComment(text);
  input.value = "";
  document.getElementById("popup").classList.add("hidden");
}

function getRelativeTime(date) {
  const now = new Date();
  const diffSec = Math.floor((now - date) / 1000);
  if (diffSec < 60) return "방금 전";
  else if (diffSec < 3600) return `${Math.floor(diffSec / 60)}분 전`;
  else
    return `${now.getHours()}:${now.getMinutes().toString().padStart(2, "0")}`;
}

function postComment(text) {
  const commentList = document.getElementById("commentList");
  const now = new Date();
  const timeStr = getRelativeTime(now);
  const el = document.createElement("div");
  el.className = "comment-item";
  el.innerHTML = `<p><strong>you</strong> <span>${timeStr}</span><br/>${text}</p>`;
  commentList.appendChild(el);
  document.getElementById("commentInput").value = "";
}
