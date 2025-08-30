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

    // ✅ HTTP 에러를 JSON 파싱 전에 먼저 처리
    if (!res.ok) {
      const errText = await res.text(); // 서버가 JSON이면 JSON 문자열, 아니면 텍스트
      throw new Error(`서버 오류 (${res.status}): ${errText}`);
    }

    const data = await res.json(); // 항상 JSON이 보장됨 (백엔드에서 HTTPException JSON 반환)

    const color = data.confidence_color;
    if (color === "red") {
      showPopup("danger");
    } else if (color === "orange") {
      showPopup("warning");
    } else {
      postComment(text);
      input.value = "";
    }
  } catch (err) {
    console.error(err);
    alert("분석 서버 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.");
  }
}

function showPopup(level) {
  const popup = document.getElementById("popup");
  const icon = document.getElementById("popupIcon");
  const title = document.getElementById("popupTitle");
  const text = document.getElementById("popupText");

  // ⬇️ 경고 레벨 클래스 설정 + 숨김 해제
  popup.classList.remove("hidden");
  popup.classList.remove("danger", "warning");
  popup.classList.add(level);

  // ⬇️ 팝업 텍스트 및 이미지 변경
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
  if (!text) return; // 빈 값이면 무시
  postComment(text);
  input.value = ""; // 입력창 초기화
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
