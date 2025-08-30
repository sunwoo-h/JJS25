async function submitComment() {
  const input = document.getElementById("commentInput");
  const text = input.value.trim();
  if (!text) return;

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const t = await res.text().catch(() => "");
    console.error("HTTP", res.status, t);
    alert("서버 오류: " + t);
    return;
  }
  const data = await res.json();

  const color = data.confidence_color;
  if (color === "red") showPopup("danger");
  else if (color === "orange") showPopup("warning");
  else {
    postComment(text);
    input.value = "";
  }
}
