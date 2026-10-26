// FlyLab suggestion relay (Cloudflare Workers, free plan).
// The app POSTs {title, text, name} here. Only this Worker knows the real Discord webhook (a secret), so the
// webhook URL is never in the open-source code. The Worker limits sizes, strips pings and rate-limits senders.
const PER_IP_PER_MIN = 3;      // suggestions one address may send per minute
const PER_DAY_TOTAL = 200;     // hard cap for everyone together per day

const clean = (s, max) => String(s ?? "")
  .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, "")
  .replace(/@/g, "@\u200b")                 // no @everyone / @here / @user
  .slice(0, max).trim();

async function bump(env, key, ttl) {
  const n = parseInt((await env.RL.get(key)) || "0", 10) + 1;
  await env.RL.put(key, String(n), { expirationTtl: ttl });
  return n;
}

export default {
  async fetch(req, env) {
    if (req.method !== "POST") return new Response("FlyLab relay", { status: 405 });
    if (parseInt(req.headers.get("content-length") || "0", 10) > 4000) return new Response("too big", { status: 413 });
    let data;
    try { data = await req.json(); } catch { return new Response("bad json", { status: 400 }); }

    const title = clean(data.title, 100), text = clean(data.text, 1500), name = clean(data.name, 40);
    if (!text) return new Response("empty", { status: 400 });
    // urgency emoji: either one of the server's custom emoji (<:Okay:123...>) or a short unicode emoji; anything else is dropped
    const e = String(data.emoji ?? "").trim();
    const emoji = /^(<a?:\w{2,32}:\d{17,20}>|[^\s<>@]{1,8})$/.test(e) ? e : "";

    const ip = req.headers.get("cf-connecting-ip") || "unknown";
    const minute = Math.floor(Date.now() / 60000), day = new Date().toISOString().slice(0, 10);
    if (await bump(env, `ip:${ip}:${minute}`, 90) > PER_IP_PER_MIN) return new Response("slow down", { status: 429 });
    if (await bump(env, `day:${day}`, 90000) > PER_DAY_TOTAL) return new Response("full today", { status: 429 });

    const content = (title ? `# ${title}\n\n` : "") + text + (emoji || name ? `\n-# ${emoji}${emoji && name ? " " : ""}${name ? `Suggested by __${name}__` : ""}` : "");
    const r = await fetch(env.WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: content.slice(0, 1990), allowed_mentions: { parse: [] }, username: "FlyLab suggestions" }),
    });
    return new Response(r.ok ? "ok" : "discord error", { status: r.ok ? 200 : 502 });
  },
};
