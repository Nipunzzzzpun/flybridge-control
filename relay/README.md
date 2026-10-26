# FlyLab suggestion relay

Why: FlyLab is open source, so a Discord webhook URL inside it (or inside a file people might commit) would be public,
and anyone could post to your channel. The relay keeps the webhook secret on a small server of yours; the app only knows the relay's address.
What a stranger can do with the relay address: send short, ping-free suggestions, 3 per minute per address and 200 per day in total.

## Set it up (about 10 minutes, free Cloudflare account)
1. Make a NEW webhook in Discord (channel settings > Integrations > Webhooks). Delete any old one.
2. In this `relay` folder:
   ```
   npm install -g wrangler
   wrangler login
   wrangler kv namespace create RL        # copy the id it prints into wrangler.toml
   wrangler secret put WEBHOOK_URL        # paste the Discord webhook URL (it is stored encrypted, not in any file)
   wrangler deploy                        # prints https://flylab-relay.<you>.workers.dev
   ```
3. Put that address in `SUGGEST_RELAY` near the top of `flylab.py` (or `"relay_url"` in `flylab_discord.json`) and commit. It is safe to be public.
4. Never commit `flylab_discord.json` with a real `webhook_url` or bot token (it is already in `.gitignore`).

If the relay gets abused: change `PER_IP_PER_MIN` / `PER_DAY_TOTAL`, or run `wrangler delete` and redeploy to get a new address.
