# Cloud Worker Setup Guide (Hetzner CX22)

Step-by-step guide to setting up the REE cloud experiment runner.
No prior cloud/DevOps experience assumed.

---

## Part 1: Create a Hetzner Cloud Account

1. Go to https://console.hetzner.cloud/
2. Click **Register** (top right)
3. Enter your email and a password, verify via the confirmation email
4. You'll be asked for payment details (credit card or PayPal).
   Hetzner bills hourly — a CX22 costs roughly EUR 0.007/hr (~$0.008).
   You won't be charged until you create a server, and you can delete
   it at any time to stop charges.
5. Once logged in, you'll see the **Cloud Console** dashboard.

---

## Part 2: Create a Project and API Token

### 2a. Create a project

1. In the Cloud Console, click **+ New project** (left sidebar)
2. Name it `REE` (or whatever you like)
3. Click into the project

### 2b. Generate an API token

You need this token for the GitHub Actions auto-scaler to control the server.

1. In the Cloud Console, click **Security** in the left sidebar
2. Click the **API tokens** tab
3. Click **Generate API token**
4. Name: `ree-scaler`
5. Permissions: **Read & Write**
6. Click **Generate API token**
7. **IMPORTANT**: Copy the token immediately and save it somewhere safe
   (e.g., a password manager). You cannot see it again after closing the dialog.
   It looks like a long string: `hc_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

## Part 3: Set Up SSH Keys (on your Mac)

SSH keys let you log into the cloud server securely without a password.

Open Terminal on your Mac and run:

```bash
# Check if you already have an SSH key
ls ~/.ssh/id_ed25519.pub 2>/dev/null && echo "Key exists" || echo "No key yet"
```

**If "No key yet"**, generate one:

```bash
ssh-keygen -t ed25519 -C "dgolden-ree-cloud"
```

Press Enter to accept the default location (`~/.ssh/id_ed25519`).
You can set a passphrase or leave it blank (Enter twice for no passphrase).

Now copy your public key to clipboard:

```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

### Upload the key to Hetzner

1. In Hetzner Cloud Console, go to **Security** > **SSH keys**
2. Click **Add SSH key**
3. Paste the key you just copied (Cmd+V)
4. Name: `macbook`
5. Click **Add SSH key**

---

## Part 4: Install the Hetzner CLI on your Mac

```bash
brew install hcloud
```

Then configure it with your API token:

```bash
hcloud context create ree
```

It will ask for your API token — paste the token from Part 2b.

Verify it works:

```bash
hcloud server list
```

Should show an empty table (no servers yet).

---

## Part 5: Create the Cloud Server

```bash
hcloud server create \
  --name ree-worker-1 \
  --type cpx22 \
  --image ubuntu-22.04 \
  --ssh-key macbook \
  --location fsn1
```

- `cpx22` = 2 shared x86 vCPUs, 4 GB RAM, 80 GB disk (~EUR 10/month if always on, ~EUR 1-3/month with auto-scaling)
- `fsn1` = Falkenstein, Germany (cheapest EU location)
- `macbook` = the SSH key name you uploaded in Part 3

The output will show the server's IP address, e.g.:

```
Server 12345678 created
IPv4: 167.235.XX.YY
```

**Write down this IP address.** You'll need it next.

Verify the server is running:

```bash
hcloud server list
```

---

## Part 6: Set Up the Server

### 6a. SSH into the server

```bash
ssh root@<YOUR_SERVER_IP>
```

Type `yes` when it asks about the fingerprint (first-time connection).

You're now logged in as root on the cloud server.

### 6b. Create a non-root user

Running experiments as root is bad practice. Create a normal user:

```bash
# Create user (replace 'ree' with whatever username you prefer)
adduser ree --disabled-password --gecos ""

# Give them sudo access (needed for systemd setup)
usermod -aG sudo ree

# Allow passwordless sudo (so the setup script works non-interactively)
echo "ree ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ree

# Copy your SSH key so you can log in as this user directly
mkdir -p /home/ree/.ssh
cp ~/.ssh/authorized_keys /home/ree/.ssh/
chown -R ree:ree /home/ree/.ssh
chmod 700 /home/ree/.ssh
chmod 600 /home/ree/.ssh/authorized_keys
```

Now log out and log back in as the new user:

```bash
exit
ssh ree@<YOUR_SERVER_IP>
```

### 6c. Set up git credentials

The runner needs to push results to GitHub. The simplest method is a
**Personal Access Token (PAT)**:

1. Go to https://github.com/settings/tokens (on GitHub, not Hetzner)
2. Click **Generate new token (classic)**
3. Name: `ree-cloud-worker`
4. Expiration: pick a duration (90 days is fine; you can renew later)
5. Scopes: tick **repo** (full control of private repositories)
6. Click **Generate token**
7. **Copy the token** (starts with `ghp_...`)

Back on the cloud server (SSH session), configure git to use this token:

```bash
git config --global user.name "REE Cloud Worker"
git config --global user.email "nooarche@users.noreply.github.com"
git config --global credential.helper store
```

Now do a test clone that will save the credentials:

```bash
mkdir -p ~/Documents/GitHub
cd ~/Documents/GitHub

# When prompted for password, paste your GitHub PAT (not your GitHub password)
# This test clone saves your credentials so remote_setup.sh won't ask again
git clone https://github.com/Latent-Fields/ree-v3.git REE_Working/ree-v3
```

It will ask:
```
Username for 'https://github.com': nooarche
Password for 'https://dgolden@github.com': <paste your PAT here>
```

The credential helper stores this so you won't be asked again.

### 6d. Run the setup script

```bash
cd ~/Documents/GitHub/REE_Working

# The script clones remaining repos and installs Python + dependencies
chmod +x ree-v3/remote_setup.sh

# Set your GitHub username (used for cloning)
# Not needed -- remote_setup.sh defaults to Latent-Fields org

# Run it -- takes 5-10 minutes
bash ree-v3/remote_setup.sh
```

**Note**: The script will try to install NVIDIA drivers. On a CX22 (no GPU),
this will print a warning and continue — that's fine, ignore it.

### 6e. Verify the setup

```bash
source ~/.venv/ree/bin/activate

cd ~/Documents/GitHub/REE_Working/ree-v3

# Validate the queue
python3 validate_queue.py

# Dry-run the runner (shows what it would do, doesn't run experiments)
python3 experiment_runner.py --dry-run --machine ree-cloud-1
```

You should see the queue contents and the smoke test listed.

### 6f. Install the runner systemd service

```bash
# Edit the service file to insert your username
cd ~/Documents/GitHub/REE_Working/ree-v3
sed "s/<user>/ree/g" ree-runner.service | sudo tee /etc/systemd/system/ree-runner.service > /dev/null

# Tell systemd about the new service
sudo systemctl daemon-reload

# Enable it to start on boot (this is what makes auto-scaling work)
sudo systemctl enable ree-runner

# Start it now to verify
sudo systemctl start ree-runner

# Check it's running
sudo systemctl status ree-runner
```

You should see `Active: active (running)`. The runner is now polling the queue.

To see live logs:

```bash
journalctl -u ree-runner -f
```

Press Ctrl+C to stop watching logs (the runner keeps running in the background).

### 6g. Power off the server (saves money)

The auto-scaler will power it back on when needed. For now:

```bash
# From your Mac (not the server!)
hcloud server shutdown ree-worker-1
```

Or from the Hetzner Cloud Console: click the server > **Power** > **Shutdown**.

---

## Part 7: Set Up the GitHub Actions Auto-Scaler

The auto-scaler is a GitHub Action that runs every 15 minutes, checks
if the experiment queue has pending items, and powers the server on/off.

### 7a. Add the Hetzner API token to GitHub

1. Go to your ree-v3 repository on GitHub:
   https://github.com/Latent-Fields/ree-v3/settings/secrets/actions
2. Click **New repository secret**
3. Name: `HCLOUD_TOKEN`
4. Value: paste the Hetzner API token from Part 2b
5. Click **Add secret**

### 7b. Push the workflow

The workflow file (`ree-v3/.github/workflows/cloud-scaler.yml`) was already
created. You need to push it to GitHub:

```bash
cd /Users/dgolden/Documents/GitHub/REE_Working/ree-v3
git add .github/workflows/cloud-scaler.yml
git add validate_queue.py
git add experiments/v3_onboard_smoke_ree_cloud_1.py
git add ree-runner.service
git add experiment_queue.json
git commit -m "cloud: add Hetzner CX22 auto-scaler + onboarding smoke test"
git push origin HEAD:main
```

### 7c. Test the auto-scaler manually

1. Go to https://github.com/Latent-Fields/ree-v3/actions
2. Click **Cloud Worker Scaler** in the left sidebar
3. Click **Run workflow** (top right) > **Run workflow**
4. Wait ~30 seconds for it to complete
5. Click into the run to see the log. It should show:
   ```
   Pending items: 8
   Server status: off
   Queue has work -- starting cloud worker...
   ```
6. Check: `hcloud server list` — the server should now show `running`
7. SSH in and check: `journalctl -u ree-runner -f` — you should see the runner
   claiming the onboarding smoke test

### 7d. Verify the full round trip

After the smoke test completes (~15 minutes):

1. Check that results appeared in REE_assembly:
   ```bash
   ls ~/Documents/GitHub/REE_Working/REE_assembly/evidence/experiments/v3_onboard_smoke_ree_cloud_1/
   ```
2. The runner should have pushed the results to GitHub automatically (--auto-sync)
3. On your Mac, pull REE_assembly to see the results:
   ```bash
   git -C /Users/dgolden/Documents/GitHub/REE_Working/REE_assembly pull origin master
   ```

---

## Part 8: Day-to-Day Usage

### How it works automatically

- Every 15 minutes, GitHub Actions checks the experiment queue
- If there are pending experiments: server powers on, runner starts, claims work
- If the queue is empty: server powers off (graceful shutdown, current experiment finishes first)
- You don't need to do anything — just queue experiments as usual

### Manual controls

```bash
# Force start the server (e.g., you just queued experiments and don't want to wait 15 min)
hcloud server poweron ree-worker-1

# Force stop the server (e.g., you want to save costs overnight)
hcloud server shutdown ree-worker-1

# Check server status
hcloud server list

# SSH in to check logs
ssh ree@<YOUR_SERVER_IP>
journalctl -u ree-runner -f
```

### Routing experiments to the cloud worker

Most experiments use `machine_affinity: "any"` and will be claimed by whichever
machine gets there first. To force an experiment to run on the cloud worker:

```json
{
  "queue_id": "V3-EXQ-999",
  "machine_affinity": "ree-cloud-1",
  ...
}
```

### Costs

- Server off: $0 (disk storage is included, no hourly charge when powered off)
- Server running: ~$0.008/hr = ~$0.19/day = ~$5.80/month if 24/7
- With auto-scaling (on only when queue has work): typically $1-3/month
- Check your bill: https://console.hetzner.cloud/ > **Billing**

---

## Troubleshooting

### "Permission denied (publickey)" when SSHing

Your SSH key isn't set up correctly. Check:
```bash
ssh -v ree@<YOUR_SERVER_IP>
```
Look for which key it's trying. Ensure `~/.ssh/id_ed25519` exists on your Mac.

### Runner says "No new items" but queue has entries

The queue_id may already be in runner_status.json (previously run). Check:
```bash
ssh ree@<YOUR_SERVER_IP>
cat ~/Documents/GitHub/REE_Working/REE_assembly/evidence/experiments/runner_status/ree-cloud-1.json
```

### git push fails on the cloud server

The PAT may have expired. Generate a new one (Part 6c) and update:
```bash
# On the cloud server
cat ~/.git-credentials
# Delete the old entry and re-clone a repo to trigger credential storage
```

### GitHub Action fails with "could not find server"

The server hasn't been created yet, or the name doesn't match. Verify:
```bash
hcloud server list
```
The server name must be exactly `ree-worker-1`.

### Server won't stop (experiment running)

The auto-scaler uses `hcloud server shutdown` which sends a graceful shutdown
signal. The runner's systemd service has a 10-minute timeout — if an experiment
is mid-episode, it will finish the current episode before stopping.
If you need to force it: `hcloud server poweroff ree-worker-1` (hard power off,
experiment will be lost but the claim TTL will recover after 6 hours).
