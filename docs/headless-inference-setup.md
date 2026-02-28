# Headless and Inference Readiness Setup Report

**Date:** 2026-02-22
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (SM120, CC 12.0)
**OS:** Ubuntu 24.04 LTS (Noble Numbat)
**Primary Goal:** Stable headless operation for `vllm` inference via Remote Desktop (RDP).

---

## 1. Remote Desktop (RDP) Configuration

Ubuntu 24.04 introduces a distinction between **Remote Login** (System-wide) and **Remote Desktop** (User-session). For a headless server, these must be configured to avoid port conflicts.

### Port Allocation
| Service | Port | Use Case |
|---------|------|----------|
| **Remote Login** | `3389` | Fresh login session (Headless / No monitor). |
| **Remote Desktop** | `3390` | Mirroring an existing active session (Admin/Shared). |

### Configuration Commands

```bash
# 1. Configure System-wide Remote Login (Port 3389)
sudo grdctl --system rdp set-credentials <REMOTE_USER> <REMOTE_PASSWORD>
sudo grdctl --system rdp set-port 3389
sudo grdctl --system rdp enable
sudo grdctl --system rdp disable-view-only
sudo systemctl restart gnome-remote-desktop.service

# 2. Configure User-session Desktop Sharing (Port 3390)
grdctl rdp set-credentials <USER> <PASSWORD>
grdctl rdp set-port 3390
grdctl rdp enable
systemctl --user restart gnome-remote-desktop.service
```

### Connection from Microsoft Remote Desktop (Mac/Windows)
- **Headless Login:** `192.168.50.132` (Default port 3389). Use system RDP credentials.
- **Shared Desktop:** `192.168.50.132:3390`. Use user-level RDP credentials.

---

## 2. Headless System Readiness

To prevent the server from disconnecting or the GPUs from idling during inference, the following system adjustments were applied.

### Power Management (Prevent Sleep)
```bash
# Disable sleep/suspend on AC power
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'

# Mask system sleep targets to prevent accidental suspension
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

### NVIDIA Persistence Mode
Ensures the NVIDIA driver remains loaded even when no applications are using the GPU, preventing initialization delays for `vllm`.
```bash
# Enable persistence mode for all GPUs
sudo nvidia-smi -pm 1
```

### Virtual Display (Dummy Monitor)
NVIDIA drivers require a display context to initialize certain features. For a headless setup, a virtual configuration is required:
```bash
sudo nvidia-xconfig --allow-empty-initial-configuration --virtual=1920x1080
```

---

## 3. Session Persistence & UX Alternatives

Running `vllm` directly in an RDP terminal will kill the process if the RDP window is closed. You must use a persistent session handler.

#### **Option A: Zellij (Best Modern UX)**
Zellij is a modern alternative to `tmux` with a built-in UI that shows shortcuts at the bottom. It does not require a "prefix" key for every action.
- **Install:** `curl -L https://github.com/zellij-org/zellij/releases/latest/download/zellij-x86_64-unknown-linux-musl.tar.gz | tar xz && sudo install zellij /usr/local/bin/`
- **Why use it:** Easier discovery of shortcuts, mouse support by default, and a much cleaner visual interface.
- **Detach:** `Ctrl + O` then `D`.

#### **Option B: Optimized Tmux (If you prefer standard tools)**
The default `tmux` shortcuts are often considered cumbersome. You can optimize them by creating a `~/.tmux.conf`:
```bash
# Change prefix to Ctrl+A (easier to reach)
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# Enable mouse support for scrolling and switching panes
set -g mouse on
```
- **Usage:** `tmux new -s vllm` -> `Ctrl+A` then `D` to detach.

#### **Option C: VS Code Remote-SSH**
If using VS Code on a client machine, the Remote-SSH extension maintains terminal persistence automatically. This is often the most comfortable UX for developers who want to avoid terminal multiplexers entirely.

---

## 4. vLLM Inference Stability

### Recommended Runtime Environment
- **Process management:** Consider a `systemd` unit for `vllm` for auto-restart on boot.
- **Memory Overhead:** Current `vLLM` configuration uses `gpu_memory_utilization: 0.80` to accommodate the 177B model and sampler warmup on the RTX 6000 Ada (96GB).

---

## 4. Troubleshooting Codes

| Error | Meaning | Resolution |
|-------|---------|------------|
| `0x4` | Internal Protocol Error | Usually an RDP credential mismatch. Check `grdctl` settings. |
| `0x207` | Handshake/Redirection Failure | Occurs when trying to connect to a logged-in user via the System Login port. Use port 3390 or log out physically. |
| `ERRINFO_LOGOFF_BY_USER` | Session Terminated | System is redirecting to an existing session that RDP cannot "hand off" to. Ensure the user is logged out for Port 3389. |
