# Repartition nvme2n1 (Samsung MZQL2 Enterprise SSD)

## Current layout
- /dev/nvme2n1p1 — 1.7TB ext4 mounted at /data (empty, only lost+found)
- /data/swapfile — 400GB swapfile (hibernate broken: GRUB/initramfs offset mismatch)

## Target layout
| Partition | Size | Type | Mount | Purpose |
|-----------|------|------|-------|---------|
| nvme2n1p1 | 400GB | linux-swap | swap | Hibernate (377GB RAM) |
| nvme2n1p2 | ~1.4TB | ext4 | /data | HF models, caches |

## Why
- Swap partition eliminates resume_offset headache (no offset needed, just UUID)
- Enterprise SSD (1 DWPD / 3500 TBW) is ideal for hibernate writes
- Current swapfile hibernate is broken: GRUB offset=110395392 vs initramfs offset=4161536

## Commands

```bash
# 1. Stop anything using /data
sudo swapoff /data/swapfile
sudo swapoff /dev/md0p2
sudo umount /data

# 2. Repartition: 400GB swap + rest ext4
sudo parted /dev/nvme2n1 --script -- \
  mklabel gpt \
  mkpart swap linux-swap 1MiB 400GiB \
  mkpart data ext4 400GiB 100%

# 3. Format both partitions
sudo mkswap /dev/nvme2n1p1
sudo mkfs.ext4 /dev/nvme2n1p2

# 4. Get new swap UUID
SWAP_UUID=$(blkid -s UUID -o value /dev/nvme2n1p1)
DATA_UUID=$(blkid -s UUID -o value /dev/nvme2n1p2)
echo "Swap UUID: ${SWAP_UUID}"
echo "Data UUID: ${DATA_UUID}"

# 5. Update /etc/fstab — replace old entries with:
#   UUID=<SWAP_UUID>  none   swap  sw  0 0
#   UUID=<DATA_UUID>  /data  ext4  defaults  0 2
#   tmpfs             /tmp   tmpfs defaults,size=32G  0 0
#   (remove old /data and swapfile entries, remove md0p2 swap entry)

# 6. Enable new swap + mount
sudo swapon /dev/nvme2n1p1
sudo mkdir -p /data
sudo mount /dev/nvme2n1p2 /data
sudo chown alex:alex /data

# 7. Update GRUB for hibernate resume
sudo sed -i "s/resume=UUID=.*/resume=UUID=${SWAP_UUID}/" /etc/default/grub
# Remove resume_offset from GRUB (not needed for partition)
sudo sed -i 's/ resume_offset=[0-9]*//' /etc/default/grub
sudo update-grub

# 8. Update initramfs resume config
echo "RESUME=UUID=${SWAP_UUID}" | sudo tee /etc/initramfs-tools/conf.d/resume
sudo update-initramfs -u

# 9. Reboot and test hibernate
sudo reboot
# After reboot: sudo systemctl hibernate
```

## After repartition: move HF cache to /data
```bash
mv ~/.cache/huggingface /data/huggingface
ln -s /data/huggingface ~/.cache/huggingface
```

## Future: break RAID 1
- nvme0n1 (9100 PRO) → OS standalone
- nvme1n1 (9100 PRO) → additional model storage
- nvme2n1 (MZQL2) → swap + /data
- Total usable: ~5.5TB NVMe
