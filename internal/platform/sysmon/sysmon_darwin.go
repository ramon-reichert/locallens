//go:build darwin

package sysmon

import (
	"os/exec"
	"strconv"
	"strings"
)

// Capture takes a best-effort snapshot of current system memory state on macOS.
func Capture() Snapshot {
	var snap Snapshot

	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return snap
	}

	pageSize := uint64(4096)
	freePages := uint64(0)
	inactivePages := uint64(0)
	speculativePages := uint64(0)
	pageFaults := uint64(0)

	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "Mach Virtual Memory Statistics:") {
			fields := strings.Fields(line)
			for i, field := range fields {
				if field == "page" && i+3 < len(fields) && fields[i+1] == "size" && fields[i+2] == "of" {
					if n, err := strconv.ParseUint(fields[i+3], 10, 64); err == nil {
						pageSize = n
					}
					break
				}
			}
			continue
		}

		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		n, err := strconv.ParseUint(strings.Trim(strings.TrimSpace(value), "."), 10, 64)
		if err != nil {
			continue
		}

		switch key {
		case "Pages free":
			freePages = n
		case "Pages inactive":
			inactivePages = n
		case "Pages speculative":
			speculativePages = n
		case "Translation faults":
			pageFaults = n
		}
	}

	availableBytes := (freePages + inactivePages + speculativePages) * pageSize
	snap.AvailableRAMinMB = availableBytes / (1024 * 1024)
	snap.PageFaults = pageFaults

	return snap
}
