//go:build !windows

package sysmon

// Snapshot holds system memory metrics at a point in time.
type Snapshot struct {
	AvailableRAM_MB uint64
	PageFaults      uint64
}

// Capture returns an empty snapshot on non-Windows platforms.
func Capture() Snapshot {
	return Snapshot{}
}
