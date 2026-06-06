//go:build !windows

package sysmon

// Snapshot holds system memory metrics at a point in time.
type Snapshot struct {
	AvailableRAMinMB uint64
	PageFaults       uint64
}
